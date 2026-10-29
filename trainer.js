// ============================================================================
// SPATIAL AI TRAINER v9 (trainer.js) — 雙階混合空間判讀與後台修正系統
//
// 完整三階段自動銜接流程 (Three-Stage Hybrid Pipeline)：
//   [Stage 1] 瀏覽器端即時初判 (In-Browser Spatial MLP + COCO-SSD + 透視地牆交界幾何演算法)：
//             約 30ms 內快速標出初步的地牆基準線、垂直轉角線、家具框與初始空間分類。
//   [Stage 2] 雲端 VLM 空間判讀 API 自動銜接更新 (Cloud VLM Spatial API Auto-Refinement)：
//             將畫面影像連同「Stage 1 的初步判讀結果 + 歷史人工修正過的 Few-Shot 範例」傳送至雲端 VLM
//             (支援 Google Gemini 2.5 Flash API / OpenAI GPT-4o-mini / Node.js 後端代理)，
//             自動修正轉角陰陽性、門扇開合、前後景深層次與動線阻擋，並無縫更新至畫面與表單！
//   [Stage 3] 後台人工審核微調與模型蒸餾訓練 (Human-in-the-Loop Override & Distillation)：
//             使用者可在後台最後確認或修改任何框選與屬性，按下「確認後台修改並回饋訓練」，
//             即可將雲端 VLM 與人工修正的結晶「蒸餾 (Distill)」回本地 TF.js MLP 模型中！
// ============================================================================
(() => {
'use strict';

const $ = (id) => document.getElementById(id);
const cv = $('workCv');
const ctx = cv.getContext('2d');

let cocoModel = null;
let spatialNet = null;
let currentImg = null;
let currentFeatVec = new Array(18).fill(0);
let elements = []; // [{ id, x, y, w, h, type, layer, action, dist, source: 'local'|'vlm'|'user' }]
let selectedElId = null;
let dataset = [];

const TYPOLOGY_KEYS = ['corridor', 'corner_space', 'cluttered_room', 'dead_end', 'open_hall'];
const PATHC_KEYS = ['clear', 'blocked_obj', 'blocked_wall', 'door_way'];
const SUPP_KEYS = ['room', 'alley', 'cover', 'wall'];

const TYPE_LABELS = {
  corner_convex: { name: '陽角 (凸轉角)', color: '#6effe1', defaultAction: 'attach_wall' },
  corner_concave: { name: '陰角 (凹轉角)', color: '#38bdf8', defaultAction: 'attach_room' },
  door_open: { name: '門扇／通道開口', color: '#a78bfa', defaultAction: 'pass_through' },
  wall_deadend: { name: '封閉牆面／死路', color: '#f87171', defaultAction: 'stop_prioritize_real' },
  obstacle_block: { name: '阻擋物 (桌椅家具)', color: '#fb923c', defaultAction: 'detour_avoid' },
  obstacle_cover: { name: '半高掩體 (矮桌/沙發)', color: '#facc15', defaultAction: 'use_as_cover' },
};

const LAYER_LABELS = {
  foreground: '前景 (0~2m)',
  midground: '中景 (2~5m)',
  background: '後景 (5m+)',
};

const ACTION_LABELS = {
  attach_room: '附著生成 2.4m 實心房間',
  attach_wall: '附著延伸 2.4m 實心牆',
  use_as_cover: '作為掩體／補充 1.1m 胸牆',
  detour_avoid: '阻擋動線（實體優先・需繞行）',
  stop_prioritize_real: '死路邊界（禁止虛擬穿牆）',
  pass_through: '保持通道淨空（禁止封堵）',
};

function log(msg) {
  const box = $('logBox');
  const time = new Date().toLocaleTimeString();
  box.innerHTML = `<div>[${time}] ${msg}</div>` + box.innerHTML;
}

function setPipelineStage(stage) {
  $('step1Indicator').className = 'pipe-step' + (stage === 1 ? ' active' : '');
  $('step2Indicator').className = 'pipe-step' + (stage === 2 ? ' vlm-active' : '');
  $('step3Indicator').className = 'pipe-step' + (stage === 3 ? ' active' : '');
}

// ============================================================================
// 0. 載入與儲存雲端 VLM API 設定
// ============================================================================
// 修正紀錄（v8）：
//   · 舊版第 402、418 行字串內含直接換行 → 整個 trainer.js 語法錯誤，所有按鈕（含「儲存」）都沒有作用。
//   · Google 目前限制「新的」API Key 使用 gemini-2.5 系列，改為：列出這把 Key 可用的模型並自動挑選 Flash。
//   · 加上 45 秒逾時與看得懂的錯誤訊息；回應若含思考段落（thought）會略過。
const LS_KEY = 'arSpatial.vlmKey', LS_PROV = 'arSpatial.vlmProvider', LS_MODEL = 'arSpatial.vlmModel';
const DEFAULT_MODEL = { gemini: 'gemini-3.6-flash', openai: 'gpt-4o-mini', server: 'gemini-3.6-flash' };
const GEMINI_BASE = 'https://generativelanguage.googleapis.com/v1beta';

function lsGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); return true; } catch (e) { return false; } }

function setVlmStatus(text, kind) {
  const el = $('vlmStatus');
  if (!el) return;
  el.textContent = text;
  el.style.color = kind === 'ok' ? 'var(--accent)' : kind === 'err' ? 'var(--danger)' : 'var(--dim)';
}

function currentModel() {
  const v = ($('vlmModel').value || '').trim().replace(/^models\//, '');
  return v || DEFAULT_MODEL[$('vlmProvider').value] || DEFAULT_MODEL.gemini;
}

function loadVLMConfig() {
  const savedKey = lsGet(LS_KEY), savedProv = lsGet(LS_PROV), savedModel = lsGet(LS_MODEL);
  if (savedKey) $('vlmApiKey').value = savedKey;
  if (savedProv) $('vlmProvider').value = savedProv;
  $('vlmModel').value = savedModel || DEFAULT_MODEL[$('vlmProvider').value] || '';
  if (savedKey) setVlmStatus('已讀取先前儲存的 Key（' + $('vlmProvider').value + '）。按「儲存並測試」確認可用。');
}

async function fetchJSON(url, opt, ms) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms || 45000);
  try {
    const resp = await fetch(url, Object.assign({}, opt, { signal: ctl.signal }));
    const text = await resp.text();
    let data = null;
    try { data = JSON.parse(text); } catch (e) { data = { raw: text }; }
    if (!resp.ok) {
      const msg = (data && data.error && (data.error.message || data.error)) || text.slice(0, 200) || resp.statusText;
      const err = new Error('HTTP ' + resp.status + '：' + msg);
      err.status = resp.status;
      throw err;
    }
    return data;
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('連線逾時（' + Math.round((ms || 45000) / 1000) + ' 秒）');
    if (e instanceof TypeError) throw new Error('網路錯誤或被瀏覽器擋下（CORS）：' + e.message);
    throw e;
  } finally { clearTimeout(timer); }
}

function versionScore(name) { // gemini-3.8-flash → 3.8；越新分數越高
  const m = name.match(/gemini-(\d+)(?:\.(\d+))?/);
  return m ? +m[1] + (m[2] ? +m[2] / 10 : 0) : 0;
}

async function listGeminiModels(key) {
  const data = await fetchJSON(GEMINI_BASE + '/models?pageSize=200', { headers: { 'x-goog-api-key': key } }, 20000);
  return (data.models || [])
    .filter((m) => (m.supportedGenerationMethods || []).includes('generateContent'))
    .map((m) => m.name.replace(/^models\//, ''))
    .filter((n) => n.startsWith('gemini'));
}

function pickFlash(names) {
  const stable = names.filter((n) => /flash/.test(n) && !/lite|preview|exp|image|tts|audio|live|thinking/.test(n));
  const pool = stable.length ? stable : names.filter((n) => /flash/.test(n));
  pool.sort((a, b) => versionScore(b) - versionScore(a));
  return pool[0] || names[0] || DEFAULT_MODEL.gemini;
}

async function testConnection(autoPick) {
  const provider = $('vlmProvider').value;
  const key = $('vlmApiKey').value.trim();
  if (provider === 'server') {
    setVlmStatus('測試後端代理中…');
    try {
      const r = await fetchJSON('/api/vlm-health', {}, 8000);
      setVlmStatus('後端代理可用' + (r.hasKey ? '（伺服器已設定 GEMINI_API_KEY）' : '（伺服器沒設 Key，會改用你填的 Key）'), 'ok');
    } catch (e) { setVlmStatus('後端代理不可用：' + e.message + '。GitHub Pages 沒有後端，請改選「Google Gemini」。', 'err'); }
    return;
  }
  if (!key) { setVlmStatus('沒有填 Key：將使用本地規則模擬 Stage 2。', 'err'); return; }
  if (provider === 'openai') { setVlmStatus('OpenAI Key 已儲存（模型：' + currentModel() + '）。實際送出照片時才會驗證。', 'ok'); return; }
  setVlmStatus('正在向 Google AI Studio 確認這把 Key…');
  try {
    const names = await listGeminiModels(key);
    const dl = $('vlmModelList');
    dl.innerHTML = names.map((n) => `<option value="${n}"></option>`).join('');
    const cur = currentModel();
    if (autoPick || !names.includes(cur)) {
      const pick = pickFlash(names);
      $('vlmModel').value = pick;
      lsSet(LS_MODEL, pick);
    }
    setVlmStatus(`連線成功 ✓ 這把 Key 可用 ${names.length} 個模型，目前使用：${currentModel()}`, 'ok');
    log('Google AI Studio 連線成功，模型：' + currentModel());
  } catch (e) {
    let hint = '';
    if (/400|403|API key|PERMISSION/i.test(e.message)) hint = '（Key 無效、未啟用 Generative Language API，或該專案沒有權限）';
    setVlmStatus('連線失敗：' + e.message + hint, 'err');
    log('[VLM 連線失敗] ' + e.message);
  }
}

$('btnSaveKey').addEventListener('click', async () => {
  const ok = lsSet(LS_KEY, $('vlmApiKey').value.trim()) && lsSet(LS_PROV, $('vlmProvider').value) && lsSet(LS_MODEL, currentModel());
  log(ok ? '已儲存雲端 VLM API 設定（存在這台裝置的瀏覽器）。' : '瀏覽器不允許儲存（無痕模式？），本次仍可使用。');
  await testConnection(false);
});
$('btnListModels').addEventListener('click', () => testConnection(true));
$('vlmProvider').addEventListener('change', () => {
  lsSet(LS_PROV, $('vlmProvider').value);
  $('vlmModel').value = DEFAULT_MODEL[$('vlmProvider').value] || '';
  setVlmStatus('已切換供應商，請按「儲存並測試」。');
});
$('vlmModel').addEventListener('change', () => lsSet(LS_MODEL, currentModel()));

function geminiText(data) {
  const parts = (data && data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts) || [];
  const txt = parts.filter((p) => p.text && !p.thought).map((p) => p.text).join('');
  if (!txt) {
    const why = (data && data.promptFeedback && data.promptFeedback.blockReason) || (data && data.candidates && data.candidates[0] && data.candidates[0].finishReason) || '空白回應';
    throw new Error('模型沒有回傳文字（' + why + '）');
  }
  return txt;
}

function parseLooseJSON(text) {
  const t = String(text).replace(/```json|```/g, '').trim();
  try { return JSON.parse(t); } catch (e) {}
  const a = t.indexOf('{'), b = t.lastIndexOf('}');
  if (a >= 0 && b > a) return JSON.parse(t.slice(a, b + 1));
  throw new Error('回應不是 JSON：' + t.slice(0, 80));
}

// ============================================================================
// 1. 初始化本地 TensorFlow.js Spatial MLP 與 COCO-SSD
// ============================================================================
async function initModels() {
  loadVLMConfig();
  log('正在初始化 Stage 1 本地模型 (TF.js Spatial MLP + COCO-SSD)...');
  try {
    try {
      spatialNet = await tf.loadLayersModel('localstorage://ar-spatial-mlp-v1');
      spatialNet.compile({ optimizer: tf.train.adam(0.015), loss: 'meanSquaredError' });
      log('已載入本地預訓練 Spatial MLP 權重。');
    } catch (e) {
      spatialNet = tf.sequential();
      spatialNet.add(tf.layers.dense({ inputShape: [18], units: 32, activation: 'relu' }));
      spatialNet.add(tf.layers.dense({ units: 24, activation: 'relu' }));
      spatialNet.add(tf.layers.dense({ units: 13, activation: 'sigmoid' }));
      spatialNet.compile({ optimizer: tf.train.adam(0.015), loss: 'meanSquaredError' });
      log('已建立全新 3 層式 Spatial MLP 神經網路。');
    }

    const rawDs = localStorage.getItem('arSpatialDataset.v1');
    if (rawDs) {
      dataset = JSON.parse(rawDs);
      $('stSamples').textContent = dataset.length;
    }

    if (window.cocoSsd) {
      cocoModel = await window.cocoSsd.load({ base: 'lite_mobilenet_v2' });
      log('COCO-SSD 物件偵測模型就緒。');
    }
  } catch (err) {
    log('模型初始化警告: ' + err.message);
  }
  loadSyntheticRoomSample();
}

// ============================================================================
// 2. [STAGE 1] 地牆交界線（材質分割 ＋ 透視）＋ COCO-SSD ＋ MLP 初判
//   舊版：只找「水平邊緣最多的一列」當交界線 → 永遠是一條水平直線，
//         房間轉角、斜看的牆、地板花紋都會讓它整條錯，後面的距離與前中後景也跟著錯。
//   新版：
//     ① 材質：每個像素取 亮度 L、紅綠軸 U、黃藍軸 V、紋理密度 D（附近邊緣多寡）
//        地板樣本＝畫面底部中央；牆面樣本＝地平線附近一帶（眼高 1.5 m，牆一定跨過地平線）
//        → 每個像素比較「較像地板還是較像牆」，從底部往上長出地板（Canny 強邊緣當圍牆）
//     ② 每一欄取地板最上緣 → 原始交界線
//     ③ 透視：Hough 直線兩兩求交點 → 消失點所在的地平線；交界線簡化成折線，
//        每一段吸附到附近方向相同的長直線（踢腳線），且一定在地平線以下
//     ④ 折線頂點：往上凸（離你較遠）＝陰角；往下凸（離你較近）＝陽角
//     ⑤ 距離：眼高 ÷ 地面俯角（以地平線為基準），用來分前景／中景／後景
// ============================================================================
const DEG = Math.PI / 180;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const r3 = (v) => Math.round(v * 1000) / 1000;
let boundary = null; // { pts:[[x,y]] 正規化, horizon, hconf, hauto, source, mask, mw, mh, mat, sep, coverage, wallBand, lines }
const camCfg = { eye: 1.5, hfov: 0 }; // hfov 0 = 依照片方向自動（iPhone 主鏡頭：橫拍 69°、直拍 54°）

// ---------- Canny ----------
function canny(gray, w, h, sens) {
  const n = w * h, tmp = new Float32Array(n), bl = new Float32Array(n), K = [1, 4, 6, 4, 1];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let s = 0; for (let i = -2; i <= 2; i++) s += K[i + 2] * gray[y * w + clamp(x + i, 0, w - 1)]; tmp[y * w + x] = s / 16; }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let s = 0; for (let i = -2; i <= 2; i++) s += K[i + 2] * tmp[clamp(y + i, 0, h - 1) * w + x]; bl[y * w + x] = s / 16; }
  const gx = new Float32Array(n), gy = new Float32Array(n), mag = new Float32Array(n);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const k = y * w + x, a = bl[k - w - 1], b = bl[k - w], c = bl[k - w + 1], d = bl[k - 1], f = bl[k + 1], g = bl[k + w - 1], hh = bl[k + w], i = bl[k + w + 1];
    gx[k] = (c + 2 * f + i - a - 2 * d - g) / 4; gy[k] = (g + 2 * hh + i - a - 2 * b - c) / 4; mag[k] = Math.hypot(gx[k], gy[k]);
  }
  const nms = new Float32Array(n), vals = [];
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const k = y * w + x, m = mag[k]; if (m < 2) continue;
    let ang = Math.atan2(gy[k], gx[k]) * 180 / Math.PI; if (ang < 0) ang += 180;
    let o1, o2;
    if (ang < 22.5 || ang >= 157.5) { o1 = k - 1; o2 = k + 1; } else if (ang < 67.5) { o1 = k - w - 1; o2 = k + w + 1; } else if (ang < 112.5) { o1 = k - w; o2 = k + w; } else { o1 = k - w + 1; o2 = k + w - 1; }
    if (m >= mag[o1] && m >= mag[o2]) { nms[k] = m; vals.push(m); }
  }
  vals.sort((a, b) => a - b);
  const hi = vals.length ? vals[Math.floor(vals.length * (1 - clamp(0.11 * sens, 0.02, 0.5)))] : 1e9, lo = hi * 0.45;
  const edge = new Uint8Array(n), st = [];
  for (let k = 0; k < n; k++) if (nms[k] >= hi) { edge[k] = 1; st.push(k); }
  while (st.length) {
    const k = st.pop(), x = k % w, y = (k / w) | 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      const q = yy * w + xx; if (!edge[q] && nms[q] >= lo) { edge[q] = 1; st.push(q); }
    }
  }
  return { edge, gx, gy, mag, hi };
}

// ---------- Hough 直線（沿梯度方向投票）----------
function houghLines(E, w, h) {
  const nT = 90, off = Math.ceil(Math.hypot(w, h)), nR = 2 * off + 1, acc = new Int32Array(nT * nR), cs = [], sn = [];
  for (let t = 0; t < nT; t++) { cs.push(Math.cos(t * Math.PI / nT)); sn.push(Math.sin(t * Math.PI / nT)); }
  const pts = [];
  for (let k = 0; k < w * h; k++) if (E.edge[k]) pts.push(k);
  for (const k of pts) {
    const x = k % w, y = (k / w) | 0;
    let th = Math.atan2(E.gy[k], E.gx[k]); th = ((th % Math.PI) + Math.PI) % Math.PI;
    const ti = Math.round(th / Math.PI * nT);
    for (let d = -1; d <= 1; d++) { const t = (ti + d + nT * 2) % nT; acc[t * nR + Math.round(x * cs[t] + y * sn[t]) + off]++; }
  }
  const cand = [], minV = Math.max(10, Math.round(w * 0.06));
  for (let t = 0; t < nT; t++) for (let r = 2; r < nR - 2; r++) {
    const v = acc[t * nR + r]; if (v < minV) continue;
    let ok = true;
    for (let dt = -2; dt <= 2 && ok; dt++) for (let dr = -3; dr <= 3; dr++) { if (!dt && !dr) continue; if (acc[((t + dt + nT) % nT) * nR + r + dr] > v) { ok = false; break; } }
    if (ok) cand.push({ t, r: r - off, v });
  }
  cand.sort((a, b) => b.v - a.v);
  const lines = [];
  for (const c of cand) {
    if (lines.length >= 24) break;
    const th = c.t * Math.PI / nT, ct = Math.cos(th), st = Math.sin(th), dx = -st, dy = ct;
    let tmin = 1e9, tmax = -1e9, cnt = 0;
    for (const k of pts) {
      const x = k % w, y = (k / w) | 0;
      if (Math.abs(x * ct + y * st - c.r) > 1.6) continue;
      let g = Math.atan2(E.gy[k], E.gx[k]) - th; g = Math.abs(((g + Math.PI * 1.5) % Math.PI) - Math.PI / 2);
      if (g > 0.22) continue;
      const tt = x * dx + y * dy; tmin = Math.min(tmin, tt); tmax = Math.max(tmax, tt); cnt++;
    }
    if (cnt < 12 || tmax - tmin < w * 0.1) continue;
    if (lines.some((L) => Math.abs(L.th - th) < 0.07 && Math.abs(L.rho - c.r) < 5)) continue;
    const bx = c.r * ct, by = c.r * st;
    lines.push({ th, rho: c.r, len: tmax - tmin, cnt, x1: bx + dx * tmin, y1: by + dy * tmin, x2: bx + dx * tmax, y2: by + dy * tmax, ang: Math.atan2(dy, dx) });
  }
  return lines;
}

// ---------- 地平線：斜線兩兩交點的 y 直方圖最高峰 ----------
function estimateHorizon(lines, w, h) {
  const sl = lines.filter((L) => { let a = Math.abs(L.ang * 180 / Math.PI) % 180; if (a > 90) a = 180 - a; return a > 7 && a < 80; });
  const lo = -1.2 * h, hiY = 2.2 * h, bins = Math.ceil((hiY - lo) / 2), hist = new Float32Array(bins);
  let total = 0, pairs = 0;
  for (let i = 0; i < sl.length; i++) for (let j = i + 1; j < sl.length; j++) {
    const A = sl[i], B = sl[j];
    if (Math.abs(Math.sin(A.ang - B.ang)) < 0.17) continue;
    const a1 = Math.cos(A.th), b1 = Math.sin(A.th), a2 = Math.cos(B.th), b2 = Math.sin(B.th), det = a1 * b2 - a2 * b1;
    if (Math.abs(det) < 1e-6) continue;
    const y = (a1 * B.rho - a2 * A.rho) / det;
    if (y < lo || y > hiY) continue;
    const wgt = Math.sqrt(A.len * B.len); total += wgt; pairs++;
    const bi = Math.floor((y - lo) / 2);
    for (let d = -3; d <= 3; d++) if (bi + d >= 0 && bi + d < bins) hist[bi + d] += wgt * Math.exp(-d * d / 4);
  }
  let best = -1, bv = 0; for (let i = 0; i < bins; i++) if (hist[i] > bv) { bv = hist[i]; best = i; }
  const conf = total > 0 ? clamp(bv / (total * 1.4), 0, 1) : 0;
  if (pairs < 2 || conf < 0.12) return { y: h * 0.48, conf: 0, auto: false };
  return { y: lo + best * 2 + 1, conf, auto: true };
}

function medianFilter(a, valid, win) {
  const out = a.slice(), r = win >> 1;
  for (let i = 0; i < a.length; i++) {
    const v = []; for (let d = -r; d <= r; d++) { const j = i + d; if (j >= 0 && j < a.length && valid[j]) v.push(a[j]); }
    if (v.length) { v.sort((p, q) => p - q); out[i] = v[v.length >> 1]; }
  }
  return out;
}
function rdpPts(pts, eps) {
  if (pts.length < 3) return pts.slice();
  const a = pts[0], b = pts[pts.length - 1], dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy) || 1e-9;
  let md = 0, mi = -1;
  for (let i = 1; i < pts.length - 1; i++) { const d = Math.abs((pts[i][0] - a[0]) * dy - (pts[i][1] - a[1]) * dx) / L; if (d > md) { md = d; mi = i; } }
  if (md > eps) return rdpPts(pts.slice(0, mi + 1), eps).slice(0, -1).concat(rdpPts(pts.slice(mi), eps));
  return [a, b];
}

function robustStats(arrs, idx) { // 中位數 ± MAD
  return arrs.map((A) => {
    const v = idx.map((k) => A[k]).sort((p, q) => p - q);
    const med = v[v.length >> 1];
    const dev = v.map((x) => Math.abs(x - med)).sort((p, q) => p - q);
    return { mu: med, sig: 1.4826 * dev[dev.length >> 1] };
  });
}

function analyzeBoundary(imgEl) {
  const iw = imgEl.naturalWidth || imgEl.width, ih = imgEl.naturalHeight || imgEl.height;
  const w = 200, h = clamp(Math.round(w * ih / iw), 60, 420), n = w * h;
  const off = document.createElement('canvas'); off.width = w; off.height = h;
  const oc = off.getContext('2d', { willReadFrequently: true });
  oc.drawImage(imgEl, 0, 0, w, h);
  const px = oc.getImageData(0, 0, w, h).data;
  const gray = new Float32Array(n), L = new Float32Array(n), U = new Float32Array(n), Vb = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    const r = px[4 * k], g = px[4 * k + 1], b = px[4 * k + 2];
    gray[k] = 0.299 * r + 0.587 * g + 0.114 * b; L[k] = gray[k] / 255; U[k] = (r - g) / 255; Vb[k] = (r + g - 2 * b) / 510;
  }
  const E = canny(gray, w, h, 1.0);
  // 紋理密度：7×7 視窗內的梯度強度平均（積分影像）
  const I = new Float64Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y++) { let row = 0; for (let x = 0; x < w; x++) { row += Math.min(E.mag[y * w + x], 40) / 40; I[(y + 1) * (w + 1) + x + 1] = I[y * (w + 1) + x + 1] + row; } }
  const D = new Float32Array(n), R = 3;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const x0 = Math.max(0, x - R), x1 = Math.min(w, x + R + 1), y0 = Math.max(0, y - R), y1 = Math.min(h, y + R + 1);
    D[y * w + x] = (I[y1 * (w + 1) + x1] - I[y0 * (w + 1) + x1] - I[y1 * (w + 1) + x0] + I[y0 * (w + 1) + x0]) / ((x1 - x0) * (y1 - y0));
  }
  const lines = houghLines(E, w, h);
  const hz = estimateHorizon(lines, w, h);
  const yh = clamp(hz.y, -0.5 * h, h * 0.85);

  // 樣本
  const floorIdx = [], wallIdx = [];
  for (let y = Math.floor(h * 0.88); y < h; y++) for (let x = Math.floor(w * 0.25); x < Math.ceil(w * 0.75); x++) { const k = y * w + x; if (!E.edge[k]) floorIdx.push(k); }
  let wy0, wy1;
  if (yh > h * 0.12) { wy0 = Math.max(1, Math.round(yh - h * 0.2)); wy1 = Math.min(Math.round(h * 0.8), Math.round(yh + h * 0.03)); }
  else { wy0 = 1; wy1 = Math.round(h * 0.15); }
  for (let y = wy0; y < wy1; y++) for (let x = Math.floor(w * 0.08); x < Math.ceil(w * 0.92); x++) { const k = y * w + x; if (!E.edge[k]) wallIdx.push(k); }
  const chans = [L, U, Vb, D], MINS = [0.05, 0.025, 0.025, 0.05], WTS = [1, 1.5, 1.5, 1.2], WS = 5.2;
  const fS = robustStats(chans, floorIdx.length ? floorIdx : [n - 1]);
  const wS = robustStats(chans, wallIdx.length ? wallIdx : [0]);
  const dist = (k, S) => { let s = 0; for (let c = 0; c < 4; c++) { const z = (chans[c][k] - S[c].mu) / Math.max(S[c].sig, MINS[c]); s += WTS[c] * z * z; } return s / WS; };
  const dF = new Float32Array(n), P = new Float32Array(n);
  for (let k = 0; k < n; k++) { dF[k] = dist(k, fS); P[k] = Math.log((dist(k, wS) + 0.5) / (dF[k] + 0.5)); } // >0：較像地板

  // 從底部長出地板
  const mask = new Uint8Array(n), st = [];
  for (const k of floorIdx) if (P[k] > 0) { mask[k] = 1; st.push(k); }
  const ymin = Math.max(0, Math.ceil(yh) + 1);
  while (st.length) {
    const k = st.pop(), x = k % w, y = (k / w) | 0;
    const nb = [x > 0 ? k - 1 : -1, x < w - 1 ? k + 1 : -1, y > ymin ? k - w : -1, y < h - 1 ? k + w : -1];
    for (const q of nb) {
      if (q < 0 || mask[q] || E.edge[q]) continue;
      if (P[q] > -0.15 && dF[q] < 9) { mask[q] = 1; st.push(q); }
    }
  }
  // 每一欄：從底部往上最長的地板段的上緣
  const yb = new Float32Array(w), valid = new Uint8Array(w);
  for (let x = 0; x < w; x++) {
    let best = 0, top = -1, runTop = -1, gap = 0, runBottom = -1;
    for (let y = h - 1; y >= 0; y--) {
      if (mask[y * w + x]) { if (runTop < 0) runBottom = y; runTop = y; gap = 0; const len = runBottom - y + 1; if (len > best && runBottom >= h * 0.55) { best = len; top = y; } }
      else if (runTop >= 0) { if (++gap > 2) { runTop = -1; gap = 0; } }
    }
    if (best >= 3) { yb[x] = top; valid[x] = 1; }
  }
  let cov = 0; for (let x = 0; x < w; x++) cov += valid[x];
  cov /= w;
  let method = 'material';
  if (cov < 0.25) { // 地板幾乎長不出來：退回「水平邊緣最強的一列」
    method = 'fallback-row';
    let bestY = h * 0.68, bs = 0;
    for (let y = Math.max(Math.ceil(yh) + 2, Math.floor(h * 0.35)); y < h * 0.95; y++) {
      let s = 0; for (let x = 1; x < w - 1; x++) { const k = y * w + x; if (E.edge[k] && Math.abs(E.gy[k]) > Math.abs(E.gx[k])) s += E.mag[k]; }
      if (s > bs) { bs = s; bestY = y; }
    }
    for (let x = 0; x < w; x++) { yb[x] = bestY; valid[x] = 1; }
  }
  let ys = medianFilter(Array.from(yb), valid, 9);
  // 沒有地板的欄位用左右插值
  let last = -1;
  for (let x = 0; x < w; x++) {
    if (!valid[x]) continue;
    if (last < 0) for (let q = 0; q < x; q++) ys[q] = ys[x];
    else for (let q = last + 1; q < x; q++) ys[q] = ys[last] + (ys[x] - ys[last]) * (q - last) / (x - last);
    last = x;
  }
  if (last >= 0) for (let q = last + 1; q < w; q++) ys[q] = ys[last];
  ys = ys.map((y) => Math.max(y, ymin));
  // 去除遮擋造成的「窄」跳動（形態學 opening / closing，只看寬度）：
  //   · 人、椅子、地上的東西擋住牆腳 → 交界線局部往下凹（y 變大）→ opening（先取最小再取最大）消掉寬度 < 16% 的凹陷
  //   · 和地板同色的門、櫃子 → 交界線局部往上凸 → closing（先取最大再取最小）消掉寬度 < 10% 的突起
  //   房間真正的轉角是很寬的 V 形，只會被削掉一點尖端，之後由踢腳線吸附與兩線交點補回。
  // 兩端用「點對稱」延伸（保留斜率），避免影像邊緣的斜牆腳被削平
  const runF = (arr, k, fn) => {
    const ext = (q) => (q < 0 ? 2 * arr[0] - arr[Math.min(w - 1, -q)] : q >= w ? 2 * arr[w - 1] - arr[Math.max(0, 2 * (w - 1) - q)] : arr[q]);
    return arr.map((_, x) => { let v = arr[x]; for (let q = x - k; q <= x + k; q++) v = fn(v, ext(q)); return v; });
  };
  const k1 = Math.round(w * 0.08), k2 = Math.round(w * 0.05);
  const raw = ys.slice();
  ys = runF(runF(ys, k1, Math.min), k1, Math.max);   // opening：去凹陷（往下的）
  ys = runF(runF(ys, k2, Math.max), k2, Math.min);   // closing：去突起（往上的）
  ys = medianFilter(ys, new Uint8Array(w).fill(1), 5);
  let occl = 0; for (let x = 0; x < w; x++) if (Math.abs(raw[x] - ys[x]) > h * 0.02) occl++;

  // 透視：折線化並吸附到踢腳線
  let poly = rdpPts(ys.map((y, x) => [x, y]), Math.max(1.5, h * 0.02));
  const segLine = [];
  for (let i = 0; i + 1 < poly.length; i++) {
    const A = poly[i], B = poly[i + 1], sa = Math.atan2(B[1] - A[1], B[0] - A[0]);
    let best = null, bd = Math.max(3, h * 0.022);
    for (const Ln of lines) {
      let da = Math.abs(Ln.ang - sa) % Math.PI; da = Math.min(da, Math.PI - da);
      if (da > 7 * DEG) continue;
      const ct = Math.cos(Ln.th), stt = Math.sin(Ln.th);
      const d = (Math.abs(A[0] * ct + A[1] * stt - Ln.rho) + Math.abs(B[0] * ct + B[1] * stt - Ln.rho)) / 2;
      if (d < bd) { bd = d; best = Ln; }
    }
    segLine.push(best);
  }
  const yOn = (Ln, x) => { const ct = Math.cos(Ln.th), stt = Math.sin(Ln.th); return Math.abs(stt) > 1e-3 ? (Ln.rho - x * ct) / stt : null; };
  poly = poly.map((p) => p.slice());
  let snapped = 0;
  for (let i = 0; i < segLine.length; i++) {
    const Ln = segLine[i]; if (!Ln) continue;
    snapped++;
    for (const j of [i, i + 1]) { const y = yOn(Ln, poly[j][0]); if (y != null && Math.abs(y - poly[j][1]) < h * 0.05) poly[j][1] = (poly[j][1] + y * 2) / 3; }
  }
  for (let i = 1; i + 1 < poly.length; i++) { // 相鄰兩段都吸附 → 頂點改成兩直線交點
    const A = segLine[i - 1], B = segLine[i];
    if (!A || !B) continue;
    const a1 = Math.cos(A.th), b1 = Math.sin(A.th), a2 = Math.cos(B.th), b2 = Math.sin(B.th), det = a1 * b2 - a2 * b1;
    if (Math.abs(det) < 1e-6) continue;
    const x = (A.rho * b2 - B.rho * b1) / det, y = (a1 * B.rho - a2 * A.rho) / det;
    if (Math.hypot(x - poly[i][0], y - poly[i][1]) < w * 0.06) poly[i] = [x, y];
  }
  poly[0][0] = 0; poly[poly.length - 1][0] = w - 1;
  for (const p of poly) p[1] = clamp(Math.max(p[1], ymin), 0, h - 1);

  let horizon = yh / h, hsrc = hz.auto ? 'vp' : 'none', hconf = hz.conf;
  if (!hz.auto) {
    const hy = horizonFromLine(poly.map((p) => [p[0] / (w - 1), p[1] / h]));
    if (hy != null) { horizon = hy; hsrc = 'floorline'; hconf = 0.5; }
  }
  // 材質摘要
  const meanRGB = (idx) => { let r = 0, g = 0, b = 0; for (const k of idx) { r += px[4 * k]; g += px[4 * k + 1]; b += px[4 * k + 2]; } const m = Math.max(1, idx.length); return [Math.round(r / m), Math.round(g / m), Math.round(b / m)]; };
  let fOk = 0, wOk = 0;
  for (const k of floorIdx) if (P[k] > 0) fOk++;
  for (const k of wallIdx) if (P[k] < 0) wOk++;
  const sep = 0.5 * (fOk / Math.max(1, floorIdx.length) + wOk / Math.max(1, wallIdx.length));
  return {
    pts: poly.map((p) => [r3(p[0] / (w - 1)), r3(p[1] / h)]), horizon: r3(horizon), hconf, hauto: hsrc !== 'none', hsrc, source: 'local', method, coverage: cov,
    mask, mw: w, mh: h, wallBand: [wy0 / h, wy1 / h], snapped, segs: poly.length - 1, occluded: r3(occl / w),
    mat: { floor: { rgb: meanRGB(floorIdx), dens: r3(fS[3].mu), L: r3(fS[0].mu) }, wall: { rgb: meanRGB(wallIdx), dens: r3(wS[3].mu), L: r3(wS[0].mu) } }, sep: r3(sep),
    lines: lines.map((Ln) => [r3(Ln.x1 / w), r3(Ln.y1 / h), r3(Ln.x2 / w), r3(Ln.y2 / h)]),
  };
}

// ---------- 交界線工具 ----------
function boundaryAt(xn) {
  if (!boundary || !boundary.pts.length) return 0.68;
  const p = boundary.pts;
  if (xn <= p[0][0]) return p[0][1];
  for (let i = 0; i + 1 < p.length; i++) if (xn <= p[i + 1][0]) { const t = (xn - p[i][0]) / Math.max(1e-6, p[i + 1][0] - p[i][0]); return p[i][1] + (p[i + 1][1] - p[i][1]) * t; }
  return p[p.length - 1][1];
}
function meanBoundaryY() { let s = 0; for (let i = 0; i <= 20; i++) s += boundaryAt(i / 20); return s / 21; }
function camF() { // 焦距（以照片高度為 1 的單位）
  const iw = currentImg ? (currentImg.naturalWidth || currentImg.width) : 4, ih = currentImg ? (currentImg.naturalHeight || currentImg.height) : 3;
  const hfov = (camCfg.hfov || (ih > iw ? 54 : 69)) * DEG;
  return ((iw / ih) / 2) / Math.tan(hfov / 2);
}
// 地面上一點（照片 y 座標）距離相機多遠（公尺，水平）
function distAtY(yn) {
  const f = camF(), yh = boundary ? boundary.horizon : 0.48;
  const v = (yn - 0.5) / f, p = Math.atan((yh - 0.5) / f);
  const dy = v * Math.cos(p) - Math.sin(p), dz = v * Math.sin(p) + Math.cos(p);
  if (dy <= 1e-3) return 25;
  return clamp(camCfg.eye * dz / dy, 0.3, 25);
}
// 牆上 2.8 m 高的位置在照片上的 y（用來決定牆、轉角框的頂端）
function wallTopY(yBottom) { const yh = boundary ? boundary.horizon : 0.48; return clamp(yh - (yBottom - yh) * (2.8 - camCfg.eye) / camCfg.eye, 0, 1); }
function layerOfDist(d) { return d < 2 ? 'foreground' : d < 5 ? 'midground' : 'background'; }

// ---------- 元素（多邊形）----------
// 距離：用物件「接觸地面」的位置（多邊形最低點）；最低點若比牆腳還高，代表它在牆上 → 用牆腳距離
function newId(pfx) { return pfx + '_' + Math.random().toString(36).substring(2, 6).toUpperCase(); }
function syncBox(el) {
  let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
  for (const p of el.pts) { x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]); }
  el.x = r3(x0); el.y = r3(y0); el.w = r3(Math.max(0.005, x1 - x0)); el.h = r3(Math.max(0.005, y1 - y0));
  return el;
}
function rectPts(x, y, w, h) { return [[r3(x), r3(y)], [r3(x + w), r3(y)], [r3(x + w), r3(y + h)], [r3(x), r3(y + h)]]; }
function mkEl(pts, type, opt) {
  opt = opt || {};
  const el = syncBox({ id: newId(opt.pfx || 'EL'), pts: pts.map((p) => [r3(clamp(p[0], 0, 1)), r3(clamp(p[1], 0, 1))]), type,
    action: opt.action || (TYPE_LABELS[type] ? TYPE_LABELS[type].defaultAction : 'detour_avoid'), source: opt.source || 'local', tag: opt.tag || '' });
  el.dist = opt.dist != null ? +(+opt.dist).toFixed(1) : +distAtY(Math.max(boundaryAt(el.x + el.w / 2), el.y + el.h)).toFixed(1);
  el.layer = opt.layer || layerOfDist(el.dist);
  el.autoLayer = !opt.layer;
  return el;
}
// 交界線的頂點 → 轉角
function boundaryCorners() {
  const out = [];
  if (!boundary) return out;
  const p = boundary.pts, asp = currentImg ? (currentImg.naturalWidth || currentImg.width) / (currentImg.naturalHeight || currentImg.height) : 1.5;
  for (let i = 1; i + 1 < p.length; i++) {
    const a = Math.atan2(p[i][1] - p[i - 1][1], (p[i][0] - p[i - 1][0]) * asp), b = Math.atan2(p[i + 1][1] - p[i][1], (p[i + 1][0] - p[i][0]) * asp);
    const turn = b - a;
    if (Math.abs(turn) < 12 * DEG) continue;
    if (p[i][0] - p[i - 1][0] < 0.06 || p[i + 1][0] - p[i][0] < 0.06) continue; // 太短的折段多半是遮擋，不算轉角
    const concave = turn > 0; // 先往上再往下＝頂點離你最遠＝陰角（房間內角）
    const x = p[i][0], yb = p[i][1], top = wallTopY(yb), hw = 0.025;
    out.push(mkEl([[x - hw, top], [x + hw, top], [x + hw, yb], [x - hw, yb]], concave ? 'corner_concave' : 'corner_convex', { tag: 'bnd', dist: distAtY(yb) }));
  }
  return out;
}


// 透視備援：左右兩側的牆腳線在真實世界是平行的地面直線 → 在照片上的交點就是消失點，它的高度＝地平線
function horizonFromLine(pts) {
  let best = null, bw = 0;
  for (let i = 0; i + 1 < pts.length; i++) for (let j = i + 1; j + 1 < pts.length; j++) {
    const A = [pts[i], pts[i + 1]], B = [pts[j], pts[j + 1]];
    const sa = (A[1][1] - A[0][1]) / Math.max(1e-6, A[1][0] - A[0][0]), sb = (B[1][1] - B[0][1]) / Math.max(1e-6, B[1][0] - B[0][0]);
    if (!(sa < -0.05 && sb > 0.05)) continue; // 左段往中間升高、右段往中間升高
    const x = (B[0][1] - A[0][1] + sa * A[0][0] - sb * B[0][0]) / (sa - sb), y = A[0][1] + sa * (x - A[0][0]);
    const minY = Math.min(...pts.map((p) => p[1]));
    if (!(y < minY - 0.005) || y < -0.5) continue;
    const wgt = (A[1][0] - A[0][0]) * (B[1][0] - B[0][0]);
    if (wgt > bw) { bw = wgt; best = y; }
  }
  return best;
}

function extractImageGeometryAndLines(imgEl) {
  const W = 160, H = 104;
  const off = document.createElement('canvas');
  off.width = W; off.height = H;
  const octx = off.getContext('2d');
  octx.drawImage(imgEl, 0, 0, W, H);
  const px = octx.getImageData(0, 0, W, H).data;
  const gray = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) gray[i] = 0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2];
  let edgeL = 0, edgeC = 0, edgeR = 0, vertEdges = 0, horizEdges = 0, lowerCenterEdge = 0, upperCenterEdge = 0;
  const colVertScore = new Float32Array(W);
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const k = y * W + x;
    const dx = (-gray[k - W - 1] + gray[k - W + 1] - 2 * gray[k - 1] + 2 * gray[k + 1] - gray[k + W - 1] + gray[k + W + 1]) / 4;
    const dy = (-gray[k - W - 1] - 2 * gray[k - W] - gray[k - W + 1] + gray[k + W - 1] + 2 * gray[k + W] + gray[k + W + 1]) / 4;
    const m = Math.hypot(dx, dy);
    if (m <= 14) continue;
    if (x < W * 0.33) edgeL++; else if (x < W * 0.66) edgeC++; else edgeR++;
    if (Math.abs(dx) > Math.abs(dy) * 1.5) { vertEdges++; colVertScore[x] += m; } else if (Math.abs(dy) > Math.abs(dx) * 1.5) horizEdges++;
    if (y > H * 0.58 && x > W * 0.28 && x < W * 0.72) lowerCenterEdge++;
    if (y <= H * 0.58 && x > W * 0.28 && x < W * 0.72) upperCenterEdge++;
  }
  const vPeaks = [];
  for (let x = 10; x < W - 10; x++) {
    const score = colVertScore[x - 1] + colVertScore[x] + colVertScore[x + 1];
    if (score <= H * 18) continue;
    let isMax = true;
    for (let d = -6; d <= 6; d++) if (colVertScore[x + d] > colVertScore[x]) { isMax = false; break; }
    if (isMax) vPeaks.push({ xNorm: x / W, score });
  }
  vPeaks.sort((a, b) => b.score - a.score);
  return {
    edgeLNorm: edgeL / (W * H * 0.33), edgeCNorm: edgeC / (W * H * 0.34), edgeRNorm: edgeR / (W * H * 0.33),
    vertRatio: vertEdges / Math.max(1, vertEdges + horizEdges), lowerCenterTexture: lowerCenterEdge / (W * H * 0.2),
    upperCenterTexture: upperCenterEdge / (W * H * 0.2), vPeaks: vPeaks.slice(0, 4),
  };
}

function computeFeatVec(geo, fgObstacleArea, centerBlockedByObj, doorDetected) {
  const fb = meanBoundaryY();
  return [
    fb, geo.edgeLNorm, geo.edgeCNorm, geo.edgeRNorm, geo.vertRatio, geo.lowerCenterTexture, geo.upperCenterTexture,
    fgObstacleArea, centerBlockedByObj, doorDetected, geo.vPeaks.length / 4,
    elements.filter((e) => e.type.startsWith('corner')).length / 3,
    elements.filter((e) => e.type.startsWith('obstacle')).length / 4,
    elements.filter((e) => e.layer === 'foreground').length / 4,
    elements.filter((e) => e.layer === 'midground').length / 4,
    elements.filter((e) => e.layer === 'background').length / 4,
    geo.edgeLNorm > geo.edgeRNorm ? 1 : 0,
    fb > 0.74 ? 1 : 0,
  ];
}
let lastGeo = null, lastCoco = [];

// opts.keepBoundary：沿用人工修正過的交界線，只重算後續判斷
async function runStage1LocalAnalysis(opts) {
  opts = opts && opts.keepBoundary !== undefined ? opts : {};
  if (!currentImg) return;
  setPipelineStage(1);
  $('vlmReasoningBox').style.display = 'none';
  const keep = opts.keepBoundary && boundary && boundary.source === 'user';
  if (!keep) boundary = analyzeBoundary(currentImg);
  const geo = (lastGeo = extractImageGeometryAndLines(currentImg));
  const userEls = elements.filter((e) => e.source === 'user');
  elements = [];

  let cocoPreds = [];
  if (cocoModel) { try { cocoPreds = await cocoModel.detect(currentImg); } catch (e) {} }
  lastCoco = cocoPreds;
  let fgObstacleArea = 0, centerBlockedByObj = 0, doorDetected = 0;
  const iw = currentImg.naturalWidth || currentImg.width, ih = currentImg.naturalHeight || currentImg.height;
  for (const p of cocoPreds) {
    const [bx, by, bw, bh] = p.bbox;
    const x = bx / iw, y = by / ih, w = bw / iw, h = bh / ih, bottomY = y + h;
    let type = 'obstacle_block';
    if (['couch', 'chair', 'bed', 'potted plant', 'bench'].includes(p.class)) type = 'obstacle_cover';
    else if (p.class === 'door') { type = 'door_open'; doorDetected = 1; }
    else if (p.class === 'person') continue;
    const el = mkEl(rectPts(x, y, w, h), type, { dist: distAtY(bottomY) });
    if (el.layer !== 'background') { fgObstacleArea += w * h; if (x + w > 0.35 && x < 0.65 && bottomY > 0.62) centerBlockedByObj = 1; }
    elements.push(el);
  }
  // 門：兩條相距適中的垂直線，底部落在交界線上
  if (geo.vPeaks.length >= 2) {
    const a = geo.vPeaks[0].xNorm, b = geo.vPeaks[1].xNorm, gap = Math.abs(a - b);
    if (gap > 0.08 && gap < 0.3) {
      const x1 = Math.min(a, b), x2 = Math.max(a, b), yb1 = boundaryAt(x1), yb2 = boundaryAt(x2);
      const top = wallTopY((yb1 + yb2) / 2) + ((yb1 + yb2) / 2 - wallTopY((yb1 + yb2) / 2)) * (0.7 / 2.8);
      elements.push(mkEl([[x1, top], [x2, top], [x2, yb2], [x1, yb1]], 'door_open', {}));
      doorDetected = 1;
    }
  }
  // 轉角：由交界線折點判斷陰角／陽角
  const corners = boundaryCorners();
  elements.push(...corners);
  // 死路：正前方的交界線近而平
  const yc = boundaryAt(0.5), slope = Math.abs(boundaryAt(0.6) - boundaryAt(0.4)) / 0.2;
  if (distAtY(yc) < 2.4 && slope < 0.15 && !doorDetected) {
    const x0 = 0.25, x1 = 0.75, top = wallTopY(yc);
    elements.push(mkEl([[x0, top], [x1, top], [x1, boundaryAt(x1)], [x0, boundaryAt(x0)]], 'wall_deadend', { tag: 'bnd' }));
  }
  elements.push(...userEls);
  currentFeatVec = computeFeatVec(geo, fgObstacleArea, centerBlockedByObj, doorDetected);
  runLocalMLPInference();
  renderCanvas(); renderElementList(); renderBoundaryPanel();
  log(`[Stage 1 完成] 地牆交界線：${boundary.source === 'user' ? '沿用人工修正' : boundary.method === 'material' ? '材質分割＋透視' : '退回水平邊緣法'}（${boundary.pts.length} 點、地平線 ${(boundary.horizon * 100).toFixed(0)}%），標出 ${elements.length} 項空間特徵。`);
  if ($('chkAutoVLM').checked && !opts.noVLM) await runStage2CloudVLM();
  else setPipelineStage(3);
}

// 交界線被修改後：重算距離、前中後景、轉角，不重跑模型
function refreshDerived() {
  if (!boundary) return;
  if (boundary.hsrc !== 'vp' && boundary.hsrc !== 'user') {
    const hy = horizonFromLine(boundary.pts);
    if (hy != null) { boundary.horizon = r3(hy); boundary.hsrc = 'floorline'; boundary.hauto = true; }
  }
  elements = elements.filter((e) => !(e.tag === 'bnd' && e.source !== 'user'));
  for (const e of elements) {
    if (e.source === 'user' && !e.autoLayer) { e.dist = +distAtY(Math.max(boundaryAt(e.x + e.w / 2), e.y + e.h)).toFixed(1); continue; }
    if (e.type.startsWith('corner') || e.type === 'door_open' || e.type === 'wall_deadend') {
      const yb = boundaryAt(e.x + e.w / 2);
      e.pts = e.pts.map((p) => [p[0], p[1] > e.y + e.h * 0.6 ? r3(boundaryAt(p[0])) : p[1]]); syncBox(e);
      e.dist = +distAtY(yb).toFixed(1);
    } else e.dist = +distAtY(e.y + e.h).toFixed(1);
    if (e.autoLayer !== false) e.layer = layerOfDist(e.dist);
  }
  elements.push(...boundaryCorners());
  if (lastGeo) {
    const fg = elements.filter((e) => e.type.startsWith('obstacle') && e.layer !== 'background').reduce((s, e) => s + e.w * e.h, 0);
    currentFeatVec = computeFeatVec(lastGeo, fg, currentFeatVec[8], currentFeatVec[9]);
  }
  renderCanvas(); renderElementList(); renderBoundaryPanel();
}

function renderBoundaryPanel() {
  const el = $('bndInfo'); if (!el) return;
  if (!boundary) { el.innerHTML = '尚未分析'; return; }
  const b = boundary, sw = (c) => `<span style="display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:middle;background:rgb(${c[0]},${c[1]},${c[2]});border:1px solid #444"></span>`;
  const src = { local: '本地（材質＋透視）', vlm: '雲端 VLM', user: '人工修正（已鎖定）' }[b.source] || b.source;
  el.innerHTML = `
    <div>來源：<b style="color:${b.source === 'user' ? 'var(--warn)' : b.source === 'vlm' ? 'var(--vlm)' : 'var(--accent)'}">${src}</b>　${b.pts.length} 個控制點</div>
    ${b.mat ? `<div>地板材質 ${sw(b.mat.floor.rgb)} 亮度 ${b.mat.floor.L} · 紋理密度 ${b.mat.floor.dens}</div>
    <div>牆面材質 ${sw(b.mat.wall.rgb)} 亮度 ${b.mat.wall.L} · 紋理密度 ${b.mat.wall.dens}</div>
    <div>材質區分度 <b>${Math.round(b.sep * 100)}%</b>${b.sep < 0.7 ? ' <span style="color:var(--warn)">（地板和牆很像，交界線可能不準，請檢查）</span>' : ''}　地板覆蓋 ${Math.round(b.coverage * 100)}% 欄位${b.method !== 'material' ? '（退回水平邊緣法）' : ''}</div>
    ${b.occluded > 0.02 ? `<div>被物體遮住而補平的牆腳：${Math.round(b.occluded * 100)}% 寬度（人、桌椅、門）</div>` : ''}
    <div>透視：地平線 ${(b.horizon * 100).toFixed(0)}%（${b.hsrc === 'user' ? '人工拖曳' : b.hsrc === 'floorline' ? '由左右牆腳線交點推得' : b.hauto ? '消失點信心 ' + Math.round(b.hconf * 100) + '%' : '<span style="color:var(--warn)">找不到消失點，假設相機水平——距離可能不準，可用 L 工具拖曳地平線</span>'}）　踢腳線吸附 ${b.snapped}/${b.segs} 段</div>` : ''}
    <div>正前方牆腳距離約 <b>${distAtY(boundaryAt(0.5)).toFixed(1)} m</b>（眼高 ${camCfg.eye} m）</div>`;
}

function runLocalMLPInference() {
  if (!spatialNet) return;
  tf.tidy(() => {
    const input = tf.tensor2d([currentFeatVec]);
    const pred = spatialNet.predict(input).dataSync();

    if (dataset.length >= 2) {
      let bestTypo = 0, bestPath = 0, bestSupp = 0;
      for (let i = 0; i < 5; i++) if (pred[i] > pred[bestTypo]) bestTypo = i;
      for (let i = 0; i < 4; i++) if (pred[5 + i] > pred[5 + bestPath]) bestPath = i;
      for (let i = 0; i < 4; i++) if (pred[9 + i] > pred[9 + bestSupp]) bestSupp = i;

      $('selTypology').value = TYPOLOGY_KEYS[bestTypo];
      $('selPathC').value = PATHC_KEYS[bestPath];
      $('selSupplement').value = SUPP_KEYS[bestSupp];
      const conf = Math.round(Math.max(pred[bestTypo], pred[5 + bestPath], pred[9 + bestSupp]) * 100);
      $('stAcc').textContent = `Stage1 MLP (${conf}%)`;
      $('aiPredBadge').textContent = `Stage 1 本地 MLP 初判 (${conf}%)`;
    } else {
      const hasBlock = elements.some((e) => e.type.startsWith('obstacle') && e.layer !== 'background');
      const hasWall = elements.some((e) => e.type === 'wall_deadend');
      const hasDoor = elements.some((e) => e.type === 'door_open');
      const hasCorner = elements.some((e) => e.type.startsWith('corner'));

      $('selTypology').value = hasWall ? 'dead_end' : hasBlock ? 'cluttered_room' : hasCorner ? 'corner_space' : 'corridor';
      $('selPathC').value = hasBlock ? 'blocked_obj' : hasWall ? 'blocked_wall' : hasDoor ? 'door_way' : 'clear';
      $('selSupplement').value = hasWall ? 'room' : hasBlock ? 'cover' : hasCorner ? 'alley' : 'wall';
      $('stAcc').textContent = 'Stage1 幾何初判';
      $('aiPredBadge').textContent = 'Stage 1 本地幾何初判';
    }
  });
}

// ============================================================================
// 3. [STAGE 2] 雲端 VLM 空間判讀 API 自動銜接與結構化更新
//    將照片 Base64 + Stage 1 初步判讀結果 + 歷史 Few-Shot 修正範例送給雲端 VLM，
//    回傳嚴格的 JSON 格式並無縫更新畫布框選與空間層次表單
// ============================================================================
function getImageBase64Jpeg(imgEl, maxDim = 640) {
  const off = document.createElement('canvas');
  const scale = Math.min(1, maxDim / Math.max(imgEl.width || 640, imgEl.height || 480));
  off.width = Math.round((imgEl.width || 640) * scale);
  off.height = Math.round((imgEl.height || 480) * scale);
  const octx = off.getContext('2d');
  octx.drawImage(imgEl, 0, 0, off.width, off.height);
  return off.toDataURL('image/jpeg', 0.82).split(',')[1];
}

function buildVLMPrompt() {
  const fewShotSummary = dataset.slice(-2).map((d, i) =>
    `範例${i + 1}: 當畫面特徵具有 ${d.elements.map((e) => e.type + '(' + e.layer + ')').join(', ')} 時，正確空間型態為 ${d.labels.typology}，中央動線為 ${d.labels.pathC}，應補充 ${d.labels.supplement}。`
  ).join('\n');
  const simplify = (pts) => { if (pts.length <= 10) return pts; const k = pts.length / 10; return Array.from({ length: 10 }, (_, i) => pts[Math.floor(i * k)]); };
  const userLine = boundary && boundary.source === 'user';
  const stage1Summary = JSON.stringify({
    typology: $('selTypology').value, pathC: $('selPathC').value, supplement: $('selSupplement').value,
    floorLine: boundary ? boundary.pts : null, horizonY: boundary ? boundary.horizon : null,
    elements: elements.map((e) => ({ type: e.type, layer: e.layer, pts: simplify(e.pts), dist: e.dist, by: e.source })),
  });
  return `你是一個專業的建築空間與 AR 戰術拓撲判讀 AI。
設計原則是「實體優先、虛擬補充」：先精準判讀真實空間中的地牆交界、轉角、門扇、牆壁死路與阻擋家具，再決定在不衝突的位置補充實心虛擬建築。

座標：照片左上角 (0,0)、右下角 (1,1)。
floorLine 是「地板與牆面交界線」由左到右的折線（每點 [x,y]），所有距離與前中後景都由它推算，非常重要。
${userLine ? '【這條 floorLine 已經由人工確認，請完全照用、不要修改，回傳時也原樣回傳】' : '請檢查 floorLine：它應該沿著牆腳（踢腳線）走，在房間轉角處轉折；若有錯，請回傳修正後的 floorLine（2～12 點，x 由 0 到 1）。'}
by = "user" 的元素是人工描邊，請保留不要刪除。

以下是【Stage 1 本地判讀】：
${stage1Summary}

${fewShotSummary ? '以下是【使用者過去人工修正的判讀經驗】：\n' + fewShotSummary : ''}

請只回傳純 JSON（不要 markdown）：
{
  "floorLine": [[0,0.7],[0.3,0.62],[1,0.7]],
  "typology": "corridor" | "corner_space" | "cluttered_room" | "dead_end" | "open_hall",
  "supplement": "room" | "alley" | "cover" | "wall",
  "pathL": "clear" | "blocked" | "corner_exit",
  "pathC": "clear" | "blocked_obj" | "blocked_wall" | "door_way",
  "pathR": "clear" | "blocked" | "corner_exit",
  "maxDepth": 數字(公尺, 1.5~20),
  "reasoning": "用繁體中文簡述空間層次、哪裡被擋住、哪裡有轉角或門，以及為何建議補充該虛擬建築(60字內)",
  "elements": [
    { "type": "corner_convex" | "corner_concave" | "door_open" | "wall_deadend" | "obstacle_block" | "obstacle_cover",
      "layer": "foreground" | "midground" | "background",
      "action": "attach_room" | "attach_wall" | "use_as_cover" | "detour_avoid" | "stop_prioritize_real" | "pass_through",
      "pts": [[x,y],[x,y],[x,y]] (沿物體外緣描出的多邊形，3～10 點),
      "dist": 估計距離公尺 }
  ]
}`;
}

async function callCloudVLMAPI(base64Jpeg, promptText) {
  const provider = $('vlmProvider').value;
  const apiKey = $('vlmApiKey').value.trim();
  const model = currentModel();

  // 1. 後端 Server 代理 (/api/vlm-analyze)：Key 放在伺服器環境變數 GEMINI_API_KEY 最安全
  if (provider === 'server') {
    return await fetchJSON('/api/vlm-analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageBase64: base64Jpeg, prompt: promptText, apiKey, model }),
    }, 60000);
  }

  // 沒填 Key：使用內建規則模擬 Stage 2
  if (!apiKey) {
    await new Promise((r) => setTimeout(r, 400));
    return simulateCloudVLMRefinement();
  }

  // 2. 瀏覽器直接呼叫 Google Gemini（GitHub Pages 純靜態也能用）
  if (provider === 'gemini') {
    const call = async (m) => fetchJSON(`${GEMINI_BASE}/models/${encodeURIComponent(m)}:generateContent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-goog-api-key': apiKey },
      body: JSON.stringify({
        contents: [{ role: 'user', parts: [{ text: promptText }, { inline_data: { mime_type: 'image/jpeg', data: base64Jpeg } }] }],
        generationConfig: { responseMimeType: 'application/json', temperature: 0.2 },
      }),
    }, 60000);
    const isBusy = (e) => e.status === 503 || e.status === 500 || /high demand|overloaded|UNAVAILABLE|try again later/i.test(e.message);
    const isQuota = (e) => e.status === 429 || /quota|RESOURCE_EXHAUSTED|rate limit/i.test(e.message);
    const isMissing = (e) => e.status === 404 || e.status === 403 || /not found|not supported|no longer available/i.test(e.message);
    const wait = (ms) => new Promise((r) => setTimeout(r, ms));
    // 同一個模型：塞車時等 2、4 秒各重試一次
    const tryModel = async (m, retries) => {
      for (let k = 0; ; k++) {
        try { return await call(m); }
        catch (e) {
          if (!(isBusy(e) || (isQuota(e) && /minute|per.?min|rate/i.test(e.message))) || k >= retries) throw e;
          const ms = 2000 * Math.pow(2, k);
          setVlmStatus(`${m} 目前塞車，${ms / 1000} 秒後重試（${k + 1}/${retries}）…`);
          log(`[Stage 2] ${m} 塞車（${e.status || ''}），${ms / 1000} 秒後重試`);
          await wait(ms);
        }
      }
    };
    let data = null, used = model, lastErr = null;
    try { data = await tryModel(model, 2); }
    catch (e) { lastErr = e; }
    if (!data && lastErr && (isBusy(lastErr) || isQuota(lastErr) || isMissing(lastErr))) {
      // 換模型：同一把 Key 可用的其他 Flash（含 Flash-Lite、舊一版），各試一次
      let names = [];
      try { names = await listGeminiModels(apiKey); } catch (e) {}
      const alts = names.filter((n) => n !== model && /flash/.test(n) && !/image|tts|audio|live|embedding/.test(n))
        .sort((x, y) => (/lite/.test(x) - /lite/.test(y)) || versionScore(y) - versionScore(x))
        .slice(0, 4);
      for (const alt of alts) {
        setVlmStatus(`${model} 暫時無法使用，改用 ${alt}…`);
        try { data = await tryModel(alt, 1); used = alt; break; }
        catch (e) { lastErr = e; }
      }
    }
    if (!data) {
      const e = lastErr || new Error('沒有回應');
      if (isBusy(e)) throw new Error('Google 的模型目前全部塞車（503），Key 是正常的，請過幾分鐘再按「立即觸發 Stage 2」。');
      if (isQuota(e)) throw new Error('這把 Key 的免費額度用完了（429）。等額度重置（通常隔天），或到 AI Studio 開啟付費方案。');
      throw e;
    }
    if (used !== model) log(`本次改用 ${used} 完成（${model} 塞車）`);
    setVlmStatus(`Stage 2 完成（模型：${used}）`, 'ok');
    return parseLooseJSON(geminiText(data));
  }

  // 3. OpenAI
  if (provider === 'openai') {
    const data = await fetchJSON('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({
        model,
        response_format: { type: 'json_object' },
        messages: [{ role: 'user', content: [
          { type: 'text', text: promptText },
          { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${base64Jpeg}` } },
        ] }],
      }),
    }, 60000);
    return parseLooseJSON(data.choices[0].message.content);
  }
  throw new Error('未知的供應商：' + provider);
}

// 當使用者尚未輸入 API Key 時，提供高擬真的深層拓撲推論模擬，確保在 GitHub Pages 上免 Key 也能完整測試三階段流程
function simulateCloudVLMRefinement() {
  const refined = elements.map((e) => Object.assign({}, e, { source: 'vlm' }));
  const hasDesk = refined.some((e) => e.type === 'obstacle_block' && e.layer === 'foreground');
  const hasDoor = refined.some((e) => e.type === 'door_open');

  return {
    typology: hasDesk ? 'cluttered_room' : hasDoor ? 'corridor' : $('selTypology').value,
    supplement: hasDesk ? 'cover' : 'room',
    pathL: 'clear',
    pathC: hasDesk ? 'blocked_obj' : hasDoor ? 'door_way' : $('selPathC').value,
    pathR: hasDesk ? 'blocked' : 'clear',
    maxDepth: +distAtY(boundary ? Math.min(...boundary.pts.map((p) => p[1])) : 0.6).toFixed(1),
    reasoning: hasDesk
      ? '【VLM 空間層次複判】前景中央偏右有一張實心桌體阻擋直行動線（需從左側繞行）；中景兩側具備內凹陰角且後方有門扇開口，建議依「真實優先、虛擬補充」於空曠側補充 1.1m 實心掩體。'
      : '【VLM 空間層次複判】已確認地牆交界透視深度與兩側轉角位置，建議以中景真實牆角為錨點補充 2.4m 正交實心隔間。',
    elements: refined,
  };
}

let vlmRunning = false;
async function runStage2CloudVLM() {
  if (!currentImg) return;
  if (vlmRunning) { log('Stage 2 已在執行中，請稍候'); return; }
  vlmRunning = true;
  try { await runStage2Inner(); } finally { vlmRunning = false; }
}
async function runStage2Inner() {
  setPipelineStage(2);
  $('modelStatus').textContent = 'VLM Analyzing...';
  $('aiPredBadge').textContent = 'Stage 2 雲端 VLM 深度判讀中...';
  log('[Stage 2 啟動] 正在將畫面與 Stage 1 初判資料傳送至雲端 VLM 進行空間層次複判...');

  try {
    const b64 = getImageBase64Jpeg(currentImg);
    const prompt = buildVLMPrompt();
    const vlmResult = await callCloudVLMAPI(b64, prompt);

    if (vlmResult) {
      if (TYPOLOGY_KEYS.includes(vlmResult.typology)) $('selTypology').value = vlmResult.typology;
      if (SUPP_KEYS.includes(vlmResult.supplement)) $('selSupplement').value = vlmResult.supplement;
      if (['clear', 'blocked', 'corner_exit'].includes(vlmResult.pathL)) $('selPathL').value = vlmResult.pathL;
      if (PATHC_KEYS.includes(vlmResult.pathC)) $('selPathC').value = vlmResult.pathC;
      if (['clear', 'blocked', 'corner_exit'].includes(vlmResult.pathR)) $('selPathR').value = vlmResult.pathR;
      if (vlmResult.maxDepth) $('inpDepth').value = vlmResult.maxDepth;

      // 地牆交界線：人工修正過的不覆蓋
      if (Array.isArray(vlmResult.floorLine) && boundary && boundary.source !== 'user') {
        const fl = vlmResult.floorLine.filter((p) => Array.isArray(p) && p.length >= 2 && isFinite(p[0]) && isFinite(p[1]))
          .map((p) => [r3(clamp(+p[0], 0, 1)), r3(clamp(+p[1], 0, 1))]).sort((a, b) => a[0] - b[0]);
        if (fl.length >= 2) { fl[0][0] = 0; fl[fl.length - 1][0] = 1; boundary.pts = fl; boundary.source = 'vlm'; log('[Stage 2] VLM 修正了地牆交界線'); }
      }
      if (Array.isArray(vlmResult.elements) && vlmResult.elements.length > 0) {
        const userEls = elements.filter((e) => e.source === 'user');
        const vlmEls = [];
        for (const el of vlmResult.elements) {
          let pts = Array.isArray(el.pts) ? el.pts.filter((p) => Array.isArray(p) && isFinite(p[0]) && isFinite(p[1])).map((p) => [+p[0], +p[1]]) : [];
          if (pts.length < 3) {
            if (![el.x, el.y, el.w, el.h].every((v) => isFinite(+v))) continue;
            pts = rectPts(+el.x, +el.y, +el.w, +el.h);
          }
          const type = TYPE_LABELS[el.type] ? el.type : 'obstacle_block';
          const e2 = mkEl(pts, type, { source: 'vlm', pfx: 'VLM', layer: LAYER_LABELS[el.layer] ? el.layer : null, dist: isFinite(+el.dist) ? +el.dist : null });
          if (ACTION_LABELS[el.action]) e2.action = el.action;
          vlmEls.push(e2);
        }
        // VLM 回傳的元素若和人工描邊重疊（中心落在裡面）就不重複加入
        elements = vlmEls.filter((v) => !userEls.some((u) => pointInPoly([v.x + v.w / 2, v.y + v.h / 2], u.pts))).concat(userEls);
      }
      if (boundary && boundary.source === 'vlm') refreshDerived();
      if (vlmResult.reasoning) {
        const rBox = $('vlmReasoningBox');
        rBox.style.display = 'block';
        rBox.textContent = vlmResult.reasoning;
      }

      $('stAcc').textContent = 'Stage2 VLM 覆寫';
      $('aiPredBadge').textContent = 'Stage 2 雲端 VLM 已更新判讀 (可於下方微調)';
      $('modelStatus').textContent = 'VLM Refined ✓';
      log(`[Stage 2 完成] 雲端 VLM 已更新空間層次與 ${elements.length} 項結構特徵！現在進入 Stage 3 供後台審核修改。`);
    }
  } catch (err) {
    log(`[Stage 2 警告] 雲端 VLM 呼叫失敗 (${err.message})，保留 Stage 1 本地判讀結果。`);
    $('modelStatus').textContent = 'Stage 1 Fallback';
    setVlmStatus('Stage 2 失敗：' + err.message, 'err');
  }

  setPipelineStage(3);
  renderCanvas();
  renderElementList();
  renderBoundaryPanel();
}

// ============================================================================
// 4. [STAGE 3] 後台人工修改確認與回饋訓練 (Distill VLM + Human Fixes -> Local MLP)
// ============================================================================
function buildTargetVectorFromUI() {
  const vec = new Array(13).fill(0);
  const tIdx = Math.max(0, TYPOLOGY_KEYS.indexOf($('selTypology').value));
  const pIdx = Math.max(0, PATHC_KEYS.indexOf($('selPathC').value));
  const sIdx = Math.max(0, SUPP_KEYS.indexOf($('selSupplement').value));
  vec[tIdx] = 1;
  vec[5 + pIdx] = 1;
  vec[9 + sIdx] = 1;
  return vec;
}

async function trainModelWithCorrection() {
  if (!spatialNet) return;
  const targetVec = buildTargetVectorFromUI();

  // 重新依據修正後的 elements 更新特徵向量後段，使本地 MLP 學到更準確的結構關聯
  currentFeatVec[11] = elements.filter((e) => e.type.startsWith('corner')).length / 3;
  currentFeatVec[12] = elements.filter((e) => e.type.startsWith('obstacle')).length / 4;
  currentFeatVec[13] = elements.filter((e) => e.layer === 'foreground').length / 4;
  currentFeatVec[14] = elements.filter((e) => e.layer === 'midground').length / 4;
  currentFeatVec[15] = elements.filter((e) => e.layer === 'background').length / 4;

  const record = {
    id: 'SAMPLE_' + (dataset.length + 1),
    timestamp: new Date().toISOString(),
    features: currentFeatVec.map((v) => +Number(v).toFixed(4)),
    target: targetVec,
    labels: {
      typology: $('selTypology').value,
      supplement: $('selSupplement').value,
      pathL: $('selPathL').value,
      pathC: $('selPathC').value,
      pathR: $('selPathR').value,
      maxDepth: parseFloat($('inpDepth').value) || 4.5,
    },
    boundary: boundary ? { pts: boundary.pts, horizon: boundary.horizon, source: boundary.source } : null,
    camera: { eye: camCfg.eye, hfov: camCfg.hfov },
    elements: JSON.parse(JSON.stringify(elements)),
  };
  dataset.push(record);
  localStorage.setItem('arSpatialDataset.v1', JSON.stringify(dataset));
  $('stSamples').textContent = dataset.length;

  const xs = tf.tensor2d(dataset.map((d) => d.features));
  const ys = tf.tensor2d(dataset.map((d) => d.target));

  $('modelStatus').textContent = 'Distilling...';
  const history = await spatialNet.fit(xs, ys, {
    epochs: 28,
    shuffle: true,
  });

  const finalLoss = history.history.loss[history.history.loss.length - 1];
  $('stLoss').textContent = finalLoss.toFixed(4);
  $('modelStatus').textContent = 'Trained ✓';

  xs.dispose();
  ys.dispose();

  try {
    await spatialNet.save('localstorage://ar-spatial-mlp-v1');
  } catch (e) {}

  log(`[Stage 3 完成] 已將雲端 VLM 與你的後台修正蒸餾訓練至本地 MLP 模型 (樣本 #${dataset.length}, Loss: ${finalLoss.toFixed(4)})！`);
}

// ============================================================================
// 5. 畫布：繪製與編輯工具
//   工具：
//     V 選取：點多邊形選取；拖曳頂點改形狀；拖曳內部整個移動；Delete 刪除
//     P 筆型描邊（像 Illustrator 鋼筆）：逐點點擊；點回第一點、雙擊或 Enter 完成；Backspace 刪上一點；Esc 取消
//     L 地牆交界線：拖曳控制點；點線段中間插入新點；雙擊控制點刪除；兩端只能上下移動
// ============================================================================
let tool = 'select';        // select | pen | line
let penPts = [];            // 描邊中的點（正規化）
let hover = null;           // 游標位置（畫布像素）
let drag = null;            // { kind: 'bnd'|'vtx'|'move', i, el, start, orig }
let showMask = false;

function setTool(t) {
  if (tool === 'pen' && t !== 'pen' && penPts.length) penPts = [];
  tool = t;
  for (const [id, k] of [['toolSelect', 'select'], ['toolPen', 'pen'], ['toolLine', 'line']]) { const b = $(id); if (b) b.classList.toggle('primary', tool === k); }
  $('penActions').style.display = tool === 'pen' ? 'flex' : 'none';
  cv.style.cursor = tool === 'pen' ? 'crosshair' : tool === 'line' ? 'ns-resize' : 'default';
  const hint = { select: '選取：點多邊形選取，拖曳頂點改形狀、拖曳內部移動，Delete 刪除', pen: '筆型描邊：逐點點擊描出範圍 → 點回第一點／雙擊／Enter 完成，Backspace 刪上一點，Esc 取消', line: '地牆交界線：拖曳控制點修正；點線段插入新點；雙擊控制點刪除；虛線地平線也能上下拖。修正後按「用修正後的交界線重新判讀」' }[tool];
  $('toolHint').textContent = hint;
  renderCanvas();
}

function setCanvasForImage(img) {
  const iw = img.naturalWidth || img.width, ih = img.naturalHeight || img.height;
  cv.width = 1000; cv.height = clamp(Math.round(1000 * ih / iw), 400, 1500);
}

const N2C = (p) => [p[0] * cv.width, p[1] * cv.height];
function handleR() { const r = cv.getBoundingClientRect(); return 11 * (cv.width / Math.max(1, r.width)); }

function drawPoly(pts, close) {
  ctx.beginPath();
  pts.forEach((p, i) => { const q = N2C(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
  if (close) ctx.closePath();
}

function renderCanvas() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (currentImg) ctx.drawImage(currentImg, 0, 0, cv.width, cv.height);
  else { ctx.fillStyle = '#0c0f16'; ctx.fillRect(0, 0, cv.width, cv.height); }
  const k = cv.width / 1000;

  // 材質分割（地板區域＋牆面樣本帶）
  if (showMask && boundary && boundary.mask) {
    const m = document.createElement('canvas'); m.width = boundary.mw; m.height = boundary.mh;
    const mc = m.getContext('2d'), im = mc.createImageData(boundary.mw, boundary.mh);
    for (let i = 0; i < boundary.mask.length; i++) if (boundary.mask[i]) { im.data[4 * i] = 60; im.data[4 * i + 1] = 230; im.data[4 * i + 2] = 120; im.data[4 * i + 3] = 90; }
    mc.putImageData(im, 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(m, 0, 0, cv.width, cv.height); ctx.imageSmoothingEnabled = true;
    ctx.fillStyle = 'rgba(255,184,108,0.12)';
    ctx.fillRect(0, boundary.wallBand[0] * cv.height, cv.width, (boundary.wallBand[1] - boundary.wallBand[0]) * cv.height);
    ctx.strokeStyle = 'rgba(255,230,60,0.55)'; ctx.lineWidth = 1.5 * k;
    for (const L of boundary.lines || []) { ctx.beginPath(); ctx.moveTo(L[0] * cv.width, L[1] * cv.height); ctx.lineTo(L[2] * cv.width, L[3] * cv.height); ctx.stroke(); }
  }

  // 地平線
  if (boundary && boundary.horizon > 0 && boundary.horizon < 1) {
    const y = boundary.horizon * cv.height;
    ctx.save(); ctx.setLineDash([6 * k, 6 * k]); ctx.strokeStyle = 'rgba(255,255,255,0.45)'; ctx.lineWidth = 1.2 * k;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cv.width, y); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(255,255,255,0.7)'; ctx.font = `${11 * k}px ui-monospace, monospace`; ctx.fillText(tool === 'line' ? '地平線（可上下拖曳）' : '地平線（消失點高度）', 8 * k, y - 5 * k);
    ctx.restore();
  }

  // 元素多邊形
  for (const el of elements) {
    const info = TYPE_LABELS[el.type] || TYPE_LABELS.obstacle_block;
    const sel = el.id === selectedElId;
    ctx.save();
    drawPoly(el.pts, true);
    ctx.fillStyle = info.color + (sel ? '40' : '22'); ctx.fill();
    ctx.strokeStyle = info.color; ctx.lineWidth = (sel ? 3 : 2) * k;
    if (el.layer === 'background') ctx.setLineDash([4 * k, 4 * k]); else if (el.layer === 'midground') ctx.setLineDash([9 * k, 4 * k]);
    ctx.stroke(); ctx.setLineDash([]);
    const srcTag = el.source === 'vlm' ? '[VLM]' : el.source === 'user' ? '[人工]' : el.tag === 'bnd' ? '[交界線]' : '[本地]';
    const tag = `${srcTag} ${info.name} · ${LAYER_LABELS[el.layer]} ${el.dist}m`;
    ctx.font = `bold ${11 * k}px -apple-system, "PingFang TC", monospace`;
    const tw = ctx.measureText(tag).width + 10 * k, tx = el.x * cv.width, ty = Math.max(18 * k, el.y * cv.height - 4 * k);
    ctx.fillStyle = 'rgba(8,9,12,0.85)'; ctx.fillRect(tx, ty - 16 * k, tw, 18 * k);
    ctx.fillStyle = info.color; ctx.fillText(tag, tx + 5 * k, ty - 3 * k);
    if (sel && tool === 'select') {
      for (const p of el.pts) { const q = N2C(p); ctx.fillStyle = '#fff'; ctx.strokeStyle = info.color; ctx.lineWidth = 2 * k; ctx.beginPath(); ctx.rect(q[0] - 5 * k, q[1] - 5 * k, 10 * k, 10 * k); ctx.fill(); ctx.stroke(); }
    }
    ctx.restore();
  }

  // 地牆交界線
  if (boundary) {
    const col = boundary.source === 'user' ? '#ffb86c' : boundary.source === 'vlm' ? '#c084fc' : '#6effe1';
    ctx.save();
    ctx.strokeStyle = 'rgba(0,0,0,0.6)'; ctx.lineWidth = 6 * k; drawPoly(boundary.pts, false); ctx.stroke();
    ctx.strokeStyle = col; ctx.lineWidth = 3 * k; drawPoly(boundary.pts, false); ctx.stroke();
    const r = tool === 'line' ? 7 * k : 4 * k;
    boundary.pts.forEach((p, i) => {
      const q = N2C(p);
      ctx.fillStyle = drag && drag.kind === 'bnd' && drag.i === i ? '#fff' : col; ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5 * k;
      ctx.beginPath(); ctx.arc(q[0], q[1], r, 0, 7); ctx.fill(); ctx.stroke();
    });
    const lbl = `地牆交界線（${{ local: '本地', vlm: 'VLM', user: '人工修正' }[boundary.source] || ''}）`;
    const q0 = N2C(boundary.pts[0]);
    ctx.font = `bold ${12 * k}px -apple-system, "PingFang TC", sans-serif`;
    const tw = ctx.measureText(lbl).width + 10 * k;
    ctx.fillStyle = 'rgba(8,9,12,0.85)'; ctx.fillRect(q0[0] + 4 * k, q0[1] + 8 * k, tw, 18 * k);
    ctx.fillStyle = col; ctx.fillText(lbl, q0[0] + 9 * k, q0[1] + 21 * k);
    // 交界線拖曳時：顯示距離
    if (drag && drag.kind === 'bnd') {
      const q = N2C(boundary.pts[drag.i]);
      const t = `${distAtY(boundary.pts[drag.i][1]).toFixed(1)} m`;
      ctx.fillStyle = 'rgba(8,9,12,0.9)'; ctx.fillRect(q[0] + 10 * k, q[1] - 26 * k, 60 * k, 20 * k);
      ctx.fillStyle = '#fff'; ctx.fillText(t, q[0] + 15 * k, q[1] - 11 * k);
    }
    ctx.restore();
  }

  // 描邊中
  if (tool === 'pen' && penPts.length) {
    ctx.save();
    const info = TYPE_LABELS[$('drawType').value] || TYPE_LABELS.obstacle_block;
    drawPoly(penPts, false);
    if (hover) ctx.lineTo(hover[0], hover[1]);
    ctx.strokeStyle = info.color; ctx.lineWidth = 2 * k; ctx.setLineDash([6 * k, 4 * k]); ctx.stroke(); ctx.setLineDash([]);
    if (penPts.length >= 3) { drawPoly(penPts, true); ctx.fillStyle = info.color + '22'; ctx.fill(); }
    penPts.forEach((p, i) => {
      const q = N2C(p);
      const closeHot = i === 0 && penPts.length >= 3 && hover && Math.hypot(hover[0] - q[0], hover[1] - q[1]) < handleR();
      ctx.fillStyle = i === 0 ? (closeHot ? '#fff' : info.color) : '#fff'; ctx.strokeStyle = info.color; ctx.lineWidth = 2 * k;
      ctx.beginPath(); ctx.rect(q[0] - (closeHot ? 7 : 4.5) * k, q[1] - (closeHot ? 7 : 4.5) * k, (closeHot ? 14 : 9) * k, (closeHot ? 14 : 9) * k); ctx.fill(); ctx.stroke();
    });
    ctx.restore();
  }
}

function renderElementList() {
  const container = $('elementList');
  $('elCount').textContent = `${elements.length} 項`;
  if (!elements.length) { container.innerHTML = `<div style="color:var(--dim);font-size:12px;padding:8px;">尚未標註物件。選「筆型描邊」在照片上逐點描出範圍。</div>`; return; }
  container.innerHTML = elements.map((el) => {
    const info = TYPE_LABELS[el.type] || TYPE_LABELS.obstacle_block;
    const srcBadge = el.source === 'vlm' ? '<span class="src-badge vlm">Stage2 VLM</span>' : el.source === 'user' ? '<span class="src-badge user">人工描邊</span>' : el.tag === 'bnd' ? '<span class="src-badge local">由交界線推得</span>' : '<span class="src-badge local">Stage1 本地</span>';
    return `
      <div class="el-item ${el.source === 'vlm' ? 'vlm-item' : 'local-item'}" data-id="${el.id}" style="${el.id === selectedElId ? 'box-shadow:0 0 0 2px ' + info.color : ''}">
        <div class="el-head">
          <span style="color:${info.color};">● ${info.name}（${el.dist} m · ${el.pts.length} 點）${srcBadge}</span>
          <button class="del-btn" data-del="${el.id}">刪除</button>
        </div>
        <div class="el-controls">
          <select data-field="type" data-id="${el.id}">${Object.keys(TYPE_LABELS).map((k) => `<option value="${k}" ${el.type === k ? 'selected' : ''}>${TYPE_LABELS[k].name}</option>`).join('')}</select>
          <select data-field="layer" data-id="${el.id}">${Object.keys(LAYER_LABELS).map((k) => `<option value="${k}" ${el.layer === k ? 'selected' : ''}>${LAYER_LABELS[k]}</option>`).join('')}</select>
          <select data-field="action" data-id="${el.id}">${Object.keys(ACTION_LABELS).map((k) => `<option value="${k}" ${el.action === k ? 'selected' : ''}>${ACTION_LABELS[k]}</option>`).join('')}</select>
        </div>
      </div>`;
  }).join('');
  container.querySelectorAll('select').forEach((sel) => sel.addEventListener('change', (e) => {
    const t = elements.find((x) => x.id === e.target.getAttribute('data-id'));
    if (!t) return;
    const f = e.target.getAttribute('data-field');
    t[f] = e.target.value; t.source = 'user';
    if (f === 'layer') t.autoLayer = false;
    renderCanvas(); renderElementList();
  }));
  container.querySelectorAll('[data-del]').forEach((btn) => btn.addEventListener('click', (e) => {
    elements = elements.filter((x) => x.id !== e.target.getAttribute('data-del'));
    renderCanvas(); renderElementList();
  }));
  container.querySelectorAll('.el-item').forEach((it) => it.addEventListener('click', (e) => {
    if (e.target.closest('select,button')) return;
    selectedElId = it.getAttribute('data-id'); setTool('select'); renderElementList();
  }));
}

function getCanvasPos(e) {
  const rect = cv.getBoundingClientRect();
  return [((e.clientX - rect.left) / rect.width) * cv.width, ((e.clientY - rect.top) / rect.height) * cv.height];
}
const C2N = (q) => [r3(clamp(q[0] / cv.width, 0, 1)), r3(clamp(q[1] / cv.height, 0, 1))];
function pointInPoly(p, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const a = pts[i], b = pts[j];
    if ((a[1] > p[1]) !== (b[1] > p[1]) && p[0] < ((b[0] - a[0]) * (p[1] - a[1])) / (b[1] - a[1] + 1e-12) + a[0]) inside = !inside;
  }
  return inside;
}
function segDist(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy || 1e-9;
  const t = clamp(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2, 0, 1);
  return [Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy), t];
}

function finishPen() {
  if (penPts.length < 3) { log('描邊至少需要 3 個點'); return; }
  const type = $('drawType').value, layerSel = $('drawLayer').value;
  const el = mkEl(penPts, type, { source: 'user', layer: layerSel === 'auto' ? null : layerSel, pfx: 'USER' });
  if (layerSel !== 'auto') el.autoLayer = false;
  elements.push(el);
  selectedElId = el.id;
  log(`[人工描邊] 新增 ${TYPE_LABELS[type].name}（${penPts.length} 點，${LAYER_LABELS[el.layer]}，約 ${el.dist} m）`);
  penPts = [];
  renderCanvas(); renderElementList();
}

function markBoundaryUser() {
  if (boundary.source !== 'user') log('[人工修正] 地牆交界線已改為人工版本（鎖定，Stage 1/VLM 不會覆蓋）');
  boundary.source = 'user';
}

cv.addEventListener('pointerdown', (e) => {
  if (!currentImg) return;
  const q = getCanvasPos(e), R = handleR();
  cv.setPointerCapture && cv.setPointerCapture(e.pointerId);
  if (tool === 'pen') {
    if (penPts.length >= 3) { const f = N2C(penPts[0]); if (Math.hypot(q[0] - f[0], q[1] - f[1]) < R) { finishPen(); return; } }
    penPts.push(C2N(q)); renderCanvas(); return;
  }
  if (tool === 'line' && boundary) {
    const P = boundary.pts.map(N2C);
    let bi = -1, bd = R * 1.3;
    P.forEach((p, i) => { const d = Math.hypot(p[0] - q[0], p[1] - q[1]); if (d < bd) { bd = d; bi = i; } });
    if (bi >= 0) { drag = { kind: 'bnd', i: bi }; renderCanvas(); return; }
    for (let i = 0; i + 1 < P.length; i++) {
      const [d] = segDist(q, P[i], P[i + 1]);
      if (d < R) { boundary.pts.splice(i + 1, 0, C2N(q)); markBoundaryUser(); drag = { kind: 'bnd', i: i + 1 }; renderCanvas(); return; }
    }
    // 點在地平線附近：拖曳地平線
    if (Math.abs(q[1] - boundary.horizon * cv.height) < R) { drag = { kind: 'hz' }; return; }
    return;
  }
  if (tool === 'select') {
    const sel = elements.find((x) => x.id === selectedElId);
    if (sel) {
      const i = sel.pts.findIndex((p) => { const c = N2C(p); return Math.hypot(c[0] - q[0], c[1] - q[1]) < R; });
      if (i >= 0) { drag = { kind: 'vtx', el: sel, i }; return; }
    }
    const n = C2N(q);
    const hit = elements.slice().reverse().find((x) => pointInPoly(n, x.pts));
    selectedElId = hit ? hit.id : null;
    if (hit) drag = { kind: 'move', el: hit, start: n, orig: hit.pts.map((p) => p.slice()) };
    renderCanvas(); renderElementList();
  }
});

window.addEventListener('pointermove', (e) => {
  const q = getCanvasPos(e);
  if (q[0] >= 0 && q[0] <= cv.width && q[1] >= 0 && q[1] <= cv.height) {
    const n = C2N(q);
    $('cursorCoord').textContent = `X: ${n[0].toFixed(2)}, Y: ${n[1].toFixed(2)}${boundary && n[1] > boundary.horizon ? ` · 地面距離 ${distAtY(n[1]).toFixed(1)} m` : ''}`;
  }
  hover = q;
  if (!drag) { if (tool === 'pen' && penPts.length) renderCanvas(); return; }
  const n = C2N(q);
  if (drag.kind === 'hz') { boundary.horizon = r3(Math.min(n[1], Math.min(...boundary.pts.map((p) => p[1])) - 0.01)); boundary.hsrc = 'user'; boundary.hauto = true; renderCanvas(); return; }
  if (drag.kind === 'bnd') {
    const p = boundary.pts, i = drag.i;
    if (i === 0 || i === p.length - 1) p[i] = [p[i][0], n[1]]; // 兩端只動上下
    else p[i] = [clamp(n[0], p[i - 1][0] + 0.005, p[i + 1][0] - 0.005), n[1]];
    markBoundaryUser();
  } else if (drag.kind === 'vtx') {
    drag.el.pts[drag.i] = n; syncBox(drag.el); drag.el.source = 'user';
  } else if (drag.kind === 'move') {
    const dx = n[0] - drag.start[0], dy = n[1] - drag.start[1];
    if (Math.abs(dx) + Math.abs(dy) > 0.003) { drag.el.pts = drag.orig.map((p) => [r3(clamp(p[0] + dx, 0, 1)), r3(clamp(p[1] + dy, 0, 1))]); syncBox(drag.el); drag.el.source = 'user'; drag.moved = true; }
  }
  renderCanvas();
});

window.addEventListener('pointerup', () => {
  if (!drag) return;
  const d = drag; drag = null;
  if (d.kind === 'bnd' || d.kind === 'hz') refreshDerived();
  else if (d.kind === 'vtx' || (d.kind === 'move' && d.moved)) {
    const el = d.el;
    el.dist = +distAtY(Math.max(boundaryAt(el.x + el.w / 2), el.y + el.h)).toFixed(1);
    if (el.autoLayer !== false) el.layer = layerOfDist(el.dist);
    renderCanvas(); renderElementList();
  } else renderCanvas();
});

cv.addEventListener('dblclick', (e) => {
  const q = getCanvasPos(e), R = handleR();
  if (tool === 'pen') {
    if (!penPts.length) return;
    const a = penPts[penPts.length - 1], b = penPts[penPts.length - 2];
    if (b && Math.hypot(a[0] - b[0], a[1] - b[1]) < 0.01) penPts.pop(); // 雙擊的第二下會重複加點
    finishPen(); return;
  }
  if (tool === 'line' && boundary && boundary.pts.length > 2) {
    const i = boundary.pts.findIndex((p) => { const c = N2C(p); return Math.hypot(c[0] - q[0], c[1] - q[1]) < R * 1.3; });
    if (i > 0 && i < boundary.pts.length - 1) { boundary.pts.splice(i, 1); markBoundaryUser(); refreshDerived(); }
  }
});

window.addEventListener('keydown', (e) => {
  const tg = e.target;
  if (tg && (/select|textarea/i.test(tg.tagName) || (/input/i.test(tg.tagName) && !/checkbox|radio|button/i.test(tg.type)))) return;
  if (e.key === 'v' || e.key === 'V') setTool('select');
  else if (e.key === 'p' || e.key === 'P') setTool('pen');
  else if (e.key === 'l' || e.key === 'L') setTool('line');
  else if (tool === 'pen' && e.key === 'Enter') finishPen();
  else if (tool === 'pen' && e.key === 'Escape') { penPts = []; renderCanvas(); }
  else if (tool === 'pen' && e.key === 'Backspace') { e.preventDefault(); penPts.pop(); renderCanvas(); }
  else if (tool === 'select' && (e.key === 'Delete' || e.key === 'Backspace') && selectedElId) {
    e.preventDefault(); elements = elements.filter((x) => x.id !== selectedElId); selectedElId = null; renderCanvas(); renderElementList();
  }
});

$('toolSelect').addEventListener('click', () => setTool('select'));
$('toolPen').addEventListener('click', () => setTool('pen'));
$('toolLine').addEventListener('click', () => setTool('line'));
$('penDone').addEventListener('click', finishPen);
$('penUndo').addEventListener('click', () => { penPts.pop(); renderCanvas(); });
$('penCancel').addEventListener('click', () => { penPts = []; renderCanvas(); });
$('chkMask').addEventListener('change', (e) => { showMask = e.target.checked; renderCanvas(); });
$('btnRedetect').addEventListener('click', () => {
  if (!currentImg) return;
  boundary = analyzeBoundary(currentImg);
  log('[重新偵測] 地牆交界線改回自動判讀（人工修正已解除）');
  refreshDerived();
});
$('btnRerunWithLine').addEventListener('click', () => runStage1LocalAnalysis({ keepBoundary: true }));
$('btnFlatten').addEventListener('click', () => { // 把交界線簡化成直線（只留兩端）
  if (!boundary) return;
  boundary.pts = [boundary.pts[0], boundary.pts[boundary.pts.length - 1]];
  markBoundaryUser(); refreshDerived();
});
$('inpEye').addEventListener('change', (e) => { camCfg.eye = clamp(parseFloat(e.target.value) || 1.5, 0.5, 3); refreshDerived(); });
$('inpFov').addEventListener('change', (e) => { camCfg.hfov = clamp(parseFloat(e.target.value) || 0, 0, 120); refreshDerived(); });
setTool('select');

// 範例空間：用真實透視畫出的房間（左右牆、後牆、磁磚地板、一張桌子、一扇門）
// 正確答案：交界線是「左牆腳斜線 → 後牆腳水平線 → 右牆腳斜線」三段折線，兩個折點都是陰角
function loadSyntheticRoomSample() {
  const w = 900, h = 640, hfov = 69 * DEG, f = (w / 2) / Math.tan(hfov / 2), cx = w / 2, cy = h / 2, eye = 1.5, yh = h * 0.44;
  const p = Math.atan((yh - cy) / f);
  const proj = (X, Y, Z) => { const dy = eye - Y, cyv = dy * Math.cos(p) + Z * Math.sin(p), czv = -dy * Math.sin(p) + Z * Math.cos(p); return [cx + f * X / czv, cy + f * cyv / czv]; };
  const off = document.createElement('canvas'); off.width = w; off.height = h;
  const c = off.getContext('2d');
  const quad = (pts, fill, edge) => { c.beginPath(); pts.forEach((q, i) => { const s = proj(...q); i ? c.lineTo(s[0], s[1]) : c.moveTo(s[0], s[1]); }); c.closePath(); c.fillStyle = fill; c.fill(); if (edge) { c.strokeStyle = edge; c.lineWidth = 2; c.stroke(); } };
  const Z0 = 0.7, Z1 = 6.8, XW = 2.1, CH = 2.8;
  c.fillStyle = '#d6d4cc'; c.fillRect(0, 0, w, h);
  quad([[-XW, 0, Z0], [XW, 0, Z0], [XW, 0, Z1], [-XW, 0, Z1]], '#7d6a55');
  // 磁磚縫（地板紋理）
  c.strokeStyle = 'rgba(40,30,20,0.35)'; c.lineWidth = 1;
  for (let x = -XW; x <= XW + 1e-6; x += 0.6) { const a = proj(x, 0, Z0), b = proj(x, 0, Z1); c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke(); }
  for (let z = Z0; z <= Z1 + 1e-6; z += 0.6) { const a = proj(-XW, 0, z), b = proj(XW, 0, z); c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke(); }
  quad([[-XW, CH, Z0], [XW, CH, Z0], [XW, CH, Z1], [-XW, CH, Z1]], '#efefea');
  quad([[-XW, 0, Z1], [XW, 0, Z1], [XW, CH, Z1], [-XW, CH, Z1]], '#c9d1d4', '#5d676b');
  quad([[-XW, 0, Z0], [-XW, 0, Z1], [-XW, CH, Z1], [-XW, CH, Z0]], '#dcd2c3', '#5d676b');
  quad([[XW, 0, Z0], [XW, 0, Z1], [XW, CH, Z1], [XW, CH, Z0]], '#bec8cc', '#5d676b');
  quad([[0.4, 0, Z1 - 0.01], [1.3, 0, Z1 - 0.01], [1.3, 2.1, Z1 - 0.01], [0.4, 2.1, Z1 - 0.01]], '#6b4a2e', '#3a2a1c'); // 門
  // 桌子
  const T = { x0: -1.3, x1: -0.1, z0: 3.0, z1: 3.8, h: 0.75 };
  quad([[T.x0, T.h, T.z0], [T.x1, T.h, T.z0], [T.x1, T.h, T.z1], [T.x0, T.h, T.z1]], '#a07850', '#4d3a24');
  quad([[T.x0, 0, T.z0], [T.x0 + 0.06, 0, T.z0], [T.x0 + 0.06, T.h, T.z0], [T.x0, T.h, T.z0]], '#4d3a24');
  quad([[T.x1 - 0.06, 0, T.z0], [T.x1, 0, T.z0], [T.x1, T.h, T.z0], [T.x1 - 0.06, T.h, T.z0]], '#4d3a24');
  quad([[T.x0, T.h - 0.1, T.z0], [T.x1, T.h - 0.1, T.z0], [T.x1, T.h, T.z0], [T.x0, T.h, T.z0]], '#6b4f33');
  const im = c.getImageData(0, 0, w, h); let s = 11;
  for (let i = 0; i < im.data.length; i += 4) { s = (s * 16807) % 2147483647; const n = (s % 13) - 6; im.data[i] += n; im.data[i + 1] += n; im.data[i + 2] += n; }
  c.putImageData(im, 0, 0);
  const img = new Image();
  img.onload = async () => {
    currentImg = img; setCanvasForImage(img);
    boundary = null; elements = []; selectedElId = null;
    camCfg.hfov = 69; $('inpFov').value = 69;
    log('已載入範例空間（真值：後牆約 6.8 m，左右牆距中線 2.1 m，桌子前緣約 3 m）');
    await runStage1LocalAnalysis();
  };
  img.src = off.toDataURL();
}

$('fileInput').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    const img = new Image();
    img.onload = () => {
      currentImg = img; setCanvasForImage(img);
      boundary = null; elements = []; selectedElId = null;
      camCfg.hfov = parseFloat($('inpFov').value) || 0;
      log(`已讀取空間照片: ${file.name} (${img.naturalWidth}x${img.naturalHeight})，啟動判讀流水線...`);
      runStage1LocalAnalysis();
    };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
});

$('btnSample').addEventListener('click', loadSyntheticRoomSample);
$('btnReAnalyze').addEventListener('click', runStage1LocalAnalysis);
$('btnRunVLM').addEventListener('click', runStage2CloudVLM);
$('btnTrain').addEventListener('click', trainModelWithCorrection);

$('btnExport').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(dataset, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `spatial_dataset_${dataset.length}samples.json`;
  a.click();
  log(`已匯出包含 ${dataset.length} 筆空間標註的訓練數據庫 JSON！`);
});

$('importJson').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async (ev) => {
    try {
      const imported = JSON.parse(ev.target.result);
      if (Array.isArray(imported) && imported.length > 0) {
        dataset = dataset.concat(imported);
        localStorage.setItem('arSpatialDataset.v1', JSON.stringify(dataset));
        $('stSamples').textContent = dataset.length;
        log(`成功匯入 ${imported.length} 筆空間數據，正在批次訓練本地模型...`);
        await trainModelWithCorrection();
      }
    } catch (err) {
      log('JSON 匯入格式錯誤: ' + err.message);
    }
  };
  reader.readAsText(file);
});

initModels();
window.__tr = { boundaryAt, distAtY, refreshDerived, camCfg, setTool, get boundary() { return boundary; }, set boundary(v) { boundary = v; }, get elements() { return elements; }, get currentImg() { return currentImg; } };
})();
