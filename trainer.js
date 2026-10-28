// ============================================================================
// SPATIAL AI TRAINER v8 (trainer.js) — 雙階混合空間判讀與後台修正系統
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
// 2. [STAGE 1] 本地幾何透視分析 + COCO-SSD + In-Browser MLP 毫秒級初判
// ============================================================================
function extractImageGeometryAndLines(imgEl) {
  const W = 160, H = 104;
  const off = document.createElement('canvas');
  off.width = W; off.height = H;
  const octx = off.getContext('2d');
  octx.drawImage(imgEl, 0, 0, W, H);
  const px = octx.getImageData(0, 0, W, H).data;

  const gray = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    gray[i] = 0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2];
  }

  let edgeL = 0, edgeC = 0, edgeR = 0;
  let vertEdges = 0, horizEdges = 0;
  let lowerCenterEdge = 0, upperCenterEdge = 0;
  const colVertScore = new Float32Array(W);
  const rowHorizScore = new Float32Array(H);

  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const k = y * W + x;
      const dx = (-gray[k - W - 1] + gray[k - W + 1] - 2 * gray[k - 1] + 2 * gray[k + 1] - gray[k + W - 1] + gray[k + W + 1]) / 4;
      const dy = (-gray[k - W - 1] - 2 * gray[k - W] - gray[k - W + 1] + gray[k + W - 1] + 2 * gray[k + W] + gray[k + W + 1]) / 4;
      const m = Math.hypot(dx, dy);

      if (m > 14) {
        if (x < W * 0.33) edgeL++;
        else if (x < W * 0.66) edgeC++;
        else edgeR++;

        if (Math.abs(dx) > Math.abs(dy) * 1.5) {
          vertEdges++;
          colVertScore[x] += m;
        } else if (Math.abs(dy) > Math.abs(dx) * 1.5) {
          horizEdges++;
          if (y > H * 0.35 && y < H * 0.92) rowHorizScore[y] += m;
        }

        if (y > H * 0.58 && x > W * 0.28 && x < W * 0.72) lowerCenterEdge++;
        if (y <= H * 0.58 && x > W * 0.28 && x < W * 0.72) upperCenterEdge++;
      }
    }
  }

  let bestFloorY = H * 0.68, bestRowScore = 0;
  for (let y = Math.floor(H * 0.40); y < Math.floor(H * 0.90); y++) {
    if (rowHorizScore[y] > bestRowScore) {
      bestRowScore = rowHorizScore[y];
      bestFloorY = y;
    }
  }
  const floorBoundaryNorm = bestFloorY / H;

  const vPeaks = [];
  for (let x = 10; x < W - 10; x++) {
    const score = colVertScore[x - 1] + colVertScore[x] + colVertScore[x + 1];
    if (score > H * 18) {
      let isMax = true;
      for (let d = -6; d <= 6; d++) {
        if (colVertScore[x + d] > colVertScore[x]) { isMax = false; break; }
      }
      if (isMax) vPeaks.push({ xNorm: x / W, score });
    }
  }
  vPeaks.sort((a, b) => b.score - a.score);

  return {
    floorBoundaryNorm,
    edgeLNorm: edgeL / (W * H * 0.33),
    edgeCNorm: edgeC / (W * H * 0.34),
    edgeRNorm: edgeR / (W * H * 0.33),
    vertRatio: vertEdges / Math.max(1, vertEdges + horizEdges),
    lowerCenterTexture: lowerCenterEdge / (W * H * 0.2),
    upperCenterTexture: upperCenterEdge / (W * H * 0.2),
    vPeaks: vPeaks.slice(0, 4),
  };
}

async function runStage1LocalAnalysis() {
  if (!currentImg) return;
  setPipelineStage(1);
  $('vlmReasoningBox').style.display = 'none';
  elements = [];
  const geo = extractImageGeometryAndLines(currentImg);

  let cocoPreds = [];
  if (cocoModel) {
    try { cocoPreds = await cocoModel.detect(currentImg); } catch (e) {}
  }

  let fgObstacleArea = 0;
  let centerBlockedByObj = 0;
  let doorDetected = 0;

  for (const p of cocoPreds) {
    const [bx, by, bw, bh] = p.bbox;
    const x = bx / currentImg.width;
    const y = by / currentImg.height;
    const w = bw / currentImg.width;
    const h = bh / currentImg.height;
    const bottomY = y + h;

    const layer = bottomY > 0.76 ? 'foreground' : bottomY > 0.52 ? 'midground' : 'background';
    const estDist = +(1.5 / Math.max(0.12, bottomY - 0.32)).toFixed(1);

    let type = 'obstacle_block';
    if (['couch', 'chair', 'bed', 'potted plant'].includes(p.class)) {
      type = 'obstacle_cover';
    } else if (['dining table', 'desk', 'refrigerator', 'tv'].includes(p.class)) {
      type = 'obstacle_block';
    } else if (p.class === 'door') {
      type = 'door_open';
      doorDetected = 1;
    }

    if (layer === 'foreground' || layer === 'midground') {
      fgObstacleArea += w * h;
      if (x + w > 0.35 && x < 0.65 && bottomY > 0.62) {
        centerBlockedByObj = 1;
      }
    }

    elements.push({
      id: 'EL_' + Math.random().toString(36).substring(2, 6).toUpperCase(),
      x: +x.toFixed(3),
      y: +y.toFixed(3),
      w: +w.toFixed(3),
      h: +h.toFixed(3),
      type,
      layer,
      action: TYPE_LABELS[type].defaultAction,
      dist: estDist,
      source: 'local',
    });
  }

  if (geo.vPeaks.length >= 2 && Math.abs(geo.vPeaks[0].xNorm - geo.vPeaks[1].xNorm) > 0.12 && Math.abs(geo.vPeaks[0].xNorm - geo.vPeaks[1].xNorm) < 0.36) {
    const x1 = Math.min(geo.vPeaks[0].xNorm, geo.vPeaks[1].xNorm);
    const x2 = Math.max(geo.vPeaks[0].xNorm, geo.vPeaks[1].xNorm);
    doorDetected = 1;
    elements.push({
      id: 'EL_' + Math.random().toString(36).substring(2, 6).toUpperCase(),
      x: +x1.toFixed(3),
      y: 0.14,
      w: +(x2 - x1).toFixed(3),
      h: +(geo.floorBoundaryNorm - 0.14).toFixed(3),
      type: 'door_open',
      layer: geo.floorBoundaryNorm > 0.72 ? 'foreground' : 'midground',
      action: 'pass_through',
      dist: +(1.6 / Math.max(0.15, geo.floorBoundaryNorm - 0.3)).toFixed(1),
      source: 'local',
    });
  } else {
    geo.vPeaks.slice(0, 2).forEach((pk, idx) => {
      const isConvex = idx === 0 && geo.edgeCNorm > geo.edgeLNorm;
      const cType = isConvex ? 'corner_convex' : 'corner_concave';
      elements.push({
        id: 'EL_' + Math.random().toString(36).substring(2, 6).toUpperCase(),
        x: +Math.max(0.02, pk.xNorm - 0.06).toFixed(3),
        y: 0.12,
        w: 0.12,
        h: +(Math.min(0.82, geo.floorBoundaryNorm) - 0.12).toFixed(3),
        type: cType,
        layer: 'midground',
        action: TYPE_LABELS[cType].defaultAction,
        dist: +(1.8 / Math.max(0.15, geo.floorBoundaryNorm - 0.28)).toFixed(1),
        source: 'local',
      });
    });
  }

  if (geo.floorBoundaryNorm > 0.76 && geo.upperCenterTexture < 0.18 && !doorDetected) {
    elements.push({
      id: 'EL_WALL',
      x: 0.22,
      y: 0.08,
      w: 0.56,
      h: +(geo.floorBoundaryNorm - 0.08).toFixed(3),
      type: 'wall_deadend',
      layer: 'foreground',
      action: 'stop_prioritize_real',
      dist: 1.6,
      source: 'local',
    });
  }

  currentFeatVec = [
    geo.floorBoundaryNorm,
    geo.edgeLNorm,
    geo.edgeCNorm,
    geo.edgeRNorm,
    geo.vertRatio,
    geo.lowerCenterTexture,
    geo.upperCenterTexture,
    fgObstacleArea,
    centerBlockedByObj,
    doorDetected,
    geo.vPeaks.length / 4,
    elements.filter((e) => e.type.startsWith('corner')).length / 3,
    elements.filter((e) => e.type.startsWith('obstacle')).length / 4,
    elements.filter((e) => e.layer === 'foreground').length / 4,
    elements.filter((e) => e.layer === 'midground').length / 4,
    elements.filter((e) => e.layer === 'background').length / 4,
    geo.edgeLNorm > geo.edgeRNorm ? 1 : 0,
    geo.floorBoundaryNorm > 0.74 ? 1 : 0,
  ];

  runLocalMLPInference();
  renderCanvas();
  renderElementList();
  log(`[Stage 1 完成] 本地幾何 + MLP 初判共標出 ${elements.length} 項空間特徵。`);

  // 若勾選「自動銜接雲端 VLM」，立即啟動 Stage 2
  if ($('chkAutoVLM').checked) {
    await runStage2CloudVLM();
  } else {
    setPipelineStage(3);
  }
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
  // 取出最近 2 筆使用者在後台人工修正過的樣本作為 Few-Shot 經驗提示
  const fewShotSummary = dataset.slice(-2).map((d, i) =>
    `範例${i + 1}: 當畫面特徵具有 ${d.elements.map((e) => e.type + '(' + e.layer + ')').join(', ')} 時，正確空間型態為 ${d.labels.typology}，中央動線為 ${d.labels.pathC}，應補充 ${d.labels.supplement}。`
  ).join('\n');

  const stage1Summary = JSON.stringify({
    typology: $('selTypology').value,
    pathC: $('selPathC').value,
    supplement: $('selSupplement').value,
    elements: elements.map((e) => ({ type: e.type, layer: e.layer, x: e.x, y: e.y, w: e.w, h: e.h, dist: e.dist })),
  });

  return `你是一個專業的建築空間與 AR 戰術拓撲判讀 AI。
我們的設計原則是「實體優先、虛擬補充」：先精準判讀真實空間中的轉角、門扇、牆壁死路與阻擋家具（如桌子/椅子），再決定如何於不衝突的位置補充實心虛擬建築。

以下是【Stage 1 本地模型與幾何演算法的初步判讀】：
${stage1Summary}

${fewShotSummary ? '以下是【使用者過去在後台人工修正的標準判讀經驗】：\n' + fewShotSummary : ''}

請仔細檢視這張空間照片，修正 Stage 1 的錯誤或漏判（例如區分陽角 corner_convex 與陰角 corner_concave、判斷門扇 door_open、前方是否為死路牆壁 wall_deadend、或是被桌子擋住 obstacle_block / 半高掩體 obstacle_cover，以及其前/中/後景層次關係），並只回傳純 JSON 格式（不要加 markdown 標籤）：
{
  "typology": "corridor" | "corner_space" | "cluttered_room" | "dead_end" | "open_hall",
  "supplement": "room" | "alley" | "cover" | "wall",
  "pathL": "clear" | "blocked" | "corner_exit",
  "pathC": "clear" | "blocked_obj" | "blocked_wall" | "door_way",
  "pathR": "clear" | "blocked" | "corner_exit",
  "maxDepth": 數字(公尺, 1.5~20),
  "reasoning": "用繁體中文簡述空間層次關係、哪裡被擋住、哪裡有轉角或門，以及為何建議補充該虛擬建築(60字內)",
  "elements": [
    {
      "type": "corner_convex" | "corner_concave" | "door_open" | "wall_deadend" | "obstacle_block" | "obstacle_cover",
      "layer": "foreground" | "midground" | "background",
      "action": "attach_room" | "attach_wall" | "use_as_cover" | "detour_avoid" | "stop_prioritize_real" | "pass_through",
      "x": 0.0~1.0 (左上角X比例),
      "y": 0.0~1.0 (左上角Y比例),
      "w": 0.05~1.0 (寬度比例),
      "h": 0.05~1.0 (高度比例),
      "dist": 估計距離公尺(數字)
    }
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
    let data;
    try {
      data = await call(model);
    } catch (e) {
      // 模型不存在或這把 Key 不能用：自動改用 Key 可用的最新 Flash 再試一次
      if (e.status === 404 || e.status === 403 || /not found|not supported|no longer available/i.test(e.message)) {
        const names = await listGeminiModels(apiKey);
        const alt = pickFlash(names);
        if (alt && alt !== model) {
          log(`模型 ${model} 無法使用，自動改用 ${alt}`);
          $('vlmModel').value = alt; lsSet(LS_MODEL, alt);
          data = await call(alt);
        } else throw e;
      } else throw e;
    }
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
    maxDepth: +(1.8 / Math.max(0.15, currentFeatVec[0] - 0.25)).toFixed(1),
    reasoning: hasDesk
      ? '【VLM 空間層次複判】前景中央偏右有一張實心桌體阻擋直行動線（需從左側繞行）；中景兩側具備內凹陰角且後方有門扇開口，建議依「真實優先、虛擬補充」於空曠側補充 1.1m 實心掩體。'
      : '【VLM 空間層次複判】已確認地牆交界透視深度與兩側轉角位置，建議以中景真實牆角為錨點補充 2.4m 正交實心隔間。',
    elements: refined,
  };
}

async function runStage2CloudVLM() {
  if (!currentImg) return;
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

      if (Array.isArray(vlmResult.elements) && vlmResult.elements.length > 0) {
        elements = vlmResult.elements.map((el) => ({
          id: 'VLM_' + Math.random().toString(36).substring(2, 6).toUpperCase(),
          x: +Number(el.x || 0.2).toFixed(3),
          y: +Number(el.y || 0.2).toFixed(3),
          w: +Number(el.w || 0.2).toFixed(3),
          h: +Number(el.h || 0.3).toFixed(3),
          type: TYPE_LABELS[el.type] ? el.type : 'obstacle_block',
          layer: LAYER_LABELS[el.layer] ? el.layer : 'midground',
          action: ACTION_LABELS[el.action] ? el.action : TYPE_LABELS[el.type]?.defaultAction || 'detour_avoid',
          dist: +Number(el.dist || 2.5).toFixed(1),
          source: 'vlm',
        }));
      }

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
// 5. 畫布繪製與滑鼠互動框選
// ============================================================================
let dragBox = null;

function renderCanvas() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (currentImg) {
    ctx.drawImage(currentImg, 0, 0, cv.width, cv.height);
  } else {
    ctx.fillStyle = '#0c0f16';
    ctx.fillRect(0, 0, cv.width, cv.height);
  }

  if (currentFeatVec[0] > 0) {
    const fy = currentFeatVec[0] * cv.height;
    ctx.save();
    ctx.strokeStyle = 'rgba(110, 255, 225, 0.45)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.moveTo(0, fy);
    ctx.lineTo(cv.width, fy);
    ctx.stroke();
    ctx.fillStyle = 'rgba(110, 255, 225, 0.85)';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText('地牆交界基準線 (Floor-Wall Horizon)', 10, fy - 6);
    ctx.restore();
  }

  for (const el of elements) {
    const info = TYPE_LABELS[el.type] || TYPE_LABELS.obstacle_block;
    const rx = el.x * cv.width, ry = el.y * cv.height;
    const rw = el.w * cv.width, rh = el.h * cv.height;

    ctx.save();
    ctx.strokeStyle = info.color;
    ctx.lineWidth = el.id === selectedElId ? 3.0 : 2.2;
    if (el.layer === 'background') ctx.setLineDash([4, 4]);
    else if (el.layer === 'midground') ctx.setLineDash([8, 4]);
    ctx.strokeRect(rx, ry, rw, rh);

    ctx.fillStyle = info.color + '26';
    ctx.fillRect(rx, ry, rw, rh);

    ctx.setLineDash([]);
    const srcTag = el.source === 'vlm' ? '[VLM]' : el.source === 'user' ? '[人工]' : '[本地]';
    const tag = `${srcTag} ${info.name} · ${LAYER_LABELS[el.layer]} (${el.dist}m)`;
    ctx.font = 'bold 11px -apple-system, "PingFang TC", monospace';
    const tw = ctx.measureText(tag).width + 10;
    ctx.fillStyle = 'rgba(8, 9, 12, 0.88)';
    ctx.fillRect(rx, Math.max(0, ry - 20), tw, 20);
    ctx.fillStyle = info.color;
    ctx.fillText(tag, rx + 5, Math.max(14, ry - 6));
    ctx.restore();
  }

  if (dragBox) {
    ctx.save();
    ctx.strokeStyle = '#ffb86c';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(dragBox.x, dragBox.y, dragBox.w, dragBox.h);
    ctx.restore();
  }
}

function renderElementList() {
  const container = $('elementList');
  $('elCount').textContent = `${elements.length} 項`;
  if (!elements.length) {
    container.innerHTML = `<div style="color:var(--dim);font-size:12px;padding:8px;">尚未標註物件，可在左側照片按住滑鼠拖曳框選。</div>`;
    return;
  }
  container.innerHTML = elements.map((el) => {
    const info = TYPE_LABELS[el.type] || TYPE_LABELS.obstacle_block;
    const srcBadge = el.source === 'vlm'
      ? '<span class="src-badge vlm">Stage2 VLM</span>'
      : el.source === 'user'
      ? '<span class="src-badge user">Stage3 人工修改</span>'
      : '<span class="src-badge local">Stage1 本地</span>';

    return `
      <div class="el-item ${el.source === 'vlm' ? 'vlm-item' : 'local-item'}" data-id="${el.id}">
        <div class="el-head">
          <span style="color:${info.color};">● ${info.name} (${el.dist}m) ${srcBadge}</span>
          <button class="del-btn" data-del="${el.id}">刪除</button>
        </div>
        <div class="el-controls">
          <select data-field="type" data-id="${el.id}">
            ${Object.keys(TYPE_LABELS).map((k) => `<option value="${k}" ${el.type === k ? 'selected' : ''}>${TYPE_LABELS[k].name}</option>`).join('')}
          </select>
          <select data-field="layer" data-id="${el.id}">
            ${Object.keys(LAYER_LABELS).map((k) => `<option value="${k}" ${el.layer === k ? 'selected' : ''}>${LAYER_LABELS[k]}</option>`).join('')}
          </select>
          <select data-field="action" data-id="${el.id}">
            ${Object.keys(ACTION_LABELS).map((k) => `<option value="${k}" ${el.action === k ? 'selected' : ''}>${ACTION_LABELS[k]}</option>`).join('')}
          </select>
        </div>
      </div>
    `;
  }).join('');

  container.querySelectorAll('select').forEach((sel) => {
    sel.addEventListener('change', (e) => {
      const id = e.target.getAttribute('data-id');
      const field = e.target.getAttribute('data-field');
      const target = elements.find((x) => x.id === id);
      if (target) {
        target[field] = e.target.value;
        target.source = 'user';
        renderCanvas();
        renderElementList();
      }
    });
  });

  container.querySelectorAll('[data-del]').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      const id = e.target.getAttribute('data-del');
      elements = elements.filter((x) => x.id !== id);
      renderCanvas();
      renderElementList();
    });
  });
}

function getCanvasPos(e) {
  const rect = cv.getBoundingClientRect();
  return {
    x: ((e.clientX - rect.left) / rect.width) * cv.width,
    y: ((e.clientY - rect.top) / rect.height) * cv.height,
  };
}

cv.addEventListener('pointerdown', (e) => {
  const p = getCanvasPos(e);
  dragBox = { startX: p.x, startY: p.y, x: p.x, y: p.y, w: 0, h: 0 };
});

window.addEventListener('pointermove', (e) => {
  const p = getCanvasPos(e);
  if (p.x >= 0 && p.x <= cv.width && p.y >= 0 && p.y <= cv.height) {
    $('cursorCoord').textContent = `X: ${(p.x / cv.width).toFixed(2)}, Y: ${(p.y / cv.height).toFixed(2)}`;
  }
  if (!dragBox) return;
  dragBox.x = Math.min(dragBox.startX, p.x);
  dragBox.y = Math.min(dragBox.startY, p.y);
  dragBox.w = Math.abs(p.x - dragBox.startX);
  dragBox.h = Math.abs(p.y - dragBox.startY);
  renderCanvas();
});

window.addEventListener('pointerup', () => {
  if (!dragBox) return;
  if (dragBox.w > 18 && dragBox.h > 18) {
    const type = $('drawType').value;
    const layer = $('drawLayer').value;
    const bottomY = (dragBox.y + dragBox.h) / cv.height;
    const estDist = +(1.6 / Math.max(0.15, bottomY - 0.28)).toFixed(1);
    const newEl = {
      id: 'USER_' + Math.random().toString(36).substring(2, 6).toUpperCase(),
      x: +(dragBox.x / cv.width).toFixed(3),
      y: +(dragBox.y / cv.height).toFixed(3),
      w: +(dragBox.w / cv.width).toFixed(3),
      h: +(dragBox.h / cv.height).toFixed(3),
      type,
      layer,
      action: TYPE_LABELS[type].defaultAction,
      dist: estDist,
      source: 'user',
    };
    elements.push(newEl);
    selectedElId = newEl.id;
    log(`[Stage 3 人工新增] 已手動框選新增 [${TYPE_LABELS[type].name}] (${LAYER_LABELS[layer]})。`);
  }
  dragBox = null;
  renderCanvas();
  renderElementList();
});

function loadSyntheticRoomSample() {
  const off = document.createElement('canvas');
  off.width = 800; off.height = 520;
  const c = off.getContext('2d');

  c.fillStyle = '#1e232d'; c.fillRect(0, 0, 800, 110);
  c.fillStyle = '#2b313e'; c.fillRect(0, 110, 800, 240);
  c.fillStyle = '#222733';
  c.beginPath(); c.moveTo(0, 0); c.lineTo(190, 110); c.lineTo(190, 350); c.lineTo(0, 520); c.closePath(); c.fill();
  c.fillStyle = '#252b38';
  c.beginPath(); c.moveTo(800, 0); c.lineTo(610, 110); c.lineTo(610, 350); c.lineTo(800, 520); c.closePath(); c.fill();
  c.fillStyle = '#363e4d';
  c.beginPath(); c.moveTo(190, 350); c.lineTo(610, 350); c.lineTo(800, 520); c.lineTo(0, 520); c.closePath(); c.fill();

  c.strokeStyle = '#6c7a96'; c.lineWidth = 3;
  c.beginPath();
  c.moveTo(190, 110); c.lineTo(190, 350);
  c.moveTo(610, 110); c.lineTo(610, 350);
  c.moveTo(190, 350); c.lineTo(610, 350);
  c.moveTo(190, 350); c.lineTo(0, 520);
  c.moveTo(610, 350); c.lineTo(800, 520);
  c.stroke();

  c.fillStyle = '#151922';
  c.fillRect(360, 150, 85, 200);
  c.strokeStyle = '#9aa8c7';
  c.strokeRect(360, 150, 85, 200);

  c.fillStyle = '#5c4332';
  c.fillRect(390, 340, 190, 110);
  c.strokeStyle = '#8c6b53';
  c.strokeRect(390, 340, 190, 110);

  const img = new Image();
  img.onload = async () => {
    currentImg = img;
    elements.push({
      id: 'EL_DEMO_DESK',
      x: +(390 / 800).toFixed(3),
      y: +(340 / 520).toFixed(3),
      w: +(190 / 800).toFixed(3),
      h: +(110 / 520).toFixed(3),
      type: 'obstacle_block',
      layer: 'foreground',
      action: 'detour_avoid',
      dist: 1.8,
      source: 'local',
    });
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
      currentImg = img;
      log(`已讀取空間照片: ${file.name} (${img.width}x${img.height})，啟動雙階空間判讀流水線...`);
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
})();
