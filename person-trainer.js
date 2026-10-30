// ==========================================================
// click clip!! — 玩家辨識訓練器
//   流程：照片 → 偵測人與關節點（MoveNet）→ 自動判斷髮長／髮色／上衣／褲子
//        → 你修正 → 存進資料庫 → 重新學習（顏色最近鄰、髮長門檻）
//        → 對新照片排名「這是誰」
//   使用：vision.js（MoveNet、外觀特徵、相似度）＋ person-attrs.js（特徵判斷）
// ==========================================================
(function () {
  'use strict';
  const V = CC.Vision, A = CC.Attr, KP = V.KP;
  const STORE = 'clickclip.personModel.v1';
  const MAXW = 960;
  const $ = (id) => document.getElementById(id);
  const cv = $('cv'), ctx = cv.getContext('2d');
  const src = document.createElement('canvas'), sctx = src.getContext('2d', { willReadFrequently: true });
  const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  const BONES = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 1], [0, 2], [1, 3], [2, 4]];
  const HAIR_OPTS = ['short', 'mid', 'long', 'unknown'];

  const S = { hasImage: false, data: null, persons: [], sel: -1, drag: null, db: load(), model: null, stream: null, truth: null };
  S.model = A.learn(S.db.samples);

  // ---------- 資料庫 ----------
  function load() {
    try { const j = JSON.parse(localStorage.getItem(STORE) || 'null'); if (j && Array.isArray(j.samples)) return j; } catch (e) {}
    return { v: 1, samples: [] };
  }
  function save() {
    try { localStorage.setItem(STORE, JSON.stringify(S.db)); return true; }
    catch (e) { setStatus('⚠ 存不進瀏覽器（空間滿了？）請先匯出 JSON'); return false; }
  }

  // ---------- 狀態列 ----------
  function setStatus(t) { $('status').textContent = t; }
  function modelStatus() {
    if (V.status === 'movenet') return '✓ MoveNet 骨架模型';
    if (V.status === 'loading') return '模型載入中…';
    return '⚠ 骨架模型沒載入（網路被擋？）：可用「手動骨架」';
  }

  // ---------- 影像 ----------
  function setImage(el, w, h) {
    const k = Math.min(1, MAXW / Math.max(w, h));
    src.width = cv.width = Math.round(w * k); src.height = cv.height = Math.round(h * k);
    sctx.drawImage(el, 0, 0, src.width, src.height);
    S.data = sctx.getImageData(0, 0, src.width, src.height).data;
    S.hasImage = true; S.persons = []; S.sel = -1; S.truth = null;
    $('empty').style.display = 'none';
  }

  async function detect() {
    if (!S.hasImage) return;
    if (V.status !== 'movenet') { await V.load(); }
    if (V.status !== 'movenet') { setStatus(modelStatus()); render(); return; }
    setStatus('偵測中…');
    const ps = await V.detectPersons(src);
    S.persons = ps.filter((p) => p.kp).map((p) => ({ kp: p.kp, box: p.box, score: p.score }));
    S.persons.sort((a, b) => a.box.x - b.box.x);
    analyzeAll();
    S.sel = S.persons.length ? 0 : -1;
    setStatus(S.persons.length ? `找到 ${S.persons.length} 人` : '沒找到人：可用「手動骨架」');
    refresh();
  }

  function analyze(p) {
    p.box = V.boxFromKp(p.kp, src.width, src.height) || p.box;
    p.attrs = A.describe(S.data, src.width, src.height, p, S.model);
    p.feat = V.features(S.data, src.width, src.height, p);
    p.edit = p.edit || {};
  }
  function analyzeAll() { S.persons.forEach(analyze); }

  // ---------- 範例照片（合成，已知正確答案，用來測試流程）----------
  const SAMPLE_PEOPLE = [
    [{ hair: 'short', hairC: '#1c1a19', top: '#c62828', pants: '#28417a' }, { hair: 'long', hairC: '#4a2f22', top: '#efefea', pants: '#1d1d1f' }, { hair: 'mid', hairC: '#151515', top: '#2e8b3e', pants: '#c3b091' }],
    [{ hair: 'long', hairC: '#151515', top: '#1f62c4', pants: '#9a9a9a' }, { hair: 'mid', hairC: '#5a3a26', top: '#f2c12e', pants: '#1f2a52' }, { hair: 'short', hairC: '#202020', top: '#7b1f2a', pants: '#1b1b1b' }],
  ];
  let sampleIdx = 0;
  function kpTemplate(cx, foot, H) {
    const y = (f) => foot - H + f * H, P = (x, f) => ({ x: cx + x * H, y: y(f), s: 0.9 });
    return [P(0, 0.088), P(0.018, 0.066), P(-0.018, 0.066), P(0.045, 0.075), P(-0.045, 0.075), P(0.115, 0.185), P(-0.115, 0.185),
      P(0.14, 0.33), P(-0.14, 0.33), P(0.15, 0.46), P(-0.15, 0.46), P(0.065, 0.5), P(-0.065, 0.5), P(0.07, 0.73), P(-0.07, 0.73), P(0.07, 0.95), P(-0.07, 0.95)];
  }
  function drawPerson(g, kp, H, o) {
    const L = (a, b, w, c) => { g.strokeStyle = c; g.lineWidth = w; g.lineCap = 'round'; g.beginPath(); g.moveTo(kp[a].x, kp[a].y); g.lineTo(kp[b].x, kp[b].y); g.stroke(); };
    const hx = kp[0].x, eyeY = kp[1].y, headTop = eyeY - 0.066 * H, hw = 0.05 * H;
    const shY = kp[5].y;
    // 長髮／中長髮（在頭後面）
    if (o.hair !== 'short') {
      const bottom = o.hair === 'long' ? eyeY + (shY - eyeY) * 1.75 : eyeY + (shY - eyeY) * 0.78;
      g.fillStyle = o.hairC; g.beginPath();
      g.moveTo(hx - hw * 1.15, eyeY - hw * 0.4); g.lineTo(hx + hw * 1.15, eyeY - hw * 0.4);
      g.lineTo(hx + hw * (o.hair === 'long' ? 1.3 : 1.15), bottom); g.lineTo(hx - hw * (o.hair === 'long' ? 1.3 : 1.15), bottom); g.closePath(); g.fill();
    }
    // 腿（褲子）、鞋
    L(11, 13, 0.07 * H, o.pants); L(12, 14, 0.07 * H, o.pants); L(13, 15, 0.06 * H, o.pants); L(14, 16, 0.06 * H, o.pants);
    g.fillStyle = '#222'; for (const i of [15, 16]) g.fillRect(kp[i].x - 0.03 * H, kp[i].y, 0.06 * H, 0.03 * H);
    // 軀幹（上衣）
    g.fillStyle = o.top; g.beginPath();
    g.moveTo(kp[6].x - 0.02 * H, kp[6].y - 0.01 * H); g.lineTo(kp[5].x + 0.02 * H, kp[5].y - 0.01 * H);
    g.lineTo(kp[11].x + 0.02 * H, kp[11].y + 0.03 * H); g.lineTo(kp[12].x - 0.02 * H, kp[12].y + 0.03 * H); g.closePath(); g.fill();
    // 手臂（袖子＋皮膚）
    L(5, 7, 0.05 * H, o.top); L(6, 8, 0.05 * H, o.top); L(7, 9, 0.04 * H, '#d9a77f'); L(8, 10, 0.04 * H, '#d9a77f');
    // 脖子、頭
    g.fillStyle = '#d9a77f'; g.fillRect(hx - 0.02 * H, eyeY + 0.03 * H, 0.04 * H, shY - eyeY - 0.03 * H);
    g.beginPath(); g.ellipse(hx, eyeY + 0.005 * H, hw * 0.92, 0.067 * H, 0, 0, Math.PI * 2); g.fill();
    // 長髮：垂在肩膀前面的兩束
    if (o.hair === 'long') {
      const bottom = eyeY + (shY - eyeY) * 1.75;
      g.fillStyle = o.hairC;
      for (const sgn of [-1, 1]) {
        g.beginPath(); g.moveTo(hx + sgn * hw * 0.62, eyeY); g.lineTo(hx + sgn * hw * 1.12, eyeY);
        g.lineTo(hx + sgn * hw * 1.35, bottom); g.lineTo(hx + sgn * hw * 0.8, bottom); g.closePath(); g.fill();
      }
    }
    // 頭頂頭髮
    g.fillStyle = o.hairC; g.beginPath();
    g.ellipse(hx, eyeY - 0.012 * H, hw * 1.02, 0.056 * H, 0, Math.PI, Math.PI * 2); g.fill();
    g.fillRect(hx - hw * 1.02, eyeY - 0.016 * H, hw * 0.22, o.hair === 'short' ? 0.03 * H : 0.02 * H);
    g.fillRect(hx + hw * 0.8, eyeY - 0.016 * H, hw * 0.22, o.hair === 'short' ? 0.03 * H : 0.02 * H);
    // 眼睛
    g.fillStyle = '#222'; for (const i of [1, 2]) g.fillRect(kp[i].x - 2, kp[i].y - 1.5, 4, 3);
  }
  function makeSample() {
    const W = 960, H = 640, c = document.createElement('canvas'); c.width = W; c.height = H;
    const g = c.getContext('2d');
    const wall = g.createLinearGradient(0, 0, 0, 330); wall.addColorStop(0, '#c9c4b8'); wall.addColorStop(1, '#b3ad9f');
    g.fillStyle = wall; g.fillRect(0, 0, W, 330);
    g.fillStyle = '#8a7b67'; g.fillRect(0, 330, W, H - 330);
    g.strokeStyle = 'rgba(0,0,0,.08)'; for (let x = -400; x < W + 400; x += 70) { g.beginPath(); g.moveTo(W / 2 + (x - W / 2) * 0.35, 330); g.lineTo(x, H); g.stroke(); }
    // 雜訊（讓顏色不是純色，比較像照片）
    const set = SAMPLE_PEOPLE[sampleIdx++ % SAMPLE_PEOPLE.length];
    const people = set.map((o, i) => { const Hh = 470 - i * 12; return { o, kp: kpTemplate(200 + i * 280, 600 - i * 6, Hh), H: Hh }; });
    people.forEach((p) => drawPerson(g, p.kp, p.H, p.o));
    const im = g.getImageData(0, 0, W, H), d = im.data;
    for (let i = 0; i < d.length; i += 4) { const n = (Math.random() - 0.5) * 18; d[i] += n; d[i + 1] += n; d[i + 2] += n; }
    g.putImageData(im, 0, 0);
    return { canvas: c, people };
  }
  function loadSample() {
    stopCam();
    const s = makeSample();
    setImage(s.canvas, s.canvas.width, s.canvas.height);
    S.truth = s.people.map((p) => ({ hair: p.o.hair, top: A.nameColor(hex(p.o.top)).name, pants: A.nameColor(hex(p.o.pants)).name, hairC: A.nameColor(hex(p.o.hairC)).name }));
    // 範例照片直接使用已知的關節點（不靠模型也能測流程）；模型有載入時也可以按「重新偵測」比較
    S.persons = s.people.map((p) => ({ kp: p.kp.map((k) => ({ ...k })), box: null, score: 1 }));
    analyzeAll(); S.sel = 0;
    setStatus('範例照片：3 人（關節點為已知值）');
    refresh();
  }
  function hex(h) { return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)); }

  // ---------- 手動骨架 ----------
  function addManual() {
    if (!S.hasImage) { setStatus('先載入照片'); return; }
    const H = src.height * 0.82;
    S.persons.push({ kp: kpTemplate(src.width / 2, src.height * 0.93, H), box: null, score: 0, manual: true });
    analyze(S.persons[S.persons.length - 1]);
    S.sel = S.persons.length - 1;
    setStatus('已放入手動骨架：把白點拖到肩、髖、膝、眼睛等位置');
    refresh();
  }

  // ---------- 相機 ----------
  async function startCam() {
    try {
      S.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 } }, audio: false });
    } catch (e) { setStatus('無法開啟相機：' + (e.message || e.name)); return; }
    const v = $('video'); v.srcObject = S.stream; v.style.display = 'block'; $('shutter').style.display = 'block'; $('empty').style.display = 'none';
    setStatus('對準玩家，按紅色快門');
  }
  function stopCam() {
    if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
    S.stream = null; $('video').style.display = 'none'; $('shutter').style.display = 'none';
  }
  function shoot() {
    const v = $('video'); if (!v.videoWidth) return;
    setImage(v, v.videoWidth, v.videoHeight); stopCam(); render(); detect();
  }

  // ---------- 繪圖 ----------
  const ok = (k) => k && k.s >= 0.3;
  function render() {
    if (!S.hasImage) { ctx.fillStyle = '#050608'; ctx.fillRect(0, 0, cv.width, cv.height); return; }
    ctx.drawImage(src, 0, 0);
    const u = cv.width / 640; // 線寬比例
    S.persons.forEach((p, i) => {
      const on = i === S.sel;
      if (on && $('chkRegions').checked && p.attrs) drawRegions(p.attrs.regions, u);
      ctx.strokeStyle = on ? 'rgba(255,255,255,.95)' : 'rgba(255,255,255,.45)'; ctx.lineWidth = (on ? 2 : 1.3) * u;
      for (const [a, b] of BONES) if (ok(p.kp[a]) && ok(p.kp[b])) { ctx.beginPath(); ctx.moveTo(p.kp[a].x, p.kp[a].y); ctx.lineTo(p.kp[b].x, p.kp[b].y); ctx.stroke(); }
      for (const k of p.kp) if (ok(k)) { ctx.fillStyle = on ? '#fff' : 'rgba(255,255,255,.6)'; ctx.beginPath(); ctx.arc(k.x, k.y, (on ? 4.5 : 3) * u, 0, 7); ctx.fill(); if (on) { ctx.strokeStyle = '#000'; ctx.lineWidth = u; ctx.stroke(); } }
      if (p.box) {
        ctx.setLineDash(on ? [8 * u, 5 * u] : [3 * u, 5 * u]); ctx.strokeStyle = on ? '#ff3b30' : 'rgba(255,255,255,.35)'; ctx.lineWidth = 2 * u;
        ctx.strokeRect(p.box.x, p.box.y, p.box.w, p.box.h); ctx.setLineDash([]);
        const tag = labelOf(p, i);
        ctx.font = `700 ${14 * u}px sans-serif`; const tw = ctx.measureText(tag).width;
        ctx.fillStyle = on ? '#ff3b30' : 'rgba(0,0,0,.6)'; ctx.fillRect(p.box.x, p.box.y - 20 * u, tw + 12 * u, 20 * u);
        ctx.fillStyle = '#fff'; ctx.fillText(tag, p.box.x + 6 * u, p.box.y - 5 * u);
      }
    });
  }
  function labelOf(p, i) {
    const r = p.rank && p.rank[0];
    return `#${i + 1}` + (p.edit && p.edit.letter ? ' ' + p.edit.letter : r && r.score > 0.72 ? ` → ${r.letter}?` : '');
  }
  function drawRegions(R, u) {
    if (R.hair && R.hair.pts) {
      ctx.fillStyle = 'rgba(255,214,10,.55)'; const s = Math.max(1.5, R.hair.step * 0.8);
      for (const [x, y] of R.hair.pts) ctx.fillRect(x - s / 2, y - s / 2, s, s);
      ctx.strokeStyle = 'rgba(255,214,10,.9)'; ctx.lineWidth = 1.5 * u; ctx.setLineDash([4 * u, 4 * u]);
      for (const yy of [R.hair.eyeY, R.hair.shY]) { ctx.beginPath(); ctx.moveTo(R.hair.crown[0] - 30 * u, yy); ctx.lineTo(R.hair.crown[2] + 30 * u, yy); ctx.stroke(); }
      ctx.setLineDash([]); ctx.strokeStyle = '#ffd60a'; ctx.lineWidth = 2.5 * u;
      ctx.beginPath(); ctx.moveTo(R.hair.crown[0] - 40 * u, R.hair.bottom); ctx.lineTo(R.hair.crown[2] + 40 * u, R.hair.bottom); ctx.stroke();
    }
    const poly = (P, c) => { ctx.strokeStyle = c; ctx.fillStyle = c.replace('1)', '.16)'); ctx.lineWidth = 2 * u; ctx.beginPath(); P.forEach((q, k) => (k ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y))); ctx.closePath(); ctx.fill(); ctx.stroke(); };
    if (R.top) poly(R.top, 'rgba(255,59,48,1)');
    if (R.pants) R.pants.forEach((P) => poly(P, 'rgba(64,156,255,1)'));
  }

  // ---------- 右側面板 ----------
  function fillSelect(el, opts, labels) {
    el.innerHTML = opts.map((o, i) => `<option value="${o}">${labels ? labels[i] : o}</option>`).join('');
  }
  fillSelect($('sHair'), HAIR_OPTS, HAIR_OPTS.map((o) => A.HAIR_LEN[o]));
  const colorNames = A.COLORS.map((c) => c[0]);
  for (const id of ['sHairC', 'sTop', 'sPants']) fillSelect($(id), ['', ...colorNames], ['（無法判斷）', ...colorNames]);

  function autoVals(p) {
    const a = p.attrs || {};
    return { hairLen: a.hair ? a.hair.len : 'unknown', hairColor: a.hair && a.hair.name ? a.hair.name : '', top: a.top ? a.top.name : '', pants: a.pants ? a.pants.name : '' };
  }
  function curVals(p) { return Object.assign(autoVals(p), p.edit || {}); }
  const sw = (rgb) => (rgb ? `<span class="sw" style="background:rgb(${rgb.join(',')})"></span>` : '');
  const pct = (v) => Math.round(v * 100) + '%';

  function renderPanel() {
    const P = S.persons;
    $('nPersons').textContent = P.length ? `${P.length} 人` : '';
    $('persons').innerHTML = P.length ? P.map((p, i) => `<span class="chip ${i === S.sel ? 'on' : ''}" data-i="${i}">${labelOf(p, i)}${p.manual ? '（手動）' : ''}</span>`).join('') : '<span class="muted">尚無</span>';
    const p = P[S.sel];
    const dis = !p;
    for (const id of ['sHair', 'sHairC', 'sTop', 'sPants', 'inLetter', 'inName', 'btnAdd', 'btnReset']) $(id).disabled = dis;
    if (!p) { for (const id of ['aHair', 'aHairC', 'aTop', 'aPants']) $(id).innerHTML = '—'; $('warn').textContent = ''; renderRank(null); return; }
    const a = p.attrs || {}, au = autoVals(p), cu = curVals(p);
    const h = a.hair || {};
    $('aHair').innerHTML = h.len && h.len !== 'unknown'
      ? `<b>${A.HAIR_LEN[h.len]}</b><small>比例 ${h.ratio}・信心 ${pct(h.conf)}</small>` : `<small>${h.why || '無法判斷'}</small>`;
    $('aHairC').innerHTML = h.rgb ? `${sw(h.rgb)}<b>${h.name}</b>` : '<small>—</small>';
    $('aTop').innerHTML = a.top ? `${sw(a.top.rgb)}<b>${a.top.name}</b><small>${a.top.by === 'learned' ? '依你的修正' : '規則'}・占 ${pct(a.top.share)}</small>` : '<small>看不到</small>';
    $('aPants').innerHTML = a.pants ? `${sw(a.pants.rgb)}<b>${a.pants.name}</b><small>${a.pants.by === 'learned' ? '依你的修正' : '規則'}・占 ${pct(a.pants.share)}</small>` : '<small>看不到</small>';
    const map = { sHair: 'hairLen', sHairC: 'hairColor', sTop: 'top', sPants: 'pants' };
    for (const id in map) { $(id).value = cu[map[id]]; $(id).classList.toggle('changed', cu[map[id]] !== au[map[id]]); }
    $('inLetter').value = (p.edit && p.edit.letter) || '';
    $('inName').value = (p.edit && p.edit.name) || '';
    $('warn').textContent = (a.warn || []).filter((w, i, arr) => arr.indexOf(w) === i).map((w) => '⚠ ' + w).join('\n');
    $('warn').style.whiteSpace = 'pre-line';
    renderRank(p);
  }

  // ---------- 這是誰 ----------
  function rankPerson(p) {
    const groups = {};
    for (const s of S.db.samples) if (s.letter) (groups[s.letter] = groups[s.letter] || []).push(s);
    const cu = curVals(p), out = [];
    for (const L in groups) {
      let sim = 0, agree = 0, best = null;
      for (const s of groups[L]) {
        const sm = p.feat && s.feat ? V.similarity(p.feat, s.feat) : 0;
        const keys = ['hairLen', 'top', 'pants'].filter((k) => s.label[k] && cu[k] && cu[k] !== 'unknown' && s.label[k] !== 'unknown');
        const ag = keys.length ? keys.filter((k) => s.label[k] === cu[k]).length / keys.length : 0.5;
        if (0.7 * sm + 0.3 * ag > 0.7 * sim + 0.3 * agree) { sim = sm; agree = ag; best = s; }
      }
      out.push({ letter: L, name: best && best.name, sim, agree, score: 0.7 * sim + 0.3 * agree, n: groups[L].length });
    }
    return out.sort((a, b) => b.score - a.score);
  }
  function renderRank(p) {
    const el = $('rank');
    if (!p) { el.innerHTML = '<span class="muted">先選一個人。</span>'; return; }
    p.rank = rankPerson(p);
    if (!p.rank.length) { el.innerHTML = '<span class="muted">資料庫裡還沒有填了字母的樣本。先幫每位玩家各存一兩張。</span>'; return; }
    const gap = p.rank.length > 1 ? p.rank[0].score - p.rank[1].score : 1;
    el.innerHTML = p.rank.slice(0, 6).map((r, i) => `<div class="rank ${i === 0 ? 'top' : ''}"><span class="letter">${r.letter}</span>
      <div><div class="barw"><i style="width:${pct(r.score)}"></i></div><small style="text-align:left;display:block">外觀 ${pct(r.sim)}・特徵 ${pct(r.agree)}${r.name ? '・' + esc(r.name) : ''}・${r.n} 張</small></div>
      <small>${pct(r.score)}</small></div>`).join('') +
      `<div class="muted" style="margin-top:6px">${p.rank[0].score < 0.6 ? '⚠ 分數偏低：可能是新玩家，或光線、角度差很多。' : gap < 0.06 ? '⚠ 前兩名很接近：這兩人外觀太像，遊戲中會改用位置判斷。' : `最可能是 <b>${p.rank[0].letter}</b>。`}</div>`;
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

  // ---------- 加入資料庫 ----------
  function thumb(p) {
    const b = p.box || { x: 0, y: 0, w: src.width, h: src.height };
    const t = document.createElement('canvas'); t.width = 90; t.height = 120;
    const g = t.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, 90, 120);
    const k = Math.min(90 / b.w, 120 / b.h);
    g.drawImage(src, b.x, b.y, b.w, b.h, (90 - b.w * k) / 2, (120 - b.h * k) / 2, b.w * k, b.h * k);
    return t.toDataURL('image/jpeg', 0.7);
  }
  const pick = (o, keys) => (o ? keys.reduce((r, k) => (o[k] != null ? ((r[k] = o[k]), r) : r), {}) : null);
  function addSample() {
    const p = S.persons[S.sel]; if (!p) return;
    const a = p.attrs || {};
    const label = curVals(p);
    const sample = {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), t: Date.now(),
      letter: (p.edit && p.edit.letter) || '', name: (p.edit && p.edit.name) || '',
      label: { hairLen: label.hairLen, hairColor: label.hairColor, top: label.top, pants: label.pants },
      auto: autoVals(p),
      attrs: { hair: pick(a.hair, ['len', 'ratio', 'rgb', 'lab', 'conf']), top: pick(a.top, ['rgb', 'lab', 'share']), pants: pick(a.pants, ['rgb', 'lab', 'share']) },
      feat: p.feat, thumb: thumb(p),
    };
    S.db.samples.push(sample);
    if (!save()) { S.db.samples.pop(); return; }
    relearn();
    const fixed = ['hairLen', 'hairColor', 'top', 'pants'].filter((k) => sample.label[k] !== sample.auto[k]).length;
    setStatus(`已加入（你修正了 ${fixed} 項）。共 ${S.db.samples.length} 筆`);
    // 自動跳到下一個人
    if (S.sel < S.persons.length - 1) S.sel++;
    refresh();
  }
  function relearn() {
    S.model = A.learn(S.db.samples);
    if (S.hasImage) analyzeAll();
  }

  // ---------- 學習狀況 ----------
  function renderStats() {
    const ss = S.db.samples, n = ss.length;
    $('stN').textContent = n;
    $('stP').textContent = new Set(ss.filter((s) => s.letter).map((s) => s.letter)).size;
    $('stC').textContent = S.model.colors.length;
    if (!n) { $('acc').innerHTML = '還沒有樣本。每存一筆，系統就會把你的修正當作新的判斷依據。'; return; }
    const keys = [['hairLen', '髮長'], ['hairColor', '髮色'], ['top', '上衣'], ['pants', '褲子']];
    const atAdd = keys.map(([k, t]) => `${t} ${pct(ss.filter((s) => s.auto[k] === s.label[k]).length / n)}`).join('・');
    let loo = '';
    if (n >= 4) { // 留一驗證：每次拿掉一筆，用其他樣本學，再判斷這一筆
      const hit = { hairLen: 0, hairColor: 0, top: 0, pants: 0 }, cnt = { hairLen: 0, hairColor: 0, top: 0, pants: 0 };
      ss.forEach((s, i) => {
        const m = A.learn(ss.filter((_, j) => j !== i));
        const h = s.attrs.hair;
        if (h && h.ratio != null && s.label.hairLen !== 'unknown') { cnt.hairLen++; const len = h.ratio < m.hair.t1 ? 'short' : h.ratio < m.hair.t2 ? 'mid' : 'long'; if (len === s.label.hairLen) hit.hairLen++; }
        if (h && h.rgb && s.label.hairColor) { cnt.hairColor++; if (A.nameColor(h.rgb, m.colors).name === s.label.hairColor) hit.hairColor++; }
        for (const k of ['top', 'pants']) if (s.attrs[k] && s.attrs[k].rgb && s.label[k]) { cnt[k]++; if (A.nameColor(s.attrs[k].rgb, m.colors).name === s.label[k]) hit[k]++; }
      });
      loo = '<br>用目前學到的規則重新判斷（每筆都用「其他樣本」學出來的規則）：' + keys.map(([k, t]) => `${t} ${cnt[k] ? pct(hit[k] / cnt[k]) : '—'}`).join('・');
    }
    const hm = S.model.hair;
    $('acc').innerHTML = `存入時自動判斷正確率：${atAdd}${loo}<br>髮長門檻：短髮 &lt; ${hm.t1} ≤ 中長髮 &lt; ${hm.t2} ≤ 長髮 ${hm.learned ? '（已依你的修正調整）' : '（預設值；短／中／長各存 2 筆以上會自動調整）'}`;
  }

  function renderDB() {
    const ss = S.db.samples.slice().reverse();
    $('db').innerHTML = ss.map((s) => `<div class="card"><img src="${s.thumb}" alt=""><button class="x" data-del="${s.id}">✕</button>
      <div class="t"><b>${s.letter || '–'}</b> ${esc(s.name || '')}<br>${A.HAIR_LEN[s.label.hairLen] || ''} ${s.label.hairColor || ''}<br>上 ${s.label.top || '?'}・褲 ${s.label.pants || '?'}</div></div>`).join('');
  }

  function refresh() { render(); renderPanel(); renderStats(); renderDB(); window.__pt.ready = true; }

  // ---------- 事件 ----------
  $('file').addEventListener('change', (e) => {
    const f = e.target.files[0]; if (!f) return; stopCam();
    const im = new Image();
    im.onload = () => { setImage(im, im.naturalWidth, im.naturalHeight); URL.revokeObjectURL(im.src); render(); detect(); };
    im.src = URL.createObjectURL(f); e.target.value = '';
  });
  $('btnCam').onclick = () => (S.stream ? stopCam() : startCam());
  $('shutter').onclick = shoot;
  $('btnSample').onclick = loadSample;
  $('btnManual').onclick = addManual;
  $('btnRedetect').onclick = detect;
  $('chkRegions').onchange = render;
  $('persons').onclick = (e) => { const c = e.target.closest('.chip'); if (c) { S.sel = +c.dataset.i; refresh(); } };
  const editMap = { sHair: 'hairLen', sHairC: 'hairColor', sTop: 'top', sPants: 'pants' };
  for (const id in editMap) $(id).onchange = () => { const p = S.persons[S.sel]; if (!p) return; p.edit[editMap[id]] = $(id).value; renderPanel(); render(); };
  $('inLetter').oninput = () => {
    const p = S.persons[S.sel]; if (!p) return;
    const v = $('inLetter').value.toUpperCase().replace(/[^A-Z]/g, ''); $('inLetter').value = v; p.edit.letter = v; render(); renderPanel();
  };
  $('inName').oninput = () => { const p = S.persons[S.sel]; if (p) p.edit.name = $('inName').value.slice(0, 16); };
  $('btnAdd').onclick = addSample;
  $('btnReset').onclick = () => { const p = S.persons[S.sel]; if (!p) return; const { letter, name } = p.edit; p.edit = { letter, name }; renderPanel(); };
  $('db').onclick = (e) => {
    const b = e.target.closest('[data-del]'); if (!b) return;
    S.db.samples = S.db.samples.filter((s) => s.id !== b.dataset.del); save(); relearn(); refresh();
  };
  $('btnExport').onclick = () => {
    const blob = new Blob([JSON.stringify({ ...S.db, model: S.model, exported: new Date().toISOString() })], { type: 'application/json' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `clickclip-players-${new Date().toISOString().slice(0, 10)}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  };
  $('fileImport').addEventListener('change', (e) => {
    const f = e.target.files[0]; if (!f) return;
    f.text().then((t) => {
      const j = JSON.parse(t); if (!j || !Array.isArray(j.samples)) throw new Error('格式不對');
      const have = new Set(S.db.samples.map((s) => s.id)); let n = 0;
      for (const s of j.samples) if (s && s.id && s.label && s.attrs && !have.has(s.id)) { S.db.samples.push(s); n++; }
      save(); relearn(); refresh(); setStatus(`匯入 ${n} 筆`);
    }).catch((err) => setStatus('匯入失敗：' + err.message));
    e.target.value = '';
  });
  $('btnClear').onclick = () => {
    if (!S.db.samples.length) return;
    if (!confirm(`刪除全部 ${S.db.samples.length} 筆樣本？（建議先匯出）`)) return;
    S.db.samples = []; save(); relearn(); refresh();
  };

  // 拖曳關節點／點選人物
  function canvasPt(e) { const r = cv.getBoundingClientRect(); return { x: (e.clientX - r.left) * cv.width / r.width, y: (e.clientY - r.top) * cv.height / r.height, k: cv.width / r.width }; }
  cv.addEventListener('pointerdown', (e) => {
    if (!S.hasImage) return;
    const q = canvasPt(e), p = S.persons[S.sel];
    if (p) {
      let bi = -1, bd = 18 * q.k;
      p.kp.forEach((k, i) => { const d = Math.hypot(k.x - q.x, k.y - q.y); if (d < bd) { bd = d; bi = i; } });
      if (bi >= 0) { S.drag = { i: bi }; cv.setPointerCapture(e.pointerId); return; }
    }
    // 點到哪個人的框
    let hit = -1, area = Infinity;
    S.persons.forEach((pp, i) => { const b = pp.box; if (b && q.x >= b.x && q.x <= b.x + b.w && q.y >= b.y && q.y <= b.y + b.h && b.w * b.h < area) { area = b.w * b.h; hit = i; } });
    if (hit >= 0 && hit !== S.sel) { S.sel = hit; refresh(); }
  });
  cv.addEventListener('pointermove', (e) => {
    if (!S.drag) return;
    const q = canvasPt(e), k = S.persons[S.sel].kp[S.drag.i];
    k.x = Math.max(0, Math.min(cv.width - 1, q.x)); k.y = Math.max(0, Math.min(cv.height - 1, q.y)); k.s = 1;
    render();
  });
  const endDrag = () => { if (!S.drag) return; S.drag = null; const p = S.persons[S.sel]; if (p) analyze(p); refresh(); };
  cv.addEventListener('pointerup', endDrag); cv.addEventListener('pointercancel', endDrag);

  // ---------- 啟動 ----------
  window.__pt = { S, loadSample, addSample, detect, relearn, rankPerson, ready: false };
  render(); renderPanel(); renderStats(); renderDB();
  setStatus('模型載入中…');
  V.load().then(() => setStatus(modelStatus()));
})();
