// ==========================================================
// click clip!! — 玩家外觀屬性（髮長、髮色、上衣顏色、褲子顏色）
//   輸入：一張照片的像素 ＋ MoveNet 的 17 個關節點
//   作法：
//     · 上衣：取「兩肩＋兩髖」圍成的四邊形，往中心縮 15%（避開背景與手臂）
//     · 褲子：沿著「髖 → 膝」兩條大腿線取樣（兩腿之間是背景，所以不取整個方框）
//     · 主色：把取樣像素轉成 Lab 色彩（接近人眼感受的色彩空間），分 3 群（k-means），取最大的一群
//     · 顏色名稱：HSV 規則（黑白灰、色相分段、深淺）＋ 你修正過的樣本（最近鄰學習）
//     · 髮長：
//         ① 頭頂（眼睛上方）取樣 → 髮色
//         ② 臉部（兩眼與鼻子之間）取樣 → 膚色
//         ③ 從頭頂往下「長」出髮色區域（不越過膚色、只在頭部左右範圍內）
//         ④ 頭髮最低點落在哪：耳下以上＝短髮、下巴到肩膀＝中長髮、過肩＝長髮
//            比例 r ＝（頭髮最低點 − 眼睛高度）÷（肩膀高度 − 眼睛高度）
//       髮色和上衣或背景太像時，會標示「信心低」
// ==========================================================
(function () {
  'use strict';
  const CC = (window.CC = window.CC || {});
  const A = (CC.Attr = {});
  const KP = { nose: 0, le: 1, re: 2, lea: 3, rea: 4, ls: 5, rs: 6, lel: 7, rel: 8, lw: 9, rw: 10, lh: 11, rh: 12, lk: 13, rk: 14, la: 15, ra: 16 };
  const ok = (k) => k && k.s >= 0.3;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const r3 = (v) => Math.round(v * 1000) / 1000;

  // ---------- 色彩 ----------
  function rgb2hsv(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
    let h = 0;
    if (d > 1e-6) { if (mx === r) h = ((g - b) / d + 6) % 6; else if (mx === g) h = (b - r) / d + 2; else h = (r - g) / d + 4; h *= 60; }
    return [h, mx ? d / mx : 0, mx];
  }
  function rgb2lab(r, g, b) {
    const f = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
    const R = f(r), G = f(g), B = f(b);
    let x = (R * 0.4124 + G * 0.3576 + B * 0.1805) / 0.95047, y = R * 0.2126 + G * 0.7152 + B * 0.0722, z = (R * 0.0193 + G * 0.1192 + B * 0.9505) / 1.08883;
    const t = (v) => (v > 0.008856 ? Math.cbrt(v) : 7.787 * v + 16 / 116);
    x = t(x); y = t(y); z = t(z);
    return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
  }
  const dE = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  A.rgb2lab = rgb2lab; A.dE = dE;

  A.COLORS = [
    ['黑', '#1b1b1b'], ['深灰', '#545454'], ['灰', '#9a9a9a'], ['白', '#f2f2f0'], ['紅', '#c62828'], ['酒紅', '#7b1f2a'], ['粉紅', '#f19bbd'],
    ['橘', '#ef6c00'], ['棕', '#6d4c41'], ['卡其', '#c3b091'], ['黃', '#f2c12e'], ['綠', '#2e8b3e'], ['軍綠', '#5b6b2f'],
    ['淺藍', '#86b6e8'], ['藍', '#1f62c4'], ['深藍', '#1f2a52'], ['紫', '#6f2b9a'],
  ];
  A.HAIR_LEN = { short: '短髮', mid: '中長髮', long: '長髮', unknown: '無法判斷' };

  // 規則命名
  function ruleName(rgb) {
    const [h, s, v] = rgb2hsv(rgb[0], rgb[1], rgb[2]);
    if (v < 0.18) return '黑';
    if (s < 0.14 || (s < 0.2 && v > 0.85)) return v > 0.82 ? '白' : v > 0.5 ? '灰' : v > 0.25 ? '深灰' : '黑';
    if (h < 12 || h >= 340) return v < 0.56 && s > 0.45 ? '酒紅' : s < 0.5 && v > 0.7 ? '粉紅' : '紅';
    if (h < 40) return v < 0.55 ? '棕' : s < 0.45 ? '卡其' : '橘';
    if (h < 68) return s < 0.45 ? '卡其' : v < 0.5 ? '軍綠' : '黃';
    if (h < 165) return v < 0.45 || s < 0.38 ? '軍綠' : '綠';
    if (h < 255) return v < 0.4 ? '深藍' : s < 0.45 && v > 0.62 ? '淺藍' : '藍';
    if (h < 290) return '紫';
    return v > 0.6 ? '粉紅' : '紫';
  }
  // 規則 ＋ 學習（你修正過的顏色樣本，最近 3 個投票）
  A.nameColor = function (rgb, learned) {
    const base = ruleName(rgb);
    if (!learned || !learned.length) return { name: base, by: 'rule' };
    const lab = rgb2lab(rgb[0], rgb[1], rgb[2]);
    const near = learned.map((q) => ({ n: q.name, d: dE(lab, q.lab) })).sort((a, b) => a.d - b.d).slice(0, 3).filter((q) => q.d < 14);
    if (!near.length) return { name: base, by: 'rule' };
    const vote = {};
    for (const q of near) vote[q.n] = (vote[q.n] || 0) + 1 / (1 + q.d);
    const best = Object.keys(vote).sort((a, b) => vote[b] - vote[a])[0];
    return { name: best, by: best === base ? 'rule' : 'learned' };
  };

  // 主色：Lab 空間 k-means（k = 3），回傳最大群
  function dominant(samples) {
    if (samples.length < 15) return null;
    const pts = samples.map((c) => ({ rgb: c, lab: rgb2lab(c[0], c[1], c[2]) }));
    const K = Math.min(3, pts.length);
    const cents = [pts[Math.floor(pts.length / 2)].lab.slice()];
    while (cents.length < K) { // 最遠點初始化
      let best = null, bd = -1;
      for (const p of pts) { const d = Math.min(...cents.map((c) => dE(c, p.lab))); if (d > bd) { bd = d; best = p; } }
      cents.push(best.lab.slice());
    }
    let assign = new Array(pts.length).fill(0);
    for (let it = 0; it < 8; it++) {
      assign = pts.map((p) => { let bi = 0, bd = 1e9; cents.forEach((c, i) => { const d = dE(c, p.lab); if (d < bd) { bd = d; bi = i; } }); return bi; });
      for (let i = 0; i < K; i++) {
        const m = pts.filter((_, j) => assign[j] === i);
        if (m.length) cents[i] = [0, 1, 2].map((q) => m.reduce((s, p) => s + p.lab[q], 0) / m.length);
      }
    }
    const counts = new Array(K).fill(0); assign.forEach((a) => counts[a]++);
    // 同一種顏色常被分成好幾群（光影、皺摺）：色差 < 12 的群合併計算
    const near = (i) => cents.map((c, j) => j).filter((j) => counts[j] && dE(cents[i], cents[j]) < 12);
    const merged = counts.map((c, i) => near(i).reduce((s, j) => s + counts[j], 0));
    const top = merged.map((c, i) => i).sort((a, b) => merged[b] - merged[a] || counts[b] - counts[a])[0];
    const grp = near(top);
    const mem = pts.filter((_, j) => grp.includes(assign[j]));
    const rgb = [0, 1, 2].map((q) => Math.round(mem.reduce((s, p) => s + p.rgb[q], 0) / mem.length));
    const lab = rgb2lab(rgb[0], rgb[1], rgb[2]);
    // 群內差異（越小越單純）
    const spread = mem.reduce((s, p) => s + dE(p.lab, lab), 0) / mem.length;
    return { rgb, lab, share: mem.length / pts.length, spread };
  }

  const px = (data, W, x, y) => { const i = (Math.round(y) * W + Math.round(x)) * 4; return [data[i], data[i + 1], data[i + 2]]; };
  function inImg(W, H, x, y) { return x >= 0 && y >= 0 && x < W && y < H; }

  // 四邊形內取樣（雙線性），shrink：往中心縮的比例
  function sampleQuad(data, W, H, q, shrink, n) {
    const c = [(q[0].x + q[1].x + q[2].x + q[3].x) / 4, (q[0].y + q[1].y + q[2].y + q[3].y) / 4];
    const Q = q.map((p) => ({ x: c[0] + (p.x - c[0]) * (1 - shrink), y: c[1] + (p.y - c[1]) * (1 - shrink) }));
    const out = [], g = Math.ceil(Math.sqrt(n));
    for (let i = 0; i < g; i++) for (let j = 0; j < g; j++) {
      const u = (i + 0.5) / g, v = (j + 0.5) / g;
      const top = { x: Q[0].x + (Q[1].x - Q[0].x) * u, y: Q[0].y + (Q[1].y - Q[0].y) * u };
      const bot = { x: Q[3].x + (Q[2].x - Q[3].x) * u, y: Q[3].y + (Q[2].y - Q[3].y) * u };
      const x = top.x + (bot.x - top.x) * v, y = top.y + (bot.y - top.y) * v;
      if (inImg(W, H, x, y)) out.push(px(data, W, x, y));
    }
    return { pts: out, poly: Q };
  }
  // 沿線段取樣（寬 width）
  function sampleStrip(data, W, H, a, b, width, n) {
    const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy) || 1, nx = -dy / L, ny = dx / L, out = [];
    const rows = Math.max(4, Math.round(n / 5));
    for (let i = 0; i < rows; i++) for (let j = -2; j <= 2; j++) {
      const t = 0.12 + 0.76 * (i + 0.5) / rows, s = (j / 2) * width * 0.5;
      const x = a.x + dx * t + nx * s, y = a.y + dy * t + ny * s;
      if (inImg(W, H, x, y)) out.push(px(data, W, x, y));
    }
    const hw = width / 2;
    return { pts: out, poly: [{ x: a.x + nx * hw, y: a.y + ny * hw }, { x: b.x + nx * hw, y: b.y + ny * hw }, { x: b.x - nx * hw, y: b.y - ny * hw }, { x: a.x - nx * hw, y: a.y - ny * hw }] };
  }
  function sampleRect(data, W, H, x0, y0, x1, y1, n) {
    const out = [], g = Math.ceil(Math.sqrt(n));
    for (let i = 0; i < g; i++) for (let j = 0; j < g; j++) {
      const x = x0 + (x1 - x0) * (i + 0.5) / g, y = y0 + (y1 - y0) * (j + 0.5) / g;
      if (inImg(W, H, x, y)) out.push(px(data, W, x, y));
    }
    return out;
  }

  // ---------- 主函式 ----------
  // data/W/H：ImageData；person：{ kp:[{x,y,s}]×17, box }；model：{ colors:[{lab,name}], hair:{t1,t2} }
  A.describe = function (data, W, H, person, model) {
    model = model || {};
    const kp = person.kp, out = { regions: {}, warn: [] };
    if (!kp) { out.warn.push('沒有關節點，無法判斷'); return out; }
    const ls = kp[KP.ls], rs = kp[KP.rs], lh = kp[KP.lh], rh = kp[KP.rh];
    const haveSh = ok(ls) && ok(rs), haveHip = ok(lh) || ok(rh);
    const shW = haveSh ? Math.hypot(ls.x - rs.x, ls.y - rs.y) : null;
    const shY = haveSh ? (ls.y + rs.y) / 2 : null;

    // 上衣
    if (haveSh && haveHip) {
      const L = ok(lh) ? lh : rh, R = ok(rh) ? rh : lh;
      let q = [ls, rs, R, L].map((p) => ({ x: p.x, y: p.y }));
      // 側身：肩膀很窄 → 用軀幹中線左右各 0.22 倍軀幹長
      const tl = Math.hypot((L.x + R.x) / 2 - (ls.x + rs.x) / 2, (L.y + R.y) / 2 - shY);
      if (shW < tl * 0.35) {
        const cx = (ls.x + rs.x + L.x + R.x) / 4, hw = tl * 0.22;
        q = [{ x: cx - hw, y: shY }, { x: cx + hw, y: shY }, { x: cx + hw, y: (L.y + R.y) / 2 }, { x: cx - hw, y: (L.y + R.y) / 2 }];
      }
      // 上緣往下移一點（避開領口與脖子）
      q[0].y += (q[3].y - q[0].y) * 0.1; q[1].y += (q[2].y - q[1].y) * 0.1;
      const s = sampleQuad(data, W, H, q, 0.18, 400);
      const dm = dominant(s.pts);
      out.regions.top = s.poly;
      if (dm) { const nm = A.nameColor(dm.rgb, model.colors); out.top = { rgb: dm.rgb, lab: dm.lab, name: nm.name, by: nm.by, conf: r3(clamp(dm.share * 1.2, 0, 1)), share: r3(dm.share) }; }
    } else out.warn.push('看不到肩膀或髖部：無法判斷上衣');

    // 褲子（沿兩條大腿；膝蓋看不到就用軀幹長往下推）
    if (haveHip) {
      const tl = haveSh ? Math.hypot(((ok(lh) ? lh.x : rh.x) + (ok(rh) ? rh.x : lh.x)) / 2 - (ls.x + rs.x) / 2, ((ok(lh) ? lh.y : rh.y) + (ok(rh) ? rh.y : lh.y)) / 2 - shY) : 80;
      const hipW = ok(lh) && ok(rh) ? Math.abs(lh.x - rh.x) : (shW || tl) * 0.6;
      const width = Math.max(4, Math.min(hipW * 0.5, tl * 0.28));
      const pts = [], polys = [];
      for (const [h, k, a] of [[lh, kp[KP.lk], kp[KP.la]], [rh, kp[KP.rk], kp[KP.ra]]]) {
        if (!ok(h)) continue;
        const end = ok(k) ? k : { x: h.x, y: h.y + tl * 0.9 };
        const s = sampleStrip(data, W, H, h, end, width, 140);
        pts.push(...s.pts); polys.push(s.poly);
        if (ok(k) && ok(a)) { const s2 = sampleStrip(data, W, H, k, a, width * 0.85, 70); pts.push(...s2.pts); polys.push(s2.poly); }
      }
      const dm = dominant(pts);
      out.regions.pants = polys;
      if (dm) {
        const nm = A.nameColor(dm.rgb, model.colors);
        out.pants = { rgb: dm.rgb, lab: dm.lab, name: nm.name, by: nm.by, conf: r3(clamp(dm.share * 1.2, 0, 1)), share: r3(dm.share) };
        if (out.top && dE(out.top.lab, dm.lab) < 8) out.warn.push('上衣和褲子顏色幾乎一樣（可能是連身或同色系）');
      }
    } else out.warn.push('看不到髖部：無法判斷褲子');

    out.hair = hairLength(data, W, H, kp, out, model);
    return out;
  };

  // ---------- 髮長 ----------
  function hairLength(data, W, H, kp, out, model) {
    const face = [KP.nose, KP.le, KP.re].map((i) => kp[i]).filter(ok);
    const ears = [KP.lea, KP.rea].map((i) => kp[i]).filter(ok);
    const ls = kp[KP.ls], rs = kp[KP.rs];
    if (!(ok(ls) && ok(rs))) return { len: 'unknown', why: '看不到肩膀' };
    const shY = (ls.y + rs.y) / 2, shW = Math.hypot(ls.x - rs.x, ls.y - rs.y);
    const headPts = face.concat(ears);
    if (!headPts.length) return { len: 'unknown', why: '看不到頭部' };
    const back = face.length === 0; // 只看得到耳朵：背面
    const hcX = headPts.reduce((s, p) => s + p.x, 0) / headPts.length;
    const eyes = [kp[KP.le], kp[KP.re]].filter(ok);
    let eyeY = eyes.length ? eyes.reduce((s, p) => s + p.y, 0) / eyes.length : ears.length ? ears.reduce((s, p) => s + p.y, 0) / ears.length : headPts[0].y;
    const earD = ears.length === 2 ? Math.abs(ears[0].x - ears[1].x) : 0;
    const eyeD = eyes.length === 2 ? Math.abs(eyes[0].x - eyes[1].x) : 0;
    const headW = Math.max(earD * 1.1, eyeD * 2.4, shW * 0.36, (shY - eyeY) * 0.6, 6);
    const span = Math.max(4, shY - eyeY); // 眼睛到肩膀

    // ① 頭頂髮色
    // 眼睛到頭頂約 0.65 倍頭寬；額頭（皮膚）在眼睛上方 0.35 倍頭寬以內
    const cy0 = eyeY - headW * 0.72, cy1 = eyeY - headW * 0.36;
    const crown = sampleRect(data, W, H, hcX - headW * 0.28, Math.max(0, cy0), hcX + headW * 0.28, Math.max(1, cy1), 120);
    const hairDm = dominant(crown);
    if (!hairDm || crown.length < 20) return { len: 'unknown', why: '頭頂超出畫面' };
    // ② 膚色
    let skin = null;
    if (!back && ok(kp[KP.nose])) {
      const n = kp[KP.nose], s = sampleRect(data, W, H, n.x - headW * 0.14, (eyeY + n.y) / 2 - headW * 0.05, n.x + headW * 0.14, n.y + headW * 0.05, 60);
      const d = dominant(s); if (d) skin = d.lab;
    }
    // 背景（人物左右兩側、頭的高度）
    const bgs = sampleRect(data, W, H, hcX - shW * 1.6, eyeY - headW * 0.4, hcX - shW * 1.1, eyeY + headW * 0.4, 40)
      .concat(sampleRect(data, W, H, hcX + shW * 1.1, eyeY - headW * 0.4, hcX + shW * 1.6, eyeY + headW * 0.4, 40));
    const bg = dominant(bgs);
    const hairLab = hairDm.lab, tol = clamp(14 + hairDm.spread * 1.2, 14, 30);
    const isHair = (lab) => { const dh = dE(lab, hairLab); if (dh > tol) return false; if (skin && dE(lab, skin) < dh + 4) return false; return true; };

    // ③ 從頭頂往下長
    const x0 = Math.max(0, Math.floor(hcX - headW * 0.85)), x1 = Math.min(W - 1, Math.ceil(hcX + headW * 0.85));
    const yTop = Math.max(0, Math.floor(cy0)), yBot = Math.min(H - 1, Math.ceil(shY + span * 1.6));
    const step = Math.max(1, Math.round(headW / 28));
    const gw = Math.floor((x1 - x0) / step) + 1, gh = Math.floor((yBot - yTop) / step) + 1;
    const lab = new Array(gw * gh), mask = new Uint8Array(gw * gh);
    for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) { const c = px(data, W, x0 + i * step, yTop + j * step); lab[j * gw + i] = rgb2lab(c[0], c[1], c[2]); }
    const st = [];
    for (let j = 0; j < gh; j++) {
      const y = yTop + j * step; if (y > cy1) break;
      for (let i = 0; i < gw; i++) { const x = x0 + i * step; if (Math.abs(x - hcX) < headW * 0.3 && isHair(lab[j * gw + i])) { mask[j * gw + i] = 1; st.push(j * gw + i); } }
    }
    // 上衣太像頭髮 → 肩膀以下不往下長（避免把黑上衣當長髮）
    const topLike = out.top && dE(out.top.lab, hairLab) < 16;
    const neckHalf = headW * 0.22;
    while (st.length) {
      const k = st.pop(), i = k % gw, j = (k / gw) | 0;
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const ii = i + di, jj = j + dj; if (ii < 0 || jj < 0 || ii >= gw || jj >= gh) continue;
        const q = jj * gw + ii; if (mask[q]) continue;
        const y = yTop + jj * step, x = x0 + ii * step;
        if (topLike && y > shY - span * 0.05) continue;
        if (y > eyeY + span * 0.35 && Math.abs(x - hcX) < neckHalf && !back) continue; // 脖子正中央通常是皮膚或衣領
        if (isHair(lab[q])) { mask[q] = 1; st.push(q); }
      }
    }
    // ④ 頭髮最低點：該列的頭髮像素比例夠多
    let bottom = eyeY;
    for (let j = gh - 1; j >= 0; j--) {
      let c = 0; for (let i = 0; i < gw; i++) c += mask[j * gw + i];
      if (c >= Math.max(2, gw * 0.08)) { bottom = yTop + j * step; break; }
    }
    const r = (bottom - eyeY) / span;
    const t1 = (model.hair && model.hair.t1) || 0.5, t2 = (model.hair && model.hair.t2) || 1.0;
    const len = r < t1 ? 'short' : r < t2 ? 'mid' : 'long';
    let conf = 0.85;
    const warn = [];
    if (topLike) { conf = 0.45; warn.push('髮色和上衣很像：過肩的長髮可能偵測不到'); }
    if (bg && dE(bg.lab, hairLab) < 15) { conf = Math.min(conf, 0.45); warn.push('髮色和背景很像：可能高估髮長'); }
    if (back) warn.push('背面'); else if (!skin) warn.push('看不到臉，無法排除膚色');
    if (Math.abs(r - t1) < 0.08 || Math.abs(r - t2) < 0.08) conf = Math.min(conf, 0.55);
    const hn = A.nameColor(hairDm.rgb, model.colors);
    out.warn.push(...warn);
    // 給畫面用：頭髮遮罩（原圖座標的點）
    const maskPts = [];
    for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) if (mask[j * gw + i]) maskPts.push([x0 + i * step, yTop + j * step]);
    out.regions.hair = { pts: maskPts, step, eyeY, shY, bottom, crown: [hcX - headW * 0.28, Math.max(0, cy0), hcX + headW * 0.28, cy1] };
    return { len, ratio: r3(r), rgb: hairDm.rgb, lab: hairDm.lab, name: hn.name, conf: r3(conf), back };
  }

  // ---------- 學習 ----------
  // samples：[{ attrs（自動量到的值）, label:{ hairLen, hairColor, top, pants } }]
  // 顏色：把你修正過的顏色加進最近鄰清單；髮長：依每類樣本的比例中位數重新決定兩個門檻
  A.learn = function (samples) {
    const colors = [];
    for (const s of samples) {
      if (s.attrs.top && s.label.top) colors.push({ lab: s.attrs.top.lab, name: s.label.top });
      if (s.attrs.pants && s.label.pants) colors.push({ lab: s.attrs.pants.lab, name: s.label.pants });
      if (s.attrs.hair && s.attrs.hair.lab && s.label.hairColor) colors.push({ lab: s.attrs.hair.lab, name: s.label.hairColor });
    }
    const by = { short: [], mid: [], long: [] };
    for (const s of samples) if (s.attrs.hair && s.attrs.hair.ratio != null && by[s.label.hairLen]) by[s.label.hairLen].push(s.attrs.hair.ratio);
    const med = (a) => { const v = a.slice().sort((p, q) => p - q); return v[v.length >> 1]; };
    let t1 = 0.5, t2 = 1.0, learned = false;
    if (by.short.length >= 2 && by.mid.length >= 2) { t1 = (med(by.short) + med(by.mid)) / 2; learned = true; }
    if (by.mid.length >= 2 && by.long.length >= 2) { t2 = (med(by.mid) + med(by.long)) / 2; learned = true; }
    if (t2 <= t1 + 0.1) t2 = t1 + 0.1;
    return { colors, hair: { t1: r3(t1), t2: r3(t2), learned }, n: samples.length };
  };
})();
