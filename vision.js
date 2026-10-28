// ==========================================================
// click clip!! — 鏡頭視覺模組
//   1. 人物偵測：MoveNet MultiPose（就是 ml5.js bodyPose 用的模型），抓出頭、軀幹、四肢 17 個關節點
//      → 直接用 TF.js 載入，不再另外載 ml5（ml5 內含第二份 TensorFlow，手機會更卡）
//      → 模型載不到時退回 COCO-SSD 的人物框
//   2. 外觀特徵：頭髮（頭頂區）、上衣（肩到髖）、褲子（髖到膝）的顏色分布 + 體型比例（肩寬/軀幹長＝胖瘦）+ 身高
//      → 用來判斷鏡頭裡的人是 A～Z 哪一位
//   3. 符號偵測：找畫面中的紅色梯形（你設計的符號），算出它在地上的位置，用來校正定位
//   4. 光流：比較前後兩格畫面的區塊位移（扣掉手機轉動造成的位移），
//      畫面「放大」= 前進、「縮小（zoom out）」= 後退、整體橫移 = 左右平移
// ==========================================================
(function () {
  'use strict';
  const CC = (window.CC = window.CC || {});
  const V = (CC.Vision = {});

  // ---------------------------------------------------------
  // 1. 人物偵測
  // ---------------------------------------------------------
  V.status = 'idle';   // idle | loading | movenet | coco | none
  V.statusText = '尚未載入';
  let detector = null, coco = null, loading = null;

  V.load = function () {
    if (loading) return loading;
    V.status = 'loading'; V.statusText = '人物偵測模型載入中…';
    loading = (async () => {
      try {
        if (window.tf && window.tf.ready) await window.tf.ready();
        if (window.poseDetection && window.tf) {
          const pd = window.poseDetection;
          detector = await pd.createDetector(pd.SupportedModels.MoveNet, {
            modelType: pd.movenet.modelType.MULTIPOSE_LIGHTNING,
            enableTracking: false,
            multiPoseMaxDimension: 256,
          });
          V.status = 'movenet'; V.statusText = 'MoveNet 人體骨架';
          return V.status;
        }
      } catch (e) { console.warn('MoveNet 載入失敗', e); }
      try {
        if (window.cocoSsd) {
          coco = await window.cocoSsd.load({ base: 'lite_mobilenet_v2' });
          V.status = 'coco'; V.statusText = 'COCO-SSD 人物框（無骨架）';
          return V.status;
        }
      } catch (e) { console.warn('COCO-SSD 載入失敗', e); }
      V.status = 'none'; V.statusText = '沒有人物偵測模型（網路被擋？）';
      return V.status;
    })();
    return loading;
  };

  // 物件（桌椅）偵測：與人物偵測分開、低頻率
  let cocoObj = null, cocoObjLoading = null;
  V.loadObjects = function () {
    if (cocoObjLoading) return cocoObjLoading;
    cocoObjLoading = (async () => {
      try {
        if (coco) { cocoObj = coco; return true; }
        if (window.cocoSsd) { cocoObj = await window.cocoSsd.load({ base: 'lite_mobilenet_v2' }); return true; }
      } catch (e) { console.warn('COCO-SSD（物件）載入失敗', e); }
      return false;
    })();
    return cocoObjLoading;
  };
  V.detectObjects = async function (canvas) {
    if (!cocoObj) return [];
    try { return await cocoObj.detect(canvas, 12, 0.4); } catch (e) { return []; }
  };

  const KP = { nose: 0, le: 1, re: 2, lea: 3, rea: 4, ls: 5, rs: 6, lel: 7, rel: 8, lw: 9, rw: 10, lh: 11, rh: 12, lk: 13, rk: 14, la: 15, ra: 16 };
  V.KP = KP;
  const KP_MIN = 0.3;

  // 回傳 [{ box:{x,y,w,h}, kp:[{x,y,s}]|null, score }]（座標＝輸入 canvas 的像素）
  V.detectPersons = async function (canvas) {
    if (detector) {
      let poses = [];
      try { poses = await detector.estimatePoses(canvas, { maxPoses: 6, flipHorizontal: false }); } catch (e) { return []; }
      const out = [];
      for (const p of poses) {
        const kp = p.keypoints.map((k) => ({ x: k.x, y: k.y, s: k.score == null ? 0 : k.score }));
        const box = boxFromKp(kp, canvas.width, canvas.height);
        if (!box) continue;
        const sc = p.score == null ? avgScore(kp) : p.score;
        if (sc < 0.22) continue;
        out.push({ box, kp, score: sc });
      }
      return out;
    }
    if (coco) {
      let preds = [];
      try { preds = await coco.detect(canvas, 8, 0.4); } catch (e) { return []; }
      return preds.filter((p) => p.class === 'person').map((p) => ({ box: { x: p.bbox[0], y: p.bbox[1], w: p.bbox[2], h: p.bbox[3] }, kp: null, score: p.score }));
    }
    return [];
  };

  function avgScore(kp) { let s = 0; for (const k of kp) s += k.s; return s / kp.length; }
  function ok(k) { return k && k.s >= KP_MIN; }
  function mid(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, s: Math.min(a.s, b.s) }; }

  function boxFromKp(kp, W, H) {
    const good = kp.filter(ok);
    if (good.length < 4) return null;
    let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (const k of good) { x0 = Math.min(x0, k.x); x1 = Math.max(x1, k.x); y0 = Math.min(y0, k.y); y1 = Math.max(y1, k.y); }
    // 頭頂在鼻子上方約「鼻到肩」的 0.9 倍；腳底在腳踝下方一點
    const sh = ok(kp[KP.ls]) && ok(kp[KP.rs]) ? mid(kp[KP.ls], kp[KP.rs]) : null;
    if (ok(kp[KP.nose]) && sh) y0 = Math.min(y0, kp[KP.nose].y - Math.abs(sh.y - kp[KP.nose].y) * 0.9);
    const ank = [kp[KP.la], kp[KP.ra]].filter(ok);
    if (ank.length) y1 = Math.max(y1, Math.max(...ank.map((k) => k.y)) + (y1 - y0) * 0.03);
    const padX = (x1 - x0) * 0.18 + 4;
    x0 = Math.max(0, x0 - padX); x1 = Math.min(W, x1 + padX);
    y0 = Math.max(0, y0); y1 = Math.min(H, y1);
    if (x1 - x0 < 6 || y1 - y0 < 12) return null;
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  }
  V.boxFromKp = boxFromKp;

  // 頭頂與腳底的像素（給引擎算距離與身高）
  V.headFoot = function (person) {
    const kp = person.kp, b = person.box;
    if (!kp) return { head: { x: b.x + b.w / 2, y: b.y }, foot: { x: b.x + b.w / 2, y: b.y + b.h }, footSeen: b.y + b.h < Infinity };
    const ank = [kp[KP.la], kp[KP.ra]].filter(ok);
    const foot = ank.length ? { x: ank.reduce((s, k) => s + k.x, 0) / ank.length, y: Math.max(...ank.map((k) => k.y)) + b.h * 0.02 } : null;
    return { head: { x: b.x + b.w / 2, y: b.y }, foot, footSeen: !!foot };
  };

  // ---------------------------------------------------------
  // 2. 外觀特徵
  // ---------------------------------------------------------
  const NB = 15; // 12 個色相 + 暗/中/亮 3 個無彩

  function regionStats(data, W, H, x0, y0, x1, y1) {
    x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0));
    x1 = Math.min(W, Math.ceil(x1)); y1 = Math.min(H, Math.ceil(y1));
    const area = (x1 - x0) * (y1 - y0);
    if (x1 - x0 < 2 || y1 - y0 < 2) return null;
    const step = Math.max(1, Math.floor(Math.sqrt(area / 500)));
    const h = new Float32Array(NB);
    let n = 0, cr = 0, cg = 0, vv = 0, tw = 0;
    for (let y = y0; y < y1; y += step) {
      for (let x = x0; x < x1; x += step) {
        const i = (y * W + x) * 4;
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
        const v = mx / 255, s = mx ? (mx - mn) / mx : 0;
        if (s < 0.2 || v < 0.13) {
          h[v < 0.3 ? 12 : v < 0.66 ? 13 : 14] += 1;
          tw += 1;
        } else {
          let hue;
          const d = mx - mn;
          if (mx === r) hue = ((g - b) / d + 6) % 6;
          else if (mx === g) hue = (b - r) / d + 2;
          else hue = (r - g) / d + 4;
          const wgt = 0.5 + 0.5 * s;
          h[Math.min(11, Math.floor(hue * 2))] += wgt;
          tw += wgt;
        }
        const sum = r + g + b + 1;
        cr += r / sum; cg += g / sum; vv += (r + g + b) / 765;
        n++;
      }
    }
    if (n < 12) return null;
    for (let k = 0; k < NB; k++) h[k] /= tw || 1;
    return { h: Array.from(h, (q) => Math.round(q * 1000) / 1000), m: [r3(cr / n), r3(cg / n), r3(vv / n)] };
  }
  const r3 = (v) => Math.round(v * 1000) / 1000;

  // data/W/H：ImageData；person：detectPersons 的一筆（座標同一張圖）；extra：{ height }
  V.features = function (data, W, H, person, extra) {
    const kp = person.kp, b = person.box;
    let head = null, torso = null, legs = null, shape = null;
    if (kp && ok(kp[KP.ls]) && ok(kp[KP.rs]) && (ok(kp[KP.lh]) || ok(kp[KP.rh]))) {
      const ls = kp[KP.ls], rs = kp[KP.rs];
      const sm = mid(ls, rs);
      const lh = ok(kp[KP.lh]) ? kp[KP.lh] : kp[KP.rh], rh = ok(kp[KP.rh]) ? kp[KP.rh] : kp[KP.lh];
      const hm = mid(lh, rh);
      const tl = Math.max(8, Math.hypot(hm.x - sm.x, hm.y - sm.y));
      const sw = Math.hypot(ls.x - rs.x, ls.y - rs.y);
      let tx0 = Math.min(ls.x, rs.x, lh.x, rh.x), tx1 = Math.max(ls.x, rs.x, lh.x, rh.x);
      if (tx1 - tx0 < tl * 0.35) { tx0 = sm.x - tl * 0.22; tx1 = sm.x + tl * 0.22; } // 側身
      else { const inset = (tx1 - tx0) * 0.15; tx0 += inset; tx1 -= inset; }
      torso = regionStats(data, W, H, tx0, sm.y + tl * 0.12, tx1, hm.y - tl * 0.06);
      const hipW = Math.max(Math.abs(lh.x - rh.x), sw * 0.5, tl * 0.3);
      const knees = [kp[KP.lk], kp[KP.rk]].filter(ok);
      const ly1 = knees.length ? knees.reduce((s, k) => s + k.y, 0) / knees.length : hm.y + tl * 0.8;
      legs = regionStats(data, W, H, hm.x - hipW * 0.6, hm.y + tl * 0.1, hm.x + hipW * 0.6, ly1);
      const face = [KP.nose, KP.le, KP.re, KP.lea, KP.rea].map((i) => kp[i]).filter(ok);
      if (face.length >= 2) {
        const cx = face.reduce((s, k) => s + k.x, 0) / face.length, cy = face.reduce((s, k) => s + k.y, 0) / face.length;
        const eyeD = ok(kp[KP.le]) && ok(kp[KP.re]) ? Math.abs(kp[KP.le].x - kp[KP.re].x) : 0;
        const s = Math.max(sw * 0.5, eyeD * 2.6, tl * 0.32);
        head = regionStats(data, W, H, cx - s * 0.45, cy - s * 0.95, cx + s * 0.45, cy - s * 0.3);
      }
      const ank = [kp[KP.la], kp[KP.ra]].filter(ok);
      const legLen = ank.length ? Math.abs(ank.reduce((s, k) => s + k.y, 0) / ank.length - hm.y) : null;
      shape = { sw: r3(sw / tl), lg: legLen ? r3(legLen / tl) : null };
    } else {
      torso = regionStats(data, W, H, b.x + b.w * 0.25, b.y + b.h * 0.22, b.x + b.w * 0.75, b.y + b.h * 0.5);
      legs = regionStats(data, W, H, b.x + b.w * 0.28, b.y + b.h * 0.56, b.x + b.w * 0.72, b.y + b.h * 0.85);
      head = regionStats(data, W, H, b.x + b.w * 0.36, b.y, b.x + b.w * 0.64, b.y + b.h * 0.08);
    }
    if (!torso && !legs) return null;
    return { v: 2, head, torso, legs, shape, hgt: extra && extra.height ? r3(extra.height) : null, src: kp ? 'kp' : 'box' };
  };

  function histInter(a, b) { let s = 0; for (let k = 0; k < NB; k++) s += Math.min(a[k] || 0, b[k] || 0); return s; }
  function regionSim(a, b) {
    if (!a || !b) return null;
    const hi = histInter(a.h, b.h);
    const dc = Math.hypot(a.m[0] - b.m[0], a.m[1] - b.m[1]) * 6 + Math.abs(a.m[2] - b.m[2]) * 0.9;
    return 0.62 * hi + 0.38 * Math.max(0, 1 - dc);
  }

  // 0～1，越高越像
  V.similarity = function (f, g) {
    if (!f || !g) return 0;
    const parts = [[regionSim(f.torso, g.torso), 0.45], [regionSim(f.legs, g.legs), 0.3], [regionSim(f.head, g.head), 0.15]];
    if (f.shape && g.shape) {
      let s = 1 - Math.min(1, Math.abs(f.shape.sw - g.shape.sw) / 0.45);
      if (f.shape.lg && g.shape.lg) s = 0.6 * s + 0.4 * (1 - Math.min(1, Math.abs(f.shape.lg - g.shape.lg) / 0.6));
      parts.push([s, 0.1]);
    }
    if (f.hgt && g.hgt) parts.push([1 - Math.min(1, Math.abs(f.hgt - g.hgt) / 0.25), 0.12]);
    let sw = 0, sv = 0;
    for (const [v, w] of parts) if (v != null) { sv += v * w; sw += w; }
    return sw ? sv / sw : 0;
  };

  // profiles：{ A: [{feat}], B: [...] }；ids：候選編號（null = 全部）
  V.identify = function (feat, profiles, ids) {
    const out = [];
    for (const id in profiles) {
      if (ids && !ids.includes(id)) continue;
      let best = 0;
      for (const p of profiles[id] || []) if (p.feat) best = Math.max(best, V.similarity(feat, p.feat));
      out.push({ id, sim: best });
    }
    out.sort((a, b) => b.sim - a.sim);
    return out;
  };

  // ---------------------------------------------------------
  // 3. 符號偵測（紅色梯形 + 白色斜線，黑底）
  // ---------------------------------------------------------
  V.symbolColor = [180, 65, 34];
  function isRed(r, g, b) { return r >= 78 && r >= g * 1.6 && r >= b * 1.35 && r - g >= 42; }

  V.detectSymbol = function (data, W, H) {
    const n = W * H, mask = new Uint8Array(n);
    let any = 0;
    for (let k = 0; k < n; k++) { const i = 4 * k; if (isRed(data[i], data[i + 1], data[i + 2])) { mask[k] = 1; any++; } }
    if (any < 8) return null;
    const lab = new Int32Array(n), queue = new Int32Array(n);
    let best = null, nl = 0;
    for (let k0 = 0; k0 < n; k0++) {
      if (!mask[k0] || lab[k0]) continue;
      nl++;
      let qh = 0, qt = 0, area = 0, sx = 0, sy = 0, x0 = W, x1 = 0, y0 = H, y1 = 0;
      lab[k0] = nl; queue[qt++] = k0;
      while (qh < qt) {
        const k = queue[qh++], x = k % W, y = (k / W) | 0;
        area++; sx += x; sy += y;
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
        if (x > 0 && mask[k - 1] && !lab[k - 1]) { lab[k - 1] = nl; queue[qt++] = k - 1; }
        if (x < W - 1 && mask[k + 1] && !lab[k + 1]) { lab[k + 1] = nl; queue[qt++] = k + 1; }
        if (y > 0 && mask[k - W] && !lab[k - W]) { lab[k - W] = nl; queue[qt++] = k - W; }
        if (y < H - 1 && mask[k + W] && !lab[k + W]) { lab[k + W] = nl; queue[qt++] = k + W; }
      }
      if (area < 10 || area > n * 0.3) continue;
      const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
      const fill = area / (bw * bh);
      if (fill < 0.3 || fill > 0.97) continue;
      // 周圍一圈：黑底比例 + 白色斜線
      const px = Math.max(2, Math.round(bw * 0.6)), py = Math.max(2, Math.round(bh * 0.6));
      let ring = 0, dark = 0, white = 0;
      for (let y = Math.max(0, y0 - py); y <= Math.min(H - 1, y1 + py); y += 1) {
        for (let x = Math.max(0, x0 - px); x <= Math.min(W - 1, x1 + px); x += 1) {
          if (x >= x0 && x <= x1 && y >= y0 && y <= y1) continue;
          const i = 4 * (y * W + x), r = data[i], g = data[i + 1], b = data[i + 2];
          const lum = 0.299 * r + 0.587 * g + 0.114 * b;
          ring++;
          if (lum < 75) dark++;
          else if (lum > 170 && Math.max(r, g, b) - Math.min(r, g, b) < 45) white++;
        }
      }
      const darkR = ring ? dark / ring : 0, whiteR = ring ? white / ring : 0;
      // 硬性條件：周圍大多是黑底（列印時連黑色方塊一起印）；形狀扁平（平放在地上會被透視壓扁）
      // → 排除「紅上衣＋深色褲子」這類直立的紅色物體
      if (darkR < 0.42 || bh / bw > 1.25) continue;
      const fillScore = fill > 0.45 && fill < 0.92 ? 1 : 0.5;
      const conf = 0.4 * fillScore + 0.35 * Math.min(1, darkR / 0.6) + 0.25 * Math.min(1, whiteR / 0.04);
      const cand = { cx: sx / area, cy: sy / area, x0, y0, x1, y1, area, fill, dark: darkR, white: whiteR, conf };
      if (conf >= 0.5 && (!best || cand.conf * Math.sqrt(cand.area) > best.conf * Math.sqrt(best.area))) best = cand;
    }
    return best;
  };

  // ---------------------------------------------------------
  // 4. 光流（區塊比對 + 扣除轉動）→ 前進／後退／左右
  // ---------------------------------------------------------
  class Flow {
    constructor() { this.prev = null; this.w = 0; this.h = 0; this.accDiv = 0; this.accLat = 0; this.frames = 0; this.last = null; }
    reset() { this.accDiv = 0; this.accLat = 0; this.frames = 0; }
    // gray：Float32Array（w×h）；project(x,y)：純轉動下，上一格 (x,y) 在這一格的位置（同一張小圖的像素）；f：小圖的焦距（像素）
    update(gray, w, h, project, f) {
      if (!this.prev || this.w !== w || this.h !== h) { this.prev = gray.slice(); this.w = w; this.h = h; return null; }
      const P = this.prev, B = 6, R = 4, step = Math.max(8, Math.floor(Math.min(w, h) / 7));
      const cx = w / 2, cy = h / 2, rmax = Math.hypot(cx, cy);
      const divs = [], lats = [];
      for (let by = R + 1; by + B + R + 1 < h; by += step) {
        for (let bx = R + 1; bx + B + R + 1 < w; bx += step) {
          // 紋理太少的區塊跳過
          let m = 0, m2 = 0;
          for (let y = 0; y < B; y++) for (let x = 0; x < B; x++) { const v = P[(by + y) * w + bx + x]; m += v; m2 += v * v; }
          m /= B * B; if (m2 / (B * B) - m * m < 30) continue;
          let best = Infinity, bdx = 0, bdy = 0, sum = 0, cnt = 0;
          const sad = new Float32Array((2 * R + 1) * (2 * R + 1));
          for (let dy = -R; dy <= R; dy++) {
            for (let dx = -R; dx <= R; dx++) {
              let s = 0;
              for (let y = 0; y < B; y++) {
                const ro = (by + y) * w + bx, rc = (by + y + dy) * w + bx + dx;
                for (let x = 0; x < B; x++) s += Math.abs(P[ro + x] - gray[rc + x]);
              }
              sad[(dy + R) * (2 * R + 1) + dx + R] = s;
              sum += s; cnt++;
              if (s < best) { best = s; bdx = dx; bdy = dy; }
            }
          }
          if (best > 0.75 * (sum / cnt)) continue; // 不明確
          // 次像素（拋物線）
          const at = (dx, dy) => sad[(dy + R) * (2 * R + 1) + dx + R];
          let fx = bdx, fy = bdy;
          if (bdx > -R && bdx < R) { const a = at(bdx - 1, bdy), c = at(bdx + 1, bdy), d = a - 2 * best + c; if (d > 1e-6) fx += 0.5 * (a - c) / d; }
          if (bdy > -R && bdy < R) { const a = at(bdx, bdy - 1), c = at(bdx, bdy + 1), d = a - 2 * best + c; if (d > 1e-6) fy += 0.5 * (a - c) / d; }
          const px = bx + B / 2, py = by + B / 2;
          let ex = 0, ey = 0;
          if (project) { const q = project(px, py); if (q) { ex = q[0] - px; ey = q[1] - py; } }
          const rx = fx - ex, ry = fy - ey; // 扣掉轉動後剩下的位移＝移動造成
          const ox = px - cx, oy = py - cy, ol = Math.hypot(ox, oy);
          if (ol > rmax * 0.15) divs.push((rx * ox + ry * oy) / (ol * ol));
          lats.push(rx);
        }
      }
      this.prev.set(gray);
      if (divs.length < 4) return null;
      const med = (a) => { a.sort((p, q) => p - q); return a[a.length >> 1]; };
      const div = med(divs), lat = med(lats) / (f || w);
      this.accDiv += div; this.accLat += lat; this.frames++;
      this.last = { div, lat, n: divs.length };
      return this.last;
    }
    // 一步發生時呼叫：依累積的放大/縮小/橫移判斷方向，然後歸零
    classify() {
      const D = this.accDiv, L = this.accLat;
      let dir = 'fwd', why = 'default';
      if (Math.abs(D) < 0.012 && Math.abs(L) < 0.02) { dir = 'fwd'; why = 'weak'; }
      else if (Math.abs(L) > 1.4 * Math.abs(D)) { dir = L < 0 ? 'right' : 'left'; why = 'lateral'; }
      else { dir = D >= 0 ? 'fwd' : 'back'; why = D >= 0 ? 'zoom-in' : 'zoom-out'; }
      const res = { dir, why, D, L, frames: this.frames };
      this.reset();
      return res;
    }
  }
  V.Flow = Flow;
})();
