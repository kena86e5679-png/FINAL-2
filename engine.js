// ==========================================================
// click clip!! — AR 引擎（沿用 v5 app.js 的感測、視野、建築繪製、牆面偵測，改為「地圖座標」）
//   · 手機方向（陀螺儀）→ 相機座標；在符號上校正後，整個畫面對齊地圖座標（原點 S1）
//   · 移動：計步器 × 光流方向（前進／後退 zoom-out／左右）
//   · 畫面：鏡頭 → 地上的點陣座標、場地邊界、符號錨點 → 伺服器生成的實心建築 → 視野限制（中央清晰、周邊點雲）
//   · 後台偵測：Canny/Sobel 梯度 → 垂直線、牆角、柱 → 回報伺服器（桌椅由 game.js 的 COCO-SSD 回報）
// ==========================================================
(function () {
  'use strict';
  const CC = (window.CC = window.CC || {});
  const E = (CC.E = {});
  const DEG = Math.PI / 180;
  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
  const lerp = (a, b, t) => a + (b - a) * t;
  const wrapPi = (a) => { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; };
  E.wrapPi = wrapPi;
  const $ = (id) => document.getElementById(id);

  // ---------- 設定 ----------
  const S = (E.S = {
    sizeMul: 1, fovLong: 65, stepLen: 0.7, dotThreshold: 14, dotDensity: 4, dotMin: 0.7, dotMax: 1.6, dotFalloff: 320,
    bandCell: [25, 12.5, 6.25], bandWidth: [50, 37.5, 25], bandPow: 1.4, limitOn: true, debug: false,
  });
  try { const raw = localStorage.getItem('clickclip.settings'); if (raw) Object.assign(S, JSON.parse(raw)); } catch (e) {}
  E.saveSettings = () => { try { localStorage.setItem('clickclip.settings', JSON.stringify({ sizeMul: S.sizeMul, fovLong: S.fovLong, stepLen: S.stepLen, dotThreshold: S.dotThreshold, limitOn: S.limitOn })); } catch (e) {} };
  const CEN_X = 0.2064, CEN_UP = 0.3058, CEN_DOWN = 0.4107, SAMPLE = 5;

  // ---------- 畫布 ----------
  let cv, ctx, video;
  const scene = document.createElement('canvas'), sctx = scene.getContext('2d');
  const sm = document.createElement('canvas'), smctx = sm.getContext('2d', { willReadFrequently: true });
  const cam = document.createElement('canvas'), camctx = cam.getContext('2d', { willReadFrequently: true }); // 偵測用的小鏡頭畫面
  const flw = document.createElement('canvas'), flwctx = flw.getContext('2d', { willReadFrequently: true });
  let W = 0, H = 0, DPR = 1, U = 1, AW = 0, AH = 0;
  let gray = new Float32Array(1), edgeMag = new Float32Array(1), bandEdge = [0], BAND_TOTAL = 1;
  let C = { l: 0, r: 0, t: 0, b: 0 };
  E.camScale = 1;

  // ---------- 姿態 ----------
  E.EYE = 1.5;
  const pos = (E.pos = { x: 0, z: 0 });
  E.hOff = 0;                 // 地圖方位 = 原始方位 + hOff（在符號上校正時決定）
  let B = { sr: [1, 0, 0], su: [0, 1, 0], F: [0, 0, -1], up: [0, 1, 0] };
  let Braw = B, Bprev = null;
  let FPX = 600, hasSensor = false, h0 = null, yawM = 0, pitchM = 0;
  const sensor = { a: 0, b: 90, g: 0, seen: false };
  let headX = 0, headZ = -1, rawHX = 0, rawHZ = -1;
  E.moveSpeed = 0; E.steps = 0;
  const cmp = (E.cmp = { nOff: 0, have: false, src: '', deg: 0 });

  E.init = function () {
    cv = $('view'); ctx = cv.getContext('2d'); video = $('cam');
    window.addEventListener('resize', resize);
    window.addEventListener('orientationchange', () => setTimeout(resize, 250));
    resize();
    // 桌機測試：拖曳轉向、WASD 移動
    let drag = null;
    cv.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, moved: 0 }; });
    window.addEventListener('pointerup', () => { drag = null; });
    window.addEventListener('pointermove', (e) => {
      if (!drag || (hasSensor && sensor.seen)) return;
      yawM += (e.clientX - drag.x) * 0.006;
      pitchM = Math.max(-1.3, Math.min(1.3, pitchM - (e.clientY - drag.y) * 0.006));
      drag.moved += Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y);
      drag.x = e.clientX; drag.y = e.clientY;
    });
    E.dragMoved = () => (drag ? drag.moved : 0);
    window.addEventListener('keydown', (e) => {
      if (e.target && /input|textarea/i.test(e.target.tagName)) return;
      const k = { KeyW: 'fwd', KeyS: 'back', KeyA: 'left', KeyD: 'right' }[e.code];
      if (k && !e.repeat) E.step(k, 'keyboard');
    });
  };

  function resize() {
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    W = Math.max(200, window.innerWidth); H = Math.max(200, window.innerHeight);
    for (const c of [cv, scene]) { c.width = Math.round(W * DPR); c.height = Math.round(H * DPR); }
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
    AW = Math.ceil(W / SAMPLE); AH = Math.ceil(H / SAMPLE);
    sm.width = AW; sm.height = AH;
    gray = new Float32Array(AW * AH); edgeMag = new Float32Array(AW * AH);
    U = Math.min(1.1, Math.max(0.55, Math.min(W, H) / 560));
    bandEdge = [0];
    for (let i = 0; i < S.bandWidth.length; i++) bandEdge.push(bandEdge[i] + S.bandWidth[i] * U);
    BAND_TOTAL = bandEdge[bandEdge.length - 1];
    const k = Math.max(W, H) / 320;           // 偵測畫面：長邊 320
    cam.width = Math.round(W / k); cam.height = Math.round(H / k); E.camScale = k;
    const kf = W / 72;                        // 光流：寬 72
    flw.width = 72; flw.height = Math.round(H / kf);
    C = centerRect();
    detectResize();
  }
  E.size = () => ({ W, H, DPR, FPX });

  // ==========================================================
  // 感測器
  // ==========================================================
  function screenAngle() {
    let a = 0;
    if (screen.orientation && typeof screen.orientation.angle === 'number') a = screen.orientation.angle;
    else if (typeof window.orientation === 'number') a = window.orientation;
    return ((a % 360) + 360) % 360;
  }
  function e2g(v) {
    const h = Math.atan2(v[0], v[1]), th = h - h0, rho = Math.hypot(v[0], v[1]);
    return [rho * Math.sin(th), v[2], -rho * Math.cos(th)];
  }
  function sensorBasis() {
    const x = sensor.b * DEG, y = sensor.g * DEG, z = sensor.a * DEG;
    const cX = Math.cos(x), cY = Math.cos(y), cZ = Math.cos(z), sX = Math.sin(x), sY = Math.sin(y), sZ = Math.sin(z);
    const m11 = cZ * cY - sZ * sX * sY, m12 = -cX * sZ, m13 = cY * sZ * sX + cZ * sY;
    const m21 = cY * sZ + cZ * sX * sY, m22 = cZ * cX, m23 = sZ * sY - cZ * cY * sX;
    const m31 = -cX * sY, m32 = sX, m33 = cX * cY;
    const Re = [m11, m21, m31], Ue = [m12, m22, m32], Fe = [-m13, -m23, -m33];
    if (h0 === null) { const hf = Math.hypot(Fe[0], Fe[1]) > 0.3 ? Fe : Ue; h0 = Math.atan2(hf[0], hf[1]); } // 整個工作階段只設一次
    const r = e2g(Re), u = e2g(Ue), f = e2g(Fe);
    const ang = screenAngle();
    let sr = r, su = u;
    if (ang === 90) { sr = u.map((q) => -q); su = r; }
    else if (ang === 180) { sr = r.map((q) => -q); su = u.map((q) => -q); }
    else if (ang === 270) { sr = u; su = r.map((q) => -q); }
    return { sr, su, F: f, up: u };
  }
  function manualBasis() {
    const cy = Math.cos(yawM), sy = Math.sin(yawM), cp = Math.cos(pitchM), sp = Math.sin(pitchM);
    return { sr: [cy, 0, sy], su: [-sy * sp, cp, cy * sp], F: [sy * cp, sp, -cy * cp], up: [-sy * sp, cp, cy * sp] };
  }
  const rotY = (v, t) => { const c = Math.cos(t), s = Math.sin(t); return [v[0] * c - v[2] * s, v[1], v[2] * c + v[0] * s]; };
  function headingOf(b) {
    let hx = b.F[0], hz = b.F[2], hl = Math.hypot(hx, hz);
    if (hl < 0.35) { const s = b.F[1] < 0 ? 1 : -1; hx = b.up[0] * s; hz = b.up[2] * s; hl = Math.hypot(hx, hz); }
    return hl > 1e-4 ? [hx / hl, hz / hl] : null;
  }
  function updateCamera() {
    Bprev = B;
    Braw = hasSensor && sensor.seen ? sensorBasis() : manualBasis();
    const t = E.hOff;
    B = { sr: rotY(Braw.sr, t), su: rotY(Braw.su, t), F: rotY(Braw.F, t), up: rotY(Braw.up, t) };
    const h = headingOf(B); if (h) { headX = h[0]; headZ = h[1]; }
    const hr = headingOf(Braw); if (hr) { rawHX = hr[0]; rawHZ = hr[1]; }
  }
  E.mapHeading = () => Math.atan2(headX, -headZ);
  E.rawHeading = () => Math.atan2(rawHX, -rawHZ);
  E.absHeading = () => (cmp.have ? wrapPi(E.rawHeading() + cmp.nOff) : null);
  E.pitch = () => Math.asin(Math.max(-1, Math.min(1, B.F[1])));
  E.head = () => [headX, headZ];
  E.hasSensor = () => hasSensor && sensor.seen;
  E.setManualView = (yaw, pitch) => { yawM = yaw; pitchM = pitch; }; // 測試／桌機用

  function cmpMeas(bearing) {
    const off = wrapPi(bearing - E.rawHeading());
    if (!cmp.have) { cmp.nOff = off; cmp.have = true; } else cmp.nOff = wrapPi(cmp.nOff + wrapPi(off - cmp.nOff) * 0.05);
  }
  function onOrient(e) {
    if (e.beta == null || e.gamma == null) return;
    sensor.a = e.alpha == null ? 0 : e.alpha; sensor.b = e.beta; sensor.g = e.gamma;
    sensor.seen = true; hasSensor = true;
    if (e.webkitCompassHeading != null && !isNaN(e.webkitCompassHeading) && Math.hypot(Braw.F[0], Braw.F[2]) > 0.5) { cmpMeas(e.webkitCompassHeading * DEG); cmp.src = 'iOS 羅盤'; }
  }
  function onOrientAbs(e) {
    if (e.alpha == null || e.beta == null || e.gamma == null) return;
    const x = e.beta * DEG, y = e.gamma * DEG, z = e.alpha * DEG;
    const cX = Math.cos(x), cY = Math.cos(y), cZ = Math.cos(z), sX = Math.sin(x), sY = Math.sin(y), sZ = Math.sin(z);
    const e0 = -(cY * sZ * sX + cZ * sY), n0 = -(sZ * sY - cZ * cY * sX);
    if (Math.hypot(e0, n0) < 0.5) return;
    cmpMeas(Math.atan2(e0, n0)); cmp.src = 'Android 羅盤';
  }

  // 計步：加速度峰值 → 一步；方向交給光流判斷
  let accBase = 9.8, prevSig = 0, lastStepT = 0;
  function onMotion(e) {
    const a = e.accelerationIncludingGravity;
    if (!a || a.x == null) return;
    const mag = Math.hypot(a.x, a.y, a.z);
    accBase += (mag - accBase) * 0.08;
    const sig = mag - accBase, now = performance.now();
    if (sig > 1.3 && prevSig <= 1.3 && now - lastStepT > 320) { lastStepT = now; E.step(null, 'motion'); }
    prevSig = sig;
  }

  // 必須在使用者點擊時呼叫（iOS 規定）
  E.requestSensors = async function () {
    let note = '';
    try {
      if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
        const r = await DeviceOrientationEvent.requestPermission();
        if (r !== 'granted') note = '沒有方向感測權限：改用拖曳畫面轉向';
      }
      if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
        try { await DeviceMotionEvent.requestPermission(); } catch (e) {}
      }
    } catch (e) { note = '感測器權限被拒絕：改用拖曳畫面轉向'; }
    window.addEventListener('deviceorientation', onOrient);
    window.addEventListener('deviceorientationabsolute', onOrientAbs);
    window.addEventListener('devicemotion', onMotion);
    return note;
  };
  E.startCamera = async function () {
    if (video.srcObject) return true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      video.srcObject = stream;
      await video.play();
      return true;
    } catch (e) { return false; }
  };
  E.cameraReady = () => !!(video && video.videoWidth && video.readyState >= 2);

  // ==========================================================
  // 投影
  // ==========================================================
  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  function toCam(vx, vy, vz) { const v = [vx, vy, vz]; return [dot3(v, B.sr), dot3(v, B.su), dot3(v, B.F)]; }
  function toScreen(c) { return [W / 2 + (FPX * c[0]) / c[2], H / 2 - (FPX * c[1]) / c[2]]; }
  function rayW(px, py, basis) {
    const b = basis || B, cx = (px - W / 2) / FPX, cy = -(py - H / 2) / FPX;
    return [cx * b.sr[0] + cy * b.su[0] + b.F[0], cx * b.sr[1] + cy * b.su[1] + b.F[1], cx * b.sr[2] + cy * b.su[2] + b.F[2]];
  }
  function floorPoint(px, py) {
    const r = rayW(px, py);
    if (r[1] > -0.035) return null;
    const t = E.EYE / -r[1];
    return { x: pos.x + r[0] * t, z: pos.z + r[2] * t, d: Math.hypot(r[0], r[2]) * t };
  }
  E.toCam = toCam; E.toScreen = toScreen; E.rayW = rayW; E.floorPoint = floorPoint;
  E.project = (x, y, z) => { const c = toCam(x - pos.x, y - E.EYE, z - pos.z); return c[2] > 0.15 ? toScreen(c) : null; };
  E.center = () => [W / 2, H / 2];
  // 畫面某點對應到的「水平方位」（地圖座標）
  E.rayHeading = (px, py) => { const r = rayW(px, py); return Math.atan2(r[0], -r[2]); };
  E.rawRayHeading = (px, py) => { const r = rayW(px, py, Braw); return Math.atan2(r[0], -r[2]); };

  // 人的身高（腳在畫面中：用腳的地面點求距離，再看頭頂仰角）
  E.personGeom = function (headPx, footPx) {
    if (footPx) {
      const fp = floorPoint(footPx.x, footPx.y);
      if (fp && fp.d > 0.5 && fp.d < 25) {
        const r = rayW(headPx.x, headPx.y), hl = Math.hypot(r[0], r[2]) || 1e-6;
        const height = E.EYE + (r[1] / hl) * fp.d;
        return { x: fp.x, z: fp.z, d: fp.d, height: height > 1.0 && height < 2.2 ? height : null };
      }
    }
    // 腳被切掉：假設身高 1.7 m，用頭頂仰角＋畫面高度估距離（粗略）
    const r = rayW(headPx.x, headPx.y), hl = Math.hypot(r[0], r[2]) || 1e-6;
    const up = r[1] / hl;
    const d = up > 0.02 ? Math.min(20, (1.7 - E.EYE) / up) : 6;
    return { x: pos.x + (r[0] / hl) * d, z: pos.z + (r[2] / hl) * d, d, height: null, rough: true };
  };

  // ==========================================================
  // 移動：一步 → 依光流判斷方向
  // ==========================================================
  const pend = { x: 0, z: 0 };
  let flowObj = null;
  E.lastStep = null;
  E.onStep = null;
  E.step = function (dir, src) {
    let why = src;
    if (!dir) {
      if (flowObj && E.cameraReady()) { const c = flowObj.classify(); dir = c.dir; why = c.why; E.lastStep = c; }
      else dir = 'fwd';
    }
    const L = S.stepLen * (dir === 'fwd' ? 1 : dir === 'back' ? 0.6 : 0.55);
    const rx = -headZ, rz = headX; // 右手方向
    let vx = headX, vz = headZ;
    if (dir === 'back') { vx = -headX; vz = -headZ; }
    else if (dir === 'right') { vx = rx; vz = rz; }
    else if (dir === 'left') { vx = -rx; vz = -rz; }
    pend.x += vx * L; pend.z += vz * L;
    E.steps++;
    if (E.onStep) E.onStep(dir, why);
  };
  function updateMove(dt) {
    const pl = Math.hypot(pend.x, pend.z);
    const mv = Math.min(pl, 2.0 * dt);
    if (pl > 1e-6) { pos.x += (pend.x / pl) * mv; pos.z += (pend.z / pl) * mv; pend.x -= (pend.x / pl) * mv; pend.z -= (pend.z / pl) * mv; }
    E.moveSpeed += (mv / Math.max(dt, 1e-3) - E.moveSpeed) * Math.min(1, dt * 6);
    if (E.moveSpeed < 0.02) E.moveSpeed = 0;
  }
  E.setPos = (x, z) => { pos.x = x; pos.z = z; pend.x = pend.z = 0; };
  E.nudgePos = (dx, dz) => { pos.x += dx; pos.z += dz; };

  // 光流：每 2 格更新一次
  let flowTick = 0;
  function updateFlow() {
    if (!CC.Vision || !E.cameraReady()) return;
    if (!flowObj) flowObj = new CC.Vision.Flow();
    if (++flowTick % 2) return;
    flwctx.drawImage(cam, 0, 0, flw.width, flw.height);
    const d = flwctx.getImageData(0, 0, flw.width, flw.height).data;
    const n = flw.width * flw.height, g = new Float32Array(n);
    for (let k = 0; k < n; k++) g[k] = 0.299 * d[4 * k] + 0.587 * d[4 * k + 1] + 0.114 * d[4 * k + 2];
    const k = W / flw.width;
    const prevB = flowObj.basis;
    const curB = B;
    const project = prevB ? (x, y) => {
      const r = rayW(x * k, y * k, prevB);
      const c = [dot3(r, curB.sr), dot3(r, curB.su), dot3(r, curB.F)];
      if (c[2] < 0.1) return null;
      return [(W / 2 + (FPX * c[0]) / c[2]) / k, (H / 2 - (FPX * c[1]) / c[2]) / k];
    } : null;
    flowObj.update(g, flw.width, flw.height, project, FPX / k);
    flowObj.basis = { sr: curB.sr.slice(), su: curB.su.slice(), F: curB.F.slice() };
  }
  E.flow = () => flowObj;

  // ==========================================================
  // 視野：中央清晰區隨速度縮放；周邊點雲
  // ==========================================================
  const fov = (E.fov = { l: 1.25, r: 1.25, u: 1.25, d: 1.25, speedScale: 1.25 });
  const FOV_MIN = 0.28;
  function centerRect() {
    const k = S.sizeMul, kx = k * (W < H ? 1.3 : 1);
    return { l: Math.max(0, W / 2 - CEN_X * W * kx * fov.l), r: Math.min(W, W / 2 + CEN_X * W * kx * fov.r),
      t: Math.max(0, H / 2 - CEN_UP * H * k * fov.u), b: Math.min(H, H / 2 + CEN_DOWN * H * k * fov.d) };
  }
  function updateVision(active) {
    let sf = 1.28;
    if (active) {
      const v = E.moveSpeed;
      sf = v < 0.05 ? 1.28 : v <= 0.75 ? lerp(1.28, 1.02, (v - 0.05) / 0.7) : lerp(1.02, 0.38, clamp01((v - 0.75) / 1.15));
    }
    fov.speedScale = sf;
    for (const k of ['l', 'r', 'u', 'd']) { const goal = Math.max(FOV_MIN, sf * E.flash.fovMul); fov[k] = lerp(fov[k], goal, goal < fov[k] ? 0.18 : 0.08); }
    C = centerRect();
  }
  function rectDist(x, y) { return Math.max(Math.max(C.l - x, 0, x - C.r), Math.max(C.t - y, 0, y - C.b)); }
  const SCAN_AZ = 72, SCAN_EL = 36, scanMap = new Float32Array(SCAN_AZ * SCAN_EL);
  function scanIdx(r) {
    const l = Math.hypot(r[0], r[1], r[2]) || 1;
    const az = Math.atan2(r[0] / l, -r[2] / l), el = Math.asin(Math.max(-1, Math.min(1, r[1] / l)));
    const bi = Math.min(SCAN_AZ - 1, Math.max(0, Math.floor(((az + Math.PI) / (2 * Math.PI)) * SCAN_AZ)));
    const bj = Math.min(SCAN_EL - 1, Math.max(0, Math.floor(((el + Math.PI / 2) / Math.PI) * SCAN_EL)));
    return bj * SCAN_AZ + bi;
  }
  function updateScanMemory(dt) {
    const sx = Math.max(12, (C.r - C.l) / 8), sy = Math.max(12, (C.b - C.t) / 8);
    for (let py = C.t; py <= C.b; py += sy) for (let px = C.l; px <= C.r; px += sx) { const i = scanIdx(rayW(px, py)); scanMap[i] = Math.min(100, scanMap[i] + dt * 18); }
  }
  E.resetScan = () => scanMap.fill(0);
  function hash(a, b, c) {
    let x = Math.imul(a | 0, 73856093) ^ Math.imul(b | 0, 19349663) ^ Math.imul(c | 0, 83492791);
    x = Math.imul(x ^ (x >>> 13), 1274126177); x ^= x >>> 16;
    return (x >>> 0) / 4294967296;
  }
  function analyzeDots() {
    smctx.drawImage(scene, 0, 0, AW, AH);
    const p = smctx.getImageData(0, 0, AW, AH).data;
    for (let k = 0; k < AW * AH; k++) gray[k] = 0.299 * p[4 * k] + 0.587 * p[4 * k + 1] + 0.114 * p[4 * k + 2];
    for (let j = 1; j < AH - 1; j++) for (let i = 1; i < AW - 1; i++) {
      const k = j * AW + i;
      const gx = -gray[k - AW - 1] + gray[k - AW + 1] - 2 * gray[k - 1] + 2 * gray[k + 1] - gray[k + AW - 1] + gray[k + AW + 1];
      const gy = -gray[k - AW - 1] - 2 * gray[k - AW] - gray[k - AW + 1] + gray[k + AW - 1] + 2 * gray[k + AW] + gray[k + AW + 1];
      edgeMag[k] = 0.6 * (Math.sqrt(gx * gx + gy * gy) / 4) + 0.4 * edgeMag[k];
    }
  }
  function blit(x, y, w, h) { if (w > 0 && h > 0) ctx.drawImage(scene, x * DPR, y * DPR, w * DPR, h * DPR, x, y, w, h); }
  function drawDots() {
    const thr = S.dotThreshold, buckets = [[], [], []];
    const ud = Math.min(1.3, Math.max(0.85, U * 1.4));
    const radii = [S.dotMin * ud, ((S.dotMin + S.dotMax) / 2) * ud, S.dotMax * ud];
    for (let j = 0; j < AH; j++) for (let i = 0; i < AW; i++) {
      const k = j * AW + i, x = i * SAMPLE + SAMPLE / 2, y = j * SAMPLE + SAMPLE / 2;
      const d = rectDist(x, y); if (d <= 0) continue;
      const edgeN = clamp01((edgeMag[k] - thr * 0.5) / (thr * 3.5));
      const v = Math.max(edgeN, Math.pow(gray[k] / 255, 2.2) * 0.3);
      if (v <= 0.02) continue;
      const prox = 0.3 + 0.7 * (1 - Math.min(1, d / (S.dotFalloff * U)));
      const scanMul = 0.15 + 2.85 * clamp01(scanMap[scanIdx(rayW(x, y))] / 25);
      const n = Math.min(9, Math.floor(v * prox * S.dotDensity * scanMul + hash(i, j, 0)));
      if (n <= 0) continue;
      const b = edgeN > 0.66 ? 2 : edgeN > 0.33 ? 1 : 0;
      for (let q = 0; q < n; q++) buckets[b].push(x + (hash(i, j, q + 1) - 0.5) * SAMPLE, y + (hash(i, j, q + 11) - 0.5) * SAMPLE);
    }
    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    for (let b = 0; b < 3; b++) {
      const pts = buckets[b], r = radii[b];
      ctx.beginPath();
      for (let q = 0; q < pts.length; q += 2) { ctx.moveTo(pts[q] + r, pts[q + 1]); ctx.arc(pts[q], pts[q + 1], r, 0, Math.PI * 2); }
      ctx.fill();
    }
  }
  function drawBand() {
    for (let L = 0; L < S.bandCell.length; L++) {
      const s = S.bandCell[L] * U, d0 = bandEdge[L], d1 = bandEdge[L + 1];
      const kx0 = Math.floor((0 - C.l) / s) - 1, kx1 = Math.ceil((W - C.l) / s) + 1;
      const ky0 = Math.floor((0 - C.t) / s) - 1, ky1 = Math.ceil((H - C.t) / s) + 1;
      for (let ky = ky0; ky <= ky1; ky++) for (let kx = kx0; kx <= kx1; kx++) {
        const x0 = C.l + kx * s, y0 = C.t + ky * s, cx = x0 + s / 2, cy = y0 + s / 2;
        if (cx < 0 || cx > W || cy < 0 || cy > H) continue;
        const d = rectDist(cx, cy);
        if (d <= d0 || d > d1) continue;
        if (hash(kx, ky, L + 50) > Math.pow(1 - d / BAND_TOTAL, S.bandPow)) continue;
        blit(x0, y0, s, s);
      }
    }
  }

  // ==========================================================
  // 地圖資料（來自伺服器）
  // ==========================================================
  E.map = null; E.anchors = []; E.obstacles = []; E.visited = new Set();
  let obs = [];                                    // 繪製用的建築
  E.structures = () => obs;
  const ARCH = { WALL_H: 2.8, SILL_H: 0.9, HEAD_H: 2.1, DOOR_H: 2.1, MODULE: 1.2 };
  function buildSegs(w, d, h, kind, withWindows) {
    const segs = [], hx = w / 2, hz = d / 2;
    const L = (x1, y1, z1, x2, y2, z2) => segs.push([x1, y1, z1, x2, y2, z2]);
    const rect = (y, ix) => { const ax = hx - (ix || 0), az = hz - (ix || 0); L(-ax, y, -az, ax, y, -az); L(ax, y, -az, ax, y, az); L(ax, y, az, -ax, y, az); L(-ax, y, az, -ax, y, -az); };
    rect(0); rect(h);
    L(-hx, 0, -hz, -hx, h, -hz); L(hx, 0, -hz, hx, h, -hz); L(hx, 0, hz, hx, h, hz); L(-hx, 0, hz, -hx, h, hz);
    if (kind === 'pillar') { rect(0.15, -0.02); rect(ARCH.DOOR_H, -0.02); return segs; }
    if (kind === 'low') { rect(0.12, 0); rect(h - 0.06, 0.03); return segs; }
    const longX = w >= d, len = longX ? w : d;
    rect(0.15);
    if (h >= 2.2) { rect(ARCH.SILL_H); rect(ARCH.HEAD_H); rect(h - 0.15); }
    const nb = Math.max(1, Math.round(len / ARCH.MODULE)), cols = [];
    for (let i = 0; i <= nb; i++) cols.push(-len / 2 + (i * len) / nb);
    for (const c0 of cols) { if (longX) { L(c0, 0, -hz, c0, h, -hz); L(c0, 0, hz, c0, h, hz); } else { L(-hx, 0, c0, -hx, h, c0); L(hx, 0, c0, hx, h, c0); } }
    if (withWindows && h >= 2.2) {
      for (let i = 0; i < nb; i++) {
        const a = cols[i] + 0.15, b = cols[i + 1] - 0.15; if (b - a < 0.3) continue;
        const y0 = ARCH.SILL_H, y1 = ARCH.HEAD_H;
        for (const sd of [-1, 1]) {
          if (longX) { const z = sd * hz; L(a, y0, z, b, y0, z); L(b, y0, z, b, y1, z); L(b, y1, z, a, y1, z); L(a, y1, z, a, y0, z); }
          else { const x = sd * hx; L(x, y0, a, x, y0, b); L(x, y0, b, x, y1, b); L(x, y1, b, x, y1, a); L(x, y1, a, x, y0, a); }
        }
      }
    }
    return segs;
  }
  function makeObs(s) {
    return { sid: s.sid, gid: s.gid, x: s.x, z: s.z, w: s.w, d: s.d, h: s.h, kind: s.kind, rot: s.rot || 0, gAx: s.gAx || 0,
      born: performance.now() - Math.min(5000, Math.max(0, (s.age || 0))), gen: true, dying: 0,
      segsFar: buildSegs(s.w, s.d, s.h, s.kind, false), segsNear: buildSegs(s.w, s.d, s.h, s.kind, true) };
  }
  E.setStructures = (list) => { obs = (list || []).map((s) => Object.assign(makeObs(s), { born: performance.now() - 5000 })); };
  E.addStructures = (list) => { for (const s of list || []) if (!obs.some((o) => o.sid === s.sid)) obs.push(makeObs(s)); };
  E.clearStructures = () => { obs = []; };
  E.setMap = (map, anchors) => { E.map = map; E.anchors = anchors || []; };
  E.setObstacles = (list) => { E.obstacles = list || []; };
  E.upsertObstacles = (list) => { for (const o of list || []) { const i = E.obstacles.findIndex((q) => q.oid === o.oid); if (i >= 0) E.obstacles[i] = o; else E.obstacles.push(o); } };

  // 遮擋：虛擬建築（高於 1.5 m）＋伺服器上的真實牆
  function rayHitBox(ox, oz, dx, dz, o, pad) {
    const r = o.rot || 0, cr = Math.cos(r), sr = Math.sin(r);
    const px = ox - o.x, pz = oz - o.z;
    const org = [px * cr + pz * sr, -px * sr + pz * cr], dir = [dx * cr + dz * sr, -dx * sr + dz * cr];
    const lo = [-o.w / 2 - pad, -o.d / 2 - pad], hi = [o.w / 2 + pad, o.d / 2 + pad];
    let t0 = 0, t1 = 60;
    for (let k = 0; k < 2; k++) {
      if (Math.abs(dir[k]) < 1e-9) { if (org[k] < lo[k] || org[k] > hi[k]) return Infinity; }
      else { let a = (lo[k] - org[k]) / dir[k], b = (hi[k] - org[k]) / dir[k]; if (a > b) { const t = a; a = b; b = t; } t0 = Math.max(t0, a); t1 = Math.min(t1, b); if (t0 > t1) return Infinity; }
    }
    return t0;
  }
  function segCross(ax, az, bx, bz, cx, cz, dx, dz) {
    const d1x = bx - ax, d1z = bz - az, d2x = dx - cx, d2z = dz - cz, den = d1x * d2z - d1z * d2x;
    if (Math.abs(den) < 1e-9) return false;
    const t = ((cx - ax) * d2z - (cz - az) * d2x) / den, u = ((cx - ax) * d1z - (cz - az) * d1x) / den;
    return t > 0.03 && t < 0.97 && u >= 0 && u <= 1;
  }
  E.occluded = function (tx, tz, opts) {
    const d = Math.hypot(tx - pos.x, tz - pos.z);
    if (d < 0.4) return false;
    const dx = (tx - pos.x) / d, dz = (tz - pos.z) / d;
    for (const o of obs) { if (o.dying || o.h < 1.5) continue; if (rayHitBox(pos.x, pos.z, dx, dz, o, 0.02) < d - 0.2) return 'virtual'; }
    if (!opts || opts.real !== false) {
      for (const w of E.obstacles) if (w.k === 'wall' && w.hits >= 2 && segCross(pos.x, pos.z, tx, tz, w.x1, w.z1, w.x2, w.z2)) return 'real';
    }
    return false;
  };

  // 點陣座標
  E.nearestGrid = function () {
    if (!E.map) return null;
    const i = Math.max(0, Math.min(E.map.cols - 1, Math.round(pos.x))), j = Math.max(0, Math.min(E.map.rows - 1, Math.round(pos.z)));
    const outside = pos.x < -0.3 || pos.z < -0.3 || pos.x > E.map.W + 0.3 || pos.z > E.map.D + 0.3;
    return { i, j, d: Math.hypot(i - pos.x, j - pos.z), label: (j + 1) + '-' + (i + 1), outside };
  };

  // ==========================================================
  // 繪製
  // ==========================================================
  function drawVideo(c) {
    c.setTransform(DPR, 0, 0, DPR, 0, 0);
    const vw = video && video.videoWidth, vh = video && video.videoHeight;
    if (vw && vh && video.readyState >= 2) {
      const s = Math.max(W / vw, H / vh), dw = vw * s, dh = vh * s;
      c.drawImage(video, (W - dw) / 2, (H - dh) / 2, dw, dh);
      FPX = (s * (Math.max(vw, vh) / 2)) / Math.tan((S.fovLong * DEG) / 2);
      // 偵測用小畫面（與螢幕同一個裁切）
      const k = E.camScale;
      camctx.drawImage(video, (W - dw) / 2 / k, (H - dh) / 2 / k, dw / k, dh / k);
    } else {
      c.fillStyle = '#15171c'; c.fillRect(0, 0, W, H);
      // 沒有鏡頭（桌機測試）：畫一個假地板方便看
      const g = c.createLinearGradient(0, H * 0.45, 0, H); g.addColorStop(0, '#1d2028'); g.addColorStop(1, '#3a3f4a');
      c.fillStyle = g; c.fillRect(0, H * 0.5 + B.F[1] * FPX, W, H);
      FPX = (Math.max(W, H) / 2) / Math.tan((S.fovLong * DEG) / 2);
      camctx.drawImage(scene, 0, 0, cam.width, cam.height);
    }
  }
  E.camCanvas = () => cam;

  function seg3(c, ax, ay, az, bx, by, bz) {
    const near = 0.15;
    let a = toCam(ax - pos.x, ay - E.EYE, az - pos.z), b = toCam(bx - pos.x, by - E.EYE, bz - pos.z);
    if (a[2] < near && b[2] < near) return;
    if (a[2] < near) { const t = (near - a[2]) / (b[2] - a[2]); a = [lerp(a[0], b[0], t), lerp(a[1], b[1], t), near]; }
    else if (b[2] < near) { const t = (near - b[2]) / (a[2] - b[2]); b = [lerp(b[0], a[0], t), lerp(b[1], a[1], t), near]; }
    const pa = toScreen(a), pb = toScreen(b);
    c.moveTo(pa[0], pa[1]); c.lineTo(pb[0], pb[1]);
  }
  function floorPoly(c, pts) { // 地上的多邊形（y = 0）
    const cam = pts.map((p) => toCam(p[0] - pos.x, 0.005 - E.EYE, p[1] - pos.z));
    const clipped = clipNear(cam, 0.15);
    if (clipped.length < 3) return false;
    const s = clipped.map(toScreen);
    c.beginPath(); c.moveTo(s[0][0], s[0][1]); for (let i = 1; i < s.length; i++) c.lineTo(s[i][0], s[i][1]); c.closePath();
    return true;
  }
  function clipNear(v, near) {
    const out = [];
    for (let i = 0; i < v.length; i++) {
      const a = v[i], b = v[(i + 1) % v.length], aIn = a[2] >= near, bIn = b[2] >= near;
      if (aIn && bIn) out.push(b);
      else if (aIn && !bIn) { const t = (near - a[2]) / (b[2] - a[2]); out.push([lerp(a[0], b[0], t), lerp(a[1], b[1], t), near]); }
      else if (!aIn && bIn) { const t = (near - a[2]) / (b[2] - a[2]); out.push([lerp(a[0], b[0], t), lerp(a[1], b[1], t), near]); out.push(b); }
    }
    return out;
  }

  // 地上的點陣座標、場地邊界、符號錨點
  function drawFloorLayer(c, opts) {
    if (!E.map) return;
    const m = E.map, R = 4.6;
    c.save();
    // 場地邊界（地上的線）
    c.strokeStyle = 'rgba(110,255,225,0.55)'; c.lineWidth = 2; c.setLineDash([10, 8]);
    c.beginPath();
    seg3(c, 0, 0, 0, m.W, 0, 0); seg3(c, m.W, 0, 0, m.W, 0, m.D); seg3(c, m.W, 0, m.D, 0, 0, m.D); seg3(c, 0, 0, m.D, 0, 0, 0);
    c.stroke(); c.setLineDash([]);
    // 點陣
    const near = E.nearestGrid();
    const list = [];
    for (let j = Math.max(0, Math.floor(pos.z - R)); j <= Math.min(m.rows - 1, Math.ceil(pos.z + R)); j++) {
      for (let i = Math.max(0, Math.floor(pos.x - R)); i <= Math.min(m.cols - 1, Math.ceil(pos.x + R)); i++) {
        const d = Math.hypot(i - pos.x, j - pos.z);
        if (d > R) continue;
        list.push({ i, j, d });
      }
    }
    list.sort((a, b) => a.d - b.d);
    c.font = '600 10px ui-monospace, Menlo, monospace'; c.textAlign = 'center'; c.textBaseline = 'middle';
    for (const g of list.slice(0, 24)) {
      if (E.occluded(g.i, g.j, { real: false })) continue;
      const lbl = (g.j + 1) + '-' + (g.i + 1);
      const isNear = near && near.i === g.i && near.j === g.j && near.d < 0.75;
      const vis = E.visited.has(lbl);
      const s = isNear ? 0.16 : 0.09;
      if (!floorPoly(c, [[g.i, g.j - s], [g.i + s, g.j], [g.i, g.j + s], [g.i - s, g.j]])) continue;
      c.fillStyle = isNear ? 'rgba(110,255,225,0.85)' : vis ? 'rgba(110,255,225,0.35)' : 'rgba(255,255,255,0.28)';
      c.strokeStyle = isNear ? '#6effe1' : 'rgba(255,255,255,0.75)';
      c.lineWidth = isNear ? 2 : 1;
      c.fill(); c.stroke();
      if (g.d < 3.2) {
        const p = E.project(g.i, 0.02, g.j);
        if (p) { c.fillStyle = isNear ? '#6effe1' : 'rgba(255,255,255,0.8)'; c.fillText(lbl, p[0], p[1] - 12); }
      }
    }
    // 符號錨點：地上畫出符號的輪廓
    for (const a of E.anchors) {
      const d = Math.hypot(a.x - pos.x, a.z - pos.z);
      if (d > 9) continue;
      const k = 0.18;
      if (floorPoly(c, [[a.x - k * 1.2, a.z + k], [a.x - k * 0.2, a.z - k], [a.x + k, a.z - k], [a.x + k, a.z + k]])) {
        c.fillStyle = 'rgba(180,65,34,0.55)'; c.fill();
        c.strokeStyle = '#ffffff'; c.lineWidth = 1.5; c.stroke();
      }
      const p = E.project(a.x, 0.05, a.z);
      if (p) { c.fillStyle = '#ff8a6a'; c.font = '700 11px ui-monospace, Menlo, monospace'; c.fillText(a.aid, p[0], p[1] - 16); }
    }
    c.restore();
  }

  // 實心不透明建築（畫家演算法）
  function drawStructures(c) {
    const near = 0.15, now = performance.now();
    c.save(); c.lineJoin = 'round';
    const sorted = obs.slice().sort((a, b) => Math.hypot(b.x - pos.x, b.z - pos.z) - Math.hypot(a.x - pos.x, a.z - pos.z));
    for (const o of sorted) {
      const dx = o.x - pos.x, dz = o.z - pos.z, dist = Math.hypot(dx, dz);
      if (dist > 30) continue;
      const reach = Math.max(o.w, o.d);
      if (toCam(dx, -E.EYE, dz)[2] < -reach && dist > 3) continue;
      const t = clamp01((now - o.born) / 1800), e = (u) => 1 - (1 - u) * (1 - u);
      let gx = e(t), gy = e(Math.max(0, (t - 0.15) / 0.85));
      if (o.dying) { const k = 1 - clamp01((now - o.dying) / 900); gx *= k; gy *= k; }
      if (gy <= 0.01 && gx <= 0.01) continue;
      const cr = Math.cos(o.rot || 0), sr = Math.sin(o.rot || 0);
      const noX = o.gAx === 9, ax = noX ? 0 : (o.gAx || 0) * o.w / 2;
      const tx = (x, z) => { const lx = noX ? x : ax + (x - ax) * gx, lz = noX ? z : z * (o.gAx ? 1 : gx); return [lx * cr - lz * sr, lx * sr + lz * cr]; };
      const hx = o.w / 2, hz = o.d / 2, h = o.h * gy;
      const lc = [[-hx, 0, -hz], [hx, 0, -hz], [hx, 0, hz], [-hx, 0, hz], [-hx, h, -hz], [hx, h, -hz], [hx, h, hz], [-hx, h, hz]];
      const cc = lc.map((p) => { const q = tx(p[0], p[2]); return toCam(dx + q[0], p[1] - E.EYE, dz + q[1]); });
      const faces = [{ idx: [0, 1, 5, 4], shade: 44 }, { idx: [1, 2, 6, 5], shade: 34 }, { idx: [2, 3, 7, 6], shade: 48 }, { idx: [3, 0, 4, 7], shade: 36 }, { idx: [4, 5, 6, 7], shade: 58 }];
      const vf = [];
      for (const f of faces) { const poly = clipNear(f.idx.map((i) => cc[i]), near); if (poly.length < 3) continue; vf.push({ poly, z: poly.reduce((s, p) => s + p[2], 0) / poly.length, shade: f.shade }); }
      vf.sort((a, b) => b.z - a.z);
      for (const f of vf) {
        const pts = f.poly.map(toScreen), lum = Math.round(lerp(f.shade, 22, clamp01(dist / 22)));
        c.fillStyle = `rgb(${lum},${lum + 3},${lum + 8})`; c.strokeStyle = 'rgba(255,255,255,0.88)'; c.lineWidth = dist < 6 ? 1.8 : 1.2;
        c.beginPath(); c.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) c.lineTo(pts[i][0], pts[i][1]); c.closePath(); c.fill(); c.stroke();
      }
      const segs = dist < 10 ? o.segsNear : o.segsFar;
      c.strokeStyle = 'rgba(255,255,255,0.72)'; c.lineWidth = dist < 6 ? 1.3 : 0.9;
      c.beginPath();
      for (const sg of segs) {
        const p1 = tx(sg[0], sg[2]), p2 = tx(sg[3], sg[5]);
        let a = toCam(dx + p1[0], sg[1] * gy - E.EYE, dz + p1[1]), b = toCam(dx + p2[0], sg[4] * gy - E.EYE, dz + p2[1]);
        if (a[2] < near && b[2] < near) continue;
        if (a[2] < near) { const t2 = (near - a[2]) / (b[2] - a[2]); a = [lerp(a[0], b[0], t2), lerp(a[1], b[1], t2), near]; }
        else if (b[2] < near) { const t2 = (near - b[2]) / (a[2] - b[2]); b = [lerp(b[0], a[0], t2), lerp(b[1], a[1], t2), near]; }
        const pa = toScreen(a), pb = toScreen(b); c.moveTo(pa[0], pa[1]); c.lineTo(pb[0], pb[1]);
      }
      c.stroke();
    }
    c.restore();
  }

  // 除錯：真實物件（伺服器上的）
  function drawObstaclesDebug(c) {
    c.save(); c.lineWidth = 2;
    c.beginPath(); c.strokeStyle = 'rgba(255,184,108,0.9)';
    for (const o of E.obstacles) if (o.k === 'wall') { seg3(c, o.x1, 0, o.z1, o.x2, 0, o.z2); seg3(c, o.x1, 2.4, o.z1, o.x2, 2.4, o.z2); }
    c.stroke();
    for (const o of E.obstacles) {
      if (o.k === 'wall') continue;
      const hw = o.w / 2, hd = o.d / 2;
      if (floorPoly(c, [[o.x - hw, o.z - hd], [o.x + hw, o.z - hd], [o.x + hw, o.z + hd], [o.x - hw, o.z + hd]])) { c.strokeStyle = 'rgba(255,184,108,0.9)'; c.stroke(); }
    }
    c.restore();
  }

  // 準心
  function drawCrosshair(state) {
    const cx = W / 2, cy = H / 2, r = 9;
    const col = state === 'lock' ? '#ff4d4d' : state === 'dim' ? 'rgba(255,255,255,0.4)' : '#ffffff';
    ctx.save();
    ctx.strokeStyle = 'rgba(0,0,0,0.6)'; ctx.lineWidth = 4;
    const draw = () => { ctx.beginPath(); ctx.moveTo(cx - r - 6, cy); ctx.lineTo(cx - 4, cy); ctx.moveTo(cx + 4, cy); ctx.lineTo(cx + r + 6, cy); ctx.moveTo(cx, cy - r - 6); ctx.lineTo(cx, cy - 4); ctx.moveTo(cx, cy + 4); ctx.lineTo(cx, cy + r + 6); ctx.stroke(); };
    draw(); ctx.strokeStyle = col; ctx.lineWidth = 2; draw();
    ctx.fillStyle = col; ctx.beginPath(); ctx.arc(cx, cy, 1.6, 0, 7); ctx.fill();
    ctx.restore();
  }

  // 敵人框：白色虛線框 + 淡紅濾鏡
  function drawPersonBox(b, label, sub, hp, mode) {
    ctx.save();
    if (mode !== 'unknown') {
      ctx.fillStyle = 'rgba(255,45,55,0.18)'; ctx.fillRect(b.x, b.y, b.w, b.h);
    }
    ctx.strokeStyle = mode === 'unknown' ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.96)';
    ctx.lineWidth = mode === 'lock' ? 3 : 2; ctx.setLineDash(mode === 'lock' ? [] : [7, 5]);
    ctx.strokeRect(b.x, b.y, b.w, b.h); ctx.setLineDash([]);
    if (label) {
      ctx.font = '800 15px "Rubik", -apple-system, "PingFang TC", sans-serif';
      const tw = ctx.measureText(label).width + 12, y = Math.max(20, b.y - 4);
      ctx.fillStyle = 'rgba(0,0,0,0.65)'; ctx.fillRect(b.x, y - 19, tw, 19);
      ctx.fillStyle = '#fff'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(label, b.x + 6, y - 2);
      if (sub) { ctx.font = '600 11px -apple-system, "PingFang TC", sans-serif'; ctx.fillStyle = 'rgba(255,255,255,0.8)'; ctx.fillText(sub, b.x + tw + 4, y - 3); }
    }
    if (hp != null) {
      const n = 10, gw = Math.max(3, Math.min(8, b.w / n - 1));
      for (let i = 0; i < n; i++) { ctx.fillStyle = i < hp ? '#ff4d4d' : 'rgba(255,255,255,0.25)'; ctx.fillRect(b.x + i * (gw + 1), b.y + b.h + 4, gw, 4); }
    }
    ctx.restore();
  }
  E.drawPersonBox = drawPersonBox;

  // 由伺服器座標投影出的人形框
  E.projectPersonBox = function (x, z, height) {
    const hgt = height || 1.72;
    const dx = x - pos.x, dz = z - pos.z, d = Math.hypot(dx, dz) || 1;
    const rx = (-dz / d) * 0.28, rz = (dx / d) * 0.28;
    let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (const y of [0, hgt]) for (const s of [-1, 1]) {
      const c = toCam(dx + rx * s, y - E.EYE, dz + rz * s);
      if (c[2] < 0.3) return null;
      const p = toScreen(c);
      x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]);
    }
    if (x1 < 0 || x0 > W || y1 < 0 || y0 > H) return null;
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  };

  // 受傷／開火效果
  E.flash = { hitUntil: 0, fireUntil: 0, markUntil: 0, fovMul: 1, killUntil: 0 };
  function drawEffects(now) {
    const f = E.flash;
    if (now < f.fireUntil) {
      const a = (f.fireUntil - now) / 120;
      ctx.fillStyle = `rgba(255,255,255,${0.12 * a})`; ctx.fillRect(0, 0, W, H);
    }
    if (now < f.markUntil) { // 命中標記
      const cx = W / 2, cy = H / 2, a = (f.markUntil - now) / 380;
      ctx.save(); ctx.strokeStyle = `rgba(255,77,77,${a})`; ctx.lineWidth = 3;
      ctx.beginPath(); for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) { ctx.moveTo(cx + sx * 8, cy + sy * 8); ctx.lineTo(cx + sx * 17, cy + sy * 17); } ctx.stroke(); ctx.restore();
    }
    if (now < f.hitUntil) {
      const a = (f.hitUntil - now) / 500;
      const g = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.25, W / 2, H / 2, Math.max(W, H) * 0.7);
      g.addColorStop(0, 'rgba(255,0,0,0)'); g.addColorStop(1, `rgba(255,20,30,${0.55 * a})`);
      ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    }
  }

  // ==========================================================
  // 後台牆面／牆角偵測（沿用 v5，改為低頻率、不畫線）
  // ==========================================================
  const DS = 4;
  let DW = 0, DH = 0;
  const dm = document.createElement('canvas'), dmctx = dm.getContext('2d', { willReadFrequently: true });
  let dgray = new Float32Array(1), dgx = new Float32Array(1), dgy = new Float32Array(1), dmag = new Float32Array(1), dR = new Float32Array(1);
  const world = (E.world = { ents: [], walls: [], nid: 1, pass: 0 });
  const det = (E.det = { feats: [], vlines: [], ms: 0, passes: 0, harrisMax: 1 });
  let angSpeed = 0, prevF = null;
  function detectResize() {
    DW = Math.ceil(W / DS); DH = Math.ceil(H / DS); dm.width = DW; dm.height = DH;
    const n = DW * DH; dgray = new Float32Array(n); dgx = new Float32Array(n); dgy = new Float32Array(n); dmag = new Float32Array(n); dR = new Float32Array(n);
  }
  function vertDir(px, py) {
    const cx = (px - W / 2) / FPX, cy = -(py - H / 2) / FPX, u0 = B.sr[1], u1 = B.su[1], u2 = B.F[1];
    let sx = FPX * (u0 - cx * u2), sy = -FPX * (u1 - cy * u2); const l = Math.hypot(sx, sy) || 1; return [sx / l, sy / l];
  }
  function trackAng(dt) {
    if (prevF) { const d = Math.max(-1, Math.min(1, dot3(prevF, B.F))); angSpeed += (Math.acos(d) / Math.max(dt, 1e-3) - angSpeed) * Math.min(1, dt * 8); }
    prevF = [B.F[0], B.F[1], B.F[2]];
  }
  E.angSpeed = () => angSpeed;
  function detGradients() {
    const px = dmctx.getImageData(0, 0, DW, DH).data, n = DW * DH;
    for (let k = 0; k < n; k++) dgray[k] = 0.299 * px[4 * k] + 0.587 * px[4 * k + 1] + 0.114 * px[4 * k + 2];
    for (let j = 1; j < DH - 1; j++) for (let i = 1; i < DW - 1; i++) {
      const k = j * DW + i;
      const gx = (-dgray[k - DW - 1] + dgray[k - DW + 1] - 2 * dgray[k - 1] + 2 * dgray[k + 1] - dgray[k + DW - 1] + dgray[k + DW + 1]) / 4;
      const gy = (-dgray[k - DW - 1] - 2 * dgray[k - DW] - dgray[k - DW + 1] + dgray[k + DW - 1] + 2 * dgray[k + DW] + dgray[k + DW + 1]) / 4;
      dgx[k] = gx; dgy[k] = gy; dmag[k] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  function detHarris() {
    let mx = 0;
    for (let j = 2; j < DH - 2; j++) for (let i = 2; i < DW - 2; i++) {
      let sxx = 0, syy = 0, sxy = 0;
      for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) { const k = (j + b) * DW + i + a, ix = dgx[k], iy = dgy[k]; sxx += ix * ix; syy += iy * iy; sxy += ix * iy; }
      const r = sxx * syy - sxy * sxy - 0.05 * (sxx + syy) * (sxx + syy);
      dR[j * DW + i] = r; if (r > mx) mx = r;
    }
    det.harrisMax = mx;
  }
  function detVertical(thr) {
    const NB = 720, votes = new Float32Array(NB), cb = [], ck = [];
    for (let j = 1; j < DH - 1; j++) for (let i = 1; i < DW - 1; i++) {
      const k = j * DW + i, m = dmag[k]; if (m < thr) continue;
      const px = (i + 0.5) * DS, py = (j + 0.5) * DS, vv = vertDir(px, py);
      const tx = -dgy[k] / m, ty = dgx[k] / m;
      if (Math.abs(tx * vv[0] + ty * vv[1]) < 0.965) continue;
      const r = rayW(px, py), az = Math.atan2(r[0], -r[2]);
      const bin = Math.min(NB - 1, Math.floor(((az + Math.PI) / (2 * Math.PI)) * NB));
      votes[bin]++; cb.push(bin); ck.push(k);
    }
    const sv = new Float32Array(NB);
    for (let b = 0; b < NB; b++) sv[b] = votes[(b + NB - 1) % NB] + votes[b] + votes[(b + 1) % NB];
    const minV = Math.max(14, 0.08 * DH), peaks = [];
    for (let b = 0; b < NB; b++) {
      if (sv[b] < minV) continue;
      let top = true;
      for (let q = -3; q <= 3; q++) if (sv[(b + q + NB) % NB] > sv[b] || (q < 0 && sv[(b + q + NB) % NB] === sv[b])) { top = false; break; }
      if (top) peaks.push(b);
    }
    const lines = [];
    for (const pb of peaks) {
      const rowN = new Uint16Array(DH), rowX = new Float32Array(DH);
      for (let q = 0; q < cb.length; q++) {
        if (Math.abs(((cb[q] - pb + NB + NB / 2) % NB) - NB / 2) > 1) continue;
        const k = ck[q], j = (k / DW) | 0; rowN[j]++; rowX[j] += k - j * DW;
      }
      let best = null, start = -1, last = -9;
      const consider = () => { if (start >= 0 && (!best || last - start > best[1] - best[0])) best = [start, last]; };
      for (let j = 0; j < DH; j++) { if (!rowN[j]) continue; if (start < 0 || j - last > 3) { consider(); start = j; } last = j; }
      consider();
      if (!best) continue;
      const jt = best[0], jb = best[1];
      if (jb - jt < Math.max(14, 0.08 * DH)) continue;
      let s = 0, nn = 0;
      for (let j = Math.max(jt, jb - 4); j <= jb; j++) if (rowN[j]) { s += rowX[j]; nn += rowN[j]; }
      if (!nn) continue;
      const xb = (s / nn + 0.5) * DS, yb = (jb + 1) * DS, ib = Math.round(xb / DS), clipped = jb >= DH - 5;
      const az = ((pb + 0.5) / NB) * 2 * Math.PI - Math.PI;
      lines.push({ xb, yb, ib, jb, jt, az, fp: clipped ? null : floorPoint(xb, yb) });
    }
    det.vlines = lines;
    return lines;
  }
  function junctionSide(ib, jb, sgn, thr) {
    const vv = vertDir(ib * DS, jb * DS);
    let C2 = 0, S2 = 0, Wt = 0, n = 0;
    for (let j = Math.max(1, jb - 7); j <= Math.min(DH - 2, jb + 5); j++) for (let s = 3; s <= 18; s++) {
      const i = ib + sgn * s; if (i < 1 || i >= DW - 1) break;
      const k = j * DW + i, m = dmag[k]; if (m < thr) continue;
      const tx = -dgy[k] / m, ty = dgx[k] / m;
      if (Math.abs(tx * vv[0] + ty * vv[1]) > 0.85) continue;
      const phi = Math.atan2(ty, tx); C2 += m * Math.cos(2 * phi); S2 += m * Math.sin(2 * phi); Wt += m; n++;
    }
    if (n < 6 || Math.hypot(C2, S2) / Wt < 0.55) return null;
    const phi = 0.5 * Math.atan2(S2, C2);
    if (Math.abs(Math.cos(phi)) < 0.25) return null;
    return { phi, n };
  }
  function marchLine(i0, j0, dx, dy, thr) {
    const alongX = Math.abs(dx) >= Math.abs(dy), step = alongX ? Math.sign(dx) : Math.sign(dy), slope = alongX ? dy / dx : dx / dy;
    let x = i0, y = j0, miss = 0, gx = x, gy = y;
    for (let s = 0; s < DW * 2; s++) {
      if (alongX) { x += step; y += slope * step; } else { y += step; x += slope * step; }
      const ii = Math.round(x), jj = Math.round(y);
      if (ii < 3 || ii >= DW - 3 || jj < 3 || jj >= DH - 3) break;
      let bm = 0, bo = 0;
      for (let o = -2; o <= 2; o++) { const m = alongX ? dmag[(jj + o) * DW + ii] : dmag[jj * DW + ii + o]; if (m > bm) { bm = m; bo = o; } }
      if (bm >= thr * 0.6) { miss = 0; if (alongX) y += bo * 0.35; else x += bo * 0.35; gx = x; gy = y; } else if (++miss > 4) break;
    }
    return [gx, gy];
  }
  function harrisNear(i, j) {
    let m = 0;
    for (let b = -3; b <= 3; b++) for (let a = -3; a <= 3; a++) { const ii = i + a, jj = j + b; if (ii < 2 || jj < 2 || ii >= DW - 2 || jj >= DH - 2) continue; if (dR[jj * DW + ii] > m) m = dR[jj * DW + ii]; }
    return m > (det.harrisMax || 1) * 0.06;
  }
  function classifyLines(lines, thr) {
    const obsE = [], obsW = [];
    const ranged = lines.filter((l) => l.fp && l.fp.d >= 0.6 && l.fp.d <= 9.0), used = new Set();
    for (let a = 0; a < ranged.length; a++) for (let b = a + 1; b < ranged.length; b++) {
      if (used.has(a) || used.has(b)) continue;
      const A = ranged[a], Bn = ranged[b], da = Math.abs(wrapPi(A.az - Bn.az)), r = (A.fp.d + Bn.fp.d) / 2, wd = r * da;
      if (wd < 0.15 || wd > 0.75 || Math.abs(A.fp.d - Bn.fp.d) > 0.38) continue;
      used.add(a); used.add(b);
      obsE.push({ type: 'column', x: (A.fp.x + Bn.fp.x) / 2, z: (A.fp.z + Bn.fp.z) / 2, w: 0.4, dirs: [] });
    }
    ranged.forEach((L, idx) => {
      if (used.has(idx)) return;
      const jb = Math.min(DH - 2, L.jb + 1);
      const sides = [-1, 1].map((sg) => {
        const js = junctionSide(L.ib, jb, sg, thr); if (!js) return null;
        let dx = Math.cos(js.phi), dy = Math.sin(js.phi); if (sg * dx < 0) { dx = -dx; dy = -dy; }
        const e = marchLine(L.ib, jb, dx, dy, thr);
        return { sg, phi: js.phi, end: floorPoint((e[0] + 0.5) * DS, (e[1] + 0.5) * DS) };
      });
      let type = 'edge';
      const dirs = [], [sl, sr] = sides;
      if (sl && sr) { let dphi = Math.abs(sl.phi - sr.phi) % Math.PI; dphi = Math.min(dphi, Math.PI - dphi); if (dphi > 14 * DEG || (dphi > 8 * DEG && harrisNear(L.ib, jb))) type = 'corner'; }
      for (const sd of sides) {
        if (!sd || !sd.end) continue;
        const len = Math.hypot(sd.end.x - L.fp.x, sd.end.z - L.fp.z);
        if (len < 0.8 || len > 10) continue;
        dirs.push(Math.atan2(sd.end.z - L.fp.z, sd.end.x - L.fp.x));
        obsW.push({ x1: L.fp.x, z1: L.fp.z, x2: sd.end.x, z2: sd.end.z });
      }
      obsE.push({ type, x: L.fp.x, z: L.fp.z, w: 0.2, dirs });
    });
    return { obsE, obsW };
  }
  function fuseEnt(o, now) {
    let best = null, bd = 0.65;
    for (const e of world.ents) { const d = Math.hypot(e.x - o.x, e.z - o.z); if (d < bd) { bd = d; best = e; } }
    if (!best) { best = { id: world.nid++, x: o.x, z: o.z, w: o.w || 0.4, hits: 0, miss: 0, votes: { corner: 0, column: 0, edge: 0 }, dirs: [], pass: 0, sent: 0 }; world.ents.push(best); }
    else { best.x += (o.x - best.x) * 0.22; best.z += (o.z - best.z) * 0.22; }
    best.hits = Math.min(best.hits + 1, 30); best.miss = 0; best.pass = world.pass; best.last = now;
    best.votes[o.type]++;
    const v = best.votes;
    best.type = v.corner >= 2 && v.corner >= v.column ? 'corner' : v.column >= 2 && v.column >= v.edge ? 'column' : v.corner > v.edge ? 'corner' : v.column > v.edge ? 'column' : 'edge';
    for (const a of o.dirs) {
      let hit = false;
      for (let q = 0; q < best.dirs.length; q++) if (Math.abs(wrapPi(best.dirs[q] - a)) < 15 * DEG) { best.dirs[q] += wrapPi(a - best.dirs[q]) * 0.3; hit = true; break; }
      if (!hit && best.dirs.length < 3) best.dirs.push(a);
    }
  }
  function fuseWall(w, now) {
    const len = Math.hypot(w.x2 - w.x1, w.z2 - w.z1); if (len < 0.8) return;
    const ang = Math.atan2(w.z2 - w.z1, w.x2 - w.x1);
    for (const e of world.walls) {
      const ea = Math.atan2(e.z2 - e.z1, e.x2 - e.x1);
      let da = Math.abs(wrapPi(ang - ea)); da = Math.min(da, Math.PI - da); if (da > 12 * DEG) continue;
      const ux = Math.cos(ea), uz = Math.sin(ea), nx = -uz, nz = ux;
      const off1 = (w.x1 - e.x1) * nx + (w.z1 - e.z1) * nz, off2 = (w.x2 - e.x1) * nx + (w.z2 - e.z1) * nz;
      if (Math.abs(off1) > 0.45 || Math.abs(off2) > 0.45) continue;
      const elen = Math.hypot(e.x2 - e.x1, e.z2 - e.z1);
      const t1 = (w.x1 - e.x1) * ux + (w.z1 - e.z1) * uz, t2 = (w.x2 - e.x1) * ux + (w.z2 - e.z1) * uz;
      const lo = Math.min(0, t1, t2), hi = Math.max(elen, t1, t2);
      if (Math.min(elen, Math.max(t1, t2)) - Math.max(0, Math.min(t1, t2)) < -0.8 || hi - lo > 14) continue;
      const off = ((off1 + off2) / 2) * 0.25;
      const bx = e.x1 + nx * off, bz = e.z1 + nz * off;
      e.x1 = bx + ux * lo; e.z1 = bz + uz * lo; e.x2 = bx + ux * hi; e.z2 = bz + uz * hi;
      e.hits = Math.min(e.hits + 1, 30); e.miss = 0; e.pass = world.pass; e.last = now;
      return;
    }
    world.walls.push({ id: world.nid++, x1: w.x1, z1: w.z1, x2: w.x2, z2: w.z2, hits: 1, miss: 0, pass: world.pass, last: now, sent: 0 });
  }
  function inView(x, z, tol) {
    const dx = x - pos.x, dz = z - pos.z, d = Math.hypot(dx, dz);
    if (d < 1.2 || d > 8) return false;
    return Math.abs(wrapPi(Math.atan2(dx, -dz) - E.mapHeading())) < tol;
  }
  function decayWorld() {
    const tol = 22 * DEG;
    for (const e of world.ents) { if (e.pass === world.pass || !inView(e.x, e.z, tol)) continue; if (++e.miss > 10) { e.hits -= 1; e.miss = 0; } }
    for (const w of world.walls) { if (w.pass === world.pass || !inView((w.x1 + w.x2) / 2, (w.z1 + w.z2) / 2, tol)) continue; if (++w.miss > 14) { w.hits -= 1; w.miss = 0; } }
    world.ents = world.ents.filter((e) => e.hits > 0);
    world.walls = world.walls.filter((w) => w.hits > 0);
  }
  function runDetection(now) {
    if (DW < 8 || angSpeed > 1.2 || E.moveSpeed > 1.2) return;
    const t0 = performance.now();
    dmctx.drawImage(cam, 0, 0, DW, DH);
    detGradients(); detHarris();
    const thr = Math.max(9, S.dotThreshold * 0.9);
    const lines = detVertical(thr);
    world.pass++;
    const { obsE, obsW } = classifyLines(lines, thr);
    for (const o of obsE) fuseEnt(o, now);
    for (const w of obsW) fuseWall(w, now);
    decayWorld();
    det.passes++; det.ms += (performance.now() - t0 - det.ms) * 0.2;
  }
  // 牆角是陰角（從角內看，例如房間的角）還是陽角（凸出的轉角）
  function cornerKind(e) {
    if (e.dirs.length < 2) return 'corner_in';
    const a = [Math.cos(e.dirs[0]), Math.sin(e.dirs[0])], b = [Math.cos(e.dirs[1]), Math.sin(e.dirs[1])];
    const v = [pos.x - e.x, pos.z - e.z];
    const cross = (p, q) => p[0] * q[1] - p[1] * q[0];
    const inside = cross(a, v) * cross(a, b) >= 0 && cross(b, v) * cross(b, a) >= 0;
    return inside ? 'corner_in' : 'corner_out';
  }
  // 給伺服器的回報（只送新的或更新過的）
  E.obstacleReport = function () {
    const out = [], now = performance.now();
    for (const w of world.walls) {
      if (w.hits < 3 || now - w.sent < 4000) continue;
      w.sent = now; out.push({ k: 'wall', x1: +w.x1.toFixed(2), z1: +w.z1.toFixed(2), x2: +w.x2.toFixed(2), z2: +w.z2.toFixed(2) });
    }
    for (const e of world.ents) {
      if (e.hits < 4 || e.type === 'edge' || now - e.sent < 4000) continue;
      e.sent = now;
      out.push({ k: e.type === 'column' ? 'column' : cornerKind(e), x: +e.x.toFixed(2), z: +e.z.toFixed(2), w: 0.4, d: 0.4 });
    }
    return out.slice(0, 20);
  };
  E.resetWorld = () => { world.ents = []; world.walls = []; };

  // ==========================================================
  // 每一格
  //   mode：'raw'（只有鏡頭＋準心，設定/校正用）、'play'（完整 AR）
  //   overlay(ctx, W, H)：game.js 疊加敵人框等
  // ==========================================================
  let detTick = 0;
  E.frame = function (dt, now, o) {
    o = o || {};
    updateCamera();
    trackAng(dt);
    updateMove(dt);
    const play = o.mode === 'play';
    updateVision(play && S.limitOn);
    if (play) updateScanMemory(dt);
    drawVideo(sctx);
    updateFlow();
    if (E.map && (o.mode === 'play' || o.mode === 'position') && ++detTick % 6 === 0) runDetection(now);
    if (E.map && o.mode !== 'raw') { drawFloorLayer(sctx, o); drawStructures(sctx); if (S.debug) drawObstaclesDebug(sctx); }
    if (play && S.limitOn) {
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
      analyzeDots(); drawDots(); drawBand();
      blit(C.l, C.t, C.r - C.l, C.b - C.t);
    } else {
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      ctx.drawImage(scene, 0, 0, W, H);
    }
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    if (o.overlay) o.overlay(ctx, W, H);
    if (o.crosshair) drawCrosshair(o.crosshair);
    drawEffects(now);
  };
})();
