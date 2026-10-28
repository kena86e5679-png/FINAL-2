// ==========================================================
// click clip!! — 房間邏輯（權威狀態）
//   同一份程式碼跑在兩個地方：
//     · Node 伺服器（server.js，require('./room.js')）
//     · 手機瀏覽器（GitHub Pages 沒有伺服器時，第一位玩家的手機當主機，透過 PeerJS 轉送）
//   所有「誰被打中、地圖多大、哪裡生成建築」都由這裡決定，玩家手機只提出請求。
//
//   座標系（地圖座標，單位公尺）：
//     原點 = 符號 S1（對角線起點），x 軸沿場地一條邊，z 軸沿另一條邊；S2 = (W, D)。
//     方位角 heading：0 = 朝 -z，π/2 = 朝 +x（與手機端引擎相同的慣例）。
//
//   流程（phase）：
//     lobby    進房、拿到 A～Z 編號、互相拍照建檔
//     setup    隨機兩位玩家放符號、站上去，量對角線 → 矩形範圍（畢氏定理）
//     position 每位玩家站上 S1 或 S2，把準心對準另一個符號 → 位置與方向校正
//     play     10 條命，點擊攻擊；伺服器生成建築
//     over     剩一人存活
// ==========================================================
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.ClickClipRoom = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  const COLORS = ['#ff5a4e', '#4ec9ff', '#ffd23f', '#7cff6b', '#ff7ad9', '#b48cff', '#ff9f43', '#5effc8'];
  const CFG = {
    MAX_PLAYERS: 8,
    HP: 10,
    SHOT_COOLDOWN_MS: 450,
    SHOT_MAX_DIST: 30,
    SHOT_SERVER_TOL: 40 * Math.PI / 180, // 伺服器端寬鬆複核（精細判定在手機端）
    POSE_MIN_MS: 60,
    POSE_BROADCAST_MS: 200,
    GEN_FIRST_MS: 1500,
    GEN_INTERVAL_MS: 7000,
    REJOIN_MS: 120000,
    PROFILE_MAX_PER_TARGET: 4,
    GPS_MAX_ACC: 12,
  };
  const TAU = Math.PI * 2;
  const wrapPi = (a) => { a = (a + Math.PI) % TAU; if (a < 0) a += TAU; return a - Math.PI; };
  const num = (v, lim) => (typeof v === 'number' && isFinite(v) ? Math.max(-lim, Math.min(lim, v)) : null);
  const str = (v, n) => (typeof v === 'string' ? v.replace(/[\u0000-\u001f<>]/g, '').slice(0, n) : '');
  const r2 = (v) => Math.round(v * 100) / 100;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  // ---------- 幾何 ----------
  // 對角線 → 矩形。d：對角線長；diagH：對角線方位；edgeH：一條邊的方位（null = 不指定 → 正方形）
  // 回傳 S2 = (W, D)，且 W, D ≥ 0，W² + D² = d²（畢氏定理）
  function computeRect(d, diagH, edgeH) {
    let delta = Math.PI / 4;
    let square = true;
    if (edgeH != null && isFinite(edgeH)) {
      const q = Math.PI / 2;
      delta = (((diagH - edgeH) % q) + q) % q; // 對角線和 +x 軸的夾角，落在 [0°, 90°)
      square = false;
    }
    const W = d * Math.cos(delta), D = d * Math.sin(delta);
    // +x 軸的方位（在量測者的方位座標裡）；地圖方位 = 原始方位 - xH + π/2
    const xH = diagH - delta;
    const thin = Math.min(W, D) < 1.5 || delta < 10 * Math.PI / 180 || delta > 80 * Math.PI / 180;
    return { W: r2(W), D: r2(D), diag: r2(d), delta, xH, square, thin, check: r2(Math.hypot(W, D)) };
  }

  function gpsVector(a, b) { // a → b，回傳 { east, north }（公尺）
    const lat = ((a.lat + b.lat) / 2) * Math.PI / 180;
    return { east: (b.lon - a.lon) * 111320 * Math.cos(lat), north: (b.lat - a.lat) * 110540 };
  }

  function gridLabel(i, j) { return (j + 1) + '-' + (i + 1); }

  // AABB（地圖軸向）
  function boxOf(o) {
    const along = Math.abs(Math.sin(o.rot || 0)) > 0.7; // 轉 90°
    const hx = (along ? o.d : o.w) / 2, hz = (along ? o.w : o.d) / 2;
    return { x0: o.x - hx, x1: o.x + hx, z0: o.z - hz, z1: o.z + hz };
  }
  function rayBox(ox, oz, dx, dz, b, pad) {
    let t0 = 0, t1 = 1e9;
    const lo = [b.x0 - pad, b.z0 - pad], hi = [b.x1 + pad, b.z1 + pad], o = [ox, oz], d = [dx, dz];
    for (let k = 0; k < 2; k++) {
      if (Math.abs(d[k]) < 1e-9) { if (o[k] < lo[k] || o[k] > hi[k]) return Infinity; }
      else {
        let a = (lo[k] - o[k]) / d[k], c = (hi[k] - o[k]) / d[k];
        if (a > c) { const t = a; a = c; c = t; }
        t0 = Math.max(t0, a); t1 = Math.min(t1, c);
        if (t0 > t1) return Infinity;
      }
    }
    return t0;
  }
  function raySeg(ox, oz, dx, dz, w) {
    const ex = w.x2 - w.x1, ez = w.z2 - w.z1;
    const den = dx * ez - dz * ex;
    if (Math.abs(den) < 1e-9) return Infinity;
    const t = ((w.x1 - ox) * ez - (w.z1 - oz) * ex) / den;
    const u = ((w.x1 - ox) * dz - (w.z1 - oz) * dx) / den;
    return t > 0 && u >= 0 && u <= 1 ? t : Infinity;
  }
  function rayCircle(ox, oz, dx, dz, cx, cz, r) {
    const fx = ox - cx, fz = oz - cz;
    const b = fx * dx + fz * dz, c = fx * fx + fz * fz - r * r;
    const disc = b * b - c;
    if (disc < 0) return Infinity;
    const t = -b - Math.sqrt(disc);
    return t > 0 ? t : (c < 0 ? 0 : Infinity);
  }
  function segBoxDist(w, b) { // 線段到 AABB 的最短距離（取樣近似）
    let best = Infinity;
    for (let i = 0; i <= 12; i++) {
      const x = w.x1 + (w.x2 - w.x1) * i / 12, z = w.z1 + (w.z2 - w.z1) * i / 12;
      const dx = Math.max(b.x0 - x, 0, x - b.x1), dz = Math.max(b.z0 - z, 0, z - b.z1);
      best = Math.min(best, Math.hypot(dx, dz));
    }
    return best;
  }
  function boxGap(a, b) {
    const dx = Math.max(a.x0 - b.x1, b.x0 - a.x1, 0), dz = Math.max(a.z0 - b.z1, b.z0 - a.z1, 0);
    return Math.hypot(dx, dz);
  }

  // ==========================================================
  class Room {
    constructor(code, opts) {
      this.code = code;
      this.out = opts.send;        // (cid, obj) => void
      this.now = opts.now || (() => Date.now());
      this.rand = opts.rand || Math.random;
      this.cfg = Object.assign({}, CFG, opts.cfg || {});
      this.players = new Map();    // cid -> player
      this.viewers = new Set();
      this.profiles = {};          // letter -> [{ by, feat, thumb, ts }]
      this.resetGame(true);
      this.lastPoseBroadcast = 0;
      this.poseDirty = false;
      this.obstDirty = [];
      this.lastObstBroadcast = 0;
      this.events = [];
    }

    resetGame(full) {
      if (full) {
        this.phase = 'lobby';
        this.setup = null;
        this.map = null;
        this.anchors = [];
        this.obstacles = [];
        this.visited = {};
        this.nextOid = 1;
      }
      this.structures = [];
      this.nextSid = 1;
      this.genNext = 0;
      this.countdownAt = 0;
      this.winner = null;
      this.playStart = 0;
      for (const p of this.players.values()) {
        p.hp = this.cfg.HP; p.alive = true; p.ready = false; p.kills = 0; p.hitsTaken = 0;
        if (full) { p.calib = false; p.posQ = 'none'; }
      }
    }

    // ---------- 傳送 ----------
    send(cid, obj) { try { this.out(cid, obj); } catch (e) { /* ignore */ } }
    toAll(obj, exceptCid) {
      for (const p of this.players.values()) if (p.connected && p.cid !== exceptCid) this.send(p.cid, obj);
      for (const v of this.viewers) if (v !== exceptCid) this.send(v, obj);
    }
    toViewers(obj) { for (const v of this.viewers) this.send(v, obj); }
    log(msg, kind) {
      const ev = { t: 'log', msg, kind: kind || 'info', ts: this.now() };
      this.events.push(ev); if (this.events.length > 60) this.events.shift();
      this.toAll(ev);
    }

    // ---------- 公開資料 ----------
    pubPlayer(p) {
      return { id: p.id, name: p.name, color: p.color, connected: p.connected, x: r2(p.x), z: r2(p.z), h: r2(p.h),
        hp: p.hp, alive: p.alive, ready: p.ready, calib: p.calib, posQ: p.posQ, kills: p.kills,
        photos: (this.profiles[p.id] || []).length };
    }
    pubProfiles(withFeat) {
      const out = {};
      for (const k in this.profiles) out[k] = this.profiles[k].map((q) => ({ by: q.by, thumb: q.thumb, feat: withFeat ? q.feat : undefined, ts: q.ts }));
      return out;
    }
    pubSetup() {
      const s = this.setup;
      if (!s) return null;
      return { a: s.a, b: s.b, standA: !!s.standA, standB: !!s.standB, edge: s.edge != null, aim: s.aim ? r2(s.aim.d) : null,
        gpsA: s.standA && s.standA.gps ? r2(s.standA.gps.acc) : null, gpsB: s.standB && s.standB.gps ? r2(s.standB.gps.acc) : null,
        proposal: s.proposal ? { W: s.proposal.W, D: s.proposal.D, diag: s.proposal.diag, source: s.proposal.source, thin: s.proposal.thin, square: s.proposal.square, warn: s.proposal.warn || '' } : null };
    }
    fullState(forViewer) {
      const players = [];
      for (const p of this.players.values()) players.push(this.pubPlayer(p));
      return { t: 'state', code: this.code, phase: this.phase, players, setup: this.pubSetup(), map: this.map, anchors: this.anchors,
        obstacles: this.obstacles, structures: this.structures, visited: Object.keys(this.visited), winner: this.winner,
        countdownAt: this.countdownAt, profiles: this.pubProfiles(true), events: forViewer ? this.events.slice(-20) : undefined,
        now: this.now(), cfg: { HP: this.cfg.HP, MAX_PLAYERS: this.cfg.MAX_PLAYERS } };
    }
    broadcastState() {
      const st = this.fullState(false);
      for (const p of this.players.values()) if (p.connected) this.send(p.cid, st);
      if (this.viewers.size) { const sv = this.fullState(true); for (const v of this.viewers) this.send(v, sv); }
    }
    setPhase(ph, why) {
      this.phase = ph;
      this.broadcastState();
      if (why) this.log(why, 'phase');
    }

    // ---------- 進出 ----------
    join(cid, m) {
      if (m.role === 'viewer') {
        this.viewers.add(cid);
        this.send(cid, Object.assign({ t: 'welcome', role: 'viewer' }, { state: this.fullState(true) }));
        const traces = {};
        for (const p of this.players.values()) traces[p.id] = p.trace;
        this.send(cid, { t: 'traces', traces });
        return true;
      }
      // 斷線重連：帶著 token 回來就恢復原本的編號與狀態
      if (m.token) {
        for (const p of this.players.values()) {
          if (p.token === m.token) {
            if (p.connected && p.cid !== cid) this.send(p.cid, { t: 'error', msg: '你在另一個分頁登入了', fatal: true });
            this.players.delete(p.cid);
            p.cid = cid; p.connected = true; p.lastSeen = this.now();
            this.players.set(cid, p);
            this.send(cid, { t: 'welcome', role: 'player', you: { id: p.id, color: p.color, token: p.token, name: p.name }, state: this.fullState(false) });
            this.broadcastState();
            this.log(`${p.id} 重新連線`);
            return true;
          }
        }
      }
      let count = 0;
      for (const p of this.players.values()) if (p.connected || this.now() - p.lastSeen < this.cfg.REJOIN_MS) count++;
      if (count >= this.cfg.MAX_PLAYERS) { this.send(cid, { t: 'error', msg: `房間已滿（最多 ${this.cfg.MAX_PLAYERS} 人）`, fatal: true }); return false; }
      const used = new Set([...this.players.values()].map((p) => p.id));
      const free = LETTERS.split('').filter((c) => !used.has(c));
      const id = free[Math.floor(this.rand() * free.length)];
      const n = used.size;
      const p = {
        cid, id, name: str(m.name, 12) || ('玩家' + id), color: COLORS[n % COLORS.length], token: Math.random().toString(36).slice(2) + Date.now().toString(36),
        connected: true, lastSeen: this.now(), x: 0, z: 0, h: 0, sp: 0, posQ: 'none', calib: false, ready: false,
        hp: this.cfg.HP, alive: true, kills: 0, hitsTaken: 0, lastShot: 0, lastPose: 0, lastProfile: 0, trace: [], lastObst: 0,
      };
      this.players.set(cid, p);
      this.send(cid, { t: 'welcome', role: 'player', you: { id: p.id, color: p.color, token: p.token, name: p.name }, state: this.fullState(false) });
      this.broadcastState();
      this.log(`${p.id}（${p.name}）加入`);
      return true;
    }

    leave(cid) {
      if (this.viewers.delete(cid)) return;
      const p = this.players.get(cid);
      if (!p) return;
      p.connected = false; p.lastSeen = this.now();
      if (this.phase === 'lobby') { this.players.delete(cid); this.log(`${p.id} 離開`); }
      else this.log(`${p.id} 斷線（${Math.round(this.cfg.REJOIN_MS / 1000)} 秒內可重連）`);
      if (this.setup && (p.id === this.setup.a || p.id === this.setup.b) && this.phase === 'setup') this.pickSetupPlayers(true);
      this.broadcastState();
      this.checkOver();
    }

    // ---------- 訊息 ----------
    message(cid, m) {
      if (!m || typeof m.t !== 'string') return;
      if (m.t === 'hello') { this.join(cid, m); return; }
      const p = this.players.get(cid);
      if (!p) return;
      p.lastSeen = this.now();
      const fn = this['on_' + m.t];
      if (typeof fn === 'function') fn.call(this, p, m);
    }

    // 拍照建檔：某位玩家替 target 拍的特徵
    on_profile(p, m) {
      const target = str(m.target, 1).toUpperCase();
      if (!LETTERS.includes(target) || target === p.id) return;
      if (![...this.players.values()].some((q) => q.id === target)) return;
      if (this.now() - p.lastProfile < 700) return;
      p.lastProfile = this.now();
      const feat = m.feat && typeof m.feat === 'object' ? m.feat : null;
      if (!feat) return;
      const thumb = typeof m.thumb === 'string' && m.thumb.length < 24000 && m.thumb.startsWith('data:image/') ? m.thumb : '';
      const list = this.profiles[target] || (this.profiles[target] = []);
      list.push({ by: p.id, feat, thumb, ts: this.now() });
      while (list.length > this.cfg.PROFILE_MAX_PER_TARGET) list.shift();
      this.toAll({ t: 'profile', target, by: p.id, feat, thumb, n: list.length });
      this.broadcastState();
    }
    on_unprofile(p, m) {
      const target = str(m.target, 1).toUpperCase();
      const list = this.profiles[target];
      if (!list) return;
      this.profiles[target] = list.filter((q) => q.by !== p.id);
      this.broadcastState();
    }

    coverage() {
      const out = {};
      for (const q of this.players.values()) out[q.id] = (this.profiles[q.id] || []).length;
      return out;
    }

    // ---------- 範圍設定 ----------
    on_startSetup(p, m) {
      if (this.phase !== 'lobby' && this.phase !== 'over') return;
      const list = [...this.players.values()].filter((q) => q.connected);
      if (!list.length) return;
      if (!m.force && list.length > 1) {
        const missing = list.filter((q) => !(this.profiles[q.id] || []).length).map((q) => q.id);
        if (missing.length) { this.send(p.cid, { t: 'need', what: 'photos', missing }); return; }
      }
      this.resetGame(true);
      this.setup = { a: null, b: null, standA: null, standB: null, edge: null, edgeAbs: null, aim: null, manual: null, proposal: null };
      this.pickSetupPlayers(false);
      this.setPhase('setup', `範圍設定：${this.setup.a} 放 S1、${this.setup.b} 放 S2`);
    }
    pickSetupPlayers(replaceGone) {
      const s = this.setup;
      const list = [...this.players.values()].filter((q) => q.connected).map((q) => q.id);
      if (!list.length) return;
      const pick = () => list.splice(Math.floor(this.rand() * list.length), 1)[0];
      if (!replaceGone) {
        s.a = pick();
        s.b = list.length ? pick() : s.a; // 只有一人時由同一人完成兩端
      } else {
        const alive = (id) => [...this.players.values()].some((q) => q.id === id && q.connected);
        if (!alive(s.a)) { s.a = list.filter((x) => x !== s.b)[0] || list[0]; s.standA = null; s.edge = null; s.aim = null; }
        if (!alive(s.b)) { s.b = list.filter((x) => x !== s.a)[0] || s.a; s.standB = null; }
      }
    }
    validGps(g) {
      if (!g || typeof g !== 'object') return null;
      const lat = num(g.lat, 90), lon = num(g.lon, 180), acc = num(g.acc, 1e5), n = num(g.n, 1e4);
      if (lat === null || lon === null || acc === null) return null;
      return { lat, lon, acc, n: n || 1 };
    }
    on_setupStand(p, m) {
      const s = this.setup;
      if (this.phase !== 'setup' || !s) return;
      const corner = m.corner === 'S2' ? 'S2' : 'S1';
      if (corner === 'S1' && p.id !== s.a) return;
      if (corner === 'S2' && p.id !== s.b) return;
      const rec = { ts: this.now(), gps: this.validGps(m.gps), abs: num(m.abs, 10) };
      if (corner === 'S1') s.standA = rec; else s.standB = rec;
      this.tryPropose();
      this.broadcastState();
    }
    on_setupGps(p, m) { // GPS 取樣完成後補送
      const s = this.setup;
      if (this.phase !== 'setup' || !s) return;
      const g = this.validGps(m.gps);
      if (!g) return;
      if (m.corner === 'S1' && p.id === s.a && s.standA) s.standA.gps = g;
      if (m.corner === 'S2' && p.id === s.b && s.standB) s.standB.gps = g;
      this.tryPropose();
      this.broadcastState();
    }
    on_setupEdge(p, m) {
      const s = this.setup;
      if (this.phase !== 'setup' || !s || p.id !== s.a) return;
      const raw = num(m.raw, 10);
      if (raw === null) { s.edge = null; s.edgeAbs = null; }
      else { s.edge = raw; s.edgeAbs = num(m.abs, 10); }
      this.tryPropose();
      this.broadcastState();
    }
    on_setupAim(p, m) { // S1 的人把準心對準 S2 的人腳下：距離 + 原始方位
      const s = this.setup;
      if (this.phase !== 'setup' || !s || p.id !== s.a) return;
      const d = num(m.d, 200), raw = num(m.raw, 10);
      if (d === null || raw === null || d < 1 || d > 80) return;
      s.aim = { d, raw, abs: num(m.abs, 10) };
      this.tryPropose();
      this.broadcastState();
    }
    on_setupManual(p, m) {
      const s = this.setup;
      if (this.phase !== 'setup' || !s || (p.id !== s.a && p.id !== s.b)) return;
      const W = num(m.W, 100), D = num(m.D, 100);
      if (W === null || D === null || W < 1 || D < 1) return;
      s.manual = { W, D };
      this.tryPropose();
      this.broadcastState();
    }
    tryPropose() {
      const s = this.setup;
      let prop = null;
      if (s.manual) {
        const d = Math.hypot(s.manual.W, s.manual.D);
        prop = { W: r2(s.manual.W), D: r2(s.manual.D), diag: r2(d), delta: Math.atan2(s.manual.D, s.manual.W), xH: null, square: false, thin: false, source: 'manual' };
      } else if (s.aim) {
        prop = computeRect(s.aim.d, s.aim.raw, s.edge);
        prop.source = 'aim';
        prop.frame = 'rawA';
      } else if (s.standA && s.standB && s.standA.gps && s.standB.gps && s.a !== s.b) {
        const g1 = s.standA.gps, g2 = s.standB.gps;
        const v = gpsVector(g1, g2);
        const d = Math.hypot(v.east, v.north);
        const acc = Math.max(g1.acc, g2.acc);
        if (d >= 1.5 && acc <= this.cfg.GPS_MAX_ACC) {
          const diagAbs = Math.atan2(v.east, v.north); // 方位角：北 0、東 π/2
          prop = computeRect(d, diagAbs, s.edgeAbs);
          prop.source = 'gps';
          prop.frame = 'abs';
          if (acc > d / 3) prop.warn = `GPS 誤差 ±${acc.toFixed(0)} m，接近場地大小，建議改用「瞄準量測」`;
        } else {
          s.gpsNote = `GPS 誤差 ±${acc.toFixed(0)} m、量到距離 ${d.toFixed(1)} m，不足以界定範圍`;
        }
      }
      s.proposal = prop;
    }
    on_setupConfirm(p) {
      const s = this.setup;
      if (this.phase !== 'setup' || !s || !s.proposal) return;
      if (p.id !== s.a && p.id !== s.b) return;
      const pr = s.proposal;
      const W = Math.max(1, pr.W), D = Math.max(1, pr.D);
      this.map = { W: r2(W), D: r2(D), diag: r2(Math.hypot(W, D)), source: pr.source, S1: [0, 0], S2: [r2(W), r2(D)],
        cols: Math.floor(W) + 1, rows: Math.floor(D) + 1, by: s.a, ts: this.now() };
      this.anchors = [{ aid: 'S1', x: 0, z: 0 }, { aid: 'S2', x: r2(W), z: r2(D) }];
      // S1 的人在瞄準時就站在 S1 上：直接幫他完成方向校正
      if (pr.source === 'aim' && pr.xH != null) {
        const a = [...this.players.values()].find((q) => q.id === s.a);
        if (a) {
          a.x = 0; a.z = 0; a.calib = true; a.posQ = 'corner';
          this.send(a.cid, { t: 'autoCalib', corner: 'S1', offset: wrapPi(Math.PI / 2 - pr.xH) });
        }
      }
      this.setPhase('position', `範圍確定：${this.map.W} × ${this.map.D} m（對角線 ${this.map.diag} m，${pr.source === 'aim' ? '瞄準量測' : pr.source === 'gps' ? 'GPS' : '手動輸入'}）`);
    }
    on_setupRedo(p) {
      const s = this.setup;
      if (this.phase !== 'setup' || !s) return;
      s.standA = s.standB = null; s.edge = s.edgeAbs = null; s.aim = null; s.manual = null; s.proposal = null;
      this.broadcastState();
    }
    on_setupReroll(p) { // 重新抽兩位設定者
      if (this.phase !== 'setup') return;
      this.setup = { a: null, b: null, standA: null, standB: null, edge: null, edgeAbs: null, aim: null, manual: null, proposal: null };
      this.pickSetupPlayers(false);
      this.setPhase('setup', `重新抽選：${this.setup.a} 放 S1、${this.setup.b} 放 S2`);
    }

    // ---------- 位置校正與準備 ----------
    on_calib(p, m) {
      if (!this.map || (this.phase !== 'position' && this.phase !== 'play')) return;
      const a = this.anchors.find((q) => q.aid === str(m.corner, 4));
      if (!a) return;
      p.x = a.x; p.z = a.z; p.calib = true; p.posQ = 'corner';
      this.poseDirty = true;
      this.broadcastState();
      this.log(`${p.id} 在 ${a.aid} 完成校正`);
    }
    on_ready(p, m) {
      if (this.phase !== 'position') return;
      p.ready = !!m.v && p.calib;
      this.broadcastState();
      this.checkAllReady(false);
    }
    on_forceStart(p) {
      if (this.phase !== 'position') return;
      if (![...this.players.values()].some((q) => q.connected && q.calib)) return;
      this.checkAllReady(true);
    }
    checkAllReady(force) {
      const list = [...this.players.values()].filter((q) => q.connected);
      if (!list.length) return;
      if (force || list.every((q) => q.ready)) {
        this.countdownAt = this.now() + 3000;
        this.broadcastState();
        this.log('3 秒後開始', 'phase');
      }
    }

    // ---------- 遊戲中 ----------
    on_pose(p, m) {
      const now = this.now();
      if (now - p.lastPose < this.cfg.POSE_MIN_MS) return;
      const x = num(m.x, 500), z = num(m.z, 500), h = num(m.h, 10);
      if (x === null || z === null || h === null) return;
      p.lastPose = now;
      p.x = x; p.z = z; p.h = wrapPi(h);
      p.sp = num(m.sp, 20) || 0;
      if (typeof m.q === 'string') p.posQ = str(m.q, 8);
      this.poseDirty = true;
      const last = p.trace[p.trace.length - 1];
      if (!last || Math.hypot(last[0] - x, last[1] - z) >= 0.25) {
        p.trace.push([r2(x), r2(z)]);
        if (p.trace.length > 800) p.trace.shift();
        if (this.viewers.size) this.toViewers({ t: 'tr', id: p.id, x: r2(x), z: r2(z) });
      }
    }
    on_locate(p, m) { // 站在點陣座標上按「定位」
      if (!this.map) return;
      const i = Math.round(num(m.i, 999)), j = Math.round(num(m.j, 999));
      if (!(i >= 0 && j >= 0 && i < this.map.cols && j < this.map.rows)) return;
      p.x = i; p.z = j; p.posQ = 'grid';
      const gid = gridLabel(i, j);
      this.visited[gid] = p.id;
      this.poseDirty = true;
      this.toAll({ t: 'located', id: p.id, gid, x: i, z: j });
    }
    on_anchor(p, m) { // 新增實體符號錨點
      if (!this.map || !p.calib) return;
      const x = num(m.x, 500), z = num(m.z, 500);
      if (x === null || z === null) return;
      if (this.anchors.some((a) => Math.hypot(a.x - x, a.z - z) < 1.2)) return;
      if (this.anchors.length >= 12) return;
      const a = { aid: 'S' + (this.anchors.length + 1), x: r2(x), z: r2(z), by: p.id };
      this.anchors.push(a);
      this.toAll({ t: 'anchor', a });
    }

    losBlocked(ax, az, bx, bz) {
      const d = Math.hypot(bx - ax, bz - az);
      if (d < 0.3) return false;
      const dx = (bx - ax) / d, dz = (bz - az) / d;
      for (const s of this.structures) {
        if (s.h < 1.5) continue;
        if (rayBox(ax, az, dx, dz, boxOf(s), 0) < d - 0.3) return true;
      }
      return false;
    }

    on_shot(p, m) {
      if (this.phase !== 'play' || !p.alive) return;
      const now = this.now();
      if (now - p.lastShot < this.cfg.SHOT_COOLDOWN_MS) return;
      p.lastShot = now;
      const h = num(m.h, 10);
      if (h !== null) p.h = wrapPi(h);
      const target = str(m.target, 1).toUpperCase();
      const tp = target ? [...this.players.values()].find((q) => q.id === target) : null;
      const shotInfo = { t: 'shot', from: p.id, x: r2(p.x), z: r2(p.z), h: r2(p.h), to: null, ok: false, why: m.why ? str(m.why, 24) : '' };
      if (!tp || tp === p || !tp.alive) { shotInfo.why = shotInfo.why || 'miss'; this.toAll(shotInfo); return; }
      const dx = tp.x - p.x, dz = tp.z - p.z, d = Math.hypot(dx, dz);
      const bearing = Math.atan2(dx, -dz);
      const err = Math.abs(wrapPi(bearing - p.h));
      const tol = this.cfg.SHOT_SERVER_TOL + Math.atan2(1.2, Math.max(0.5, d));
      if (d > this.cfg.SHOT_MAX_DIST) { shotInfo.why = 'too_far'; this.toAll(shotInfo); return; }
      if (err > tol) { shotInfo.why = 'angle'; this.toAll(shotInfo); return; }
      if (this.losBlocked(p.x, p.z, tp.x, tp.z)) { shotInfo.why = 'wall'; this.toAll(shotInfo); return; }
      tp.hp = Math.max(0, tp.hp - 1);
      tp.hitsTaken++;
      this.poseDirty = true;
      shotInfo.to = tp.id; shotInfo.ok = true; shotInfo.hp = tp.hp; shotInfo.d = r2(d);
      this.toAll(shotInfo);
      if (tp.hp <= 0) {
        tp.alive = false;
        p.kills++;
        this.log(`${tp.id} 被 ${p.id} 淘汰`, 'out');
        this.toAll({ t: 'out', id: tp.id, by: p.id });
        this.broadcastState();
        this.checkOver();
      }
    }

    checkOver() {
      if (this.phase !== 'play') return;
      const now = this.now();
      const inGame = [...this.players.values()].filter((q) => q.connected || now - q.lastSeen < 30000);
      if (inGame.length < 2) return; // 單人測試不結束
      const alive = inGame.filter((q) => q.alive);
      if (alive.length <= 1) {
        this.winner = alive[0] ? alive[0].id : null;
        this.setPhase('over', this.winner ? `${this.winner} 獲勝！` : '平手');
      }
    }
    on_restart(p, m) {
      if (this.phase !== 'over' && this.phase !== 'play') return;
      if (m.newMap) { this.resetGame(true); this.setPhase('lobby', '回到大廳'); return; }
      this.resetGame(false);
      for (const q of this.players.values()) q.ready = false;
      this.setPhase('position', '再來一局：請重新站上符號校正');
    }

    // 玩家回報偵測到的真實物件（桌椅、牆、轉角）
    on_obst(p, m) {
      if (!this.map || !Array.isArray(m.list)) return;
      const now = this.now();
      if (now - p.lastObst < 600) return;
      p.lastObst = now;
      const KINDS = ['block', 'cover', 'corner_in', 'corner_out', 'column', 'wall'];
      for (const it of m.list.slice(0, 24)) {
        if (!it || !KINDS.includes(it.k)) continue;
        if (it.k === 'wall') {
          const w = { x1: num(it.x1, 500), z1: num(it.z1, 500), x2: num(it.x2, 500), z2: num(it.z2, 500) };
          if (Object.values(w).some((v) => v === null)) continue;
          if (Math.hypot(w.x2 - w.x1, w.z2 - w.z1) < 0.8) continue;
          if (!this.inRect((w.x1 + w.x2) / 2, (w.z1 + w.z2) / 2, 2.5)) continue;
          this.upsertWall(w, p.id);
        } else {
          const x = num(it.x, 500), z = num(it.z, 500);
          if (x === null || z === null || !this.inRect(x, z, 1.5)) continue;
          const w = clamp(num(it.w, 10) || 0.5, 0.2, 4), d = clamp(num(it.d, 10) || w, 0.2, 4);
          this.upsertPoint({ k: it.k, x, z, w, d, label: str(it.label, 16) }, p.id);
        }
      }
    }
    inRect(x, z, margin) {
      if (!this.map) return false;
      return x >= -margin && z >= -margin && x <= this.map.W + margin && z <= this.map.D + margin;
    }
    upsertPoint(o, by) {
      const R = o.k === 'block' || o.k === 'cover' ? 0.9 : 0.6;
      let best = null, bd = R;
      for (const q of this.obstacles) {
        if (q.k === 'wall' || q.k !== o.k) continue;
        const d = Math.hypot(q.x - o.x, q.z - o.z);
        if (d < bd) { bd = d; best = q; }
      }
      if (best) {
        const a = 1 / Math.min(8, best.hits + 1);
        best.x = r2(best.x + (o.x - best.x) * a); best.z = r2(best.z + (o.z - best.z) * a);
        best.w = r2(best.w + (o.w - best.w) * a); best.d = r2(best.d + (o.d - best.d) * a);
        best.hits = Math.min(99, best.hits + 1); best.ts = this.now();
        this.obstDirty.push(best);
      } else {
        if (this.obstacles.length >= 300) return;
        const q = { oid: this.nextOid++, k: o.k, x: r2(o.x), z: r2(o.z), w: r2(o.w), d: r2(o.d), label: o.label, hits: 1, by, ts: this.now() };
        this.obstacles.push(q);
        this.obstDirty.push(q);
      }
    }
    upsertWall(w, by) {
      const ang = Math.atan2(w.z2 - w.z1, w.x2 - w.x1);
      for (const e of this.obstacles) {
        if (e.k !== 'wall') continue;
        const ea = Math.atan2(e.z2 - e.z1, e.x2 - e.x1);
        let da = Math.abs(wrapPi(ang - ea)); da = Math.min(da, Math.PI - da);
        if (da > 14 * Math.PI / 180) continue;
        const ux = Math.cos(ea), uz = Math.sin(ea), nx = -uz, nz = ux;
        const o1 = (w.x1 - e.x1) * nx + (w.z1 - e.z1) * nz, o2 = (w.x2 - e.x1) * nx + (w.z2 - e.z1) * nz;
        if (Math.abs(o1) > 0.5 || Math.abs(o2) > 0.5) continue;
        const el = Math.hypot(e.x2 - e.x1, e.z2 - e.z1);
        const t1 = (w.x1 - e.x1) * ux + (w.z1 - e.z1) * uz, t2 = (w.x2 - e.x1) * ux + (w.z2 - e.z1) * uz;
        if (Math.min(el, Math.max(t1, t2)) - Math.max(0, Math.min(t1, t2)) < -0.8) continue;
        const lo = Math.min(0, t1, t2), hi = Math.max(el, t1, t2);
        if (hi - lo > 16) continue;
        const off = ((o1 + o2) / 2) * 0.2;
        const bx = e.x1 + nx * off, bz = e.z1 + nz * off;
        e.x1 = r2(bx + ux * lo); e.z1 = r2(bz + uz * lo); e.x2 = r2(bx + ux * hi); e.z2 = r2(bz + uz * hi);
        e.hits = Math.min(99, e.hits + 1); e.ts = this.now();
        this.obstDirty.push(e);
        return;
      }
      if (this.obstacles.length >= 300) return;
      const q = { oid: this.nextOid++, k: 'wall', x1: r2(w.x1), z1: r2(w.z1), x2: r2(w.x2), z2: r2(w.z2), hits: 1, by, ts: this.now() };
      this.obstacles.push(q);
      this.obstDirty.push(q);
    }

    // ---------- 建築生成（伺服器決定，所有人看到同一份）----------
    // 規則：只在「空曠、視野好、地面空間多」的位置生長；狹小擁擠處不生成；
    //       不生成在真實物件上（例如牆上不再長牆）；與玩家保持距離；建築之間留通道。
    openness(cx, cz) {
      const N = 16, MAX = 6;
      let sum = 0, mn = MAX;
      for (let k = 0; k < N; k++) {
        const a = (k / N) * TAU, dx = Math.sin(a), dz = -Math.cos(a);
        let t = MAX;
        // 場地邊界
        const tx = dx > 1e-6 ? (this.map.W - cx) / dx : dx < -1e-6 ? -cx / dx : Infinity;
        const tz = dz > 1e-6 ? (this.map.D - cz) / dz : dz < -1e-6 ? -cz / dz : Infinity;
        t = Math.min(t, tx, tz);
        for (const o of this.obstacles) {
          if (o.k === 'wall') t = Math.min(t, raySeg(cx, cz, dx, dz, o));
          else if (o.k === 'block' || o.k === 'cover') t = Math.min(t, rayBox(cx, cz, dx, dz, boxOf(o), 0));
          else t = Math.min(t, rayCircle(cx, cz, dx, dz, o.x, o.z, 0.3));
        }
        for (const s of this.structures) t = Math.min(t, rayBox(cx, cz, dx, dz, boxOf(s), 0));
        t = Math.max(0, t);
        sum += t; mn = Math.min(mn, t);
      }
      let crowd = 0;
      for (const o of this.obstacles) {
        const ox = o.k === 'wall' ? (o.x1 + o.x2) / 2 : o.x, oz = o.k === 'wall' ? (o.z1 + o.z2) / 2 : o.z;
        if (Math.hypot(ox - cx, oz - cz) < 2.5) crowd++;
      }
      for (const s of this.structures) if (Math.hypot(s.x - cx, s.z - cz) < 2.5) crowd++;
      return { mean: sum / N, min: mn, crowd };
    }

    partsFor(kind, cx, cz, axis) {
      const R = axis ? Math.PI / 2 : 0;
      const ux = axis ? 0 : 1, uz = axis ? 1 : 0; // 沿軸方向
      const vx = axis ? 1 : 0, vz = axis ? 0 : 1; // 垂直方向
      const P = (x, z, w, d, h, k, rot, gAx) => ({ x: r2(x), z: r2(z), w, d, h, kind: k, rot, gAx: gAx || 0 });
      if (kind === 'low') return [P(cx, cz, 1.2, 0.4, 1.1, 'low', R)];
      if (kind === 'pillar') return [P(cx, cz, 0.4, 0.4, 2.8, 'pillar', 0, 9), P(cx + ux * 1.2, cz + uz * 1.2, 1.2, 0.4, 1.1, 'low', R)];
      if (kind === 'wall') return [P(cx, cz, 2.4, 0.2, 2.8, 'wall', R)];
      if (kind === 'corner') return [
        P(cx + ux * 1.2, cz + uz * 1.2, 2.4, 0.2, 2.8, 'wall', R),
        P(cx + vx * 1.2, cz + vz * 1.2, 2.4, 0.2, 2.8, 'wall', axis ? 0 : Math.PI / 2),
      ];
      return [];
    }

    partsValid(parts) {
      const m = this.map;
      for (const s of parts) {
        const b = boxOf(s);
        if (b.x0 < 0.3 || b.z0 < 0.3 || b.x1 > m.W - 0.3 || b.z1 > m.D - 0.3) return false;
        for (const p of this.players.values()) {
          if (!p.calib) continue;
          const dx = Math.max(b.x0 - p.x, 0, p.x - b.x1), dz = Math.max(b.z0 - p.z, 0, p.z - b.z1);
          if (Math.hypot(dx, dz) < 1.5) return false;
        }
        for (const o of this.obstacles) {
          if (o.k === 'wall') { if (segBoxDist(o, b) < 0.5) return false; }  // 牆上不再生牆
          else if (o.k === 'block' || o.k === 'cover') { if (boxGap(boxOf(o), b) < 0.5) return false; }
          else { const dx = Math.max(b.x0 - o.x, 0, o.x - b.x1), dz = Math.max(b.z0 - o.z, 0, o.z - b.z1); if (Math.hypot(dx, dz) < 0.5) return false; }
        }
        for (const q of this.structures) if (boxGap(boxOf(q), b) < 0.9) return false; // 留 0.9 m 通道
        for (const a of this.anchors) { const dx = Math.max(b.x0 - a.x, 0, a.x - b.x1), dz = Math.max(b.z0 - a.z, 0, a.z - b.z1); if (Math.hypot(dx, dz) < 1.0) return false; }
      }
      return true;
    }

    generateOne() {
      if (!this.map) return false;
      const area = this.map.W * this.map.D;
      const groups = new Set(this.structures.map((s) => s.gid)).size;
      const budget = clamp(Math.floor(area / 10), 1, 14);
      if (groups >= budget) return false;
      const cands = [];
      for (let i = 1; i < this.map.W - 0.5; i += 1) {
        for (let j = 1; j < this.map.D - 0.5; j += 1) {
          const cx = i + 0.5 * (this.rand() - 0.5), cz = j + 0.5 * (this.rand() - 0.5);
          const op = this.openness(cx, cz);
          if (op.mean < 2.0 || op.min < 0.9 || op.crowd > 2) continue; // 狹小擁擠：不生成
          cands.push({ cx, cz, op, s: op.mean + 0.5 * op.min - 0.6 * op.crowd + this.rand() * 0.6 });
        }
      }
      cands.sort((a, b) => b.s - a.s);
      for (const c of cands.slice(0, 12)) {
        const pool = c.op.mean >= 3.6 ? ['corner', 'wall', 'pillar', 'low'] : c.op.mean >= 2.8 ? ['wall', 'pillar', 'low'] : ['low'];
        const kinds = pool.slice().sort(() => this.rand() - 0.5);
        for (const kind of kinds) {
          for (const axis of this.rand() < 0.5 ? [0, 1] : [1, 0]) {
            const parts = this.partsFor(kind, c.cx, c.cz, axis);
            if (!parts.length || !this.partsValid(parts)) continue;
            const gid = this.nextSid;
            const born = this.now();
            const add = parts.map((pt) => Object.assign({ sid: this.nextSid++, gid, born }, pt));
            this.structures.push(...add);
            this.toAll({ t: 'struct', add });
            const NAMES = { low: '1.1m 胸牆掩體', pillar: '40cm 柱＋胸牆', wall: '2.4m 實心牆', corner: 'L 型轉角牆' };
            this.log(`生成 ${NAMES[kind]} @ ${gridLabel(Math.round(c.cx), Math.round(c.cz))}（空曠度 ${c.op.mean.toFixed(1)} m）`, 'gen');
            return true;
          }
        }
      }
      return false;
    }

    // ---------- 週期 ----------
    tick() {
      const now = this.now();
      if (this.phase === 'position' && this.countdownAt && now >= this.countdownAt) {
        this.countdownAt = 0;
        this.playStart = now;
        this.genNext = now + this.cfg.GEN_FIRST_MS;
        for (const q of this.players.values()) { q.hp = this.cfg.HP; q.alive = true; }
        this.setPhase('play', '開始！');
      }
      if (this.phase === 'play' && now >= this.genNext) {
        const area = this.map ? this.map.W * this.map.D : 0;
        const first = now - this.playStart < 4000;
        let made = 0;
        const n = first ? Math.max(1, Math.ceil(clamp(Math.floor(area / 10), 1, 14) / 3)) : 1;
        for (let k = 0; k < n; k++) if (this.generateOne()) made++;
        this.genNext = now + (made ? this.cfg.GEN_INTERVAL_MS : 4000);
      }
      if (this.poseDirty && now - this.lastPoseBroadcast >= this.cfg.POSE_BROADCAST_MS) {
        this.poseDirty = false;
        this.lastPoseBroadcast = now;
        const ps = [];
        for (const p of this.players.values()) ps.push([p.id, r2(p.x), r2(p.z), r2(p.h), p.hp, p.alive ? 1 : 0, p.connected ? 1 : 0, p.posQ]);
        this.toAll({ t: 'poses', ps });
      }
      if (this.obstDirty.length && now - this.lastObstBroadcast > 500) {
        this.lastObstBroadcast = now;
        const uniq = [...new Set(this.obstDirty)];
        this.obstDirty = [];
        this.toAll({ t: 'obst', up: uniq });
      }
      // 清掉太久沒回來的斷線玩家（大廳以外保留一段時間）
      for (const [cid, p] of this.players) {
        if (!p.connected && now - p.lastSeen > this.cfg.REJOIN_MS) {
          this.players.delete(cid);
          this.log(`${p.id} 已移出房間`);
          this.broadcastState();
          this.checkOver();
        }
      }
    }

    isEmpty() { return ![...this.players.values()].some((p) => p.connected) && this.viewers.size === 0; }
  }

  return { Room, CFG, computeRect, gpsVector, gridLabel, wrapPi, LETTERS };
});
