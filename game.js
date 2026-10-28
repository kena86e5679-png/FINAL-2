// ==========================================================
// click clip!! — 遊戲流程
//   封面 → 進房（拿到 A～Z 編號）→ 拍其他玩家建檔 → 隨機兩人放符號界定範圍
//   → 每個人站上符號、準心對準另一個符號校正 → 10 條命對戰 → 結算
// ==========================================================
(function () {
  'use strict';
  const CC = window.CC, E = CC.E, V = CC.Vision;
  const $ = (id) => document.getElementById(id);
  const DEG = Math.PI / 180;
  const wrapPi = E.wrapPi;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const Q = new URLSearchParams(location.search);

  const G = (window.__G = {
    conn: null, me: null, room: '', phase: 'off', players: {}, profiles: {}, setup: null, map: null, anchors: [], winner: null,
    countdownAt: 0, tOff: 0, calib: false, ready: false, persons: [], personsAt: 0, sym: null,
    lastPoseSent: 0, lastObstSent: 0, lastObjDet: 0, lastDet: 0, lastSym: 0, lastSymFix: 0, lastFire: 0,
    fixType: 'none', walked: 0, lastPos: null, dead: false, killer: null,
    setupLocal: { symSeen: false, stood: false, gpsBusy: false, edgeSet: false },
    calibStep: 0, calibCorner: null, recal: false, furniture: [], frame: 0, started: false,
  });
  if (Q.has('debug')) E.S.debug = true;

  // ---------- 小工具 ----------
  function toast(msg, ok, ms) {
    const el = document.createElement('div');
    el.textContent = msg; if (ok) el.className = 'ok';
    $('toast').appendChild(el);
    setTimeout(() => el.remove(), ms || 3200);
    while ($('toast').children.length > 3) $('toast').firstChild.remove();
  }
  function feed(msg) {
    const el = document.createElement('div'); el.textContent = msg;
    $('feed').appendChild(el); setTimeout(() => el.remove(), 5000);
    while ($('feed').children.length > 5) $('feed').firstChild.remove();
  }
  function buzz(p) { try { navigator.vibrate && navigator.vibrate(p); } catch (e) {} }
  let ac = null;
  function sfx(kind) {
    try {
      ac = ac || new (window.AudioContext || window.webkitAudioContext)();
      if (ac.state === 'suspended') ac.resume();
      const o = ac.createOscillator(), g = ac.createGain(), t = ac.currentTime;
      if (kind === 'fire') { o.type = 'square'; o.frequency.setValueAtTime(880, t); o.frequency.exponentialRampToValueAtTime(220, t + 0.08); g.gain.setValueAtTime(0.08, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.1); }
      else if (kind === 'hit') { o.type = 'triangle'; o.frequency.setValueAtTime(1400, t); g.gain.setValueAtTime(0.12, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.15); }
      else if (kind === 'hurt') { o.type = 'sawtooth'; o.frequency.setValueAtTime(140, t); o.frequency.exponentialRampToValueAtTime(60, t + 0.25); g.gain.setValueAtTime(0.15, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.3); }
      else { o.type = 'sine'; o.frequency.setValueAtTime(660, t); g.gain.setValueAtTime(0.08, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.12); }
      o.connect(g); g.connect(ac.destination); o.start(t); o.stop(t + 0.35);
    } catch (e) {}
  }
  function loadScript(src) {
    return new Promise((res) => { const s = document.createElement('script'); s.src = src; s.async = false; s.onload = () => res(true); s.onerror = () => res(false); document.head.appendChild(s); });
  }
  let mlLoading = null;
  function loadML() { // 進房後才在背景載入 TF.js（封面不卡）
    if (mlLoading) return mlLoading;
    mlLoading = (async () => {
      if (!window.tf) await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js');
      if (!window.poseDetection) await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.1.3/dist/pose-detection.min.js');
      await V.load();
      if (V.status !== 'movenet' && !window.cocoSsd) { await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js'); await V.load(); }
      renderLobby();
      return V.status;
    })();
    return mlLoading;
  }
  let cocoScript = null;
  async function ensureCoco() {
    if (!window.cocoSsd) { cocoScript = cocoScript || loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js'); await cocoScript; }
    return V.loadObjects();
  }

  const SCREENS = ['scrCover', 'scrJoin', 'scrLobby', 'scrSetup', 'scrPos', 'hud', 'scrOver'];
  function show(id) { for (const s of SCREENS) $(s).classList.toggle('on', s === id); }
  const others = () => Object.values(G.players).filter((p) => G.me && p.id !== G.me.id);
  const meP = () => (G.me ? G.players[G.me.id] : null);
  const letterBtn = (p) => `<div class="l" style="background:${p.color}"><span>${p.id}</span></div>`;
  const serverNow = () => Date.now() + G.tOff;

  // ==========================================================
  // 封面 → 進房
  // ==========================================================
  $('btnCoverStart').addEventListener('click', () => {
    sfx('ui');
    try { $('inName').value = localStorage.getItem('clickclip.name') || ''; } catch (e) {}
    $('inRoom').value = (Q.get('room') || (() => { try { return localStorage.getItem('clickclip.room'); } catch (e) { return ''; } })() || 'CLIP1').toUpperCase();
    show('scrJoin');
  });

  $('btnJoin').addEventListener('click', async () => {
    const name = $('inName').value.trim().slice(0, 12);
    const room = CC.Net.sanitizeRoom($('inRoom').value);
    try { localStorage.setItem('clickclip.name', name); localStorage.setItem('clickclip.room', room); } catch (e) {}
    $('joinErr').textContent = '';
    $('btnJoin').disabled = true;
    const note = await E.requestSensors();        // iOS：一定要在點擊當下請求
    const camOk = await E.startCamera();
    if (note) toast(note);
    if (!camOk) toast('沒有取得相機（桌機可用拖曳轉向、WASD 移動、空白鍵開火測試）');
    try { if (navigator.wakeLock) await navigator.wakeLock.request('screen'); } catch (e) {}
    G.room = room; G.name = name;
    G.started = true;
    $('topbar').classList.remove('hidden');
    $('myName').textContent = (name || '玩家') + ' · 房間 ' + room;
    show('scrLobby');
    loadML();
    connect();
    $('btnJoin').disabled = false;
  });

  function connect() {
    if (G.conn) G.conn.close();
    G.conn = CC.Net.connect({
      room: G.room,
      onStatus: (s) => {
        $('netStatus').textContent = s.text;
        if (s.type === 'open') {
          let token = '';
          try { token = sessionStorage.getItem('clickclip.token.' + G.room) || ''; } catch (e) {}
          G.conn.send({ t: 'hello', role: 'player', name: G.name, token });
        }
        if (s.type === 'error') toast(s.text, false, 6000);
      },
      onMessage: onMsg,
    });
  }

  // ==========================================================
  // 訊息
  // ==========================================================
  function applyState(st) {
    if (st.now) G.tOff = st.now - Date.now();
    const prev = G.phase;
    G.players = {};
    for (const p of st.players) G.players[p.id] = p;
    G.profiles = st.profiles || {};
    G.setup = st.setup;
    G.countdownAt = st.countdownAt || 0;
    G.winner = st.winner;
    if (JSON.stringify(st.map) !== JSON.stringify(G.map)) { G.map = st.map; }
    G.anchors = st.anchors || [];
    E.setMap(G.map, G.anchors);
    E.setObstacles(st.obstacles || []);
    E.setStructures(st.structures || []);
    E.visited = new Set(st.visited || []);
    const me = meP();
    if (me) {
      if (G.calib && !me.calib && performance.now() - (G.calibSentAt || 0) > 4000) G.calib = false; // 伺服器重置（新地圖）
      G.ready = me.ready; G.dead = !me.alive && st.phase === 'play';
    }
    if (st.phase !== prev) onPhase(st.phase, prev);
    G.phase = st.phase;
    renderAll();
  }

  function onMsg(m) {
    switch (m.t) {
      case 'welcome':
        if (m.role === 'player') {
          G.me = m.you;
          try { sessionStorage.setItem('clickclip.token.' + G.room, m.you.token); } catch (e) {}
          $('myLetter').textContent = m.you.id;
          $('myBadge').style.background = m.you.color;
          toast(`你的編號是 ${m.you.id}`, true);
        }
        applyState(m.state);
        break;
      case 'state': applyState(m); break;
      case 'profile':
        (G.profiles[m.target] = G.profiles[m.target] || []).push({ by: m.by, feat: m.feat, thumb: m.thumb });
        if (m.target === (G.me && G.me.id)) toast(`${m.by} 幫你建檔了`, true);
        break;
      case 'need':
        if (m.what === 'photos') {
          toast(`還沒被拍到：${m.missing.join('、')}。再按一次會略過檢查直接開始。`, false, 5000);
          G.forceNext = true;
        }
        break;
      case 'poses':
        for (const r of m.ps) {
          const p = G.players[r[0]]; if (!p) continue;
          p.x = r[1]; p.z = r[2]; p.h = r[3]; p.hp = r[4]; p.alive = !!r[5]; p.connected = !!r[6]; p.posQ = r[7];
        }
        renderHudTop();
        break;
      case 'struct':
        E.addStructures(m.add);
        break;
      case 'obst': E.upsertObstacles(m.up); break;
      case 'anchor': G.anchors.push(m.a); E.setMap(G.map, G.anchors); toast(`新增符號錨點 ${m.a.aid}`, true); break;
      case 'located': E.visited.add(m.gid); if (m.id !== (G.me && G.me.id)) feed(`${m.id} 定位 ${m.gid}`); break;
      case 'autoCalib':
        E.hOff = m.offset; E.setPos(0, 0); G.calib = true; G.calibSentAt = performance.now(); G.fixType = 'corner'; G.walked = 0;
        toast('你在 S1 上量測，已自動完成校正', true);
        break;
      case 'shot': onShot(m); break;
      case 'out':
        feed(`${m.id} 被 ${m.by} 淘汰`);
        if (G.me && m.id === G.me.id) { G.dead = true; G.killer = m.by; buzz([80, 60, 200]); }
        renderHudTop();
        break;
      case 'log':
        if (m.kind === 'gen' && G.phase === 'play') feed(m.msg);
        else if (m.kind === 'phase' || m.kind === 'out') toast(m.msg, true);
        break;
      case 'error':
        toast(m.msg, false, 6000);
        if (m.fatal) { G.conn && G.conn.close(); G.started = false; $('topbar').classList.add('hidden'); $('joinErr').textContent = m.msg; show('scrJoin'); }
        break;
    }
  }

  const WHY = { angle: '沒對準', wall: '被牆擋住', no_person: '準心上沒有人', miss: '沒打中', too_far: '太遠', unknown_person: '認不出是誰', geo: '', 'cam+id': '', 'cam+geo': '', 'geo-only': '' };
  function onShot(m) {
    const meId = G.me && G.me.id;
    if (m.from === meId) {
      if (m.ok) { E.flash.markUntil = performance.now() + 380; sfx('hit'); buzz(20); feed(`命中 ${m.to}（剩 ${m.hp}）`); }
      else feed(WHY[m.why] != null && WHY[m.why] ? WHY[m.why] : '沒打中');
    } else if (m.to === meId && m.ok) {
      E.flash.hitUntil = performance.now() + 500; sfx('hurt'); buzz([40, 30, 40]);
      feed(`被 ${m.from} 打中（剩 ${m.hp}）`);
    }
    if (m.ok && G.players[m.to]) G.players[m.to].hp = m.hp;
    renderHudTop();
  }

  function onPhase(ph, prev) {
    if (ph === 'lobby') { G.calib = false; G.calibSentAt = 0; G.ready = false; G.calibCorner = null; show('scrLobby'); }
    else if (ph === 'setup') { G.setupLocal = { symSeen: false, stood: false, gpsBusy: false, edgeSet: false }; show('scrSetup'); }
    else if (ph === 'position') {
      G.ready = false; G.calibStep = G.calib ? 2 : 0; show('scrPos');
      E.resetWorld();
    }
    else if (ph === 'play') {
      G.dead = false; G.killer = null; G.recal = !G.calib; G.calibStep = G.calib ? 2 : 0;
      show(G.calib ? 'hud' : 'scrPos');
      E.resetScan();
      sfx('ui'); buzz(60);
    }
    else if (ph === 'over') show('scrOver');
  }

  // ==========================================================
  // 畫面
  // ==========================================================
  function renderAll() {
    $('myHp').classList.toggle('hidden', G.phase !== 'play');
    $('minimap').classList.toggle('hidden', !(G.phase === 'play' || G.phase === 'position'));
    $('alive').classList.toggle('hidden', G.phase !== 'play');
    if (G.phase === 'lobby') renderLobby();
    if (G.phase === 'setup') renderSetup();
    if (G.phase === 'position' || (G.phase === 'play' && !G.calib) || G.recal) renderPos();
    if (G.phase === 'play') renderHudTop();
    if (G.phase === 'over') renderOver();
  }

  // ---------- 大廳 ----------
  function renderLobby() {
    if (!G.me) return;
    const list = Object.values(G.players).sort((a, b) => a.id.localeCompare(b.id));
    $('lobbyCards').innerHTML = list.map((p) => {
      const ph = (G.profiles[p.id] || []).slice(-3).map((q) => (q.thumb ? `<img src="${q.thumb}" alt="">` : '')).join('');
      return `<div class="pcard ${p.id === G.me.id ? 'me' : ''} ${p.connected ? '' : 'off'}">${letterBtn(p)}
        <div class="n">${p.id === G.me.id ? '我' : escapeHtml(p.name)}</div><div class="ph">${ph}</div>
        <div class="cnt" style="color:${p.photos ? 'var(--cyan)' : '#ff9c7f'}">${p.photos ? p.photos + ' 張' : '未建檔'}</div></div>`;
    }).join('');
    const missing = list.filter((p) => !p.photos).length;
    const st = V.status === 'loading' ? '人物偵測模型載入中…' : V.status === 'movenet' ? '偵測到人時會出現骨架，按快門拍下' : V.status === 'coco' ? '（簡易模式）對準人按快門' : V.status === 'none' ? '人物模型載不到：拍照時請把人放在畫面中央' : '';
    $('lobbyHint').textContent = (list.length < 2 ? '等其他人進房… ' : missing ? `還有 ${missing} 人未建檔。` : '全部建檔完成！') + st;
  }
  function escapeHtml(s) { return String(s || '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]); }

  $('btnStartSetup').addEventListener('click', () => {
    if (!G.conn) return;
    G.conn.send({ t: 'startSetup', force: !!G.forceNext });
    G.forceNext = false;
  });

  // 拍照建檔
  let pendingShot = null;
  $('btnShutter').addEventListener('click', async () => {
    if (!G.me) return;
    sfx('ui'); E.flash.fireUntil = performance.now() + 120;
    const src = E.camCanvas();
    const shot = document.createElement('canvas'); shot.width = src.width; shot.height = src.height;
    shot.getContext('2d').drawImage(src, 0, 0);
    const img = shot.getContext('2d').getImageData(0, 0, shot.width, shot.height);
    let persons = [];
    if (V.status === 'movenet' || V.status === 'coco') persons = await V.detectPersons(shot);
    let person = null;
    if (persons.length) {
      const cx = shot.width / 2, cy = shot.height / 2;
      persons.sort((a, b) => (b.box.w * b.box.h) / (1 + Math.hypot(b.box.x + b.box.w / 2 - cx, b.box.y + b.box.h / 2 - cy) / 60) - (a.box.w * a.box.h) / (1 + Math.hypot(a.box.x + a.box.w / 2 - cx, a.box.y + a.box.h / 2 - cy) / 60));
      person = persons[0];
    } else if (V.status === 'movenet' || V.status === 'coco') {
      toast('沒有偵測到人，請把對方整個人放進畫面再拍');
      return;
    } else {
      person = { box: { x: shot.width * 0.3, y: shot.height * 0.12, w: shot.width * 0.4, h: shot.height * 0.76 }, kp: null, score: 0.5 };
    }
    // 身高（需要方向感測：腳在畫面中才算）
    const k = E.camScale, hf = V.headFoot(person);
    const geom = E.hasSensor() ? E.personGeom({ x: hf.head.x * k, y: hf.head.y * k }, hf.foot ? { x: hf.foot.x * k, y: hf.foot.y * k } : null) : null;
    const feat = V.features(img.data, shot.width, shot.height, person, { height: geom && geom.height });
    if (!feat) { toast('畫面太暗或人太小，請再拍一次'); return; }
    // 縮圖
    const b = person.box, th = document.createElement('canvas'); th.width = 72; th.height = 96;
    const tctx = th.getContext('2d'), s = Math.max(72 / b.w, 96 / b.h);
    tctx.fillStyle = '#222'; tctx.fillRect(0, 0, 72, 96);
    tctx.drawImage(shot, b.x + b.w / 2 - 36 / s, b.y + b.h / 2 - 48 / s, 72 / s, 96 / s, 0, 0, 72, 96);
    const thumb = th.toDataURL('image/jpeg', 0.6);
    // 先猜是誰
    const guess = V.identify(feat, G.profiles, others().map((p) => p.id))[0];
    pendingShot = { feat, thumb };
    $('whoImg').src = thumb;
    $('whoNote').textContent = (feat.src === 'kp' ? '已抓到骨架：頭髮、上衣、褲子顏色＋體型' : '用人物框擷取顏色') + (feat.hgt ? `，身高約 ${feat.hgt.toFixed(2)} m` : '') + (guess && guess.sim > 0.6 ? `。看起來像 ${guess.id}` : '');
    $('whoLetters').innerHTML = others().map((p) => `<button data-id="${p.id}" style="background:${p.color}">${p.id}</button>`).join('') || '<div class="note">房間裡還沒有其他玩家</div>';
    $('whoModal').classList.add('on');
  });
  $('whoLetters').addEventListener('click', (e) => {
    const id = e.target && e.target.getAttribute('data-id');
    if (!id || !pendingShot) return;
    G.conn.send({ t: 'profile', target: id, feat: pendingShot.feat, thumb: pendingShot.thumb });
    pendingShot = null;
    $('whoModal').classList.remove('on');
    toast(`已替 ${id} 建檔`, true);
  });
  $('whoCancel').addEventListener('click', () => { pendingShot = null; $('whoModal').classList.remove('on'); });

  // ---------- 範圍設定 ----------
  function role() {
    const s = G.setup, id = G.me && G.me.id;
    if (!s || !id) return 'other';
    if (s.a === id && s.b === id) return 'solo';
    if (s.a === id) return 'A';
    if (s.b === id) return 'B';
    return 'other';
  }
  function stepHtml(k, text, state) { return `<div class="step ${state}"><div class="k">${k}</div><div class="t">${text}</div></div>`; }
  function proposalHtml(s) {
    const p = s.proposal;
    if (!p) return '';
    const src = p.source === 'aim' ? '瞄準量測' : p.source === 'gps' ? 'GPS' : '手動輸入';
    return `<div class="row"><div class="bigval">${p.W.toFixed(1)} × ${p.D.toFixed(1)}<small>m</small></div>
      <div class="note">對角線 ${p.diag.toFixed(1)} m（${src}）${p.square ? '・未指定邊方向→正方形' : ''}<br>√(${p.W.toFixed(1)}² + ${p.D.toFixed(1)}²) = ${Math.hypot(p.W, p.D).toFixed(1)}</div></div>
      ${p.thin ? '<div class="pill warn">形狀很細長：確認邊方向是否正確</div>' : ''}${p.warn ? `<div class="pill warn">${p.warn}</div>` : ''}`;
  }
  function renderSetup() {
    const s = G.setup; if (!s) return;
    const r = role(), L = G.setupLocal;
    const nameOf = (id) => (G.players[id] ? `${id}` : id);
    let h = '';
    if (r === 'A' || r === 'solo') {
      h += `<h3>你負責 S1（對角線起點）</h3><div class="sub">${r === 'solo' ? '只有你一個人：S2 也由你放置，或直接手動輸入尺寸。' : `S2 由 ${nameOf(s.b)} 負責。`}</div><div class="steps">`;
      h += stepHtml(1, `把符號放在場地的一個角，鏡頭對準它 <span id="symState" class="pill ${L.symSeen ? 'ok' : ''}">${L.symSeen ? '已偵測到符號 ✓' : '尋找符號…'}</span>`, L.symSeen ? 'done' : 'now');
      h += stepHtml(2, '站到符號上 → 按「我站在 S1」（同時記錄 GPS）', s.standA ? 'done' : 'now');
      h += stepHtml(3, '（建議）面向場地的一條邊，例如沿著牆 → 按「設定邊方向」', s.edge ? 'done' : s.standA ? 'now' : '');
      h += stepHtml(4, `${r === 'solo' ? '把準心對準對角的 S2 符號' : `等 ${nameOf(s.b)} 站上 S2，把準心對準他的腳（或 S2 符號）`} → 按「測量對角線」 ${s.standB ? '<span class="pill ok">S2 已就位</span>' : r === 'solo' ? '' : '<span class="pill">S2 尚未就位</span>'}`, s.aim ? 'done' : s.standA ? 'now' : '');
      h += '</div>' + proposalHtml(s);
      h += `<div class="row">
        ${!s.standA ? '<button class="btn red" data-act="standA"><span>我站在 S1</span></button>' : ''}
        ${s.standA ? `<button class="btn ghost small" data-act="edge"><span>${s.edge ? '重設邊方向' : '設定邊方向'}</span></button>` : ''}
        ${s.standA ? '<button class="btn red" data-act="aim"><span>測量對角線</span></button>' : ''}
        <button class="btn ghost small" data-act="manual"><span>手動輸入</span></button>
        ${s.proposal ? '<button class="btn ok" data-act="confirm"><span>確認範圍</span></button>' : ''}
        ${s.standA ? '<button class="btn ghost small" data-act="redo"><span>重來</span></button>' : ''}
      </div>`;
    } else if (r === 'B') {
      h += `<h3>你負責 S2（對角線終點）</h3><div class="sub">S1 由 ${nameOf(s.a)} 負責。把符號放在他的對角。</div><div class="steps">`;
      h += stepHtml(1, `把符號放在 S1 的對角，鏡頭對準它 <span id="symState" class="pill ${L.symSeen ? 'ok' : ''}">${L.symSeen ? '已偵測到符號 ✓' : '尋找符號…'}</span>`, L.symSeen ? 'done' : 'now');
      h += stepHtml(2, '站到符號上 → 按「我站在 S2」（同時記錄 GPS）', s.standB ? 'done' : 'now');
      h += stepHtml(3, `站好不要動，等 ${nameOf(s.a)} 用準心對準你的腳量測`, s.aim ? 'done' : s.standB ? 'now' : '');
      h += '</div>' + proposalHtml(s);
      h += `<div class="row">${!s.standB ? '<button class="btn red" data-act="standB"><span>我站在 S2</span></button>' : ''}
        <button class="btn ghost small" data-act="manual"><span>手動輸入</span></button>
        ${s.proposal ? '<button class="btn ok" data-act="confirm"><span>確認範圍</span></button>' : ''}</div>`;
    } else {
      h += `<h3>等待範圍設定</h3><div class="sub">系統隨機選了 <b>${nameOf(s.a)}</b> 放 S1、<b>${nameOf(s.b)}</b> 放 S2。他們站上符號後，S1 的人會用準心量對角線，再用畢氏定理算出矩形場地。</div>
        <div class="row"><span class="pill ${s.standA ? 'ok' : ''}">S1 ${s.standA ? '就位' : '準備中'}</span><span class="pill ${s.standB ? 'ok' : ''}">S2 ${s.standB ? '就位' : '準備中'}</span>
        <span class="pill ${s.aim ? 'ok' : ''}">對角線 ${s.aim ? s.aim.toFixed(1) + ' m' : '未量'}</span></div>` + proposalHtml(s);
      h += '<div class="row"><button class="btn ghost small" data-act="reroll"><span>重新抽選</span></button></div>';
    }
    $('setupSheet').innerHTML = h;
  }
  $('setupSheet').addEventListener('click', async (e) => {
    const b = e.target.closest('[data-act]'); if (!b) return;
    const act = b.getAttribute('data-act');
    sfx('ui');
    if (act === 'standA' || act === 'standB') {
      const corner = act === 'standA' ? 'S1' : 'S2';
      G.conn.send({ t: 'setupStand', corner, abs: E.absHeading() });
      toast(`已記錄：你站在 ${corner}。正在取得 GPS…`, true);
      sampleGps(6000).then((g) => {
        if (g) { G.conn.send({ t: 'setupGps', corner, gps: g }); toast(`GPS ±${g.acc.toFixed(0)} m（${g.n} 筆）`, g.acc < 8); }
        else toast('沒有 GPS（或被拒絕）：請用瞄準量測');
      });
      if (role() === 'solo' && corner === 'S1') { /* 單人：之後自己量 S2 */ }
    } else if (act === 'edge') {
      G.conn.send({ t: 'setupEdge', raw: E.rawHeading(), abs: E.absHeading() });
      toast('邊方向已記錄', true);
    } else if (act === 'aim') {
      const a = aimPoint();
      if (!a.fp) { toast('準心要對準地面（腳或符號），再按一次'); return; }
      G.conn.send({ t: 'setupAim', d: a.fp.d, raw: E.rawRayHeading(a.px, a.py), abs: E.cmp.have ? wrapPi(E.rawRayHeading(a.px, a.py) + E.cmp.nOff) : null });
      toast(`對角線 ${a.fp.d.toFixed(2)} m${a.sym ? '（對準符號）' : ''}`, true);
    } else if (act === 'manual') { $('manualModal').classList.add('on'); }
    else if (act === 'confirm') G.conn.send({ t: 'setupConfirm' });
    else if (act === 'redo') G.conn.send({ t: 'setupRedo' });
    else if (act === 'reroll') G.conn.send({ t: 'setupReroll' });
  });
  $('manualCancel').addEventListener('click', () => $('manualModal').classList.remove('on'));
  $('manualOk').addEventListener('click', () => {
    const W = parseFloat($('inW').value), D = parseFloat($('inD').value);
    if (!(W >= 1 && D >= 1)) { toast('請輸入 1 公尺以上'); return; }
    G.conn.send({ t: 'setupManual', W, D });
    $('manualModal').classList.remove('on');
  });

  function sampleGps(ms) {
    return new Promise((res) => {
      if (!navigator.geolocation) return res(null);
      const pts = [];
      let id = null;
      const done = () => {
        if (id != null) navigator.geolocation.clearWatch(id);
        if (!pts.length) return res(null);
        let sw = 0, la = 0, lo = 0;
        for (const p of pts) { const w = 1 / Math.max(1, p.acc * p.acc); sw += w; la += p.lat * w; lo += p.lon * w; }
        const accs = pts.map((p) => p.acc).sort((a, b) => a - b);
        res({ lat: la / sw, lon: lo / sw, acc: accs[accs.length >> 1], n: pts.length });
      };
      try {
        id = navigator.geolocation.watchPosition((p) => pts.push({ lat: p.coords.latitude, lon: p.coords.longitude, acc: p.coords.accuracy }), () => {}, { enableHighAccuracy: true, maximumAge: 0, timeout: ms });
      } catch (e) { return res(null); }
      setTimeout(done, ms);
    });
  }

  // 準心瞄準的地面點（準心附近有符號就用符號中心，比較準）
  function aimPoint() {
    const [cx, cy] = E.center();
    let px = cx, py = cy, sym = false;
    if (G.sym && performance.now() - G.sym.t < 600 && Math.hypot(G.sym.sx - cx, G.sym.sy - cy) < Math.min(cx, cy) * 0.18) { px = G.sym.sx; py = G.sym.sy; sym = true; }
    return { px, py, sym, fp: E.floorPoint(px, py) };
  }

  // ---------- 站位校正 ----------
  function renderPos() {
    if (!G.map) return;
    const meId = G.me && G.me.id;
    const list = Object.values(G.players).sort((a, b) => a.id.localeCompare(b.id));
    let h = `<h3>${G.recal ? '重新校正' : '站上符號，校正位置與方向'}</h3>
      <div class="sub">場地 ${G.map.W.toFixed(1)} × ${G.map.D.toFixed(1)} m，S1 與 S2 在對角。站上其中一個，準心對準另一個，系統就知道你在哪、面向哪。</div><div class="steps">`;
    h += stepHtml(1, `走到 S1 或 S2 的符號上站好 ${G.calibCorner ? `<span class="pill ok">${G.calibCorner}</span>` : ''}`, G.calibStep > 0 ? 'done' : 'now');
    h += stepHtml(2, `把準心對準 ${G.calibCorner === 'S2' ? 'S1' : G.calibCorner === 'S1' ? 'S2' : '另一個'} 符號（或站在上面的人的腳）→ 按「對準」`, G.calibStep > 1 ? 'done' : G.calibStep === 1 ? 'now' : '');
    h += stepHtml(3, G.phase === 'play' ? '回到遊戲' : '按「準備好」，等所有人', G.ready ? 'done' : G.calibStep > 1 ? 'now' : '');
    h += '</div>';
    if (G.phase === 'position') {
      h += '<div class="row">' + list.map((p) => `<span class="pill ${p.ready ? 'ok' : ''}">${p.id}${p.id === meId ? '(我)' : ''} ${p.ready ? '準備好' : p.calib ? '已校正' : '未校正'}</span>`).join('') + '</div>';
    }
    h += '<div class="row">';
    if (G.calibStep === 0) h += '<button class="btn red" data-act="c1"><span>我在 S1</span></button><button class="btn red" data-act="c2"><span>我在 S2</span></button>';
    if (G.calibStep === 1) h += '<button class="btn red" data-act="aimc"><span>對準</span></button><button class="btn ghost small" data-act="back"><span>上一步</span></button>';
    if (G.calibStep === 2) {
      if (G.phase === 'position') h += `<button class="btn ${G.ready ? 'ghost' : 'ok'}" data-act="ready"><span>${G.ready ? '取消準備' : '準備好'}</span></button><button class="btn ghost small" data-act="again"><span>重新校正</span></button><button class="btn ghost small" data-act="anchor"><span>新增符號錨點</span></button><button class="btn ghost small" data-act="force"><span>全員開始</span></button>`;
      else h += '<button class="btn ok" data-act="toplay"><span>回到遊戲</span></button>';
    }
    h += '</div>';
    $('posSheet').innerHTML = h;
  }
  $('posSheet').addEventListener('click', (e) => {
    const b = e.target.closest('[data-act]'); if (!b) return;
    const act = b.getAttribute('data-act');
    sfx('ui');
    if (act === 'c1' || act === 'c2') { G.calibCorner = act === 'c1' ? 'S1' : 'S2'; G.calibStep = 1; }
    else if (act === 'back') { G.calibStep = 0; }
    else if (act === 'aimc') {
      const from = G.anchors.find((a) => a.aid === G.calibCorner), to = G.anchors.find((a) => a.aid === (G.calibCorner === 'S1' ? 'S2' : 'S1'));
      if (!from || !to) return;
      const a = aimPoint();
      const rawH = E.rawRayHeading(a.px, a.py);
      const targetH = Math.atan2(to.x - from.x, -(to.z - from.z));
      E.hOff = wrapPi(targetH - rawH);
      E.setPos(from.x, from.z);
      const diag = Math.hypot(to.x - from.x, to.z - from.z);
      let eyeNote = '';
      if (a.fp && a.fp.d > 1) {
        const ratio = diag / a.fp.d;
        if (ratio > 0.6 && ratio < 1.6) { E.EYE = clamp(E.EYE * ratio, 1.1, 1.9); eyeNote = `，眼高 ${E.EYE.toFixed(2)} m`; }
      }
      G.calib = true; G.calibSentAt = performance.now(); G.fixType = 'corner'; G.walked = 0; G.calibStep = 2;
      G.conn.send({ t: 'calib', corner: G.calibCorner });
      toast(`校正完成：在 ${G.calibCorner}${a.sym ? '（對準符號）' : ''}${eyeNote}`, true);
    }
    else if (act === 'ready') { G.ready = !G.ready; G.conn.send({ t: 'ready', v: G.ready }); }
    else if (act === 'again') { G.calibStep = 0; G.calibCorner = null; G.ready = false; G.conn.send({ t: 'ready', v: false }); }
    else if (act === 'force') G.conn.send({ t: 'forceStart' });
    else if (act === 'anchor') {
      if (!G.sym || performance.now() - G.sym.t > 800) { toast('準心附近沒有偵測到符號'); return; }
      const fp = E.floorPoint(G.sym.sx, G.sym.sy);
      if (!fp) { toast('符號要在地上'); return; }
      G.conn.send({ t: 'anchor', x: fp.x, z: fp.z });
    }
    else if (act === 'toplay') { G.recal = false; show('hud'); }
    renderPos();
  });

  // ---------- 遊戲中 ----------
  function renderHudTop() {
    const me = meP();
    if (!me) return;
    if (G.phase === 'play') {
      let h = '';
      for (let i = 0; i < 10; i++) h += `<i class="${i < me.hp ? '' : 'off'}"></i>`;
      $('myHp').innerHTML = h;
      const alive = Object.values(G.players).filter((p) => p.alive && p.connected).length;
      $('alive').textContent = `存活 ${alive}`;
      $('dead').classList.toggle('hidden', !G.dead);
      if (G.dead) { $('deadTitle').textContent = G.killer ? `被 ${G.killer} 淘汰` : '你被淘汰了'; }
    }
  }
  $('btnLocate').addEventListener('click', () => {
    const n = E.nearestGrid(); if (!n) return;
    if (n.d > 1.0) { toast(`離最近的座標 ${n.label} 還有 ${n.d.toFixed(1)} m，站到點上再按`); return; }
    E.setPos(n.i, n.j);
    G.fixType = 'grid'; G.walked = 0;
    G.conn.send({ t: 'locate', i: n.i, j: n.j });
    toast(`定位 ${n.label}`, true); sfx('ui');
  });
  $('btnRecal').addEventListener('click', () => { G.recal = true; G.calibStep = 0; G.calibCorner = null; show('scrPos'); renderPos(); });
  $('btnLimit').addEventListener('click', () => { E.S.limitOn = !E.S.limitOn; E.saveSettings(); toast(E.S.limitOn ? '視野限制：開' : '視野限制：關'); });
  $('btnSettings').addEventListener('click', () => $('setModal').classList.add('on'));
  $('setClose').addEventListener('click', () => $('setModal').classList.remove('on'));
  $('btnDebug').addEventListener('click', () => { E.S.debug = !E.S.debug; });
  $('btnLeave').addEventListener('click', () => { try { sessionStorage.removeItem('clickclip.token.' + G.room); } catch (e) {} location.reload(); });
  function bindRange(id, out, get, set, fmt) {
    const el = $(id), o = $(out);
    el.value = get(); o.textContent = fmt(get());
    el.addEventListener('input', () => { set(parseFloat(el.value)); o.textContent = fmt(get()); E.saveSettings(); });
  }
  bindRange('sFov', 'oFov', () => E.S.fovLong, (v) => { E.S.fovLong = v; }, (v) => v + '°');
  bindRange('sStep', 'oStep', () => E.S.stepLen, (v) => { E.S.stepLen = v; }, (v) => v.toFixed(2));
  bindRange('sSize', 'oSize', () => E.S.sizeMul, (v) => { E.S.sizeMul = v; }, (v) => v.toFixed(2));
  bindRange('sEye', 'oEye', () => E.EYE, (v) => { E.EYE = v; }, (v) => v.toFixed(2));

  $('btnAgain').addEventListener('click', () => G.conn.send({ t: 'restart' }));
  $('btnLobby').addEventListener('click', () => G.conn.send({ t: 'restart', newMap: true }));
  function renderOver() {
    $('overWinner').textContent = G.winner || '平手';
    $('overWinner').style.color = G.winner && G.players[G.winner] ? G.players[G.winner].color : '#fff';
    $('overTable').innerHTML = Object.values(G.players).sort((a, b) => b.kills - a.kills || b.hp - a.hp)
      .map((p) => `<tr><td><b style="color:${p.color}">${p.id}</b> ${escapeHtml(p.name)}</td><td>淘汰 ${p.kills}</td><td>剩 ${p.hp} 命</td></tr>`).join('');
  }

  // 開火：點螢幕（桌機：點擊或空白鍵）
  $('view').addEventListener('pointerup', () => { if (E.dragMoved() < 8) fire(); });
  window.addEventListener('keydown', (e) => { if (e.code === 'Space' || e.code === 'KeyF') { e.preventDefault(); fire(); } });

  function geoCandidates() {
    const pos = E.pos, h = E.mapHeading(), out = [];
    for (const q of others()) {
      if (!q.alive || !q.connected || !q.calib) continue;
      const dx = q.x - pos.x, dz = q.z - pos.z, d = Math.hypot(dx, dz);
      if (d < 0.3) continue;
      const err = Math.abs(wrapPi(Math.atan2(dx, -dz) - h));
      const sigma = q.posQ === 'est' || G.fixType === 'none' || G.walked > 3 ? 1.0 : 0.6;
      const tol = clamp(Math.atan2(0.45 + sigma, d), 4 * DEG, 30 * DEG);
      out.push({ id: q.id, d, err, tol, occl: E.occluded(q.x, q.z) });
    }
    return out.sort((a, b) => a.err / a.tol - b.err / b.tol);
  }
  function inBox(x, y, b, pad) { const px = b.w * (pad || 0), py = b.h * (pad || 0); return x >= b.x - px && x <= b.x + b.w + px && y >= b.y - py && y <= b.y + b.h + py; }

  function fire() {
    if (G.phase !== 'play' || G.dead || !G.calib || G.recal) return;
    const now = performance.now();
    if (now - G.lastFire < 450) return;
    G.lastFire = now;
    E.flash.fireUntil = now + 120; sfx('fire'); buzz(12);
    const [cx, cy] = E.center(), h = E.mapHeading();
    const cands = geoCandidates();
    const visionOn = (V.status === 'movenet' || V.status === 'coco') && now - G.personsAt < 800;
    let target = null, why = 'miss';
    if (visionOn) {
      const under = G.persons.filter((p) => !p.hidden && inBox(cx, cy, p.sb, 0.12)).sort((a, b) => a.geom.d - b.geom.d);
      const hiddenUnder = G.persons.some((p) => p.hidden && inBox(cx, cy, p.sb, 0.12));
      if (under.length) {
        const p = under[0];
        if (p.id) { target = p.id; why = 'cam+id'; }
        else {
          const ok = cands.filter((c) => c.err < 2.5 * c.tol && !c.occl);
          if (ok.length) { target = ok[0].id; why = 'cam+geo'; } else why = 'unknown_person';
        }
      } else if (hiddenUnder) why = 'wall';
      else {
        const ok = cands.filter((c) => c.err < 0.6 * c.tol && !c.occl);
        if (ok.length) { target = ok[0].id; why = 'geo'; } else why = 'no_person';
      }
    } else {
      const ok = cands.filter((c) => c.err < c.tol && !c.occl);
      if (ok.length) { target = ok[0].id; why = 'geo-only'; }
      else why = cands.some((c) => c.err < c.tol && c.occl) ? 'wall' : 'miss';
    }
    G.conn.send({ t: 'shot', target, h, why });
  }

  // ==========================================================
  // 偵測排程（一次只跑一個模型推論）
  // ==========================================================
  let busy = false;
  const FURN = { 'dining table': 'block', desk: 'block', refrigerator: 'block', tv: 'block', bed: 'block', couch: 'cover', chair: 'cover', bench: 'cover', 'potted plant': 'cover', toilet: 'cover', suitcase: 'cover' };
  async function detectTick(now) {
    if (busy || !E.cameraReady() || !(V.status === 'movenet' || V.status === 'coco')) return;
    const wantPersons = G.phase === 'lobby' || G.phase === 'play' || G.phase === 'position';
    const interval = G.phase === 'play' ? 200 : 300;
    const wantObj = G.map && (G.phase === 'play' || G.phase === 'position') && now - G.lastObjDet > 3000 && E.moveSpeed < 0.8;
    if (wantObj && now - G.lastDet > 120) {
      busy = true; G.lastObjDet = now;
      try {
        if (await ensureCoco()) {
          const preds = await V.detectObjects(E.camCanvas());
          const k = E.camScale, sz = E.size();
          for (const p of preds) {
            const kind = FURN[p.class]; if (!kind) continue;
            const bx = (p.bbox[0] + p.bbox[2] / 2) * k, by = (p.bbox[1] + p.bbox[3]) * k;
            const fp = E.floorPoint(bx, by); if (!fp || fp.d > 7) continue;
            const w = clamp((p.bbox[2] * k / sz.FPX) * fp.d, 0.3, 3);
            G.furniture.push({ k: kind, x: +fp.x.toFixed(2), z: +fp.z.toFixed(2), w: +w.toFixed(2), d: +clamp(w * 0.6, 0.3, 2).toFixed(2), label: p.class });
          }
        }
      } catch (e) {}
      busy = false;
      return;
    }
    if (!wantPersons || now - G.lastDet < interval) return;
    busy = true; G.lastDet = now;
    try {
      const cvs = E.camCanvas();
      const persons = await V.detectPersons(cvs);
      let img = null;
      if (persons.length && G.phase !== 'position') img = cvs.getContext('2d').getImageData(0, 0, cvs.width, cvs.height);
      processPersons(persons, img, cvs.width, cvs.height);
      G.personsAt = performance.now();
    } catch (e) {}
    busy = false;
  }

  function processPersons(persons, img, w, h) {
    const k = E.camScale;
    const out = [];
    for (const p of persons) {
      const sb = { x: p.box.x * k, y: p.box.y * k, w: p.box.w * k, h: p.box.h * k };
      const hf = V.headFoot(p);
      const geom = E.personGeom({ x: hf.head.x * k, y: hf.head.y * k }, hf.foot ? { x: hf.foot.x * k, y: hf.foot.y * k } : null);
      const feat = img ? V.features(img.data, w, h, p, { height: geom.height }) : null;
      out.push({ raw: p, sb, geom, feat, id: null, conf: 0, hidden: false });
    }
    // 對應到玩家編號：外觀相似度（拍照建檔）＋ 位置吻合（伺服器座標）
    if (G.phase === 'play' || G.phase === 'lobby') {
      const cands = others().filter((q) => G.phase === 'lobby' || (q.alive && q.connected));
      const pairs = [];
      for (let i = 0; i < out.length; i++) {
        for (const q of cands) {
          const hasProf = (G.profiles[q.id] || []).length > 0;
          const app = out[i].feat && hasProf ? V.identify(out[i].feat, G.profiles, [q.id])[0].sim : 0.5;
          let geo = 0.5;
          if (G.phase === 'play' && q.calib && G.calib) {
            const dd = Math.hypot(out[i].geom.x - q.x, out[i].geom.z - q.z), sg = out[i].geom.rough ? 2.2 : 1.4;
            geo = Math.exp(-(dd * dd) / (2 * sg * sg));
          }
          const score = G.phase === 'lobby' ? app : 0.55 * app + 0.45 * geo;
          pairs.push({ i, id: q.id, score, app, geo });
        }
      }
      pairs.sort((a, b) => b.score - a.score);
      const usedP = new Set(), usedQ = new Set();
      for (const pr of pairs) {
        if (usedP.has(pr.i) || usedQ.has(pr.id) || pr.score < (G.phase === 'lobby' ? 0.62 : 0.42)) continue;
        usedP.add(pr.i); usedQ.add(pr.id);
        out[pr.i].id = pr.id; out[pr.i].conf = pr.score;
      }
    }
    // 被虛擬建築擋住：不顯示（真實的牆會擋住鏡頭本身，人自然不會被偵測到）
    if (G.map && G.calib) for (const o of out) o.hidden = E.occluded(o.geom.x, o.geom.z, { real: false }) === 'virtual';
    G.persons = out;
  }

  // 符號：顯示、校正
  function symbolTick(now) {
    if (!E.cameraReady() || now - G.lastSym < 160) return;
    if (!(G.phase === 'setup' || G.phase === 'position' || G.phase === 'play')) return;
    G.lastSym = now;
    const c = E.camCanvas();
    const img = c.getContext('2d').getImageData(0, 0, c.width, c.height);
    const s = V.detectSymbol(img.data, c.width, c.height);
    if (!s) return;
    const k = E.camScale;
    G.sym = { sx: s.cx * k, sy: s.cy * k, box: { x: s.x0 * k, y: s.y0 * k, w: (s.x1 - s.x0 + 1) * k, h: (s.y1 - s.y0 + 1) * k }, conf: s.conf, t: now };
    if (G.phase === 'setup' && !G.setupLocal.symSeen) { G.setupLocal.symSeen = true; sfx('ui'); renderSetup(); }
    // 遊戲中：準心往符號移動 → 以實體符號重新定位
    if (G.phase === 'play' && G.calib && !G.recal && now - G.lastSymFix > 1200) {
      const [cx, cy] = E.center();
      if (Math.hypot(G.sym.sx - cx, G.sym.sy - cy) < Math.min(cx, cy) * 0.25) {
        const fp = E.floorPoint(G.sym.sx, G.sym.sy);
        if (fp && fp.d < 6) {
          let best = null, bd = 2.5;
          for (const a of G.anchors) { const d = Math.hypot(a.x - fp.x, a.z - fp.z); if (d < bd) { bd = d; best = a; } }
          if (best && bd > 0.15) {
            E.nudgePos((best.x - fp.x) * 0.75, (best.z - fp.z) * 0.75);
            G.lastSymFix = now; G.fixType = 'anchor'; G.walked = 0;
            toast(`以符號 ${best.aid} 校正 ${bd.toFixed(2)} m`, true, 1800);
          } else if (best) { G.lastSymFix = now; G.fixType = 'anchor'; G.walked = 0; }
        }
      }
    }
  }

  // ==========================================================
  // 疊加繪製
  // ==========================================================
  const SK = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 1], [0, 2], [1, 3], [2, 4]];
  function drawSkeleton(c, p) {
    if (!p.raw.kp) return;
    const k = E.camScale;
    c.save(); c.strokeStyle = 'rgba(110,255,225,0.9)'; c.lineWidth = 2.5; c.beginPath();
    for (const [a, b] of SK) { const A = p.raw.kp[a], Bk = p.raw.kp[b]; if (A.s < 0.3 || Bk.s < 0.3) continue; c.moveTo(A.x * k, A.y * k); c.lineTo(Bk.x * k, Bk.y * k); }
    c.stroke();
    c.fillStyle = '#fff';
    for (const q of p.raw.kp) if (q.s >= 0.3) { c.beginPath(); c.arc(q.x * k, q.y * k, 3, 0, 7); c.fill(); }
    c.restore();
  }
  function drawSymbolRing(c) {
    if (!G.sym || performance.now() - G.sym.t > 500) return;
    const b = G.sym.box;
    c.save(); c.strokeStyle = '#ff8a6a'; c.lineWidth = 2; c.setLineDash([5, 4]);
    c.strokeRect(b.x - 6, b.y - 6, b.w + 12, b.h + 12); c.setLineDash([]);
    c.fillStyle = '#ff8a6a'; c.font = '700 11px ui-monospace, Menlo, monospace'; c.fillText('符號', b.x - 6, b.y - 10);
    c.restore();
  }
  function overlayLobby(c) {
    if (performance.now() - G.personsAt > 900) return;
    for (const p of G.persons) {
      drawSkeleton(c, p);
      if (p.id) E.drawPersonBox(p.sb, p.id, '已建檔', null, 'unknown');
    }
  }
  function overlayPlay(c) {
    drawSymbolRing(c);
    const visionOn = (V.status === 'movenet' || V.status === 'coco') && performance.now() - G.personsAt < 900;
    const [cx, cy] = E.center();
    let lock = false;
    if (visionOn) {
      for (const p of G.persons) {
        if (p.hidden) continue; // 被虛擬建築擋住
        const q = p.id ? G.players[p.id] : null;
        const under = inBox(cx, cy, p.sb, 0.12);
        if (under) lock = true;
        if (q && !q.alive) continue;
        E.drawPersonBox(p.sb, p.id || '?', `${p.geom.d.toFixed(1)}m`, q ? q.hp : null, under ? 'lock' : p.id ? 'enemy' : 'unknown');
        if (E.S.debug) drawSkeleton(c, p);
      }
    } else {
      // 沒有人物模型：用伺服器座標投影（被虛擬或真實牆擋住就不畫）
      for (const q of others()) {
        if (!q.alive || !q.connected || !q.calib) continue;
        if (E.occluded(q.x, q.z)) continue;
        const b = E.projectPersonBox(q.x, q.z);
        if (!b) continue;
        const under = inBox(cx, cy, b, 0.1);
        if (under) lock = true;
        E.drawPersonBox(b, q.id, `${Math.hypot(q.x - E.pos.x, q.z - E.pos.z).toFixed(1)}m`, q.hp, under ? 'lock' : 'enemy');
      }
    }
    return lock;
  }

  function drawMinimap() {
    const mc = $('minimap'); if (!G.map || mc.classList.contains('hidden')) return;
    const c = mc.getContext('2d'), S = mc.width, m = G.map, pad = 14;
    const sc = Math.min((S - 2 * pad) / m.W, (S - 2 * pad) / m.D);
    const ox = (S - m.W * sc) / 2, oz = (S - m.D * sc) / 2;
    const P = (x, z) => [ox + x * sc, oz + z * sc];
    c.clearRect(0, 0, S, S);
    c.strokeStyle = 'rgba(110,255,225,0.6)'; c.lineWidth = 2; c.strokeRect(ox, oz, m.W * sc, m.D * sc);
    c.fillStyle = 'rgba(255,255,255,0.18)';
    for (let j = 0; j < m.rows; j++) for (let i = 0; i < m.cols; i++) { const p = P(i, j); c.fillRect(p[0] - 1, p[1] - 1, 2, 2); }
    c.fillStyle = 'rgba(110,255,225,0.6)';
    for (const g of E.visited) { const [r, col] = g.split('-').map(Number); const p = P(col - 1, r - 1); c.fillRect(p[0] - 2, p[1] - 2, 4, 4); }
    c.fillStyle = '#c8ccd6';
    for (const o of E.structures()) { const rot = Math.abs(Math.sin(o.rot || 0)) > 0.7; const w = (rot ? o.d : o.w) * sc, d = (rot ? o.w : o.d) * sc; const p = P(o.x, o.z); c.fillRect(p[0] - w / 2, p[1] - d / 2, Math.max(2, w), Math.max(2, d)); }
    c.strokeStyle = 'rgba(255,184,108,0.85)'; c.lineWidth = 2;
    for (const o of E.obstacles) { if (o.k === 'wall') { const a = P(o.x1, o.z1), b = P(o.x2, o.z2); c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke(); } }
    for (const a of G.anchors) { const p = P(a.x, a.z); c.fillStyle = '#e0552d'; c.fillRect(p[0] - 3, p[1] - 3, 6, 6); }
    const me = P(E.pos.x, E.pos.z), h = E.mapHeading();
    c.fillStyle = G.me ? G.me.color : '#fff';
    c.beginPath(); c.moveTo(me[0] + Math.sin(h) * 9, me[1] - Math.cos(h) * 9); c.lineTo(me[0] + Math.sin(h + 2.4) * 6, me[1] - Math.cos(h + 2.4) * 6); c.lineTo(me[0] + Math.sin(h - 2.4) * 6, me[1] - Math.cos(h - 2.4) * 6); c.closePath(); c.fill();
  }

  // ==========================================================
  // 主迴圈
  // ==========================================================
  let lastT = performance.now();
  E.init();
  E.onStep = (dir, why) => {
    if (E.S.debug) feed(`步：${{ fwd: '前進', back: '後退', left: '左移', right: '右移' }[dir]}（${why}）`);
  };

  function loop(t) {
    requestAnimationFrame(loop);
    const dt = Math.min(0.1, Math.max(0.001, (t - lastT) / 1000));
    lastT = t;
    if (!G.started) return;
    const now = performance.now();
    G.frame++;
    const ph = G.phase;
    const inCalib = ph === 'position' || G.recal || (ph === 'play' && !G.calib);
    const mode = ph === 'play' && !inCalib ? 'play' : ph === 'position' || inCalib ? 'position' : 'raw';
    const before = { x: E.pos.x, z: E.pos.z };
    E.frame(dt, now, {
      mode,
      crosshair: ph === 'lobby' ? null : ph === 'play' && !inCalib ? (G.dead ? 'dim' : G._lock ? 'lock' : 'normal') : 'normal',
      overlay: (c) => {
        if (ph === 'lobby') overlayLobby(c);
        else if (mode === 'play') G._lock = overlayPlay(c);
        else drawSymbolRing(c);
      },
    });
    G.walked += Math.hypot(E.pos.x - before.x, E.pos.z - before.z);

    detectTick(now);
    if (G.frame % 3 === 0) symbolTick(now);

    // 準心讀數
    const ri = $('reticleInfo');
    if (ph === 'setup' || inCalib) {
      const a = aimPoint();
      ri.classList.remove('hidden');
      ri.textContent = a.fp ? `${a.fp.d.toFixed(2)} m${a.sym ? '・符號' : ''}` : '對準地面';
    } else ri.classList.add('hidden');

    // 送出位置（8 Hz）
    if (G.calib && (ph === 'position' || ph === 'play') && now - G.lastPoseSent > 125) {
      G.lastPoseSent = now;
      const q = G.walked > 3 ? 'est' : G.fixType;
      G.conn.send({ t: 'pose', x: +E.pos.x.toFixed(2), z: +E.pos.z.toFixed(2), h: +E.mapHeading().toFixed(3), sp: +E.moveSpeed.toFixed(2), q });
    }
    // 回報真實物件（1.5 秒一次）
    if (G.calib && G.map && (ph === 'position' || ph === 'play') && now - G.lastObstSent > 1500) {
      G.lastObstSent = now;
      const list = E.obstacleReport().concat(G.furniture.splice(0, 12));
      if (list.length) G.conn.send({ t: 'obst', list });
    }
    // 點陣資訊
    if (mode === 'play' && G.frame % 6 === 0) {
      const n = E.nearestGrid();
      $('gridInfo').innerHTML = n ? `最近座標 <b>${n.label}</b>　${n.d.toFixed(1)} m${n.outside ? '（場外）' : ''}${G.fixType === 'anchor' ? '　符號校正 ✓' : ''}` : '';
      $('btnLocate').disabled = !n || n.d > 1.0;
    }
    if (G.frame % 2 === 0) drawMinimap();
    // 倒數
    const cd = $('countdown');
    if (G.countdownAt && ph === 'position') {
      const left = Math.ceil((G.countdownAt - serverNow()) / 1000);
      cd.classList.toggle('hidden', left <= 0); cd.textContent = left > 0 ? left : '';
    } else cd.classList.add('hidden');
    // 除錯
    const dbg = $('dbg');
    dbg.classList.toggle('hidden', !E.S.debug);
    if (E.S.debug && G.frame % 10 === 0) {
      const fl = E.flow() && E.flow().last, ls = E.lastStep;
      dbg.textContent = [
        `phase ${ph}  mode ${mode}  net ${G.conn ? G.conn.mode : '-'}`,
        `pos ${E.pos.x.toFixed(2)},${E.pos.z.toFixed(2)}  h ${(E.mapHeading() / DEG).toFixed(0)}°  hOff ${(E.hOff / DEG).toFixed(0)}°  eye ${E.EYE.toFixed(2)}`,
        `walked ${G.walked.toFixed(1)}  fix ${G.fixType}  steps ${E.steps}  speed ${E.moveSpeed.toFixed(2)}`,
        `flow ${fl ? `div ${fl.div.toFixed(3)} lat ${fl.lat.toFixed(3)} n ${fl.n}` : '-'}  last ${ls ? ls.dir + ' ' + ls.why : '-'}`,
        `vision ${V.status}  persons ${G.persons.length}  ids ${G.persons.map((p) => p.id || '?').join(',')}`,
        `walls ${E.world.walls.length} ents ${E.world.ents.length}  det ${E.det.ms.toFixed(1)}ms  obst ${E.obstacles.length}  structs ${E.structures().length}`,
        `compass ${E.cmp.have ? E.cmp.src : '無'}  sym ${G.sym && now - G.sym.t < 500 ? G.sym.conf.toFixed(2) : '-'}`,
      ].join('\n');
    }
  }
  requestAnimationFrame(loop);

  window.__cc = { G, E, V, fire, toast };
})();
