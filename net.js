// ==========================================================
// click clip!! — 連線層
//   三種模式，自動選擇：
//     1. ws        ：網頁由 Node 伺服器提供（server.js），或網址加 ?server=wss://你的伺服器
//     2. peer-host ：GitHub Pages 沒有伺服器 → 第一位進房的裝置當「主機」，房間邏輯（room.js）跑在這支手機/電腦上
//     3. peer-client：其他人透過 PeerJS（WebRTC）連到主機
//   注意：peer 模式下主機的螢幕關掉或切到背景，整個房間會暫停；建議用一台電腦開 dashboard 當主機。
// ==========================================================
(function () {
  'use strict';
  const CC = (window.CC = window.CC || {});

  function sanitizeRoom(code) {
    return String(code || '').trim().toUpperCase().replace(/[^A-Z0-9_-]/g, '').slice(0, 16) || 'CLIP';
  }

  function connect(opts) {
    const room = sanitizeRoom(opts.room);
    const onMessage = opts.onMessage || (() => {});
    const onStatus = opts.onStatus || (() => {});
    const q = new URLSearchParams(location.search);
    const conn = { mode: 'connecting', room, isHost: false, open: false, send: () => {}, close: () => {} };

    const serverParam = q.get('server');
    const sameHostOk = location.protocol.startsWith('http') && !location.hostname.endsWith('github.io') && !q.has('p2p');

    if (serverParam) wsMode(serverParam);
    else if (sameHostOk) wsMode((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host, true);
    else peerMode();
    return conn;

    // ---------- WebSocket ----------
    function wsMode(url, fallbackToPeer) {
      let ws = null, everOpen = false, closedByUser = false, retry = 0;
      const open = () => {
        conn.mode = 'ws';
        onStatus({ type: 'connecting', mode: 'ws', text: '連線伺服器中…' });
        try { ws = new WebSocket(url); } catch (e) { if (fallbackToPeer && !everOpen) return peerMode(); return; }
        const timer = setTimeout(() => {
          if (!everOpen && ws.readyState !== 1) { try { ws.close(); } catch (e) {} if (fallbackToPeer) peerMode(); }
        }, 2500);
        ws.onopen = () => {
          clearTimeout(timer);
          everOpen = true; retry = 0;
          conn.open = true;
          conn.send = (obj) => { if (ws.readyState === 1) ws.send(JSON.stringify(Object.assign({ room }, obj))); };
          onStatus({ type: 'open', mode: 'ws', text: '已連上伺服器' });
        };
        ws.onmessage = (ev) => { let m; try { m = JSON.parse(ev.data); } catch (e) { return; } onMessage(m); };
        ws.onclose = () => {
          clearTimeout(timer);
          conn.open = false;
          if (closedByUser) return;
          if (!everOpen) { if (fallbackToPeer) peerMode(); else onStatus({ type: 'error', text: '連不到伺服器：' + url }); return; }
          retry++;
          onStatus({ type: 'closed', mode: 'ws', text: `連線中斷，${Math.min(5, retry)} 秒後重連…` });
          setTimeout(open, Math.min(5, retry) * 1000);
        };
        ws.onerror = () => {};
      };
      conn.close = () => { closedByUser = true; try { ws && ws.close(); } catch (e) {} };
      open();
    }

    // ---------- PeerJS ----------
    function peerMode() {
      if (!window.Peer) {
        onStatus({ type: 'error', text: 'PeerJS 載入失敗（網路被擋？）。請改用 Node 伺服器模式，或檢查網路後重新整理。' });
        return;
      }
      if (!window.ClickClipRoom) { onStatus({ type: 'error', text: 'room.js 沒有載入' }); return; }
      const hostId = 'clickclip-v6-' + room.toLowerCase() + '-host';
      conn.mode = 'peer';
      onStatus({ type: 'connecting', mode: 'peer', text: '尋找房間主機…' });
      const asHost = new window.Peer(hostId, { debug: 0 });
      let decided = false;
      asHost.on('open', () => { decided = true; becomeHost(asHost); });
      asHost.on('error', (err) => {
        if (decided) { onStatus({ type: 'warn', text: '主機連線問題：' + (err.type || err.message) }); return; }
        decided = true;
        try { asHost.destroy(); } catch (e) {}
        if (err.type === 'unavailable-id') becomeClient(hostId);
        else onStatus({ type: 'error', text: '無法連到 PeerJS 雲端（' + (err.type || err.message) + '）' });
      });
    }

    function becomeHost(peer) {
      conn.mode = 'peer-host';
      conn.isHost = true;
      const remotes = new Map(); // cid -> DataConnection
      let n = 0;
      const R = new window.ClickClipRoom.Room(room, {
        send: (cid, obj) => {
          if (cid === 'local') { setTimeout(() => onMessage(obj), 0); return; }
          const c = remotes.get(cid);
          if (c && c.open) { try { c.send(obj); } catch (e) {} }
        },
      });
      conn.roomObj = R;
      const tick = setInterval(() => R.tick(), 100);
      peer.on('connection', (c) => {
        const cid = 'p' + (++n);
        c.on('open', () => remotes.set(cid, c));
        c.on('data', (d) => { if (d && typeof d === 'object') R.message(cid, d); });
        c.on('close', () => { remotes.delete(cid); R.leave(cid); });
        c.on('error', () => {});
      });
      peer.on('disconnected', () => { try { peer.reconnect(); } catch (e) {} });
      conn.open = true;
      conn.send = (obj) => R.message('local', obj);
      conn.close = () => { clearInterval(tick); try { peer.destroy(); } catch (e) {} };
      onStatus({ type: 'open', mode: 'peer-host', text: '你是這個房間的主機（請保持這個畫面開著）' });
    }

    function becomeClient(hostId) {
      conn.mode = 'peer-client';
      const peer = new window.Peer({ debug: 0 });
      let c = null, retry = 0, closedByUser = false;
      const dial = () => {
        c = peer.connect(hostId, { reliable: true, serialization: 'json' });
        const timer = setTimeout(() => { if (!c.open) onStatus({ type: 'warn', text: '主機回應很慢…（同一個 Wi-Fi 最穩）' }); }, 6000);
        c.on('open', () => {
          clearTimeout(timer); retry = 0;
          conn.open = true;
          conn.send = (obj) => { if (c.open) c.send(obj); };
          onStatus({ type: 'open', mode: 'peer-client', text: '已連上房間主機' });
        });
        c.on('data', (d) => { if (d && typeof d === 'object') onMessage(d); });
        c.on('close', () => {
          conn.open = false;
          if (closedByUser) return;
          retry++;
          onStatus({ type: 'closed', mode: 'peer-client', text: '主機離線，重新連線中…' });
          if (retry > 4) {
            onStatus({ type: 'error', text: '主機已離開。請所有人重新進房（第一個進房的人會成為新主機）。' });
            return;
          }
          setTimeout(dial, 1500 * retry);
        });
        c.on('error', () => {});
      };
      peer.on('open', dial);
      peer.on('error', (err) => {
        if (err.type === 'peer-unavailable') {
          onStatus({ type: 'closed', text: '找不到主機，改由你當主機…' });
          try { peer.destroy(); } catch (e) {}
          setTimeout(peerMode, 800);
        } else onStatus({ type: 'warn', text: 'PeerJS：' + (err.type || err.message) });
      });
      conn.close = () => { closedByUser = true; try { peer.destroy(); } catch (e) {} };
    }
  }

  CC.Net = { connect, sanitizeRoom };
})();
