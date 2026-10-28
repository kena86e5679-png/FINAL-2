// ==========================================================
// click clip!! — Node 伺服器
//   · 提供網頁檔案（index.html、dashboard.html、trainer.html …）
//   · WebSocket 房間（遊戲邏輯在 room.js，和手機主機模式共用）
//   · /api/vlm-analyze：訓練工作台的 Gemini 代理（Key 放在環境變數 GEMINI_API_KEY 最安全）
//
//   啟動：npm install → npm start（預設 http://localhost:3000）
//   iPhone 需要 https / wss：用 cloudflared tunnel --url http://localhost:3000，或部署到雲端主機
// ==========================================================
const http = require('http');
const fs = require('fs');
const path = require('path');
const { WebSocketServer } = require('ws');
const { Room } = require('./room.js');

const PORT = parseInt(process.env.PORT || '3000', 10);
const ROOT = __dirname;
const GEMINI_KEY = process.env.GEMINI_API_KEY || '';
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-3.6-flash';

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.ico': 'image/x-icon', '.md': 'text/plain; charset=utf-8', '.webmanifest': 'application/manifest+json',
};
const PUBLIC = new Set(['index.html', 'dashboard.html', 'trainer.html', 'trainer.js', 'room.js', 'net.js', 'vision.js',
  'engine.js', 'game.js', 'symbol.svg', 'symbol.png', 'symbol-print.html', 'README.md']);

function readBody(req, limit) {
  return new Promise((resolve, reject) => {
    let size = 0; const chunks = [];
    req.on('data', (c) => { size += c.length; if (size > limit) { reject(new Error('too large')); req.destroy(); } else chunks.push(c); });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

async function vlmProxy(req, res) {
  const json = (code, obj) => { res.writeHead(code, { 'content-type': 'application/json; charset=utf-8' }); res.end(JSON.stringify(obj)); };
  try {
    const body = JSON.parse(await readBody(req, 15 * 1024 * 1024));
    const key = GEMINI_KEY || body.apiKey;
    if (!key) return json(400, { error: { message: '伺服器沒有設定 GEMINI_API_KEY，請求裡也沒有 apiKey' } });
    const model = String(body.model || GEMINI_MODEL).replace(/[^a-z0-9.\-]/gi, '');
    const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key },
      body: JSON.stringify({
        contents: [{ role: 'user', parts: [{ text: String(body.prompt || '') }, { inline_data: { mime_type: 'image/jpeg', data: String(body.imageBase64 || '') } }] }],
        generationConfig: { responseMimeType: 'application/json', temperature: 0.2 },
      }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) return json(r.status, { error: { message: (data.error && data.error.message) || ('Gemini HTTP ' + r.status) } });
    const parts = (data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts) || [];
    const text = parts.filter((p) => p.text && !p.thought).map((p) => p.text).join('').replace(/```json|```/g, '').trim();
    const a = text.indexOf('{'), b = text.lastIndexOf('}');
    return json(200, JSON.parse(a >= 0 ? text.slice(a, b + 1) : text));
  } catch (e) {
    return json(500, { error: { message: e.message } });
  }
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  if (url.pathname === '/api/vlm-analyze' && req.method === 'POST') return vlmProxy(req, res);
  if (url.pathname === '/api/vlm-health') { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify({ ok: true, hasKey: !!GEMINI_KEY })); return; }
  if (url.pathname === '/api/health') { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify({ ok: true, rooms: rooms.size })); return; }
  let name = decodeURIComponent(url.pathname.replace(/^\/+/, '')) || 'index.html';
  if (name === 'dashboard') name = 'dashboard.html';
  if (!PUBLIC.has(name)) { res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' }); res.end('找不到 ' + name); return; }
  fs.readFile(path.join(ROOT, name), (err, buf) => {
    if (err) { res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' }); res.end('找不到 ' + name); return; }
    res.writeHead(200, { 'content-type': MIME[path.extname(name)] || 'application/octet-stream', 'cache-control': 'no-cache' });
    res.end(buf);
  });
});

// ---------- WebSocket 房間 ----------
const wss = new WebSocketServer({ server, maxPayload: 96 * 1024 });
const rooms = new Map();   // code -> { room, sockets: Map(cid -> ws), empty: ts }
let nextCid = 1;

function getRoom(code) {
  let r = rooms.get(code);
  if (!r) {
    const sockets = new Map();
    const room = new Room(code, {
      send: (cid, obj) => { const ws = sockets.get(cid); if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); },
    });
    r = { room, sockets, empty: 0 };
    rooms.set(code, r);
  }
  return r;
}

wss.on('connection', (ws) => {
  const cid = 'c' + (nextCid++);
  let entry = null;
  ws.isAlive = true;
  ws.on('pong', () => { ws.isAlive = true; });
  ws.on('message', (raw, isBinary) => {
    if (isBinary) return;
    let m; try { m = JSON.parse(raw.toString()); } catch (e) { return; }
    if (!m || typeof m.t !== 'string') return;
    if (!entry) {
      if (m.t !== 'hello') return;
      const code = String(m.room || '').trim().toUpperCase().replace(/[^A-Z0-9_-]/g, '').slice(0, 16) || 'CLIP';
      entry = getRoom(code);
      entry.sockets.set(cid, ws);
      entry.empty = 0;
    }
    entry.room.message(cid, m);
  });
  ws.on('close', () => {
    if (!entry) return;
    entry.sockets.delete(cid);
    entry.room.leave(cid);
  });
});

setInterval(() => {
  const now = Date.now();
  for (const [code, r] of rooms) {
    r.room.tick();
    if (r.room.isEmpty() && !r.sockets.size) {
      if (!r.empty) r.empty = now;
      else if (now - r.empty > 10 * 60 * 1000) rooms.delete(code);
    } else r.empty = 0;
  }
}, 100);

setInterval(() => {
  for (const ws of wss.clients) {
    if (!ws.isAlive) { ws.terminate(); continue; }
    ws.isAlive = false;
    try { ws.ping(); } catch (e) { /* ignore */ }
  }
}, 15000);

server.listen(PORT, () => {
  console.log('==================================================');
  console.log(`click clip!! 伺服器：http://localhost:${PORT}`);
  console.log(`地圖監控端：     http://localhost:${PORT}/dashboard.html`);
  console.log(`訓練工作台：     http://localhost:${PORT}/trainer.html`);
  console.log(GEMINI_KEY ? 'GEMINI_API_KEY 已設定' : '（未設定 GEMINI_API_KEY，訓練工作台需自行輸入 Key）');
  console.log('==================================================');
});
