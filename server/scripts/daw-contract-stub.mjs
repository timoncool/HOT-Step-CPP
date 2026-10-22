#!/usr/bin/env node
// daw-contract-stub.mjs - teeth check for daw-contract-test.mjs.
// Serves a fake /api/v1/daw whose /status renames non_turbo_available to
// nonTurboAvailable. The contract test must FAIL against it.
//   node server/scripts/daw-contract-stub.mjs [--port 3911]
//   node server/scripts/daw-contract-test.mjs --base http://127.0.0.1:3911 --quick
// --ok serves a well-formed /status instead (used by the HOT-STEP LINK heartbeat test),
// plus a fake job flow for the HOT-STEP LINK job client:
//   POST /generate (JSON or multipart "request" + "src_audio") -> 202 queued.
//   Validates mode and takes (400 bad_request / unsupported_mode, like the facade).
//   GET /jobs/:id walks queued -> running -> done over 3 polls. Caption "hold"
//   stays running until cancelled; caption "partial" ends partial with half the takes.
//   GET /jobs/:id/audio?take=n serves a 0.25 s 48 kHz stereo float32 WAV.
//   POST /understand (multipart "audio") -> 202, then GET /understand/:id is done.

import http from 'http';

const i = process.argv.indexOf('--port');
const port = i >= 0 ? Number(process.argv[i + 1]) : 3911;
const ok = process.argv.includes('--ok');
const send = (res, code, body) => { res.writeHead(code, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(body)); };

const MODES = ['lego', 'repaint', 'text2music'];
const jobs = new Map();
const unders = new Map();
let nextId = 1;
const err = (res, code, c, message) => send(res, code, { api: 1, error: { code: c, message } });

function wav() {
  const frames = 12000, ch = 2, data = frames * ch * 4;
  const b = Buffer.alloc(44 + data);
  b.write('RIFF', 0); b.writeUInt32LE(36 + data, 4); b.write('WAVE', 8);
  b.write('fmt ', 12); b.writeUInt32LE(16, 16); b.writeUInt16LE(3, 20); b.writeUInt16LE(ch, 22);
  b.writeUInt32LE(48000, 24); b.writeUInt32LE(48000 * ch * 4, 28); b.writeUInt16LE(ch * 4, 32); b.writeUInt16LE(32, 34);
  b.write('data', 36); b.writeUInt32LE(data, 40);
  for (let i = 0; i < frames; i++) for (let c = 0; c < ch; c++) b.writeFloatLE(0.25 * Math.sin(2 * Math.PI * 440 * i / 48000), 44 + (i * ch + c) * 4);
  return b;
}

function readBody(req) {
  return new Promise((resolve) => { const parts = []; req.on('data', (d) => parts.push(d)); req.on('end', () => resolve(Buffer.concat(parts).toString('latin1'))); });
}

function jobView(j) {
  const n = j.state === 'done' ? j.takes : j.state === 'partial' ? Math.floor(j.takes / 2) : 0;
  const takes = Array.from({ length: n }, (_, i) => ({ n: i, seed: i, duration_s: 0.25, ready: true }));
  return { api: 1, id: j.id, state: j.state, stage: null, progress: j.state === 'done' ? 1 : j.polls / 4,
    take_count: j.takes, takes_done: n, takes, eta_s: ['queued', 'running'].includes(j.state) ? 3 - j.polls : 0,
    error: j.state === 'partial' ? 'stub: take 3 failed' : null };
}

async function handleOk(req, res, url) {
  const path = url.split('?')[0];
  if (req.method === 'POST' && path === '/api/v1/daw/generate') {
    const body = await readBody(req);
    let r;
    try {
      if ((req.headers['content-type'] || '').startsWith('multipart/')) {
        const m = /name="request"\r\n(?:[^\r\n]*\r\n)*?\r\n([\s\S]*?)\r\n--/.exec(body);
        if (!m) return err(res, 400, 'bad_request', 'multipart field "request" is required');
        if (!/name="src_audio"/.test(body)) return err(res, 400, 'bad_request', 'stub: multipart without src_audio');
        r = JSON.parse(m[1]);
        r.__src = true;
      } else r = JSON.parse(body);
    } catch { return err(res, 400, 'bad_request', 'request is not valid JSON'); }
    if (!MODES.includes(r.mode)) return err(res, 400, 'unsupported_mode', `mode must be one of ${MODES.join(', ')}`);
    const takes = r.takes === undefined ? 1 : Number(r.takes);
    if (!Number.isInteger(takes) || takes < 1 || takes > 8) return err(res, 400, 'bad_request', 'takes must be an integer 1..8');
    const id = `stub-${nextId++}`;
    jobs.set(id, { id, state: 'queued', polls: 0, takes, caption: String(r.caption || ''), req: r });
    return send(res, 202, { api: 1, id, state: 'queued', eta_s: 3 });
  }
  let m = /^\/api\/v1\/daw\/jobs\/([^/]+)(\/audio|\/cancel|\/request)?$/.exec(path);
  if (m) {
    const j = jobs.get(m[1]);
    if (!j) return err(res, 404, 'not_found', `unknown job ${m[1]}`);
    if (m[2] === '/cancel') { if (['queued', 'running'].includes(j.state)) j.state = 'cancelled'; return send(res, 200, { api: 1, id: j.id, state: j.state }); }
    if (m[2] === '/request') return send(res, 200, { api: 1, request: j.req });
    if (m[2] === '/audio') {
      const n = Number(new URL(url, 'http://x').searchParams.get('take') ?? 0);
      if (n >= jobView(j).takes.length) return err(res, 409, 'not_ready', `take ${n} not ready`);
      const b = wav();
      res.writeHead(200, { 'Content-Type': 'audio/wav', 'Content-Length': b.length });
      return res.end(b);
    }
    if (['queued', 'running'].includes(j.state)) {
      j.polls++;
      if (j.caption === 'hold') j.state = 'running';
      else if (j.polls >= 3) j.state = j.caption === 'partial' ? 'partial' : 'done';
      else if (j.polls >= 1) j.state = 'running';
    }
    return send(res, 200, jobView(j));
  }
  if (req.method === 'POST' && path === '/api/v1/daw/understand') {
    const body = await readBody(req);
    if (!/name="audio"/.test(body)) return err(res, 400, 'bad_request', 'multipart field "audio" is required');
    const id = `an-${nextId++}`;
    unders.set(id, 0);
    return send(res, 202, { api: 1, analysis_id: id, state: 'running' });
  }
  m = /^\/api\/v1\/daw\/understand\/([^/]+)$/.exec(path);
  if (m) {
    if (!unders.has(m[1])) return err(res, 404, 'not_found', 'unknown analysis');
    const n = unders.get(m[1]) + 1;
    unders.set(m[1], n);
    if (n < 2) return send(res, 200, { api: 1, id: m[1], state: 'running' });
    return send(res, 200, { api: 1, id: m[1], state: 'done', bpm: 128, key: 'A minor', timesig: '4/4', tags: ['stub'] });
  }
  return null;
}

http.createServer(async (req, res) => {
  const url = req.url || '';
  if (ok && url !== '/api/v1/daw/status' && (await handleOk(req, res, url)) !== null) return;
  if (res.writableEnded) return;
  if (url === '/api/v1/daw/status' && ok) {
    const e = { base_s: 5, per_take_s: 2, per_audio_s: 0.1, cold_load_s: 10, source: 'default' };
    return send(res, 200, {
      api: 1, ace_server_alive: true, ready: true, boot_status: 'stub', version: 'stub',
      loaded_model: 'stub-model', dit_models: [], non_turbo_available: true, base_available: true,
      capabilities: null, eta_by_backend: { ace_turbo: e, ace_base: e, sa3: e, yue2: e },
    });
  }
  if (url === '/api/v1/daw/status') {
    return send(res, 200, {
      api: 1, ace_server_alive: true, ready: false, boot_status: 'stub', version: 'stub',
      loaded_model: null, dit_models: [], nonTurboAvailable: false, capabilities: null,
    });
  }
  if (url === '/api/v1/daw/generate') return send(res, 400, { api: 1, error: { code: 'unsupported_mode', message: 'stub' } });
  return send(res, 404, { api: 1, error: { code: 'not_found', message: 'stub' } });
}).listen(port, '127.0.0.1', () => console.log(`stub on http://127.0.0.1:${port}`));
