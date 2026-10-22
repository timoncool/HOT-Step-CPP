#!/usr/bin/env node
// daw-contract-test.mjs - contract test for the /api/v1/daw facade.
//
// Usage:
//   node server/scripts/daw-contract-test.mjs [--base http://127.0.0.1:3001] [--key KEY] [--quick]
//
// --quick runs only the checks that need no GPU work (status shape, 400/404 validation).
// Prints PASS/FAIL per check with the actual values; exits 1 if any check fails.

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 && args[i + 1] ? args[i + 1] : def; };
const BASE = opt('--base', 'http://127.0.0.1:3001').replace(/\/+$/, '');
const KEY = opt('--key', process.env.HOTSTEP_DAW_KEY || '');
const QUICK = args.includes('--quick');
const API = `${BASE}/api/v1/daw`;

const results = [];
function check(name, ok, detail) {
  results.push({ name, ok: !!ok });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
  return !!ok;
}

const headers = (extra = {}) => (KEY ? { 'X-DAW-Key': KEY, ...extra } : extra);
async function req(method, path, body) {
  const init = { method, headers: headers() };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) { init.body = JSON.stringify(body); init.headers = headers({ 'Content-Type': 'application/json' }); }
  const r = await fetch(`${API}${path}`, init);
  const buf = Buffer.from(await r.arrayBuffer());
  let json = null;
  try { json = JSON.parse(buf.toString('utf8')); } catch { /* binary */ }
  return { status: r.status, json, buf };
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── WAV helpers ───────────────────────────────────────────────────────────

/** 16-bit mono PCM click track: short 1 kHz blips on every beat. */
function clickWav(seconds = 4, bpm = 120, sr = 48000) {
  const n = seconds * sr;
  const data = Buffer.alloc(n * 2);
  const period = Math.round((60 / bpm) * sr), blip = Math.round(0.02 * sr);
  for (let i = 0; i < n; i++) {
    const p = i % period;
    const v = p < blip ? Math.sin((2 * Math.PI * 1000 * i) / sr) * (1 - p / blip) * 0.8 : 0;
    data.writeInt16LE(Math.round(v * 32767), i * 2);
  }
  const h = Buffer.alloc(44);
  h.write('RIFF', 0); h.writeUInt32LE(36 + data.length, 4); h.write('WAVE', 8);
  h.write('fmt ', 12); h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20); h.writeUInt16LE(1, 22);
  h.writeUInt32LE(sr, 24); h.writeUInt32LE(sr * 2, 28); h.writeUInt16LE(2, 32); h.writeUInt16LE(16, 34);
  h.write('data', 36); h.writeUInt32LE(data.length, 40);
  return Buffer.concat([h, data]);
}

/** Parse RIFF chunks; returns { riff, wave, format, sub, sr, ch, bits, seconds }. */
function parseWav(buf) {
  const out = { riff: buf.toString('ascii', 0, 4), wave: buf.toString('ascii', 8, 12) };
  let off = 12, dataLen = null;
  while (off + 8 <= buf.length) {
    const id = buf.toString('ascii', off, off + 4), len = buf.readUInt32LE(off + 4);
    if (id === 'fmt ') {
      out.format = buf.readUInt16LE(off + 8); out.ch = buf.readUInt16LE(off + 10);
      out.sr = buf.readUInt32LE(off + 12); out.bits = buf.readUInt16LE(off + 22);
      if (out.format === 0xfffe && len >= 40) out.sub = buf.readUInt16LE(off + 8 + 24);
    } else if (id === 'data') { dataLen = Math.min(len, buf.length - off - 8); break; }
    off += 8 + len + (len & 1);
  }
  if (dataLen !== null && out.sr && out.ch && out.bits) out.seconds = dataLen / (out.sr * out.ch * (out.bits / 8));
  return out;
}

async function pollJob(id, capS = 300, until = (s) => ['done', 'failed', 'cancelled', 'partial'].includes(s)) {
  const t0 = Date.now();
  let last = null;
  while ((Date.now() - t0) / 1000 < capS) {
    last = (await req('GET', `/jobs/${id}`)).json;
    if (last && until(last.state)) break;
    await sleep(1000);
  }
  return { job: last, elapsed: (Date.now() - t0) / 1000 };
}

function genForm(request, src) {
  const fd = new FormData();
  fd.append('request', JSON.stringify(request));
  if (src) fd.append('src_audio', new Blob([src], { type: 'audio/wav' }), 'src.wav');
  return fd;
}

// ── Checks ────────────────────────────────────────────────────────────────

async function main() {
  console.log(`DAW contract test against ${API}${QUICK ? ' (quick)' : ''}`);
  let status;
  try { status = await req('GET', '/status'); } catch (e) {
    check('status reachable', false, e.message); return finish();
  }
  const s = status.json || {};
  if (!status.json) console.log('note: /status did not return JSON; the server may predate the /api/v1/daw facade (restart it)');
  check('status http 200', status.status === 200, `http ${status.status}`);
  check('status api == 1', s.api === 1, `api=${JSON.stringify(s.api)}`);
  const want = { ace_server_alive: 'boolean', ready: 'boolean', non_turbo_available: 'boolean', base_available: 'boolean', dit_models: 'array' };
  for (const [k, t] of Object.entries(want)) {
    const got = Array.isArray(s[k]) ? 'array' : typeof s[k];
    check(`status.${k} is ${t}`, got === t, `got ${got}`);
  }
  for (const k of ['boot_status', 'version', 'loaded_model', 'capabilities']) check(`status has ${k}`, k in s, k in s ? '' : 'missing');

  const bad = await req('POST', '/generate', { mode: 'not_a_mode', caption: 'x' });
  check('bad mode -> 400', bad.status === 400 && bad.json?.api === 1 && !!bad.json?.error?.code, `http ${bad.status} code=${bad.json?.error?.code}`);
  const cm = await req('POST', '/generate', { mode: 'complete', caption: 'x' });
  check('complete -> 400 unsupported_mode', cm.status === 400 && cm.json?.error?.code === 'unsupported_mode', `http ${cm.status} code=${cm.json?.error?.code}`);
  const nj = await req('GET', '/jobs/00000000-0000-0000-0000-000000000000');
  check('unknown job -> 404', nj.status === 404 && nj.json?.error?.code === 'not_found', `http ${nj.status} code=${nj.json?.error?.code}`);
  const nu = await req('GET', '/understand/00000000-0000-0000-0000-000000000000');
  check('unknown analysis -> 404', nu.status === 404 && nu.json?.error?.code === 'not_found', `http ${nu.status} code=${nu.json?.error?.code}`);

  if (QUICK) return finish();
  if (s.ready !== true) { check('engine ready for full run', false, `ready=${s.ready} boot=${s.boot_status}`); return finish(); }

  // understand
  const fd = new FormData();
  fd.append('audio', new Blob([clickWav()], { type: 'audio/wav' }), 'click.wav');
  let u = await req('POST', '/understand', fd);
  let uj = u.json;
  const t0 = Date.now();
  while (uj?.state === 'running' && Date.now() - t0 < 120000) { await sleep(1000); uj = (await req('GET', `/understand/${uj.analysis_id}`)).json; }
  check('understand bpm within 2 of 120', uj?.state === 'done' && Math.abs(Number(uj.bpm) - 120) <= 2, `state=${uj?.state} bpm=${uj?.bpm}`);

  // text2music
  const dur = 10;
  const g = await req('POST', '/generate', genForm({ mode: 'text2music', caption: 'upbeat electronic drum groove, instrumental', duration_s: dur, seed: 12345, skip_lm: true }));
  check('text2music accepted 202', g.status === 202 && !!g.json?.id, `http ${g.status}`);
  let wav = null;
  if (g.json?.id) {
    const { job, elapsed } = await pollJob(g.json.id);
    check('text2music done within 300 s', job?.state === 'done', `state=${job?.state} t=${elapsed.toFixed(1)}s err=${job?.error ?? ''}`);
    if (job?.state === 'done') {
      const a = await req('GET', `/jobs/${g.json.id}/audio`);
      const w = parseWav(a.buf);
      wav = a.buf;
      check('audio RIFF/WAVE', w.riff === 'RIFF' && w.wave === 'WAVE', `${w.riff}/${w.wave}`);
      check('audio float format', w.format === 3 || (w.format === 0xfffe && w.sub === 3), `format=${w.format} sub=${w.sub ?? '-'} bits=${w.bits}`);
      check('audio 48000 Hz', w.sr === 48000, `sr=${w.sr}`);
      check('audio 2 ch', w.ch === 2, `ch=${w.ch}`);
      check(`audio length within 0.5 s of ${dur}`, Math.abs((w.seconds ?? 0) - dur) <= 0.5, `len=${w.seconds?.toFixed(3)}s`);
    }
  }

  // repaint
  if (wav) {
    const r = await req('POST', '/generate', genForm({ mode: 'repaint', caption: 'upbeat electronic drum groove, instrumental', region: { start_s: 3, end_s: 6 }, seed: 777, skip_lm: true }, wav));
    check('repaint accepted 202', r.status === 202 && !!r.json?.id, `http ${r.status} ${r.json?.error?.message ?? ''}`);
    if (r.json?.id) {
      const { job, elapsed } = await pollJob(r.json.id);
      check('repaint done within 300 s', job?.state === 'done', `state=${job?.state} t=${elapsed.toFixed(1)}s err=${job?.error ?? ''}`);
    }
  } else check('repaint (needs text2music audio)', false, 'skipped: no source audio');

  // lego (needs a pure base DiT model)
  if (!wav) check('lego (needs text2music audio)', false, 'skipped: no source audio');
  else if (s.base_available !== true) {
    console.log('SKIP  lego: base_available is false (install acestep-v15-base from Model Manager); not counted as PASS');
    const lb = await req('POST', '/generate', genForm({ mode: 'lego', caption: 'bass line', track: 'bass', seed: 99 }, wav));
    check('lego without base -> 422 base_model_required', lb.status === 422 && lb.json?.error?.code === 'base_model_required', `http ${lb.status} code=${lb.json?.error?.code}`);
  } else {
    const lg = await req('POST', '/generate', genForm({ mode: 'lego', caption: 'bass line', track: 'bass', seed: 99 }, wav));
    check('lego accepted 202', lg.status === 202 && !!lg.json?.id, `http ${lg.status} ${lg.json?.error?.message ?? ''}`);
    if (lg.json?.id) {
      const { job, elapsed } = await pollJob(lg.json.id);
      check('lego done within 300 s', job?.state === 'done', `state=${job?.state} t=${elapsed.toFixed(1)}s err=${job?.error ?? ''}`);
    }
    const ln = await req('POST', '/generate', genForm({ mode: 'lego', caption: 'bass', dit_model: 'acestep-v15-turbo-Q8_0.gguf' }, wav));
    check('lego with non-base dit_model -> 422', ln.status === 422 && ln.json?.error?.code === 'base_model_required', `http ${ln.status} code=${ln.json?.error?.code}`);
  }

  // model field: unknown name and turbo+lego are 400 (no render needed)
  const mu = await req('POST', '/generate', genForm({ mode: 'text2music', caption: 'x', model: 'no-such-model.gguf' }));
  check('unknown model -> 400 unknown_model', mu.status === 400 && mu.json?.error?.code === 'unknown_model', `http ${mu.status} code=${mu.json?.error?.code}`);
  const mt = await req('POST', '/generate', genForm({ mode: 'lego', caption: 'bass', model: 'turbo' }, wav));
  check('lego with model turbo -> 400', mt.status === 400 && mt.json?.error?.code === 'bad_request', `http ${mt.status} code=${mt.json?.error?.code}`);

  // cancel
  const c = await req('POST', '/generate', genForm({ mode: 'text2music', caption: 'slow ambient pad', duration_s: 60, seed: 4242, skip_lm: true }));
  if (c.json?.id) {
    await sleep(1000);
    const cr = await req('POST', `/jobs/${c.json.id}/cancel`);
    const { job, elapsed } = await pollJob(c.json.id, 10);
    check('cancel reaches cancelled within 10 s', job?.state === 'cancelled', `cancel http ${cr.status} state=${job?.state} t=${elapsed.toFixed(1)}s`);
  } else check('cancel job accepted', false, `http ${c.status}`);

  return finish();
}

function finish() {
  const failed = results.filter((r) => !r.ok).length;
  console.log(`\n${results.length - failed}/${results.length} passed${failed ? `, ${failed} FAILED` : ''}`);
  process.exit(failed ? 1 : 0);
}

main().catch((e) => { console.error(e); check('unexpected error', false, e.message); finish(); });
