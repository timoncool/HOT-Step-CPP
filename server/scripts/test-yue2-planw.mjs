#!/usr/bin/env node
// Feature C1 test: plan-strength curve w(step) on the YuE2 NAR plan conditioning
// (yue2:plan_w_start / plan_w_end / plan_w_curve, and the Lua plan_strength
// override via md_plan_loose_yue2), plus a CADS low-strength sweep.
//
// node server/scripts/test-yue2-planw.mjs --phase pre|post [--seed N] [--frames N]
//                                         [--base URL] [--timeout S]
import fs from 'fs';
import path from 'path';

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const PHASE      = arg('--phase', 'post');
const BASE       = arg('--base', 'http://127.0.0.1:8085');
const SEED       = Number(arg('--seed', '1234'));
const FRAMES     = Number(arg('--frames', '500'));   // 20 s at 25 fps
const CHUNK      = 250;
const TIMEOUT_MS = Number(arg('--timeout', '400')) * 1000;

const REPO = path.resolve(import.meta.dirname, '../..');
const OUT_DIR = path.join(REPO, 'temp', 'planw-test');
fs.mkdirSync(OUT_DIR, { recursive: true });
const ABC_TEXT = fs.readFileSync(
  path.join(REPO, 'server/data/audio/0005ea93-ca4f-4c6f-8b15-784040165f22.score.abc'), 'utf8');

// ── engine log tail ─────────────────────────────────────────────────────────
const LOGS_DIR = path.join(REPO, 'logs');
function engineLog() {
  const dirs = fs.readdirSync(LOGS_DIR).filter(d => /^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$/.test(d)).sort();
  return dirs.length ? path.join(LOGS_DIR, dirs[dirs.length - 1], 'ace_engine.log') : null;
}
const logSize = () => { const f = engineLog(); return f && fs.existsSync(f) ? fs.statSync(f).size : 0; };
function logSince(off) {
  const f = engineLog(); if (!f || !fs.existsSync(f)) return '';
  const size = fs.statSync(f).size; if (size <= off) return '';
  const fd = fs.openSync(f, 'r'); const b = Buffer.alloc(size - off);
  fs.readSync(fd, b, 0, b.length, off); fs.closeSync(fd); return b.toString('utf8');
}

const tfetch = (url, opt = {}) => fetch(url, { ...opt, signal: AbortSignal.timeout(30000) });

function decodeWav(buf, ct) {
  const off = buf.indexOf('RIFF');
  if (off < 0) throw new Error(`no RIFF in result (content-type ${ct}, ${buf.length} bytes)`);
  let p = off + 12, fmt = null;
  while (p + 8 <= buf.length) {
    const id = buf.toString('ascii', p, p + 4), sz = buf.readUInt32LE(p + 4);
    if (id === 'fmt ') fmt = { tag: buf.readUInt16LE(p + 8), bits: buf.readUInt16LE(p + 22) };
    if (id === 'data') {
      const d = buf.subarray(p + 8, p + 8 + sz);
      if (fmt.tag === 3 && fmt.bits === 32) return new Float32Array(d.buffer.slice(d.byteOffset, d.byteOffset + d.length));
      if (fmt.bits === 16) { const o = new Float32Array(d.length >> 1); for (let i = 0; i < o.length; i++) o[i] = d.readInt16LE(i * 2) / 32768; return o; }
      throw new Error(`unsupported wav fmt ${JSON.stringify(fmt)}`);
    }
    p += 8 + sz + (sz & 1);
  }
  throw new Error('no data chunk');
}

async function render(extra, label) {
  const body = {
    style: 'pop rock, female vocals, energetic', seed: SEED,
    lyrics: '[verse]\nWalking down the empty street tonight\nNeon signs are flickering bright\n[chorus]\nWe are alive, we are alive\nHold on tight and feel the drive\n',
    abc: ABC_TEXT, cot: 'melody',
    nar_chunk_frames: CHUNK, ode_steps: 16, preview_max_frames: FRAMES,
    ...extra,
  };
  const t0 = Date.now();
  const r = await tfetch(`${BASE}/yue2/synth`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`synth ${r.status}: ${await r.text()}`);
  const sub = await r.json(); const id = sub.job_id ?? sub.id;
  for (;;) {
    if (Date.now() - t0 > TIMEOUT_MS) { await tfetch(`${BASE}/job?id=${id}&cancel=1`, { method: 'POST' }).catch(() => {}); throw new Error(`timeout: ${label}`); }
    await new Promise(res => setTimeout(res, 1000));
    const s = await (await tfetch(`${BASE}/job?id=${id}`)).json();
    const st = String(s.status ?? s.state ?? '').toLowerCase();
    if (['done', 'complete', 'completed', 'succeeded'].includes(st)) break;
    if (st === 'failed' || st === 'error' || st.startsWith('cancel')) throw new Error(`job ${st}: ${JSON.stringify(s)}`);
  }
  const res = await tfetch(`${BASE}/job?id=${id}&result=1`);
  const audio = decodeWav(Buffer.from(await res.arrayBuffer()), res.headers.get('content-type'));
  console.error(`render ${label}: ${((Date.now() - t0) / 1000).toFixed(1)}s, ${audio.length} samples`);
  return audio;
}

function relDb(a, b) {
  const n = Math.min(a.length, b.length); let num = 0, den = 0;
  for (let i = 0; i < n; i++) { const d = a[i] - b[i]; num += d * d; den += a[i] * a[i]; }
  if (a.length !== b.length) console.log(`  (length mismatch ${a.length} vs ${b.length})`);
  return num === 0 ? -Infinity : 20 * Math.log10(Math.sqrt(num) / Math.sqrt(den));
}
const f = d => (d === -Infinity ? '-inf' : d.toFixed(2));
const saveF32 = (name, a) => fs.writeFileSync(path.join(OUT_DIR, name), Buffer.from(a.buffer));
const loadF32 = name => { const b = fs.readFileSync(path.join(OUT_DIR, name)); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); };


const pw = (s, extra = {}) => ({ plugin_params: { 'yue2:plan_w_start': s, 'yue2:plan_w_end': 1, ...extra } });
const cads = s => ({ plugin_params: { 'yue2:cads_strength': s } });

if (PHASE === 'pre') {
  saveF32('baseline_pre.f32', await render({}, 'baseline (pre-change build)'));
  console.log('pre phase saved to', OUT_DIR);
} else {
  const rows = []; let pass = true;
  const pre = loadF32('baseline_pre.f32');
  const w1 = await render(pw(1), 'w=1');
  const a = relDb(pre, w1); const aOk = a < -90; pass &&= aOk;
  rows.push({ check: '(a) w=1 vs pre-change build', rel_db: f(a), ok: aOk });
  let prev = -Infinity, mono = true; const sw = {};
  for (const s of [0.9, 0.7, 0.5, 0.3]) {
    sw[s] = await render(pw(s), `w_start ${s}`);
    const d = relDb(w1, sw[s]); if (!(d > prev)) mono = false; prev = d;
    const okMag = d > -60; pass &&= okMag;
    rows.push({ check: `(b) w_start=${s} vs w=1`, rel_db: f(d), ok: okMag });
  }
  rows.push({ check: '(b) monotonic growth', rel_db: '', ok: mono });
  const loose = await render({ scheduler: 'md_plan_loose_yue2', plugin_params: { 'md_plan_loose_yue2:loose': 0.5 } }, 'plan_loose 0.5');
  const c = relDb(w1, loose); const cOk = c > -60; pass &&= cOk;
  rows.push({ check: '(c) md_plan_loose_yue2 loose=0.5 vs stock', rel_db: f(c), ok: cOk });
  const again = await render(pw(0.5), 'w_start 0.5 repeat');
  const d = relDb(sw[0.5], again); const dOk = d < -90; pass &&= dOk;
  rows.push({ check: '(d) w_start=0.5 same seed twice', rel_db: f(d), ok: dOk });
  for (const s of [0.005, 0.01, 0.02, 0.05]) {
    rows.push({ check: `(cads) strength ${s} vs 0`, rel_db: f(relDb(w1, await render(cads(s), `cads ${s}`))), ok: '' });
  }
  console.table(rows);
  console.log(pass ? 'OVERALL PASS' : 'OVERALL FAIL');
  process.exit(pass ? 0 : 1);
}
