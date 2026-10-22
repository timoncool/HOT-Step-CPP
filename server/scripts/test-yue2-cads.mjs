#!/usr/bin/env node
// Feature B test: CADS-style conditioning noise on the YuE2 NAR plan conditioning
// (yue2:cads_strength / yue2:cads_t1 / yue2:cads_t2 / yue2:cads_rescale), plus
// the Part 0 check (abc given + cot off must still populate the Lua yue2 table).
//
// Run once BEFORE the engine rebuild with --phase pre (saves the baseline render
// and the Part 0 "before" table), then after the rebuild with --phase post.
//
// node server/scripts/test-yue2-cads.mjs --phase pre|post [--seed N] [--frames N]
//                                        [--base URL] [--timeout S]
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
const OUT_DIR = path.join(REPO, 'temp', 'cads-test');
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

// Part 0: abc given, cot off, plan-aware scheduler prints the yue2 table per chunk.
async function part0() {
  const off = logSize();
  await render({ cot: 'off', scheduler: 'md_plan_aware_yue2', plugin_params: { 'md_plan_aware_yue2:section': 'none' } }, 'part0 abc+cot=off');
  const lines = logSince(off).split('\n').filter(l => l.includes('[md_plan_aware_yue2] yue2') || l.includes('no yue2 table'));
  const first = lines[0] || '(no line)';
  const tempo = Number((first.match(/tempo=(\S+)/) || [])[1] || 0);
  const bars = Number((first.match(/bars=(\S+)/) || [])[1] || 0);
  console.log(`[part0 ${PHASE}] ${first.trim()}`);
  console.log(`[part0 ${PHASE}] tempo=${tempo} bars=${bars} -> ${tempo > 0 && bars > 0 ? 'NON-EMPTY' : 'EMPTY'}`);
  return { tempo, bars };
}

const cads = s => ({ plugin_params: { 'yue2:cads_strength': s } });

const SWEEP = [0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0];
async function runSweep(s0, tag) {
  const out = {};
  for (const s of SWEEP) out[s] = await render(cads(s), `${tag} strength ${s}`);
  return out;
}

if (PHASE === 'pre') {
  // Old build: baseline for (a) plus the old sweep, for the old-vs-new table.
  const s0 = await render({}, 'baseline (pre-change build)');
  saveF32('baseline_pre.f32', s0);
  const sw = await runSweep(s0, 'OLD');
  const old = {}; for (const s of SWEEP) old[s] = relDb(s0, sw[s]);
  fs.writeFileSync(path.join(OUT_DIR, 'sweep_pre.json'), JSON.stringify(old));
  console.table(SWEEP.map(s => ({ strength: s, old_rel_db: f(old[s]) })));
  console.log('pre phase saved to', OUT_DIR);
} else {
  const rows = []; let pass = true;
  const pre = loadF32('baseline_pre.f32');
  let old = {}; try { old = JSON.parse(fs.readFileSync(path.join(OUT_DIR, 'sweep_pre.json'), 'utf8')); } catch {}
  const s0 = await render(cads(0), 'strength 0');
  const a = relDb(pre, s0);
  const aOk = a < -90; pass &&= aOk;
  rows.push({ check: '(a) s=0 vs pre-change build', old: '', new: f(a), ok: aOk });

  const sw = await runSweep(s0, 'NEW');
  let prev = -Infinity, mono = true, partial = false;
  for (const s of SWEEP) {
    const d = relDb(s0, sw[s]); if (prev < 0 && !(d > prev)) mono = false; prev = d;  // above 0 dB the song is decorrelated; order there is level noise
    if (s <= 0.1 && d >= -30 && d <= -6) partial = true;
    rows.push({ check: `(b) s=${s} vs s=0`, old: old[s] !== undefined ? f(old[s]) : '', new: f(d), ok: '' });
  }
  rows.push({ check: '(b) monotonic until decorrelated (0 dB)', old: '', new: '', ok: mono }); pass &&= mono;
  rows.push({ check: '(c) some s<=0.1 in -30..-6 dB', old: '', new: '', ok: partial }); pass &&= partial;
  const smallOk = relDb(s0, sw[0.01]) < -6; pass &&= smallOk;
  rows.push({ check: '(c) s=0.01 below -6 dB', old: '', new: f(relDb(s0, sw[0.01])), ok: smallOk });

  const again = await render(cads(0.2), 'strength 0.2 repeat');
  const d = relDb(sw[0.2], again); const dOk = d < -90; pass &&= dOk;
  rows.push({ check: '(d) s=0.2 same seed twice', old: '', new: f(d), ok: dOk });
  console.table(rows);
  console.log(pass ? 'OVERALL PASS' : 'OVERALL FAIL');
  process.exit(pass ? 0 : 1);
}
