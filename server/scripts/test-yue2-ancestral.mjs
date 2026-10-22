#!/usr/bin/env node
// A/B determinism test for the md_ancestral_yue2 NAR solver (churn).
// Talks to ace-server directly (default http://127.0.0.1:8085).
// Usage: node server/scripts/test-yue2-ancestral.mjs [--base URL] [--seed N] [--frames N]
// rel diff dB = 20*log10(||a-b|| / ||a||)
import { writeFileSync, mkdirSync } from 'node:fs';

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const BASE = arg('--base', 'http://127.0.0.1:8085');
const SEED = Number(arg('--seed', '1234'));
const FRAMES = Number(arg('--frames', '500')); // 25 frames/s -> 20 s
const OUT = arg('--out', '');
const CHUNK = Number(arg('--chunk', '250')); // nar_chunk_frames: 500/250 -> 2 NAR chunks
const TIMEOUT_MS = Number(arg('--timeout', '300')) * 1000; // hard cap per render
const tfetch = (url, opt = {}) => fetch(url, { ...opt, signal: AbortSignal.timeout(30000) });

async function render(churn, seed = SEED, solver = 'md_ancestral_yue2') {
  const body = {
    style: 'instrumental, lo-fi hip hop, mellow piano, soft drums',
    lyrics: '[instrumental]',
    cot: 'off',
    seed,
    nar_chunk_frames: CHUNK,
    ode_steps: 16,
    preview_max_frames: FRAMES,
    infer_method: solver,
    plugin_params: { [`${solver}:churn`]: churn },
  };
  const t0 = Date.now();
  const r = await tfetch(`${BASE}/yue2/synth`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`synth ${r.status}: ${await r.text()}`);
  const sub = await r.json();
  const job_id = sub.job_id ?? sub.id;
  if (!job_id) throw new Error(`no job id in synth response: ${JSON.stringify(sub)}`);
  let last = '';
  for (;;) {
    if (Date.now() - t0 > TIMEOUT_MS) {
      await tfetch(`${BASE}/job?id=${job_id}&cancel=1`, { method: 'POST' }).catch(() => {});
      throw new Error(`job ${job_id} timed out after ${TIMEOUT_MS / 1000}s (last status ${last})`);
    }
    await new Promise((res) => setTimeout(res, 1000));
    const s = await (await tfetch(`${BASE}/job?id=${job_id}`)).json();
    const st = String(s.status ?? s.state ?? '').toLowerCase();
    last = JSON.stringify(s);
    if (st === 'done' || st === 'complete' || st === 'completed' || st === 'succeeded') break;
    if (st === 'failed' || st === 'error' || st.startsWith('cancel')) throw new Error(`job ${job_id} ${st}: ${last}`);
  }
  console.error(`render churn=${churn} seed=${seed}: ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  const res = await tfetch(`${BASE}/job?id=${job_id}&result=1`);
  if (!res.ok) throw new Error(`result ${res.status}: ${await res.text()}`);
  const buf = Buffer.from(await res.arrayBuffer());
  return decodeWav(buf, res.headers.get('content-type'));
}

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
      if (fmt.bits === 32) { const o = new Float32Array(d.length >> 2); for (let i = 0; i < o.length; i++) o[i] = d.readInt32LE(i * 4) / 2147483648; return o; }
      throw new Error(`unsupported wav fmt ${JSON.stringify(fmt)}`);
    }
    p += 8 + sz + (sz & 1);
  }
  throw new Error('no data chunk');
}

function relDb(a, b) {
  const n = Math.min(a.length, b.length);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { const d = a[i] - b[i]; num += d * d; den += a[i] * a[i]; }
  if (num === 0) return -Infinity;
  return 20 * Math.log10(Math.sqrt(num) / Math.sqrt(den));
}

const rows = [];
const check = (name, a, b, pass) => {
  const db = relDb(a, b);
  const ok = pass(db);
  rows.push({ test: name, samples: `${a.length}/${b.length}`, rel_diff_db: db === -Infinity ? '-inf' : db.toFixed(2), result: ok ? 'PASS' : 'FAIL' });
};

const c0a = await render(0);
const c0b = await render(0);
const c5a = await render(0.5);
const c5b = await render(0.5);
const c5s = await render(0.5, SEED + 1);
check('(a) churn0 vs churn0', c0a, c0b, (d) => d < -90);
check('(b) churn0.5 vs churn0', c0a, c5a, (d) => d > -20);
check('(c) churn0.5 vs churn0.5', c5a, c5b, (d) => d < -90);
check('(e) churn0.5 seed S vs S+1', c5a, c5s, (d) => d > -20);
// (f) churn-induced difference, chunk 1 vs chunk 2 segment (same length, aligned
// from each chunk start). Identical per-chunk solver noise shows up as a high
// correlation of the two segments; independent noise stays near 0.
{
  const n = Math.min(c0a.length, c5a.length);
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = c5a[i] - c0a[i];
  const ch = 2, sr = 48000, spf = sr / 25; // interleaved stereo, 25 latent frames/s
  const seg = Math.floor(CHUNK * spf) * ch;
  const L = Math.min(seg, n - seg);
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < L; i++) { const a = d[i], b = d[seg + i]; sab += a * b; saa += a * a; sbb += b * b; }
  const corr = sab / Math.sqrt(saa * sbb || 1);
  rows.push({ test: '(f) chunk1 vs chunk2 churn-diff corr', samples: `${L}`, rel_diff_db: corr.toFixed(4), result: Math.abs(corr) < 0.3 ? 'PASS' : 'FAIL' });
}
console.table(rows);
if (OUT) { mkdirSync(OUT, { recursive: true }); writeFileSync(`${OUT}/rows.json`, JSON.stringify(rows, null, 2)); }
process.exit(rows.every((r) => r.result === 'PASS') ? 0 : 1);
