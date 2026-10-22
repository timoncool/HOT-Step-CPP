#!/usr/bin/env node
// Headless engine smoke test: renders two short fixed-seed clips with
// ace-synth --check (no UI, no server) and fails unless each raw output is
// finite, free of subnormals, audible, free of DC offset and not clipping, and
// the WAV on disk is fresh and exactly the length the latent grid predicts.
// See the engine-dev-loop skill.
//
//   node tools/engine-smoke.mjs [--models <dir>] [--dit <file.gguf>] [--self-test]
//
// Cases: "round" is 10 s (250 latent frames, one VAE tile). "odd" is 7.33 s:
// 183.25 frames floors to 183, pads to 184 for the DiT patch, and decodes in
// ragged 96-frame VAE tiles, so padding and tile-edge indexing get exercised.
// --self-test renders 6 tracks, corrupts 5 inside ace-synth
// (HOTSTEP_CHECK_INJECT), then forces a write failure; it passes only if every
// fault is caught.
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const opt = (name) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : undefined; };
const selfTest = args.includes('--self-test');

const exe = ['engine/build/Release/ace-synth.exe', 'engine/build/ace-synth.exe', 'engine/build/ace-synth', 'engine/ace-synth.exe', 'engine/ace-synth']
  .map((p) => join(root, p)).find(existsSync);
if (!exe) { console.error('engine-smoke: no ace-synth binary under engine/build; build the engine first'); process.exit(1); }

let models = opt('--models');
if (!models) {
  const r = spawnSync(process.execPath, ['--import', 'tsx', 'scripts/data-paths.ts'], { cwd: join(root, 'server'), encoding: 'utf8' });
  try { models = JSON.parse(r.stdout).children.models; } catch { /* reported below */ }
  if (!models) { console.error('engine-smoke: could not resolve the models folder; pass --models <dir>\n' + (r.stderr || '')); process.exit(1); }
}

// Each run gets its own folder, so overlapping runs never share a WAV, log or
// squat path. Folders from runs older than a day are pruned.
const smokeDir = join(root, 'logs', 'engine-smoke');
mkdirSync(smokeDir, { recursive: true });
for (const d of readdirSync(smokeDir)) {
  const p = join(smokeDir, d);
  // Best effort: a concurrent run may be pruning the same folder.
  try { if (/^\d{14}-\d+$/.test(d) && statSync(p).mtimeMs < Date.now() - 86_400_000) rmSync(p, { recursive: true, force: true }); } catch { /* gone or busy */ }
}
const runId = `${new Date().toISOString().replace(/\D/g, '').slice(0, 14)}-${process.pid}`;
const outDir = join(smokeDir, runId);
mkdirSync(outDir);
const rel = `logs/engine-smoke/${runId}`;
console.log(`engine-smoke: ${exe}\n  models  ${models}\n  output  ${rel}`);

// frames = floor(duration * 25) rounded up to the DiT patch (2); 1920 samples per frame.
const expectSamples = (duration) => Math.ceil(Math.floor(duration * 25) / 2) * 2 * 1920;

// Data chunk length of a 16-bit stereo WAV, in samples per channel.
function wavSamples(path) {
  const b = readFileSync(path);
  for (let p = 12; p + 8 <= b.length; p += 8 + b.readUInt32LE(p + 4)) {
    if (b.toString('ascii', p, p + 4) === 'data') return b.readUInt32LE(p + 4) / 4;
  }
  return -1;
}

function stageTimes(log) {
  const t = {};
  const add = (k, v) => { t[k] = (t[k] ?? 0) + Number(v); };
  for (const m of log.matchAll(/\[Store\] Load ([\w-]+): (\d+) ms/g)) add(`load ${m[1]}`, m[2]);
  for (const m of log.matchAll(/\[Encode-Text Batch\d+\].*?([\d.]+) ms/g)) add('encode', m[1]);
  for (const m of log.matchAll(/\[DiT-Generate\] Total: ([\d.]+) ms/g)) add('dit', m[1]);
  for (const m of log.matchAll(/\[VAE-Decode Batch\d+\] Decode: ([\d.]+) ms/g)) add('vae', m[1]);
  return t;
}

// One ace-synth run. Returns { status, lines, log, times, secs }.
function render(name, req, { env = {}, extra = [] } = {}) {
  const request = join(outDir, `${name}.json`);
  writeFileSync(request, JSON.stringify({
    caption: 'upbeat electronic pop, synth bass, drums, bright lead',
    lyrics: '[Instrumental]',
    bpm: 120,
    inference_steps: 8,
    synth_batch_size: 1,
    synth_model: opt('--dit') ?? 'acestep-v15-turbo-BF16.gguf',
    output_format: 'wav16',
    ...req,
  }, null, 2));
  const t0 = Date.now();
  const r = spawnSync(exe, ['--models', models, '--request', request, '--check', ...extra],
    { env: { ...process.env, ...env }, encoding: 'utf8', timeout: 15 * 60_000, maxBuffer: 256 << 20 });
  const secs = (Date.now() - t0) / 1000;
  const log = (r.stdout ?? '') + (r.stderr ?? '');
  writeFileSync(join(outDir, `${name}.log`), log);
  const lines = log.split(/\r?\n/).filter((l) => l.startsWith('[Check Batch') || l.includes('FATAL'));
  const times = stageTimes(log);
  console.log(`\n[${name}] exit ${r.status ?? r.error?.message} after ${secs.toFixed(1)}s (log ${rel}/${name}.log)`);
  lines.forEach((l) => console.log('  ' + l));
  console.log('  stages: ' + (Object.entries(times).map(([k, v]) => `${k} ${v.toFixed(0)} ms`).join(', ') || 'none found'));
  return { status: r.status, lines, log, times, secs, t0 };
}

const fails = [];
const expect = (cond, what) => { if (!cond) fails.push(what); };

if (selfTest) {
  const faults = ['none', 'nan', 'subnormal', 'silence', 'dc', 'clip'];
  const tags = ['PASS', 'nonfinite=1 [FAIL]', 'subnormal=1 [FAIL]', '[FAIL silence]', '[FAIL dc]', '[FAIL clip]'];
  const r = render('selftest', { duration: 10, seed: 42, synth_batch_size: faults.length }, { env: { HOTSTEP_CHECK_INJECT: faults.join(',') } });
  faults.forEach((f, b) => {
    const caught = r.lines.some((l) => l.startsWith(`[Check Batch${b}]`) && l.includes(tags[b]));
    console.log(`  self-test track ${b} (${f}): ${caught ? 'caught' : 'MISSED'}`);
    expect(caught, `self-test ${f}`);
  });
  expect(r.status === 2, `self-test exit ${r.status}, want 2`);

  // Write failure: a directory squats on the output path, so the WAV open fails.
  const squat = join(outDir, 'writefail0.wav');
  rmSync(squat, { recursive: true, force: true });
  mkdirSync(squat);
  const w = render('writefail', { duration: 10, seed: 42 });
  rmSync(squat, { recursive: true, force: true });
  const caught = w.status === 1 && w.lines.some((l) => l.includes('FATAL: failed to write'));
  console.log(`  self-test write failure: ${caught ? 'caught' : 'MISSED'} (exit ${w.status}, want 1)`);
  expect(caught, 'self-test write failure');
} else {
  const cases = [
    { name: 'round', req: { duration: 10, seed: 42 }, extra: [] },
    { name: 'odd', req: { duration: 7.33, seed: 43 }, extra: ['--vae-chunk', '96', '--vae-overlap', '16'] },
  ];
  for (const c of cases) {
    const wav = join(outDir, `${c.name}0.wav`);
    rmSync(wav, { force: true });
    const r = render(c.name, c.req, { extra: c.extra });
    const want = expectSamples(c.req.duration);
    const fresh = existsSync(wav) && statSync(wav).mtimeMs >= r.t0 - 2000;
    const got = fresh ? wavSamples(wav) : -1;
    console.log(`  wav: ${fresh ? `${got} samples/ch (want ${want}, ${(want / 48000).toFixed(2)} s)` : 'MISSING or stale'}`);
    expect(r.status === 0, `${c.name} exit ${r.status}`);
    expect(r.lines.length === 1 && r.lines[0].includes(' PASS '), `${c.name} check`);
    expect(fresh && got === want, `${c.name} wav length ${got} != ${want}`);
    expect(r.times.dit !== undefined && r.times.vae !== undefined, `${c.name} stage times missing from log`);
  }
}

if (fails.length) console.log('\nfailed: ' + fails.join('; '));
console.log(fails.length ? 'ENGINE SMOKE: FAIL' : 'ENGINE SMOKE: PASS');
process.exit(fails.length ? 1 : 0);
