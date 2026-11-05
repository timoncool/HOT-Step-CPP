#!/usr/bin/env node
// SA3 resume continuity proof for checkpoint previews (HOTSTEP-674 plan, point 8).
//
// Checkpoint previews stop the trainer at a cadence boundary, render, and
// relaunch it with --resume and a larger absolute --steps. That only works if
// the resumed run is the run that never stopped. This proof trains the same
// fixture twice on the same backend:
//   reference: one uninterrupted --steps 12, state.gguf copied at steps 4 and 8
//   segmented: --steps 4, then --resume --steps 8, then --resume --steps 12
// and checks
//   - state at each restart (parameters and AdamW m/v) is bitwise equal to the
//     reference at that step, and the file the resume reads is the file the
//     previous segment wrote (sha256 before relaunch == archived copy)
//   - every step 1..12 is reported once, resumes start at 4 and 8, and the
//     emitted draws (sample, cropStart, t, maskType, cfgDrop) match exactly
//   - loss on the 8 steps after the first boundary, and the final parameter and
//     moment tensors, satisfy abs(seg - ref) <= 1e-6 + 1e-5 * abs(ref)
// Each segment's adapter.gguf is archived as snapshots/step-N.gguf, the capture
// the server makes before the next segment overwrites it. EMA resets on resume
// and is display state only, so it is not compared.
//
// The fixture is a finished SA3 run's prepared dataset and encoding cache, with
// its first clip added a second time under a new name so the sample draw has two
// choices. Receipts (resume-proof.json, tensor-diff.json) stay in --out.
//   node tools/sa3-resume-proof.mjs --models <dir> --source <run dir> --out <dir>
//        [--backend cpu|gpu] [--steps 4,8,12] [--keep]

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : dflt;
};
const models = arg('models'), source = arg('source'), out = arg('out');
const backend = arg('backend', 'gpu');
const bounds = arg('steps', '4,8,12').split(',').map(Number);
const keep = process.argv.includes('--keep');
if (!models || !source || !out || !['cpu', 'gpu'].includes(backend) || bounds.some(n => !Number.isSafeInteger(n) || n < 1)) {
  console.error('usage: node tools/sa3-resume-proof.mjs --models <dir> --source <run dir> --out <dir> [--backend cpu|gpu] [--steps 4,8,12] [--keep]');
  process.exit(2);
}
const total = bounds[bounds.length - 1];
const exe = path.join(repo, 'engine', 'build', 'Release', process.platform === 'win32' ? 'ace-train.exe' : 'ace-train');
const tol = (ref) => 1e-6 + 1e-5 * Math.abs(ref);

let failed = 0;
const check = (name, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${detail ? `: ${detail}` : ''}`);
  if (!ok) failed++;
  return ok;
};
const sha = (file) => createHash('sha256').update(fs.readFileSync(file)).digest('hex');

// ── minimal GGUF v3 reader (F32 tensors, scalar/string/array metadata) ──
function readGguf(file) {
  const b = fs.readFileSync(file);
  let o = 0;
  const u32 = () => { const v = b.readUInt32LE(o); o += 4; return v; };
  const u64 = () => { const v = b.readBigUInt64LE(o); o += 8; return v; };
  const str = () => { const n = Number(u64()); const s = b.toString('utf8', o, o + n); o += n; return s; };
  const SIZES = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
  const val = (t) => {
    if (t === 8) return str();
    if (t === 9) { const et = u32(), n = Number(u64()); const a = []; for (let i = 0; i < n; i++) a.push(val(et)); return a; }
    if (t === 4) return u32();
    if (t === 10) return u64();
    const v = b.subarray(o, o + SIZES[t]); o += SIZES[t]; return v;
  };
  if (b.toString('ascii', 0, 4) !== 'GGUF') throw new Error(`${file}: not GGUF`);
  o = 4; if (u32() !== 3) throw new Error(`${file}: not GGUF v3`);
  const nt = Number(u64()), nkv = Number(u64());
  const kv = {};
  for (let i = 0; i < nkv; i++) { const k = str(); kv[k] = val(u32()); }
  const infos = [];
  for (let i = 0; i < nt; i++) {
    const name = str(), nd = u32(); let n = 1;
    for (let d = 0; d < nd; d++) n *= Number(u64());
    infos.push({ name, n, type: u32(), offset: Number(u64()) });
  }
  const align = typeof kv['general.alignment'] === 'number' ? kv['general.alignment'] : 32;
  const base = Math.ceil(o / align) * align;
  const tensors = new Map();
  for (const t of infos) {
    if (t.type !== 0) throw new Error(`${file}: ${t.name} is not F32`);
    const s = b.subarray(base + t.offset, base + t.offset + 4 * t.n);
    tensors.set(t.name, new Float32Array(s.buffer.slice(s.byteOffset, s.byteOffset + s.length)));
  }
  return { kv, tensors };
}

// Per-tensor comparison: bitwise equality, max abs difference, and the worst
// ratio of |seg - ref| to the point-8 tolerance (<= 1 passes).
function compareStates(refFile, segFile) {
  const ref = readGguf(refFile), seg = readGguf(segFile);
  const rows = [];
  for (const [name, r] of ref.tensors) {
    const s = seg.tensors.get(name);
    if (!s || s.length !== r.length) { rows.push({ name, n: r.length, missing: true }); continue; }
    const rb = new Uint32Array(r.buffer), sb = new Uint32Array(s.buffer);
    let bitwise = true, maxAbs = 0, worst = 0;
    for (let i = 0; i < r.length; i++) {
      if (rb[i] !== sb[i]) bitwise = false;
      const d = Math.abs(s[i] - r[i]);
      if (d > maxAbs) maxAbs = d;
      const q = d / tol(r[i]);
      if (q > worst || Number.isNaN(q)) worst = Number.isNaN(q) ? Infinity : q;
    }
    rows.push({ name, n: r.length, bitwise, maxAbs, worstTolRatio: worst });
  }
  for (const name of seg.tensors.keys()) if (!ref.tensors.has(name)) rows.push({ name, extra: true });
  const step = (g) => Number(g.kv['sa3t.step']);
  return {
    refStep: step(ref), segStep: step(seg), tensors: rows.length,
    bitwise: rows.every(r => r.bitwise), withinTol: rows.every(r => r.worstTolRatio <= 1),
    maxAbs: Math.max(0, ...rows.map(r => r.maxAbs ?? Infinity)),
    worstTolRatio: Math.max(0, ...rows.map(r => r.worstTolRatio ?? Infinity)),
    adamBitwise: rows.filter(r => /\.adam_[mv]$/.test(r.name)).every(r => r.bitwise),
    rows,
  };
}

// Run one trainer process; calls onEvent for each JSONL event.
function train(dir, steps, resume, onEvent = () => {}) {
  const saved = JSON.parse(fs.readFileSync(path.join(source, 'run.json'), 'utf8'));
  const args = ['sa3-train', '--models', models, '--base-dit', path.join(models, 'sa3-dit-base-BF16.gguf'),
    '--inference-dit', path.join(models, 'sa3-dit-BF16.gguf'), '--manifest', path.join(dir, 'train-manifest.json'),
    '--out', dir, '--steps', String(steps), '--rank', String(saved.rank), '--alpha', String(saved.alpha),
    '--lr', String(saved.learningRate), '--duration', String(saved.duration), '--seed', String(saved.seed),
    '--checkpoint-every', String(bounds[0]), '--export-every', '0', '--bwd', 'mm', ...(resume ? ['--resume'] : []), '--jsonl'];
  const env = { ...process.env, ...(backend === 'cpu' ? { GGML_BACKEND: 'CPU' } : {}) };
  return new Promise((resolve, reject) => {
    const child = spawn(exe, args, { env, windowsHide: true });
    const events = [], stderr = [];
    child.stderr.on('data', d => { stderr.push(d.toString()); if (stderr.length > 40) stderr.shift(); });
    readline.createInterface({ input: child.stdout }).on('line', line => {
      let ev; try { ev = JSON.parse(line); } catch { return; }
      events.push(ev); onEvent(ev);
    });
    const timer = setTimeout(() => child.kill(), 60 * 60_000);
    child.once('error', reject);
    child.once('close', code => {
      clearTimeout(timer);
      if (code !== 0) reject(new Error(`ace-train --steps ${steps} exited ${code}: ${stderr.join('').slice(-1500)}`));
      else resolve(events);
    });
  });
}

function makeFixture(dir) {
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  for (const name of ['dataset', 'cache']) fs.cpSync(path.join(source, name), path.join(dir, name), { recursive: true });
  const manifest = JSON.parse(fs.readFileSync(path.join(source, 'train-manifest.json'), 'utf8'));
  const first = manifest.samples[0];
  const twin = first.audio_path.replace(/(\.\w+)$/, '-twin$1');
  fs.copyFileSync(path.join(dir, first.audio_path), path.join(dir, twin));
  manifest.samples = [...manifest.samples, { ...first, audio_path: twin }];
  fs.writeFileSync(path.join(dir, 'train-manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
}

const DRAWS = ['sample', 'cropStart', 't', 'maskType', 'cfgDrop'];
const stepsOf = (events) => events.filter(e => e.type === 'step');

fs.mkdirSync(out, { recursive: true });
const refDir = path.join(out, 'reference'), segDir = path.join(out, 'segmented');
const receipt = { bounds, backend, source: path.basename(source), tolerance: 'abs <= 1e-6 + 1e-5*abs(ref)', exe };
const tensorDiff = {};
try {
  makeFixture(refDir); makeFixture(segDir);

  // Reference: uninterrupted. The trainer computes `bounds[0]` more steps before
  // it next rewrites state.gguf, so the copy reads a settled file; its step
  // field is checked anyway.
  const refStates = {};
  const t0 = Date.now();
  const refEvents = await train(refDir, total, false, ev => {
    if (ev.type === 'checkpoint' && bounds.includes(ev.step) && ev.step < total) {
      const copy = path.join(out, `ref-state-${ev.step}.gguf`);
      fs.copyFileSync(path.join(refDir, 'state.gguf'), copy);
      refStates[ev.step] = copy;
    }
  });
  receipt.referenceMs = Date.now() - t0;
  refStates[total] = path.join(refDir, 'state.gguf');
  const refStart = refEvents.find(e => e.type === 'start');
  receipt.backendName = refStart?.backend;
  console.log(`reference: ${stepsOf(refEvents).length} steps on ${refStart?.backend} in ${receipt.referenceMs} ms`);

  // Segmented: each segment ends at a boundary, archives its export, resumes.
  const segEvents = [], resumes = [], restarts = [];
  fs.mkdirSync(path.join(segDir, 'snapshots'), { recursive: true });
  let prev = 0;
  for (const b of bounds) {
    const before = prev ? sha(path.join(segDir, 'state.gguf')) : null;
    const t1 = Date.now();
    const evs = await train(segDir, b, prev > 0);
    const done = evs.find(e => e.type === 'done');
    check(`segment to ${b}: trainer reported done at ${b}`, done?.steps === b, JSON.stringify(done ?? null));
    const resume = evs.find(e => e.type === 'resume');
    if (prev) {
      resumes.push(resume?.step);
      check(`segment to ${b}: resumed at ${prev}`, resume?.step === prev && evs.find(e => e.type === 'start')?.startStep === prev);
      restarts[restarts.length - 1].shaBeforeRelaunch = before;
      check(`restart at ${prev}: resume read the bytes the previous segment saved`, before === restarts[restarts.length - 1].sha);
    }
    segEvents.push(...evs);
    // Immutable capture before the next segment overwrites adapter.gguf.
    const snap = path.join(segDir, 'snapshots', `step-${b}.gguf`);
    fs.copyFileSync(path.join(segDir, 'adapter.gguf'), snap, fs.constants.COPYFILE_EXCL);
    if (b < total) {
      const copy = path.join(out, `seg-state-${b}.gguf`);
      fs.copyFileSync(path.join(segDir, 'state.gguf'), copy);
      restarts.push({ step: b, sha: sha(copy), ms: Date.now() - t1 });
    }
    prev = b;
  }
  receipt.resumes = resumes;
  receipt.snapshots = fs.readdirSync(path.join(segDir, 'snapshots')).sort();

  // Restart states: bitwise equal to the uninterrupted run at the same step.
  for (const r of restarts) {
    const d = compareStates(refStates[r.step], path.join(out, `seg-state-${r.step}.gguf`));
    tensorDiff[`restart-${r.step}`] = d;
    r.bitwise = d.bitwise; r.maxAbs = d.maxAbs; r.tensors = d.tensors;
    check(`restart at ${r.step}: state steps`, d.refStep === r.step && d.segStep === r.step, `${d.refStep}/${d.segStep}`);
    check(`restart at ${r.step}: ${d.tensors} parameter and AdamW tensors bitwise equal`, d.bitwise,
      `maxAbs ${d.maxAbs.toExponential(3)}, worst tol ratio ${d.worstTolRatio.toExponential(3)}`);
  }
  receipt.restarts = restarts;
  // Negative control: the comparator must see two different steps as different.
  if (restarts.length > 1) {
    const ctl = compareStates(path.join(out, `seg-state-${restarts[0].step}.gguf`), path.join(out, `seg-state-${restarts[1].step}.gguf`));
    receipt.negativeControl = { steps: [restarts[0].step, restarts[1].step], bitwise: ctl.bitwise, maxAbs: ctl.maxAbs };
    check(`comparator flags step ${restarts[0].step} vs step ${restarts[1].step} as different`, !ctl.bitwise && ctl.maxAbs > 0,
      `maxAbs ${ctl.maxAbs.toExponential(3)}`);
  }

  // Steps and draws.
  const rs = stepsOf(refEvents), ss = stepsOf(segEvents);
  check(`reference reports steps 1..${total} once`, rs.map(e => e.step).join() === Array.from({ length: total }, (_, i) => i + 1).join());
  check(`segmented reports steps 1..${total} once`, ss.map(e => e.step).join() === Array.from({ length: total }, (_, i) => i + 1).join(),
    ss.map(e => e.step).join());
  const table = [];
  for (let i = 0; i < Math.min(rs.length, ss.length); i++) {
    const r = rs[i], s = ss[i], d = Math.abs(s.loss - r.loss);
    table.push({ step: r.step, refLoss: r.loss, segLoss: s.loss, absDiff: d, tol: tol(r.loss),
      drawsEqual: DRAWS.every(k => r[k] === s[k]), draws: Object.fromEntries(DRAWS.map(k => [k, [r[k], s[k]]])) });
  }
  receipt.steps = table;
  check('draws (sample, cropStart, t, maskType, cfgDrop) identical on every step', table.every(t => t.drawsEqual));
  check('sample draw exercised both clips', new Set(rs.map(e => e.sample)).size > 1, [...new Set(rs.map(e => e.sample))].join());
  const after = table.filter(t => t.step > bounds[0]);
  check(`loss within tolerance on the ${after.length} steps after the first boundary`, after.length === total - bounds[0]
    && after.every(t => t.absDiff <= t.tol), `max abs diff ${Math.max(...after.map(t => t.absDiff)).toExponential(3)}`);

  // Final parameters and moments.
  const fin = compareStates(refStates[total], path.join(segDir, 'state.gguf'));
  tensorDiff[`final-${total}`] = fin;
  receipt.final = { tensors: fin.tensors, bitwise: fin.bitwise, withinTol: fin.withinTol, maxAbs: fin.maxAbs, worstTolRatio: fin.worstTolRatio };
  check(`final step-${total} state`, fin.refStep === total && fin.segStep === total, `${fin.refStep}/${fin.segStep}`);
  check(`final ${fin.tensors} parameter and AdamW tensors within tolerance`, fin.withinTol,
    `bitwise ${fin.bitwise}, maxAbs ${fin.maxAbs.toExponential(3)}, worst tol ratio ${fin.worstTolRatio.toExponential(3)}`);
  receipt.finalAdapterSha = { reference: sha(path.join(refDir, 'adapter.gguf')), segmented: sha(path.join(segDir, 'adapter.gguf')) };
  console.log(`final adapter.gguf ${receipt.finalAdapterSha.reference === receipt.finalAdapterSha.segmented ? 'byte-identical' : 'differs'}`);
} catch (err) {
  check('proof ran to completion', false, err.message);
} finally {
  receipt.failed = failed;
  receipt.verdict = failed ? 'FAIL' : 'PASS';
  fs.writeFileSync(path.join(out, 'resume-proof.json'), `${JSON.stringify(receipt, null, 2)}\n`);
  fs.writeFileSync(path.join(out, 'tensor-diff.json'), `${JSON.stringify(tensorDiff, null, 2)}\n`);
  if (!keep) {
    for (const d of [refDir, segDir]) fs.rmSync(d, { recursive: true, force: true });
    for (const f of fs.readdirSync(out)) if (f.endsWith('.gguf')) fs.rmSync(path.join(out, f));
  }
}
console.log(`sa3-resume-proof: ${failed ? `FAIL (${failed})` : 'PASS'} on ${receipt.backendName ?? backend}; receipts in ${out}`);
process.exit(failed ? 1 : 0);
