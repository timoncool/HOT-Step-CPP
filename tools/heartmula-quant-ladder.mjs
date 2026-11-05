#!/usr/bin/env node
// HeartMuLa quant ladder: builds the global and local LM files at every level from the
// F16 pair, each with its own importance matrix, and scores them against the F16 pair.
//
//   node tools/heartmula-quant-ladder.mjs build --models <dir> --variant <3B|3B-rl|...>
//        --imatrix <dir> [--quantize <exe>] [--levels Q8_0,Q4_K_M,...] [--parts global,local]
//        [--jobs N] [--out-dir <dir>]    (--variant takes a comma list)
//   node tools/heartmula-quant-ladder.mjs eval  --models <dir> --variant <V> --ref-dir <dir>
//        [--exe <heartmula-quant>] [--levels ...] [--out report.json]
//
// build: <dir>/heartmula/heartmula-<v>-<part>-<LEVEL>.gguf from heartmula-<v>-<part>-F16.gguf,
//   imatrix <imatrix>/heartmula-<v>-<part>.imatrix.gguf (collect it with heartmula-quant
//   --mode collect). Existing outputs are kept. Every rung uses the imatrix: the engine
//   reads the level from the last dash segment of the name, so there is no -imat twin.
//   The metadata keys quantize.imatrix / quantize.imatrix_file record it.
//   --jobs runs that many quantize processes at once over every variant x part x level (default 6; the longest single level takes ~27 min, so more adds nothing).
//   Failed levels leave no .partial and the run exits 1 after the rest finish. --out-dir writes the
//   outputs and manifest elsewhere (sources and imatrices are still read from --models).
//   Writes heartmula-<v>-quant-manifest.json once, after the pool drains (parts, levels, bytes, hashes).
// eval: runs heartmula-quant --mode eval for each level on the held-out prompts, scoring
//   the quantized part alone (other part F16) and both together, against the --ref-dir
//   tracks written by --mode ref on the F16 pair. Prints one JSON line per run.
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, existsSync, mkdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const [cmd, ...rest] = process.argv.slice(2);
const opt = (name, dflt) => { const i = rest.indexOf(name); return i >= 0 ? rest[i + 1] : dflt; };

// The existing ladder (engine/quantize.sh), minus the F16 source.
const ALL = ['Q8_0', 'Q6_K', 'Q5_K_M', 'Q5_K_S', 'Q4_K_M', 'Q4_K_S', 'Q3_K_L', 'Q3_K_M', 'Q3_K_S', 'Q2_K',
  'IQ4_XS', 'IQ3_XXS', 'IQ2_XS', 'IQ2_XXS', 'IQ1_M', 'IQ1_S', 'MXFP4', 'NVFP4'];
const uniq = (list) => [...new Set(list)];
const levels = uniq((opt('--levels') ?? ALL.join(',')).split(','));
const parts = uniq((opt('--parts') ?? 'global,local').split(','));
const models = opt('--models');
// Variants differing only in case name the same files, so keep the first spelling of each.
const variant_list = (opt('--variant') ?? '').split(',').filter((x, i, a) => x && a.findIndex((y) => y.toLowerCase() === x.toLowerCase()) === i);
const variant = variant_list[0];
if (!['build', 'eval'].includes(cmd) || !models || !variant) {
  console.error(readFileSync(fileURLToPath(import.meta.url), 'utf8').split('\n').slice(1, 20).join('\n').replace(/^\/\/ ?/gm, ''));
  process.exit(2);
}
const dir = existsSync(join(models, 'heartmula')) ? join(models, 'heartmula') : models;
const v = variant.toLowerCase();
const sha = (p) => new Promise((ok, no) => {
  const h = createHash('sha256');
  createReadStream(p).on('data', (d) => h.update(d)).on('error', no).on('end', () => ok(h.digest('hex')));
});
const mb = (p) => (statSync(p).size / 1048576).toFixed(1);

if (cmd === 'build') {
  const imdir = opt('--imatrix');
  const exe = opt('--quantize') ?? ['engine/build/Release/quantize.exe', 'engine/build/quantize'].map((p) => join(root, p)).find(existsSync);
  if (!imdir || !exe) { console.error('build needs --imatrix <dir> and a quantize binary (--quantize)'); process.exit(2); }
  // quantize is single-threaded (one ggml_quantize_chunk call per tensor, no thread flag), about 1.5 cores
  // per run, so the speed-up is all in running many at once. Each job reads its own F16 source (6.5 GB for a
  // global part), so --jobs is bounded by RAM before it is by cores.
  const jobs = Number(opt('--jobs', 6));
  if (!Number.isInteger(jobs) || jobs < 1) { console.error(`--jobs must be a positive integer, got ${opt('--jobs')}`); process.exit(2); }
  const outDir = opt('--out-dir', dir); // measure into a scratch folder without touching the live quants
  mkdirSync(outDir, { recursive: true });
  const outName = (v, part, level) => join(outDir, `heartmula-${v}-${part}-${level}.gguf`);
  // Run one quantize child; resolves with its exit status. stderr is dropped unless it fails.
  const run = (args) => new Promise((ok) => {
    let err = '';
    const c = spawn(exe, args, { stdio: ['ignore', 'ignore', 'pipe'] });
    c.stderr.on('data', (d) => { err = (err + d).slice(-2000); });
    c.on('error', (e) => ok({ error: String(e), err }));
    c.on('close', (status) => ok({ status, err }));
  });
  const tasks = [], manifests = {};
  for (const variant of variant_list) {
    const v = variant.toLowerCase();
    const mpath = join(outDir, `heartmula-${v}-quant-manifest.json`);
    let m = { variant, rungs: [] };
    try { m = JSON.parse(readFileSync(mpath, 'utf8')); } catch { /* first build */ }
    m.variant = variant;
    m.quantize = exe;
    manifests[variant] = { mpath, m };
    for (const part of parts) {
      const src = join(dir, `heartmula-${v}-${part}-F16.gguf`);
      const im = join(imdir, `heartmula-${v}-${part}.imatrix.gguf`);
      for (const f of [src, im]) if (!existsSync(f)) { console.error(`missing ${f}`); process.exit(1); }
      // quantize does not embed hashes, so the manifest carries them; an output is reused only when
      // its recorded source, imatrix and output hashes all match the files on disk now.
      const [source_sha256, imatrix_sha256] = await Promise.all([sha(src), sha(im)]);
      const f16 = { part, level: 'F16', bytes: statSync(src).size, source_sha256 };
      m.rungs = m.rungs.filter((r) => !(r.part === part && r.level === 'F16')).concat(f16);
      for (const level of levels) tasks.push({ variant, v, part, level, src, im, source_sha256, imatrix_sha256, mpath });
    }
  }
  // Longest first (measured: a global IQ level takes ~10 min, a K-quant under 1, a local part a few seconds
  // to 2 min), so the slow jobs start together and short ones fill the tail.
  const cost = (t) => (t.part === 'global' ? 10 : 1) * (t.level.startsWith('IQ') ? 5 : 1);
  tasks.sort((a, b) => cost(b) - cost(a));
  const failed = [];
  const work = async (t) => {
    const out = outName(t.v, t.part, t.level), tag = `${t.variant} ${t.part} ${t.level}`;
    const { m } = manifests[t.variant];
    const old = m.rungs.find((r) => r.part === t.part && r.level === t.level);
    const reusable = existsSync(out) && old && old.source_sha256 === t.source_sha256 && old.imatrix_sha256 === t.imatrix_sha256
      && old.bytes === statSync(out).size && old.output_sha256 === await sha(out);
    let output_sha256 = reusable ? old.output_sha256 : undefined;
    if (reusable) console.log(`${tag}: kept ${mb(out)} MB (hashes match)`);
    else {
      if (existsSync(out)) console.log(`${tag}: stale or unrecorded, rebuilding`);
      const t0 = Date.now();
      const r = await run([t.src, `${out}.partial`, t.level, '--imatrix', t.im]);
      if (r.error || r.status !== 0) {
        rmSync(`${out}.partial`, { force: true });
        failed.push(tag);
        console.error(`FAILED ${tag} (${r.error ?? 'exit ' + r.status})\n${r.err}`);
        return;
      }
      renameSync(`${out}.partial`, out);
      output_sha256 = await sha(out);
      console.log(`${tag}: ${mb(out)} MB in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
    }
    // Only the parent touches the manifest object, and the file is written once after the pool drains.
    m.rungs = m.rungs.filter((r) => !(r.part === t.part && r.level === t.level))
      .concat({ part: t.part, level: t.level, bytes: statSync(out).size, source_sha256: t.source_sha256, imatrix: t.im, imatrix_sha256: t.imatrix_sha256, output_sha256 });
  };
  const queue = tasks.slice(), t00 = Date.now();
  await Promise.all(Array.from({ length: Math.min(jobs, queue.length) }, async () => { for (let t; (t = queue.shift());) await work(t); }));
  for (const { mpath, m } of Object.values(manifests)) {
    writeFileSync(`${mpath}.partial`, JSON.stringify(m, null, 1));
    renameSync(`${mpath}.partial`, mpath);
  }
  console.log(`build: ${tasks.length - failed.length}/${tasks.length} ok in ${((Date.now() - t00) / 1000).toFixed(0)} s (jobs ${jobs})`);
  if (failed.length) { console.error(`failed: ${failed.join(', ')}`); process.exit(1); }
} else {
  const exe = opt('--exe') ?? ['engine/build/Release/heartmula-quant.exe', 'engine/build/heartmula-quant'].map((p) => join(root, p)).find(existsSync);
  const refDir = opt('--ref-dir');
  const prompts = opt('--prompts') ?? join(root, 'engine/tools/heartmula-quant-prompts.json');
  if (!exe || !refDir) { console.error('eval needs --ref-dir and a heartmula-quant binary (--exe)'); process.exit(2); }
  const rows = [];
  for (const level of levels) {
    for (const [g, l] of [[level, 'F16'], ['F16', level], [level, level]]) {
      const r = spawnSync(exe, ['--mode', 'eval', '--models', models, '--prompts', prompts, '--set', 'heldout', '--variant', variant,
        '--global', g, '--local', l, '--ref-dir', refDir], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'] });
      if (r.status !== 0) { rows.push({ global: g, local: l, error: `exit ${r.status}` }); console.log(JSON.stringify(rows.at(-1))); continue; }
      rows.push(JSON.parse(r.stdout.trim().split('\n').at(-1)));
      console.log(JSON.stringify(rows.at(-1)));
    }
  }
  if (opt('--out')) writeFileSync(opt('--out'), JSON.stringify(rows, null, 1));
}
