#!/usr/bin/env node
// SA3 checkpoint-preview loader proof (engine/src/sa3-preview-resolver.h).
//
// 1. Copies a finished SA3 run's prepared dataset into a fixture run and
//    trains it fresh for 4 steps with --export-every 3, so the trainer writes a
//    real snapshots/step-3.gguf. (A copied state.gguf cannot --resume: the run
//    fingerprint hashes absolute audio paths.)
// 2. Renders /sa3-synth at a fixed seed: base twice (determinism), then with
//    the snapshot, which must differ from base.
// 3. Rejections: traversal and bad charsets, a missing step, a junction that
//    escapes the runs root, wrong base (another generation DiT, forged base id),
//    a non-adapter GGUF.
// 4. Stale cache: rewrites the step file with the source run's final export;
//    the same request must render differently, and match the public adapter
//    path when that export is published byte-identically.
// 5. The public adapter-name parser and /sa3/props catalogue are unchanged.
// 0. First, with no GPU: builds a small C++ test of the resolver with clang++
//    against ggml-base and swaps the step file between the resolver's read and
//    its validate/hash/load steps. The resolver must accept or reject the bytes
//    it read, never the replacement. Also rejects non-finite and junk scales.
//    --unit-only stops after this part.
//
// Needs the rebuilt dev engine on an idle GPU. Removes its fixtures unless --keep.
//   node tools/sa3-preview-proof.mjs --adapters <dir> --models <dir>
//        [--engine http://127.0.0.1:8085] [--source <dataset>/<adapter>/<stamp>]
//   node tools/sa3-preview-proof.mjs --unit-only

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : dflt;
};
const engine = arg('engine', 'http://127.0.0.1:8085');
const adapters = arg('adapters');
const models = arg('models');
const source = arg('source', 'sa3-integrated-proof/proof-1792701425811/2026-10-22_21-37-05');
const keep = process.argv.includes('--keep');
const unitOnly = process.argv.includes('--unit-only');
if (!unitOnly && (!adapters || !models)) {
  console.error('usage: node tools/sa3-preview-proof.mjs --adapters <dir> --models <dir> [--engine url] [--source run] [--keep]');
  process.exit(2);
}

let failed = 0;
const check = (name, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${detail ? `: ${detail}` : ''}`);
  if (!ok) failed++;
};

// 0. Resolver unit: replacement between reads, scale parsing.
const UNIT_CPP = String.raw`
#include "sa3-preview-resolver.h"
#include <cstdio>
namespace fs = std::filesystem;
static int fails = 0;
static void check(const std::string & name, bool ok, const std::string & d = "") {
    printf("%s unit: %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(), d.empty() ? "" : ": ", d.c_str());
    if (!ok) fails++;
}
static void put(const fs::path & p, const std::string & base, const char * pad) {
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "sa3-adapter");
    gguf_set_val_str(g, "sa3.adapter_format", "dora-rows-v1");
    gguf_set_val_str(g, "sa3.inference_base_id", base.c_str());
    gguf_set_val_str(g, "test.pad", pad);
    fs::create_directories(p.parent_path());
    gguf_write_to_file(g, p.u8string().c_str(), true);
    gguf_free(g);
}
int main(int, char ** argv) {
    const fs::path root = fs::u8path(argv[1]);
    fs::remove_all(root);
    fs::create_directories(root / "sa3-medium");
    const fs::path dit = root / "dit.gguf";
    { std::ofstream(dit, std::ios::binary) << "stand-in generation DiT"; }
    const std::string good = "stabilityai/stable-audio-3-medium sha256:" + sa3_preview_sha(dit);
    const std::string bad  = "stabilityai/stable-audio-3-medium sha256:" + std::string(64, '0');
    const fs::path step = root / "sa3-medium" / "d" / "a" / "1" / "snapshots" / "step-1.gguf";
    const std::string ad = root.u8string();
    std::string path, sig, err;

    put(step, good, "A");
    const std::string a_sha = sa3_preview_sha(step);
    err = sa3_resolve_snapshot(ad.c_str(), "d/a/1", "1", dit.u8string(), 1.0f, path, sig, [&] { put(step, bad, "B"); });
    check("valid A swapped for wrong-base B after read: A accepted", err.empty(), err);
    check("loader path holds A's bytes", sa3_preview_sha(fs::u8path(path)) == a_sha);
    check("cache key carries A's hash", sig.find(a_sha) != std::string::npos);
    check("loader path is not the step file", fs::u8path(path) != fs::canonical(step));

    put(step, bad, "B");
    err = sa3_resolve_snapshot(ad.c_str(), "d/a/1", "1", dit.u8string(), 1.0f, path, sig, [&] { put(step, good, "A"); });
    check("wrong-base B swapped for valid A after read: B rejected", err.find("different DiT") != std::string::npos, err);

    put(step, good, "A");
    std::string path_a, path_c;
    err = sa3_resolve_snapshot(ad.c_str(), "d/a/1", "1", dit.u8string(), 1.0f, path_a, sig);
    put(step, good, "C");
    const std::string c_sha = sa3_preview_sha(step);
    err += sa3_resolve_snapshot(ad.c_str(), "d/a/1", "1", dit.u8string(), 1.0f, path_c, sig);
    check("rewrite gets a new staged copy", err.empty() && path_c != path_a && sa3_preview_sha(fs::u8path(path_c)) == c_sha, err);
    check("earlier staged copy is untouched", sa3_preview_sha(fs::u8path(path_a)) == a_sha);

    float v = 0;
    for (const char * s : { "nan", "NaN", "inf", "-inf", "1e40", "-1e40", "1x", "", " ", "0.5 ", "0x" })
        check(std::string("scale rejects '") + s + "'", !sa3_parse_snapshot_scale(s, v));
    for (const char * s : { "1", "0.5", "-0.25", "1e-3", "2" })
        check(std::string("scale accepts '") + s + "'", sa3_parse_snapshot_scale(s, v) && std::isfinite(v));
    fs::remove_all(root);
    return fails;
}
`;
function unit() {
  const dir = path.join(repo, '.tmp', 'sa3-preview-unit');
  fs.mkdirSync(dir, { recursive: true });
  const src = path.join(dir, 'unit.cpp'), exe = path.join(dir, 'unit.exe');
  fs.writeFileSync(src, UNIT_CPP);
  const clang = process.platform === 'win32' ? 'C:/Program Files/LLVM/bin/clang++.exe' : 'clang++';
  const lib = path.join(repo, 'engine', 'build', 'ggml', 'src', 'Release', 'ggml-base.lib');
  const c = spawnSync(clang, ['-std=c++17', '-O1', '-D_CRT_SECURE_NO_WARNINGS', `-I${path.join(repo, 'engine', 'src')}`,
    `-I${path.join(repo, 'engine', 'ggml', 'include')}`, src, lib, '-o', exe], { encoding: 'utf8' });
  check('unit test builds', c.status === 0, c.status ? String(c.stderr || c.error).slice(-600) : '');
  if (c.status === 0) {
    const env = { ...process.env, PATH: `${path.join(repo, 'engine', 'build', 'Release')}${path.delimiter}${process.env.PATH}` };
    const r = spawnSync(exe, [path.join(dir, 'root')], { encoding: 'utf8', env });
    process.stdout.write(r.stdout || '');
    check('unit test ran clean', r.status === 0, `exit ${r.status} ${String(r.stderr || '').slice(-300)}`);
  }
  fs.rmSync(dir, { recursive: true, force: true });
}
unit();
if (unitOnly) {
  console.log(failed ? `sa3-preview-proof: FAIL (${failed})` : 'sa3-preview-proof: PASS (unit only)');
  process.exit(failed ? 1 : 0);
}

const [slug, , stamp] = source.split('/');
const runsRoot = path.join(adapters, 'sa3-medium');
const srcDir = path.join(runsRoot, ...source.split('/'));
const fixRun = `${slug}/sa3-preview-proof/${stamp}`;
const fixDir = path.join(runsRoot, ...fixRun.split('/'));
const escRun = `${slug}/sa3-preview-escape/${stamp}`;
const escLink = path.join(runsRoot, ...escRun.split('/'));
const outside = path.join(repo, '.tmp', 'sa3-preview-proof-outside');
const sha = f => createHash('sha256').update(fs.readFileSync(f)).digest('hex');

function cleanup() {
  if (fs.existsSync(escLink)) fs.rmdirSync(escLink); // junction only, never its target
  fs.rmSync(path.dirname(escLink), { recursive: true, force: true });
  fs.rmSync(outside, { recursive: true, force: true });
  fs.rmSync(path.dirname(fixDir), { recursive: true, force: true });
}

// Minimal GGUF v3 with string KVs only (no tensors): enough for the metadata gate.
function writeGguf(file, kv) {
  const parts = [Buffer.from('GGUF'), u32(3), u64(0), u64(Object.keys(kv).length)];
  for (const [k, v] of Object.entries(kv)) parts.push(str(k), u32(8), str(v));
  fs.writeFileSync(file, Buffer.concat(parts));
  function u32(n) { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b; }
  function u64(n) { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(n)); return b; }
  function str(s) { const b = Buffer.from(s, 'utf8'); return Buffer.concat([u64(b.length), b]); }
}

const manifest = JSON.parse(fs.readFileSync(path.join(srcDir, 'train-manifest.json'), 'utf8'));
const { token_ids: tokens, n_tokens: nTokens } = manifest.samples[0];

async function synth(extra) {
  const q = new URLSearchParams({ duration: '8', steps: '8', seed: '1234', out_fmt: 'f32',
    tokens: tokens.join(','), n_tokens: String(nTokens), ...extra });
  const res = await fetch(`${engine}/sa3-synth?${q}`, { method: 'POST' });
  const body = Buffer.from(await res.arrayBuffer());
  return { status: res.status, body, text: res.ok ? '' : body.toString('utf8') };
}

function pcm(wav) {
  const i = wav.indexOf('data', 12);
  const b = wav.subarray(i + 8);
  return new Float32Array(b.buffer, b.byteOffset, Math.floor(b.length / 4));
}

function diff(a, b) {
  const x = pcm(a.body), y = pcm(b.body);
  let max = 0, sq = 0;
  for (let i = 0; i < Math.min(x.length, y.length); i++) { const d = Math.abs(x[i] - y[i]); max = Math.max(max, d); sq += d * d; }
  return { max, rms: Math.sqrt(sq / Math.max(1, x.length)), sameLength: x.length === y.length };
}

async function expect400(name, extra, needle) {
  const r = await synth(extra);
  check(name, r.status === 400 && r.text.includes(needle), `${r.status} ${r.text.slice(0, 140)}`);
}

async function catalogue() {
  const r = await fetch(`${engine}/sa3/props`);
  return (await r.json()).adapters;
}

try {
  cleanup();
  const catBefore = await catalogue();

  // 1. Real trainer snapshot.
  const saved = JSON.parse(fs.readFileSync(path.join(srcDir, 'run.json'), 'utf8'));
  fs.mkdirSync(fixDir, { recursive: true });
  for (const name of ['train-manifest.json', 'dataset', 'cache']) {
    if (fs.existsSync(path.join(srcDir, name))) fs.cpSync(path.join(srcDir, name), path.join(fixDir, name), { recursive: true });
  }
  const step = 3;
  const exe = path.join(repo, 'engine', 'build', 'Release', process.platform === 'win32' ? 'ace-train.exe' : 'ace-train');
  const trainArgs = ['sa3-train', '--models', models, '--base-dit', path.join(models, 'sa3-dit-base-BF16.gguf'),
    '--inference-dit', path.join(models, 'sa3-dit-BF16.gguf'), '--manifest', path.join(fixDir, 'train-manifest.json'),
    '--out', fixDir, '--steps', '4', '--rank', String(saved.rank), '--alpha', String(saved.alpha),
    '--lr', '0.001', '--duration', String(saved.duration), '--seed', String(saved.seed),
    '--checkpoint-every', '4', '--export-every', '3', '--bwd', 'mm', '--jsonl'];
  const t = spawnSync(exe, trainArgs, { encoding: 'utf8', timeout: 20 * 60_000, windowsHide: true });
  const snap = path.join(fixDir, 'snapshots', `step-${step}.gguf`);
  check('trainer wrote snapshot', t.status === 0 && fs.existsSync(snap),
    `exit ${t.status} ${path.relative(runsRoot, snap)}${t.status ? ` ${String(t.stderr).slice(-400)}` : ''}`);
  if (!fs.existsSync(snap)) throw new Error('no snapshot, stopping');

  // 2. Fixed-seed effect.
  const base1 = await synth({});
  const base2 = await synth({});
  const s1 = await synth({ snapshot_run: fixRun, snapshot_step: String(step) });
  check('base render ok', base1.status === 200 && base2.status === 200, `${base1.status}/${base2.status}`);
  const d0 = diff(base1, base2);
  check('base is deterministic at fixed seed', d0.max === 0, JSON.stringify(d0));
  check('snapshot render ok', s1.status === 200, s1.text.slice(0, 200));
  const d1 = diff(base1, s1);
  check('snapshot changes the render', d1.sameLength && d1.max > 0, JSON.stringify(d1));
  const s1b = await synth({ snapshot_run: fixRun, snapshot_step: String(step) });
  check('snapshot render repeats (cache hit)', diff(s1, s1b).max === 0);

  // 3. Rejections.
  await expect400('traversal in run', { snapshot_run: `../${fixRun}`, snapshot_step: String(step) }, 'Invalid snapshot_run');
  await expect400('backslash in run', { snapshot_run: fixRun.replace(/\//g, '\\'), snapshot_step: String(step) }, 'Invalid snapshot_run');
  await expect400('absolute run', { snapshot_run: srcDir, snapshot_step: String(step) }, 'Invalid snapshot_run');
  await expect400('traversal in step', { snapshot_run: fixRun, snapshot_step: `${step}/../1` }, 'Invalid snapshot_step');
  await expect400('step zero', { snapshot_run: fixRun, snapshot_step: '0' }, 'Invalid snapshot_step');
  await expect400('missing step', { snapshot_run: fixRun, snapshot_step: '999999' }, 'not found');
  await expect400('run without step', { snapshot_run: fixRun }, 'Invalid snapshot_step');
  for (const v of ['nan', 'inf', '-inf', '1e40', '1x', ''])
    await expect400(`snapshot_scale '${v}'`, { snapshot_run: fixRun, snapshot_step: String(step), snapshot_scale: v }, 'Invalid snapshot_scale');

  fs.mkdirSync(path.join(outside, 'snapshots'), { recursive: true });
  fs.copyFileSync(snap, path.join(outside, 'snapshots', `step-${step}.gguf`));
  fs.mkdirSync(path.dirname(escLink), { recursive: true });
  fs.symlinkSync(outside, escLink, 'junction');
  await expect400('junction escape', { snapshot_run: escRun, snapshot_step: String(step) }, 'outside the training runs');

  const q8 = fs.readdirSync(models).find(f => /^sa3-dit-(Q|F16)[^/]*\.gguf$/.test(f));
  if (q8) await expect400(`wrong base (${q8})`, { snapshot_run: fixRun, snapshot_step: String(step), synth_model: q8 }, 'different DiT');
  else check('wrong base (other DiT)', false, 'no second sa3-dit-*.gguf installed');
  writeGguf(path.join(fixDir, 'snapshots', 'step-900001.gguf'), { 'general.architecture': 'sa3-adapter',
    'sa3.adapter_format': 'dora-rows-v1', 'sa3.inference_base_id': `stabilityai/stable-audio-3-medium sha256:${'0'.repeat(64)}` });
  await expect400('wrong base (forged id)', { snapshot_run: fixRun, snapshot_step: '900001' }, 'different DiT');
  writeGguf(path.join(fixDir, 'snapshots', 'step-900002.gguf'), { 'general.architecture': 'sa3-lora' });
  await expect400('not an inference export', { snapshot_run: fixRun, snapshot_step: '900002' }, 'not a dora-rows-v1');

  // 4. Stale cache: same path, new bytes.
  const srcExport = path.join(srcDir, 'adapter.gguf');
  check('replacement differs from snapshot', sha(srcExport) !== sha(snap));
  fs.copyFileSync(srcExport, snap);
  const s2 = await synth({ snapshot_run: fixRun, snapshot_step: String(step) });
  const d2 = diff(s1, s2);
  check('rewritten step file is not served from cache', s2.status === 200 && d2.max > 0, JSON.stringify(d2));
  const pubDir = path.join(models, 'sa3-adapters');
  const twin = (fs.existsSync(pubDir) ? fs.readdirSync(pubDir) : [])
    .find(f => f.endsWith('.gguf') && fs.statSync(path.join(pubDir, f)).size === fs.statSync(srcExport).size &&
      sha(path.join(pubDir, f)) === sha(srcExport));
  if (twin) {
    const pub = await synth({ adapters: twin.slice(0, -5) });
    check(`snapshot path matches public adapter ${twin}`, pub.status === 200 && diff(s2, pub).max === 0, JSON.stringify(diff(s2, pub)));
  } else {
    console.log('SKIP public-twin equality: source export is not published byte-identically');
  }

  // 5. Public surface unchanged.
  await expect400('public parser: traversal', { adapters: '../x' }, 'Invalid adapter name');
  await expect400('public parser: missing', { adapters: 'no-such-adapter-zz' }, 'SA3 adapter not found');
  await expect400('public parser: snapshot by name', { adapters: `step-${step}` }, 'SA3 adapter not found');
  const catAfter = await catalogue();
  check('catalogue unchanged, no snapshots', JSON.stringify(catBefore) === JSON.stringify(catAfter) &&
    !catAfter.some(n => /^step-/.test(n)), `${catAfter.length} adapters`);
} catch (e) {
  check('proof ran to completion', false, e.message);
} finally {
  if (!keep) cleanup();
}
console.log(failed ? `sa3-preview-proof: FAIL (${failed})` : 'sa3-preview-proof: PASS');
process.exit(failed ? 1 : 0);
