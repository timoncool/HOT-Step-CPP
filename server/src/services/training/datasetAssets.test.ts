import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { readDatasetAssets, readSa3DatasetAssets } from './datasetAssets.js';
import { SA3_AUDIO } from './sa3Stems.js';
import type { TrainingDatasetRow } from './types.js';

const hash = (data: string | Buffer) => createHash('sha256').update(data).digest('hex');
const baseHash = 'a'.repeat(64);

test('dataset responses include the frozen SA3 block even before training', () => {
  const assets = readDatasetAssets({ id: 'sa3-asset-missing', slug: 'sa3-asset-missing', labeledCount: 0,
    status: 'draft', builtAt: null } as unknown as TrainingDatasetRow);
  assert.deepEqual(assets.sa3, { audioPrepared: false, preparedClips: 0, encodingCacheValid: false,
    encodedClips: 0, snapshots: 0, adapter: null });
});

function gguf(file: string) {
  const u32 = (n: number) => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b; };
  const u64 = (n: number) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(n)); return b; };
  const str = (s: string) => Buffer.concat([u64(Buffer.byteLength(s)), Buffer.from(s)]);
  const kv = (k: string, v: string) => Buffer.concat([str(k), u32(8), str(v)]);
  const tensor = (name: string, dims: number[], off: number) => Buffer.concat([str(`model.preprocess_conv.weight.${name}`),
    u32(dims.length), ...dims.map(u64), u32(0), u64(off)]);
  const header = Buffer.concat([Buffer.from('GGUF'), u32(3), u64(3), u64(3),
    kv('general.architecture', 'sa3-adapter'), kv('sa3.adapter_format', 'dora-rows-v1'),
    kv('sa3.inference_base_id', `stabilityai/stable-audio-3-medium sha256:${baseHash}`),
    tensor('lora_a', [2, 1], 0), tensor('lora_b', [1, 2], 32), tensor('row_scale', [2], 64)]);
  const data = Buffer.alloc(72); data.writeFloatLE(1, 64); data.writeFloatLE(1, 68);
  fs.writeFileSync(file, Buffer.concat([header, Buffer.alloc((32 - header.length % 32) % 32), data]));
}

function fixture(t: test.TestContext) {
  const root = fs.mkdtempSync(path.join(process.cwd(), '.test-sa3-assets-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const roots = { runs: path.join(root, 'runs'), models: path.join(root, 'models') };
  const sourceDir = path.join(root, 'source'), dir = path.join(roots.runs, 'adapter', '2026-01-01');
  const prepared = path.join(dir, 'dataset'), snapshots = path.join(dir, 'snapshots');
  for (const p of [sourceDir, prepared, snapshots, path.join(roots.models, 'supersep'), path.join(roots.models, 'sa3-adapters')]) {
    fs.mkdirSync(p, { recursive: true });
  }
  const source = path.join(sourceDir, 'clip.wav');
  fs.writeFileSync(source, 'original audio');
  fs.writeFileSync(path.join(prepared, '000001.wav'), 'prepared audio');
  fs.writeFileSync(path.join(prepared, '000001.txt'), 'warm pad\n');
  const modelFiles = ['bs_leap_xe_voc-F32.gguf', 'bs_leap_xe_inst-F32.gguf'].map(f => path.join(roots.models, 'supersep', f));
  modelFiles.forEach(f => fs.writeFileSync(f, 'weights'));
  const sourceHash = hash('original audio'), modelHashes = modelFiles.map(f => hash(fs.readFileSync(f)));
  const manifest = { version: 1, sample_rate: 44100, channels: 2, separation_level: 5, samples: [{
    source_path: source, source_sha256: sourceHash, audio_path: '000001.wav', audio_sha256: hash('prepared audio'),
    caption_path: '000001.txt', prompt: 'warm pad', seconds_total: 1, model_sha256: modelHashes,
    cache_fingerprint: hash(JSON.stringify({ sourceHash, modelHashes, settings: SA3_AUDIO })),
  }] };
  const manifestFile = path.join(prepared, 'manifest.json');
  fs.writeFileSync(manifestFile, JSON.stringify(manifest));
  const run = { datasetId: 'fixture', published: 'published', inferenceSha256: baseHash, status: 'done' };
  const saveRun = () => fs.writeFileSync(path.join(dir, 'run.json'), JSON.stringify(run));
  saveRun();
  const snapshot = path.join(snapshots, 'step-4.gguf'), published = path.join(roots.models, 'sa3-adapters', 'published.gguf');
  gguf(snapshot); gguf(published);
  const ds = { id: 'fixture', slug: 'sa3-asset-fixture', sourceDir, recursive: false };
  return { roots, dir, source, sourceDir, prepared, modelFiles, manifest, manifestFile, snapshot, published, run, saveRun,
    read: () => readSa3DatasetAssets(ds, roots) };
}

test('SA3 chips verify preparation, snapshots and publication independently of completed runs', t => {
  const f = fixture(t), assets = f.read();
  assert.equal(assets.audioPrepared, true);
  assert.equal(assets.preparedClips, 1);
  assert.equal(assets.snapshots, 1);
  assert.equal(assets.adapter?.name, 'published');
  assert.equal(assets.adapter?.run, 'adapter/2026-01-01');
  fs.rmSync(f.snapshot); fs.rmSync(f.published); fs.rmSync(f.manifestFile);
  assert.deepEqual(f.read(), { audioPrepared: false, preparedClips: 0, snapshots: 0,
    encodingCacheValid: false, encodedClips: 0, adapter: null });
});

for (const change of ['source', 'derived', 'model', 'selection', 'caption', 'fingerprint'] as const) {
  test(`SA3 preparation becomes unready after ${change} changes`, t => {
    const f = fixture(t);
    assert.equal(f.read().audioPrepared, true);
    if (change === 'source') fs.writeFileSync(f.source, 'replaced audio');
    if (change === 'derived') fs.writeFileSync(path.join(f.prepared, '000001.wav'), 'tampered audio');
    if (change === 'model') fs.writeFileSync(f.modelFiles[0], 'changed');
    if (change === 'selection') fs.writeFileSync(path.join(f.sourceDir, 'new.wav'), 'new clip');
    if (change === 'caption') fs.writeFileSync(path.join(f.prepared, '000001.txt'), 'changed prompt');
    if (change === 'fingerprint') {
      f.manifest.samples[0].cache_fingerprint = 'invalid';
      fs.writeFileSync(f.manifestFile, JSON.stringify(f.manifest));
    }
    assert.equal(f.read().audioPrepared, false);
    assert.equal(f.read().preparedClips, 0);
    assert.equal(f.read().snapshots, 1);
    assert.ok(f.read().adapter);
  });
}

test('SA3 cache files do not prove the current trainer fingerprint', t => {
  const f = fixture(t);
  fs.mkdirSync(path.join(f.dir, 'cache'));
  fs.writeFileSync(path.join(f.dir, 'cache', 'old.sa3c'), 'SA3C0001');
  assert.equal(f.read().encodingCacheValid, false);
  assert.equal(f.read().encodedClips, 0);
});

test('SA3 rejects corrupt exports and wrong dataset or inference base', t => {
  const f = fixture(t);
  fs.writeFileSync(f.snapshot, 'GGUF truncated');
  assert.equal(f.read().snapshots, 0);
  f.run.inferenceSha256 = 'b'.repeat(64); f.saveRun();
  assert.equal(f.read().adapter, null);
  f.run.datasetId = 'other'; f.saveRun();
  assert.equal(f.read().audioPrepared, false);
});

test('SA3 rejects preparation traversal and publication path injection', t => {
  const f = fixture(t);
  f.manifest.samples[0].audio_path = '../../outside.wav';
  fs.writeFileSync(f.manifestFile, JSON.stringify(f.manifest));
  f.run.published = '../../outside'; f.saveRun();
  assert.equal(f.read().audioPrepared, false);
  assert.equal(f.read().adapter, null);
});

test('SA3 missing roots and legacy incomplete records remain unready', t => {
  const f = fixture(t);
  fs.writeFileSync(path.join(f.dir, 'run.json'), '{');
  assert.equal(f.read().audioPrepared, false);
  assert.equal(f.read().adapter, null);
  fs.rmSync(f.roots.runs, { recursive: true });
  assert.equal(f.read().snapshots, 0);
});

test('SA3 invalidates cached export checks after file replacement', t => {
  const f = fixture(t);
  assert.equal(f.read().snapshots, 1);
  assert.ok(f.read().adapter);
  fs.writeFileSync(f.snapshot, 'corrupt');
  fs.writeFileSync(f.published, 'corrupt');
  assert.equal(f.read().snapshots, 0);
  assert.equal(f.read().adapter, null);
});

test('SA3 rejects snapshot directory junctions that escape the runs root', t => {
  const f = fixture(t);
  const outside = path.join(f.sourceDir, 'exports');
  fs.mkdirSync(outside);
  gguf(path.join(outside, 'step-8.gguf'));
  fs.rmSync(path.join(f.dir, 'snapshots'), { recursive: true });
  fs.symlinkSync(outside, path.join(f.dir, 'snapshots'), 'junction');
  assert.equal(f.read().snapshots, 0);
});
