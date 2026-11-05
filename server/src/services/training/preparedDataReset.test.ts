import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { clearPreparedCaches, listPreparedCaches } from './preparedDataReset.js';
import { datasetDir } from './paths.js';
import { tensorsRoot } from './aceTrain.js';

test('HeartMuLa reset measures and deletes only its prepare folder', () => {
  const slug = `heartmula-reset-${process.pid}-${Date.now()}`;
  const root = datasetDir(slug);
  const source = path.join(root, 'source');
  const owned = path.join(root, 'heartmula', 'prepare');
  const kept = [path.join(root, 'yue2-latents'), path.join(root, 'mm3-codes'), tensorsRoot(slug), source,
    path.join(root, 'heartmula', 'other')];
  try {
    for (const folder of [owned, ...kept]) {
      fs.mkdirSync(folder, { recursive: true }); fs.writeFileSync(path.join(folder, 'sample.bin'), 'four');
    }
    const caches = listPreparedCaches(slug, source, { target: 'heartmula' });
    assert.equal(caches.length, 1);
    assert.equal(caches[0].path, owned);
    assert.equal(caches[0].files, 1); assert.equal(caches[0].bytes, 4);
    assert.deepEqual(clearPreparedCaches(slug, source, { target: 'heartmula', includeYue2Core: true }), caches);
    assert.equal(fs.existsSync(owned), false);
    for (const folder of kept) assert.equal(fs.readFileSync(path.join(folder, 'sample.bin'), 'utf8'), 'four');
  } finally {
    fs.rmSync(root, { recursive: true, force: true }); fs.rmSync(tensorsRoot(slug), { recursive: true, force: true });
  }
});

test('HeartMuLa reset refuses source audio inside its preparation folder', () => {
  const slug = `heartmula-reset-guard-${process.pid}-${Date.now()}`;
  const root = datasetDir(slug);
  const source = path.join(root, 'heartmula', 'prepare', 'source');
  try {
    fs.mkdirSync(source, { recursive: true }); fs.writeFileSync(path.join(source, 'song.wav'), 'source');
    assert.throws(() => clearPreparedCaches(slug, source, { target: 'heartmula' }), /source files/);
    assert.equal(fs.readFileSync(path.join(source, 'song.wav'), 'utf8'), 'source');
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('prepared-data reset removes generated caches but preserves labels and source files', () => {
  const slug = `reset-fixture-${process.pid}-${Date.now()}`;
  const root = datasetDir(slug);
  const tensor = tensorsRoot(slug);
  const source = fs.mkdtempSync(path.join(os.tmpdir(), 'hotstep-reset-source-'));
  try {
    fs.mkdirSync(path.join(root, 'labels'), { recursive: true });
    fs.writeFileSync(path.join(root, 'labels', 'edited.json'), '{}');
    fs.mkdirSync(path.join(root, 'yue2-latents', 'codes'), { recursive: true });
    fs.writeFileSync(path.join(root, 'yue2-latents', 'codes', 'track.codes'), 'data');
    fs.mkdirSync(tensor, { recursive: true });
    fs.writeFileSync(path.join(tensor, 'tensor.bin'), 'data');
    fs.writeFileSync(path.join(source, 'track.wav'), 'source');
    fs.writeFileSync(path.join(source, 'track.abc'), 'X:1\nK:C\nC');
    // The batch reset removes the same YuE2 cache entries returned by this
    // inventory. Its cache paths must never include the source sidecar.
    assert.equal(listPreparedCaches(slug, source).some(c => c.path === path.join(source, 'track.abc')), false);
    // Default: YuE2's core prepared data is kept, the rest goes.
    assert.equal(listPreparedCaches(slug, source).length, 1);
    assert.equal(clearPreparedCaches(slug, source).length, 1);
    assert.equal(fs.existsSync(path.join(root, 'yue2-latents', 'codes', 'track.codes')), true);
    // Asked for: the core goes too.
    assert.equal(clearPreparedCaches(slug, source, { includeYue2Core: true }).length, 1);
    assert.equal(fs.existsSync(path.join(root, 'yue2-latents')), false);
    assert.equal(fs.existsSync(tensor), false);
    assert.equal(fs.existsSync(path.join(root, 'labels', 'edited.json')), true);
    assert.equal(fs.existsSync(path.join(source, 'track.wav')), true);
    assert.equal(fs.readFileSync(path.join(source, 'track.abc'), 'utf8'), 'X:1\nK:C\nC');
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
    fs.rmSync(tensor, { recursive: true, force: true });
    fs.rmSync(source, { recursive: true, force: true });
  }
});

test('prepared-data reset rejects a source directory inside a cache', () => {
  const slug = `reset-guard-${process.pid}-${Date.now()}`;
  const root = datasetDir(slug);
  const source = path.join(root, 'yue2-latents', 'source');
  try {
    fs.mkdirSync(source, { recursive: true });
    fs.writeFileSync(path.join(source, 'track.wav'), 'source');
    assert.throws(() => clearPreparedCaches(slug, source), /source files/);
    assert.equal(fs.existsSync(path.join(source, 'track.wav')), true);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('the batch YuE2 cache selection leaves the source ABC beside its audio', () => {
  const slug = `batch-reset-${process.pid}-${Date.now()}`;
  const root = datasetDir(slug);
  const source = fs.mkdtempSync(path.join(os.tmpdir(), 'hotstep-batch-source-'));
  try {
    fs.mkdirSync(path.join(root, 'yue2-stems'), { recursive: true });
    fs.writeFileSync(path.join(root, 'yue2-stems', 'stem.wav'), 'derived');
    fs.writeFileSync(path.join(source, 'track.wav'), 'source');
    fs.writeFileSync(path.join(source, 'track.abc'), 'X:1\nK:C\nC');
    // Same guarded inventory and filter used by the batch reset.
    for (const cache of listPreparedCaches(slug, source)) {
      if (cache.name.startsWith('yue2-')) fs.rmSync(cache.path, { recursive: true, force: false });
    }
    assert.equal(fs.existsSync(path.join(root, 'yue2-stems')), false);
    assert.equal(fs.readFileSync(path.join(source, 'track.abc'), 'utf8'), 'X:1\nK:C\nC');
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
    fs.rmSync(source, { recursive: true, force: true });
  }
});
