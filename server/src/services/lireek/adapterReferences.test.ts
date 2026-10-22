import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import Database from 'better-sqlite3';
import { updateAdapterReferences } from './adapterReferences.js';

function fixture(t: test.TestContext) {
  const scratch = path.resolve('.scratch');
  fs.mkdirSync(scratch, { recursive: true });
  const root = fs.mkdtempSync(path.join(scratch, 'adapter-references-'));
  const db = new Database(':memory:');
  t.after(() => { db.close(); fs.rmSync(root, { recursive: true, force: true }); });
  db.exec(`CREATE TABLE artists (id INTEGER PRIMARY KEY, name TEXT);
    CREATE TABLE lyrics_sets (id INTEGER PRIMARY KEY, artist_id INTEGER, album TEXT);
    CREATE TABLE album_presets (lyrics_set_id INTEGER PRIMARY KEY,
      adapter_path TEXT, lm_adapter_path TEXT, mm3_adapter_path TEXT,
      yue2_ar_adapter_path TEXT, yue2_nar_adapter_path TEXT,
      adapter_scale REAL DEFAULT 0.7, adapter_group_scales TEXT DEFAULT '{"mlp":0.5}',
      lm_adapter_scale REAL DEFAULT 0.6, reference_track_path TEXT DEFAULT 'reference.wav');
    INSERT INTO artists VALUES (1, 'Example artist');`);
  const file = (name: string) => {
    const target = path.join(root, name);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, 'weights');
    return target;
  };
  const preset = (id: number, fields: Record<string, string | null>) => {
    db.prepare('INSERT INTO lyrics_sets VALUES (?, 1, ?)').run(id, `Example preset ${id}`);
    const keys = Object.keys(fields);
    db.prepare(`INSERT INTO album_presets (lyrics_set_id, ${keys.join(',')}) VALUES (?, ${keys.map(() => '?').join(',')})`)
      .run(id, ...Object.values(fields));
  };
  const read = (id: number) => db.prepare('SELECT * FROM album_presets WHERE lyrics_set_id = ?').get(id) as Record<string, unknown>;
  return { root, db, file, preset, read };
}

test('deleted adapters are cleared, reported per preset, and load without phantom paths', t => {
  const { root, db, file, preset, read } = fixture(t);
  const gone = file('deleted.safetensors');
  const ar = file('yue2/ar.safetensors');
  const nar = file('yue2/nar.safetensors');
  preset(1, { adapter_path: gone, lm_adapter_path: path.join(root, 'missing-planner'),
    mm3_adapter_path: 'missing/run.safetensors', yue2_ar_adapter_path: ar, yue2_nar_adapter_path: nar });
  fs.unlinkSync(gone);
  fs.unlinkSync(nar);
  // Reproduce the stale saved preset before the maintenance command.
  assert.equal(read(1).adapter_path, gone);
  const before = read(1);
  const result = updateAdapterReferences(db, root);
  assert.equal(result.scanned, 1);
  assert.equal(result.updated, 1);
  assert.equal(result.changes.length, 4);
  assert.deepEqual(result.changes.map(c => c.field), ['adapter_path', 'lm_adapter_path', 'mm3_adapter_path', 'yue2_nar_adapter_path']);
  assert.ok(result.changes.every(c => c.presetName === 'Example artist / Example preset 1'));
  assert.equal(result.changes[0].adapter, gone);
  const loaded = read(1);
  assert.equal(loaded.yue2_ar_adapter_path, ar);
  assert.deepEqual(loaded, { ...before, adapter_path: null, lm_adapter_path: null, mm3_adapter_path: null, yue2_nar_adapter_path: null });
  assert.deepEqual(updateAdapterReferences(db, root).changes, []);
});

test('live files, registry names, PEFT and LoKR directories remain byte-for-byte unchanged', t => {
  const { root, db, file, preset, read } = fixture(t);
  file('sound/adapter_model.safetensors');
  file('lm/planner/lokr_weights.safetensors');
  file('mm3-lm-adapters/run/model.safetensors');
  const ar = file('yue2/ar.safetensors');
  const nar = file('yue2/nar.safetensors');
  preset(1, { adapter_path: 'sound', lm_adapter_path: 'planner', mm3_adapter_path: 'run/model.safetensors',
    yue2_ar_adapter_path: ar, yue2_nar_adapter_path: nar });
  preset(2, { adapter_path: file('flat.safetensors'), lm_adapter_path: path.dirname(file('lm-4b/run/adapter_model.safetensors')) });
  preset(3, { adapter_path: '', lm_adapter_path: null });
  const before = [read(1), read(2), read(3)];
  assert.deepEqual(updateAdapterReferences(db, root), { scanned: 3, updated: 0, changes: [] });
  assert.deepEqual([read(1), read(2), read(3)], before);
});

test('empty adapter directories count as missing weights and reports include every changed preset', t => {
  const { root, db, preset, read } = fixture(t);
  const empty = path.join(root, 'empty');
  fs.mkdirSync(empty);
  preset(1, { adapter_path: empty });
  preset(2, { lm_adapter_path: empty });
  const result = updateAdapterReferences(db, root);
  assert.equal(result.updated, 2);
  assert.deepEqual(result.changes.map(c => c.lyricsSetId), [1, 2]);
  assert.equal(read(1).adapter_path, null);
  assert.equal(read(2).lm_adapter_path, null);
});

test('unavailable root and write failures leave all presets intact', t => {
  const { root, db, preset, read } = fixture(t);
  preset(1, { adapter_path: 'deleted' });
  const before = read(1);
  assert.throws(() => updateAdapterReferences(db, path.join(root, 'unavailable')));
  assert.deepEqual(read(1), before);
  preset(2, { adapter_path: 'also-deleted' });
  db.exec(`CREATE TRIGGER reject_update BEFORE UPDATE ON album_presets
    WHEN OLD.lyrics_set_id = 2 BEGIN SELECT RAISE(ABORT, 'test write failure'); END;`);
  assert.throws(() => updateAdapterReferences(db, root), /test write failure/);
  assert.deepEqual(read(1), before);
  assert.equal(read(2).adapter_path, 'also-deleted');
});
