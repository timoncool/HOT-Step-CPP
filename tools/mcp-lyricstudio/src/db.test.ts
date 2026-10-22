import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import Database from 'better-sqlite3';
import { getDb, initDb, resolveDbPath } from './db.js';
import { pointerLocation } from '../../../server/src/dataroot/locate.js';
import { newPending } from '../../../server/src/dataroot/paths.js';

const packageDir = fileURLToPath(new URL('../', import.meta.url));

function scratch(t: test.TestContext): string {
  const dir = mkdtempSync(path.join(packageDir, '.db-test-'));
  t.after(() => {
    assert.equal(path.dirname(path.resolve(dir)), path.resolve(packageDir));
    assert.ok(path.basename(dir).startsWith('.db-test-'));
    rmSync(dir, { recursive: true, force: true });
  });
  return dir;
}

test('missing HOTSTEP_DB exits with guidance and creates no database', t => {
  const dir = scratch(t);
  const dbPath = path.join(dir, 'missing.db');
  const result = spawnSync(process.execPath, ['--import', 'tsx', 'src/index.ts'], {
    cwd: packageDir,
    env: { ...process.env, HOTSTEP_DB: dbPath },
    encoding: 'utf8',
    input: '',
    timeout: 15_000,
  });
  assert.ifError(result.error);
  assert.equal(result.status, 1);
  assert.ok(result.stderr.includes(dbPath));
  assert.match(result.stderr, /not found/i);
  assert.match(result.stderr, /HOTSTEP_DB/);
  assert.equal(existsSync(dbPath), false);
  assert.deepEqual(readdirSync(dir), []);
});

function pointerFixture(t: test.TestContext) {
  const dir = scratch(t);
  const root = path.join(dir, 'data');
  const profile = path.join(dir, 'profile');
  mkdirSync(profile);
  mkdirSync(path.join(root, 'library'), { recursive: true });
  const env: NodeJS.ProcessEnv = {
    ...process.env, HOTSTEP_DB: '', HOT_STEP_ROOT: '', HOT_STEP_DATA_ROOT: '',
    HOT_STEP_PROFILE_DIR: profile, DATA_DIR: '',
  };
  const pointerPath = pointerLocation(env).pointerPath;
  writeFileSync(pointerPath, JSON.stringify({ version: 1, root, pending: null }));
  return { root, env, pointerPath, dbPath: path.join(root, 'library', 'hotstep.db') };
}

function startup(env: NodeJS.ProcessEnv) {
  return spawnSync(process.execPath, ['--import', 'tsx', 'src/index.ts'], {
    cwd: packageDir, env, encoding: 'utf8', input: '', timeout: 15_000,
  });
}

test('HOTSTEP_DB overrides even an unreadable data-folder pointer', t => {
  const f = pointerFixture(t);
  writeFileSync(f.pointerPath, 'invalid json');
  const fixture = new Database(f.dbPath);
  fixture.close();
  f.env.HOTSTEP_DB = f.dbPath;
  assert.deepEqual(resolveDbPath(f.env), { dbPath: f.dbPath, source: 'HOTSTEP_DB override' });
  const result = startup(f.env);
  assert.ifError(result.error);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stderr, /Server started/);
});

test('temporary pointer opens its library and follows DATA_DIR child overrides', t => {
  const f = pointerFixture(t);
  assert.equal(resolveDbPath(f.env).dbPath, f.dbPath);
  assert.ok(resolveDbPath(f.env).source.includes(f.pointerPath));
  const fixture = new Database(f.dbPath);
  fixture.close();
  const result = startup(f.env);
  assert.ifError(result.error);
  assert.equal(result.status, 0, result.stderr);
  assert.ok(result.stderr.includes(f.dbPath));
  assert.match(result.stderr, /Server started/);
  assert.equal(resolveDbPath({ ...f.env, DATA_DIR: 'custom-library' }).dbPath,
    path.join(f.root, 'custom-library', 'hotstep.db'));
});

test('missing pointer-selected database exits 1 and creates nothing', t => {
  const f = pointerFixture(t);
  const before = readdirSync(f.root, { recursive: true });
  const result = startup(f.env);
  assert.ifError(result.error);
  assert.equal(result.status, 1);
  assert.ok(result.stderr.includes(f.dbPath));
  assert.ok(result.stderr.includes(f.pointerPath));
  assert.match(result.stderr, /Database not found/);
  assert.deepEqual(readdirSync(f.root, { recursive: true }), before);
});

test('pending move exits 1 without opening an existing DB or creating files', t => {
  const f = pointerFixture(t);
  // A non-database file makes an accidental SQLite open observable.
  writeFileSync(f.dbPath, 'not a sqlite database');
  const pending = newPending('move', f.root, path.join(f.root, 'destination'));
  pending.error = 'copy unfinished';
  writeFileSync(f.pointerPath, JSON.stringify({ version: 1, root: f.root, pending }));
  const before = readdirSync(f.root, { recursive: true });
  const result = startup(f.env);
  assert.ifError(result.error);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Pending data-folder move/);
  assert.match(result.stderr, /copy unfinished/);
  assert.ok(result.stderr.includes(f.pointerPath));
  assert.ok(result.stderr.includes(pending.destination));
  assert.doesNotMatch(result.stderr, /Opening database/);
  assert.deepEqual(readdirSync(f.root, { recursive: true }), before);
});

test('invalid pointer exits 1 with the reason and creates nothing', t => {
  const f = pointerFixture(t);
  writeFileSync(f.pointerPath, 'invalid json');
  const before = readdirSync(f.root, { recursive: true });
  const result = startup(f.env);
  assert.ifError(result.error);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Cannot read data-folder pointer/);
  assert.ok(result.stderr.includes(f.pointerPath));
  assert.doesNotMatch(result.stderr, /Opening database/);
  assert.deepEqual(readdirSync(f.root, { recursive: true }), before);
});

test('existing HOTSTEP_DB opens with its data and connection pragmas', t => {
  const dir = scratch(t);
  const dbPath = path.join(dir, 'existing.db');
  const fixture = new Database(dbPath);
  fixture.exec('CREATE TABLE marker (value TEXT); INSERT INTO marker VALUES (\'existing\')');
  fixture.close();
  const previous = process.env.HOTSTEP_DB;
  t.after(() => {
    if (previous === undefined) delete process.env.HOTSTEP_DB;
    else process.env.HOTSTEP_DB = previous;
  });
  process.env.HOTSTEP_DB = dbPath;
  initDb();
  try {
    assert.deepEqual(getDb().prepare('SELECT value FROM marker').get(), { value: 'existing' });
    assert.equal(getDb().pragma('journal_mode', { simple: true }), 'wal');
    assert.equal(getDb().pragma('foreign_keys', { simple: true }), 1);
  } finally {
    getDb().close();
  }
});
