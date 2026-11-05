// Two connections from two processes on one hotstep.db (dev and prod copies),
// and the verified backup.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import Database from 'better-sqlite3';
import { applyConnectionPragmas, BUSY_TIMEOUT_MS } from './database.js';
import { backupDatabase } from './backup.js';

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'shared-db-'));
test.after(() => fs.rmSync(dir, { recursive: true, force: true }));

test('a write waits for the other process to release its write lock instead of failing', async () => {
  const file = path.join(dir, 'contention.db');
  const db = new Database(file);
  applyConnectionPragmas(db);
  assert.equal(db.pragma('busy_timeout', { simple: true }), BUSY_TIMEOUT_MS);
  db.exec('CREATE TABLE t (v INTEGER)');

  // The other process takes the write lock and holds it for 800 ms.
  const holder = spawn(process.execPath, ['--input-type=module', '-e', `
    import Database from 'better-sqlite3';
    const db = new Database(${JSON.stringify(file)});
    db.exec('BEGIN IMMEDIATE'); db.exec('INSERT INTO t VALUES (1)');
    console.log('locked');
    setTimeout(() => { db.exec('COMMIT'); db.close(); }, 800);
  `], { cwd: path.resolve('.'), stdio: ['ignore', 'pipe', 'inherit'] });
  const done = new Promise(resolve => holder.on('exit', resolve));
  await new Promise<void>(resolve => holder.stdout!.on('data', d => { if (String(d).includes('locked')) resolve(); }));

  // Without a timeout the write fails at once; that is what busy_timeout prevents.
  const impatient = new Database(file);
  impatient.pragma('busy_timeout = 0');
  assert.throws(() => impatient.exec('INSERT INTO t VALUES (2)'), (err: any) => err.code === 'SQLITE_BUSY');
  impatient.close();

  const started = Date.now();
  db.exec('INSERT INTO t VALUES (3)');
  const waited = Date.now() - started;
  assert.ok(waited >= 200 && waited < BUSY_TIMEOUT_MS, `waited ${waited} ms for the lock`);
  assert.equal(await done, 0);
  assert.deepEqual(db.prepare('SELECT v FROM t ORDER BY v').all().map((r: any) => r.v), [1, 3]);
  db.close();
});

test('a backup made while the source is open in WAL mode is complete, verified, and never overwrites', async () => {
  const source = path.join(dir, 'source.db');
  const db = new Database(source);
  applyConnectionPragmas(db);
  db.pragma('wal_autocheckpoint = 0');   // keep every row in the -wal file
  db.exec('CREATE TABLE songs (id INTEGER PRIMARY KEY, title TEXT); CREATE TABLE other (x)');
  const insert = db.prepare('INSERT INTO songs (title) VALUES (?)');
  db.transaction(() => { for (let i = 0; i < 500; i++) insert.run(`song ${i}`); })();
  assert.ok(fs.statSync(`${source}-wal`).size > 0, 'rows are still in the WAL');

  const destination = path.join(dir, 'backups', 'copy.db');
  const result = await backupDatabase(source, destination);
  assert.equal(result.tables, 2);
  const copy = new Database(destination, { readonly: true });
  assert.equal((copy.prepare('SELECT COUNT(*) AS n FROM songs').get() as { n: number }).n, 500);
  copy.close();

  await assert.rejects(backupDatabase(source, destination), /Refusing to overwrite/);
  db.close();
});
