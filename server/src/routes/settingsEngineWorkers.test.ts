import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const directory = fs.mkdtempSync(path.resolve('.test-settings-workers-'));
process.env.DATA_DIR = directory;
const express = (await import('express')).default;
const settings = (await import('./settings.js')).default;
const workers = await import('../services/engineWorkers.js');

test('GET /engine-workers reports the table and busy only while a call is bound', async t => {
  t.after(() => { workers.setEngineWorkersForTest(null); fs.rmSync(directory, { recursive: true, force: true }); });
  const { workers: table } = workers.resolveWorkers('', [], 8085);
  workers.setEngineWorkersForTest(table);
  const app = express();
  app.use('/settings', settings);
  const server = app.listen(0, '127.0.0.1');
  await new Promise<void>(resolve => server.once('listening', resolve));
  t.after(() => server.close());
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  const get = async () => (await fetch(`http://127.0.0.1:${address.port}/settings/engine-workers`)).json() as Promise<
    { workers: Array<{ id: string; primary: boolean; gpuUuid: string | null }>; busy: boolean }>;

  const idle = await get();
  assert.equal(idle.workers.length, 1);
  assert.deepEqual([idle.workers[0].id, idle.workers[0].primary, idle.workers[0].gpuUuid], ['w0', true, null]);
  assert.equal(idle.busy, false);

  await workers.runWithWorker(table[0], async () => { assert.equal((await get()).busy, true); });
  assert.equal((await get()).busy, false);
});
