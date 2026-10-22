import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { REPO, apiRaw, pluginDeclaration } from './lib.mjs';

test('plugin declarations use metadata names and exclude companion modules', () => {
  const read = (file) => fs.readFileSync(path.join(REPO, 'engine', 'plugins', file), 'utf8');
  assert.deepEqual(pluginDeclaration(read('schedulers/md_ht_scheduler_v5_1.lua')), {
    type: 'scheduler', name: 'md_ht_scheduler v5.1',
  });
  assert.equal(pluginDeclaration(read('schedulers/beta_math.lua')), null);
  assert.equal(pluginDeclaration(read('solvers/stork4_constants.lua')), null);
  assert.deepEqual(pluginDeclaration("solver = { name = 'custom solver' }"), {
    type: 'solver', name: 'custom solver',
  });
});

test('apiRaw cancels an open SSE body after reading its headers', { timeout: 1000 }, async (t) => {
  let cancelled = false;
  const stream = new ReadableStream({
    start(controller) { controller.enqueue(new TextEncoder().encode('data: hello\n\n')); },
    cancel() { cancelled = true; },
  });
  t.mock.method(globalThis, 'fetch', async () => new Response(stream, {
    status: 200, headers: { 'content-type': 'text/event-stream; charset=utf-8' },
  }));
  assert.deepEqual(await apiRaw('GET', '/api/logs?after=-1'), {
    status: 200, contentType: 'text/event-stream; charset=utf-8', body: null,
  });
  assert.equal(cancelled, true);
});

test('apiRaw still reads finite JSON and text error bodies', async (t) => {
  const responses = [
    new Response('{"error":"unavailable"}', { status: 503, headers: { 'content-type': 'application/json' } }),
    new Response('missing', { status: 404, headers: { 'content-type': 'text/plain' } }),
  ];
  t.mock.method(globalThis, 'fetch', async () => responses.shift());
  assert.deepEqual(await apiRaw('GET', '/json'), {
    status: 503, contentType: 'application/json', body: { error: 'unavailable' },
  });
  assert.deepEqual(await apiRaw('GET', '/text'), {
    status: 404, contentType: 'text/plain', body: 'missing',
  });
});
