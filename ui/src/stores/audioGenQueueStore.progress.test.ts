// audioGenQueueStore.progress.test.ts — a running row shows the engine's stage
// and progress in both queue modes: a server-owned queue projects it from the
// server's audio items; a browser-owned queue follows a studio's workflow-backed
// row (Cover, Auto-Gen) to the render that workflow queued. See
// audioGenQueueStore.migration.test.ts for why the store is imported this way.
import test from 'node:test';
import assert from 'node:assert/strict';
import { register } from 'node:module';
import { createFakeIndexedDB, AudioQueueApiFake, MemoryStorage } from './audioQueueStoreTestFake.js';

register('./cssStubLoader.mjs', import.meta.url);
Object.assign(globalThis, {
  window: { addEventListener() {}, removeEventListener() {} },
  localStorage: new MemoryStorage(),
  indexedDB: createFakeIndexedDB().indexedDB,
});

const {
  OWNER_KEY, addManualQueueItem, updateManualQueueItem, completeManualQueueItem, resumeQueue,
  _queueItemsForTests, _resetAudioQueueForTests,
} = await import('./audioGenQueueStore.js');

const STAGE = 'YuE2: Composing: Step 3/10';
const ok = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
const settle = () => new Promise(resolve => setTimeout(resolve, 300));

async function withEngine(owner: string, run: (api: AudioQueueApiFake) => Promise<void>) {
  localStorage.setItem(OWNER_KEY, owner);
  const api = new AudioQueueApiFake();
  // The render a cover workflow queued: a server audio item with an engine job.
  api.serverItems = [{ id: 'srv1', status: 'submitted', jobId: 'job1', request: { title: 'Cover' },
    meta: { source: 'cover-studio', workflowJobId: 'wf1', workflowKey: 'render' }, createdAt: 1 }];
  const uninstall = api.install();
  const queue = globalThis.fetch;
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    if (String(url).includes('/generate/status/job1')) return ok({ jobId: 'job1', status: 'running', stage: STAGE, progress: 40 });
    if (String(url).includes('/workflows/jobs/wf1')) return ok({ job: { id: 'wf1', status: 'running', audioIntentIds: ['srv1'] }, events: [], gap: false });
    return queue(url, init);
  }) as typeof fetch;
  _resetAudioQueueForTests([]);
  try { await run(api); } finally { _resetAudioQueueForTests(); uninstall(); }
}

test('browser-owned queue: a workflow-backed Cover row shows the engine stage and progress', async () => {
  await withEngine('browser', async () => {
    const id = addManualQueueItem({ title: 'Cover' });
    updateManualQueueItem(id, { workflowJob: { id: 'wf1', token: 't' }, stage: 'Generating…' });
    await settle();
    const row = _queueItemsForTests().find(item => item.id === id)!;
    assert.equal(row.stage, STAGE);
    assert.equal(row.progress, 40);
    completeManualQueueItem(id, { audioUrl: '/audio/a.wav' });
    await new Promise(resolve => setTimeout(resolve, 1700));
    assert.equal(row.stage, 'Complete!', 'the follower stops once the row has finished');
  });
});

test('server-owned queue: the projected workflow render shows the engine stage and progress', async () => {
  await withEngine('server', async () => {
    await resumeQueue('t');
    await settle();
    const row = _queueItemsForTests().find(item => item.serverQueueId === 'srv1')!;
    assert.equal(row.stage, STAGE);
    assert.equal(row.progress, 40);
  });
});
