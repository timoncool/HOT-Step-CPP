import assert from 'node:assert/strict';
import test, { after } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';

const root = fs.mkdtempSync(path.join(process.cwd(), '.test-heartmula-batch-'));
process.env.DATA_DIR = path.join(root, 'library');
process.env.TRAINING_DIR = path.join(root, 'training');
const { config } = await import('../../config.js');
config.data.dir = path.join(root, 'library');
const { initDb, getDb, closeDb } = await import('../../db/database.js');
initDb();
const runner = await import('./yue2BatchRunner.js');
const queue = await import('./labelingQueue.js');
const { trainingBaseDir } = await import('./paths.js');
for (const id of ['first', 'second']) getDb().prepare('INSERT INTO training_datasets (id, slug, name, source_dir) VALUES (?, ?, ?, ?)').run(id, id, id, path.join(root, id));
after(() => { closeDb(); fs.rmSync(root, { recursive: true, force: true }); });

async function until(done: () => boolean) {
  for (let n = 0; n < 200; n++) { if (done()) return; await new Promise(resolve => setTimeout(resolve, 20)); }
  throw new Error('Batch did not finish');
}

test('HeartMuLa batch prepares stale audio, skips valid audio, and trains datasets in order with a persisted run ID', async () => {
  const calls: string[] = [];
  const original = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    const parts = String(url).split('/'); const id = parts.at(-2)!; const route = parts.at(-1)!;
    if (!init?.method) return Response.json({ prepare: { cacheValid: id === 'second', preparedSamples: 1, staleSamples: id === 'first' ? 1 : 0 }, runs: [] });
    calls.push(`${id}/${route}`);
    if (route === 'heartmula-train') assert.deepEqual(JSON.parse(String(init.body)), { recipe: { steps: 1 } });
    const job = queue.createJob(route === 'heartmula-prepare' ? 'heartmula-prepare' : 'heartmula-train', id, [], {});
    queue.finishJob(job, 'done');
    return Response.json({ jobId: job.id, ...(route === 'heartmula-train' ? { run: `${id}/2026-10-26` } : {}) });
  };
  try {
    const batch = runner.startBatch({ family: 'heartmula', datasetIds: ['first', 'second'], lyricTiming: false, recipe: { recipe: { steps: 1 } } });
    assert.ok(!('error' in batch));
    await until(() => runner.getBatch(batch.id)?.status === 'done');
    assert.deepEqual(calls, ['first/heartmula-prepare', 'first/heartmula-train', 'second/heartmula-train']);
    const saved = JSON.parse(fs.readFileSync(path.join(trainingBaseDir, 'yue2-batches', `${batch.id}.json`), 'utf8'));
    assert.equal(saved.family, 'heartmula'); assert.equal(saved.items[0].stages[1].runId, 'first/2026-10-26');
    assert.ok(saved.items.every((item: { status: string }) => item.status === 'done'));
  } finally { globalThis.fetch = original; }
});

test('a recovered HeartMuLa batch resumes its saved run and keeps finished preparation', async () => {
  const id = 'recovered';
  fs.writeFileSync(path.join(trainingBaseDir, 'yue2-batches', `${id}.json`), JSON.stringify({
    id, family: 'heartmula', status: 'paused', lyricTiming: false, recipe: {}, createdAt: Date.now(), finishedAt: null,
    items: [{ datasetId: 'first', name: 'first', status: 'pending', currentStage: null, error: null, stages: [
      { stage: 'prepare', status: 'done', jobId: 'old-prepare' },
      { stage: 'train', status: 'pending', jobId: '', runId: 'first/2026-10-26' },
    ] }],
  }));
  const original = globalThis.fetch; const bodies: unknown[] = [];
  globalThis.fetch = async (_url, init) => {
    if (!init?.method) return Response.json({ runs: [{ run: 'first/2026-10-26', state: 'failed', resumable: true }] });
    bodies.push(JSON.parse(String(init.body)));
    const job = queue.createJob('heartmula-train', 'first', [], {}); queue.finishJob(job, 'done');
    return Response.json({ jobId: job.id, run: 'first/2026-10-26' });
  };
  try {
    assert.equal(runner.resumeBatch(id), 'ok');
    await until(() => runner.getBatch(id)?.status === 'done');
    assert.deepEqual(bodies, [{ resumeRun: 'first/2026-10-26' }]);
  } finally { globalThis.fetch = original; }
});

test('pause persists after the current stage; resume continues without preparing twice', async () => {
  const original = globalThis.fetch;
  let prepareJob: ReturnType<typeof queue.createJob> | undefined;
  const posts: string[] = [];
  globalThis.fetch = async (url, init) => {
    if (!init?.method) return Response.json({ prepare: { cacheValid: false }, runs: [] });
    const route = String(url).split('/').at(-1)!; posts.push(route);
    const job = queue.createJob(route === 'heartmula-prepare' ? 'heartmula-prepare' : 'heartmula-train', 'first', [], {});
    if (route === 'heartmula-prepare') prepareJob = job; else queue.finishJob(job, 'done');
    return Response.json({ jobId: job.id, run: 'first/2026-10-26' });
  };
  try {
    const batch = runner.startBatch({ family: 'heartmula', datasetIds: ['first'], lyricTiming: false, recipe: {} });
    assert.ok(!('error' in batch));
    await until(() => !!runner.getBatch(batch.id)?.items[0].stages[0].jobId);
    assert.equal(runner.pauseBatch(batch.id), 'ok'); queue.finishJob(prepareJob!, 'done');
    await until(() => runner.getBatch(batch.id)?.status === 'paused');
    const saved = JSON.parse(fs.readFileSync(path.join(trainingBaseDir, 'yue2-batches', `${batch.id}.json`), 'utf8'));
    assert.equal(saved.status, 'paused'); assert.equal(saved.items[0].stages[0].status, 'done');
    assert.equal(runner.resumeBatch(batch.id), 'ok');
    await until(() => runner.getBatch(batch.id)?.status === 'done');
    assert.deepEqual(posts, ['heartmula-prepare', 'heartmula-train']);
  } finally { globalThis.fetch = original; }
});

test('cancel before the queue starts leaves every item cancelled', async () => {
  const batch = runner.startBatch({ family: 'heartmula', datasetIds: ['first', 'second'], lyricTiming: false, recipe: {} });
  assert.ok(!('error' in batch));
  assert.equal(runner.cancelBatch(batch.id), true);
  await until(() => runner.getBatch(batch.id)?.status === 'cancelled');
  assert.ok(runner.getBatch(batch.id)?.items.every(item => item.status === 'cancelled'));
});
