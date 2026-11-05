// Generation jobs on two engine workers (GH #13), through the real route entry
// point: overlapping jobs each run on their own worker, /status reports it,
// the ledger keeps it, and a cancel or MM3 stream arriving outside the job's
// lease reaches that job's engine. Two fake HTTP engines stand in for two
// cards; the backend's generate is stubbed, so no GPU is touched.
// Run: npx tsx --test src/routes/generate.workers.test.ts

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import Database from 'better-sqlite3';

const DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'route-workers-'));
process.env.DATA_DIR = DATA_DIR;
process.env.ACESTEPCPP_HOST = '127.0.0.1';
process.env.ACESTEPCPP_WORKER_GPUS = '';
const database = await import('../db/database.js');
database.initDb();
const { config } = await import('../config.js');
const workers = await import('../services/engineWorkers.js');
const { setEngineReady } = await import('../engineState.js');
const registry = await import('../services/backends/registry.js');
const lane = await import('../services/generation/gpuLane.js');
const { GenerationJobLedger } = await import('../services/generation/jobLedger.js');
const gen = await import('./generate.js');

const hits: string[] = [];
function fakeEngine(name: string) {
  return http.createServer((req, res) => {
    hits.push(`${name} ${req.method} ${req.url}`);
    res.setHeader('content-type', 'application/json');
    res.end('{}');
  });
}
async function listen(srv: http.Server, port = 0): Promise<number> {
  await new Promise<void>((resolve, reject) => {
    srv.once('error', reject);
    srv.listen(port, '127.0.0.1', () => { srv.off('error', reject); resolve(); });
  });
  return (srv.address() as AddressInfo).port;
}
let a!: http.Server, b!: http.Server, pa = 0;
for (let i = 0; i < 20 && !pa; i++) {
  a = fakeEngine('A'); b = fakeEngine('B');
  const p = await listen(a);
  try { await listen(b, p + 1); pa = p; } catch { a.close(); }
}
assert.ok(pa, 'no adjacent ports free');

const GPUS = [
  { index: 0, uuid: 'GPU-aaaa', name: 'Card A', memoryMB: 24576 },
  { index: 1, uuid: 'GPU-bbbb', name: 'Card B', memoryMB: 24576 },
];
const savedPort = config.aceServer.port;
config.aceServer.port = pa;
const { workers: table } = workers.resolveWorkers('GPU-aaaa,GPU-bbbb', GPUS, pa, []);
workers.setEngineWorkersForTest(table);
setEngineReady(true, 'test');
table[1].ready = true;
const db = new Database(':memory:');
const ledger = new GenerationJobLedger(db);
gen.setGenerationJobLedgerForTests(ledger);

// Stub ACE: record the worker each job runs on, expose an engine job id and an
// MM3-style stream, and hold until the job is cancelled.
const ran: Array<{ id: string; worker: string }> = [];
const ace = registry.getBackend('ace') as any;
ace.capabilities = async () => ({ extensions: [] });
/** Per-test override of the stub; a coalesced follower's turn returns at once, as YuE2's does. */
let onGenerate: ((job: any, deps: any) => Promise<unknown>) | undefined;
const done = { endReason: 'succeeded', stages: [], artifacts: [], songIds: [] };
ace.generate = async (job: any, deps: any) => {
  if (job.coalescedInto) { ran.push({ id: job.id, worker: `follower:${workers.currentWorker().id}` }); return done; }
  if (onGenerate) return onGenerate(job, deps);
  ran.push({ id: job.id, worker: workers.currentWorker().id });
  job.aceJobId = `eng-${ran.length}`;
  job.mm3Streaming = true;
  job.status = 'running';
  await new Promise<void>(resolve => deps.signal.addEventListener('abort', () => resolve(), { once: true }));
  throw new Error('Cancelled: stub backend');
};
for (const backend of registry.listBackends() as any[]) if (backend.releaseVram) backend.releaseVram = async () => {};

test.after(() => {
  lane.setWorkerConcurrency(false);
  workers.setEngineWorkersForTest(null);
  gen.setGenerationJobLedgerForTests(null);
  config.aceServer.port = savedPort;
  a.close(); b.close();
  database.closeDb();
  fs.rmSync(DATA_DIR, { recursive: true, force: true });
});

async function until(cond: () => boolean, what: string): Promise<void> {
  for (let i = 0; i < 500 && !cond(); i++) await new Promise(r => setTimeout(r, 10));
  assert.ok(cond(), `timed out waiting for ${what}`);
}
async function callRoute(method: string, routePath: string, params: Record<string, string>, query: Record<string, string> = {}) {
  const layer = (gen.default as any).stack.find((l: any) => l.route?.path === routePath && l.route.methods[method]);
  assert.ok(layer, `route ${method.toUpperCase()} ${routePath} not found`);
  const out: { status: number; body?: any } = { status: 200 };
  const res: any = {
    headersSent: false, writableEnded: false,
    status(code: number) { out.status = code; return res; },
    json(value: unknown) { out.body = value; res.writableEnded = true; return res; },
    type() { return res; }, send(v: unknown) { out.body = v; res.writableEnded = true; return res; },
    setHeader() { res.headersSent = true; }, write() { return true; }, end() { res.writableEnded = true; },
    on() { return res; }, once() { return res; },
  };
  await layer.route.stack.at(-1).handle({ body: {}, headers: {}, query, params }, res, () => {});
  return out;
}
const submit = async () => {
  const r = await gen.submitGeneration({ caption: 'test', lyrics: '[Instrumental]', duration: 30, backendOverride: 'ace' }, 'user-1');
  assert.equal(r.ok, true, JSON.stringify(r));
  return (r as { jobId: string }).jobId;
};

test('two overlapping jobs run on two workers; status, ledger, cancel and stream follow each job', async () => {
  lane.setWorkerConcurrency(true);
  const j1 = await submit();
  const j2 = await submit();
  await until(() => ran.length === 2, 'both jobs to start side by side');
  const workerOf = (id: string) => ran.find(r => r.id === id)!.worker;
  assert.deepEqual([workerOf(j1), workerOf(j2)].sort(), ['w0', 'w1']);
  const onW1 = workerOf(j1) === 'w1' ? j1 : j2;
  const onW0 = onW1 === j1 ? j2 : j1;

  const s1 = await callRoute('get', '/status/:id', { id: onW1 });
  assert.equal(s1.body.workerId, 'w1');
  assert.equal((await callRoute('get', '/status/:id', { id: onW0 })).body.workerId, 'w0');
  assert.equal(ledger.get(onW1)?.workerId, 'w1');

  // Out-of-lease requests: the MM3 stream and the cancel reach the job's engine.
  hits.length = 0;
  const engId = gen.getGenerationJob(onW1)!.aceJobId!;
  await callRoute('get', '/mm3/stream/:id', { id: onW1 });
  assert.ok(hits.some(h => h === `B GET /mm3/stream?id=${engId}`), `stream went elsewhere: ${hits}`);
  hits.length = 0;
  assert.equal(gen.cancelGenerationJob(onW1), true);
  await until(() => hits.length > 0, 'the cancel to reach an engine');
  assert.deepEqual(hits, [`B POST /job?id=${engId}&cancel=1`]);

  assert.equal(gen.cancelGenerationJob(onW0), true);
  await until(() => !lane.gpuLaneBusy(), 'the lane to go idle');
  assert.equal(ledger.get(onW1)?.workerId, 'w1', 'the ledger keeps the worker past the job');
});

test('concurrency off: jobs queue on the primary, which /status reports', async () => {
  lane.setWorkerConcurrency(false);
  ran.length = 0;
  const j1 = await submit();
  const j2 = await submit();
  await until(() => ran.length === 1, 'the first job to start');
  assert.equal((await callRoute('get', '/status/:id', { id: j1 })).body.workerId, 'w0');
  assert.equal((await callRoute('get', '/status/:id', { id: j2 })).body.workerId, null, 'a queued job has no worker yet');
  gen.cancelGenerationJob(j1);
  await until(() => ran.length === 2, 'the second job to start after the first');
  assert.deepEqual(ran.map(r => r.worker), ['w0', 'w0']);
  gen.cancelGenerationJob(j2);
  await until(() => !lane.gpuLaneBusy(), 'the lane to go idle');
});

test('a job no worker can take fails with the error and stays failed in the ledger', async () => {
  lane.setWorkerConcurrency(true);
  ran.length = 0;
  let releaseExclusive!: () => void;
  const exclusive = lane.runOnGpuLane(() => new Promise<void>(r => { releaseExclusive = r; }), { label: 'training' });
  const id = await submit();
  // Both cards drop out while the exclusive task holds the lane.
  setEngineReady(false, 'crashed');
  table[1].ready = false;
  try {
    releaseExclusive();
    await exclusive;
    await until(() => gen.getGenerationJob(id)!.status !== 'pending', 'the job to end');
    const job = gen.getGenerationJob(id)!;
    assert.equal(job.status, 'failed');
    assert.match(job.error ?? '', /No engine worker is ready/);
    await until(() => ledger.get(id)?.status === 'failed', 'the ledger to record the failure');
    assert.equal(ran.length, 0);
  } finally {
    setEngineReady(true, 'test');
    table[1].ready = true;
  }
});

test('batching: a dispatched job is never claimed; followers carry the lead worker and are requeued when handed back', async () => {
  lane.setWorkerConcurrency(true);
  ran.length = 0;
  let lead!: { job: any; deps: any };
  let releaseLead!: () => void, releaseOther!: () => void;
  onGenerate = async (job, deps) => {
    ran.push({ id: job.id, worker: workers.currentWorker().id });
    // The second job stays "pending" while it prepares, as YuE2's preflight does.
    if (!lead) { lead = { job, deps }; job.status = 'running'; await new Promise<void>(r => { releaseLead = r; }); }
    else if (ran.length === 2) await new Promise<void>(r => { releaseOther = r; });
    else onGenerate = undefined;  // the requeued follower: end as a plain stub job
    if (job.status === 'running' || job.status === 'pending') job.status = 'succeeded';
    return done;
  };
  try {
    const j1 = await submit();
    const j2 = await submit();
    await until(() => ran.length === 2, 'both jobs dispatched');
    const j3 = await submit();
    await new Promise(r => setTimeout(r, 20));
    const candidates = lead.deps.pendingJobs().map((j: any) => j.id);
    assert.deepEqual(candidates, [j3], 'a job already dispatched to a worker was offered for coalescing');

    // Claim j3 into the lead's batch, as runYue2Generation does.
    const f = gen.getGenerationJob(j3)!;
    f.coalescedInto = j1; lead.deps.assignFollower(f, lead.job); f.status = 'running';
    assert.equal(ledger.get(j3)?.workerId, lead.job.workerId, 'the running follower must carry the lead worker durably');
    assert.equal((await callRoute('get', '/status/:id', { id: j3 })).body.workerId, lead.job.workerId);

    // j2's worker frees: j3's own turn runs there and returns at once (it is in a batch).
    releaseOther();
    await until(() => ran.some(r => r.id === j3), 'the follower turn');
    // The batch fails and hands j3 back; it must render on its own afterwards.
    f.coalescedInto = undefined; lead.deps.assignFollower(f, null); f.status = 'pending';
    assert.equal(ledger.get(j3)?.workerId, null);
    releaseLead();
    await until(() => ran.filter(r => r.id === j3).length === 2, 'the handed-back follower to be requeued');
    await until(() => !lane.gpuLaneBusy(), 'the lane to go idle');
    assert.equal(gen.getGenerationJob(j3)!.status, 'succeeded');
    assert.ok(ledger.get(j3)?.workerId, 'the requeued follower records its own worker');
    void j2;
  } finally {
    onGenerate = undefined;
  }
});

test('batching: a follower whose own turn ends first is recorded when its batch succeeds', async () => {
  lane.setWorkerConcurrency(true);
  ran.length = 0;
  let lead!: { job: any; deps: any };
  let releaseLead!: () => void, releaseOther!: () => void;
  onGenerate = async (job, deps) => {
    ran.push({ id: job.id, worker: workers.currentWorker().id });
    if (!lead) { lead = { job, deps }; job.status = 'running'; await new Promise<void>(r => { releaseLead = r; }); }
    else await new Promise<void>(r => { releaseOther = r; });
    job.status = 'succeeded';
    return done;
  };
  try {
    const j1 = await submit();
    await submit();
    await until(() => ran.length === 2, 'both jobs dispatched');
    const j3 = await submit();
    const f = gen.getGenerationJob(j3)!;
    f.coalescedInto = j1; lead.deps.assignFollower(f, lead.job); f.status = 'running';
    // j3's own turn runs on the freed worker and returns at once, while the batch renders.
    releaseOther();
    await until(() => ran.some(r => r.id === j3), 'the follower turn');
    await new Promise(r => setTimeout(r, 20));
    assert.equal(ledger.get(j3)?.status, 'open', 'the follower was recorded before its batch ended');
    // The batch renders the follower's song too.
    f.status = 'succeeded';
    f.result = { audioUrls: ['/audio/follower.wav'] } as any;
    releaseLead();
    await until(() => !lane.gpuLaneBusy(), 'the lane to go idle');
    await until(() => ledger.get(j3)?.status === 'succeeded', 'the follower to be recorded');
    const row = ledger.get(j3)!;
    assert.ok(row.finishedAt, 'finishedAt');
    assert.deepEqual(row.result, f.result);
    assert.equal(row.workerId, lead.job.workerId);
    assert.equal(ran.filter(r => r.id === j3).length, 1, 'a finished follower must not be requeued');
  } finally {
    onGenerate = undefined;
  }
});
