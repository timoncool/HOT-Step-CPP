// The GPU lane with two engine workers (GH #13): exclusive by default, worker
// leases overlap only with worker concurrency on, nested calls reuse their
// reservation, and an exclusive waiter waits for every worker and for a YuE2
// job released early to drain. Two fake HTTP engines stand in for two cards;
// no GPU is touched.
// Run: npx tsx --test src/services/generation/gpuLane.test.ts

import { test, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { EventEmitter } from 'node:events';
import type { AddressInfo } from 'node:net';

process.env.ACESTEPCPP_HOST = '127.0.0.1';
process.env.ACESTEPCPP_WORKER_GPUS = '';
process.env.ACESTEPCPP_EXE = process.execPath;   // the training tests respawn fake engines; it only needs to exist

const { config } = await import('../../config.js');
const workers = await import('../engineWorkers.js');
const { setEngineReady } = await import('../../engineState.js');
const { aceClient } = await import('../aceClient.js');
const lane = await import('./gpuLane.js');
type LaneLease = import('./gpuLane.js').LaneLease;
const { trackBackendRelease, awaitBackendRelease, releaseOnEveryWorker, _resetBackendReleases } = await import('./backendRelease.js');
const proc = await import('../aceEngineProcess.js');

function fakeEngine(name: string) {
  return http.createServer((req, res) => {
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify({ status: 'ok', engine: name }));
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
  { index: 1, uuid: 'GPU-bbbb', name: 'Card B', memoryMB: 12288 },
];
const savedPort = config.aceServer.port;
config.aceServer.port = pa;
const { workers: table } = workers.resolveWorkers('GPU-aaaa,GPU-bbbb', GPUS, pa, []);
workers.setEngineWorkersForTest(table);
const [w0, w1] = table;
setEngineReady(true, 'test');
w1.ready = true;

after(() => {
  lane.setWorkerConcurrency(false);
  workers.setEngineWorkersForTest(null);
  config.aceServer.port = savedPort;
  a.close(); b.close();
});
beforeEach(() => {
  assert.equal(lane.gpuLaneBusy(), false, 'previous test left the lane busy');
  lane.setWorkerConcurrency(false);
  setEngineReady(true, 'test');
  w1.ready = true;
});

function fakeChild(): any {
  const c: any = new EventEmitter();
  c.stdout = new EventEmitter();
  c.stderr = new EventEmitter();
  // No pid: stop() then forgets it as already gone. A real-looking pid would make
  // stop() taskkill that pid on Windows.
  c.pid = undefined;
  c.exitCode = null;
  c.signalCode = null;
  c.kill = () => true;
  return c;
}
proc.testHooks.spawn = (() => fakeChild()) as any;
proc.testHooks.listGpus = () => GPUS;

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>(r => { resolve = r; });
  return { promise, resolve };
}
const tick = () => new Promise(r => setTimeout(r, 5));
async function until(cond: () => boolean, what: string): Promise<void> {
  for (let i = 0; i < 500 && !cond(); i++) await new Promise(r => setTimeout(r, 10));
  assert.ok(cond(), `timed out waiting for ${what}`);
}
/** Which fake engine a bare-client call reaches from here. */
async function engineHit(): Promise<string> {
  return ((await aceClient.health()) as { engine?: string }).engine ?? '?';
}

test('concurrency off: worker leases run one at a time on the primary', async () => {
  const order: string[] = [];
  const gate = deferred();
  const one = lane.runOnWorkerLane(async l => { order.push(`1:${l.worker?.id}:${await engineHit()}`); await gate.promise; order.push('1 end'); });
  const two = lane.runOnWorkerLane(async l => { order.push(`2:${l.worker?.id}:${await engineHit()}`); });
  await until(() => order.length > 0, 'the first job');
  await tick(); await tick();
  assert.deepEqual(order, ['1:w0:A'], 'second job started beside the first with concurrency off');
  gate.resolve();
  await Promise.all([one, two]);
  assert.deepEqual(order, ['1:w0:A', '1 end', '2:w0:A']);
});

test('concurrency on: two jobs overlap, each bare client follows its lease worker', async () => {
  lane.setWorkerConcurrency(true);
  const gate = deferred();
  const seen: string[] = [];
  const job = (n: number) => lane.runOnWorkerLane(async l => {
    seen.push(`${n}:${l.worker!.id}:${await engineHit()}:${workers.currentWorker().id}`);
    await gate.promise;
  }, { label: `job ${n}` });
  const both = Promise.all([job(1), job(2)]);
  await until(() => seen.length === 2, 'both jobs to reach their engines');
  assert.deepEqual(seen.sort(), ['1:w0:A:w0', '2:w1:B:w1']);
  assert.equal(lane.gpuLaneWorkerLeases().size, 2);
  assert.equal(w1.busy, 1, 'the lease holds the worker scope for its whole run');
  gate.resolve();
  await both;
  assert.equal(w1.busy, 0);
});

test('a bound call cannot switch to another worker mid-lease', async () => {
  lane.setWorkerConcurrency(true);
  const gate = deferred();
  const first = lane.runOnWorkerLane(async () => { await gate.promise; });
  await lane.runOnWorkerLane(async l => {
    assert.equal(l.worker, w1);
    await assert.rejects(workers.runWithWorker(w0, async () => {}), /bound to w1 cannot switch to w0/);
  });
  gate.resolve();
  await first;
});

test('an exclusive waiter blocks new worker leases and waits for every worker', async () => {
  lane.setWorkerConcurrency(true);
  const order: string[] = [];
  const g0 = deferred(), g1 = deferred();
  const j0 = lane.runOnWorkerLane(async () => { await g0.promise; order.push('w0 done'); });
  const j1 = lane.runOnWorkerLane(async () => { await g1.promise; order.push('w1 done'); });
  await tick();
  const ex = lane.runOnGpuLane(async l => { order.push(`exclusive:${l.worker ?? 'all'}:${await engineHit()}`); }, { label: 'training' });
  const late = lane.runOnWorkerLane(async () => { order.push('late worker job'); });
  g0.resolve();
  await tick(); await tick();
  assert.deepEqual(order, ['w0 done'], 'a worker lease started past the exclusive waiter, or it did not wait for w1');
  g1.resolve();
  await Promise.all([j0, j1, ex, late]);
  assert.deepEqual(order, ['w0 done', 'w1 done', 'exclusive:all:A', 'late worker job']);
});

test('nested reservations reuse the outer one instead of deadlocking', async () => {
  lane.setWorkerConcurrency(true);
  const gate = deferred();
  const other = lane.runOnWorkerLane(async () => { await gate.promise; });  // keeps w0 busy
  const inner = await lane.runOnWorkerLane(async outer =>
    lane.runOnGpuLane(async l => ({ same: l === outer, worker: l.worker?.id, hit: await engineHit() })));
  assert.deepEqual(inner, { same: true, worker: 'w1', hit: 'B' });
  gate.resolve();
  await other;
  const nestedExclusive = await lane.runOnGpuLane(async outer =>
    lane.runOnWorkerLane(async l => lane.runOnGpuLane(async l2 => l === outer && l2 === outer)));
  assert.equal(nestedExclusive, true);
});

test('YuE2 early release: the next job gets the worker, an exclusive waiter waits for the render to drain', async () => {
  const order: string[] = [];
  const render = deferred(), nextGate = deferred();
  const first = lane.runOnWorkerLane(async l => {
    assert.equal(lane.releaseGpuLane(l), true);
    assert.equal(l.isCurrent(), false);
    await render.promise;
    // Post-render GPU work takes its own worker back, ahead of the exclusive waiter.
    await lane.runOnGpuLane(async l2 => { order.push(`finish:${l2.worker?.id}:${await engineHit()}`); });
    order.push('first settled');
  }, { family: 'yue2' });
  await tick();
  const next = lane.runOnWorkerLane(async () => { order.push('next started'); await nextGate.promise; order.push('next done'); }, { family: 'yue2' });
  await tick();
  const ex = lane.runOnGpuLane(async () => { order.push('exclusive'); });
  await tick();
  assert.deepEqual(order, ['next started']);
  nextGate.resolve();
  await tick(); await tick();
  assert.deepEqual(order, ['next started', 'next done'], 'exclusive ran while a released render was still on the engine');
  render.resolve();
  await Promise.all([first, next, ex]);
  assert.deepEqual(order, ['next started', 'next done', 'finish:w0:A', 'first settled', 'exclusive']);
});

test('reset rejects queued work and a draining worker is not reused before it settles', async () => {
  lane.setWorkerConcurrency(true);
  const g0 = deferred(), g1 = deferred();
  let lease0!: LaneLease;
  const j0 = lane.runOnWorkerLane(async l => { lease0 = l; await g0.promise; return l.isCurrent(); });
  const j1 = lane.runOnWorkerLane(async () => { await g1.promise; });
  await tick();
  const queued = lane.runOnWorkerLane(async () => 'ran');
  assert.equal(lane.resetGpuLane(), 1);
  await assert.rejects(queued, lane.LaneResetError);
  assert.equal(lease0.draining, true);
  const after = lane.runOnWorkerLane(async l => l.worker!.id);
  await tick();
  assert.equal(lane.gpuLaneWorkerLeases().size, 2, 'a draining worker was handed to new work');
  g1.resolve();
  assert.equal(await after, 'w1');
  g0.resolve();
  assert.equal(await j0, false, 'a reset lease must not report current');
  await j1;
});

test('eligibility: only matching cards take the job; none eligible or ready fails once', async () => {
  lane.setWorkerConcurrency(true);
  const bigCard = (w: { gpuUuid: string | null }) => w.gpuUuid === 'GPU-aaaa';
  const smallCard = (w: { gpuUuid: string | null }) => w.gpuUuid === 'GPU-bbbb';
  assert.equal(await lane.runOnWorkerLane(async l => l.worker!.id, { eligible: smallCard }), 'w1');
  await assert.rejects(lane.runOnWorkerLane(async () => 'ran', { eligible: () => false, label: 'huge model' }),
    /No engine worker can run huge model/);
  w1.ready = false;
  await assert.rejects(lane.runOnWorkerLane(async () => 'ran', { eligible: smallCard }), /No engine worker is ready/);
  // A not-ready card is skipped when a ready one is eligible.
  assert.equal(await lane.runOnWorkerLane(async l => l.worker!.id), 'w0');
  // Waits (rather than fails) while the only eligible ready card is busy.
  const gate = deferred();
  const busy = lane.runOnWorkerLane(async () => { await gate.promise; }, { eligible: bigCard });
  const waiting = lane.runOnWorkerLane(async l => l.worker!.id, { eligible: bigCard });
  await tick();
  gate.resolve();
  await busy;
  assert.equal(await waiting, 'w0');
});

test('single-GPU fallback: the legacy one-worker table ignores eligibility and readiness', async () => {
  workers.setEngineWorkersForTest(null);
  try {
    assert.equal(workers.engineWorkers().length, 1);
    lane.setWorkerConcurrency(true);
    assert.equal(await lane.runOnWorkerLane(async l => `${l.worker!.id}:${await engineHit()}`), 'w0:A');
    lane.setWorkerConcurrency(false);
    setEngineReady(false, 'booting');
    assert.equal(await lane.runOnWorkerLane(async l => l.worker!.id, { eligible: () => false }), 'w0');
  } finally {
    workers.setEngineWorkersForTest(table);
  }
});

test('backend evictions are per worker', async () => {
  _resetBackendReleases();
  const unload = deferred();
  await workers.runWithWorker(w1, async () => { trackBackendRelease('yue2', () => unload.promise); });
  await awaitBackendRelease();  // w0 is not held up by w1's unload
  let w1Waited = false;
  const w1Wait = workers.runWithWorker(w1, () => awaitBackendRelease()).then(() => { w1Waited = true; });
  await tick();
  assert.equal(w1Waited, false);
  unload.resolve();
  await w1Wait;
  _resetBackendReleases();
});

test('training barrier: suspend waits for every worker and holds the lane until resume', async () => {
  lane.setWorkerConcurrency(true);
  const order: string[] = [];
  const g0 = deferred(), g1 = deferred();
  const j0 = lane.runOnWorkerLane(async () => { await g0.promise; order.push('w0 job done'); });
  const j1 = lane.runOnWorkerLane(async () => { await g1.promise; order.push('w1 job done'); });
  await tick();
  const suspended = proc.suspendEngines('Paused for test training').then(ok => { order.push(`suspended:${ok}`); });
  const late = lane.runOnWorkerLane(async l => { order.push(`late job:${l.worker!.id}`); });
  g0.resolve();
  await tick(); await tick();
  assert.deepEqual(order, ['w0 job done'], 'training stopped the engines while w1 was still rendering');
  g1.resolve();
  await suspended;
  await tick();
  assert.deepEqual(order, ['w0 job done', 'w1 job done', 'suspended:true'], 'a job started on a card during training');
  assert.equal(proc.isEngineSuspended(), true);
  assert.equal(w1.ready, false, 'the secondary kept running through training');
  assert.equal(await proc.resumeEngines(), true);
  await Promise.all([j0, j1, late]);
  assert.equal(order.at(-1)?.startsWith('late job:'), true);
  assert.equal(w1.ready, true);
});

test('training barrier waits for a YuE2 render released early to drain', async () => {
  lane.setWorkerConcurrency(true);
  const order: string[] = [];
  const render = deferred();
  const job = lane.runOnWorkerLane(async l => {
    lane.releaseGpuLane(l);
    await render.promise;
    await lane.runOnGpuLane(async l2 => { order.push(`finish:${l2.worker?.id}`); });
    order.push('render settled');
  }, { family: 'yue2' });
  await tick();
  const suspended = proc.suspendEngines('Paused for test training').then(() => { order.push('suspended'); });
  await tick(); await tick();
  assert.deepEqual(order, [], 'training stopped the engine under a draining YuE2 render');
  render.resolve();
  await Promise.all([job, suspended]);
  assert.deepEqual(order, ['finish:w0', 'render settled', 'suspended']);
  assert.equal(await proc.resumeEngines(), true);
});

test('a backend switch during a worker job evicts that worker only after the job', async () => {
  _resetBackendReleases();
  lane.setWorkerConcurrency(true);
  const order: string[] = [];
  const gate = deferred();
  const onW1 = { eligible: (w: { id: string }) => w.id === 'w1' };
  const job = lane.runOnWorkerLane(async () => { await gate.promise; order.push('w1 job done'); }, onW1);
  await tick();
  releaseOnEveryWorker('mm3', async () => { order.push(`release:${workers.currentWorker().id}:${await engineHit()}`); });
  await until(() => order.length === 1, 'the idle worker to release');
  assert.deepEqual(order, ['release:w0:A'], 'the busy worker was evicted mid-job');
  const next = lane.runOnWorkerLane(async l => { await awaitBackendRelease(); order.push(`next job:${l.worker!.id}`); }, onW1);
  gate.resolve();
  await Promise.all([job, next]);
  await until(() => !lane.gpuLaneBusy(), 'the releases to settle');
  assert.deepEqual(order, ['release:w0:A', 'w1 job done', 'release:w1:B', 'next job:w1']);
  _resetBackendReleases();
});
