// sseBridge.test.ts — coalescing, validated updates, replay and teardown.
// Run with the ui suite: node --import ../server/node_modules/tsx/dist/loader.mjs --test "src/**/*.test.ts"
import assert from 'node:assert/strict';
import test, { after } from 'node:test';
import { createQueryClient as makeClient } from './client';

// gcTime keeps a 5 min timer per query; clear every client so node can exit.
const made: ReturnType<typeof makeClient>[] = [];
const createQueryClient = () => { const c = makeClient(); made.push(c); return c; };
after(() => made.forEach(c => c.clear()));
import { serverKey } from './keys';
import { bridgeEvents } from './sseBridge';

type Sub = { message?: (d: string) => void; error?: () => void; open?: () => void };

/** A stand-in for subscribeEventSource that the test drives by hand. */
function fakeSource() {
  const subs = new Map<string, Set<Sub>>();
  return {
    subscribe(url: string, sub: Sub) {
      if (!subs.has(url)) subs.set(url, new Set());
      subs.get(url)!.add(sub);
      return () => { subs.get(url)!.delete(sub); };
    },
    send(url: string, event: unknown) { for (const s of subs.get(url) ?? []) s.message?.(JSON.stringify(event)); },
    drop(url: string) { for (const s of subs.get(url) ?? []) s.error?.(); },
    open(url: string) { for (const s of subs.get(url) ?? []) s.open?.(); },
    count(url: string) { return subs.get(url)?.size ?? 0; },
  };
}
const wait = (ms: number) => new Promise(r => setTimeout(r, ms));

type JobEvent = { type: string; jobId: string; progress?: number };
const jobKey = (worker: string | null, id: string) => serverKey('training-runs/job', { worker, id });
const jobsKey = (worker: string | null) => serverKey('training-runs/jobs', { worker });

function bridgeFor(client: ReturnType<typeof createQueryClient>, src: ReturnType<typeof fakeSource>, worker: string | null, onDone: () => void) {
  return bridgeEvents<JobEvent>(client, {
    url: `/stream/${worker}`,
    subscribe: src.subscribe,
    coalesceMs: 10,
    reconnectInvalidate: [jobsKey(worker)],
    map: e => {
      if (e.type === 'progress') return { update: [{ key: jobKey(worker, e.jobId), apply: (old) => {
        const o = old as { progress: number };
        return (e.progress ?? 0) >= o.progress ? { ...o, progress: e.progress } : undefined;
      } }] };
      if (e.type === 'done') return { invalidate: [jobsKey(worker), jobKey(worker, e.jobId)], once: { id: `${worker}:${e.jobId}:done`, run: onDone } };
    },
  });
}

function countInvalidations(client: ReturnType<typeof createQueryClient>) {
  const seen: string[] = [];
  const real = client.invalidateQueries.bind(client);
  client.invalidateQueries = ((filters?: { queryKey?: readonly unknown[] }) => {
    seen.push(JSON.stringify(filters?.queryKey));
    return real(filters);
  }) as typeof client.invalidateQueries;
  return seen;
}

test('a burst of events coalesces to one invalidation per key', async () => {
  const client = createQueryClient(), src = fakeSource();
  const seen = countInvalidations(client);
  const stop = bridgeFor(client, src, 'A', () => {});
  for (const id of ['j1', 'j1', 'j1']) src.send('/stream/A', { type: 'done', jobId: id });
  await wait(30);
  assert.equal(seen.length, 2);
  stop();
});

test('updates apply only to existing entries and only when validated', () => {
  const client = createQueryClient(), src = fakeSource();
  const stop = bridgeFor(client, src, 'A', () => {});
  client.setQueryData(jobKey('A', 'j1'), { progress: 0.5 });
  src.send('/stream/A', { type: 'progress', jobId: 'j1', progress: 0.7 });
  src.send('/stream/A', { type: 'progress', jobId: 'j1', progress: 0.6 });  // stale: rejected
  src.send('/stream/A', { type: 'progress', jobId: 'j2', progress: 0.1 });  // no entry: not created
  assert.deepEqual(client.getQueryData(jobKey('A', 'j1')), { progress: 0.7 });
  assert.equal(client.getQueryData(jobKey('A', 'j2')), undefined);
  stop();
});

test('a replayed terminal event never repeats its side effect', () => {
  const client = createQueryClient(), src = fakeSource();
  let done = 0;
  const stop = bridgeFor(client, src, 'A', () => done++);
  src.send('/stream/A', { type: 'done', jobId: 'j1' });
  src.drop('/stream/A'); src.open('/stream/A');
  src.send('/stream/A', { type: 'done', jobId: 'j1' });                     // server replay on reconnect
  stop();
  const again = bridgeFor(client, src, 'A', () => done++);                  // remount, replay on join
  src.send('/stream/A', { type: 'done', jobId: 'j1' });
  assert.equal(done, 1);
  again();
});

test('reconnect invalidates authoritative snapshots; the first open does not', async () => {
  const client = createQueryClient(), src = fakeSource();
  const seen = countInvalidations(client);
  const stop = bridgeFor(client, src, 'A', () => {});
  src.open('/stream/A');
  await wait(20);
  assert.equal(seen.length, 0);
  src.drop('/stream/A'); src.open('/stream/A');
  await wait(20);
  assert.deepEqual(seen, [JSON.stringify(jobsKey('A'))]);
  stop();
});

test('teardown unsubscribes, drops pending invalidations and ignores late events', async () => {
  const client = createQueryClient(), src = fakeSource();
  const seen = countInvalidations(client);
  let done = 0;
  const stop = bridgeFor(client, src, 'A', () => done++);
  src.send('/stream/A', { type: 'done', jobId: 'j1' });
  stop();
  assert.equal(src.count('/stream/A'), 0);
  await wait(30);
  assert.equal(seen.length, 0);
  assert.equal(done, 1);  // the effect ran when the event arrived, before teardown
  stop();                 // idempotent
});

test('after a worker switch, a delayed event for the old worker touches nothing', async () => {
  const client = createQueryClient(), src = fakeSource();
  let late: Sub | undefined;
  const capture = (url: string, sub: Sub) => { if (url === '/stream/A') late = sub; return src.subscribe(url, sub); };
  const stopA = bridgeEvents<JobEvent>(client, {
    url: '/stream/A', subscribe: capture, coalesceMs: 1,
    map: e => ({ invalidate: [jobsKey('A')], update: [{ key: jobKey('A', e.jobId), apply: () => ({ progress: 1 }) }] }),
  });
  stopA();
  const stopB = bridgeFor(client, src, 'B', () => {});
  client.setQueryData(jobKey('A', 'j1'), { progress: 0 });
  client.setQueryData(jobKey('B', 'j1'), { progress: 0 });
  const seen = countInvalidations(client);
  late!.message!(JSON.stringify({ type: 'progress', jobId: 'j1', progress: 1 }));
  await wait(20);
  assert.deepEqual(client.getQueryData(jobKey('A', 'j1')), { progress: 0 });
  assert.deepEqual(client.getQueryData(jobKey('B', 'j1')), { progress: 0 });
  assert.equal(seen.length, 0);
  stopB();
});
