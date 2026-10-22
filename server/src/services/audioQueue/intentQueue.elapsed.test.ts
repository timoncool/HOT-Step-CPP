// intentQueue.elapsed.test.ts — elapsedSeconds is computed authoritatively by
// the server's own follow() loop, never inferred from a client's poll
// cadence. A client (ui/src/stores/audioGenQueueStore.ts) used to run its own
// timer against whatever status it happened to observe, which missed a job
// that started and finished between two of its polls (F19 review). The
// server's follow() loop runs on its own fixed interval regardless of any
// browser being open, so it cannot miss a transition the way a client can —
// see intentQueue.test.ts for the harness shape this borrows.
//
// Round 2 (F19): the frozen value itself must come from the job's own
// lifecycle timestamps (GenerationAttempt.startedAt/endedAt, routes/
// generate.ts), never from this queue's own poll cadence or submission
// timestamp — a job that sat queued for 600s and ran for 1s must read 1s,
// not 601s, and a terminal row must never resume counting after it froze.
import test from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { AudioIntentQueue, ensureAudioQueueSchema, type AudioQueueDeps } from './intentQueue.js';

type Attempt = { startedAt: number; endedAt?: number };
type Job = { status: string; error?: string; result?: unknown; attempts?: Attempt[] };

/** Same harness shape as intentQueue.test.ts, but with a clock the test
 *  drives directly instead of one keyed to submit() calls — elapsedSeconds
 *  needs distinct timestamps across ticks that don't submit anything new. */
function harness(db = new Database(':memory:')) {
  const jobs = new Map<string, Job>();
  let next = 0;
  let time = 1000;
  const deps: AudioQueueDeps = {
    db,
    submit: async () => { const jobId = `job-${++next}`; jobs.set(jobId, { status: 'pending' }); return { ok: true, jobId }; },
    getJob: id => jobs.get(id),
    cancelJob: id => { const j = jobs.get(id); if (!j) return false; j.status = 'cancelled'; return true; },
    activeEngine: () => 'yue2',
    isEngine: id => ['ace', 'minimax-m3', 'yue2'].includes(id),
    now: () => time,
  };
  return { db, deps, jobs, queue: new AudioIntentQueue(deps), advance: (ms: number) => { time += ms; } };
}

const req = (extra: Record<string, unknown> = {}) => ({ caption: 'c', lyrics: 'l', seed: 1, expectedBackend: 'yue2', ...extra });

test('elapsedSeconds is null while queued, counts up once running, freezes on success from the job\'s own timestamps', async () => {
  const h = harness();
  const item = h.queue.enqueue({ idempotencyKey: 'a', request: req() }, 'u').item;
  await h.queue.tick(); // claims + submits -> job-1, status 'pending'
  assert.equal(h.queue.get(item.id).elapsedSeconds, null, 'still queued at the engine, not generating yet');

  h.jobs.get('job-1')!.status = 'lm_running';
  h.jobs.get('job-1')!.attempts = [{ startedAt: h.deps.now!() }];
  h.advance(2000);
  await h.queue.tick(); // follow() sees it running for the first time -> started_at set here (live ticker only)
  assert.equal(h.queue.get(item.id).elapsedSeconds, 0, 'just started; no real time has passed yet');

  h.advance(5000);
  assert.equal(h.queue.get(item.id).elapsedSeconds, 5, 'live count reflects real time since the engine actually started');

  h.jobs.get('job-1')!.status = 'succeeded';
  h.jobs.get('job-1')!.result = { audioUrls: ['/a.wav'] };
  h.jobs.get('job-1')!.attempts![0].endedAt = h.deps.now!() + 3000;
  h.advance(3000);
  await h.queue.tick(); // follow() detects success and freezes elapsedSeconds here
  const frozen = h.queue.get(item.id).elapsedSeconds;
  // The job's own startedAt (1000) predates our first 'running' observation
  // (3000, after the 2000ms tick gap above) — frozen value is 10s, the full
  // real runtime, not 8s (which is what our own poll-based started_at would
  // have undercounted it to).
  assert.equal(frozen, 10, 'frozen from the job\'s own startedAt/endedAt, including time our own poll missed');

  h.advance(60_000);
  assert.equal(h.queue.get(item.id).elapsedSeconds, frozen, 'a completed row must not keep counting up forever');
});

test('a job that goes straight to terminal with no recorded attempt has no elapsedSeconds, never a submission-time guess', async () => {
  // The race the client-side fix could not close: job.status is already
  // terminal the very first time anything looks at it, so follow() never
  // observes a 'running' status itself. Previously this fell back to the
  // submission timestamp; Lead's F19 round-2 decision: no observed running
  // state and no job timestamps means null, not a wrong number.
  const h = harness();
  const item = h.queue.enqueue({ idempotencyKey: 'a', request: req() }, 'u').item;
  await h.queue.tick(); // submits -> job-1, 'pending'
  h.jobs.get('job-1')!.status = 'succeeded';
  h.jobs.get('job-1')!.result = { audioUrls: ['/a.wav'] };
  // No attempts recorded: the engine never told us when it actually started.
  h.advance(4000);
  await h.queue.tick();
  assert.equal(h.queue.get(item.id).elapsedSeconds, null, 'no observed running state and no job timestamps: hide the badge');
});

test('elapsedSeconds never counts queue wait: 600s queued, 1s running reads as 1s, not 601s', async () => {
  const h = harness();
  const item = h.queue.enqueue({ idempotencyKey: 'a', request: req() }, 'u').item;
  await h.queue.tick(); // submits -> job-1, 'pending'
  h.advance(600_000); // 600s queued at the engine; our own poll never observes it running
  const ranAt = h.deps.now!();
  h.jobs.get('job-1')!.status = 'succeeded';
  h.jobs.get('job-1')!.result = { audioUrls: ['/a.wav'] };
  h.jobs.get('job-1')!.attempts = [{ startedAt: ranAt, endedAt: ranAt + 1000 }];
  h.advance(1000);
  await h.queue.tick();
  assert.equal(h.queue.get(item.id).elapsedSeconds, 1, 'the job itself only ran for 1s; queue wait must never be counted');
});

test('elapsedSeconds freezes on cancel mid-run from the job\'s own startedAt, and retry clears it for the new attempt', async () => {
  const h = harness();
  const item = h.queue.enqueue({ idempotencyKey: 'a', request: req() }, 'u').item;
  await h.queue.tick(); // submits -> job-1, 'pending'
  h.jobs.get('job-1')!.status = 'synth_running';
  h.jobs.get('job-1')!.attempts = [{ startedAt: h.deps.now!() }];
  h.advance(6000);
  h.queue.cancel(item.id);
  assert.equal(h.queue.get(item.id).status, 'cancelled');
  const frozen = h.queue.get(item.id).elapsedSeconds;
  assert.equal(frozen, 6, 'frozen at cancel time, from the job\'s own startedAt, not our poll cadence');
  h.advance(60_000);
  assert.equal(h.queue.get(item.id).elapsedSeconds, frozen, 'a cancelled row must not keep counting either');

  h.queue.retry(item.id);
  assert.equal(h.queue.get(item.id).elapsedSeconds, null, 'a fresh attempt starts with a clean clock, not the last attempt\'s frozen time');
});

test('cancelling an item that never started running leaves elapsedSeconds null, not a queue-wait number', async () => {
  const h = harness();
  const item = h.queue.enqueue({ idempotencyKey: 'a', request: req() }, 'u').item;
  await h.queue.tick(); // submits -> job-1, still 'pending' (queued) at the engine
  h.advance(600_000);
  h.queue.cancel(item.id);
  assert.equal(h.queue.get(item.id).status, 'cancelled');
  assert.equal(h.queue.get(item.id).elapsedSeconds, null, 'generation never actually started; nothing to show');
});

test('reconcileAfterRestart freezes elapsed time instead of leaving it to grow forever', () => {
  const h = harness();
  const item = h.queue.enqueue({ idempotencyKey: 'a', request: req() }, 'u').item;
  h.db.prepare(`UPDATE audio_intents SET status = 'submitted', job_id = 'job-1', started_at = ? WHERE id = ?`).run(h.deps.now!(), item.id);
  h.advance(40_000); // the process dies 40s into the render
  const n = h.queue.reconcileAfterRestart();
  assert.equal(n, 1);
  const frozen = h.queue.get(item.id).elapsedSeconds;
  assert.equal(frozen, 40, 'frozen at the moment the server noticed it had been running, not left ticking');
  h.advance(60_000); // a minute passes before anyone looks again
  assert.equal(h.queue.get(item.id).elapsedSeconds, frozen, 'the same finished item must never grow — the exact 2s-to-62s regression');
});

test('reconcileAfterRestart with no observed running state leaves elapsedSeconds null', () => {
  const h = harness();
  const item = h.queue.enqueue({ idempotencyKey: 'a', request: req() }, 'u').item;
  h.db.prepare(`UPDATE audio_intents SET status = 'submitting' WHERE id = ?`).run(item.id);
  const n = h.queue.reconcileAfterRestart();
  assert.equal(n, 1);
  assert.equal(h.queue.get(item.id).elapsedSeconds, null, 'it never reached the engine; nothing ran');
});

test('ensureAudioQueueSchema adds started_at/elapsed_seconds to a database that predates them', () => {
  const db = new Database(':memory:');
  // Build the pre-F19 table by hand, missing the two new columns.
  db.exec(`CREATE TABLE audio_intents (
    id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL UNIQUE, user_id TEXT NOT NULL, status TEXT NOT NULL,
    engine TEXT NOT NULL, request TEXT NOT NULL, request_hash TEXT NOT NULL, meta TEXT, job_id TEXT,
    attempt INTEGER NOT NULL DEFAULT 1, previous_job_ids TEXT NOT NULL DEFAULT '[]', waiting TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0, error TEXT, result TEXT,
    created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
  )`);
  db.prepare(`INSERT INTO audio_intents (id, idempotency_key, user_id, status, engine, request, request_hash, created_at, updated_at)
    VALUES ('old-1', 'k1', 'u', 'succeeded', 'ace', '{}', 'h', 1, 1)`).run();
  ensureAudioQueueSchema(db); // must not throw, and must not drop the pre-existing row
  const cols = (db.prepare(`SELECT name FROM pragma_table_info('audio_intents')`).all() as { name: string }[]).map(c => c.name);
  assert.ok(cols.includes('started_at') && cols.includes('elapsed_seconds'));
  assert.equal((db.prepare(`SELECT COUNT(*) AS c FROM audio_intents`).get() as { c: number }).c, 1);
  // Re-running on an already-migrated database must stay a no-op, not error.
  ensureAudioQueueSchema(db);
});
