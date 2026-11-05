// Two servers on one database and one training folder (dev and prod copies).
// A second real process (instanceOwner.fixture.ts) owns one job of each kind;
// this process must leave them alone while it lives and recover them once it
// is dead. Covers the startup sweeps, the polling paths, claims and cancels.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import Database from 'better-sqlite3';

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'instance-owner-'));
process.env.TRAINING_DIR = path.join(dir, 'training');
const dbPath = path.join(dir, 'shared.db');

const { applyConnectionPragmas } = await import('../db/database.js');
const instance = await import('./instance.js');
const { AudioIntentQueue } = await import('./audioQueue/intentQueue.js');
const { WorkflowJobs, WorkflowError } = await import('./workflows/workflowJobs.js');
const { GenerationJobLedger, ledgerStatusBody } = await import('./generation/jobLedger.js');
const training = await import('./training/labelingQueue.js');

const db = new Database(dbPath);
applyConnectionPragmas(db);
instance.ensureInstanceSchema(db);

const here = path.dirname(fileURLToPath(import.meta.url));
const child = spawn(process.execPath, ['--import', 'tsx/esm', path.join(here, 'instanceOwner.fixture.ts')], {
  cwd: path.join(here, '..', '..'), env: { ...process.env, OWNER_DB: dbPath }, stdio: ['ignore', 'pipe', 'inherit'],
});
const exited = new Promise<void>(resolve => child.on('exit', () => resolve()));
const made = await new Promise<{ owner: { instance: string; pid: number }; audio: string; workflow: string; jobs: Record<string, string> }>((resolve, reject) => {
  let out = '';
  child.stdout!.on('data', chunk => {
    out += chunk;
    const line = out.split('\n').find(l => l.startsWith('{'));
    if (line) resolve(JSON.parse(line));
  });
  child.on('exit', code => reject(new Error(`fixture exited early (${code})`)));
});
instance.startInstance(db, { copy: 'parent:1', heartbeat: false });

test.after(async () => {
  child.kill();
  await exited;
  instance.resetInstanceForTests();
  db.close();
  fs.rmSync(dir, { recursive: true, force: true });
});

const audioQueue = () => new AudioIntentQueue({
  db, submit: async () => ({ ok: true, jobId: 'parent-gen' }), getJob: () => undefined,
  cancelJob: () => false, activeEngine: () => 'ace', isEngine: () => true,
});
const workflowJobs = () => new WorkflowJobs({ db, audio: { enqueue: () => { throw new Error('unused'); }, get: () => { throw new Error('unused'); }, cancel: () => { throw new Error('unused'); } } });
const meta = (id: string) => JSON.parse(fs.readFileSync(path.join(process.env.TRAINING_DIR!, 'jobs', id, '_meta.json'), 'utf8'));
const audioRow = (id: string) => db.prepare('SELECT status, cancel_requested FROM audio_intents WHERE id = ?').get(id) as { status: string; cancel_requested: number };

test('a live owner in another process is left alone by every sweep, poll, claim and cancel', async () => {
  assert.equal(instance.ownerAlive(made.owner), true);

  // Audio: the startup reconcile and a polling pass leave the item submitted;
  // a cancel here only flags it for its owner.
  const audio = audioQueue();
  assert.equal(audio.reconcileAfterRestart(), 0);
  await audio.tick();
  assert.equal(audioRow(made.audio).status, 'submitted');
  audio.cancel(made.audio);
  assert.deepEqual(audioRow(made.audio), { status: 'submitted', cancel_requested: 1 });

  // Workflow: not interrupted, and a cancel from here is refused.
  const workflows = workflowJobs();
  assert.equal(workflows.reconcileAfterRestart(), 0);
  workflows.pump();
  assert.equal(workflows.get(made.workflow).status, 'running');
  assert.throws(() => workflows.cancel(made.workflow), (err: unknown) => err instanceof WorkflowError && err.status === 409);

  // Generation ledger: the row stays open and reads as running elsewhere.
  const ledger = new GenerationJobLedger(db);
  const entry = ledger.get('child-gen')!;
  assert.equal(entry.status, 'open');
  assert.equal(ledgerStatusBody(entry).status, 'running');

  // Training and label jobs: kept running; the datasets are busy here too.
  training.recoverStaleJobs();
  for (const kind of ['train-dit', 'label']) assert.equal(meta(made.jobs[kind]).status, 'running', kind);
  assert.equal(training.activeJobForDataset('ds-train')?.id, made.jobs['train-dit']);
  assert.equal(training.activeJobForDataset('ds-label')?.id, made.jobs.label);
  assert.throws(() => training.createJob('build', 'ds-label', [], {}), /busy in the other HOT-Step copy \(child:1/);
  assert.match(training.jobRunsElsewhere(made.jobs['train-dit']) ?? '', /running in the other HOT-Step copy/);

  // A cancelled trainer still exiting keeps its dataset until cleanup ends.
  assert.deepEqual([meta(made.jobs.stopping).status, meta(made.jobs.stopping).cleanupPending], ['cancelled', true]);
  assert.equal(training.activeJobForDataset('ds-cleanup')?.id, made.jobs.stopping);
  assert.throws(() => training.createJob('build', 'ds-cleanup', [], {}), /busy in the other HOT-Step copy/);

  // Legacy rule: a running record with no owner predates ownership, so its
  // process is gone. It is recovered even while another copy is alive.
  const legacy = path.join(process.env.TRAINING_DIR!, 'jobs', 'legacy-job');
  fs.mkdirSync(legacy, { recursive: true });
  fs.writeFileSync(path.join(legacy, '_meta.json'), JSON.stringify({ id: 'legacy-job', datasetId: 'ds-old', kind: 'build', status: 'running', createdAt: 1 }));
  training.recoverStaleJobs();
  assert.deepEqual([meta('legacy-job').status, meta('legacy-job').error], ['failed', 'Server restarted']);
});

test('once the owner is dead, polling and startup sweeps recover its jobs', async () => {
  child.kill();
  await exited;
  assert.equal(instance.ownerAlive(made.owner), false, 'a dead PID is a dead owner, before its heartbeat lapses');

  // Audio: the next polling pass interrupts it.
  await audioQueue().tick();
  assert.equal(audioRow(made.audio).status, 'interrupted');

  // Workflow: the next pump interrupts it.
  const workflows = workflowJobs();
  workflows.pump();
  assert.equal(workflows.get(made.workflow).status, 'interrupted');

  // Generation ledger: the next startup marks it interrupted.
  assert.equal(new GenerationJobLedger(db).get('child-gen')?.status, 'interrupted');

  // Training and label jobs: failed at startup; the datasets are free.
  training.recoverStaleJobs();
  for (const kind of ['train-dit', 'label']) {
    assert.deepEqual([meta(made.jobs[kind]).status, meta(made.jobs[kind]).error], ['failed', 'Server restarted'], kind);
  }
  assert.equal(training.activeJobForDataset('ds-train'), undefined);
  assert.equal(training.activeJobForDataset('ds-cleanup'), undefined, 'the pending cleanup of a dead owner frees the dataset');
  assert.equal(training.jobRunsElsewhere(made.jobs['train-dit']), null);
  const job = training.createJob('build', 'ds-train', [], {});
  assert.equal(training.activeJobForDataset('ds-train')?.id, job.id);
  training.finishJob(job, 'done');
});

test('a stale heartbeat is a dead owner even when its PID is alive (PID reuse)', () => {
  // This process's own PID, under an instance id whose last beat is old.
  db.prepare(`INSERT INTO server_instances (id, pid, host, copy, started_at, heartbeat_at) VALUES ('reused', ?, ?, 'x:1', 0, ?)`)
    .run(process.pid, os.hostname(), Date.now() - instance.STALE_MS - 1000);
  assert.equal(instance.ownerAlive({ instance: 'reused', pid: process.pid }), false);
  db.prepare(`UPDATE server_instances SET heartbeat_at = ? WHERE id = 'reused'`).run(Date.now());
  assert.equal(instance.ownerAlive({ instance: 'reused', pid: process.pid }), true);
  db.prepare(`UPDATE server_instances SET stopped_at = ? WHERE id = 'reused'`).run(Date.now());
  assert.equal(instance.ownerAlive({ instance: 'reused', pid: process.pid }), false, 'a clean stop frees it at once');
});
