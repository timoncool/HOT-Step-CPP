// Fixture for instance.test.ts, run as its own process: a second server on
// the shared database and training folder. It owns one job of each kind
// (audio item, workflow job, generation ledger row, training job, label job),
// prints what it made, then stays alive until the test kills it.

import Database from 'better-sqlite3';
import { z } from 'zod/v4';
import { applyConnectionPragmas } from '../db/database.js';
import { owner, startInstance } from './instance.js';
import { AudioIntentQueue } from './audioQueue/intentQueue.js';
import { WorkflowJobs } from './workflows/workflowJobs.js';
import { GenerationJobLedger } from './generation/jobLedger.js';
import * as training from './training/labelingQueue.js';

const db = new Database(process.env.OWNER_DB!);
applyConnectionPragmas(db);
startInstance(db, { copy: 'child:1' });

const audio = new AudioIntentQueue({
  db, submit: async () => ({ ok: true, jobId: 'child-gen' }), getJob: () => ({ status: 'running' }),
  cancelJob: () => true, activeEngine: () => 'ace', isEngine: () => true,
});
const item = audio.enqueue({ idempotencyKey: 'child-audio', request: { caption: 'x' } }, 'u').item;
await audio.tick();

const workflows = new WorkflowJobs({ db, audio: {
  enqueue: (input, userId) => audio.enqueue(input, userId), get: id => audio.get(id), cancel: id => audio.cancel(id),
} });
workflows.register({ kind: 'hang', input: z.object({}), run: () => new Promise(() => {}) });
const workflow = workflows.submit({ kind: 'hang', idempotencyKey: 'child-wf', input: {} }, 'u').job;

new GenerationJobLedger(db).opened({ id: 'child-gen', userId: 'u', createdAt: Date.now() });

const jobs: Record<string, string> = {};
for (const [kind, dataset] of [['train-dit', 'ds-train'], ['label', 'ds-label']] as const) {
  const job = training.createJob(kind, dataset, [], {});
  job.status = 'running';
  job.startedAt = Date.now();
  training.emitJob(job);
  jobs[kind] = job.id;
}
// A graceful trainer that was cancelled but has not exited yet: the job is
// over, the dataset is not free until its cleanup releases it.
const stopping = training.createJob('heartmula-train', 'ds-cleanup', [], {});
stopping.status = 'running';
stopping.cleanupPending = true;
stopping.cancelChild = () => {};
training.emitJob(stopping);
training.cancelJob(stopping.id);
jobs.stopping = stopping.id;

console.log(JSON.stringify({ owner: owner(), audio: item.id, workflow: workflow.id, jobs }));
setInterval(() => {}, 60_000);
