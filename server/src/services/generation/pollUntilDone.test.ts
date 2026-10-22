import assert from 'node:assert/strict';
import { test, type TestContext } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import type { AceJobStatus } from '../aceClient.js';
import type { GenerationJob } from './jobTypes.js';
// Engine lifecycle imports also load training recovery. Keep it off real jobs.
const trainingDir = fs.mkdtempSync(path.resolve('.watchdog-test-'));
process.env.TRAINING_DIR = trainingDir;
const { aceClient } = await import('../aceClient.js');
const { pollUntilDone } = await import('./pollUntilDone.js');
test.after(() => fs.rmSync(trainingDir, { recursive: true, force: true }));

function clock(t: TestContext, phase = 'nar', finishAt = Infinity, ticking = false) {
  let now = 1_000_000;
  const start = now;
  t.mock.method(Date, 'now', () => now);
  t.mock.method(globalThis, 'setTimeout', (callback: () => void) => {
    now += 1000;
    callback();
    return {};
  });
  t.mock.method(aceClient, 'pollJob', async () => ({
    status: now - start >= finishAt ? 'done' : 'running',
    phase, phase_step: ticking ? Math.floor((now - start) / 700_000) : 1, phase_total: 32,
  } as AceJobStatus));
  t.mock.method(console, 'warn', () => {});
  const cancel = t.mock.method(aceClient, 'cancelJob', async () => {});
  const job = { status: 'running', stage: 'YuE2: Rendering: Step 1/32' } as GenerationJob;
  return { cancel, job, elapsed: () => now - start };
}

test('a long NAR chunk survives 900 seconds of silence, then returns its result', async t => {
  const { cancel, job } = clock(t, 'nar', 1200_000);
  await pollUntilDone('long-nar', job, new AbortController().signal, 45, { narChunkFrames: 4500 });
  assert.equal(cancel.mock.callCount(), 0);
});

test('a frozen NAR step exhausts its frame-scaled allowance', async t => {
  const { cancel, job, elapsed } = clock(t);
  await assert.rejects(pollUntilDone('hung-nar', job, new AbortController().signal, 45,
    { narChunkFrames: 4500 }), /Generation stalled/);
  assert.ok(elapsed() >= 1920_000 && elapsed() < 1980_000);
  assert.equal(cancel.mock.callCount(), 1);
});

test('the wall-clock limit still catches an uncapped hung NAR chunk', async t => {
  const { cancel, job, elapsed } = clock(t);
  await assert.rejects(pollUntilDone('wall-limit', job, new AbortController().signal, 5,
    { narChunkFrames: 9000 }), /Generation timed out \(5 min limit\)/);
  assert.ok(elapsed() < 310_000);
  assert.equal(cancel.mock.callCount(), 1);
});

test('NAR options leave other phases on the existing quiet window', async t => {
  const { job, elapsed } = clock(t, 'loading_dit');
  await assert.rejects(pollUntilDone('other-stage', job, new AbortController().signal, 45,
    { narChunkFrames: 9000 }), /Generation stalled/);
  assert.ok(elapsed() >= 900_000 && elapsed() < 960_000);
});

test('real ODE step updates reset the watchdog across a long solve', async t => {
  const { cancel, job } = clock(t, 'nar', 2400_000, true);
  await pollUntilDone('steps', job, new AbortController().signal, 45, { narChunkFrames: 1500 });
  assert.equal(cancel.mock.callCount(), 0);
});
