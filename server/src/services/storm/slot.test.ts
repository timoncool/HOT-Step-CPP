// slot.test.ts -- Storm slot builder (YuE2 drift, SA3 resolver parity) and
// GPU lane ownership for a slot render.
// Run: cd server && npx tsx --test src/services/storm/slot.test.ts

import { strict as assert } from 'node:assert';
import { after, describe, it } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import type { StreamControlSnapshot, Sa3SlotResult } from './slot.js';
import type { DriftRuntime } from './drift.js';

// mapYue2Params reads the persisted YuE2 pick, so the YuE2 cases need a DB.
const DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'storm-slot-'));
process.env.DATA_DIR = DATA_DIR;
const database = await import('../../db/database.js');
database.initDb();
after(() => { database.closeDb(); fs.rmSync(DATA_DIR, { recursive: true, force: true }); });

const { buildSlotRequest, renderSlotOnLane, waitForSlotJob } = await import('./slot.js');
const { initDrift, YUE2_PARAM_SPECS } = await import('./drift.js');
const { gpuLaneBusy, runOnGpuLane } = await import('../generation/gpuLane.js');
const { SA3_DEFAULT_STEPS, SA3_DEFAULT_SOLVER, SA3_DEFAULT_SEED_BLEND } = await import('../backends/sa3/generate.js');
const { SA3_MAX_DURATION_SEC } = await import('../backends/sa3/index.js');

function ctrl(over: Partial<StreamControlSnapshot> = {}): StreamControlSnapshot {
  return {
    guidanceScale: 7, lssStrength: 0, inferenceSteps: 0, duration: 0, bpm: 0,
    seedLock: false, nextSeed: null, nextPrompt: null, nextLyrics: null,
    nextBpm: null, nextDuration: null, baseCaption: null, baseLyrics: null,
    baseInstrumental: null, streamPaused: false, extraPluginParams: {},
    inferMethod: null, scheduler: null, guidanceMode: null,
    ...over,
  };
}

function sa3(base: Record<string, unknown>, over: Partial<StreamControlSnapshot> = {}): Sa3SlotResult {
  const r = buildSlotRequest({ caption: 'warm synth pad', ...base }, ctrl(over), 0, 1234, 'sa3');
  assert.equal(r.kind, 'sa3');
  return r as Sa3SlotResult;
}

describe('YuE2 slot cfg_scale', () => {
  const base = { caption: 'indie rock', lyrics: '[verse]\nla la', instrumental: false };

  it('without drift, the guidance knob reaches cfg_scale', () => {
    const r = buildSlotRequest(base, ctrl({ guidanceScale: 7 }), 0, 1, 'yue2');
    assert.equal(r.kind, 'yue2');
    if (r.kind === 'yue2') assert.equal(r.yue2Req.cfg_scale, 7);
  });

  it('a drifted yue2CfgScale reaches mapping.req.cfg_scale', () => {
    const spec = YUE2_PARAM_SPECS.find(s => s.key === 'yue2CfgScale')!;
    const drift: DriftRuntime = {
      state: initDrift(42),
      pressure: {
        targets: { yue2CfgScale: 1 }, targetCaption: [], captionPool: [],
        pull: 1, rate: 0.5, jitter: 0, locks: {}, seedMode: 'lock',
      },
    };
    const r = buildSlotRequest(base, ctrl({ guidanceScale: 7, drift }), 0, 1, 'yue2');
    assert.equal(r.kind, 'yue2');
    // Starts from the knob (7), moves one capped step (0.125 of the range) toward max.
    const expected = 7 + 0.125 * (spec.max - spec.min);
    if (r.kind === 'yue2') assert.ok(Math.abs(Number(r.yue2Req.cfg_scale) - expected) < 1e-9,
      `cfg_scale ${r.yue2Req.cfg_scale}, expected ${expected}`);
  });
});

describe('SA3 slot matches normal generation resolvers', () => {
  it('defaults: 8 steps, pingpong wire with the default Lua solver, no seed B', () => {
    const r = sa3({});
    assert.equal(r.steps, SA3_DEFAULT_STEPS);
    assert.equal(r.sampler, 'pingpong');
    assert.equal(r.solver, SA3_DEFAULT_SOLVER);
    assert.equal(r.seedB, undefined);
    assert.equal(r.seedBlend, undefined);
    assert.equal(r.durationSec, 30);
  });

  it('explicit overrides: euler, steps, plugin solver', () => {
    const e = sa3({ sa3Sampler: 'euler', sa3Steps: 12 });
    assert.equal(e.sampler, 'euler');
    assert.equal(e.solver, undefined);
    assert.equal(e.steps, 12);
    const native = sa3({ sa3Sampler: 'pingpong' });
    assert.equal(native.sampler, 'pingpong');
    assert.equal(native.solver, undefined);
    const plug = sa3({ samplerPluginsEnabled: true, inferMethod: 'my_solver' });
    assert.equal(plug.solver, 'my_solver');
  });

  it('bounds: steps 1..64, duration 1..max', () => {
    assert.equal(sa3({ sa3Steps: 500 }).steps, 64);
    assert.equal(sa3({ sa3Steps: 0 }).steps, SA3_DEFAULT_STEPS);
    assert.equal(sa3({ duration: 99999 }).durationSec, SA3_MAX_DURATION_SEC);
    assert.equal(sa3({ duration: 0.5 }).durationSec, 30);
    assert.equal(sa3({}, { duration: 45 }).durationSec, 45);
  });

  it('seed B and blend: default 0.5, explicit 0 kept, clamped', () => {
    const d = sa3({ seedB: 77 });
    assert.equal(d.seedB, 77);
    assert.equal(d.seedBlend, SA3_DEFAULT_SEED_BLEND);
    assert.equal(sa3({ seedB: 77, seedBlend: 0 }).seedBlend, 0);
    assert.equal(sa3({ seedB: 77, seedBlend: 3 }).seedBlend, 1);
    assert.equal(sa3({ seedB: '' }).seedB, undefined);
    assert.equal(sa3({ seedB: -1 }).seedB, undefined);
  });
});

function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>(r => { resolve = r; });
  return { promise, resolve };
}
const tick = () => new Promise(r => setImmediate(r));

describe('renderSlotOnLane', () => {
  it('evicts other families, then renders, under the lane', async () => {
    const order: string[] = [];
    const buf = await renderSlotOnLane({
      label: 't', family: 'sa3', isRunning: () => true,
      evict: async () => { order.push('evict'); assert.ok(gpuLaneBusy()); },
      render: async () => { order.push('render'); return Buffer.from('wav'); },
    });
    assert.deepEqual(order, ['evict', 'render']);
    assert.equal(buf?.toString(), 'wav');
    assert.equal(gpuLaneBusy(), false);
  });

  it('Stop mid-slot keeps the lane until the engine finishes, then discards the output', async () => {
    let running = true;
    const engine = deferred<Buffer>();
    let started = false;
    const p = renderSlotOnLane({
      label: 't', family: 'sa3', isRunning: () => running, evict: async () => {},
      render: () => { started = true; return engine.promise; },
    });
    while (!started) await tick();
    running = false;               // Stop / disconnect
    await tick();
    assert.equal(gpuLaneBusy(), true, 'lane released before the engine finished');
    engine.resolve(Buffer.from('wav'));
    assert.equal(await p, null);
    assert.equal(gpuLaneBusy(), false);
  });

  it('Stop during eviction does not render', async () => {
    let running = true;
    let rendered = false;
    const buf = await renderSlotOnLane({
      label: 't', family: 'yue2', isRunning: () => running,
      evict: async () => { await tick(); running = false; },
      render: async () => { rendered = true; return Buffer.from('wav'); },
    });
    assert.equal(buf, null);
    assert.equal(rendered, false, 'rendered after Stop during eviction');
    assert.equal(gpuLaneBusy(), false);
  });

  it('a failing poll keeps the lane until the engine job is terminal', async () => {
    const statuses: Array<string | Error> = [new Error('timeout'), new Error('timeout'), 'running', 'done'];
    let laneHeldOnEveryPoll = true;
    const buf = await renderSlotOnLane({
      label: 't', family: 'ace', isRunning: () => true, evict: async () => {},
      render: async () => {
        const end = await waitForSlotJob({
          poll: async () => {
            laneHeldOnEveryPoll &&= gpuLaneBusy();
            const s = statuses.shift()!;
            if (s instanceof Error) throw s;
            return { status: s };
          },
          cancel: async () => {}, isRunning: () => true, onTimeout: () => {},
          engineCrashSeq: () => 0, timeoutMs: 60_000, intervalMs: 0,
        });
        return end === 'done' ? Buffer.from('wav') : null;
      },
    });
    assert.equal(statuses.length, 0, 'stopped polling before the job was terminal');
    assert.ok(laneHeldOnEveryPoll);
    assert.equal(buf?.toString(), 'wav');
    assert.equal(gpuLaneBusy(), false);
  });

  it('Stop while queued for the lane starts no next slot', async () => {
    let running = true;
    const blocker = deferred<void>();
    const other = runOnGpuLane(() => blocker.promise, { label: 'generation' });
    let touched = false;
    const p = renderSlotOnLane({
      label: 't', family: 'ace', isRunning: () => running,
      evict: async () => { touched = true; },
      render: async () => { touched = true; return Buffer.from('wav'); },
    });
    await tick();
    running = false;
    blocker.resolve();
    await other;
    assert.equal(await p, null);
    assert.equal(touched, false, 'slot rendered after Stop');
    assert.equal(gpuLaneBusy(), false);
  });
});

describe('waitForSlotJob', () => {
  const base = {
    cancel: async () => {}, isRunning: () => true, onTimeout: () => {},
    engineCrashSeq: () => 0, timeoutMs: 60_000, intervalMs: 0,
  };

  it('after Stop, cancels once and keeps polling through errors to a terminal state', async () => {
    const statuses: Array<string | Error> = ['running', new Error('timeout'), 'running', 'cancelled'];
    let running = true;
    let cancels = 0;
    const end = await waitForSlotJob({
      ...base,
      isRunning: () => running,
      cancel: async () => { cancels++; },
      poll: async () => {
        running = false;
        const s = statuses.shift()!;
        if (s instanceof Error) throw s;
        return { status: s };
      },
    });
    assert.equal(end, 'cancelled');
    assert.equal(cancels, 1);
    assert.equal(statuses.length, 0);
  });

  it('retries a cancel that failed to reach the engine', async () => {
    let cancels = 0;
    let polls = 0;
    const end = await waitForSlotJob({
      ...base,
      isRunning: () => false,
      cancel: async () => { if (++cancels === 1) throw new Error('timeout'); },
      poll: async () => ({ status: ++polls < 3 ? 'running' : 'cancelled' }),
    });
    assert.equal(end, 'cancelled');
    assert.equal(cancels, 2);
  });

  it('watchdog stops the stream, cancels and waits for the engine', async () => {
    let timeouts = 0;
    let cancels = 0;
    const statuses = ['running', 'running', 'cancelled'];
    const end = await waitForSlotJob({
      ...base,
      timeoutMs: -1,
      onTimeout: () => { timeouts++; },
      cancel: async () => { cancels++; },
      poll: async () => ({ status: statuses.shift()! }),
    });
    assert.equal(end, 'cancelled');
    assert.deepEqual([timeouts, cancels, statuses.length], [1, 1, 0]);
  });

  it('ends as lost only on confirmed engine loss', async () => {
    let seq = 0;
    let polls = 0;
    const crashed = await waitForSlotJob({
      ...base,
      engineCrashSeq: () => seq,
      poll: async () => { if (++polls === 2) seq++; throw new Error('ECONNREFUSED'); },
    });
    assert.equal(crashed, 'lost');

    let n = 0;
    const gone = await waitForSlotJob({
      ...base,
      poll: async () => { n++; throw new Error('ace-server GET /job failed (404)'); },
    });
    assert.equal(gone, 'lost');
    assert.equal(n, 3);
  });
});
