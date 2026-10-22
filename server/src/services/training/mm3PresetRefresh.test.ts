import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { config } from '../../config.js';
import { refreshLinkedMm3Presets } from './mm3TrainRunner.js';
import type { TrainingJob } from './labelingQueue.js';

function setup() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'mm3-preset-'));
  const prev = config.aceServer.adapters;
  config.aceServer.adapters = root;
  const ckpt = path.join(root, 'mm3-lm-adapters', 'ds-1', 'final');
  fs.mkdirSync(ckpt, { recursive: true });
  fs.writeFileSync(path.join(ckpt, 'adapter_model.safetensors'), 'x');
  const calls: string[] = [];
  const deps = {
    refresh: (_ds: { slug: string }, ref: string) => { calls.push(ref); return 1; },
    lookup: () => ({ slug: 'ds', lyricsSetId: 7 }) as never,
  };
  const job = { id: 'job-mm3-preset-test', kind: 'mm3-train-lm', datasetId: 'd', events: [], listeners: new Set() } as unknown as TrainingJob;
  return { ckpt, calls, deps, job, done: () => { config.aceServer.adapters = prev; fs.rmSync(root, { recursive: true, force: true }); } };
}

test('linked presets are refreshed when updateLinkedPresets is on or absent', () => {
  const t = setup();
  try {
    refreshLinkedMm3Presets(t.job, { updateLinkedPresets: true }, t.ckpt, t.deps);
    refreshLinkedMm3Presets(t.job, {}, t.ckpt, t.deps);
    assert.equal(t.calls.length, 2);
    assert.match(t.calls[0], /^ds-1[\\/]final[\\/]adapter_model\.safetensors$/);
  } finally { t.done(); }
});

test('linked presets are left alone when updateLinkedPresets is off, with one log line', () => {
  const t = setup();
  try {
    refreshLinkedMm3Presets(t.job, { updateLinkedPresets: false }, t.ckpt, t.deps);
    assert.equal(t.calls.length, 0);
    const logs = t.job.events.filter((e: { type: string }) => e.type === 'log');
    assert.equal(logs.length, 1);
    assert.match((logs[0] as { message: string }).message, /album presets left as they were/);
  } finally { t.done(); }
});
