// What this guards: monitor control commands must not be dropped. A pause must
// keep a pending track, a seek must keep a pending pause, seq must increase on
// every write, and a seek must be tagged with its own seekSeq so the host can
// apply it exactly once. Monitor start/restart/stop transitions must not
// interleave across their awaits, or two hosts get spawned. A late exit from a
// replaced host must not delete the chain file its replacement reads.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { createControlWriter, createSerializer, mergeControl, releaseMonitor } from './vstControl.js';

test('pause merges into state and keeps the track', () => {
  const s1 = mergeControl({}, { track: 'a.wav', action: 'play' }, 1, true);
  const s2 = mergeControl(s1, { action: 'pause' }, 2);
  assert.equal(s2.track, 'a.wav');
  assert.equal(s2.action, 'pause');
  assert.equal(s2.seq, 2);
});

test('seek keeps a pending pause and gets its own seekSeq', () => {
  const s1 = mergeControl({ track: 'a.wav', action: 'pause', seq: 5 }, { seek: 12.5 }, 6);
  assert.equal(s1.action, 'pause');
  assert.equal(s1.seek, 12.5);
  assert.equal(s1.seekSeq, 6);
  const s2 = mergeControl(s1, { action: 'play' }, 7);
  assert.equal(s2.seekSeq, 6, 'a later non-seek write must not re-arm the seek');
  assert.equal(s2.seq, 7);
});

test('writer produces increasing seq and a parseable file with no temp left', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vstctl-'));
  const file = path.join(dir, 'monitor_control.json');
  const w = createControlWriter(() => file);
  w.write({ track: 'a.wav', action: 'play' }, true);
  const first = JSON.parse(fs.readFileSync(file, 'utf-8'));
  for (let i = 0; i < 10; i++) w.write({ seek: i });
  w.write({ action: 'pause' });
  const last = JSON.parse(fs.readFileSync(file, 'utf-8'));
  assert.ok(last.seq > first.seq);
  assert.equal(last.seq - first.seq, 11);
  assert.equal(last.track, 'a.wav');
  assert.equal(last.seek, 9);
  assert.equal(last.action, 'pause');
  assert.deepEqual(fs.readdirSync(dir), ['monitor_control.json']);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('serializer runs overlapping transitions one at a time, in order', async () => {
  const serial = createSerializer();
  const log: string[] = [];
  let running = 0;
  const transition = (name: string, waitMs: number, fail = false) => serial(async () => {
    running++;
    assert.equal(running, 1, `${name} overlapped another transition`);
    log.push(`${name}:start`);
    await new Promise(r => setTimeout(r, waitMs));  // stands in for awaiting the old host's exit
    log.push(`${name}:spawn`);
    running--;
    if (fail) throw new Error(name);
    return name;
  });
  const results = await Promise.allSettled([
    transition('start1', 20),
    transition('start2', 5, true),
    transition('restart', 1),
  ]);
  assert.deepEqual(log, ['start1:start', 'start1:spawn', 'start2:start', 'start2:spawn', 'restart:start', 'restart:spawn']);
  assert.equal(results[0].status, 'fulfilled');
  assert.equal(results[1].status, 'rejected');      // a failed transition does not block the next
  assert.equal(results[2].status, 'fulfilled');
});

test('late exit of a replaced host keeps the new host and its chain file', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vstctl-'));
  const chain = path.join(dir, '_monitor_chain.json');
  const oldHost = {}, newHost = {};
  fs.writeFileSync(chain, '{}');
  assert.equal(releaseMonitor(oldHost, newHost, chain), newHost);
  assert.ok(fs.existsSync(chain), 'old exit deleted the replacement chain file');
  assert.equal(releaseMonitor(newHost, newHost, chain), null);
  assert.ok(!fs.existsSync(chain), 'last host exit must clean up the chain file');
  fs.rmSync(dir, { recursive: true, force: true });
});
