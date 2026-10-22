// An empty models folder makes ace-server exit 1 after "ERROR: no models found".
// That is a first run, not a crash: no restart budget burnt, no respawn, and a
// later restartAceServer() (a finished download) clears the state.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';

function fakeChild(): any {
  const c: any = new EventEmitter();
  c.stdout = new EventEmitter();
  c.stderr = new EventEmitter();
  c.pid = 4243;
  c.exitCode = null;
  c.signalCode = null;
  c.kill = () => true;
  return c;
}

test('no-models exit is classified, spends no crash budget and does not respawn', async () => {
  process.env.ACESTEPCPP_EXE = process.execPath;   // only needs to exist
  const proc = await import('./aceEngineProcess.js');
  const state = await import('../engineState.js');
  const kids: any[] = [];
  proc.testHooks.spawn = (() => { const c = fakeChild(); kids.push(c); return c; }) as any;

  try {
    proc.startAceServer();
    const k = kids[0];
    const seq = proc.engineCrashSeq();
    k.stderr.emit('data', Buffer.from('[Server] ERROR: no models found in C:\data\models\n'));
    k.exitCode = 1;
    k.emit('exit', 1, null);

    assert.equal(state.engineNoModels, true);
    assert.equal(state.engineReady, false);
    assert.match(state.engineBootStatus, /Model Manager/);
    assert.equal(proc.engineCrashSeq(), seq, 'not counted as a crash');
    await new Promise(r => setTimeout(r, 3300));
    assert.equal(kids.length, 1, 'no respawn timer was armed');

    // A download finishing brings it back; the flag clears.
    proc.retryIfGaveUp();
    assert.equal(kids.length, 2);
    assert.equal(state.engineNoModels, false);

    // A plain exit 1 without the stderr line is still a crash.
    const k2 = kids[1];
    k2.exitCode = 1;
    k2.emit('exit', 1, null);
    assert.equal(state.engineNoModels, false);
    assert.equal(proc.engineCrashSeq(), seq + 1);
  } finally {
    for (const k of kids) k.exitCode = 0;
    await proc.stopAceServer('test over', 100).catch(() => {});
  }
});

test('no-models marker split across stderr chunks, with exit before stream end', async () => {
  const proc = await import('./aceEngineProcess.js');
  const state = await import('../engineState.js');
  const kids: any[] = [];
  proc.testHooks.spawn = (() => { const c = fakeChild(); kids.push(c); return c; }) as any;
  proc.testHooks.restartHealthTimeoutMs = 100;
  await proc.restartAceServer();                   // clear suspended/gaveUp left by the previous test
  for (const k of kids) k.exitCode = 0;
  kids.length = 0;
  try {
    proc.startAceServer();
    const k = kids[0];
    const seq = proc.engineCrashSeq();
    k.stderr.readableEnded = false;                  // stream still open when 'exit' fires
    k.stderr.emit('data', Buffer.from('[Server] ERROR: no mod'));
    k.exitCode = 1;
    k.emit('exit', 1, null);
    k.stderr.emit('data', Buffer.from('els found in X\n'));
    k.stderr.emit('close');
    assert.equal(state.engineNoModels, true);
    assert.equal(proc.engineCrashSeq(), seq);
  } finally {
    for (const k of kids) k.exitCode = 0;
    await proc.stopAceServer('test over', 100).catch(() => {});
  }
});

test('a model copied into the folder restarts a waiting engine', async () => {
  const fs = await import('node:fs');
  const os = await import('node:os');
  const path = await import('node:path');
  const { config } = await import('../config.js');
  const proc = await import('./aceEngineProcess.js');
  const kids: any[] = [];
  proc.testHooks.spawn = (() => { const c = fakeChild(); kids.push(c); return c; }) as any;

  proc.testHooks.restartHealthTimeoutMs = 100;
  await proc.restartAceServer();                   // clear suspended/gaveUp left by the previous test
  for (const k of kids) k.exitCode = 0;
  kids.length = 0;
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hs-models-'));
  const prev = config.aceServer.models;
  config.aceServer.models = dir;
  try {
    proc.startAceServer();
    const k = kids[0];
    k.stderr.emit('data', Buffer.from('ERROR: no models found\n'));
    k.exitCode = 1;
    k.emit('exit', 1, null);
    const f = path.join(dir, 'm.gguf');
    fs.writeFileSync(f, 'x');
    const old = new Date(Date.now() - 60_000);
    fs.utimesSync(f, old, old);
    const until = Date.now() + 9000;
    while (kids.length < 2 && Date.now() < until) await new Promise(r => setTimeout(r, 100));
    assert.equal(kids.length, 2, 'engine respawned after a model appeared');
  } finally {
    config.aceServer.models = prev;
    for (const k of kids) k.exitCode = 0;
    await proc.stopAceServer('test over', 100).catch(() => {});
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
