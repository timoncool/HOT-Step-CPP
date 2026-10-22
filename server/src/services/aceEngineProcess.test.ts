// Respawn after the crash limiter gave up (first run, empty models folder):
// the engine that comes back must be marked ready even when its /health answers
// only AFTER the restart timeout (HOTSTEP-328).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { EventEmitter } from 'node:events';
import type { AddressInfo } from 'node:net';

function fakeChild(): any {
  const c: any = new EventEmitter();
  c.stdout = new EventEmitter();
  c.stderr = new EventEmitter();
  c.pid = 4242;
  c.exitCode = null;
  c.signalCode = null;
  c.kill = () => true;
  return c;
}

test('restart that outruns the health timeout still ends ready', async () => {
  let healthUp = false;
  let gate: Promise<void> | null = null;   // holds /health open until released
  const srv = http.createServer(async (req, res) => {
    if (req.url === '/health' && healthUp) {
      if (gate) await gate; res.setHeader('content-type', 'application/json'); res.end('{"status":"ok"}'); }
    else { res.statusCode = 503; res.end('booting'); }
  });
  await new Promise<void>(r => srv.listen(0, '127.0.0.1', r));
  process.env.ACESTEPCPP_PORT = String((srv.address() as AddressInfo).port);
  process.env.ACESTEPCPP_HOST = '127.0.0.1';
  process.env.ACESTEPCPP_EXE = process.execPath;   // only needs to exist

  const proc = await import('./aceEngineProcess.js');
  const state = await import('../engineState.js');
  const kids: any[] = [];
  let restores = 0;
  proc.onEngineRestarted(() => { restores++; });
  proc.testHooks.spawn = (() => { const c = fakeChild(); kids.push(c); return c; }) as any;
  proc.testHooks.restartHealthTimeoutMs = 300;
  proc.testHooks.lateReadyPollMs = 50;

  try {
    // Empty folder: three crashes inside the window → the limiter gives up.
    proc.startAceServer();
    for (let i = 0; i < 3; i++) {
      const k = kids[kids.length - 1];
      k.exitCode = 1;
      k.emit('exit', 1, null);
      if (i < 2) await new Promise(r => setTimeout(r, 3100));   // the 3 s respawn timer
    }
    assert.equal(state.engineReady, false);

    // A model lands; the engine respawns but its /health is slow.
    assert.equal(await proc.restartAceServer(), false, 'restart times out while health is 503');
    assert.equal(state.engineReady, false);

    healthUp = true;
    const until = Date.now() + 3000;
    while (!state.engineReady && Date.now() < until) await new Promise(r => setTimeout(r, 25));
    assert.equal(state.engineReady, true, 'late /health 200 flips the engine to ready');

    // Stop while a late-watch /health request is in flight: its success is stale.
    healthUp = false;
    for (const k of kids) k.exitCode = 0;            // retire the first engine so restart spawns a new one
    assert.equal(await proc.restartAceServer(), false);
    const restoresBefore = restores;
    let release!: () => void;
    gate = new Promise<void>(r => { release = r; });
    healthUp = true;
    await new Promise(r => setTimeout(r, 200));      // watcher's request is now parked on the gate
    for (const k of kids) k.exitCode = 0;            // fake pid: never let stop reach taskkill
    await proc.stopAceServer('stopped mid-health', 100);
    release();
    await new Promise(r => setTimeout(r, 200));
    assert.equal(state.engineReady, false, 'stale health success must not mark a stopped engine ready');
    assert.equal(restores, restoresBefore, 'no restore hook after a stop');
  } finally {
    for (const k of kids) k.exitCode = 0;   // fake pid: never let stop reach taskkill
    await proc.stopAceServer('test over', 100).catch(() => {});
    srv.close();
  }
});
