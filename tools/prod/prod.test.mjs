import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { assertIdle, assertScratchEnv, inside, parseOptions } from './promote.mjs';
import { maintenanceHandler } from './runtime.mjs';
import { resolveSharedData, readDevData, validateSharedConfig } from './data-mode.mjs';

test('mode selection refuses ambiguous, missing and misspelled options', () => {
  for (const args of [['--scratch', 'root', '--data-root', 'root'], ['--data-root'],
    ['--scratch', 'root', '--data-root'], ['--shared', 'root'], ['--dev-profile', 'profile'],
    ['--data-root', 'root', '--data-root', 'other'], ['--start', '--data-root', 'root']]) {
    assert.throws(() => parseOptions(args));
  }
  assert.deepEqual(parseOptions(['--data-root', 'root']), { '--data-root': 'root' });
});

test('shared paths come from the dev pointer and env overrides, without creating missing targets', () => {
  const checkout = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
  const fixture = fs.mkdtempSync(path.join(checkout, '.tmp/prod-mode-test-'));
  try {
    const root = path.join(fixture, 'shared data');
    const profile = path.join(fixture, 'profile');
    const adapters = path.join(fixture, 'external adapters');
    for (const dir of [profile, adapters, root, path.join(root, 'library/training'), path.join(root, 'models'), path.join(root, 'settings')]) fs.mkdirSync(dir, { recursive: true });
    // Use the real resolver and the checkout-specific pointer format.
    const initial = readDevData(profile);
    fs.writeFileSync(initial.pointerPath, JSON.stringify({ version: 1, root, pending: null, checkout }));
    fs.writeFileSync(path.join(root, 'settings/.env'), `ACESTEPCPP_ADAPTERS="${adapters.replaceAll('\\', '/')}"\n`);
    const report = readDevData(profile);
    const selected = resolveSharedData(root, report);
    assert.equal(selected.profile, profile);
    assert.equal(selected.library, path.join(root, 'library'));
    assert.equal(selected.training, path.join(root, 'library/training'));
    assert.equal(selected.adapters, adapters);
    const privateProfile = path.join(fixture, 'prod-profile');
    fs.mkdirSync(privateProfile);
    const config = { devCheckout: checkout, devProfile: profile, data: { ...selected, profile: privateProfile } };
    assert.doesNotThrow(() => validateSharedConfig(config));
    assert.throws(() => validateSharedConfig({ ...config, data: { ...config.data, adapters: root } }), /adapters changed/);
    assert.throws(() => resolveSharedData('relative', report), /absolute/);
    for (const changed of [{ root: fixture }, { pendingMove: {} }, { pointerError: 'bad JSON' }, { source: 'fallback' }, { children: null }]) {
      assert.throws(() => resolveSharedData(root, { ...report, ...changed }), /dev pointer/);
    }
    for (const key of ['library', 'training', 'adapters', 'models']) {
      const missing = path.join(fixture, `missing-${key}`);
      assert.throws(() => resolveSharedData(root, { ...report, children: { ...report.children, [key]: missing } }), /directory/);
      assert.equal(fs.existsSync(missing), false);
    }
    assert.throws(() => resolveSharedData(root, { ...report, profileDir: path.join(fixture, 'missing-profile') }), /directory/);
  } finally { fs.rmSync(fixture, { recursive: true }); }
});

test('promotion fails closed for busy, unknown, or moving data', () => {
  assert.doesNotThrow(() => assertIdle({ blockers: [], pending: null }));
  for (const report of [{}, null, { blockers: ['training'] }, { blockers: [], pending: {} }]) {
    assert.throws(() => assertIdle(report));
  }
});
test('path boundary excludes a sibling with the same prefix', () => {
  assert.equal(inside('D:/app', 'D:/app/engine/a.exe'), true);
  assert.equal(inside('D:/app', 'D:/app-other/engine/a.exe'), false);
  assert.equal(inside('D:/app', 'D:/app'), false);
});
test('repeat promotion accepts generated relative paths and rejects escaped scratch writes', () => {
  const scratch = path.resolve('scratch-data');
  assert.doesNotThrow(() => assertScratchEnv('LYRICS_EXPORT_DIR=./data/lyrics\n', scratch));
  assert.throws(() => assertScratchEnv('LYRICS_EXPORT_DIR=../real-data\n', scratch));
  assert.throws(() => assertScratchEnv(`DATA_DIR="${path.resolve('real-data')}"\n`, scratch));
});
test('maintenance drains existing submissions and rejects new work until resumed', async () => {
  let release;
  const server = http.createServer(maintenanceHandler((req, res) => {
    if (req.url === '/slow') release = () => res.end('done');
    else res.end('ok');
  }, 'test-token'));
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  const gate = action => fetch(`${base}/__prod/maintenance`, { method: 'POST', headers: { 'x-hotstep-prod-token': 'test-token', 'x-hotstep-prod-action': action } });
  try {
    const slow = fetch(`${base}/slow`, { method: 'POST' });
    while (!release) await new Promise(resolve => setTimeout(resolve, 5));
    assert.equal((await fetch(`${base}/__prod/maintenance`, { method: 'POST' })).status, 403);
    const paused = gate('pause');
    // Observe the gate, rather than assume timing of the control request.
    for (let attempt = 0; attempt < 100; attempt++) {
      const response = await fetch(`${base}/submit`, { method: 'POST' });
      if (response.status === 503) break;
      assert.ok(attempt < 99, 'gate did not pause');
      await new Promise(resolve => setTimeout(resolve, 5));
    }
    assert.equal((await fetch(`${base}/audio`)).status, 200);
    release(); await slow;
    assert.deepEqual(await (await paused).json(), { paused: true, drained: true });
    await gate('resume');
    assert.equal((await fetch(`${base}/submit`, { method: 'POST' })).status, 200);
  } finally {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
});
test('Windows rebuild dry run preserves an inert sibling engine and lock excludes rebuild', { skip: process.platform !== 'win32' }, () => {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
  const result = spawnSync('pwsh', ['-NoProfile', '-File', path.join(root, 'tools/prod/isolation-test.ps1')], { encoding: 'utf8', timeout: 60_000 });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.match(result.stdout, /ISOLATION PASS/);
});
