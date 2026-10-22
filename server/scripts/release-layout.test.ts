// Disposable layout fixtures. Never read the real profile or move user data.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { dataLayout, devPointerFile, layoutDefaults, profileDir, RESTART_MARKER_FILE } from '../src/dataroot/paths.js';
import { resolveRuntime } from '../src/dataroot/bootstrap.js';
// JS packaging helpers intentionally have no runtime TypeScript dependency.
// @ts-expect-error packaging helper is plain JavaScript
import { restartMarker } from '../../release/profile-path.mjs';
// @ts-expect-error packaging checker is plain JavaScript
import { checkReleaseLayout } from '../../release/check-layout.mjs';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

test('all five assemblers retain immutable resources and portable defaults', () => {
  assert.deepEqual(checkReleaseLayout(repo), []);
});

for (const platform of ['win32', 'linux', 'darwin']) {
  test(`${platform}: stable profile marker and platform defaults`, () => {
    const paths = platform === 'win32' ? path.win32 : path.posix;
    const home = platform === 'win32' ? 'C:\\Users\\fixture' : '/home/fixture';
    const env = platform === 'win32' ? { LOCALAPPDATA: 'C:\\Users\\fixture\\Local' } : {};
    const expected = platform === 'win32' ? paths.join(env.LOCALAPPDATA!, 'HOT-Step')
      : platform === 'darwin' ? paths.join(home, 'Library', 'Application Support', 'HOT-Step')
      : paths.join(home, '.config', 'hot-step');
    assert.equal(restartMarker(env, platform, home, paths), paths.join(expected, 'restart-requested'));
  });
}

test('launcher and resolver agree on profile override, XDG and fallback', () => {
  for (const env of [{}, { HOT_STEP_PROFILE_DIR: path.join(os.tmpdir(), 'profile with spaces') }, { XDG_CONFIG_HOME: '/custom/config' }]) {
    assert.equal(restartMarker(env), path.join(profileDir(env), RESTART_MARKER_FILE));
  }
});

test('fresh release from unrelated cwd reuses data across version folders', async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'hs-release-layout-'));
  try {
    const app = path.join(base, 'version one');
    fs.mkdirSync(path.join(app, 'server', 'data'), { recursive: true });
    fs.writeFileSync(path.join(app, 'server', 'data', 'model-registry.json'), 'bundled');
    const env = { HOT_STEP_ROOT: app, HOT_STEP_PROFILE_DIR: path.join(base, 'profile') };
    const first = await resolveRuntime({ env, log: () => {} });
    assert.ok(first.ok);
    const layout = dataLayout(path.join(base, 'HOT-Step-Data'));
    assert.equal(first.runtime.dataRoot, layout.root);
    for (const writable of Object.values(layoutDefaults(layout))) {
      assert.ok(!writable.startsWith(app + path.sep));
      assert.ok(writable.startsWith(layout.root + path.sep));
    }
    assert.equal(fs.readFileSync(path.join(app, 'server', 'data', 'model-registry.json'), 'utf8'), 'bundled');
    assert.ok(!fs.existsSync(path.join(app, 'models')));
    const app2 = path.join(base, 'version two');
    fs.mkdirSync(app2);
    const second = await resolveRuntime({ env: { ...env, HOT_STEP_ROOT: app2 }, log: () => {} });
    assert.ok(second.ok);
    assert.equal(second.runtime.dataRoot, layout.root);
    assert.equal(second.runtime.restartMarkerPath, first.runtime.restartMarkerPath);
    const devProfile = path.join(base, 'dev profile');
    const checkoutRoot = path.join(base, 'checkout');
    const dev = await resolveRuntime({ env: { HOT_STEP_PROFILE_DIR: devProfile }, checkoutRoot, log: () => {} });
    assert.ok(dev.ok);
    assert.equal(dev.runtime.mode, 'dev');
    assert.equal(dev.runtime.dataRoot, null);
    assert.equal(dev.runtime.pointerPath, devPointerFile(devProfile, checkoutRoot));
    assert.ok(!fs.existsSync(devProfile), 'dev profile untouched');
    assert.match(fs.readFileSync(path.join(repo, '.env.example'), 'utf8'), /^DATA_DIR=\.\/server\/data$/m);
  } finally {
    fs.rmSync(base, { recursive: true, force: true });
  }
});

test('Windows launcher restarts a stub server using the profile marker from another cwd', { skip: process.platform !== 'win32' }, () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'hs-launcher-'));
  try {
    const app = path.join(base, 'app with spaces');
    const profile = path.join(base, 'profile with spaces');
    for (const dir of ['runtime', 'server']) fs.mkdirSync(path.join(app, dir), { recursive: true });
    fs.mkdirSync(profile);
    fs.copyFileSync(process.execPath, path.join(app, 'runtime', 'node.exe'));
    fs.copyFileSync(path.join(repo, 'release', 'profile-path.mjs'), path.join(app, 'profile-path.mjs'));
    // Disable desktop browser opening and the final interactive pause only.
    const launcher = fs.readFileSync(path.join(repo, 'release/HOT-Step.bat'), 'utf8')
      .replace(/^start \/MIN .*$/m, 'REM Browser disabled in fixture')
      .replace('pause >nul', 'exit /b 0');
    fs.writeFileSync(path.join(app, 'HOT-Step.bat'), launcher.replace(/\r?\n/g, '\r\n'));
    fs.writeFileSync(path.join(app, 'server', 'server.mjs'), `
      import fs from 'node:fs';
      import path from 'node:path';
      import { restartMarker } from '../profile-path.mjs';
      const trace = path.join(process.env.HOT_STEP_PROFILE_DIR, 'starts');
      const first = !fs.existsSync(trace);
      fs.appendFileSync(trace, process.env.HOT_STEP_ROOT + '\\n');
      if (first) fs.writeFileSync(restartMarker(), 'restart');
    `);
    const output = execFileSync('cmd.exe', ['/d', '/c', path.join(app, 'HOT-Step.bat')], {
      cwd: base, encoding: 'utf8', timeout: 15_000,
      env: { ...process.env, HOT_STEP_PROFILE_DIR: profile },
    });
    assert.match(output, /Restarting/);
    assert.equal(fs.readFileSync(path.join(profile, 'starts'), 'utf8').trim().split('\n').length, 2);
    assert.ok(!fs.existsSync(path.join(profile, 'restart-requested')));
    assert.ok(!fs.existsSync(path.join(app, 'models')));
    assert.ok(!fs.existsSync(path.join(app, '.env')));
  } finally {
    fs.rmSync(base, { recursive: true, force: true });
  }
});
