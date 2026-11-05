// Launcher contract for the persistent data root (GH #228): the launchers pin
// only their own folder and resolve the restart marker from the profile, so
// every unpacked release finds the same pointer. No app launch.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import { restartMarker } from './profile-path.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));

test('restart marker sits in the platform profile, spaces included', () => {
  const win = path.win32;
  const home = win.join('C:', 'Users', 'Jo Bloggs');
  const local = win.join(home, 'AppData', 'Local');
  const want = win.join(local, 'HOT-Step', 'restart-requested');
  assert.equal(restartMarker({ LOCALAPPDATA: local }, 'win32', home, win), want);
  assert.equal(restartMarker({}, 'win32', home, win), want);
  const posix = path.posix;
  assert.equal(restartMarker({}, 'darwin', '/Users/Jo Bloggs', posix), '/Users/Jo Bloggs/Library/Application Support/HOT-Step/restart-requested');
  assert.equal(restartMarker({}, 'linux', '/home/jo', posix), '/home/jo/.config/hot-step/restart-requested');
  assert.equal(restartMarker({ XDG_CONFIG_HOME: '/cfg dir' }, 'linux', '/home/jo', posix), '/cfg dir/hot-step/restart-requested');
});

test('the helper prints the same marker from two release folders sharing one profile', t => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'hs launcher '));
  t.after(() => fs.rmSync(base, { recursive: true, force: true }));
  const profile = path.join(base, 'my profile');
  const run = name => {
    // A real unpacked release: its own parent folder, its own copy of the helper, its own HOT_STEP_ROOT.
    const app = path.join(base, `${name} parent`, 'HOT-Step v2');
    fs.mkdirSync(app, { recursive: true });
    fs.copyFileSync(path.join(here, 'profile-path.mjs'), path.join(app, 'profile-path.mjs'));
    return execFileSync(process.execPath, [path.join(app, 'profile-path.mjs')],
      { env: { ...process.env, HOT_STEP_ROOT: app, HOT_STEP_PROFILE_DIR: profile }, cwd: app, encoding: 'utf8' }).trim();
  };
  const a = run('old');
  assert.equal(a, path.join(profile, 'restart-requested'));
  assert.equal(run('new'), a);
});

test('launchers set HOT_STEP_ROOT to their own folder and never pin the data root or profile', () => {
  const bat = fs.readFileSync(path.join(here, 'HOT-Step.bat'), 'utf8');
  const sh = fs.readFileSync(path.join(here, 'HOT-Step.sh'), 'utf8');
  assert.match(bat, /set "HOT_STEP_ROOT=%~dp0"/);
  assert.match(sh, /export HOT_STEP_ROOT="\$DIR"/);
  for (const src of [bat, sh]) {
    assert.doesNotMatch(src, /HOT_STEP_DATA_ROOT|HOT_STEP_PROFILE_DIR/);
    assert.doesNotMatch(src, /HOT-Step-Data/);
  }
});
