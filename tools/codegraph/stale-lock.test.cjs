'use strict';
// Regression: the launcher must neither block on a dead writer's codegraph.lock
// nor touch a live writer's. Runs the real launcher and codegraph binary on a
// throwaway one-file project. node tools/codegraph/stale-lock.test.cjs
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawn, spawnSync } = require('node:child_process');

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'cg-lock-'));
const tools = path.join(root, 'tools/codegraph');
fs.mkdirSync(tools, { recursive: true });
fs.copyFileSync(path.join(__dirname, 'launcher.cjs'), path.join(tools, 'launcher.cjs'));
fs.symlinkSync(path.join(__dirname, 'node_modules'), path.join(tools, 'node_modules'), 'junction');
const git = (...args) => spawnSync('git', args, { cwd: root, encoding: 'utf8' });
git('init', '-q');
fs.writeFileSync(path.join(root, '.git/info/exclude'), '.codegraph/\ntools/\n');
fs.writeFileSync(path.join(root, 'a.c'), 'int alpha(void) { return 1; }\n');
git('add', 'a.c');
git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'x');
spawnSync(process.execPath, [path.join(tools, 'node_modules/@colbymchenry/codegraph/npm-shim.js'), 'init', root], { cwd: root });

const lock = path.join(root, '.codegraph/codegraph.lock');
const launcher = path.join(tools, 'launcher.cjs');
const cg = (...args) => new Promise(resolve => {
  const child = spawn(process.execPath, [launcher, ...args], { cwd: root });
  let out = '';
  child.stdout.on('data', d => { out += d; });
  child.stderr.on('data', d => { out += d; });
  child.on('close', status => resolve({ status, out }));
});
// Each case edits a file so the launcher has to sync, which is what takes the lock.
let edits = 0;
const touch = () => fs.writeFileSync(path.join(root, 'a.c'), `int alpha(void) { return ${++edits}; }\n`);

(async () => {
  try {
    // A reader that reads, renames or unlinks the lock can race a live writer
    // (two cleaners, one stale PID). The launcher must leave the file to codegraph.
    assert(!fs.readFileSync(launcher, 'utf8').includes('codegraph.lock'), 'launcher must not handle codegraph.lock');

    const dead = String(spawnSync(process.execPath, ['-e', '']).pid);

    // A writer killed mid-sync left its lock behind: the reader still answers.
    touch();
    fs.writeFileSync(lock, dead);
    let r = await cg('context', 'alpha');
    assert.strictEqual(r.status, 0, r.out);
    assert.match(r.out, /alpha/);
    assert(!fs.existsSync(lock), 'dead lock cleared');

    // Several readers arrive together on a dead lock: all answer, no lock left.
    touch();
    fs.writeFileSync(lock, dead);
    const all = await Promise.all([1, 2, 3, 4].map(() => cg('context', 'alpha')));
    for (const each of all) assert.strictEqual(each.status, 0, each.out);
    assert(!fs.existsSync(lock), 'dead lock cleared after concurrent readers');

    // A live writer holds the lock (this process stands in for it): the reader
    // gives up with "index busy" and the writer's lock is still there, unchanged.
    touch();
    fs.writeFileSync(lock, String(process.pid));
    const before = fs.statSync(lock);
    r = await cg('context', 'alpha');
    assert.strictEqual(r.status, 1, r.out);
    assert.match(r.out, /index busy/);
    assert.strictEqual(fs.readFileSync(lock, 'utf8'), String(process.pid), 'live lock kept');
    assert.strictEqual(fs.statSync(lock).ino, before.ino, 'live lock never replaced');
    fs.unlinkSync(lock);

    // The writer finishes: the next reader answers.
    r = await cg('context', 'alpha');
    assert.strictEqual(r.status, 0, r.out);
    console.log('stale-lock: ok');
  } finally {
    fs.unlinkSync(path.join(tools, 'node_modules'));
    fs.rmSync(root, { recursive: true, force: true });
  }
})().catch(error => { console.error(error); process.exit(1); });
