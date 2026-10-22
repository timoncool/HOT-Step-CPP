// node --test tools/docs/check-docs.symlink.test.mjs
// check-docs skips an untracked dangling symlink .md, but a missing tracked doc must still fail.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const SCRIPT = path.join(path.dirname(fileURLToPath(import.meta.url)), 'check-docs.mjs');

// Throwaway git repo with the script copied in (ROOT is script-relative); returns the run helper.
function sandbox() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'check-docs-'));
  const git = (...a) => assert.equal(spawnSync('git', a, { cwd: dir }).status, 0, `git ${a.join(' ')}`);
  fs.mkdirSync(path.join(dir, 'tools/docs'), { recursive: true });
  fs.mkdirSync(path.join(dir, 'ui/src/components'), { recursive: true });
  fs.mkdirSync(path.join(dir, '.claude/skills'), { recursive: true });
  fs.copyFileSync(SCRIPT, path.join(dir, 'tools/docs/check-docs.mjs'));
  fs.writeFileSync(path.join(dir, 'tools/docs/build-docs.mjs'), ''); // stub: generated-table check is out of scope here
  fs.writeFileSync(path.join(dir, '.claude/skills/README.md'), '');
  fs.writeFileSync(path.join(dir, 'keep.md'), 'hi\n');
  git('init', '-q');
  git('add', '-A');
  git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'init');
  const run = () => spawnSync('node', ['tools/docs/check-docs.mjs'], { cwd: dir, encoding: 'utf8' });
  return { dir, run, cleanup: () => fs.rmSync(dir, { recursive: true, force: true }) };
}

test('untracked dangling symlink is skipped with a warning', (t) => {
  const s = sandbox();
  t.after(s.cleanup);
  try { fs.symlinkSync(path.join(s.dir, 'nowhere.md'), path.join(s.dir, 'dangling.md')); }
  catch (e) { if (e.code === 'EPERM') return t.skip('symlinks need privileges here'); throw e; }
  const r = s.run();
  assert.match(r.stdout + r.stderr, /dangling\.md: unreadable \(skipped - broken symlink\)/);
  assert.doesNotMatch(r.stderr, /ENOENT/);
  assert.equal(r.status, 0, r.stdout + r.stderr);
});

test('missing tracked doc still fails', (t) => {
  const s = sandbox();
  t.after(s.cleanup);
  fs.rmSync(path.join(s.dir, 'keep.md'));
  const r = s.run();
  assert.notEqual(r.status, 0);
  assert.match(r.stderr, /ENOENT/);
});
