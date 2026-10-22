// node --test tools/docs/build-docs.skip.test.mjs
// The archive/debug/_INTERNAL skip filter must hold for both path separators.
import test from 'node:test';
import assert from 'node:assert/strict';
import { isSkippedPluginPath } from './build-docs.mjs';

const cases = [
  ['docs/archive/old.md', true],
  ['docs/debug/x.md', true],
  ['docs\\archive\\old.md', true],
  ['docs\\debug\\x.md', true],
  ['plugins/solvers/md_x_INTERNAL.lua', true],
  ['plugins/solvers/md_x.lua', false],
  ['plugins\\solvers\\md_x.lua', false],
];

for (const [p, want] of cases) {
  test(`skip(${JSON.stringify(p)}) === ${want}`, () => assert.equal(isSkippedPluginPath(p), want));
}
