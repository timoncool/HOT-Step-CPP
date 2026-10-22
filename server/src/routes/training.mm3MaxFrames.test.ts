import test from 'node:test';
import assert from 'node:assert/strict';
import { mm3MaxFramesError } from './training.js';

test('mm3 maxFrames: absent uses the default, in-range passes', () => {
  for (const v of [undefined, null, 64, 1500, 9000, '750']) assert.equal(mm3MaxFramesError(v), '', String(v));
});

test('mm3 maxFrames: out-of-range or non-numeric is rejected, not clamped', () => {
  for (const v of [32, 63, 9001, -1, 'abc', NaN, {}]) assert.match(mm3MaxFramesError(v), /between 64 and 9000/, String(v));
});
