import { test } from 'node:test';
import assert from 'node:assert/strict';
import { applyYue2StyleTemplate, yue2StyleSelfCheck } from './style.js';

test('branch table still matches the engine', () => {
  assert.deepEqual(yue2StyleSelfCheck(), []);
});

// #238 is fixed at the origin (the create path no longer prefixes the trigger for
// YuE2), so a bare "<trigger>, " is caption content and must survive.
test('a first genre equal to the trigger word is kept', () => {
  assert.equal(applyYue2StyleTemplate({ trigger: 'rock', caption: 'rock, pop' }), 'rock, in the style of rock. rock, pop');
});

test('an already composed caption keeps a first genre equal to the trigger word', () => {
  const composed = 'rock, in the style of rock. rock, pop';
  assert.equal(applyYue2StyleTemplate({ trigger: 'rock', caption: composed }), composed);
});

test('#238: the trigger appears once in the opener when the caption carries none', () => {
  const style = applyYue2StyleTemplate({ trigger: 't', caption: 'warm synths' });
  assert.equal(style, 't, in the style of t. warm synths');
  assert.equal(style.split('t,').length - 1, 1);
});

test('stays idempotent', () => {
  const once = applyYue2StyleTemplate({ trigger: 't', caption: 't, rock' });
  assert.equal(applyYue2StyleTemplate({ trigger: 't', caption: once }), once);
});
