import { test } from 'node:test';
import assert from 'node:assert/strict';

import { yue2CoverGuide, yue2GuideSkipReason, yue2GuideWire, YUE2_COVER_GUIDE_DEFAULTS } from './sourceGuide.js';
import { yue2CoverFromSubmission } from './index.js';

test('Follow source: validation, skip rule and engine fields', () => {
  assert.equal(yue2CoverGuide(undefined), undefined);
  assert.deepEqual(yue2CoverGuide({}), YUE2_COVER_GUIDE_DEFAULTS);
  assert.throws(() => yue2CoverGuide({ scale: 1.5 }), /scale/);
  assert.throws(() => yue2CoverGuide({ k: 65 }), /k must/);
  assert.throws(() => yue2CoverGuide({ schedule: 'wobble' }), /schedule/);

  assert.equal(yue2GuideSkipReason({ key: 'source', tempo: 'free' }), '');
  assert.match(yue2GuideSkipReason({ keyShift: 2, key: 'D' }), /key/);
  assert.match(yue2GuideSkipReason({ tempo: 120 }), /BPM/);

  const g = yue2CoverGuide({ scale: 0.75, k: 20, restrict: true, lfo: 'sine', lfoDepth: 1 })!;
  assert.deepEqual(yue2GuideWire(g, 'C:/g.y2gd'), {
    guide_path: 'C:/g.y2gd', guide_scale: 0.75, guide_k: 20, guide_outside: -100,
    guide_lfo: 'sine', guide_lfo_period_sec: 8, guide_lfo_depth: 1,
  });
  // Off-by-default knobs stay off the wire.
  assert.deepEqual(yue2GuideWire(YUE2_COVER_GUIDE_DEFAULTS, 'x'), { guide_path: 'x', guide_scale: 0.5, guide_k: 32 });

  // Carried through the cover marker; refused for a score with no recording.
  assert.deepEqual(yue2CoverFromSubmission({ yue2Cover: { sourceId: 's', guide: { scale: 0.3 } } })?.guide,
    { ...YUE2_COVER_GUIDE_DEFAULTS, scale: 0.3 });
  assert.throws(() => yue2CoverFromSubmission({ yue2Cover: { sourceKind: 'musicxml', guide: {} } }), /MusicXML/);
});
