// drift-integration-check.mjs
// Verifies that applyDrift is called by buildSlotRequest:
//   1. guidanceScale strictly increases toward target over 3 slots (drift is active).
//   2. A one-shot nextPrompt still wins even when drift sets a caption.
//
// Run: cd server && node scripts/drift-integration-check.mjs

import { initDrift } from '../src/services/storm/drift.js';
import { buildSlotRequest } from '../src/services/storm/slot.js';

// ---------------------------------------------------------------------------
// Minimal stubs for translateParams and mapYue2Params
// (we do not need the real implementations for this wiring check)
// ---------------------------------------------------------------------------

// Patch the module resolution so the slot helper finds stubs.
// Because slot.ts imports from '../generation/translateParams.js' and
// '../backends/yue2/generate.js', we need the real files or we get an import
// error. Since we ARE running inside the server tree the real files exist.
// We just load the real translateParams (it's pure) and a minimal mapYue2Params
// is exported from the real module.

// ---------------------------------------------------------------------------
// Test 1: guidanceScale strictly increases over 3 slots with cfg drift
// ---------------------------------------------------------------------------

const baseSeed = 100;
const driftState = initDrift(baseSeed);

const pressure = {
  targets: { guidanceScale: (9 - 1) / (15 - 1) },  // target 9 normalised to 1..15 range
  targetCaption: [],
  captionPool: [],
  pull: 1,
  rate: 0.5,
  jitter: 0,
  locks: {},
  seedMode: 'lock',
};

const drift = { state: driftState, pressure };

const baseParams = {
  seed: baseSeed,
  guidanceScale: 5,
  randomSeed: false,
};

const ctrl = {
  guidanceScale: 5,
  lssStrength: 0,
  inferenceSteps: 0,
  duration: 0,
  bpm: 0,
  seedLock: true,
  nextSeed: null,
  nextPrompt: null,
  nextLyrics: null,
  nextBpm: null,
  nextDuration: null,
  baseCaption: null,
  baseLyrics: null,
  baseInstrumental: null,
  streamPaused: false,
  extraPluginParams: {},
  inferMethod: null,
  scheduler: null,
  guidanceMode: null,
  drift,
};

const vals = [];
for (let i = 0; i < 3; i++) {
  const result = buildSlotRequest(baseParams, ctrl, i, baseSeed, 'ace');
  const gs = result.aceReq.guidance_scale ?? result.aceReq.guidanceScale;
  vals.push(gs);
  console.log(`  slot ${i}: guidanceScale = ${gs}`);
}

let strictlyIncreasing = true;
for (let i = 1; i < vals.length; i++) {
  if (vals[i] <= vals[i - 1]) {
    strictlyIncreasing = false;
    console.error(`FAIL: not strictly increasing at step ${i}: ${vals[i - 1]} -> ${vals[i]}`);
  }
}
if (strictlyIncreasing) {
  console.log('PASS: guidanceScale strictly increases over 3 slots');
} else {
  process.exitCode = 1;
}

// ---------------------------------------------------------------------------
// Test 2: one-shot nextPrompt wins over drift caption
// ---------------------------------------------------------------------------

const driftState2 = initDrift(baseSeed);
const pressure2 = {
  targets: {},
  targetCaption: ['epic', 'dark'],
  captionPool: ['epic', 'dark', 'chill'],
  pull: 1,
  rate: 1,
  jitter: 0,
  locks: {},
  seedMode: 'lock',
};
const drift2 = { state: driftState2, pressure: pressure2 };

const ctrl2 = {
  ...ctrl,
  drift: drift2,
  nextPrompt: 'user override caption',
};

const result2 = buildSlotRequest(baseParams, ctrl2, 0, baseSeed, 'ace');
// translateParams maps caption -> prompt_text or similar; check the raw aceReq
const caption = result2.aceReq.caption ?? result2.aceReq.prompt_text ?? result2.aceReq.prompt;
console.log(`  one-shot caption in aceReq = ${JSON.stringify(caption)}`);

if (caption === 'user override caption') {
  console.log('PASS: user one-shot nextPrompt wins over drift caption');
} else {
  console.error(`FAIL: expected 'user override caption', got ${JSON.stringify(caption)}`);
  process.exitCode = 1;
}
