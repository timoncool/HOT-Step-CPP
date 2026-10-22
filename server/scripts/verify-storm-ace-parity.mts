// verify-storm-ace-parity.mts -- throwaway parity check for the Storm slot helper.
// Run with: npx tsx scripts/verify-storm-ace-parity.mts
//
// Proves that buildSlotRequest(..., 'ace') produces the same AceRequest as the
// old inline logic in generate.ts for the same inputs.
// Prints PASS and the key list on success; exits 1 on mismatch.

import { buildSlotRequest, type StreamControlSnapshot } from '../src/services/storm/slot.js';
import { translateParams } from '../src/services/generation/translateParams.js';

// ── Shared test fixtures ───────────────────────────────────────────────────

const baseParams: Record<string, unknown> = {
  caption: 'epic cinematic orchestral',
  prompt: 'epic cinematic orchestral',
  seed: 12345,
  instrumental: true,
  guidanceScale: 7.0,
  pluginParams: { 'md_wasserstein:tau': 0.5 },
  coResident: false,
};

const ctrl: StreamControlSnapshot = {
  guidanceScale: 8.5,
  lssStrength: 0.3,
  inferenceSteps: 20,
  duration: 30,
  bpm: 120,
  seedLock: false,
  nextSeed: null,
  nextPrompt: null,
  nextLyrics: null,
  nextBpm: null,
  nextDuration: null,
  baseCaption: null,
  baseLyrics: null,
  baseInstrumental: null,
  streamPaused: false,
  extraPluginParams: { 'md_guidance:strength': 0.8 },
  inferMethod: 'rk4',
  scheduler: 'dpmpp',
  guidanceMode: 'cfg',
};

const slotIndex = 3;
const baseSeed = 12345;

// ── Old inline logic (verbatim copy from pre-helper generate.ts) ───────────

function buildSlotReqInline(
  baseP: Record<string, unknown>,
  c: StreamControlSnapshot,
  idx: number,
  bseed: number,
): Record<string, unknown> {
  const mergedPluginParams = {
    ...(typeof baseP.pluginParams === 'object' && baseP.pluginParams !== null
      ? baseP.pluginParams as Record<string, unknown> : {}),
    ...c.extraPluginParams,
  };
  const slotReq: Record<string, unknown> = {
    ...baseP,
    randomSeed: false,
    seed: c.nextSeed !== null
      ? c.nextSeed
      : (c.seedLock ? bseed : bseed + idx),
    guidanceScale: c.guidanceScale,
    ...(c.lssStrength > 0 ? { lssStrength: c.lssStrength } : {}),
    ...(c.inferenceSteps > 0 ? { inferenceSteps: c.inferenceSteps } : {}),
    ...(c.nextBpm !== null ? { bpm: c.nextBpm } : c.bpm > 0 ? { bpm: c.bpm } : {}),
    ...(c.nextDuration !== null ? { duration: c.nextDuration } : c.duration > 0 ? { duration: c.duration } : {}),
    ...(c.inferMethod ? { inferMethod: c.inferMethod } : {}),
    ...(c.scheduler ? { scheduler: c.scheduler } : {}),
    ...(c.guidanceMode ? { guidanceMode: c.guidanceMode } : {}),
    ...(Object.keys(mergedPluginParams).length > 0 ? { pluginParams: mergedPluginParams } : {}),
  };
  if (c.nextPrompt !== null) {
    slotReq.caption = c.nextPrompt;
    slotReq.prompt = c.nextPrompt;
  } else if (c.baseCaption !== null) {
    slotReq.caption = c.baseCaption;
    slotReq.prompt = c.baseCaption;
  }
  if (c.nextLyrics !== null) {
    slotReq.lyrics = c.nextLyrics;
    slotReq.instrumental = false;
  } else if (c.baseLyrics !== null) {
    slotReq.lyrics = c.baseLyrics;
    slotReq.instrumental = false;
  } else if (c.baseInstrumental === false) {
    slotReq.instrumental = false;
  }
  return slotReq;
}

// ── Compare ────────────────────────────────────────────────────────────────

const inlineSlotReq = buildSlotReqInline(baseParams, ctrl, slotIndex, baseSeed);
const inlineAceReq = translateParams(inlineSlotReq);

const helperResult = buildSlotRequest(baseParams, ctrl, slotIndex, baseSeed, 'ace');
if (helperResult.kind !== 'ace') {
  console.error('FAIL: helper returned non-ace result for ace backend');
  process.exit(1);
}
const helperAceReq = helperResult.aceReq;

const inlineJson = JSON.stringify(inlineAceReq, null, 2);
const helperJson = JSON.stringify(helperAceReq, null, 2);

if (inlineJson !== helperJson) {
  console.error('FAIL: ACE requests differ');
  console.error('--- inline ---');
  console.error(inlineJson);
  console.error('--- helper ---');
  console.error(helperJson);
  process.exit(1);
}

const keys = Object.keys(inlineAceReq as object).sort();
console.log('PASS: ACE request from helper == inline logic');
console.log('Keys:', keys.join(', '));
console.log(`Seed in request: ${(inlineAceReq as any).seed ?? (inlineAceReq as any).random_seed}`);

// ── SA3 slot check ─────────────────────────────────────────────────────────

const sa3BaseParams: Record<string, unknown> = {
  caption:       'moody ambient electronic',
  prompt:        'moody ambient electronic',
  seed:          99999,
  instrumental:  true,
  guidanceScale: 5.0,
  sa3Steps:      24,
  duration:      30,
  sa3Sampler:    'pingpong',
};

const sa3Ctrl: StreamControlSnapshot = {
  guidanceScale:      5.0,
  lssStrength:        0,
  inferenceSteps:     0,
  duration:           30,
  bpm:                0,
  seedLock:           false,
  nextSeed:           null,
  nextPrompt:         null,
  nextLyrics:         null,
  nextBpm:            null,
  nextDuration:       null,
  baseCaption:        'moody ambient electronic',
  baseLyrics:         null,
  baseInstrumental:   null,
  streamPaused:       false,
  extraPluginParams:  {},
  inferMethod:        null,
  scheduler:          null,
  guidanceMode:       null,
};

const sa3SlotIndex = 2;
const sa3BaseSeed  = 99999;

const sa3Result = buildSlotRequest(sa3BaseParams, sa3Ctrl, sa3SlotIndex, sa3BaseSeed, 'sa3');

let sa3Fail = false;
const sa3Errors: string[] = [];

if (sa3Result.kind !== 'sa3') {
  sa3Errors.push(`kind: expected 'sa3', got '${sa3Result.kind}'`);
  sa3Fail = true;
} else {
  const r = sa3Result;
  if (!r.caption || r.caption.length === 0) {
    sa3Errors.push(`caption: expected non-empty, got '${r.caption}'`);
    sa3Fail = true;
  } else {
    console.log(`SA3 caption ok: '${r.caption}'`);
  }
  if (r.durationSec !== 30) {
    sa3Errors.push(`durationSec: expected 30, got ${r.durationSec}`);
    sa3Fail = true;
  } else {
    console.log(`SA3 durationSec ok: ${r.durationSec}`);
  }
  // Seed per slot: seedLock=false so seed = baseSeed + slotIndex
  const expectedSeed = sa3BaseSeed + sa3SlotIndex;
  if (r.seed !== expectedSeed) {
    sa3Errors.push(`seed: expected ${expectedSeed}, got ${r.seed}`);
    sa3Fail = true;
  } else {
    console.log(`SA3 seed ok: ${r.seed} (baseSeed ${sa3BaseSeed} + slotIndex ${sa3SlotIndex})`);
  }
}

if (sa3Fail) {
  console.error('FAIL: SA3 slot check');
  for (const e of sa3Errors) console.error(' ', e);
  process.exit(1);
}
console.log('PASS: SA3 slot check');
