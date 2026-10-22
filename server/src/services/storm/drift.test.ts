// drift.test.ts -- unit tests for the Storm Directed Drift core.
// Run: cd server && npx tsx --test src/services/storm/drift.test.ts
//
// TEETH CHECK: these tests were first run against a stub applyDrift that
// returned the input unchanged, and verified to FAIL with the numbers printed
// before the real implementation was written.

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import {
  type DriftPressure,
  type DriftRuntime,
  ACE_PARAM_SPECS,
  YUE2_PARAM_SPECS,
  MOOD_PRESETS,
  initDrift,
  updateDriftPressure,
  applyDrift,
  snapshot,
  resolveStartDrift,
  defaultDriftPressure,
} from './drift.js';

import { buildSlotRequest, type StreamControlSnapshot } from './slot.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a minimal ACE runtime with one numeric target. */
function makeRuntime(
  startVal: number,
  targetVal: number,
  paramKey: string,
  overrides: Partial<DriftPressure> = {},
): DriftRuntime {
  const spec = ACE_PARAM_SPECS.find(s => s.key === paramKey);
  if (!spec) throw new Error(`Unknown param: ${paramKey}`);
  const norm = (v: number) => (v - spec.min) / (spec.max - spec.min);
  const state = initDrift(42);
  state.pos[paramKey] = norm(startVal);
  state.seed = 100;
  const pressure: DriftPressure = {
    targets: { [paramKey]: norm(targetVal) },
    targetCaption: [],
    captionPool: [],
    pull: 1,
    rate: 0.5,
    jitter: 0,
    locks: {},
    seedMode: 'lock',
    ...overrides,
  };
  return { state, pressure };
}

/** Denormalize a param value from state.pos. */
function denorm(runtime: DriftRuntime, key: string): number {
  const spec =
    ACE_PARAM_SPECS.find(s => s.key === key) ??
    YUE2_PARAM_SPECS.find(s => s.key === key);
  if (!spec) throw new Error(`Unknown spec: ${key}`);
  return spec.min + runtime.state.pos[key] * (spec.max - spec.min);
}

// ---------------------------------------------------------------------------
// Test 1: Convergence -- cfg 3 to 8 with pull=1, rate=0.5, jitter=0
//
// With guidanceScale range 1..15 (range=14), cap per slot = 0.125*14 = 1.75
// raw. From 3, slot 1 is capped to 3+1.75 = 4.75 -> rounded 4.8.
// After 6 slots the gap to 8 is < 0.2 (verified: slot 6 raw ~7.9, gap ~0.1).
// ---------------------------------------------------------------------------
describe('convergence', () => {
  it('slot 1 lands in [4.6, 4.9]', () => {
    const rt = makeRuntime(3, 8, 'guidanceScale');
    const req: Record<string, unknown> = { guidanceScale: 3 };
    applyDrift(req, rt, 'ace');
    const val = Number(req['guidanceScale']);
    console.log(`  convergence slot1: guidanceScale = ${val.toFixed(4)} (expected ~4.75)`);
    assert.ok(
      val >= 4.6 && val <= 4.9,
      `Expected [4.6, 4.9], got ${val}`,
    );
  });

  it('gap is under 0.2 after 6 slots', () => {
    const rt = makeRuntime(3, 8, 'guidanceScale');
    const req: Record<string, unknown> = { guidanceScale: 3 };
    for (let i = 0; i < 6; i++) applyDrift(req, rt, 'ace');
    const val = Number(req['guidanceScale']);
    const gap = Math.abs(8 - val);
    console.log(`  convergence 6slots: guidanceScale = ${val.toFixed(4)}, gap = ${gap.toFixed(4)} (expected < 0.2)`);
    assert.ok(gap < 0.2, `Expected gap < 0.2, got ${gap}`);
  });
});

// ---------------------------------------------------------------------------
// Test 2: Gradual -- target 3 to 15 (max), slot 1 at or below 6.0
//
// With range 1..15, cap = 1.75/slot. From 3, slot 1 = 4.8 which is << 6.0.
// The cap ensures a large jump never teleports more than 1.75 raw units.
// ---------------------------------------------------------------------------
describe('gradual', () => {
  it('slot 1 is at or below 6.0 for target jump 3->15', () => {
    const rt = makeRuntime(3, 15, 'guidanceScale');
    const req: Record<string, unknown> = { guidanceScale: 3 };
    applyDrift(req, rt, 'ace');
    const val = Number(req['guidanceScale']);
    console.log(`  gradual slot1: guidanceScale = ${val.toFixed(4)} (expected <= 6.0, cap = 1.75 raw/slot)`);
    assert.ok(val <= 6.0, `Expected <= 6.0, got ${val}`);
  });
});

// ---------------------------------------------------------------------------
// Test 3: Lock -- bit-identical over 20 slots; unlocked moves >= 20% of range
// ---------------------------------------------------------------------------
describe('lock', () => {
  it('locked param is bit-identical over 20 slots', () => {
    const rt = makeRuntime(3, 9, 'guidanceScale', { locks: { guidanceScale: true } });
    const req: Record<string, unknown> = { guidanceScale: 3 };
    for (let i = 0; i < 20; i++) applyDrift(req, rt, 'ace');
    const val = Number(req['guidanceScale']);
    console.log(`  lock: guidanceScale after 20 slots = ${val.toFixed(4)} (expected 3.0)`);
    assert.strictEqual(val, 3, `Locked param changed: ${val}`);
  });

  it('unlocked param moves at least 20% of range over 20 slots', () => {
    const rt = makeRuntime(3, 9, 'guidanceScale');
    const req: Record<string, unknown> = { guidanceScale: 3 };
    for (let i = 0; i < 20; i++) applyDrift(req, rt, 'ace');
    const val = Number(req['guidanceScale']);
    const spec = ACE_PARAM_SPECS.find(s => s.key === 'guidanceScale')!;
    const range = spec.max - spec.min;
    const moved = Math.abs(val - 3);
    const pct = moved / range;
    console.log(`  unlock: guidanceScale moved ${moved.toFixed(3)} of range ${range}, ${(pct * 100).toFixed(1)}% (expected >= 20%)`);
    assert.ok(pct >= 0.2, `Expected >= 20% of range, got ${(pct * 100).toFixed(1)}%`);
  });
});

// ---------------------------------------------------------------------------
// Test 4: Zero pull -- stays in bounds; std over 5% of range across 50 slots
// ---------------------------------------------------------------------------
describe('zero pull', () => {
  it('stays in [min,max] and std > 5% of range over 50 slots', () => {
    const spec = ACE_PARAM_SPECS.find(s => s.key === 'guidanceScale')!;
    const state = initDrift(77);
    state.pos['guidanceScale'] = 0.5;
    state.seed = 50;
    const pressure: DriftPressure = {
      targets: { guidanceScale: 0.5 },
      targetCaption: [],
      captionPool: [],
      pull: 0,
      rate: 1,
      jitter: 1,
      locks: {},
      seedMode: 'lock',
    };
    const rt: DriftRuntime = { state, pressure };
    const vals: number[] = [];
    for (let i = 0; i < 50; i++) {
      const req: Record<string, unknown> = {};
      applyDrift(req, rt, 'ace');
      const v = Number(req['guidanceScale']);
      assert.ok(v >= spec.min && v <= spec.max, `Out of bounds: ${v}`);
      vals.push(v);
    }
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
    const range = spec.max - spec.min;
    const stdPct = std / range;
    console.log(`  zeroPull: std = ${std.toFixed(3)}, range = ${range}, std/range = ${(stdPct * 100).toFixed(1)}% (expected > 5%)`);
    assert.ok(stdPct > 0.05, `Expected std > 5% of range, got ${(stdPct * 100).toFixed(1)}%`);
  });
});

// ---------------------------------------------------------------------------
// Test 5: Caption adoption within ceil(ln .5 / ln(1 - rate*pull)) + 1 slots
// ---------------------------------------------------------------------------
describe('caption', () => {
  it('target tag appears in caption within the expected slots', () => {
    const rate = 0.5;
    const pull = 0.8;
    const maxSlots = Math.ceil(Math.log(0.5) / Math.log(1 - rate * pull)) + 1;
    console.log(`  caption: maxSlots = ${maxSlots}`);

    const state = initDrift(11);
    state.seed = 1;
    const pressure: DriftPressure = {
      targets: {},
      targetCaption: ['epic'],
      captionPool: ['epic', 'dark', 'chill'],
      pull,
      rate,
      jitter: 0,
      locks: {},
      seedMode: 'lock',
    };
    const rt: DriftRuntime = { state, pressure };

    let adoptedAt = -1;
    for (let i = 0; i < maxSlots + 2; i++) {
      const req: Record<string, unknown> = { caption: '', prompt: '' };
      applyDrift(req, rt, 'ace');
      const caption = String(req['caption'] ?? '');
      if (caption.includes('epic')) {
        adoptedAt = i + 1;
        break;
      }
    }
    console.log(`  caption: 'epic' adopted at slot ${adoptedAt} (expected <= ${maxSlots})`);
    assert.ok(adoptedAt !== -1 && adoptedAt <= maxSlots, `Expected adoption within ${maxSlots} slots, got ${adoptedAt}`);
  });
});

// ---------------------------------------------------------------------------
// Test 6: BPM delta and Camelot step caps
// ---------------------------------------------------------------------------
describe('limits', () => {
  it('BPM delta per slot stays within max(1, 4*rate)', () => {
    const state = initDrift(5);
    state.pos['bpm'] = 0.3;
    state.seed = 1;
    const pressure: DriftPressure = {
      targets: { bpm: 1.0 },
      targetCaption: [],
      captionPool: [],
      pull: 1,
      rate: 0.5,
      jitter: 0,
      locks: {},
      seedMode: 'lock',
    };
    const rt: DriftRuntime = { state, pressure };
    const spec = ACE_PARAM_SPECS.find(s => s.key === 'bpm')!;
    let prevBpm = spec.min + state.pos['bpm'] * (spec.max - spec.min);
    const maxDelta = Math.max(1, 4 * pressure.rate);
    for (let i = 0; i < 10; i++) {
      const req: Record<string, unknown> = { bpm: prevBpm };
      applyDrift(req, rt, 'ace');
      const bpm = Number(req['bpm']);
      const delta = Math.abs(bpm - prevBpm);
      console.log(`  bpm slot ${i + 1}: ${prevBpm.toFixed(1)} -> ${bpm.toFixed(1)}, delta=${delta.toFixed(2)} (max=${maxDelta})`);
      assert.ok(delta <= maxDelta + 0.001, `BPM delta ${delta} exceeds max ${maxDelta}`);
      prevBpm = bpm;
    }
  });

  it('Camelot step is at most 1 wheel position per slot AND moves at least once over 20 slots', () => {
    // TEETH CHECK: this test was designed to fail against a stub camelotStep
    // that jumps 2 positions. A stub returning "C -> D -> E" (2 steps each)
    // would give circular distance 2, failing the <= 1 assertion. The "moves
    // at least once" assertion also fails against a stub that never changes the
    // key. Both were verified to fail before the real implementation was added.

    // Camelot wheel: each key maps to a number 1..12 + mode A/B.
    // Circular distance on the 12-number wheel:
    const KEY_TO_NUM: Record<string, number> = {
      C:8, G:9, D:10, A:11, E:12, B:1, 'F#':2, 'C#':3, 'G#':4, 'D#':5, 'A#':6, F:7,
      Am:8, Em:9, Bm:10, 'F#m':11, 'C#m':12, 'G#m':1, 'D#m':2, 'A#m':3, Fm:4, Cm:5, Gm:6, Dm:7,
    };
    function camDist(a: string, b: string): number {
      if (a === b) return 0;
      const na = KEY_TO_NUM[a]; const nb = KEY_TO_NUM[b];
      if (na === undefined || nb === undefined) return 0; // unknown key -- skip
      const d = Math.abs(na - nb);
      return Math.min(d, 12 - d);
    }

    const state = initDrift(3);
    state.seed = 1;
    const startKey = 'C';
    const pressure: DriftPressure = {
      targets: {},
      targetCaption: [],
      captionPool: [],
      pull: 1,
      rate: 1,
      jitter: 0,
      locks: {},
      seedMode: 'lock',
    };
    const rt: DriftRuntime = { state, pressure };

    let prevKey = startKey;
    let moved = 0;
    for (let i = 0; i < 20; i++) {
      const req: Record<string, unknown> = { keyScale: prevKey };
      applyDrift(req, rt, 'ace');
      const newKey = String(req['keyScale'] ?? prevKey);
      const dist = camDist(prevKey, newKey);
      console.log(`  camelot slot ${i + 1}: ${prevKey} -> ${newKey} (dist=${dist})`);
      assert.ok(dist <= 1, `Camelot jumped ${dist} positions at slot ${i + 1}: ${prevKey} -> ${newKey}`);
      if (newKey !== prevKey) moved++;
      prevKey = newKey;
    }
    console.log(`  camelot: moved ${moved}/20 slots, final key = ${prevKey} (expected >= 1 move)`);
    assert.ok(moved >= 1, `Expected key to move at least once in 20 slots with pull=1 rate=1, got 0`);
  });
});

// ---------------------------------------------------------------------------
// Test 7: Determinism -- same rngSeed gives same sequence
// ---------------------------------------------------------------------------
describe('determinism', () => {
  it('same rngSeed produces identical sequences', () => {
    function runSequence(rngSeed: number): number[] {
      const state = initDrift(rngSeed);
      state.pos['guidanceScale'] = 0.3;
      state.seed = 1;
      const pressure: DriftPressure = {
        targets: { guidanceScale: 0.7 },
        targetCaption: [],
        captionPool: [],
        pull: 0.5,
        rate: 0.5,
        jitter: 0.5,
        locks: {},
        seedMode: 'lock',
      };
      const rt: DriftRuntime = { state, pressure };
      const vals: number[] = [];
      for (let i = 0; i < 10; i++) {
        const req: Record<string, unknown> = {};
        applyDrift(req, rt, 'ace');
        vals.push(Number(req['guidanceScale']));
      }
      return vals;
    }

    const a = runSequence(42);
    const b = runSequence(42);
    const c = runSequence(99);
    console.log(`  determinism: a[0]=${a[0].toFixed(4)}, b[0]=${b[0].toFixed(4)}, c[0]=${c[0].toFixed(4)}`);
    assert.deepStrictEqual(a, b, 'Same seed should produce identical sequence');
    assert.notDeepStrictEqual(a, c, 'Different seeds should differ');
  });
});

// ---------------------------------------------------------------------------
// Test 8: Integration -- guidanceScale strictly increases over 3 slots
// ---------------------------------------------------------------------------
describe('integration', () => {
  it('guidanceScale strictly increases over 3 slots when start < target', () => {
    const rt = makeRuntime(5, 12, 'guidanceScale');
    const vals: number[] = [5];
    const req: Record<string, unknown> = { guidanceScale: 5 };
    for (let i = 0; i < 3; i++) {
      applyDrift(req, rt, 'ace');
      vals.push(Number(req['guidanceScale']));
    }
    console.log(`  integration: guidanceScale sequence = ${vals.map(v => v.toFixed(3)).join(' -> ')}`);
    for (let i = 1; i < vals.length; i++) {
      assert.ok(vals[i] > vals[i - 1], `Not strictly increasing at step ${i}: ${vals[i - 1]} -> ${vals[i]}`);
    }
  });
});

// ---------------------------------------------------------------------------
// Test 9: Ear -- 6+ slots with a genre pull (caption reflects target genre)
// ---------------------------------------------------------------------------
describe('ear', () => {
  it('genre caption appears within 6 slots', () => {
    const state = initDrift(22);
    state.seed = 1;
    const pressure: DriftPressure = {
      targets: {},
      targetCaption: ['electronic', 'synth', 'techno'],
      captionPool: ['acoustic', 'jazz', 'electronic', 'synth', 'techno', 'folk'],
      pull: 1,
      rate: 1,
      jitter: 0,
      locks: {},
      seedMode: 'lock',
    };
    const rt: DriftRuntime = { state, pressure };

    let found = false;
    for (let i = 0; i < 6; i++) {
      const req: Record<string, unknown> = { caption: '', prompt: '' };
      applyDrift(req, rt, 'ace');
      const caption = String(req['caption'] ?? '');
      if (['electronic', 'synth', 'techno'].some(t => caption.includes(t))) {
        found = true;
        console.log(`  ear: genre tag appeared at slot ${i + 1}: "${caption}"`);
        break;
      }
    }
    assert.ok(found, 'Expected at least one genre tag to appear within 6 slots');
  });
});

// ---------------------------------------------------------------------------
// Test 10: Plugin toggle off -- never changes solver over 100 slots
// ---------------------------------------------------------------------------
describe('plugin toggle', () => {
  it('plugin drift off: solver unchanged over 100 slots', () => {
    const state = initDrift(99);
    state.seed = 1;
    const pressure: DriftPressure = {
      targets: {},
      targetCaption: [],
      captionPool: [],
      pull: 1,
      rate: 1,
      jitter: 1,
      locks: {},
      seedMode: 'lock',
      pluginDrift: false,
      pluginCandidates: {
        inferMethod: ['euler', 'heun', 'midpoint', 'dpm2'],
      },
    };
    const rt: DriftRuntime = { state, pressure };
    const startSolver = 'euler';
    let changed = 0;
    for (let i = 0; i < 100; i++) {
      const req: Record<string, unknown> = { inferMethod: startSolver };
      applyDrift(req, rt, 'ace');
      if (req['inferMethod'] !== startSolver) changed++;
    }
    console.log(`  pluginOff: solver changes in 100 slots = ${changed} (expected 0)`);
    assert.strictEqual(changed, 0, `Expected 0 solver changes with toggle off, got ${changed}`);
  });

  it('plugin drift on (jitter=1, rate=1): solver changes at least once in 100 slots', () => {
    const state = initDrift(99);
    state.seed = 1;
    const pressure: DriftPressure = {
      targets: {},
      targetCaption: [],
      captionPool: [],
      pull: 1,
      rate: 1,
      jitter: 1,
      locks: {},
      seedMode: 'lock',
      pluginDrift: true,
      pluginCandidates: {
        inferMethod: ['euler', 'heun', 'midpoint', 'dpm2'],
      },
    };
    const rt: DriftRuntime = { state, pressure };
    const startSolver = 'euler';
    let changed = 0;
    for (let i = 0; i < 100; i++) {
      const req: Record<string, unknown> = { inferMethod: startSolver };
      applyDrift(req, rt, 'ace');
      if (req['inferMethod'] !== startSolver) changed++;
    }
    console.log(`  pluginOn: solver changes in 100 slots = ${changed} (expected >= 1)`);
    assert.ok(changed >= 1, `Expected at least 1 solver change with toggle on, got ${changed}`);
  });
});

// ---------------------------------------------------------------------------
// Test 11: Mood presets exist
// ---------------------------------------------------------------------------
describe('mood presets', () => {
  it('exports dark/bright/chill/intense/dreamy presets', () => {
    for (const key of ['dark', 'bright', 'chill', 'intense', 'dreamy']) {
      assert.ok(Array.isArray(MOOD_PRESETS[key]), `Missing preset: ${key}`);
      assert.ok(MOOD_PRESETS[key].length > 0, `Empty preset: ${key}`);
    }
    console.log(`  moodPresets: keys = ${Object.keys(MOOD_PRESETS).join(', ')}`);
  });
});

// ---------------------------------------------------------------------------
// Test 12: snapshot() returns a copy of the current runtime
// ---------------------------------------------------------------------------
describe('snapshot', () => {
  it('snapshot returns pos and pressure pull', () => {
    const rt = makeRuntime(5, 9, 'guidanceScale');
    const snap = snapshot(rt);
    assert.ok('pos' in snap, 'snapshot missing pos');
    assert.ok('pull' in snap, 'snapshot missing pull');
    console.log(`  snapshot: pull = ${snap.pull}, pos.guidanceScale = ${snap.pos['guidanceScale']?.toFixed(4)}`);
  });
});

// ---------------------------------------------------------------------------
// Test 13 (Issue 2): initDrift seeds from baseSeed so walk starts there
// TEETH CHECK: fails on seed:0 (slot0 seed = 1 < 12345), passes after fix.
// ---------------------------------------------------------------------------
describe('initDrift baseSeed', () => {
  it('walk mode: first slot seed > baseSeed and < baseSeed+10', () => {
    const baseSeed = 12345;
    const state = initDrift(baseSeed);
    const pressure: DriftPressure = {
      targets: {},
      targetCaption: [],
      captionPool: [],
      pull: 1,
      rate: 1,
      jitter: 0,
      locks: {},
      seedMode: 'walk',
    };
    const rt: DriftRuntime = { state, pressure };
    const req: Record<string, unknown> = {};
    applyDrift(req, rt, 'ace');
    const seed = Number(req['seed']);
    console.log(`  initDrift baseSeed: seed after 1 walk slot = ${seed} (expected > ${baseSeed} and < ${baseSeed + 10})`);
    assert.ok(seed > baseSeed, `Expected seed > ${baseSeed}, got ${seed}`);
    assert.ok(seed < baseSeed + 10, `Expected seed < ${baseSeed + 10}, got ${seed}`);
  });
});

// ---------------------------------------------------------------------------
// Test 14 (Issue 4): updateDriftPressure validates and clamps inputs
// TEETH CHECK: fails with Object.assign (rate becomes "abc", NaN enters
// targets), passes after full validation is added.
// ---------------------------------------------------------------------------
describe('updateDriftPressure validation', () => {
  it('rate "abc" and targets {guidanceScale: NaN, bogus: 5} leave state valid', () => {
    const state = initDrift(1);
    const pressure: DriftPressure = {
      targets: { guidanceScale: 0.5 },
      targetCaption: [],
      captionPool: [],
      pull: 0.5,
      rate: 0.5,
      jitter: 0.1,
      locks: {},
      seedMode: 'walk',
    };
    const rt: DriftRuntime = { state, pressure };

    updateDriftPressure(rt, {
      rate: 'abc' as unknown as number,
      targets: { guidanceScale: NaN, bogus: 5 } as Record<string, number>,
    } as Partial<DriftPressure>);

    const rateValid = typeof rt.pressure.rate === 'number' && Number.isFinite(rt.pressure.rate);
    const gsValid = typeof rt.pressure.targets['guidanceScale'] === 'number'
      && Number.isFinite(rt.pressure.targets['guidanceScale']);
    const noBogus = !('bogus' in rt.pressure.targets);
    console.log(`  validation: rate=${rt.pressure.rate} (finite=${rateValid}), guidanceScale=${rt.pressure.targets['guidanceScale']} (finite=${gsValid}), bogus absent=${noBogus}`);
    assert.ok(rateValid, `rate should be finite, got ${rt.pressure.rate}`);
    assert.ok(gsValid, `guidanceScale target should be finite, got ${rt.pressure.targets['guidanceScale']}`);
    assert.ok(noBogus, `bogus key should not appear in targets`);
  });
});

// ---------------------------------------------------------------------------
// Test 15 (Issue 1): one-shot fields win over drift in buildSlotRequest
// TEETH CHECK: fails in old slot.ts where nextSeed/bpm/duration are placed
// in slotReq before applyDrift, so drift overwrites them. Passes after the
// fix that applies one-shots after drift.
// ---------------------------------------------------------------------------
describe('one-shot priority in buildSlotRequest', () => {
  it('nextSeed=777, nextBpm=140, nextDuration=90 survive walk-mode drift', () => {
    const baseSeed = 9999;
    const state = initDrift(baseSeed);
    const pressure: DriftPressure = {
      targets: { bpm: 1.0, duration: 1.0 },
      targetCaption: [],
      captionPool: [],
      pull: 1,
      rate: 1,
      jitter: 0,
      locks: {},
      seedMode: 'walk',
    };
    const driftRt: DriftRuntime = { state, pressure };

    const baseParams: Record<string, unknown> = {
      guidanceScale: 7,
      randomSeed: false,
      seed: baseSeed,
    };
    const ctrl: StreamControlSnapshot = {
      guidanceScale: 7,
      lssStrength: 0,
      inferenceSteps: 0,
      duration: 0,
      bpm: 0,
      seedLock: false,
      nextSeed: 777,
      nextBpm: 140,
      nextDuration: 90,
      nextPrompt: null,
      nextLyrics: null,
      baseCaption: null,
      baseLyrics: null,
      baseInstrumental: null,
      streamPaused: false,
      extraPluginParams: {},
      inferMethod: null,
      scheduler: null,
      guidanceMode: null,
      drift: driftRt,
    };

    const result = buildSlotRequest(baseParams, ctrl, 0, baseSeed, 'ace');
    assert.strictEqual(result.kind, 'ace');
    const aceReq = result.aceReq as unknown as Record<string, unknown>;
    const seed = Number(aceReq.seed);
    const bpm = Number(aceReq.bpm);
    const duration = Number(aceReq.duration);
    console.log(`  oneshot priority: seed=${seed} (expected 777), bpm=${bpm} (expected 140), duration=${duration} (expected 90)`);
    assert.strictEqual(seed, 777, `seed should be 777 (one-shot), got ${seed}`);
    assert.strictEqual(bpm, 140, `bpm should be 140 (one-shot), got ${bpm}`);
    assert.strictEqual(duration, 90, `duration should be 90 (one-shot), got ${duration}`);
  });

  it('seedLock=true keeps seed at baseSeed=12345 for 10 slots with walk drift', () => {
    const baseSeed = 12345;
    const state = initDrift(baseSeed);
    const pressure: DriftPressure = {
      targets: {},
      targetCaption: [],
      captionPool: [],
      pull: 1,
      rate: 1,
      jitter: 0,
      locks: {},
      seedMode: 'walk',
    };
    const driftRt: DriftRuntime = { state, pressure };

    const baseParams: Record<string, unknown> = { guidanceScale: 7, randomSeed: false, seed: baseSeed };
    const ctrl: StreamControlSnapshot = {
      guidanceScale: 7,
      lssStrength: 0,
      inferenceSteps: 0,
      duration: 0,
      bpm: 0,
      seedLock: true,
      nextSeed: null,
      nextBpm: null,
      nextDuration: null,
      nextPrompt: null,
      nextLyrics: null,
      baseCaption: null,
      baseLyrics: null,
      baseInstrumental: null,
      streamPaused: false,
      extraPluginParams: {},
      inferMethod: null,
      scheduler: null,
      guidanceMode: null,
      drift: driftRt,
    };

    const seeds: number[] = [];
    for (let i = 0; i < 10; i++) {
      const result = buildSlotRequest(baseParams, ctrl, i, baseSeed, 'ace');
      assert.strictEqual(result.kind, 'ace');
      seeds.push(Number((result.aceReq as unknown as Record<string, unknown>).seed));
    }
    console.log(`  seedLock: seeds over 10 slots = ${seeds.join(', ')} (expected all ${baseSeed})`);
    for (let i = 0; i < 10; i++) {
      assert.strictEqual(seeds[i], baseSeed, `slot ${i}: expected seed=${baseSeed}, got ${seeds[i]}`);
    }
  });
});

// ---------------------------------------------------------------------------
// Test 16: resolveStartDrift (stream-start drift resolution).
//
// TEETH CHECK: run against a helper that mimicked the old generate.ts logic
// (any existing ctrl.drift treated as pending, start patch merged over it);
// the leftover and stale-lock tests FAILED, then PASSED on the real helper.
// ---------------------------------------------------------------------------
function staleRuntime(): DriftRuntime {
  const rt: DriftRuntime = { state: initDrift(7), pressure: defaultDriftPressure() };
  updateDriftPressure(rt, {
    pull: 0.9, rate: 0.8, targets: { guidanceScale: 0.9 },
    locks: { guidanceScale: true }, captionPool: ['stale tag'],
  });
  rt.state.slot = 12;
  return rt;
}

describe('resolveStartDrift', () => {
  it('leftover (not pending) and no start drift gives undefined', () => {
    const out = resolveStartDrift(staleRuntime(), false, undefined, 123);
    console.log(`  leftover/no-start: ${out === undefined ? 'undefined' : 'pull=' + out.pressure.pull}`);
    assert.strictEqual(out, undefined);
  });

  it('leftover plus start drift gives fresh state without stale locks or targets', () => {
    const out = resolveStartDrift(staleRuntime(), false, { pull: 0.3 }, 123);
    assert.ok(out);
    const p = out.pressure;
    console.log(`  leftover+start: pull=${p.pull} rate=${p.rate} targets=${JSON.stringify(p.targets)} locks=${JSON.stringify(p.locks)} pool=${JSON.stringify(p.captionPool)} slot=${out.state.slot}`);
    assert.strictEqual(p.pull, 0.3);
    assert.strictEqual(p.rate, 0.5);
    assert.deepStrictEqual(p.targets, {});
    assert.deepStrictEqual(p.locks, {});
    assert.deepStrictEqual(p.captionPool, []);
    assert.strictEqual(out.state.slot, 0);
    assert.strictEqual(out.state.rngSeed, 123);
  });

  it('pending plus no start drift keeps pending pressure, re-seeded to baseSeed', () => {
    const out = resolveStartDrift(staleRuntime(), true, undefined, 555);
    assert.ok(out);
    console.log(`  pending: pull=${out.pressure.pull} rate=${out.pressure.rate} rngSeed=${out.state.rngSeed} slot=${out.state.slot}`);
    assert.strictEqual(out.pressure.pull, 0.9);
    assert.strictEqual(out.pressure.rate, 0.8);
    assert.strictEqual(out.state.rngSeed, 555);
    assert.strictEqual(out.state.slot, 0);
  });

  it('out-of-range start values are clamped', () => {
    const out = resolveStartDrift(undefined, false, { pull: 2, rate: -0.5, jitter: 9 }, 1);
    assert.ok(out);
    console.log(`  clamp: pull=${out.pressure.pull} rate=${out.pressure.rate} jitter=${out.pressure.jitter}`);
    assert.strictEqual(out.pressure.pull, 1);
    assert.strictEqual(out.pressure.rate, 0);
    assert.strictEqual(out.pressure.jitter, 1);
  });
});

// ---------------------------------------------------------------------------
// Test 17: SA3 drift specs and the held drift seed.
//
// TEETH CHECK: both SA3 tests failed before SA3_PARAM_SPECS joined the
// target/lock validation (sa3Steps was dropped); both seed tests failed while
// Lock left the slot on baseSeed + slotIndex.
// ---------------------------------------------------------------------------
function slotCtrl(drift: DriftRuntime, over: Partial<StreamControlSnapshot> = {}): StreamControlSnapshot {
  return {
    guidanceScale: 7, lssStrength: 0, inferenceSteps: 0, duration: 0, bpm: 0,
    seedLock: false, nextSeed: null, nextBpm: null, nextDuration: null,
    nextPrompt: null, nextLyrics: null, baseCaption: null, baseLyrics: null,
    baseInstrumental: null, streamPaused: false, extraPluginParams: {},
    inferMethod: null, scheduler: null, guidanceMode: null, drift, ...over,
  };
}

describe('SA3 drift specs', () => {
  it('sa3Steps target from start drift pulls SA3 steps', () => {
    const rt = resolveStartDrift(undefined, false, { targets: { sa3Steps: 1 }, pull: 1, rate: 1 }, 500);
    assert.ok(rt);
    assert.strictEqual(rt.pressure.targets['sa3Steps'], 1);
    const ctrl = slotCtrl(rt);
    const r = buildSlotRequest({ seed: 500, sa3Steps: 8 }, ctrl, 0, 500, 'sa3');
    assert.strictEqual(r.kind, 'sa3');
    console.log(`  sa3 target: steps=${r.kind === 'sa3' ? r.steps : '?'} (from 8, target 64)`);
    assert.ok(r.kind === 'sa3' && r.steps > 8, 'steps should move toward the target');
  });

  it('sa3Steps lock holds SA3 steps', () => {
    const rt = resolveStartDrift(undefined, false, { targets: { sa3Steps: 1 }, locks: ['sa3Steps'], pull: 1, rate: 1 }, 500);
    assert.ok(rt);
    assert.strictEqual(rt.pressure.locks['sa3Steps'], true);
    const r = buildSlotRequest({ seed: 500, sa3Steps: 8 }, slotCtrl(rt), 0, 500, 'sa3');
    assert.ok(r.kind === 'sa3' && r.steps === 8, 'locked steps should stay at 8');
  });
});

describe('drift seed Lock', () => {
  const seedOf = (r: ReturnType<typeof buildSlotRequest>) =>
    Number(r.kind === 'ace' ? (r.aceReq as unknown as Record<string, unknown>).seed : r.kind === 'sa3' ? r.seed : NaN);

  it('Lock renders the drift seed on every slot', () => {
    const rt = resolveStartDrift(undefined, false, { seedMode: 'lock', rate: 1 }, 500);
    assert.ok(rt);
    const ctrl = slotCtrl(rt);
    const seeds = [0, 1, 2].map(i => seedOf(buildSlotRequest({ seed: 500 }, ctrl, i, 500, 'ace')));
    console.log(`  lock seeds: ${seeds.join(', ')} (control ${rt.state.seed})`);
    assert.deepStrictEqual(seeds, [500, 500, 500]);
    assert.strictEqual(rt.state.seed, 500);
  });

  it('Lock returns to the drift seed after a one-shot seed', () => {
    const rt = resolveStartDrift(undefined, false, { seedMode: 'lock' }, 500);
    assert.ok(rt);
    const one = seedOf(buildSlotRequest({ seed: 500 }, slotCtrl(rt, { nextSeed: 777 }), 0, 500, 'sa3'));
    const next = seedOf(buildSlotRequest({ seed: 500 }, slotCtrl(rt), 1, 500, 'sa3'));
    console.log(`  one-shot then lock: ${one}, ${next}`);
    assert.strictEqual(one, 777);
    assert.strictEqual(next, 500);
  });

  it('Jump with zero jitter holds the drift seed', () => {
    const rt = resolveStartDrift(undefined, false, { seedMode: 'jump', jitter: 0 }, 500);
    assert.ok(rt);
    const ctrl = slotCtrl(rt);
    const seeds = [0, 1, 2].map(i => seedOf(buildSlotRequest({ seed: 500 }, ctrl, i, 500, 'ace')));
    assert.deepStrictEqual(seeds, [500, 500, 500]);
  });
});
