// server/src/services/storm/drift.ts
// Storm Directed Drift -- pure TS, no Express, no I/O.
//
// Provides DriftState, DriftPressure, per-backend ParamSpec tables (ACE and
// YuE2), seeded mulberry32 RNG, initDrift, updateDriftPressure, applyDrift,
// and snapshot.
//
// applyDrift mutates both the slotReq and the DriftRuntime state in place.
// It snapshots the pressure at entry so a concurrent POST cannot change it
// mid-slot.

// ---------------------------------------------------------------------------
// Camelot wheel (same source of truth as StormPage.tsx)
// ---------------------------------------------------------------------------

const KEY_TO_CAMELOT: Record<string, string> = {
  C: '8B', G: '9B', D: '10B', A: '11B', E: '12B', B: '1B',
  'F#': '2B', 'C#': '3B', 'G#': '4B', 'D#': '5B', 'A#': '6B', F: '7B',
  Am: '8A', Em: '9A', Bm: '10A', 'F#m': '11A', 'C#m': '12A', 'G#m': '1A',
  'D#m': '2A', 'A#m': '3A', Fm: '4A', Cm: '5A', Gm: '6A', Dm: '7A',
};

// Reverse map: Camelot notation -> canonical key string
const CAMELOT_TO_KEY: Record<string, string> = Object.fromEntries(
  Object.entries(KEY_TO_CAMELOT).map(([k, v]) => [v, k]),
);

/**
 * Step a Camelot key by +1 or -1 on the number (wrapping 1-12), preserving
 * mode (A/B). Returns the canonical key string (e.g. "C", "Am").
 * If the key is unknown, returns it unchanged.
 */
function camelotStep(key: string, direction: 1 | -1): string {
  const cam = KEY_TO_CAMELOT[key];
  if (!cam) return key;
  const num = parseInt(cam, 10);
  const mode = cam.slice(-1); // 'A' or 'B'
  const next = ((num - 1 + direction + 12) % 12) + 1;
  const nextCam = `${next}${mode}`;
  return CAMELOT_TO_KEY[nextCam] ?? key;
}

// ---------------------------------------------------------------------------
// ParamSpec
// ---------------------------------------------------------------------------

export interface ParamSpec {
  /** slotReq field name */
  key: string;
  min: number;
  max: number;
  /** Rounding step for output (e.g. 1 for integers). */
  step: number;
  /** Round to integer on denormalize. */
  int?: boolean;
}

/**
 * Numeric params that drift for the ACE backend.
 *
 * guidanceScale range matches the UI slider max (StormPage.tsx, StormLiveControls.tsx):
 * min=1, max=15. Range=14 in raw units; cap=0.125*14=1.75 per slot at rate=0.5.
 */
export const ACE_PARAM_SPECS: ParamSpec[] = [
  { key: 'guidanceScale',   min: 1,  max: 15,  step: 0.1  },
  { key: 'lssStrength',     min: 0,  max: 1,   step: 0.05 },
  { key: 'inferenceSteps',  min: 10, max: 100, step: 1, int: true },
  { key: 'bpm',             min: 60, max: 220, step: 1, int: true },
  { key: 'duration',        min: 15, max: 120, step: 1, int: true },
];

/** Numeric params that drift for the YuE2 backend. */
export const YUE2_PARAM_SPECS: ParamSpec[] = [
  { key: 'yue2CfgScale',      min: 1, max: 25,  step: 0.1  },
  { key: 'yue2OdeSteps',      min: 8, max: 64,  step: 1, int: true },
  { key: 'yue2NarCacheRatio', min: 0, max: 0.9, step: 0.05 },
];

/**
 * Numeric params that drift for the SA3 backend.
 *
 * guidanceScale is the live knob value; mapSa3SamplerPlugins gates it behind
 * samplerPluginsEnabled before sending to the engine. sa3Steps maps to the
 * steps param; duration is shared with ACE.
 */
export const SA3_PARAM_SPECS: ParamSpec[] = [
  { key: 'guidanceScale', min: 1, max: 15, step: 0.1       },
  { key: 'sa3Steps',      min: 4, max: 64, step: 1, int: true },
  { key: 'duration',      min: 5, max: 600, step: 1, int: true },
];

const SPEC_BY_BACKEND: Record<string, ParamSpec[]> = {
  ace: ACE_PARAM_SPECS,
  yue2: YUE2_PARAM_SPECS,
  sa3: SA3_PARAM_SPECS,
};

// ---------------------------------------------------------------------------
// Mood presets
// ---------------------------------------------------------------------------

/** Fixed mood preset tag sets. Keys are preset names; values are tag arrays. */
export const MOOD_PRESETS: Record<string, string[]> = {
  dark:    ['dark', 'melancholic', 'atmospheric', 'minor key', 'slow'],
  bright:  ['bright', 'uplifting', 'major key', 'energetic', 'upbeat'],
  chill:   ['chill', 'relaxed', 'ambient', 'smooth', 'lo-fi'],
  intense: ['intense', 'powerful', 'driving', 'aggressive', 'fast'],
  dreamy:  ['dreamy', 'ethereal', 'floating', 'reverb', 'lush'],
};

// ---------------------------------------------------------------------------
// DriftState / DriftPressure / DriftRuntime
// ---------------------------------------------------------------------------

/**
 * Per-slot mutable state owned by the route (one instance per stream).
 * All numeric param positions are stored normalized to [0, 1].
 */
export interface DriftState {
  /** Normalized [0,1] position for each driftable param key. */
  pos: Record<string, number>;
  /** Per-tag weight in [0,1]. Tags with weight > 0.5 are active. */
  captionWeights: Record<string, number>;
  /** Current generation seed. */
  seed: number;
  /** Monotonically increasing slot counter (used to advance RNG). */
  slot: number;
  /** Base RNG seed (set at stream start, determines the full sequence). */
  rngSeed: number;
}

/** SeedMode controls how the seed changes each slot. */
export type SeedMode = 'lock' | 'walk' | 'jump';

/**
 * User-controlled pressure: targets, rates, and locks.
 * Updated via POST /storm/control; read by applyDrift.
 */
export interface DriftPressure {
  /** Target normalized [0,1] for each numeric param key. */
  targets: Record<string, number>;
  /** Tags that should be active in the generated caption. */
  targetCaption: string[];
  /** Full tag pool (all tags that may appear in the output). */
  captionPool: string[];
  /** Pull strength in [0, 1]: how hard each slot is pulled toward the target. */
  pull: number;
  /** Rate in [0, 1]: scales step size and overall drift speed. */
  rate: number;
  /** Jitter in [0, 1]: scales the random noise term. */
  jitter: number;
  /** Which param keys are locked (no change this pressure cycle). */
  locks: Record<string, boolean>;
  /** How the seed evolves each slot. */
  seedMode: SeedMode;
  /**
   * Optional plugin choice drift (default off).
   * When true, inferMethod/scheduler/guidanceMode may be swapped to another
   * entry from pluginCandidates with probability p = 0.1 * jitter * rate per
   * slot.
   */
  pluginDrift?: boolean;
  /**
   * Candidate lists for plugin drift.
   * Each field is a list of valid values for the corresponding slotReq field.
   */
  pluginCandidates?: {
    inferMethod?: string[];
    scheduler?: string[];
    guidanceMode?: string[];
  };
  /** User-supplied custom tag sets (name -> tags). */
  customTagSets?: Record<string, string[]>;
}

/**
 * The full runtime object passed to applyDrift.
 * Typically stored as ctrl.drift inside StreamControlState.
 */
export interface DriftRuntime {
  state: DriftState;
  pressure: DriftPressure;
}

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------

/**
 * Mulberry32 seeded PRNG. Returns a () => number in [0, 1).
 * Each call to mulberry32() creates an independent sequence from that seed.
 */
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return function (): number {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), s | 1);
    t = (t + Math.imul(t ^ (t >>> 7), t | 61)) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 0x100000000;
  };
}

/**
 * Box-Muller transform: produces one standard normal sample from two uniforms.
 */
function randNormal(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-10); // avoid log(0)
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Return jitter * rate * 0.15 * N(0,1).
 * At jitter=0 returns exactly 0 without consuming RNG values so that
 * jitter=0 runs are fully deterministic even if N(0,1) is expensive.
 */
function jitterNoise(rng: () => number, jitter: number, rate: number): number {
  if (jitter === 0) return 0;
  return jitter * rate * 0.15 * randNormal(rng);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Create a fresh DriftState with empty pos and captionWeights.
 * Call once at stream start; pass the same object to every applyDrift call.
 */
export function initDrift(rngSeed: number): DriftState {
  return {
    pos: {},
    captionWeights: {},
    seed: rngSeed,
    slot: 0,
    rngSeed,
  };
}

// Keys that are valid lock targets.
const ALL_PARAM_SPECS = Object.values(SPEC_BY_BACKEND).flat();
const LOCKABLE_KEYS = new Set<string>([
  ...ALL_PARAM_SPECS.map(s => s.key),
  'keyScale', 'seed',
]);

/**
 * Merge a partial pressure patch into the runtime with full validation.
 * Called by the control-endpoint handler on POST /storm/control.
 * Silently ignores unknown keys and invalid values; never throws.
 */
export function updateDriftPressure(
  runtime: DriftRuntime,
  patch: Partial<DriftPressure>,
): void {
  const p = runtime.pressure;

  // pull / rate / jitter: finite numbers clamped to [0, 1].
  if (typeof patch.pull === 'number' && Number.isFinite(patch.pull)) {
    p.pull = Math.max(0, Math.min(1, patch.pull));
  }
  if (typeof patch.rate === 'number' && Number.isFinite(patch.rate)) {
    p.rate = Math.max(0, Math.min(1, patch.rate));
  }
  if (typeof patch.jitter === 'number' && Number.isFinite(patch.jitter)) {
    p.jitter = Math.max(0, Math.min(1, patch.jitter));
  }

  // targets: only known ParamSpec keys with finite normalized values [0, 1].
  if (patch.targets !== null && typeof patch.targets === 'object' && !Array.isArray(patch.targets)) {
    for (const [k, v] of Object.entries(patch.targets)) {
      const spec = ALL_PARAM_SPECS.find(s => s.key === k);
      if (!spec) continue;
      const num = Number(v);
      if (!Number.isFinite(num)) continue;
      p.targets[k] = Math.max(0, Math.min(1, num));
    }
  }

  // locks: array of known key strings, or Record<string, boolean>.
  if (patch.locks !== undefined) {
    if (Array.isArray(patch.locks)) {
      const newLocks: Record<string, boolean> = {};
      for (const k of patch.locks) {
        if (typeof k === 'string' && LOCKABLE_KEYS.has(k)) newLocks[k] = true;
      }
      p.locks = newLocks;
    } else if (typeof patch.locks === 'object' && patch.locks !== null) {
      const newLocks: Record<string, boolean> = {};
      for (const [k, v] of Object.entries(patch.locks)) {
        if (LOCKABLE_KEYS.has(k) && typeof v === 'boolean') newLocks[k] = v;
      }
      p.locks = newLocks;
    }
  }

  // seedMode: one of lock | walk | jump.
  const VALID_SEED_MODES = new Set<string>(['lock', 'walk', 'jump']);
  if (typeof patch.seedMode === 'string' && VALID_SEED_MODES.has(patch.seedMode)) {
    p.seedMode = patch.seedMode as SeedMode;
  }

  // targetCaption / captionPool: string arrays.
  if (Array.isArray(patch.targetCaption)) {
    p.targetCaption = patch.targetCaption.filter((t): t is string => typeof t === 'string');
  }
  if (Array.isArray(patch.captionPool)) {
    p.captionPool = patch.captionPool.filter((t): t is string => typeof t === 'string');
  }

  // pluginDrift boolean.
  if (typeof patch.pluginDrift === 'boolean') p.pluginDrift = patch.pluginDrift;

  // pluginCandidates: string arrays per known field.
  if (patch.pluginCandidates !== null && typeof patch.pluginCandidates === 'object') {
    const cands = patch.pluginCandidates;
    const validated: DriftPressure['pluginCandidates'] = {};
    for (const field of ['inferMethod', 'scheduler', 'guidanceMode'] as const) {
      if (Array.isArray(cands[field])) {
        validated[field] = (cands[field] as unknown[]).filter(
          (v): v is string => typeof v === 'string',
        );
      }
    }
    p.pluginCandidates = validated;
  }

  // customTagSets: Record<string, string[]>.
  if (patch.customTagSets !== null && typeof patch.customTagSets === 'object'
      && !Array.isArray(patch.customTagSets)) {
    const sets: Record<string, string[]> = {};
    for (const [k, v] of Object.entries(patch.customTagSets)) {
      if (Array.isArray(v)) sets[k] = v.filter((t): t is string => typeof t === 'string');
    }
    p.customTagSets = sets;
  }
}

/**
 * Apply one slot of drift to slotReq, mutating both slotReq and
 * runtime.state in place.
 *
 * The pressure is snapshotted at entry so a concurrent POST to the control
 * endpoint cannot alter it mid-slot (JS is single-threaded, but the route
 * may atomically replace runtime.pressure with a new object).
 */
export function applyDrift(
  slotReq: Record<string, unknown>,
  runtime: DriftRuntime,
  backend: string,
): void {
  // Snapshot pressure at entry.
  const p: DriftPressure = {
    ...runtime.pressure,
    targets:          { ...runtime.pressure.targets },
    locks:            { ...(runtime.pressure.locks ?? {}) },
    captionPool:      [...(runtime.pressure.captionPool ?? [])],
    targetCaption:    [...(runtime.pressure.targetCaption ?? [])],
    pluginCandidates: runtime.pressure.pluginCandidates
      ? { ...runtime.pressure.pluginCandidates }
      : undefined,
    customTagSets: runtime.pressure.customTagSets
      ? { ...runtime.pressure.customTagSets }
      : undefined,
  };

  const st = runtime.state;
  // Per-slot RNG: unique but deterministic for (rngSeed, slot).
  const rng = mulberry32(st.rngSeed + st.slot * 997);

  const specs = SPEC_BY_BACKEND[backend] ?? ACE_PARAM_SPECS;

  // ---- Numeric params ----
  for (const spec of specs) {
    if (p.locks[spec.key]) continue;

    // Initialize position from slotReq if not yet tracked.
    if (!(spec.key in st.pos)) {
      const raw = Number(slotReq[spec.key]);
      st.pos[spec.key] = Number.isFinite(raw)
        ? Math.max(0, Math.min(1, (raw - spec.min) / (spec.max - spec.min)))
        : 0.5;
    }

    const u = st.pos[spec.key];
    const targetU = spec.key in p.targets ? p.targets[spec.key] : u;

    // Per-slot rule (normalized space):
    //   step = rate*pull*(target-u) + jitter*rate*0.15*N(0,1)
    //   clamped to +-0.25*rate
    const noise = jitterNoise(rng, p.jitter, p.rate);
    let step = p.rate * p.pull * (targetU - u) + noise;
    const cap = 0.25 * p.rate;
    step = Math.max(-cap, Math.min(cap, step));

    let newU = Math.max(0, Math.min(1, u + step));

    // BPM: additional cap on raw delta (max(1, 4*rate) beats per slot).
    if (spec.key === 'bpm') {
      const prevRaw = spec.min + u * (spec.max - spec.min);
      const newRaw  = spec.min + newU * (spec.max - spec.min);
      const rawDelta = newRaw - prevRaw;
      const bpmCap = Math.max(1, 4 * p.rate);
      const clampedRaw = prevRaw + Math.max(-bpmCap, Math.min(bpmCap, rawDelta));
      newU = Math.max(0, Math.min(1, (clampedRaw - spec.min) / (spec.max - spec.min)));
    }

    st.pos[spec.key] = newU;

    // Denormalize and round.
    let val = spec.min + newU * (spec.max - spec.min);
    if (spec.int) {
      val = Math.round(val);
    } else {
      val = Math.round(val / spec.step) * spec.step;
    }
    slotReq[spec.key] = val;
  }

  // ---- Key / Camelot (ACE only) ----
  if (backend === 'ace' && !p.locks['keyScale']) {
    const currentKey = String(slotReq['keyScale'] ?? '');
    if (currentKey && KEY_TO_CAMELOT[currentKey]) {
      if (rng() < p.rate * p.pull) {
        const dir: 1 | -1 = rng() < 0.5 ? 1 : -1;
        slotReq['keyScale'] = camelotStep(currentKey, dir);
      }
    }
  }

  // ---- Caption / tag weights ----
  if (p.captionPool.length > 0) {
    for (const tag of p.captionPool) {
      // Initialize weight: target tags start slightly above 0.5; others below.
      if (!(tag in st.captionWeights)) {
        st.captionWeights[tag] = p.targetCaption.includes(tag) ? 0.55 : 0.45;
      }
      const w = st.captionWeights[tag];
      const targetW = p.targetCaption.includes(tag) ? 1.0 : 0.0;
      const noise2 = jitterNoise(rng, p.jitter, p.rate) * 0.33; // scale down for weights
      let wStep = p.rate * p.pull * (targetW - w) + noise2;
      const wCap = 0.25 * p.rate;
      wStep = Math.max(-wCap, Math.min(wCap, wStep));
      st.captionWeights[tag] = Math.max(0, Math.min(1, w + wStep));
    }

    const activeTags = p.captionPool.filter(t => (st.captionWeights[t] ?? 0) > 0.5);
    const caption = activeTags.join(', ');
    slotReq['caption'] = caption;
    slotReq['prompt']  = caption;
  }

  // ---- Seed ----
  if (!p.locks['seed']) {
    if (p.seedMode === 'walk') {
      st.seed = st.seed + 1 + Math.floor(p.jitter * 7 * rng());
    } else if (p.seedMode === 'jump') {
      if (rng() < 0.5 * p.jitter) {
        st.seed = (mulberry32(st.rngSeed + st.slot * 1009)() * 0xffffffff) >>> 0;
      }
    }
  }
  // Lock, a locked seed and a non-jumping Jump hold the drift seed. Without
  // this the slot keeps buildSlotRequest's baseSeed + slotIndex default.
  slotReq['seed'] = st.seed;

  // ---- Plugin drift (optional, off by default) ----
  if (p.pluginDrift === true && p.pluginCandidates) {
    const pSwap = 0.1 * p.jitter * p.rate;
    const cands = p.pluginCandidates;

    if (cands.inferMethod && cands.inferMethod.length > 1 && rng() < pSwap) {
      const pool = cands.inferMethod.filter(v => v !== slotReq['inferMethod']);
      if (pool.length > 0) {
        slotReq['inferMethod'] = pool[Math.floor(rng() * pool.length)];
      }
    }
    if (cands.scheduler && cands.scheduler.length > 1 && rng() < pSwap) {
      const pool = cands.scheduler.filter(v => v !== slotReq['scheduler']);
      if (pool.length > 0) {
        slotReq['scheduler'] = pool[Math.floor(rng() * pool.length)];
      }
    }
    if (cands.guidanceMode && cands.guidanceMode.length > 1 && rng() < pSwap) {
      const pool = cands.guidanceMode.filter(v => v !== slotReq['guidanceMode']);
      if (pool.length > 0) {
        slotReq['guidanceMode'] = pool[Math.floor(rng() * pool.length)];
      }
    }
  }

  // Advance slot counter.
  st.slot += 1;
}

/**
 * Return a snapshot of the current runtime for the GET /storm/control readout.
 * Contains normalized positions, caption weights, seed state, and key pressure
 * values. Returns copies so the caller cannot mutate the live state.
 */
export function snapshot(runtime: DriftRuntime): {
  pos: Record<string, number>;
  captionWeights: Record<string, number>;
  seed: number;
  slot: number;
  pull: number;
  rate: number;
  jitter: number;
  targetCaption: string[];
} {
  return {
    pos:            { ...runtime.state.pos },
    captionWeights: { ...runtime.state.captionWeights },
    seed:           runtime.state.seed,
    slot:           runtime.state.slot,
    pull:           runtime.pressure.pull,
    rate:           runtime.pressure.rate,
    jitter:         runtime.pressure.jitter,
    targetCaption:  [...(runtime.pressure.targetCaption ?? [])],
  };
}

/** Fresh default pressure (no targets, locks or caption pool). */
export function defaultDriftPressure(): DriftPressure {
  return {
    targets: {}, targetCaption: [], captionPool: [],
    pull: 0, rate: 0.5, jitter: 0, locks: {}, seedMode: 'lock',
  };
}

/**
 * Resolve the drift runtime at stream start.
 * - ctrlDrift is kept only when driftPending is true (set by a POST
 *   /storm/control while no stream was live). It is re-seeded with baseSeed
 *   and any startDrift patch is applied on top.
 * - Otherwise any leftover runtime is discarded: a fresh runtime is built from
 *   the default pressure plus startDrift, or undefined when there is no startDrift.
 * startDrift is always validated through updateDriftPressure (clamped).
 */
export function resolveStartDrift(
  ctrlDrift: DriftRuntime | undefined,
  driftPending: boolean,
  startDrift: unknown,
  baseSeed: number,
): DriftRuntime | undefined {
  const hasStart = !!startDrift && typeof startDrift === 'object' && !Array.isArray(startDrift);
  if (driftPending && ctrlDrift) {
    ctrlDrift.state = initDrift(baseSeed);
    if (hasStart) updateDriftPressure(ctrlDrift, startDrift as Partial<DriftPressure>);
    return ctrlDrift;
  }
  if (!hasStart) return undefined;
  const rt: DriftRuntime = { state: initDrift(baseSeed), pressure: defaultDriftPressure() };
  updateDriftPressure(rt, startDrift as Partial<DriftPressure>);
  return rt;
}
