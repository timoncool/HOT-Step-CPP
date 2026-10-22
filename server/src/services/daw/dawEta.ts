// Wall-time estimates for DAW facade jobs, per backend.
// Defaults come from the warm (keep-loaded) bench of 2026-10-02 on GPU0,
// an RTX 4070 Ti Super 16 GB (docs/plans/hotstep-link-bench-2026-10-02.md,
// "Final warm table"). Cold-load figures are rough guesses, not measured.
// Real runs refine the figures in memory with an EMA (alpha 0.3).

export type EtaBackend = 'ace_turbo' | 'ace_base' | 'sa3' | 'yue2';
export const ETA_BACKENDS: EtaBackend[] = ['ace_turbo', 'ace_base', 'sa3', 'yue2'];

export interface EtaFigures {
  base_s: number;
  per_take_s: number;
  per_audio_s: number;
  cold_load_s: number;
  /** True when every take is a full render of its own (SA3 runs takes one by one). */
  full_per_take?: boolean;
  source: 'measured' | 'default';
}

const DEFAULTS: Record<EtaBackend, Omit<EtaFigures, 'source'>> = {
  // 10 s b1 4.1, 30 s b1 4.2, b4 2.2 to 3.3 s per extra take
  ace_turbo: { base_s: 4.05, per_take_s: 2.3, per_audio_s: 0.005, cold_load_s: 15 },
  // 10 s b1 4.1, 30 s b1 6.4, b4 2.3 to 4.7 s per extra take
  ace_base: { base_s: 2.95, per_take_s: 3.5, per_audio_s: 0.115, cold_load_s: 15 },
  // single take per job; extra takes run sequentially, so each take costs a
  // full render (base + per_audio * duration) and per_take_s is only overhead
  sa3: { base_s: 2.95, per_take_s: 0, per_audio_s: 0.115, cold_load_s: 20, full_per_take: true },
  // ignores duration_s; b1 65 to 82 s, b4 about 126 s
  yue2: { base_s: 73, per_take_s: 17.7, per_audio_s: 0, cold_load_s: 40 },
};

export const EMA_ALPHA = 0.3;

let table: Record<EtaBackend, EtaFigures>;
/** Backend whose weights are resident, or null when unknown. */
let loaded: EtaBackend | null = null;

export function resetEta(): void {
  table = Object.fromEntries(ETA_BACKENDS.map((b) => [b, { ...DEFAULTS[b], source: 'default' }])) as Record<EtaBackend, EtaFigures>;
  loaded = null;
}
resetEta();

export function getFigures(b: EtaBackend): EtaFigures { return { ...table[b] }; }
export function getLoaded(): EtaBackend | null { return loaded; }
export function setLoaded(b: EtaBackend | null): void { loaded = b; }

/** Map the active backend id plus DiT model name to an ETA key. */
export function etaKey(backendId: string | undefined, ditModel?: string | null): EtaBackend {
  if (backendId === 'sa3') return 'sa3';
  if (backendId === 'yue2') return 'yue2';
  return ditModel && /base/i.test(ditModel) && !/turbo/i.test(ditModel) ? 'ace_base' : 'ace_turbo';
}

/** Warm estimate, without the cold load. */
export function warmEstimate(f: Omit<EtaFigures, 'source'>, takes: number, durationS: number): number {
  const render = f.base_s + f.per_audio_s * durationS;
  return render * (f.full_per_take ? Math.max(1, takes) : 1) + f.per_take_s * Math.max(0, takes - 1);
}

/**
 * Estimated wall seconds. Any switch of backend counts as cold: SA3 and YuE2
 * do not both fit in 16 GB, so going between them reloads.
 */
export function estimate(b: EtaBackend, takes: number, durationS: number, loadedNow: EtaBackend | null = loaded): number {
  const f = table[b];
  return warmEstimate(f, takes, durationS) + (loadedNow === b ? 0 : f.cold_load_s);
}

/**
 * Learn from a finished job. The observed wall time (minus the cold load when
 * it was cold) pulls the warm estimate EMA_ALPHA of the way toward it, by
 * shifting base_s. Marks the backend measured. Marks it loaded only when the
 * caller says it is still the resident backend (markLoaded).
 */
export function recordRun(b: EtaBackend, takes: number, durationS: number, wallS: number, wasCold: boolean, markLoaded = false): void {
  if (!Number.isFinite(wallS) || wallS <= 0) return;
  const f = table[b];
  const observedWarm = Math.max(0, wallS - (wasCold ? f.cold_load_s : 0));
  const est = warmEstimate(f, takes, durationS);
  // base_s counts once per take when each take is a full render
  const scale = f.full_per_take ? Math.max(1, takes) : 1;
  f.base_s = Math.max(0, f.base_s + EMA_ALPHA * (observedWarm - est) / scale);
  f.source = 'measured';
  if (markLoaded) loaded = b;
}

export function etaByBackend(): Record<EtaBackend, EtaFigures> {
  return Object.fromEntries(ETA_BACKENDS.map((b) => [b, getFigures(b)])) as Record<EtaBackend, EtaFigures>;
}
