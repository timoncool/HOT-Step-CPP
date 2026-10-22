// levelMatch.ts - pure level-match gain math for the A/B engine.
// No test runner exists in ui/, so this stays pure and dependency free.

/** Per-take gain in dB: targetLufs - takeLufs, where target is the quietest
 *  measured take. Takes without a measurement get 0 dB. */
export function levelMatchGainsDb(lufsByTake: Record<string, number | null | undefined>): Record<string, number> {
  const vals = Object.values(lufsByTake).filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
  const out: Record<string, number> = {};
  const target = vals.length ? Math.min(...vals) : 0;
  for (const [k, v] of Object.entries(lufsByTake)) {
    out[k] = typeof v === 'number' && Number.isFinite(v) ? target - v : 0;
  }
  return out;
}

export function dbToLinear(db: number): number {
  return Math.pow(10, db / 20);
}
