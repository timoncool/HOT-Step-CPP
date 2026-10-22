/**
 * Pure helpers for the generation-progress badge, the tab title and the
 * "job finished" notification. No DOM, no React: unit-checked with node.
 */

export interface JobLike {
  id: string;
  status: string;
  progress?: number;
  songId?: string;
  generation?: { title?: string } | unknown;
}

export interface JobSummary {
  /** Jobs waiting or rendering. */
  running: number;
  /** Rounded % of the job currently generating, or null when none reports one. */
  activePct: number | null;
}

const RUNNING = new Set(['pending', 'loading-adapter', 'generating']);

export function summarizeJobs(items: readonly JobLike[]): JobSummary {
  let running = 0;
  let activePct: number | null = null;
  for (const it of items) {
    if (!RUNNING.has(it.status)) continue;
    running++;
    if (activePct === null && it.status === 'generating' && typeof it.progress === 'number' && isFinite(it.progress)) {
      activePct = Math.round(Math.min(100, Math.max(0, it.progress)));
    }
  }
  return { running, activePct };
}

/** Short badge text: "2 · 43%", "1", or null when idle. */
export function formatBadge(s: JobSummary): string | null {
  if (s.running <= 0) return null;
  return s.activePct === null ? String(s.running) : `${s.running} · ${s.activePct}%`;
}

/** Tab title while running, e.g. "(2) 43% · HOT-Step"; the base title when idle. */
export function formatTitle(s: JobSummary, baseTitle: string, appName = 'HOT-Step'): string {
  if (s.running <= 0) return baseTitle;
  return s.activePct === null ? `(${s.running}) ${appName}` : `(${s.running}) ${s.activePct}% · ${appName}`;
}

/** Items that moved into 'succeeded' since the previous status snapshot.
 *  Items not seen before are ignored, so a restored queue never notifies. */
export function newlySucceeded<T extends JobLike>(prev: ReadonlyMap<string, string>, items: readonly T[]): T[] {
  const out: T[] = [];
  for (const it of items) {
    const before = prev.get(it.id);
    if (before !== undefined && before !== 'succeeded' && it.status === 'succeeded') out.push(it);
  }
  return out;
}

export function statusSnapshot(items: readonly JobLike[]): Map<string, string> {
  return new Map(items.map(i => [i.id, i.status]));
}
