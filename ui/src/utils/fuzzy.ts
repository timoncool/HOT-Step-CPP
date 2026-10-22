/**
 * Fuzzy matching for the command palette. Pure: no DOM or React imports,
 * so it can be unit-checked in plain node.
 */

/**
 * Score how well `query` matches `text`. Returns null when the query
 * characters do not all appear in order. Higher is better.
 * Rewards: exact match, prefix, contiguous substring, matches at word starts,
 * consecutive runs; penalises gaps and long texts slightly.
 */
export function fuzzyScore(query: string, text: string): number | null {
  const q = query.trim().toLowerCase();
  const s = text.toLowerCase();
  if (!q) return 0;
  if (s === q) return 1000;
  let score = 0;
  const sub = s.indexOf(q);
  if (sub === 0) score += 500;
  else if (sub > 0) score += isWordStart(s, sub) ? 350 : 250;

  let qi = 0;
  let prev = -2;
  let firstAt = -1;
  for (let i = 0; i < s.length && qi < q.length; i++) {
    if (s[i] !== q[qi]) continue;
    if (firstAt < 0) firstAt = i;
    score += 10;
    if (i === prev + 1) score += 15;
    else if (prev >= 0) score -= Math.min(10, i - prev - 1);
    if (isWordStart(s, i)) score += 20;
    prev = i;
    qi++;
  }
  if (qi < q.length) return null;
  score -= Math.min(20, firstAt);
  score -= Math.min(20, Math.floor((s.length - q.length) / 4));
  return score;
}

function isWordStart(s: string, i: number): boolean {
  if (i === 0) return true;
  const p = s[i - 1];
  return p === ' ' || p === '-' || p === '_' || p === '/' || p === '.' || p === '(';
}

/** Filter and sort items by score, best first; ties keep input order. */
export function fuzzyRank<T>(items: readonly T[], query: string, getText: (t: T) => string): T[] {
  if (!query.trim()) return [...items];
  const scored: Array<{ item: T; score: number; idx: number }> = [];
  items.forEach((item, idx) => {
    const sc = fuzzyScore(query, getText(item));
    if (sc !== null) scored.push({ item, score: sc, idx });
  });
  scored.sort((a, b) => b.score - a.score || a.idx - b.idx);
  return scored.map(x => x.item);
}
