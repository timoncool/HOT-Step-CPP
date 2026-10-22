// abPair.ts - pure state for the Compare A/B quick switch.
// The pair is two explicit take slots {a, b}. Invariant: whenever 2+ takes
// exist, a and b are both valid take ids and a !== b.
// - Init: baseline (if set) or the first take in A, the next other take in B.
// - Set A / Set B: put a take into a slot; if it already sits in the other
//   slot, the two slots swap.
// - Tapping a row that is in neither slot replaces the slot that is NOT the
//   currently active take, so the A/B button flips between the take you were
//   on and the one you just tapped.
// - Removal: a slot whose take is gone picks the next other take.

export interface AbPair { a: string | null; b: string | null; }

/** Restore the invariant for the current take list. */
export function normalizePair(pair: AbPair, ids: string[], baselineId: string | null): AbPair {
  if (ids.length === 0) return { a: null, b: null };
  const first = baselineId && ids.includes(baselineId) ? baselineId : ids[0];
  let a = pair.a && ids.includes(pair.a) ? pair.a : null;
  let b = pair.b && ids.includes(pair.b) ? pair.b : null;
  if (a !== null && a === b) b = null;
  if (a === null) a = b !== first ? first : ids.find(id => id !== b) ?? null;
  if (b === null) b = ids.find(id => id !== a) ?? null;
  if (a === b) b = null;
  if (a === null && b !== null) { a = b; b = null; }
  return a === pair.a && b === pair.b ? pair : { a, b };
}

export function initialPair(ids: string[], baselineId: string | null): AbPair {
  return normalizePair({ a: null, b: null }, ids, baselineId);
}

/** Put a take into slot A or B; swaps if it already sits in the other slot. */
export function setSlot(pair: AbPair, slot: 'a' | 'b', id: string): AbPair {
  const other = slot === 'a' ? 'b' : 'a';
  if (pair[slot] === id) return pair;
  if (pair[other] === id) return slot === 'a' ? { a: id, b: pair.a } : { a: pair.b, b: id };
  return { ...pair, [slot]: id };
}

/** Row tap: a take outside the pair replaces the slot that is not active. */
export function tapTake(pair: AbPair, active: string | null, id: string): AbPair {
  if (id === pair.a || id === pair.b) return pair;
  if (active === pair.b) return { a: id, b: pair.b };
  return { a: pair.a, b: id };
}

/** Take to switch to when the A/B button is pressed. */
export function toggleTarget(pair: AbPair, current: string | null): string | null {
  if (current === pair.a) return pair.b;
  return pair.a;
}
