// rangeSelect.ts: pure helpers for list multi-select (click, ctrl-toggle, shift-range).

export interface SelectState {
  selected: Set<string>;
  anchor: string | null;
}

/** Ids between a and b inclusive, in `order`. Empty if either id is not in the list. */
export function idsInRange(order: string[], a: string, b: string): string[] {
  const i = order.indexOf(a);
  const j = order.indexOf(b);
  if (i < 0 || j < 0) return [];
  const [lo, hi] = i <= j ? [i, j] : [j, i];
  return order.slice(lo, hi + 1);
}

/**
 * Apply one selection click.
 * - range: add every id from the anchor to target (in visible order); anchor stays.
 *   Falls back to a toggle when there is no usable anchor.
 * - otherwise: toggle target and make it the new anchor.
 */
export function applySelectClick(
  order: string[],
  state: SelectState,
  target: string,
  range: boolean,
): SelectState {
  const next = new Set(state.selected);
  if (range && state.anchor) {
    const ids = idsInRange(order, state.anchor, target);
    if (ids.length > 0) {
      for (const id of ids) next.add(id);
      return { selected: next, anchor: state.anchor };
    }
  }
  if (next.has(target)) next.delete(target); else next.add(target);
  return { selected: next, anchor: target };
}
