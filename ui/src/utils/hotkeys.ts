/**
 * Global desktop hotkeys: pure key-to-action mapping and the "should ignore" check.
 * No DOM or React imports, so it can be unit-checked in plain node.
 */

export type HotkeyAction =
  | 'togglePlay'
  | 'seekBack'
  | 'seekForward'
  | 'focusPrev'
  | 'focusNext'
  | 'playFocused'
  | 'deleteSelection'
  | 'clearSelection'
  | 'showHelp'
  | 'openPalette';

export interface KeyLike {
  key: string;
  ctrlKey?: boolean;
  metaKey?: boolean;
  altKey?: boolean;
  shiftKey?: boolean;
  repeat?: boolean;
}

export interface TargetLike {
  tagName?: string;
  isContentEditable?: boolean;
}

export interface IgnoreContext {
  target: TargetLike | null | undefined;
  modalOpen: boolean;
  pathname: string;
}

export const SEEK_STEP_SEC = 10;

/** Library-list actions are dispatched to the mounted SongList. */
export const LIBRARY_ACTIONS: ReadonlySet<HotkeyAction> = new Set<HotkeyAction>([
  'focusPrev', 'focusNext', 'playFocused', 'deleteSelection', 'clearSelection',
]);

/** Rows for the cheat-sheet modal: key label and i18n key of the action. */
export const HOTKEY_HELP: ReadonlyArray<{ keys: string; action: HotkeyAction }> = [
  { keys: 'Space / K', action: 'togglePlay' },
  { keys: 'J', action: 'seekBack' },
  { keys: 'L', action: 'seekForward' },
  { keys: 'Up / Down', action: 'focusPrev' },
  { keys: 'Enter', action: 'playFocused' },
  { keys: 'Delete', action: 'deleteSelection' },
  { keys: 'Esc', action: 'clearSelection' },
  { keys: '?', action: 'showHelp' },
  { keys: 'Ctrl+K / Cmd+K', action: 'openPalette' },
];

/** Map a key event to an action, or null. Modified chords (ctrl/meta/alt) never match. */
export function mapKey(e: KeyLike): HotkeyAction | null {
  if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && (e.key === 'k' || e.key === 'K')) {
    return e.repeat ? null : 'openPalette';
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return null;
  switch (e.key) {
    case ' ':
    case 'Spacebar':
      return e.repeat ? null : 'togglePlay';
    case 'k':
    case 'K':
      return e.repeat ? null : 'togglePlay';
    case 'j':
    case 'J':
      return 'seekBack';
    case 'l':
    case 'L':
      return 'seekForward';
    case 'ArrowUp':
      return 'focusPrev';
    case 'ArrowDown':
      return 'focusNext';
    case 'Enter':
      return e.repeat ? null : 'playFocused';
    case 'Delete':
      return e.repeat ? null : 'deleteSelection';
    case 'Escape':
      return 'clearSelection';
    case '?':
      return 'showHelp';
    default:
      return null;
  }
}

const EDITABLE_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);

/** True when the global handler must stay out of the way. */
export function shouldIgnore(ctx: IgnoreContext): boolean {
  if (ctx.modalOpen) return true;
  // Compare studio owns its own keys (B, Tab, ...).
  if (ctx.pathname === '/compare' || ctx.pathname.startsWith('/compare/')) return true;
  const t = ctx.target;
  if (t) {
    if (t.isContentEditable) return true;
    if (t.tagName && EDITABLE_TAGS.has(t.tagName.toUpperCase())) return true;
  }
  return false;
}

/** Clamp a seek target into [0, duration] (duration 0 or NaN = unknown, only floor at 0). */
export function seekTarget(current: number, delta: number, duration: number): number {
  const t = Math.max(0, current + delta);
  return Number.isFinite(duration) && duration > 0 ? Math.min(t, duration) : t;
}

/** Move a focus index through a list of ids; -1 / missing starts at the edge. */
export function moveFocus(order: readonly string[], focusedId: string | null, dir: 1 | -1): string | null {
  if (order.length === 0) return null;
  const i = focusedId ? order.indexOf(focusedId) : -1;
  if (i < 0) return dir === 1 ? order[0] : order[order.length - 1];
  const n = Math.min(order.length - 1, Math.max(0, i + dir));
  return order[n];
}
