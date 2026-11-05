// backends/registry.ts — id → EngineBackend map, active-backend selection
//
// Shape copied from services/lireek/llm/registry.ts (the provider registry
// blueprint named in the plan, §3.6). Active id persists through the same
// generic settings key-value table training/activeModels.ts already uses
// (server/src/db/lireekDb.ts `settings` table) — no new persistence
// mechanism, no schema change.

import { getSetting, setSetting } from '../../db/lireekDb.js';
import type { EngineBackend } from './types.js';
import { aceBackend } from './ace/index.js';
import { minimaxBackend } from './minimax/index.js';
import { yue2Backend } from './yue2/index.js';
import { sa3Backend } from './sa3/index.js';
import { heartmulaBackend } from './heartmula/index.js';

const SETTING_KEY = 'active_backend_id';
const DEFAULT_BACKEND_ID = 'ace';

// Registration order is the UI's toggle order. ACE stays first and stays the
// default — an existing install must behave exactly as before until the user
// deliberately switches.
const backends: Record<string, EngineBackend> = {
  ace: aceBackend,
  'minimax-m3': minimaxBackend,
  yue2: yue2Backend,
  // HOT-Step: Stable Audio 3 as a primary backend (backends/sa3/). Last in
  // the toggle, after the two it does not change.
  sa3: sa3Backend,
  heartmula: heartmulaBackend,
};

/** Look up a backend by id, or undefined if unregistered. */
export function getBackend(id: string): EngineBackend | undefined {
  // Own properties only: 'toString' and friends are not backends.
  return Object.hasOwn(backends, id) ? backends[id] : undefined;
}

/** All registered backends, in registration order. */
export function listBackends(): EngineBackend[] {
  return Object.values(backends);
}

/** The persisted active backend id. Falls back to the default if the stored
 *  value doesn't name a registered backend (e.g. a backend was removed). */
export function getActiveBackendId(): string {
  const id = getSetting(SETTING_KEY, DEFAULT_BACKEND_ID);
  return getBackend(id) ? id : DEFAULT_BACKEND_ID;
}

/** Set the active backend id. Returns false (no-op) if `id` isn't registered. */
export function setActiveBackendId(id: string): boolean {
  if (!getBackend(id)) return false;
  setSetting(SETTING_KEY, id);
  return true;
}

/** The active EngineBackend instance. */
export function getActiveBackend(): EngineBackend {
  return backends[getActiveBackendId()] ?? backends[DEFAULT_BACKEND_ID];
}
