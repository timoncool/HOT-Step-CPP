// Wires the update checker to the real app: GitHub over fetch, state in the
// profile (data folder), identity from this install.
import fs from 'node:fs';
import path from 'node:path';
import { APP_VERSION, DATA_ROOT, PORTABLE_MODE, PROJECT_ROOT, config } from '../../config.js';
import { writeJsonAtomic } from '../../dataroot/paths.js';
import { dirtyCheckout } from '../training/workerUpdate.js';
import { createChecker, freshState } from './checker.js';
import type { UpdateState } from './checker.js';
import { detectInstall, gitRelation, realGit, realReadFile } from './install.js';

// Profile state, outside the replaceable app folder. A dev checkout with no
// data folder falls back to its data directory.
const STATE_FILE = DATA_ROOT ? path.join(DATA_ROOT, 'settings', 'updates.json') : path.join(config.data.dir, 'updates.json');
const git = realGit(PROJECT_ROOT);

export const updates = createChecker({
  fetch: (url, init) => fetch(url, init),
  now: Date.now,
  load: () => {
    try {
      const s = JSON.parse(fs.readFileSync(STATE_FILE, 'utf-8')) as UpdateState;
      return s?.version === 1 && s.release && s.master && (s.channel === 'release' || s.channel === 'master') ? s : freshState();
    } catch { return null; }
  },
  save: s => writeJsonAtomic(STATE_FILE, s),
  install: () => detectInstall({ root: PROJECT_ROOT, portable: PORTABLE_MODE, version: APP_VERSION, readFile: realReadFile, git, dirty: dirtyCheckout }),
  relation: (id, tip) => gitRelation(id.git, tip, git),
  platform: process.platform,
});

/** Startup check: only goes to GitHub when the cached result is stale. */
export function startUpdateCheck(): void {
  updates.check({ manual: false }).catch(e => console.warn('[Updates] startup check failed:', (e as Error).message));
}
