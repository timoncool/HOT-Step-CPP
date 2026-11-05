/**
 * Unconfirmed VRAM evictions from backend switches (#204).
 *
 * A backend switch answers at once and evicts the outgoing family in the
 * background; no render may load weights while any such eviction is still
 * unconfirmed. Every family's eviction is tracked on its own, so a later
 * switch can neither hide an earlier failure nor be skipped by a wait that
 * started before it.
 *
 * Evictions are per engine worker (multi-GPU, GH #13): one is tracked on the
 * worker the caller is bound to, and a render waits only for its own worker's,
 * since another card's memory is not in its way.
 */

import { currentWorker } from '../engineWorkers.js';
import { laneWorkers, runOnWorkerLane } from './gpuLane.js';

interface Eviction { family: string; release: () => Promise<void>; done: Promise<void> }

const pending = new Map<string, Eviction>();
const keyFor = (family: string) => `${currentWorker().id}/${family}`;

/** Start evicting `family` (the outgoing backend of a switch). A second switch
 *  away from the same family queues behind the first. */
export function trackBackendRelease(family: string, release: () => Promise<void>): Promise<void> {
  const key = keyFor(family);
  const prior = pending.get(key)?.done.catch(() => {}) ?? Promise.resolve();
  const done = prior.then(release);
  done.catch(() => {});  // observed by awaitBackendRelease; never an unhandled rejection
  pending.set(key, { family, release, done });
  return done;
}

/** A backend switch: evict the outgoing family on every worker, each in its
 *  own lease on that worker, so a job already running there finishes first.
 *  Tracking starts inside the lease, so a render queued ahead of it never
 *  waits on it (that render evicts other families itself). */
export function releaseOnEveryWorker(family: string, release: () => Promise<void>): void {
  for (const worker of laneWorkers()) {
    runOnWorkerLane(() => trackBackendRelease(family, release),
      { label: `release ${family}`, eligible: w => w === worker })
      .catch(err => console.warn(`[backends] releaseVram not confirmed for outgoing '${family}' on ${worker.id}:`, err?.message || err));
  }
}

/** Resolve once no eviction is unconfirmed. A failed eviction is tried once
 *  more (the engine may have finished it after the client gave up); if that
 *  fails too, this throws naming the family and leaves it tracked, so the
 *  next render tries again rather than loading next to it. Evictions started
 *  while this waits are waited for as well. */
export async function awaitBackendRelease(): Promise<void> {
  const mine = `${currentWorker().id}/`;
  for (;;) {
    const entry = [...pending.entries()].find(([key]) => key.startsWith(mine));
    if (!entry) return;
    const [key, ev] = entry;
    const family = ev.family;
    try {
      await ev.done;
    } catch {
      try {
        await ev.release();
      } catch (err: any) {
        throw new Error(`The previous backend (${family}) has not confirmed it freed its GPU memory, so this `
          + `render was not started: ${err?.message || err}`);
      }
    }
    // A newer eviction of the same family may have replaced this one meanwhile.
    if (pending.get(key) === ev) pending.delete(key);
  }
}

/** Test hook. */
export function _resetBackendReleases(): void { pending.clear(); }
