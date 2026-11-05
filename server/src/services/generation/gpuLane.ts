// The GPU lane: who may drive the engine(s), including while reset work drains.
//
// Two kinds of reservation share one FIFO queue:
// - Exclusive (runOnGpuLane): every engine worker at once. The default for
//   every caller; previews, post-processing, training and maintenance stay
//   here. Its engine calls use the primary worker (no worker scope).
// - Worker (runOnWorkerLane): one engine worker, bound with runWithWorker for
//   the whole lease, so the bare client inside follows that worker. Leases on
//   different workers overlap only when worker concurrency is on (multi-GPU,
//   GH #13); off, the primary is the only worker and the lane behaves as the
//   single lane always has.
//
// An exclusive task at the head of the queue blocks every task behind it, and
// starts only once no worker lease is held and no early-released lease (a
// YuE2 job rendering on the engine's own lane) is still running.
// A lane call made inside a held reservation reuses it instead of queueing
// behind itself; inside an early-released lease it takes that lease's worker
// back ahead of the queue (the drain everyone else is waiting for).

import { AsyncLocalStorage } from 'node:async_hooks';
import { engineWorkerScope } from '../../config.js';
import { engineWorkers, primaryWorker, runWithWorker, type EngineWorker } from '../engineWorkers.js';

export interface LaneLease {
  readonly id: number;
  readonly generation: number;
  readonly label: string;
  readonly family?: string;
  /** The worker a worker lease is bound to; undefined for an exclusive one. */
  readonly worker?: EngineWorker;
  readonly draining: boolean;
  isCurrent(): boolean;
}

export class LaneResetError extends Error {
  constructor() {
    super('Queue reset by user');
    this.name = 'LaneResetError';
  }
}

export interface WorkerLaneOptions {
  label?: string;
  family?: string;
  /** Workers this task can run on (each model must fit one card). Ignored
   *  while worker concurrency is off: the primary is then the only worker. */
  eligible?: (worker: EngineWorker) => boolean;
}

interface Pending {
  label: string;
  family?: string;
  exclusive: boolean;
  eligible?: (worker: EngineWorker) => boolean;
  /** Re-acquire after an early release: this worker only, ahead of the queue. */
  worker?: EngineWorker;
  start(lease: LaneLease): void;
  reject(error: Error): void;
}

const pending: Pending[] = [];
const reacquire: Pending[] = [];
let generation = 0;
let nextLeaseId = 1;
let exclusive: LaneLease | null = null;
/** Worker id -> the worker lease holding it. */
const slots = new Map<string, LaneLease>();
/** Released early, still running on the engine: an exclusive task waits for these. */
const released = new Set<LaneLease>();
const leaseScope = new AsyncLocalStorage<LaneLease>();
let workerConcurrency = false;

/** Multi-GPU switch. Off (the default) keeps every worker lease on the primary. */
export function setWorkerConcurrency(on: boolean): void { workerConcurrency = on; pump(); }
export function workerConcurrencyEnabled(): boolean { return workerConcurrency; }

const held = (lease: LaneLease) => exclusive === lease || (!!lease.worker && slots.get(lease.worker.id) === lease);

export function gpuLaneBusy(): boolean { return exclusive !== null || slots.size > 0 || released.size > 0; }
export function gpuLaneDepth(): number { return pending.length + reacquire.length; }
/** The exclusive owner, or the oldest worker lease. */
export function gpuLaneOwner(): LaneLease | null { return exclusive ?? slots.values().next().value ?? null; }
/** Every worker lease held now, by worker id. */
export function gpuLaneWorkerLeases(): ReadonlyMap<string, LaneLease> { return slots; }
/** The family of the task next in line, if any. */
export function gpuLaneNextFamily(): string | undefined { return pending[0]?.family; }

/** Hand the lane (or this lease's worker) to the next task while this one
 *  carries on without it (a YuE2 job whose render the engine runs on its own
 *  lane). The lease stops being current; an exclusive waiter still waits for
 *  it to settle, and a lane call it makes later takes its worker back. */
export function releaseGpuLane(lease: LaneLease): boolean {
  if (!held(lease)) return false;
  if (lease === exclusive) exclusive = null;
  else slots.delete(lease.worker!.id);
  released.add(lease);
  pump();
  return true;
}

/** Invalidates callbacks and rejects queued tasks. Current tasks retain
 * ownership until their promises settle, including any external
 * post-processing, so a worker is never reused before it drains. There is no
 * timer release or release based only on an engine restart. */
export function resetGpuLane(): number {
  generation++;
  const drained = [...reacquire.splice(0), ...pending.splice(0)];
  for (const task of drained) task.reject(new LaneResetError());
  return drained.length;
}

/** The workers lane leases can land on: every worker with concurrency on, else the primary. */
export function laneWorkers(): EngineWorker[] {
  return workerConcurrency ? engineWorkers() : [primaryWorker()];
}

/** A free worker for `task`, null to wait, or an Error when no worker can ever
 *  take it (so an unavailable or undersized card fails the job once instead
 *  of redispatching it forever). */
function placeWorker(task: Pending): EngineWorker | null | Error {
  if (!workerConcurrency) return slots.size ? null : primaryWorker();
  const eligible = laneWorkers().filter(w => !task.eligible || task.eligible(w));
  if (!eligible.length) return new Error(`No engine worker can run ${task.label}`);
  const ready = eligible.filter(w => w.ready);
  // A leased worker that is no longer ready settles and pumps again; only give
  // up when nothing eligible is either ready or still working.
  if (!ready.length) return eligible.some(w => slots.has(w.id)) ? null : new Error(`No engine worker is ready to run ${task.label}`);
  return ready.find(w => !slots.has(w.id)) ?? null;
}

function grant(task: Pending, worker?: EngineWorker): void {
  const lease: LaneLease = {
    id: nextLeaseId++, generation, label: task.label, family: task.family, worker,
    get draining() { return lease.generation !== generation; },
    isCurrent: () => held(lease) && lease.generation === generation,
  };
  Object.freeze(lease);
  if (worker) slots.set(worker.id, lease);
  else exclusive = lease;
  task.start(lease);
}

function pump(): void {
  if (exclusive) return;
  for (let i = 0; i < reacquire.length; i++) {
    const task = reacquire[i];
    if (slots.has(task.worker!.id)) continue;
    reacquire.splice(i--, 1);
    grant(task, task.worker);
  }
  for (let i = 0; i < pending.length; i++) {
    const task = pending[i];
    if (task.exclusive) {
      // Head-of-line: nothing behind an exclusive waiter starts before it.
      if (slots.size || released.size || reacquire.length) return;
      pending.splice(i, 1);
      grant(task);
      return;
    }
    const worker = placeWorker(task);
    if (worker === null) continue;  // ponytail: later worker tasks may pass a blocked one; age-ordered placement if a card starves
    pending.splice(i--, 1);
    if (worker instanceof Error) task.reject(worker);
    else grant(task, worker);
  }
}

function finish(lease: LaneLease, settle: () => void): void {
  // An old completion can never release another owner's lease.
  if (exclusive === lease) exclusive = null;
  else if (lease.worker && slots.get(lease.worker.id) === lease) slots.delete(lease.worker.id);
  released.delete(lease);
  settle();
  pump();
}

function enqueue<T>(fn: (lease: LaneLease) => Promise<T>, task: Omit<Pending, 'start' | 'reject'>): Promise<T> {
  const outer = leaseScope.getStore();
  if (outer && held(outer)) return fn(outer);
  if (outer && released.has(outer) && outer.worker) task = { ...task, worker: outer.worker };
  return new Promise<T>((resolve, reject) => {
    const entry: Pending = {
      ...task, reject,
      start(lease) {
        Promise.resolve().then(() => {
          // Reset may have invalidated the lease before its first callback.
          if (!lease.isCurrent()) throw new LaneResetError();
          // pump() runs in whichever lease just finished; start from no worker
          // scope, so an exclusive task means the primary and a worker lease binds cleanly.
          const run = () => leaseScope.run(lease, () => fn(lease));
          return engineWorkerScope.exit(() => lease.worker ? runWithWorker(lease.worker, run) : run());
        }).then(
          value => finish(lease, () => resolve(value)),
          error => finish(lease, () => reject(error)),
        );
      },
    };
    (entry.worker ? reacquire : pending).push(entry);
    pump();
  });
}

/** Exclusive: every engine worker, the primary's client. */
export function runOnGpuLane<T>(
  fn: (lease: LaneLease) => Promise<T>,
  opts: { label?: string; family?: string } = {},
): Promise<T> {
  return enqueue(fn, { label: opts.label ?? 'gpu task', family: opts.family, exclusive: true });
}

/** One engine worker for the whole lease: a ready, eligible, free one. */
export function runOnWorkerLane<T>(
  fn: (lease: LaneLease) => Promise<T>,
  opts: WorkerLaneOptions = {},
): Promise<T> {
  return enqueue(fn, { label: opts.label ?? 'gpu task', family: opts.family, exclusive: false, eligible: opts.eligible });
}

/** Exclusive like runOnGpuLane, held outside a callback until the returned
 *  release runs: training owns every card from its engine stop to its restart.
 *  Rejects with LaneResetError if a reset drops it while it waits. */
export function holdGpuLane(label: string): Promise<() => void> {
  return new Promise((held, failed) => {
    runOnGpuLane(() => new Promise<void>(release => held(release)), { label }).catch(failed);
  });
}
