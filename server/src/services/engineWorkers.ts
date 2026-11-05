// engineWorkers.ts — one ace-server per opted-in GPU (GH #13)
//
// The engine runs one job at a time on one device. Multi-GPU dispatch runs
// whole jobs side by side, each on its own ace-server pinned to one card by
// UUID. This module is the worker table and the scope that binds engine calls
// to one worker; aceEngineProcess.ts owns the processes behind it.
//
// Empty opt-in (the default) is one worker, "w0", that IS today's engine:
// configured port, the Settings/Auto GPU pick of #153, ace_engine.log. A
// non-empty opt-in pins w0 to the first listed UUID on the same port and adds
// w1.. on the next free ports, one per further UUID.
//
// Binding: config.aceServer.url, aceClient, engineCrashSeq() and the live log
// feed all resolve through the worker scope, so the backends' existing calls
// follow runWithWorker() without each taking a client argument. Outside any
// scope they all mean w0, which is every caller today.

import { config, engineWorkerScope } from '../config.js';
import { listGpusSync, type NvidiaGpu } from './gpuDevices.js';
import * as engineState from '../engineState.js';

export interface EngineWorker {
  /** "w0" is the primary; secondaries are w1, w2... in opt-in order. */
  readonly id: string;
  readonly index: number;
  readonly primary: boolean;
  /** Pinned card, or null for the primary under an empty opt-in (Settings/Auto pick). */
  readonly gpuUuid: string | null;
  readonly port: number;
  readonly url: string;
  /** Engine log file name in the session log folder. */
  readonly logFile: string;
  /** What config.aceServer.url resolves to inside this worker's scope; undefined for w0. */
  readonly scopedUrl?: string;
  // Live state, written by aceEngineProcess.ts.
  pid: number | null;
  ready: boolean;
  status: string;
  /** Bumped on every unexpected exit; a job polling across a change is lost. */
  crashSeq: number;
  /** ggml device from the worker's last "[Load] ... backend:" line. */
  device: string | null;
  /** Calls in flight under runWithWorker(); restartWorker() needs 0. */
  busy: number;
}

/** Serialisable snapshot for the status route and tools/multi-gpu-check.mjs. */
export type EngineWorkerStatus = Pick<EngineWorker,
  'id' | 'index' | 'primary' | 'gpuUuid' | 'port' | 'url' | 'logFile' | 'pid' | 'ready' | 'status' | 'crashSeq' | 'device' | 'busy'>;

/** Opt-in UUIDs in order, deduplicated; anything not GPU-/MIG-shaped is dropped. */
export function parseWorkerGpus(setting: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of (setting || '').split(',')) {
    const id = raw.trim();
    if (!/^(GPU-|MIG-)/i.test(id) || seen.has(id.toLowerCase())) continue;
    seen.add(id.toLowerCase());
    out.push(id);
  }
  return out;
}

function makeWorker(index: number, gpuUuid: string | null, port?: number): EngineWorker {
  const primary = index === 0;
  const worker: EngineWorker = {
    id: `w${index}`, index, primary, gpuUuid,
    // w0 tracks config live, exactly as the single engine always has.
    get port() { return primary ? config.aceServer.port : port!; },
    get url() { return `http://${config.aceServer.host}:${this.port}`; },
    get scopedUrl() { return primary ? undefined : this.url; },
    logFile: primary ? 'ace_engine.log' : `ace_engine.w${index}.log`,
    pid: null, ready: false, status: 'Not started', crashSeq: 0, device: null, busy: 0,
  };
  // w0's readiness and device ARE the global engine state that bootstrap,
  // training and /api/health already drive; a copy would drift from them.
  if (primary) {
    Object.defineProperties(worker, {
      ready: { enumerable: true, get: () => engineState.engineReady,
        set: (v: boolean) => engineState.setEngineReady(v, engineState.engineBootStatus) },
      status: { enumerable: true, get: () => engineState.engineBootStatus,
        set: (v: string) => engineState.setEngineReady(engineState.engineReady, v) },
      device: { enumerable: true, get: () => engineState.engineDevice,
        set: (v: string | null) => { if (v === null) engineState.resetEngineDevice(); } },
    });
  }
  return worker;
}

/**
 * Worker table for an opt-in setting. Only UUIDs nvidia-smi reports survive,
 * so a non-CUDA machine, a pulled card or a typo falls back to the legacy
 * single worker instead of an engine that cannot start.
 */
export function resolveWorkers(setting: string, gpus: NvidiaGpu[], basePort: number, reservedPorts: number[] = []): { workers: EngineWorker[]; log: string | null } {
  const wanted = parseWorkerGpus(setting);
  if (!wanted.length) return { workers: [makeWorker(0, null)], log: null };
  const known = wanted
    .map(id => gpus.find(g => g.uuid.toLowerCase() === id.toLowerCase()))
    .filter((g): g is NvidiaGpu => !!g);
  const missing = wanted.length - known.length;
  if (!known.length) {
    return { workers: [makeWorker(0, null)],
      log: `[Workers] none of the ${wanted.length} opted-in GPU(s) is present; running the single default engine` };
  }
  const taken = new Set([basePort, ...reservedPorts]);
  let next = basePort;
  const workers = known.map((g, i) => {
    if (i === 0) return makeWorker(0, g.uuid);
    while (taken.has(next)) next++;
    taken.add(next);
    return makeWorker(i, g.uuid, next);
  });
  const line = workers.map(w => `${w.id}=GPU ${known[w.index].index} ${known[w.index].name} :${w.port}`).join(', ');
  return { workers, log: `[Workers] ${workers.length} engine worker(s): ${line}`
    + (missing ? `; ${missing} opted-in GPU(s) not present, skipped` : '') };
}

let table: EngineWorker[] | null = null;

/** The worker table, resolved once on first use (the opt-in is restart-required). */
export function engineWorkers(): EngineWorker[] {
  if (!table) {
    const { workers, log } = resolveWorkers(config.aceServer.workerGpus,
      config.aceServer.workerGpus.trim() ? listGpusSync() : [],
      config.aceServer.port, [config.server.port]);
    if (log) console.log(log);
    table = workers;
  }
  return table;
}

/** Test seam: install a table (or null to re-resolve from config). */
export function setEngineWorkersForTest(workers: EngineWorker[] | null): void {
  table = workers;
}

export function primaryWorker(): EngineWorker {
  return engineWorkers()[0];
}

export function workerById(id: string | undefined): EngineWorker | undefined {
  return id ? engineWorkers().find(w => w.id === id) : undefined;
}

/** The worker the current async chain is bound to; the primary outside any scope. */
export function currentWorker(): EngineWorker {
  const w = engineWorkerScope.getStore() as EngineWorker | undefined;
  return w && 'id' in w ? w : primaryWorker();
}

/** Worker a job's engine calls went to; the primary for a job that never bound one. */
export function workerForJob(job: { workerId?: string }): EngineWorker {
  return workerById(job.workerId) ?? primaryWorker();
}

/**
 * Run `fn` with every engine call bound to `worker`: aceClient, backend
 * clients built on config.aceServer.url, crash epochs and the live log feed.
 * Nesting the same worker is a no-op; nesting a different one throws, so a
 * job can never drift to another card halfway through.
 */
export async function runWithWorker<T>(worker: EngineWorker, fn: () => T | Promise<T>): Promise<T> {
  const outer = engineWorkerScope.getStore() as EngineWorker | undefined;
  if (outer && 'id' in outer && outer !== worker) {
    throw new Error(`engine call bound to ${outer.id} cannot switch to ${worker.id}`);
  }
  worker.busy++;
  try {
    return await engineWorkerScope.run(worker, fn);
  } finally {
    worker.busy--;
  }
}

/** Internal scope without the busy count: health probes and restore hooks. */
export function inWorkerScope<T>(worker: EngineWorker, fn: () => T): T {
  return engineWorkerScope.run(worker, fn);
}

export function engineWorkerStatus(): EngineWorkerStatus[] {
  return engineWorkers().map(w => ({
    id: w.id, index: w.index, primary: w.primary, gpuUuid: w.gpuUuid, port: w.port, url: w.url,
    logFile: w.logFile, pid: w.pid, ready: w.ready, status: w.status, crashSeq: w.crashSeq,
    device: w.device, busy: w.busy,
  }));
}
