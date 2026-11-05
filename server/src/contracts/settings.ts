// contracts/settings.ts — wire shapes for /api/settings (server/src/routes/settings.ts).

import type { EngineWorkerStatus } from '../services/engineWorkers.js';
import type { NvidiaGpu } from '../services/gpuDevices.js';

export interface EnvResponse {
  /** Exposed .env keys only; unset keys are backfilled with their resolved default. */
  values: Record<string, string>;
  /** Keys that need a restart to take effect — the UI badges them. */
  restartKeys: string[];
}

export interface EnvUpdateResponse {
  updated: string[];
  restartRequired: boolean;
}

/** `services/gpuDevices.ts` already names this shape; re-exported under the
 *  name GET /settings/gpus docs it by, rather than a second definition. */
export type GpuInfo = NvidiaGpu;

export interface GpusResponse { gpus: GpuInfo[] }

/** One row per engine worker (multi-GPU is experimental). `workers[0]` is the
 *  primary; a single row means the opt-in is empty or matched no detected card. */
export type EngineWorkerInfo = EngineWorkerStatus;

export interface EngineWorkersResponse {
  workers: EngineWorkerInfo[];
  /** Restart now would be refused: a call is bound to some worker. */
  busy: boolean;
}
