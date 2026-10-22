// trainingOperations.ts — client for Node-owned training operations
// (/api/training/ops/<domain>, server/src/services/training/operations.ts).
//
// Operations always go to this machine (never through API_BASE, which follows
// "Train on"). The worker is part of the command: snapshotFor() reads the
// current "Train on" choice once, when the command is built, and the server
// keeps that captured worker whatever this or another client selects later.
import type {
  TrainingDatasetRef, TrainingSnapshot, TrainingSourceRef, TrainingWorkerRef,
} from '../../../server/src/contracts/trainingOperation';
import { getTrainingWorker } from './trainingApi';
import { apiClient, type ClientOptions } from './httpClient';

export type { TrainingSnapshot, TrainingWorkerRef } from '../../../server/src/contracts/trainingOperation';

/** The worker "Train on" points at now, as an explicit reference. */
export const currentWorkerRef = (): TrainingWorkerRef => {
  const name = getTrainingWorker();
  return name ? { kind: 'remote', name } : { kind: 'local' };
};

/** Build an operation envelope. `worker` defaults to "Train on" now. */
export function snapshotFor<P>(op: {
  kind: string; idempotencyKey: string; payload: P;
  worker?: TrainingWorkerRef; dataset?: TrainingDatasetRef; sources?: TrainingSourceRef[];
}): TrainingSnapshot<P> {
  return {
    version: 1,
    operation: { kind: op.kind, idempotencyKey: op.idempotencyKey },
    worker: op.worker ?? currentWorkerRef(),
    ...(op.dataset ? { dataset: op.dataset } : {}),
    sources: op.sources ?? [],
    payload: op.payload,
  };
}

/** Call a domain's operation route on this machine. Throws ApiError with the
 *  server's status, reason and currentRevision (training ops use string
 *  revisions — they ride in `err.currentRevision`); the full body, including
 *  `issues`, is on `err.body`. */
export async function trainingOperation<T>(domain: string, path: string,
  init: { method?: string; body?: unknown } & ClientOptions = {}): Promise<T> {
  const { body, ...opts } = init;
  const method = init.method ?? (body === undefined ? 'GET' : 'POST');
  return apiClient.request<T>(method, `/training/ops/${encodeURIComponent(domain)}${path}`, body, opts);
}
