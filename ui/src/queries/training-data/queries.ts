// queries.ts — training-data server snapshots, keyed by the worker they came from.
//
// Every key carries `worker` (null = this PC), and every queryFn reads the
// target back from its own key, so a request scheduled for worker A lands in
// A's entry whatever "Train on" says by the time it answers. Local-only reads
// (the worker list, dispatch lines) carry no worker.
//
// trainingStore's loaders fetch through these options and project the result
// into the store only while (worker, dataset) still match the selection. That
// projection is the compatibility surface for training-runs until it moves to
// Query; nothing else fetches these snapshots.

import { queryOptions, type QueryClient, type QueryKey } from '@tanstack/react-query';
import { scopeOf, serverKey, sharedKeys, type Resource } from '../keys';
import * as api from '../../services/trainingApi';
import * as preparation from '../../services/trainingPreparationApi';
import { adapterApi } from '../../services/api';
import type { LmSize } from '../../services/trainingApi';

export const TD = {
  capabilities: 'training-data/capabilities',
  datasets: 'training-data/datasets',
  dataset: 'training-data/dataset',
  job: 'training-data/job',
  preprocessStatus: 'training-data/preprocess-status',
  trainLmStatus: 'training-data/train-lm-status',
  trainDitStatus: 'training-data/train-dit-status',
  auditions: 'training-data/auditions',
  lsGenerations: 'training-data/ls-generations',
  preparedData: 'training-data/prepared-data',
  pipelines: 'training-data/pipelines',
  yue2Batches: 'training-data/yue2-batches',
  workers: 'training-data/workers',
  workerDispatch: 'training-data/worker-dispatch',
  lmAdapters: 'training-data/lm-adapters',
  scanPreview: 'training-data/scan-preview',
  sampleMm3: 'training-data/sample-mm3',
  sampleYue2: 'training-data/sample-yue2',
  lyricStudioPreview: 'training-data/lyric-studio-preview',
} as const satisfies Record<string, Resource>;

type Worker = string | null;
/** The worker captured in a key; reads never consult the live selection. */
const workerOf = (key: QueryKey): Worker => scopeOf(key)?.worker ?? null;

export const trainingQueries = {
  capabilities: (worker: Worker) => queryOptions({
    queryKey: serverKey(TD.capabilities, { worker }),
    queryFn: ({ queryKey, signal }) => api.getCapabilities({ worker: workerOf(queryKey), signal }),
  }),
  datasets: (worker: Worker) => queryOptions({
    queryKey: serverKey(TD.datasets, { worker }),
    queryFn: ({ queryKey, signal }) => api.listDatasets({ worker: workerOf(queryKey), signal }),
  }),
  dataset: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey(TD.dataset, { worker, id }),
    queryFn: ({ queryKey, signal }) => api.getDataset(id, { worker: workerOf(queryKey), signal }),
  }),
  job: (worker: Worker, jobId: string) => queryOptions({
    queryKey: serverKey(TD.job, { worker, id: jobId }),
    queryFn: ({ queryKey, signal }) => api.getJob(jobId, { worker: workerOf(queryKey), signal }),
    // A job's status is live: every read asks the server, concurrent reads
    // still share one request.
    staleTime: 0,
  }),
  preprocessStatus: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey(TD.preprocessStatus, { worker, id }),
    queryFn: ({ queryKey, signal }) => api.getPreprocessStatus(id, { worker: workerOf(queryKey), signal }),
  }),
  trainLmStatus: (worker: Worker, id: string, q: { variantKey?: string; adapterName?: string; lmSize?: LmSize } = {}) => queryOptions({
    queryKey: serverKey(TD.trainLmStatus, { worker, id, filters: q }),
    queryFn: ({ queryKey, signal }) => api.getTrainLmStatus(id, q, { worker: workerOf(queryKey), signal }),
  }),
  trainDitStatus: (worker: Worker, id: string, q: { variantKey?: string; adapterName?: string } = {}) => queryOptions({
    queryKey: serverKey(TD.trainDitStatus, { worker, id, filters: q }),
    queryFn: ({ queryKey, signal }) => api.getTrainDitStatus(id, q, { worker: workerOf(queryKey), signal }),
  }),
  auditions: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey(TD.auditions, { worker, id }),
    queryFn: ({ queryKey, signal }) => api.getAuditions(id, 20, { worker: workerOf(queryKey), signal }),
  }),
  lsGenerations: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey(TD.lsGenerations, { worker, id }),
    queryFn: ({ queryKey, signal }) => api.getLsGenerations(id, { worker: workerOf(queryKey), signal }),
  }),
  preparedData: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey(TD.preparedData, { worker, id }),
    queryFn: ({ queryKey, signal }) => api.getPreparedData(id, { worker: workerOf(queryKey), signal }),
  }),
  pipelines: (worker: Worker) => queryOptions({
    queryKey: serverKey(TD.pipelines, { worker }),
    queryFn: ({ queryKey, signal }) => api.listPipelines({ worker: workerOf(queryKey), signal }),
  }),
  yue2Batches: (worker: Worker) => queryOptions({
    queryKey: serverKey(TD.yue2Batches, { worker }),
    queryFn: ({ queryKey, signal }) => api.listYue2Batches({ worker: workerOf(queryKey), signal }),
  }),
  /** Settings' worker list, as this PC sees it. Local. */
  workers: () => queryOptions({
    queryKey: serverKey(TD.workers),
    queryFn: ({ signal }) => api.listTrainingWorkers({ signal }),
  }),
  /** Every LM adapter under adapters/lm on this PC, for the pickers. Local. */
  lmAdapters: () => queryOptions({
    queryKey: serverKey(TD.lmAdapters),
    queryFn: ({ signal }) => adapterApi.lmList(undefined, { signal }),
  }),
  workerDispatch: (name: string) => queryOptions({
    queryKey: serverKey(TD.workerDispatch, { id: name }),
    queryFn: ({ signal }) => api.getWorkerDispatch(name, { signal }),
  }),

  // One-shot reads below are fetched with query(), never observed: they
  // seed a form or a wizard row, so no background refetch can write over the
  // user's edits. staleTime 0 asks the server on every open; concurrent reads
  // still share a request. A caller that moves on cancels its old key.

  /** A server YuE2 preparation chain and its current job. Local (shared keys). */
  preparation: (id: string) => queryOptions({
    queryKey: sharedKeys.preparation(id),
    queryFn: ({ signal }) => preparation.getYue2Preparation(id, { signal }),
    staleTime: 0,
  }),
  preparationJob: (id: string) => queryOptions({
    queryKey: sharedKeys.preparationJob(id),
    queryFn: ({ signal }) => preparation.getYue2PreparationJob<api.TrainingJobSummary>(id, { signal }),
    staleTime: 0,
  }),
  /** The folder picker's listing, on this PC (shared key). */
  browse: (dir: string, filter?: string) => queryOptions({
    queryKey: sharedKeys.browse(dir, filter),
    queryFn: ({ signal }) => adapterApi.browse(dir, filter, { signal }),
    staleTime: 0,
  }),
  /** What a folder holds before a dataset is created from it. */
  scanPreview: (worker: Worker, path: string, recursive = true) => queryOptions({
    queryKey: serverKey(TD.scanPreview, { worker, filters: { path, recursive } }),
    queryFn: ({ queryKey, signal }) => api.scanPreview(path, recursive, { worker: workerOf(queryKey), signal }),
    staleTime: 0,
  }),
  sampleMm3: (worker: Worker, id: string, sampleId: string) => queryOptions({
    queryKey: serverKey(TD.sampleMm3, { worker, id, filters: { sampleId } }),
    queryFn: ({ queryKey, signal }) => api.getSampleMm3(id, sampleId, { worker: workerOf(queryKey), signal }),
    staleTime: 0,
  }),
  sampleYue2: (worker: Worker, id: string, sampleId: string) => queryOptions({
    queryKey: serverKey(TD.sampleYue2, { worker, id, filters: { sampleId } }),
    queryFn: ({ queryKey, signal }) => api.getSampleYue2(id, sampleId, { worker: workerOf(queryKey), signal }),
    staleTime: 0,
  }),
  lyricStudioPreview: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey(TD.lyricStudioPreview, { worker, id }),
    queryFn: ({ queryKey, signal }) => api.getLyricStudioPreview(id, { worker: workerOf(queryKey), signal }),
    staleTime: 0,
  }),
};

/** Every training-data entry scoped to one worker (null = this PC), for
 *  invalidation after a job or cancellation after a "Train on" switch. */
export function workerReads(worker: Worker) {
  return {
    queryKey: ['hs'] as const,
    predicate: (q: { queryKey: QueryKey }) => String(q.queryKey[1]).startsWith('training-data/')
      && scopeOf(q.queryKey)?.worker !== undefined && workerOf(q.queryKey) === worker,
  };
}

/** Cancel every training-data read still in flight for a worker. Its answers
 *  would be dropped anyway once "Train on" has moved away from it. */
export function cancelWorkerReads(client: QueryClient, worker: Worker): Promise<void> {
  return client.cancelQueries(workerReads(worker));
}
