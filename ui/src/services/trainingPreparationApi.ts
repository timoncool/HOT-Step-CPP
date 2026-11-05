import type { TrainingSnapshot } from '../../../server/src/contracts/trainingOperation';
import type { TrainingDatasetRef, TrainingSourceRef } from '../../../server/src/contracts/trainingOperation';
import type { Yue2PreparationPayload, Yue2PreparationSummary } from '../../../server/src/contracts/trainingPreparation';
import { trainingOperation } from './trainingOperations';
import type { ReadOptions } from './httpClient';

export type { Yue2PreparationPayload, Yue2PreparationSummary } from '../../../server/src/contracts/trainingPreparation';

const domain = 'preparation';
type Snapshot = TrainingSnapshot<Yue2PreparationPayload>;

export async function getYue2PreparationContext(datasetId: string, opts: ReadOptions = {}): Promise<{ dataset: TrainingDatasetRef; source: TrainingSourceRef }> {
  return trainingOperation(domain, `/context/${encodeURIComponent(datasetId)}`, opts);
}

export async function startYue2Preparation(snapshot: Snapshot): Promise<Yue2PreparationSummary> {
  // silent: Yue2AitkTrainCard's train error line is the sole notifier.
  return (await trainingOperation<{ pipeline: Yue2PreparationSummary }>(domain, '/', { body: snapshot, silent: true })).pipeline;
}

export async function listYue2Preparations(datasetId?: string, opts: ReadOptions = {}): Promise<Yue2PreparationSummary[]> {
  const suffix = datasetId ? `?datasetId=${encodeURIComponent(datasetId)}` : '';
  return (await trainingOperation<{ pipelines: Yue2PreparationSummary[] }>(domain, `/${suffix}`, opts)).pipelines;
}

export async function getYue2Preparation(id: string, opts: ReadOptions = {}): Promise<Yue2PreparationSummary> {
  return (await trainingOperation<{ pipeline: Yue2PreparationSummary }>(domain, `/${encodeURIComponent(id)}`, opts)).pipeline;
}

export async function getYue2PreparationJob<T>(id: string, opts: ReadOptions = {}): Promise<T | null> {
  return (await trainingOperation<{ job: T | null }>(domain, `/${encodeURIComponent(id)}/job`, opts)).job;
}

export async function commandYue2Preparation(id: string, action: 'pause' | 'resume' | 'cancel' | 'retry'): Promise<Yue2PreparationSummary> {
  // silent: Yue2TrainStages' capture-error banner and Yue2AitkTrainCard's
  // stop error line are the sole notifiers.
  return (await trainingOperation<{ pipeline: Yue2PreparationSummary }>(domain,
    `/${encodeURIComponent(id)}/${action}`, { method: 'POST', silent: true })).pipeline;
}
