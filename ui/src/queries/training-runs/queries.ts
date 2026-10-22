import { queryOptions, type QueryClient } from '@tanstack/react-query';
import * as api from '../../services/trainingApi';
import * as preparation from '../../services/trainingPreparationApi';
import { getReviewLadder } from '../../services/trainingReviewApi';
import { trainingOperation } from '../../services/trainingOperations';
import type { TrainingRecipe } from '../../services/trainingRecipesApi';
import type { TrainingRecipeFamily } from '../../../../server/src/contracts/trainingRecipes';
import { settingsApi } from '../../services/api';
import { scopeOf, serverKey, sharedKeys, resourceKey, SHARED } from '../keys';
import { pollWhile } from '../contract';
import { queryManager } from '../invalidation';

export type Worker = string | null;
export const activeJob = (job: { status: string } | null | undefined) =>
  !job || job.status === 'queued' || job.status === 'running';

// Local indexes deliberately omit worker scope. A remote job's mirrored run
// and review records belong to this PC, regardless of the Train on selection.
export const runQueries = {
  recipe: <T>(worker: Worker, family: TrainingRecipeFamily) => queryOptions({
    queryKey: serverKey('training-runs/recipe', { worker, id: family }),
    queryFn: ({ queryKey, signal }) => {
      const target = scopeOf(queryKey)!.worker;
      return trainingOperation<TrainingRecipe<T>>('recipes', `/${family}${target ? `?worker=${encodeURIComponent(target)}` : ''}`, { signal });
    },
  }),
  gpus: () => queryOptions({
    queryKey: sharedKeys.gpus(),
    queryFn: ({ signal }) => settingsApi.getGpus({ signal }),
  }),
  mm3Status: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey('training-runs/mm3-status', { worker, id }),
    queryFn: ({ queryKey, signal }) => api.getMm3Status(id, { worker: scopeOf(queryKey)!.worker, signal }),
  }),
  yue2Status: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey('training-runs/yue2-status', { worker, id }),
    queryFn: ({ queryKey, signal }) => api.getYue2Status(id, { worker: scopeOf(queryKey)!.worker, signal }),
  }),
  arStatus: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey('training-runs/ar-status', { worker, id }),
    queryFn: ({ queryKey, signal }) => api.getYue2ArStatus(id, { worker: scopeOf(queryKey)!.worker, signal }),
  }),
  mm3Runs: (id: string) => queryOptions({
    queryKey: serverKey('training-runs/mm3-runs', { id }),
    queryFn: ({ signal }) => api.listMm3Runs(id, { signal }),
  }),
  narRuns: (id: string) => queryOptions({
    queryKey: serverKey('training-runs/nar-runs', { id }),
    queryFn: ({ signal }) => api.listYue2Runs(id, { signal }),
  }),
  arRuns: (id: string) => queryOptions({
    queryKey: serverKey('training-runs/ar-runs', { id }),
    queryFn: ({ signal }) => api.listYue2ArRuns(id, { signal }),
  }),
  jointRuns: (id: string) => queryOptions({
    queryKey: serverKey('training-runs/joint-runs', { id }),
    queryFn: ({ signal }) => api.listYue2AitkRuns(id, { signal }),
    refetchInterval: pollWhile<Awaited<ReturnType<typeof api.listYue2AitkRuns>>>(data =>
      !data || !!data.activeJob && activeJob(data.activeJob) || data.runs.some(run => run.live), 5000),
  }),
  previews: (id: string, run: string) => queryOptions({
    queryKey: serverKey('training-runs/previews', { id, filters: { run } }),
    queryFn: ({ signal }) => api.listYue2JointPreviewsLocal(id, run, { signal }),
  }),
  review: () => queryOptions({
    queryKey: serverKey('training-runs/review'),
    queryFn: ({ signal }) => api.listYue2Review({ signal }),
    refetchInterval: pollWhile(() => true, 30_000),
  }),
  ladder: (id: string, run: string) => queryOptions({
    queryKey: serverKey('training-runs/ladder', { id, filters: { run } }),
    queryFn: ({ signal }) => getReviewLadder(id, run, { signal }),
  }),
  scores: (id: string, run: string) => queryOptions({
    queryKey: serverKey('training-runs/scores', { id, filters: { run } }),
    queryFn: ({ signal }) => api.listYue2RungScores(id, run, { signal }),
  }),
  album: (id: string, run: string) => queryOptions({
    queryKey: serverKey('training-runs/album', { id, filters: { run } }),
    queryFn: ({ signal }) => api.getYue2AlbumScore(id, run, { signal }),
  }),
  localBatches: () => queryOptions({
    queryKey: serverKey('training-runs/local-batches'),
    queryFn: ({ signal }) => api.listYue2BatchesLocal({ signal }),
    refetchInterval: pollWhile(() => true, 10_000),
  }),
  mirror: (name: string) => queryOptions({
    queryKey: serverKey('training-runs/mirror', { id: name }),
    queryFn: ({ signal }) => api.getWorkerMirror(name, { signal }),
    refetchInterval: pollWhile(() => true, 15_000),
  }),
  workerUpdate: (name: string) => queryOptions({
    queryKey: serverKey('training-runs/worker-update', { id: name }),
    queryFn: ({ signal }) => api.getWorkerUpdate(name, { signal }),
    refetchInterval: pollWhile<Awaited<ReturnType<typeof api.getWorkerUpdate>>>(data =>
      !!data && !['done', 'failed', 'cancelled'].includes(data.status), 5000),
  }),
  optimise: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey('training-runs/optimise', { worker, id }),
    queryFn: ({ queryKey, signal }) => api.getYue2Optimise(id, { worker: scopeOf(queryKey)!.worker, signal }),
  }),
  prepare: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey('training-runs/prepare', { worker, id }),
    queryFn: ({ queryKey, signal }) => api.getYue2AitkPrepare(id, undefined, { worker: scopeOf(queryKey)!.worker, signal }),
  }),
  jobs: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey('training-runs/jobs', { worker, id }),
    queryFn: ({ queryKey, signal }) => api.listJobs(id, { worker: scopeOf(queryKey)!.worker, signal }),
  }),
  sources: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey('training-runs/sources', { worker, id }),
    queryFn: ({ queryKey, signal }) => api.listYue2SheetSources(id, { worker: scopeOf(queryKey)!.worker, signal }),
  }),
  source: (worker: Worker, id: string, name: string) => queryOptions({
    queryKey: serverKey('training-runs/source', { worker, id, filters: { name } }),
    queryFn: ({ queryKey, signal }) => api.getYue2SheetSource(id, name, { worker: scopeOf(queryKey)!.worker, signal }),
  }),
  preparations: (id: string) => queryOptions({
    queryKey: serverKey('training-runs/preparations', { id }),
    queryFn: ({ signal }) => preparation.listYue2Preparations(id, { signal }),
    refetchInterval: pollWhile<Awaited<ReturnType<typeof preparation.listYue2Preparations>>>(data =>
      !data || data.some(p => !['done', 'failed', 'cancelled', 'interrupted'].includes(p.status))),
  }),
  preparation: (id: string) => queryOptions({
    queryKey: sharedKeys.preparation(id),
    queryFn: ({ signal }) => preparation.getYue2Preparation(id, { signal }),
    refetchInterval: pollWhile<Awaited<ReturnType<typeof preparation.getYue2Preparation>>>(data =>
      !data || !['done', 'failed', 'cancelled', 'interrupted'].includes(data.status)),
  }),
  preparationJob: (id: string) => queryOptions({
    queryKey: sharedKeys.preparationJob(id),
    queryFn: ({ signal }) => preparation.getYue2PreparationJob<api.TrainingJobSummary>(id, { signal }),
  }),
};

/** Commands execute once; only successful commands refresh their snapshots.
 * Capture the worker before starting a command, including across awaits. */
export async function runCommand<T>(client: QueryClient, worker: Worker, command: () => Promise<T>): Promise<T> {
  const result = await command();
  await Promise.all([
    queryManager.keys(client, ...[SHARED.preparation, SHARED.preparationJob].map(r => resourceKey(r))),
    queryManager.filter(client, { queryKey: ['hs'], predicate: q =>
      String(q.queryKey[1]).startsWith('training-runs/') &&
      (scopeOf(q.queryKey)?.worker === undefined || scopeOf(q.queryKey)?.worker === worker) }),
    queryManager.filter(client, { queryKey: ['hs'], predicate: q =>
      String(q.queryKey[1]).startsWith('training-data/') && scopeOf(q.queryKey)?.worker === worker }),
  ]);
  return result;
}

