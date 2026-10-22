import { useQuery } from '@tanstack/react-query';
import { useRunEvents } from '../../queries/training-runs/hooks';
import { runQueries } from '../../queries/training-runs/queries';
import { useTrainingStore } from '../../stores/trainingStore';

export function useMm3Status(datasetId: string) {
  const worker = useTrainingStore(s => s.trainingWorker);
  const job = useTrainingStore(s => s.activeJob);
  useRunEvents(job?.id, worker);
  const query = useQuery({ ...runQueries.mm3Status(worker, datasetId), enabled: !!datasetId });
  return { status: query.data ?? null, error: query.error?.message ?? null,
    reload: () => { void query.refetch(); } };
}
