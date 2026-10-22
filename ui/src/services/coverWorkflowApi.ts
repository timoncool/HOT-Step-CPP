import { changed, songsChanged } from '../queries/cover-lyrics/reads';
import { waitForSnapshot } from '../queries/cover-lyrics/wait';
import { workflowOptions } from '../queries/cover-lyrics/workflows';
import { apiClient } from './httpClient';
import type { WorkflowReplay } from '../../../server/src/contracts/workflow';
import { coverLyricWorkflowApi as workflowApi } from '../queries/cover-lyrics/workflows';
import type { WorkflowJob } from '../../../server/src/contracts/workflow';

export interface CoverDraftResult {
  documentId: string;
  revision: number;
  metadata?: { artist: string; title: string; album: string; duration: number | null };
  analysis?: { bpm: number; key: string; scale?: string } | null;
  abc?: string;
  scoreSource?: string;
  caption?: string;
}

export async function waitForCoverJob(jobId: string, cancelled: () => boolean,
  progress?: (job: WorkflowJob) => void): Promise<WorkflowJob> {
  const result = await waitForSnapshot(workflowOptions(jobId).queryKey,
    signal => apiClient.get<WorkflowReplay>(`/workflows/jobs/${encodeURIComponent(jobId)}?after=0`, { signal }),
    data => !['succeeded', 'failed', 'cancelled', 'interrupted'].includes(data.job.status),
    data => {
      if (cancelled()) {
        void workflowApi.cancel(jobId).catch(() => {});
        throw new Error('Cover operation cancelled');
      }
      progress?.(data.job);
    });
  if (result.job.status !== 'succeeded') throw new Error(result.job.error || `Cover operation ${result.job.status}`);
  if (result.job.kind === 'cover-render') await songsChanged();
  return result.job;
}

export const coverWorkflowApi = {
  open: (assetId: string, cached?: {
    metadata: { artist: string; title: string; album: string; duration: number | null };
    analysis: { bpm: number; key: string; scale?: string };
  }) => workflowApi.submit('cover-open', crypto.randomUUID(), { assetId, cached }),
  transcribe: (documentId: string, revision: number, force: boolean) =>
    workflowApi.submit('cover-transcribe', crypto.randomUUID(), { documentId, revision, force }),
  caption: (documentId: string, revision: number, artistId: number,
    provider: string, model: string, force: boolean) =>
    workflowApi.submit('cover-caption', crypto.randomUUID(),
      { documentId, revision, artistId, provider, model, force }),
  render: (input: Record<string, unknown>) =>
    workflowApi.submit('cover-render', crypto.randomUUID(), input),
  approveScore: async (documentId: string, revision: number, abc: string) => {
    const data = await apiClient.post<{ document: { id: string; revision: number }; sourceId: string; sourceLabel: string; abc: string }>(
      `/yue2-cover/drafts/${encodeURIComponent(documentId)}/approve-score`, { revision, abc });
    await Promise.all([changed('drafts'), changed('cover')]);
    return data;
  },
  saveScoreDetails: async (documentId: string, revision: number,
    abc: string, factor: number) => {
    const data = await apiClient.post<{ saved: true; bpm: number; key: string }>(
      `/yue2-cover/drafts/${encodeURIComponent(documentId)}/save-score-details`, { revision, abc, factor });
    await changed('cover');
    return data;
  },
};
