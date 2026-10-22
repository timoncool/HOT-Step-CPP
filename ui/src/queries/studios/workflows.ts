import { queryOptions } from '@tanstack/react-query';
import { apiClient } from '../../services/httpClient';
import { followJob, workflowApi } from '../../services/workflowApi';
import type { WorkflowDocument, WorkflowReplay } from '../../../../server/src/contracts/workflow';
import { queryClient } from '../client';
import { resourceKey, sharedKeys, getSessionEpoch, SHARED } from '../keys';
import { queryManager } from '../invalidation';

export const workflowOptions = (id: string, after = 0) => {
  return queryOptions({
  queryKey: sharedKeys.workflowJob(id, after),
  queryFn: ({ signal }) => apiClient.get<WorkflowReplay>(`/workflows/jobs/${encodeURIComponent(id)}?after=${after}`, { signal }),
  staleTime: 0,
  });
};
export const documentOptions = (id: string) => {
  return queryOptions({
  queryKey: sharedKeys.workflowDocument(id),
  queryFn: ({ signal }) => apiClient.get<{ document: WorkflowDocument }>(`/workflows/documents/${encodeURIComponent(id)}`, { signal }),
  staleTime: 0,
  });
};

export const studioWorkflowApi = {
  ...workflowApi,
  get: (id: string, after = 0) => queryClient.query(workflowOptions(id, after)),
  getDocument: (id: string) => queryClient.query(documentOptions(id)),
  updateDocument: async (...args: Parameters<typeof workflowApi.updateDocument>) => {
    const key = documentOptions(args[0]).queryKey;
    await queryClient.cancelQueries({ queryKey: key });
    const result = await workflowApi.updateDocument(...args);
    queryClient.setQueryData(key, result);
    await queryManager.keys(queryClient, key);
    return result;
  },
};

/** Streaming reducers retain sequencing; events invalidate captured snapshots.
 *  Reconnects always reconcile, including when the replay history has a gap. */
export const followStudioJob: typeof followJob = (id, handlers, opts) => {
  const jobKey = workflowOptions(id).queryKey;
  const epoch = getSessionEpoch();
  const controller = new AbortController();
  const abort = () => controller.abort();
  opts?.signal?.addEventListener('abort', abort, { once: true });
  if (opts?.signal?.aborted) controller.abort();
  const current = () => {
    if (getSessionEpoch() === epoch) return true;
    controller.abort();
    return false;
  };
  const jobsKey = resourceKey(SHARED.workflowJob, jobKey[2], id);
  const documentsKey = resourceKey(SHARED.workflowDocument, jobKey[2]);
  let timer: ReturnType<typeof setTimeout> | undefined;
  const flush = () => {
    clearTimeout(timer);
    timer = undefined;
    if (getSessionEpoch() !== epoch) return;
    void queryManager.keys(queryClient, jobsKey);
    void queryManager.keys(queryClient, documentsKey);
    void queryManager.songs.views(queryClient);
  };
  const invalidate = () => { timer ??= setTimeout(flush, 50); };
  return followJob(id, {
    ...handlers,
    onSnapshot: (job, gap) => { if (current()) { invalidate(); handlers.onSnapshot?.(job, gap); } },
    onEvent: event => { if (current()) { invalidate(); handlers.onEvent(event); } },
  }, { ...opts, signal: controller.signal }).then(status => {
    if (!current()) throw new DOMException('Auth session changed', 'AbortError');
    return status;
  }).finally(() => {
    opts?.signal?.removeEventListener('abort', abort);
    flush();
  });
};
