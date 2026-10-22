import { queryOptions } from '@tanstack/react-query';
import { apiClient, type ReadOptions } from '../../services/httpClient';
import { followJob, workflowApi } from '../../services/workflowApi';
import type { WorkflowDocument, WorkflowReplay } from '../../../../server/src/contracts/workflow';
import { queryClient } from '../client';
import { resourceKey, sharedKeys, getSessionEpoch, SHARED } from '../keys';
import { queryManager } from '../invalidation';

export const workflowOptions = (id: string, after = 0, options: ReadOptions = {}) => {
  return queryOptions({
  queryKey: sharedKeys.workflowJob(id, after),
  queryFn: ({ signal }) => apiClient.get<WorkflowReplay>(`/workflows/jobs/${encodeURIComponent(id)}?after=${after}`, { signal: options.signal ? AbortSignal.any([signal, options.signal]) : signal }),
  staleTime: 0,
  });
};
export const documentOptions = (id: string, options: ReadOptions = {}) => {
  return queryOptions({
  queryKey: sharedKeys.workflowDocument(id),
  queryFn: ({ signal }) => apiClient.get<{ document: WorkflowDocument }>(`/workflows/documents/${encodeURIComponent(id)}`, { signal: options.signal ? AbortSignal.any([signal, options.signal]) : signal }),
  staleTime: 0,
  });
};

export const coverLyricWorkflowApi = {
  ...workflowApi,
  get: (id: string, after = 0, options: ReadOptions = {}) => queryClient.query(workflowOptions(id, after, options)),
  getDocument: (id: string, options: ReadOptions = {}) => queryClient.query(documentOptions(id, options)),
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
export const followCoverLyricJob: typeof followJob = (id, handlers, opts) => {
  const jobScope = sharedKeys.workflowJob(id)[2];
  const jobKey = resourceKey(SHARED.workflowJob, jobScope, id);
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
  const documentsKey = resourceKey(SHARED.workflowDocument, jobScope);
  let timer: ReturnType<typeof setTimeout> | undefined;
  const flush = () => {
    clearTimeout(timer);
    timer = undefined;
    if (getSessionEpoch() !== epoch) return;
    void queryManager.keys(queryClient, jobKey);
    void queryManager.keys(queryClient, documentsKey);
    void queryManager.songs.views(queryClient);
    void queryManager.keys(queryClient, resourceKey('cover-lyrics/lireek'));
    void queryManager.keys(queryClient, resourceKey('cover-lyrics/drafts', jobScope));
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
