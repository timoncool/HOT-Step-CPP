// workflowApi.ts — client for /api/workflows: durable workflow jobs and
// revisioned documents (server/src/routes/workflows.ts).
//
// JSON goes through the shared client (httpClient.ts → axiosClient.ts), so a
// stale-revision 409 comes back as ApiError with currentRevision/reason — the
// old WorkflowRequestError class was folded into it. followJob reads the job's
// event stream with fetch rather than EventSource, so it can send the bearer
// token (now read from the auth session accessor). It resumes from the last
// event it saw after a drop, and hands back a fresh snapshot (gap: true) when
// the server no longer holds the events in between.

import { getAuthToken } from './axiosClient';
import { ApiError } from './apiError';
import { apiClient, type ClientOptions } from './httpClient';
import type {
  WorkflowDocument, WorkflowEvent, WorkflowJob, WorkflowJobStatus, WorkflowReplay,
} from '../../../server/src/contracts/workflow';

export type {
  DocumentImportReceipt, DocumentProvenance, DocumentScope, TypedDocument, UnsupportedDocumentVersion,
} from '../../../server/src/contracts/workflow';

/** A JSON request that throws ApiError carrying the stale revision or
 *  unsupported-version reason. Shared by the typed-document domain clients,
 *  whose routes live outside /api/workflows. `url` may carry the `/api` base;
 *  it is stripped because the shared client adds its own. */
export async function revisionedRequest<T>(url: string, method = 'GET', body?: unknown, opts: ClientOptions = {}): Promise<T> {
  return apiClient.request<T>(method, url.replace(/^\/api/, ''), body, opts);
}

const call = <T>(path: string, method = 'GET', body?: unknown, opts?: ClientOptions): Promise<T> =>
  revisionedRequest<T>(`/api/workflows${path}`, method, body, opts);

const id = (v: string) => encodeURIComponent(v);
const FINAL = new Set<WorkflowJobStatus>(['succeeded', 'failed', 'cancelled', 'interrupted']);

export const workflowApi = {
  submit: (kind: string, idempotencyKey: string, input: Record<string, unknown>) =>
    call<{ job: WorkflowJob; created: boolean }>('/jobs', 'POST', { kind, idempotencyKey, input }),
  list: (filter: { kind?: string; status?: WorkflowJobStatus } = {}) =>
    call<{ jobs: WorkflowJob[] }>(`/jobs?${new URLSearchParams(filter as Record<string, string>)}`),
  get: (jobId: string, after = 0, opts?: ClientOptions) =>
    call<WorkflowReplay>(`/jobs/${id(jobId)}?after=${after}`, 'GET', undefined, opts),
  cancel: (jobId: string) => call<{ job: WorkflowJob }>(`/jobs/${id(jobId)}/cancel`, 'POST', {}),
  retry: (jobId: string) => call<{ job: WorkflowJob }>(`/jobs/${id(jobId)}/retry`, 'POST', {}),

  createDocument: (kind: string, data: Record<string, unknown>) =>
    call<{ document: WorkflowDocument }>('/documents', 'POST', { kind, data }),
  listDocuments: (kind: string) => call<{ documents: WorkflowDocument[] }>(`/documents?kind=${id(kind)}`),
  getDocument: (docId: string) => call<{ document: WorkflowDocument }>(`/documents/${id(docId)}`),
  /** Throws ApiError with status 409 and currentRevision when stale. */
  updateDocument: (docId: string, expectedRevision: number, data: Record<string, unknown>) =>
    call<{ document: WorkflowDocument }>(`/documents/${id(docId)}`, 'PUT', { expectedRevision, data }),
  removeDocument: (docId: string, expectedRevision: number) =>
    call<{ removed: true }>(`/documents/${id(docId)}?expectedRevision=${expectedRevision}`, 'DELETE'),
};

export interface FollowHandlers {
  /** The job's state on every (re)connect. `gap` means events since the last
   *  one seen were dropped: rebuild the view from `job`. */
  onSnapshot?: (job: WorkflowJob, gap: boolean) => void;
  /** Each event, once, in order. */
  onEvent: (event: WorkflowEvent) => void;
  /** A connection dropped; it will be retried. */
  onError?: (error: unknown) => void;
}

/** Follow a job until it finishes or `signal` aborts. Resolves with the
 *  final status, or null when aborted. `after` resumes from a known event. */
export async function followJob(jobId: string, handlers: FollowHandlers,
  opts: { after?: number; signal?: AbortSignal; retryMs?: number } = {}): Promise<WorkflowJobStatus | null> {
  let last = opts.after ?? 0;
  let delay = opts.retryMs ?? 1000;
  while (!opts.signal?.aborted) {
    try {
      const token = getAuthToken();
      const response = await fetch(`/api/workflows/jobs/${id(jobId)}/events?after=${last}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {}, signal: opts.signal,
      });
      if (!response.ok || !response.body) {
        const value = await response.json().catch(() => ({}));
        // A missing job or a bad request will not fix itself by retrying.
        if (response.status === 404 || response.status === 401 || response.status === 400) {
          throw new ApiError(response.status, value.error || `Cannot follow job (${response.status})`, value);
        }
        throw new Error(value.error || `Cannot follow job (${response.status})`);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let finalStatus: WorkflowJobStatus | null = null;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let at;
        while ((at = buffer.indexOf('\n\n')) >= 0) {
          const line = buffer.slice(0, at).split('\n').find(l => l.startsWith('data: '));
          buffer = buffer.slice(at + 2);
          if (!line) continue;
          const frame = JSON.parse(line.slice(6));
          if (frame.type === 'snapshot') {
            const job = frame.job as WorkflowJob;
            // On a gap the kept events still follow; seq checks skip none of them.
            handlers.onSnapshot?.(job, frame.gap);
            if (FINAL.has(job.status) && job.lastSeq <= last) finalStatus = job.status;
          } else if (frame.type === 'event') {
            const event = frame.event as WorkflowEvent;
            if (event.seq <= last) continue;
            last = event.seq;
            handlers.onEvent(event);
            const status = (event.data as { status?: WorkflowJobStatus } | null)?.status;
            // A retry follows an earlier final status with a new attempt.
            if (event.type === 'status' && status) finalStatus = FINAL.has(status) ? status : null;
          }
        }
      }
      if (finalStatus) return finalStatus;
      delay = opts.retryMs ?? 1000;  // a clean end without a final status: reconnect promptly
    } catch (err) {
      if (opts.signal?.aborted) return null;
      if (err instanceof ApiError) throw err;
      handlers.onError?.(err);
      delay = Math.min(delay * 2, 10_000);
    }
    await new Promise(resolve => setTimeout(resolve, delay));
  }
  return null;
}
