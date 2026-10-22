// studio-drafts domain client: reads go through the shared query cache
// (readSnapshot), writes hit the transport directly and then invalidate the
// domain's keys (changed). The bearer comes from the auth session accessor.

import type { PlaylistBody, StudioDraftBody } from '../../../server/src/contracts/studioDrafts';
import type { DocumentImportReceipt, TypedDocument, WorkflowDocument } from '../../../server/src/contracts/workflow';
import { apiClient, type ReadOptions } from './httpClient';
import { readSnapshot, changed } from '../queries/cover-lyrics/reads';
import { getSessionEpoch } from '../queries/keys';

async function request<T>(url: string, method = 'GET', body?: unknown, options: ReadOptions = {}): Promise<T> {
  const target = { session: getSessionEpoch(), api: apiClient.baseUrl };
  const path = url.replace(/^\/api/, '');
  if (method === 'GET') return readSnapshot<T>('drafts', path, options);
  const result = method === 'DELETE' ? await apiClient.delete<T>(path, options)
    : method === 'PUT' ? await apiClient.put<T>(path, body, options)
    : await apiClient.post<T>(path, body, options);
  await changed('drafts', target);
  return result;
}

const path = '/api/studio-drafts';
const id = (value: string) => encodeURIComponent(value);
export type PlaylistDocument = TypedDocument<PlaylistBody>;
export type StudioDraftDocument = TypedDocument<StudioDraftBody>;

export const studioDraftsApi = {
  playlist: (options: ReadOptions = {}) => request<{ document: PlaylistDocument | null }>(`${path}/playlist`, 'GET', undefined, options),
  playlistCommand: (expectedRevision: number, command: unknown) =>
    request<{ document: PlaylistDocument }>(`${path}/playlist/commands`, 'POST', { expectedRevision, command }),
  handoff: (documentId: string, options: ReadOptions = {}) =>
    request<{ document: WorkflowDocument }>(`${path}/handoffs/${id(documentId)}`, 'GET', undefined, options),
  list: (studio?: StudioDraftBody['studio'], options: ReadOptions = {}) =>
    request<{ documents: StudioDraftDocument[] }>(`${path}/drafts${studio ? `?studio=${id(studio)}` : ''}`, 'GET', undefined, options),
  create: (body: StudioDraftBody) =>
    request<{ document: StudioDraftDocument }>(`${path}/drafts`, 'POST', { body }),
  get: (documentId: string, options: ReadOptions = {}) =>
    request<{ document: StudioDraftDocument; sourceError: string | null }>(`${path}/drafts/${id(documentId)}`, 'GET', undefined, options),
  update: (documentId: string, expectedRevision: number, body: StudioDraftBody) =>
    request<{ document: StudioDraftDocument }>(`${path}/drafts/${id(documentId)}`, 'PUT', { expectedRevision, body }),
  remove: (documentId: string, expectedRevision: number) =>
    request<{ removed: true }>(`${path}/drafts/${id(documentId)}?expectedRevision=${expectedRevision}`, 'DELETE'),
  importValue: (input: { storageKey: string; raw: string; sourceHash: string; schemaVersion: 1; expectedRevision: number; resolution?: 'keep-both' | 'replace' }) =>
    request<{ receipt: DocumentImportReceipt; document: PlaylistDocument | StudioDraftDocument | null; created: boolean }>(`${path}/import`, 'POST', input),
};
