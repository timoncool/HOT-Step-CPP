// sa3PreviewApi.ts — the SA3 checkpoint-preview routes (server/src/contracts/training.ts).
//
//   GET  /datasets/:id/sa3-previews?run=   -> Sa3PreviewListResponse
//   POST /datasets/:id/sa3-previews/render <- Sa3PreviewRenderRequest
//
// Audio is played from each record's server-validated `audioUrl`; the browser
// never builds a filesystem path. Reads go to the selected worker, like
// every other SA3 training read.
import type { Sa3PreviewListResponse, Sa3PreviewRenderRequest } from '../../../server/src/contracts/training';
import { apiClient, type ReadOptions } from './httpClient';

export type { Sa3PreviewListResponse, Sa3PreviewRecord, Sa3PreviewRenderRequest, Sa3PreviewSettings } from '../../../server/src/contracts/training';

export interface Sa3PreviewTarget extends ReadOptions { worker: string | null }

const base = (worker: string | null) => worker ? `/api/workers/${encodeURIComponent(worker)}/api/training` : '/api/training';
const path = (datasetId: string) => `/datasets/${encodeURIComponent(datasetId)}/sa3-previews`;

export function listSa3Previews(datasetId: string, run: string, target: Sa3PreviewTarget): Promise<Sa3PreviewListResponse> {
  return apiClient.request<Sa3PreviewListResponse>('GET', `${path(datasetId)}?run=${encodeURIComponent(run)}`, undefined,
    { signal: target.signal, baseURL: base(target.worker) });
}

/** Render, or retry, one saved step. Resolves with the refreshed list. */
export function renderSa3Preview(datasetId: string, body: Sa3PreviewRenderRequest, worker: string | null): Promise<Sa3PreviewListResponse> {
  return apiClient.request<Sa3PreviewListResponse>('POST', `${path(datasetId)}/render`, body, { baseURL: base(worker) });
}
