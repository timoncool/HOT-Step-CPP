import { apiClient, type ReadOptions } from './httpClient';
import type { AudioIntentItem, AudioQueueState, ImportAudioQueue } from '../../../server/src/contracts/audioQueue';

async function call<T>(path: string, body?: unknown, opts: ReadOptions = {}): Promise<T> {
  return body === undefined
    ? apiClient.get<T>(`/audio-queue${path}`, opts)
    : apiClient.post<T>(`/audio-queue${path}`, body, opts);
}

export const audioQueueApi = {
  list: (opts?: ReadOptions) => call<{ items: AudioIntentItem[] }>('/items', undefined, opts),
  state: (opts?: ReadOptions) => call<AudioQueueState>('/state', undefined, opts),
  enqueue: (idempotencyKey: string, request: Record<string, unknown>, meta?: Record<string, unknown>) =>
    call<{ item: AudioIntentItem; created: boolean }>('/items', { idempotencyKey, request, meta }),
  cancel: (id: string) => call<{ item: AudioIntentItem }>(`/items/${encodeURIComponent(id)}/cancel`, {}),
  retry: (id: string) => call<{ item: AudioIntentItem }>(`/items/${encodeURIComponent(id)}/retry`, {}),
  dismiss: async (id: string): Promise<void> => {
    await apiClient.delete(`/audio-queue/items/${encodeURIComponent(id)}`);
  },
  importLegacy: (body: ImportAudioQueue) => call<{
    backupId: string; imported: number; existing: number;
    items: { legacyId: string; itemId: string; status: string }[];
  }>('/migration/import', body),
  resumeHeld: (ids: string[]) => call('/migration/resume-held', { ids }),
  rollbackExport: () => call<{ version: number; exportedAt: number; state: AudioQueueState; items: AudioIntentItem[] }>(
    '/migration/rollback-export', {}),
  pause: () => call<AudioQueueState>('/pause', {}),
  resume: () => call<AudioQueueState>('/resume', {}),
};
