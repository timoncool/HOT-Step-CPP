import { apiClient, type ReadOptions } from './httpClient';
// streamSessionsApi.ts — client for /api/stream-sessions (Batch 5 row 8a).
//
// The STORM and MM3 stream responses name their Node session in the
// X-Stream-Session header (useStreamAudio's sessionId, mm3StreamSessionId).
// Node analyses every chunk and, between start and stop, records the engine's
// own samples; export returns that canonical recording. Browser device capture
// (useStreamAudio's startRecording) is separate and unchanged.
import type { StreamExportFormat, StreamSessionSummary } from '../../../server/src/contracts/streamSessions';

export type { StreamExportFormat, StreamSessionSummary } from '../../../server/src/contracts/streamSessions';

const base = (id: string) => `/stream-sessions/${encodeURIComponent(id)}`;

export async function getStreamSession(id: string, opts?: ReadOptions): Promise<StreamSessionSummary> {
  return (await apiClient.get<{ session: StreamSessionSummary }>(base(id), opts)).session;
}

export async function streamRecording(id: string, action: 'start' | 'stop' | 'discard'): Promise<StreamSessionSummary> {
  return (await apiClient.post<{ session: StreamSessionSummary }>(`${base(id)}/recording`, { action })).session;
}

/** URL of a stopped recording as a download; open it in an <a download>. */
export function streamRecordingExportUrl(id: string, format: StreamExportFormat = 'wav', bitrate?: number): string {
  const query = new URLSearchParams({ format, ...(bitrate ? { bitrate: String(bitrate) } : {}) });
  return apiClient.eventSourceUrl(`${base(id)}/recording/export?${query}`);
}
