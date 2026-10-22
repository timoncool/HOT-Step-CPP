import { queryClient } from '../queries/client';
import { serverKey, resourceKey } from '../queries/keys';
import { queryManager } from '../queries/invalidation';
import { ACTIVE_POLL_MS } from '../queries/contract';
import { waitForSnapshot } from '../queries/studios/wait';
import { apiClient, type ReadOptions } from './httpClient';
// stemStudioApi.ts — API client for Stem Studio extraction
//
// Communicates with Node server /api/stem-studio endpoints.

const API_BASE = '/api/stem-studio';
/** Path prefix for apiClient calls (the client already adds `/api`). */
const P = '/stem-studio';

// ── Types ────────────────────────────────────────────────────────────────

export interface ExtractStemInfo {
  trackName: string;
  audioUrl: string;
  durationSec: number;
  index: number;
  sizeBytes?: number;
}

export interface ExtractJobResult {
  id: string;
  stems: ExtractStemInfo[];
}

export interface ExtractProgress {
  status: 'pending' | 'extracting' | 'separating' | 'saving' | 'done' | 'failed' | 'cancelled';
  progress: number;
  currentTrack: string;
  completedStems: string[];
  totalTracks: number;
  warning?: string;
  error?: string;
  sepMessage?: string;       // SuperSep status message from ace-server
}

export interface ExtractJobSummary {
  id: string;
  type?: 'extract' | 'supersep';
  sourceFileName: string;
  tracks: string[];
  completedStems: string[];
  createdAt: string;
  sepLevel?: number;
}

export interface ExtractParams {
  sourceAudioUrl: string;
  sourceFileName: string;
  tracks: string[];
  style?: string;
  lyrics?: string;
  ditSettings: Record<string, any>;
}

export interface SupersepParams {
  sourceAudioUrl: string;
  sourceFileName: string;
  level: number;   // 0=Basic, 1=Vocal Split, 2=Full, 3=Maximum
}

export interface StemStats {
  totalBytes: number;
  jobCount: number;
  stemCount: number;
}

// ── Constants ────────────────────────────────────────────────────────────

export const EXTRACT_TRACKS = [
  'vocals', 'backing_vocals', 'drums', 'bass', 'guitar', 'keyboard',
  'percussion', 'strings', 'synth', 'fx', 'brass', 'woodwinds',
] as const;

export type ExtractTrack = typeof EXTRACT_TRACKS[number];

export const TRACK_CATEGORIES: Record<string, 'vocals' | 'instruments' | 'drums' | 'other'> = {
  vocals: 'vocals',
  backing_vocals: 'vocals',
  drums: 'drums',
  percussion: 'drums',
  bass: 'instruments',
  guitar: 'instruments',
  keyboard: 'instruments',
  strings: 'instruments',
  synth: 'instruments',
  brass: 'instruments',
  woodwinds: 'instruments',
  fx: 'other',
};

/** Human-readable track labels */
export const TRACK_LABELS: Record<string, string> = {
  vocals: 'Vocals',
  backing_vocals: 'Backing Vocals',
  drums: 'Drums',
  bass: 'Bass',
  guitar: 'Guitar',
  keyboard: 'Keyboard',
  percussion: 'Percussion',
  strings: 'Strings',
  synth: 'Synth',
  fx: 'FX',
  brass: 'Brass',
  woodwinds: 'Woodwinds',
};

// ── API Functions ────────────────────────────────────────────────────────

/** Submit a new extraction job. Returns the job ID. */
export async function submitExtraction(params: ExtractParams): Promise<string> {
  const data = await apiClient.post<{ id: string }>(`${P}/extract`, params);
  return data.id;
}

/** Submit a new SuperSep separation job. Returns the job ID. */
export async function submitSupersep(params: SupersepParams): Promise<string> {
  const data = await apiClient.post<{ id: string }>(`${P}/supersep`, params);
  return data.id;
}

/** Poll extraction job progress. */
export async function getExtractProgress(jobId: string, opts?: ReadOptions): Promise<ExtractProgress> {
  return apiClient.get<ExtractProgress>(`${P}/${jobId}/progress`, opts);
}

/** Get completed extraction result with stem metadata. */
export async function getExtractResult(jobId: string, opts?: ReadOptions): Promise<ExtractJobResult> {
  return apiClient.get<ExtractJobResult>(`${P}/${jobId}/result`, opts);
}

/** Get the direct URL for a specific stem's audio. */
export function getStemUrl(jobId: string, trackName: string): string {
  return `${API_BASE}/${jobId}/stem/${trackName}`;
}

/** Get the URL for downloading all stems as a ZIP. */
export function getDownloadAllUrl(jobId: string): string {
  return `${API_BASE}/${jobId}/download-all`;
}

/** List all past extraction jobs (newest first). */
export async function listJobs(opts?: ReadOptions): Promise<ExtractJobSummary[]> {
  return apiClient.get<ExtractJobSummary[]>(`${P}/jobs`, opts);
}

/** Delete a single extraction job and its files. */
export async function deleteJob(jobId: string): Promise<void> {
  await apiClient.delete(`${P}/${jobId}`);
}

/** Delete ALL extraction jobs and stems. */
export async function deleteAllJobs(): Promise<void> {
  await apiClient.delete(`${P}/all`);
}

/** Get stem storage statistics for the Settings page. */
export async function getStemStats(opts?: ReadOptions): Promise<StemStats> {
  return apiClient.get<StemStats>(`${P}/stats`, opts);
}

/**
 * Poll until extraction completes, calling onProgress along the way.
 * Returns the final result with stem metadata.
 */
export async function waitForExtraction(
  jobId: string,
  onProgress?: (p: ExtractProgress) => void,
  pollMs = ACTIVE_POLL_MS,
): Promise<ExtractJobResult> {
  const p = await waitForSnapshot(serverKey('studios/extract-progress', { id: jobId }),
    signal => getExtractProgress(jobId, { signal }),
    p => !['done', 'failed', 'cancelled'].includes(p.status), onProgress, { pollMs });
  if (p.status !== 'done') throw new Error(p.error || `Extraction ${p.status}`);
  const result = await queryClient.query({
    queryKey: serverKey('studios/extract-result', { id: jobId }),
    queryFn: ({ signal }) => getExtractResult(jobId, { signal }),
  });
  await queryManager.keys(queryClient, resourceKey('studios/extractions'));
  return result;
}

/** Format bytes for display */
export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}
