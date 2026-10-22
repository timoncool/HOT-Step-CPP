// queue.ts — the generation queue's server reads.
//
// The queue executor (stores/audioGenQueueStore.ts) keeps its own sequencing,
// legacy owner/migration rules and completion effects; Query owns the reads
// it makes. Every read goes through query(), so two loops asking for the
// same thing at once share one request, and a loop that is happy with a
// recent answer takes the cache instead of asking again. Reads never retry
// here: each executor loop already decides what a failed read means.
// Submits, cancels and other commands stay direct transport calls.

import { queryOptions } from '@tanstack/react-query';
import { generateApi, songApi } from '../../services/api';
import { audioQueueApi } from '../../services/audioQueueApi';
import { apiClient } from '../../services/httpClient';
import type { WorkflowReplay } from '../../../../server/src/contracts/workflow';
import { queryClient } from '../client';
import { ACTIVE_POLL_MS } from '../contract';
import { serverKey, sharedKeys } from '../keys';
import { songOptions } from './songs';

export const audioQueueItemsOptions = () => queryOptions({
  queryKey: serverKey('shell-queue/audio-queue', { auth: true }),
  queryFn: ({ signal }) => audioQueueApi.list({ signal }),
  retry: false,
});

export const generationStatusOptions = (jobId: string) => queryOptions({
  queryKey: serverKey('shell-queue/generation-status', { id: jobId }),
  queryFn: ({ signal }) => generateApi.status(jobId, { signal }),
  retry: false,
});

export const workflowJobOptions = (jobId: string) => queryOptions({
  queryKey: sharedKeys.workflowJob(jobId),
  queryFn: ({ signal }) => apiClient.get<WorkflowReplay>(`/workflows/jobs/${encodeURIComponent(jobId)}?after=0`, { signal }),
  retry: false,
});

export const songIdsOptions = () => queryOptions({
  queryKey: serverKey('shell-queue/song-ids', { auth: true }),
  queryFn: ({ signal }) => songApi.listIds({ signal }),
  retry: false,
});

export const selectedPathOptions = () => queryOptions({
  queryKey: serverKey('shell-queue/selected-path', { auth: true }),
  queryFn: ({ signal }) => generateApi.selectedPath({ signal }),
  retry: false,
});

/** The server queue's items. `maxAgeMs` 0 always asks the server (joining a
 *  request already in flight); a waiter that only needs "recent" passes the
 *  poll cadence and rides the projection's own reads. */
export const readAudioQueue = (maxAgeMs = 0) =>
  queryClient.query({ ...audioQueueItemsOptions(), staleTime: maxAgeMs });

/** One job's status. Every loop that follows a job polls at about the active
 *  cadence, so answers younger than a poll are shared rather than re-asked. */
export const readGenerationStatus = (jobId: string) =>
  queryClient.query({ ...generationStatusOptions(jobId), staleTime: ACTIVE_POLL_MS - 500 });

export const readWorkflowJob = (jobId: string) =>
  queryClient.query({ ...workflowJobOptions(jobId), staleTime: 0 });

/** A finished render's song row, fresh: the queue reads it the moment the
 *  render is saved. Shares the canonical song key with every other reader. */
export const readSong = (id: string) =>
  queryClient.query({ ...songOptions(id), staleTime: 0 });

export const readSongIds = () =>
  queryClient.query({ ...songIdsOptions(), staleTime: 0 });

/** Which submit path the server runs. Asked fresh: Settings can switch it,
 *  and the next submit must follow. */
export const readSelectedPath = () =>
  queryClient.query({ ...selectedPathOptions(), staleTime: 0 });
