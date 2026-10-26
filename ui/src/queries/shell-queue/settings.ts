// settings.ts — Settings and shared-dialog snapshots: server health/version,
// the .env form's source values, GPUs, stem storage, the data folder and the
// file browser's directory listings.

import { queryOptions } from '@tanstack/react-query';
import { adapterApi, healthApi, reportLogsApi, settingsApi } from '../../services/api';
import { audioQueueApi } from '../../services/audioQueueApi';
import { dataFolderApi } from '../../services/dataFolderApi';
import { updatesApi } from '../../services/updatesApi';
import { getStemStats } from '../../services/stemStudioApi';
import { serverKey, sharedKeys } from '../keys';

export const healthOptions = () => queryOptions({
  queryKey: serverKey('shell-queue/health'),
  queryFn: ({ signal }) => healthApi.check({ signal }),
});

/** Seeds the editable .env form. Asked fresh each time the form loads, so
 *  it never starts from values another tab or a restart has replaced. */
export const envOptions = () => queryOptions({
  queryKey: serverKey('shell-queue/env'),
  queryFn: ({ signal }) => settingsApi.getEnv({ signal }),
  staleTime: 0,
});

export const gpusOptions = () => queryOptions({
  queryKey: sharedKeys.gpus(),
  queryFn: ({ signal }) => settingsApi.getGpus({ signal }),
});

/** Log tails for the Report an issue tab; fresh each visit so the report shows the latest lines. */
export const reportLogsOptions = () => queryOptions({
  queryKey: serverKey('shell-queue/report-logs'),
  queryFn: ({ signal }) => reportLogsApi.snapshot({ signal }),
  staleTime: 0,
});

/** Newest failed audio queue item, for the Report an issue tab. Auth. */
export const reportFailedJobOptions = () => queryOptions({
  queryKey: serverKey('shell-queue/report-failed-job', { auth: true }),
  queryFn: async ({ signal }) => {
    const { items } = await audioQueueApi.list({ signal });
    return items.filter(i => i.status === 'failed').sort((a, b) => b.updatedAt - a.updatedAt)[0] ?? null;
  },
  staleTime: 0,
});

export const stemStatsOptions = () => queryOptions({
  queryKey: serverKey('shell-queue/stem-stats'),
  queryFn: ({ signal }) => getStemStats({ signal }),
  staleTime: 0,
});

/** The data folder and any pending move. Fresh on every visit: a scheduled
 *  move changes it, and a restart applies it. */
export const dataFolderOptions = () => queryOptions({
  queryKey: serverKey('shell-queue/data-folder'),
  queryFn: ({ signal }) => dataFolderApi.location({ signal }),
  staleTime: 0,
});

/** An explicit adapter-folder scan (a read-only POST). Fresh per click. */
export const adapterScanOptions = (folder: string) => queryOptions({
  queryKey: sharedKeys.adapterScan(folder),
  queryFn: ({ signal }) => adapterApi.scan(folder, { signal }),
  staleTime: 0,
});

/** One directory listing in the file browser. Fresh each time it is opened,
 *  since files come and go; the cache only joins repeated requests. */
export const browseOptions = (dir: string, filter?: string) => queryOptions({
  queryKey: sharedKeys.browse(dir, filter),
  queryFn: ({ signal }) => adapterApi.browse(dir, filter, { signal }),
  staleTime: 0,
});

/** The server's cached update notice. Never calls GitHub; a manual check
 *  writes its result straight into this entry. */
export const updatesStatusOptions = () => queryOptions({
  queryKey: serverKey('shell-queue/updates'),
  queryFn: ({ signal }) => updatesApi.status({ signal }),
  staleTime: 0,
});
