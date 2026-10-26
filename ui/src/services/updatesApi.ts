import { apiClient, type ReadOptions } from './httpClient';

// Wire shapes of /api/updates (server/src/contracts/updates.ts). Notices only.
export type UpdateChannel = 'release' | 'master';
export type UpdateStatusKind = 'never-checked' | 'up-to-date' | 'update-available' | 'ahead' | 'unknown' | 'could-not-check';
export interface UpdatesStatus {
  channel: UpdateChannel;
  installed: {
    kind: 'portable' | 'git' | 'unknown';
    version: string;
    tag: string | null;
    sha: string | null;
    variant: string | null;
    buildInfoProblem: string | null;
    git: { head: string; branch: string | null; dirty: boolean; originOk: boolean } | null;
  };
  status: UpdateStatusKind;
  latest: {
    release: { tag: string; name: string; url: string; publishedAt: string | null } | null;
    master: { sha: string; url: string } | null;
  };
  gitRelation: 'up-to-date' | 'behind' | 'ahead' | 'diverged' | 'unknown';
  notesUrl: string | null;
  lastSuccessAt: string | null;
  lastAttemptAt: string | null;
  stale: boolean;
  error: string | null;
  retryAfter: string | null;
  checking: boolean;
}
export interface UpdatesCheckResult extends UpdatesStatus { throttled: boolean }

export const updatesApi = {
  status: (opts?: ReadOptions) => apiClient.get<UpdatesStatus>('/updates/status', opts),
  check: (channel?: UpdateChannel) => apiClient.post<UpdatesCheckResult>('/updates/check', channel ? { channel } : {}),
};
