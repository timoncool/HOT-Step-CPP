// contracts/updates.ts — wire shapes for /api/updates (routes/updates.ts).
// Notices only: nothing here downloads, prepares or installs anything.

export type UpdateChannel = 'release' | 'master';

export const BUILD_VARIANTS = ['cuda13.1', 'cuda12.8', 'cuda12-volta', 'vulkan', 'rocm', 'cpu', 'metal'] as const;
export type BuildVariant = (typeof BUILD_VARIANTS)[number];

/** build-info.json beside VERSION (format 1). */
export interface BuildInfo {
  format: 1;
  tag: string | null;
  sha: string;
  channel: UpdateChannel;
  os: 'windows' | 'linux' | 'macos';
  arch: string;
  variant: BuildVariant;
  engineSha: string;
  updaterProtocol: number;
}

export type GitRelation = 'up-to-date' | 'behind' | 'ahead' | 'diverged' | 'unknown';

export interface GitInstall {
  head: string;
  /** null = detached HEAD. */
  branch: string | null;
  /** Tracked, untracked or submodule changes. */
  dirty: boolean;
  /** origin is scragnog/HOT-Step-CPP. */
  originOk: boolean;
}

export interface InstalledIdentity {
  /** 'portable' = release package, 'git' = source checkout, 'unknown' = neither could be proven. */
  kind: 'portable' | 'git' | 'unknown';
  /** VERSION file content ('dev' when absent). */
  version: string;
  /** Exact release tag from build-info.json, else null. */
  tag: string | null;
  /** Full source commit: build-info.json first, else the checkout's HEAD. */
  sha: string | null;
  /** Exact asset variant from build-info.json only; never guessed. */
  variant: BuildVariant | null;
  /** null = no usable build-info.json (older release or a git clone). */
  buildInfo: BuildInfo | null;
  /** Why build-info.json was ignored, when it exists but is unusable. */
  buildInfoProblem: string | null;
  git: GitInstall | null;
}

export interface LatestRelease {
  tag: string;
  name: string;
  url: string;
  publishedAt: string | null;
}

export interface LatestMaster {
  sha: string;
  url: string;
}

export type UpdateStatusKind =
  | 'never-checked'
  | 'up-to-date'
  | 'update-available'
  /** Installed is newer than the feed (dev build, unreleased commit). */
  | 'ahead'
  /** Installed identity or relation can't be compared; the latest is shown with its link. */
  | 'unknown'
  /** Last attempt failed. Never read this as up to date; `latest` is the stale result. */
  | 'could-not-check';

export interface UpdatesStatusResponse {
  channel: UpdateChannel;
  installed: InstalledIdentity;
  status: UpdateStatusKind;
  /** Latest known feed result for the selected channel (may be stale). */
  latest: { release: LatestRelease | null; master: LatestMaster | null };
  /** Installed commit versus the master tip (Master channel on a checkout only). */
  gitRelation: GitRelation;
  /** Where to read what changed; null until a feed answered. */
  notesUrl: string | null;
  lastSuccessAt: string | null;
  lastAttemptAt: string | null;
  /** Cached result is older than 24 h or the last attempt failed. */
  stale: boolean;
  /** Message from the last failed attempt, else null. */
  error: string | null;
  /** No GitHub request before this time (403/429 back-off), else null. */
  retryAfter: string | null;
  /** A refresh is running now. */
  checking: boolean;
  apply: {
    /** Always false in the notices batch. */
    offered: false;
    reason: 'notices-only';
    /** What would also stop an automatic update on this install (empty = nothing else). */
    blockers: string[];
  };
}

export interface UpdatesCheckRequest {
  /** Switch the check feed first; never changes the installation. */
  channel?: UpdateChannel;
}

export interface UpdatesCheckResponse extends UpdatesStatusResponse {
  /** True when a manual check was inside the one-minute limit and the cached result was returned. */
  throttled: boolean;
}
