// GitHub update checker: 24 h cache with ETags, coalesced requests, a one-minute
// manual limit and 403/429 back-off. One request per check, for the selected
// channel's feed only. Every dependency is injected so tests run offline.
import type {
  GitRelation, InstalledIdentity, LatestMaster, LatestRelease, UpdateChannel,
  UpdateStatusKind, UpdatesCheckResponse, UpdatesStatusResponse,
} from '../../contracts/updates.js';
import { applyBlockers, REPO } from './install.js';

export const STALE_MS = 24 * 3600_000;
export const MANUAL_MIN_MS = 60_000;
export const AUTO_RETRY_MS = 3600_000;
export const BACKOFF_MIN_MS = 3600_000;
const TIMEOUT_MS = 10_000;
const MAX_BODY = 1_000_000;

export interface HttpResponse {
  status: number;
  headers: { get(name: string): string | null };
  text(): Promise<string>;
}
export type FetchLike = (url: string, init: { headers: Record<string, string>; signal: AbortSignal }) => Promise<HttpResponse>;

interface Feed<T> {
  etag: string | null;
  data: T | null;
  /** ms epoch of the last answer (200 or 304). */
  okAt: number | null;
  attemptAt: number | null;
  error: string | null;
  retryUntil: number | null;
}
export interface UpdateState {
  version: 1;
  channel: UpdateChannel;
  release: Feed<LatestRelease>;
  master: Feed<LatestMaster>;
}

const emptyFeed = <T>(): Feed<T> => ({ etag: null, data: null, okAt: null, attemptAt: null, error: null, retryUntil: null });
export const freshState = (): UpdateState => ({ version: 1, channel: 'release', release: emptyFeed(), master: emptyFeed() });

export interface CheckerDeps {
  fetch: FetchLike;
  now(): number;
  load(): UpdateState | null;
  save(s: UpdateState): void;
  install(): InstalledIdentity;
  relation(install: InstalledIdentity, masterSha: string | null): GitRelation;
  platform: NodeJS.Platform;
}

const STABLE = /^v?(\d+)\.(\d+)\.(\d+)$/;
const SHA = /^[0-9a-f]{40}$/i;
const parseVer = (s: string | null): [number, number, number] | null => {
  const m = s ? STABLE.exec(s.trim()) : null;
  return m ? [+m[1], +m[2], +m[3]] : null;
};
const cmpVer = (a: number[], b: number[]) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2];

function parseRelease(body: string): LatestRelease {
  const j = JSON.parse(body);
  if (!j || typeof j.tag_name !== 'string' || !parseVer(j.tag_name)) throw new Error('latest release is not a stable version tag');
  if (j.draft === true || j.prerelease === true) throw new Error('latest release is not a published stable release');
  if (typeof j.html_url !== 'string' || !j.html_url.startsWith(`https://github.com/${REPO}/`)) throw new Error('latest release has an unexpected link');
  return { tag: j.tag_name, name: typeof j.name === 'string' && j.name ? j.name : j.tag_name, url: j.html_url, publishedAt: typeof j.published_at === 'string' ? j.published_at : null };
}

function parseMaster(body: string): LatestMaster {
  const sha = JSON.parse(body)?.object?.sha;
  if (typeof sha !== 'string' || !SHA.test(sha)) throw new Error('master ref has no commit sha');
  return { sha: sha.toLowerCase(), url: `https://github.com/${REPO}/commit/${sha.toLowerCase()}` };
}

/** ms to wait from the headers, never below the one-hour floor. */
function backoffMs(res: HttpResponse, now: number): number {
  let ms = 0;
  const ra = res.headers.get('retry-after');
  if (ra) {
    const secs = Number(ra);
    const at = Number.isFinite(secs) ? now + secs * 1000 : Date.parse(ra);
    if (Number.isFinite(at)) ms = Math.max(ms, at - now);
  }
  const reset = Number(res.headers.get('x-ratelimit-reset'));
  if (res.headers.get('x-ratelimit-remaining') === '0' && Number.isFinite(reset) && reset > 0) ms = Math.max(ms, reset * 1000 - now);
  return Math.max(ms, BACKOFF_MIN_MS);
}

export function createChecker(deps: CheckerDeps) {
  let state: UpdateState | null = null;
  const inflight: Partial<Record<UpdateChannel, Promise<void>>> = {};
  const get = (): UpdateState => (state ??= deps.load() ?? freshState());
  const save = () => { try { deps.save(get()); } catch (e) { console.warn('[Updates] could not save state:', (e as Error).message); } };

  async function refresh(channel: UpdateChannel): Promise<void> {
    const s = get();
    const feed = s[channel] as Feed<any>;
    const now = deps.now();
    feed.attemptAt = now;
    const url = channel === 'release'
      ? `https://api.github.com/repos/${REPO}/releases/latest`
      : `https://api.github.com/repos/${REPO}/git/ref/heads/master`;
    const headers: Record<string, string> = { Accept: 'application/vnd.github+json', 'User-Agent': 'HOT-Step-update-check', 'X-GitHub-Api-Version': '2022-11-28' };
    if (feed.etag && feed.data) headers['If-None-Match'] = feed.etag;
    try {
      const res = await deps.fetch(url, { headers, signal: AbortSignal.timeout(TIMEOUT_MS) });
      if (res.status === 304 && feed.data) {
        feed.okAt = now; feed.error = null; feed.retryUntil = null;
      } else if (res.status === 403 || res.status === 429) {
        feed.retryUntil = now + backoffMs(res, now);
        feed.error = `GitHub is rate limiting this network (HTTP ${res.status})`;
      } else if (res.status !== 200) {
        feed.error = `GitHub answered HTTP ${res.status}`;
      } else {
        const body = await res.text();
        if (body.length > MAX_BODY) throw new Error('GitHub response is too large');
        feed.data = channel === 'release' ? parseRelease(body) : parseMaster(body);
        feed.etag = res.headers.get('etag');
        feed.okAt = now; feed.error = null; feed.retryUntil = null;
      }
    } catch (e) {
      // Offline, timeout or malformed: keep the last result and say we couldn't check.
      feed.error = `Could not reach GitHub: ${(e as Error).message}`;
    }
    save();
  }

  function status(): UpdatesStatusResponse {
    const s = get();
    const channel = s.channel;
    const feed = s[channel] as Feed<any>;
    const installed = deps.install();
    const release = s.release.data;
    const master = s.master.data;
    const relation: GitRelation = channel === 'master' && installed.kind === 'git' ? deps.relation(installed, master?.sha ?? null) : 'unknown';
    let kind: UpdateStatusKind;
    let notesUrl: string | null = null;
    if (!feed.data) kind = feed.error ? 'could-not-check' : 'never-checked';
    else {
      if (channel === 'release') {
        const have = parseVer(installed.tag ?? installed.version);
        const want = parseVer(release!.tag)!;
        kind = !have ? 'unknown' : cmpVer(have, want) < 0 ? 'update-available' : cmpVer(have, want) > 0 ? 'ahead' : 'up-to-date';
        notesUrl = release!.url;
      } else {
        const have = installed.sha;
        const tip = master!.sha;
        notesUrl = have && have !== tip ? `https://github.com/${REPO}/compare/${have}...${tip}` : master!.url;
        if (!have) kind = 'unknown';
        else if (have.toLowerCase() === tip) kind = 'up-to-date';
        else if (installed.kind === 'git') kind = relation === 'behind' ? 'update-available' : relation === 'ahead' ? 'ahead' : 'unknown';
        else kind = 'update-available';
      }
      if (feed.error) kind = 'could-not-check';
    }
    const iso = (t: number | null) => (t === null ? null : new Date(t).toISOString());
    return {
      channel,
      installed,
      status: kind,
      latest: { release, master },
      gitRelation: relation,
      notesUrl,
      lastSuccessAt: iso(feed.okAt),
      lastAttemptAt: iso(feed.attemptAt),
      stale: !!feed.error || feed.okAt === null || deps.now() - feed.okAt > STALE_MS,
      error: feed.error,
      retryAfter: feed.retryUntil && feed.retryUntil > deps.now() ? iso(feed.retryUntil) : null,
      checking: !!inflight[channel],
      apply: { offered: false, reason: 'notices-only', blockers: applyBlockers(installed, relation, deps.platform) },
    };
  }

  /** Manual: at most once a minute per feed. Auto (startup): only when stale and not retried within the hour. */
  async function check(opts: { manual: boolean; channel?: UpdateChannel }): Promise<UpdatesCheckResponse> {
    const s = get();
    if (opts.channel && opts.channel !== s.channel) { s.channel = opts.channel; save(); }
    const feed = s[s.channel] as Feed<any>;
    const now = deps.now();
    let skip: boolean;
    let throttled = false;
    if (inflight[s.channel]) skip = true;
    else if (feed.retryUntil && feed.retryUntil > now) skip = true;
    else if (opts.manual) skip = throttled = feed.attemptAt !== null && now - feed.attemptAt < MANUAL_MIN_MS;
    else skip = (feed.okAt !== null && now - feed.okAt <= STALE_MS) || (feed.attemptAt !== null && now - feed.attemptAt < AUTO_RETRY_MS);
    const ch = s.channel;
    if (!skip) inflight[ch] = refresh(ch).finally(() => { delete inflight[ch]; });
    // Concurrent callers for a feed share its running refresh; the other feed is not blocked.
    await inflight[ch];
    return { ...status(), throttled };
  }

  return { status, check };
}
