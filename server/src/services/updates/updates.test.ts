import test from 'node:test';
import assert from 'node:assert/strict';
import { createChecker, BACKOFF_MIN_MS, STALE_MS } from './checker.js';
import type { CheckerDeps, FetchLike, UpdateState } from './checker.js';
import { trustedOrigin } from './origin.js';
import { applyBlockers, detectInstall, gitRelation, parseBuildInfo } from './install.js';
import type { InstallDeps } from './install.js';
import type { InstalledIdentity } from '../../contracts/updates.js';

const SHA_A = 'a'.repeat(40);
const SHA_B = 'b'.repeat(40);
const REL = (tag: string) => JSON.stringify({ tag_name: tag, name: tag, html_url: `https://github.com/scragnog/HOT-Step-CPP/releases/tag/${tag}`, draft: false, prerelease: false });
const REF = (sha: string) => JSON.stringify({ object: { sha } });
const ok = (body: string, headers: Record<string, string> = {}) => ({ status: 200, headers: { get: (k: string) => headers[k.toLowerCase()] ?? null }, text: async () => body });
const code = (status: number, headers: Record<string, string> = {}) => ({ status, headers: { get: (k: string) => headers[k.toLowerCase()] ?? null }, text: async () => '' });

const portable = (over: Partial<InstalledIdentity> = {}): InstalledIdentity => ({
  kind: 'portable', version: '1.0.0', tag: null, sha: null, variant: null, buildInfo: null, buildInfoProblem: null, git: null, ...over,
});

function setup(responses: Array<ReturnType<typeof ok> | Error>, id: InstalledIdentity = portable()) {
  let t = 1_000_000_000_000;
  const calls: Array<{ url: string; headers: Record<string, string> }> = [];
  let saved: UpdateState | null = null;
  const fetch: FetchLike = async (url, init) => {
    calls.push({ url, headers: init.headers });
    const r = responses.shift();
    if (!r) throw new Error('unexpected fetch');
    if (r instanceof Error) throw r;
    return r;
  };
  const deps: CheckerDeps = {
    fetch, now: () => t, load: () => null, save: s => { saved = structuredClone(s); },
    install: () => id, relation: () => 'behind', platform: 'win32',
  };
  return { c: createChecker(deps), calls, advance: (ms: number) => { t += ms; }, saved: () => saved, deps };
}

test('release: newer stable tag is an update; first check sends no ETag, the next sends it', async () => {
  const s = setup([ok(REL('v1.2.0'), { etag: 'W/"x"' }), code(304)]);
  const r = await s.c.check({ manual: true });
  assert.equal(r.status, 'update-available');
  assert.equal(r.latest.release?.tag, 'v1.2.0');
  assert.equal(s.calls[0].headers['If-None-Match'], undefined);
  s.advance(61_000);
  const r2 = await s.c.check({ manual: true });
  assert.equal(s.calls[1].headers['If-None-Match'], 'W/"x"');
  assert.equal(r2.status, 'update-available');
  assert.equal(r2.error, null);
});

test('release: same or newer installed version is up-to-date / ahead; unparseable is unknown', async () => {
  assert.equal((await setup([ok(REL('v1.0.0'))]).c.check({ manual: true })).status, 'up-to-date');
  assert.equal((await setup([ok(REL('v0.9.0'))]).c.check({ manual: true })).status, 'ahead');
  assert.equal((await setup([ok(REL('v1.0.0'))], portable({ version: 'dev' })).c.check({ manual: true })).status, 'unknown');
});

test('release: prerelease or non-semver latest is malformed and never "up to date"', async () => {
  const bad = JSON.stringify({ tag_name: 'v2.0.0-beta1', html_url: 'https://github.com/scragnog/HOT-Step-CPP/x' });
  const r = await setup([ok(bad)]).c.check({ manual: true });
  assert.equal(r.status, 'could-not-check');
  assert.equal(r.latest.release, null);
});

test('offline keeps the stale result and reports could-not-check', async () => {
  const s = setup([ok(REL('v1.0.0')), new Error('ENOTFOUND')]);
  assert.equal((await s.c.check({ manual: true })).status, 'up-to-date');
  s.advance(61_000);
  const r = await s.c.check({ manual: true });
  assert.equal(r.status, 'could-not-check');
  assert.equal(r.latest.release?.tag, 'v1.0.0');
  assert.match(r.error!, /ENOTFOUND/);
  assert.equal(r.stale, true);
});

test('manual check is limited to once a minute and does not call GitHub', async () => {
  const s = setup([ok(REL('v1.0.0'))]);
  await s.c.check({ manual: true });
  s.advance(30_000);
  const r = await s.c.check({ manual: true });
  assert.equal(r.throttled, true);
  assert.equal(s.calls.length, 1);
});

test('concurrent checks share one request', async () => {
  const s = setup([ok(REL('v1.0.0'))]);
  const [a, b] = await Promise.all([s.c.check({ manual: true }), s.c.check({ manual: true })]);
  assert.equal(s.calls.length, 1);
  assert.equal(a.throttled, false);
  assert.equal(b.throttled, false);
});

test('403 backs off at least an hour, honours a longer Retry-After, and blocks requests meanwhile', async () => {
  const s = setup([code(429, { 'retry-after': '30' })]);
  const r = await s.c.check({ manual: true });
  assert.equal(r.status, 'could-not-check');
  assert.ok(Date.parse(r.retryAfter!) - Date.parse(r.lastAttemptAt!) >= BACKOFF_MIN_MS);
  s.advance(120_000);
  await s.c.check({ manual: true });
  assert.equal(s.calls.length, 1);
  s.advance(BACKOFF_MIN_MS);
  const long = setup([code(403, { 'retry-after': '7200' })]);
  const r2 = await long.c.check({ manual: true });
  assert.equal(Date.parse(r2.retryAfter!) - Date.parse(r2.lastAttemptAt!), 7200_000);
});

test('startup check only runs when the cache is stale, and not within an hour of a failed attempt', async () => {
  const s = setup([ok(REL('v1.0.0')), new Error('down'), ok(REL('v1.0.0'))]);
  await s.c.check({ manual: false });
  s.advance(STALE_MS - 1000);
  await s.c.check({ manual: false });
  assert.equal(s.calls.length, 1);
  s.advance(2000);
  await s.c.check({ manual: false });
  assert.equal(s.calls.length, 2);
  s.advance(60_000);
  await s.c.check({ manual: false });
  assert.equal(s.calls.length, 2);
  s.advance(3600_000);
  await s.c.check({ manual: false });
  assert.equal(s.calls.length, 3);
});

test('state is saved and a channel switch changes only the feed', async () => {
  const s = setup([ok(REF(SHA_B))], portable({ sha: SHA_A }));
  const r = await s.c.check({ manual: true, channel: 'master' });
  assert.equal(r.channel, 'master');
  assert.match(s.calls[0].url, /git\/ref\/heads\/master$/);
  assert.equal(r.status, 'update-available');
  assert.match(r.notesUrl!, new RegExp(`compare/${SHA_A}\\.\\.\\.${SHA_B}$`));
  assert.equal(s.saved()?.channel, 'master');
  assert.equal(r.installed.kind, 'portable');
});

test('master on a checkout follows the git relation', async () => {
  const git = { head: SHA_A, branch: 'master', dirty: false, originOk: true };
  const id = portable({ kind: 'git', sha: SHA_A, git });
  const seed = (rel: 'behind' | 'ahead' | 'diverged' | 'unknown') => {
    const s = setup([ok(REF(SHA_B))], id);
    s.deps.relation = () => rel;
    return createChecker(s.deps).check({ manual: true, channel: 'master' });
  };
  assert.equal((await seed('behind')).status, 'update-available');
  assert.equal((await seed('ahead')).status, 'ahead');
  assert.equal((await seed('diverged')).status, 'unknown');
});

// ---- install detection ----

const gitFake = (map: Record<string, string | Error>) => (args: string[]) => {
  const v = map[args.join(' ')];
  if (v === undefined) throw new Error(`unfaked git ${args.join(' ')}`);
  if (v instanceof Error) throw v;
  return v;
};
const base = (git: InstallDeps['git'], files: Record<string, string> = {}, over: Partial<InstallDeps> = {}): InstallDeps => ({
  root: process.cwd(), portable: false, version: 'dev', readFile: f => files[f.replace(/\\/g, '/').split('/').pop()!] ?? null, git, dirty: () => false, ...over,
});
const cleanGit = { 'rev-parse --show-toplevel': process.cwd(), 'rev-parse HEAD': SHA_A, 'symbolic-ref -q --short HEAD': 'master', 'remote get-url origin': 'https://github.com/scragnog/HOT-Step-CPP.git' };

test('clean master checkout: git kind, no build-info, no blockers beyond relation', () => {
  const id = detectInstall(base(gitFake(cleanGit)));
  assert.equal(id.kind, 'git');
  assert.equal(id.sha, SHA_A);
  assert.equal(id.variant, null);
  assert.deepEqual(applyBlockers(id, 'behind', 'win32'), []);
});

test('dirty, detached and foreign-origin checkouts report blockers', () => {
  const dirty = detectInstall(base(gitFake(cleanGit), {}, { dirty: () => true }));
  assert.deepEqual(applyBlockers(dirty, 'behind', 'win32'), ['The checkout has local changes']);
  const detached = detectInstall(base(gitFake({ ...cleanGit, 'symbolic-ref -q --short HEAD': new Error('x') })));
  assert.equal(detached.git?.branch, null);
  assert.deepEqual(applyBlockers(detached, 'behind', 'win32'), ['HEAD is detached']);
  const foreign = detectInstall(base(gitFake({ ...cleanGit, 'remote get-url origin': 'https://github.com/someone/else.git' })));
  assert.match(applyBlockers(foreign, 'behind', 'win32').join(), /origin/);
  assert.match(applyBlockers(detached, 'diverged', 'win32').join(), /diverged/);
});

test('a git root elsewhere (app folder inside another repo) is not a checkout', () => {
  const id = detectInstall(base(gitFake({ ...cleanGit, 'rev-parse --show-toplevel': process.cwd() + '-other' }), {}, { portable: true }));
  assert.equal(id.kind, 'portable');
  assert.equal(id.git, null);
});

test('portable without build-info is notice-only and never guesses a variant', () => {
  const id = detectInstall(base(() => { throw new Error('not a repo'); }, {}, { portable: true, version: '1.2.0' }));
  assert.equal(id.kind, 'portable');
  assert.equal(id.variant, null);
  assert.match(applyBlockers(id, 'unknown', 'win32')[0], /predates updater/);
});

test('build-info.json is validated', () => {
  const good = { format: 1, tag: 'v1.2.0', sha: SHA_A, channel: 'release', os: 'windows', arch: 'x64', variant: 'cuda12-volta', engineSha: SHA_B, updaterProtocol: 0 };
  assert.equal(parseBuildInfo(JSON.stringify(good)).info?.variant, 'cuda12-volta');
  assert.equal(parseBuildInfo(JSON.stringify({ ...good, variant: 'cuda13' })).info, null);
  assert.equal(parseBuildInfo(JSON.stringify({ ...good, sha: 'abc' })).info, null);
  assert.equal(parseBuildInfo(JSON.stringify({ ...good, format: 2 })).info, null);
  assert.equal(parseBuildInfo('{nope').info, null);
  assert.deepEqual(parseBuildInfo(null), { info: null, problem: null });
  const id = detectInstall(base(() => { throw new Error('x'); }, { 'build-info.json': JSON.stringify(good) }, { portable: true }));
  assert.equal(id.variant, 'cuda12-volta');
  assert.equal(id.sha, SHA_A);
  assert.deepEqual(applyBlockers(id, 'unknown', 'win32'), ['This build has no updater support yet']);
  assert.match(applyBlockers(id, 'unknown', 'linux').join(), /different operating system/);
});

test('gitRelation maps merge-base exit codes; a missing object is unknown, not diverged', () => {
  const git = { head: SHA_A, branch: 'master', dirty: false, originOk: true };
  const exit = (status: number) => Object.assign(new Error('exit'), { status });
  assert.equal(gitRelation(git, SHA_A, () => { throw new Error('unused'); }), 'up-to-date');
  assert.equal(gitRelation(git, SHA_B, () => ''), 'behind');
  assert.equal(gitRelation(git, SHA_B, a => { if (a[2] === SHA_A) throw exit(1); return ''; }), 'ahead');
  assert.equal(gitRelation(git, SHA_B, () => { throw exit(1); }), 'diverged');
  assert.equal(gitRelation(git, SHA_B, () => { throw exit(128); }), 'unknown');
  assert.equal(gitRelation(null, SHA_B, () => ''), 'unknown');
});

test('a held Release fetch does not drop a Master check', async () => {
  let release!: (r: ReturnType<typeof ok>) => void;
  const calls: string[] = [];
  const s = setup([]);
  s.deps.fetch = (url) => {
    calls.push(url);
    return url.endsWith('/releases/latest') ? new Promise(r => { release = r; }) : Promise.resolve(ok(REF(SHA_B)));
  };
  const c = createChecker(s.deps);
  const startup = c.check({ manual: false });
  const master = await c.check({ manual: true, channel: 'master' });
  assert.equal(calls.length, 2);
  assert.match(calls[1], /git\/ref\/heads\/master$/);
  assert.equal(master.latest.master?.sha, SHA_B);
  release(ok(REL('v1.0.0')));
  await startup;
});

test('origin guard: own origin and the Vite dev proxy pass, other sites do not', () => {
  assert.equal(trustedOrigin(undefined, '127.0.0.1:3001'), true);
  assert.equal(trustedOrigin('http://localhost:3001', 'localhost:3001'), true);
  assert.equal(trustedOrigin('http://localhost:3000', '127.0.0.1:3001'), true);
  assert.equal(trustedOrigin('http://127.0.0.1:3000', '127.0.0.1:3001'), true);
  assert.equal(trustedOrigin('https://evil.example', '127.0.0.1:3001'), false);
  assert.equal(trustedOrigin('http://localhost:8080', '127.0.0.1:3001'), false);
  assert.equal(trustedOrigin('http://evil.example:3000', '127.0.0.1:3001'), false);
  assert.equal(trustedOrigin('http://localhost:3000', 'evil.example'), false);
  assert.equal(trustedOrigin('null', '127.0.0.1:3001'), false);
});
