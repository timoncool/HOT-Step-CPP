// Installed identity: build-info.json plus bounded git calls. Pure over its
// deps so tests can fake the filesystem and git.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { BUILD_VARIANTS } from '../../contracts/updates.js';
import type { BuildInfo, GitInstall, GitRelation, InstalledIdentity } from '../../contracts/updates.js';

export const REPO = 'scragnog/HOT-Step-CPP';

export interface InstallDeps {
  root: string;
  portable: boolean;
  version: string;
  readFile(file: string): string | null;
  /** Throws on failure or timeout; the error's `status` is git's exit code. */
  git(args: string[]): string;
  dirty(): boolean;
}

export const realGit = (root: string) => (args: string[]) =>
  execFileSync('git', args, { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], timeout: 15000, windowsHide: true }).trim();

export const realReadFile = (file: string): string | null => { try { return fs.readFileSync(file, 'utf-8'); } catch { return null; } };

const OS_NAME: Partial<Record<NodeJS.Platform, BuildInfo['os']>> = { win32: 'windows', linux: 'linux', darwin: 'macos' };
const SHA = /^[0-9a-f]{40}$/i;

/** Returns the manifest, or why it was rejected. A missing file is not a problem (null, null). */
export function parseBuildInfo(raw: string | null): { info: BuildInfo | null; problem: string | null } {
  if (raw === null) return { info: null, problem: null };
  let j: any;
  try { j = JSON.parse(raw); } catch { return { info: null, problem: 'build-info.json is not valid JSON' }; }
  if (!j || j.format !== 1) return { info: null, problem: 'build-info.json has an unknown format' };
  const bad = (k: string) => ({ info: null, problem: `build-info.json field "${k}" is missing or invalid` });
  if (typeof j.sha !== 'string' || !SHA.test(j.sha)) return bad('sha');
  if (j.tag !== null && typeof j.tag !== 'string') return bad('tag');
  if (j.channel !== 'release' && j.channel !== 'master') return bad('channel');
  if (j.os !== 'windows' && j.os !== 'linux' && j.os !== 'macos') return bad('os');
  if (typeof j.arch !== 'string' || !j.arch) return bad('arch');
  if (!BUILD_VARIANTS.includes(j.variant)) return bad('variant');
  if (typeof j.engineSha !== 'string' || !j.engineSha) return bad('engineSha');
  if (!Number.isInteger(j.updaterProtocol) || j.updaterProtocol < 0) return bad('updaterProtocol');
  return { info: j as BuildInfo, problem: null };
}

const sameDir = (a: string, b: string) => {
  const n = (p: string) => { let r = path.resolve(p); try { r = fs.realpathSync(r); } catch { /* keep resolved */ } return process.platform === 'win32' ? r.toLowerCase() : r; };
  return n(a) === n(b);
};

export const isRepoOrigin = (url: string) =>
  new RegExp(`github\\.com[:/]${REPO.replace('/', '\\/')}(\\.git)?/?$`, 'i').test(url.trim());

/** null = the app folder is not the root of a git checkout. */
export function detectGit(d: InstallDeps): GitInstall | null {
  try {
    if (!sameDir(d.git(['rev-parse', '--show-toplevel']), d.root)) return null;
    const head = d.git(['rev-parse', 'HEAD']);
    if (!SHA.test(head)) return null;
    let branch: string | null = null;
    try { branch = d.git(['symbolic-ref', '-q', '--short', 'HEAD']) || null; } catch { /* detached */ }
    let originOk = false;
    try { originOk = isRepoOrigin(d.git(['remote', 'get-url', 'origin'])); } catch { /* no origin */ }
    return { head, branch, dirty: d.dirty(), originOk };
  } catch { return null; }
}

export function detectInstall(d: InstallDeps): InstalledIdentity {
  const { info, problem } = parseBuildInfo(d.readFile(path.join(d.root, 'build-info.json')));
  const git = detectGit(d);
  // A checkout is never a portable install, whatever launched it.
  const kind = git ? 'git' : d.portable ? 'portable' : 'unknown';
  return {
    kind,
    version: d.version,
    tag: info?.tag ?? null,
    sha: info?.sha ?? git?.head ?? null,
    variant: info?.variant ?? null,
    buildInfo: info,
    buildInfoProblem: problem,
    git,
  };
}

/** How the checkout's HEAD relates to the master tip; 'unknown' when the tip isn't in the local object store (no fetch is run). */
export function gitRelation(git: GitInstall | null, masterSha: string | null, run: (args: string[]) => string): GitRelation {
  if (!git || !masterSha) return 'unknown';
  if (git.head.toLowerCase() === masterSha.toLowerCase()) return 'up-to-date';
  // true/false = git's answer (exit 0/1); null = it couldn't tell (missing object, timeout).
  const isAncestor = (a: string, b: string): boolean | null => {
    try { run(['merge-base', '--is-ancestor', a, b]); return true; } catch (e: any) { return e?.status === 1 ? false : null; }
  };
  const behind = isAncestor(git.head, masterSha);
  if (behind) return 'behind';
  const ahead = isAncestor(masterSha, git.head);
  if (ahead) return 'ahead';
  return behind === false && ahead === false ? 'diverged' : 'unknown';
}

/** What would also stop an automatic update on this install (the apply phase is not built yet). */
export function applyBlockers(id: InstalledIdentity, relation: GitRelation, platform: NodeJS.Platform): string[] {
  const out: string[] = [];
  if (id.kind === 'portable') {
    if (!id.buildInfo) out.push(id.buildInfoProblem ?? 'No build-info.json: this package predates updater support, so one manual upgrade is needed');
    else {
      if (id.buildInfo.os !== OS_NAME[platform]) out.push('build-info.json is for a different operating system');
      if (id.buildInfo.updaterProtocol < 1) out.push('This build has no updater support yet');
    }
  } else if (id.kind === 'git' && id.git) {
    if (id.git.dirty) out.push('The checkout has local changes');
    if (!id.git.branch) out.push('HEAD is detached');
    else if (id.git.branch !== 'master') out.push(`The checkout is on branch ${id.git.branch}, not master`);
    if (!id.git.originOk) out.push('origin is not the HOT-Step-CPP repository');
    if (relation === 'ahead') out.push('The checkout has commits master does not');
    else if (relation === 'diverged') out.push('The checkout has diverged from master');
    else if (relation === 'unknown') out.push('Could not tell how the checkout relates to master');
  } else out.push('Could not identify this installation');
  return out;
}
