// Windows local-master promotion. No fetch from the network and no push.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { readDevData, resolveSharedData } from './data-mode.mjs';

const source = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const upstream = 'https://github.com/scragnog/HOT-Step-CPP.git';
const { parse: parseEnv } = createRequire(new URL('../../server/package.json', import.meta.url))('dotenv');
export function run(command, args, cwd = source, options = {}) {
  const result = spawnSync(command, args, { cwd, encoding: 'utf8', timeout: 900_000, ...options });
  if (result.error || result.status !== 0) throw new Error(`${command} failed: ${result.error?.message ?? result.stderr ?? result.status}`);
  return result.stdout?.trim() ?? '';
}
const git = (cwd, ...args) => run('git', ['-C', cwd, ...args]);
const writeJson = (file, value) => { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, JSON.stringify(value, null, 2) + '\n'); };
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
export function inside(parent, child) {
  const rel = path.relative(path.resolve(parent), path.resolve(child));
  return rel !== '' && !rel.startsWith('..') && !path.isAbsolute(rel);
}
export function assertIdle(report) {
  if (!Array.isArray(report?.blockers) || report.pending || report.blockers.length) {
    throw new Error(`Production is busy or its idle state is unknown: ${JSON.stringify(report?.blockers)}`);
  }
}
export function assertScratchEnv(text, scratch) {
  const values = parseEnv(text);
  for (const key of ['DATA_DIR', 'TRAINING_DIR', 'ACESTEPCPP_ADAPTERS', 'LYRICS_EXPORT_DIR', 'MUSCRIPTOR_MODELS_DIR']) {
    if (!values[key]) continue;
    const resolved = path.resolve(scratch, values[key]);
    if (resolved !== path.resolve(scratch) && !inside(scratch, resolved)) {
      throw new Error(`Scratch .env redirects ${key} outside its data root.`);
    }
  }
}
export function parseOptions(args) {
  const values = {};
  const valued = ['--prod', '--scratch', '--data-root', '--dev-profile'];
  for (let i = 0; i < args.length; i++) {
    const flag = args[i];
    if (Object.hasOwn(values, flag)) throw new Error(`Repeated option: ${flag}`);
    if (valued.includes(flag)) {
      const value = args[++i];
      if (!value || value.startsWith('--')) throw new Error(`Missing value for ${flag}`);
      values[flag] = value;
    } else if (['--help', '--start', '--stop'].includes(flag)) values[flag] = true;
    else throw new Error(`Unknown option: ${flag}`);
  }
  if (values['--scratch'] && values['--data-root']) throw new Error('Choose exactly one explicit data mode.');
  if (values['--dev-profile'] && (!values['--data-root'] || !path.isAbsolute(values['--dev-profile']))) throw new Error('--dev-profile requires shared mode and an absolute profile path.');
  if ((values['--start'] || values['--stop']) && (values['--scratch'] || values['--data-root'] || values['--start'] && values['--stop'])) throw new Error('Start/stop uses the recorded mode. Promote to change data paths.');
  return values;
}
async function request(endpoint) {
  const response = await fetch(`http://127.0.0.1:3101${endpoint}`, { signal: AbortSignal.timeout(15_000) });
  if (!response.ok) throw new Error(`${endpoint}: HTTP ${response.status}`);
  return response.json();
}
function powershell(script, args = [], cwd = source) {
  return run('pwsh', ['-NoProfile', '-File', path.join(source, 'tools/prod', script), ...args], cwd);
}
function ownedServer(prod) {
  const statePath = path.join(prod, '.prod/process.json');
  if (!fs.existsSync(statePath)) return null;
  const saved = JSON.parse(fs.readFileSync(statePath, 'utf8').replace(/^\uFEFF/, ''));
  const proc = JSON.parse(powershell('process.ps1', ['-ProcessId', String(saved.pid)]));
  if (!proc) return null;
  if (!proc.command?.toLowerCase().includes(path.join(prod, 'server/src/index.ts').toLowerCase()) ||
      Math.abs(Date.parse(proc.started) - Date.parse(saved.started)) > 5) {
    throw new Error('Production PID identity changed. Refusing to stop it.');
  }
  return saved;
}
function ensurePortsFree() {
  powershell('process.ps1', ['-CheckPorts']);
}
function stop(prod) {
  if (!ownedServer(prod)) {
    // No saved live Node PID authorizes an HTTP shutdown. Reap only an
    // orphan engine by executable path; an unknown Node listener aborts.
    powershell('process.ps1', ['-CheckServerPort']);
    powershell('stop-checkout.ps1', ['-Root', prod, '-Port', '3101']);
    ensurePortsFree(); return;
  }
  const config = JSON.parse(fs.readFileSync(path.join(prod, '.prod/config.json')));
  powershell('stop-checkout.ps1', ['-Root', prod, '-Port', '3101', '-Token', config.token]);
  ensurePortsFree();
}
async function maintenance(prod, action) {
  const { token } = JSON.parse(fs.readFileSync(path.join(prod, '.prod/config.json')));
  const response = await fetch('http://127.0.0.1:3101/__prod/maintenance', {
    method: 'POST', signal: AbortSignal.timeout(20_000),
    headers: { 'x-hotstep-prod-token': token, 'x-hotstep-prod-action': action },
  });
  if (!response.ok) throw new Error(`Could not ${action} production admission: HTTP ${response.status}`);
}
async function start(prod, commit) {
  ensurePortsFree();
  console.log(run('pwsh', ['-NoProfile', '-File', path.join(prod, 'tools/detach.ps1'), 'tools/prod/run.bat'], prod));
  const deadline = Date.now() + 90_000;
  while (Date.now() < deadline) {
    try {
      const health = await request('/api/health');
      if (health.status === 'ok' && health.commit === commit && ownedServer(prod)) {
        console.log(`PROD HEALTH PASS commit=${commit} url=http://localhost:3101 engine=${health.aceServer.status}`);
        return;
      }
    } catch { /* Startup has a bounded readiness window. */ }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  throw new Error('Production failed readiness. See .prod/server.stdout.log and server.stderr.log.');
}
function pointer(profile, prod, root) {
  const key = crypto.createHash('sha256').update(path.resolve(prod).toLowerCase()).digest('hex').slice(0, 12);
  writeJson(path.join(profile, `data-location.dev-${key}.json`), { version: 1, root, pending: null, checkout: prod });
}
function copyEngine(stage, commit) {
  if (git(source, 'status', '--porcelain', '--', 'engine')) throw new Error('Engine source/submodule is dirty. Rebuild a clean engine first.');
  if (git(source, 'rev-parse', 'HEAD:engine') !== git(source, 'rev-parse', `${commit}:engine`)) throw new Error('Engine source differs from the selected commit.');
  const release = path.join(source, 'engine/build/Release');
  const exe = path.join(release, 'ace-server.exe');
  const builtAt = fs.statSync(exe).mtimeMs;
  // Reject stale binaries, including changed submodule inputs. The source tree
  // and every binary hash are recorded so a prebuilt snapshot is inspectable.
  const inputs = git(source, 'ls-files', '--recurse-submodules', 'engine').split('\n')
    .filter(f => /\.(?:h|hpp|c|cc|cpp|cu|cuh|cmake)$|CMakeLists\.txt$/i.test(f));
  for (const input of inputs) {
    if (fs.statSync(path.join(source, input)).mtimeMs > builtAt) throw new Error(`Engine is stale: ${input} is newer than ace-server.exe. Run dev-rebuild first.`);
  }
  const binaries = fs.readdirSync(release).filter(f => /\.(exe|dll)$/i.test(f));
  if (!binaries.includes('ace-server.exe') || !binaries.some(f => /\.dll$/i.test(f))) throw new Error('Incomplete engine output.');
  const dest = path.join(stage, 'engine/build/Release');
  fs.mkdirSync(dest, { recursive: true });
  const hashes = {};
  for (const name of binaries) {
    const from = path.join(release, name), to = path.join(dest, name);
    const before = hash(from);
    fs.copyFileSync(from, to);
    if (hash(to) !== before || hash(from) !== before) throw new Error(`Engine changed during copy: ${name}`);
    hashes[name] = before;
  }
  return { sourceTree: git(source, 'rev-parse', `${commit}:engine`), builtAt: new Date(builtAt).toISOString(), hashes };
}
async function main(args) {
  const options = parseOptions(args);
  if (args.includes('--help')) {
    console.log('node tools/prod/promote.mjs (--scratch <absolute-data-root> | --data-root <absolute-shared-root>) [--prod <clone-path>]\nShared mode reads the dev pointer and overrides. --dev-profile <absolute-profile> selects a stand-in dev profile.\n--stop stops the managed prod; --start starts its recorded artifact.');
    return;
  }
  const option = name => options[name];
  const prod = path.resolve(option('--prod') ?? 'D:/Ace-Step-Latest/hot-step-prod');
  if (!inside(path.dirname(source), prod) || inside(source, prod) || inside(prod, source) || prod === source) throw new Error('Prod must be a separate sibling clone.');
  if (process.env.HOTSTEP_ENGINE_LOCK_ROOT !== source) {
    console.log(powershell('engine-lock.ps1', ['-Promote', ...args]));
    return;
  }
  if (args.includes('--stop')) {
    if (ownedServer(prod)) {
      await maintenance(prod, 'pause');
      try { assertIdle(await request('/api/settings/data-folder')); stop(prod); }
      catch (error) { if (ownedServer(prod)) await maintenance(prod, 'resume'); throw error; }
    } else ensurePortsFree();
    console.log('Production stopped.'); return;
  }
  if (args.includes('--start')) {
    const config = JSON.parse(fs.readFileSync(path.join(prod, '.prod/config.json')));
    await start(prod, config.commit); return;
  }
  const scratchArg = option('--scratch');
  const sharedArg = option('--data-root');
  if (Boolean(scratchArg) === Boolean(sharedArg)) throw new Error('Choose exactly one explicit mode: --scratch or --data-root.');
  const mode = sharedArg ? 'shared' : 'scratch';
  const scratch = scratchArg ? path.resolve(scratchArg) : null;
  let data;
  let devProfile;
  if (mode === 'shared') {
    data = resolveSharedData(sharedArg, readDevData(option('--dev-profile')));
    devProfile = data.profile;
    data.profile = path.join(prod, '.prod/profile');
  } else {
    if (!path.isAbsolute(scratchArg)) throw new Error('An absolute --scratch data root is required.');
    // Scratch validation only authorizes disposable roots in this checkout.
    if (!inside(path.join(source, '.tmp'), scratch)) throw new Error('Scratch root must be under the dev checkout .tmp directory.');
    fs.mkdirSync(scratch, { recursive: true });
    const scratchEnv = path.join(scratch, 'settings/.env');
    if (fs.existsSync(scratchEnv)) assertScratchEnv(fs.readFileSync(scratchEnv, 'utf8'), scratch);
    data = { root: scratch, profile: path.join(prod, '.prod/scratch-profile'), library: path.join(scratch, 'library'),
      training: path.join(scratch, 'library/training'), adapters: path.join(scratch, 'adapters'), models: 'N:\\HOT-Step Data\\models' };
  }
  console.log(`PROD DATA mode=${mode}\n${JSON.stringify(data, null, 2)}`);
  if (git(source, 'branch', '--show-current') !== 'master') throw new Error('Source must be on master.');
  const commit = git(source, 'rev-parse', 'refs/heads/master');
  const existing = fs.existsSync(prod);
  if (existing) {
    if (!fs.existsSync(path.join(prod, '.prod/config.json')) || git(prod, 'branch', '--show-current') !== 'master' || git(prod, 'status', '--porcelain', '--untracked-files=no')) throw new Error('Prod must be a clean, managed master clone.');
    if (ownedServer(prod)) assertIdle(await request('/api/settings/data-folder'));
    else ensurePortsFree();
  } else ensurePortsFree();
  const id = crypto.randomUUID();
  const stage = `${prod}.staging-${id}`;
  const previous = `${prod}.previous-${id}`;
  console.log(`Preparing ${commit} at ${stage}`);
  git(source, 'clone', '--no-hardlinks', '--branch', 'master', source, stage);
  if (git(stage, 'rev-parse', 'HEAD') !== commit) throw new Error('Master moved during clone. Retry promotion.');
  git(stage, 'remote', 'set-url', 'origin', upstream);
  const npm = (cwd, ...commands) => {
    // npm.cmd needs cmd.exe on Windows. All arguments here are fixed constants.
    const result = spawnSync('cmd.exe', ['/d', '/c', `npm.cmd ${commands.join(' ')}`], { cwd, stdio: 'inherit', timeout: 900_000 });
    if (result.error || result.status !== 0) throw new Error(`npm ${commands.join(' ')} failed: ${result.error?.message ?? result.status}`);
  };
  for (const part of ['server', 'ui']) {
    npm(path.join(stage, part), 'ci', '--no-audit', '--no-fund');
    npm(path.join(stage, part), 'run', 'build');
  }
  const engine = copyEngine(stage, commit);
  const config = { commit, mode, data, devCheckout: source, devProfile, scratch, token: crypto.randomBytes(32).toString('hex'), models: data.models, ports: { server: 3101, engine: 8186 } };
  writeJson(path.join(stage, '.prod/config.json'), config);
  writeJson(path.join(stage, '.prod/artifact.json'), { commit, preparedAt: new Date().toISOString(), engine });
  fs.writeFileSync(path.join(stage, '.env'), 'SERVER_PORT=3101\nACESTEPCPP_PORT=8186\nACESTEPCPP_HOST=127.0.0.1\nACESTEPCPP_MODELS="N:\\HOT-Step Data\\models"\n');
  pointer(path.join(stage, mode === 'shared' ? '.prod/profile' : '.prod/scratch-profile'), prod, data.root);
  if (!fs.existsSync(path.join(stage, 'ui/dist/index.html')) || !fs.existsSync(path.join(stage, 'server/dist/index.js'))) throw new Error('Missing built UI or server.');
  // Recheck after preparation. A failed build leaves the live clone untouched.
  if (existing && ownedServer(prod)) {
    await maintenance(prod, 'pause');
    try { assertIdle(await request('/api/settings/data-folder')); }
    catch (error) { await maintenance(prod, 'resume'); throw error; }
  }
  stop(prod);
  let installed = false;
  try {
    if (existing) fs.renameSync(prod, previous);
    fs.renameSync(stage, prod); installed = true;
    await start(prod, commit);
    console.log(`PROMOTE PASS ${commit}\nArtifact: ${path.join(prod, '.prod/artifact.json')}\nPrevious clone retained: ${existing ? previous : 'none'}\nStop: node tools/prod/promote.mjs --stop`);
  } catch (error) {
    if (installed) {
      stop(prod);
      fs.renameSync(prod, `${prod}.failed-${id}`);
    }
    if (existing && fs.existsSync(previous)) {
      fs.renameSync(previous, prod);
      const prior = JSON.parse(fs.readFileSync(path.join(prod, '.prod/config.json')));
      await start(prod, prior.commit);
    }
    throw error;
  }
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch(error => { console.error(error.message); process.exitCode = 1; });
}
