#!/usr/bin/env node
// Read-only observation. Start this while independently submitted jobs run.
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';

const exec = promisify(execFile);
const UUID = /^(?:GPU|MIG)-[a-f0-9-]{8,80}$/i;
const workerId = v => typeof v === 'string' && /^w\d{1,6}$/.test(v) ? v : null;
const uuid = v => typeof v === 'string' && UUID.test(v) ? v.toLowerCase() : null;
const integer = v => Number.isSafeInteger(v) && v > 0 ? v : null;
const states = new Set(['pending', 'running', 'lm_running', 'synth_running', 'saving', 'succeeded', 'failed', 'cancelled', 'interrupted']);
const terminal = s => ['succeeded', 'failed', 'cancelled', 'interrupted'].includes(s);
const active = j => ['running', 'lm_running', 'synth_running'].includes(j.status) && j.engineJob && j.workerId;

function csv(raw, kind) {
  if (typeof raw !== 'string') return null;
  const rows = [];
  for (const line of raw.trim().split(/\r?\n/).filter(Boolean)) {
    const parts = line.split(',').map(s => s.trim());
    if (parts.length !== 2) return null;
    const device = uuid(parts[kind === 'gpus' ? 1 : 0]);
    const numberText = parts[kind === 'gpus' ? 0 : 1];
    const number = /^\d+$/.test(numberText) ? Number(numberText) : NaN;
    if (!device || !Number.isSafeInteger(number) || number < (kind === 'gpus' ? 0 : 1)) return null;
    rows.push(kind === 'gpus' ? { index: number, uuid: device } : { uuid: device, pid: number });
  }
  return rows;
}

// Whitelist every field: never serialize API results, free-form errors or paths.
function snapshot(atMs, rawWorkers, rawJobs, gpuCsv, processCsv) {
  const list = Array.isArray(rawWorkers) ? rawWorkers : rawWorkers?.workers;
  return {
    atMs,
    workers: Array.isArray(list) ? list.map(w => ({
      id: workerId(w?.id), gpuUuid: uuid(w?.gpuUuid), affinity: w?.gpuUuid == null ? 'auto' : 'assigned',
      pid: integer(w?.pid), port: integer(w?.port), ready: w?.ready === true,
      crashSeq: Number.isSafeInteger(w?.crashSeq) && w.crashSeq >= 0 ? w.crashSeq : null,
    })) : null,
    jobs: rawJobs.map((j, i) => ({
      id: `j${i}`, workerId: workerId(j?.workerId),
      status: states.has(j?.status) ? j.status : 'unknown',
      engineJob: typeof j?.ace_job_id === 'string' && j.ace_job_id.length > 0,
    })),
    gpus: csv(gpuCsv, 'gpus'), processes: csv(processCsv, 'processes'),
  };
}

function affinity(s, w) {
  if (!w || w.affinity === 'auto') return 'no-affinity-claim';
  if (!w.id || !w.pid || !w.port || w.port > 65535 || !w.gpuUuid || !w.ready || w.crashSeq === null
      || !s.gpus?.length || !s.processes) return 'missing-telemetry';
  if (!s.gpus.some(g => g.uuid === w.gpuUuid)) return 'missing-device';
  const devices = s.processes.filter(p => p.pid === w.pid).map(p => p.uuid);
  if (!devices.length) return 'missing-process';
  if (devices.some(d => d !== w.gpuUuid)) return 'wrong-device';
  return 'matched';
}

function evaluate(samples) {
  const failures = new Set();
  const gaps = new Set();
  const intervals = [];
  const overlaps = [];
  for (const s of samples) {
    if (!s.workers?.length) gaps.add('worker-status-unavailable');
    if (!s.gpus?.length || !s.processes) gaps.add('device-telemetry-unavailable');
    for (const w of s.workers ?? []) {
      const result = affinity(s, w);
      if (result === 'wrong-device') failures.add('worker-on-wrong-device');
      else if (result !== 'matched') gaps.add(result);
    }
    for (const j of s.jobs) {
      if (['failed', 'cancelled', 'interrupted'].includes(j.status)) failures.add('job-unsuccessful');
      if (j.status === 'unknown') gaps.add('job-status-unavailable');
      if (active({ ...j, workerId: 'w0' }) && !j.workerId) gaps.add('job-worker-unavailable');
    }
  }
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1], b = samples[i];
    if (b.atMs <= a.atMs) continue;
    const simultaneous = [];
    for (const j of a.jobs.filter(active)) {
      const end = b.jobs.find(k => k.id === j.id);
      if (!end || !active(end) || end.workerId !== j.workerId) continue;
      const w = a.workers?.find(w => w.id === j.workerId);
      const next = b.workers?.find(w => w.id === j.workerId);
      if (!w || !next || w.pid !== next.pid || w.gpuUuid !== next.gpuUuid || w.port !== next.port
          || w.crashSeq !== next.crashSeq) { gaps.add('worker-changed'); continue; }
      const item = { job: j.id, worker: w.id, pid: w.pid, gpuUuid: w.gpuUuid,
        fromMs: a.atMs, toMs: b.atMs,
        affinityVerified: affinity(a, w) === 'matched' && affinity(b, next) === 'matched' };
      intervals.push(item);
      if (item.affinityVerified) simultaneous.push(item);
    }
    for (let x = 0; x < simultaneous.length; x++) for (let y = x + 1; y < simultaneous.length; y++) {
      const one = simultaneous[x], two = simultaneous[y];
      if (one.worker !== two.worker && one.pid !== two.pid && one.gpuUuid !== two.gpuUuid) {
        overlaps.push({ jobs: [one.job, two.job], fromMs: a.atMs, toMs: b.atMs });
      }
    }
  }
  const outcomes = samples.at(-1)?.jobs.map(j => ({ job: j.id, status: j.status })) ?? [];
  if (outcomes.length < 2) gaps.add('need-at-least-two-jobs');
  if (outcomes.some(j => j.status !== 'succeeded')) gaps.add('successful-outcomes-not-observed');
  if (!overlaps.length) gaps.add('no-verified-overlap');
  return { verdict: failures.size ? 'fail' : gaps.size ? 'inconclusive' : 'pass',
    reasons: [...failures, ...gaps], intervals, overlaps, outcomes };
}

function options(args) {
  const out = { base: 'http://localhost:3000', seconds: 60, intervalMs: 1000,
    output: 'multi-gpu-receipt.json', jobs: [] };
  for (let i = 0; i < args.length; i++) {
    const flag = args[i];
    if (flag === '--self-test' || flag === '--help') { out[flag.slice(2)] = true; continue; }
    if (!['--base', '--seconds', '--interval-ms', '--output', '--job'].includes(flag)
        || !args[i + 1] || args[i + 1].startsWith('--')) throw new Error('Invalid option; use --help.');
    const value = args[++i];
    if (flag === '--job') out.jobs.push(value);
    else out[({ '--interval-ms': 'intervalMs' })[flag] ?? flag.slice(2)] = value;
  }
  out.seconds = Number(out.seconds); out.intervalMs = Number(out.intervalMs);
  if (!Number.isFinite(out.seconds) || out.seconds < 1 || out.seconds > 3600
      || !Number.isFinite(out.intervalMs) || out.intervalMs < 250 || out.intervalMs > 60000
      || out.jobs.length > 32 || new Set(out.jobs).size !== out.jobs.length) throw new Error('Invalid observation limits.');
  const url = new URL(out.base);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash)
    throw new Error('Invalid server URL.');
  out.base = url.origin;
  return out;
}

async function observe(o) {
  const started = performance.now();
  let stop = false;
  const interrupt = () => { stop = true; };
  process.on('SIGINT', interrupt);
  const headers = {};
  async function get(path) {
    try {
      const res = await fetch(o.base + path, { headers, signal: AbortSignal.timeout(5000), redirect: 'error' });
      return res.ok ? await res.json() : null;
    } catch { return null; }
  }
  async function smi(query) {
    try {
      return (await exec('nvidia-smi', [query, '--format=csv,noheader,nounits'],
        { timeout: 5000, maxBuffer: 1024 * 1024, windowsHide: true })).stdout;
    } catch { return null; }
  }
  const samples = [];
  try {
    const auth = await get('/api/auth/auto');
    if (typeof auth?.token === 'string') headers.Authorization = `Bearer ${auth.token}`;
    do {
      const [workers, jobs, gpus, processes] = await Promise.all([
        get('/api/settings/engine-workers'),
        Promise.all(o.jobs.map(id => get(`/api/generate/status/${encodeURIComponent(id)}`))),
        smi('--query-gpu=index,uuid'), smi('--query-compute-apps=gpu_uuid,pid'),
      ]);
      samples.push(snapshot(Math.round(performance.now() - started), workers, jobs, gpus, processes));
      if (jobs.length && jobs.every(j => terminal(j?.status))) break;
      const remaining = o.seconds * 1000 - (performance.now() - started);
      if (stop || remaining <= 0) break;
      await new Promise(resolve => setTimeout(resolve, Math.min(o.intervalMs, remaining)));
    } while (!stop && performance.now() - started < o.seconds * 1000);
  } finally { process.off('SIGINT', interrupt); }
  const receipt = { schema: 1, observation: 'Sampled job activity and CUDA process residency; not a throughput or kernel-execution measurement.',
    ...evaluate(samples), samples };
  await writeFile(o.output, JSON.stringify(receipt, null, 2) + '\n', { flag: 'wx' });
  console.log(`MULTI-GPU: ${receipt.verdict.toUpperCase()} (${receipt.reasons.join(', ') || 'verified sampled overlap and successful jobs'})`);
  console.log('Receipt written. Job aliases j0, j1, ... follow --job argument order.');
  return receipt.verdict === 'pass' ? 0 : receipt.verdict === 'fail' ? 1 : 2;
}

function selfTest() {
  const a = 'GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', b = 'GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb';
  const workers = [a, b].map((gpuUuid, i) => ({ id: `w${i}`, gpuUuid, pid: 100 + i, port: 8085 + i, ready: true, crashSeq: 0 }));
  const jobs = workers.map(w => ({ status: 'synth_running', workerId: w.id, ace_job_id: 'engine-job' }));
  const gpuCsv = `0, ${a}\r\n1, ${b}\r\n`, processCsv = `${a}, 100\r\n${b}, 101\r\n`;
  const sample = (at, js = jobs) => snapshot(at, { workers }, js, gpuCsv, processCsv);
  const base = () => [sample(0), sample(1000), sample(2000, jobs.map(j => ({ ...j, status: 'succeeded' })))];
  let count = 0;
  const check = (name, mutate, expected) => { const s = base(); mutate(s); assert.equal(evaluate(s).verdict, expected, name); count++; };
  check('two GPUs and successful jobs', () => {}, 'pass');
  check('wrong device', s => { s[0].processes[0].uuid = b.toLowerCase(); }, 'fail');
  check('extra wrong-device context', s => { s[0].processes.push({ pid: 100, uuid: b.toLowerCase() }); }, 'fail');
  check('failed job', s => { s[2].jobs[0].status = 'failed'; }, 'fail');
  check('no telemetry', s => { s.forEach(x => { x.processes = null; }); }, 'inconclusive');
  check('missing PID', s => { s.forEach(x => { x.processes = []; }); }, 'inconclusive');
  check('missing GPU', s => { s.forEach(x => { x.gpus = []; }); }, 'inconclusive');
  check('no worker route', s => { s.forEach(x => { x.workers = null; }); }, 'inconclusive');
  check('Auto has no claim', s => { s.forEach(x => { x.workers[0].affinity = 'auto'; x.workers[0].gpuUuid = null; }); }, 'inconclusive');
  check('missing job ownership', s => { s.forEach(x => { x.jobs[0].workerId = null; }); }, 'inconclusive');
  check('no overlap', s => { s[0].jobs[1].status = 'pending'; }, 'inconclusive');
  check('pending is not work', s => { s.forEach(x => x.jobs.forEach(j => { j.status = 'pending'; })); }, 'inconclusive');
  check('saving is not GPU work', s => { s[0].jobs[0].status = 'saving'; }, 'inconclusive');
  check('same worker', s => { s.forEach(x => { x.jobs[1].workerId = 'w0'; }); }, 'inconclusive');
  check('two workers on one GPU', s => { s.forEach(x => {
    x.workers[1].gpuUuid = a.toLowerCase(); x.processes[1].uuid = a.toLowerCase();
  }); }, 'inconclusive');
  check('no engine job accepted', s => { s.forEach(x => { x.jobs[0].engineJob = false; }); }, 'inconclusive');
  check('worker not ready', s => { s.forEach(x => { x.workers[0].ready = false; }); }, 'inconclusive');
  check('telemetry missing at one boundary', s => { s[1].processes = null; }, 'inconclusive');
  check('worker restart', s => { s[1].workers[0].crashSeq++; }, 'inconclusive');
  check('unfinished jobs', s => { s.pop(); }, 'inconclusive');
  check('no jobs', s => { s.forEach(x => { x.jobs = []; }); }, 'inconclusive');
  assert.equal(csv('N/A, N/A', 'processes'), null); count++;
  const secret = 'private-path-name-song-text';
  const safe = snapshot(0, [{ ...workers[0], url: secret, logFile: secret, status: secret, device: secret }],
    [{ ...jobs[0], result: secret, error: secret, stage: secret, jobId: secret }], gpuCsv, processCsv);
  assert.ok(!JSON.stringify(safe).includes(secret)); count++;
  assert.throws(() => options(['--seconds', 'NaN'])); count++;
  console.log(`MULTI-GPU SELF-TEST: PASS (${count} cases)`);
}

async function main() {
  const o = options(process.argv.slice(2));
  if (o['self-test']) { selfTest(); return 0; }
  if (o.help) {
    console.log(`Usage: node tools/multi-gpu-check.mjs --job JOB_ID --job JOB_ID --seconds 120 --output receipt.json
Options: --base http://localhost:3000 --interval-ms 1000 --self-test --help
Read-only: submit jobs separately, then observe them while running. No installs.
Needs GET /api/settings/engine-workers (array or {workers}) and job status with
workerId, status and ace_job_id. Missing fields or NVIDIA telemetry are inconclusive.
Intervals bound consecutive active samples, not exact job start/end timestamps.
Receipt omits URLs, paths, names, prompts, results and raw errors; jobs use aliases.
An existing output file is never overwritten. Exit: 0 pass, 1 fail/error, 2 inconclusive.`);
    return 0;
  }
  return observe(o);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().then(code => { process.exitCode = code; }).catch(() => {
    console.error('MULTI-GPU: ERROR. Check options and that the output file is new and writable.');
    process.exitCode = 1;
  });
}
