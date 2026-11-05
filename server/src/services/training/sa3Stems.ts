/** SA3 preparation only. Call from the serialized GPU lane with the engine ready.
 * Originals are read-only. Cache entries are immutable and keyed by audio/model
 * bytes and settings, including the post-separation 44.1 kHz stereo crop.
 * Pass the actual level-5 model files, not catalogue names, as modelFiles.
 */
import fs from 'node:fs';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { setTimeout as delay } from 'node:timers/promises';
import { parseFile } from 'music-metadata';

export const SA3_AUDIO = Object.freeze({ level: 5, sampleRate: 44100, channels: 2, maxSeconds: 384, version: 1 });
export interface Sa3AudioInfo { duration: number; sampleRate: number; channels: number }
export interface Sa3StemDependencies {
  fetch: typeof fetch;
  convert(input: string, output: string, signal?: AbortSignal, crop?: boolean): Promise<void>;
  inspect(file: string): Promise<Sa3AudioInfo>;
}
export interface Sa3Instrumental {
  audioPath: string;
  fingerprint: string;
  sourceHash: string;
  modelHashes: string[];
  audioHash: string;
  seconds_total: number;
  cached: boolean;
}

export async function sa3Hash(file: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of fs.createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}

export async function inspectSa3Audio(file: string): Promise<Sa3AudioInfo> {
  const { format } = await parseFile(file, { duration: true, skipCovers: true });
  return { duration: format.duration ?? 0, sampleRate: format.sampleRate ?? 0, channels: format.numberOfChannels ?? 0 };
}

export function validateSa3Audio(info: Sa3AudioInfo): void {
  if (!Number.isFinite(info.duration) || info.duration <= 0 || info.duration > SA3_AUDIO.maxSeconds
    || info.sampleRate !== SA3_AUDIO.sampleRate || info.channels !== SA3_AUDIO.channels) {
    throw new Error('SA3 audio must be 44.1 kHz stereo with a measured duration in (0, 384] seconds');
  }
}

async function convertAudio(input: string, output: string, signal?: AbortSignal, crop = true): Promise<void> {
  const { getFFmpegPath } = await import('../../config.js');
  const ffmpeg = getFFmpegPath();
  if (!ffmpeg) throw new Error('ffmpeg is required for SA3 dataset preparation');
  await promisify(execFile)(ffmpeg, ['-y', '-v', 'error', '-i', input, '-vn',
    ...(crop ? ['-t', String(SA3_AUDIO.maxSeconds)] : []), '-ar', String(SA3_AUDIO.sampleRate),
    '-ac', String(SA3_AUDIO.channels), '-c:a', 'pcm_s16le', '-f', 'wav', output],
  { signal, timeout: 300_000, maxBuffer: 1024 * 1024, windowsHide: true });
}

/** Reject a derived directory inside the source tree, including symlink aliases. */
export function assertSa3DerivedDir(dir: string, source: string): void {
  let existing = path.resolve(dir);
  const suffix: string[] = [];
  while (!fs.existsSync(existing)) {
    suffix.unshift(path.basename(existing));
    existing = path.dirname(existing);
  }
  const resolved = path.join(fs.realpathSync(existing), ...suffix);
  const parent = path.dirname(fs.realpathSync(source));
  const rel = path.relative(parent, resolved);
  if (!rel || (!rel.startsWith(`..${path.sep}`) && rel !== '..' && !path.isAbsolute(rel))) {
    throw new Error('SA3 derived folders must be outside the source audio directory');
  }
}

/** Required model fingerprints invalidate stems after a weight replacement.
 * force bypasses reuse, but never deletes a previous good entry on failure.
 * Tests inject every engine/conversion dependency; importing this module does
 * not load configuration, open the database, or contact the engine.
 */
export async function prepareSa3Instrumental(opts: {
  source: string;
  cacheDir: string;
  modelFiles: string[];
  engineUrl: string;
  signal?: AbortSignal;
  force?: boolean;
  onProgress?: (fraction: number) => void;
  pollMs?: number;
  timeoutMs?: number;
}, injected: Partial<Sa3StemDependencies> = {}): Promise<Sa3Instrumental> {
  const deps: Sa3StemDependencies = { fetch: globalThis.fetch, convert: convertAudio, inspect: inspectSa3Audio, ...injected };
  opts.signal?.throwIfAborted();
  assertSa3DerivedDir(opts.cacheDir, opts.source);
  if (!opts.modelFiles.length) throw new Error('SA3 preparation requires the actual SuperSep level-5 model files');
  const sourceHash = await sa3Hash(opts.source);
  const modelHashes = await Promise.all(opts.modelFiles.map(sa3Hash));
  const fingerprint = createHash('sha256').update(JSON.stringify({ sourceHash, modelHashes, settings: SA3_AUDIO })).digest('hex');
  const entry = path.join(opts.cacheDir, fingerprint);
  const audioPath = path.join(entry, 'instrumental.wav');
  if (!opts.force) {
    try {
      const receipt = JSON.parse(await fs.promises.readFile(path.join(entry, 'manifest.json'), 'utf8')) as Sa3Instrumental;
      const info = await deps.inspect(audioPath);
      validateSa3Audio(info);
      if (receipt.fingerprint === fingerprint && receipt.sourceHash === sourceHash
        && JSON.stringify(receipt.modelHashes) === JSON.stringify(modelHashes)
        && receipt.seconds_total === info.duration && receipt.audioHash === await sa3Hash(audioPath)) {
        opts.signal?.throwIfAborted();
        return { ...receipt, audioPath, cached: true };
      }
    } catch (error) {
      opts.signal?.throwIfAborted(); // Invalid or partial cache entries are recomputed.
    }
  }
  opts.signal?.throwIfAborted();
  await fs.promises.mkdir(opts.cacheDir, { recursive: true });
  const scratch = await fs.promises.mkdtemp(path.join(opts.cacheDir, '.sa3-'));
  let id: string | undefined;
  let failure: unknown;
  const signal = AbortSignal.any([...(opts.signal ? [opts.signal] : []), AbortSignal.timeout(opts.timeoutMs ?? 7_200_000)]);
  const base = opts.engineUrl.replace(/\/$/, '');
  async function request(endpoint: string, init: RequestInit = {}, cleanup = false): Promise<Response> {
    const res = await deps.fetch(`${base}/supersep/${endpoint}`, {
      ...init, signal: cleanup ? AbortSignal.timeout(15_000) : AbortSignal.any([signal, AbortSignal.timeout(120_000)]),
    });
    if (!res.ok) throw new Error(`SuperSep ${endpoint}: ${res.status} ${await res.text()}`);
    return res;
  }
  try {
    // A fresh conversion avoids the basename-only legacy input cache. This is
    // also needed for formats the engine cannot decode, such as FLAC.
    const input = path.join(scratch, 'input.wav');
    await deps.convert(opts.source, input, signal, false);
    signal.throwIfAborted();
    // Collect the job id before observing caller cancellation so release can
    // reclaim a job accepted while the POST was in flight.
    const started = await deps.fetch(`${base}/supersep/separate?level=${SA3_AUDIO.level}`, {
      method: 'POST', body: await fs.promises.readFile(input),
      headers: { 'Content-Type': 'application/octet-stream' }, signal: AbortSignal.timeout(120_000),
    });
    if (!started.ok) throw new Error(`SuperSep separate: ${started.status} ${await started.text()}`);
    const job = await started.json() as { id?: string };
    if (typeof job.id !== 'string' || !job.id) throw new Error('SuperSep returned no job id');
    id = job.id;
    signal.throwIfAborted();
    const query = `id=${encodeURIComponent(id)}`;
    for (;;) {
      const p = await (await request(`progress?${query}`)).json() as { status: string; progress?: number; error?: string };
      if (typeof p.progress === 'number') opts.onProgress?.(p.progress);
      if (p.status === 'done') break;
      if (p.status === 'failed' || p.status === 'cancelled') throw new Error(p.error || `SuperSep ${p.status}`);
      await delay(opts.pollMs ?? 500, undefined, { signal });
    }
    const { stems } = await (await request(`result?${query}`)).json() as { stems: Array<{ name: string; index: number }> };
    const instrumental = stems.find(s => /^(?:\d+[_ ]+)?instrumental$/i.test(s.name.trim()));
    if (!instrumental || !Number.isInteger(instrumental.index)) throw new Error('SuperSep returned no instrumental stem');
    const raw = path.join(scratch, 'separated.wav');
    await fs.promises.writeFile(raw, Buffer.from(await (await request(`serve?${query}&stem=${instrumental.index}`)).arrayBuffer()));
    const output = path.join(scratch, 'instrumental.wav');
    await deps.convert(raw, output, signal); // Resample after separation, even when the input was already 44.1 kHz.
    const info = await deps.inspect(output);
    validateSa3Audio(info);
    signal.throwIfAborted();
    if (await sa3Hash(opts.source) !== sourceHash
      || JSON.stringify(await Promise.all(opts.modelFiles.map(sa3Hash))) !== JSON.stringify(modelHashes)) {
      throw new Error('Source audio or SuperSep weights changed during preparation');
    }
    const result: Sa3Instrumental = { audioPath, fingerprint, sourceHash, modelHashes,
      audioHash: await sa3Hash(output), seconds_total: info.duration, cached: false };
    // Publish a unique audio file first and the receipt last. A failed/cancelled
    // conversion cannot make an incomplete cache entry look reusable.
    await fs.promises.mkdir(entry, { recursive: true });
    const receiptTmp = path.join(entry, `${randomUUID()}.json`);
    try {
      await fs.promises.writeFile(receiptTmp, JSON.stringify(result, null, 2));
      signal.throwIfAborted();
      await fs.promises.rename(output, audioPath);
      await fs.promises.rename(receiptTmp, path.join(entry, 'manifest.json'));
    } finally {
      await fs.promises.rm(receiptTmp, { force: true });
    }
    return result;
  } catch (error) {
    failure = error;
    throw error;
  } finally {
    try {
      if (id) await request(`release?id=${encodeURIComponent(id)}`, { method: 'POST' }, true);
    } catch (error) {
      if (!failure) throw error;
    } finally {
      await fs.promises.rm(scratch, { recursive: true, force: true });
    }
  }
}
