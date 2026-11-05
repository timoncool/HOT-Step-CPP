/** Internal preparation contract for the SA3 runner. Consume parsed sidecars
 * or dataset.json rows; never pass raw key/value .txt bodies to SA3. Exports
 * contain only derived WAVs, plain same-stem prompts and manifest.json.
 * Routes, trainer integration and user documentation belong to later slices.
 */
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { assertSa3DerivedDir, prepareSa3Instrumental, inspectSa3Audio, validateSa3Audio, sa3Hash,
  type Sa3Instrumental, type Sa3StemDependencies } from './sa3Stems.js';

type Metadata = Record<string, unknown>;
const text = (value: unknown): string => typeof value === 'string' ? value.replace(/\s+/g, ' ').trim() : '';

/** Allowlist only. Lyrics, trigger/repeat, prompt_override and guessed duration
 * never enter the prompt. dataset.json stores only the signature numerator;
 * preserve that value rather than inventing a denominator. Merge the parsed
 * sidecar's explicit signature into metadata to retain the full signature.
 */
export function sa3Prompt(meta: Metadata): string {
  const bpm = typeof meta.bpm === 'number' ? meta.bpm : Number(text(meta.bpm));
  const signature = text(meta.signature) || text(meta.timesignature) || text(meta.time_signature);
  return [text(meta.caption), text(meta.genre), Number.isFinite(bpm) && bpm > 0 ? `${bpm} BPM` : '',
    text(meta.key) || text(meta.keyscale), signature].filter(Boolean).join(', ');
}

export interface Sa3DatasetSample {
  audio_path: string;
  metadata: Metadata;
  excluded?: boolean;
}
export interface Sa3DatasetManifest {
  version: 1;
  sample_rate: 44100;
  channels: 2;
  separation_level: 5;
  samples: Array<{
    audio_path: string;
    caption_path: string;
    prompt: string;
    seconds_total: number;
    source_path: string;
    source_sha256: string;
    model_sha256: string[];
    cache_fingerprint: string;
    audio_sha256: string;
  }>;
}

/** Publish a NEW derived directory only after every sample succeeds. Cancellation
 * or one failure leaves no export; good instrumentals stay cached for retry.
 * seconds_total comes from the exported clip, not sidecar/manifest duration.
 */
export async function prepareSa3Dataset(opts: {
  samples: Sa3DatasetSample[];
  exportDir: string;
  cacheDir: string;
  modelFiles: string[];
  engineUrl: string;
  signal?: AbortSignal;
  onProgress?: (done: number, total: number, fraction: number) => void;
}, deps: Partial<Sa3StemDependencies> & {
  prepare?: (source: string) => Promise<Sa3Instrumental>;
} = {}): Promise<Sa3DatasetManifest> {
  opts.signal?.throwIfAborted();
  const samples = opts.samples.filter(s => !s.excluded);
  if (!samples.length) throw new Error('SA3 dataset has no included audio');
  if (fs.existsSync(opts.exportDir)) throw new Error('SA3 export directory must be new');
  for (const sample of samples) {
    assertSa3DerivedDir(opts.exportDir, sample.audio_path);
    assertSa3DerivedDir(opts.cacheDir, sample.audio_path);
    if (!sa3Prompt(sample.metadata)) throw new Error('SA3 sample has no prompt metadata');
  }
  const exportDir = path.resolve(opts.exportDir);
  await fs.promises.mkdir(path.dirname(exportDir), { recursive: true });
  const scratch = path.join(path.dirname(exportDir), `.sa3-export-${randomUUID()}`);
  await fs.promises.mkdir(scratch);
  const manifest: Sa3DatasetManifest = { version: 1, sample_rate: 44100, channels: 2, separation_level: 5, samples: [] };
  try {
    for (let i = 0; i < samples.length; i++) {
      opts.signal?.throwIfAborted();
      const sample = samples[i];
      const prepared = deps.prepare ? await deps.prepare(sample.audio_path) : await prepareSa3Instrumental({
        source: sample.audio_path, cacheDir: opts.cacheDir, modelFiles: opts.modelFiles,
        engineUrl: opts.engineUrl, signal: opts.signal,
        onProgress: fraction => opts.onProgress?.(i, samples.length, fraction),
      }, deps);
      opts.signal?.throwIfAborted();
      // Index names avoid basename collisions and Windows reserved filenames.
      const stem = String(i + 1).padStart(6, '0');
      const audio = `${stem}.wav`;
      const caption = `${stem}.txt`;
      await fs.promises.copyFile(prepared.audioPath, path.join(scratch, audio));
      const info = await (deps.inspect ?? inspectSa3Audio)(path.join(scratch, audio));
      validateSa3Audio(info);
      const audioHash = await sa3Hash(path.join(scratch, audio));
      if (audioHash !== prepared.audioHash) throw new Error('SA3 cached instrumental changed during export');
      const prompt = sa3Prompt(sample.metadata);
      await fs.promises.writeFile(path.join(scratch, caption), `${prompt}\n`, 'utf8');
      manifest.samples.push({ audio_path: audio, caption_path: caption, prompt, seconds_total: info.duration,
        source_path: path.resolve(sample.audio_path), source_sha256: prepared.sourceHash,
        model_sha256: prepared.modelHashes, cache_fingerprint: prepared.fingerprint, audio_sha256: audioHash });
      opts.onProgress?.(i + 1, samples.length, 1);
    }
    await fs.promises.writeFile(path.join(scratch, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
    opts.signal?.throwIfAborted();
    if (fs.existsSync(exportDir)) throw new Error('SA3 export directory was created during preparation');
    await fs.promises.rename(scratch, exportDir);
    return manifest;
  } finally {
    await fs.promises.rm(scratch, { recursive: true, force: true });
  }
}
