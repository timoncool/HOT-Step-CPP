// training/heartmulaPreview.ts — HeartMuLa checkpoint previews: catalogue,
// provenance and the render itself.
//
// One record per take, kept in <run>/previews/index.json beside the WAVs
// (shape: contracts/heartmulaTraining.ts). Audio is written first and the
// record committed after, so a crash never leaves a record pointing at a
// partial file. A render is one synth through heartmulaClient with the take's
// adapter GGUF, then an unload; the trainer and the engine never hold the GPU
// together because the runner (heartmulaRunner) only calls this between
// trainer segments, with the engine back up.
//
// Every take uses the run's resolved prompt (tags, lyrics, seed, sampler; the
// source is heartmulaPreviewSource), so base, step and final takes differ only
// by their weights.

import fs from 'node:fs';
import path from 'node:path';
import { createHash, randomBytes, randomUUID } from 'node:crypto';
import { aceClient } from '../aceClient.js';
import { heartmulaClient, type HeartmulaRequest } from '../backends/heartmula/client.js';
import { heartmulaAdapterInfo, heartmulaPick, heartmulaSelection, withHeartmulaTrigger } from '../backends/heartmula/models.js';
import type {
  HeartmulaPreviewKind, HeartmulaPreviewRecord, HeartmulaPreviewSettings, HeartmulaPreviewSource, HeartmulaRunId,
} from '../../contracts/heartmulaTraining.js';
import { heartmulaAdapterFile, isGgufAdapter } from './heartmulaRuns.js';

const MAX_RECORDS = 512;
const RENDER_TIMEOUT_MS = 30 * 60_000;
const POLL_MS = 500;

/** On-disk record: the wire record minus its URL, plus its id and audio file name. */
export type HeartmulaStoredPreview = Omit<HeartmulaPreviewRecord, 'audioUrl'> & { id: string; file?: string };

/** The engine could not be left clean after a render; the run must stop rather than train beside a loaded model. */
export class HeartmulaPreviewCleanupError extends Error { readonly code = 'PREVIEW_CLEANUP'; }

/** Runs whose previews are rendering in this process. Any other queued or rendering record was cut off. */
const live = new Set<string>();
export const heartmulaPreviewBusy = (runDir: string): boolean => live.has(path.resolve(runDir));

// ── catalogue ────────────────────────────────────────────────────────────

const indexPath = (runDir: string): string => path.join(runDir, 'previews', 'index.json');

function readRecords(runDir: string): HeartmulaStoredPreview[] {
  try {
    const value = JSON.parse(fs.readFileSync(indexPath(runDir), 'utf8')) as unknown;
    return Array.isArray(value) ? value.filter((r): r is HeartmulaStoredPreview => !!r && typeof r === 'object'
      && typeof r.id === 'string' && Number.isSafeInteger(r.step) && typeof r.kind === 'string'
      && typeof r.status === 'string' && !!r.settings && typeof r.settings === 'object') : [];
  } catch { return []; }
}

function writeRecords(runDir: string, records: HeartmulaStoredPreview[]): void {
  fs.mkdirSync(path.dirname(indexPath(runDir)), { recursive: true });
  const temp = `${indexPath(runDir)}.${randomUUID()}.tmp`;
  fs.writeFileSync(temp, `${JSON.stringify(records.slice(-MAX_RECORDS), null, 2)}\n`, 'utf8');
  fs.renameSync(temp, indexPath(runDir));
}

function upsert(runDir: string, record: HeartmulaStoredPreview): void {
  writeRecords(runDir, [...readRecords(runDir).filter(r => r.id !== record.id), record]);
}

/** Takes oldest step first (base, then the checkpoints, then final); a cut-off render reads as failed so it can be retried. */
export function listHeartmulaPreviewRecords(runDir: string): HeartmulaStoredPreview[] {
  const alive = heartmulaPreviewBusy(runDir);
  return readRecords(runDir)
    .map(r => !alive && (r.status === 'queued' || r.status === 'rendering')
      ? { ...r, status: 'failed' as const, error: 'Interrupted before it finished' } : r)
    .sort((a, b) => a.step - b.step || a.createdAt - b.createdAt);
}

/** True when a take at this step already rendered (or is rendering): nothing is owed there. */
export function heartmulaPreviewDone(runDir: string, step: number): boolean {
  return readRecords(runDir).some(r => r.step === step && r.status === 'done' && !!r.file);
}

/** Record the pause a preview cost the run (known only once the trainer is back). */
export function setHeartmulaPreviewPauseMs(runDir: string, id: string, pauseMs: number): void {
  const record = readRecords(runDir).find(r => r.id === id);
  if (record) upsert(runDir, { ...record, pauseMs: Math.round(pauseMs) });
}

/** <run>/previews with links resolved, or null when it resolves outside the run. */
function previewsDir(runDir: string): string | null {
  try {
    const run = fs.realpathSync(runDir), dir = fs.realpathSync(path.join(runDir, 'previews'));
    return dir === path.join(run, 'previews') ? dir : null;
  } catch { return null; }
}

/** The audio of a done record, or null for any other name, and for a folder or file that a link takes outside the run. */
export function heartmulaPreviewAudioPath(runDir: string, file: string): string | null {
  if (!/^(?:base|step|final)-[0-9]+-[a-f0-9]{8}\.wav$/.test(file)) return null;
  if (!readRecords(runDir).some(r => r.status === 'done' && r.file === file)) return null;
  const dir = previewsDir(runDir);
  if (!dir) return null;
  try {
    const real = fs.realpathSync(path.join(dir, file));
    return path.dirname(real) === dir && fs.statSync(real).isFile() ? real : null;
  } catch { return null; }
}

/** Which kind a take at `step` is: 0 is the base model, the last step the final adapter. */
export const heartmulaPreviewKind = (step: number, totalSteps: number): HeartmulaPreviewKind =>
  step === 0 ? 'base' : step >= totalSteps ? 'final' : 'step';

// ── render ───────────────────────────────────────────────────────────────

function sha256(file: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    fs.createReadStream(file).on('data', d => hash.update(d)).on('error', reject).on('end', () => resolve(hash.digest('hex')));
  });
}

export interface HeartmulaPreviewDeps {
  synth: (wire: HeartmulaRequest) => Promise<string>;
  poll: (id: string) => Promise<{ status: string }>;
  detail: (id: string) => Promise<{ status: string; error?: { message: string } | null }>;
  result: (id: string) => Promise<Response>;
  cancel: (id: string) => Promise<unknown>;
  unload: () => Promise<void>;
  /** Free the other backends' VRAM before the model loads. */
  beforeRender: () => Promise<void>;
  /** The model pick the render uses, on the run's base; the adapter fields are replaced per take. */
  pick: (base: HeartmulaTrainBasePick) => HeartmulaRequest;
}

/**
 * The base the trainer loaded (the run's `--variant` and `--type`, heartmula-train-run.h).
 * An adapter is bound to that base by fingerprint, so every take renders on it
 * whatever the user generates with; the codec stages follow their own picks.
 */
export interface HeartmulaTrainBasePick { variant: string; quant: string }
export const HEARTMULA_TRAIN_BASE: HeartmulaTrainBasePick = { variant: '3B', quant: 'F16' };

function defaultDeps(): HeartmulaPreviewDeps {
  return {
    synth: w => heartmulaClient.synth(w),
    poll: id => aceClient.pollJob(id),
    detail: id => heartmulaClient.job(id),
    result: id => aceClient.getJobResult(id),
    cancel: id => aceClient.cancelJob(id),
    unload: () => heartmulaClient.unload(),
    beforeRender: async () => { await (await import('../generation/residency.js')).evictOtherFamilies('heartmula'); },
    // The trainer's base and no generation adapter: a take carries only the run's weights.
    pick: base => ({ ...heartmulaPick({ ...heartmulaSelection(), variant: base.variant, global: base.quant, local: base.quant, lmAdapter: '' }) } as HeartmulaRequest),
  };
}

export interface HeartmulaPreviewInput {
  runDir: string;
  runId: HeartmulaRunId;
  /** Training step the weights were saved at; 0 renders the base model. */
  step: number;
  kind: HeartmulaPreviewKind;
  settings: HeartmulaPreviewSettings;
  /** Where `settings` tags and lyrics came from; kept on the record. */
  source?: HeartmulaPreviewSource;
  /** The run's base; absent = HEARTMULA_TRAIN_BASE, what every run used before the choice existed. */
  base?: HeartmulaTrainBasePick;
  signal?: AbortSignal;
  deps?: Partial<HeartmulaPreviewDeps>;
}

/**
 * Render one take. A failed render is recorded and returned (the caller decides
 * whether training goes on); a cleanup failure throws HeartmulaPreviewCleanupError.
 * Takes an earlier failed or skipped record of the same step with it, so a
 * retry replaces it; a done take is never touched.
 */
export async function renderHeartmulaPreview(input: HeartmulaPreviewInput): Promise<HeartmulaStoredPreview> {
  const deps: HeartmulaPreviewDeps = { ...defaultDeps(), ...input.deps };
  const key = path.resolve(input.runDir);
  const started = Date.now();
  const adapterFile = input.kind === 'base' ? null : heartmulaAdapterFile(input.runDir, input.step);
  let record: HeartmulaStoredPreview = {
    id: randomUUID(), run: input.runId, kind: input.kind, step: input.step, settings: input.settings, ...(input.source ? { source: input.source } : {}), status: 'rendering',
    adapter: adapterFile ? path.relative(input.runDir, adapterFile).split(path.sep).join('/') : null, adapterSha256: null,
    renderMs: null, pauseMs: null, error: null, createdAt: started,
  };
  live.add(key);
  let engineJob: string | undefined;
  let terminal = true;
  let cleanup: unknown;
  try {
    writeRecords(input.runDir, [...readRecords(input.runDir)
      .filter(r => !(r.kind === input.kind && r.step === input.step && (r.status === 'failed' || r.status === 'skipped' || r.status === 'queued' || r.status === 'rendering'))),
      record]);
    if (input.signal?.aborted) throw new Error('Cancelled');
    if (adapterFile) {
      if (!isGgufAdapter(adapterFile)) throw new Error(`No valid adapter was saved at step ${input.step}`);
      record = { ...record, adapterSha256: await sha256(adapterFile) };
      upsert(input.runDir, record);
    }
    await deps.beforeRender();
    const s = input.settings;
    const tags = adapterFile ? withHeartmulaTrigger(s.tags, heartmulaAdapterInfo(adapterFile).trigger) : s.tags;
    if (Buffer.byteLength(tags) > 4096) throw new Error('tags with the adapter trigger exceed 4096 bytes');
    const wire: HeartmulaRequest = {
      ...deps.pick(input.base ?? HEARTMULA_TRAIN_BASE), adapter: adapterFile ?? '', adapter_scale: 1,
      tags, lyrics: s.lyrics, duration_s: s.seconds, seed: s.seed,
      temperature: s.temperature, topk: s.topk, cfg_scale: s.cfgScale, codec_steps: s.codecSteps, codec_guidance: s.codecGuidance,
    };
    engineJob = await deps.synth(wire);
    terminal = false;
    const deadline = started + RENDER_TIMEOUT_MS;
    for (;;) {
      if (input.signal?.aborted) throw new Error('Cancelled');
      if (Date.now() >= deadline) throw new Error('The preview took longer than 30 minutes');
      const status = (await deps.poll(engineJob)).status;
      if (status === 'done') { terminal = true; break; }
      if (status === 'failed' || status === 'cancelled') {
        terminal = true;
        const detail = await deps.detail(engineJob).catch(() => null);
        throw new Error(detail?.error?.message || `The engine job ${status}`);
      }
      await new Promise(resolve => setTimeout(resolve, POLL_MS));
    }
    const response = await deps.result(engineJob);
    if (!response.ok) throw new Error(`HeartMuLa audio result HTTP ${response.status}`);
    const wav = Buffer.from(await response.arrayBuffer());
    if (wav.length < 44 || wav.toString('ascii', 0, 4) !== 'RIFF' || wav.toString('ascii', 8, 12) !== 'WAVE') {
      throw new Error('HeartMuLa returned an invalid WAV result');
    }
    const dir = previewsDir(input.runDir) ?? (fs.mkdirSync(path.join(input.runDir, 'previews'), { recursive: true }), previewsDir(input.runDir));
    if (!dir) throw new Error('The previews folder resolves outside the run');
    const file = `${input.kind}-${input.step}-${randomBytes(4).toString('hex')}.wav`;
    fs.writeFileSync(path.join(dir, file), wav, { flag: 'wx' });
    record = { ...record, status: 'done', file, renderMs: Date.now() - started };
    upsert(input.runDir, record);
  } catch (err: any) {
    const cancelled = !!input.signal?.aborted;
    record = { ...record, status: cancelled ? 'skipped' : 'failed', error: cancelled ? 'Cancelled' : err?.message || String(err) };
    upsert(input.runDir, record);
  } finally {
    if (engineJob && !terminal) {
      await deps.cancel(engineJob).catch(() => {});
      const until = Date.now() + 30_000;
      while (Date.now() < until && !terminal) {
        try { const st = (await deps.poll(engineJob)).status; terminal = st === 'done' || st === 'failed' || st === 'cancelled'; } catch { /* transient */ }
        if (!terminal) await new Promise(resolve => setTimeout(resolve, 250));
      }
      if (!terminal) cleanup = new Error('the preview job did not reach a terminal state');
    }
    // Free the model even after a failed render: the next thing is the trainer, or the user's own generation.
    // A refused connection (fetch's TypeError) means no engine is up, so nothing is resident.
    if (!cleanup) { try { await deps.unload(); } catch (err) { if (!(err instanceof TypeError)) cleanup = err; } }
    live.delete(key);
  }
  if (cleanup) {
    throw new HeartmulaPreviewCleanupError(`HeartMuLa preview cleanup failed: ${cleanup instanceof Error ? cleanup.message : String(cleanup)}`);
  }
  return record;
}
