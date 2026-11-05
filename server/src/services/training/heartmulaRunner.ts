// training/heartmulaRunner.ts — runs one `ace-train heartmula-prepare` or
// `heartmula-train` job.
//
// The spawn, engine stop/restore and idle watchdog are runYue2AceTrain's (one
// implementation for every ace-train subcommand). What differs is the output:
// heartmula-train speaks `HMTRAIN {json}` events on stdout (the contract is
// engine/docs/HEARTMULA_TRAINING_CONTRACT.md), and exit codes carry meaning:
// 0 done, 3 paused at a checkpoint, 4 cancelled, anything else a failure.
//
// Checkpoint previews are segments: the trainer runs with --stop-after to the
// next preview step (exit 3), the engine comes back up, heartmulaPreview renders
// the saved adapter and unloads, and the next segment resumes. The base model
// is rendered once before the first segment, the final adapter after the last,
// even when that step is off the cadence. A user's checkpoint-now ends the run
// as paused instead (checkpoint reason 'now'), never as a preview pause.

import fs from 'node:fs';
import path from 'node:path';
import { config } from '../../config.js';
import { getDataset } from './datasetsRepo.js';
import { buildSamples } from './datasetScan.js';
import {
  buildHeartmulaPrepareArgs, buildHeartmulaTrainArgs, findHeartmulaEncoderDir, missingEncoderMessage, missingHeartmulaBase, missingHeartmulaModels,
  resolveHeartmulaPreviewSettings,
} from './heartmulaTrain.js';
import { HEARTMULA_PREPARE_MANIFEST, snapshotSources, writeHeartmulaSamples, writeSkippedFingerprints } from './heartmulaPrepare.js';
import {
  appendLoss, blankStatus, heartmulaAdapterFile, heartmulaCheckpointSteps, heartmulaPrepareDir, publishHeartmulaAdapter,
  readStoredStatus, saveHeartmulaRunRecord, writeStoredStatus, type HeartmulaRunRecord,
} from './heartmulaRuns.js';
import { ensureHeartmulaRunPreview } from './heartmulaPreviewSource.js';
import {
  heartmulaPreviewDone, heartmulaPreviewKind, renderHeartmulaPreview, setHeartmulaPreviewPauseMs,
} from './heartmulaPreview.js';
import { HEARTMULA_PREVIEW_DEFAULTS, type HeartmulaPreviewSettings, type HeartmulaPreviewSource } from '../../contracts/heartmulaTraining.js';
import type { HeartmulaRecipe, HeartmulaRunState } from '../../contracts/heartmulaTraining.js';
import { log, runYue2AceTrain, YUE2_IDLE_MS, type RelayState } from './yue2TrainRunner.js';
import {
  emitProgress, finishJob, isCancelled, killJobChild, pushEvent, releaseJobCleanup, type TrainingJob,
} from './labelingQueue.js';

// ── prepare ──────────────────────────────────────────────────────────────

/**
 * The tool's lines (heartmula-prepare.h): `stage <what>`, `%zu/%zu <id>: working` before a
 * sample, `%zu/%zu <id>: <result>` after it, and `skip <id>: <reason> (<detail>)`.
 */
const RE_PREP_STAGE = /^\[HeartMuLa-Prepare\]\s+stage\s+(.+)$/;
const RE_PREP_WORK = /^\[HeartMuLa-Prepare\]\s+(\d+)\/(\d+)\s+(\S+):\s+working$/;
const RE_PREP_FILE = /^\[HeartMuLa-Prepare\]\s+(\d+)\/(\d+)\s/;
const RE_PREP_SKIP = /^\[HeartMuLa-Prepare\]\s+skip\s+(\S+):\s+(.+)$/;
/** The tool's own fatal sentences: `heartmula-prepare: <message>`. */
const RE_FATAL = /^(?:heartmula-prepare|heartmula-train):\s+(.+)$/;

export interface HeartmulaPrepareJobOpts { force?: boolean }

export async function runHeartmulaPrepareJob(job: TrainingJob): Promise<void> {
  const opts = (job.opts ?? {}) as HeartmulaPrepareJobOpts;
  const st: RelayState = { fatalMessage: '', doneSeen: false, lastStep: 0, totalSteps: 0 };
  try {
    const ds = getDataset(job.datasetId);
    if (!ds) throw new Error('Dataset not found');
    const missing = missingHeartmulaModels();
    if (missing.length) throw new Error(`HeartMuLa models are missing: ${missing.join(', ')}`);
    // No fallback: without the encoder pack the codes cannot be made.
    const encoderDir = findHeartmulaEncoderDir();
    if (!encoderDir) throw new Error(missingEncoderMessage());
    const samples = await buildSamples(ds);
    if (isCancelled(job)) return;
    const before = snapshotSources(ds, samples);
    const list = writeHeartmulaSamples(ds, samples);
    if (list.count === 0) throw new Error('The dataset has no usable audio to prepare');
    const outDir = heartmulaPrepareDir(ds.slug);
    const args = buildHeartmulaPrepareArgs({
      samplesJson: list.file, outDir, modelsDir: config.aceServer.models, encoderDir, force: opts.force,
    });
    job.total = list.count; job.done = 0; job.failed = 0; job.currentSampleId = null;
    log(job, 'info', `Encoding ${list.count} sample(s) to HeartCodec codes.`);
    let at = 0;  // the 1-based sample the tool is on
    await runYue2AceTrain(job, 'heartmula-prepare', args, YUE2_IDLE_MS, () =>
      fs.existsSync(path.join(outDir, HEARTMULA_PREPARE_MANIFEST)) ? null : 'heartmula-prepare finished but wrote no manifest',
    (line, state) => {
      let m: RegExpExecArray | null;
      if ((m = RE_PREP_STAGE.exec(line))) { job.phase = m[1].startsWith('encoding') ? 'encoding' : 'loading-models'; emitProgress(job); log(job, 'info', `Prepare: ${m[1]}.`); return; }
      if ((m = RE_PREP_WORK.exec(line))) {
        at = Number(m[1]); job.done = at - 1; job.total = Number(m[2]) || job.total; job.phase = 'encoding';
        job.currentSampleId = m[3]; emitProgress(job);
        return;
      }
      if ((m = RE_PREP_SKIP.exec(line))) {
        // A skipped sample is finished work that will not train: count it done and failed.
        job.failed++; job.done = at; emitProgress(job);
        log(job, 'warn', `Skipped ${m[1]}: ${m[2]}`);
        return;
      }
      if ((m = RE_PREP_FILE.exec(line))) {
        job.done = Number(m[1]); job.total = Number(m[2]) || job.total; job.phase = 'encoding'; emitProgress(job);
        return;
      }
      if ((m = RE_FATAL.exec(line))) state.fatalMessage ||= m[1];
    }, st);
    if (!isCancelled(job)) {
      writeSkippedFingerprints(ds.slug, before);
      finishJob(job, 'done');
    }
  } catch (err: any) {
    if (!isCancelled(job)) finishJob(job, 'failed', err?.message || String(err));
  }
}

// ── train ────────────────────────────────────────────────────────────────

export interface HeartmulaTrainJobOpts {
  runId: string;
  runDir: string;
  record: HeartmulaRunRecord;
  resume: boolean;
}

interface HmState extends RelayState {
  started: boolean;
  checkpointStep: number | null;
  /** Why the newest checkpoint was written: every | now | stop | final. */
  checkpointReason: string;
  stepMsSum: number;
  stepMsCount: number;
  /** When the trainer paused for a preview, and the take it paused for (set once that take has a record). */
  pauseT0: number;
  pausePreview?: string;
}

/** Jobs whose trainer has printed `start`; control files before that are deleted by the engine. */
const started = new Set<string>();

function touch(file: string): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, 'x\n');
}

/** Ask the running trainer to checkpoint and exit (paused, resumable). False if it is not running yet. */
export function requestHeartmulaCheckpoint(job: TrainingJob): boolean {
  const opts = job.opts as HeartmulaTrainJobOpts | undefined;
  if (!opts || !started.has(job.id)) return false;
  touch(path.join(opts.runDir, 'control', 'checkpoint-now'));
  return true;
}

function setStatus(opts: HeartmulaTrainJobOpts, patch: Partial<ReturnType<typeof blankStatus>>): void {
  const cur = readStoredStatus(opts.runDir) ?? blankStatus(opts.record.recipe.steps);
  writeStoredStatus(opts.runDir, { ...cur, ...patch });
}

function relayTrainLine(job: TrainingJob, opts: HeartmulaTrainJobOpts, line: string, st: HmState): void {
  if (!line.startsWith('HMTRAIN ')) {
    const m = RE_FATAL.exec(line);
    if (m) st.fatalMessage ||= m[1];
    return;
  }
  const ev = JSON.parse(line.slice('HMTRAIN '.length)) as Record<string, any>;
  const cur = readStoredStatus(opts.runDir) ?? blankStatus(opts.record.recipe.steps);
  switch (ev.event) {
    case 'start':
      st.started = true;
      started.add(job.id);
      st.totalSteps = Number(ev.total) || opts.record.recipe.steps;
      job.phase = 'training'; job.total = st.totalSteps; job.done = Number(ev.step) || 0; emitProgress(job);
      setStatus(opts, { state: 'running', step: Number(ev.step) || 0, totalSteps: st.totalSteps, startedAt: cur.startedAt ?? Date.now() });
      pushEvent(job, { type: 'metric', metric: 'data', ts: Date.now(), samples: Number(ev.samples) || undefined, totalSteps: st.totalSteps });
      log(job, 'info', `Training ${ev.samples ?? '?'} sample(s), ${st.totalSteps} steps.`);
      if (ev.alloc_mib !== undefined || ev.attention) {
        log(job, 'info', `Trainer: ${ev.attention ?? '?'} attention, ${ev.alloc_mib !== undefined ? `${Math.round(Number(ev.alloc_mib))} MiB allocated` : 'memory not reported'}.`);
      }
      if (st.pausePreview) {
        // Training is back: the whole pause (engine up, render, engine down, model reload) is what the preview cost.
        const pauseMs = Date.now() - st.pauseT0;
        setHeartmulaPreviewPauseMs(opts.runDir, st.pausePreview, pauseMs);
        log(job, 'info', `That preview paused training for ${(pauseMs / 1000).toFixed(1)} s.`);
        st.pausePreview = undefined;
      }
      break;
    case 'step': {
      const step = Number(ev.step), loss = Number(ev.loss), ms = Number(ev.ms_step);
      if (!Number.isFinite(loss)) { st.fatalMessage ||= 'The trainer reported a non-finite loss'; break; }
      st.lastStep = step; st.lastLoss = loss;
      if (Number.isFinite(ms)) { st.stepMsSum += ms; st.stepMsCount++; }
      const peak = Number(ev.peak_alloc_mib);
      pushEvent(job, { type: 'metric', metric: 'step', ts: Date.now(), step, totalSteps: st.totalSteps, loss,
        stepMs: Number.isFinite(ms) ? ms : undefined, gradNorm: Number.isFinite(Number(ev.grad_norm)) ? Number(ev.grad_norm) : undefined,
        usedMb: Number.isFinite(Number(ev.device_used_mib)) ? Number(ev.device_used_mib) : undefined });
      job.done = step; emitProgress(job);
      setStatus(opts, {
        state: 'running', step, totalSteps: st.totalSteps, loss, lossHistory: appendLoss(cur.lossHistory, step, loss),
        msPerStep: st.stepMsCount ? st.stepMsSum / st.stepMsCount : cur.msPerStep,
        peakVramMiB: Number.isFinite(peak) ? Math.max(peak, cur.peakVramMiB ?? 0) : cur.peakVramMiB,
      });
      break;
    }
    case 'skip':
      log(job, 'warn', `Step ${ev.step} skipped: ${ev.reason}`);
      break;
    case 'checkpoint':
      st.checkpointStep = Number(ev.step);
      st.checkpointReason = String(ev.reason ?? '');
      if (st.checkpointReason === 'stop') st.pauseT0 = Date.now();
      setStatus(opts, { checkpointStep: st.checkpointStep });
      pushEvent(job, { type: 'metric', metric: 'milestone', ts: Date.now(), step: st.checkpointStep, loss: st.lastLoss });
      log(job, 'info', `Checkpoint at step ${ev.step} (${ev.reason}).`);
      break;
    case 'done':
      st.doneSeen = true; st.lastStep = Number(ev.step) || st.lastStep;
      break;
    case 'error':
      st.fatalMessage ||= String(ev.message || 'The trainer reported an error');
      break;
  }
}

/** The run's preview fixture; a run made before previews existed has none, which means off. */
function previewOf(record: HeartmulaRunRecord): HeartmulaPreviewSettings {
  try { return resolveHeartmulaPreviewSettings(record.preview, { ...HEARTMULA_PREVIEW_DEFAULTS, everySteps: 0 }); }
  catch { return { ...HEARTMULA_PREVIEW_DEFAULTS, everySteps: 0 }; }
}

/** The next step a preview pause is due at, after `from`; 0 when the run finishes first. */
export function nextPreviewStep(from: number, every: number, total: number): number {
  if (every <= 0) return 0;
  const next = (Math.floor(from / every) + 1) * every;
  return next < total ? next : 0;
}

export async function runHeartmulaTrainJob(job: TrainingJob): Promise<void> {
  const opts = job.opts as HeartmulaTrainJobOpts | undefined;
  if (!opts?.runDir || !opts.record) { finishJob(job, 'failed', 'heartmula-train job is missing its run'); return; }
  const recipe: HeartmulaRecipe = opts.record.recipe;
  let preview = previewOf(opts.record);
  let previewSource: HeartmulaPreviewSource | undefined;
  const state = (s: HeartmulaRunState, extra: Partial<ReturnType<typeof blankStatus>> = {}) => setStatus(opts, { state: s, ...extra });
  const st: HmState = {
    fatalMessage: '', doneSeen: false, lastStep: 0, totalSteps: recipe.steps, started: false,
    checkpointStep: null, checkpointReason: '', stepMsSum: 0, stepMsCount: 0, pauseT0: 0,
  };
  job.total = recipe.steps; job.done = 0; job.cleanupPending = true;
  // Cancel is graceful once the trainer is up: control/cancel, then a bounded kill.
  let cancelTimer: ReturnType<typeof setTimeout> | undefined;
  job.cancelChild = () => {
    if (!started.has(job.id)) { killJobChild(job); return; }
    try { touch(path.join(opts.runDir, 'control', 'cancel')); } catch { killJobChild(job); return; }
    cancelTimer ??= setTimeout(() => killJobChild(job), 30_000);
  };

  /** One take. A failed render is logged and training goes on (the checkpoints are what matters); a dirty engine throws. */
  const takePreview = async (step: number): Promise<string> => {
    const kind = heartmulaPreviewKind(step, recipe.steps);
    state('previewing'); job.phase = 'preview'; emitProgress(job);
    log(job, 'info', `Rendering the ${kind === 'base' ? 'base model' : kind === 'final' ? 'final adapter' : `step ${step}`} preview (up to ${preview.seconds} s).`);
    const take = await renderHeartmulaPreview({
      runDir: opts.runDir, runId: opts.runId, step, kind, settings: preview, source: previewSource, signal: job.controller.signal,
      base: { variant: recipe.variant ?? '3B', quant: recipe.quant ?? 'F16' },
    });
    if (take.status === 'done') log(job, 'info', `Preview rendered in ${((take.renderMs ?? 0) / 1000).toFixed(1)} s.`);
    else if (take.status === 'failed') log(job, 'warn', `Preview at step ${step} failed; training goes on: ${take.error}`);
    return take.id;
  };

  try {
    const missing = missingHeartmulaModels();
    if (missing.length) throw new Error(`HeartMuLa models are missing: ${missing.join(', ')}`);
    const baseMissing = missingHeartmulaBase({ variant: recipe.variant ?? '3B', quant: recipe.quant ?? 'F16' });
    if (baseMissing) throw new Error(baseMissing);
    const dataDir = heartmulaPrepareDir(getDataset(job.datasetId)?.slug ?? '');
    if (!fs.existsSync(path.join(dataDir, HEARTMULA_PREPARE_MANIFEST))) {
      throw new Error('The HeartMuLa prepare cache is gone — run Prepare again');
    }
    const ds = getDataset(job.datasetId);
    if (preview.everySteps > 0 && ds) {
      // Resolved once and saved to the run, so every take (and a resume) sings the same words.
      ({ settings: preview, source: previewSource } = await ensureHeartmulaRunPreview(ds, { dir: opts.runDir, record: opts.record }));
      const s = previewSource;
      log(job, 'info', `Preview prompt: tags from ${s.tags === 'override' ? 'the override' : `"${s.sampleName}"`}; lyrics from `
        + `${s.lyrics === 'override' ? 'the override' : s.lyrics === 'lyric-studio' ? `Lyric Studio generation #${s.generationId}` : `"${s.sampleName}"`}.`);
    }
    state('queued');
    log(job, 'info', `${opts.resume ? 'Resuming' : 'Starting'}: ${recipe.steps} steps at rank ${recipe.rank}/alpha ${recipe.alpha}`
      + `${preview.everySteps > 0 ? `, a full-song preview (up to ${preview.seconds} s) every ${preview.everySteps} steps` : ', no checkpoint previews'}.`);

    let resume = opts.resume;
    for (;;) {
      // The trainer resumes from its newest checkpoint, whatever the last step event said.
      const from = resume ? Math.max(0, ...heartmulaCheckpointSteps(opts.runDir)) : 0;
      if (preview.everySteps > 0 && !isCancelled(job)) {
        // Owed takes: the base model before anything trained, and a cadence step a pause or crash cut short.
        const owed = from === 0 || (from < recipe.steps && from % preview.everySteps === 0) ? from : -1;
        if (owed >= 0 && !heartmulaPreviewDone(opts.runDir, owed)) {
          const id = await takePreview(owed);
          if (owed > 0) st.pausePreview = id;  // the base take precedes the first start; only a mid-run pause has a cost
        }
      }
      if (isCancelled(job)) { state('cancelled'); return; }
      const stopAfter = nextPreviewStep(from, preview.everySteps, recipe.steps);
      const args = buildHeartmulaTrainArgs({
        dataDir, runDir: opts.runDir, modelsDir: config.aceServer.models, recipe, name: opts.record.adapterName, resume, stopAfter,
        trigger: getDataset(job.datasetId)?.customTag ?? '',
      });
      st.checkpointReason = ''; st.exitCode = undefined; st.started = false;
      await runYue2AceTrain(job, 'heartmula-train', args, YUE2_IDLE_MS, () => null,
        (line, s) => relayTrainLine(job, opts, line, s), st, undefined, true, [0, 3]);

      if (isCancelled(job)) { state('cancelled'); return; }
      if (st.exitCode === 3) {
        if (stopAfter && st.checkpointReason === 'stop' && st.checkpointStep === stopAfter) { resume = true; continue; }
        // Paused at a checkpoint by checkpoint-now: resumable, not finished.
        state('cancelled', { checkpointStep: st.checkpointStep ?? readStoredStatus(opts.runDir)?.checkpointStep ?? null });
        log(job, 'info', `Paused at step ${st.lastStep}. Resume the run to continue.`);
        job.phase = 'paused';
        finishJob(job, 'done');
        return;
      }
      break;
    }

    // Exit 0 is finished; a resume of a finished run prints no steps, so fall back to the stored one.
    const finalStep = st.lastStep || readStoredStatus(opts.runDir)?.step || recipe.steps;
    const final = heartmulaAdapterFile(opts.runDir, finalStep);
    if (!fs.existsSync(final)) throw new Error('The trainer finished but wrote no adapter');
    const published = publishHeartmulaAdapter(final, `${opts.record.adapterName}-${path.basename(opts.runDir)}`);
    saveHeartmulaRunRecord(opts.runDir, { ...opts.record, published });
    log(job, 'info', `Published HeartMuLa adapter: ${published}`);
    // The final take is owed even off the cadence. The adapter is already published, so a failed render loses nothing.
    if (preview.everySteps > 0 && !heartmulaPreviewDone(opts.runDir, finalStep)) await takePreview(finalStep);
    if (isCancelled(job)) { state('cancelled', { step: finalStep, adapterFile: published }); return; }
    state('done', { step: finalStep, adapterFile: published, error: null });
    finishJob(job, 'done');
  } catch (err: any) {
    const message = err?.message || String(err);
    if (isCancelled(job)) state('cancelled'); else state('failed', { error: message });
    if (!isCancelled(job)) finishJob(job, 'failed', message);
  } finally {
    clearTimeout(cancelTimer);
    started.delete(job.id);
    job.cancelChild = undefined;
    releaseJobCleanup(job);
  }
}
