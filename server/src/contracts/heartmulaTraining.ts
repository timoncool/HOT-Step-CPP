// HeartMuLa backbone-LoRA training contracts shared by the Node server and the
// React UI (types and constants only). Slice 2 of HOTSTEP-767.
//
// Owned by the HeartMuLa training batch. The server slices (prepare, train,
// runner, preview) and the UI leaves build on these read-only; a change goes
// back to the owner of this file. The FROZEN SA3 shapes in ./training.ts are
// untouched.
//
// Routes these types describe (all scoped by the dataset, never by a path the
// browser supplies):
//   POST /datasets/:id/heartmula-prepare          <- HeartmulaPrepareRequest
//   POST /datasets/:id/heartmula-train            <- HeartmulaTrainRequest
//   POST /datasets/:id/heartmula-checkpoint       <- HeartmulaCheckpointNowRequest
//   GET  /datasets/:id/heartmula-status           -> HeartmulaStatusResponse
//   GET  /datasets/:id/heartmula-previews?run=    -> HeartmulaPreviewListResponse
//   POST /datasets/:id/heartmula-previews/render  <- HeartmulaPreviewRenderRequest, -> HeartmulaPreviewListResponse
//   GET  /datasets/:id/heartmula-previews/audio?run=&file= -> audio bytes

/** Training target id; a member of TRAINING_TARGET_IDS in ./training.ts. */
export const HEARTMULA_TARGET_ID = 'heartmula' as const;
export type HeartmulaTargetId = typeof HEARTMULA_TARGET_ID;

/** Job kinds the HeartMuLa pipeline submits (already members of TrainingJobKind). */
export const HEARTMULA_JOB_KINDS = ['heartmula-prepare', 'heartmula-train'] as const;
export type HeartmulaJobKind = typeof HEARTMULA_JOB_KINDS[number];

/** `<adapterName>/<timestamp>`, the same shape as Sa3RunId. */
export type HeartmulaRunId = string;

/** Base model sizes the trainer targets. v1 is the 3B base only. */
export type HeartmulaBase = '3B';

// ── Recipe ───────────────────────────────────────────────────────────────

/** Backbone LoRA recipe. Every key is optional on the wire; omitted means the default below. */
export interface HeartmulaRecipe {
  rank: number;
  alpha: number;
  learningRate: number;
  steps: number;
  /** Micro-batches accumulated per optimizer step. */
  gradAccum: number;
  seed: number;
  /** Write a resumable checkpoint every N steps; 0 = only on stop/finish. */
  checkpointEvery: number;
  /** LoRA dropout on the adapter inputs, 0 <= p < 1. Advanced: 0.05 is the proven recipe (+8% step time, +1.3 GiB). */
  loraDropout: number;
  /** Also LoRA the backbone MLP (w1/w2/w3), not only attention. Bigger adapter, more VRAM; unlocks the MLP dial. */
  trainMlp: boolean;
  /** Train on the first this-many seconds of each song (start-anchored, so starts stay right),
   *  with every 4th sample trained whole so endings are learned too; 0 = whole songs. Faster steps;
   *  peak memory still follows the longest song. */
  cropSeconds: number;
  /** Train the end-of-song frame after every song's last frame, so the adapter learns to end.
   *  Always on for new runs; false only on runs saved before it existed. */
  eos: boolean;
  /** The base model trained on: its variant (as the GGUF names it) and the quant of its global+local pair. */
  variant: string;
  quant: string;
}

/** An installed base the trainer can load: a variant and the quants it has a global+local pair in. */
export interface HeartmulaTrainBase {
  variant: string;
  quants: string[];
}

/** r64/alpha64 as decided; the rest is the proven recipe (contract section 1: four micro-batches per step). */
/** What a run saved before a recipe key existed actually trained with, where that is not
 *  today's default (resume checks and trainer arguments read a missing key as this). */
export const HEARTMULA_RECIPE_LEGACY: Partial<HeartmulaRecipe> = { trainMlp: false, cropSeconds: 0, eos: false, loraDropout: 0 };

export const HEARTMULA_RECIPE_DEFAULTS: HeartmulaRecipe = {
  rank: 64,
  alpha: 64,
  learningRate: 1e-4,
  steps: 480,
  gradAccum: 4,
  seed: 0,
  checkpointEvery: 100,
  loraDropout: 0.05,
  trainMlp: true,
  cropSeconds: 0,
  eos: true,
  variant: '3B',
  quant: 'F16',
};

/** The trainer ends each epoch with a short update: ceil(samples / accumulation). */
export function heartmulaDefaultSteps(preparedSamples: number, gradAccum = HEARTMULA_RECIPE_DEFAULTS.gradAccum): number {
  if (!Number.isSafeInteger(preparedSamples) || preparedSamples < 1
    || !Number.isSafeInteger(gradAccum) || gradAccum < 1) return HEARTMULA_RECIPE_DEFAULTS.steps;
  return Math.min(HEARTMULA_RECIPE_DEFAULTS.steps, 30 * Math.ceil(preparedSamples / gradAccum));
}

// ── Prepare ──────────────────────────────────────────────────────────────

/** Why a sample was left out of the prepared set. */
export type HeartmulaSkipReason = 'invalid' | 'over-context' | 'no-audio' | 'encode-failed';

export interface HeartmulaPrepareRequest {
  /** Rebuild every sample even when the cache fingerprint validates. */
  force?: boolean;
}

export interface HeartmulaSkippedSample {
  sampleId: string;
  reason: HeartmulaSkipReason;
  detail: string;
}

/** What the prepare stage has on disk for a dataset. */
export interface HeartmulaPrepareStatus {
  /** True only when the versioned cache fingerprint validates. */
  cacheValid: boolean;
  /** Cache fingerprint version the engine wrote; null when none. */
  cacheVersion: number | null;
  preparedSamples: number;
  /** Samples whose source changed since they were prepared. */
  staleSamples: number;
  skipped: HeartmulaSkippedSample[];
  /** Model context limit in tokens the over-context check used; null when unknown. */
  contextLimit: number | null;
  /** ISO timestamp of the last successful prepare, null when never. */
  preparedAt: string | null;
}

// ── Train ────────────────────────────────────────────────────────────────

export interface HeartmulaPreviewSettings {
  /** Render every N training steps; 0 turns checkpoint previews off. */
  everySteps: number;
  /** Tags override. Blank = the tags of the dataset's first sung track (what training saw), resolved once per run. */
  tags: string;
  /** Lyrics override. Blank = per `lyricsSource`. */
  lyrics: string;
  /**
   * Where blank lyrics come from, same meaning as YuE2's preview. Absent: the newest Lyric Studio generation for
   * the dataset's artist, falling back to the dataset track's lyrics. 'dataset': the track's own lyrics.
   */
  lyricsSource?: 'dataset';
  /** Longest a take may run, in seconds. A full song ends earlier on the model's own end-of-audio; 240 is the engine's maximum. */
  seconds: number;
  seed: number;
  /** Sampler, same keys and meaning as the HeartMuLa engine request. */
  temperature: number;
  topk: number;
  cfgScale: number;
  codecSteps: number;
  codecGuidance: number;
}

/** Blank tags and lyrics mean "take them from the dataset" (HeartmulaPreviewSource); sampler values are the generation defaults. */
export const HEARTMULA_PREVIEW_DEFAULTS: HeartmulaPreviewSettings = {
  everySteps: 100,
  tags: '',
  lyrics: '',
  seconds: 240,
  seed: 42,
  temperature: 1,
  topk: 50,
  cfgScale: 1.5,
  codecSteps: 10,
  codecGuidance: 1.25,
};

export interface HeartmulaTrainRequest {
  /** Adapter name; omitted = the dataset slug. */
  adapterName?: string;
  recipe?: Partial<HeartmulaRecipe>;
  preview?: Partial<HeartmulaPreviewSettings>;
  /** Continue this run from its latest checkpoint. Omitted = a new run. */
  resumeRun?: HeartmulaRunId;
}

/** Ask the running trainer to checkpoint atomically, then exit (resumable). */
export interface HeartmulaCheckpointNowRequest {
  run: HeartmulaRunId;
}

export type HeartmulaRunState =
  | 'idle'
  | 'queued'
  | 'running'
  | 'previewing'
  | 'checkpointing'
  | 'done'
  | 'failed'
  | 'cancelled';

/** One loss sample on the curve. */
export interface HeartmulaLossPoint {
  step: number;
  /** Mean of the eight per-codebook mean CEs. */
  loss: number;
}

/** Live and final numbers for a run. Step time and memory are first-class: they are the batch's targets. */
export interface HeartmulaRunStatus {
  run: HeartmulaRunId;
  state: HeartmulaRunState;
  step: number;
  totalSteps: number;
  /** Latest loss, null before the first step. */
  loss: number | null;
  lossHistory: HeartmulaLossPoint[];
  /** Mean wall time per optimizer step so far; null before the first step. */
  msPerStep: number | null;
  /** Peak VRAM the trainer has used, MiB; null when the device does not report it. */
  peakVramMiB: number | null;
  /** Step of the newest resumable checkpoint, null when none. */
  checkpointStep: number | null;
  /** The published GGUF adapter, relative to the adapter root; null until exported. */
  adapterFile: string | null;
  /** Failure text when state is 'failed'. */
  error: string | null;
  /** Epoch ms. */
  startedAt: number | null;
  updatedAt: number;
}

// ── Dataset asset block (chips) ──────────────────────────────────────────

/** A published HeartMuLa adapter, or null when none exists. */
export interface HeartmulaAdapterAsset {
  name: string;
  /** ISO timestamp of the publish. */
  trainedAt: string;
  run: HeartmulaRunId | null;
}

/** Whether training can start here, and what is missing when it cannot. */
export interface HeartmulaReadiness {
  /** CUDA build, ace-train present and every model file installed. */
  available: boolean;
  backend: string;
  /** Base-model files that are not installed. */
  missingModels: string[];
  /** Why the selected GPU cannot train, null when it can. */
  deviceProblem: string | null;
  /** The HeartCodec encoder pack is not installed (Prepare needs it). */
  encoderMissing: boolean;
  /** Missing file names and the install step, null when the pack is installed. */
  encoderHelp: string | null;
}

/** One run for the run picker. */
export interface HeartmulaRunSummary {
  run: HeartmulaRunId;
  adapterName: string;
  recipe: HeartmulaRecipe;
  state: HeartmulaRunState;
  step: number;
  /** True when a checkpoint exists and the run has not published. */
  resumable: boolean;
  published: string | null;
}

/** Everything the status route returns for one dataset; every flag is a verified fact. */
export interface HeartmulaStatusResponse {
  ready: HeartmulaReadiness;
  runs: HeartmulaRunSummary[];
  base: HeartmulaBase;
  /** Installed variants and quants the trainer can load, F16 first. */
  bases: HeartmulaTrainBase[];
  prepare: HeartmulaPrepareStatus;
  /** The active or newest run, null when the dataset has none. */
  run: HeartmulaRunStatus | null;
  adapter: HeartmulaAdapterAsset | null;
  /** Recipe the next run uses when the request omits keys. */
  defaults: HeartmulaRecipe;
  previewDefaults: HeartmulaPreviewSettings;
}

/** The HeartMuLa block of the dataset asset chips; every flag is a verified fact. */
export interface HeartmulaDatasetAssets {
  /** True only when the prepare cache validates against the dataset. */
  prepared: boolean;
  preparedSamples: number;
  adapter: HeartmulaAdapterAsset | null;
}

// ── Checkpoint previews ──────────────────────────────────────────────────

/** `base` renders the unadapted model; `step` a mid-run checkpoint; `final` the finished adapter. */
export type HeartmulaPreviewKind = 'base' | 'step' | 'final';

export type HeartmulaPreviewStatus = 'queued' | 'rendering' | 'done' | 'failed' | 'skipped';

/** Where a run's preview prompt came from. Resolved once at the start of a run, so every take sings the same words. */
export interface HeartmulaPreviewSource {
  tags: 'override' | 'dataset';
  lyrics: 'override' | 'lyric-studio' | 'dataset';
  /** The dataset track the tags (and dataset lyrics) came from; null when none qualified. */
  sampleId: string | null;
  sampleName: string | null;
  /** The Lyric Studio generation the lyrics came from; null otherwise. */
  generationId: number | null;
  generationTitle: string | null;
}

export interface HeartmulaPreviewRecord {
  run: HeartmulaRunId;
  kind: HeartmulaPreviewKind;
  /** Completed training step the snapshot was taken at; 0 for 'base'. */
  step: number;
  /** The resolved settings this take was rendered with (conditioning, seed, sampler). Per take: a retry with overrides can differ from its neighbours. */
  settings: HeartmulaPreviewSettings;
  /** Where the tags and lyrics in `settings` came from; absent on takes rendered before sources existed. */
  source?: HeartmulaPreviewSource;
  status: HeartmulaPreviewStatus;
  /** Server-validated URL under the audio route; null until `status` is 'done'. */
  audioUrl: string | null;
  /** Provenance: the adapter file this take was rendered with, relative to the run folder; null for 'base'. */
  adapter: string | null;
  /** SHA-256 of that adapter file, so a take can be matched to the exact weights; null for 'base'. */
  adapterSha256: string | null;
  /** Wall time of the render itself (queue, synth, fetch), ms; null until done. */
  renderMs: number | null;
  /** What this preview cost the run: trainer exit to the next trainer start (engine restart, render, engine stop, model reload), ms; null when it did not pause a run. */
  pauseMs: number | null;
  error: string | null;
  /** Epoch ms. */
  createdAt: number;
}

export interface HeartmulaPreviewListResponse {
  /** Echo of the resolved run, or null when the dataset has none. */
  run: HeartmulaRunId | null;
  /** The run's current defaults for new takes; each record carries its own resolved `settings`. */
  settings: HeartmulaPreviewSettings;
  /** Where `settings` tags and lyrics came from; null until the run has resolved them. */
  source: HeartmulaPreviewSource | null;
  previews: HeartmulaPreviewRecord[];
}

/** Ask for a render (or a retry) at one saved step; omitted settings use the run's. */
export interface HeartmulaPreviewRenderRequest {
  run: HeartmulaRunId;
  step: number;
  settings?: Partial<HeartmulaPreviewSettings>;
}
