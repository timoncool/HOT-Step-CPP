// Training Studio contracts shared by the Node server and the React UI (types only).
//
// FROZEN for the Training unification batch (HOTSTEP-674): the SA3 asset block,
// the SA3 recipe and the SA3 checkpoint-preview request/response shapes. The
// asset reader (datasetAssets), the preview orchestration (trainSa3Runner,
// routes) and the SA3 UI build on these read-only. A change goes back to the
// owner of this file, not into a downstream slice.
//
// Routes these types describe (all scoped by the dataset, never by a path the
// browser supplies):
//   GET  /datasets/:id/sa3-previews?run=            -> Sa3PreviewListResponse
//   POST /datasets/:id/sa3-previews/render          <- Sa3PreviewRenderRequest, -> Sa3PreviewListResponse
//   GET  /datasets/:id/sa3-previews/audio?run=&file= -> audio bytes

/** The things Training Studio can train for. Matches the backend registry ids. */
export const TRAINING_TARGET_IDS = ['ace', 'yue2', 'minimax-m3', 'sa3', 'heartmula'] as const;
export type TrainingTargetId = typeof TRAINING_TARGET_IDS[number];

/** `<adapterName>/<timestamp>`, the id `POST /sa3-train` already takes as `resumeRun`. */
export type Sa3RunId = string;

/** SA3 Medium DoRA-rows recipe. Same keys the trainer route accepts. */
export interface Sa3Recipe {
  steps: number;
  rank: number;
  alpha: number;
  learningRate: number;
  duration: number;
  seed: number;
  checkpointEvery: number;
  exportEvery: number;
}

// ── Dataset asset block (chips) ──────────────────────────────────────────

/** A published SA3 adapter, or null when none exists. */
export interface Sa3AdapterAsset {
  name: string;
  /** ISO timestamp of the publish. */
  trainedAt: string;
  run: Sa3RunId | null;
}

/**
 * What exists on disk for one dataset, read fresh like the other backends'
 * blocks. Every flag is a verified fact: readiness is never inferred from a
 * finished run alone.
 */
export interface Sa3DatasetAssets {
  /** Source audio (and stems) prepared for the trainer; clips counted below. */
  audioPrepared: boolean;
  preparedClips: number;
  /** The trainer's encoding cache, true only when its fingerprint validates. */
  encodingCacheValid: boolean;
  encodedClips: number;
  /** Inference snapshots (exportEvery) across the dataset's runs. */
  snapshots: number;
  /** The newest published adapter. */
  adapter: Sa3AdapterAsset | null;
}

// ── Checkpoint previews ──────────────────────────────────────────────────

/** Settings for one preview round; the same knobs the YuE2 preview card shows. */
export interface Sa3PreviewSettings {
  /** Render every N training steps; 0 turns checkpoint previews off. */
  everySteps: number;
  prompts: string[];
  /** Seconds of audio per render. */
  seconds: number;
  seeds: number[];
}

export type Sa3PreviewStatus = 'queued' | 'rendering' | 'done' | 'failed' | 'skipped';

/** One rendered (or pending) clip for a (step, prompt, seed) cell. */
export interface Sa3PreviewRecord {
  run: Sa3RunId;
  /** Completed training step the snapshot was taken at. */
  step: number;
  promptIndex: number;
  prompt: string;
  seed: number;
  seconds: number;
  status: Sa3PreviewStatus;
  /** Server-validated URL under the audio route; null until `status` is 'done'. */
  audioUrl: string | null;
  error: string | null;
  /** True for the render after the logical final step. */
  final: boolean;
  /** Epoch ms. */
  createdAt: number;
}

export interface Sa3PreviewListResponse {
  /** Echo of the resolved run, or null when the dataset has none. */
  run: Sa3RunId | null;
  settings: Sa3PreviewSettings;
  previews: Sa3PreviewRecord[];
}

/** Ask for a render (or a retry) at one saved step; omitted settings use the run's. */
export interface Sa3PreviewRenderRequest {
  run: Sa3RunId;
  step: number;
  settings?: Partial<Sa3PreviewSettings>;
}
