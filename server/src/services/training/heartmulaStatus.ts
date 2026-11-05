// training/heartmulaStatus.ts — GET /datasets/:id/heartmula-status, and the
// HeartMuLa block of the dataset asset chips.

import fs from 'node:fs';
import path from 'node:path';
import {
  HEARTMULA_PREVIEW_DEFAULTS, HEARTMULA_RECIPE_DEFAULTS, heartmulaDefaultSteps,
  type HeartmulaDatasetAssets, type HeartmulaReadiness, type HeartmulaStatusResponse,
} from '../../contracts/heartmulaTraining.js';
import { aceTrainExe, engineGpuBackend } from './aceTrain.js';
import { listGpusSync, selectedGpuMemoryMB, type NvidiaGpu } from '../gpuDevices.js';
import { buildSamples } from './datasetScan.js';
import { activeJobForDataset } from './labelingQueue.js';
import { findHeartmulaEncoderDir, heartmulaTrainBases, missingEncoderMessage, missingHeartmulaModels } from './heartmulaTrain.js';
import { heartmulaPrepareStatus, readHeartmulaManifest } from './heartmulaPrepare.js';
import {
  heartmulaRunStatus, latestHeartmulaAdapter, listHeartmulaRuns,
} from './heartmulaRuns.js';
import type { HeartmulaTrainJobOpts } from './heartmulaRunner.js';
import type { TrainingDatasetRow } from './types.js';

/**
 * The smallest card the trainer is expected to fit on. The trainer's own peak
 * allocation is 10.41 GiB with every block recomputed (`--keep-layers -1`, the
 * smallest mode) and 15.36 GiB with none (the default on a big card), measured
 * on clip B at rank 64 (engine/docs/HEARTMULA_TRAINING_CONTRACT.md section 8.1).
 * `--keep-layers auto` picks the largest mode that fits the free VRAM, so the
 * floor is the smallest mode plus 1.5 GiB for the CUDA context and fragmentation;
 * a card near it trains slower, not worse. LoRA dropout adds 1.3 GiB, which the
 * trainer's own admission check rejects if it does not fit.
 */
export const HEARTMULA_MIN_VRAM_MB = Math.ceil((10.41 + 1.5) * 1024);

/**
 * Why the card training would run on cannot train, or null. Gated per device:
 * the selected GPU (Settings, else the only one) must be known and have enough
 * memory. An unidentified card is refused rather than trusted.
 */
export function heartmulaDeviceProblem(gpus: NvidiaGpu[] = listGpusSync()): string | null {
  let mb: number | null;
  try { mb = selectedGpuMemoryMB(undefined, gpus); } catch (err: any) { return err.message; }
  if (mb === null) return 'The selected GPU could not be identified, so HeartMuLa training cannot check that it fits.';
  return mb >= HEARTMULA_MIN_VRAM_MB ? null
    : `The selected GPU has ${(mb / 1024).toFixed(1)} GB; HeartMuLa training needs at least ${(HEARTMULA_MIN_VRAM_MB / 1024).toFixed(1)} GB.`;
}

/** CUDA builds on a big enough card only: the trainer is validated nowhere else. */
export function heartmulaReadiness(gpus?: NvidiaGpu[]): HeartmulaReadiness {
  const backend = engineGpuBackend();
  const missingModels = missingHeartmulaModels();
  const deviceProblem = backend === 'cuda' ? heartmulaDeviceProblem(gpus) : null;
  const encoderMissing = findHeartmulaEncoderDir() === null;
  return {
    available: backend === 'cuda' && deviceProblem === null && missingModels.length === 0 && !!aceTrainExe(),
    backend, missingModels, deviceProblem, encoderMissing, encoderHelp: encoderMissing ? missingEncoderMessage() : null,
  };
}

/** The id of the run the dataset's live heartmula-train job is working on, if any. */
function liveRunId(ds: TrainingDatasetRow): string | null {
  const job = activeJobForDataset(ds.id);
  return job?.kind === 'heartmula-train' ? (job.opts as HeartmulaTrainJobOpts | undefined)?.runId ?? null : null;
}

export async function heartmulaStatus(ds: TrainingDatasetRow): Promise<HeartmulaStatusResponse> {
  const live = liveRunId(ds);
  const runs = listHeartmulaRuns(ds.id, ds.slug);
  const summaries = runs.map(r => {
    const status = heartmulaRunStatus(r, r.id === live);
    return {
      run: r.id, adapterName: r.record.adapterName, recipe: r.record.recipe, state: status.state, step: status.step,
      resumable: !r.record.published && fs.existsSync(path.join(r.dir, 'hm_train_state.bin')), published: r.record.published ?? null,
    };
  });
  const current = runs.find(r => r.id === live) ?? runs[0];
  const prepare = heartmulaPrepareStatus(ds, await buildSamples(ds));
  return {
    ready: heartmulaReadiness(), runs: summaries, base: '3B', bases: heartmulaTrainBases(),
    prepare,
    run: current ? heartmulaRunStatus(current, current.id === live) : null,
    adapter: latestHeartmulaAdapter(ds.id, ds.slug),
    defaults: { ...HEARTMULA_RECIPE_DEFAULTS, steps: heartmulaDefaultSteps(prepare.preparedSamples) }, previewDefaults: HEARTMULA_PREVIEW_DEFAULTS,
  };
}

/**
 * The chips' block. Cheap on purpose (no source scan, it runs for every row of
 * the dataset list): `prepared` means a manifest with samples exists; whether it
 * still matches the sources is the status route's `cacheValid`.
 */
export function readHeartmulaDatasetAssets(ds: TrainingDatasetRow): HeartmulaDatasetAssets {
  const manifest = readHeartmulaManifest(ds.slug);
  const preparedSamples = manifest?.samples.length ?? 0;
  return {
    prepared: preparedSamples > 0, preparedSamples, adapter: latestHeartmulaAdapter(ds.id, ds.slug),
  };
}
