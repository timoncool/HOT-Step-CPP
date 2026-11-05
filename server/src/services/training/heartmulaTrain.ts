// training/heartmulaTrain.ts — recipe validation, model discovery and argv for
// `ace-train heartmula-prepare` / `heartmula-train`.
//
// The CLI contract is engine/docs/HEARTMULA_TRAINING_CONTRACT.md and the header
// of engine/src/train/heartmula-train-run.h. Recipe defaults live in
// contracts/heartmulaTraining.ts and nowhere else.

import fs from 'node:fs';
import path from 'node:path';
import { config } from '../../config.js';
import {
  HEARTMULA_PREVIEW_DEFAULTS, HEARTMULA_RECIPE_DEFAULTS, heartmulaDefaultSteps, type HeartmulaPreviewSettings, type HeartmulaRecipe,
  type HeartmulaTrainBase,
} from '../../contracts/heartmulaTraining.js';

export const HEARTMULA_RECIPE_KEYS = Object.keys(HEARTMULA_RECIPE_DEFAULTS) as (keyof HeartmulaRecipe)[];

// Prepare needs only these; the LM pair is the run's own choice (heartmulaTrainBases).
const MODEL_FILES = [/^heartmula-tokenizer\.gguf$/i, /^heartcodec-rvq-.+\.gguf$/i];
const ENCODER_FILES = ['muencoder', 'wavlm', 'whisper', 'query'].map(role => `heartcodec-${role}-F16.gguf`);

/** Where the engine looks for models: `<models>/heartmula`, then `<models>`. */
export function heartmulaModelDirs(modelsDir = config.aceServer.models): string[] {
  return [path.join(modelsDir, 'heartmula'), modelsDir];
}

function namesIn(dir: string): string[] {
  try { return fs.readdirSync(dir); } catch { return []; }
}

/** Human names of the base-model files that are not installed (empty = ready). */
export function missingHeartmulaModels(modelsDir = config.aceServer.models): string[] {
  const names = heartmulaModelDirs(modelsDir).flatMap(namesIn);
  const missing = ['heartmula-tokenizer.gguf', 'heartcodec-rvq-<quant>.gguf'].filter((_, i) => !names.some(n => MODEL_FILES[i].test(n)));
  return heartmulaTrainBases(modelsDir).length ? missing : [...missing, 'a heartmula-<variant>-global/local-<quant>.gguf pair'];
}

const variantCache = new Map<string, { mtimeMs: number; variant: string }>();

/** `heartmula.variant` from a model's GGUF header, cached by mtime (the status route polls). */
function ggufVariant(file: string): string {
  let mtimeMs = 0;
  try { mtimeMs = fs.statSync(file).mtimeMs; } catch { return ''; }
  const hit = variantCache.get(file);
  if (hit && hit.mtimeMs === mtimeMs) return hit.variant;
  let variant = '', fd: number | undefined;
  try {
    fd = fs.openSync(file, 'r');
    const buf = Buffer.alloc(64 * 1024);
    const head = buf.subarray(0, fs.readSync(fd, buf, 0, buf.length, 0));
    // A GGUF string KV: u64 key length, key, u32 type (8 = string), u64 length, bytes.
    const key = Buffer.from('heartmula.variant');
    const at = head.indexOf(key);
    if (at >= 8 && head.readBigUInt64LE(at - 8) === BigInt(key.length) && head.readUInt32LE(at + key.length) === 8) {
      const len = Number(head.readBigUInt64LE(at + key.length + 4));
      if (len > 0 && len < 64) variant = head.toString('utf8', at + key.length + 12, at + key.length + 12 + len);
    }
  } catch { /* unreadable: not offered */ }
  finally { if (fd !== undefined) fs.closeSync(fd); }
  variantCache.set(file, { mtimeMs, variant });
  return variant;
}

/** Variants with a global+local pair installed, and the quants each pair exists in (F16 first). */
export function heartmulaTrainBases(modelsDir = config.aceServer.models): HeartmulaTrainBase[] {
  const pairs = new Map<string, Set<string>>();
  for (const dir of heartmulaModelDirs(modelsDir)) {
    const names = namesIn(dir);
    for (const n of names) {
      const m = /^heartmula-(.+)-global-([A-Za-z0-9_]+)\.gguf$/.exec(n);
      if (!m || !names.some(l => l.toLowerCase() === `heartmula-${m[1]}-local-${m[2]}.gguf`.toLowerCase())) continue;
      const variant = ggufVariant(path.join(dir, n)) || m[1];
      if (!pairs.has(variant)) pairs.set(variant, new Set());
      pairs.get(variant)!.add(m[2]);
    }
  }
  const rank = (q: string) => q === 'F16' ? 0 : q === 'BF16' ? 1 : 2;
  return [...pairs].map(([variant, q]) => ({ variant, quants: [...q].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b)) }))
    .sort((a, b) => a.variant.localeCompare(b.variant));
}

/** Why the run's base cannot load, or '' when its global+local pair is installed. */
export function missingHeartmulaBase(recipe: Pick<HeartmulaRecipe, 'variant' | 'quant'>, modelsDir = config.aceServer.models): string {
  const lower = (s: string) => s.toLowerCase();
  const base = heartmulaTrainBases(modelsDir).find(b => lower(b.variant) === lower(recipe.variant));
  return base?.quants.some(q => lower(q) === lower(recipe.quant)) ? ''
    : `The HeartMuLa ${recipe.variant} ${recipe.quant} global+local model pair is not installed`;
}

/**
 * The directory holding the HeartCodec encoder pack (all four F16 files), or
 * null. Only the models folder is searched: the oracle checkout is never a
 * source. Readiness reports a missing pack before Prepare is offered.
 */
export function findHeartmulaEncoderDir(modelsDir = config.aceServer.models): string | null {
  for (const dir of heartmulaModelDirs(modelsDir)) {
    const names = new Set(namesIn(dir).map(n => n.toLowerCase()));
    if (ENCODER_FILES.every(f => names.has(f.toLowerCase()))) return dir;
  }
  return null;
}

/**
 * What the user is told when the pack is missing: the exact files and the one
 * install step. Packs reach users the way the generation packs do, by running
 * the offline converter on the upstream weights (engine/tools/convert-heartmula.py).
 */
export const missingEncoderMessage = (modelsDir = config.aceServer.models): string =>
  `The HeartCodec encoder pack is not installed (missing: ${ENCODER_FILES.join(', ')}). `
  + 'Build it from the upstream HeartCodec weights with '
  + '`python engine/tools/convert-heartmula.py --src <checkpoint-root> --out <dir> --only encoder`, '
  + `then copy the four heartcodec-*-F16.gguf files into ${path.join(modelsDir, 'heartmula')}. See engine/docs/HEARTMULA_CONTRACT.md (Files and discovery).`;

// ── Recipe ───────────────────────────────────────────────────────────────

function int(v: unknown, key: string, min: number, max: number): number {
  const n = Number(v);
  if (!Number.isSafeInteger(n) || n < min || n > max) throw new Error(`${key} must be a whole number from ${min} to ${max}`);
  return n;
}

/** Defaults under any given keys; unknown keys and out-of-range values throw. */
export function resolveHeartmulaRecipe(input: Record<string, unknown> | undefined, preparedSamples = 0): HeartmulaRecipe {
  const src = input ?? {};
  for (const key of Object.keys(src)) {
    if (!(HEARTMULA_RECIPE_KEYS as string[]).includes(key)) throw new Error(`Unsupported HeartMuLa recipe key: ${key}`);
  }
  const r: HeartmulaRecipe = { ...HEARTMULA_RECIPE_DEFAULTS };
  if (src.rank !== undefined) r.rank = int(src.rank, 'rank', 1, 256);
  if (src.steps !== undefined) r.steps = int(src.steps, 'steps', 1, 100_000);
  if (src.gradAccum !== undefined) r.gradAccum = int(src.gradAccum, 'gradAccum', 1, 64);
  if (src.seed !== undefined) r.seed = int(src.seed, 'seed', 0, Number.MAX_SAFE_INTEGER);
  if (src.checkpointEvery !== undefined) r.checkpointEvery = int(src.checkpointEvery, 'checkpointEvery', 0, 100_000);
  for (const key of ['alpha', 'learningRate'] as const) {
    if (src[key] === undefined) continue;
    const n = Number(src[key]);
    if (!Number.isFinite(n) || n <= 0 || n > (key === 'learningRate' ? 1 : 1024)) throw new Error(`${key} is out of range`);
    r[key] = n;
  }
  if (src.loraDropout !== undefined) {
    const n = Number(src.loraDropout);
    if (!Number.isFinite(n) || n < 0 || n >= 1) throw new Error('loraDropout must be at least 0 and below 1');
    r.loraDropout = n;
  }
  for (const key of ['variant', 'quant'] as const) {
    if (src[key] === undefined) continue;
    if (typeof src[key] !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,47}$/.test(src[key] as string)) throw new Error(`${key} must be a model ${key} name`);
    r[key] = src[key] as string;
  }
  if (src.cropSeconds !== undefined) {
    const n = Number(src.cropSeconds);
    if (!Number.isFinite(n) || n !== 0 && (n < 10 || n > 600)) throw new Error('cropSeconds must be 0 (off) or 10 to 600 seconds');
    r.cropSeconds = n;
  }
  if (src.eos !== undefined) {
    if (typeof src.eos !== 'boolean') throw new Error('eos must be a boolean');
    r.eos = src.eos;
  }
  if (src.trainMlp !== undefined) {
    if (typeof src.trainMlp !== 'boolean') throw new Error('trainMlp must be a boolean');
    r.trainMlp = src.trainMlp;
  }
  if (src.steps === undefined) r.steps = heartmulaDefaultSteps(preparedSamples, r.gradAccum);
  return r;
}

/** Validate now, but defer an omitted update budget until each dataset is prepared. */
export function resolveHeartmulaBatchRecipe(input: Record<string, unknown> | undefined): Partial<HeartmulaRecipe> {
  const recipe: Partial<HeartmulaRecipe> = resolveHeartmulaRecipe(input);
  if (input?.steps === undefined) delete recipe.steps;
  return recipe;
}

const PREVIEW_KEYS = [...Object.keys(HEARTMULA_PREVIEW_DEFAULTS), 'lyricsSource'];

function num(v: unknown, key: string, min: number, max: number, integer: boolean): number {
  const n = Number(v);
  if (typeof v === 'boolean' || v === null || !Number.isFinite(n) || n < min || n > max || (integer && !Number.isSafeInteger(n))) {
    throw new Error(`preview ${key} must be ${integer ? 'a whole number' : 'a number'} from ${min} to ${max}`);
  }
  return n;
}

/**
 * Preview settings over `base`; unknown keys and out-of-range values throw. Blank tags
 * and lyrics are legal here: the run resolves them from the dataset (heartmulaPreviewSource).
 * Ranges are the engine request's (mapHeartmulaRequest), so a saved fixture
 * can never be refused at render time.
 */
export function resolveHeartmulaPreviewSettings(
  input: unknown, base: HeartmulaPreviewSettings = HEARTMULA_PREVIEW_DEFAULTS,
): HeartmulaPreviewSettings {
  if (input === undefined) return base;
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('preview settings must be an object');
  const src = input as Record<string, unknown>;
  for (const key of Object.keys(src)) if (!PREVIEW_KEYS.includes(key)) throw new Error(`Unsupported HeartMuLa preview option: ${key}`);
  const r: HeartmulaPreviewSettings = { ...base };
  for (const key of ['tags', 'lyrics'] as const) {
    if (src[key] === undefined) continue;
    const limit = key === 'tags' ? 4096 : 32768;
    if (typeof src[key] !== 'string' || Buffer.byteLength(src[key]) > limit) throw new Error(`preview ${key} must be text of at most ${limit} bytes`);
    r[key] = src[key];
  }
  if (src.lyricsSource !== undefined) {
    if (src.lyricsSource !== 'dataset' && src.lyricsSource !== 'generated' && src.lyricsSource !== null) throw new Error("preview lyricsSource must be 'dataset' or 'generated'");
    if (src.lyricsSource === 'dataset') r.lyricsSource = 'dataset'; else delete r.lyricsSource;
  }
  if (src.everySteps !== undefined) r.everySteps = num(src.everySteps, 'everySteps', 0, 100_000, true);
  if (src.seconds !== undefined) r.seconds = num(src.seconds, 'seconds', 5, 240, true);
  if (src.seed !== undefined) r.seed = num(src.seed, 'seed', 0, 0xffffffff, true);
  if (src.temperature !== undefined) r.temperature = num(src.temperature, 'temperature', Number.MIN_VALUE, 4, false);
  if (src.topk !== undefined) r.topk = num(src.topk, 'topk', 1, 8197, true);
  if (src.cfgScale !== undefined) r.cfgScale = num(src.cfgScale, 'cfgScale', 1, 10, false);
  if (src.codecSteps !== undefined) r.codecSteps = num(src.codecSteps, 'codecSteps', 1, 100, true);
  if (src.codecGuidance !== undefined) r.codecGuidance = num(src.codecGuidance, 'codecGuidance', 1, 10, false);
  return r;
}

// ── argv ─────────────────────────────────────────────────────────────────

export function buildHeartmulaPrepareArgs(o: {
  samplesJson: string; outDir: string; modelsDir: string; encoderDir: string; ffmpeg?: string; force?: boolean;
}): string[] {
  return ['heartmula-prepare', '--dataset', o.samplesJson, '--out', o.outDir, '--models', o.modelsDir,
    '--encoder', o.encoderDir, ...(o.ffmpeg ? ['--ffmpeg', o.ffmpeg] : []), ...(o.force ? ['--force'] : [])];
}

export function buildHeartmulaTrainArgs(o: {
  dataDir: string; runDir: string; modelsDir: string; recipe: HeartmulaRecipe; name: string; resume: boolean;
  /** Pause at this step with a checkpoint (exit 3); a preview segment ends here. */
  stopAfter?: number;
  /** The dataset's trigger word, stored in the adapter so generation can lead the tags with it. */
  trigger?: string;
}): string[] {
  const r = o.recipe;
  return ['heartmula-train', '--data', o.dataDir, '--out', o.runDir, '--models', o.modelsDir,
    '--variant', r.variant ?? '3B', '--type', r.quant ?? 'F16',
    '--steps', String(r.steps), '--rank', String(r.rank), '--alpha', String(r.alpha), '--lr', String(r.learningRate),
    '--grad-accum', String(r.gradAccum), '--seed', String(r.seed), '--save-every', String(r.checkpointEvery),
    // 12.5 codec frames per second. Runs from before the end-of-song frame keep the random
    // windows they were trained with; newer ones crop from each song's start (heartmula-train-state.h).
    ...(r.cropSeconds ? [r.eos ? '--crop-frames' : '--max-frames', String(Math.round(r.cropSeconds * 12.5))] : []),
    ...(r.eos ? ['--eos'] : []),
    // auto picks the fastest attention the card has and keeps as many blocks un-recomputed as the free VRAM allows,
    // so the same run fits a small card and is quick on a big one. Losses are identical in every mode.
    '--attn', 'auto', '--keep-layers', 'auto', '--lora-dropout', String(r.loraDropout ?? 0),
    ...(r.trainMlp ? ['--lora-mlp'] : []),
    ...(o.trigger?.trim() ? ['--trigger', o.trigger.trim()] : []),
    '--name', o.name, ...(o.stopAfter ? ['--stop-after', String(o.stopAfter)] : []), ...(o.resume ? ['--resume'] : [])];
}
