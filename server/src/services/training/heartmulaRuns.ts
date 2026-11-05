// training/heartmulaRuns.ts — on-disk layout of HeartMuLa training runs.
//
//   <datasetDir>/heartmula/prepare/        heartmula-prepare output (+ samples.json)
//   <adapters>/heartmula-runs/<slug>/<adapterName>/<stamp>/
//       run.json            what the run was asked to do (recipe, dataset)
//       status.json         what the runner has seen (HeartmulaRunStatus minus run/updatedAt)
//       hm_train_state.bin, adapters/step-NNNNNN.gguf, control/   the trainer's own files
//   <adapters>/heartmula-lm-adapters/<adapterName>-<stamp>.gguf   the published adapter
//
// Checkpoints stay under heartmula-runs, outside the folder the generation
// dropdown scans, so only a finished run reaches the picker.

import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { config } from '../../config.js';
import { datasetDir, slugify } from './paths.js';
import { newRunDir } from './adapterLayout.js';
import { resolveHeartmulaPreviewSettings, resolveHeartmulaRecipe } from './heartmulaTrain.js';
import { readHeartmulaManifest } from './heartmulaPrepare.js';
import type {
  HeartmulaAdapterAsset, HeartmulaLossPoint, HeartmulaPreviewSource, HeartmulaRecipe, HeartmulaRunId, HeartmulaRunState, HeartmulaRunStatus,
} from '../../contracts/heartmulaTraining.js';
import { HEARTMULA_RECIPE_DEFAULTS, HEARTMULA_RECIPE_LEGACY } from '../../contracts/heartmulaTraining.js';

/** `<adapterName>/<timestamp>`, the id the routes and `resumeRun` take. */
export const HEARTMULA_RUN_ID = /^[a-z0-9][a-z0-9_-]{0,63}\/[0-9][0-9_-]*$/i;
export const HEARTMULA_ADAPTER_NAME = /^[a-z0-9][a-z0-9_-]{0,63}$/i;

export const heartmulaPrepareDir = (slug: string): string => path.join(datasetDir(slug), 'heartmula', 'prepare');
export const heartmulaAdapterRoot = (): string => path.join(config.aceServer.adapters, 'heartmula-lm-adapters');

export function heartmulaRunsRoot(slug: string): string {
  return path.join(config.aceServer.adapters, 'heartmula-runs', slugify(slug));
}

export interface HeartmulaRunRecord {
  datasetId: string;
  adapterName: string;
  recipe: HeartmulaRecipe;
  /** ISO time the run was created. */
  createdAt: string;
  /** Resolved checkpoint-preview settings the run started with (HeartmulaPreviewSettings); absent on runs made before previews. */
  preview?: unknown;
  /** Where `preview`'s tags and lyrics came from (HeartmulaPreviewSource); set when the run resolved them, absent before. */
  previewSource?: HeartmulaPreviewSource;
  /** Published adapter, relative to the adapter root; set once the run finished. */
  published?: string;
}

type StoredStatus = Omit<HeartmulaRunStatus, 'run' | 'updatedAt'>;

function writeJson(file: string, value: unknown): void {
  const part = `${file}.${randomUUID()}.part`;
  fs.writeFileSync(part, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  fs.renameSync(part, file);
}

function readJson<T>(file: string): T | null {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')) as T; } catch { return null; }
}

/** Reserve a fresh run folder and write its record. */
export function createHeartmulaRun(
  datasetId: string, slug: string, adapterName: string, recipe: HeartmulaRecipe, preview?: unknown,
): { id: HeartmulaRunId; dir: string; record: HeartmulaRunRecord } {
  if (!HEARTMULA_ADAPTER_NAME.test(adapterName)) throw new Error('adapterName must match [A-Za-z0-9_-]{1,64}');
  const dir = newRunDir(path.join(heartmulaRunsRoot(slug), adapterName));
  fs.mkdirSync(dir, { recursive: true });
  const record: HeartmulaRunRecord = { datasetId, adapterName, recipe, createdAt: new Date().toISOString(), ...(preview === undefined ? {} : { preview }) };
  writeJson(path.join(dir, 'run.json'), record);
  return { id: `${adapterName}/${path.basename(dir)}`, dir, record };
}

/** Resolve a run id under this dataset's runs; refuses ids that escape the root or belong to another dataset. */
export function heartmulaRunById(datasetId: string, slug: string, id: string): { id: HeartmulaRunId; dir: string; record: HeartmulaRunRecord } {
  if (!HEARTMULA_RUN_ID.test(id)) throw new Error('Invalid HeartMuLa run');
  const root = heartmulaRunsRoot(slug);
  const dir = path.join(root, ...id.split('/'));
  let realRoot: string, realDir: string;
  try { realRoot = fs.realpathSync(root); realDir = fs.realpathSync(dir); } catch { throw new Error('HeartMuLa run not found'); }
  if (!realDir.startsWith(realRoot + path.sep)) throw new Error('HeartMuLa run is outside the dataset runs');
  const record = readJson<HeartmulaRunRecord>(path.join(dir, 'run.json'));
  if (!record) throw new Error('HeartMuLa run not found');
  if (record.datasetId !== datasetId) throw new Error('HeartMuLa run belongs to another dataset');
  return { id, dir, record };
}

/** Every run of the dataset, newest first. */
export function listHeartmulaRuns(datasetId: string, slug: string): { id: HeartmulaRunId; dir: string; record: HeartmulaRunRecord }[] {
  const root = heartmulaRunsRoot(slug);
  const out: { id: HeartmulaRunId; dir: string; record: HeartmulaRunRecord }[] = [];
  const dirs = (p: string): string[] => {
    try { return fs.readdirSync(p, { withFileTypes: true }).filter(e => e.isDirectory()).map(e => e.name); } catch { return []; }
  };
  for (const name of dirs(root)) {
    for (const stamp of dirs(path.join(root, name))) {
      const dir = path.join(root, name, stamp);
      const record = readJson<HeartmulaRunRecord>(path.join(dir, 'run.json'));
      if (record && record.datasetId === datasetId) out.push({ id: `${name}/${stamp}`, dir, record });
    }
  }
  return out.sort((a, b) => b.id.split('/')[1].localeCompare(a.id.split('/')[1]));
}

export function saveHeartmulaRunRecord(dir: string, record: HeartmulaRunRecord): void {
  writeJson(path.join(dir, 'run.json'), record);
}

/**
 * Admission for POST /heartmula-train: a new run folder, or the folder of the
 * run to resume. A resume keeps the saved recipe (the trainer refuses changed
 * settings) and needs a checkpoint to continue from.
 */
export function resolveHeartmulaRun(
  ds: { id: string; slug: string }, body: Record<string, unknown>,
): { id: HeartmulaRunId; dir: string; record: HeartmulaRunRecord; resume: boolean } {
  for (const key of Object.keys(body)) {
    if (!['adapterName', 'recipe', 'preview', 'resumeRun'].includes(key)) throw new Error(`Unsupported HeartMuLa option: ${key}`);
  }
  const given = body.recipe && typeof body.recipe === 'object' && !Array.isArray(body.recipe) ? body.recipe as Record<string, unknown> : undefined;
  if (body.recipe !== undefined && !given) throw new Error('recipe must be an object');
  if (body.resumeRun !== undefined) {
    if (typeof body.resumeRun !== 'string') throw new Error('Invalid HeartMuLa resumeRun');
    if (body.preview !== undefined) throw new Error('A resume keeps the preview settings the run started with');
    const run = heartmulaRunById(ds.id, ds.slug, body.resumeRun);
    if (body.adapterName !== undefined && body.adapterName !== run.record.adapterName) throw new Error('A resume cannot change adapterName');
    for (const [key, value] of Object.entries(given ?? {})) {
      // A run saved before a recipe key existed ran with that key's default.
      const saved = (run.record.recipe as unknown as Record<string, unknown>)[key]
        ?? (HEARTMULA_RECIPE_LEGACY as unknown as Record<string, unknown>)[key]
        ?? (HEARTMULA_RECIPE_DEFAULTS as unknown as Record<string, unknown>)[key];
      if (value !== undefined && saved !== value) throw new Error(`A resume cannot change ${key}`);
    }
    if (run.record.published) throw new Error('This run already finished and published its adapter');
    if (!fs.existsSync(path.join(run.dir, 'hm_train_state.bin'))) throw new Error('This run has no checkpoint to resume from');
    return { ...run, resume: true };
  }
  const adapterName = body.adapterName === undefined ? ds.slug : body.adapterName;
  if (typeof adapterName !== 'string' || !HEARTMULA_ADAPTER_NAME.test(adapterName)) throw new Error('adapterName must match [A-Za-z0-9_-]{1,64}');
  const recipe = resolveHeartmulaRecipe(given, readHeartmulaManifest(ds.slug)?.samples.length ?? 0);
  // Validated now, so a bad fixture refuses the start instead of failing a render hours in.
  const run = createHeartmulaRun(ds.id, ds.slug, adapterName, recipe, resolveHeartmulaPreviewSettings(body.preview));
  return { ...run, resume: false };
}

// ── status ───────────────────────────────────────────────────────────────

export const blankStatus = (totalSteps: number): StoredStatus => ({
  state: 'queued', step: 0, totalSteps, loss: null, lossHistory: [], msPerStep: null, peakVramMiB: null,
  checkpointStep: null, adapterFile: null, error: null, startedAt: null,
});

export function readStoredStatus(dir: string): StoredStatus | null {
  return readJson<StoredStatus>(path.join(dir, 'status.json'));
}

export function writeStoredStatus(dir: string, s: StoredStatus): void {
  writeJson(path.join(dir, 'status.json'), s);
}

const LIVE_STATES: readonly HeartmulaRunState[] = ['queued', 'running', 'previewing', 'checkpointing'];

/**
 * The run as the status route reports it. A stored live state with no job
 * behind it means the server stopped mid-run: shown as failed, and still
 * resumable from its last checkpoint.
 */
export function heartmulaRunStatus(
  run: { id: HeartmulaRunId; dir: string; record: HeartmulaRunRecord }, jobActive: boolean,
): HeartmulaRunStatus {
  const stored = readStoredStatus(run.dir) ?? blankStatus(run.record.recipe.steps);
  const orphaned = !jobActive && LIVE_STATES.includes(stored.state);
  let mtime = Date.now();
  try { mtime = fs.statSync(path.join(run.dir, 'status.json')).mtimeMs; } catch { /* fresh run */ }
  return { ...stored, run: run.id, updatedAt: mtime,
    ...(orphaned ? { state: 'failed' as const, error: stored.error ?? 'Interrupted: the server stopped during this run' } : {}) };
}

export function appendLoss(history: HeartmulaLossPoint[], step: number, loss: number): HeartmulaLossPoint[] {
  return [...history.filter(p => p.step !== step), { step, loss }];
}

// ── checkpoints and publication ──────────────────────────────────────────

const STEP_FILE = /^step-(\d{6})\.gguf$/;

/** Steps that have an immutable adapter file, ascending. */
export function heartmulaCheckpointSteps(dir: string): number[] {
  try {
    return fs.readdirSync(path.join(dir, 'adapters')).map(f => STEP_FILE.exec(f)).filter((m): m is RegExpExecArray => !!m)
      .map(m => Number(m[1])).sort((a, b) => a - b);
  } catch { return []; }
}

export const heartmulaAdapterFile = (dir: string, step: number): string =>
  path.join(dir, 'adapters', `step-${String(step).padStart(6, '0')}.gguf`);

/** Cheap structural check: a GGUF v3 file with at least one tensor. */
export function isGgufAdapter(file: string): boolean {
  let fd: number | undefined;
  try {
    fd = fs.openSync(file, 'r');
    const h = Buffer.alloc(24);
    return fs.readSync(fd, h, 0, 24, 0) === 24 && h.toString('ascii', 0, 4) === 'GGUF'
      && h.readUInt32LE(4) === 3 && h.readBigUInt64LE(8) > 0n;
  } catch { return false; } finally { if (fd !== undefined) fs.closeSync(fd); }
}

/** Copy a checkpoint into the adapter folder without overwriting; returns the path relative to that folder. */
export function publishHeartmulaAdapter(file: string, name: string): string {
  if (!isGgufAdapter(file)) throw new Error('The trainer wrote an invalid adapter file');
  const root = heartmulaAdapterRoot();
  fs.mkdirSync(root, { recursive: true });
  const temp = path.join(root, `.${randomUUID()}.part`);
  try {
    fs.copyFileSync(file, temp, fs.constants.COPYFILE_EXCL);
    for (let i = 1; i < 10_000; i++) {
      const dest = path.join(root, `${name}${i === 1 ? '' : `-${i}`}.gguf`);
      try { fs.linkSync(temp, dest); return path.basename(dest); } catch (err: any) { if (err.code !== 'EEXIST') throw err; }
    }
    throw new Error('HeartMuLa adapter name exhausted');
  } finally { fs.rmSync(temp, { force: true }); }
}

/** The newest published adapter of the dataset whose file is still there. */
export function latestHeartmulaAdapter(datasetId: string, slug: string): HeartmulaAdapterAsset | null {
  for (const run of listHeartmulaRuns(datasetId, slug)) {
    const pub = run.record.published;
    if (!pub) continue;
    try {
      const file = path.join(heartmulaAdapterRoot(), pub);
      if (isGgufAdapter(file)) return { name: pub.replace(/\.gguf$/i, ''), trainedAt: fs.statSync(file).mtime.toISOString(), run: run.id };
    } catch { /* file removed */ }
  }
  return null;
}
