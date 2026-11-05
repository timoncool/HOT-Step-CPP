// training/sa3Preview.ts — SA3 checkpoint-preview settings, records and renders.
//
// One record per (step, prompt, seed) cell, kept in <run>/previews/index.json
// beside the WAVs (types in contracts/training.ts). Audio is written first and
// the record committed after, so a crash never leaves a record pointing at a
// partial file. Renders go to /sa3-synth with snapshot_run/snapshot_step: the
// engine resolves and validates the captured snapshots/step-N.gguf itself
// (engine/src/sa3-preview-resolver.h), so no filesystem path crosses the wire.
// The runner (trainSa3Runner) owns the engine around these calls.
import fs from 'node:fs';
import path from 'node:path';
import { randomBytes, randomUUID } from 'node:crypto';
import { config } from '../../config.js';
import type { Sa3PreviewRecord, Sa3PreviewSettings, Sa3RunId } from '../../contracts/training.js';
import { tokenizeForSa3 } from '../sa3Tokenizer.js';
import { SA3_DEFAULT_STEPS } from '../backends/sa3/generate.js';

export const SA3_PREVIEW_DEFAULTS: Sa3PreviewSettings = { everySteps: 0, prompts: [], seconds: 30, seeds: [424242] };
const MAX_PROMPTS = 8, MAX_SEEDS = 4, MAX_RECORDS = 2048, RENDER_TIMEOUT_MS = 30 * 60_000;

/** On-disk record: the wire record minus its URL, plus the audio file name. */
export type Sa3StoredPreview = Omit<Sa3PreviewRecord, 'audioUrl'> & { file?: string };

/** Validate preview settings over `base`; throws on anything out of range. */
export function parseSa3PreviewSettings(raw: unknown, base: Sa3PreviewSettings = SA3_PREVIEW_DEFAULTS): Sa3PreviewSettings {
  if (raw === undefined) return base;
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('SA3 preview settings must be an object');
  const b = raw as Record<string, unknown>;
  for (const key of Object.keys(b)) if (!['everySteps', 'prompts', 'seconds', 'seeds'].includes(key)) throw new Error(`Unsupported SA3 preview option: ${key}`);
  const everySteps = b.everySteps ?? base.everySteps, seconds = b.seconds ?? base.seconds;
  if (typeof everySteps !== 'number' || !Number.isSafeInteger(everySteps) || everySteps < 0 || everySteps > 1_000_000) throw new Error('Invalid SA3 preview everySteps');
  if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds < 1 || seconds > 384) throw new Error('Invalid SA3 preview seconds');
  const prompts = b.prompts ?? base.prompts, seeds = b.seeds ?? base.seeds;
  if (!Array.isArray(prompts) || prompts.length > MAX_PROMPTS
    || prompts.some(p => typeof p !== 'string' || !p.trim() || p.length > 2000)) throw new Error('Invalid SA3 preview prompts');
  if (!Array.isArray(seeds) || !seeds.length || seeds.length > MAX_SEEDS
    || seeds.some(s => typeof s !== 'number' || !Number.isSafeInteger(s) || s < 0 || s > 0x7fffffff)) throw new Error('Invalid SA3 preview seeds');
  if (everySteps > 0 && !prompts.length) throw new Error('SA3 checkpoint previews need at least one prompt');
  return { everySteps, prompts: prompts.map(p => (p as string).trim()), seconds, seeds: [...seeds] as number[] };
}

/** Runs whose previews are rendering in this process. A queued or rendering
 *  record of any other run was cut off (cancel, crash or server restart). */
const live = new Set<string>();

function indexPath(runDir: string): string { return path.join(runDir, 'previews', 'index.json'); }

function readRecords(runDir: string): Sa3StoredPreview[] {
  try {
    const value = JSON.parse(fs.readFileSync(indexPath(runDir), 'utf8')) as unknown;
    return Array.isArray(value) ? value.filter((r): r is Sa3StoredPreview => !!r && typeof r === 'object'
      && Number.isSafeInteger(r.step) && Number.isSafeInteger(r.promptIndex) && typeof r.prompt === 'string'
      && Number.isSafeInteger(r.seed) && typeof r.seconds === 'number' && typeof r.status === 'string') : [];
  } catch { return []; }
}

function writeRecords(runDir: string, records: Sa3StoredPreview[]): void {
  fs.mkdirSync(path.dirname(indexPath(runDir)), { recursive: true });
  const temp = `${indexPath(runDir)}.${randomUUID()}.tmp`;
  fs.writeFileSync(temp, `${JSON.stringify(records.slice(-MAX_RECORDS), null, 2)}\n`, 'utf8');
  fs.renameSync(temp, indexPath(runDir));
}

const sameCell = (a: Sa3StoredPreview, b: Sa3StoredPreview) => a.step === b.step && a.promptIndex === b.promptIndex
  && a.prompt === b.prompt && a.seed === b.seed && a.seconds === b.seconds;

function upsert(runDir: string, record: Sa3StoredPreview): void {
  writeRecords(runDir, [...readRecords(runDir).filter(r => !sameCell(r, record)), record]);
}

/** Records, newest step first. Cut-off renders read as failed, so they can be retried. */
export function listSa3PreviewRecords(runDir: string): Sa3StoredPreview[] {
  const alive = live.has(path.resolve(runDir));
  return readRecords(runDir).map(r => !alive && (r.status === 'queued' || r.status === 'rendering')
    ? { ...r, status: 'failed' as const, error: 'Interrupted before it finished' } : r)
    .sort((a, b) => b.step - a.step || a.promptIndex - b.promptIndex || a.seed - b.seed);
}

function cells(runId: Sa3RunId, step: number, settings: Sa3PreviewSettings, final: boolean): Sa3StoredPreview[] {
  return settings.prompts.flatMap((prompt, promptIndex) => settings.seeds.map(seed => ({ run: runId, step, promptIndex, prompt, seed,
    seconds: settings.seconds, status: 'queued' as const, error: null, final, createdAt: Date.now() })));
}

/** True when some cell of `step` has no finished render (done or failed). A
 *  skipped or cut-off cell is still owed. */
export function sa3PreviewPending(runDir: string, runId: Sa3RunId, step: number, settings: Sa3PreviewSettings): boolean {
  // Raw statuses: the list view shows a cut-off render as failed, but it is still owed.
  const records = readRecords(runDir);
  return cells(runId, step, settings, false).some(c => !records.some(r => sameCell(r, c) && (r.status === 'done' || r.status === 'failed')));
}

/** <run>/previews with links resolved, or null when it resolves outside the run. */
function previewsDir(runDir: string): string | null {
  try {
    const run = fs.realpathSync(runDir), dir = fs.realpathSync(path.join(runDir, 'previews'));
    return dir === path.join(run, 'previews') ? dir : null;
  } catch { return null; }
}

/** The audio of a done record, or null for any other name, and for a
 *  previews folder or file that a junction or symlink takes outside the run. */
export function sa3PreviewAudioPath(runDir: string, file: string): string | null {
  if (!/^step-[0-9]+-p[0-9]+-s[0-9]+-[a-f0-9]{8}\.wav$/.test(file)) return null;
  if (!readRecords(runDir).some(r => r.status === 'done' && r.file === file)) return null;
  const dir = previewsDir(runDir);
  if (!dir) return null;
  try {
    const real = fs.realpathSync(path.join(dir, file));
    return path.dirname(real) === dir && fs.statSync(real).isFile() ? real : null;
  } catch { return null; }
}

export interface Sa3PreviewDeps {
  tokenize: typeof tokenizeForSa3;
  synth: (query: URLSearchParams, signal: AbortSignal) => Promise<Buffer>;
}

async function engineSynth(query: URLSearchParams, signal: AbortSignal): Promise<Buffer> {
  const res = await fetch(`${config.aceServer.url}/sa3-synth?${query}`, { method: 'POST', signal });
  if (!res.ok) throw new Error(`SA3 preview render failed (${res.status}): ${(await res.text().catch(() => '')).slice(0, 300)}`);
  return Buffer.from(await res.arrayBuffer());
}

export interface Sa3PreviewInput {
  runDir: string; runId: Sa3RunId;
  /** `<dataset slug>/<run id>`: the engine's snapshot_run. */
  snapshotRun: string; step: number; settings: Sa3PreviewSettings;
  /** File name of the generation DiT the snapshot was exported against. */
  inferenceDit: string; final?: boolean; signal?: AbortSignal; deps?: Partial<Sa3PreviewDeps>;
}

/**
 * Render every (prompt, seed) cell of one captured step that has no done
 * record yet, one at a time. A done cell is never rendered twice, which is
 * what keeps the final preview from repeating a cadence preview. A failed
 * cell is recorded and the next one still renders; a cancel marks the rest
 * skipped and stops. Needs the engine up.
 */
export async function renderSa3Previews(input: Sa3PreviewInput): Promise<{ done: number; failed: number; skipped: number }> {
  const deps: Sa3PreviewDeps = { tokenize: tokenizeForSa3, synth: engineSynth, ...input.deps };
  const key = path.resolve(input.runDir), counts = { done: 0, failed: 0, skipped: 0 };
  const existing = readRecords(input.runDir);
  const todo = cells(input.runId, input.step, input.settings, !!input.final)
    .filter(c => !existing.some(r => sameCell(r, c) && r.status === 'done' && r.file && sa3PreviewAudioPath(input.runDir, r.file)));
  if (!todo.length) return counts;
  live.add(key);
  try {
    for (const c of todo) upsert(input.runDir, c);
    for (const c of todo) {
      if (input.signal?.aborted) { upsert(input.runDir, { ...c, status: 'skipped', error: 'Cancelled' }); counts.skipped++; continue; }
      upsert(input.runDir, { ...c, status: 'rendering' });
      try {
        const tokens = await deps.tokenize(c.prompt);
        const query = new URLSearchParams({ tokens: tokens.ids.join(','), n_tokens: String(tokens.nTokens),
          duration: String(c.seconds), seed: String(c.seed), steps: String(SA3_DEFAULT_STEPS),
          synth_model: input.inferenceDit, snapshot_run: input.snapshotRun, snapshot_step: String(c.step) });
        const signal = input.signal ? AbortSignal.any([input.signal, AbortSignal.timeout(RENDER_TIMEOUT_MS)]) : AbortSignal.timeout(RENDER_TIMEOUT_MS);
        const wav = await deps.synth(query, signal);
        if (wav.length < 44 || wav.toString('ascii', 0, 4) !== 'RIFF') throw new Error('SA3 preview render returned no WAV');
        const file = `step-${c.step}-p${c.promptIndex}-s${c.seed}-${randomBytes(4).toString('hex')}.wav`;
        const dir = previewsDir(input.runDir);
        if (!dir) throw new Error('SA3 previews folder resolves outside the run');
        fs.writeFileSync(path.join(dir, file), wav, { flag: 'wx' });
        upsert(input.runDir, { ...c, status: 'done', file });
        counts.done++;
      } catch (err: any) {
        const cancelled = !!input.signal?.aborted;
        upsert(input.runDir, { ...c, status: cancelled ? 'skipped' : 'failed', error: cancelled ? 'Cancelled' : err?.message || String(err) });
        counts[cancelled ? 'skipped' : 'failed']++;
      }
    }
  } finally { live.delete(key); }
  return counts;
}
