// training/heartmulaPrepare.ts — what the dataset hands `ace-train heartmula-prepare`,
// and what the server reads back from its manifest.
//
// The engine owns the cache fingerprint (codes keyed on audio bytes, mtime and
// encoder hashes). The server only needs to know, without starting the engine,
// whether a prepared sample still matches its source: the contract's rule is
// "size or whole-second mtime of the audio, or tags or lyrics, differ".

import fs from 'node:fs';
import path from 'node:path';
import type {
  HeartmulaPrepareStatus, HeartmulaSkipReason, HeartmulaSkippedSample,
} from '../../contracts/heartmulaTraining.js';
import type { TrainingDatasetRow, TrainingSample } from './types.js';
import { heartmulaPrepareDir } from './heartmulaRuns.js';

export const HEARTMULA_PREPARE_MANIFEST = 'heartmula_prepare.json';
const SKIP_REASONS: readonly string[] = ['invalid', 'over-context', 'no-audio', 'encode-failed'];

export interface HeartmulaSampleInput { id: string; audio: string; tags: string; lyrics: string }

/** Samples that can train: not excluded, audio on disk. */
export const trainableSamples = (samples: TrainingSample[]): TrainingSample[] =>
  samples.filter(s => !s.excluded && !s.fileMissing);

/**
 * Tags are the caption and genre as comma-separated tags, led by the trigger
 * word (the sample's own, else the dataset's). Lyrics are empty for an
 * instrumental. The same text goes to prepare and to the stale check.
 */
export function heartmulaSampleInput(ds: Pick<TrainingDatasetRow, 'customTag'>, s: TrainingSample): HeartmulaSampleInput {
  const parts = [s.customTag || ds.customTag || '', s.caption, s.genre].map(p => p.trim()).filter(Boolean);
  const seen = new Set<string>();
  const tags = parts.filter(p => !seen.has(p.toLowerCase()) && seen.add(p.toLowerCase())).join(', ');
  return { id: s.sampleId, audio: s.audioPath, tags, lyrics: s.isInstrumental ? '' : s.lyrics.trim() };
}

/** Write the sample list the engine reads; returns its path and the count. */
export function writeHeartmulaSamples(ds: TrainingDatasetRow, samples: TrainingSample[]): { file: string; count: number } {
  const dir = heartmulaPrepareDir(ds.slug);
  fs.mkdirSync(dir, { recursive: true });
  const list = trainableSamples(samples).map(s => heartmulaSampleInput(ds, s));
  const file = path.join(dir, 'samples.json');
  fs.writeFileSync(`${file}.part`, JSON.stringify({ samples: list }), 'utf8');
  fs.renameSync(`${file}.part`, file);
  return { file, count: list.length };
}

interface ManifestRow { id: string; audio: string; audio_bytes: number; audio_mtime_s: number; tags: string; lyrics: string }
interface Manifest {
  version: number; created_at?: string; context_limit?: number;
  samples: ManifestRow[]; skipped?: { id: string; reason: string; detail: string }[];
}

export function readHeartmulaManifest(slug: string): Manifest | null {
  try {
    const m = JSON.parse(fs.readFileSync(path.join(heartmulaPrepareDir(slug), HEARTMULA_PREPARE_MANIFEST), 'utf8')) as Manifest;
    return Array.isArray(m.samples) ? m : null;
  } catch { return null; }
}

/** A sample's source as the stale check sees it: audio size and whole-second mtime, tags and lyrics. */
type Fingerprint = Pick<ManifestRow, 'audio_bytes' | 'audio_mtime_s' | 'tags' | 'lyrics'>;

function fingerprint(input: HeartmulaSampleInput): Fingerprint | null {
  try {
    const st = fs.statSync(input.audio);
    return { audio_bytes: st.size, audio_mtime_s: Math.floor(st.mtimeMs / 1000), tags: input.tags, lyrics: input.lyrics };
  } catch { return null; }
}

const sameSource = (a: Fingerprint | null, b: Fingerprint | undefined): boolean =>
  !!a && !!b && a.audio_bytes === b.audio_bytes && a.audio_mtime_s === b.audio_mtime_s && a.tags === b.tags && a.lyrics === b.lyrics;

const SKIPPED_FILE = 'skipped-sources.json';

/**
 * Record the sources of the samples the last prepare skipped, so a corrected
 * sample reads as stale instead of staying hidden behind its skip. The runner
 * calls this after a successful prepare.
 */
export type SourceSnapshot = Record<string, Fingerprint>;

/** Every trainable sample's source now. The runner takes it BEFORE spawning Prepare, so a file replaced while Prepare runs cannot pass for the version it skipped. */
export function snapshotSources(ds: Pick<TrainingDatasetRow, 'customTag'>, samples: TrainingSample[]): SourceSnapshot {
  const out: SourceSnapshot = {};
  for (const s of trainableSamples(samples)) {
    const fp = fingerprint(heartmulaSampleInput(ds, s));
    if (fp) out[s.sampleId] = fp;
  }
  return out;
}

/** Keep the pre-Prepare fingerprints of the samples the completed manifest skipped. */
export function writeSkippedFingerprints(slug: string, before: SourceSnapshot): void {
  const m = readHeartmulaManifest(slug);
  if (!m) return;
  const out: SourceSnapshot = {};
  for (const { id } of m.skipped ?? []) if (before[id]) out[id] = before[id];
  fs.writeFileSync(path.join(heartmulaPrepareDir(slug), SKIPPED_FILE), JSON.stringify(out), 'utf8');
}

function readSkippedFingerprints(slug: string): Record<string, Fingerprint> {
  try { return JSON.parse(fs.readFileSync(path.join(heartmulaPrepareDir(slug), SKIPPED_FILE), 'utf8')); } catch { return {}; }
}

/**
 * Prepare state of a dataset, from the manifest and the live samples. Stale =
 * a prepared sample whose source changed or went away; a sample the dataset
 * has that the manifest lacks makes the cache invalid too. `cacheValid` needs
 * at least one prepared sample and nothing stale or missing.
 */
export function heartmulaPrepareStatus(ds: TrainingDatasetRow, samples: TrainingSample[]): HeartmulaPrepareStatus {
  const m = readHeartmulaManifest(ds.slug);
  if (!m) return { cacheValid: false, cacheVersion: null, preparedSamples: 0, staleSamples: 0, skipped: [], contextLimit: null, preparedAt: null };
  const live = new Map(trainableSamples(samples).map(s => [s.sampleId, heartmulaSampleInput(ds, s)]));
  const stale = m.samples.filter(r => {
    const input = live.get(r.id);
    return !sameSource(input ? fingerprint(input) : null, r);
  }).length;
  const skippedIds = new Set((m.skipped ?? []).map(s => s.id));
  const unseen = [...live.keys()].filter(id => !m.samples.some(r => r.id === id) && !skippedIds.has(id)).length;
  // A skipped sample is fine while its source is what was skipped. One with no
  // record (a cache from before records existed) counts as changed, conservatively.
  const recorded = readSkippedFingerprints(ds.slug);
  const changedSkips = [...skippedIds].filter(id => live.has(id) && !sameSource(fingerprint(live.get(id)!), recorded[id])).length;
  const skipped: HeartmulaSkippedSample[] = (m.skipped ?? []).map(s => ({
    sampleId: s.id, reason: (SKIP_REASONS.includes(s.reason) ? s.reason : 'invalid') as HeartmulaSkipReason, detail: s.detail,
  }));
  return {
    cacheValid: m.samples.length > 0 && stale === 0 && unseen === 0 && changedSkips === 0,
    cacheVersion: m.version, preparedSamples: m.samples.length, staleSamples: stale + unseen + changedSkips, skipped,
    contextLimit: m.context_limit ?? null, preparedAt: m.created_at ?? null,
  };
}
