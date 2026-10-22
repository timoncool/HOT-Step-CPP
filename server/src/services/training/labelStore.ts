// training/labelStore.ts — studio-private per-sample state
//
// data/training/datasets/<slug>/labels/<sampleId>.json holds everything
// Side-Step does not understand: audio_codes, the raw Essentia and /understand
// results, provenance, exclude flags, per-file label status and the duration
// cache (D6/D7).
//
// `audioCodes` is the phase-4 payload — storing it now means LM-LoRA training
// can skip the FSQ tokenizer pass for already-labeled datasets. It must NEVER
// enter the sidecar: `audio_codes` is not a known sidecar key, so Side-Step's
// parser would glue it onto the previous field's value and corrupt it.
//
// Spec: docs/plans/2026-07-27-dataset-studio-implementation.md §4.4

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { labelsDir, datasetDir } from './paths.js';
import type { FieldSource, SampleLabelStatus } from './types.js';

export interface SampleLabelRecord {
  sampleId: string;
  relPath: string;
  excluded: boolean;
  labelStatus: SampleLabelStatus;
  error: string | null;
  labeledAt: string;
  sources: Partial<Record<string, FieldSource>>;
  audioCodes: string | null;        // comma-separated FSQ codes from /understand
  audioCodesModel: string | null;   // which LM/DiT produced them
  essentia: {                       // raw subset, kept for later use
    bpm: number | null; key: string; scale: string;
    loudness: number | null; danceability: number | null;
  } | null;
  essentiaTempo?: { minTempo?: number; maxTempo?: number } | null;
  understand: {                     // raw /understand result, unmerged
    caption: string; lyrics: string; bpm: number | null;
    keyscale: string; timesignature: string; vocalLanguage: string;
    duration: number | null; seed: number | null;
  } | null;
  durationCache: { seconds: number; sizeBytes: number; mtimeMs: number } | null;
  tags: { artist: string; title: string; album: string } | null;
  /** Epoch ms of the first scan that could not find this sample's audio file.
   *  Orphans are only pruned after two consecutive misses (§7.4). */
  missingSince?: number | null;
  /** Why this sample ended up with no lyrics, or null when it has them.
   *
   *  Per-sample Genius outcomes only ever went to the job's SSE stream, which
   *  nobody is listening to during a bulk run and which is never persisted —
   *  the job _meta.json keeps counts alone. So 166 blank-lyric tracks across 36
   *  albums had to be diagnosed by re-querying the API by hand afterwards
   *  (2026-08-05). Keep the reason with the sample it belongs to. */
  lyricsError?: string | null;
}

/** A blank record — every field explicit so partial writes stay predictable. */
export function emptyLabel(sampleId: string, relPath: string): SampleLabelRecord {
  return {
    sampleId,
    relPath,
    excluded: false,
    labelStatus: 'unlabeled',
    error: null,
    labeledAt: '',
    sources: {},
    audioCodes: null,
    audioCodesModel: null,
    essentia: null,
    understand: null,
    durationCache: null,
    tags: null,
    missingSince: null,
    lyricsError: null,
  };
}

function labelPath(slug: string, sampleId: string): string {
  // sampleId is always a server-generated sha1 hex slice; guard anyway.
  const safe = String(sampleId).replace(/[^a-f0-9]/gi, '').slice(0, 32);
  return path.join(labelsDir(slug), `${safe}.json`);
}

export function readLabel(slug: string, sampleId: string): SampleLabelRecord | null {
  try {
    const p = labelPath(slug, sampleId);
    if (!fs.existsSync(p)) return null;
    const parsed = JSON.parse(fs.readFileSync(p, 'utf-8')) as Partial<SampleLabelRecord>;
    if (!parsed || typeof parsed !== 'object') return null;
    return { ...emptyLabel(sampleId, parsed.relPath || ''), ...parsed, sampleId };
  } catch {
    return null;
  }
}

/** Atomic write (temp + rename) into the dataset's labels dir. */
export function writeLabel(slug: string, rec: SampleLabelRecord): void {
  const dir = labelsDir(slug);
  fs.mkdirSync(dir, { recursive: true });
  const target = labelPath(slug, rec.sampleId);
  const tmp = path.join(dir, `.label_${crypto.randomBytes(6).toString('hex')}.tmp`);
  try {
    fs.writeFileSync(tmp, JSON.stringify(rec, null, 2), 'utf-8');
    fs.renameSync(tmp, target);
  } catch (err) {
    try { fs.unlinkSync(tmp); } catch { /* already gone */ }
    throw err;
  }
}

export function patchLabel(
  slug: string,
  sampleId: string,
  patch: Partial<SampleLabelRecord>,
): SampleLabelRecord {
  const existing = readLabel(slug, sampleId) ?? emptyLabel(sampleId, patch.relPath || '');
  const merged: SampleLabelRecord = {
    ...existing,
    ...patch,
    sampleId,
    sources: { ...existing.sources, ...(patch.sources || {}) },
  };
  writeLabel(slug, merged);
  return merged;
}

/** One readdir pass over labels/ — used by every dataset read. */
export function readAllLabels(slug: string): Map<string, SampleLabelRecord> {
  const out = new Map<string, SampleLabelRecord>();
  const dir = labelsDir(slug);
  let entries: string[];
  try {
    if (!fs.existsSync(dir)) return out;
    entries = fs.readdirSync(dir);
  } catch {
    return out;
  }
  for (const name of entries) {
    if (!name.endsWith('.json')) continue;
    const sampleId = name.slice(0, -5);
    const rec = readLabel(slug, sampleId);
    if (rec) out.set(sampleId, rec);
  }
  return out;
}

export function deleteLabel(slug: string, sampleId: string): void {
  try { fs.unlinkSync(labelPath(slug, sampleId)); } catch { /* already gone */ }
}

// ── Transient job status (in-memory ONLY) ────────────────────────────────
//
// §2.0 defines 'pending' as "queued in an active job" and 'processing' as
// "currently being labeled" — both describe a LIVE job, so neither may ever be
// persisted. Writing them into labels/<sampleId>.json would freeze the row
// read-only forever after a cancel, a crash or a server restart, because
// nothing on disk would clear it again.

type TransientStatus = 'pending' | 'processing';

const transientStatus = new Map<string, TransientStatus>();

function transientKey(datasetId: string, sampleId: string): string {
  return `${datasetId}:${sampleId}`;
}

export function setTransientStatus(datasetId: string, sampleId: string, status: TransientStatus): void {
  transientStatus.set(transientKey(datasetId, sampleId), status);
}

export function clearTransientStatus(datasetId: string, sampleId: string): void {
  transientStatus.delete(transientKey(datasetId, sampleId));
}

/** Drop every transient marker a job owns — called when it ends, however it ends. */
export function clearTransientStatuses(datasetId: string, sampleIds: readonly string[]): void {
  for (const sampleId of sampleIds) transientStatus.delete(transientKey(datasetId, sampleId));
}

export function getTransientStatus(datasetId: string, sampleId: string): TransientStatus | null {
  return transientStatus.get(transientKey(datasetId, sampleId)) ?? null;
}

/** Remove the whole studio-private directory for a dataset (D20). */
export function deleteLabels(slug: string): void {
  try {
    fs.rmSync(datasetDir(slug), { recursive: true, force: true });
  } catch (err: any) {
    console.warn(`[Training] Could not remove ${datasetDir(slug)}: ${err.message}`);
  }
}

/** Persist the dataset-level meta blob (D6). Best-effort, never throws. */
export function writeDatasetMeta(slug: string, meta: unknown): void {
  try {
    const dir = datasetDir(slug);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'dataset_meta.json'), JSON.stringify(meta, null, 2), 'utf-8');
  } catch (err: any) {
    console.warn(`[Training] Could not write dataset_meta.json for ${slug}: ${err.message}`);
  }
}
