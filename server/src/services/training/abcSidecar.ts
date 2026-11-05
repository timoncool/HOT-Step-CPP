import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { stemPathFor } from './paths.js';
import { backupPath } from './sidecarIO.js';
import { hasSoundingScore, hasSoundingVocalBar } from '../backends/yue2/scoreHealth.js';

export function abcSidecarPath(audioPath: string): string {
  return `${stemPathFor(audioPath)}.abc`;
}

export function readAbcSidecar(audioPath: string): string {
  try { return fs.readFileSync(abcSidecarPath(audioPath), 'utf8').trim(); }
  catch { return ''; }
}

/** The score's opening key in the caption format (`K:F#m` → `F# Minor`), '' when unreadable. */
export function abcKeyString(abc: string): string {
  const m = abc.match(/^K:\s*([A-G][#b]?)(m|min|minor|maj|major)?\s*$/m);
  if (!m) return '';
  return `${m[1]} ${m[2] && !m[2].startsWith('maj') ? 'Minor' : 'Major'}`;
}

/** The score with its header tempo (`Q:1/4=N`) set to a measured `bpm`, at the
 *  score's own beat level: a score written at 80 stays near 80 when the
 *  measurement counts the same song at 157, or its notes would play twice as
 *  fast. Unchanged when it has no tempo. */
export function withAbcTempo(abc: string, bpm: number): string {
  return abc.replace(/^(Q:\s*1\/4\s*=\s*)(\d+(?:\.\d+)?)/m, (_all, head: string, q: string) => {
    const written = Number(q);
    const near = [bpm / 2, bpm, bpm * 2].reduce((a, b) => Math.abs(Math.log(b / written)) < Math.abs(Math.log(a / written)) ? b : a);
    return `${head}${Math.round(near)}`;
  });
}

/** A failed or empty transcription never replaces a usable score. With
 *  `keepBackups` the score it replaces is kept (see backupPath), and no write
 *  happens without that copy. */
export function writeAbcSidecar(audioPath: string, abc: string, opts: { keepBackups?: boolean; anySounding?: boolean } = {}): boolean {
  if (!(opts.anySounding ? hasSoundingScore(abc) : hasSoundingVocalBar(abc))) return false;
  const target = abcSidecarPath(audioPath);
  if (opts.keepBackups && fs.existsSync(target)) fs.copyFileSync(target, backupPath(target, true));
  const tmp = path.join(path.dirname(target), `.abc_${crypto.randomBytes(6).toString('hex')}.tmp`);
  try {
    fs.writeFileSync(tmp, `${abc.trim()}\n`, 'utf8');
    fs.renameSync(tmp, target);
    return true;
  } finally {
    try { fs.unlinkSync(tmp); } catch { /* renamed or never created */ }
  }
}
