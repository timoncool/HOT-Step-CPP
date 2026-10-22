import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { stemPathFor } from './paths.js';
import { backupPath } from './sidecarIO.js';
import { hasSoundingVocalBar } from '../backends/yue2/scoreHealth.js';

export function abcSidecarPath(audioPath: string): string {
  return `${stemPathFor(audioPath)}.abc`;
}

export function readAbcSidecar(audioPath: string): string {
  try { return fs.readFileSync(abcSidecarPath(audioPath), 'utf8').trim(); }
  catch { return ''; }
}

/** A failed or empty transcription never replaces a usable score. With
 *  `keepBackups` the score it replaces is kept (see backupPath), and no write
 *  happens without that copy. */
export function writeAbcSidecar(audioPath: string, abc: string, opts: { keepBackups?: boolean } = {}): boolean {
  if (!hasSoundingVocalBar(abc)) return false;
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
