// exportZip.ts - plan and stream a Library "Export as zip" download.
//
// planZipEntries is a pure-ish planner: song rows in, the files that go in the
// archive and their names out. streamZip pipes them to the response and tears
// the archive down if the client goes away.

import fs from 'fs';
import path from 'path';
import archiver from 'archiver';
import type { Readable, Writable } from 'stream';

export const MAX_ZIP_IDS = 500;
/** Total size of the source files in one export. Checked before any header is sent. */
export const MAX_ZIP_BYTES = 256 * 1024 * 1024;

export interface ExportSongRow {
  id: string;
  title?: string | null;
  audio_url?: string | null;
}

export interface ZipEntry {
  file: string;
  name: string;
  size: number;
}

export interface ZipPlan {
  entries: ZipEntry[];
  totalBytes: number;
}

/** Every id must be a nonempty string before it reaches SQLite. */
export function parseZipIds(ids: unknown): string[] | null {
  if (!Array.isArray(ids) || ids.length === 0 || ids.length > MAX_ZIP_IDS) return null;
  if (!ids.every(id => typeof id === 'string' && id.length > 0)) return null;
  return ids as string[];
}

function safeName(s: string): string {
  const cleaned = s.replace(/[\/:*?"<>|\x00-\x1f]/g, '_').replace(/\s+/g, ' ').trim();
  return cleaned.slice(0, 120) || 'untitled';
}

/** Resolve a stored audio_url to a regular file that really lives inside the
 *  audio directory (symlinks and junctions followed), or null. */
function resolveAudioFile(audioRoot: string, audioUrl: string): { file: string; size: number } | null {
  const base = path.basename(audioUrl.split('?')[0]);
  if (!base) return null;
  try {
    const real = fs.realpathSync(path.join(audioRoot, base));
    const rel = path.relative(audioRoot, real);
    if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) return null;
    const st = fs.statSync(real);
    return st.isFile() ? { file: real, size: st.size } : null;
  } catch {
    return null;
  }
}

/** Build zip entries: one per song with an audio file on disk. Names are the
 *  song title plus the file extension; every final name is reserved
 *  case-folded, so "Track", "track" and an explicit "Track (2)" never collide. */
export function planZipEntries(songs: ExportSongRow[], audioDir: string): ZipPlan {
  let audioRoot: string;
  try { audioRoot = fs.realpathSync(audioDir); } catch { return { entries: [], totalBytes: 0 }; }
  const used = new Set<string>();
  const entries: ZipEntry[] = [];
  let totalBytes = 0;
  for (const s of songs) {
    if (!s.audio_url) continue;
    const found = resolveAudioFile(audioRoot, s.audio_url);
    if (!found) continue;
    const ext = path.extname(found.file) || '.wav';
    const stem = safeName(s.title || s.id);
    let n = 1;
    let name = `${stem}${ext}`;
    while (used.has(name.toLowerCase())) name = `${stem} (${++n})${ext}`;
    used.add(name.toLowerCase());
    entries.push({ file: found.file, name, size: found.size });
    totalBytes += found.size;
  }
  return { entries, totalBytes };
}

/** Pipe the entries into `res` as a zip, one source stream open at a time.
 *  If the client disconnects before the archive ends, destroy the open source
 *  stream and the archive: abort() alone leaves a backpressured file open. */
export function streamZip(
  res: Writable & { destroy(err?: Error): unknown },
  entries: ZipEntry[],
  onError: (err: Error) => void,
  open: (file: string) => Readable = f => fs.createReadStream(f),
): archiver.Archiver {
  const archive = archiver('zip', { zlib: { level: 1 } });
  let current: Readable | null = null;
  let i = 0;
  const fail = (err: Error) => { onError(err); res.destroy(err); };
  const next = () => {
    if (i >= entries.length) { void archive.finalize(); return; }
    const e = entries[i++];
    current = open(e.file);
    // append() does not forward source errors; a file that vanished since
    // planning must fail the download, not crash the process.
    current.once('error', fail);
    archive.append(current, { name: e.name });
  };
  archive.on('entry', next);
  archive.on('error', fail);
  res.on('close', () => {
    if (res.writableFinished) return;
    archive.abort();
    current?.destroy();
    archive.destroy();
  });
  archive.pipe(res);
  next();
  return archive;
}
