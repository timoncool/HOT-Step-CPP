// training/sa3Snapshots.ts — the SA3 checkpoint-preview segments on disk.
//
// A preview run trains in segments: ace-train runs to an absolute --steps
// bound, saves state.gguf and writes its inference export to adapter.gguf,
// then exits; the next segment relaunches with --resume and a larger bound.
// tools/sa3-resume-proof.mjs shows the segmented run equals the uninterrupted
// one. Between segments the export is captured, once, as
// <run>/snapshots/step-N.gguf, the file the engine's snapshot resolver loads.
// A capture is never replaced: a later file under the same name is refused.
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { validateSa3Export } from './adapterLayout.js';

/** The completed step saved in `<runDir>/state.gguf` (`sa3t.step`), or null
 *  when the run has no state yet. Reads the GGUF metadata only. */
export function sa3StateStep(runDir: string): number | null {
  let fd: number;
  try { fd = fs.openSync(path.join(runDir, 'state.gguf'), 'r'); } catch { return null; }
  try {
    const head = Buffer.alloc(65536);
    const n = fs.readSync(fd, head, 0, head.length, 0);
    let pos = 0;
    const need = (k: number) => { if (pos + k > n) throw new Error('SA3 state.gguf metadata is truncated'); };
    const u32 = () => { need(4); const v = head.readUInt32LE(pos); pos += 4; return v; };
    const u64 = () => { need(8); const v = head.readBigUInt64LE(pos); pos += 8; return v; };
    const str = () => { const len = Number(u64()); need(len); const s = head.toString('utf8', pos, pos + len); pos += len; return s; };
    if (n < 24 || head.toString('ascii', 0, 4) !== 'GGUF') throw new Error('SA3 state.gguf is not a GGUF file');
    pos = 4;
    if (u32() !== 3) throw new Error('SA3 state.gguf is not GGUF v3');
    u64();
    const keys = Number(u64());
    for (let i = 0; i < keys; i++) {
      const key = str(), type = u32();
      if (key === 'sa3t.step' && type === 10) return Number(u64());
      // sa3t_save_state writes only strings and the u64 step.
      if (type === 8) str(); else if (type === 10) u64(); else break;
    }
    throw new Error('SA3 state.gguf has no sa3t.step');
  } finally { fs.closeSync(fd); }
}

export function sa3SnapshotPath(runDir: string, step: number): string {
  return path.join(runDir, 'snapshots', `step-${step}.gguf`);
}

/** Captured steps, ascending. */
export function sa3SnapshotSteps(runDir: string): number[] {
  let names: string[] = [];
  try { names = fs.readdirSync(path.join(runDir, 'snapshots')); } catch { return []; }
  return names.map(n => /^step-([1-9][0-9]*)\.gguf$/.exec(n)?.[1]).filter((s): s is string => !!s).map(Number).sort((a, b) => a - b);
}

/** The next segment bound after `done`: the next multiple of `cadence`, at
 *  most `total`. Multiples, not done + cadence, so a run resumed from an
 *  off-cadence checkpoint lands back on the same preview steps. */
export function sa3NextBound(done: number, cadence: number, total: number): number {
  return Math.min(total, (Math.floor(done / cadence) + 1) * cadence);
}

/**
 * Capture the segment's `adapter.gguf` as `snapshots/step-N.gguf`. The export
 * is validated against the inference base first, copied to a temp name and
 * hard-linked into place, so the name never holds a partial file and an
 * existing capture is never overwritten. Returns false when step N was
 * already captured (that file stays as it is).
 */
export function captureSa3Snapshot(runDir: string, step: number, expectedBase: string): boolean {
  const dest = sa3SnapshotPath(runDir, step);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  if (fs.existsSync(dest)) return false;
  const temp = path.join(path.dirname(dest), `.${randomUUID()}.part`);
  try {
    fs.copyFileSync(path.join(runDir, 'adapter.gguf'), temp, fs.constants.COPYFILE_EXCL);
    validateSa3Export(temp, expectedBase);
    try { fs.linkSync(temp, dest); }
    catch (err: any) {
      if (err.code === 'EEXIST') return false;
      // No hard links on this volume: an exclusive copy is still write-once.
      if (err.code !== 'EPERM' && err.code !== 'ENOTSUP' && err.code !== 'EXDEV') throw err;
      fs.copyFileSync(temp, dest, fs.constants.COPYFILE_EXCL);
    }
    return true;
  } finally { fs.rmSync(temp, { force: true }); }
}

/** A segment that is not the last leaves no adapter in the run folder: only
 *  the logical final step's export is ever published or reported. */
export function clearSa3SegmentExport(runDir: string): void {
  for (const name of ['adapter.gguf', 'adapter-raw.gguf']) fs.rmSync(path.join(runDir, name), { force: true });
}
