// Monitor control file helpers for the VST routes.
//
// The monitor host polls monitor_control.json. Two rules keep commands from
// being lost:
//  - The file is written atomically (temp file, then rename), so the host never
//    reads a truncated or half-written file.
//  - Every write carries a monotonically increasing `seq`, and every write
//    merges into the current state, so a pause cannot drop a pending track and
//    a seek cannot drop a pending pause. A seek carries its own `seekSeq`, so
//    the host applies each seek once even though the field stays in the file.

import fs from 'fs';

export type MonitorControl = Record<string, unknown>;

/** Merge `patch` into `state` (or replace it when `reset` is set) and stamp the
 *  given sequence number. Pure, so it is unit-testable. */
export function mergeControl(state: MonitorControl, patch: MonitorControl, seq: number, reset = false): MonitorControl {
  const next: MonitorControl = reset ? { ...patch } : { ...state, ...patch };
  next.seq = seq;
  if (typeof patch.seek === 'number') next.seekSeq = seq;
  return next;
}

/** Write JSON atomically: temp file in the same directory, then rename over the
 *  target. On Windows the rename can fail briefly while the reader holds the
 *  file open, so retry a few times and fall back to a direct write. */
export function writeJsonAtomic(file: string, data: unknown): void {
  const json = JSON.stringify(data);
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, json, 'utf-8');
  for (let i = 0; i < 5; i++) {
    try { fs.renameSync(tmp, file); return; } catch { /* reader has it open, retry */ }
  }
  try { fs.unlinkSync(tmp); } catch {}
  fs.writeFileSync(file, json, 'utf-8');
}

/** Runs async tasks one at a time, in call order. A task starts only after the
 *  previous one has settled, whether it resolved or threw, so a monitor
 *  start/restart/stop cannot interleave with another across its awaits. */
export function createSerializer() {
  let tail: Promise<unknown> = Promise.resolve();
  return <T>(task: () => Promise<T>): Promise<T> => {
    const run = tail.then(task, task);
    tail = run.catch(() => {});
    return run;
  };
}

/** Exit handler for a monitor host. Every host reads the same chain file, and
 *  a host that ignored stop can exit after its replacement has rewritten it, so
 *  remove the file only when no other host owns the slot. Returns the new value
 *  for the current-host slot. */
export function releaseMonitor<T>(child: T, current: T | null, chainFile: string): T | null {
  if (current === child) current = null;
  if (current === null) { try { fs.unlinkSync(chainFile); } catch {} }
  return current;
}

/** Stateful writer: holds the current control state and the next seq. The seq
 *  starts at the current time so it keeps increasing across server restarts. */
export function createControlWriter(fileFn: () => string) {
  let state: MonitorControl = {};
  let seq = Date.now();
  return {
    write(patch: MonitorControl, reset = false): MonitorControl {
      state = mergeControl(state, patch, ++seq, reset);
      writeJsonAtomic(fileFn(), state);
      return state;
    },
    get state(): MonitorControl { return state; },
  };
}
