import fs from 'node:fs';
import path from 'node:path';

/** True when `filename` exists in any of the runtime dirs (data folder tools/runtime, beside the exe). */
export function hasRuntimeFile(dirs: string[], filename: string): boolean {
  return dirs.some(d => fs.existsSync(path.join(d, filename)));
}
