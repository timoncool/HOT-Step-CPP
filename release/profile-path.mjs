// Match dataroot/paths.ts profileDir and RESTART_MARKER_FILE.
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export function restartMarker(env = process.env, platform = process.platform, home = os.homedir(), paths = path) {
  const profile = env.HOT_STEP_PROFILE_DIR ? paths.resolve(env.HOT_STEP_PROFILE_DIR)
    : platform === 'win32' ? paths.join(env.LOCALAPPDATA || paths.join(home, 'AppData', 'Local'), 'HOT-Step')
    : platform === 'darwin' ? paths.join(home, 'Library', 'Application Support', 'HOT-Step')
    : paths.join(env.XDG_CONFIG_HOME || paths.join(home, '.config'), 'hot-step');
  return paths.join(profile, 'restart-requested');
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.stdout.write(restartMarker() + '\n');
}
