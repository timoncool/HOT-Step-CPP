import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const source = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

export function requireDirectories(paths) {
  for (const [name, value] of Object.entries(paths)) {
    if (!value || !path.isAbsolute(value) || !fs.statSync(value, { throwIfNoEntry: false })?.isDirectory()) {
      throw new Error(`Missing or invalid shared ${name} directory: ${value}`);
    }
  }
}

export function resolveSharedData(root, report) {
  if (!root || !path.isAbsolute(root)) throw new Error('Shared mode requires an absolute --data-root.');
  if (report.pointerError || report.pendingMove || report.source !== 'pointer' || !report.root ||
      path.resolve(report.root) !== path.resolve(root) || !report.children) {
    throw new Error('Shared root must match the dev pointer, with no pointer error or pending move.');
  }
  const paths = { root: path.resolve(root), profile: report.profileDir,
    library: report.children.library, training: report.children.training,
    adapters: report.children.adapters, models: report.children.models };
  requireDirectories(paths);
  return paths;
}

export function readDevData(profile, checkout = source) {
  const env = { ...process.env };
  for (const key of ['HOT_STEP_ROOT', 'HOT_STEP_DEV', 'HOT_STEP_DATA_ROOT']) delete env[key];
  if (profile) env.HOT_STEP_PROFILE_DIR = profile;
  const result = spawnSync(process.execPath, ['--import', 'tsx', 'scripts/data-paths.ts'], {
    cwd: path.join(checkout, 'server'), env, encoding: 'utf8', timeout: 30_000,
  });
  if (result.error || result.status !== 0) throw new Error(`Dev path inspection failed: ${result.error?.message ?? result.stderr}`);
  return JSON.parse(result.stdout);
}

// Called before the launcher starts Node, including recorded-artifact restarts.
export function validateSharedConfig(config) {
  const current = resolveSharedData(config.data.root, readDevData(config.devProfile, config.devCheckout));
  current.profile = config.data.profile;
  requireDirectories({ profile: current.profile });
  for (const key of Object.keys(current)) {
    if (current[key] !== config.data[key]) throw new Error(`Shared ${key} changed. Promote again to select the new paths.`);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const config = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
    if (config.mode === 'shared') validateSharedConfig(config);
    else if (config.mode !== 'scratch') throw new Error('Unknown production data mode. Promote again.');
    console.log(`PROD DATA mode=${config.mode}\n${JSON.stringify(config.data, null, 2)}`);
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
