import fs from 'node:fs';
import path from 'node:path';
import { config } from '../../../config.js';
import { getSetting, setSetting } from '../../../db/lireekDb.js';
import type { BackendModels } from '../types.js';

export const SA3_DEFAULT_DIT = 'sa3-dit-BF16.gguf';
const DIT_SETTING = 'sa3_dit_model';

export function isSa3GenerationDit(name: string): boolean {
  return /^sa3-dit-[A-Za-z0-9_-]+\.gguf$/.test(name) && !name.startsWith('sa3-dit-base-');
}

/** The engine loads generation DiTs from the models root, not subfolders. */
export function sa3ModelCatalogue(dir = config.aceServer.models): BackendModels {
  const entries: Array<{ name: string; bytes: number }> = [];
  if (fs.existsSync(dir)) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.isFile() && isSa3GenerationDit(entry.name)) {
        entries.push({ name: entry.name, bytes: fs.statSync(path.join(dir, entry.name)).size });
      }
    }
  }
  entries.sort((a, b) => a.name === SA3_DEFAULT_DIT ? -1
    : b.name === SA3_DEFAULT_DIT ? 1 : a.name.localeCompare(b.name));
  const selected = sa3SelectedDit();
  const installed = entries.map(entry => entry.name);
  return {
    buckets: { dit: installed },
    defaults: { dit: installed.includes(selected) ? selected
      : installed.includes(SA3_DEFAULT_DIT) ? SA3_DEFAULT_DIT : '' },
    meta: { dit: Object.fromEntries(entries.map(entry => [entry.name, {
      label: entry.name, bytes: entry.bytes,
    }])) },
  };
}

export function sa3SelectedDit(): string {
  const selected = getSetting(DIT_SETTING, SA3_DEFAULT_DIT);
  return isSa3GenerationDit(selected) ? selected : SA3_DEFAULT_DIT;
}

export async function selectSa3Model(selection: Record<string, string>): Promise<{ changed: boolean; dit: string }> {
  const previous = sa3SelectedDit();
  const dit = selection.dit === undefined ? previous : selection.dit || SA3_DEFAULT_DIT;
  if (!isSa3GenerationDit(dit) || !sa3ModelCatalogue().buckets.dit.includes(dit)) {
    throw new Error(`SA3 generation DiT not installed: ${dit}`);
  }
  setSetting(DIT_SETTING, dit);
  return { changed: previous !== dit, dit };
}
