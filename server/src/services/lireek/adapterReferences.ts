import fs from 'node:fs';
import path from 'node:path';
import type Database from 'better-sqlite3';

const fields = ['adapter_path', 'lm_adapter_path', 'mm3_adapter_path', 'yue2_ar_adapter_path', 'yue2_nar_adapter_path'] as const;
type AdapterField = typeof fields[number];
type Preset = Record<AdapterField, string | null> & {
  lyrics_set_id: number; artist_name: string; album: string | null;
};

export interface DroppedAdapterReference {
  lyricsSetId: number;
  presetName: string;
  field: AdapterField;
  adapter: string;
}

function stat(file: string): fs.Stats | null {
  try { return fs.statSync(file); }
  catch (error) {
    if (['ENOENT', 'ENOTDIR'].includes((error as NodeJS.ErrnoException).code ?? '')) return null;
    throw error;
  }
}

function hasWeights(file: string): boolean {
  const entry = stat(file);
  if (entry?.isFile()) return true;
  return !!entry?.isDirectory() && ['adapter_model.safetensors', 'lokr_weights.safetensors']
    .some(leaf => stat(path.join(file, leaf))?.isFile());
}

function exists(field: AdapterField, ref: string, root: string): boolean {
  if (field === 'mm3_adapter_path') {
    const dir = path.join(root, 'mm3-lm-adapters');
    const file = path.resolve(dir, ref);
    const rel = path.relative(dir, file);
    return !rel.startsWith('..') && !path.isAbsolute(rel) && !!stat(file)?.isFile();
  }
  if (field === 'yue2_ar_adapter_path' || field === 'yue2_nar_adapter_path') {
    return path.isAbsolute(ref) && !!stat(ref)?.isFile();
  }
  // ACE accepts registry names as well as explicit file/PEFT/LoKR paths.
  const registryRoot = field === 'lm_adapter_path' ? path.join(root, 'lm') : root;
  return hasWeights(path.resolve(registryRoot, ref))
    || ((ref.includes('/') || ref.includes('\\')) && hasWeights(path.resolve(ref)));
}

/** Explicit maintenance command. Only missing adapter columns are changed. */
export function updateAdapterReferences(db: Database.Database, adaptersRoot: string) {
  // An unavailable adapter drive must not be mistaken for deleted weights.
  if (!fs.statSync(adaptersRoot).isDirectory()) throw new Error('Adapters directory is unavailable');
  return db.transaction(() => {
    const presets = db.prepare(`SELECT ap.*, a.name AS artist_name, ls.album
      FROM album_presets ap JOIN lyrics_sets ls ON ls.id = ap.lyrics_set_id
      JOIN artists a ON a.id = ls.artist_id`).all() as Preset[];
    const changes: DroppedAdapterReference[] = [];
    for (const preset of presets) {
      for (const field of fields) {
        const ref = preset[field];
        if (!ref || exists(field, ref, adaptersRoot)) continue;
        db.prepare(`UPDATE album_presets SET ${field} = NULL WHERE lyrics_set_id = ?`).run(preset.lyrics_set_id);
        changes.push({ lyricsSetId: preset.lyrics_set_id,
          presetName: preset.album ? `${preset.artist_name} / ${preset.album}` : preset.artist_name,
          field, adapter: ref });
      }
    }
    return { scanned: presets.length, updated: new Set(changes.map(change => change.lyricsSetId)).size, changes };
  })();
}
