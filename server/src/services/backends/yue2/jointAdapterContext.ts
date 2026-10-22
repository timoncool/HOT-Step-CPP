import fs from 'fs';
import path from 'path';
import { getSetting, setSetting } from '../../../db/lireekDb.js';
import { getDataset } from '../../training/datasetsRepo.js';
import { jointRunForAdapter, type Yue2AitkRunRecord } from '../../training/yue2AitkRuns.js';
import { readSafetensorsMeta } from '../../training/yue2Runs.js';

const fixedRecipeTriggers = new Map<string, { mtimeMs: number; trigger: string }>();

/** Fixed-recipe (HOT-Approved) preparation prefixes the trigger onto every
 * caption in TS, so the native manifest and the exported files carry no
 * `trigger` field. The prepared styles are the evidence: when every one opens
 * with the run's trigger, the adapter was trained on it in the bare shape. */
function fixedRecipeTrigger(run: Yue2AitkRunRecord): string {
  const prep = run.options.preparation as Record<string, unknown> | undefined;
  const trigger = typeof prep?.trigger === 'string' ? prep.trigger.normalize('NFC').trim() : '';
  const file = run.options.dataset;
  if (run.options.method !== 'fixed-recipe' || !trigger || typeof file !== 'string') return '';
  try {
    const { mtimeMs, size } = fs.statSync(file);
    const hit = fixedRecipeTriggers.get(file);
    if (hit?.mtimeMs === mtimeMs) return hit.trigger;
    if (size > 16 * 1024 * 1024) return '';
    const items = (JSON.parse(fs.readFileSync(file, 'utf8')) as { items?: unknown }).items;
    const trained = Array.isArray(items) && items.length > 0 && items.every(item => {
      const style = (item as { style?: unknown } | null)?.style;
      return typeof style === 'string' && (style === trigger || style.startsWith(`${trigger}, `));
    });
    fixedRecipeTriggers.set(file, { mtimeMs, trigger: trained ? trigger : '' });
    return trained ? trigger : '';
  } catch { return ''; }
}

// Trigger words typed in the Adapters dropdown for adapters that carry none,
// keyed by the adapter's resolved path. Only consulted when the file, its run
// and its dataset give no trigger at all.
const MANUAL_TRIGGERS_SETTING = 'yue2_manual_triggers';

function manualTriggers(): Record<string, string> {
  try {
    const parsed = JSON.parse(getSetting(MANUAL_TRIGGERS_SETTING, '') || '{}');
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed as Record<string, string> : {};
  } catch { return {}; }
}

/** Set ('' clears) the typed trigger for one adapter file. */
export function setYue2ManualTrigger(ref: string, trigger: string): void {
  const all = manualTriggers();
  const key = path.resolve(ref);
  const value = trigger.normalize('NFC').trim().slice(0, 200);
  if (value) all[key] = value; else delete all[key];
  setSetting(MANUAL_TRIGGERS_SETTING, JSON.stringify(all));
}

/** Older native joint exports lack trigger metadata. Recover the dataset's
 * configured tag for prompt composition, but report that it is inferred: the
 * prepared style text may not have contained it during training. With none
 * of those, a trigger the user typed for this file (`manual`). Pass `runs`
 * when calling per adapter in a loop: each lookup otherwise rescans every run
 * folder on disk. */
export function yue2AdapterTrigger(ref: string, runs?: Yue2AitkRunRecord[]): { trigger: string; inferred: boolean; manual?: boolean } {
  if (!ref) return { trigger: '', inferred: false };
  const recorded = readSafetensorsMeta(ref)?.trigger?.normalize('NFC').trim() || '';
  if (recorded) return { trigger: recorded, inferred: false };
  const run = jointRunForAdapter(ref, runs);
  const trained = run ? fixedRecipeTrigger(run) : '';
  if (trained) return { trigger: trained, inferred: false };
  const dataset = run ? getDataset(run.datasetId) : undefined;
  const tag = dataset?.customTag?.normalize('NFC').trim() || '';
  if (tag) return { trigger: tag, inferred: true };
  const typed = manualTriggers()[path.resolve(ref)] || '';
  return typed ? { trigger: typed, inferred: false, manual: true } : { trigger: '', inferred: false };
}
