import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import type { Yue2AitkRunRecord } from '../../training/yue2AitkRuns.js';

// A fresh DATA_DIR and database: the fallback path reads the dataset table.
process.env.DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-trigger-data-'));
(await import('../../../db/database.js')).initDb();
const { yue2AdapterTrigger } = await import('./jointAdapterContext.js');

function run(styles: string[], method = 'fixed-recipe'): { ref: string; runs: Yue2AitkRunRecord[] } {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-trigger-'));
  const dataset = path.join(dir, 'dataset.json');
  fs.writeFileSync(dataset, JSON.stringify({ items: styles.map(style => ({ style })) }));
  const ref = path.join(dir, 'yue2_joint_ar_step10.safetensors');
  return { ref, runs: [{
    version: 1, jobId: 'j', datasetId: 'no-such-dataset', datasetSlug: 's', method: 'aitk', output: dir,
    options: { method, dataset, preparation: { trigger: 'tag_a' } }, status: 'done', createdAt: 0, updatedAt: 0,
    checkpoints: [{ step: 10, arPath: ref, narPath: path.join(dir, 'nar.safetensors') }],
  } as unknown as Yue2AitkRunRecord] };
}

test('fixed-recipe styles that open with the trigger count as trained, not inferred', () => {
  const { ref, runs } = run(['tag_a, nu metal', 'tag_a']);
  assert.deepEqual(yue2AdapterTrigger(ref, runs), { trigger: 'tag_a', inferred: false });
});

test('a fixed-recipe style without the trigger is not claimed as trained', () => {
  const { ref, runs } = run(['tag_a, nu metal', 'nu metal']);
  assert.equal(yue2AdapterTrigger(ref, runs).trigger, '');
});

test('non-fixed-recipe runs keep the old inference path', () => {
  const { ref, runs } = run(['tag_a, nu metal'], 'aitk');
  assert.equal(yue2AdapterTrigger(ref, runs).trigger, '');
});
