import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { validateProfileImport } from './contracts.js';

// The YuE2 Plan Play presets ship as import files (docs/user/presets). Each
// must go through the normal profile import unchanged and carry YuE2 knobs.
const DIR = process.env.HS_PRESET_DIR
  ?? resolve(dirname(fileURLToPath(import.meta.url)), '../../../../docs/user/presets/yue2-plan-play');
const EXPECTED = ['plan-play-loose.json', 'plan-play-sections.json', 'plan-play-subtle.json', 'plan-play-wander.json'];

test('YuE2 Plan Play preset files import through validateProfileImport', () => {
  const files = readdirSync(DIR).filter((f) => f.endsWith('.json')).sort();
  assert.deepEqual(files, EXPECTED);
  for (const filename of files) {
    const profile = JSON.parse(readFileSync(join(DIR, filename), 'utf8'));
    const { name, data } = validateProfileImport({ filename, profile });
    assert.equal(name, filename.replace(/\.json$/, ''));
    assert.equal(data._format, 'hot-step-preset', filename);
    const bp = data.backendParams as Record<string, unknown>;
    assert.ok(bp && typeof bp === 'object' && !Array.isArray(bp), `${filename}: backendParams map`);
    const keys = Object.keys(bp);
    assert.ok(keys.length >= 10 && keys.every((k) => k.startsWith('yue2')), `${filename}: yue2 knobs only, got ${keys.length}`);
    assert.equal(typeof bp.yue2PlanCfgLevelMatch, 'boolean', `${filename}: final (level match) content`);
    console.log(`  ${filename}: ok, ${keys.length} yue2 knobs, plan_cfg ${bp.yue2PlanCfg}`);
  }
});
