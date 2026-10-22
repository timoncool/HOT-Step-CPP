// What this guards: the opt-in per-job `backendOverride` routes one job to a
// named registered backend, while the legacy `backend` field keeps its
// upstream log-only behavior (it is recorded as submittedBackendMismatch and
// the route logs it; it never selects a backend).
// Run: npx tsx --test src/services/generation/backendOverride.test.ts
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'backend-override-'));
process.env.DATA_DIR = DATA_DIR;
const database = await import('../../db/database.js');
database.initDb();
const registry = await import('../backends/registry.js');
const { buildEnvelope } = await import('./envelope.js');
test.after(() => { database.closeDb(); fs.rmSync(DATA_DIR, { recursive: true, force: true }); });

const body = { caption: 'test', lyrics: '[Instrumental]', duration: 30 };
let n = 0;
const envelopeFor = (raw: Record<string, unknown>) => buildEnvelope(raw, 'user-1', `job-${++n}`, Date.now());

test('backendOverride routes the job to the override backend', () => {
  assert.equal(registry.setActiveBackendId('ace'), true);
  const env = envelopeFor({ ...body, backendOverride: 'yue2' });
  console.log(`override: active=${registry.getActiveBackendId()} envelope.backendId=${env.backendId}`);
  assert.equal(env.backendId, 'yue2');
  assert.equal(registry.getActiveBackendId(), 'ace', 'active backend must not change');
});

test('unknown backendOverride is rejected', () => {
  assert.throws(() => envelopeFor({ ...body, backendOverride: 'not-a-backend' }), /backendOverride/);
});

test('without the override, raw.backend stays log-only (upstream behavior)', () => {
  assert.equal(registry.setActiveBackendId('yue2'), true);
  const env = envelopeFor({ ...body, backend: 'ace' });
  console.log(`no override: envelope.backendId=${env.backendId} submittedBackendMismatch=${env.submittedBackendMismatch}`);
  assert.equal(env.backendId, 'yue2');
  assert.equal(env.submittedBackendMismatch, 'ace');
});

test('the submittedBackendMismatch log call is still in the route', () => {
  const route = fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), '../../routes/generate.ts'), 'utf8').replace(/\r\n/g, '\n');
  assert.ok(route.includes("if (envelope.submittedBackendMismatch !== undefined) {\n    console.log(`[Generate] Job ${job.id} submitted for '${envelope.submittedBackendMismatch}' but runs on '${envelope.backendId}'`);"));
});
