// node --test tools/docs/build-docs.test.mjs
// Plugin metadata and training operation route discovery, on fixture trees and on
// the real repository. Nothing here writes a generated doc.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { parsePlugin, trainingOperationRoutes } from './build-docs.mjs';

function fixture(files) {
  fs.mkdirSync(path.join(import.meta.dirname, '..', '..', '.scratch'), { recursive: true });
  const root = fs.mkdtempSync(path.join(import.meta.dirname, '..', '..', '.scratch', 'build-docs-'));
  const base = {
    'server/src/index.ts': "import trainingRoutes from './routes/training.js';\napp.use('/api/training', trainingRoutes);\n",
    'server/src/routes/training.ts': "router.use('/ops', trainingOperationsRouter);\n",
  };
  for (const [name, text] of Object.entries({ ...base, ...files })) {
    fs.mkdirSync(path.dirname(path.join(root, name)), { recursive: true });
    fs.writeFileSync(path.join(root, name), text);
  }
  return root;
}

const routes = (root) => trainingOperationRoutes(root).map((r) => `${r.verb} ${r.path}`);

for (const eol of ['\n', '\r\n']) {
  test(`plugin declarations longer than 2500 characters are complete with ${JSON.stringify(eol)}`, () => {
    const source = [
      'solver = {',
      '  name = "long_plugin",',
      '  params = {',
      '    { key = "first", options = { { value = "one", label = "One" } },',
      `      hint = "${'Long parameter hint. '.repeat(150)}" },`,
      '    { key = "last", type = "slider", default = 1 },',
      '  },',
      '  display = "Long plugin",',
      '}',
      'local unrelated = {',
      '  { key = "not_a_parameter" },',
      '}',
    ].join(eol);
    const root = fixture({ 'plugin.lua': source, 'empty.lua': 'scheduler = {\n  name = "empty",\n}\n', 'helper.lua': 'local helper = {}\n' });
    try {
      const plugin = parsePlugin(path.join(root, 'plugin.lua'));
      assert.equal(plugin.params, 2);
      assert.equal(plugin.display, 'Long plugin');
      assert.equal(parsePlugin(path.join(root, 'empty.lua')).params, 0);
      assert.equal(parsePlugin(path.join(root, 'helper.lua')), null);
    } finally { fs.rmSync(root, { recursive: true, force: true }); }
  });
}

test('registered domains are read from their mount functions, loops expanded', () => {
  const root = fixture({
    'server/src/services/training/things.ts': [
      "router.get('/not-mounted', handler);",
      'export function mountThings(router: Router, deps: Deps): void {',
      "  router.get('/', trainingOp(async () => ({ ok: { nested: true } })));",
      '  router.post("/start", trainingOp(async req => { if (x) { return 1; } return 2; }));',
      "  for (const action of ['pause', 'resume'] as const) {",
      '    router.post(`/:id/${action}`, trainingOp(async () => ({})));',
      '  }',
      '}',
      "registerTrainingOperations('things', mountThings);",
    ].join('\n'),
    'server/src/services/training/deep/other.ts':
      "function mountOther(r: Router) { r.delete('/:id', h); }\nregisterTrainingOperations('other', mountOther);\n",
    'server/src/services/training/things.test.ts': "registerTrainingOperations('test-only', mountThings);\n",
  });
  try {
    assert.deepEqual(routes(root).sort(), [
      'DELETE /api/training/ops/other/:id',
      'GET /api/training/ops/things',
      'POST /api/training/ops/things/:id/pause',
      'POST /api/training/ops/things/:id/resume',
      'POST /api/training/ops/things/start',
    ]);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('a path it cannot expand, or a mount function it cannot find, is an error', () => {
  for (const [file, pattern] of [
    ["function mountX(router) { router.get(`/${kind}`, h); }\nregisterTrainingOperations('x', mountX);\n", /without a literal loop array/],
    ["registerTrainingOperations('x', mountElsewhere);\n", /not a function in that file/],
    // A path held in a variable, or any non-verb call, is never skipped silently.
    ["function mountX(router) { const route = '/start'; router.post(route, h); }\nregisterTrainingOperations('x', mountX);\n", /cannot index router\.post\(route/],
    ["function mountX(router) { router.use('/sub', other); }\nregisterTrainingOperations('x', mountX);\n", /cannot index router\.use/],
  ]) {
    const root = fixture({ 'server/src/services/training/x.ts': file });
    try { assert.throws(() => trainingOperationRoutes(root), pattern); }
    finally { fs.rmSync(root, { recursive: true, force: true }); }
  }
  const root = fixture({ 'server/src/routes/training.ts': '// no ops mount\n' });
  try { assert.throws(() => trainingOperationRoutes(root), /trainingOperationsRouter mount/); }
  finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('the repository yields every training operation domain once, without duplicates', () => {
  const all = routes(path.resolve(import.meta.dirname, '..', '..'));
  assert.equal(new Set(all).size, all.length);
  for (const domain of ['recipes', 'preparation', 'review']) {
    assert.ok(all.some((r) => r.includes(`/api/training/ops/${domain}`)), domain);
  }
  for (const action of ['pause', 'resume', 'retry', 'cancel']) {
    assert.ok(all.includes(`POST /api/training/ops/preparation/:id/${action}`), action);
  }
});
