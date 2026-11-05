import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

test('SA3 catalogue, persisted picks and queued snapshots use generation files only', (t) => {
  const root = fs.mkdtempSync(path.resolve('.tmp-sa3-models-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const modelsDir = path.join(root, 'models');
  fs.mkdirSync(modelsDir);
  // Test-only entries. Generation quants are not yet in the shipped registry.
  const entries = [
    { filename: 'sa3-dit-BF16.gguf', quant: 'BF16' },
    { filename: 'sa3-dit-Q4_K_M-imat.gguf', quant: 'Q4_K_M-imat' },
  ];
  for (const entry of entries) fs.writeFileSync(path.join(modelsDir, entry.filename), 'fixture');
  fs.writeFileSync(path.join(modelsDir, 'sa3-dit-base-BF16.gguf'), 'training');
  fs.writeFileSync(path.join(modelsDir, 'sa3-dit-Q8_0.gguf.part'), 'partial');
  fs.mkdirSync(path.join(modelsDir, 'sa3-dit-NVFP4.gguf'));
  const script = `
    import assert from 'node:assert/strict';
    import fs from 'node:fs';
    import path from 'node:path';
    import { initDb, closeDb } from './src/db/database.js';
    import { getSetting, setSetting } from './src/db/lireekDb.js';
    import { sa3Backend } from './src/services/backends/sa3/index.js';
    import { SA3_DEFAULT_DIT, sa3ModelCatalogue } from './src/services/backends/sa3/models.js';
    globalThis.fetch = async () => Response.json({ synth_ready: true, adapters: ['fixture-adapter'] });
    initDb();
    try {
      assert.equal((await sa3Backend.capabilities()).features.models, true);
      const catalogue = await sa3Backend.models();
      assert.deepEqual(catalogue.buckets.dit, ${JSON.stringify(entries.map(e => e.filename))});
      assert.deepEqual(catalogue.adapters, ['fixture-adapter']);
      assert.equal(catalogue.defaults.dit, SA3_DEFAULT_DIT);
      const quant = ${JSON.stringify(entries[1].filename)};
      assert.equal(catalogue.meta.dit[quant].label, quant);
      assert.equal(catalogue.meta.dit[quant].bytes, 7);
      const first = sa3Backend.resolveRequest({ caption: 'fixture' });
      assert.equal((await sa3Backend.selectModel({dit: quant})).changed, true);
      assert.equal(getSetting('sa3_dit_model'), quant);
      const queued = sa3Backend.resolveRequest({ caption: 'fixture' });
      assert.equal(queued.models.dit, quant);
      assert.equal((await sa3Backend.selectModel({dit: SA3_DEFAULT_DIT})).changed, true);
      assert.equal(first.models.dit, SA3_DEFAULT_DIT);
      assert.equal(queued.models.dit, quant, 'later picks must not alter queued jobs');
      assert.equal(sa3Backend.resolveRequest({sa3DitModel: quant}).models.dit, quant);
      for (const dit of ['sa3-dit-base-BF16.gguf', '../sa3-dit-BF16.gguf', 'sa3-dit-Q8_0.gguf']) {
        await assert.rejects(sa3Backend.selectModel({dit}), /not installed/);
        assert.equal(getSetting('sa3_dit_model'), SA3_DEFAULT_DIT);
      }
      await sa3Backend.selectModel({dit: quant});
      fs.unlinkSync(path.join(process.env.ACESTEPCPP_MODELS, quant));
      assert.equal(sa3ModelCatalogue().defaults.dit, SA3_DEFAULT_DIT);
      setSetting('sa3_dit_model', 'invalid');
      assert.equal(sa3Backend.resolveRequest({}).models.dit, SA3_DEFAULT_DIT);
      fs.unlinkSync(path.join(process.env.ACESTEPCPP_MODELS, SA3_DEFAULT_DIT));
      assert.deepEqual(sa3ModelCatalogue().buckets.dit, []);
      assert.equal(sa3ModelCatalogue().defaults.dit, '');
    } finally { closeDb(); }
  `;
  execFileSync(process.execPath, ['--import', 'tsx/esm', '--eval', script], {
    timeout: 30_000,
    env: { ...process.env, DATA_DIR: path.join(root, 'library'),
      ACESTEPCPP_MODELS: modelsDir, TRAINING_DIR: path.join(root, 'training') },
    stdio: 'pipe',
  });
});
