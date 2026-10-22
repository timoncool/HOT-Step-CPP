import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { ensureYue2PreparedDataset, preparationFingerprint } from './yue2AutoPrepare.js';
import type { TrainingJob } from './labelingQueue.js';
import type { ResolvedYue2AitkPrepareOptions } from './yue2AitkPrepareRunner.js';
import { FIXED_RECIPE_CAPTION_GRAMMAR } from './fixedRecipeCaption.js';

function fixture() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-auto-prepare-'));
  const file = (name: string, value = 'fixture') => { const p = path.join(dir, name); fs.writeFileSync(p, value); return p; };
  const o: ResolvedYue2AitkPrepareOptions = {
    legacyManifest: file('legacy.json', JSON.stringify({ sources: [{ latents: 'latents', codec_ids: 'codes', cursor_words: 'words' }] })),
    checkpoint: file('base'), tokenizer: file('tokenizer'), output: path.join(dir, 'unused'),
    models: { vae: file('vae'), semantic: file('semantic'), sheetsage: file('sheet') }, lyricTiming: true,
  };
  file('latents'); file('codes'); file('words');
  const job = (): TrainingJob => ({ id: 'fixture', datasetId: 'fixture', kind: 'yue2-joint-train', status: 'queued',
    total: 1, done: 0, failed: 0, currentSampleId: null, phase: '', engineQueueDepth: 0, error: null,
    createdAt: Date.now(), startedAt: null, finishedAt: null, sampleIds: [], opts: {}, events: [], listeners: new Set(), controller: new AbortController() });
  let calls = 0;
  const prepare = async (_job: TrainingJob, opts?: ResolvedYue2AitkPrepareOptions) => {
    ++calls; assert.ok(opts);
    fs.mkdirSync(opts.output);
    fs.writeFileSync(path.join(opts.output, 'latent.bin'), 'prepared');
    fs.writeFileSync(path.join(opts.output, 'dataset.json'), JSON.stringify({ schema_version: 1,
      ...(opts.captionGrammar ? { caption_grammar: FIXED_RECIPE_CAPTION_GRAMMAR, trigger: opts.trigger, style_template: 'bare' } : {}),
      recipe_version: 'aitk-yue2-2026-09-16', items: [{ latent_file: 'latent.bin', prefix_full_nocap_ids: [1], prefix_full_uncond_ids: [1] }] }));
  };
  return { dir, o, job, prepare, calls: () => calls };
}

test('automatic preparation reuses unchanged inputs, but invalidates changed settings, caches and models', async () => {
  const f = fixture();
  const first = await ensureYue2PreparedDataset(f.job(), f.o, f.prepare);
  assert.equal(await ensureYue2PreparedDataset(f.job(), f.o, f.prepare), first);
  assert.equal(f.calls(), 1);
  assert.notEqual(await ensureYue2PreparedDataset(f.job(), { ...f.o, trigger: 'artist_album' }, f.prepare), first);
  assert.notEqual(await ensureYue2PreparedDataset(f.job(), { ...f.o, lyricTiming: false }, f.prepare), first);
  fs.appendFileSync(path.join(f.dir, 'codes'), 'new');
  await ensureYue2PreparedDataset(f.job(), { ...f.o, lyricTiming: false }, f.prepare);
  fs.appendFileSync(f.o.checkpoint, 'changed');
  await ensureYue2PreparedDataset(f.job(), { ...f.o, lyricTiming: false }, f.prepare);
  fs.appendFileSync(f.o.legacyManifest, ' ');
  await ensureYue2PreparedDataset(f.job(), { ...f.o, lyricTiming: false }, f.prepare);
  assert.equal(f.calls(), 6);
});

test('Fixed-recipe preparation uses a separate cache key and leaves the old fingerprint intact', () => {
  const f = fixture();
  const old = preparationFingerprint(f.o);
  assert.equal(preparationFingerprint({ ...f.o, captionGrammar: undefined }), old);
  assert.notEqual(preparationFingerprint({ ...f.o, captionGrammar: 'fixed-recipe' }), old);
});

test('missing prepared payload is regenerated instead of reused', async () => {
  const f = fixture();
  const first = await ensureYue2PreparedDataset(f.job(), f.o, f.prepare);
  assert.ok(first);
  fs.unlinkSync(path.join(path.dirname(first), 'latent.bin'));
  assert.notEqual(await ensureYue2PreparedDataset(f.job(), f.o, f.prepare), first);
  assert.equal(f.calls(), 2);
});

test('Fixed recipe automatically replaces the old grammar cache and then reuses the new one', async () => {
  const f = fixture();
  const options = { ...f.o, captionGrammar: 'fixed-recipe' as const, trigger: 'sample_slug', lyricTiming: false };
  const first = await ensureYue2PreparedDataset(f.job(), options, f.prepare);
  assert.ok(first);
  const manifest = JSON.parse(fs.readFileSync(first, 'utf8'));
  assert.equal(manifest.caption_grammar, 'fixed-recipe-v3');
  manifest.caption_grammar = 'fixed-recipe-v2';
  const oldText = JSON.stringify(manifest);
  fs.writeFileSync(first, oldText);
  // Give the old output a matching hash so the grammar check must reject it.
  const indexPath = path.join(f.dir, 'aitk-auto-prepare-fixed-recipe-v1.json');
  const index = JSON.parse(fs.readFileSync(indexPath, 'utf8'));
  const latent = path.join(path.dirname(first), 'latent.bin');
  const st = fs.statSync(latent);
  index.outputFingerprint = createHash('sha256').update(oldText)
    .update(JSON.stringify([[path.resolve(latent), st.size, st.mtimeMs, st.ctimeMs]])).digest('hex');
  fs.writeFileSync(indexPath, JSON.stringify(index));
  const second = await ensureYue2PreparedDataset(f.job(), options, f.prepare);
  assert.notEqual(second, first);
  assert.equal(await ensureYue2PreparedDataset(f.job(), options, f.prepare), second);
  assert.equal(f.calls(), 2);
  assert.notEqual(await ensureYue2PreparedDataset(f.job(), { ...options, trigger: 'changed_tag' }, f.prepare), second);
  assert.equal(f.calls(), 3);
});

test('a Fixed recipe dataset that does not record its trigger is regenerated', async () => {
  const f = fixture();
  const options = { ...f.o, captionGrammar: 'fixed-recipe' as const, trigger: 'sample_slug', lyricTiming: false };
  const first = await ensureYue2PreparedDataset(f.job(), options, f.prepare);
  assert.ok(first);
  const manifest = JSON.parse(fs.readFileSync(first, 'utf8'));
  delete manifest.trigger; delete manifest.style_template;
  const oldText = JSON.stringify(manifest);
  fs.writeFileSync(first, oldText);
  const indexPath = path.join(f.dir, 'aitk-auto-prepare-fixed-recipe-v1.json');
  const index = JSON.parse(fs.readFileSync(indexPath, 'utf8'));
  const latent = path.join(path.dirname(first), 'latent.bin');
  const st = fs.statSync(latent);
  index.outputFingerprint = createHash('sha256').update(oldText)
    .update(JSON.stringify([[path.resolve(latent), st.size, st.mtimeMs, st.ctimeMs]])).digest('hex');
  fs.writeFileSync(indexPath, JSON.stringify(index));
  assert.notEqual(await ensureYue2PreparedDataset(f.job(), options, f.prepare), first);
  assert.equal(f.calls(), 2);
});

test('a dataset prepared before trigger-only prefixes is regenerated, not reused', async () => {
  const f = fixture();
  const first = await ensureYue2PreparedDataset(f.job(), f.o, f.prepare);
  assert.ok(first);
  fs.writeFileSync(first, JSON.stringify({ schema_version: 1, recipe_version: 'aitk-yue2-2026-09-16', items: [{ latent_file: 'latent.bin' }] }));
  assert.notEqual(await ensureYue2PreparedDataset(f.job(), f.o, f.prepare), first);
  assert.equal(f.calls(), 2);
});

test('cancelled or failed preparation cannot hand a dataset to training', async () => {
  const f = fixture();
  const cancelled = f.job(); cancelled.controller.abort();
  assert.equal(await ensureYue2PreparedDataset(cancelled, f.o, f.prepare), undefined);
  assert.equal(f.calls(), 0);
  const during = f.job();
  assert.equal(await ensureYue2PreparedDataset(during, f.o, async j => { j.controller.abort(); }), undefined);
  const failed = f.job();
  assert.equal(await ensureYue2PreparedDataset(failed, f.o, async j => { j.status = 'failed'; }), undefined);
  assert.equal(fs.existsSync(path.join(f.dir, 'aitk-auto-prepare-v1.json')), false);
});

test('source changes during preparation prevent handoff and cache publication', async () => {
  const f = fixture();
  await assert.rejects(ensureYue2PreparedDataset(f.job(), f.o, async (j, o) => {
    await f.prepare(j, o); fs.appendFileSync(path.join(f.dir, 'words'), 'updated');
  }), /changed during preparation/);
  assert.equal(fs.existsSync(path.join(f.dir, 'aitk-auto-prepare-v1.json')), false);
});

test('captions are stale when a .yue2.txt sidecar differs from the cut, or the cut mode is not yue2', async () => {
  const { yue2CaptionsStale } = await import('./yue2AitkPrepareRunner.js');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-captions-'));
  const audio = path.join(dir, 'track.flac');
  fs.writeFileSync(path.join(dir, 'track.yue2.txt'), 'Pop-punk,  nasal tenor.\n');
  const manifest = path.join(dir, 'yue2_preprocess.json');
  const write = (mode: string, caption: string) =>
    fs.writeFileSync(manifest, JSON.stringify({ caption_mode: mode, sources: [{ source: audio, caption }] }));
  write('yue2', 'Pop-punk, nasal tenor.'); assert.equal(yue2CaptionsStale(manifest), false);
  write('yue2', 'An ACE caption.'); assert.equal(yue2CaptionsStale(manifest), true);
  write('ace', 'Pop-punk, nasal tenor.'); assert.equal(yue2CaptionsStale(manifest), true);
  fs.unlinkSync(path.join(dir, 'track.yue2.txt'));
  write('ace', 'An ACE caption.'); assert.equal(yue2CaptionsStale(manifest), false);
  fs.rmSync(dir, { recursive: true, force: true });
});
