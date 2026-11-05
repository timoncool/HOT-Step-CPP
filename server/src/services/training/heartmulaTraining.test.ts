import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const root = fs.mkdtempSync(path.join(process.cwd(), '.test-heartmula-train-'));
process.env.TRAINING_DIR = path.join(root, 'training');
process.env.DATA_DIR = path.join(root, 'library');
const { config } = await import('../../config.js');
config.aceServer.adapters = path.join(root, 'adapters');
config.aceServer.models = path.join(root, 'models');
const train = await import('./heartmulaTrain.js');
const runs = await import('./heartmulaRuns.js');
const prep = await import('./heartmulaPrepare.js');
const { HEARTMULA_RECIPE_DEFAULTS, HEARTMULA_PREVIEW_DEFAULTS } = await import('../../contracts/heartmulaTraining.js');
after(() => fs.rmSync(root, { recursive: true, force: true }));

const ds = { id: 'ds1', slug: 'demo', customTag: 'trig' };

test('recipe: defaults, bad values and unknown keys throw', () => {
  assert.equal(HEARTMULA_RECIPE_DEFAULTS.gradAccum, 4);
  assert.deepEqual(train.resolveHeartmulaRecipe({}), HEARTMULA_RECIPE_DEFAULTS);
  assert.equal(train.resolveHeartmulaRecipe({ steps: 50, learningRate: 2e-4 }).steps, 50);
  assert.throws(() => train.resolveHeartmulaRecipe({ steps: 0 }), /steps/);
  assert.throws(() => train.resolveHeartmulaRecipe({ rank: 1.5 }), /rank/);
  assert.throws(() => train.resolveHeartmulaRecipe({ learningRate: -1 }), /learningRate/);
  assert.throws(() => train.resolveHeartmulaRecipe({ nope: 1 }), /Unsupported/);
  assert.equal(HEARTMULA_RECIPE_DEFAULTS.loraDropout, 0.05);
  assert.equal(train.resolveHeartmulaRecipe({ loraDropout: 0.05 }).loraDropout, 0.05);
  assert.throws(() => train.resolveHeartmulaRecipe({ loraDropout: 1 }), /loraDropout/);
  assert.throws(() => train.resolveHeartmulaRecipe({ loraDropout: -0.1 }), /loraDropout/);
  assert.equal(HEARTMULA_RECIPE_DEFAULTS.trainMlp, true);
  assert.equal(HEARTMULA_RECIPE_DEFAULTS.cropSeconds, 0);
  assert.equal(HEARTMULA_RECIPE_DEFAULTS.eos, true);
  assert.throws(() => train.resolveHeartmulaRecipe({ eos: 'yes' }), /eos/);
  assert.equal(train.resolveHeartmulaRecipe({ cropSeconds: 120 }).cropSeconds, 120);
  for (const bad of [5, 601, -1, 'x']) assert.throws(() => train.resolveHeartmulaRecipe({ cropSeconds: bad }), /cropSeconds/);
  assert.equal(train.resolveHeartmulaRecipe({ trainMlp: true }).trainMlp, true);
  assert.throws(() => train.resolveHeartmulaRecipe({ trainMlp: 'yes' }), /trainMlp/);
  assert.deepEqual([HEARTMULA_RECIPE_DEFAULTS.variant, HEARTMULA_RECIPE_DEFAULTS.quant], ['3B', 'F16']);
  assert.equal(train.resolveHeartmulaRecipe({ variant: '3B-happy-new-year', quant: 'Q8_0' }).variant, '3B-happy-new-year');
  assert.throws(() => train.resolveHeartmulaRecipe({ variant: '../x' }), /variant/);
  assert.throws(() => train.resolveHeartmulaRecipe({ quant: 8 }), /quant/);
});

test('new run defaults cap at 30 epochs using actual updates, with an overall 480 ceiling', () => {
  for (const [samples, accum, steps] of [[1, 4, 30], [4, 4, 30], [5, 4, 60], [12, 4, 90], [60, 4, 450], [61, 4, 480], [64, 4, 480], [65, 4, 480], [5, 1, 150], [5, 8, 30]]) {
    assert.equal(train.resolveHeartmulaRecipe({ gradAccum: accum }, samples).steps, steps);
  }
  assert.equal(train.resolveHeartmulaRecipe({ steps: 2000, loraDropout: 0, gradAccum: 1 }, 1).steps, 2000);
  const batch = train.resolveHeartmulaBatchRecipe({ gradAccum: 4 });
  assert.equal(batch.steps, undefined);
  assert.equal(train.resolveHeartmulaRecipe(batch, 12).steps, 90);
  assert.equal(train.resolveHeartmulaBatchRecipe({ steps: 480 }).steps, 480);
});

test('API starts use prepared sample count; a legacy resume retains its recipe', () => {
  const dsCap = { id: 'cap', slug: 'cap' };
  const dir = runs.heartmulaPrepareDir(dsCap.slug);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'heartmula_prepare.json'), JSON.stringify({ samples: Array.from({ length: 5 }, (_, id) => ({ id })) }));
  const fresh = runs.resolveHeartmulaRun(dsCap, {});
  assert.equal(fresh.record.recipe.steps, 60);
  const oldRecipe = { ...HEARTMULA_RECIPE_DEFAULTS, steps: 2000, loraDropout: 0, gradAccum: 8 };
  const old = runs.createHeartmulaRun(dsCap.id, dsCap.slug, 'legacy', oldRecipe);
  fs.writeFileSync(path.join(old.dir, 'hm_train_state.bin'), 'checkpoint');
  const saved = JSON.parse(fs.readFileSync(path.join(old.dir, 'run.json'), 'utf8'));
  delete saved.recipe.trainMlp; delete saved.recipe.cropSeconds; delete saved.recipe.eos; delete saved.recipe.loraDropout;
  fs.writeFileSync(path.join(old.dir, 'run.json'), JSON.stringify(saved));
  const resumed = runs.resolveHeartmulaRun(dsCap, { resumeRun: old.id, recipe: { loraDropout: 0, gradAccum: 8 } });
  assert.deepEqual(resumed.record.recipe, saved.recipe);
  assert.equal(resumed.record.recipe.steps, 2000);
  assert.throws(() => runs.resolveHeartmulaRun(dsCap, { resumeRun: old.id, recipe: { loraDropout: 0.05 } }), /cannot change loraDropout/);
});

test('preview settings: defaults, overrides, and every bad value refused', () => {
  const D = HEARTMULA_PREVIEW_DEFAULTS;
  assert.deepEqual(train.resolveHeartmulaPreviewSettings(undefined), D);
  assert.deepEqual(train.resolveHeartmulaPreviewSettings({}), D);
  assert.equal(train.resolveHeartmulaPreviewSettings({ everySteps: 0 }).everySteps, 0);
  assert.equal(train.resolveHeartmulaPreviewSettings({ seed: 7, seconds: 60 }).seconds, 60);
  for (const bad of [{ nope: 1 }, { everySteps: -1 }, { everySteps: 1.5 }, { seconds: 1 }, { seconds: 241 }, { temperature: 0 }, { topk: 0 },
    { cfgScale: 0.5 }, { codecSteps: 0 }, { codecGuidance: 11 }, { tags: 5 }, { lyricsSource: 'nope' }, { seed: -1 }, { seed: true }]) {
    assert.throws(() => train.resolveHeartmulaPreviewSettings(bad), /preview|Unsupported/, JSON.stringify(bad));
  }
  assert.throws(() => train.resolveHeartmulaPreviewSettings([]), /object/);
  // full-song length and blank (= from the dataset) prompt by default; lyricsSource only ever stores 'dataset'
  assert.equal(D.seconds, 240);
  assert.equal(D.tags, '');
  assert.equal(train.resolveHeartmulaPreviewSettings({ lyricsSource: 'dataset' }).lyricsSource, 'dataset');
  assert.equal(train.resolveHeartmulaPreviewSettings({ lyricsSource: 'generated' }, { ...D, lyricsSource: 'dataset' }).lyricsSource, undefined);
  // a new run stores the resolved fixture, and a bad one refuses the start
  const made = runs.resolveHeartmulaRun(ds, { preview: { everySteps: 50 } });
  assert.equal((made.record.preview as typeof D).everySteps, 50);
  assert.equal((made.record.preview as typeof D).tags, D.tags);
  assert.throws(() => runs.resolveHeartmulaRun(ds, { preview: { seconds: 0 } }), /preview seconds/);
});

test('argv matches the trainer CLI contract', () => {
  const plain = { ...HEARTMULA_RECIPE_DEFAULTS, trainMlp: false, eos: false, loraDropout: 0 };
  assert.ok(train.buildHeartmulaTrainArgs({ dataDir: 'D', runDir: 'R', modelsDir: 'M', recipe: HEARTMULA_RECIPE_DEFAULTS, name: 'a', resume: false })
    .includes('--lora-mlp'), 'MLP layers train by default');
  const args = train.buildHeartmulaTrainArgs({
    dataDir: 'D', runDir: 'R', modelsDir: 'M', recipe: plain, name: 'a', resume: true,
  });
  assert.deepEqual(args.slice(0, 11), ['heartmula-train', '--data', 'D', '--out', 'R', '--models', 'M', '--variant', '3B', '--type', 'F16']);
  assert.ok(args.join(' ').includes('--grad-accum 4 --seed 0 --save-every 100 --attn auto --keep-layers auto --lora-dropout 0 --name a --resume'));
  assert.ok(!args.includes('--stop-after'));
  const seg = train.buildHeartmulaTrainArgs({ dataDir: 'D', runDir: 'R', modelsDir: 'M', recipe: { ...plain, loraDropout: 0.05 }, name: 'a', resume: false, stopAfter: 100 });
  assert.ok(seg.join(' ').includes('--lora-dropout 0.05 --name a --stop-after 100'));
  const crop = train.buildHeartmulaTrainArgs({ dataDir: 'D', runDir: 'R', modelsDir: 'M', recipe: { ...plain, cropSeconds: 120 }, name: 'a', resume: false });
  assert.ok(crop.join(' ').includes('--max-frames 1500'), 'a run from before the end frame keeps its random windows');
  const modern = train.buildHeartmulaTrainArgs({ dataDir: 'D', runDir: 'R', modelsDir: 'M', recipe: { ...HEARTMULA_RECIPE_DEFAULTS, cropSeconds: 120 }, name: 'a', resume: false }).join(' ');
  assert.ok(modern.includes('--crop-frames 1500 --eos'), 'new runs: start-anchored crops (120 s at 12.5 frames per second) and the end frame');
  assert.ok(!modern.includes('--max-frames'));
  assert.ok(train.buildHeartmulaTrainArgs({ dataDir: 'D', runDir: 'R', modelsDir: 'M', recipe: HEARTMULA_RECIPE_DEFAULTS, name: 'a', resume: false })
    .includes('--eos'), 'the end-of-song frame trains by default');
  assert.ok(!args.includes('--max-frames'), 'no crop, no flag');
  const mlp = train.buildHeartmulaTrainArgs({ dataDir: 'D', runDir: 'R', modelsDir: 'M', recipe: { ...HEARTMULA_RECIPE_DEFAULTS, trainMlp: true }, name: 'a', resume: false });
  assert.ok(mlp.join(' ').includes('--lora-dropout 0.05 --lora-mlp --name a'));
  const trig = train.buildHeartmulaTrainArgs({ dataDir: 'D', runDir: 'R', modelsDir: 'M', recipe: plain, name: 'a', resume: false, trigger: ' zx9 ' });
  assert.ok(trig.join(' ').includes('--trigger zx9 --name a'));
  assert.ok(!args.includes('--trigger'), 'no dataset trigger, no flag');
  assert.ok(!args.includes('--lora-mlp'), 'attention-only runs keep their old command line');
  const prepare = train.buildHeartmulaPrepareArgs({ samplesJson: 'S', outDir: 'O', modelsDir: 'M', encoderDir: 'E', force: true });
  assert.deepEqual(prepare, ['heartmula-prepare', '--dataset', 'S', '--out', 'O', '--models', 'M', '--encoder', 'E', '--force']);
});

test('models and encoder pack are found under the models folder only', () => {
  const models = config.aceServer.models;
  assert.equal(train.missingHeartmulaModels(models).length, 3);
  assert.equal(train.findHeartmulaEncoderDir(models), null);
  const dir = path.join(models, 'heartmula');
  fs.mkdirSync(dir, { recursive: true });
  for (const f of ['heartmula-3b-global-F16', 'heartmula-3b-local-F16', 'heartmula-tokenizer', 'heartcodec-rvq-F32']) {
    fs.writeFileSync(path.join(dir, `${f}.gguf`), 'x');
  }
  assert.deepEqual(train.missingHeartmulaModels(models), []);
  assert.equal(train.findHeartmulaEncoderDir(models), null);
  // Train bases: global+local pairs only, F16 first; the variant falls back to the filename without a header.
  fs.writeFileSync(path.join(dir, 'heartmula-3b-global-Q8_0.gguf'), 'x');
  assert.deepEqual(train.heartmulaTrainBases(models), [{ variant: '3b', quants: ['F16'] }], 'a global without its local is not a base');
  fs.writeFileSync(path.join(dir, 'heartmula-3b-local-Q8_0.gguf'), 'x');
  assert.deepEqual(train.heartmulaTrainBases(models), [{ variant: '3b', quants: ['F16', 'Q8_0'] }]);
  assert.equal(train.missingHeartmulaBase({ variant: '3B', quant: 'q8_0' }, models), '');
  assert.match(train.missingHeartmulaBase({ variant: '3B-RL', quant: 'F16' }, models), /3B-RL F16 .* not installed/);
  for (const r of ['muencoder', 'wavlm', 'whisper', 'query']) fs.writeFileSync(path.join(dir, `heartcodec-${r}-F16.gguf`), 'x');
  assert.equal(train.findHeartmulaEncoderDir(models), dir);
});

test('runs: new run, resume rules, foreign dataset refused', () => {
  const first = runs.resolveHeartmulaRun(ds, { recipe: { steps: 20 } });
  assert.equal(first.resume, false);
  assert.match(first.id, runs.HEARTMULA_RUN_ID);
  assert.equal(first.record.recipe.steps, 20);
  assert.throws(() => runs.resolveHeartmulaRun(ds, { resumeRun: first.id }), /no checkpoint/);
  fs.writeFileSync(path.join(first.dir, 'hm_train_state.bin'), 'x');
  assert.throws(() => runs.resolveHeartmulaRun(ds, { resumeRun: first.id, recipe: { steps: 30 } }), /cannot change steps/);
  const again = runs.resolveHeartmulaRun(ds, { resumeRun: first.id, recipe: { steps: 20 } });
  assert.equal(again.resume, true);
  // A run saved before trainMlp and cropSeconds existed trained attention-only on whole songs.
  const recordFile = path.join(first.dir, 'run.json');
  const legacy = JSON.parse(fs.readFileSync(recordFile, 'utf8'));
  delete legacy.recipe.trainMlp; delete legacy.recipe.cropSeconds; delete legacy.recipe.eos;
  fs.writeFileSync(recordFile, JSON.stringify(legacy));
  assert.equal(runs.resolveHeartmulaRun(ds, { resumeRun: first.id, recipe: { trainMlp: false, cropSeconds: 0 } }).resume, true);
  assert.throws(() => runs.resolveHeartmulaRun(ds, { resumeRun: first.id, recipe: { trainMlp: true } }), /cannot change trainMlp/);
  assert.throws(() => runs.resolveHeartmulaRun(ds, { resumeRun: first.id, recipe: { eos: true } }), /cannot change eos/);
  assert.equal(again.dir, first.dir);
  assert.throws(() => runs.resolveHeartmulaRun({ id: 'other', slug: 'demo' }, { resumeRun: first.id }), /another dataset/);
  assert.throws(() => runs.resolveHeartmulaRun(ds, { resumeRun: '../../x/1' }), /Invalid/);
  assert.throws(() => runs.resolveHeartmulaRun(ds, { adapterName: '../x' }), /adapterName/);
  assert.throws(() => runs.resolveHeartmulaRun(ds, { bogus: 1 }), /Unsupported/);
  assert.ok(runs.listHeartmulaRuns(ds.id, ds.slug).some(r => r.id === first.id));
});

test('status: a stored live state with no job behind it reads as failed and resumable', () => {
  const run = runs.listHeartmulaRuns(ds.id, ds.slug)[0];
  runs.writeStoredStatus(run.dir, { ...runs.blankStatus(20), state: 'running', step: 7 });
  assert.equal(runs.heartmulaRunStatus(run, true).state, 'running');
  const orphan = runs.heartmulaRunStatus(run, false);
  assert.equal(orphan.state, 'failed');
  assert.equal(orphan.step, 7);
});

test('publish copies a valid GGUF once and never overwrites', () => {
  const run = runs.listHeartmulaRuns(ds.id, ds.slug)[0];
  const file = runs.heartmulaAdapterFile(run.dir, 20);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const header = Buffer.alloc(24);
  header.write('GGUF'); header.writeUInt32LE(3, 4); header.writeBigUInt64LE(2n, 8);
  fs.writeFileSync(file, header);
  assert.deepEqual(runs.heartmulaCheckpointSteps(run.dir), [20]);
  assert.equal(runs.publishHeartmulaAdapter(file, 'demo-1'), 'demo-1.gguf');
  assert.equal(runs.publishHeartmulaAdapter(file, 'demo-1'), 'demo-1-2.gguf');
  fs.writeFileSync(file, 'nope');
  assert.throws(() => runs.publishHeartmulaAdapter(file, 'demo-2'), /invalid adapter/);
  runs.saveHeartmulaRunRecord(run.dir, { ...run.record, published: 'demo-1.gguf' });
  assert.equal(runs.latestHeartmulaAdapter(ds.id, ds.slug)?.name, 'demo-1');
});

test('prepare status: stale when audio or text changed, valid otherwise', () => {
  const audio = path.join(root, 'a.wav');
  fs.writeFileSync(audio, 'abcd');
  const st = fs.statSync(audio);
  const sample = {
    sampleId: 's1', audioPath: audio, caption: 'rock', genre: 'Rock', lyrics: 'la la', isInstrumental: false,
    customTag: '', excluded: false, fileMissing: false,
  } as never;
  const input = prep.heartmulaSampleInput(ds as never, sample);
  assert.equal(input.tags, 'trig, rock');
  const dir = path.join(root, 'training', 'datasets', 'demo', 'heartmula', 'prepare');
  fs.mkdirSync(dir, { recursive: true });
  const manifest = {
    version: 1, created_at: 'now', context_limit: 8192, skipped: [{ id: 'x', reason: 'over-context', detail: 'long' }],
    samples: [{ id: 's1', audio, audio_bytes: st.size, audio_mtime_s: Math.floor(st.mtimeMs / 1000), tags: input.tags, lyrics: input.lyrics }],
  };
  fs.writeFileSync(path.join(dir, 'heartmula_prepare.json'), JSON.stringify(manifest));
  const ok = prep.heartmulaPrepareStatus(ds as never, [sample]);
  assert.equal(ok.cacheValid, true);
  assert.equal(ok.preparedSamples, 1);
  assert.equal(ok.skipped[0].reason, 'over-context');
  assert.equal(prep.heartmulaPrepareStatus(ds as never, [{ ...(sample as object), lyrics: 'changed' } as never]).staleSamples, 1);
  const added = prep.heartmulaPrepareStatus(ds as never, [sample, { ...(sample as object), sampleId: 's2' } as never]);
  assert.equal(added.cacheValid, false);
  assert.equal(added.staleSamples, 1);
});

test('prepare status: correcting a skipped sample reads as stale, an untouched skip does not', () => {
  const audio = path.join(root, 'long.wav');
  fs.writeFileSync(audio, 'abcdefgh');
  const sample = {
    sampleId: 's9', audioPath: audio, caption: 'rock', genre: '', lyrics: 'a very long song', isInstrumental: false,
    customTag: '', excluded: false, fileMissing: false,
  } as never;
  const dir = path.join(root, 'training', 'datasets', 'demo2', 'heartmula', 'prepare');
  fs.mkdirSync(dir, { recursive: true });
  const other = path.join(root, 'ok.wav');
  fs.writeFileSync(other, 'zz');
  const ds2 = { id: 'ds2', slug: 'demo2', customTag: '' } as never;
  const okSample = { ...(sample as object), sampleId: 's1', audioPath: other, lyrics: 'ok' } as never;
  const okInput = prep.heartmulaSampleInput(ds2, okSample);
  const ost = fs.statSync(other);
  fs.writeFileSync(path.join(dir, 'heartmula_prepare.json'), JSON.stringify({
    version: 1, skipped: [{ id: 's9', reason: 'over-context', detail: 'long' }],
    samples: [{ id: 's1', audio: other, audio_bytes: ost.size, audio_mtime_s: Math.floor(ost.mtimeMs / 1000), tags: okInput.tags, lyrics: okInput.lyrics }],
  }));
  // A cache with no skip record is treated as changed.
  assert.equal(prep.heartmulaPrepareStatus(ds2, [okSample, sample]).cacheValid, false);
  prep.writeSkippedFingerprints('demo2', prep.snapshotSources(ds2, [okSample, sample]));
  assert.equal(prep.heartmulaPrepareStatus(ds2, [okSample, sample]).cacheValid, true);
  // The user shortens the lyrics: the skipped sample is stale and Train must wait for Prepare.
  const fixed = prep.heartmulaPrepareStatus(ds2, [okSample, { ...(sample as object), lyrics: 'short' } as never]);
  assert.equal(fixed.cacheValid, false);
  assert.equal(fixed.staleSamples, 1);
  fs.writeFileSync(audio, 'different audio bytes');
  assert.equal(prep.heartmulaPrepareStatus(ds2, [okSample, sample]).cacheValid, false);
  // Timing: the audio is replaced after the pre-Prepare snapshot but before the record is written.
  // The record must hold the version that was skipped, so the replacement reads as stale.
  fs.writeFileSync(audio, 'abcdefgh');
  const snap = prep.snapshotSources(ds2, [okSample, sample]);
  fs.writeFileSync(audio, 'replaced during prepare');
  prep.writeSkippedFingerprints('demo2', snap);
  const raced = prep.heartmulaPrepareStatus(ds2, [okSample, sample]);
  assert.equal(raced.cacheValid, false);
  assert.equal(raced.staleSamples, 1);
});

test('device gate: refuses a small, absent or unidentifiable selected GPU', async () => {
  const { heartmulaDeviceProblem } = await import('./heartmulaStatus.js');
  const gpu = (index: number, memoryMB: number) => ({ index, uuid: `GPU-${index}`, name: 'g', memoryMB });
  assert.equal(heartmulaDeviceProblem([gpu(0, 24 * 1024)]), null);
  assert.match(heartmulaDeviceProblem([gpu(0, 8 * 1024)]) ?? '', /needs at least 11\.9 GB/);
  assert.match(heartmulaDeviceProblem([]) ?? '', /could not be identified/);
  const two = [gpu(0, 24 * 1024), gpu(1, 8 * 1024)];
  assert.equal(heartmulaDeviceProblem(two), null); // auto picks the larger card
  config.aceServer.cudaVisibleDevices = 'GPU-1';
  try { assert.match(heartmulaDeviceProblem(two) ?? '', /needs at least 11\.9 GB/); } finally { config.aceServer.cudaVisibleDevices = ''; }
});

test('a missing encoder pack is described with the file names and the install step', () => {
  const message = train.missingEncoderMessage(config.aceServer.models);
  for (const role of ['muencoder', 'wavlm', 'whisper', 'query']) assert.ok(message.includes(`heartcodec-${role}-F16.gguf`), role);
  assert.ok(message.includes('convert-heartmula.py'));
  assert.ok(message.includes('--only encoder'));
});

test('preview prompt: tags from the first sung track, lyrics from Lyric Studio then the track, overrides win', async () => {
  const { resolveHeartmulaPreviewPrompt } = await import('./heartmulaPreviewSource.js');
  const sourceDir = path.join(root, 'pushed');
  fs.mkdirSync(sourceDir, { recursive: true });
  const dsx = { customTag: 'trig', lyricsSetId: undefined, sourceDir } as never;
  const mk = (sampleId: string, over: object) => ({ sampleId, filename: sampleId + '.wav', caption: 'rock', genre: '', lyrics: '', isInstrumental: false, excluded: false, fileMissing: false, ...over }) as never;
  const samples = [mk('intro', { isInstrumental: true }), mk('song', { caption: 'folk', lyrics: 'own words' }), mk('gone', { lyrics: 'x', excluded: true })];
  const D = HEARTMULA_PREVIEW_DEFAULTS;
  // no Lyric Studio generation: the dataset track gives both, and the instrumental intro is skipped
  let r = resolveHeartmulaPreviewPrompt(dsx, samples, D);
  assert.deepEqual([r.settings.tags, r.settings.lyrics, r.source.lyrics, r.source.sampleId], ['trig, folk', 'own words', 'dataset', 'song']);
  // a pushed generation wins, unless lyricsSource is 'dataset' or lyrics are overridden
  fs.writeFileSync(path.join(sourceDir, '_hotstep-preview-lyrics.json'), JSON.stringify({ id: 7, title: 'New', lyrics: 'fresh words' }));
  r = resolveHeartmulaPreviewPrompt(dsx, samples, D);
  assert.deepEqual([r.settings.lyrics, r.source.lyrics, r.source.generationId, r.source.generationTitle], ['fresh words', 'lyric-studio', 7, 'New']);
  assert.equal(resolveHeartmulaPreviewPrompt(dsx, samples, { ...D, lyricsSource: 'dataset' }).settings.lyrics, 'own words');
  r = resolveHeartmulaPreviewPrompt(dsx, samples, { ...D, tags: 'jazz', lyrics: 'mine' });
  assert.deepEqual([r.settings.tags, r.settings.lyrics, r.source.tags, r.source.lyrics, r.source.generationId], ['jazz', 'mine', 'override', 'override', null]);
  // nothing usable and no override: refused with a clear message
  assert.throws(() => resolveHeartmulaPreviewPrompt(dsx, [], D), /needs tags/);
});
