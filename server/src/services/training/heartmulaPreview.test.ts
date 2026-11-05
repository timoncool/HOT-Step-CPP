import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const root = fs.mkdtempSync(path.join(process.cwd(), '.test-heartmula-preview-'));
process.env.TRAINING_DIR = path.join(root, 'training');
process.env.DATA_DIR = path.join(root, 'library');
const { config } = await import('../../config.js');
config.aceServer.adapters = path.join(root, 'adapters');
const runs = await import('./heartmulaRuns.js');
const pv = await import('./heartmulaPreview.js');
const { nextPreviewStep } = await import('./heartmulaRunner.js');
const { HEARTMULA_PREVIEW_DEFAULTS, HEARTMULA_RECIPE_DEFAULTS } = await import('../../contracts/heartmulaTraining.js');
after(() => fs.rmSync(root, { recursive: true, force: true }));

const run = runs.createHeartmulaRun('ds1', 'demo', 'demo', { ...HEARTMULA_RECIPE_DEFAULTS, steps: 250 });

/** A header-only GGUF v3 with one tensor: all isGgufAdapter reads. */
function fakeAdapter(step: number, trigger?: string): string {
  const file = runs.heartmulaAdapterFile(run.dir, step);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const h = Buffer.alloc(32);
  h.write('GGUF', 0, 'ascii'); h.writeUInt32LE(3, 4); h.writeBigUInt64LE(1n, 8);
  if (trigger !== undefined) {
    const key = Buffer.from('heartmula.lora.trigger');
    const value = Buffer.from(trigger);
    const kv = Buffer.alloc(8 + key.length + 4 + 8 + value.length);
    kv.writeBigUInt64LE(BigInt(key.length)); key.copy(kv, 8); kv.writeUInt32LE(8, 8 + key.length);
    kv.writeBigUInt64LE(BigInt(value.length), 12 + key.length); value.copy(kv, 20 + key.length);
    fs.writeFileSync(file, Buffer.concat([h, kv]));
  } else fs.writeFileSync(file, h);
  return file;
}

const source = { tags: 'dataset', lyrics: 'lyric-studio', sampleId: 's1', sampleName: 's1.wav', generationId: 3, generationTitle: 'T' } as const;
const wav = Buffer.concat([Buffer.from('RIFF'), Buffer.alloc(4), Buffer.from('WAVE'), Buffer.alloc(40)]);

function deps(over: Partial<import('./heartmulaPreview.js').HeartmulaPreviewDeps> & { wires?: any[]; unloads?: number[] } = {}) {
  const wires = over.wires ?? [];
  const unloads = over.unloads ?? [];
  return {
    synth: async (w: any) => { wires.push(w); return 'job1'; },
    poll: async () => ({ status: 'done' }),
    detail: async () => ({ status: 'done' }),
    result: async () => new Response(wav),
    cancel: async () => ({}),
    unload: async () => { unloads.push(1); },
    beforeRender: async () => {},
    pick: () => ({ variant: '3B', global_type: 'F16' } as any),
    ...over,
  };
}

test('a base take renders with no adapter, a step take with its checkpoint, both with provenance', async () => {
  const wires: any[] = []; const unloads: number[] = [];
  const base = await pv.renderHeartmulaPreview({ runDir: run.dir, runId: run.id, step: 0, kind: 'base', settings: HEARTMULA_PREVIEW_DEFAULTS, source, deps: deps({ wires, unloads }) });
  assert.equal(base.status, 'done');
  assert.deepEqual(base.source, source);
  assert.equal(base.adapter, null);
  assert.equal(base.adapterSha256, null);
  assert.equal(wires[0].adapter, '');
  assert.equal(wires[0].duration_s, HEARTMULA_PREVIEW_DEFAULTS.seconds);
  assert.equal(wires[0].seed, HEARTMULA_PREVIEW_DEFAULTS.seed);
  assert.equal(wires[0].cfg_scale, HEARTMULA_PREVIEW_DEFAULTS.cfgScale);

  const file = fakeAdapter(100);
  const take = await pv.renderHeartmulaPreview({ runDir: run.dir, runId: run.id, step: 100, kind: 'step', settings: HEARTMULA_PREVIEW_DEFAULTS, deps: deps({ wires, unloads }) });
  assert.equal(take.status, 'done');
  assert.equal(wires[1].adapter, file);
  assert.equal(wires[1].adapter_scale, 1);
  assert.equal(take.adapter, 'adapters/step-000100.gguf');
  assert.match(take.adapterSha256 ?? '', /^[a-f0-9]{64}$/);
  assert.ok((take.renderMs ?? -1) >= 0);
  assert.equal(unloads.length, 2, 'the model is unloaded after every take');

  const list = pv.listHeartmulaPreviewRecords(run.dir);
  assert.deepEqual(list.map(r => [r.kind, r.step]), [['base', 0], ['step', 100]], 'base first, then by step');
  assert.ok(pv.heartmulaPreviewDone(run.dir, 100));
  assert.ok(!pv.heartmulaPreviewDone(run.dir, 200));
});

test('checkpoint tags match normal render trigger handling, including overrides and byte limits', async () => {
  const { withHeartmulaTrigger } = await import('../backends/heartmula/models.js');
  for (const [index, tags] of ['folk', 'ZX9, folk', 'folk, zx9', '', 'é'.repeat(2044)].entries()) {
    const step = 300 + index;
    fakeAdapter(step, 'zx9');
    const wires: any[] = [];
    const take = await pv.renderHeartmulaPreview({ runDir: run.dir, runId: run.id, step, kind: 'step',
      settings: { ...HEARTMULA_PREVIEW_DEFAULTS, tags }, deps: deps({ wires }) });
    assert.equal(take.status, 'done');
    assert.equal(wires[0].tags, withHeartmulaTrigger(tags, 'zx9'));
    assert.equal(take.settings.tags, tags, 'saved settings retain the original override');
  }
  fakeAdapter(310, 'zx9');
  const wires: any[] = [];
  const failed = await pv.renderHeartmulaPreview({ runDir: run.dir, runId: run.id, step: 310, kind: 'step',
    settings: { ...HEARTMULA_PREVIEW_DEFAULTS, tags: 'é'.repeat(2046) }, deps: deps({ wires }) });
  assert.equal(failed.status, 'failed');
  assert.match(failed.error ?? '', /4096 bytes/);
  assert.equal(wires.length, 0);
  const base = await pv.renderHeartmulaPreview({ runDir: run.dir, runId: run.id, step: 0, kind: 'base',
    settings: { ...HEARTMULA_PREVIEW_DEFAULTS, tags: 'folk' }, deps: deps({ wires }) });
  assert.equal(base.status, 'done');
  assert.equal(wires[0].tags, 'folk');
});

test('audio is served by the file name of a done record only', async () => {
  const take = pv.listHeartmulaPreviewRecords(run.dir).find(r => r.step === 100)!;
  assert.ok(take.file);
  assert.ok(pv.heartmulaPreviewAudioPath(run.dir, take.file!));
  assert.equal(pv.heartmulaPreviewAudioPath(run.dir, '../index.json'), null);
  assert.equal(pv.heartmulaPreviewAudioPath(run.dir, 'step-100-deadbeef.wav'), null, 'a name no record holds');
  assert.equal(pv.heartmulaPreviewAudioPath(run.dir, 'index.json'), null);
});

test('a failed render is recorded, training can go on, and a retry replaces it', async () => {
  fakeAdapter(200);
  const failed = await pv.renderHeartmulaPreview({ runDir: run.dir, runId: run.id, step: 200, kind: 'step', settings: HEARTMULA_PREVIEW_DEFAULTS,
    deps: deps({ synth: async () => { throw new Error('vram_alloc_failed'); } }) });
  assert.equal(failed.status, 'failed');
  assert.match(failed.error ?? '', /vram_alloc_failed/);
  assert.ok(!pv.heartmulaPreviewDone(run.dir, 200));
  const again = await pv.renderHeartmulaPreview({ runDir: run.dir, runId: run.id, step: 200, kind: 'step', settings: HEARTMULA_PREVIEW_DEFAULTS, deps: deps() });
  assert.equal(again.status, 'done');
  assert.equal(pv.listHeartmulaPreviewRecords(run.dir).filter(r => r.step === 200).length, 1, 'the failed record is gone');
});

test('a missing or invalid adapter fails the take instead of rendering the base model under its name', async () => {
  const wires: any[] = [];
  const take = await pv.renderHeartmulaPreview({ runDir: run.dir, runId: run.id, step: 150, kind: 'step', settings: HEARTMULA_PREVIEW_DEFAULTS, deps: deps({ wires }) });
  assert.equal(take.status, 'failed');
  assert.match(take.error ?? '', /No valid adapter/);
  assert.equal(wires.length, 0);
});

test('cancel marks the take skipped and cancels the engine job', async () => {
  fakeAdapter(100);
  const ctl = new AbortController();
  const cancelled: string[] = [];
  const take = await pv.renderHeartmulaPreview({ runDir: run.dir, runId: run.id, step: 100, kind: 'step', settings: { ...HEARTMULA_PREVIEW_DEFAULTS, seed: 9 }, signal: ctl.signal,
    deps: deps({
      poll: async (id: string) => { ctl.abort(); return { status: cancelled.includes(id) ? 'cancelled' : 'running' }; },
      cancel: async (id: string) => { cancelled.push(id); return {}; },
    }) });
  assert.equal(take.status, 'skipped');
  assert.deepEqual(cancelled, ['job1']);
});

test('an unload the engine refuses is a cleanup error; no engine at all is not', async () => {
  await assert.rejects(pv.renderHeartmulaPreview({ runDir: run.dir, runId: run.id, step: 0, kind: 'base', settings: { ...HEARTMULA_PREVIEW_DEFAULTS, seed: 1 },
    deps: deps({ unload: async () => { throw new Error('did not confirm'); } }) }), pv.HeartmulaPreviewCleanupError);
  const gone = await pv.renderHeartmulaPreview({ runDir: run.dir, runId: run.id, step: 0, kind: 'base', settings: { ...HEARTMULA_PREVIEW_DEFAULTS, seed: 2 },
    deps: deps({ unload: async () => { throw new TypeError('fetch failed'); } }) });
  assert.equal(gone.status, 'done');
  assert.ok(!pv.heartmulaPreviewBusy(run.dir), 'the busy flag is released on every path');
});

test('a render cut off by a restart reads as failed, and the pause cost lands on its take', () => {
  const dir = runs.createHeartmulaRun('ds1', 'demo', 'cut', HEARTMULA_RECIPE_DEFAULTS).dir;
  fs.mkdirSync(path.join(dir, 'previews'));
  fs.writeFileSync(path.join(dir, 'previews', 'index.json'), JSON.stringify([{
    id: 'a', run: 'x', kind: 'step', step: 100, settings: HEARTMULA_PREVIEW_DEFAULTS, status: 'rendering', adapter: null, adapterSha256: null,
    renderMs: null, pauseMs: null, error: null, createdAt: 1,
  }]));
  const [r] = pv.listHeartmulaPreviewRecords(dir);
  assert.equal(r.status, 'failed');
  assert.match(r.error ?? '', /Interrupted/);
  pv.setHeartmulaPreviewPauseMs(dir, 'a', 41234.6);
  assert.equal(pv.listHeartmulaPreviewRecords(dir)[0].pauseMs, 41235);
});

test('preview cadence: the step after `from`, never at or past the end', () => {
  assert.equal(nextPreviewStep(0, 100, 480), 100);
  assert.equal(nextPreviewStep(100, 100, 480), 200);
  assert.equal(nextPreviewStep(150, 100, 480), 200);
  assert.equal(nextPreviewStep(400, 100, 480), 0, 'the next pause (500) is past the end: run to the finish');
  assert.equal(nextPreviewStep(300, 100, 400), 0, 'a pause at the final step is the final take, not a pause');
  assert.equal(nextPreviewStep(0, 0, 480), 0, 'previews off');
});
