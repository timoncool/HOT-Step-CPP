import test from 'node:test';
import assert from 'node:assert/strict';
import { barsToDurationS, mapJobStatus, parseDawRequest, toGenerateParams, DawError } from './dawSchema.js';

test('bars convert to duration_s from bpm and time signature', () => {
  assert.equal(barsToDurationS(4, 120, '4/4'), 8);
  assert.equal(barsToDurationS(2, 90, '3/4'), 4);
  assert.equal(barsToDurationS(4, 120, '6/8'), 6);
  const r = parseDawRequest({ mode: 'text2music', caption: 'x', bars: 8, bpm: 120, timesig: '4/4' });
  assert.equal(r.duration_s, 16);
});

test('every DAW field maps to the /api/generate name translateParams reads', () => {
  const req = parseDawRequest({
    mode: 'repaint', caption: 'warm keys', lyrics: 'la', negative: 'noise', duration_s: 12, bpm: 97,
    timesig: '7/8', key: 'C minor', seed: 42, seed_b: 7, seed_blend: 0.3, seed_blend_end: 0.8, takes: 2,
    region: { start_s: 2, end_s: 6 }, track: 'drums', cover_strength: 0.5, steps: 30, guidance: 4.5,
    solver: 'euler', scheduler: 'linear', guidance_mode: 'apg', dit_model: 'base.gguf', expected_backend: 'ace',
  });
  const p = toGenerateParams(req, '/tmp/src.wav');
  const expected: Record<string, unknown> = {
    taskType: 'repaint', caption: 'warm keys', skipLm: true, lyrics: 'la', negative_prompt: 'noise',
    duration: 12, bpm: 97, timeSignature: '7/8', keyScale: 'C minor', seed: 42, randomSeed: false,
    seedB: 7, seedBlend: 0.3, seedBlendEnd: 0.8, batchSize: 2, repaintingStart: 2, repaintingEnd: 6,
    trackName: 'drums', audioCoverStrength: 0.5, inferenceSteps: 30, guidanceScale: 4.5, inferMethod: 'euler',
    scheduler: 'linear', guidanceMode: 'apg', ditModel: 'base.gguf', expectedBackend: 'ace',
    sourceAudioUrl: '/tmp/src.wav',
  };
  console.log('mapped params:', JSON.stringify(p));
  assert.deepEqual(p, expected);
});

test('skip_lm defaults true for source modes, false for text2music, and is overridable', () => {
  assert.equal(toGenerateParams(parseDawRequest({ mode: 'lego', caption: '' })).skipLm, true);
  assert.equal(toGenerateParams(parseDawRequest({ mode: 'text2music', caption: '' })).skipLm, false);
  assert.equal(toGenerateParams(parseDawRequest({ mode: 'lego', caption: '', skip_lm: false })).skipLm, false);
  assert.equal(toGenerateParams(parseDawRequest({ mode: 'text2music', caption: '' })).instrumental, true);
});

test('bad input is rejected with the envelope codes', () => {
  const code = (raw: unknown) => { try { parseDawRequest(raw); return null; } catch (e) { return (e as DawError).code; } };
  assert.equal(code({ mode: 'extract' }), 'unsupported_mode');
  assert.equal(code({ mode: 'repaint', caption: '' }), 'bad_request');
  assert.equal(code({ mode: 'lego', bars: 4 }), 'bad_request');
  assert.equal(code({ mode: 'lego', steps: 'many' }), 'bad_request');
});

test('upstream status maps to DAW job shape with progress 0..1', () => {
  const run = mapJobStatus('j', { status: 'synth_running', stage: 'DiT', progress: 40, ace_phase_progress: 'step 12/50' });
  assert.equal(run.state, 'running');
  assert.equal(run.progress, 0.4);
  assert.equal(run.step, 12);
  assert.equal(run.step_total, 50);
  assert.equal(mapJobStatus('j', { status: 'pending', progress: 0 }).state, 'queued');
  const done = mapJobStatus('j', { status: 'succeeded', progress: 90, result: { audioUrls: ['/audio/a.mp3', '/audio/b.mp3'], durations: [10, 11] } }, [5, 5]);
  console.log('done job:', JSON.stringify(done));
  assert.equal(done.state, 'done');
  assert.equal(done.progress, 1);
  assert.deepEqual(done.takes, [{ n: 0, seed: 5, duration_s: 10, ready: true }, { n: 1, seed: 5, duration_s: 11, ready: true }]);
  assert.equal(done.api, 1);
});

test('only acestep-v15-base-* and acestep-v15-xl-base-* count as base (SFT and merges do not)', async () => {
  const s: any = await import('./dawSchema.js');
  const names = ['acestep-v15-base-Q8_0.gguf', 'acestep-v15-xl-base-BF16.gguf', 'acestep-v15-sft-Q8_0.gguf',
    'acestep-v15-merge-base-turbo-Q8_0.gguf', 'my-custom-merge.gguf', 'acestep-v15-turbo-Q8_0.gguf'];
  const kinds = names.map((n) => s.ditKind(n));
  console.log('ditKind:', JSON.stringify(kinds));
  assert.deepEqual(kinds, ['base', 'base', 'sft', 'turbo', 'other', 'turbo']);
});

test('lego with no base model installed -> 422 base_model_required', async () => {
  const s: any = await import('./dawSchema.js');
  const req = s.parseDawRequest({ mode: 'lego', caption: 'bass' });
  const run = (r: any, list: string[]) => { try { return s.resolveDitModel(r, list, null); } catch (e: any) { return `${e.status} ${e.code}`; } };
  const none = run(req, ['acestep-v15-turbo-Q8_0.gguf', 'acestep-v15-sft-Q8_0.gguf', 'my-merge.gguf']);
  const nonBase = run({ ...req, dit_model: 'acestep-v15-sft-Q8_0.gguf' }, ['acestep-v15-sft-Q8_0.gguf', 'acestep-v15-base-Q8_0.gguf']);
  console.log('lego no base:', none, '| lego sft passed:', nonBase);
  assert.equal(none, '422 base_model_required');
  assert.equal(nonBase, '422 base_model_required');
});

test('lego auto-picks a base model (loaded one first); repaint and complete gating', async () => {
  const s: any = await import('./dawSchema.js');
  const lego = s.parseDawRequest({ mode: 'lego', caption: 'bass' });
  const list = ['acestep-v15-turbo-Q8_0.gguf', 'acestep-v15-xl-base-Q8_0.gguf', 'acestep-v15-base-Q8_0.gguf'];
  const auto = s.resolveDitModel(lego, list, null);
  const loaded = s.resolveDitModel(lego, list, 'acestep-v15-turbo-Q8_0.gguf');
  const rep = s.resolveDitModel(s.parseDawRequest({ mode: 'repaint', caption: '', region: { start_s: 0, end_s: 1 } }), ['acestep-v15-turbo-Q8_0.gguf'], null);
  let complete = '';
  try { s.parseDawRequest({ mode: 'complete', caption: '' }); } catch (e: any) { complete = `${e.status} ${e.code}`; }
  console.log('auto:', auto, '| loaded:', loaded, '| repaint:', rep, '| complete:', complete);
  assert.equal(auto, 'acestep-v15-xl-base-Q8_0.gguf'); // first installed base
  assert.equal(s.resolveDitModel(lego, list, 'acestep-v15-base-Q8_0.gguf'), 'acestep-v15-base-Q8_0.gguf');
  assert.equal(loaded, 'acestep-v15-xl-base-Q8_0.gguf'); // loaded turbo is ignored
  assert.equal(rep, undefined);
  assert.equal(complete, '400 unsupported_mode');
});

test('model field: turbo, base, exact name, unknown 400, turbo+lego 400', async () => {
  const s: any = await import('./dawSchema.js');
  const list = ['acestep-v15-base-Q8_0.gguf', 'acestep-v15-turbo-Q8_0.gguf', 'acestep-v15-turbo-shift3-Q8_0.gguf', 'acestep-v15-sft-Q8_0.gguf'];
  const run = (raw: any, loaded: string | null = null) => {
    try { return s.resolveDitModel(s.parseDawRequest(raw), list, loaded); } catch (e: any) { return `${e.status} ${e.code}: ${e.message}`; }
  };
  const t2m = { mode: 'text2music', caption: 'x' };
  const turbo = run({ ...t2m, model: 'turbo' }, 'acestep-v15-base-Q8_0.gguf');
  const turboLoaded = run({ ...t2m, model: 'turbo' }, 'acestep-v15-turbo-shift3-Q8_0.gguf');
  const base = run({ ...t2m, model: 'base' }, 'acestep-v15-turbo-Q8_0.gguf');
  const exact = run({ ...t2m, model: 'acestep-v15-sft-Q8_0.gguf' });
  const unknown = run({ ...t2m, model: 'nope.gguf' });
  const legoTurbo = run({ mode: 'lego', caption: 'b', model: 'turbo' });
  const legoBase = run({ mode: 'lego', caption: 'b', model: 'base' });
  console.log('model turbo:', turbo, '| turbo loaded:', turboLoaded, '| base:', base, '| exact:', exact);
  console.log('model unknown:', unknown, '| lego turbo:', legoTurbo, '| lego base:', legoBase);
  assert.equal(turbo, 'acestep-v15-turbo-Q8_0.gguf');
  assert.equal(turboLoaded, 'acestep-v15-turbo-shift3-Q8_0.gguf');
  assert.equal(base, 'acestep-v15-base-Q8_0.gguf');
  assert.equal(exact, 'acestep-v15-sft-Q8_0.gguf');
  assert.match(String(unknown), /^400 unknown_model: .*acestep-v15-sft-Q8_0\.gguf/);
  assert.match(String(legoTurbo), /^400 bad_request: lego needs a base/);
  assert.equal(legoBase, 'acestep-v15-base-Q8_0.gguf');
  // model type check
  assert.match(String(run({ ...t2m, model: 5 })), /^400 bad_request/);
  const p = s.toGenerateParams({ ...s.parseDawRequest(t2m), dit_model: turbo });
  assert.equal(p.ditModel, 'acestep-v15-turbo-Q8_0.gguf');
});

test('typed fields are checked: strings, booleans, seed ranges, 0..1 fractions', () => {
  const bad = (extra: Record<string, unknown>, field: string) => {
    assert.throws(() => parseDawRequest({ mode: 'text2music', caption: 'x', ...extra }),
      (e) => e instanceof DawError && e.status === 400 && e.code === 'bad_request' && e.message.includes(field), JSON.stringify(extra));
  };
  bad({ caption: 5 }, 'caption');
  bad({ lyrics: ['a'] }, 'lyrics');
  bad({ key: 3 }, 'key');
  bad({ expected_backend: {} }, 'expected_backend');
  bad({ skip_lm: 'false' }, 'skip_lm');
  bad({ seed: -1 }, 'seed');
  bad({ seed: 1.5 }, 'seed');
  bad({ seed: 4294967296 }, 'seed');
  bad({ seed: true }, 'seed');
  bad({ seed_b: -3 }, 'seed_b');
  bad({ seed_blend: 1.5 }, 'seed_blend');
  bad({ cover_strength: -0.1 }, 'cover_strength');
  bad({ steps: 0 }, 'steps');
  bad({ steps: 2.5 }, 'steps');
  bad({ guidance: -1 }, 'guidance');
  bad({ bpm: 0 }, 'bpm');
  bad({ bpm: [120] }, 'bpm');
  const ok = parseDawRequest({ mode: 'text2music', caption: 'x', seed: '42', seed_b: 0, seed_blend: 1, steps: 8, skip_lm: false });
  assert.equal(ok.seed, 42);
});

test('region and derived duration: strict numbers, objects only, finite result', () => {
  const bad = (extra: Record<string, unknown>) => assert.throws(
    () => parseDawRequest({ mode: 'repaint', caption: 'x', ...extra }),
    (e) => e instanceof DawError && e.status === 400 && e.code === 'bad_request', JSON.stringify(extra));
  bad({ region: { start_s: false, end_s: true } });
  bad({ region: { start_s: null, end_s: 4 } });
  bad({ region: { end_s: 4 } });
  bad({ region: { start_s: 1, end_s: [4] } });
  bad({ region: null });
  bad({ region: [] });
  bad({ region: 'x' });
  bad({ region: { start_s: 4, end_s: 2 } });
  assert.deepEqual(parseDawRequest({ mode: 'repaint', caption: 'x', region: { start_s: '1', end_s: 3 } }).region, { start_s: 1, end_s: 3 });
  bad({ mode: 'text2music', bars: 1e308, bpm: 1 });
});
