import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { abcKeyString, abcSidecarPath, readAbcSidecar, withAbcTempo, writeAbcSidecar } from './abcSidecar.js';
import { seedAbcManifestFromSidecars, saveAbcManifestSidecars, buildYue2SheetArgs, readYue2CoverAbc } from './yue2Sheet.js';
import { runYue2CoverSheetJob, runYue2SheetJob, transcribeLabelSheets } from './yue2ArTrainRunner.js';
import { jobsDir } from './paths.js';
import { createJob } from './labelingQueue.js';
import { runYue2AceTrain } from './yue2TrainRunner.js';
import { writeWav } from '../generation/audioLevel.js';

test('lead sheets survive manifest reset and seed the next stage by source name', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'abc-sidecar-'));
  try {
    const audio = path.join(dir, 'track.wav');
    const manifest = path.join(dir, 'preprocess.json');
    fs.writeFileSync(audio, 'audio');
    assert.equal(writeAbcSidecar(audio, 'X:1\nK:C\nC'), true);
    assert.equal(path.basename(abcSidecarPath(audio)), 'track.abc');
    fs.writeFileSync(manifest, JSON.stringify({ sources: [{ name: 'track.wav', source: audio }] }));
    assert.equal(seedAbcManifestFromSidecars(manifest), 1);
    assert.equal(JSON.parse(fs.readFileSync(manifest, 'utf8')).sources[0].abc, 'X:1\nK:C\nC');
    assert.equal(buildYue2SheetArgs({ manifest, only: '', force: false, fast: false, datasetSlug: '' }).includes('--force'), false);
    fs.rmSync(manifest);
    assert.equal(readAbcSidecar(audio), 'X:1\nK:C\nC');
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test('a staged manifest (preprocess copies) seeds from and saves to the ORIGINAL audio folder', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'abc-staged-'));
  try {
    // The dataset folder, with a sheet the Label step saved.
    const original = path.join(root, 'dataset', 'track.mp3');
    fs.mkdirSync(path.dirname(original), { recursive: true });
    fs.writeFileSync(original, 'audio');
    writeAbcSidecar(original, 'X:1\nK:G\nG');
    // The YuE2 cache: a staged copy, the staging map, and a manifest pointing at the copy.
    const cache = path.join(root, 'yue2-latents');
    const staged = path.join(cache, '_staged_audio', 'track.mp3');
    fs.mkdirSync(path.dirname(staged), { recursive: true });
    fs.copyFileSync(original, staged);
    fs.writeFileSync(path.join(cache, '_staged_audio.sources.json'),
      JSON.stringify([{ name: 'track.mp3', source: original, relPath: 'track.mp3' }]));
    const manifest = path.join(cache, 'yue2_preprocess.json');
    fs.writeFileSync(manifest, JSON.stringify({ sources: [{ name: 'track.mp3', source: staged }] }));

    assert.equal(seedAbcManifestFromSidecars(manifest), 1);
    assert.equal(JSON.parse(fs.readFileSync(manifest, 'utf8')).sources[0].abc, 'X:1\nK:G\nG');

    fs.writeFileSync(manifest, JSON.stringify({ sources: [{ name: 'track.mp3', source: staged, abc: 'X:1\nK:A\nA' }] }));
    assert.equal(saveAbcManifestSidecars(manifest), 1);
    assert.equal(readAbcSidecar(original), 'X:1\nK:A\nA');
    assert.equal(readAbcSidecar(staged), '');   // never beside the copy
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('successful force result replaces the sidecar, while empty and error rows do not', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'abc-sidecar-'));
  try {
    const audio = path.join(dir, 'track.wav');
    const manifest = path.join(dir, 'preprocess.json');
    fs.writeFileSync(audio, 'audio');
    writeAbcSidecar(audio, 'X:1\nK:C\nC');
    const row = { name: 'track.wav', source: audio, abc: '', abc_error: 'failed' };
    fs.writeFileSync(manifest, JSON.stringify({ sources: [row] }));
    assert.equal(saveAbcManifestSidecars(manifest), 0);
    assert.equal(readAbcSidecar(audio), 'X:1\nK:C\nC');
    row.abc = 'X:1\nK:D\nD';
    fs.writeFileSync(manifest, JSON.stringify({ sources: [row] }));
    assert.equal(saveAbcManifestSidecars(manifest), 0);
    delete (row as { abc_error?: string }).abc_error;
    fs.writeFileSync(manifest, JSON.stringify({ sources: [row] }));
    assert.equal(buildYue2SheetArgs({ manifest, only: '', force: true, fast: false, datasetSlug: '' }).includes('--force'), true);
    assert.equal(saveAbcManifestSidecars(manifest), 1);
    assert.equal(readAbcSidecar(audio), 'X:1\nK:D\nD');
    assert.equal(writeAbcSidecar(audio, '  '), false);
    assert.equal(writeAbcSidecar(audio, 'X:1\nV: Vocal\nZ4|z4|\nV: Ins\nC4|'), false);
    assert.equal(readAbcSidecar(audio), 'X:1\nK:D\nD');
    // Cover caching opts in to a sounding Ins under a silent Vocal.
    assert.equal(writeAbcSidecar(audio, 'X:1\nV: Vocal\nZ4|z4|\nV: Ins\nC4|', { anySounding: true }), true);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test('cover transcription fails for all-rest Vocal bars and accepts a sounding bar', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cover-sheet-job-'));
  try {
    const audio = path.join(dir, 'track.wav');
    fs.writeFileSync(audio, 'audio');
    let abc = 'X:1\nV: Vocal\nZ4|z4|\nV: Ins\nz4|';
    const run = (async (_job, _kind, _args, _idleMs, verify) => {
      const manifest = path.join(dir, 'cover-sheet.json');
      const data = JSON.parse(fs.readFileSync(manifest, 'utf8'));
      data.sources[0].abc = abc;
      fs.writeFileSync(manifest, JSON.stringify(data));
      if (verify?.()) throw new Error(verify()!);
    }) as typeof runYue2AceTrain;
    const deps = { missingModels: () => [], run };
    const silent = createJob('yue2-sheet', 'fixture', [], {});
    await assert.rejects(runYue2CoverSheetJob(silent, audio, dir, deps), /The transcriber heard no melody in this source/);
    assert.equal(silent.status, 'failed');
    assert.equal(fs.existsSync(abcSidecarPath(audio)), false);
    abc = 'X:1\nV: Vocal\nz4|C4|\nV: Ins\nC4|';
    const sounding = createJob('yue2-sheet', 'fixture', [], {});
    assert.equal(await runYue2CoverSheetJob(sounding, audio, dir, deps), abc);
    assert.equal(sounding.status, 'done');

    // Rap: no sung melody, but the vocal stem gives the rhythm. Bar 2 holds a
    // syllable on its first two 16ths at 120 BPM.
    abc = 'X:1\nM:4/4\nL:1/16\nQ:1/4=120\nV: Vocal\nV: Ins\nK:Am\n% verse\nV: Vocal\nZ|"Am"z16|\nV: Ins\nE16|E16|';
    const stem = path.join(dir, 'vocals.wav');
    const samples = new Float32Array(4 * 8000);
    for (let i = 16000; i < 18000; i++) samples[i] = 0.5 * Math.sin(i);
    writeWav(stem, { channels: [samples], sampleRate: 8000, numChannels: 1, sourceFormat: 's16' }, 's16');
    const rapped = createJob('yue2-sheet', 'fixture', [], {});
    const filled = await runYue2CoverSheetJob(rapped, audio, dir, { ...deps, vocalStem: async () => stem });
    assert.ok(filled.endsWith('V: Vocal\nz16|"Am"A2z12z2|\nV: Ins\nE16|E16|'), filled);
    assert.equal(rapped.status, 'done');
    assert.equal(readYue2CoverAbc(path.join(dir, 'cover-sheet.json'), 'track.wav'), filled);
    const noStem = createJob('yue2-sheet', 'fixture', [], {});
    // Sounding Ins survives a failed separation (production always passes vocalStem).
    const failStem = { ...deps, vocalStem: async () => { throw new Error('separation found no vocals'); } };
    assert.equal(await runYue2CoverSheetJob(noStem, audio, dir, failStem), abc);
    // Nothing sounding anywhere still fails.
    abc = ['X:1', 'V: Vocal', 'Z4|z4|', 'V: Ins', 'z4|'].join(String.fromCharCode(10));
    await assert.rejects(runYue2CoverSheetJob(createJob('yue2-sheet', 'fixture', [], {}), audio, dir, failStem),
      /heard no melody in this source, and its vocal rhythm could not be read: separation found no vocals/);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test('dataset sheet job seeds saved rows, writes new scores and replaces them on force', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'abc-sheet-job-'));
  try {
    const first = path.join(dir, 'first.wav');
    const second = path.join(dir, 'second.wav');
    const manifest = path.join(dir, 'preprocess.json');
    fs.writeFileSync(first, 'audio 1'); fs.writeFileSync(second, 'audio 2');
    writeAbcSidecar(first, 'X:1\nK:C\nC');
    const sources = [{ name: 'first.wav', source: first }, { name: 'second.wav', source: second }];
    fs.writeFileSync(manifest, JSON.stringify({ sources }));
    let force = false;
    const run = (async (_job, _kind, args, _idleMs, _verify, _onLine, state) => {
      assert.equal(args.includes('--force'), force);
      const data = JSON.parse(fs.readFileSync(manifest, 'utf8'));
      assert.equal(data.sources[0].abc, 'X:1\nK:C\nC');
      data.sources[0].abc = force ? 'X:1\nK:D\nD' : data.sources[0].abc;
      data.sources[1].abc = 'X:1\nK:G\nG';
      fs.writeFileSync(manifest, JSON.stringify(data));
      return state;
    }) as typeof runYue2AceTrain;
    const deps = { missingModels: () => [], run };
    const job = createJob('yue2-sheet', 'fixture', [], { manifest, only: '', force: false, fast: false, datasetSlug: '' });
    await runYue2SheetJob(job, deps);
    assert.equal(job.status, 'done');
    assert.equal(readAbcSidecar(first), 'X:1\nK:C\nC');
    assert.equal(readAbcSidecar(second), 'X:1\nK:G\nG');
    force = true;
    const retry = createJob('yue2-sheet', 'fixture', [], { manifest, only: '', force: true, fast: false, datasetSlug: '' });
    await runYue2SheetJob(retry, deps);
    assert.equal(retry.status, 'done');
    assert.equal(readAbcSidecar(first), 'X:1\nK:D\nD');
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test('label key pass transcribes only files without a lead sheet and reads key and tempo helpers', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'label-sheets-'));
  const job = createJob('label', 'fixture', [], {});
  try {
    const saved = path.join(dir, 'saved.wav');
    const fresh = path.join(dir, 'fresh.wav');
    const instrumental = path.join(dir, 'instrumental.wav');
    for (const f of [saved, fresh, instrumental]) fs.writeFileSync(f, 'audio');
    writeAbcSidecar(saved, 'X:1\nQ:1/4=135\nV: Vocal\nK:Em\nV: Vocal\nE4|');
    const run = (async (_job, _kind, args, _idleMs, _verify, _onLine, state) => {
      const manifest = args[args.indexOf('--manifest') + 1];
      const data = JSON.parse(fs.readFileSync(manifest, 'utf8'));
      assert.deepEqual(data.sources.map((s: { name: string }) => s.name), ['fresh.wav', 'instrumental.wav']);
      data.sources[0].abc = 'X:1\nQ:1/4=96\nV: Vocal\nK:F#m\nV: Vocal\nF4|';
      data.sources[1].abc = 'X:1\nQ:1/4=120\nV: Vocal\nK:Bb\nV: Vocal\nZ4|';
      fs.writeFileSync(manifest, JSON.stringify(data));
      return state;
    }) as typeof runYue2AceTrain;
    const scores = new Map<string, string>();
    await transcribeLabelSheets(job, [saved, fresh, instrumental], (audio, abc) => {
      // Each score is saved before it is handed on.
      if (audio === fresh) assert.equal(abcKeyString(readAbcSidecar(fresh)), 'F# Minor');
      scores.set(audio, abc);
    }, { missingModels: () => [], run, exe: () => 'ace-train' });
    assert.equal(abcKeyString(readAbcSidecar(fresh)), 'F# Minor');
    // An instrumental score has no sung bar, so it is not saved, but its key still counts.
    assert.equal(readAbcSidecar(instrumental), '');
    assert.equal(abcKeyString(scores.get(instrumental)!), 'Bb Major');
    assert.equal(scores.has(saved), false);
    assert.equal(withAbcTempo(readAbcSidecar(saved), 123.4), 'X:1\nQ:1/4=123\nV: Vocal\nK:Em\nV: Vocal\nE4|');
    assert.equal(withAbcTempo('X:1\nK:C', 100), 'X:1\nK:C');
    // The score's beat level wins: a double- or half-time measurement is folded onto it.
    assert.equal(withAbcTempo('Q:1/4=80\nK:Bb', 157), 'Q:1/4=79\nK:Bb');
    assert.equal(withAbcTempo('Q:1/4=183\nK:Eb', 88.5), 'Q:1/4=177\nK:Eb');
    assert.equal(abcKeyString('X:1\nK:Dmaj'), 'D Major');
    assert.equal(abcKeyString('X:1\nK:none'), '');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
    fs.rmSync(path.join(jobsDir(), job.id), { recursive: true, force: true });
  }
});
