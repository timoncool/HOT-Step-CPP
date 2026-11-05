import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { sa3Prompt, prepareSa3Dataset } from './sa3Dataset.js';
import { sa3Hash, type Sa3Instrumental } from './sa3Stems.js';

test('SA3 maps parsed sidecars and dataset rows to an allowlisted prompt', () => {
  const expected = 'warm analogue pad, synthwave, 96 BPM, A minor, 4/4';
  assert.equal(sa3Prompt({ caption: ' warm analogue\n pad ', genre: 'synthwave', bpm: '96', key: 'A minor', signature: '4/4',
    lyrics: 'Never include these lyrics', prompt_override: 'override', custom_tag: 'tag', repeat: 10, duration: 999 }), expected);
  assert.equal(sa3Prompt({ caption: 'warm analogue pad', genre: 'synthwave', bpm: 96, keyscale: 'A minor', timesignature: '4/4' }), expected);
  assert.equal(sa3Prompt({ caption: 'pad', timesignature: '6' }), 'pad, 6');
  assert.equal(sa3Prompt({ caption: 'pad', signature: '6/4', timesignature: '6' }), 'pad, 6/4');
  assert.equal(sa3Prompt({ caption: 'pad', bpm: null, genre: '', key: '', signature: '' }), 'pad');
  assert.equal(sa3Prompt({ bpm: 'not tempo', lyrics: 'lyrics' }), '');
  assert.equal(sa3Prompt({ bpm: -1, duration: 40 }), '');
});

async function fixture(t: test.TestContext) {
  const root = await fs.promises.mkdtemp(path.join(process.cwd(), '.test-sa3-dataset-'));
  t.after(() => fs.promises.rm(root, { recursive: true, force: true }));
  const source = path.join(root, 'originals', 'track.wav');
  await fs.promises.mkdir(path.dirname(source));
  await fs.promises.writeFile(source, 'original');
  await fs.promises.writeFile(path.join(path.dirname(source), 'track.txt'), 'caption: original\nlyrics: words\n');
  const derived = path.join(root, 'instrumental.wav');
  await fs.promises.writeFile(derived, 'instrumental');
  const prepared: Sa3Instrumental = { audioPath: derived, sourceHash: await sa3Hash(source), modelHashes: ['model'],
    fingerprint: 'fingerprint', audioHash: await sa3Hash(derived), seconds_total: 999, cached: true };
  const opts = { samples: [{ audio_path: source, metadata: { caption: 'pad', duration: 999, lyrics: 'words' } }],
    cacheDir: path.join(root, 'cache'), exportDir: path.join(root, 'export'), modelFiles: ['unused-mock'], engineUrl: 'http://mock' };
  return { root, source, opts, prepared, deps: {
    prepare: async () => prepared,
    inspect: async () => ({ duration: 17.5, channels: 2, sampleRate: 44100 }),
  } };
}

test('SA3 derived export uses measured clip timing and plain captions without changing source sidecars', async t => {
  const f = await fixture(t);
  const manifest = await prepareSa3Dataset(f.opts, f.deps);
  assert.equal(manifest.samples[0].seconds_total, 17.5);
  assert.equal(manifest.samples[0].prompt, 'pad');
  assert.equal(await fs.promises.readFile(path.join(f.opts.exportDir, '000001.txt'), 'utf8'), 'pad\n');
  assert.equal(await fs.promises.readFile(path.join(f.opts.exportDir, '000001.wav'), 'utf8'), 'instrumental');
  assert.equal(await fs.promises.readFile(path.join(path.dirname(f.source), 'track.txt'), 'utf8'), 'caption: original\nlyrics: words\n');
  assert.deepEqual(JSON.parse(await fs.promises.readFile(path.join(f.opts.exportDir, 'manifest.json'), 'utf8')), manifest);
  await assert.rejects(prepareSa3Dataset(f.opts, f.deps), /must be new/);
});

test('SA3 duplicate basenames get distinct export pairs and excluded samples never prepare', async t => {
  const f = await fixture(t);
  let calls = 0;
  const samples = [f.opts.samples[0], { ...f.opts.samples[0], excluded: true }, f.opts.samples[0]];
  const manifest = await prepareSa3Dataset({ ...f.opts, samples }, { ...f.deps, prepare: async () => { calls++; return f.prepared; } });
  assert.equal(calls, 2);
  assert.deepEqual(manifest.samples.map(s => s.audio_path), ['000001.wav', '000002.wav']);
});

for (const failure of ['cancel', 'separation', 'corrupt', 'invalid duration']) {
  test(`SA3 ${failure} cannot publish a partial export`, async t => {
    const f = await fixture(t);
    const controller = new AbortController();
    const deps = { ...f.deps, prepare: async () => {
      if (failure === 'cancel') controller.abort();
      if (failure === 'separation') throw new Error('separation failed');
      if (failure === 'corrupt') await fs.promises.writeFile(f.prepared.audioPath, 'corrupt');
      return f.prepared;
    }, inspect: async () => ({ duration: failure === 'invalid duration' ? 0 : 17.5, sampleRate: 44100, channels: 2 }) };
    await assert.rejects(prepareSa3Dataset({ ...f.opts, signal: controller.signal }, deps));
    assert.equal(fs.existsSync(f.opts.exportDir), false);
    assert.ok(!(await fs.promises.readdir(f.root)).some(n => n.startsWith('.sa3-export-')));
  });
}
