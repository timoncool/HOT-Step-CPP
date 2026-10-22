import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { createCoverStemCache, isCompleteWav } from './coverStemCache.js';
import { NoVocalsError } from './training/yue2Stems.js';

/** A real 16-bit mono PCM WAV with `frames` frames; `chunks` can override the layout. */
function wav(frames = 8, opts: { fmt?: boolean; dataSize?: number; riffSize?: number; tag?: number; bits?: number; align?: number; rate?: number } = {}): Buffer {
  const fmt = Buffer.alloc(24);
  fmt.write('fmt ', 0, 'ascii'); fmt.writeUInt32LE(16, 4); fmt.writeUInt16LE(opts.tag ?? 1, 8); fmt.writeUInt16LE(1, 10);
  fmt.writeUInt32LE(opts.rate ?? 44100, 12); fmt.writeUInt32LE(88200, 16); fmt.writeUInt16LE(opts.align ?? 2, 20); fmt.writeUInt16LE(opts.bits ?? 16, 22);
  const pcm = Buffer.alloc(frames * (opts.align ?? 2), 1);
  const data = Buffer.concat([Buffer.from('data'), Buffer.alloc(4), pcm]);
  data.writeUInt32LE(opts.dataSize ?? pcm.length, 4);
  const body = Buffer.concat([Buffer.from('WAVE'), opts.fmt === false ? Buffer.alloc(0) : fmt, data]);
  const head = Buffer.alloc(8);
  head.write('RIFF', 0, 'ascii'); head.writeUInt32LE(opts.riffSize ?? body.length, 4);
  return Buffer.concat([head, body]);
}

const MALFORMED: Array<[string, Buffer]> = [
  ['RIFF header only', (() => { const b = Buffer.alloc(64); b.write('RIFF', 0); b.writeUInt32LE(56, 4); b.write('WAVE', 8); return b; })()],
  ['no fmt chunk', wav(8, { fmt: false })],
  ['unsupported format tag', wav(8, { tag: 2 })],
  ['empty data', wav(0)],
  ['data past end of file', wav(8, { dataSize: 4000 })],
  ['partial frame', wav(8, { dataSize: 15 })],
  ['truncated RIFF size', wav(8, { riffSize: 9999 })],
  ['PCM with 7 bits', wav(8, { bits: 7 })],
  ['zero sample rate', wav(8, { rate: 0 })],
  ['block align disagrees with bits', wav(8, { align: 4 })],
  ['chunks outside a tiny RIFF', wav(8, { riffSize: 4 })],
  ['extensible with a short fmt', wav(8, { tag: 0xfffe })],
];

function setup(mode: 'ok' | 'none' | 'bad' = 'ok', bad: Buffer = Buffer.from('junk')) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'stemcache-'));
  const root = path.join(dir, 'cache');
  let calls = 0;
  const separate = async (_src: string, out: string) => {
    calls++;
    await new Promise(r => setTimeout(r, 10));
    if (mode === 'none') throw new NoVocalsError('none');
    fs.writeFileSync(out, mode === 'bad' ? bad : wav());
  };
  const audio = (name: string, body: string) => { const p = path.join(dir, name); fs.writeFileSync(p, body); return p; };
  const make = (level = 4, version?: number) => createCoverStemCache({ root, level, separate, version });
  return { dir, root, audio, make, calls: () => calls };
}

test('identical bytes under a new path, a new mtime and a new cache instance are hits', async () => {
  const t = setup();
  const first = await t.make().vocalStem(t.audio('a.wav', 'same bytes'));
  const b = t.audio('b.wav', 'same bytes');
  fs.utimesSync(b, new Date(2000, 0, 1), new Date(2000, 0, 1));
  assert.equal(await t.make().vocalStem(b), first);  // new instance = restart
  assert.equal(t.calls(), 1);
});

test('changed bytes, level or version are misses; nothing is evicted', async () => {
  const t = setup();
  const a = t.audio('a.wav', 'one');
  await t.make().vocalStem(a);
  await t.make().vocalStem(t.audio('b.wav', 'two'));
  await t.make(0).vocalStem(a);
  await t.make(4, 2).vocalStem(a);
  assert.equal(t.calls(), 4);
  assert.equal(fs.readdirSync(t.root).filter(f => f.endsWith('.wav')).length, 4);
});

test('concurrent misses share one separation and leave no part files', async () => {
  const t = setup();
  const cache = t.make();
  const a = t.audio('a.wav', 'x');
  const paths = await Promise.all([cache.vocalStem(a), cache.vocalStem(t.audio('b.wav', 'x'))]);
  assert.equal(paths[0], paths[1]);
  assert.equal(t.calls(), 1);
  assert.deepEqual(fs.readdirSync(t.root).filter(f => f.endsWith('.part')), []);
});

test('a corrupt entry is rejected and rebuilt', async () => {
  const t = setup();
  const a = t.audio('a.wav', 'x');
  const out = await t.make().vocalStem(a);
  fs.writeFileSync(out, 'truncated');
  assert.equal(await t.make().vocalStem(a), out);
  assert.equal(t.calls(), 2);
  assert.ok(fs.readFileSync(out).equals(wav()));
});

test('a separator that writes a bad stem publishes nothing', async () => {
  const t = setup('bad');
  await assert.rejects(t.make().vocalStem(t.audio('a.wav', 'x')));
  assert.deepEqual(fs.readdirSync(t.root), []);
});

test('a no-vocals result is cached and replays as NoVocalsError', async () => {
  const t = setup('none');
  const a = t.audio('a.wav', 'x');
  await assert.rejects(t.make().vocalStem(a), NoVocalsError);
  await assert.rejects(t.make().vocalStem(t.audio('b.wav', 'x')), NoVocalsError);
  assert.equal(t.calls(), 1);
});

test('malformed WAVs are rejected by the validator and the real one passes', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'stemwav-'));
  const f = path.join(dir, 'x.wav');
  fs.writeFileSync(f, wav());
  assert.equal(isCompleteWav(f), true);
  fs.writeFileSync(f, wav(8, { bits: 24, align: 3 }));
  assert.equal(isCompleteWav(f), true);
  for (const [name, body] of MALFORMED) {
    fs.writeFileSync(f, body);
    assert.equal(isCompleteWav(f), false, name);
  }
});

test('a malformed stem is never published and a malformed cached stem is rebuilt', async () => {
  for (const [name, body] of MALFORMED) {
    const t = setup('bad', body);
    const a = t.audio('a.wav', 'x');
    await assert.rejects(t.make().vocalStem(a));
    assert.deepEqual(fs.readdirSync(t.root), [], name);
    // hit path: plant the same malformed file where the entry would live
    const good = setup();
    const out = await good.make().vocalStem(good.audio('a.wav', 'x'));
    fs.writeFileSync(out, body);
    await good.make().vocalStem(good.audio('b.wav', 'x'));
    assert.equal(good.calls(), 2, name);
    assert.ok(fs.readFileSync(out).equals(wav()), name);
  }
});
