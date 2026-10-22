import assert from 'node:assert/strict';
import test from 'node:test';
import abcjs from 'abcjs';
import { instrumentalScore, instrumentalStyle } from './instrumentalScore.js';

const pitches = (abc: string) => (abcjs.parseOnly(abc)[0] as any).setUpAudio().tracks
  .flatMap((track: any[]) => track.filter(e => e.cmd === 'note').map(e => e.pitch));

const HEAD = 'X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=82\nV: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
  + 'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:Bb\n';

test('sung bars move to Ins; Vocal keeps its chords on rests; silent bars keep Ins', () => {
  const abc = HEAD + '% intro\nV: Vocal\n"Bb"z32|"Gm"z32|\nV: Ins\nB,4F2B6f8B2c2d2c2B4|G4G2B6f8B2c2d2c2B4|\n'
    + '% verse\nV: Vocal\n"Bb"z4d2d2d4d4d2c2B2B6z4|"G"B8d8"D"f16|Z2|\nV: Ins\nZ4|\n';
  const result = instrumentalScore(abc);
  assert.equal(result.abc, HEAD + '% intro\nV: Vocal\n"Bb"z32|"Gm"z32|\nV: Ins\nB,4F2B6f8B2c2d2c2B4|G4G2B6f8B2c2d2c2B4|\n'
    + '% verse\nV: Vocal\n"Bb"z32|"G"z16"D"z16|Z2|\nV: Ins\nz4d2d2d4d4d2c2B2B6z4|B8d8f16|Z2|\n');
  assert.equal(result.movedBars, 2);
  assert.equal(result.replacedInsBars, 0);
  assert.deepEqual(instrumentalScore(result.abc), { abc: result.abc, movedBars: 0, replacedInsBars: 0 });
});

test('Vocal wins a bar both voices play in, and an unchorded sung bar becomes Z', () => {
  const abc = HEAD + 'V: Vocal\nc8d8e8-e8|\nV: Ins\nB,32|\n';
  const result = instrumentalScore(abc);
  assert.equal(result.abc, HEAD + 'V: Vocal\nZ|\nV: Ins\nc8d8e8-e8|\n');
  assert.equal(result.replacedInsBars, 1);
});

test('Ins keeps the parts of its notes Vocal does not cover', () => {
  // G24 sounds after Vocal's C8; a long Ins note is cut around a Vocal note.
  const a = instrumentalScore(HEAD + 'V: Vocal\nC8z24|z8c8z16|\nV: Ins\nz8G24|F32|\n');
  assert.equal(a.abc, HEAD + 'V: Vocal\nZ2|\nV: Ins\nC8G24|F8c8F16|\n');
  assert.equal(a.replacedInsBars, 1);
  // A tied Ins note whose continuation Vocal covers loses its tie.
  const b = instrumentalScore(HEAD + 'V: Vocal\nZ|d32|\nV: Ins\nz16G16-|G16z16|\n');
  assert.equal(b.abc, HEAD + 'V: Vocal\nZ2|\nV: Ins\nz16G16|d32|\n');
  // Accidentals are respelled for the merged bar: Vocal's C# (spelled Db in Bb, as upstream) leaves Ins's c natural.
  const c = instrumentalScore(HEAD + 'V: Vocal\n^c8z24|\nV: Ins\nz16c16|\n');
  assert.equal(c.abc, HEAD + 'V: Vocal\nZ|\nV: Ins\n_d8z8c16|\n');
});

test('a respelled accidental reads the same per letter (YuE2) and per octave (abcjs)', () => {
  const head = HEAD.replace('K:Bb', 'K:G');
  const abc = head + 'V: Vocal\n=F8=f8z16|\nV: Ins\nZ|\n';
  const result = instrumentalScore(abc);
  assert.equal(result.abc, head + 'V: Vocal\nZ|\nV: Ins\n=F8=f8z16|\n');
  assert.deepEqual(pitches(abc), [65, 77]);
  assert.deepEqual(pitches(result.abc), [65, 77]);
  // abcjs ignores the mark on a tie target, so the next G# in that bar is marked again.
  // (Ab is respelled G# in C, as upstream prefers sharps there.)
  const inC = HEAD.replace('K:Bb', 'K:C');
  const tied = instrumentalScore(inC + 'V: Vocal\n_A32-|_A16A16|\nV: Ins\nZ2|\n');
  assert.equal(tied.abc, inC + 'V: Vocal\nZ2|\nV: Ins\n^G32-|^G16^G16|\n');
  assert.deepEqual(pitches(tied.abc), [68, 68]);
});

test('a group split over several lines keeps Vocal and Ins bar counts equal', () => {
  const result = instrumentalScore(HEAD + 'V: Vocal\nC32|\nC32|\nV: Ins\nZ|\nZ|\n');
  assert.equal(result.abc, HEAD + 'V: Vocal\nZ2|\nV: Ins\nC32|C32|\n');
  assert.equal(result.movedBars, 2);
  assert.deepEqual(instrumentalScore(result.abc), { abc: result.abc, movedBars: 0, replacedInsBars: 0 });
});

test('a melody-only score gains an Ins voice to carry the melody', () => {
  const head = 'X:1\nM:4/4\nL:1/32\nV: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nK:D\n';
  const result = instrumentalScore(head + '% verse\nV: Vocal\nf16B16|\n');
  assert.equal(result.abc, 'X:1\nM:4/4\nL:1/32\nV: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
    + 'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:D\n% verse\nV: Vocal\nZ|\nV: Ins\nf16B16|\n');
});

test('a score with no Vocal voice is left alone', () => {
  const abc = 'X:1\nL:1/4\nK:D\nf2 B c | d4 |\n';
  assert.deepEqual(instrumentalScore(abc), { abc, movedBars: 0, replacedInsBars: 0 });
});

test('instrumental style follows upstream wording and is idempotent', () => {
  const style = instrumentalStyle('gentle solo piano, slow and minimal.');
  assert.equal(style, 'Instrumental, gentle solo piano, slow and minimal, no vocals, no singing, no choir, no spoken words.');
  assert.equal(instrumentalStyle(style), style);
  // No caption stays no caption, so the missing-caption check still fires.
  assert.equal(instrumentalStyle(' . '), '');
});
