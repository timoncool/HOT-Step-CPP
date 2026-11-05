import assert from 'node:assert/strict';
import test from 'node:test';
import abcjs from 'abcjs';
import {
  coverScoreChoices, coverTempo, coverTranspose, coverVocalOnly, remeterTwoFourToFourFour, shiftBarLines, transformCoverScore,
} from './coverScoreTransform.js';
import { scoreBarClock } from './scoreClock.js';
import { scoreSections } from './scoreSections.js';

// The two-voice layout, fields, rests and section markers follow a transcribed
// excerpt from _experiments/yue2-cover-qual-2026-10-02/approved.abc.
const score = [
  'X:1', 'M:4/4', 'L:1/16', 'Q:1/4=122',
  'V: Vocal clef=treble name="Vocal Melody"',
  'V: Ins clef=treble name="Ins Melody"',
  'K:Em', '% intro', 'V: Vocal', '"Em/B"E2 F2 [K:G]"G/D"G2 _B2|',
  'V: Ins', 'z12E,2E,2|E,2E,2E2E,2|',
  '% verse', 'V: Vocal', 'B4A2G2A4A2G2|',
  'V: Ins', 'Z4|', '',
].join('\n');

test('vocal-only removes every Ins declaration and body block, preserving the rest', () => {
  const expected = score.replace('V: Ins clef=treble name="Ins Melody"\n', '')
    .replace('V: Ins\nz12E,2E,2|E,2E,2E2E,2|\n', '')
    .replace('V: Ins\nZ4|\n', '');
  assert.equal(coverVocalOnly(score), expected);
  assert.equal(coverVocalOnly(expected), expected);
});

test('tempo free removes Q:; set keeps the beat unit and is idempotent', () => {
  const free = coverTempo(score, 'free');
  assert.equal(free.includes('Q:'), false);
  assert.equal(coverTempo(free, 'free'), free);
  const set = coverTempo(score, 138);
  assert.match(set, /^Q:1\/4=138$/m);
  assert.equal(coverTempo(set, 138), set);
  assert.equal(coverTempo(score, 'source'), score);
});

test('transpose moves header and inline keys, notes, and chord root/bass; +2 and -5 round-trip', () => {
  for (const [target, header, inline, chord] of [
    ['F#m', 'K:F#m', '[K:A]', '"F#m/C#"'],
    ['Bm', 'K:Bm', '[K:D]', '"Bm/F#"'],
  ] as const) {
    const moved = coverTranspose(score, target);
    assert.match(moved, new RegExp(`^${header}$`, 'm'));
    assert.ok(moved.includes(inline));
    assert.ok(moved.includes(chord));
    assert.notEqual(moved, score);
    assert.equal(coverTranspose(moved, target), moved);
    assert.equal(coverTranspose(moved, 'Em'), score);
  }
});

test('transpose preserves K: modifiers and accepts a separate mode word', () => {
  const withClef = score.replace('K:Em', 'K:Em clef=treble');
  const moved = coverTranspose(withClef, 'F#m');
  assert.match(moved, /^K:F#m clef=treble$/m);
  assert.match(moved, /^"F#m\/C#"F2 G2/m);
  assert.equal(coverTranspose(moved, 'Em'), withClef);
  const wordMode = score.replace('K:Em', 'K:E minor clef=treble');
  assert.match(coverTranspose(wordMode, 'F#m'), /^K:F# minor clef=treble$/m);
});

test('transpose keeps the signature abcjs spelled the notes for, so every pitch moves by the interval', () => {
  const pitches = (abc: string) => (abcjs.parseOnly(abc)[0] as any).setUpAudio().tracks
    .flatMap((track: any[]) => track.filter(e => e.cmd === 'note').map(e => e.pitch));
  const inB = 'X:1\nL:1/8\nK:B\n"B"B c d e f g a b|"E"E F G A B c d e|\n';
  for (const [target, header, steps] of [['C#', 'K:Db', 2], ['A#', 'K:Bb', -1]] as const) {
    const moved = coverTranspose(inB, target);
    assert.match(moved, new RegExp(`^${header}$`, 'm'));
    assert.deepEqual(pitches(moved), pitches(inB).map((p: number) => p + steps));
  }
});

test('keyShift carries the exact signed semitone delta past the nearest-key fold', () => {
  // Chord-symbol text has no register of its own: abcjs re-derives a chord
  // track's playback octave from scratch rather than offsetting it by the
  // literal shift, so a chord-bearing score cannot assert exact MIDI deltas.
  // A plain, chord-free score isolates the note transposition this fix touches.
  const pitches = (abc: string) => (abcjs.parseOnly(abc)[0] as any).setUpAudio().tracks
    .flatMap((track: any[]) => track.filter(e => e.cmd === 'note').map(e => e.pitch));
  const chordFree = 'X:1\nL:1/8\nK:B\nB c d e f g a b|E F G A B c d e|\n';
  const base = pitches(chordFree);

  // B +7 and B -5 both name "F#". Without keyShift, coverTranspose always
  // takes the <=6-semitone route (-5), not the requested direction.
  assert.deepEqual(pitches(coverTranspose(chordFree, 'F#')), base.map((p: number) => p - 5));
  // abcjs may spell the result as F# or its enharmonic Gb depending on the
  // literal shift used internally; either is the same requested pitch class.
  const up7 = coverTranspose(chordFree, 'F#', 7);
  assert.match(up7, /^K:(F#|Gb)$/m);
  assert.deepEqual(pitches(up7), base.map((p: number) => p + 7));

  // +12 names the same key as no shift at all ("B"); the fold path is a
  // no-op here, silently dropping a whole-octave request.
  assert.equal(coverTranspose(chordFree, 'B'), chordFree);
  const upOctave = coverTranspose(chordFree, 'B', 12);
  assert.match(upOctave, /^K:B$/m);
  assert.notEqual(upOctave, chordFree);
  assert.deepEqual(pitches(upOctave), base.map((p: number) => p + 12));

  // The same requests downward: B -7 names E, B -12 names B again.
  const down7 = coverTranspose(chordFree, 'E', -7);
  assert.match(down7, /^K:E$/m);
  assert.deepEqual(pitches(down7), base.map((p: number) => p - 7));
  const downOctave = coverTranspose(chordFree, 'B', -12);
  assert.match(downOctave, /^K:B$/m);
  assert.deepEqual(pitches(downOctave), base.map((p: number) => p - 12));

  assert.throws(() => coverTranspose(chordFree, 'F#', 1), /does not land on the requested key/);
  assert.throws(() => coverScoreChoices({ keyShift: 13 }), /keyShift must/);
  assert.throws(() => coverScoreChoices({ keyShift: 1.5 }), /keyShift must/);

  const choices = coverScoreChoices({ key: 'F#', keyShift: 7 });
  assert.deepEqual(pitches(transformCoverScore(chordFree, choices).renderedAbc), base.map((p: number) => p + 7));
});

test('submit-time defaults retain both score voices, then apply chords, tempo, key in order', () => {
  const choices = coverScoreChoices({});
  assert.deepEqual(choices, { voices: 'both', keepChords: false, tempo: 'free', key: 'source', cfgScale: 1 });
  const result = transformCoverScore(score, choices);
  assert.equal(result.fullScore, score);
  assert.equal(result.renderedAbc.includes('V: Ins'), true);
  assert.equal(result.renderedAbc.includes('"Em/B"'), false);
  assert.equal(result.renderedAbc.includes('Q:'), false);
  assert.match(result.renderedAbc, /^K:Em$/m);
  assert.equal(transformCoverScore(result.renderedAbc, choices).renderedAbc, result.renderedAbc);
  const changed = coverScoreChoices({ voices: 'vocal', keepChords: false, tempo: 140, key: 'F#m' });
  const rendered = transformCoverScore(score, changed).renderedAbc;
  assert.equal(transformCoverScore(rendered, changed).renderedAbc, rendered);
  assert.throws(() => coverScoreChoices({ tempo: 301 }), /tempo must/);
  assert.throws(() => coverScoreChoices({ cfgScale: 0 }), /cfgScale must/);
});

test('every combination of voices/chords/tempo/key reaches the wire together, nothing silently dropped', () => {
  // Both voices + chords kept + a non-source tempo + a non-source key, all at
  // once: each selected part must still be present AND correctly transformed
  // by the others, not just "not crash". This is the combination a real
  // cover job sends when nothing is toggled off.
  const keepAll = coverScoreChoices({ voices: 'both', keepChords: true, tempo: 96, key: 'F#m' });
  const keepAllResult = transformCoverScore(score, keepAll).renderedAbc;
  assert.equal(keepAllResult.includes('V: Ins'), true); // voices: both -> Ins voice present
  assert.ok(keepAllResult.includes('"F#m/C#"')); // chords: kept -> transposed chord root/bass, not stripped
  assert.match(keepAllResult, /^Q:1\/4=96$/m); // tempo: set
  assert.match(keepAllResult, /^K:F#m$/m); // key: changed
  assert.match(keepAllResult, /^"F#m\/C#"F2 G2/m); // melody itself moved too, under the new key

  // vocal-only + chords stripped + free tempo + a non-source key: the three
  // "off" choices must each take effect without blocking the key change.
  const dropAll = coverScoreChoices({ voices: 'vocal', keepChords: false, tempo: 'free', key: 'Bm' });
  const dropAllResult = transformCoverScore(score, dropAll).renderedAbc;
  assert.equal(dropAllResult.includes('V: Ins'), false); // voices: vocal -> Ins voice gone
  // chords: stripped -> neither original chord symbol survives, transposed or not
  assert.equal(dropAllResult.includes('Em/B'), false);
  assert.equal(dropAllResult.includes('G/D'), false);
  assert.equal(dropAllResult.includes('Q:'), false); // tempo: free -> no Q: field
  assert.match(dropAllResult, /^K:Bm$/m); // key: still changed despite every other toggle being off
});

test('explicit vocal-only choice remains available and removes the instrumental voice', () => {
  const choices = coverScoreChoices({ voices: 'vocal' });
  assert.equal(choices.voices, 'vocal');
  const result = transformCoverScore(score, choices);
  assert.equal(result.fullScore, score);
  assert.equal(result.renderedAbc.includes('V: Ins'), false);
  assert.equal(result.renderedAbc.includes('V: Vocal'), true);
});

// Two voices in the transcribed layout: a two-bar intro, then a five-bar verse
// whose last bar has no partner.
const twoFour = [
  'X:1', 'M:2/4', 'L:1/16', 'Q:1/4=100',
  'V: Vocal clef=treble name="Vocal Melody"',
  'V: Ins clef=treble name="Ins Melody"',
  'K:G', '% intro', 'V: Vocal', '"G"z8|"D"z8|', 'V: Ins', 'G8|Z|',
  '% verse', 'V: Vocal', '"G"=F4G4|"C"F4[CEG]4-|[CEG]4d4|"D"d8-|d8|', 'V: Ins', 'Z5|', '',
].join('\n');

test('re-meter pairs 2/4 bars in every voice, keeps sections and leaves an odd last bar in 2/4', () => {
  const result = remeterTwoFourToFourFour(twoFour, false);
  assert.equal(result.abc, [
    'X:1', 'M:4/4', 'L:1/16', 'Q:1/4=100',
    'V: Vocal clef=treble name="Vocal Melody"',
    'V: Ins clef=treble name="Ins Melody"',
    // A bar with chord symbols keeps its rests; a full rest pair becomes one Z.
    'K:G', '% intro', 'V: Vocal', '"G"z8"D"z8|', 'V: Ins', 'G8z8|',
    // The removed bar line would carry =F into the second half: it gets ^F back.
    // The tied chord crosses what is now the middle of a bar.
    '% verse', 'V: Vocal', '"G"=F4G4"C"^F4[CEG]4-|[CEG]4d4"D"d8-|', 'V: Ins', 'Z2|',
    // Off the header meter, a rest bar is written out: abcjs times Z by the header.
    'V: Vocal', 'M:2/4', 'd8|', 'V: Ins', 'M:2/4', 'z8|', '',
  ].join('\n'));
  assert.equal(result.twoFourBars, 1);
  assert.deepEqual(scoreSections(twoFour), [{ label: 'intro', startBar: 1 }, { label: 'verse', startBar: 3 }]);
  assert.deepEqual(scoreSections(result.abc), [{ label: 'intro', startBar: 1 }, { label: 'verse', startBar: 2 }]);
  const before = scoreBarClock(twoFour), after = scoreBarClock(result.abc);
  assert.equal(after.length, 4);
  assert.ok(Math.abs(after.at(-1)!.end - before.at(-1)!.end) < 1e-9);
  assert.ok(Math.abs(after[1].start - before[2].start) < 1e-9);
});

test('re-meter with a pickup leaves the first bar in 2/4 and pairs from bar 2', () => {
  const abc = 'X:1\nM:2/4\nL:1/8\nK:C\nc4|d4|e4|f4|g4|';
  assert.equal(remeterTwoFourToFourFour(abc, false).abc, 'X:1\nM:4/4\nL:1/8\nK:C\nc4d4|e4f4|\nM:2/4\ng4|');
  const pickup = remeterTwoFourToFourFour(abc, true);
  assert.equal(pickup.abc, 'X:1\nM:4/4\nL:1/8\nK:C\nM:2/4\nc4|\nM:4/4\nd4e4|f4g4|');
  assert.equal(pickup.twoFourBars, 1);
});

test('re-meter writes the default length out, keeps CRLF and marks only notes the carry would change', () => {
  // Under 2/4 the default L: is 1/16; under 4/4 it would be 1/8 and double every note.
  assert.equal(remeterTwoFourToFourFour('X:1\r\nM:2/4\r\nK:C\r\nc8|d8|\r\n', false).abc,
    'X:1\r\nM:4/4\r\nL:1/16\r\nK:C\r\nc8d8|\r\n');
  // A tie continuation keeps its pitch and carries nothing; the next c is natural again.
  assert.match(remeterTwoFourToFourFour('X:1\nM:2/4\nL:1/16\nK:C\nz4^c4-|c4c4|', false).abc, /^z4\^c4-c4=c4\|$/m);
  // YuE2 carries an accidental across octaves, so f gets its natural written in.
  assert.match(remeterTwoFourToFourFour('X:1\nM:2/4\nL:1/16\nK:C\n^F4z4|f8|', false).abc, /^\^F4z4=f8\|$/m);
});

test('re-meter turns only full bars of rests into Z and keeps short ones as written', () => {
  // Two one-beat rest bars are not a 4/4 bar's worth of silence.
  assert.equal(remeterTwoFourToFourFour('X:1\nM:2/4\nL:1/8\nK:C\nc4|d4|z2|z2|', false).abc,
    'X:1\nM:4/4\nL:1/8\nK:C\nc4d4|z2z2|');
  // Rests that do add up to a bar each, fractions included, still become one Z.
  assert.equal(remeterTwoFourToFourFour('X:1\nM:2/4\nL:1/8\nK:C\nc4|d4|z2z/z/z|z4|', false).abc,
    'X:1\nM:4/4\nL:1/8\nK:C\nc4d4|Z|');
});

test('re-meter moves a key change before a second half inside the bar', () => {
  assert.equal(remeterTwoFourToFourFour('X:1\nM:2/4\nL:1/16\nK:C\nc8|\nK:G\nf8|', false).abc,
    'X:1\nM:4/4\nL:1/16\nK:C\nc8[K:G]f8|');
  assert.equal(remeterTwoFourToFourFour('X:1\nM:2/4\nL:1/16\nK:C\nc8|d8|\nK:G\nf8|g8|', false).abc,
    'X:1\nM:4/4\nL:1/16\nK:C\nc8d8|\nK:G\nf8g8|');
});

test('re-meter rejects what it cannot convert exactly, and returns nothing to apply', () => {
  for (const [abc, pickup, message] of [
    ['X:1\nM:4/4\nL:1/16\nK:C\nc16|d16|', false, /no 2\/4 bars/],
    ['X:1\nM:2/4\nL:1/16\nK:C\n|:c8|d8:|', false, /Repeats/],
    [twoFour.replace('Z5|', 'Z4|'), false, /voices disagree/],
    ['X:1\nM:2/4\nL:1/16\nK:C\nc8|\n\nd8|', false, /blank line/],
    ['X:1\nM:2/4\nL:1/16\nK:C\nc8|d8|\nw: la la', false, /w: line/],
    ['X:1\nM:2/4\nL:1/16\nK:C\nc8|d8', false, /end with a bar line/],
    ['X:1\nM:3/4\nL:1/16\nK:C\nc12|\nM:2/4\nc8|d8|', true, /pickup/],
    ['X:1\nM:2/4\nL:1/16\nK:C\nc8|\nM:3/4\nc12|', false, /paired/],
  ] as const) {
    assert.throws(() => remeterTwoFourToFourFour(abc, pickup), message);
  }
});

test('bar-line shift moves every bar line a beat, ties the notes it cuts and writes the odd bars as beats', () => {
  const abc = 'X:1\nM:4/4\nL:1/8\nK:C\n"C"c2d2e2f2|"G"g8|\n';
  assert.equal(shiftBarLines(abc, 1).abc,
    'X:1\nM:4/4\nL:1/8\nK:C\nM:1/4\n"C"c2|\nM:4/4\nd2e2f2"G"g2-|\nM:3/4\ng6|\n');
  assert.equal(shiftBarLines(abc, -1).abc,
    'X:1\nM:4/4\nL:1/8\nK:C\nM:3/4\n"C"c2d2e2|\nM:4/4\nf2"G"g6-|\nM:1/4\ng2|\n');
  // The carried sharp is written out in its new bar; the tie continuation needs no mark.
  assert.equal(shiftBarLines('X:1\nM:4/4\nL:1/8\nK:C\n^F2F2G2A2|F8|', 1).abc,
    'X:1\nM:4/4\nL:1/8\nK:C\nM:1/4\n^F2|\nM:4/4\n^F2G2A2=F2-|\nM:3/4\nF6|');
});

test('bar-line shift moves both voices and every section but the first with the bar lines', () => {
  const abc = [
    'X:1', 'M:4/4', 'L:1/8', 'V: Vocal', 'V: Ins', 'K:G', '% intro', 'V: Vocal', 'Z2|', 'V: Ins', 'G8|z6d2|',
    '% verse', 'V: Vocal', '"G"B2A2G2A2|"D"A8|', 'V: Ins', 'Z2|', '',
  ].join('\n');
  const later = shiftBarLines(abc, 1).abc;
  assert.equal(later, [
    'X:1', 'M:4/4', 'L:1/8', 'V: Vocal', 'V: Ins', 'K:G', '% intro',
    'V: Vocal', 'M:1/4', 'z2|', 'V: Ins', 'M:1/4', 'G2-|', 'V: Vocal', 'M:4/4', 'Z|z6"G"B2|', 'V: Ins', 'M:4/4', 'G6z2|z4d2z2|',
    '% verse', 'V: Vocal', 'A2G2A2"D"A2-|', 'V: Ins', 'Z|', 'V: Vocal', 'M:3/4', 'A6|', 'V: Ins', 'M:3/4', 'z6|', '',
  ].join('\n'));
  assert.deepEqual(scoreSections(later).map(s => s.startBar), [1, 4]);
  // Back again plays the same notes (the shift checks that itself) with the sections where they were.
  assert.deepEqual(scoreSections(shiftBarLines(later, -1).abc), scoreSections(abc));
});

test('bar-line shift refuses a zero or whole-bar shift and notation it cannot cut', () => {
  for (const [abc, beats, message] of [
    ['X:1\nM:4/4\nL:1/8\nK:C\nc8|', 0, /whole number/],
    ['X:1\nM:4/4\nL:1/8\nK:C\nc8|', 4, /whole number/],
    ['X:1\nM:4/4\nL:1/8\nK:C\n(3ccc c6|', 1, /tuplets/],
    ['X:1\nM:4/4\nL:1/8\nK:C\nc8|]', 1, /double bars/],
  ] as const) {
    assert.throws(() => shiftBarLines(abc, beats), message);
  }
});
