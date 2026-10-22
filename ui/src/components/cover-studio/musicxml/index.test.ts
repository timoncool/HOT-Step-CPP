import assert from 'node:assert/strict';
import { test } from 'node:test';
import { DOMParser } from 'linkedom';
import abcjs from 'abcjs';
import { zipSync, strToU8 } from 'fflate';
import { importMusicXml, MusicXmlImportError } from './index';

Object.assign(globalThis, { DOMParser });
const header = '<score-partwise version="4.0"><work><work-title>Test song</work-title></work><identification><creator type="composer">Composer</creator></identification><part-list><score-part id="P1"><part-name>Voice</part-name></score-part>';
const notes = '<attributes><divisions>4</divisions><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>' +
  '<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><lyric number="1"><syllabic>begin</syllabic><text>Hel</text></lyric></note>' +
  '<note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><lyric number="1"><syllabic>end</syllabic><text>lo</text></lyric></note>' +
  '<note><rest/><duration>8</duration></note>';
const melody = `${header}</part-list><part id="P1"><measure number="1">${notes}</measure></part></score-partwise>`;
const blob = (s: string) => new Blob([s]);

test('melody, rests and first lyric verse produce a parseable native YuE2 score', async () => {
  const result = await importMusicXml(blob(melody), 'song.musicxml');
  assert.equal(result.partId, 'P1');
  assert.equal(result.title, 'Test song');
  assert.equal(result.abc, 'X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=120\nV: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
    + 'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n% verse\nV: Vocal\nC8D8z16|\nV: Ins\nZ|\n');
  assert.equal(result.lyrics, '[Verse]\nHello\n');
  assert.equal(abcjs.parseOnly(result.abc)[0].warnings?.length ?? 0, 0);
});

test('an absent lyric verse never blocks score import', async () => {
  const noLyrics = melody.replace(/<lyric number="1">.*?<\/lyric>/g, '');
  const result = await importMusicXml(blob(noLyrics), 'song.musicxml', { partId: 'P1', verse: '' });
  assert.ok(result.abc);
  assert.equal(result.lyrics, '');
  const afterVerseTwo = await importMusicXml(blob(noLyrics), 'song.musicxml', { partId: 'P1', verse: '2' });
  assert.ok(afterVerseTwo.abc);
  assert.equal(afterVerseTwo.lyrics, '');
  const afterNoLyrics = await importMusicXml(blob(melody), 'song.musicxml', { partId: 'P1', verse: '2' });
  assert.equal(afterNoLyrics.lyrics, '[Verse]\nHello\n');
});

test('accidentals read the same per octave or per letter, and long notes split with ties', async () => {
  const note = (step: string, octave: number, alter: number, duration: number) =>
    `<note><pitch><step>${step}</step>${alter ? `<alter>${alter}</alter>` : ''}<octave>${octave}</octave></pitch><duration>${duration}</duration></note>`;
  const score = `${header}</part-list><part id="P1"><measure number="1" implicit="yes"><attributes><divisions>4</divisions><key><fifths>2</fifths><mode>minor</mode></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>`
    + `${note('F', 4, 0, 4)}</measure><measure number="2">${note('F', 5, 1, 4)}${note('C', 4, 1, 5)}${note('C', 4, 1, 7)}</measure></part></score-partwise>`;
  const result = await importMusicXml(blob(score), 'keys.musicxml');
  assert.match(result.abc, /^K:Bm$/m);
  assert.match(result.abc, /\nz24=F8\|f8C8-C2C12-C2\|\n/);
  const tune = abcjs.parseOnly(result.abc)[0] as unknown as { setUpAudio(): { tracks: { cmd: string; pitch: number }[][] } };
  const pitches = tune.setUpAudio().tracks[0].filter(e => e.cmd === 'note').map(e => e.pitch);
  assert.deepEqual(pitches, [65, 78, 61, 61]);
});

test('two parts require a part choice', async () => {
  const score = `${header}<score-part id="P2"><part-name>Piano</part-name></score-part></part-list><part id="P1"><measure number="1">${notes}</measure></part><part id="P2"><measure number="1">${notes}</measure></part></score-partwise>`;
  const result = await importMusicXml(blob(score), 'song.xml');
  assert.equal(result.abc, '');
  assert.equal(result.partId, null);
  assert.deepEqual(result.parts.map(p => p.id), ['P1', 'P2']);
  assert.equal((await importMusicXml(blob(score), 'song.xml', { partId: 'P2' })).partId, 'P2');
});

test('grand staff defaults to top staff and allows switching', async () => {
  const score = `${header}</part-list><part id="P1"><measure number="1"><attributes><divisions>4</divisions><staves>2</staves></attributes><note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><staff>1</staff></note><backup><duration>4</duration></backup><note><pitch><step>C</step><octave>3</octave></pitch><duration>4</duration><staff>2</staff></note></measure></part></score-partwise>`;
  const result = await importMusicXml(blob(score), 'piano.musicxml');
  assert.equal(result.parts[0].staves, 2);
  assert.match(result.abc, /\n"C"C8z24\|\n/);
  assert.equal(result.chords, 'inferred');
  const lower = await importMusicXml(blob(score), 'piano.musicxml', { staff: 2 });
  assert.match(lower.abc, /C,8z24\|/);
  assert.match(result.abc, /V: Ins\nC,8z24\|/);
});

const harm = (step: string, kind: string, extra = '') => `<harmony><root><root-step>${step}</root-step></root><kind>${kind}</kind>${extra}</harmony>`;
const n = (step: string, duration: number, extra = '') => `<note><pitch><step>${step}</step><octave>4</octave></pitch><duration>${duration}</duration>${extra}</note>`;
const attrs = (fifths: number) => `<attributes><divisions>4</divisions><key><fifths>${fifths}</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>`;

test('chord symbols land where SheetSage2 writes them, spelled the same way', async () => {
  const score = `${header}</part-list><part id="P1">`
    // A change inside a held note splits it with a tie; the chord still sounding is repeated at the next bar line.
    + `<measure number="1">${attrs(-2)}${harm('G', 'minor-seventh')}${n('D', 8)}${harm('C', 'suspended-fourth', '<degree><degree-value>7</degree-value><degree-alter>0</degree-alter><degree-type>add</degree-type></degree>')}<offset>-4</offset>${n('C', 8)}</measure>`
    // A melody rest bar keeps its chord rather than becoming Z; an <offset> moves the change; a slash bass is kept.
    + `<measure number="2"><note><rest/><duration>16</duration></note></measure>`
    + `<measure number="3">${harm('A', 'major', '<bass><bass-step>C</bass-step><bass-alter>1</bass-alter></bass>')}${n('A', 16)}</measure>`
    // A ninth is simplified to its seventh; N.C. ends the chord.
    + `<measure number="4">${harm('F', 'dominant-ninth')}${n('C', 8)}${harm('C', 'none')}${n('F', 8)}</measure>`
    + `<measure number="5">${n('F', 16)}</measure></part></score-partwise>`;
  const result = await importMusicXml(blob(score.replace('</harmony><offset>-4</offset>', '<offset>-4</offset></harmony>')), 'chords.musicxml');
  assert.equal(result.chords, 'score');
  assert.match(result.abc, /\nV: Vocal\n"Gm7"D8-"C7sus4"D8C16\|"C7sus4"z32\|"A\/C#"A32\|"F7"C16F16\|\nV: Ins\nZ4\|\nV: Vocal\nF32\|\n/);
  assert.ok(result.warnings.some(w => w.includes("outside YuE2's chord set")));
  assert.equal(abcjs.parseOnly(result.abc)[0].warnings?.length ?? 0, 0);
});

test('chord symbols in other parts are used, one part per measure, the melody part first', async () => {
  const list = '<score-part id="P2"><part-name>Guitar</part-name></score-part><score-part id="P3"><part-name>Bass</part-name></score-part></part-list>';
  const rests = (count: number) => Array.from({ length: count }, (_, i) => `<measure number="${i + 1}"><note><rest/><duration>16</duration></note></measure>`);
  const withHarmony = (bars: string[], at: Record<number, string>) => bars.map((bar, i) => at[i] ? bar.replace('<note>', `${at[i]}<note>`) : bar).join('');
  const melody = [`<measure number="1">${attrs(0)}${n('C', 16)}</measure>`, `<measure number="2">${n('D', 16)}</measure>`, `<measure number="3">${n('E', 16)}</measure>`];
  const score = `${header}${list}`
    + `<part id="P1">${withHarmony(melody, { 1: harm('C', 'major') })}</part>`
    // P2 has G7 in bar 1 and F in bar 2, where the melody part's own C wins.
    + `<part id="P2">${withHarmony(rests(3), { 0: harm('G', 'dominant'), 1: harm('F', 'major') })}</part>`
    // P3 conflicts with P2 in bar 1 (P2 comes first) and alone supplies bar 3.
    + `<part id="P3">${withHarmony(rests(3), { 0: harm('D', 'minor'), 2: harm('A', 'minor') })}</part></score-partwise>`;
  const result = await importMusicXml(blob(score), 'band.musicxml', { partId: 'P1' });
  assert.equal(result.chords, 'score');
  assert.match(result.abc, /\nV: Vocal\n"G7"C32\|"C"D32\|"Am"E32\|\n/);
  assert.ok(!result.warnings.some(w => w.startsWith('Chords are inferred')));
});

test('roots are respelled under the key, as correct_chord_spelling does', async () => {
  // An A# major chord in Bb is written Bb; a Gb minor chord in Eb stays Gbm (its tones sit nearer the key).
  const score = `${header}</part-list><part id="P1"><measure number="1">${attrs(-2)}`
    + `<harmony><root><root-step>A</root-step><root-alter>1</root-alter></root><kind>major</kind></harmony>${n('D', 16)}</measure></part></score-partwise>`;
  assert.match((await importMusicXml(blob(score), 'spell.musicxml')).abc, /"Bb"D32\|/);
  const inEb = score.replace('<fifths>-2</fifths>', '<fifths>-3</fifths>').replace('<root-step>A</root-step><root-alter>1</root-alter></root><kind>major', '<root-step>F</root-step><root-alter>1</root-alter></root><kind>minor');
  assert.match((await importMusicXml(blob(inEb), 'spell.musicxml')).abc, /"Gbm"D32\|/);
});

test('without chord symbols, chords are inferred per half bar from the other staff and flagged', async () => {
  // F-A-C under a held E reads as Fmaj7: the melody counts at half weight.
  const low = (step: string, octave = 3, chord = false) => `<note>${chord ? '<chord/>' : ''}<pitch><step>${step}</step><octave>${octave}</octave></pitch><duration>8</duration><staff>2</staff></note>`;
  const score = `${header}</part-list><part id="P1"><measure number="1"><attributes><divisions>4</divisions><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time><staves>2</staves></attributes>`
    + `${n('E', 16, '<staff>1</staff>')}<backup><duration>16</duration></backup>${low('C')}${low('E', 3, true)}${low('G', 3, true)}${low('F', 2)}${low('A', 3, true)}${low('C', 4, true)}</measure>`
    + `<measure number="2">${n('E', 16, '<staff>1</staff>')}<backup><duration>16</duration></backup>${low('C')}${low('E', 3, true)}${low('G', 3, true)}${low('C')}${low('E', 3, true)}${low('G', 3, true)}</measure></part></score-partwise>`;
  const result = await importMusicXml(blob(score), 'piano.musicxml');
  assert.equal(result.chords, 'inferred');
  assert.match(result.abc, /\n"C"E16-"Fmaj7"E16\|"C"E32\|\n/);
  assert.ok(result.warnings.some(w => w.startsWith('Chords are inferred')));
});

test('a tie is kept only into the next melody note at the same pitch', async () => {
  const tie = '<tie type="start"/>';
  const score = `${header}</part-list><part id="P1"><measure number="1">${attrs(0)}${n('C', 4, tie)}${n('D', 4)}${n('E', 4, tie)}<note><rest/><duration>4</duration></note></measure>`
    + `<measure number="2">${n('F', 16, tie)}</measure><measure number="3">${n('F', 16)}</measure></part></score-partwise>`;
  assert.match((await importMusicXml(blob(score), 'ties.musicxml')).abc, /\nC8D8E8z8\|F32-\|F32\|\n/);
});

test('MXL follows container rootfile and rejects a missing rootfile', async () => {
  const container = '<container><rootfiles><rootfile full-path="scores/main.musicxml"/></rootfiles></container>';
  const good = zipSync({ 'META-INF/container.xml': strToU8(container), 'scores/main.musicxml': strToU8(melody) });
  assert.equal((await importMusicXml(new Blob([new Uint8Array(good)]), 'song.mxl')).partId, 'P1');
  const bad = zipSync({ 'META-INF/container.xml': strToU8(container) });
  await assert.rejects(importMusicXml(new Blob([new Uint8Array(bad)]), 'bad.mxl'), MusicXmlImportError);
});

test('distinct notes prove first and second endings play in order', async () => {
  const bar = (number: number, step: string, before = '', after = '') =>
    `<measure number="${number}">${before}${number === 1 ? attrs(0) : ''}${n(step, 16)}${after}</measure>`;
  const score = `${header}</part-list><part id="P1">`
    + bar(1, 'C', '<barline location="left"><repeat direction="forward"/></barline>')
    + bar(2, 'D', '<barline location="left"><ending number="1" type="start"/></barline>',
      '<barline location="right"><ending number="1" type="stop"/><repeat direction="backward"/></barline>')
    + bar(3, 'E', '<barline location="left"><ending number="2" type="start"/></barline>',
      '<barline location="right"><ending number="2" type="discontinue"/></barline>')
    + bar(4, 'F') + '</part></score-partwise>';
  const result = await importMusicXml(blob(score), 'repeat.musicxml');
  const vocal = [...result.abc.matchAll(/V: Vocal\n([^\n]*)/g)].map(m => m[1]).join('');
  assert.equal(vocal, 'C32|D32|C32|E32|F32|');
  assert.equal(abcjs.parseOnly(result.abc)[0].warnings?.length ?? 0, 0);
});

test('D.S. al Fine returns to segno and stops on the second pass', async () => {
  const direction = (body: string, sound: string) => `<direction><direction-type>${body}</direction-type><sound ${sound}/></direction>`;
  const score = `${header}</part-list><part id="P1"><measure number="1">${attrs(0)}${n('C', 16)}</measure>`
    + `<measure number="2">${direction('<segno/>', 'segno="s1"')}${n('D', 16)}</measure>`
    + `<measure number="3">${direction('<words>Fine</words>', 'fine="yes"')}${n('E', 16)}</measure>`
    + `<measure number="4">${direction('<words>D.S. al Fine</words>', 'dalsegno="s1"')}${n('F', 16)}</measure></part></score-partwise>`;
  const result = await importMusicXml(blob(score), 'segno.musicxml');
  const vocal = [...result.abc.matchAll(/V: Vocal\n([^\n]*)/g)].map(m => m[1]).join('');
  assert.equal(vocal, 'C32|D32|E32|F32|D32|E32|');
  assert.equal(abcjs.parseOnly(result.abc)[0].warnings?.length ?? 0, 0);
});

test('tuplets snap to the 32nd grid with a warning', async () => {
  const score = `${header}</part-list><part id="P1"><measure number="1"><attributes><divisions>3</divisions></attributes>${n('C', 1)}${n('D', 1)}${n('E', 1)}</measure></part></score-partwise>`;
  const result = await importMusicXml(blob(score), 'triplet.musicxml');
  assert.ok(result.warnings.some(w => w.includes('rounded')));
  assert.equal(abcjs.parseOnly(result.abc)[0].warnings?.length ?? 0, 0);
});

test('long scores warn at an estimated played bar', async () => {
  const bars = Array.from({ length: 600 }, (_, i) => `<measure number="${i + 1}">${i === 0 ? attrs(0) : ''}${n('C', 16)}</measure>`).join('');
  const score = `${header}</part-list><part id="P1">${bars}</part></score-partwise>`;
  const result = await importMusicXml(blob(score), 'long.musicxml');
  assert.ok(result.warnings.some(w => /5,120-token limit.*played bar \d+/.test(w)));
});
