import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { DOMParser } from 'linkedom';
import abcjs from 'abcjs';
import { importMusicXml } from './index';

Object.assign(globalThis, { DOMParser });
const fixture = (name: string) => {
  const path = fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url));
  return new Blob([new Uint8Array(readFileSync(path))]);
};
const parseable = (abc: string) => {
  const tune = abcjs.parseOnly(abc)[0];
  assert.ok(tune);
  assert.deepEqual(tune.warnings ?? [], []);
};
const vocal = (abc: string) => [...abc.matchAll(/V: Vocal\n([^\n]*)/g)].map(m => m[1]).join('');

test('MuseScore MXL follows its container and offers voice and piano parts', async () => {
  const name = 'openscore_nun_seh_ich_wohl.mxl';
  const first = await importMusicXml(fixture(name), name);
  assert.equal(first.abc, '');
  assert.equal(first.parts.length, 2);
  assert.ok(first.parts[0].hasLyrics);
  const chosen = await importMusicXml(fixture(name), name, { partId: first.parts[0].id });
  assert.ok(chosen.lyrics.startsWith('[Verse]\n'));
  assert.ok(chosen.warnings.some(w => w.includes('Multiple voices')));
  parseable(chosen.abc);
});

test('Sibelius choir excerpt offers five parts and a lyric-bearing part parses', async () => {
  const name = 'kyrie_excerpt.xml';
  const first = await importMusicXml(fixture(name), name);
  assert.equal(first.abc, '');
  assert.equal(first.parts.length, 5);
  const chosen = await importMusicXml(fixture(name), name, { partId: 'P2' });
  assert.ok(chosen.lyrics.startsWith('[Verse]\n'));
  parseable(chosen.abc);
});

test('Finale tablature offers part choice and applies transposition', async () => {
  const name = 'w3c_tutorial-tablature.musicxml';
  const first = await importMusicXml(fixture(name), name);
  assert.equal(first.abc, '');
  assert.equal(first.parts.length, 2);
  const chosen = await importMusicXml(fixture(name), name, { partId: 'P1' });
  assert.ok(chosen.abc.includes('V: Vocal'));
  parseable(chosen.abc);
});

test('Finale chord symbols become native chords, extensions simplified', async () => {
  const name = 'w3c_tutorial-chord-symbols.musicxml';
  const chosen = await importMusicXml(fixture(name), name);
  assert.equal(chosen.parts.length, 1);
  assert.equal(chosen.chords, 'score');
  assert.deepEqual([...chosen.abc.matchAll(/"([^"]+)"/g)].map(m => m[1]).filter(c => !/Melody|Vocal|Inst/.test(c)), ['G6/D', 'G6/D', 'A', 'A7']);
  assert.ok(chosen.warnings.some(w => w.includes("outside YuE2's chord set")));
  parseable(chosen.abc);
});

test('repeat endings are played in order', async () => {
  const name = 'testsuite_45b-RepeatWithAlternatives.musicxml';
  const result = await importMusicXml(fixture(name), name);
  assert.equal((vocal(result.abc).match(/c32\|/g) ?? []).length, 5);
  parseable(result.abc);
});

test('mid-piece key and meter changes survive as inline fields', async () => {
  for (const [name, field] of [
    ['testsuite_13e-KeySignatures-MidMeasure-Change.musicxml', '[K:'],
    ['testsuite_11a-TimeSignatures.musicxml', '[M:'],
  ]) {
    const result = await importMusicXml(fixture(name), name);
    assert.ok(result.abc.includes(field), name);
    parseable(result.abc);
  }
});

test('later tempo is ignored with a warning', async () => {
  const name = 'testsuite_31c-MetronomeMarks.musicxml';
  const result = await importMusicXml(fixture(name), name);
  assert.match(result.abc, /^Q:1\/4=100$/m);
  assert.ok(result.warnings.some(w => w.includes('tempo')));
  parseable(result.abc);
});

test('verse selection changes the imported words', async () => {
  const name = 'testsuite_61b-MultipleLyrics.musicxml';
  const first = await importMusicXml(fixture(name), name);
  assert.deepEqual(first.parts[0].verses, ['1', '2', 'iij']);
  const second = await importMusicXml(fixture(name), name, { verse: '2' });
  assert.notEqual(first.lyrics, second.lyrics);
  parseable(second.abc);
});

test('a verse from another score falls back on lyric and instrumental parts', async () => {
  const multi = 'testsuite_61b-MultipleLyrics.musicxml';
  const instrumental = 'testsuite_72a-TransposingInstruments.musicxml';
  const first = await importMusicXml(fixture(instrumental), instrumental);
  const noLyrics = await importMusicXml(fixture(instrumental), instrumental, { partId: first.parts[0].id, verse: '2' });
  assert.equal(noLyrics.lyrics, '');
  assert.ok(noLyrics.abc);
  const fallback = await importMusicXml(fixture(multi), multi, { verse: '' });
  const verseOne = await importMusicXml(fixture(multi), multi, { verse: '1' });
  assert.equal(fallback.lyrics, verseOne.lyrics);
});

test('Bb instrument is converted to sounding pitch', async () => {
  const name = 'testsuite_72a-TransposingInstruments.musicxml';
  const first = await importMusicXml(fixture(name), name);
  const result = await importMusicXml(fixture(name), name, { partId: first.parts[0].id });
  assert.match(result.abc, /^K:C$/m);
  assert.match(vocal(result.abc).replace(/"[^"]+"/g, ''), /^C8D8E8F8\|/);
  parseable(result.abc);
});

test('timewise score is converted', async () => {
  const name = 'w3c_score-timewise-element.musicxml';
  const result = await importMusicXml(fixture(name), name);
  assert.ok(result.abc);
  parseable(result.abc);
});

test('D.C. al Coda resolves to played order', async () => {
  const name = 'synthetic_dc_al_coda.musicxml';
  const result = await importMusicXml(fixture(name), name);
  assert.equal(vocal(result.abc), 'C32|D32|E32|C32|D32|F32|G32|');
  parseable(result.abc);
});

test('lead sheet keeps chords and selected lyric verse', async () => {
  const name = 'synthetic_lead_sheet.musicxml';
  const result = await importMusicXml(fixture(name), name, { verse: '2' });
  assert.equal(result.chords, 'score');
  assert.match(result.abc, /"C"C16"G7"D16\|/);
  assert.match(result.lyrics, /Second verse/);
  parseable(result.abc);
});
