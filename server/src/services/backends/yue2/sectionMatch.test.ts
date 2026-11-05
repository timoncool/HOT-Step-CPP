import assert from 'node:assert/strict';
import test from 'node:test';
import { matchSectionsToScore, planScoreFromLyrics, suggestionLyrics, suggestSections } from './sectionMatch.js';
import type { Yue2AlignWord } from './align.js';

// Two 3-second bars per section: intro 0, verse 6, chorus 12, verse 18,
// chorus 24, outro 30. Intro and outro have no Vocal note.
const score = [
  'X:1', 'M:3/4', 'L:1/8', 'Q:1/8=120', 'K:C',
  '% intro', 'V: Vocal', 'z6|z6|',
  '% verse', 'V: Vocal', 'C6|D6|',
  '% chorus', 'V: Vocal', 'E6|F6|',
  '% verse', 'V: Vocal', 'C6|D6|',
  '% chorus', 'V: Vocal', 'E6|F6|',
  '% outro', 'V: Vocal', 'z6|z6|', '',
].join('\n');

function words(lyrics: string, times: Array<[string, number, number?]>): Yue2AlignWord[] {
  return times.map(([text, start, score = 0.9]) => {
    const char0 = Array.from(lyrics.slice(0, lyrics.indexOf(text))).length;
    return { char0, char1: char0 + Array.from(text).length, start, end: start + 0.4, score };
  });
}

test('retags by sung time with the score labels as written, fills a missing chorus and instrumental tags', () => {
  const lyrics = '[Intro]\n\n[Verse 1]\nfirst line\n\n[Hook]\nsecond line\n\n[verse]\nthird line';
  // "second" lands half a bar before the chorus: a pickup into it.
  const result = matchSectionsToScore(score, lyrics, words(lyrics, [['first', 6.2], ['second', 11.5], ['third', 18.3]]));
  assert.equal(result.lyrics, '[intro]\n\n[verse]\nfirst line\n\n[chorus]\nsecond line\n\n[verse]\nthird line\n\n[chorus]\nsecond line\n\n[outro]');
  assert.deepEqual(result.blocks.map(b => b.status), ['dropped', 'renamed', 'renamed', 'kept']);
  assert.deepEqual(result.filled.map(f => [f.section, f.copiedFrom]), [[1, null], [5, 3], [6, null]]);
  assert.equal(result.unchanged, false);
});

test('a block with no aligned words keeps its tag and place, and blocks copies into its gap', () => {
  const lyrics = '[verse]\nfirst line\n[hook]\nsecond line\n[verse]\nthird line';
  const result = matchSectionsToScore(score, lyrics, words(lyrics, [['first', 6.2], ['second', 12.1]]));
  assert.equal(result.lyrics, '[intro]\n\n[verse]\nfirst line\n\n[chorus]\nsecond line\n\n[verse]\nthird line\n\n[outro]');
  assert.equal(result.blocks[2].status, 'unsure');
  assert.equal(result.blocks[2].section, null);
});

test('a block after an unsure one is never merged under the unsure tag', () => {
  const lyrics = '[Verse]\nfirst line\n\n[Bridge]\nlost line\n\n[Verse 2]\nthird line';
  const result = matchSectionsToScore(score, lyrics, words(lyrics, [['first', 6.1], ['third', 7.5]]));
  assert.equal(result.lyrics.includes('[Bridge]\nlost line\n\n[Verse 2]\nthird line'), true);
  assert.deepEqual(result.blocks.map(b => b.status), ['renamed', 'unsure', 'unsure']);
});

test('two blocks sung in one section share one tag; words are untouched', () => {
  const lyrics = 'first line\n\n[Pre]\nsecond line';
  const result = matchSectionsToScore(score, lyrics, words(lyrics, [['first', 6.1], ['second', 7.5]]));
  assert.equal(result.lyrics.startsWith('[intro]\n\n[verse]\nfirst line\n\nsecond line'), true);
  assert.deepEqual(result.blocks.map(b => b.status), ['renamed', 'merged']);
});

test('only chorus-type sections get copies, and a block sung where the score has no Vocal stays unsure', () => {
  // A verse the lyrics leave out stays empty-handed; the chorus is copied.
  const twoBlocks = '[Verse 1]\nfirst line\n\n[Chorus]\nsecond line';
  const copied = matchSectionsToScore(score, twoBlocks, words(twoBlocks, [['first', 6.2], ['second', 12.1]]));
  assert.equal(copied.lyrics, '[intro]\n\n[verse]\nfirst line\n\n[chorus]\nsecond line\n\n[chorus]\nsecond line\n\n[outro]');
  assert.deepEqual(copied.filled.map(f => [f.section, f.copiedFrom]), [[1, null], [5, 2], [6, null]]);

  // "third" starts in the outro, which has no Vocal note: tag and place kept,
  // flagged, and the outro still gets its own empty tag before it.
  const lyrics = '[Verse 1]\nfirst line\n\n[Chorus]\nsecond line\n\n[Verse 2]\nthird line';
  const result = matchSectionsToScore(score, lyrics, words(lyrics, [['first', 6.2], ['second', 12.1], ['third', 30.2]]));
  assert.deepEqual(result.blocks.map(b => b.status), ['renamed', 'renamed', 'unsure']);
  assert.equal(result.blocks[2].newTag, 'Verse 2');
  assert.equal(result.lyrics.endsWith('[outro]\n\n[Verse 2]\nthird line'), true);
  assert.equal(result.filled.some(f => f.copiedFrom === 1), false);
});

test('a note held over into a section\'s first bar does not make it sung', () => {
  // The outro opens on the last chorus's held note; it still gets its tag.
  const held = score.replace('% outro\nV: Vocal\nz6|z6|', '% outro\nV: Vocal\nF6|z6|');
  const lyrics = '[verse]\nfirst line\n\n[chorus]\nsecond line\n\n[verse]\nthird line\n\n[chorus]\nfourth line';
  const result = matchSectionsToScore(held, lyrics,
    words(lyrics, [['first', 6.2], ['second', 12.1], ['third', 18.3], ['fourth', 24.2]]));
  assert.notEqual(held, score);
  assert.equal(result.lyrics, `[intro]\n\n${lyrics}\n\n[outro]`);
});

test('a score whose clock runs late still gives each block its own section', () => {
  // SheetSage's one tempo drifts: every block after the first is sung more
  // than a pickup bar before the score says its section starts.
  const lyrics = '[verse]\nfirst line\n\n[chorus]\nsecond line\n\n[verse]\nthird line\n\n[chorus]\nfourth line';
  const result = matchSectionsToScore(score, lyrics,
    words(lyrics, [['first', 5.5], ['second', 8.5], ['third', 14.5], ['fourth', 20.5]]));
  assert.equal(result.lyrics, `[intro]\n\n${lyrics}\n\n[outro]`);
  assert.deepEqual(result.blocks.map(b => b.status), ['kept', 'kept', 'kept', 'kept']);
});

test('a block tagged like a section with no Vocal note is placed there, not merged before it', () => {
  // Backing ad-libs over the outro: SheetSage hears no Vocal there.
  const lyrics = '[verse]\nfirst line\n\n[chorus]\nsecond line\n\n[Outro]\nna na na';
  const result = matchSectionsToScore(score, lyrics,
    words(lyrics, [['first', 6.2], ['second', 12.1], ['na na', 29.5]]));
  // The chorus the lyrics wrote once is still copied into the gap.
  assert.equal(result.lyrics, '[intro]\n\n[verse]\nfirst line\n\n[chorus]\nsecond line\n\n[chorus]\nsecond line\n\n[outro]\nna na na');
  assert.deepEqual(result.blocks.map(b => b.status), ['kept', 'kept', 'renamed']);
});

test('score to lyrics: blocks open their own sections where the score has too few', () => {
  // One long verse (bars 3-8) holds a verse, a chorus and a verse.
  const coarse = [
    'X:1', 'M:3/4', 'L:1/8', 'Q:1/8=120', 'K:C',
    '% intro', 'V: Vocal', 'z6|z6|',
    '% verse', 'V: Vocal', 'C6|D6|E6|F6|C6|D6|',
    '% chorus', 'V: Vocal', 'E6|F6|',
    '% outro', 'V: Vocal', 'z6|z6|', '',
  ].join('\n');
  const lyrics = '[Verse 1]\nfirst line\n\n[Chorus]\nhook line\n\n[Verse 2]\nthird line\n\n[Chorus]\nhook again';
  const plan = planScoreFromLyrics(coarse, lyrics, words(lyrics,
    [['first', 6.2], ['hook line', 12.1], ['third', 18.2], ['hook again', 24.1]]));
  assert.deepEqual(plan.sections.map(s => `${s.label}@${s.bar}`),
    ['intro@1', 'verse@3', 'chorus@5', 'verse@7', 'chorus@9', 'outro@11']);
  assert.deepEqual(plan.sections.map(s => s.tag), [null, 'Verse 1', 'Chorus', 'Verse 2', 'Chorus', null]);
  assert.equal(plan.lyrics, '[intro]\n\n[verse]\nfirst line\n\n[chorus]\nhook line\n\n[verse]\nthird line\n\n[chorus]\nhook again\n\n[outro]');
  assert.deepEqual(plan.unsure, []);
});

test('score to lyrics: an instrumental outro stays when the last word is stretched over it', () => {
  const lyrics = '[Verse]\nfirst line\n\n[Chorus]\nhook line';
  const timed = words(lyrics, [['first', 6.2], ['hook line', 12.1]]);
  timed[1].end = 35;  // the aligner drags the last word past the outro's start (30 s)
  const plan = planScoreFromLyrics(score, lyrics, timed);
  assert.deepEqual(plan.sections.map(s => `${s.label}@${s.bar}`).at(-1), 'outro@11');
});

test('score to lyrics: a block with no timing joins the section before it', () => {
  const lyrics = '[Verse]\nfirst line\n\n[Chorus]\nlost line';
  const plan = planScoreFromLyrics(score, lyrics, words(lyrics, [['first', 6.2]]));
  assert.deepEqual(plan.unsure, [{ block: 2, tag: 'Chorus' }]);
  assert.match(plan.lyrics, /\[verse\]\nfirst line\n\nlost line/);
});

test('score to lyrics: a kept marker before a same-label sung block folds into it', () => {
  // The intro is marked at bar 1 but its block is sung from bar 5.
  const raw = [
    'X:1', 'M:3/4', 'L:1/8', 'Q:1/8=120', 'K:C',
    '% intro', 'V: Vocal', 'z6|z6|z6|z6|C6|D6|',
    '% chorus', 'V: Vocal', 'E6|F6|', '',
  ].join('\n');
  const lyrics = '[Intro]\nla la\n\n[Chorus]\nhook line';
  const plan = planScoreFromLyrics(raw, lyrics, words(lyrics, [['la la', 12.1], ['hook line', 18.1]]));
  assert.deepEqual(plan.sections.map(s => `${s.label}@${s.bar}`), ['intro@1', 'chorus@7']);
  assert.equal(plan.sections[0].block, 1);
});

test('score to lyrics: a double chorus stays two sections', () => {
  const lyrics = '[Verse]\nfirst line\n\n[Chorus]\nhook one\n\n[Chorus]\nhook two';
  const plan = planScoreFromLyrics(score, lyrics, words(lyrics, [['first', 6.2], ['hook one', 12.1], ['hook two', 24.1]]));
  assert.deepEqual(plan.sections.map(s => `${s.label}@${s.bar}`).filter(s => s.startsWith('chorus')), ['chorus@5', 'chorus@9']);
});

test('score to lyrics: a kept marker with another label is untouched', () => {
  const lyrics = '[Verse]\nfirst line\n\n[Chorus]\nhook line';
  const plan = planScoreFromLyrics(score, lyrics, words(lyrics, [['first', 6.2], ['hook line', 12.1]]));
  assert.equal(plan.sections[0].label, 'intro');
  assert.equal(plan.sections[0].block, null);
});

const ids = ['s0', 's1', 's2', 's3', 's4', 's5'];
const b = (id: string, tag: string | null, text = `${id} words`) => ({ id, tag, text });

test('instant suggestions go by tag kind and order, offer one pasted chorus as a reuse, and leave the score alone', () => {
  const result = suggestSections(score, ids, [b('b1', 'Verse 1'), b('b2', 'Chorus'), b('b3', 'Verse 2')]);
  assert.deepEqual(result.assignments, { s0: 'instrumental', s1: ['b1'], s2: ['b2'], s3: ['b3'], s5: 'instrumental' });
  assert.deepEqual(result.reuse, [{ sectionId: 's4', blockId: 'b2' }]);
  assert.deepEqual([result.uncertain, result.unplaced, result.recording], [[], [], false]);
});

test('a repeat present in the paste stays its own block; nothing is reused for it', () => {
  const result = suggestSections(score, ids, [b('b1', 'verse'), b('b2', 'chorus'), b('b3', 'verse'), b('b4', 'hook')]);
  assert.deepEqual(result.assignments.s4, ['b4']);
  assert.deepEqual(result.reuse, []);
});

test('untagged words are a guess, and a block with no section of its kind is left unplaced, not forced', () => {
  const result = suggestSections(score, ids, [b('b1', null), b('b2', 'bridge')]);
  assert.deepEqual(result.assignments.s1, ['b1']);
  assert.deepEqual(result.uncertain, ['s1']);
  assert.deepEqual(result.unplaced, ['b2']);
  assert.equal('s2' in result.assignments, false);
});

test('an empty tag keeps its section as instrumental instead of vanishing', () => {
  const result = suggestSections(score, ids, [b('b1', 'verse', ''), b('b2', 'verse'), b('b3', 'chorus')]);
  assert.equal(result.assignments.s1, 'instrumental');
  assert.deepEqual(result.assignments.s3, ['b2']);
  assert.deepEqual(result.assignments.s4, ['b3']);
});

test('with the recording, sung time beats the pasted tag and two blocks in one section are marked uncertain', () => {
  const blocks = [b('b1', 'chorus'), b('b2', 'verse'), b('b3', 'verse')];
  const lyrics = suggestionLyrics(blocks);
  // b1 is sung in the first verse, b2 right after it, b3 in the second.
  const result = suggestSections(score, ids, blocks, words(lyrics, [['b1 words', 6.2], ['b2 words', 7], ['b3 words', 18.3]]));
  assert.equal(result.recording, true);
  assert.deepEqual(result.assignments.s1, ['b1', 'b2']);
  assert.deepEqual(result.assignments.s3, ['b3']);
  assert.deepEqual(result.uncertain, ['s1']);
});

test('with the recording, an empty tag still places by its tag', () => {
  const blocks = [b('b1', 'intro', ''), b('b2', 'verse'), b('b3', 'chorus')];
  const lyrics = suggestionLyrics(blocks);
  const result = suggestSections(score, ids, blocks, words(lyrics, [['b2 words', 6.2], ['b3 words', 12.1]]));
  assert.equal(result.recording, true);
  assert.equal(result.assignments.s0, 'instrumental');
  assert.deepEqual(result.assignments.s1, ['b2']);
});

test('no reuse while an unplaced chorus could belong to the section', () => {
  const blocks = [b('b1', 'verse'), b('b2', 'chorus'), b('b3', 'chorus')];
  const lyrics = suggestionLyrics(blocks);
  // b2 is sung in the first chorus; b3 has no timing, so it is unplaced.
  const result = suggestSections(score, ids, blocks, words(lyrics, [['b1 words', 6.2], ['b2 words', 12.1]]));
  assert.deepEqual(result.unplaced, ['b3']);
  assert.deepEqual(result.reuse, []);
});

test('untagged words skip a silent section whatever its label', () => {
  const silentVerse = score.replace('% verse\nV: Vocal\nC6|D6|', '% verse\nV: Vocal\nz6|z6|');
  const result = suggestSections(silentVerse, ids, [b('b1', null)]);
  assert.equal(result.assignments.s1, 'instrumental');
  assert.deepEqual(result.assignments.s2, ['b1']);
});

test('with the recording, an empty tag takes the next free section of its kind, not one a worded block holds', () => {
  const twoVerses = score.replace('% chorus', '% verse');
  const blocks = [b('b1', 'verse'), b('b2', 'verse', '')];
  const result = suggestSections(twoVerses, ids, blocks, words(suggestionLyrics(blocks), [['b1 words', 6.2]]));
  assert.equal(result.recording, true);
  assert.deepEqual(result.assignments.s1, ['b1']);
  assert.equal(result.assignments.s2, 'instrumental');
});
