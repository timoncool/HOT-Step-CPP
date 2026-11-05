import assert from 'node:assert/strict';
import test from 'node:test';
import { collapseLoops, hasSectionTags, looksLooped, normaliseLyricTags, transferTags } from './lyricTags.js';

test('Lyric tags: Genius, Flamingo and transcriber dialects all land on the unnumbered vocabulary', () => {
  const input = [
    '[Verse 1: Some Singer]', 'Line one', '', '[Chrous]', 'Hook line', '', 'Chorus:', 'Again',
    '', '(Verse 2)', 'Two', '', '*piano solo*', '', '[Chorus - Layered Male Vocals]', 'Last',
    '', '[Spoken Intro]', 'Hi', '', '[Outro Chorus: Two Names]', 'Bye',
  ].join('\n');
  assert.equal(normaliseLyricTags(input), [
    '[Verse]', 'Line one', '', '[Chorus]', 'Hook line', '', '[Chorus]', 'Again', '',
    '[Verse]', 'Two', '', '[Instrumental]', '', '[Chorus]', 'Last', '', '[Intro]', 'Hi', '', '[Outro]', 'Bye',
  ].join('\n'));
});

test('Lyric tags: speaker tags, cues and junk drop; spoken and sung lines stay', () => {
  const input = [
    '[Verse 1]', 'Richard:', '(Spoken:) Hello there', '[Duet]', 'Sung line', '(rpt 1,2)', '[x3]',
    '(It\'s the lost verse!)', 'A spectacle monopolized:', '[And good night.]', '*Cough*',
  ].join('\n');
  assert.equal(normaliseLyricTags(input), [
    '[Verse]', 'Hello there', 'Sung line', '(It\'s the lost verse!)', 'A spectacle monopolized:', 'And good night.',
  ].join('\n'));
});

test('Lyric tags: a decode loop within a line keeps two copies; real repeats survive', () => {
  assert.equal(
    collapseLoops('Friss mich auf (She is the sun and the moon and the stars and the sun and the moon and the stars and the sun and the moon and the stars and the sun and the moon and the stars)'),
    // The repeating unit is "and the sun ... stars", which starts at the second
    // cycle: the first stays, then two copies.
    'Friss mich auf (She is the sun and the moon and the stars and the sun and the moon and the stars and the sun and the moon and the stars)');
  assert.equal(collapseLoops("She's my honey, she's my honey"), "She's my honey, she's my honey");
  assert.equal(collapseLoops('la la la la la la'), 'la la la la la la');   // under 3-word phrases: left alone
  assert.equal(normaliseLyricTags('[Chorus]\nI want it all, I want it all, I want it all, I want it all'),
    '[Chorus]\nI want it all, I want it all');
});

test('Lyric tags: looksLooped catches repeated lines, looping phrases and a tacked-on parenthetical', () => {
  assert.equal(looksLooped(Array(6).fill('(She is a whore)').join('\n')), true);
  assert.equal(looksLooped(['A (my honey, my honey)', 'B (my honey, my honey)', 'C (my honey, my honey)',
    'D (my honey, my honey)', 'E (my honey, my honey)'].join('\n')), true);
  assert.equal(looksLooped('the sun and the moon, the sun and the moon, the sun and the moon'), true);
  assert.equal(looksLooped('[Chorus]\nDö-dö-dö\nDö-dö-dö\nDö-dö-dö\nDö-dö-dö\n\n[Verse]\nWords'), false);
});

test('Lyric tags: [Hook] is the chorus, unless the track also has a chorus', () => {
  assert.equal(normaliseLyricTags('[Hook]\nA\n\n[Verse]\nB'), '[Chorus]\nA\n\n[Verse]\nB');
  assert.equal(normaliseLyricTags('[Chorus]\nA\n\n[Hook]\nB'), '[Chorus]\nA\n\n[Post-Chorus]\nB');
});

test('Lyric tags: back-to-back and trailing empty tags collapse', () => {
  assert.equal(normaliseLyricTags('[Intro]\n[Synth arpeggio intro]\n[Verse 1]\nWords\n[Outro]'), '[Verse]\nWords');
  assert.equal(hasSectionTags('just words\nmore words'), false);
  assert.equal(hasSectionTags('Chorus:\nwords'), true);
  assert.equal(hasSectionTags('words\n*piano solo*\nmore words\n[Instrumental]'), false);
});

test('Lyric tags: transferTags copies section starts from a tagged transcript onto untagged lyrics', () => {
  const untagged = [
    'When you were here before', 'I could not look you in the eye', '',
    'But I am a creep, I am a weirdo', 'What am I doing here', '',
    'Whatever makes you happy', 'Whatever you want',
  ].join('\n');
  const tagged = [
    '[Verse 1]', 'When you were here before', 'Couldnt look you in the eye',
    '[Chorus]', 'But Im a creep Im a weirdo', 'What am I doing here',
    '[Instrumental]', '[Verse 2]', 'Whatever makes you happy', 'Whatever you want',
  ].join('\n');
  assert.equal(transferTags(untagged, tagged), [
    '[Verse]', 'When you were here before', 'I could not look you in the eye', '',
    '[Chorus]', 'But I am a creep, I am a weirdo', 'What am I doing here', '',
    '[Verse]', 'Whatever makes you happy', 'Whatever you want',
  ].join('\n'));
  assert.equal(transferTags('totally different words here', tagged), 'totally different words here');
});
