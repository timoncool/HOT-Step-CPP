import assert from 'node:assert/strict';
import test from 'node:test';
import { cleanFixedRecipeLyrics, fixedRecipeTrigger, parseFixedRecipeCaption, prepareFixedRecipeCaption } from './fixedRecipeCaption.js';
import { fixedRecipeLegacyManifest } from './yue2AitkPrepareRunner.js';

test('Fixed-recipe lyric cleaning keeps known sections and words, removes cues, and keeps unknown headings in Title case', () => {
  const input = '(quiet mood)\n[EXTENDED OUTRO]\n(wind)\nA word (whispered) stays\n[Unknown]\nHum\n[VERSE 2]\nNext line';
  assert.deepEqual(cleanFixedRecipeLyrics(input), {
    lyrics: '[Outro]\nA word stays\n[Unknown]\nHum\n[Verse]\nNext line', hints: ['quiet mood'],
  });
  assert.equal(cleanFixedRecipeLyrics('[breakdown]\nHey\n[pre-chorus]\nGo\n[half time]\nYo').lyrics,
    '[Breakdown]\nHey\n[Pre-Chorus]\nGo\n[Half Time]\nYo');
});

test('Fixed-recipe caption parser separates style and lyrics and normalizes section case', () => {
  assert.deepEqual(parseFixedRecipeCaption('A style\n[Lyrics]\n[pre-chorus]\nWords\n[Duration]\n120'),
    { style: 'A style', lyrics: '[Pre-Chorus]\nWords' });
  assert.deepEqual(prepareFixedRecipeCaption('A  style', '[VERSE]\nWords'),
    { style: 'A style', lyrics: '[Verse]\nWords' });
});

test('Fixed-recipe preparation cleans the legacy caption without changing cached audio paths', () => {
  const source = { latents: 'latents/track.f32', codec_ids: 'codes/track.i32',
    caption: 'A  style\n[Lyrics]\n[CHORUS]\nHey (echo)', lyrics: '' };
  const result = JSON.parse(fixedRecipeLegacyManifest(JSON.stringify({ caption_format: 'plain', sources: [source] })));
  assert.equal(result.caption_format, 'ace-sidecar');
  assert.deepEqual(result.sources[0], { ...source, caption: 'A style', lyrics: '[Chorus]\nHey' });
});

test('Fixed-recipe preparation prefixes the bare trigger once before native tokenization', () => {
  for (const caption of ['A style', 'sample_style, A style', 'sample_style', 'sample_style_extra music']) {
    const result = JSON.parse(fixedRecipeLegacyManifest(JSON.stringify({ sources: [{ caption }] }), 'sample_style'));
    const style = result.sources[0].caption;
    assert.equal(style, caption.startsWith('sample_style') && caption !== 'sample_style_extra music'
      ? caption : `sample_style, ${caption}`);
    assert.equal(style.includes('in the style of'), false);
    if (caption !== 'sample_style_extra music') assert.equal(style.match(/sample_style/g)?.length, 1);
  }
});

test('Fixed recipe uses the dataset slug when no trigger is set', () => {
  for (const trigger of [undefined, '', '  ']) {
    const prefix = fixedRecipeTrigger(trigger, 'sample_slug');
    assert.equal(prepareFixedRecipeCaption('A style', '', prefix).style, 'sample_slug, A style');
    assert.equal(prepareFixedRecipeCaption('', '', prefix).style, 'sample_slug');
  }
  assert.equal(fixedRecipeTrigger(' custom_tag ', 'sample_slug'), 'custom_tag');
});
