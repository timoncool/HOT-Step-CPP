import assert from 'node:assert/strict';
import test from 'node:test';
import { fixedRecipeTrigger } from './fixedRecipeCaption.js';
import {
  buildYue2AitkPrepareArgs, parseYue2AitkModels,
  type ResolvedYue2AitkPrepareOptions,
} from './yue2AitkPrepareRunner.js';

const options: ResolvedYue2AitkPrepareOptions = {
  legacyManifest: 'legacy/yue2_preprocess.json',
  checkpoint: 'models/yue2.safetensors',
  tokenizer: 'models/tokenizer.gguf',
  output: 'prepared/new-run',
  models: { vae: 'models/vae.gguf', semantic: 'models/semantic.gguf', sheetsage: 'models/sheetsage.gguf' },
};

test('AITK preparation CLI keeps stable paths and repeatable model arguments', () => {
  assert.deepEqual(buildYue2AitkPrepareArgs(options), [
    'yue2-prepare-aitk', '--legacy-manifest', 'legacy/yue2_preprocess.json',
    '--checkpoint', 'models/yue2.safetensors', '--tokenizer', 'models/tokenizer.gguf',
    '--output', 'prepared/new-run', '--model', 'vae=models/vae.gguf',
    '--model', 'semantic=models/semantic.gguf', '--model', 'sheetsage=models/sheetsage.gguf',
    '--lyric-timing', '1',
  ]);
});

test('AITK preparation passes the dataset trigger before native tokenization', () => {
  assert.deepEqual(buildYue2AitkPrepareArgs({ ...options, trigger: 'artist_album' }).slice(-4),
    ['--lyric-timing', '1', '--trigger', 'artist_album']);
});

test('Fixed recipe prepares bare caption-dropout twins with the same trigger or slug as its captions', () => {
  for (const trigger of ['sample_style', ' custom_tag ', '', '  ', undefined]) {
    const prefix = fixedRecipeTrigger(trigger, 'sample_slug');
    const args = buildYue2AitkPrepareArgs({ ...options, trigger: prefix, captionGrammar: 'fixed-recipe', lyricTiming: false });
    assert.equal(args.includes('--trigger'), false);
    assert.deepEqual(args.slice(-4), ['--lyric-timing', '0', '--caption-dropout-trigger', prefix]);
  }
});

test('request parser accepts concrete model map and repeatable model fields', () => {
  assert.deepEqual(parseYue2AitkModels({
    vae: 'v', semantic: 's', sheetsage: 'h',
  }, undefined), { vae: 'v', semantic: 's', sheetsage: 'h' });
  assert.deepEqual(parseYue2AitkModels(undefined, [
    'vae=v', 'semantic=s', 'sheetsage=h',
  ]), { vae: 'v', semantic: 's', sheetsage: 'h' });
});

test('request parser rejects missing, duplicate, malformed, and foreign model names', () => {
  assert.equal(parseYue2AitkModels(undefined, ['vae=v', 'semantic=s']), null);
  assert.equal(parseYue2AitkModels(undefined, ['vae=v', 'vae=again', 'semantic=s', 'sheetsage=h']), null);
  assert.equal(parseYue2AitkModels(undefined, ['vae=v', 'semantic=s', 'sheetsage']), null);
  assert.equal(parseYue2AitkModels({ vae: 'v', semantic: 's', sheetsage: 'h', extra: 'x' }, undefined), null);
});
