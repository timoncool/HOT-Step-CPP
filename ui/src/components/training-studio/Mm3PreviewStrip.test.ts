import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import type { TrainingPreviewRow } from '../../stores/trainingStore';
import { Mm3PreviewList } from './Mm3PreviewStrip';

await i18n.use(initReactI18next).init({ lng: 'en', fallbackLng: 'en', resources: {}, interpolation: { escapeValue: false } });

const row = (overrides: Partial<TrainingPreviewRow> = {}): TrainingPreviewRow => ({
  id: 'take', run: 'run / one', step: 20, totalSteps: 100, kind: 'artist', base: false,
  file: 'takes/a & b.wav', seconds: 40, seed: 424242, caption: 'A held-out caption',
  bytes: 100, ms: 16000, ts: 1, ...overrides,
});
const render = (previews: TrainingPreviewRow[]) => renderToStaticMarkup(createElement(Mm3PreviewList, { previews }));

test('MM3 keeps both kinds, puts base before checkpoints and uses encoded existing audio routes', () => {
  const html = render([
    row(), row({ id: 'base', base: true, step: 0 }),
    row({ id: 'early', step: 10, loss: 0 }), row({ id: 'control', kind: 'control' }),
  ]);
  assert.match(html, /Artist caption[\s\S]*Base[\s\S]*Step 10[\s\S]*Step 20[\s\S]*Control caption/);
  assert.match(html, /Loss 0\.000/);
  assert.match(html, /40 s · seed 424242 · rendered in 16 s/);
  assert.match(html, /A held-out caption/);
  assert.match(html, /src="\/api\/training\/mm3\/preview\?run=run%20%2F%20one&amp;file=takes%2Fa%20%26%20b\.wav"/);
  assert.equal(html.match(/<audio/g)?.length, 4);
  assert.doesNotMatch(html, /<audio[^>]*controls/);
});

test('MM3 keeps repeated steps from different runs separate and renders nothing without previews', () => {
  const html = render([row(), row({ run: 'other run' })]);
  assert.equal(html.match(/Step 20/g)?.length, 2);
  assert.match(html, /run=other%20run/);
  assert.equal(render([]), '');
});
