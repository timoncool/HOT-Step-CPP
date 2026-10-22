import assert from 'node:assert/strict';
import test from 'node:test';
import { essentiaTempoProfile, reconcileEssentiaBpm } from './essentiaClient.js';

test('embedded BPM resolves a half or double Essentia tempo only', () => {
  assert.equal(reconcileEssentiaBpm(136, 68), 68);
  assert.equal(reconcileEssentiaBpm(68, 136), 136);
  assert.equal(reconcileEssentiaBpm(140, 93), 140);
  assert.equal(reconcileEssentiaBpm(140, null), 140);
  assert.equal(reconcileEssentiaBpm(null, 68), null);
});

test('tempo bounds become extractor profile keys only when set', () => {
  assert.equal(essentiaTempoProfile(), null);
  assert.equal(essentiaTempoProfile({ minTempo: 60, maxTempo: 100 }), 'rhythm:\n  minTempo: 60\n  maxTempo: 100\n');
  assert.equal(essentiaTempoProfile({ maxTempo: 100 }), 'rhythm:\n  maxTempo: 100\n');
  assert.throws(() => essentiaTempoProfile({ minTempo: 120, maxTempo: 100 }), /below maxTempo/);
});
