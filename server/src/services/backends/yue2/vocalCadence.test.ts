import assert from 'node:assert/strict';
import test from 'node:test';
import type { FloatWav } from '../../generation/audioLevel.js';
import { hasSoundingVocalBar } from './scoreHealth.js';
import { vocalCadenceAbc } from './vocalCadence.js';

const HEADER = 'X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\nV: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nV: Ins clef=treble name="Ins Melody" snm="Inst."\nK:Em\n';

/** A stem with a 440 Hz burst over each [startSlot, endSlot) of 16ths at 120 BPM. */
function stem(seconds: number, bursts: [number, number][]): FloatWav {
  const rate = 8000;
  const samples = new Float32Array(seconds * rate);
  for (const [from, to] of bursts) {
    for (let i = Math.round(from * 0.125 * rate); i < Math.round(to * 0.125 * rate); i++) samples[i] = 0.5 * Math.sin(2 * Math.PI * 440 * i / rate);
  }
  return { channels: [samples], sampleRate: rate, numChannels: 1, sourceFormat: 's16' };
}

test('a silent Vocal line takes the stem rhythm on the tonic, keeping chords, Ins and rests', () => {
  const abc = `${HEADER}% verse\nV: Vocal\nZ|"Em"z8"C"z8|\nV: Ins\nB16|B16|`;
  assert.equal(hasSoundingVocalBar(abc), false);
  // Bar 2 starts at slot 16: syllables on slots 0-1, 4 and 6-9 of it.
  const out = vocalCadenceAbc(abc, stem(4, [[16, 18], [20, 21], [22, 26]]));
  assert.equal(out, `${HEADER}% verse\nV: Vocal\nz16|"Em"E2z2EzE2-"C"E2z6|\nV: Ins\nB16|B16|`);
  assert.equal(hasSoundingVocalBar(out), true);
});

test('a silent stem leaves only rests, so the empty-score refusal still applies', () => {
  const abc = `${HEADER}% verse\nV: Vocal\nZ2|\nV: Ins\nB16|B16|`;
  const out = vocalCadenceAbc(abc, stem(4, []));
  assert.equal(hasSoundingVocalBar(out), false);
});
