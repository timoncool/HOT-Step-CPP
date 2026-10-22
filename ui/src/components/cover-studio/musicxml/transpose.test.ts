import assert from 'node:assert/strict';
import { test } from 'node:test';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { DOMParser } from 'linkedom';
import { importMusicXml } from './index';
import { soundingPitches, transposeAbc } from '../../../../../server/src/services/backends/yue2/abcTranspose.js';

Object.assign(globalThis, { DOMParser });
const dir = fileURLToPath(new URL('./fixtures/', import.meta.url));
// A user's score that abcjs strTranspose could not move at any shift; read in
// place when this machine has it, never copied into the repo.
const reported = 'L:/Downloads/in-the-hall-of-the-mountain-king-dovregubbens-hall.mxl';
const STANDARD = /^(Cb|Gb|Db|Ab|Eb|Bb|F|C|G|D|A|E|B|F#|C#)$|^(Ab|Eb|Bb|F|C|G|D|A|E|B|F#|C#|G#|D#|A#)m$/;

const NATURAL: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const pc = (n: string) => NATURAL[n[0]] + (n[1] === '#' ? 1 : n[1] === 'b' ? -1 : 0);
/** Chord symbols in the music lines (header V: names are quoted too). */
const chords = (abc: string) => abc.split('\n').filter(l => !/^[A-Za-z]:/.test(l)).join('\n')
  .match(/"[A-G][#b]?[^"/]*(?:\/[A-G][#b]?)?"/g) ?? [];

async function scores(path: string): Promise<{ name: string; abc: string }[]> {
  const name = path.split(/[\\/]/).pop()!;
  const blob = () => new Blob([new Uint8Array(readFileSync(path))]);
  const first = await importMusicXml(blob(), name);
  if (first.abc) return [{ name, abc: first.abc }];
  return Promise.all(first.parts.map(async p => ({ name: `${name} ${p.id}`, abc: (await importMusicXml(blob(), name, { partId: p.id })).abc })));
}

async function everyShift(path: string) {
  for (const { name, abc } of await scores(path)) {
    const before = soundingPitches(abc);
    const doubles = (abc.match(/\^\^|__/g) ?? []).length;
    for (let s = -12; s <= 12; s++) {
      const out = transposeAbc(abc, s);
      const after = soundingPitches(out);
      before.forEach((track, i) => track.forEach((p, j) => {
        assert.equal(after[i][j] - p, s, `${name} ${s}: track ${i + 1} note ${j + 1}`);
      }));
      for (const [, k] of out.matchAll(/(?:^K:|\[K:)([^\]\r\n]*)/gm)) assert.match(k.trim(), STANDARD, `${name} ${s}: K:${k}`);
      assert.equal((out.match(/\^\^|__/g) ?? []).length, doubles, `${name} ${s}: a double accidental appeared`);
      const from = chords(abc), to = chords(out);
      assert.equal(to.length, from.length, `${name} ${s}: chord symbols`);
      from.forEach((c, n) => {
        const [x, y] = [c, to[n]].map(t => t.match(/^"([A-G][#b]?)([^"/]*)(?:\/([A-G][#b]?))?"$/)!);
        assert.equal(y[2], x[2], `${name} ${s}: ${c} became ${to[n]}`);
        for (const i of [1, 3]) if (x[i]) assert.equal(((pc(y[i]) - pc(x[i]) - s) % 12 + 12) % 12, 0, `${name} ${s}: ${c} became ${to[n]}`);
      });
    }
  }
}

for (const file of readdirSync(dir).filter(f => /\.(xml|mxl|musicxml)$/.test(f))) {
  test(`${file}: every shift from -12 to +12 moves each note exactly`, () => everyShift(dir + file));
}

test('the reported mountain-king score: every shift from -12 to +12', { skip: !existsSync(reported) && 'score not on this machine' },
  () => everyShift(reported));
