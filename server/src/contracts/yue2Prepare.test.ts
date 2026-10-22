// The Prepare document contract (contracts/yue2Prepare.ts): hash, shape,
// completeness and the shared submission check.
import test from 'node:test';
import assert from 'node:assert/strict';
import { preparedPair, prepareSectionLabels, prepareV2Problem, structureHash, type PrepareV2 } from './yue2Prepare.js';
import { scoreSections } from '../services/backends/yue2/scoreSections.js';

const ABC = 'X:1\nM:4/4\nK:C\n% intro\nV: Vocal\nZ|\nV: Ins\nZ|\n% verse\nV: Vocal\nC4|\nV: Ins\nZ|\n% chorus\nV: Vocal\nD4|\nV: Ins\nZ|';

function confirmedPrepare(over: Partial<PrepareV2> = {}): PrepareV2 {
  const p: PrepareV2 = {
    version: 2, sourceKey: 'src', abc: ABC, sectionIds: ['s1', 's2', 's3'], paste: '[Verse]\nhello\n\n[Chorus]\nla la',
    blocks: [{ id: 'b1', text: 'hello', tag: 'Verse' }, { id: 'b2', text: 'la la', tag: 'Chorus' }, { id: 'b3', text: 'spoken', tag: null }],
    assignments: { s1: { kind: 'instrumental' }, s2: { kind: 'words', blockIds: ['b1'] }, s3: { kind: 'words', blockIds: ['b2'] } },
    excluded: ['b3'], lyricRevision: 4, flagged: [], nextId: 7,
    confirmed: { structure: null, lyrics: null }, ...over,
  };
  const hash = structureHash(p.abc);
  return { ...p, confirmed: over.confirmed ?? { structure: { structureHash: hash }, lyrics: { structureHash: hash, lyricRevision: p.lyricRevision } } };
}

test('structureHash ignores line endings and outer whitespace, not content', () => {
  assert.equal(structureHash(ABC), structureHash(`\n${ABC.replace(/\n/g, '\r\n')}\n  `));
  assert.notEqual(structureHash(ABC), structureHash(ABC.replace('% chorus', '% bridge')));
  assert.notEqual(structureHash(ABC), structureHash(ABC.replace('C4|', 'C2C2|')));
  assert.match(structureHash(ABC), /^v1:[0-9a-f]{16}$/);
});

test('section labels count as the server counts them', () => {
  for (const abc of [ABC, `${ABC}\n%%MIDI program 1\n%  outro \nV: Vocal\nZ|`, '% above\nX:1\nK:C\nC|'])
    assert.deepEqual(prepareSectionLabels(abc), scoreSections(abc).map(s => s.label));
});

test('a complete, confirmed document exports tagged lyrics in score order', () => {
  const pair = preparedPair(confirmedPrepare(), `${ABC}\n`);
  assert.ok(!('error' in pair));
  assert.equal(pair.lyrics, '[intro]\n\n[verse]\nhello\n\n[chorus]\nla la');
  assert.equal(pair.abc, ABC);
  assert.equal(pair.lyricRevision, 4);
  // Several blocks fill one section in their order; labels come from the score, never the paste.
  const two = confirmedPrepare({ assignments: { s1: { kind: 'instrumental' }, s2: { kind: 'words', blockIds: ['b2', 'b1'] }, s3: { kind: 'instrumental' } } });
  const out = preparedPair(two);
  assert.ok(!('error' in out));
  assert.equal(out.lyrics, '[intro]\n\n[verse]\nla la\nhello\n\n[chorus]');
});

test('submission refuses stale or incomplete documents', () => {
  const p = confirmedPrepare();
  const err = (v: unknown, abc?: string) => { const r = preparedPair(v, abc); return 'error' in r ? r.error : ''; };
  assert.match(err({ ...p, lyricRevision: 5 }), /Confirm the lyrics/);
  assert.match(err({ ...p, confirmed: { ...p.confirmed, structure: null } }), /structure/);
  assert.match(err(p, ABC.replace('C4|', 'E4|')), /score changed/);
  // A structure edit after confirming lyrics: the old lyric confirmation no longer matches.
  const edited = ABC.replace('% chorus', '% bridge');
  assert.match(err({ ...p, abc: edited, confirmed: { structure: { structureHash: structureHash(edited) }, lyrics: p.confirmed.lyrics } }), /Confirm the lyrics/);
  const { s3: _, ...partial } = p.assignments;
  assert.match(err({ ...p, assignments: partial }), /chorus \(section 3\) needs words/);
  assert.match(err({ ...p, excluded: [] }), /Assign or exclude every lyric block/);
  assert.match(err({ ...p, excluded: ['b3', 'b1'] }), /both assigned and excluded/);
  assert.match(err({ ...p, assignments: { ...p.assignments, s3: { kind: 'words', blockIds: ['b1'] } }, excluded: ['b3', 'b2'] }), /assigned twice/);
  assert.match(err({ ...p, blocks: p.blocks.map(b => b.id === 'b1' ? { ...b, text: 'hello\n[Bridge]' } : b) }), /looks like a section tag/);
});

test('shape check catches what a draft must never hold', () => {
  const p = confirmedPrepare();
  assert.equal(prepareV2Problem(p), null);
  assert.match(prepareV2Problem({ ...p, version: 1 })!, /version/);
  assert.match(prepareV2Problem({ ...p, sectionIds: ['s1', 's2'] })!, /do not match/);
  assert.match(prepareV2Problem({ ...p, sectionIds: ['s1', 's1', 's2'] })!, /share an id/);
  assert.match(prepareV2Problem({ ...p, assignments: { zz: { kind: 'instrumental' } } })!, /does not have/);
  assert.match(prepareV2Problem({ ...p, assignments: { s1: { kind: 'words', blockIds: [] } } })!, /malformed/);
  assert.match(prepareV2Problem({ ...p, excluded: ['nope'] })!, /excluded block/);
  assert.match(prepareV2Problem({ ...p, lyricRevision: -1 })!, /whole numbers/);
  assert.match(prepareV2Problem({ ...p, confirmed: { structure: 'x', lyrics: null } })!, /structure confirmation/);
});
