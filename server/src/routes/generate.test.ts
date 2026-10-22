import test from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { expectedBackendMismatch, markGenerationJobCancelled, setGenerationJobLedgerForTests } from './generate.js';
import { GenerationJobLedger } from '../services/generation/jobLedger.js';

test('expectedBackendMismatch is null when the caller named no expectation', () => {
  assert.equal(expectedBackendMismatch({}, 'ace'), null);
  assert.equal(expectedBackendMismatch({ backend: 'yue2' }, 'ace'), null); // backend is log-only, not read here
  assert.equal(expectedBackendMismatch(null, 'ace'), null);
  assert.equal(expectedBackendMismatch(undefined, 'ace'), null);
});

test('expectedBackendMismatch is null when the expectation already matches the active backend', () => {
  assert.equal(expectedBackendMismatch({ expectedBackend: 'ace' }, 'ace'), null);
});

test('expectedBackendMismatch reports both ids when the active backend moved on', () => {
  assert.deepEqual(
    expectedBackendMismatch({ expectedBackend: 'yue2' }, 'ace'),
    { expectedBackend: 'yue2', activeBackend: 'ace' },
  );
});

test('expectedBackendMismatch ignores a non-string expectedBackend rather than comparing it', () => {
  for (const bad of [42, true, {}, [], null]) {
    assert.equal(expectedBackendMismatch({ expectedBackend: bad }, 'ace'), null);
  }
});

test('a job cancelled before its lane turn stays cancelled across a restart', () => {
  const db = new Database(':memory:');
  const ledger = new GenerationJobLedger(db);
  setGenerationJobLedgerForTests(ledger);
  try {
    const job = { id: 'waiting', userId: 'u', createdAt: Date.now(), status: 'pending' } as Parameters<typeof markGenerationJobCancelled>[0];
    ledger.opened(job);
    markGenerationJobCancelled(job);
    assert.equal(job.status, 'cancelled');
    // Restart before the lane ever ran the job: a new process opens the same database.
    assert.equal(new GenerationJobLedger(db).get('waiting')!.status, 'cancelled');
  } finally { setGenerationJobLedgerForTests(null); }
});

test('a direct YuE2 submission with a Prepare document is checked, then queued without it', async () => {
  const { withoutPrepareDocument } = await import('./generate.js');
  const { structureHash } = await import('../contracts/yue2Prepare.js');
  const abc = ['X:1', 'K:C', '% intro', 'V: Vocal', 'Z|', '% verse', 'V: Vocal', 'C4|'].join(String.fromCharCode(10));
  const hash = structureHash(abc);
  const doc = { version: 2, sourceKey: 'xml', abc, sectionIds: ['s1', 's2'], paste: '',
    blocks: [{ id: 'b1', text: 'hello', tag: 'Verse' }],
    assignments: { s1: { kind: 'instrumental' }, s2: { kind: 'words', blockIds: ['b1'] } },
    excluded: [], lyricRevision: 1, flagged: [], nextId: 4,
    confirmed: { structure: { structureHash: hash }, lyrics: { structureHash: hash, lyricRevision: 1 } } };
  const lyrics = ['[intro]', '', '[verse]', 'hello'].join(String.fromCharCode(10));
  const params = { backend: 'yue2', yue2Abc: abc, lyrics, instrumental: false, yue2Prepare: doc };
  const ok = withoutPrepareDocument(params);
  assert.ok('submission' in ok);
  assert.equal(ok.submission.yue2Prepare, undefined);
  assert.equal(ok.submission.lyrics, lyrics);
  const error = (p: object) => { const r = withoutPrepareDocument(p); return 'error' in r ? r.error : ''; };
  assert.match(error({ ...params, yue2Prepare: { ...doc, lyricRevision: 2 } }), /Confirm the lyrics/);
  assert.match(error({ ...params, yue2Abc: abc.replace('C4|', 'E4|') }), /score changed/);
  assert.match(error({ ...params, lyrics: '[verse]' }), /not the confirmed Prepare lyrics/);
  assert.match(error({ ...params, instrumental: true }), /turn off Instrumental/);
  // Callers without a document are untouched.
  const plain = { backend: 'yue2', yue2Abc: abc, lyrics: 'anything' };
  assert.deepEqual(withoutPrepareDocument(plain), { submission: plain });
});
