import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

test('prompt lyrics use Ollama and save to an empty album; bad output and cancellation leave no partial library rows', async () => {
  const root = fs.mkdtempSync(path.resolve('.test-scratch-lyrics-'));
  process.env.DATA_DIR = root;
  const database = await import('../../db/database.js');
  const db = await import('../../db/lireekDb.js');
  const { scratchLyricsInput, writeScratchLyrics, runScratchLyrics } = await import('./scratchLyrics.js');
  const { WorkflowJobs } = await import('../workflows/workflowJobs.js');
  const originalFetch = globalThis.fetch;
  let response = JSON.stringify({ title: 'Home again', lyrics: '[Verse]\nThe road brought me home\n\n[Chorus]\nI left the porch light on', caption: 'Gentle acoustic folk', bpm: 90, key: 'C major' });
  let calls = 0;
  globalThis.fetch = (async (url, options) => {
    assert.match(String(url), /\/api\/chat$/);
    const payload = JSON.parse(String(options?.body));
    assert.equal(payload.model, 'fixture-local');
    assert.equal(payload.stream, false);
    assert.equal(payload.think, false);
    const prompt = payload.messages[1].content;
    for (const field of ['coming home', 'folk', 'Verse, Chorus', 'porch light']) assert.ok(prompt.includes(field));
    calls++;
    return new Response(JSON.stringify({ message: { content: `<think>private reasoning</think>\n\x60\x60\x60json\n${response}\n\x60\x60\x60` } }), { status: 200 });
  }) as typeof fetch;
  try {
    database.initDb();
    const artist = db.getOrCreateArtist('Fixture writer');
    const album = db.saveLyricsSet(artist.id as number, 'Originals', 0, []);
    const input = scratchLyricsInput.parse({ lyricsSetId: album.id, model: 'fixture-local', theme: 'coming home', style: 'folk', structure: 'Verse, Chorus', existingLyrics: 'porch light' });
    assert.equal(input.provider, 'ollama');
    assert.equal(scratchLyricsInput.safeParse({ lyricsSetId: album.id, theme: '   ' }).success, false);
    assert.equal(scratchLyricsInput.safeParse({ lyricsSetId: album.id, existingLyrics: 'A first verse' }).success, true);
    const jobs = new WorkflowJobs({ db: database.getDb(), audio: {
      enqueue: () => { throw new Error('unexpected audio'); },
      get: () => { throw new Error('unexpected audio'); },
      cancel: () => { throw new Error('unexpected audio'); },
    } });
    jobs.register({ kind: 'scratch-test', input: scratchLyricsInput, run: runScratchLyrics });
    const submitted = jobs.submit({ kind: 'scratch-test', idempotencyKey: 'one', input }, 'user');
    assert.equal(jobs.submit({ kind: 'scratch-test', idempotencyKey: 'one', input }, 'user').created, false);
    await jobs.settled();
    assert.equal(jobs.get(submitted.job.id).status, 'succeeded');
    assert.equal(calls, 1);
    const saved = db.getGenerations(undefined, album.id as number);
    assert.equal(saved.length, 1);
    assert.equal(saved[0].title, 'Home again');
    assert.equal(saved[0].provider, 'ollama');
    assert.equal(saved[0].model, 'fixture-local');
    assert.match(saved[0].lyrics, /\[Verse\]/);
    assert.doesNotMatch(saved[0].lyrics, /private reasoning/);
    assert.ok(saved[0].duration > 0);
    assert.equal(db.getAllGenerationsWithContext()[0].lyrics_set_id, album.id);
    assert.equal(db.getLyricsSet(album.id as number)?.songs.length, 0);
    const repeated = await runScratchLyrics({
      jobId: submitted.job.id, userId: 'user', input, attempt: 2, signal: new AbortController().signal,
      emit: () => {}, audio: {} as any, throwIfCancelled: () => {},
    }, async () => { throw new Error('must reuse the saved lyric'); });
    assert.equal(repeated.id, saved[0].id);
    assert.equal(db.getGenerations().length, 1);

    for (response of ['bad json', JSON.stringify({ title: 'Empty', lyrics: '' }), JSON.stringify({ title: 'Tags', lyrics: '[Verse]\n[Chorus]' })]) {
      await assert.rejects(writeScratchLyrics(input), /model returned/);
    }
    const controller = new AbortController();
    let checks = 0;
    await assert.rejects(runScratchLyrics({
      jobId: 'cancel-test', userId: 'user', input, attempt: 1, signal: controller.signal,
      emit: () => {}, audio: {} as any,
      throwIfCancelled: () => { if (++checks === 2) throw new Error('cancelled'); },
    }, async () => ({ title: 'Discard', lyrics: 'Words', caption: '', key: '', model: 'fixture-local', systemPrompt: '', userPrompt: '' })), /cancelled/);
    assert.equal(db.getGenerations().length, 1);
    assert.equal(db.getProfiles().length, 1);
    await assert.rejects(runScratchLyrics({
      jobId: 'delete-test', userId: 'user', input, attempt: 1, signal: controller.signal,
      emit: () => {}, audio: {} as any, throwIfCancelled: () => {},
    }, async () => {
      db.deleteLyricsSet(album.id as number);
      return { title: 'Discard', lyrics: 'Words', caption: '', key: '', model: 'fixture-local', systemPrompt: '', userPrompt: '' };
    }), /deleted/);
    assert.equal(db.getProfiles().length, 0);
  } finally {
    globalThis.fetch = originalFetch;
    database.closeDb();
    fs.rmSync(root, { recursive: true, force: true });
  }
});
