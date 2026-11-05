import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { prepareSa3Instrumental, sa3Hash, type Sa3StemDependencies } from './sa3Stems.js';

async function fixture(t: test.TestContext) {
  const root = await fs.promises.mkdtemp(path.join(process.cwd(), '.test-sa3-stems-'));
  t.after(() => fs.promises.rm(root, { recursive: true, force: true }));
  const source = path.join(root, 'originals', 'track.flac');
  await fs.promises.mkdir(path.dirname(source));
  await fs.promises.writeFile(source, 'original audio');
  const model = path.join(root, 'model.gguf');
  await fs.promises.writeFile(model, 'weights');
  const calls: string[] = [];
  const conversions: Array<{ input: string; crop?: boolean }> = [];
  let mode = 'done';
  let duration = 12.25;
  let channels = 2;
  let failConvert = false;
  const deps: Sa3StemDependencies = {
    fetch: (async (input: string | URL | Request) => {
      const url = String(input);
      calls.push(url);
      if (url.includes('/separate?')) return Response.json({ id: 'job-1' });
      if (url.includes('/progress?')) return Response.json({ status: mode, progress: 1, error: mode === 'failed' ? 'bad weights' : undefined });
      if (url.includes('/result?')) return Response.json({ stems: mode === 'missing' ? [{ name: 'Vocals', index: 0 }] : [{ name: 'Vocals', index: 0 }, { name: '02_Instrumental', index: 1 }] });
      if (url.includes('/serve?')) return new Response('instrumental neural output');
      if (url.includes('/release?')) return Response.json({ released: true });
      throw new Error(`Unexpected request ${url}`);
    }) as typeof fetch,
    convert: async (input, output, signal, crop) => {
      signal?.throwIfAborted();
      conversions.push({ input, crop });
      if (failConvert && input.endsWith('separated.wav')) throw new Error('conversion failed');
      await fs.promises.copyFile(input, output);
    },
    inspect: async () => ({ duration, sampleRate: 44100, channels }),
  };
  return { root, source, model, calls, conversions, deps,
    opts: { source, cacheDir: path.join(root, 'cache'), modelFiles: [model], engineUrl: 'http://mock', pollMs: 1 },
    mode: (value: string) => { mode = value; },
    duration: (value: number) => { duration = value; },
    channels: (value: number) => { channels = value; },
    failConvert: () => { failConvert = true; },
  };
}

test('SA3 level 5 takes instrumental, converts after separation, preserves originals and reuses verified cache', async t => {
  const f = await fixture(t);
  const before = await sa3Hash(f.source);
  const result = await prepareSa3Instrumental(f.opts, f.deps);
  assert.equal(result.seconds_total, 12.25);
  assert.equal(result.cached, false);
  assert.equal(await fs.promises.readFile(result.audioPath, 'utf8'), 'instrumental neural output');
  assert.ok(f.calls[0].endsWith('separate?level=5'));
  assert.ok(f.calls.some(c => c.endsWith('serve?id=job-1&stem=1')));
  assert.ok(f.calls.at(-1)?.includes('/release?'));
  assert.equal(f.conversions.length, 2);
  assert.equal(f.conversions[0].crop, false);
  assert.equal(path.basename(f.conversions[1].input), 'separated.wav');
  const callCount = f.calls.length;
  assert.equal((await prepareSa3Instrumental(f.opts, f.deps)).cached, true);
  assert.equal(f.calls.length, callCount);
  assert.equal(await sa3Hash(f.source), before);
  assert.deepEqual(await fs.promises.readdir(path.dirname(f.source)), ['track.flac']);
});

test('SA3 fingerprints source bytes and model bytes even when size and mtime are unchanged', async t => {
  const f = await fixture(t);
  const first = await prepareSa3Instrumental(f.opts, f.deps);
  const stat = await fs.promises.stat(f.source);
  await fs.promises.writeFile(f.source, 'changed! audio');
  await fs.promises.utimes(f.source, stat.atime, stat.mtime);
  const second = await prepareSa3Instrumental(f.opts, f.deps);
  assert.notEqual(first.fingerprint, second.fingerprint);
  await fs.promises.writeFile(f.model, 'updated');
  const third = await prepareSa3Instrumental(f.opts, f.deps);
  assert.notEqual(second.fingerprint, third.fingerprint);
  assert.equal(third.cached, false);
});

test('SA3 corrupt cache audio is recomputed; force also bypasses a good entry', async t => {
  const f = await fixture(t);
  const result = await prepareSa3Instrumental(f.opts, f.deps);
  await fs.promises.writeFile(result.audioPath, 'corrupt');
  assert.equal((await prepareSa3Instrumental(f.opts, f.deps)).cached, false);
  assert.equal((await prepareSa3Instrumental({ ...f.opts, force: true }, f.deps)).cached, false);
  assert.equal(f.calls.filter(c => c.includes('/separate?')).length, 3);
});

for (const failure of ['failed', 'missing', 'conversion', 'format', 'too long']) {
  test(`SA3 ${failure} leaves no reusable entry and releases the engine job`, async t => {
    const f = await fixture(t);
    if (failure === 'conversion') f.failConvert();
    else if (failure === 'format') f.channels(1);
    else if (failure === 'too long') f.duration(385);
    else f.mode(failure);
    // A missing stem is discovered after a successful completion status.
    if (failure === 'missing') {
      const fetch = f.deps.fetch;
      f.deps.fetch = (async (input, init) => String(input).includes('/progress?')
        ? Response.json({ status: 'done' }) : fetch(input, init)) as typeof fetch;
    }
    await assert.rejects(prepareSa3Instrumental(f.opts, f.deps));
    assert.ok(f.calls.at(-1)?.includes('/release?'));
    const files = await fs.promises.readdir(f.opts.cacheDir);
    assert.deepEqual(files, []);
  });
}

test('SA3 cancellation while submission is in flight still collects and releases the accepted job', async t => {
  const f = await fixture(t);
  const controller = new AbortController();
  const fetch = f.deps.fetch;
  f.deps.fetch = (async (input, init) => {
    const res = await fetch(input, init);
    if (String(input).includes('/separate?')) controller.abort();
    return res;
  }) as typeof fetch;
  await assert.rejects(prepareSa3Instrumental({ ...f.opts, signal: controller.signal }, f.deps), { name: 'AbortError' });
  assert.ok(f.calls.at(-1)?.includes('/release?'));
  assert.equal(f.calls.length, 2);
  assert.deepEqual(await fs.promises.readdir(f.opts.cacheDir), []);
});

test('SA3 timeout during polling cancels by release and removes scratch audio', async t => {
  const f = await fixture(t);
  f.mode('running');
  await assert.rejects(prepareSa3Instrumental({ ...f.opts, timeoutMs: 40, pollMs: 100 }, f.deps));
  assert.ok(f.calls.at(-1)?.includes('/release?'));
  assert.deepEqual(await fs.promises.readdir(f.opts.cacheDir), []);
});

test('SA3 rejects source-tree output and missing model provenance before engine work', async t => {
  const f = await fixture(t);
  await assert.rejects(prepareSa3Instrumental({ ...f.opts, cacheDir: path.join(path.dirname(f.source), 'cache') }, f.deps), /outside/);
  await assert.rejects(prepareSa3Instrumental({ ...f.opts, modelFiles: [] }, f.deps), /model files/);
  assert.equal(f.calls.length, 0);
});

test('SA3 changing source during separation refuses publication', async t => {
  const f = await fixture(t);
  const fetch = f.deps.fetch;
  f.deps.fetch = (async (input, init) => {
    if (String(input).includes('/serve?')) await fs.promises.writeFile(f.source, 'replaced while running');
    return fetch(input, init);
  }) as typeof fetch;
  await assert.rejects(prepareSa3Instrumental(f.opts, f.deps), /changed during/);
  assert.ok(f.calls.at(-1)?.includes('/release?'));
});

test('SA3 cancellation during polling uses an independent signal to release the job', async t => {
  const f = await fixture(t);
  const controller = new AbortController();
  f.mode('running');
  let releasedWithLiveSignal = false;
  const fetch = f.deps.fetch;
  f.deps.fetch = (async (input, init) => {
    const res = await fetch(input, init);
    if (String(input).includes('/progress?')) controller.abort();
    if (String(input).includes('/release?')) releasedWithLiveSignal = !init?.signal?.aborted;
    return res;
  }) as typeof fetch;
  await assert.rejects(prepareSa3Instrumental({ ...f.opts, signal: controller.signal }, f.deps), { name: 'AbortError' });
  assert.equal(releasedWithLiveSignal, true);
  assert.deepEqual(await fs.promises.readdir(f.opts.cacheDir), []);
});

test('SA3 a download HTTP error releases the job and never caches the error body', async t => {
  const f = await fixture(t);
  const fetch = f.deps.fetch;
  f.deps.fetch = (async (input, init) => String(input).includes('/serve?')
    ? new Response('download failed', { status: 503 }) : fetch(input, init)) as typeof fetch;
  await assert.rejects(prepareSa3Instrumental(f.opts, f.deps), /503 download failed/);
  assert.ok(f.calls.at(-1)?.includes('/release?'));
  assert.deepEqual(await fs.promises.readdir(f.opts.cacheDir), []);
});

test('SA3 force failure keeps a previous good cache entry intact', async t => {
  const f = await fixture(t);
  const first = await prepareSa3Instrumental(f.opts, f.deps);
  f.failConvert();
  await assert.rejects(prepareSa3Instrumental({ ...f.opts, force: true }, f.deps), /conversion failed/);
  assert.equal(await sa3Hash(first.audioPath), first.audioHash);
  assert.equal((await prepareSa3Instrumental(f.opts, f.deps)).cached, true);
});
