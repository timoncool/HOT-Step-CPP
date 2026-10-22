import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { execFile } from 'child_process';
import { __setCaller, __setConverter, startUnderstand, getUnderstand, __setPollMs, __setRetryDelays, __test, cancelJob, convertOnce, fitPlan, getJob, isInsideDir, isLoopback, sniffAudio, checkUploadMagic, submitGenerate, sweepTmp, dawAuthError } from './dawFacade.js';
import { getFFmpegPath } from '../../config.js';
import { DawError, parseDawRequest } from './dawSchema.js';
import { estimate, getFigures, getLoaded, resetEta, setLoaded } from './dawEta.js';

test('sweep removes an old upload-less job', () => {
  const now = Date.now();
  __test.addJob({ id: 'old-no-upload', upstreamId: 'u1', seeds: [], createdAt: now - 7 * 3600_000 });
  __test.addJob({ id: 'fresh', upstreamId: 'u2', seeds: [], createdAt: now });
  __test.addJob({ id: 'failed-old', upstreamId: 'u3', seeds: [], createdAt: now - 20 * 60_000, finalState: 'failed', finishedAt: now - 15 * 60_000 });
  sweepTmp(now);
  const r = { old: __test.hasJob('old-no-upload'), fresh: __test.hasJob('fresh'), failedOld: __test.hasJob('failed-old') };
  console.log('sweep result:', JSON.stringify(r));
  assert.deepEqual(r, { old: false, fresh: true, failedOld: false });
});

test('cancel on a done job keeps done', async () => {
  __test.addJob({ id: 'done-job', upstreamId: 'u4', seeds: [], createdAt: Date.now(), finalState: 'done', finishedAt: Date.now() });
  const r = await cancelJob('done-job');
  console.log('cancel on done:', JSON.stringify(r));
  assert.equal(r.state, 'done');
});

test('two concurrent take requests trigger one conversion', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dawconv-'));
  const out = path.join(dir, 'take_0.wav');
  let calls = 0;
  const parts: string[] = [];
  const prev = __setConverter(async (_src, dst) => {
    calls++; parts.push(dst);
    await new Promise((r) => setTimeout(r, 50));
    fs.writeFileSync(dst, 'x');
  });
  try {
    await Promise.all([convertOnce('src.flac', out), convertOnce('src.flac', out)]);
    console.log('conversions:', calls, 'part:', path.basename(parts[0]), 'out exists:', fs.existsSync(out));
    assert.equal(calls, 1);
    assert.ok(fs.existsSync(out));
    assert.notEqual(parts[0], `${out}.part.wav`);
    assert.deepEqual(fs.readdirSync(dir), ['take_0.wav']);
  } finally {
    __setConverter(prev);
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('failed conversion cleans the part file', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dawconv-'));
  const out = path.join(dir, 'take_0.wav');
  const prev = __setConverter(async (_src, dst) => { fs.writeFileSync(dst, 'half'); throw new Error('ffmpeg died'); });
  try {
    await assert.rejects(convertOnce('src.flac', out));
    const left = fs.readdirSync(dir);
    console.log('left after failure:', JSON.stringify(left));
    assert.deepEqual(left, []);
  } finally {
    __setConverter(prev);
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('IPv4-mapped loopback is loopback, mapped private is not', () => {
  const r = {
    mapped127: isLoopback('::ffff:127.0.0.1'), mapped127_5: isLoopback('::ffff:127.0.0.5'),
    plain127_2: isLoopback('127.0.0.2'), v6: isLoopback('::1'), mapped10: isLoopback('::ffff:10.0.0.1'),
  };
  console.log('loopback:', JSON.stringify(r));
  assert.deepEqual(r, { mapped127: true, mapped127_5: true, plain127_2: true, v6: true, mapped10: false });
});

test('magic-byte sniff rejects a .wav that is really text', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dawsniff-'));
  try {
    const fake = path.join(dir, 'audio.wav');
    fs.writeFileSync(fake, 'hello, this is not audio at all');
    let code: number | null = null;
    try { checkUploadMagic(fake); } catch (e) { if (e instanceof DawError) code = e.status; }
    const real = path.join(dir, 'ok.wav');
    const hdr = Buffer.alloc(44); hdr.write('RIFF', 0, 'latin1'); hdr.write('WAVE', 8, 'latin1');
    fs.writeFileSync(real, hdr);
    checkUploadMagic(real);
    console.log('fake wav status:', code, 'sniff flac:', sniffAudio(Buffer.from('fLaC\0\0\0\0')), 'sniff aifc:', sniffAudio(Buffer.from('FORM\0\0\0\0AIFC')));
    assert.equal(code, 400);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

// ── SA3 sequential takes / YuE2 trim ──────────────────────────────────────

/**
 * Fake loopback server: each POST /api/generate is a job that succeeds after
 * `ticks` status polls. `waitFor` resolves once a predicate over the call log
 * holds, so tests sync on events instead of sleeping. `statusFail(id)` returns
 * a status code to fail that poll with, `gatePost(n)` holds the n-th POST open.
 */
function fakeUpstream(backend: string, ticks = 2, opts: { statusFail?: (id: string, n: number) => number | null; gatePost?: (n: number) => Promise<void> | null; nUrls?: number; batchMax?: number } = {}) {
  const posts: any[] = [];
  const cancels: string[] = [];
  const statusCalls: string[] = [];
  const polls = new Map<string, number>();
  const waiters: Array<{ pred: () => boolean; resolve: () => void }> = [];
  const notify = () => { for (const w of [...waiters]) if (w.pred()) { waiters.splice(waiters.indexOf(w), 1); w.resolve(); } };
  const waitFor = (pred: () => boolean) => new Promise<void>((resolve) => { if (pred()) resolve(); else waiters.push({ pred, resolve }); });
  const handle = async (method: string, route: string, body?: any) => {
    if (route === '/api/backends') return { status: 200, data: { activeId: backend } };
    if (route.startsWith('/api/capabilities')) return { status: 200, data: { core: { batch: { max: opts.batchMax ?? 1 } } } };
    if (method === 'POST' && route === '/api/generate') {
      const n = posts.length;
      posts.push(body);
      notify();
      const gate = opts.gatePost?.(n);
      if (gate) await gate;
      return { status: 200, data: { jobId: `up${n}` } };
    }
    const c = /^\/api\/generate\/cancel\/(.+)$/.exec(route);
    if (c) { cancels.push(c[1]); polls.set(c[1], -1); return { status: 200, data: {} }; }
    const s = /^\/api\/generate\/status\/(up\d+)$/.exec(route);
    if (s) {
      const k = s[1];
      statusCalls.push(k);
      const fail = opts.statusFail?.(k, statusCalls.filter((x) => x === k).length);
      if (fail) return { status: fail, data: { error: 'boom' } };
      const n = polls.get(k) ?? 0;
      if (n < 0) return { status: 200, data: { status: 'cancelled' } };
      polls.set(k, n + 1);
      if (n < ticks) return { status: 200, data: { status: 'running', progress: 50 } };
      return { status: 200, data: { status: 'succeeded', result: { audioUrls: Array.from({ length: opts.nUrls ?? 1 }, (_, i) => `/audio/${k}${i ? `_${i}` : ''}.flac`), durations: Array.from({ length: opts.nUrls ?? 1 }, () => backend === 'yue2' ? 63.2 : 10) } } };
    }
    return { status: 404, data: { error: route } };
  };
  const caller = async (method: string, route: string, body?: any) => {
    const r = await handle(method, route, body);
    notify();
    return r;
  };
  return { caller, posts, cancels, statusCalls, polls, waitFor };
}

test('SA3 takes=4 runs 4 single-take jobs with distinct seeds under one facade id', async () => {
  const up = fakeUpstream('sa3');
  const prevC = __setCaller(up.caller); const prevP = __setPollMs(1);
  try {
    await submitGenerate('sa3-4', { mode: 'text2music', caption: 'x', takes: 4, seed: 100 } as any, { batchSize: 4, seed: 100 });
    await __test.getJob('sa3-4')!.seq?.done;
    const r = await getJob('sa3-4') as any;
    const seeds = up.posts.map((p) => p.seed);
    console.log('sa3 posts:', up.posts.length, 'seeds:', JSON.stringify(seeds), 'batchSizes:', JSON.stringify(up.posts.map((p) => p.batchSize)),
      'state:', r.state, 'take_count:', r.take_count, 'takes:', r.takes.length, 'job seeds:', JSON.stringify(r.takes.map((t: any) => t.seed)));
    assert.equal(up.posts.length, 4);
    assert.deepEqual(seeds, [100, 101, 102, 103]);
    assert.equal(new Set(seeds).size, 4);
    assert.ok(up.posts.every((p) => p.batchSize === 1 && p.randomSeed === false));
    assert.equal(r.state, 'done');
    assert.equal(r.take_count, 4);
    assert.equal(r.takes.length, 4);
    assert.deepEqual(r.takes.map((t: any) => t.seed), [100, 101, 102, 103]);
  } finally { __setCaller(prevC); __setPollMs(prevP); }
});

test('SA3 takes without a seed draw 4 distinct random seeds', async () => {
  const up = fakeUpstream('sa3', 0);
  const prevC = __setCaller(up.caller); const prevP = __setPollMs(1);
  try {
    await submitGenerate('sa3-rand', { mode: 'text2music', caption: 'x', takes: 4 } as any, {});
    await __test.getJob('sa3-rand')!.seq?.done;
    const seeds = up.posts.map((p) => p.seed);
    console.log('sa3 random seeds:', JSON.stringify(seeds));
    assert.equal(seeds.length, 4);
    assert.equal(new Set(seeds).size, 4);
  } finally { __setCaller(prevC); __setPollMs(prevP); }
});

test('SA3 cancel mid-run stops the current take and skips the rest', async () => {
  const up = fakeUpstream('sa3', 1000);
  const prevC = __setCaller(up.caller); const prevP = __setPollMs(1);
  try {
    await submitGenerate('sa3-cancel', { mode: 'text2music', caption: 'x', takes: 4, seed: 7 } as any, {});
    await up.waitFor(() => (up.polls.get('up0') ?? 0) >= 1);
    const mid = await getJob('sa3-cancel') as any;
    const c = await cancelJob('sa3-cancel');
    await __test.getJob('sa3-cancel')!.seq?.done;
    console.log('cancel: progress before', Number(mid.progress).toFixed(3), 'state', c.state, 'posts', up.posts.length, 'cancelled', JSON.stringify(up.cancels));
    assert.equal(c.state, 'cancelled');
    assert.equal(up.posts.length, 1);
    assert.deepEqual(up.cancels, ['up0']);
    assert.ok(mid.progress > 0.1 && mid.progress < 0.25, `progress ${mid.progress}`);
  } finally { __setCaller(prevC); __setPollMs(prevP); }
});

test('SA3 cancel during an in-flight submit cancels only the new take, never the finished one', async () => {
  let release!: () => void;
  const gate = new Promise<void>((r) => { release = r; });
  const up = fakeUpstream('sa3', 0, { gatePost: (n) => n === 1 ? gate : null });
  const prevC = __setCaller(up.caller); const prevP = __setPollMs(1);
  try {
    await submitGenerate('sa3-inflight', { mode: 'text2music', caption: 'x', takes: 4, seed: 7 } as any, {});
    await up.waitFor(() => up.posts.length === 2);
    const c = await cancelJob('sa3-inflight');
    release();
    await __test.getJob('sa3-inflight')!.seq?.done;
    const toFinished = up.cancels.filter((x) => x === 'up0').length, toNew = up.cancels.filter((x) => x === 'up1').length;
    console.log('inflight cancel: state', c.state, 'cancels', JSON.stringify(up.cancels), 'to finished up0', toFinished, 'to new up1', toNew, 'posts', up.posts.length);
    assert.equal(c.state, 'cancelled');
    assert.equal(toFinished, 0);
    assert.equal(toNew, 1);
    assert.equal(up.posts.length, 2);
  } finally { __setCaller(prevC); __setPollMs(prevP); }
});

test('SA3 status poll retries 2 transient 500s and still gives 4 takes', async () => {
  const up = fakeUpstream('sa3', 1, { statusFail: (id, n) => id === 'up1' && n <= 2 ? 500 : null });
  const prevC = __setCaller(up.caller); const prevP = __setPollMs(1); const prevR = __setRetryDelays([1, 1, 1]);
  try {
    await submitGenerate('sa3-retry', { mode: 'text2music', caption: 'x', takes: 4, seed: 1 } as any, {});
    await __test.getJob('sa3-retry')!.seq?.done;
    const r = await getJob('sa3-retry') as any;
    console.log('retry: state', r.state, 'takes', r.takes.length, 'up1 status calls', up.statusCalls.filter((x) => x === 'up1').length, 'error', r.error);
    assert.equal(r.state, 'done');
    assert.equal(r.takes.length, 4);
  } finally { __setCaller(prevC); __setPollMs(prevP); __setRetryDelays(prevR); }
});

test('SA3 failing the 3rd of 4 takes leaves the job partial with 2 takes', async () => {
  const up = fakeUpstream('sa3', 0, { statusFail: (id) => id === 'up2' ? 500 : null });
  const prevC = __setCaller(up.caller); const prevP = __setPollMs(1); const prevR = __setRetryDelays([1, 1, 1]);
  try {
    await submitGenerate('sa3-partial', { mode: 'text2music', caption: 'x', takes: 4, seed: 1 } as any, {});
    await __test.getJob('sa3-partial')!.seq?.done;
    const r = await getJob('sa3-partial') as any;
    console.log('partial: state', r.state, 'takes', r.takes.length, 'ready', JSON.stringify(r.takes.map((t: any) => t.ready)), 'up2 status calls', up.statusCalls.filter((x) => x === 'up2').length, 'posts', up.posts.length, 'error', r.error);
    assert.equal(r.state, 'partial');
    assert.equal(r.takes.length, 2);
    assert.ok(r.takes.every((t: any) => t.ready));
    assert.ok(r.error);
    assert.equal(up.posts.length, 3);
  } finally { __setCaller(prevC); __setPollMs(prevP); __setRetryDelays(prevR); }
});

test('SA3 cancel after the last take finished keeps the job done', async () => {
  const up = fakeUpstream('sa3', 1);
  const prevC = __setCaller(up.caller); const prevP = __setPollMs(1);
  try {
    await submitGenerate('sa3-late-cancel', { mode: 'text2music', caption: 'x', takes: 2, seed: 3 } as any, {});
    await __test.getJob('sa3-late-cancel')!.seq?.done;
    const c = await cancelJob('sa3-late-cancel');
    const st = await getJob('sa3-late-cancel') as any;
    console.log('late cancel: cancel state', c.state, 'job state', st.state, 'takes', st.takes.length, 'upstream cancels', up.cancels.length);
    assert.equal(c.state, 'done');
    assert.equal(st.state, 'done');
    assert.equal(st.takes.length, 2);
    assert.equal(up.cancels.length, 0);
  } finally { __setCaller(prevC); __setPollMs(prevP); }
});

test('SA3 job swept from the map stops submitting takes', async () => {
  const up = fakeUpstream('sa3', 2);
  const prevC = __setCaller(up.caller); const prevP = __setPollMs(1);
  try {
    await submitGenerate('sa3-swept', { mode: 'text2music', caption: 'x', takes: 4, seed: 1 } as any, {});
    const done = __test.getJob('sa3-swept')!.seq!.done;
    await up.waitFor(() => (up.polls.get('up0') ?? 0) >= 1);
    const before = up.posts.length;
    __test.removeJob('sa3-swept');
    await done;
    console.log('swept: posts before sweep', before, 'after', up.posts.length, 'upstream cancels', JSON.stringify(up.cancels));
    assert.equal(up.posts.length, before);
    assert.deepEqual(up.cancels, ['up0']);
  } finally { __setCaller(prevC); __setPollMs(prevP); }
});

test('YuE2 duration_s marks takes trimmed_from_s and reports the requested length', async () => {
  const up = fakeUpstream('yue2', 0);
  const prevC = __setCaller(up.caller);
  try {
    await submitGenerate('yue-trim', { mode: 'text2music', caption: 'x', duration_s: 10 } as any, { duration: 10 });
    const r = await getJob('yue-trim') as any;
    console.log('yue2 take:', JSON.stringify(r.takes[0]));
    assert.equal(r.takes[0].trimmed_from_s, 63.2);
    assert.equal(r.takes[0].duration_s, 10);
  } finally { __setCaller(prevC); }
});

/** Data chunk length of a WAV file, in sample frames. */
function wavFrames(file: string): { frames: number; rate: number } {
  const b = fs.readFileSync(file);
  let off = 12, rate = 0, ch = 0, bits = 0;
  while (off + 8 <= b.length) {
    const id = b.toString('latin1', off, off + 4); const size = b.readUInt32LE(off + 4);
    if (id === 'fmt ') { ch = b.readUInt16LE(off + 10); rate = b.readUInt32LE(off + 12); bits = b.readUInt16LE(off + 22); }
    if (id === 'data') return { frames: size / (ch * bits / 8), rate };
    off += 8 + size + (size % 2);
  }
  throw new Error('no data chunk');
}

test('trim export cuts a 12 s render to 10.000 s at 48 kHz with a fade, and never pads', { skip: !getFFmpegPath() && 'ffmpeg not available' }, async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dawtrim-'));
  try {
    const src = path.join(dir, 'src.wav');
    await new Promise<void>((res, rej) => execFile(getFFmpegPath()!, ['-y', '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=44100:duration=12', '-ac', '2', src], (e) => e ? rej(e) : res()));
    const out = path.join(dir, 'take_0.wav');
    await convertOnce(src, out, { trimS: 10 });
    const w = wavFrames(out);
    const b = fs.readFileSync(out);
    const lastAbs = Math.abs(b.readFloatLE(b.length - 8));
    const short = path.join(dir, 'take_1.wav');
    await convertOnce(src, short, { trimS: 20 });
    const s = wavFrames(short);
    console.log('trim: frames', w.frames, 'rate', w.rate, 'seconds', (w.frames / w.rate).toFixed(5), 'last sample', lastAbs.toExponential(2), '| no-pad frames', s.frames, (s.frames / s.rate).toFixed(5), 's');
    assert.equal(w.rate, 48000);
    assert.ok(Math.abs(w.frames - 480000) <= 48, `frames ${w.frames}`);
    assert.ok(lastAbs < 0.01, `last sample ${lastAbs}`);
    assert.ok(Math.abs(s.frames - 576000) <= 48, `no-pad frames ${s.frames}`);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

/** Upstream whose job succeeds on the first status poll, with a chosen active backend and DiT. */
function instantUpstream(activeId: string, loadedModel: string | null, final = 'succeeded') {
  let n = 0;
  const state = { activeId, loadedModel };
  const caller = async (method: string, route: string) => {
    if (route === '/api/backends') return { status: 200, data: { activeId: state.activeId } };
    if (route === '/api/models') return { status: 200, data: { defaults: { synth_model: state.loadedModel } } };
    if (method === 'POST' && route === '/api/generate') return { status: 200, data: { jobId: `iu${n++}` } };
    if (route.startsWith('/api/generate/status/')) return { status: 200, data: { status: final, result: { audioUrls: ['/a.flac'], durations: [10] } } };
    return { status: 404, data: {} };
  };
  return { caller, state };
}

test('a 3600 s poll gap does not teach the EMA', async () => {
  resetEta();
  const up = instantUpstream('ace', 'acestep-v15-turbo-Q8.gguf');
  const prevC = __setCaller(up.caller);
  try {
    const before = getFigures('ace_turbo').base_s;
    __test.addJob({ id: 'slow-poll', upstreamId: 'iu9', seeds: [], createdAt: Date.now() - 3600_000,
      eta: { key: 'ace_turbo', takes: 1, durationS: 30, estS: 4.2, cold: false } } as any);
    const r = await getJob('slow-poll') as any;
    const after = getFigures('ace_turbo').base_s;
    console.log('poll gap 3600 s: state', r.state, 'base_s before/after', before, after);
    assert.equal(r.state, 'done');
    assert.equal(after, before);
  } finally { __setCaller(prevC); }
});

test('a fast poll records the run', async () => {
  resetEta();
  const up = instantUpstream('ace', 'acestep-v15-turbo-Q8.gguf');
  const prevC = __setCaller(up.caller);
  try {
    const before = getFigures('ace_turbo').base_s;
    __test.addJob({ id: 'fast-poll', upstreamId: 'iu8', seeds: [], createdAt: Date.now() - 20_000, lastPollAt: Date.now() - 1000,
      eta: { key: 'ace_turbo', takes: 1, durationS: 30, estS: 4.2, cold: false } } as any);
    await getJob('fast-poll');
    const after = getFigures('ace_turbo').base_s;
    console.log('fast poll 20 s run: base_s before/after', before, after);
    assert.ok(after - before > 3, `moved ${after - before}`);
  } finally { __setCaller(prevC); }
});

for (const final of ['failed', 'cancelled']) {
  test(`${final} jobs are never recorded`, async () => {
    resetEta();
    const up = instantUpstream('ace', 'acestep-v15-turbo-Q8.gguf', final);
    const prevC = __setCaller(up.caller);
    try {
      const before = getFigures('ace_turbo');
      __test.addJob({ id: `rec-${final}`, upstreamId: 'iu7', seeds: [], createdAt: Date.now() - 20_000, lastPollAt: Date.now() - 500,
        eta: { key: 'ace_turbo', takes: 1, durationS: 30, estS: 4.2, cold: false } } as any);
      const r = await getJob(`rec-${final}`) as any;
      const after = getFigures('ace_turbo');
      console.log(`${final}: state`, r.state, 'base_s', before.base_s, after.base_s, 'source', after.source);
      assert.equal(after.base_s, before.base_s);
      assert.equal(after.source, 'default');
    } finally { __setCaller(prevC); }
  });
}

test('cold/warm comes from the real resident backend, not the last facade job', async () => {
  resetEta();
  // The facade last ran ace_base, then the UI switched away and back, leaving turbo loaded.
  setLoaded('ace_base');
  const up = instantUpstream('ace', 'acestep-v15-turbo-Q8.gguf');
  const prevC = __setCaller(up.caller);
  try {
    await submitGenerate('ui-switch', { mode: 'text2music', caption: 'x', dit_model: 'acestep-v15-base-Q8.gguf' } as any, {});
    const e = __test.getJob('ui-switch')!.eta!;
    const warm = estimate('ace_base', 1, 30, 'ace_base');
    console.log('ui switch: key', e.key, 'cold', e.cold, 'est', e.estS, 'warm', warm, 'cold_load_s', getFigures('ace_base').cold_load_s);
    assert.equal(e.cold, true);
    assert.ok(Math.abs(e.estS - warm - getFigures('ace_base').cold_load_s) < 1e-9);
    // No dit_model: the loaded DiT picks the key, and it is warm.
    up.state.loadedModel = 'acestep-v15-base-Q8.gguf';
    await submitGenerate('no-dit', { mode: 'text2music', caption: 'x' } as any, {});
    const e2 = __test.getJob('no-dit')!.eta!;
    console.log('no dit_model: key', e2.key, 'cold', e2.cold);
    assert.equal(e2.key, 'ace_base');
    assert.equal(e2.cold, false);
  } finally { __setCaller(prevC); }
});

test('recording does not mark loaded when the backend changed since', async () => {
  resetEta();
  const up = instantUpstream('sa3', null);
  const prevC = __setCaller(up.caller);
  try {
    __test.addJob({ id: 'moved-on', upstreamId: 'iu6', seeds: [], createdAt: Date.now() - 5000, lastPollAt: Date.now() - 500,
      eta: { key: 'ace_turbo', takes: 1, durationS: 30, estS: 4.2, cold: false } } as any);
    await getJob('moved-on');
    console.log('loaded after record with sa3 active:', getLoaded());
    assert.notEqual(getLoaded(), 'ace_turbo');
  } finally { __setCaller(prevC); }
});

test('job response reports the chosen dit_model and its eta backend', async () => {
  resetEta();
  const up = instantUpstream('ace', 'acestep-v15-base-Q8.gguf');
  const prevC = __setCaller(up.caller);
  try {
    await submitGenerate('pick-turbo', { mode: 'text2music', caption: 'x', model: 'turbo', dit_model: 'acestep-v15-turbo-Q8.gguf' } as any, {});
    const job: any = await getJob('pick-turbo');
    console.log('pick-turbo: dit_model', job.dit_model, 'eta_backend', job.eta_backend);
    assert.equal(job.dit_model, 'acestep-v15-turbo-Q8.gguf');
    assert.equal(job.eta_backend, 'ace_turbo');
  } finally { __setCaller(prevC); }
});

test('ACE fit pads 9.44 s to exactly 480000 frames and cuts 11 s to 480000, with a fade', { skip: !getFFmpegPath() && 'ffmpeg not available' }, async () => {
  const short = fitPlan(Math.round(48000 * 9.44), 10);
  const long = fitPlan(48000 * 11, 10);
  console.log('fitPlan 9.44->10:', JSON.stringify(short), '| 11->10:', JSON.stringify(long));
  assert.deepEqual(short, { frames: 480000, pad: 26880, cut: 0, fadeEndFrame: 453120 });
  assert.deepEqual(long, { frames: 480000, pad: 0, cut: 48000, fadeEndFrame: 480000 });
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dawfit-'));
  try {
    const res: Record<string, number> = {};
    for (const [name, dur] of [['s944', 9.44], ['s11', 11]] as const) {
      const src = path.join(dir, `${name}.wav`);
      await new Promise<void>((r, j) => execFile(getFFmpegPath()!, ['-y', '-f', 'lavfi', '-i', `sine=frequency=440:sample_rate=48000:duration=${dur}`, '-ac', '2', '-c:a', 'pcm_s16le', src], (e) => e ? j(e) : r()));
      const out = path.join(dir, `${name}_out.wav`);
      await convertOnce(src, out, { fitS: 10, srcS: dur });
      const w = wavFrames(out);
      const b = fs.readFileSync(out);
      // Sample just before the original end (pad case) must be faded.
      const dataStart = b.length - w.frames * 8;
      const preEnd = Math.min(Math.round(dur * 48000), 480000) - 1;
      res[`${name}_frames`] = w.frames;
      res[`${name}_rate`] = w.rate;
      res[`${name}_preEndAbs`] = Math.abs(b.readFloatLE(dataStart + preEnd * 8));
      res[`${name}_lastAbs`] = Math.abs(b.readFloatLE(b.length - 8));
    }
    console.log('fit export:', JSON.stringify(res));
    assert.equal(res.s944_frames, 480000);
    assert.equal(res.s11_frames, 480000);
    assert.equal(res.s944_rate, 48000);
    assert.ok(res.s944_preEndAbs < 0.01 && res.s944_lastAbs === 0, 'pad tail faded and silent');
    assert.ok(res.s11_lastAbs < 0.01, 'cut end faded');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('YuE2 takes=2 reports seed + i per take', async () => {
  const up = fakeUpstream('yue2', 0, { nUrls: 2, batchMax: 2 });
  const prevC = __setCaller(up.caller); const prevP = __setPollMs(1);
  try {
    await submitGenerate('yue2-seeds', { mode: 'text2music', caption: 'x', takes: 2, seed: 4242 } as any, { batchSize: 2, seed: 4242 });
    const r = await getJob('yue2-seeds') as any;
    const seeds = r.takes.map((t: any) => t.seed);
    console.log('yue2 job seeds:', JSON.stringify(seeds), 'state:', r.state);
    assert.equal(r.takes.length, 2);
    assert.ok(seeds.every((s: unknown) => typeof s === 'number'));
    assert.deepEqual(seeds, [4242, 4243]);
  } finally { __setCaller(prevC); __setPollMs(prevP); }
});

test('YuE2 takes=2 capped at 1 song per job runs 2 single-take jobs with seeds base+i', async () => {
  const up = fakeUpstream('yue2', 0, { batchMax: 1 });
  const prevC = __setCaller(up.caller); const prevP = __setPollMs(1);
  try {
    await submitGenerate('yue2-cap', { mode: 'text2music', caption: 'x', takes: 2, seed: 4242 } as any, { batchSize: 2, seed: 4242 });
    await __test.getJob('yue2-cap')!.seq?.done;
    const r = await getJob('yue2-cap') as any;
    const seeds = r.takes.map((t: any) => t.seed);
    console.log('yue2 capped posts:', up.posts.length, 'takes:', r.takes.length, 'seeds:', JSON.stringify(seeds),
      'postSeeds:', JSON.stringify(up.posts.map((p) => p.seed)), 'urls:', JSON.stringify(r.takes.map((t: any) => t.audio_url ?? t.url)), 'state:', r.state);
    assert.equal(up.posts.length, 2);
    assert.equal(r.takes.length, 2);
    assert.deepEqual(seeds, [4242, 4243]);
    assert.deepEqual(up.posts.map((p) => p.seed), [4242, 4243]);
    assert.deepEqual(up.posts.map((p) => p.batchSize), [1, 1]);
  } finally { __setCaller(prevC); __setPollMs(prevP); }
});

test('audio path containment rejects a sibling dir and accepts a real child', () => {
  const base = path.resolve(os.tmpdir(), 'daw-audio');
  const sibling = isInsideDir(base, base + 'X' + path.sep + 'a.flac');
  const child = isInsideDir(base, path.join(base, 'sub', 'a.flac'));
  const up = isInsideDir(base, path.join(base, '..', 'a.flac'));
  console.log('containment sibling:', sibling, 'child:', child, 'parent:', up);
  assert.equal(sibling, false);
  assert.equal(child, true);
  assert.equal(up, false);
});

test('a seed at the uint32 max with takes 2 is rejected, not run past the engine range', async () => {
  const raw = { mode: 'text2music', caption: 'x', takes: 2, seed: 4294967295 };
  let err: unknown = null;
  let seeds: number[] = [];
  try {
    const req = parseDawRequest(raw);
    const up = fakeUpstream('sa3');
    const prevC = __setCaller(up.caller); const prevP = __setPollMs(1);
    try {
      await submitGenerate('seed-max', req, { seed: req.seed });
      await __test.getJob('seed-max')!.seq?.done;
      seeds = up.posts.map((p) => p.seed);
    } finally { __setCaller(prevC); __setPollMs(prevP); __test.removeJob('seed-max'); }
  } catch (e) { err = e; }
  console.log('seed max: rejected', err instanceof DawError ? (err as DawError).status : null, 'seeds sent:', JSON.stringify(seeds));
  assert.ok(err instanceof DawError);
  assert.equal((err as DawError).status, 400);
  assert.deepEqual(seeds, []);
  // The last valid start for 2 takes still parses.
  assert.equal(parseDawRequest({ ...raw, seed: 4294967294 }).seed, 4294967294);
});

test('a job still loading its model reports eta_s null and phase loading_model, a progressing job a number', async () => {
  resetEta();
  const stageFor: Record<string, any> = {
    'up-load': { status: 'running', stage: 'Loading DiT...', progress: 10 },
    'up-run': { status: 'running', stage: 'Synthesizing', progress: 50, ace_phase_progress: 'Step 4/8' },
  };
  const prevC = __setCaller(async (_m: string, route: string) => {
    const id = route.split('/').pop()!;
    return stageFor[id] ? { status: 200, data: stageFor[id] } : { status: 404, data: {} };
  });
  try {
    // Loading: 30 s since submit against a 4.2 s estimate, so the old code clamps to 0.
    __test.addJob({ id: 'eta-load', upstreamId: 'up-load', seeds: [], createdAt: Date.now() - 30_000,
      eta: { key: 'ace_turbo', takes: 1, durationS: 30, estS: 4.2, cold: true } } as any);
    __test.addJob({ id: 'eta-run', upstreamId: 'up-run', seeds: [], createdAt: Date.now() - 1_000,
      eta: { key: 'ace_turbo', takes: 1, durationS: 30, estS: 20, cold: false } } as any);
    const l = await getJob('eta-load') as any;
    const r = await getJob('eta-run') as any;
    console.log('loading job: state', l.state, 'eta_s', l.eta_s, 'phase', l.phase, '| running job: eta_s', r.eta_s, 'phase', r.phase);
    assert.equal(l.state, 'running');
    assert.equal(l.eta_s, null);
    assert.equal(l.phase, 'loading_model');
    assert.equal(typeof r.eta_s, 'number');
    assert.ok(r.eta_s > 15, 'progressing job keeps its remaining estimate');
    assert.equal(r.phase, 'running');
  } finally { __setCaller(prevC); }
});

test('an SA3 render past its estimate reports eta_s null and phase running, ACE keeps a number', async () => {
  resetEta();
  const stageFor: Record<string, any> = {
    'up-sa3': { status: 'running', stage: 'SA3: rendering 30s (8 steps)...', progress: 5 },
    'up-ace': { status: 'running', stage: 'Synthesizing', progress: 90, ace_phase_progress: 'Step 7/8' },
  };
  const prevC = __setCaller(async (_m: string, route: string) => {
    const id = route.split('/').pop()!;
    return stageFor[id] ? { status: 200, data: stageFor[id] } : { status: 404, data: {} };
  });
  try {
    // 30 s since submit against a 6 s estimate: the old code clamps both to 0.
    __test.addJob({ id: 'eta-sa3', upstreamId: 'up-sa3', seeds: [], createdAt: Date.now() - 30_000,
      eta: { key: 'sa3', takes: 1, durationS: 30, estS: 6, cold: false } } as any);
    __test.addJob({ id: 'eta-ace', upstreamId: 'up-ace', seeds: [], createdAt: Date.now() - 30_000,
      eta: { key: 'ace_turbo', takes: 1, durationS: 30, estS: 6, cold: false } } as any);
    const s = await getJob('eta-sa3') as any;
    const a = await getJob('eta-ace') as any;
    console.log('sa3 job: state', s.state, 'eta_s', s.eta_s, 'phase', s.phase, '| ace job: eta_s', a.eta_s, 'phase', a.phase);
    assert.equal(s.state, 'running');
    assert.equal(s.eta_s, null);
    assert.equal(s.phase, 'running');
    assert.equal(a.eta_s, 0, 'ACE keeps its numeric clamp');
    assert.equal(a.phase, 'running');
  } finally { __setCaller(prevC); }
});


test('fast sync understand response carries analysis_id equal to id', async () => {
  const prevC = __setCaller((async () => ({ status: 200, data: { bpm: 120, key: 'C', scale: 'major', tags: [] } })) as any);
  try {
    const { id, result } = await startUnderstand('und-sync-1', '/tmp/x.wav', 1000);
    assert.ok(result, 'expected sync result');
    console.log('understand sync result:', JSON.stringify(result));
    assert.equal(result.analysis_id, id);
    assert.equal(result.id, id);
    assert.equal(getUnderstand(id).analysis_id, id);
  } finally { __setCaller(prevC); }
});

test('auth: a configured key is required from loopback and proxied callers too', () => {
  const lo = '127.0.0.1';
  assert.equal(dawAuthError('k', undefined, lo), 'X-DAW-Key is missing or wrong');
  assert.equal(dawAuthError('k', 'wrong', lo), 'X-DAW-Key is missing or wrong');
  assert.equal(dawAuthError('k', 'k', lo), null);
  assert.equal(dawAuthError('k', 'k', '10.0.0.9'), null);
  assert.equal(dawAuthError('k', undefined, '::1', { xff: '127.0.0.1' }), 'X-DAW-Key is missing or wrong');
});

test('auth: with no key only demonstrably local callers pass; forwarded hops count', () => {
  const lo = '127.0.0.1';
  assert.equal(dawAuthError('', undefined, lo), null);
  assert.equal(dawAuthError('', undefined, '::ffff:127.0.0.1', { xff: '127.0.0.1, ::1' }), null);
  assert.equal(dawAuthError('', undefined, '::1', { forwarded: 'for="[::1]:4711"' }), null);
  assert.notEqual(dawAuthError('', undefined, '192.168.1.20'), null);
  assert.notEqual(dawAuthError('', undefined, lo, { xff: '192.168.1.20' }), null);
  assert.notEqual(dawAuthError('', undefined, lo, { xff: '127.0.0.1, 8.8.8.8' }), null);
  assert.notEqual(dawAuthError('', undefined, lo, { xff: '127.0.0.1,' }), null);
  assert.notEqual(dawAuthError('', undefined, lo, { realIp: '10.1.1.1' }), null);
  assert.notEqual(dawAuthError('', undefined, lo, { forwarded: 'for=10.1.1.1;proto=http' }), null);
  assert.notEqual(dawAuthError('', undefined, undefined), null);
});

test('YuE2 sequential takes carry trimTo like the batched branch', async () => {
  const up = fakeUpstream('yue2', 0, { batchMax: 1 });
  const prevC = __setCaller(up.caller); const prevP = __setPollMs(1);
  try {
    await submitGenerate('yue2-seq-trim', { mode: 'text2music', caption: 'x', takes: 2, seed: 9, duration_s: 10 } as any, { batchSize: 2, seed: 9, duration: 10 });
    await __test.getJob('yue2-seq-trim')!.seq?.done;
    const r = await getJob('yue2-seq-trim') as any;
    console.log('yue2 sequential takes:', JSON.stringify(r.takes.map((t: any) => [t.duration_s, t.trimmed_from_s])));
    assert.equal(r.takes.length, 2);
    assert.ok(r.takes.every((t: any) => t.duration_s === 10 && t.trimmed_from_s === 63.2));
  } finally { __setCaller(prevC); __setPollMs(prevP); }
});
