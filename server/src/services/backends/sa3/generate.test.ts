import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  SA3_LOADING_STAGE, sa3RenderStage, resolveSa3Sampler, resolveSa3SeedBlend,
} from './generate.js';

test('SA3 sends the captured DiT and reports the response filename, including fallback', async () => {
  const { aceClient } = await import('../../aceClient.js');
  const originalFetch = globalThis.fetch;
  const requested = 'sa3-dit-Q4_K_M-imat.gguf';
  const seen: string[] = [];
  let returned: string | undefined = requested;
  globalThis.fetch = (async (url: string | URL | Request) => {
    assert.equal(new URL(String(url)).searchParams.get('synth_model'), requested);
    return new Response(new Uint8Array([1, 2, 3]), {
      headers: returned ? { 'X-SA3-DiT': returned } : {},
    });
  }) as typeof fetch;
  try {
    for (returned of [requested, 'sa3-dit-BF16.gguf', undefined]) {
      const wav = await aceClient.submitSa3Synth({
        tokens: [1], nTokens: 1, durationSec: 1, synthModel: requested,
        onModelResolved: filename => seen.push(filename),
      });
      assert.deepEqual([...wav], [1, 2, 3]);
    }
    assert.deepEqual(seen, [requested, 'sa3-dit-BF16.gguf']);
  } finally { globalThis.fetch = originalFetch; }
});

test('SA3 selects its default Lua sampler while preserving native and plugin overrides', () => {
  assert.deepEqual(resolveSa3Sampler({}, {}), { sampler: 'pingpong', solver: 'md_pingpong_blend' });
  assert.deepEqual(resolveSa3Sampler({ sa3Sampler: 'pingpong' }, {}), { sampler: 'pingpong' });
  assert.deepEqual(resolveSa3Sampler({ sa3Sampler: 'euler' }, {}), { sampler: 'euler' });
  assert.deepEqual(resolveSa3Sampler({}, { solver: 'heun' }), { sampler: 'pingpong' });
});

test('SA3 blend defaults preserve explicit endpoints and reject non-finite values', () => {
  for (const v of [undefined, null, '', NaN, Infinity, 'invalid']) {
    assert.equal(resolveSa3SeedBlend(v), 0.5);
  }
  for (const [v, expected] of [[0, 0], ['0', 0], [1, 1], ['1', 1], [-1, 0], [2, 1], [0.25, 0.25]] as const) {
    assert.equal(resolveSa3SeedBlend(v), expected);
  }
});

test('SA3 synth sends the default and full blend, and disables breeding at zero or without seed B', async () => {
  const { aceClient } = await import('../../aceClient.js');
  const realFetch = globalThis.fetch;
  const requests: URL[] = [];
  globalThis.fetch = (async (url: string | URL | Request) => {
    requests.push(new URL(String(url)));
    return new Response(new Uint8Array([1, 2, 3]));
  }) as typeof fetch;
  try {
    for (const input of [undefined, 0, 1]) {
      await aceClient.submitSa3Synth({
        tokens: [1], nTokens: 1, durationSec: 1, seed: 123,
        seedB: 456, seedBlend: resolveSa3SeedBlend(input),
      });
    }
    await aceClient.submitSa3Synth({ tokens: [1], nTokens: 1, durationSec: 1, seedBlend: 1 });
    assert.equal(requests.length, 4);
    assert.ok(requests.every(url => url.pathname === '/sa3-synth'));
    assert.equal(requests[0].searchParams.get('seed_b'), '456');
    assert.equal(requests[0].searchParams.get('seed_blend'), '0.5');
    assert.equal(requests[1].searchParams.has('seed_b'), false);
    assert.equal(requests[1].searchParams.has('seed_blend'), false);
    assert.equal(requests[2].searchParams.get('seed_b'), '456');
    assert.equal(requests[2].searchParams.get('seed_blend'), '1');
    assert.equal(requests[3].searchParams.has('seed_b'), false);
    assert.equal(requests[3].searchParams.has('seed_blend'), false);
  } finally {
    globalThis.fetch = realFetch;
  }
});

// Copied from the DAW facade (pr/29 dawFacade.ts:457): a stage matching this
// reports phase loading_model and eta_s null. pr/06 has no facade, so the
// regex is asserted directly here.
const FACADE_LOADING_RE = /^loading\b/i;

test('SA3 stage at submit reads as loading', () => {
  console.log('submit stage:', SA3_LOADING_STAGE);
  assert.match(SA3_LOADING_STAGE, FACADE_LOADING_RE);
});

test('SA3 stage once the render fetch is sent does not read as loading', () => {
  const s = sa3RenderStage(30.4, 8);
  console.log('render stage:', s);
  assert.doesNotMatch(s, FACADE_LOADING_RE);
  assert.ok(s.startsWith('SA3: rendering'));
  assert.ok(s.includes('30s') && s.includes('8 steps'));
});

test('real progress (saving audio) replaces the loading stage', () => {
  const saving = 'Stable Audio 3: saving audio...';
  assert.doesNotMatch(saving, FACADE_LOADING_RE);
});

// The lane must stay held until the engine's worker has provably finished the
// SA3 call. Each test scripts the engine: the render POST, then the replies to
// successive /worker-drain calls ('fail' = socket error, '404' = an engine
// without the route, 'ok' = held until released by the test).
type DrainReply = 'fail' | '404' | 'ok';
async function withScriptedEngine(
  post: 'hang' | 'socket-error',
  drains: DrainReply[],
  body: (h: { calls: string[]; releaseDrain: () => void }) => Promise<void>,
) {
  const realFetch = globalThis.fetch;
  const calls: string[] = [];
  let releaseDrain: () => void = () => {};
  const socketError = () => Object.assign(new TypeError('fetch failed'), { cause: { code: 'UND_ERR_SOCKET' } });
  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    const path = new URL(String(url)).pathname;
    calls.push(path);
    if (path === '/worker-drain') {
      const reply = drains.shift() ?? 'ok';
      if (reply === 'fail') throw socketError();
      if (reply === '404') return new Response('Not Found', { status: 404 });
      await new Promise<void>(r => { releaseDrain = r; });
      return new Response('{"status":"ok"}');
    }
    if (post === 'socket-error') throw socketError();
    return new Promise<Response>((_, reject) => {
      init!.signal!.addEventListener('abort',
        () => reject(new DOMException('aborted', 'AbortError')), { once: true });
    });
  }) as typeof fetch;
  try {
    await body({ calls, releaseDrain: () => releaseDrain() });
  } finally {
    globalThis.fetch = realFetch;
  }
}

async function startSynth(signal?: AbortSignal) {
  const { aceClient } = await import('../../aceClient.js');
  const state = { settled: false };
  const p = aceClient.submitSa3Synth({ tokens: [1], nTokens: 1, durationSec: 1, signal })
    .catch((e: Error) => { state.settled = true; return e; });
  return { p, state };
}

const tick = (ms = 20) => new Promise(r => setTimeout(r, ms));

test('an aborted SA3 render settles only after the engine worker drains', async () => {
  await withScriptedEngine('hang', ['ok'], async ({ calls, releaseDrain }) => {
    const ac = new AbortController();
    const { p, state } = await startSynth(ac.signal);
    await tick(0);
    ac.abort();
    await tick();
    assert.deepEqual(calls, ['/sa3-synth', '/worker-drain']);
    assert.equal(state.settled, false, 'must not settle while the engine is still stopping');
    releaseDrain();
    assert.equal(((await p) as Error).name, 'AbortError');
  });
});

test('a failed drain is retried, not taken as proof the engine stopped', async () => {
  await withScriptedEngine('hang', ['fail', 'ok'], async ({ calls, releaseDrain }) => {
    const ac = new AbortController();
    const { p, state } = await startSynth(ac.signal);
    await tick(0);
    ac.abort();
    await tick(1300);  // past the 1 s retry delay
    assert.deepEqual(calls, ['/sa3-synth', '/worker-drain', '/worker-drain']);
    assert.equal(state.settled, false, 'a socket error on the drain must keep the lane held');
    releaseDrain();
    assert.equal(((await p) as Error).name, 'AbortError');
  });
});

test('a render POST that dies on a socket error drains before it rejects', async () => {
  await withScriptedEngine('socket-error', ['ok'], async ({ calls, releaseDrain }) => {
    const { p, state } = await startSynth();
    await tick();
    assert.deepEqual(calls, ['/sa3-synth', '/worker-drain']);
    assert.equal(state.settled, false);
    releaseDrain();
    assert.equal(((await p) as Error).message, 'fetch failed');
  });
});

test('a 404 from /worker-drain keeps the lane held', async () => {
  await withScriptedEngine('hang', ['404', 'ok'], async ({ calls, releaseDrain }) => {
    const ac = new AbortController();
    const { p, state } = await startSynth(ac.signal);
    await tick(0);
    ac.abort();
    await tick(1300);  // past the 1 s retry delay
    assert.deepEqual(calls, ['/sa3-synth', '/worker-drain', '/worker-drain']);
    assert.equal(state.settled, false, 'a 404 proves nothing about the render');
    releaseDrain();
    assert.equal(((await p) as Error).name, 'AbortError');
  });
});
