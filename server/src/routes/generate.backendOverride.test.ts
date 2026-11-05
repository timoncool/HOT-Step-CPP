// What this guards: the per-job `backendOverride` through the real route entry
// point (submitGeneration), not just buildEnvelope. Backends are stubbed so no
// engine or network is touched: `capabilities` and `generate` record calls.
// Run: npx tsx --test src/routes/generate.backendOverride.test.ts
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'route-backend-override-'));
process.env.DATA_DIR = DATA_DIR;
const database = await import('../db/database.js');
database.initDb();
const { setEngineReady } = await import('../engineState.js');
const registry = await import('../services/backends/registry.js');
const { submitGeneration, getGenerationJob, default: generateRouter } = await import('./generate.js');
const { default: authRouter } = await import('./auth.js');
const { trackBackendRelease, awaitBackendRelease, _resetBackendReleases } = await import('../services/generation/backendRelease.js');

// Engine transport: every engine call goes through `engineFetch`, which fails
// unless a test installs a handler, so no test can reach a live engine.
type EngineHandler = (url: string, init?: RequestInit) => Response | Promise<Response>;
let engineFetch: EngineHandler | undefined;
const realFetch = globalThis.fetch;
globalThis.fetch = (async (input: any, init?: RequestInit) => {
  const url = String(input instanceof Request ? input.url : input);
  if (!/^https?:\/\/(127\.0\.0\.1|localhost):8085\//.test(url)) return realFetch(input, init);
  if (!engineFetch) throw new Error(`test: no engine (${url})`);
  return engineFetch(url, init);
}) as typeof fetch;
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

const { gpuLaneBusy } = await import('../services/generation/gpuLane.js');

setEngineReady(true, 'test');
const capabilityCalls: string[] = [];
const generateCalls: string[] = [];
// Ordered record of generate starts and confirmed unloads, for the residency cases.
const events: string[] = [];
// Per-test hooks: a held generate or unload, or a backend switch mid-lookup.
const hold: Record<string, Promise<void> | undefined> = {};
const holdRelease: Record<string, Promise<void> | undefined> = {};
let duringCapabilities: ((id: string) => void) | undefined;
for (const id of ['ace', 'yue2']) {
  const backend = registry.getBackend(id) as any;
  assert.ok(backend, `backend ${id} must be registered`);
  const realCaps = backend.capabilities.bind(backend);
  backend.capabilities = async () => {
    capabilityCalls.push(id);
    duringCapabilities?.(id);
    try { return await realCaps(); } catch { return { extensions: [] }; }
  };
  // 'Cancelled' ends the job without the 2 s retry pauses.
  backend.generate = async () => {
    generateCalls.push(id); events.push(`generate:${id}`);
    await hold[id];
    throw new Error('Cancelled: stub backend, no engine in tests');
  };
}
// Unloads are recorded stubs; `realRelease` keeps ACE's own implementation for
// the cases that drive it through the mocked transport above.
const realRelease: Record<string, () => Promise<void>> = {};
const useRealRelease = new Set<string>();
for (const backend of registry.listBackends() as any[]) {
  if (!backend.releaseVram) continue;
  realRelease[backend.id] = backend.releaseVram.bind(backend);
  backend.releaseVram = async () => {
    if (useRealRelease.has(backend.id)) await realRelease[backend.id]();
    await holdRelease[backend.id];
    events.push(`released:${backend.id}`);
  };
}
// Only the two families these override cases switch between. Other
// registered backends also release residency before either family loads.
const aceYue2 = () => events.filter(e => /:(ace|yue2)$/.test(e));

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>(r => { resolve = r; });
  return { promise, resolve };
}
async function until(cond: () => boolean, what: string): Promise<void> {
  for (let i = 0; i < 500 && !cond(); i++) await new Promise(r => setTimeout(r, 10));
  assert.ok(cond(), `timed out waiting for ${what}`);
}
const laneIdle = () => until(() => !gpuLaneBusy(), 'the GPU lane to go idle');

const logs: string[] = [];
const realLog = console.log;
console.log = (...args: unknown[]) => { logs.push(args.map(String).join(' ')); };
test.after(() => { console.log = realLog; database.closeDb(); fs.rmSync(DATA_DIR, { recursive: true, force: true }); });

const flat = { caption: 'test', lyrics: '[Instrumental]', duration: 30 };
const intent = (extra: Record<string, unknown> = {}) => ({ contract: 'generation-intent/1', params: { caption: 'test', lyrics: '[Instrumental]', duration: 30 }, ...extra });

async function submit(raw: unknown) {
  const result = await submitGeneration(raw, 'user-1');
  realLog(`  submit -> ${JSON.stringify(result.ok ? { ok: true, backendId: getGenerationJob(result.jobId)?.envelope.backendId } : { ok: false, status: result.status, error: (result.body as any).error })}`);
  return result;
}
function queuedBackend(result: Awaited<ReturnType<typeof submitGeneration>>): string | undefined {
  assert.equal(result.ok, true, `expected a queued job, got ${JSON.stringify(result)}`);
  return result.ok ? getGenerationJob(result.jobId)?.envelope.backendId : undefined;
}

test('1. flat body with backendOverride queues on that non-active backend', async () => {
  assert.equal(registry.setActiveBackendId('ace'), true);
  const r = await submit({ ...flat, backendOverride: 'yue2' });
  assert.equal(queuedBackend(r), 'yue2');
  assert.equal(registry.getActiveBackendId(), 'ace', 'active backend must not change');
});

test('2. intent body with backendOverride resolves capabilities for the override backend', async () => {
  assert.equal(registry.setActiveBackendId('ace'), true);
  capabilityCalls.length = 0;
  const r = await submit(intent({ backendOverride: 'yue2' }));
  realLog(`  capabilities asked of: ${JSON.stringify(capabilityCalls)}`);
  assert.deepEqual(capabilityCalls, ['yue2']);
  assert.equal(queuedBackend(r), 'yue2');
});

test('3. unknown backendOverride returns 400 on the flat and intent paths', async () => {
  for (const raw of [{ ...flat, backendOverride: 'not-a-backend' }, intent({ backendOverride: 'not-a-backend' })]) {
    const r = await submit(raw);
    assert.equal(r.ok, false);
    if (!r.ok) { assert.equal(r.status, 400); assert.match(String((r.body as any).error), /Unknown backendOverride 'not-a-backend'/); }
  }
});

test('4. empty or non-string backendOverride returns 400', async () => {
  for (const bad of ['', 42, true, {}, [], null]) {
    for (const raw of [{ ...flat, backendOverride: bad }, intent({ backendOverride: bad })]) {
      const r = await submit(raw);
      assert.equal(r.ok, false, `backendOverride=${JSON.stringify(bad)} must be rejected`);
      if (!r.ok) { assert.equal(r.status, 400); assert.match(String((r.body as any).error), /backendOverride/); }
    }
  }
});

test('5. backend alone runs on the active backend and logs the mismatch', async () => {
  assert.equal(registry.setActiveBackendId('ace'), true);
  logs.length = 0;
  const r = await submit({ ...flat, backend: 'yue2' });
  assert.equal(queuedBackend(r), 'ace');
  const job = r.ok ? getGenerationJob(r.jobId) : undefined;
  assert.equal(job?.envelope.submittedBackendMismatch, 'yue2');
  assert.ok(logs.some(l => l.includes(`submitted for 'yue2' but runs on 'ace'`)), `mismatch log missing: ${JSON.stringify(logs)}`);
});

test('6. backend and backendOverride together: the override wins', async () => {
  assert.equal(registry.setActiveBackendId('ace'), true);
  const r = await submit({ ...flat, backend: 'ace', backendOverride: 'yue2' });
  assert.equal(queuedBackend(r), 'yue2');
});

test('7. the job is generated by the override backend, and legacy backend by the active one', async () => {
  await laneIdle();
  assert.equal(registry.setActiveBackendId('ace'), true);
  generateCalls.length = 0;
  queuedBackend(await submit({ ...flat, backendOverride: 'yue2' }));
  await laneIdle();
  assert.deepEqual(generateCalls, ['yue2']);
  generateCalls.length = 0;
  queuedBackend(await submit({ ...flat, backend: 'yue2' }));
  await laneIdle();
  assert.deepEqual(generateCalls, ['ace']);
});

test('8. a nested backendOverride never selects the backend; the top-level one wins', async () => {
  assert.equal(registry.setActiveBackendId('ace'), true);
  for (const nested of [{ input: { backendOverride: 'yue2' } }, { params: { ...intent().params, backendOverride: 'yue2' } }]) {
    capabilityCalls.length = 0;
    const r = await submit(intent(nested));
    assert.equal(queuedBackend(r), 'ace');
    assert.deepEqual(capabilityCalls, ['ace']);
    assert.equal(r.ok && getGenerationJob(r.jobId)?.params.backendOverride, undefined, 'the nested copy must be stripped');
  }
  for (const [top, inner] of [['ace', 'yue2'], ['yue2', 'ace']]) {
    capabilityCalls.length = 0;
    const r = await submit(intent({ backendOverride: top, input: { backendOverride: inner } }));
    assert.equal(queuedBackend(r), top);
    assert.deepEqual(capabilityCalls, [top]);
    assert.equal(r.ok && getGenerationJob(r.jobId)?.params.backendOverride, top);
  }
});

test('9. inherited property names are unknown backends (400), not 500s', async () => {
  for (const name of ['toString', 'constructor', '__proto__', 'hasOwnProperty']) {
    assert.equal(registry.getBackend(name), undefined);
    assert.equal(registry.setActiveBackendId(name), false);
    for (const raw of [{ ...flat, backendOverride: name }, intent({ backendOverride: name })]) {
      const r = await submit(raw);
      assert.equal(r.ok, false, name);
      if (!r.ok) { assert.equal(r.status, 400, name); assert.match(String((r.body as any).error), /Unknown backendOverride/); }
    }
  }
});

test('10. expectedBackend is checked against the override when there is one, else the active backend', async () => {
  assert.equal(registry.setActiveBackendId('ace'), true);
  const cases: Array<[string, unknown, number | string]> = [
    ['flat override=expected', { ...flat, backendOverride: 'yue2', expectedBackend: 'yue2' }, 'yue2'],
    ['flat override!=expected', { ...flat, backendOverride: 'yue2', expectedBackend: 'ace' }, 409],
    ['intent override=expected', intent({ backendOverride: 'yue2', input: { expectedBackend: 'yue2' } }), 'yue2'],
    ['intent override!=expected', intent({ backendOverride: 'yue2', input: { expectedBackend: 'ace' } }), 409],
    ['flat no override, expected=active', { ...flat, expectedBackend: 'ace' }, 'ace'],
    ['flat no override, expected!=active', { ...flat, expectedBackend: 'yue2' }, 409],
    ['intent no override, expected!=active', intent({ input: { expectedBackend: 'yue2' } }), 409],
    ['unknown override wins over expectedBackend', { ...flat, backendOverride: 'nope', expectedBackend: 'ace' }, 400],
    ['flat malformed override wins over expectedBackend', { ...flat, backendOverride: 42, expectedBackend: 'yue2' }, 400],
    ['intent malformed override wins over expectedBackend', intent({ backendOverride: 42, input: { expectedBackend: 'yue2' } }), 400],
  ];
  for (const [name, raw, want] of cases) {
    const r = await submit(raw);
    if (typeof want === 'string') { assert.equal(queuedBackend(r), want, name); continue; }
    assert.equal(r.ok, false, name);
    if (!r.ok) assert.equal(r.status, want, name);
    if (!r.ok && want === 409) {
      const body = r.body as any;
      assert.equal(body.activeBackend, 'ace', name);
      if (name.includes('override!=')) assert.equal(body.backendOverride, 'yue2', name);
    }
  }
});

test('11. an override to a backend that cannot do the requested operation is a 400', async () => {
  assert.equal(registry.setActiveBackendId('ace'), true);
  for (const raw of [{ ...flat, taskType: 'repaint', backendOverride: 'yue2' }, intent({ backendOverride: 'yue2', input: { taskType: 'repaint' } })]) {
    const r = await submit(raw);
    assert.equal(r.ok, false);
    if (!r.ok) { assert.equal(r.status, 400); assert.match(String((r.body as any).error), /does not support generation operation 'repaint'/); }
  }
  // Legacy `backend` is still log-only: the repaint stays on the active ACE.
  assert.equal(queuedBackend(await submit({ ...flat, taskType: 'repaint', backend: 'yue2' })), 'ace');
});

for (const [first, second] of [['yue2', 'ace'], ['ace', 'yue2']]) {
  test(`12. queued ${first} then ${second} override: ${first} is unloaded and confirmed before ${second} loads`, async () => {
    await laneIdle();
    // The selector stays on `first` throughout: only the overrides change family.
    assert.equal(registry.setActiveBackendId(first), true);
    const firstHeld = deferred();
    const releaseHeld = deferred();
    hold[first] = firstHeld.promise;
    holdRelease[first] = releaseHeld.promise;
    try {
      events.length = 0;
      queuedBackend(await submit({ ...flat, backendOverride: first }));
      await until(() => events.at(-1) === `generate:${first}`, `${first} to start`);
      events.length = 0;
      queuedBackend(await submit({ ...flat, backendOverride: second }));
      firstHeld.resolve();
      // The unload is scheduled but unconfirmed: the next family must not load.
      await new Promise(r => setTimeout(r, 50));
      assert.equal(aceYue2().join(), '', `${second} started before ${first}'s unload was confirmed`);
      releaseHeld.resolve();
      await until(() => events.includes(`generate:${second}`), `${second} to start`);
      assert.deepEqual(aceYue2(), [`released:${first}`, `generate:${second}`]);
      assert.equal(registry.getActiveBackendId(), first, 'the selector must not change');
    } finally {
      firstHeld.resolve(); releaseHeld.resolve();
      hold[first] = undefined; holdRelease[first] = undefined;
      await laneIdle();
    }
  });
}

test('13. every job evicts the other families, even between two jobs of the same family', async () => {
  // Something other than a generation (a score preview under keep-loaded) can
  // leave YuE2 resident between two ACE jobs, so no job trusts what last ran.
  await laneIdle();
  assert.equal(registry.setActiveBackendId('ace'), true);
  queuedBackend(await submit({ ...flat, backendOverride: 'ace' }));
  await laneIdle();
  events.length = 0;
  queuedBackend(await submit({ ...flat, backendOverride: 'ace' }));
  queuedBackend(await submit({ ...flat, backendOverride: 'ace' }));
  await laneIdle();
  assert.deepEqual(aceYue2(), ['released:yue2', 'generate:ace', 'released:yue2', 'generate:ace']);
  assert.equal(events.filter(e => e === 'released:minimax-m3').length, 2);
});

test('14. active backend switched during capability lookup: an override keeps its backend, no override is a 409', async () => {
  try {
    duringCapabilities = () => { registry.setActiveBackendId('yue2'); };
    assert.equal(registry.setActiveBackendId('ace'), true);
    assert.equal(queuedBackend(await submit(intent({ backendOverride: 'ace' }))), 'ace');
    assert.equal(registry.setActiveBackendId('ace'), true);
    const r = await submit(intent());
    assert.equal(r.ok, false);
    if (!r.ok) { assert.equal(r.status, 409); assert.match(String((r.body as any).error), /Active backend changed/); }
  } finally {
    duringCapabilities = undefined;
    registry.setActiveBackendId('ace');
    await laneIdle();
  }
});

/** Call a router's handler directly with a minimal req/res. */
async function callRoute(router: any, method: string, routePath: string, body: unknown, headers: Record<string, string> = {}) {
  const layer = router.stack.find((l: any) => l.route?.path === routePath && l.route.methods[method]);
  assert.ok(layer, `route ${method.toUpperCase()} ${routePath} not found`);
  const out: { status: number; body?: any } = { status: 200 };
  const res: any = {
    writableFinished: false,
    status(code: number) { out.status = code; return res; },
    json(value: unknown) { out.body = value; res.writableFinished = true; return res; },
    on() { return res; },
  };
  await layer.route.stack.at(-1).handle({ body, headers, query: {}, params: {} }, res, () => {});
  return out;
}

test('15. a YuE2 score preview evicts the other families before it loads, and the next ACE job evicts YuE2', async () => {
  await laneIdle();
  const { body: auth } = await callRoute(authRouter, 'get', '/auto', undefined);
  assert.equal(registry.setActiveBackendId('yue2'), true);
  try {
    events.length = 0;
    // An empty caption fails inside the preview, after the lane work has begun.
    const r = await callRoute(generateRouter, 'post', '/yue2/plan', { caption: '', lyrics: '' }, { authorization: `Bearer ${auth.token}` });
    realLog(`  preview -> ${r.status} ${JSON.stringify(r.body)}`);
    assert.equal(r.status, 500);
    assert.deepEqual(aceYue2(), ['released:ace']);
    events.length = 0;
    queuedBackend(await submit({ ...flat, backendOverride: 'ace' }));
    await laneIdle();
    assert.deepEqual(aceYue2(), ['released:yue2', 'generate:ace']);
  } finally {
    registry.setActiveBackendId('ace');
  }
});

test('16. ACE release rejects unless the engine confirms nothing is left resident', async () => {
  await laneIdle();
  const unloads: string[] = [];
  const probe = (answers: Array<() => Response>) => {
    let n = 0;
    engineFetch = (url, init) => {
      if (url.endsWith('/models/loaded')) return answers[Math.min(n++, answers.length - 1)]();
      if (url.endsWith('/models/unload')) {
        const label = JSON.parse(String(init?.body)).label;
        unloads.push(label);
        return json({ unloaded: label !== 'stuck', label });
      }
      throw new Error(`unexpected engine call ${url}`);
    };
  };
  const loaded = (...labels: Array<string | [string, boolean]>) => () => json({
    loaded: labels.map(l => Array.isArray(l) ? { label: l[0], in_use: l[1] } : { label: l, in_use: false }),
  });
  const cases: Array<[string, () => void, RegExp | null]> = [
    ['probe 503', () => probe([() => json({ error: 'busy' }, 503)]), /HTTP 503/],
    ['engine unreachable', () => { engineFetch = () => { throw new TypeError('fetch failed'); }; }, /fetch failed/],
    ['unload did not take', () => probe([loaded('dit', 'stuck'), loaded('stuck')]), /still resident after unload: stuck/],
    ['module in use', () => probe([loaded('dit', ['vae', true]), loaded(['vae', true])]), /still resident after unload: vae \(in use\)/],
    ['all freed', () => probe([loaded('dit', 'vae'), loaded()]), null],
    ['nothing loaded', () => probe([loaded()]), null],
  ];
  try {
    for (const [name, setup, want] of cases) {
      setup();
      unloads.length = 0;
      _resetBackendReleases();
      trackBackendRelease('ace', realRelease.ace).catch(() => {});
      if (want) await assert.rejects(awaitBackendRelease(), want, name);
      else await awaitBackendRelease();
      if (name === 'all freed') assert.deepEqual(unloads, ['dit', 'vae']);
      if (name === 'module in use') assert.deepEqual(unloads, ['dit'], 'an in-use module is not unloaded');
    }

    // Through the lane: a YuE2 job never starts while ACE cannot confirm.
    useRealRelease.add('ace');
    probe([() => json({ error: 'busy' }, 503)]);
    events.length = 0;
    const r = await submit({ ...flat, backendOverride: 'yue2' });
    queuedBackend(r);
    await laneIdle();
    assert.equal(events.includes('generate:yue2'), false, 'YuE2 loaded beside unconfirmed ACE weights');
    const job = r.ok ? getGenerationJob(r.jobId) : undefined;
    assert.equal(job?.status, 'failed');
    assert.match(String(job?.error), /previous backend \(ace\) has not confirmed/);
  } finally {
    useRealRelease.delete('ace');
    engineFetch = undefined;
    _resetBackendReleases();
  }
});
