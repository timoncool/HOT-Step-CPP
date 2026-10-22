// What this guards: the /api/v1/daw router wired as index.ts mounts it (JSON parser, router, body-error
// handler) on a real socket. Auth, the DAW error envelope for a malformed body, and that the per-job
// `backendOverride` of POST /api/generate is not forwarded by the DAW routes. Upstream is stubbed.
// Run: npx tsx --test src/services/daw/dawV1.test.ts
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import express from 'express';

process.env.DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'daw-route-'));
const { default: dawRouter, dawBodyErrorHandler } = await import('../../routes/dawV1.js');
const { __setCaller } = await import('./dawFacade.js');

const app = express();
app.use(express.json({ limit: '50mb' }));
app.use('/api/v1/daw', dawRouter, dawBodyErrorHandler);
const server = http.createServer(app);
await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}/api/v1/daw`;
test.after(() => { server.close(); });

const posts: any[] = [];
__setCaller(async (method, route, body) => {
  if (route === '/api/backends') return { status: 200, data: { activeId: 'ace' } };
  if (method === 'POST' && route === '/api/generate') { posts.push(body); return { status: 200, data: { jobId: 'up0' } }; }
  return { status: 404, data: { error: route } };
});

const post = (headers: Record<string, string>, body: string) =>
  fetch(`${base}/generate`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...headers }, body });

test('no key set: a plain loopback call passes, a forwarded remote client does not', async () => {
  delete process.env.HOTSTEP_DAW_KEY;
  assert.equal((await fetch(`${base}/nope`)).status, 404);
  const r = await fetch(`${base}/nope`, { headers: { 'X-Forwarded-For': '192.168.1.50' } });
  assert.equal(r.status, 401);
  assert.equal((await r.json() as any).error.code, 'unauthorized');
});

test('key set: required even from loopback, wrong key refused, right key passes', async () => {
  process.env.HOTSTEP_DAW_KEY = 'secret';
  try {
    assert.equal((await fetch(`${base}/nope`)).status, 401);
    assert.equal((await fetch(`${base}/nope`, { headers: { 'X-DAW-Key': 'wrong' } })).status, 401);
    assert.equal((await fetch(`${base}/nope`, { headers: { 'X-DAW-Key': 'secret' } })).status, 404);
    // Without the key a well-formed generate call never reaches the engine.
    const denied = await post({}, JSON.stringify({ mode: 'text2music', caption: 'x' }));
    assert.equal(denied.status, 401);
    assert.equal(posts.length, 0);
  } finally { delete process.env.HOTSTEP_DAW_KEY; }
});

test('malformed JSON and bad typed fields come back in the DAW envelope', async () => {
  const bad = await post({}, '{"mode": ');
  const body = await bad.json() as any;
  console.log('malformed json:', bad.status, JSON.stringify(body));
  assert.equal(bad.status, 400);
  assert.deepEqual(Object.keys(body).sort(), ['api', 'error']);
  assert.equal(body.api, 1);
  assert.equal(body.error.code, 'bad_request');
  const typed = await post({}, JSON.stringify({ mode: 'text2music', caption: 'x', seed: -4 }));
  assert.equal(typed.status, 400);
  assert.equal(((await typed.json()) as any).error.code, 'bad_request');
  assert.equal(posts.length, 0);
});

test('DAW routes ignore a per-job backendOverride (flat, nested or named): the engine request never carries it', async () => {
  const res = await post({}, JSON.stringify({
    mode: 'text2music', caption: 'x', backendOverride: 'yue2', backend: 'yue2',
    params: { backendOverride: 'yue2' }, input: { backendOverride: 'yue2' },
  }));
  assert.equal(res.status, 202);
  assert.equal(posts.length, 1);
  console.log('forwarded keys:', Object.keys(posts[0]).join(','));
  const text = JSON.stringify(posts[0]);
  assert.ok(!/backendOverride|yue2/.test(text), text);
});

test('an invalid backendOverride is ignored too, not a 400 as on /api/generate', async () => {
  posts.length = 0;
  const res = await post({}, JSON.stringify({ mode: 'text2music', caption: 'x', backendOverride: 42, expected_backend: 'ace' }));
  assert.equal(res.status, 202);
  assert.equal(posts.length, 1);
  assert.equal(posts[0].expectedBackend, 'ace');
  assert.ok(!('backendOverride' in posts[0]));
});
