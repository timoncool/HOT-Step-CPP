import test from 'node:test';
import assert from 'node:assert/strict';
import { estimate, etaKey, getFigures, recordRun, resetEta, warmEstimate, EMA_ALPHA } from './dawEta.js';
import { __test, jobEtaS } from './dawFacade.js';

test('YuE2 batch-4 warm estimate is about 125 s', () => {
  resetEta();
  const e = estimate('yue2', 4, 30, 'yue2');
  console.log('yue2 b4 warm estimate s:', e);
  assert.ok(Math.abs(e - 125) <= 15, `got ${e}`);
});

test('formula: base + per_take*(takes-1) + per_audio*duration, plus cold load', () => {
  resetEta();
  const f = getFigures('ace_base');
  const warm = estimate('ace_base', 4, 30, 'ace_base');
  const cold = estimate('ace_base', 4, 30, 'sa3');
  console.log('ace_base b4 30s warm/cold s:', warm, cold);
  assert.ok(Math.abs(warm - (f.base_s + 3 * f.per_take_s + 30 * f.per_audio_s)) < 1e-9);
  assert.ok(warm > 15 && warm < 25, `bench says about 20.4, got ${warm}`);
  assert.equal(cold - warm, f.cold_load_s);
  assert.ok(f.cold_load_s >= 10);
  // SA3 to YuE2 is a cold load
  assert.ok(estimate('yue2', 1, 30, 'sa3') - estimate('yue2', 1, 30, 'yue2') >= 10);
});

test('EMA moves the estimate 30% toward the observed wall time', () => {
  resetEta();
  const before = estimate('ace_turbo', 1, 30, 'ace_turbo');
  recordRun('ace_turbo', 1, 30, 14.1, false);
  const after = estimate('ace_turbo', 1, 30, 'ace_turbo');
  const moved = (after - before) / (14.1 - before);
  console.log('ema before/after/fraction:', before, after, moved);
  assert.equal(EMA_ALPHA, 0.3);
  assert.ok(Math.abs(moved - 0.3) < 1e-6);
  assert.ok(after - before > 2.5, 'moved by a real amount');
  assert.equal(getFigures('ace_turbo').source, 'measured');
  assert.equal(getFigures('ace_base').source, 'default');
});

test('cold run subtracts the cold load before the EMA', () => {
  resetEta();
  const f = getFigures('yue2');
  const before = warmEstimate(f, 4, 30);
  recordRun('yue2', 4, 30, before + f.cold_load_s, true);
  assert.ok(Math.abs(warmEstimate(getFigures('yue2'), 4, 30) - before) < 1e-9);
});

test('etaKey maps backend ids and DiT names', () => {
  assert.equal(etaKey('sa3'), 'sa3');
  assert.equal(etaKey('yue2'), 'yue2');
  assert.equal(etaKey('ace', 'acestep-v15-base-Q8.gguf'), 'ace_base');
  assert.equal(etaKey('ace', 'acestep-v15-turbo-Q8.gguf'), 'ace_turbo');
});

test('job eta_s is estimate minus elapsed, floor 0', () => {
  const now = Date.now();
  __test.addJob({ id: 'eta-job', upstreamId: 'e1', seeds: [], createdAt: now - 40_000, eta: { key: 'yue2', takes: 4, durationS: 30, estS: 125, cold: false } } as any);
  __test.addJob({ id: 'eta-late', upstreamId: 'e2', seeds: [], createdAt: now - 400_000, eta: { key: 'yue2', takes: 4, durationS: 30, estS: 125, cold: false } } as any);
  const a = jobEtaS('eta-job', now), b = jobEtaS('eta-late', now);
  console.log('eta_s 40s in / 400s in:', a, b);
  assert.equal(a, 85);
  assert.equal(b, 0);
});

test('4 SA3 takes at 30 s estimate about 4 full renders (about 25 s)', () => {
  resetEta();
  const one = estimate('sa3', 1, 30, 'sa3');
  const four = estimate('sa3', 4, 30, 'sa3');
  console.log('sa3 30 s: 1 take', one, '4 takes', four, 'ratio', four / one);
  assert.ok(Math.abs(four - 25) <= 4, `got ${four}`);
  assert.ok(Math.abs(four / one - 4) < 0.05);
});
