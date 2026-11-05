#!/usr/bin/env node
// Run against an idle dev stack with a dedicated test dataset. Keeps all audio.
// node tools/sa3-training-check.mjs --dataset <id> --out <new directory>
// node tools/sa3-training-check.mjs --self-test
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

function audioStats(b) {
  assert.equal(b.toString('ascii', 0, 4), 'RIFF', 'Expected WAV');
  assert.equal(b.toString('ascii', 8, 12), 'WAVE');
  let format, data;
  for (let p = 12; p + 8 <= b.length;) {
    const size = b.readUInt32LE(p + 4), end = p + 8 + size;
    assert.ok(end <= b.length, 'Truncated WAV');
    const tag = b.toString('ascii', p, p + 4);
    if (tag === 'fmt ') format = b.subarray(p + 8, end);
    if (tag === 'data') data = b.subarray(p + 8, end);
    p = end + (size % 2);
  }
  assert.ok(format?.length >= 16 && data?.length, 'Missing WAV chunks');
  const codec = format.readUInt16LE(0), channels = format.readUInt16LE(2);
  const rate = format.readUInt32LE(4), bits = format.readUInt16LE(14);
  assert.ok(channels > 0 && rate > 0);
  assert.ok((codec === 1 && bits === 16) || (codec === 3 && bits === 32), 'Expected PCM16 or float32 WAV');
  const width = bits / 8;
  assert.equal(data.length % (width * channels), 0);
  let sum = 0, peak = 0;
  for (let i = 0; i < data.length; i += width) {
    const x = codec === 1 ? data.readInt16LE(i) / 32768 : data.readFloatLE(i);
    assert.ok(Number.isFinite(x), 'Non-finite audio');
    sum += x * x; peak = Math.max(peak, Math.abs(x));
  }
  const rms = Math.sqrt(sum / (data.length / width));
  assert.ok(rms > 1e-5, `Silent audio: rms=${rms}`);
  return { channels, rate, seconds: data.length / width / channels / rate, rms, peak };
}

function selfTest() {
  const wav = Buffer.alloc(52);
  wav.write('RIFF'); wav.writeUInt32LE(44, 4); wav.write('WAVEfmt ', 8);
  wav.writeUInt32LE(16, 16); wav.writeUInt16LE(3, 20); wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(44100, 24); wav.writeUInt32LE(176400, 28);
  wav.writeUInt16LE(4, 32); wav.writeUInt16LE(32, 34);
  wav.write('data', 36); wav.writeUInt32LE(8, 40);
  wav.writeFloatLE(0.1, 44); wav.writeFloatLE(-0.1, 48);
  assert.ok(audioStats(wav).rms > 0.09);
  const bad = Buffer.from(wav); bad.writeFloatLE(NaN, 44);
  assert.throws(() => audioStats(bad), /Non-finite/);
  bad.fill(0, 44); assert.throws(() => audioStats(bad), /Silent/);
  assert.throws(() => audioStats(wav.subarray(0, 50)), /Truncated/);
  console.log('SA3 CHECK SELF-TEST: PASS (valid, NaN, silence, truncation)');
}

async function main() {
  const args = process.argv.slice(2);
  if (args.includes('--self-test')) return selfTest();
  const opt = k => args[args.indexOf(k) + 1];
  const dataset = args.includes('--dataset') ? opt('--dataset') : '';
  assert.match(dataset, /^[a-f0-9-]{36}$/i, '--dataset is required');
  assert.ok(args.includes('--out'), '--out is required');
  const out = path.resolve(opt('--out'));
  fs.mkdirSync(out, { recursive: false });
  const base = 'http://localhost:3000';
  const receipt = { head: spawnSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).stdout.trim(), dataset, jobs: [], renders: [] };
  const save = () => fs.writeFileSync(path.join(out, 'receipt.json'), JSON.stringify(receipt, null, 2));
  let token, current;
  async function request(endpoint, body, method = body === undefined ? 'GET' : 'POST') {
    const r = await fetch(base + endpoint, { method, headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: AbortSignal.timeout(120000) });
    if (!r.ok) throw new Error(`${method} ${endpoint}: ${r.status} ${await r.text()}`);
    return r.json();
  }
  async function waitJob(endpoint, id, success) {
    const deadline = Date.now() + 30 * 60_000;
    let last;
    while (Date.now() < deadline) {
      const j = await request(endpoint + id);
      const state = `${j.status}/${j.phase ?? ''}/${j.done ?? ''}`;
      if (state !== last) console.log(`${id}: ${state}`);
      last = state;
      if (['done', 'succeeded', 'failed', 'cancelled'].includes(j.status)) {
        receipt.jobs.push(j); save(); current = undefined;
        assert.equal(j.status, success, j.error || 'Job did not succeed');
        return j;
      }
      await delay(2000);
    }
    throw new Error(`Timed out waiting for ${id}`);
  }
  try {
    token = (await request('/api/auth/auto')).token;
    const queue = await request('/api/generate/queue');
    assert.ok(!queue.running && !queue.pending && !queue.depth, 'Generation queue is busy');
    const jobs = await request('/api/training/jobs');
    assert.ok(!jobs.jobs.some(j => ['running', 'queued'].includes(j.status)), 'Training queue is busy');
    const endpoint = `/api/training/datasets/${dataset}/sa3-train`;
    const ready = await request(endpoint);
    assert.ok(ready.available, `Missing prerequisites: ${ready.missing}`);
    const adapterName = `proof-${Date.now()}`;
    receipt.recipe = { steps: 3, rank: 16, alpha: 16, learningRate: 0.0001, duration: 12, seed: 42, checkpointEvery: 1, exportEvery: 0, adapterName };
    save();
    const start = await request(endpoint, receipt.recipe);
    current = { kind: 'training', id: start.jobId };
    await waitJob('/api/training/jobs/', start.jobId, 'done');
    let run = (await request(endpoint)).runs.find(r => r.adapterName === adapterName);
    assert.ok(run?.resumable && run.published, 'Missing checkpoint or published export');
    const resumed = await request(endpoint, { resumeRun: run.run, steps: 5 });
    current = { kind: 'training', id: resumed.jobId };
    await waitJob('/api/training/jobs/', resumed.jobId, 'done');
    run = (await request(endpoint)).runs.find(r => r.adapterName === adapterName);
    receipt.run = run; save();
    const picker = await request('/api/models/stablestep/adapters');
    assert.ok(picker.adapters.some(a => a.name === run.published), 'Export missing from picker');
    for (const caption of ['soft piano and brushed drums, spacious instrumental jazz', 'warm analogue pad, slow instrumental ambient']) {
      const pair = [];
      for (const scale of [0, 1]) {
        const generation = await request('/api/generate', { backend: 'sa3', backendOverride: 'sa3', expectedBackend: 'sa3', caption, lyrics: '', instrumental: true,
          duration: 12, seed: 42, randomSeed: false, batchSize: 1, postProcessingEnabled: false,
          samplerPluginsEnabled: false, sa3Steps: 8, sa3Sampler: 'md_pingpong_blend',
          sa3Adapters: scale === 0 ? [] : [{ name: run.published, scale }], sa3LufsEnabled: false });
        current = { kind: 'generate', id: generation.jobId };
        const result = await waitJob('/api/generate/status/', generation.jobId, 'succeeded');
        assert.ok(result.result?.audioUrls?.length === 1, 'Expected one audio result');
        const r = await fetch(new URL(result.result.audioUrls[0], base), { headers: { Authorization: `Bearer ${token}` }, signal: AbortSignal.timeout(120000) });
        assert.ok(r.ok, 'Audio download failed');
        const bytes = Buffer.from(await r.arrayBuffer());
        const file = `${receipt.renders.length}-strength-${scale}.wav`;
        fs.writeFileSync(path.join(out, file), bytes);
        receipt.renders.push({ caption, scale, seed: 42, file, ...audioStats(bytes) });
        pair.push(bytes); save();
      }
      assert.ok(!pair[0].equals(pair[1]), 'Strength 0 and 1 are identical');
    }
    receipt.status = 'PASS'; save(); console.log('SA3 TRAINING CHECK: PASS');
  } catch (error) {
    if (current) {
      try {
        if (current.kind === 'training') await request(`/api/training/jobs/${current.id}`, undefined, 'DELETE');
        else await request(`/api/generate/cancel/${current.id}`, {});
      } catch (cleanup) { receipt.cleanupError = String(cleanup); }
    }
    receipt.status = 'FAIL'; receipt.error = String(error); save(); throw error;
  }
}
main().catch(e => { console.error(e.message); process.exitCode = 1; });
