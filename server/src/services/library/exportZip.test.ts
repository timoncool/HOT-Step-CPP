import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { PassThrough } from 'node:stream';
import { planZipEntries, parseZipIds, streamZip, MAX_ZIP_IDS, MAX_ZIP_BYTES } from './exportZip.js';

function tmp(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'export-zip-'));
}

function audioDirWith(files: Record<string, number>): string {
  const dir = tmp();
  for (const [name, size] of Object.entries(files)) fs.writeFileSync(path.join(dir, name), Buffer.alloc(size, 1));
  return dir;
}

test('names stay unique case-folded, including an explicit "(2)" title', () => {
  const dir = audioDirWith({ "a.wav": 1, "b.wav": 1, "c.wav": 1, "d.wav": 1, "e.wav": 1 });
  const { entries } = planZipEntries([
    { id: '1', title: 'Track', audio_url: '/audio/a.wav' },
    { id: '2', title: 'Track', audio_url: '/audio/b.wav' },
    { id: '3', title: 'track', audio_url: '/audio/c.wav' },
    { id: '4', title: 'Track (2)', audio_url: '/audio/d.wav' },
    { id: '5', title: 'TRACK', audio_url: '/audio/e.wav' },
  ], dir);
  const names = entries.map(e => e.name);
  assert.equal(new Set(names.map(n => n.toLowerCase())).size, 5);
  assert.deepEqual(names.slice(0, 2), ['Track.wav', 'Track (2).wav']);
});

test('parseZipIds accepts nonempty strings only', () => {
  assert.deepEqual(parseZipIds(['a', 'b']), ['a', 'b']);
  for (const bad of [undefined, 'x', [], [''], [1], [null], [{}], ['a', 2], Array(MAX_ZIP_IDS + 1).fill('a')]) {
    assert.equal(parseZipIds(bad), null);
  }
});

test('totalBytes sums the source files and the cap is 256 MiB', () => {
  const dir = audioDirWith({ "a.wav": 100, "b.wav": 50 });
  const plan = planZipEntries([
    { id: '1', audio_url: '/audio/a.wav' },
    { id: '2', audio_url: '/audio/b.wav' },
  ], dir);
  assert.equal(plan.totalBytes, 150);
  assert.equal(MAX_ZIP_BYTES, 256 * 1024 * 1024);
});

test('traversal, directories and missing files are skipped', () => {
  const dir = audioDirWith({ "ok.wav": 1 });
  fs.mkdirSync(path.join(dir, 'sub.wav'));
  const { entries } = planZipEntries([
    { id: '1', audio_url: '/audio/../../etc/passwd' },
    { id: '2', audio_url: '/audio/sub.wav' },
    { id: '3', audio_url: '/audio/missing.wav' },
    { id: '4', audio_url: '/audio/ok.wav?x=1' },
  ], dir);
  assert.deepEqual(entries.map(e => path.basename(e.file)), ['ok.wav']);
});

test('a symlink pointing outside the audio dir is not exported', (t) => {
  const dir = audioDirWith({});
  const outside = path.join(tmp(), 'secret.wav');
  fs.writeFileSync(outside, 'secret');
  try { fs.symlinkSync(outside, path.join(dir, 'link.wav')); } catch { t.skip('symlinks not permitted'); return; }
  const { entries } = planZipEntries([{ id: '1', audio_url: '/audio/link.wav' }], dir);
  assert.equal(entries.length, 0);
});

test('mid-transfer disconnect closes the open source file', async () => {
  const dir = tmp();
  fs.writeFileSync(path.join(dir, 'big.wav'), crypto.randomBytes(8 * 1024 * 1024));
  const { entries } = planZipEntries([{ id: '1', audio_url: '/audio/big.wav' }], dir);
  const res = new PassThrough({ highWaterMark: 1 });
  let src: fs.ReadStream | undefined;
  const archive = streamZip(res, entries, () => {}, f => (src = fs.createReadStream(f)));
  await new Promise<void>(r => src!.once('data', () => r()));
  const closed = new Promise<void>(r => src!.once('close', () => r()));
  res.destroy();
  await closed;
  assert.equal(src!.destroyed, true);
  assert.equal((archive as any)._state.aborted, true);
});

test('a full zip still writes every entry', async () => {
  const dir = audioDirWith({ 'a.wav': 10, 'b.wav': 10 });
  const { entries } = planZipEntries([
    { id: '1', title: 'x', audio_url: '/audio/a.wav' },
    { id: '2', title: 'y', audio_url: '/audio/b.wav' },
  ], dir);
  const res = new PassThrough();
  const chunks: Buffer[] = [];
  res.on('data', c => chunks.push(c));
  streamZip(res, entries, () => {});
  await new Promise(r => res.once('end', r));
  const zip = Buffer.concat(chunks).toString('latin1');
  assert.ok(zip.includes('x.wav') && zip.includes('y.wav'));
});

test('a source file that vanishes after planning fails the download cleanly', async () => {
  const dir = audioDirWith({ 'a.wav': 10 });
  const { entries } = planZipEntries([{ id: '1', audio_url: '/audio/a.wav' }], dir);
  fs.unlinkSync(entries[0].file);
  const res = new PassThrough();
  res.on('error', () => {});
  const errors: Error[] = [];
  streamZip(res, entries, e => errors.push(e));
  await new Promise(r => res.once('close', r));
  assert.equal(errors.length >= 1, true);
  assert.equal((errors[0] as NodeJS.ErrnoException).code, 'ENOENT');
});
