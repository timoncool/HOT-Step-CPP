import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { parseBuildInfo } from './config.js';
// @ts-expect-error packaging helper is plain JavaScript
import { buildInfoErrors } from '../../release/build-info.mjs';
// @ts-expect-error packaging checker is plain JavaScript
import { checkPackageBuildInfo } from '../../release/check-layout.mjs';

const SHA = 'a'.repeat(40);
const GOOD = { format: 1, tag: 'v1.2.3', sha: SHA, channel: 'release', os: 'windows', arch: 'x64', variant: 'cuda12-volta', engineSha: SHA, updaterProtocol: 0 };
const BAD: Array<[string, object]> = [
  ['empty sha', { sha: '' }],
  ['short sha', { sha: 'abc123' }],
  ['missing engineSha', { engineSha: undefined }],
  ['unknown variant', { variant: 'banana' }],
  ['variant from another os', { variant: 'metal' }],
  ['unknown arch', { arch: 'arm64' }],
  ['missing channel', { channel: undefined }],
  ['invalid channel', { channel: 'nightly' }],
  ['release channel without tag', { tag: null }],
  ['master channel with tag', { channel: 'master' }],
  ['tag without v', { tag: '1.2.3' }],
  ['future protocol', { updaterProtocol: 99 }],
  ['future format', { format: 2 }],
];

test('runtime and package check both accept a contract build-info', () => {
  assert.deepEqual(parseBuildInfo(JSON.stringify(GOOD)), GOOD);
  assert.deepEqual(buildInfoErrors(GOOD), []);
  const master = { ...GOOD, tag: null, channel: 'master' };
  assert.deepEqual(parseBuildInfo(JSON.stringify(master)), master);
});

for (const [label, patch] of BAD) {
  test(`build-info ${label}: unknown at runtime, rejected by the package check`, () => {
    const info = { ...GOOD, ...patch };
    assert.equal(parseBuildInfo(JSON.stringify(info)), null);
    assert.notDeepEqual(buildInfoErrors(info), []);
  });
}

test('runtime treats unparseable build-info as unknown', () => {
  for (const text of ['', 'null', '[]', '{']) assert.equal(parseBuildInfo(text), null);
});

test('package check: good fixture passes, protocol 99 and wrong asset fail', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hs-build-info-'));
  try {
    fs.mkdirSync(path.join(root, 'engine'));
    fs.writeFileSync(path.join(root, 'engine', 'ggml-cuda.dll'), '');
    fs.writeFileSync(path.join(root, 'VERSION'), '1.2.3');
    const write = (info: object) => fs.writeFileSync(path.join(root, 'build-info.json'), JSON.stringify(info));
    const expect = { os: 'windows', variant: 'cuda12-volta' };
    assert.deepEqual(checkPackageBuildInfo(root, { os: 'windows', variant: 'cuda12-volta' }), ['build-info.json: missing']);
    write(GOOD);
    assert.deepEqual(checkPackageBuildInfo(root, expect), []);
    write({ ...GOOD, updaterProtocol: 99 });
    assert.notDeepEqual(checkPackageBuildInfo(root, expect), []);
    write(GOOD);
    assert.notDeepEqual(checkPackageBuildInfo(root, { os: 'windows', variant: 'cuda13.1' }), []);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
