import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildReport, redact, MAX_URL_LENGTH } from './reportIssue';

const names = { home: 'C:\\Users\\jdoe', user: 'jdoe' };

test('redact masks home dir in both slash styles, user name, other profiles', () => {
  const out = redact('open C:\\Users\\jdoe\\Music\\a.wav and C:/Users/jdoe/x and /home/other/y by jdoe', names);
  assert.ok(!/jdoe|other/.test(out), out);
  assert.match(out, /~\\Music\\a\.wav/);
});

test('redact masks IPs, emails, tokens', () => {
  const out = redact([
    'listening on 192.168.1.20:8085 from ::ffff:10.0.0.1 and fe80:0:0:0:1:2:3:4',
    'mail me@example.com',
    'Authorization: Bearer abcdef1234567890abcdef',
    'HF_TOKEN=hf_abcdefghijklmnop1234567',
    'ghp_abcdefghijklmnopqrstuvwx',
    'sha 0123456789abcdef0123456789abcdef01234567',
  ].join('\n'), names);
  assert.ok(!/192\.168|10\.0\.0\.1|fe80|example\.com|abcdef1234567890|hf_abc|ghp_abc|0123456789abcdef0123456789abcdef01234567/.test(out), out);
  assert.match(out, /listening on <ip>:8085/);
});

test('redact leaves ordinary log text alone', () => {
  for (const line of ['[engine] loaded acestep-v15-turbo in 3.2 s on CUDA0', 'step 12:30:45 done', 'a::b namespace std::vector']) {
    assert.equal(redact(line, names), line);
  }
});

test('buildReport keeps the URL under the cap by trimming the oldest log lines first', () => {
  const lines = (tag: string) => Array.from({ length: 400 }, (_, i) => `${tag} line ${i} ` + 'x'.repeat(60));
  const parts = { title: 'Crash on render', description: 'It broke', system: 'OS: Windows', sections: [{ name: 'Server log', lines: lines('s') }, { name: 'Engine log', lines: lines('e') }] };
  const r = buildReport(parts);
  assert.ok(r.url.length <= MAX_URL_LENGTH, String(r.url.length));
  assert.ok(r.trimmed);
  assert.match(r.body, /s line 399/, 'newest line survives');
  assert.ok(!r.body.includes('s line 0 '), 'oldest line dropped');
  assert.match(r.url, /template=in-app-report\.md/);
});

test('buildReport leaves a short report untouched', () => {
  const r = buildReport({ title: 't', description: 'd', system: 'OS: x', sections: [{ name: 'Server log', lines: ['a'] }] });
  assert.equal(r.trimmed, false);
  assert.match(r.body, /## Description\n\nd/);
});

test('buildReport cuts the description only after the logs are gone', () => {
  const r = buildReport({ title: 't', description: 'é'.repeat(5000), system: 'OS: x', sections: [{ name: 'Server log', lines: ['a', 'b'] }] });
  assert.ok(r.url.length <= MAX_URL_LENGTH, String(r.url.length));
  assert.ok(!r.body.includes('## Logs'));
});

test('redact masks prefixed credential keys, short usernames and ::1', () => {
  const out = redact('WORKER_TOKEN=sample-private-value\nHF_TOKEN=sample-private-value\n"api_key": "sample-private-value"\nmax_tokens=512\nbind ::1 port 80\nby ab', { home: 'C:\\Users\\ab', user: 'ab' });
  assert.ok(!/sample-private-value|::1|\bab\b/.test(out), out);
  assert.match(out, /max_tokens=512/);
});

test('buildReport enforces the cap when system info or title alone is oversized', () => {
  const big = '漢'.repeat(900);
  const a = buildReport({ title: 't', description: '', system: big, sections: [] });
  assert.ok(a.url.length <= MAX_URL_LENGTH, String(a.url.length));
  assert.ok(a.trimmed);
  const b = buildReport({ title: big, description: 'd', system: 'OS: x', sections: [] });
  assert.ok(b.url.length <= MAX_URL_LENGTH, String(b.url.length));
  assert.ok(b.title.length < big.length);
});

test('redact masks credential values of any length or shape', () => {
  const out = redact('WORKER_TOKEN=12345678\npassword=abcde\n"api_key": "1234567890"\nsecret=1\nnum_tokens=64', names);
  assert.ok(!/12345678|abcde|1234567890|secret=1\b/.test(out), out);
  assert.match(out, /num_tokens=64/);
});

import { modelLines } from './reportIssue';

test('modelLines follows the engine: ACE-Step models for ace, catalogue for YuE2', () => {
  const ace = { ditModel: 'C:\\m\\acestep-dit.gguf', lmModel: 'lm.gguf', vaeModel: 'vae.gguf' };
  const cat = { buckets: { lm: ['q8_0'], vae: ['standard'] }, meta: { lm: { q8_0: { label: 'yue2-lm-q8_0.gguf' } } }, defaults: { lm: 'q8_0', vae: 'standard', lmAdapterAr: 'D:\\a\\x_ar.safetensors', lmAdapterNar: '' } };
  assert.deepEqual(modelLines('ace', ace, cat, '?').map(l => l[0]), ['DiT model', 'LM model', 'VAE model']);
  assert.deepEqual(modelLines('yue2', ace, cat, '?'), [['lm model', 'yue2-lm-q8_0.gguf'], ['vae model', 'standard'], ['AR LoRA', 'x_ar.safetensors']]);
  assert.deepEqual(modelLines('yue2', ace, undefined, '?'), []);
});
