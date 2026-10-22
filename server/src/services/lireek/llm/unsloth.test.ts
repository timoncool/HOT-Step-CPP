import test from 'node:test';
import assert from 'node:assert/strict';
import { config } from '../../../config.js';
import { UnslothProvider } from './unsloth.js';

// Regression guard for the availability check. The shipped default base URL is
// already loopback (http://127.0.0.1:8888), so "points at localhost" must never
// be treated as "Studio is installed" — or every fresh install would show the
// provider as available with nothing configured.

// Only the data props we mutate are saved/restored (config.lireek also has
// getter-only props like dbPath, which Object.assign cannot write back).
async function withUnsloth(patch: { unslothBaseUrl?: string; unslothUsername?: string; unslothPassword?: string; unslothKeyless?: boolean }, fn: () => void | Promise<void>) {
  const saved = {
    unslothBaseUrl: config.lireek.unslothBaseUrl,
    unslothUsername: config.lireek.unslothUsername,
    unslothPassword: config.lireek.unslothPassword,
    unslothKeyless: config.lireek.unslothKeyless,
  };
  Object.assign(config.lireek, patch);
  try { await fn(); } finally { Object.assign(config.lireek, saved); }
}

test('is unavailable on a fresh install (default loopback URL, no credentials, no keyless flag)', async () => {
  await withUnsloth({ unslothBaseUrl: 'http://127.0.0.1:8888', unslothUsername: '', unslothPassword: '', unslothKeyless: false },
    () => assert.equal(new UnslothProvider().isAvailable(), false));
});

test('requires BOTH username and password, not just one', async () => {
  await withUnsloth({ unslothBaseUrl: 'http://127.0.0.1:8888', unslothUsername: 'admin', unslothPassword: '', unslothKeyless: false },
    () => assert.equal(new UnslothProvider().isAvailable(), false));
  await withUnsloth({ unslothBaseUrl: 'http://127.0.0.1:8888', unslothUsername: '', unslothPassword: 'secret', unslothKeyless: false },
    () => assert.equal(new UnslothProvider().isAvailable(), false));
});

test('is available once both credentials are set (local or remote URL)', async () => {
  await withUnsloth({ unslothBaseUrl: 'http://127.0.0.1:8888', unslothUsername: 'admin', unslothPassword: 'secret', unslothKeyless: false },
    () => assert.equal(new UnslothProvider().isAvailable(), true));
  await withUnsloth({ unslothBaseUrl: 'https://studio.example.com', unslothUsername: 'admin', unslothPassword: 'secret', unslothKeyless: false },
    () => assert.equal(new UnslothProvider().isAvailable(), true));
});

test('keyless flag makes a local Studio usable without credentials', async () => {
  // The real-world setup this exists for: default loopback URL, no
  // username/password, but Studio's opt-in "keyless API access" is on.
  await withUnsloth({ unslothBaseUrl: 'http://127.0.0.1:8888', unslothUsername: '', unslothPassword: '', unslothKeyless: true },
    () => assert.equal(new UnslothProvider().isAvailable(), true));
});

test('keyless authenticate() returns the approved dummy bearer without any network call', async () => {
  await withUnsloth({ unslothBaseUrl: 'http://127.0.0.1:8888', unslothUsername: '', unslothPassword: '', unslothKeyless: true },
    async () => {
      const token = await (new UnslothProvider() as any).authenticate();
      assert.equal(token, 'not-needed'); // in Unsloth's APPROVED_DUMMY_BEARERS whitelist
    });
});

test('real credentials take precedence over the keyless flag', async () => {
  // Both set: the keyless short-circuit must NOT fire — authenticate() goes to
  // the login flow and fails against a dead port instead of returning the dummy.
  await withUnsloth({ unslothBaseUrl: 'http://127.0.0.1:9', unslothUsername: 'admin', unslothPassword: 'secret', unslothKeyless: true },
    async () => {
      await assert.rejects((new UnslothProvider() as any).authenticate());
    });
});
