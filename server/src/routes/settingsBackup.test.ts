import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const directory = fs.mkdtempSync(path.resolve('.test-settings-backup-'));
process.env.DATA_DIR = directory;
const express = (await import('express')).default;
const database = await import('../db/database.js');
const auth = (await import('./auth.js')).default;
const backupModule = await import('./settingsBackup.js');
const backup = backupModule.default;
const backupParser = backupModule.backupBodyParser;
const service = await import('../services/settingsBackup.js');
const rows = await import('../db/lireekDb.js');

const appSettings = {
  coResident: false, cacheLmCodes: true, speculativeDecoding: true, downloadFormat: 'flac',
  downloadMp3Bitrate: 192, downloadOpusBitrate: 192, downloadVersion: 'mastered', downloadIncludeLatent: false,
  triggerUseFilename: false, triggerPlacement: 'prepend', discoKickExtract: false,
  parallelWhisper: false, parallelQualityEval: false, parallelCoverArt: false, generationTimeoutMinutes: 30,
};

test('settings backup validates, previews and atomically restores only the exported data', async t => {
  database.initDb();
  const db = database.getDb();
  const app = express();
  app.use(express.json());
  app.use('/auth', auth);
  app.use('/backup', backup);
  const server = app.listen(0, '127.0.0.1');
  await new Promise<void>(resolve => server.once('listening', resolve));
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  const base = `http://127.0.0.1:${address.port}`;
  try {
    const { token } = await (await fetch(`${base}/auth/auto`)).json() as { token: string };
    const post = async (action: string, body: unknown, authenticated = true) => {
      const response = await fetch(`${base}/backup/${action}`, { method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(authenticated ? { Authorization: `Bearer ${token}` } : {}) },
        body: JSON.stringify(body) });
      return { status: response.status, body: await response.json() };
    };
    const artist = rows.getOrCreateArtist('Backup fixture');
    const collection = rows.saveLyricsSet(artist.id, 'Collection', 1, [{ title: 'Track', lyrics: 'Original lyrics' }]);
    const profile = rows.saveProfile(collection.id, 'fixture', 'model', { themes: ['night'] });
    const generation = rows.saveGeneration({ profileId: profile.id, provider: 'fixture', model: 'model', lyrics: 'Original lyrics' });
    db.prepare('INSERT INTO audio_generations (generation_id, hotstep_job_id, audio_url) VALUES (?, ?, ?)')
      .run(generation.id, 'fixture-job', '/audio/fixture.wav');
    rows.upsertPreset(collection.id, { adapterPath: 'fixture.safetensors' });
    const lyricPreferences = { prompt_profile: 'Profile instructions', prompt_lyrics: 'Lyric instructions', instagen_system_prompt: 'Inspire instructions' };
    for (const [key, value] of Object.entries(lyricPreferences)) rows.setSetting(key, value);
    const localPreferences = {
      active_backend_id: 'yue2', yue2_lm_adapter_ar: 'local/ar.safetensors',
      yue2_lm_adapter_nar: 'local/nar.safetensors', yue2_lm_adapter_folder: 'local/adapters',
      sa3_dit_model: 'local-dit', active_models_v1: '{"local":"model"}',
      training_defaults_v1: '{"trainLm":{"steps":100}}',
      promptXunrelated: 'Literal underscore required', Prompt_custom: 'Case-sensitive prefix required',
    };
    for (const [key, value] of Object.entries(localPreferences)) rows.setSetting(key, value);
    const usersBefore = db.prepare('SELECT * FROM users').all();
    db.prepare("INSERT INTO songs (id, user_id, title, audio_url) SELECT 'backup-library-sentinel', id, 'Library fixture', '/audio/fixture.wav' FROM users LIMIT 1").run();
    const songsBefore = db.prepare('SELECT * FROM songs').all();
    const exported = await post('export', { app: appSettings });
    assert.equal(exported.status, 200, JSON.stringify(exported.body));
    const file = JSON.parse(JSON.stringify(exported.body));

    await t.test('export includes only Lyric Studio prompt preferences', () => {
      assert.deepEqual(file.settings.lyricStudio, lyricPreferences);
    });
    await t.test('round trip includes the actual migrated schema and leaves users alone', async () => {
      rows.setSetting('prompt_profile', 'changed');
      rows.setSetting('prompt_new_local', 'Remove this prompt on replacement');
      rows.setSetting('active_backend_id', 'ace');
      localPreferences.active_backend_id = 'ace';
      rows.getOrCreateArtist('Replacement fixture');
      const result = await post('import', { backup: file, confirmed: true });
      assert.equal(result.status, 200, JSON.stringify(result.body));
      assert.deepEqual(result.body.app, appSettings);
      const again = (await post('export', { app: result.body.app })).body;
      assert.deepEqual(again.settings, file.settings);
      assert.deepEqual(again.lyricStudio, file.lyricStudio);
      assert.deepEqual(db.prepare('SELECT * FROM users').all(), usersBefore);
      assert.deepEqual(db.prepare('SELECT * FROM songs').all(), songsBefore);
      for (const [key, value] of Object.entries(localPreferences)) assert.equal(rows.getSetting(key), value, key);
      assert.equal(rows.getSetting('prompt_profile'), lyricPreferences.prompt_profile);
      assert.equal(rows.getSetting('instagen_system_prompt'), lyricPreferences.instagen_system_prompt);
      assert.equal(rows.getSetting('prompt_new_local'), '');
    });
    await t.test('export falls back to defaults for missing or stale stored settings, and the result re-imports', async () => {
      const { downloadVersion, ...older } = appSettings;
      for (const app of [{ ...appSettings, downloadVersion: '' }, { ...appSettings, downloadVersion: 'legacy', triggerPlacement: 7 }, older, null]) {
        const res = await post('export', { app });
        assert.equal(res.status, 200, JSON.stringify(res.body));
        assert.equal(res.body.settings.app.downloadVersion, 'mastered');
        assert.equal((await post('preview', { backup: res.body })).status, 200);
      }
      assert.equal((await post('export', { app: { ...appSettings, downloadVersion: 'both' } })).body.settings.app.downloadVersion, 'both');
    });
    await t.test('preview reports replacement counts without writing', async () => {
      rows.getOrCreateArtist('Current extra fixture');
      const result = await post('preview', { backup: file });
      assert.equal(result.status, 200);
      assert.equal(result.body.current.artists, 2);
      assert.equal(result.body.incoming.artists, 1);
      assert.equal(result.body.current.settings, Object.keys(lyricPreferences).length);
      assert.equal(result.body.incoming.settings, Object.keys(lyricPreferences).length);
      assert.equal((db.prepare('SELECT COUNT(*) AS n FROM artists').get() as { n: number }).n, 2);
    });
    await t.test('missing confirmation cannot overwrite', async () => {
      assert.equal((await post('import', { backup: file })).status, 400);
    });
    await t.test('preview and import reject settings outside the Lyric Studio allowlist without writing', async () => {
      for (const key of [...Object.keys(localPreferences), 'unknown_key']) {
        const bad = structuredClone(file);
        bad.settings.lyricStudio[key] = 'foreign setting';
        for (const action of ['preview', 'import']) {
          const before = db.prepare('SELECT * FROM settings ORDER BY key').all();
          const result = await post(action, { backup: bad, confirmed: true });
          assert.equal(result.status, 400, `${action}: ${key}`);
          assert.match(result.body.error, /settings\.lyricStudio/);
          assert.deepEqual(db.prepare('SELECT * FROM settings ORDER BY key').all(), before);
        }
      }
    });
    await t.test('unknown versions and wrong shapes give a clear 400', async () => {
      const version = await post('preview', { backup: { ...file, version: 99 } });
      assert.equal(version.status, 400);
      assert.match(version.body.error, /Unsupported backup version 99/);
      for (const bad of [null, [], {}, { ...file, settings: [] }, { ...file, lyricStudio: {} },
        { ...file, settings: { ...file.settings, app: { ...appSettings, cacheLmCodes: 'yes' } } }]) {
        assert.equal((await post('import', { backup: bad, confirmed: true })).status, 400);
      }
    });
    await t.test('broken relationships and duplicate IDs are rejected before replacement', async () => {
      const bad = structuredClone(file);
      bad.lyricStudio.profiles[0].lyrics_set_id = 999;
      assert.equal((await post('import', { backup: bad, confirmed: true })).status, 400);
      const duplicate = structuredClone(file);
      duplicate.lyricStudio.artists.push(duplicate.lyricStudio.artists[0]);
      assert.equal((await post('preview', { backup: duplicate })).status, 400);
      const badJson = structuredClone(file);
      badJson.lyricStudio.profiles[0].profile_data = 'invalid json';
      assert.equal((await post('preview', { backup: badJson })).status, 400);
    });
    await t.test('constraint failure rolls back all deletions and settings changes', () => {
      const bad = structuredClone(file);
      bad.lyricStudio.artists.push({ ...bad.lyricStudio.artists[0], id: 999 });
      const before = service.exportSettingsBackup(db, appSettings);
      assert.throws(() => service.importSettingsBackup(db, bad), /UNIQUE/);
      const after = service.exportSettingsBackup(db, appSettings);
      assert.deepEqual(after.settings, before.settings);
      assert.deepEqual(after.lyricStudio, before.lyricStudio);
    });
    await t.test('all operations require the current local auth session', async () => {
      for (const action of ['export', 'preview', 'import']) {
        assert.equal((await post(action, { app: appSettings, backup: file, confirmed: true }, false)).status, 401);
      }
    });
    await t.test('a later parent generation restores with deferred foreign keys', () => {
      const data = structuredClone(file);
      data.lyricStudio.generations.push({ ...data.lyricStudio.generations[0], id: 999 });
      data.lyricStudio.generations[0].parent_generation_id = 999;
      service.importSettingsBackup(db, data);
      assert.equal((db.prepare('SELECT parent_generation_id FROM generations WHERE id = ?').get(generation.id) as { parent_generation_id: number }).parent_generation_id, 999);
      assert.deepEqual(db.pragma('foreign_key_check'), []);
    });
    await t.test('a backup past the global 50 MB body limit still previews', async () => {
      const big = structuredClone(file);
      big.lyricStudio.generations[0].lyrics = 'x'.repeat(60 * 1024 * 1024);
      const limited = express();
      limited.use(backupParser);
      limited.use(express.json({ limit: '50mb' }));
      limited.use('/backup', backup);
      const srv = limited.listen(0, '127.0.0.1');
      await new Promise<void>(resolve => srv.once('listening', resolve));
      const addr = srv.address();
      assert.ok(addr && typeof addr !== 'string');
      try {
        const response = await fetch(`http://127.0.0.1:${addr.port}/backup/preview`, { method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }, body: JSON.stringify({ backup: big }) });
        assert.equal(response.status, 200);
      } finally { await new Promise<void>(resolve => srv.close(() => resolve())); }
    });
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()));
    database.closeDb();
    // This test creates this exact directory beneath its working directory.
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
