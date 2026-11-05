# server-feature-dev — reference

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

Deep detail supporting [SKILL.md](SKILL.md). All paths repo-relative; all line numbers verified against `master` at `dda5f6b6` on 2026-10-23.

## 1. aceClient method catalog (`server/src/services/aceClient.ts`)

`const BASE = config.aceServer.url` (aceClient.ts:13), `http://127.0.0.1:8085` by default.

| Method | Engine endpoint | Timeout tier | Returns |
|---|---|---|---|
| `health()` | `GET /health` | QUICK | `{ status }` |
| `props()` | `GET /props` | QUICK | models/adapters/defaults (`AceProps`, :66-81) |
| `plugins()` | `GET /plugins` | QUICK | Lua plugin registry (solvers/schedulers/guidance/postprocess) |
| `warm(req, keepLoaded=true)` | `POST /warm[?keep_loaded=1]` | QUICK | job id; pre-loads DiT+VAE+adapter; no-op under strict eviction |
| `listJobs()` | `GET /jobs` | QUICK | every job in the engine's in-memory table (reconcile after disconnect) |
| `submitLm(req, mode?, keepLoaded?)` | `POST /lm[?keep_loaded=1]` | QUICK | job id; `mode: 'inspire' \| 'format'` sets `lm_mode` |
| `submitSynth(req\|req[], format='wav16', keepLoaded?)` | `POST /synth[?format=...]` | QUICK | job id; JSON body may be an array (batch) |
| `submitSynthMultipart(req, srcAudio?, refAudio?, srcLatents?, refLatents?, format?, keepLoaded?, seedLatents?)` | `POST /synth` multipart | RESULT | job id; **single JSON object only** in the `request` part (:525-529) |
| `submitUnderstand(audioBuffer, params?)` | `POST /understand` multipart | RESULT | job id; `params` becomes the `request` part when present |
| `pollJob(id)` | `GET /job?id=N` | POLL | `AceJobStatus` (:236-254) |
| `getJobResult(id)` | `GET /job?id=N&result=1` | RESULT | raw `Response` (audio body) |
| `getJobLatent(id)` | `GET /job?id=N&latent=1` | RESULT | `Buffer \| null` (soft-fail) |
| `cancelJob(id)` | `POST /job?id=N&cancel=1` | POLL | void |
| `dropSongs(id, mask)` | `POST /job?id=N&drop_songs=MASK` | POLL | void; takes songs out of a running YuE2 batch |
| `isReachable()` | via `health()` | QUICK | boolean (soft-fail) |
| `submitSpectralLifter(wav, params)` | `POST /spectral-lifter?...` | RESULT | processed WAV `Buffer` (synchronous endpoint) |
| `submitPpVaeReencode(wav, blend, outFmt?)` | `POST /pp-vae-reencode?...` | RESULT | processed WAV `Buffer` (synchronous endpoint) |
| `submitSa3Refine(wav, opts)` | `POST /sa3-refine?...` | SA3 | refined WAV `Buffer` (synchronous, via `workerPost`) |
| `sa3Props()` | `GET /sa3/props` | 5 s | `Sa3Props \| null` (soft-fail) |
| `submitSa3Synth(opts)` | `POST /sa3-synth?...` | SA3 | WAV `Buffer` (synchronous, via `workerPost`) |
| `submitSuperSepSeparate(audio, level)` | `POST /supersep/separate?level=N` | RESULT | SuperSep job id |
| `superSepProgress(id)` / `superSepResult(id)` | `GET /supersep/progress`, `/supersep/result` | POLL | progress / stem list |
| `superSepStem(id, index, outFmt?)` | `GET /supersep/serve` | RESULT | stem WAV `Buffer` |
| `superSepRecombine(id, stems, outFmt?)` | `POST /supersep/recombine` | RESULT | mixed WAV `Buffer` |
| `vaeEncode(audio, vaeModel?)` | `POST /vae` multipart, then poll + result | RESULT | raw f32 latent `Buffer` |
| `codesDecode(req, timeoutMs?, isCancelled?)` | `POST /codes-decode`, then poll + result | QUICK submit | decoded audio `Buffer` |
| `restoreEvictPolicy()` | `POST /models/restore-policy` | QUICK | boolean (soft-fail) |
| `listLoadedModels()` | `GET /models/loaded` | QUICK | `{label, mb, in_use}[]` (soft-fail) |
| `unloadLabel(label)` | `POST /models/unload` | QUICK | boolean (soft-fail) |

Timeout tiers (aceClient.ts:18-21): `TIMEOUT_QUICK = 15_000`, `TIMEOUT_POLL = 30_000`, `TIMEOUT_RESULT = 300_000`, `TIMEOUT_SA3 = 1_800_000` ms, via `AbortSignal.timeout` (`withTimeout` combines a caller signal with one, :26-41).

Engine job phases (`AceJobPhase`, aceClient.ts:258-273, mirrors `job_phase_str()` in the C++ `hot-step-server.cpp`): `queued, loading_text_enc, encoding_text, loading_cond_enc, encoding_cond, loading_dit, loading_adapter, adapter_precompute, dit_inference, loading_vae, vae_decode, encoding_output, done, failed, cancelled`. YuE2 jobs report their own phases (`semantic`, `nar`, which `pollUntilDone` reads). `adapter_precompute` is the "~17 s cold-start LoKr precompute" phase: a job sitting there is working, not stuck, and `adapter_progress` moves while it runs.

Multipart part names accepted by the engine: `request` (JSON), `audio` (source WAV), `ref_audio` (timbre reference WAV), `src_latents`, `ref_latents`, `seed_latents` (raw float32), aceClient.ts:525-554. Boundary is hand-rolled: `'----HotStepBoundary' + Date.now()` (:513).

### AceRequest field groups (aceClient.ts:84-233)

- Core: `caption, lyrics, bpm, duration, keyscale, timesignature, vocal_language, seed, lm_seed`, seed breeding `seed_b, seed_blend, seed_blend_end`
- LM: `lm_batch_size, lm_temperature, lm_cfg_scale, lm_cfg_cutoff_ratio, lm_top_p, lm_top_k, lm_min_p, lm_negative_prompt, use_cot_caption, audio_codes`, repetition control `lm_rep_penalty, lm_rep_window, lm_rep_mode, lm_dry_base, lm_dry_min_len`, CFG refinements `lm_cfg_min_p, lm_cfg_logprob, lm_cfg_rescale`
- Synth/sampler: `inference_steps, guidance_scale, shift, dit_sliding_window, infer_method, scheduler, guidance_mode, custom_timesteps, cfg_cutoff_ratio, cache_ratio`
- Cover/repaint: `task_type, audio_cover_strength, cover_noise_strength, cover_noise_method, repainting_start, repainting_end, seed_strength`
- Model routing: `synth_model, lm_model, vae_model, emb_model`, planner-LM LoRA `lm_adapter, lm_adapter_scale`
- Adapters: `adapter, adapter_scale`, multi-adapter `adapters: {name, scale, gain_curve?, gain_domain?}[]` (supersedes single), per-section `adapter_sections`, `adapter_section_align_at`, `adapter_section_isolation` (engine feature reverted; inert), `adapter_group_scales`, `adapter_mode` ("merge"|"runtime"|"runtime_lowrank"), `adapter_runtime_quant`, `adapter_runtime_dora`, `adapter_merge_lowvram`, basin re-base `rebase_source`/`rebase_beta` (**stack-level fields, applied once per adapter stack, never per adapter; do not move them into the `adapters[]` entries**, issue #72)
- Plugins: `plugin_params: Record<string, string|number|boolean>`, `postprocess_plugin`
- Post: `latent_shift, latent_rescale, denoise_*`, `lss_strength, lss_var_thresh, lss_dc_remove`, `pp_vae_reencode`, `get_lrc`, `vae_chunk`

Adding a field: extend the interface here, then map it in `services/generation/translateParams.ts`. The engine ignores unknown fields, so a typo fails silently; verify end-to-end in `gen_*.log` (params are dumped by `logGenerationParams`).

## 2. Engine child-process lifecycle (`server/src/services/aceEngineProcess.ts`)

- `index.ts` calls `startAceServer()` once from its bootstrap block (`index.ts:354-384`), after the portable CUDA runtime download.
- Spawn args built at aceEngineProcess.ts:161-202: `--models`, `--host`, `--port` always; `--adapters` if the dir exists; `--user-plugins` with a release data folder; `--keep-loaded` if `config.aceServer.keepLoaded`; `--noise-profile`, `--draft-lm`, `--vae-chunk`, `--vae-overlap` conditionally.
- The child env is rebuilt on every spawn by `buildGpuEnv()` (:220), so an empty `CUDA_VISIBLE_DEVICES` (the Settings GPU picker's "Auto") is dropped instead of hiding every GPU. TensorRT and TRT-LLM lib dirs are prepended to PATH when present; the Windows `Path`-key gotcha and its fix live at :209-242.
- stdout/stderr piped to console (with GGML noise filtered: `CUDA graph warmup`, `CUDA Graph id`, `ggml_backend_cuda_graph_compute`; these are also dropped at the engine source by `acestep_ggml_log` in engine/src/backend.h, which discards GGML DEBUG-level lines unless `HOTSTEP_GGML_DEBUG=1`; the Node filter is belt-and-braces), plus `logEngine()` → `ace_engine.log` and `pushLog(line, 'engine')` → UI SSE stream (:247-274). The `[Load] <label> backend:` lines also record the engine device (`engineState.ts`, `backendLineFeeder`).
- Crash respawn (:295-339): non-clean exit → restart after 3 s, capped at 3 crashes per 30 s window; on cap, `setEngineReady(false, '...')` and give up. An exit with code 1 after the engine's `ERROR: no models found` line (an empty models folder) is a first run, not a crash (:298-307): no budget is spent, `setEngineNoModels(true)` makes `/api/health` report `engine.noModels`, and the engine waits. `retryIfGaveUp()` (:438) starts it again once a model download finishes (`index.ts:389-395`) or once `watchForModels()` (:44-68), a 5 s check of the models folder, finds a top-level `.gguf` that has stopped growing. Note `setEngineReady(ready, status)` takes **two** args (engineState.ts:20-23).
- Deliberate stop/restart (:366-460): `stopAceServer(reason, timeoutMs, { suspend })` sets `suspended` first so the exit handler never respawns, then tree-kills on Windows (`taskkill /PID <pid> /T /F`, :387). Training jobs call it to borrow the GPU and `restartAceServer()` afterwards; `isEngineSuspended()` makes `/api/generate` answer 503 meanwhile. Restore hooks registered with `onEngineRestarted()` (:131) run after each restart; `index.ts:397-422` registers the MM3 and YuE2 model-selection restores.
- Shutdown (`shutdown()`, index.ts:487-522, on SIGINT/SIGTERM): kill training children first (`killActiveChildren`), `stopAceServer(..., { suspend: false })`, `server.close()`, `closeDb()`, `closeLogger()`, forced `process.exit(0)` after 1 s. `POST /api/shutdown` runs the same steps from routes/shutdown.ts (`stopEverything`). This clean shutdown path is why C++ rebuilds must go through `dev-rebuild.bat`.

## 3. Settings hot-reload internals

- `GET /api/settings/env` returns only whitelisted keys (seeded with resolved defaults) plus `restartKeys` (settings.ts:98-128).
- `POST /api/settings/env` (settings.ts:136-209; header comment says PUT, code wins): filters body to `EXPOSED_ENV_KEYS`, validates `DOWNLOAD_DIR` first, updates matching `.env` lines in place, appends new keys, writes values through `formatEnvValue()`, preserves original EOL style (`\r\n` vs `\n`, :191), writes the file, calls `reloadEnvConfig()`, responds `{ updated, restartRequired }` where `restartRequired = changed.some(k => RESTART_REQUIRED_KEYS.has(k))`.
- `reloadEnvConfig()` (config.ts:494-636) re-parses `.env` with `dotenvParse` and hot-patches the live `config` object via per-key `apply(envKey, setter, getter)` calls (`applyInt` for the labeling throughput numbers), compares resolved values before and after, logs `[Config] Hot-reloaded N setting(s): ...`, runs every `configReloadListeners` callback, and returns the changed keys.
- Restart-required keys (config.ts:483-488): `ACESTEPCPP_MODELS`, `ACESTEPCPP_ADAPTERS`, `ACESTEPCPP_PORT`, `ACESTEPCPP_HOST`, `ACESTEPCPP_VAE_CHUNK`, `ACESTEPCPP_VAE_OVERLAP`, `ACESTEPCPP_KEEP_LOADED`, `CUDA_VISIBLE_DEVICES`, `SERVER_PORT`, `DATA_DIR`, `MUSCRIPTOR_MODELS_DIR`. Their values still hot-patch the config object (where an `apply()` line exists), but the running engine child was spawned with the old args. `MUSCRIPTOR_MODELS_DIR` has no `apply()` line; `services/muscriptor.ts` reads it from `process.env` at load.

## 4. DB idioms (copy-paste ready)

Checked column-add (the preferred idiom; database.ts:365-430):

```ts
const songsMigrations: Array<{ check: string; alter: string }> = [
  {
    check: `SELECT COUNT(*) as c FROM pragma_table_info('songs') WHERE name='my_col'`,
    alter: `ALTER TABLE songs ADD COLUMN my_col TEXT DEFAULT ''`,
  },
];
for (const m of songsMigrations) {
  const row = db.prepare(m.check).get() as any;
  if (row.c === 0) {
    db.exec(m.alter);
    console.log(`[DB] Migration: ${m.alter}`);
  }
}
```

Auth-scoped list with dynamic filter (songs.ts:35-58):

```ts
router.get('/', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }
  const source = req.query.source as string | undefined;
  let query = 'SELECT * FROM songs WHERE user_id = ?';
  const params: any[] = [userId];
  if (source) {
    query += ` AND json_extract(generation_params, '$.source') = ?`;
    params.push(source);
  }
  query += ' ORDER BY created_at DESC';
  const songs = getDb().prepare(query).all(...params);
  const parsed = songs.map((s: any) => ({
    ...s,
    tags: JSON.parse(s.tags || '[]'),
    is_public: !!s.is_public,
  }));
  res.json({ songs: parsed });
});
```

Insert with UUID PK and JSON-in-TEXT (services/backends/ace/generate.ts:1247-1259): `uuidv4()` id, `JSON.stringify(trackParams)` into `generation_params`, plus `backend`.

Service-owned table (services/generation/jobLedger.ts:35-47): a class whose constructor takes the `Database`, runs `CREATE TABLE IF NOT EXISTS`, and fixes up rows a previous process left behind; the caller constructs it lazily (`ledgerInstance ??= new GenerationJobLedger(getDb())`, generate.ts:68).

Bulk cross-DB migration reference (`migrateLireekData`, database.ts:532-625): `ATTACH DATABASE ... AS lireek_old`, copy common columns via `INSERT OR IGNORE ... SELECT` (columns intersected via `PRAGMA table_info`), `DETACH`, rename the old file to `.migrated`; `foreign_keys = OFF` during, restored in `finally`.

## 5. Logger and SSE log stream

`services/logger.ts`:
- `initLogger()` (called at index.ts:74 **before any console output**) creates `<logs>/<YYYY-MM-DD_HH-MM-SS>/` (`LOGS_DIR`, config.ts:22) with `node_console.log` + `ace_engine.log` streams and a `generations/` dir, then monkey-patches `process.stdout.write`/`process.stderr.write` to mirror everything into `node_console.log` (logger.ts:45-83).
- `logEngine(line)` appends to `ace_engine.log` (logger.ts:89-94).
- `logTraining(jobId, kind, line)` writes `training/<kind>_<jobId>.log` live; `closeTrainingLog(jobId)` closes it (logger.ts:96-123).
- Per-generation: `startGenerationLog(jobId, taskType)` seeds an in-memory buffer; `logGeneration(jobId, level, msg)` appends (levels `INFO|DEBUG|WARNING|ERROR`) and also pushes `[Gen:<8-char-id>] msg` to the SSE stream; `logGenerationParams(jobId, obj)` pretty-prints JSON line-by-line; `finishGenerationLog(jobId, taskType)` / `failGenerationLog(jobId, error, taskType)` write `gen_<jobId>_<taskType>.log` and drop the buffer (logger.ts:125-212). Buffer is lost if the process dies before finish/fail.

`routes/logs.ts`:
- Ring buffer of `MAX_LINES = 2000` `LogLine` objects `{ id, ts, text, source: 'engine'|'server' }` (:15-25).
- `pushLog(text, source)`: suppresses engine noise patterns (`ENGINE_NOISE`, :28-32), appends, notifies subscribers (:35-47).
- `subscribeLines(cb)` returns an unsubscribe function (:50-53). **Global pub/sub, no job tagging**: the reason GPU work is serialized. A backend's `generate()` subscribes during a run to parse engine progress lines into `job.stage`/`job.progress`.
- `GET /api/logs`: SSE; replays backlog (optionally `?after=<id>`), streams `data: {json}\n\n` events, `: keepalive` ping every 15 s, cleans up on close (:57-100). The UI also uses this stream as its presence beacon (`countPresence`, routes/health.ts).
- `GET /api/logs/vram`: proxies the engine's `GET /vram` with a 3 s timeout, soft-failing to zeros (:105-121). `GET /models-loaded` (:125) and `POST /models-unload` (:138) proxy the engine's model residency routes.

## 6. Generation orchestration details beyond the SKILL summary

- LM-cache hit path (services/backends/ace/generate.ts:96-118): reconstructs each synth request as `{ ...aceReq, audio_codes, caption, lyrics, bpm, duration, keyscale, timesignature, lm_seed }` from the cached LM outputs, the concrete embodiment of the "LM echo sideband" rule. Cache entries are LRU-refreshed on hit; the key is the LM seed (the DiT seed when the two are tied) plus the LM-affecting params (`computeLmCacheKey`, services/generation/lmCache.ts).
- Skip-LM defaults (ace/generate.ts:81-89): when LM is skipped for a non-cover task, missing metadata is defaulted (bpm 120, duration 120 s, `C major`, timesig `4`) because `/lm` cannot be called for metadata only (it always generates audio codes).
- Retry policy (routes/generate.ts:198-263): attempts come from `job.envelope.policy.retry` (`maxAttempts: 2, reseedOnRetry: true` on every backend); on a retryable failure the job is reset to `pending`, the seed randomized (`Math.floor(Math.random() * 2_147_483_647)`) because a bad same-seed LM output may have caused the stall, 2 s pause, retry. Terminal failure calls `failGenerationLog`.
- Duration (ace/generate.ts:1214-1225): every track's duration is measured from the output WAV via `wavDurationSec`, falling back to the LM's figure; cover/repaint tasks skip the LM so the LM figure is 0, and Song Builder places its seams from this value.
- Multipart-vs-JSON decision (ace/generate.ts:815-827): multipart iff any of source audio, timbre-ref audio, source latents, ref latents, or seed latents buffers are present.
- Status values and their UI meaning: `pending` (queued), `running` (YuE2 and post-processing re-runs), `lm_running`, `synth_running`, `saving`, then terminal `succeeded|failed|cancelled`. Progress is 0 to 100, set by log-subscription callbacks and phase milestones.

## 7. Server directory census (verified 2026-10-23)

- `server/src/routes/`: adapters, analyze, assistant, audio, audioQueue, auth, backends, compare, coverArt, dawV1, download, exportImport, generate, health, inspire, lireek.ts (+ `lireek/` subfolder: crudRoutes, llmRoutes, mm3Routes), listening, logs, mastering, midiStudio, modelManager, models, plugins, preferences, profiles, resolve, seeds, settings (+ settingsDataFolder), shutdown, songBuilder, songs, stemStudio, streamSessions, studioDrafts, supersep, training, updates, upload, vst (+ vstControl helpers), workers, workflows, yue2Cover.
- `server/src/services/` files: aceClient, aceEngineProcess, audioConvert, audioCrop, audioMetadata, autoTrim, coverDatasetSource, coverStemCache, disco-analyzer, downloadSave, gpuDevices, languageCodes, latentFormat, logger, lyricsReconcile, midiLyrics, midiParser, modelDownloadService, muscriptor, pathMapper, runtimeFiles, sa3Tokenizer, spectralLifter, whisperTranscribe, yue2Cover, yue2Takes.
- `server/src/services/` subfolders: adapters, assets, audio, audioQueue, backends (ace, minimax, sa3, yue2, plus registry.ts and types.ts), compare, coverArt, daw, exportImport, generation (incl. `resolve/`), library, lireek (incl. `llm/`), preferences, songBuilder, storm, streamSessions, studioDrafts, training, updates, workflows.
- Other `server/src/` folders: `contracts/` (wire shapes shared with the UI), `data/` (runtime data shipped with the app), `dataroot/` (data folder resolution), `db/`.
- Mount-path ↔ filename mismatches: `/api/builder` → `songBuilder.ts`, `/api/v1/daw` → `dawV1.ts`, `/api` → `backends.ts`, `/api/training/worker` → `workers.ts` (`workerRouter`), `/listening` → `listening.ts`.

## 8. Known non-obvious states

- **Draft-LM speculative decoding: DISABLED** (config.ts:232-237). GGML per-call overhead (~10 ms) makes sequential 0.6B forwards nearly as costly as the 4B target. Re-enable via `ACESTEPCPP_DRAFT_LM`; the auto-detect block is commented out at config.ts:274-287. The spawn still passes `--draft-lm` if the config value is set and the file exists (aceEngineProcess.ts:190-194).
- **`lireekDb.ts` shims**: `initLireekDb`/`closeLireekDb`/`getLireekDb` are deprecated no-ops/aliases (lireekDb.ts:14-25) kept for compile compat. Do not create new separate-DB patterns.
- **`adapter_section_isolation`** (aceClient.ts:174-176): field exists; the engine parses it but never reads it, because the feature was reverted (commit ee041e1, "broke musical continuity"). Treat as inert unless the engine re-lands it.
- **Warm-on-startup** is gated on `warmOnStartup && keepLoaded && warmDit`; with keep-loaded off (the default) it never fires (index.ts:429-433, config.ts:247-263).
- **`uncaughtException`/`unhandledRejection` log-and-continue** (index.ts:526-531); no global Express error middleware exists, so per-handler try/catch is mandatory for clean JSON errors.

## 9. Unverified notes (from the original research dossier)

- Engine route registration was checked against `hot-step-server.cpp` on 2026-10-23, but handler semantics still come mostly from aceClient.ts's own comments (which state they mirror `job_phase_str()` / `request_parse_json`). Verify against engine source before changing wire behavior.
- `docs/plans/` (gitignored, local-only) may contain design docs that supersede details here; it was not inspected.
- The backend `generate.ts` files are long (ACE ~1390 lines, MM3 and YuE2 longer); the orchestration skeleton and all cited line numbers were verified directly, the rest was sampled.
