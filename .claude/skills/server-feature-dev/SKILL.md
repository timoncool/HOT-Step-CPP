---
name: server-feature-dev
description: Guides adding or modifying a feature in the HOT-Step CPP Node/TypeScript server (Express route + service pattern, better-sqlite3 schema changes, engine calls via aceClient, logging, tsx watch dev loop). Use when creating/editing files under server/src/, adding an /api/* endpoint, changing the SQLite schema, adding a config/env setting, wiring a server feature to the C++ engine, or debugging server-side failures (stalled/stuck generation jobs, engine HTTP timeouts, SQLite errors, engine crash-respawn loops).
---

# Node Server Feature Development

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

The Node server (`server/src/`) is the middle tier of HOT-Step CPP, a local AI music-generation app. It serves the React UI, owns the SQLite database, and orchestrates the C++ inference engine (`ace-server.exe`, spawned as a child process). This skill covers how to add or change a server feature the way this codebase does it.

**Glossary (no prior context assumed):**
- **Engine / ace-server**: the C++ binary doing actual AI inference, HTTP API on port **8085**. The Node server spawns it (`services/aceEngineProcess.ts`) and talks to it over HTTP.
- **DiT**: Diffusion Transformer, the engine's main music-synthesis model. **LM**: the language model that generates "audio codes" before synthesis. **VAE**: decodes latents to audio. ACE-Step pipeline: LM → DiT → VAE.
- **Backend**: one generation model family behind the shared registry (`services/backends/registry.ts`): `ace` (ACE-Step 1.5, default), `minimax-m3`, `yue2`, `sa3`. Each has `index.ts` (capabilities, envelope) and `generate.ts` under `services/backends/<id>/`.
- **AceRequest**: the canonical JSON request shape the engine's ACE-Step routes accept, defined in `server/src/services/aceClient.ts:84-233`.
- **aceReq**: the AceRequest instance built for a generation job (from UI params via `translateParams`).
- **Adapter**: a LoRA/LoKr fine-tune applied to the DiT at generation time.

```
UI (/api/* via axios/fetch) → Express :3001 (dev: Vite on :3000 proxies /api to it) → routes/*.ts → services/*.ts
                                        ├→ getDb()  (better-sqlite3, synchronous)
                                        └→ aceClient (HTTP → ace-server :8085)
```

## When to use this skill

- Adding a new `/api/*` endpoint or a whole feature (route + service).
- Modifying SQLite schema or queries (`server/src/db/database.ts`, or a service that owns its tables).
- Adding a config value or a Settings-UI-exposed `.env` key (`server/src/config.ts`).
- Making the server call the engine (new AceRequest field, new engine endpoint).
- Debugging server-side failures (jobs stalling, engine timeouts, DB errors).

## Golden rules (hard constraints — each prevents expensive damage)

1. **Type-check with `npx tsc --noEmit` during dev; only `npm run build` before user testing.** (Institutional rule, verbatim.) `tsx watch` runs TypeScript directly, so a build is wasted work mid-dev, but tsx also tolerates errors that `tsc` rejects, so type-check before committing. `server/tsconfig.json` only covers `server/src/`; a touched file under `server/scripts/` needs `npx tsc --noEmit -p scripts/tsconfig.json` as well.
   ```powershell
   cd d:\Ace-Step-Latest\hot-step-cpp\server; npx tsc --noEmit
   ```
2. **Node 20 to 24 LTS, 24 recommended.** `server/package.json:6-8` enforces `>=20.0.0 <25.0.0`. better-sqlite3 is a native module; `postinstall` runs `npm rebuild better-sqlite3` (package.json:15).
3. **ESM everywhere: relative imports MUST end in `.js` even though sources are `.ts`.** `server/package.json:9` sets `"type": "module"`. `import { config } from '../config.js'`; omitting `.js` may run under tsx but fails `tsc`. There is no `__dirname`; the pattern is `const __dirname = path.dirname(fileURLToPath(import.meta.url))` (config.ts:10, index.ts:71).
4. **Never block a request on long work.** The engine has one worker; generations take minutes. Return a job id immediately and let the UI poll (see the worked example), or register a workflow kind for multi-step studio work. Generations and other GPU work run one at a time on the shared GPU lane (`services/generation/gpuLane.ts`; the reason is the comment at generate.ts:185-190).
5. **Never call `getDb()` at module import time.** `initDb()` runs at index.ts:85; a query executed during module load throws `Database not initialized. Call initDb() first.` (database.ts:17-22). Query only inside handlers/functions, or construct a store lazily on first use.
6. **Declare static routes before `/:id` params.** Express matches in order: `GET /recent` after `GET /:id` returns a 404 for id "recent". See the explicit warnings at songs.ts:63 and :75, and seeds.ts (`/favorites`, `/random` at lines 137/149 precede `/:name` at 162).
7. **Git: all work on `master`, stage explicit paths (never `git add -A`, never `git add -f` on gitignored paths: `/data/`, `logs/`, `docs/plans/`, `.agents/` and `node_modules/` are gitignored on purpose), commit locally often, push only with explicit user approval.** Pushing any `v*` tag triggers a full multi-platform CI release build.
8. **PowerShell syntax: `;` not `&&`** for command chaining on this machine.
9. **Don't visually verify the UI with a browser agent**; ask the human user, who provides screenshots. (Hitting API endpoints programmatically is fine.)
10. **If your change touches C++ (`engine/src/`), rebuild via `dev-rebuild.bat` at repo root, never `engine/build.cmd` directly** (you cannot reliably tell whether the app is running; the Node server auto-respawns ace-server on crash, `services/aceEngineProcess.ts:295-339` → infinite respawn + file-lock loop). **Never `cmake --clean-first`** (20+ min CUDA recompile); for stale `.obj` issues delete only `engine/build/acestep-core.dir/` and `engine/build/Release/acestep-core.lib`.
11. **Never delete generated audio or test outputs, even ones you predict are bad**; the human verifies results by ear. Leave all artifacts in place unless the user explicitly asks for cleanup.

## Dev loop

1. Start dev mode detached: `pwsh -NoProfile -File tools\detach.ps1 dev.bat` from the repo root. `dev.bat` runs Vite (UI, :3000, HMR) plus `server\restart-loop.cmd`, which loops `npx tsx watch src/index.ts` (restart-loop.cmd:6). Develop against `http://localhost:3000` and poll `/api/health` until it answers. `LAUNCH.bat` (Node :3001 serving prebuilt `ui/dist/`) is the end-user prod path; agents don't run it.
2. Save any `server/src/**/*.ts` file → tsx watch restarts the **whole Node process**, which kills ace-server and any running training child too (both are child processes; `shutdown()` in index.ts:489-522). Expect a few seconds of engine downtime after every save; a generation or training run in flight will die. Check `GET /api/training/jobs` before editing server code while training runs.
3. The restart loop re-enters if a `.restart-requested` marker file exists at repo root (`RESTART_MARKER_PATH`, config.ts:23); this is how the in-app "restart server" (`POST /api/shutdown/restart`) works without killing Vite.
4. Type-check before committing: `cd d:\Ace-Step-Latest\hot-step-cpp\server; npx tsc --noEmit`.
5. Logs land in `<data root>/logs/<YYYY-MM-DD_HH-MM-SS>/` (the checkout's `logs/` when there is no data folder): `node_console.log` (server), `ace_engine.log` (engine), `generations/gen_<jobId>_<task>.log` (per generation), `training/<kind>_<jobId>.log` (per training job). Newest folder = current session.

## Procedure: add a new feature (route + service)

1. **Create the route file** `server/src/routes/myFeature.ts`. Convention: open with a comment block stating the mount point and a route table (best example: seeds.ts:1-20).
   ```ts
   // myFeature.ts — one-line purpose
   // Mounts at: /api/my-feature
   // Routes:
   //   GET  /api/my-feature        — list things
   import { Router } from 'express';
   import { config } from '../config.js';   // note the .js extension

   const router = Router();

   router.get('/', (req, res) => {
     try {
       res.json({ ok: true });
     } catch (err: any) {
       console.error('[MyFeature] list failed:', err.message);
       res.status(500).json({ error: err.message });
     }
   });

   export default router;
   ```
2. **Register it in `server/src/index.ts`**: two edits. Add the import to the block at index.ts:19-69 (`import myFeatureRoutes from './routes/myFeature.js';`) and mount it in the block at index.ts:104-151 (`app.use('/api/my-feature', myFeatureRoutes);`), after the body parsers at index.ts:101-102. URL segments are kebab-case (`/api/model-manager`, `/api/cover-art`, `/api/stem-studio`). Exceptions to the filename rule: `songBuilder.ts` mounts at `/api/builder`, `dawV1.ts` at `/api/v1/daw`, `backends.ts` at `/api` (it spells its own sub-paths), `workers.ts` exports a second router mounted at `/api/training/worker`, and `listening.ts` mounts at `/listening` outside `/api`.
3. **Put non-trivial logic in a service**: `server/src/services/myFeature.ts`. When a feature grows, use a sub-folder: `services/generation/` holds `translateParams.ts`, `lmCache.ts`, `postProcessing.ts`, `pollUntilDone.ts`, `gpuLane.ts` and more, all extracted from the once-monolithic `generate.ts`.
4. **Put the wire shapes in `server/src/contracts/myFeature.ts`** when the UI reads them. The UI imports those types by relative path (`import type { … } from '../../../server/src/contracts/…'`), and routes check their responses with `satisfies` (seeds.ts:133). This keeps one copy of each shape; see the shared-contracts notes in `docs/dev/api-contracts.md`.
5. **Auth-scope if the data is per-user**: `import { getUserId } from './auth.js';` then
   ```ts
   const userId = getUserId(req);
   if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }
   ```
   (exact pattern: songs.ts:36-37). Auth is deliberately lightweight: an in-memory `Map<token, userId>` that resets on restart, with a single auto-created local user "Producer" (auth.ts:14-27). `getUserId` is auth.ts:99-103.
6. **Express 5 handler style**: end early with `res.status(...).json(...); return;`; do not `return res...` from typed handlers. The SPA fallback uses Express-5 wildcard syntax `app.get('/{*splat}', ...)` (index.ts:198); Express 4 `'*'` patterns do not work.
7. **Log with a bracketed feature tag**: `console.log('[MyFeature] ...')`. Existing tags: `[Server]`, `[DB]`, `[Config]`, `[Generate]`, `[Settings]`, `[Logger]`, `[ace-server]`. This convention is what makes `node_console.log` greppable. To also surface lines in the UI's live log panel, `import { pushLog } from './logs.js';` (from a service: `'../routes/logs.js'`) and call `pushLog('text', 'server');`.
8. **Type-check** (`npx tsc --noEmit`), test via the running dev server, regenerate the route index with `node tools/docs/build-docs.mjs` (it rewrites `docs/dev/api.md`), run `node tools/docs/check-docs.mjs`, then commit explicit paths:
   ```powershell
   cd d:\Ace-Step-Latest\hot-step-cpp; git add server/src/routes/myFeature.ts server/src/services/myFeature.ts server/src/index.ts docs/dev/api.md; git commit -m "feat(server): my feature"
   ```

**Not every feature needs SQLite.** `routes/seeds.ts` stores JSON files under `<data dir>/seeds/` on purpose (ComfyUI-compatible, import/export-friendly) with a filename-sanitization regex at seeds.ts:50. Use seeds.ts as the template for small self-contained features; use generate.ts for engine-orchestrating ones.

**Multi-step studio work belongs in a workflow kind.** `registerWorkflowKind({ kind, input, run })` (routes/workflows.ts:166; `WorkflowKind` in services/workflows/workflowJobs.ts:75-85) gives a durable job with a zod-validated input, cancellation (`ctx.signal`, `ctx.throwIfCancelled()`), an event stream (`ctx.emit`, served as SSE at `GET /api/workflows/jobs/:id/events`), retry, and `ctx.audio.enqueue()` to queue `/api/generate` bodies on the durable audio queue. Register from the studio's own module; examples are `services/workflows/repaintLayerWorkflow.ts` (registered at index.ts:144) and `routes/songBuilder.ts:25`. Contract: [docs/dev/api-contracts.md](../../../docs/dev/api-contracts.md#workflow-jobs-and-revisions).

## Procedure: SQLite schema change

The core schema lives in `server/src/db/database.ts`. One connection, **synchronous** better-sqlite3 API, module singleton (`initDb()` once at startup, `getDb()` everywhere else, `closeDb()` on shutdown at index.ts:514). Pragmas at open: WAL, `synchronous = NORMAL`, `foreign_keys = ON` (database.ts:32-34).

1. **New core table**: add a `CREATE TABLE IF NOT EXISTS` block inside the big `db.exec(...)` at database.ts:37-316. Schema is re-run idempotently at every startup; there is **no migration framework and no version table**.
2. **A table one feature owns** can live with that feature instead: a store class whose constructor takes the `Database` and runs its own `CREATE TABLE IF NOT EXISTS` (`ensureRevisionSchema`, services/workflows/revisions.ts:53; `ensureAudioQueueSchema`, services/audioQueue/intentQueue.ts:68; the `GenerationJobLedger` constructor, services/generation/jobLedger.ts:35-47). Construct it lazily on first use (`ledgerInstance ??= new GenerationJobLedger(getDb())`, generate.ts:68), which also lets tests hand it a test database.
3. **New column on an existing table**: add an entry to the checked-migration list (the `songsMigrations` array pattern, database.ts:365-430; `trainingMigrations` at :320-363 is the same shape):
   ```ts
   {
     check: `SELECT COUNT(*) as c FROM pragma_table_info('songs') WHERE name='my_col'`,
     alter: `ALTER TABLE songs ADD COLUMN my_col TEXT DEFAULT ''`,
   },
   ```
   The loop runs `ALTER` only when the column is absent and logs `[DB] Migration: ...`. **Use this idiom for new columns.** (An older idiom, `try { db.exec("ALTER TABLE ...") } catch {}`, exists for the lireek tables at database.ts:432-473 and two `audio_generations` columns at :504-508; don't add to it.)
4. **Column conventions**: TEXT primary keys are UUIDs (`uuidv4()`) for core tables; lireek (Lyric Studio) tables use `INTEGER PRIMARY KEY AUTOINCREMENT`; core-table timestamps are `TEXT DEFAULT (datetime('now'))`, while the newer service-owned tables (`workflow_documents`, `generation_jobs`, `audio_intents`) store epoch-ms INTEGERs; booleans are INTEGER 0/1, re-hydrated as `!!row.is_public` (songs.ts:54); complex values are JSON-in-TEXT parsed with `JSON.parse(s.tags || '[]')`.
5. **Query style**: inline `getDb().prepare(sql).get/all/run(...params)` with `?` placeholders; no statement-caching layer. Dynamic filters append to the SQL string and a params array (songs.ts:39-47), including JSON queries like `json_extract(generation_params, '$.source') = ?`.
6. **Don't imitate the two-database pattern.** `db/lireekDb.ts` is a query-helper module over the *same* connection; its `initLireekDb`/`closeLireekDb`/`getLireekDb` are deprecated no-ops or aliases (lireekDb.ts:14-25). The old separate `lireek.db` was consolidated into `<data dir>/hotstep.db` (one-time `ATTACH DATABASE` migration `migrateLireekData` at database.ts:532-625, a good reference if you ever need bulk data migration).

## Procedure: add a config / .env setting

`server/src/config.ts` builds one exported `config` object (config.ts:211-422) grouping `aceServer` (engine exe autodetected from 4 candidate paths at config.ts:133-139; port default 8085), `essentia`, `labeling`, `server` (port 3001), `data` (`dir` from `resolveDataDir()`, config.ts:27-30: `<data root>/library` with a data folder, otherwise `DATA_DIR` resolved against `server/`; getter `dbPath` → `<data dir>/hotstep.db`. A stale legacy `data/` dir may exist at repo root with an outdated hotstep.db; ignore it), `lireek` (LLM keys/endpoints), `vst`, `training`, `workers`, `whisper`. `.env` lives at the repo root in a checkout without a data folder and at `<data root>/settings/.env` with one (config.ts:36-40); it is auto-bootstrapped from `.env.example` on first launch (config.ts:95-109).

To expose a new env setting in the Settings UI, you must touch **three places** in config.ts or it silently won't work:

1. Add the key to the `EXPOSED_ENV_KEYS` whitelist (config.ts:442-470); "nothing else leaks" to the UI.
2. If it only takes effect at process/engine spawn time, also add it to `RESTART_REQUIRED_KEYS` (config.ts:483-488) so the UI shows "restart required".
3. Add an `apply('MY_KEY', setter, getter)` line (or `applyInt` for numbers) inside `reloadEnvConfig()` (config.ts:494-636). **Without this the value saves to `.env` but never hot-patches the live config**, and because `restartRequired` is computed from the keys `reloadEnvConfig()` reports changed, a restart-required key with no `apply()` line never reports `restartRequired: true` either.

A module that caches state derived from config (rather than reading `config` on each use) registers a callback in `configReloadListeners` (config.ts:480) so a Settings save reaches it without a restart. Settings that are not meant for users stay out of `EXPOSED_ENV_KEYS` entirely (see the note on `training` at config.ts:398-400).

The Settings route (`POST /api/settings/env`, settings.ts:136-209; the header comment says PUT, the code is POST, code wins) rewrites `.env` preserving comments and line endings, writes values through `formatEnvValue()` (config.ts:430-434) so `#` and quotes survive, calls `reloadEnvConfig()`, and returns `restartRequired`.

## Procedure: call the engine (aceClient)

`server/src/services/aceClient.ts` is the typed wrapper over ace-server's HTTP API (`BASE = config.aceServer.url`, i.e. `http://127.0.0.1:8085` by default). MM3 and YuE2 have their own clients for their route families (`services/backends/minimax/client.ts`, `services/backends/yue2/client.ts`).

**Critical constraint** (aceClient.ts:6-8, verbatim comment): *"ace-server uses single-threaded httplib. During heavy compute (DiT generation, adapter merge, VAE decode) it cannot respond to HTTP requests."* Hence timeout tiers via `AbortSignal.timeout` (aceClient.ts:18-21); pick one deliberately for any new call:

| Tier | Value | For |
|---|---|---|
| `TIMEOUT_QUICK` | 15 s | health, props, job submit |
| `TIMEOUT_POLL` | 30 s | job polling; fail fast, let the watchdog decide |
| `TIMEOUT_RESULT` | 300 s | fetching large audio bodies, multipart submits |
| `TIMEOUT_SA3` | 30 min | `/sa3-refine`, `/sa3-synth` (the first call may build a TensorRT engine) |

- **Transport**: every call goes through `aceFetch` (aceClient.ts:342-356), which retries a refused connection on any method and a reset on GET, up to three attempts.
- **Async-job pattern**: `submitLm` / `submitSynth` / `submitSynthMultipart` / `submitUnderstand` / `warm` POST and return a job id string; poll with `pollJob(id)` → `AceJobStatus` (aceClient.ts:236-254: `status: 'running'|'done'|'failed'|'cancelled'`, `phase?`, `phase_step?`, `phase_total?`, `adapter_progress?`, YuE2 `end_reason`/`songs_done`); phases like `adapter_precompute`, `dit_inference` are `AceJobPhase` (aceClient.ts:258-273). Fetch the result with `getJobResult(id)` (`GET /job?id=N&result=1`); cancel with `cancelJob(id)`; `dropSongs(id, mask)` takes songs out of a running YuE2 batch. `vaeEncode` and `codesDecode` wrap the submit-and-poll loop for you. Sync endpoints (`/spectral-lifter`, `/pp-vae-reencode`, `/sa3-refine`, `/sa3-synth`) return processed WAV bytes directly; the two SA3 calls go through `workerPost` (aceClient.ts:364-372), which waits on `GET /worker-drain` after a failed or aborted call so the caller's GPU lane is released only once the engine has stopped.
- **A poll timeout is NOT a failure.** The engine may simply be busy mid-DiT-step. Catch transient poll errors and retry; only treat explicit `failed`/`cancelled` statuses as terminal (see `pollUntilDone`, services/generation/pollUntilDone.ts:156-193).
- **New engine parameter?** Add the field to the `AceRequest` interface first (aceClient.ts:84-233), then plumb it through `services/generation/translateParams.ts` (`translateParams()` at :47, UI params → AceRequest). Known gotcha (institutional, from the LM-echo bug): server-only fields do **not** survive the engine's `/lm` round trip. Synth requests are rebuilt from the original aceReq plus only the LM-generated fields (`audio_codes`, caption, lyrics, bpm, duration, keyscale, timesignature, and the per-output `lm_seed`; services/backends/ace/generate.ts:192-213). Never assume a field you sent to `/lm` comes back.
- **File upload to the engine**: multipart is hand-rolled with a manual boundary because the engine's parser expects a `request` JSON part plus named binary parts (`audio`, `ref_audio`, `src_latents`, `ref_latents`, `seed_latents`). The multipart path takes a **single JSON object, not an array** (aceClient.ts:525-529). Copy `submitSynthMultipart` (aceClient.ts:498-575).
- **Error convention**: non-OK responses `throw new Error('ace-server POST /x failed (status): body')` (`aceGet`/`acePost`, aceClient.ts:390-422); soft-fail helpers return `null`/`false`/`[]` (`getJobLatent`, `isReachable`, `sa3Props`, `listLoadedModels`, `unloadLabel`, `restoreEvictPolicy`).
- **Readiness gate**: routes that hit the engine should return 503 while the engine is booting or stopped for training. `import { engineReady, engineBootStatus } from '../engineState.js';` (this module exists solely to break a circular import between index.ts and routes, engineState.ts:1-4) and `isEngineSuspended()` from `services/aceEngineProcess.ts`. Pattern: `submitGeneration` (generate.ts:344-357).

## Worked example (dissected): the generation feature

`routes/generate.ts` (~1060 lines) plus `services/backends/<id>/generate.ts` is the canonical "server orchestrates engine" feature. Copy its skeleton for anything long-running.

1. **In-memory job map**: the `GenerationJob` type (services/generation/jobTypes.ts: id = uuid, userId, envelope, attempts, status `pending|running|lm_running|synth_running|saving|succeeded|failed|cancelled`, `stage`/`progress` for the UI, `aceJobId`, `acePhase`, result, error, params) in `const jobs = new Map()` (generate.ts:60). A TTL sweeper prunes terminal jobs older than 1 h every 10 min, with `.unref()` so the interval never blocks process exit (generate.ts:85-98). The `generation_jobs` ledger (services/generation/jobLedger.ts) records each outcome, so `/status/:id` still answers after a restart or the prune (generate.ts:483-492); rows a dead process left open are marked `interrupted` at startup.
2. **POST `/api/generate`** (generate.ts:476-480) calls `submitGeneration()` (generate.ts:344-464): 503 if the engine is suspended or not ready; 401 if no user; 409 if the caller's `expectedBackend` no longer matches; a `generation-intent/1` body is resolved server-side; `buildEnvelope()` (services/generation/envelope.ts) captures the backend id, models and retry policy; `jobs.set(...)`; `enqueueGeneration(job)`; respond `{ jobId, status }` **immediately**. The durable audio queue (services/audioQueue/intentQueue.ts) calls the same exported `submitGeneration`.
3. **Serialized GPU lane + retry** (generate.ts:185-273): `runOnGpuLane()` (services/generation/gpuLane.ts) runs one owner at a time. The in-code comment explains why: the engine is single-GPU, `subscribeLines()` is a global log pub/sub with no job tagging, and post-processing re-runs share the lane. Retry follows `job.envelope.policy.retry` (every backend sets `maxAttempts: 2, reseedOnRetry: true`, e.g. services/backends/ace/index.ts:170): a retryable failure resets the job to `pending` with a fresh random seed and a 2 s pause; `Cancelled`/`Unauthorized` are non-retryable.
4. **Pipeline**: `runGeneration()` (generate.ts:156-184) evicts other model families (services/generation/residency.ts), then calls the backend's `generate()`. For ACE (services/backends/ace/generate.ts): `translateParams(job.params)` → aceReq (:51); **write the resolved seeds back into `job.params`** so the DB stores the actual seed used (reproducibility, :58-65); decide whether to skip the LM phase (explicit skipLm, `audio_codes` already present, or a cover-type task, :70-72); `startGenerationLog(job.id, taskType)` + `logGenerationParams(job.id, aceReq)` (:76-77); check the LM cache (`services/generation/lmCache.ts`, :91-118); submit via `aceClient.submitSynth` or, when any source/reference audio or latents are present, `submitSynthMultipart` (:815-827); stage timings via a small `timed()` helper.
5. **Watchdog polling** (`pollUntilDone`, services/generation/pollUntilDone.ts): 500 ms interval; honors an `AbortSignal` for cancellation; **stall detection**: a stage that ticks ("Step n/total") and shows no change for 120 s triggers a probe of the engine, and the job is cancelled as stalled only if the engine no longer reports it running; quiet stages (model loads, adapter merge, VAE decode) get a 15 min window; wall-clock timeout clamped to 5 to 360 min, default 45; an engine crash or three 404s on the job id fail the job at once; transient poll errors are logged and retried.
6. **Persist**: on success, `INSERT INTO songs (...)` with a fresh uuid, `generation_params` = the per-track params as JSON (services/backends/ace/generate.ts:1247-1259); duration measured from the output WAV (:1214-1225); audio files go under `config.data.audioDir` and are served statically at `/audio/*` (index.ts:154).
7. **Companion endpoints**: `GET /status/:id` returns `{ jobId, status, stage, progress, result, error, attempts, ace_job_id, ace_phase, ace_phase_progress, batch, mm3_* }` (generate.ts:483-529); `POST /cancel/:id` marks the job cancelled, cancels the engine job and fires the AbortController (`cancelGenerationJob`, generate.ts:612); plus `/cancel-all`, `/queue`, `/reset-queue`, `GET /mm3/stream/:id` (live MM3 audio pipe), `POST /yue2/plan` (YuE2 score preview) and the STORM `/storm/stream|stop|control` routes.

**Per-generation logging** (services/logger.ts): `startGenerationLog(jobId, taskType)` → `logGeneration(jobId, 'INFO'|'DEBUG'|'WARNING'|'ERROR', msg)` / `logGenerationParams(jobId, obj)` → `finishGenerationLog(jobId, taskType)` or `failGenerationLog(jobId, error, taskType)`. Lines are **buffered in memory and flushed to `<data root>/logs/<session>/generations/gen_<jobId>_<task>.log` only on completion** (logger.ts:125-212); a crash before finish loses the buffer, but every line is also streamed live via `pushLog`. Training jobs write `training/<kind>_<jobId>.log` live instead (`logTraining`, logger.ts:104-117).

## Key files

| Path | Role |
|---|---|
| `server/src/index.ts` | App entry: route registration (imports :19-69, mounts :98-151), static serving, CUDA runtime bootstrap (:218-384), graceful shutdown (:487-522) |
| `server/src/config.ts` | All configuration; `.env` hot-reload (`EXPOSED_ENV_KEYS` / `RESTART_REQUIRED_KEYS` / `reloadEnvConfig`) |
| `server/src/dataroot/` | Data folder resolution (`bootstrap.ts` `RUNTIME`, `locate.ts`); `server/scripts/data-paths.ts` prints it |
| `server/src/db/database.ts` | SQLite singleton, core schema (`CREATE TABLE IF NOT EXISTS`), column migrations |
| `server/src/db/lireekDb.ts` | Lyric-Studio query helpers over the same connection (init/close are deprecated no-ops) |
| `server/src/contracts/` | Wire shapes shared with the UI |
| `server/src/services/aceClient.ts` | Typed HTTP client for the engine; `AceRequest`; timeout tiers; multipart |
| `server/src/services/aceEngineProcess.ts` | Engine child process: spawn args, log fan-out, crash-respawn limiter, deliberate stop/restart |
| `server/src/engineState.ts` | `engineReady` / `engineBootStatus` flags (circular-import breaker) |
| `server/src/routes/generate.ts` | Worked example: job map, GPU lane queue, retry, status |
| `server/src/services/backends/` | Backend registry and the four backend implementations |
| `server/src/services/generation/translateParams.ts` | UI params → AceRequest translation |
| `server/src/services/generation/pollUntilDone.ts` | Engine job polling with the stall and wall-clock watchdogs |
| `server/src/routes/seeds.ts` | Template for a small self-contained feature (filesystem storage, route-table header) |
| `server/src/routes/songs.ts` | Template for auth-scoped SQLite CRUD (incl. the static-before-`/:id` warnings) |
| `server/src/routes/workflows.ts` | Durable workflow jobs and revisioned documents; `registerWorkflowKind` |
| `server/src/routes/auth.ts` | Token-lite auth; `getUserId(req)` helper |
| `server/src/services/logger.ts` | Session log dirs, `logEngine`, per-generation buffered logs, live training logs |
| `server/src/routes/logs.ts` | 2000-line ring buffer + SSE at `GET /api/logs`; `pushLog` / `subscribeLines` |
| `server/src/routes/settings.ts` | `.env` read/rewrite + hot-reload endpoint (`POST /api/settings/env`), GPUs, data folder |
| `server/restart-loop.cmd` | Dev-mode tsx watch wrapper with `.restart-requested` re-entry |
| `server/package.json` | Scripts (`dev`/`typecheck`), `"type": "module"`, Node `<25` engines pin |

## Failure signatures

| Symptom | Cause | Fix |
|---|---|---|
| `Database not initialized. Call initDb() first.` | A module ran a DB query at import time, before index.ts:85 | Move queries inside handlers/functions, or construct the store lazily |
| Engine requests time out during generation | Normal: the engine answers HTTP on one thread and can't respond mid-compute | Use the right timeout tier; retry transient poll errors; never treat a poll timeout as job failure |
| `Generation stalled — no progress for Ns` | The watchdog fired: a ticking stage went quiet and the engine no longer reported it running, or a quiet stage passed its 15 min window | Check `gen_*.log` + `ace_engine.log` in the newest `<data root>/logs/` session; the lane is released automatically |
| `Generation lost: the engine crashed mid-render` / `no longer knows this job` | The engine process died or restarted under the job | `ace_engine.log` for the crash; a training stop/start also restarts the engine |
| `Engine is paused for training preprocessing` (503) | A training job stopped the engine to use the GPU (`isEngineSuspended()`) | Wait for the job to finish |
| `[ace-server] Crashed 3 times within 30s — giving up.` | Missing engine DLL, broken build, or a model that fails to load; the crash limiter halts respawn and sets engine not-ready (aceEngineProcess.ts:320-326) | Fix the engine build (via `dev-rebuild.bat`); check `ace_engine.log` |
| `No models installed — waiting for a download in the Model Manager.` | Empty models folder: a first run, not a crash (aceEngineProcess.ts:298-307); `/api/health` reports `engine.noModels: true` | Download a model or copy a `.gguf` into the models folder; the engine starts by itself after the download or the 5 s folder check (`watchForModels`, aceEngineProcess.ts:44-68) |
| New env setting saves but never takes effect | Key missing from the `apply(...)` list in `reloadEnvConfig()` or from `EXPOSED_ENV_KEYS` | Touch all three places in config.ts (see procedure above) |
| Setting saved, hot-reload reports OK, engine ignores it | Key is spawn-time (in `RESTART_REQUIRED_KEYS`) | Restart the app so the engine child respawns with new args |
| `GET /api/things/recent` → 404 with id "recent" | Static route declared after `/:id` | Reorder: static routes first (songs.ts:63) |
| A new route sees `req.body` undefined | Mounted before `express.json()` (index.ts:101) | Mount after the body parsers; only the worker proxy goes before them |
| better-sqlite3 native/ABI error after `npm install` or Node switch | Addon built for a different Node ABI or skipped native rebuild | Use Node 24; `npm rebuild better-sqlite3` when the app is idle |
| `tsc` errors on imports that run fine under tsx | Missing `.js` extension on a relative ESM import | Add `.js` to the import path |
| Spawned engine can't find DLLs despite a PATH edit | Spreading `process.env` on Windows creates a case-sensitive duplicate (`Path` vs `PATH`) that shadows | Find the real key case-insensitively before prepending; pattern at aceEngineProcess.ts:227-231 |
| Multipart submit to `/synth` fails with a parse error | Sent a JSON **array** as the multipart `request` part | Engine multipart expects a single object (aceClient.ts:525-529) |

## Institutional knowledge

- **VALIDATED (rule, verbatim):** Type-check with `npx tsc --noEmit` during dev; only `npm run build` before user testing.
- **VALIDATED (code comment, aceClient.ts:6-8):** the engine is single-threaded httplib and cannot answer HTTP during heavy compute; every timeout, watchdog, and retry in the server exists because of this.
- **VALIDATED (bug fixed, per project memory + services/backends/ace/generate.ts:192-213):** LM echo sideband gotcha: server-only AceRequest fields don't survive the engine's `/lm` round trip. Rebuild synth requests from the original aceReq plus LM-generated fields only; never whitelist-copy from the LM response.
- **VALIDATED (in-code warning, aceEngineProcess.ts:211-213):** Windows `process.env` is a case-insensitive proxy but spreading it makes plain case-sensitive keys; mutating `PATH` blindly creates a shadowing duplicate of `Path`.
- **VALIDATED:** GPU work is intentionally single-file because `subscribeLines()` is global pub/sub with no job tagging (generate.ts:185-190); don't "fix" concurrency without solving log attribution first.
- **VALIDATED (config.ts:232-237):** draft-LM speculative decoding is DISABLED; GGML per-call overhead negates the speedup. `ACESTEPCPP_DRAFT_LM` env re-enables it; auto-detect is commented out (config.ts:274-287).
- **Field present, feature reverted:** `adapter_section_isolation` still exists in AceRequest (aceClient.ts:176) and the engine parses it into `g_hotstep_params` (hot-step-server.cpp:836-837, :1475), but nothing in `engine/src/` reads it; the regional self-attn isolation was reverted for breaking musical continuity. Treat the field as inert.
- **VALIDATED:** `uncaughtException`/`unhandledRejection` are logged but do **not** exit the process (index.ts:526-531), and there is no global Express error middleware (only the DAW facade's body-error handler, index.ts:147), so a throw in an async handler will not produce a clean JSON 500: catch locally in every handler.

More depth (full aceClient method catalog, engine process lifecycle, settings hot-reload internals, DB idiom code, SSE log stream details): [reference.md](reference.md).

## Deeper reading

- `AGENTS.md` (repo root; `CLAUDE.md` only imports it): orientation map, build/git rules.
- `docs/dev/architecture.md` (tiers, request path, backends), `docs/dev/api-contracts.md` (request, job, audio queue and workflow contracts), `docs/dev/config.md` (every env var), `docs/dev/api.md` (generated route list).
- `engine/docs/ARCHITECTURE.md`: engine internals, CLI, request JSON, generation modes (committed).
- `FEATURES.md`: full feature catalogue (committed).
- `docs/dev/plugins-authoring.md`: Lua plugin authoring; solvers/schedulers/guidance are plugins, not C++ (committed).
- `docs/plans/`: internal design/investigation docs. **Gitignored, local-only; may be absent on a fresh clone.**
- `server/src/data/assistant-knowledge.md`: in-app assistant knowledge base (committed).
