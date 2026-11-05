# Validating Changes — Reference

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

Deep detail backing [SKILL.md](SKILL.md). Paths and line refs first verified
against the repo on 2026-07-02 and re-checked on 2026-10-23, after generation
moved into per-backend services. Line numbers drift as code changes — treat them
as anchors, re-verify before relying on exact positions.

## 1. Health endpoints

`GET http://localhost:3000/api/health` (dev app — :3001 is the raw prod server) (`server/src/routes/health.ts:33-82`, shape in `server/src/contracts/health.ts`) returns, among other fields:

```json
{
  "status": "ok",
  "version": "<VERSION file, or dev>",
  "commit": "<startup HEAD>",
  "dirty": false,
  "build": { "tag": null, "sha": "...", "variant": null, "channel": null },
  "aceServer": { "status": "ok|disconnected", "url": "...", "version": "..." },
  "server":    { "port": 3001, "uptime": 123.4 },
  "clients": 1,
  "engine":    { "ready": true, "bootStatus": "...", "noModels": false, "backend": "cuda", "device": "CUDA0" }
}
```

Require `engine.ready === true` before submitting work. `engine.noModels: true`
means the engine is down only because no model is installed yet.
`aceServer.status: "disconnected"` during an active generation may just mean the
engine is busy — ace-server is **single-threaded httplib** and cannot answer HTTP
mid-DiT-step (`server/src/services/aceClient.ts:6-21`; the client uses generous
timeouts for exactly this reason). Direct engine probe, bypassing Node:
`Invoke-RestMethod http://127.0.0.1:8085/health`.

## 2. Auth

All in `server/src/routes/auth.ts`:

- `GET /api/auth/auto` (lines 36-41) — get-or-create local user, returns
  `{ user, token }`. No password. Tokens are in-memory UUIDs (`tokens` Map),
  **reset on Node server restart**.
- Route guard: `getUserId(req)` (lines 98-103) reads `Authorization: Bearer <token>`.
- `POST /api/generate` requires it (`submitGeneration`,
  `server/src/routes/generate.ts:359` → 401 without).
- The status/queue endpoints do **not** require auth (they never call `getUserId`).

## 3. POST /api/generate — request semantics

Route at `server/src/routes/generate.ts:476-480`; it calls `submitGeneration`
(lines 344-464), which validates, stores a `structuredClone` of the body as
`job.params` and queues the job. `translateParams`
(`server/src/services/generation/translateParams.ts`) maps UI names → engine
(`AceRequest`) names:

- Caption source precedence: `params.prompt || params.songDescription ||
  params.caption || params.style` (translateParams.ts:49).
- `instrumental: true` ⇒ `lyrics: "[Instrumental]"` (line 54).
- `randomSeed: true` ⇒ random seed; else `seed` passed through (line 90).
- `duration` gets `durationBuffer` added only when `autoTrimEnabled` (line 66).

Phase selection (`server/src/services/backends/ace/generate.ts:69-89`):

- `skipLm: true` ⇒ `needsLm = false`; defaults backfilled: `bpm 120`,
  **`duration 120` if unset** (always pass a short `duration` for smokes),
  `keyscale "C major"`, `timesignature "4"`.
- Cover-family task types (`cover`, `cover-nofsq`, `repaint`, `lego`, `extract`)
  also skip the LM, as does a request that already carries `audio_codes`.

Responses:

- `200 { jobId, status: "pending" }` (`generate.ts:479`).
- `503 { error: "Engine not ready: <bootStatus>", detail }` while bootstrapping
  (`generate.ts:352-357`), and `503 { error: "Engine is paused for training
  preprocessing — try again when the job finishes" }` while a preprocess job
  owns the GPU (lines 347-349).
- `401 { error: "Unauthorized" }` without a valid token.
- `409` when an `expectedBackend` field disagrees with the active backend or the
  `backendOverride`; `400` for a malformed override or intent, or a missing
  timbre reference file.

## 4. Queueing, retries, watchdogs

- **Global serial queue** (`enqueueGeneration`, `generate.ts:191-273`): one job at
  a time on the shared GPU lane (`server/src/services/generation/gpuLane.ts`) —
  the engine is single-GPU, progress-line subscription is a global pub/sub that
  would cross-contaminate concurrent jobs, and post-processing re-runs share the
  lane. Your smoke waits behind the user's job.
- **1 automatic retry** on transient failure with a randomized seed
  (`generate.ts:203-259`; the ACE backend's `policy.retry` is
  `{ maxAttempts: 2, reseedOnRetry: true }`,
  `server/src/services/backends/ace/index.ts:170`) — a smoke that "succeeded"
  may have succeeded on attempt 2 with a different seed; check the gen log for
  `[Retry]` lines (or the `attempts` array in `/status`) before claiming
  seed-reproducibility.
- **Watchdogs** (`server/src/services/generation/pollUntilDone.ts`):
  - stall: a ticking stage (`... Step n/total`) with no stage/progress change for
    2 min (`STALE_TIMEOUT_MS = 120_000`) makes the server poll the engine; `done`
    is kept, and a still-running engine gets the 15 min quiet window before
    cancel + `Generation stalled — no progress for Ns (last stage: "...")`.
    Non-ticking stages (model loads, adapter merge, DiT setup) and VAE decode get
    15 min from the start; YuE2's NAR stage scales with the chunk size.
  - wall clock: default 45 min, clamped to 5 to 360 min via the
    `generationTimeoutMinutes` param. A render that finished right at the limit
    is kept.
  - transient poll errors against :8085 are logged and retried, not fatal.
  - an engine crash mid-render fails the job at once with `Generation lost: the
    engine crashed mid-render (see the engine log)`; three 404s in a row from
    `/job` give `Generation lost: the engine no longer knows this job`.
- **TRT first run:** stage becomes `Building TRT engine (first run only, ~5-10 min)...`
  (`server/src/services/backends/ace/generate.ts:772-788`) and is then mutated
  with elapsed seconds specifically to defeat the stall detector. Normal; do not
  cancel.

## 5. Endpoints summary (all in generate.ts)

| Endpoint | Notes |
|---|---|
| `POST /api/generate` | auth required; returns `{ jobId, status }` |
| `GET /api/generate/status/:id` | `{ jobId, status, stage, progress, result, error, attempts, ace_job_id, ace_phase, ace_phase_progress, batch, mm3_* }` (lines 483-529); a job gone from memory (restart, or the one-hour prune) answers from the job ledger |
| `GET /api/generate/queue` | `{ depth, running, owner, draining, current, pending }` — check before smoking (line 651) |
| `POST /api/generate/cancel/:id` | cancel one job (line 627) |
| `POST /api/generate/cancel-all` | cancel every active job, whoever submitted it (line 633) |
| `POST /api/generate/reset-queue` | force-drain — destructive to the user's queue, ask first (line 675) |
| `GET /api/generate/mm3/stream/:id` | live audio of a streaming MM3 render, once `/status` says `mm3_streaming: true` (line 541) |

Status values (`server/src/services/generation/jobTypes.ts:16`):
`pending | running | lm_running | synth_running | saving | succeeded | failed | cancelled`.

Success `result` shape (`jobTypes.ts:27-50`): `{ audioUrls: ["/audio/<uuid>.wav"],
songIds, bpm?, duration?, keyScale?, timeSignature?, masteredAudioUrl?,
masteredAudioUrls?, noAdapterAudioUrl?, noAdapterAudioUrls?, durations?,
timing?, totalMs? }`.

`ace_phase` / `ace_phase_progress` surface the engine's fine-grained phase (e.g.
`adapter_precompute`, `step 12/50`) so you can tell "slow but alive" from "wedged".

## 6. Output files

**Live data dir = `<data root>/library`** — `config.data.dir` is
`resolveDataDir(process.env.DATA_DIR)` (`server/src/config.ts:27-30`, set at
line 319): `<data root>/library` when a data folder exists (a relative
`DATA_DIR` resolves against the data root), `server/data` in a dev checkout that
has none. The repo-root `data\` directory is not the library. Never verify
outputs against root `data\`.

Per successful track, written into `<data root>/library\audio\`
(`server/src/services/backends/ace/generate.ts`):

| File | When |
|---|---|
| `<uuid>.wav` (or `.mp3`) | always — served at `http://localhost:3000/audio/<uuid>.wav` in dev (line 842) |
| `<uuid>.lrc` | only if lyrics + engine alignment (skipped for instrumental) (line 931) |
| `<uuid>.latent` | HSLAT post-DiT latent companion (line 985) |
| `<uuid>_mastered.wav` | when the post-processing chain renders one (`server/src/services/generation/postProcessing.ts:406`) |
| `<uuid>.lyrics.json` | only if Whisper transcription enabled (line 579) |

DB: row inserted into `songs` in `<data root>/library\hotstep.db`
(`server/src/services/backends/ace/generate.ts:1249`); `generation_params` column
holds the full request params JSON (your reproduction recipe), `quality_scores`
holds advisory evaluator telemetry.

## 7. Log anatomy

Session folders (`initLogger`, `server/src/services/logger.ts:45-63`), name-sorted = time-sorted:

```
<data root>/logs/YYYY-MM-DD_HH-MM-SS/
  ├── node_console.log                        # all Node stdout/stderr, mirrored
  ├── ace_engine.log                          # ace-server (C++ engine) stdout/stderr
  ├── generations/gen_<jobId>_<taskType>.log  # per-generation, flushed ON COMPLETION
  └── training/<kind>_<jobId>.log             # per training job, written live
```

- The gen log opens with a `GENERATION STARTED` header, then the full resolved
  engine params JSON (`logGenerationParams`, logger.ts:158), includes a
  `[Timing]` pipeline-breakdown table, and ends `GENERATION COMPLETED.`
  (logger.ts:179) or `GENERATION FAILED: <error>` (logger.ts:201).
- **Buffered:** the gen log lives in memory until finish/fail
  (`fs.writeFileSync` in `finishGenerationLog`/`failGenerationLog`). Mid-run,
  its lines are mirrored to the terminal SSE/console prefixed `[Gen:<jobId8>]`
  (logger.ts:152) — tail `node_console.log` or `ace_engine.log` instead.
- Failure triage order: matching `gen_*.log` → cross-ref `ace_engine.log` +
  `node_console.log`. Startup/crash issues: `node_console.log` + `ace_engine.log`.
- Clean engine start looks like (verified against a real session log on
  2026-07-01): model registry scan lines
  (`[Server] Scanning models in ...`, `[Registry] ... -> DiT/LM/Text-Enc`), then
  `[Server] Listening on 127.0.0.1:8085`, with no exception in between.

## 8. Crash respawn mechanics (why dev-rebuild.bat exists)

`onExited` in `server/src/services/aceEngineProcess.ts:295-338`: on ace-server
exiting non-cleanly (not SIGTERM/SIGINT/code 0), Node logs
`[ace-server] Restarting in 3 seconds... (crash N/3)` and respawns after 3 s.
The crash limiter (`MAX_CRASHES = 3` within `CRASH_WINDOW_MS = 30_000`, lines
83-84) then gives up and sets bootStatus to `Engine crashed 3 times — download a
model in the Model Manager, or check the logs`. An exit at the model scan with no
models installed is not counted: bootStatus says `No models installed — download
one in the Model Manager`, and `retryIfGaveUp()` starts the engine again once a
model download finishes. This respawn is why you must never kill/rebuild
ace-server externally while the app runs: the respawned exe locks the file the
linker needs.

`dev-rebuild.bat` (repo root, verified): POST `/api/shutdown` on :3001
(`server/src/routes/shutdown.ts` — stops the ace-server child it owns, the dev
Vite server on :3000 when `dev.bat` set `HOT_STEP_DEV`, then its own process
tree), poll `tasklist` for `ace-server.exe` up to 10 s, `taskkill /F` at 10 s,
abort at 15 s, then `call engine\build.cmd` (exit 1 with `BUILD FAILED` when it
fails). Final message: *"Start the app with LAUNCH.bat to pick up changes"* — it
does **not** relaunch; relaunch with `dev.bat` (detached) and check
`http://localhost:3000`. `LAUNCH.bat` is the end-user prod path.

## 9. LM echo sideband gotcha (regression trap for param changes)

When the LM phase runs, ace-server echoes back a C++ `AceRequest` — but
**ServerFields-only "sideband" params** (fields the Node server adds that the C++
struct doesn't know: `adapter_runtime_quant`, alignment timing, rebase,
`plugin_params`, …) do not survive that round trip. The fix
(`server/src/services/backends/ace/generate.ts:192-213`) rebuilds each synth
request from the **current** `aceReq` spread plus only the LM-generated fields
(`audio_codes`, `caption`, `lyrics`, `bpm`, `duration`, `keyscale`,
`timesignature`, and the per-output `lm_seed`) from the echo; the LM-cache hit
path (lines 96-112) does the same.

Validation implication: any change adding a new server-side param must be smoke
tested **with the LM phase on** (omit `skipLm`) to prove the param reaches
`/synth` — and the mechanism is the spread reconstruction, never a whitelist.

## 10. Known feature states (don't misreport)

- `adapter_section_isolation` — wire plumbing still present
  (`aceClient.ts`, `translateParams.ts`) but the feature was **reverted**
  (commit `ee041e1`; it broke musical continuity). Inactive by design.
- `rebase_source` / `rebase_beta` (basin re-base, `translateParams.ts`) — exists,
  **unvalidated** for the cross-base LoKR problem. Never report it as working.
- Draft-LM speculative decoding — **deliberately disabled** by comment in
  `server/src/config.ts` (~line 232). Not a bug to fix.

## 11. Not verified in this document

- No live smoke generation was run while authoring or re-checking this
  reference. The request/poll flow is derived from code (`generate.ts`,
  `services/backends/ace/generate.ts`, `pollUntilDone.ts`, `translateParams.ts`,
  `auth.ts`); the log and output shapes were confirmed against real 2026-07-01
  session logs and actual `<data root>/library\audio` contents. If a live smoke
  contradicts this document, trust the live run and update this file.
