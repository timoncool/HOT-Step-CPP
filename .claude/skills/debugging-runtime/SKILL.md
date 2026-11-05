---
name: debugging-runtime
description: Diagnoses HOT-Step CPP generation failures, engine crashes, hangs, and startup problems from the <data root>/logs/ session folders. Use when a music generation failed, ace-server crashed or keeps respawning, the app hangs with no progress, an API call returns 500/503, or you need to trace a gen_<uuid> log back to engine output.
---

# Debugging generation failures & crashes (log-driven playbook)

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

HOT-Step CPP has two runtime processes: the **Node server** (Express, port 3001) and its child **ace-server.exe** (the C++ inference engine, port 8085). Agents work in dev mode (`dev.bat`), where the Vite dev server on **3000** fronts the app and proxies `/api`, `/audio` and `/references` to 3001 — every URL below is the 3000 address. The Node server orchestrates every generation through the active backend (ACE-Step, MiniMax-Music3, YuE2 or SA3, each in `server/src/services/backends/<id>/generate.ts`). For ACE-Step it optionally calls the engine's LM (language model that expands your caption/lyrics into audio codes), then submits synthesis (DiT — the Diffusion Transformer that generates audio latents — followed by VAE decode to a WAV), polls until done, then saves the result to SQLite. Almost every runtime problem is diagnosable from the per-session log folders under `<data root>/logs/`. This skill tells you which file to open first for each symptom, what the real failure strings mean, and how to correlate the three log files.

All `path:line` references verified against the code on 2026-10-23. Line numbers drift; grep for the named symbol when one does not match.

## When to use this skill

- A generation failed in the UI, or a job silently disappeared.
- ace-server crashed, keeps restarting, or "gave up" after repeated crashes.
- The app hangs — progress bar stuck, queue wedged.
- `POST /api/generate` returns 503 "Engine not ready", or any API route 500s.
- The app fails to start, or starts without the engine.
- You need to reproduce a failed generation exactly (seed, adapters, solver).

## Golden rules

1. **NEVER kill ace-server.exe externally (Task Manager / `taskkill` / `Stop-Process`) while the Node server is running.** The Node server auto-respawns the engine on non-zero exit (the exit handler in `startAceServer()`, [server/src/services/aceEngineProcess.ts:280-339](../../../server/src/services/aceEngineProcess.ts)); if crashes are spaced more than 30 s apart, the crash limiter's window resets and the respawn loop continues indefinitely, holding file locks on the exe. Stops made through the server itself (`stopAceServer()`: shutdown, restart, training preprocess) never respawn. To rebuild the engine, use `dev-rebuild.bat` at the repo root — it shuts the whole app down cleanly first, then builds. To just stop the engine, use `Invoke-RestMethod -Method Post http://localhost:3000/api/shutdown` (kills engine + server + Vite).
2. **Never rebuild the C++ engine via `engine/build.cmd` directly, under any circumstances** — you cannot reliably tell whether the app is running; same respawn/file-lock reason. `dev-rebuild.bat` wraps it safely (and is a harmless no-op shutdown when nothing runs).
3. **Never `cmake --build . --clean-first`** — CUDA kernel recompilation takes 20+ minutes. For stale `.obj` problems, delete only `engine/build/acestep-core.dir/` and `engine/build/Release/acestep-core.lib`.
4. **Do not assume the engine is dead because HTTP to :8085 hangs.** Requests to a busy engine can go unanswered during DiT steps, adapter precompute or VAE decode; the client allows 30 s per poll and the watchdog treats a failed poll as blindness, not a stall ([server/src/services/aceClient.ts:6-21](../../../server/src/services/aceClient.ts), `pollUntilDone.ts`). Check the tail of `ace_engine.log` for advancing `[DiT] Step N/M` lines before declaring it hung.
   <!-- TODO(verify): aceClient.ts:6-8 blames "single-threaded httplib", but the vendored cpp-httplib defaults to a thread pool (CPPHTTPLIB_THREAD_POOL_COUNT, at least 8) and GPU work runs on one worker thread (hot-step-server.cpp worker_main). Why polls stall is unconfirmed. -->
5. **Do not delete or "clean up" log folders or generated test audio** — the user verifies experiments by ear and by log history. Logs are the only forensic record (gen logs especially: they contain the exact request JSON needed for repro).
6. **Do not visually verify the UI with a browser agent** — ask the user to check. Hitting API endpoints programmatically is fine.
7. **Windows PowerShell syntax**: separate commands with `;`, never `&&`.

## The log session layout

One folder per Node-server session, named so name-sort = time-sort. Created at server boot by `initLogger()` ([server/src/services/logger.ts:45](../../../server/src/services/logger.ts)). The logs folder can also hold loose files that are not sessions (stray build logs, for one), and those sort after the dated folders, so filter on the `YYYY-MM-DD_HH-MM-SS` name when picking the newest session:

```
<data root>/logs/YYYY-MM-DD_HH-MM-SS/
  node_console.log            all Node stdout+stderr, mirrored transparently
  ace_engine.log              raw ace-server child stdout+stderr
  generations/
    gen_<jobId>_<taskType>.log   one per generation (and per re-post-process or Stem Studio extract); jobId = server-side UUID
  training/
    <kind>_<jobId>.log           one per training job, written live: `> $ <trainer command>`,
                                 every job log line (INFO/WARN/ERROR), the trainer's raw
                                 stdout/stderr (`> ...`), and `END | <status>`
```

Facts you must know before reading them:

- **`node_console.log` contains every engine line EXCEPT the filtered GGML noise patterns** (which appear only in `ace_engine.log` — `startAceServer()` in aceEngineProcess.ts:247-274 gates console output on `isNoise()` but writes `ace_engine.log` unconditionally). Engine lines appear prefixed `[ace-server] `, interleaved with server lines in real arrival order — it is the **only** file where engine and server output are time-ordered relative to each other.
- **`ace_engine.log` and `node_console.log` have NO per-line timestamps.** `gen_*.log` and `training/*.log` lines carry local-time ISO timestamps with the UTC offset (`2026-07-01T11:28:33.514+01:00 | INFO    | ...`, `isoTimestamp()` in logger.ts:32-39). `node_console.log` gets one at its first line only (`[Logger] Session started at ...`).
- **`gen_*.log` is buffered in RAM and only written to disk when the generation completes or fails** (`finishGenerationLog` / `failGenerationLog`, logger.ts:173-212 — both do a single `fs.writeFileSync`). Failed and cancelled generations DO get their log written. **If Node itself crashes or is hard-killed mid-generation, the gen log is never written.** A missing `gen_*.log` for a generation you know started = Node died mid-flight; fall back to `node_console.log`. Training logs are the opposite: written line by line as the job runs (`logTraining`, logger.ts:104-117).
- `<taskType>` in the filename depends on the backend. ACE-Step uses the engine request's `task_type`: `text2music` (default), `cover`, `cover-nofsq`, `repaint`, `lego`, `extract`, `complete` (services/backends/ace/generate.ts:70-76). The other backends use `mm3-text2music`, `yue2-text2music` and `sa3-text2music`; post-processing re-runs use `postprocess`; Stem Studio writes `gen_<jobId>_<track>_extract.log`. The final-failure path in the queue writes taskType `'unknown'` (routes/generate.ts:257), but only for a throw that escaped the backend, and it writes nothing unless that backend had already opened the job's log, so a `gen_<id>_unknown.log` is rare.
- Repetitive GGML noise (`CUDA graph warmup`, `CUDA Graph id`, `ggml_backend_cuda_graph_compute`) is dropped **at the engine source since 2026-07-17**: `acestep_ggml_log` (engine/src/backend.h) discards all GGML DEBUG-level messages (set `HOTSTEP_GGML_DEBUG=1` to pass them through) and digit-insensitively dedups consecutive near-identical lines, so these no longer reach ANY log file. The Node-side filters (`isNoise()` in aceEngineProcess.ts, `ENGINE_NOISE` in [server/src/routes/logs.ts:27-31](../../../server/src/routes/logs.ts)) remain as belt-and-braces for older engine binaries. If you need CUDA-graph-layer logging, use the env var.
- Live tail without touching files: `GET http://localhost:3000/api/logs` is an SSE stream backed by a 2000-line ring buffer, each line tagged `source: 'engine' | 'server'` (logs.ts:21-80); `?after=<id>` skips the backlog up to that line id.

## Which log to open first, per symptom

| Symptom | Open first | Then | Looking for |
|---|---|---|---|
| Generation failed (UI error) | Newest `<data root>/logs/<session>/generations/gen_<uuid>_*.log` — last line is `GENERATION FAILED: <reason>` | `node_console.log` around that job; `ace_engine.log` for C++ detail | The failure reason string (table below) |
| Engine crash | `node_console.log` | `ace_engine.log` tail | `[ace-server] Process exited with code N` + the FATAL/assert lines just before it |
| Server 500 / API error | `node_console.log` | — | Express stack traces; `[Server] Uncaught exception` / `Unhandled rejection` (logged and swallowed — process keeps running, index.ts:526-531) |
| Startup failure | `node --version` FIRST, then `node_console.log` | `ace_engine.log` | **Node 20 to 24 LTS, 24 recommended** (`engines` in server/package.json enforces `<25`; use matching native modules). Then: `[Server] ace-server not found at:`, CUDA-runtime download banner, `No models installed — waiting for a download in the Model Manager.`, `Crashed 3 times within 30s — giving up`, DB errors |
| Hang / no progress | `GET /api/generate/queue` (live), then `node_console.log` tail | `ace_engine.log` tail | Last engine line = the wedged phase. The stall watchdog (120 s on a stage that ticks `Step n/total`, 15 min on quiet stages) usually converts hangs into a `Generation stalled` failure on its own |
| No gen log exists at all | `node_console.log` | — | Node died mid-generation (gen logs only flush at the end) |

No session folder at all → the server never reached `initLogger()`; run `npx tsx src/index.ts` from `server/` and read the terminal directly.

## Triage procedure

1. Find the newest session and tail the logs (repo root, PowerShell 7). The first line asks the data-folder resolver where the logs live:
   ```powershell
   Push-Location server; $logs = (npx tsx scripts/data-paths.ts | Out-String | ConvertFrom-Json).children.logs; Pop-Location
   $s = (Get-ChildItem $logs -Directory | Where-Object Name -match '^\d{4}-\d{2}-\d{2}_' | Sort-Object Name -Descending | Select-Object -First 1).FullName
   Get-Content "$s\node_console.log" -Tail 80
   Get-Content "$s\ace_engine.log" -Tail 60
   Get-ChildItem "$s\generations" | Sort-Object LastWriteTime -Descending | Select-Object -First 3
   ```
2. If a generation failed, read its gen log bottom-up: the last line is `GENERATION FAILED: <reason>`; the top of the file embeds the **full resolved engine request JSON** (seed, adapters, solver, scheduler — `logGenerationParams`, logger.ts:158-167). That JSON is ground truth for reproduction.
3. Match the reason against the failure-signature tables below.
4. If the reason is generic (`Generation failed on ace-server`), the real cause is engine-side only — grep the session:
   ```powershell
   Select-String -Path "$s\node_console.log" -Pattern "exited with code|Crashed|FATAL|CUDA error"
   ```
5. For live triage while the app runs:
   ```powershell
   Invoke-RestMethod http://localhost:3000/api/health | ConvertTo-Json -Depth 4
   Invoke-RestMethod http://localhost:3000/api/generate/queue
   ```
   `/api/health` reports `aceServer.status` (`ok`/`disconnected`) and `engine.{ready, bootStatus, noModels, backend, device}` ([server/src/routes/health.ts:33-82](../../../server/src/routes/health.ts)). Remember rule 4: a `disconnected` aceServer during heavy compute can be a busy engine, not a dead one.
6. To unwedge a stuck queue without restarting:
   ```powershell
   Invoke-RestMethod -Method Post http://localhost:3000/api/generate/reset-queue
   ```
   This fails all non-terminal jobs with `Queue reset by user` and drains the pending GPU lane (`resetGpuLane()`, routes/generate.ts:674-705). Also available: `POST /api/generate/cancel/:id`, `POST /api/generate/cancel-all`.
7. To search all history for a pattern:
   ```powershell
   Select-String -Path "$logs\*\generations\*.log" -Pattern "GENERATION FAILED" | Select-Object -Last 10
   ```

## Failure signatures — gen-log `GENERATION FAILED: <X>`

| Signature | Cause / next step |
|---|---|
| `Cancelled by user` | Benign — user hit cancel (each backend's catch, e.g. services/backends/ace/generate.ts:1379). Most common "failure" in history. |
| `Generation stalled — no progress for Ns (last stage: "...")` | The stall watchdog in `pollUntilDone()` (services/generation/pollUntilDone.ts) fired. The window is 120 s on a stage that ticks (`...: Step n/total`) and 15 min on quiet stages (model loads, adapter merge, VAE decode); YuE2 NAR gets more by chunk size. Before failing, it asks the engine: a job the engine reports `done` is kept, and a ticking stage the engine reports still running gets the quiet window. The quoted stage names the wedged phase (e.g. `"Loading CondEnc..."`); check `ace_engine.log` tail for what the engine was doing. |
| `Generation timed out (N min limit)` | Wall-clock watchdog: default 45 min when the request has no `generationTimeoutMinutes`, clamped to 5–360 (pollUntilDone.ts). Settings offers 10 min to 6 hours, default 30. Huge duration/steps, or a first-run TensorRT engine build without a cached engine. A render that finishes right at the limit is kept. |
| `Generation lost: the engine crashed mid-render (see the engine log)` | The engine process exited while the job ran (`engineCrashSeq()` changed, #179). Find the `Process exited with code` line in `node_console.log` and the engine lines before it. |
| `Generation lost: the engine no longer knows this job (it restarted mid-render)` | Three 404s in a row from the engine's `/job`: it restarted (a training stop, a crash) and the new process never had this job. |
| `Generation failed on ace-server` | Engine set job status `failed` without crashing (pollUntilDone.ts). Deliberately generic — **the real reason is only in `ace_engine.log`**; grep it for `FATAL`, `[Server]`, `failed`. |
| `Cancelled by ace-server` | The engine reported the job `cancelled` on a poll (pollUntilDone.ts). A user cancel normally shows as `Cancelled by user` instead. |
| `...POST /synth... failed (413)` | Engine rejected the upload: `audio exceeds max duration (10 min)` or `src_latents exceeds max frames` ([engine/tools/hot-step-server.cpp:2498, 2526](../../../engine/tools/hot-step-server.cpp)); global payload cap is 256 MB (hot-step-server.cpp:3434). Source audio for a cover/repaint is too long. |
| `Engine not ready: <status>` (HTTP 503, never reaches a gen log) | Bootstrap incomplete (CUDA DLL download still running), no models installed yet, or the crash limiter gave up (`submitGeneration()`, routes/generate.ts:351-357). Check `node_console.log`. |
| `Engine is paused for training preprocessing — try again when the job finishes` (HTTP 503) | A training job stopped the engine to own the GPU (`isEngineSuspended()`, routes/generate.ts:347-349). Wait for the job. |
| `Queue reset by user` | Someone hit `/api/generate/reset-queue`. Benign. |

## Failure signatures — engine side (`ace_engine.log` / `[ace-server]` lines)

| Signature | Cause |
|---|---|
| `[VAE] FATAL: tensor 'decoder.conv1.weight_v' not found in safetensors` then process exit | Wrong or corrupt VAE model file (real occurrence 2026-06-04). |
| `[Server] FATAL: LM load failed` / `[Server] FATAL: synth load failed` | Model file missing/corrupt or OOM during load (hot-step-server.cpp:1133, 1574). Job goes to `failed`; the engine process usually survives. |
| `[Server] ERROR: no models found in <dir>` then exit code 1 | Empty models folder (no ACE-Step model and no MM3/YuE2 weights). Not a crash since 2026-10-18: the server logs `[ace-server] No models installed — waiting for a download in the Model Manager.`, spends no restart budget, sets `engine.noModels: true` in `/api/health` with bootStatus `No models installed — download one in the Model Manager`, and starts the engine again once a Model Manager download finishes or a top-level `.gguf` that has stopped growing appears in the models folder (checked every 5 s; aceEngineProcess.ts:44-68, 298-307). |
| `[Server] ERROR: no usable pipeline, synth missing: <DiT, Text-Enc, VAE>` then exit code 1 | An incomplete ACE-Step set (for example a VAE alone) with no LM and no other family's weights (hot-step-server.cpp:3317-3347). This one **is** counted as a crash, so three in 30 s trip the crash limiter; the engine is retried when the next download finishes. |
| `[Safetensors] Cannot open <path>/adapter_model.safetensors` | Usually **benign filename probing** — the adapter loader tries candidate filenames. Thousands appear in healthy sessions. Only meaningful if the adapter then actually fails to load. |
| `CUDA error: ...` (any) | GGML/CUDA fault; typically followed by process exit and respawn. |
| `Process exited with code 3221225786` (0xC000013A) | Console Ctrl+C / window closed — usually deliberate. |
| `Process exited with code 3221226505` (0xC0000409) | Fail-fast / abort / stack-buffer-overrun — a genuine C++ crash. Get the engine lines immediately preceding it. |
| `Process exited with code 1, signal null` **right after** `[Shutdown] Killed ace-server PID ...` or `[Server] Shutting down...` | Sessions before 2026-09-21 only: **benign**, `taskkill /F` yields exit code 1, not a signal. Since then, shutdown, restart and training stops go through `stopAceServer()` (`[Server] Stopping ace-server (pid N) — <reason>`), which marks the stop deliberate, so the exit handler prints no exit line at all. In a current session every `Process exited with code` line is an unexpected exit. |
| `[ace-server] Restarting in 3 seconds... (crash N/3)` | Routine supervised respawn — dozens exist in healthy log history. Investigate the crash cause, not the respawn itself. |
| `[ace-server] Crashed 3 times within 30s — giving up.` | Crash limiter tripped (aceEngineProcess.ts:320-327). The message that follows still says "usually no models downloaded yet", but a truly empty models folder no longer gets here (see `ERROR: no models found` above). Look at the engine lines before each exit: an incomplete model set (`ERROR: no usable pipeline`), a DLL missing from the engine directory (cuBLAS runtime on portable CUDA builds; reconnect to the internet and restart, or re-extract the release zip), or a startup crash. `engineReady` becomes false and `/api/generate` returns 503; the engine starts again by itself when a model download finishes. |
| All solvers/schedulers/guidance behave identically or ignore settings, logs completely clean — especially after an upstream acestep.cpp sync | The `hot-step-sampler.h` hook in `pipeline-synth-ops.cpp` was silently overwritten (compiles fine, all plugins dead — **invisible in logs**). Run `engine/verify-hooks.ps1`; see the upstream-sync skill. |

## How the Node server supervises ace-server

The child lifecycle lives in [server/src/services/aceEngineProcess.ts](../../../server/src/services/aceEngineProcess.ts), moved out of `index.ts` so training jobs can stop and restart the engine.

- **Spawn**: `startAceServer()` (aceEngineProcess.ts:149-347) launches `config.aceServer.exe` with `--models`, `--host`, `--port` (default **8085**, [server/src/config.ts:217](../../../server/src/config.ts)) plus optional `--adapters`, `--user-plugins` (when a data folder is active), `--keep-loaded`, `--noise-profile`, `--draft-lm`, `--vae-chunk`, `--vae-overlap`. The exe is auto-detected among `engine/ace-server.exe`, `engine/build/Release/ace-server.exe`, `engine/build/ace-server.exe`, `engine/build/Debug/ace-server.exe` (config.ts:133-139); override with `ACESTEPCPP_EXE` in `.env`.
- **Respawn**: child exit with non-zero code (and signal not SIGTERM/SIGINT) → respawn after 3 s. **Crash limiter**: 3 crashes within a rolling 30 s window → give up, set `engineReady=false` with bootStatus `Engine crashed 3 times — download a model in the Model Manager, or check the logs` (aceEngineProcess.ts:80-86, 308-338). Crashes spaced >30 s apart reset the window, so slow-cycle crashes (e.g. crash-on-first-request) still loop forever — hence Golden rule 1. An exit with `ERROR: no models found` is not counted (see the engine table above).
- **Deliberate stops**: `stopAceServer()` cancels any pending respawn and marks the stop before it kills (tree kill on Windows, SIGTERM then SIGKILL elsewhere), so the exit handler returns without respawning; `restartAceServer()` respawns and waits up to 90 s for `/health`, then keeps watching and marks the engine ready if it answers later. While a training preprocess job holds the stop, `isEngineSuspended()` is true and `/api/generate` answers 503 with the training message. Each unexpected exit bumps `engineCrashSeq()`, which is what makes a running poll fail with `Generation lost: the engine crashed mid-render`.
- **Clean shutdown**: Ctrl-C/SIGTERM → `shutdown()` (index.ts:489-522) kills training children, then `stopAceServer()`. `POST /api/shutdown` ([server/src/routes/shutdown.ts:131-164](../../../server/src/routes/shutdown.ts)) stops the audio queue, kills training children, stops ace-server through `stopAceServer()` (waits up to 15 s), kills Vite on :3000 only when `dev.bat` set `HOT_STEP_DEV` and the pid really is a Node process running Vite, then tree-kills its own launcher after checking the parent pid is still a launcher process. `POST /api/restart` writes the restart marker (`RESTART_MARKER_PATH`, `<repo>/.restart-requested` in dev) and refuses with 409 while a data-folder move waits on busy work; in dev mode `server/restart-loop.cmd` loops on the marker and relaunches Node (`LAUNCH.bat` is the end-user prod equivalent).
- **`dev-rebuild.bat`** = POST `/api/shutdown` → wait up to 10 s for ace-server.exe to die (force-kill at 10 s, abort at 15 s) → `engine\build.cmd`, then `BUILD FAILED` and exit 1 if the build failed. It does **not** restart the app — run `dev.bat` afterwards (agents never use `LAUNCH.bat`).
- **Bootstrap gate**: on portable Windows CUDA builds, the server first downloads cuBLAS DLLs from HuggingFace before the engine is usable; until then `POST /api/generate` returns 503 `Engine not ready: <bootStatus>`. No internet → CPU-only start with a banner in `node_console.log`.

## Generation orchestration facts that change your diagnosis

- **One GPU lane**: only one generation runs at a time (`enqueueGeneration`, routes/generate.ts:191-273, on `runOnGpuLane()` in services/generation/gpuLane.ts), because engine log parsing is a global untagged pub/sub. The lane is shared with post-processing re-runs and other GPU work, so `GET /api/generate/queue` reports its `owner` label and whether it is `draining`.
- **The auto-retry loop is dead code for generation failures**: `enqueueGeneration` retries with a fresh seed per the backend's `policy.retry` (`maxAttempts: 2, reseedOnRetry: true` for every backend), but each backend's generate function catches its own errors and records the failure on the job without rethrowing (ACE: `runAceGeneration`, services/backends/ace/generate.ts:1375-1387), so the retry never fires for a failed render. The throws that do escape (`translateParams`, an unregistered backend) happen before any engine submission. **One UI-visible failure = exactly ONE engine attempt**, and the seed in the gen log's single params block is directly trustworthy for repro. Zero `[Retry]` lines exist across 3,666 session folders (checked 2026-10-23). If retry is ever *intended* behavior, the swallowed rethrow is the server bug to fix.
- **Stall watchdog**: `pollUntilDone()` (services/generation/pollUntilDone.ts) polls the engine every 500 ms. A stage that ticks `Step n/total` may stay unchanged for 120 s, a quiet stage (model load, adapter merge, VAE decode) for 15 min; the engine's own `phase`/`phase_step`/`adapter_progress` changes count as progress too. Past the window it asks the engine before cancelling (see `Generation stalled` above). First-run TensorRT builds also feed the stage string (`[TRT-WARN]`, `[TRT-ERROR]` and `[DiT-TRT] Engine build in progress` lines, services/backends/ace/generate.ts:778-779) so a long TRT compile doesn't trip it.
- **Transient poll errors are normal**: `[Generate] Poll error for job <id>: ... (will retry)` (pollUntilDone.ts) just means the busy engine missed a 30 s poll window. Three 404s in a row are not transient: that is the `Generation lost` failure.
- **Fine-grained engine phase** is exposed at `GET /api/generate/status/:id` as `ace_phase` — one of `queued, loading_text_enc, encoding_text, loading_cond_enc, encoding_cond, loading_dit, loading_adapter, adapter_precompute, dit_inference, loading_vae, vae_decode, encoding_output, done, failed, cancelled` (aceClient.ts:258-273), or `plan`, `semantic`, `nar` for YuE2 (`job_phase_str()` in hot-step-server.cpp). `ace_phase_progress` carries `"step N/M"`. These fields are optional on the wire for older engine builds; absence is not an error.
- **Job TTL**: the in-memory job map prunes terminal jobs after 1 h (routes/generate.ts:85-98). `/status/:id` then answers from the job ledger in SQLite (`GenerationJobLedger`, services/generation/jobLedger.ts, rows kept a week); a 404 means the ledger has no row either. A job that a server restart killed reports `failed` with `interrupted: true` and `Generation failed: the server restarted while this job was queued or running. It was not resubmitted.` The song row and gen log persist regardless.
- **Post-processing failures are non-fatal**: mastering / Spectral Lifter / PP-VAE / Whisper / LRC / cover-art failures log a `WARNING` and the job still succeeds with raw audio. A "succeeded" job can still have WARNINGs worth reading in its gen log.

## Correlating a gen_<uuid> log with engine + node logs

There are **two job-ID namespaces**: the server UUID (the `gen_<uuid>` filename; SSE lines show only its first 8 chars as `[Gen:xxxxxxxx]`, logger.ts:152) and the engine's own job id (`ace_job_id` in `/status/:id`; returned by the engine's `/lm` and `/synth`). The engine id is NOT in the gen filename.

1. Open `gen_<uuid>_*.log`; note the ISO timestamps of the LM/synth submission lines and the final `GENERATION FAILED` line.
2. `ace_engine.log` has no timestamps, so pivot through **`node_console.log`**: search for `[Generate] Job <uuid>` — for ACE-Step the submission line logs ditModel/synth_model/emb_model/seed/lm_seed/source (services/backends/ace/generate.ts:66) and the failure logs `[Generate] Job <uuid> failed:` (:1384); the other backends tag the same lines with their name, e.g. `[Generate] Job <uuid> (minimax-m3) failed:`. The block of `[ace-server] ...` lines between your job's submit and fail lines IS that generation's engine output.
3. If you need the exact region of `ace_engine.log` (e.g. to see noise-filtered lines), grep it for a distinctive engine line found in step 2 (the last `[DiT] Step N/M`, a model-load line) and read forward from there.
4. Repro comes from the params JSON at the top of the gen log — it is the fully resolved request actually sent to the engine.

Dev-mode note: under `dev.bat` (tsx watch), every source-change auto-restart creates a **new** session folder. A burst of near-identical `<data root>/logs/` folders seconds apart means tsx was restarting, not that the app was crashing.

## Key files

| Path | Role |
|---|---|
| `server/src/services/logger.ts` | Session folders, console/engine mirroring, gen-log buffering & flush, live training logs |
| `server/src/services/aceEngineProcess.ts` | Engine spawn, respawn + crash limiter, no-models wait, `stopAceServer`/`restartAceServer`, `engineCrashSeq` |
| `server/src/index.ts` | CUDA DLL bootstrap, first `startAceServer()` call, SIGINT/SIGTERM `shutdown()` (:489), uncaught-error handlers |
| `server/src/routes/generate.ts` | Queue and endpoints: `runGeneration` :156, `enqueueGeneration` (retry) :191, `submitGeneration` :344, status/queue/reset :482-705 |
| `server/src/services/backends/ace/generate.ts` | ACE-Step pipeline (`runAceGeneration`): LM phase, synth, timing table, gen-log start/finish |
| `server/src/services/generation/pollUntilDone.ts` | Engine poller: stall and wall-clock watchdogs, crash and 404 detection |
| `server/src/services/generation/gpuLane.ts` | The single GPU lane every generation and re-run waits on |
| `server/src/services/aceClient.ts` | Typed engine HTTP client; timeouts; `AceJobPhase` list |
| `server/src/routes/logs.ts` | `GET /api/logs` SSE, 2000-line ring buffer, noise filter |
| `server/src/routes/health.ts` | `GET /api/health` |
| `server/src/routes/shutdown.ts` | `POST /api/shutdown` (engine via `stopAceServer`, then Vite, then self), `POST /api/restart` marker |
| `server/src/engineState.ts` | `engineReady` / `engineBootStatus` gate |
| `server/src/config.ts` | Engine exe auto-detect, port 8085, env-var overrides |
| `engine/tools/hot-step-server.cpp` | C++ engine HTTP server; FATAL emit sites; 413/payload limits |
| `dev-rebuild.bat` | The ONLY sanctioned way to rebuild the engine, always |
| `dev.bat` | The launcher agents use: dev mode (Vite :3000 HMR + tsx watch on :3001), started detached |
| `LAUNCH.bat` | End-user prod launcher (Node :3001 serving `ui/dist/`) — agents don't run it |

## Institutional knowledge

- **VALIDATED**: Killing ace-server externally while Node runs causes a respawn/file-lock loop. The crash limiter (3 crashes / 30 s) caps *fast* crash loops, but slow-cycle crashes (>30 s apart) reset the window and loop indefinitely — the "infinite respawn" warning in AGENTS.md predates the limiter but the rule stands: rebuild only via `dev-rebuild.bat`.
- **VALIDATED**: gen logs are lost if Node dies mid-generation (flush-on-completion only). Failed/cancelled generations DO get their gen log.
- **VALIDATED**: `ace_engine.log` and `node_console.log` carry no per-line timestamps; only gen and training logs do. All time correlation goes through `node_console.log` line ordering.
- **VALIDATED**: in sessions before 2026-09-21, `Process exited with code 1, signal null` after a shutdown line is a taskkill artifact, not a crash (dozens in history). Respawns themselves are routine (`Restarting in 3 seconds... (crash N/3)` appears across many healthy sessions).
- **VALIDATED**: `[Safetensors] Cannot open ...adapter_model.safetensors` is usually benign filename probing.
- **VALIDATED**: one UI failure = exactly one engine attempt. The retry loop in `enqueueGeneration` is unreachable for generation failures (each backend's generate function swallows its own errors); no `[Retry]` line has ever appeared in log history. The gen-log params block is the real, only attempt.
- **UNVALIDATED**: exact engine-side text accompanying `status:'failed'` for every failure class — `hot-step-server.cpp` has many FATAL/failed emit sites; the tables above list only those with real log occurrences. Linux/macOS shutdown paths exist in code (`stopAceServer()`'s SIGTERM-then-SIGKILL branch; `killSelf()` is Windows-only) but are unconfirmed by logs. `gen_<id>_unknown.log` files have a code path but none have been observed on disk.
- Draft-LM speculative decoding is **disabled by default** (config.ts:232-237) — `--draft-lm` is only passed if `ACESTEPCPP_DRAFT_LM` is set. Don't chase it as a crash suspect unless that env var is set.

## Deeper reading

- [reference.md](reference.md) (this folder) — copy-paste triage command pack, status JSON shapes, engine phase reference, watchdog timing table.
- [AGENTS.md](../../../AGENTS.md) — build/git rules that constrain what you may do while debugging (`CLAUDE.md` only imports it).
- [engine/docs/ARCHITECTURE.md](../../../engine/docs/ARCHITECTURE.md) — engine internals, request JSON, generation modes.
- [docs/dev/plugins-authoring.md](../../../docs/dev/plugins-authoring.md) — Lua solver/scheduler plugins (a bad plugin can wedge DiT inference).
- `docs/plans/` — internal investigation docs. **Gitignored, local-only — may be absent on a fresh clone.**
