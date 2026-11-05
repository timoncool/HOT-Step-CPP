# debugging-runtime — reference

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

Companion to [SKILL.md](SKILL.md). Copy-paste command pack, wire formats, and timing constants. All line refs verified 2026-10-23.

## Triage command pack (PowerShell 7, repo root)

```powershell
# Where the logs live (data-folder resolver; read-only)
Push-Location server; $logs = (npx tsx scripts/data-paths.ts | Out-String | ConvertFrom-Json).children.logs; Pop-Location

# Newest session folder (session names only; the logs folder can hold loose files too)
$s = (Get-ChildItem $logs -Directory | Where-Object Name -match '^\d{4}-\d{2}-\d{2}_' | Sort-Object Name -Descending | Select-Object -First 1).FullName

# Tails
Get-Content "$s\node_console.log" -Tail 80
Get-Content "$s\ace_engine.log" -Tail 60

# Recent generation logs
Get-ChildItem "$s\generations" | Sort-Object LastWriteTime -Descending | Select-Object -First 3

# All failures across history
Select-String -Path "$logs\*\generations\*.log" -Pattern "GENERATION FAILED" | Select-Object -Last 10

# Crash forensics in current session
Select-String -Path "$s\node_console.log" -Pattern "exited with code|Crashed|FATAL|CUDA error"

# Trace one job end-to-end (paste the full UUID from the gen filename)
Select-String -Path "$s\node_console.log" -Pattern "Job <uuid>"

# Live health / queue (dev app: Vite :3000 proxies /api to the Node server on :3001)
Invoke-RestMethod http://localhost:3000/api/health | ConvertTo-Json -Depth 4
Invoke-RestMethod http://localhost:3000/api/generate/queue

# Engine direct (:8085) — WARNING: may hang mid-compute; that is normal, not a crash
Invoke-RestMethod http://localhost:8085/health
Invoke-RestMethod http://localhost:8085/jobs      # engine-side job list: id, status, phase, phase_step, phase_total

# Unwedge
Invoke-RestMethod -Method Post http://localhost:3000/api/generate/cancel-all
Invoke-RestMethod -Method Post http://localhost:3000/api/generate/reset-queue

# Clean full shutdown (engine + Vite + Node)
Invoke-RestMethod -Method Post http://localhost:3000/api/shutdown
```

If the app never started, restart it detached with `dev.bat` and read the newest
`<data root>/logs/<session>/node_console.log`. If there is no session folder at all, run the Node
server in the foreground for one diagnostic pass to see the error (this bypasses dev mode
— do not use it as the normal way to run the app):

```powershell
Set-Location server; npx tsx src/index.ts
```

## Watchdog & timing constants (server/src/services/generation/pollUntilDone.ts, routes/generate.ts, services/aceEngineProcess.ts, services/aceClient.ts)

| Constant | Value | Where | Effect |
|---|---|---|---|
| Poll interval | 500 ms | pollUntilDone.ts | Engine `GET /job?id=N` cadence |
| Stall timeout, ticking stage | 120 s with no stage/progress change on a stage matching `: Step n` | pollUntilDone.ts `STALE_TIMEOUT_MS` | Engine asked first; then → `Generation stalled — no progress for Ns (last stage: "...")` |
| Stall timeout, quiet stage | 15 min (model loads, adapter merge, VAE decode, `Synthesizing...`; also a ticking stage the engine says is still running) | pollUntilDone.ts `STALE_TIMEOUT_QUIET_MS`, `STALE_TIMEOUT_VAE_MS` | Same failure, later |
| Stall timeout, YuE2 NAR | at least 15 min; 400 ms per submitted chunk frame plus 2 min | pollUntilDone.ts `narQuietMs` | Same failure |
| Blind grace | 10 s since the last answered poll | pollUntilDone.ts `BLIND_GRACE_MS` | No stall verdict while polls are failing; the wall clock is the backstop |
| Wall clock | 45 min default, clamped 5–360 (`generationTimeoutMinutes` in request; Settings default 30) | pollUntilDone.ts | → `Generation timed out (N min limit)`; a job the engine reports `done` is kept |
| Not-found limit | 3 consecutive 404s from `/job` | pollUntilDone.ts `NOT_FOUND_LIMIT` | → `Generation lost: the engine no longer knows this job (it restarted mid-render)` |
| Retry | UNREACHABLE for failed renders | routes/generate.ts:198-263 | `policy.retry` allows 2 attempts, but every backend's generate function swallows its own errors; expect exactly one attempt |
| Job TTL | 1 h after terminal state, pruned every 10 min | routes/generate.ts:85-98 | `/status/:id` then answers from the job ledger (rows kept 7 days), 404 only when the ledger has none |
| HTTP quick timeout | 15 s (health, props, submit) | aceClient.ts:18 | |
| HTTP poll timeout | 30 s | aceClient.ts:19 | Missed polls logged `Poll error ... (will retry)` — benign |
| HTTP result timeout | 300 s (large audio fetch) | aceClient.ts:20 | |
| HTTP SA3 timeout | 30 min (`/sa3-refine`; a first call may build a TensorRT engine) | aceClient.ts:21 | |
| Respawn delay | 3 s | aceEngineProcess.ts:329-337 | |
| Crash limiter | 3 crashes / rolling 30 s window | aceEngineProcess.ts:80-86, 308-327 | Then gives up, `engineReady=false` |
| Restart health wait | 90 s, then a 2 s watcher while the child lives | aceEngineProcess.ts:101-104, 474-512 | `restartAceServer()` returns false, the engine is marked ready later if `/health` answers |
| Log ring buffer | 2000 lines | logs.ts:21 | Backs `GET /api/logs` SSE |
| Engine payload cap | 256 MB | hot-step-server.cpp:3434 | Multipart uploads (cover/repaint source) |
| Engine audio cap | 10 min source audio | hot-step-server.cpp:2498 | 413 `audio exceeds max duration (10 min)` |

## `GET /api/generate/status/:id` response shape (routes/generate.ts:482-529)

```json
{
  "jobId": "<server uuid>",
  "status": "pending | lm_running | synth_running | saving | succeeded | failed | cancelled",
  "stage": "<human-readable stage string>",
  "progress": 0,
  "result": {},
  "error": "<message or undefined>",
  "attempts": [],
  "ace_job_id": "<engine-side job id or null>",
  "ace_phase": "<AceJobPhase or null>",
  "ace_phase_progress": "step N/M",
  "batch": null,
  "mm3_streaming": false
}
```

The live body also carries the YuE2 `batch` grouping and the MM3 fields (`mm3_interleaved`, `mm3_duration`, `mm3_takes`, `mm3_take_seeds`, `mm3_ending`), null or default on other backends. A job no longer in memory gets the ledger's body instead (`ledgerStatusBody()`, services/generation/jobLedger.ts).

`ace_phase` values (aceClient.ts:258-273, mirrors `job_phase_str()` in hot-step-server.cpp):

```
queued, loading_text_enc, encoding_text, loading_cond_enc, encoding_cond,
loading_dit, loading_adapter, adapter_precompute, dit_inference,
loading_vae, vae_decode, encoding_output, done, failed, cancelled
```

YuE2 jobs also report `plan`, `semantic` and `nar` (the engine's `YUE2_*` phases; not in the TypeScript union).

`phase`/`phase_step`/`phase_total` are optional on the wire for older engine builds (aceClient.ts:236-254) — absence is not an error. `adapter_progress` (live delta index during an adapter precompute, -1 otherwise) counts as watchdog progress.

## `GET /api/generate/queue` response shape (routes/generate.ts:650-672)

```json
{
  "depth": 1,
  "running": true,
  "owner": "generate:<job id>",
  "draining": false,
  "current": { "id": "...", "status": "synth_running", "stage": "...", "progress": 42, "age": 130, "aceJobId": "..." },
  "pending": 0
}
```

`running` and `owner` describe the GPU lane, which other GPU work (post-processing re-runs, YuE2 previews, Storm slots) also takes, so `running: true` with `current: null` can be legitimate. `age` is seconds since job creation. A `running: true, current: null` that persists after a Node error is what `reset-queue` fixes.

## `GET /api/health` response shape (health.ts:33-82)

```json
{
  "status": "ok",
  "version": "<app version>",
  "commit": "<startup commit>",
  "dirty": false,
  "build": { "tag": null, "sha": "...", "variant": null, "channel": null },
  "engineBuiltAt": "<ace-server exe mtime, ISO>",
  "aceServer": { "status": "ok | disconnected", "url": "http://127.0.0.1:8085", "version": "" },
  "server": { "port": 3001, "uptime": 1234.5 },
  "clients": 1,
  "engine": { "ready": true, "bootStatus": "...", "noModels": false, "backend": "cuda", "device": "..." }
}
```

`aceServer.status: "disconnected"` while `engine.ready: true` during a generation usually means the engine is busy computing, not dead. `engineBuiltAt` is a quick check that a rebuild produced a new exe.

## Engine spawn CLI (aceEngineProcess.ts:161-203)

```
ace-server.exe --models <dir> --host 127.0.0.1 --port 8085
               [--adapters <dir>] [--user-plugins <dir>] [--keep-loaded]
               [--noise-profile <wav>] [--draft-lm <path>] [--vae-chunk N] [--vae-overlap N]
```

`--user-plugins` is the data folder's `plugins/` overlay, passed only when a data folder is active. Exe candidates in order (config.ts:133-138): `engine/ace-server.exe` (portable), `engine/build/Release/ace-server.exe` (VS), `engine/build/ace-server.exe` (Ninja), `engine/build/Debug/ace-server.exe`. Env overrides in `.env`: `ACESTEPCPP_EXE`, `ACESTEPCPP_MODELS`, `ACESTEPCPP_PORT`, `ACESTEPCPP_HOST`, `ACESTEPCPP_ADAPTERS`, `ACESTEPCPP_KEEP_LOADED`, `ACESTEPCPP_NOISE_PROFILE`, `ACESTEPCPP_DRAFT_LM`, `ACESTEPCPP_VAE_CHUNK` (default 1024), `ACESTEPCPP_VAE_OVERLAP` (default 64), `TENSORRT_LIBS`, `CUDA_VISIBLE_DEVICES`.

## Windows exit codes seen in `Process exited with code N`

| Code (decimal) | Hex | Meaning |
|---|---|---|
| 1 | — | Before 2026-09-21, after a shutdown line: benign `taskkill /F` artifact (deliberate stops no longer print an exit line). Otherwise: engine returned error from main (e.g. VAE FATAL); `ERROR: no models found` is handled as a first run, not a crash. |
| 3221225786 | 0xC000013A | Console Ctrl+C / window closed — usually deliberate |
| 3221226505 | 0xC0000409 | Fail-fast / abort / stack-buffer-overrun — genuine C++ crash; read the preceding engine lines |

## Anatomy of a gen log

```
<ISO ts> | INFO    | ============================================================
<ISO ts> | INFO    | GENERATION STARTED: Job <uuid>
<ISO ts> | INFO    | Task Type: text2music
<ISO ts> | INFO    | Parameters:
<ISO ts> | INFO    | {  ...full resolved engine request JSON: seed, adapters, solver, scheduler...  }
<ISO ts> | INFO    | ============================================================
... phase lines, [Timing] table on success (ACE-Step) ...
<ISO ts> | INFO    | GENERATION COMPLETED.        ← success
<ISO ts> | ERROR   | GENERATION FAILED: <reason>  ← failure/cancel
```

`<ISO ts>` is local time with the UTC offset, e.g. `2026-10-23T19:37:13.513+01:00`. Buffered in `generationBuffers` (logger.ts:24) and flushed in one `writeFileSync` by `finishGenerationLog`/`failGenerationLog` (logger.ts:173-212). Each gen log contains exactly one params block. (The retry loop in routes/generate.ts is unreachable for failed renders — and even if it fired, attempt 2 calls `startGenerationLog` again, which starts a fresh buffer for the same filename, so a two-attempt log is structurally impossible.)

## Known-good real-world log excerpts (from `<data root>/logs/` history)

- Crash + supervised respawn (2026-06-04, wrong VAE file):
  ```
  [ace-server] [VAE] FATAL: tensor 'decoder.conv1.weight_v' not found in safetensors
  [ace-server] Process exited with code 1, signal null
  [ace-server] Restarting in 3 seconds... (crash 1/3)
  ```
- Benign shutdown artifact (sessions before 2026-09-21, when shutdown still killed the engine by port):
  ```
  [Shutdown] Killed ace-server PID 1234
  [ace-server] Process exited with code 1, signal null
  ```
- Clean stop in a current session (no exit line follows):
  ```
  [Server] Shutdown requested via API
  [Server] Stopping ace-server (pid 1234) — Server shutting down
  ```
