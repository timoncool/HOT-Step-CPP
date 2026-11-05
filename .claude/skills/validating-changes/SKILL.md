---
name: validating-changes
description: Defines the verification bar every HOT-Step CPP change must clear before being called done — per-tier checks (TypeScript, C++ engine, UI, audio, end-to-end smoke generation) and honest result reporting. Use when finishing any code change, deciding whether work is "done", running a smoke/e2e test, or reporting test/validation results.
---

# Validating Changes to the Required Standard

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

HOT-Step CPP is a 3-tier app: a C++17/CUDA/GGML **engine** (`engine/`, spawned as
`ace-server.exe` on port 8085), a Node/TypeScript/Express **server** (`server/src/`,
port 3001), and a React 19 **UI** (`ui/src/`, Vite dev server on port 3000). Agents run the
app with `dev.bat`, detached, and verify through `http://localhost:3000`; :3001 is the
end-user prod address. A change
is NOT done until it clears the bar for **every tier it touches**. This skill defines
those bars, how to run a headless end-to-end smoke generation, and how to report
results honestly.

All commands are Windows PowerShell (`;` as separator, never `&&`).
Deep API/log detail lives in [reference.md](reference.md) in this folder.

## When to use this skill

- You just finished a code change and need to decide whether it is "done".
- You need to verify a C++ engine change, a server/UI TypeScript change, or a
  Lua plugin change actually works.
- You need to run an end-to-end generation without the UI (headless smoke test).
- You are about to report validation results to the user.

## Golden rules (hard constraints — each prevents expensive damage)

1. **ONLY the human can judge audio quality, by ear.** Never claim audio "sounds
   fine" or "is broken" from metrics, file size, duration, RMS, spectrograms, log
   silence, or the pipeline's quality-evaluator scores (stored in the `quality_scores`
   DB column — advisory telemetry only). Say: "generation completed mechanically;
   audio at `<path>` awaits your ear." WHY: metrics have repeatedly failed to predict
   what the user hears; false verdicts destroy trust and mislead debugging.

2. **PRESERVE EXPERIMENT ARTIFACTS.** Never delete generated test audio, latents,
   or other outputs because *you* predict they are bad. The human verifies by ear and
   has lost artifacts to premature cleanup before. WHY: a generation costs GPU time and
   a seed; a deleted artifact may be unreproducible evidence.

3. **Report outcomes faithfully.** Failed tests are reported as FAILED with the actual
   output/error pasted. Skipped steps are reported as SKIPPED with the reason. Never
   write hedged "should work" / "probably fine". Distinguish what you **ran and saw**
   vs what you **inferred** vs what you **did not verify**. WHY: the user makes
   release and debugging decisions off your report.

4. **Rebuild C++ only via `dev-rebuild.bat` at repo root — never `engine\build.cmd`
   directly, under any circumstances.** WHY: you cannot reliably tell whether the app
   is running; the Node server auto-respawns ace-server 3 s after a non-clean exit
   (`onExited` in `server/src/services/aceEngineProcess.ts:295-338`), and the
   respawned exe file-locks the linker output — infinite loop. `dev-rebuild.bat`
   is safe even when the app is stopped (its shutdown request is a suppressed
   no-op).

5. **Never push a `v*` tag to "test" a build.** WHY: any pushed `v*` tag triggers the
   full multi-platform Release workflow and drafts a GitHub Release. Throwaway compile
   checks use a `-CI-Test` suffix; any push still requires explicit user approval.

6. **Never `cmake --build . --clean-first`** unless the GGML/CUDA layer itself
   changed. WHY: CUDA kernel recompilation takes 20+ minutes. For stale-`.obj`
   symptoms delete only `engine/build/acestep-core.dir/` and
   `engine/build/Release/acestep-core.lib`.

7. **No `npm run build` during dev.** Type-check with `tsc` only (§ below). WHY:
   builds are slow and unnecessary until the user tests a prod (`LAUNCH.bat`) flow.

8. **Never visually verify UI with a browser agent** — ask the human; they provide
   screenshots. Browser/`Invoke-RestMethod` IS fine for non-visual API checks.
   WHY: the browser agent is too slow/unreliable in this environment.

9. **A feature is not done until a USER can run it.** Anything the app resolves
   at runtime — model weights, a data file, a binary — must be reachable by
   someone who downloaded a release, not just present on this machine. Run
   `node server/scripts/check-release-prereqs.mjs` before any push (§ Tier 6).
   WHY: this has shipped broken twice. MM3 training asked for two GGUFs that
   existed only on the dev box (#137), and the MM3 caption corpus was committed
   but never copied into the archives (#139). Both passed tsc, both built green,
   both were dead on arrival for every user.

10. **Don't clobber the user's generation queue.** Jobs run one at a time on a single
   GPU. Check `GET /api/generate/queue` before submitting a smoke generation, and
   never call `POST /api/generate/reset-queue` (force-drains everything) without
   asking. WHY: it cancels the user's in-flight work.

## Tier 1 — TypeScript (server and UI)

Bar: type-check clean. Zero errors.

> **Node 20 to 24 LTS, 24 recommended** (`engines: >=20.0.0 <25.0.0`
> in `server/package.json`). If tsc/tsx/npm fails oddly, check `node --version`
> BEFORE blaming the change — a wrong Node version produces false validation verdicts.

```powershell
cd D:\Ace-Step-Latest\hot-step-cpp\server; npx tsc --noEmit
cd D:\Ace-Step-Latest\hot-step-cpp\server; npx tsc --noEmit -p scripts/tsconfig.json   # only if server/scripts/ changed
cd D:\Ace-Step-Latest\hot-step-cpp\ui; npx tsc -b
```

- `server/package.json` defines `"typecheck": "tsc --noEmit"`, which covers
  `server/src/` only (`server/tsconfig.json` includes `src/**/*`); a touched
  script under `server/scripts/` is checked only by the second command. The UI
  uses project references (`ui/tsconfig.json` → `tsconfig.app.json` +
  `tsconfig.node.json`, both `noEmit`-safe), so `npx tsc -b` type-checks without
  emitting JS; `npx tsc --noEmit -p tsconfig.app.json` checks the app sources
  alone. A bare `npx tsc --noEmit` in `ui/` checks nothing and exits 0, because
  the root `ui/tsconfig.json` has `"files": []`.
- Unit tests sit beside the code as `*.test.ts` (`node:test`). For the server, run
  the files for the area you touched from `server/` with
  `npx tsx --test <files>`; release gate tier 0 runs every `server/src` test
  that way.
- In dev mode (`dev.bat`) the server runs under `tsx watch` (`server/package.json`
  `"dev": "tsx watch src/index.ts"`) — it auto-restarts on save. Server-side bar =
  passing tsc **plus** a clean restart visible in the newest
  `<data root>/logs/<session>/node_console.log` (no stack trace after the restart).

## Tier 2 — C++ engine

Bar: rebuild compiles, the render receipt passes, AND the engine starts clean after
relaunch (log check).

1. Recompile **immediately** after editing any `engine/src/` or `engine/tools/`
   file — don't wait to be asked:

   ```powershell
   D:\Ace-Step-Latest\hot-step-cpp\dev-rebuild.bat
   ```

   What it does (verified in the script): POSTs `http://localhost:3001/api/shutdown`
   (stops ace-server, the dev Vite server and the Node server cleanly), waits up to
   10 s for `ace-server.exe` to exit, force-kills at 10 s, aborts at 15 s, then runs
   `engine\build.cmd`, which runs `engine\verify-hooks.ps1` first and aborts on a
   broken hook. A failed build prints `[dev-rebuild] BUILD FAILED` and exits 1.
   Per AGENTS.md, never take a batch exit code as proof it ran: check the output
   for the script's own `[dev-rebuild]` lines and a fresh `ace-server.exe`
   timestamp (engine-dev-loop Procedure 1).

   Then the render receipt AGENTS.md requires: with the GPU idle, run
   `node tools/engine-smoke.mjs` (two fixed-seed `ace-synth --check` renders, no
   server; exit 1 on NaN, silence, DC offset, clipping or a wrong length) and paste
   its `[Check Batch0]` line and `ENGINE SMOKE:` verdict into the hand-off. A change
   to the checker itself also runs `--self-test`. Details: engine-dev-loop
   Procedure 4.

2. **The rebuild does NOT restart the app.** Relaunch with
   `pwsh -NoProfile -File tools\detach.ps1 dev.bat` (detached) and poll
   `http://localhost:3000/api/health` before any runtime verification. `LAUNCH.bat`
   is the end-user prod launcher — agents don't run it.

3. Clean-start check — read the newest `<data root>/logs/<session>/ace_engine.log` and confirm:
   - model registry scan ran (`[Server] Scanning models in ...` / `[Registry] ...` lines),
   - `[Server] Listening on 127.0.0.1:8085` appears,
   - no crash/exception before the Listening line,
   - `node_console.log` has **no** `[ace-server] Restarting in 3 seconds...`
     (that exact string = crash respawn, `server/src/services/aceEngineProcess.ts:329`).

   ```powershell
   $sess = (Get-ChildItem <data root>/logs | Sort-Object Name -Descending | Select-Object -First 1).FullName
   Select-String -Path "$sess\ace_engine.log" -Pattern 'Listening on'
   Select-String -Path "$sess\node_console.log" -Pattern 'Restarting in 3 seconds'
   ```

4. After any **upstream sync** (the engine is a patched fork of acestep.cpp):

   ```powershell
   powershell -File D:\Ace-Step-Latest\hot-step-cpp\engine\verify-hooks.ps1
   ```

   Exit 0 = good. AGENTS.md's table lists the four upstream files whose hooks
   break; the script checks more than that (Hooks 1 to 18, the script is the
   truth): `pipeline-synth-ops.cpp → hot-step-sampler.h` (loss is **SILENT** —
   compiles fine but all solvers/guidance/schedulers go dead),
   `model-store.h → hot-step-params.h`, `dit.h → adapter-merge.h` +
   `adapter-runtime.h`, `tools/hot-step-server.cpp → hot-step-params.h` plus each
   backend family's include and route registration and the family weights gate,
   the shared FSQ helpers, the linker sentinel `hotstep_sampler_linked_` in
   `hot-step-sampler.h`, the HOT-ggml changes, and `engine/ggml` being clean at its
   pinned commit. `engine\build.cmd` runs it before every compile.

5. Lua plugins (`engine/plugins/`) need **no** rebuild, but they load at engine
   start — relaunch the app for a new/edited plugin to appear.

## Tier 3 — UI visuals

Bar: `npx tsc -b` (or `npx tsc --noEmit -p tsconfig.app.json`) clean + Vite HMR
applies the change without console errors + **the human confirms the visuals**.
You cannot close this tier yourself. Ask the user to look and wait for their
screenshot/feedback.

## Tier 4 — Audio quality

Bar: **the human's ear.** There is no mechanical substitute — see Golden rules 1–2.
Your job ends at: generation succeeded mechanically, here is the file path
(`<data root>/library/audio/<uuid>.wav` or `http://localhost:3000/audio/<uuid>.wav` in dev), here
are the seed and params for reproduction. Then wait.

## Tier 5 — End-to-end headless smoke generation

Bar: job reaches `status: "succeeded"`, a WAV exists on disk, and the per-generation
log ends with `GENERATION COMPLETED.` (Mechanical success only — quality is Tier 4.)

Full request/response/log detail: [reference.md](reference.md).

1. **Pre-flight health** — require `engine.ready -eq $true`:

   ```powershell
   Invoke-RestMethod http://localhost:3000/api/health
   ```

2. **Check the queue is free** (don't queue behind/ahead of the user silently):

   ```powershell
   Invoke-RestMethod http://localhost:3000/api/generate/queue
   ```

3. **Get an auth token** (`POST /api/generate` returns 401 without a Bearer token;
   tokens are in-memory and reset on server restart):

   ```powershell
   $auth = Invoke-RestMethod http://localhost:3000/api/auth/auto
   $hdr = @{ Authorization = "Bearer $($auth.token)" }
   ```

4. **Submit a fast smoke request** — `skipLm: $true` skips the language-model phase
   (the "LM" that writes audio codes/metadata before the DiT diffusion transformer
   runs). Always pass `duration` explicitly: with skipLm the server defaults an
   unset duration to **120 s** (`server/src/services/backends/ace/generate.ts:84`).

   ```powershell
   $body = @{
     prompt         = 'minimal ambient techno, warm pads'
     instrumental   = $true
     skipLm         = $true
     duration       = 30
     inferenceSteps = 8
     seed           = 42      # fixed seed = reproducible; or randomSeed = $true
   } | ConvertTo-Json
   $job = Invoke-RestMethod -Method Post -Uri http://localhost:3000/api/generate -Headers $hdr -ContentType 'application/json' -Body $body
   $job.jobId
   ```

   Omit `skipLm` if your change touches the LM phase — then the smoke must
   exercise it.

5. **Poll to completion** (status values: `pending | running | lm_running |
   synth_running | saving | succeeded | failed | cancelled`):

   ```powershell
   do { Start-Sleep 3; $s = Invoke-RestMethod "http://localhost:3000/api/generate/status/$($job.jobId)";
        "{0} {1}% {2}" -f $s.status, $s.progress, $s.stage
   } while ($s.status -notin 'succeeded','failed','cancelled')
   $s | ConvertTo-Json -Depth 5
   ```

   Built-in watchdogs (`server/src/services/generation/pollUntilDone.ts`): a
   stage that ticks (`... Step n/total`) and goes quiet for 2 min makes the server
   ask the engine; if the engine says it is still running, the job gets the 15 min
   quiet window before it fails with "Generation stalled". Quiet stages (model
   loads, adapter merge, VAE decode) get that window from the start, and the wall
   clock defaults to 45 min. An engine crash or restart mid-render fails the job
   with "Generation lost: ..." instead of waiting. A first-ever TensorRT run
   legitimately shows `Building TRT engine ... (first run only, ~5-10 min)` — do
   not panic-cancel it. There is 1 automatic retry with a randomized seed on
   transient failure (the ACE backend's `policy.retry`, `maxAttempts: 2`).

6. **Confirm outputs.** On success `$s.result.audioUrls` is like
   `["/audio/<uuid>.wav"]`. **The live data dir is `<data root>/library`, NOT the repo-root
   `data\`** (`config.data.dir` comes from `resolveDataDir()`,
   `server/src/config.ts:27-30`: `<data root>/library` when a data folder exists,
   `server/data` in a dev checkout that has none; the repo-root `data/` is not the
   library — never "verify" outputs there). Files land in
   `<data root>/library\audio\`; DB row in the `songs` table of
   `<data root>/library\hotstep.db` with the full request in `generation_params`.

7. **Confirm the log.** Per-generation logs are buffered in memory and flushed
   **only at finish/fail** — mid-run, tail `node_console.log` (lines prefixed
   `[Gen:<jobId8>]`) or `ace_engine.log` instead.

   ```powershell
   $sess = (Get-ChildItem <data root>/logs | Sort-Object Name -Descending | Select-Object -First 1).FullName
   Get-Content (Get-ChildItem "$sess\generations" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName -Tail 40
   ```

   Success ends `GENERATION COMPLETED.`; failure ends `GENERATION FAILED: <error>`
   (`server/src/services/logger.ts:179,201`).

8. **Hand off to the human** (Tier 4): file path + seed + params. Do not delete
   the artifacts.

## Tier 6 — Distributability (does it work for someone who is not you?)

Bar: `node server/scripts/check-release-prereqs.mjs` exits 0.

Applies to **any change that adds a file the app looks for at runtime** — a new
model, a new data file, a new registry entry, a new packaged binary. This tier is
invisible to every other bar: the path is resolved at runtime, so tsc passes, the
build is green, and the feature is dead in every download.

```powershell
node D:\Ace-Step-Latest\hot-step-cpp\server\scripts\check-release-prereqs.mjs
# --offline skips the Hugging Face and TensorRT checks
```

What it checks:

1. **Weights** — every entry in `server/src/data/model-registry.json` exists in
   the Hugging Face repo it names, at the size the registry claims (and the
   sha256, when the entry declares one), in a repo that is public. Companion
   files (LICENSE) too. Each entry also needs an id, filename, repo, positive
   `sizeBytes` and a `tags` array.
2. **Packs** — no pack references a file id that does not exist (that is how the
   Model Manager blanked in #120).
3. **Runtime data** — everything in `server/src/data/` is packaged by
   `.github/workflows/release.yml`, which copies the directory wholesale for all
   three platforms. Put new runtime data there and it ships automatically; put it
   anywhere else and it will not.
4. **Engine binaries** — every CMake executable target that `server/src` names
   (dev-only tools excepted) appears in `release.yml`'s package lists. v1.2.1
   shipped without `ace-train` and every release through v1.3 without
   `ace-caption`.
5. **The rest of the release path** — the layout checks below, the TensorRT
   files `release.yml` downloads from Hugging Face, the latest Cache Warm run on
   `master` (via `gh`), and `tools/docs/check-docs.mjs`.

What it CANNOT check, so do it by hand:

- A model resolved by prefix scan rather than by registry id. `resolveMm3TrainModels`
  in `server/src/services/training/mm3Train.ts` takes the newest
  `mm3-rvq-*.gguf` on disk — the registry has to publish something that matches
  the prefix, and only a human knows that. **`fs.existsSync` or a directory scan
  against a models dir is the smell**; grep for those when adding a feature.
- Anything gated behind an HF token or a licence the user has to accept.
- Weights you converted locally. Uploading is a deliberate act: `huggingface_hub`
  is installed and the token is in `~/.cache/huggingface/token`, but ask before
  pushing anything to a public repo, and credit the original author in the model
  card if the weights are not ours.

## Tier 7 — Documentation (does the owning page say what the code now does?)

Bar: `node tools/docs/check-docs.mjs` exits 0, and the page that owns the changed
area describes the new behaviour.

Applies to any change a user or developer can see. The ownership table in
`AGENTS.md` ("Documentation is part of the change") names the page per source area.
Routes, plugins and the model registry feed generated tables, so those need
`node tools/docs/build-docs.mjs` rather than a hand edit. The generated blocks in
`server/src/data/assistant-knowledge.md` come from `docs/user/`, so a user-facing
page change also refreshes what the in-app assistant knows (picked up on the next
server start). Template and rules: `docs/dev/docs-contributing.md`.

```powershell
node tools/docs/build-docs.mjs
node tools/docs/check-docs.mjs
```

The checker fails on a UI folder with no page, a stale generated table, a broken
relative link, an unindexed skill, a studio page `FEATURES.md` does not link to,
or a new native `<select>` or checkbox under `ui/src` (files that still have one
are listed in its baseline, which `--update-ui-baseline` rewrites after a
conversion). It also runs inside `check-release-prereqs.mjs` and in CI
(`docs.yml`).

## Key files

| Path | Role |
|---|---|
| `dev-rebuild.bat` | The ONLY sanctioned C++ rebuild entry point (shutdown → build) |
| `tools/engine-smoke.mjs` | Headless render receipt after an engine rebuild (`ENGINE SMOKE: PASS/FAIL`) |
| `engine/verify-hooks.ps1` | Post-upstream-sync check of the fork hooks and the HOT-ggml pin (Hooks 1 to 18) |
| `server/src/routes/generate.ts` | Generation queue, POST `/api/generate`, status/queue/cancel endpoints |
| `server/src/services/backends/ace/generate.ts` | The ACE pipeline: LM phase, skip-LM defaults, synth rebuild, output files, `songs` insert |
| `server/src/services/generation/pollUntilDone.ts` | Stall, quiet-window and wall-clock watchdogs |
| `server/src/services/aceEngineProcess.ts` | Spawns ace-server; crash respawn and the crash limiter |
| `server/src/routes/health.ts` | `GET /api/health` — engine readiness |
| `server/src/routes/auth.ts` | `GET /api/auth/auto` — free Bearer token; `getUserId` guard |
| `server/src/services/generation/translateParams.ts` | UI param names → engine `AceRequest` names |
| `server/src/services/aceClient.ts` | HTTP client to ace-server :8085 (single-threaded httplib caveat) |
| `server/src/services/logger.ts` | Session log folders + buffered per-generation logs |
| `server/src/config.ts` | Ports, paths; `config.data.dir` = `<data root>/library` |
| `<data root>/logs/YYYY-MM-DD_HH-MM-SS/` | Per-session logs: `ace_engine.log`, `node_console.log`, `generations/gen_<jobId>_<task>.log`, `training/<kind>_<jobId>.log` |
| `<data root>/library/audio/` | Where generated audio actually lands |

## Failure signatures

| Symptom | Cause → fix |
|---|---|
| 503 `Engine not ready: ...` on POST /generate | ace-server still bootstrapping — wait; check `/api/health` `engine.bootStatus` |
| 503 `Engine is paused for training preprocessing` | a training preprocess job owns the GPU and stopped the engine — wait for the job |
| 401 Unauthorized | missing/stale Bearer token (in-memory, resets on Node restart) — re-hit `/api/auth/auto` |
| `Generation stalled — no progress for Ns (last stage: "...")` | stall watchdog fired after the quiet window; engine wedged or emitting unrecognized progress — check `ace_engine.log` tail |
| `Generation lost: the engine crashed mid-render` / `... no longer knows this job` | the engine died or restarted under the job — read `ace_engine.log` for the crash |
| `Generation failed on ace-server` | engine-side failure — the real error is in `ace_engine.log`, not the Node response |
| `[ace-server] Restarting in 3 seconds...` in node_console.log | engine crashed; repeated ⇒ missing DLL or a bad C++ change |
| `/api/health` bootStatus `Engine crashed 3 times — download a model in the Model Manager, or check the logs` | crash limiter gave up after 3 crashes in 30 s (`aceEngineProcess.ts:320-326`) — fix the crash, relaunch; it also restarts by itself once a model download lands |
| `/api/health` `engine.noModels: true`, bootStatus `No models installed ...` | first run with an empty models folder, not a crash — download a model in the Model Manager |
| Solvers/schedulers/guidance silently ignored after upstream sync (no error!) | lost `hot-step-sampler.h` hook — run `engine/verify-hooks.ps1` |
| Adapter/sideband params vanish between LM and synth phases | LM echo drops ServerFields-only params; synth requests are rebuilt from the current request + LM fields only (`server/src/services/backends/ace/generate.ts:192-213`) — new sideband params must flow through that reconstruction, never a whitelist |
| First run stuck on `Building TRT engine` 5–10 min | normal one-time TensorRT compile, not a hang |
| HTTP timeouts against :8085 during a generation | normal — ace-server is single-threaded httplib and can't answer mid-DiT-step (`aceClient.ts:6-8`) |
| Output file "missing" from repo-root `data\` | wrong directory — live outputs are in `<data root>/library\audio\` |

## Institutional knowledge

- **VALIDATED (ground truth):** PRESERVE EXPERIMENT ARTIFACTS — never delete
  generated test audio/outputs on your own prediction that they are bad; the human
  verifies by ear and has lost artifacts to premature cleanup before.
- **VALIDATED (ground truth):** report outcomes faithfully — failed = FAILED with
  output pasted; skipped = SKIPPED with reason; no hedged "should work".
- **VALIDATED:** audio quality is human-ear-only; `quality_scores` telemetry is
  advisory, never a verdict.
- **VALIDATED:** live data dir is `<data root>/library`, root `data\` is not the
  library (`resolveDataDir()`, `config.ts:27-30`; run
  `cd server; npx tsx scripts/data-paths.ts` for the active paths).
- **VALIDATED:** `verify-hooks.ps1` checks 18 numbered hooks, not only the 3
  upstream files AGENTS.md lists — the script is the truth.
- **VALIDATED (code):** `adapter_section_isolation` still exists on the wire
  (`aceClient.ts`, `translateParams.ts`) but the feature was reverted (commit
  `ee041e1`, broke musical continuity) — don't "fix" it back in or count it as active.
- **UNVALIDATED:** basin re-base params `rebase_source`/`rebase_beta` exist in
  `translateParams.ts` but the cross-base fix is unvalidated — never report it working.
- **DELIBERATE, not a bug:** draft-LM speculative decoding is disabled by a code
  comment in `server/src/config.ts` (~line 232).
- **VALIDATED (twice, in production):** local presence is not distribution.
  MM3 training required `mm3-rvq-*.gguf` + `mm3-enc-*.gguf` that had never been
  uploaded (#137, fixed 96d442fb); `mm3-corpus.json` was committed but not copied
  into release archives (#139). Tier 6 exists because of these.
- **VALIDATED:** smoke generations queue globally one-at-a-time behind the user's
  jobs — check `GET /api/generate/queue` first; `reset-queue` is destructive.

## Deeper reading

- [reference.md](reference.md) — same folder: full API shapes, output-file table,
  log anatomy, all generate endpoints.
- `AGENTS.md` (repo root, imported by `CLAUDE.md`) — orientation map, build/git rules.
- `engine/docs/ARCHITECTURE.md` — engine internals, request JSON, generation modes.
- `docs/dev/plugins-authoring.md` — Lua plugin authoring (plugins hot-load, no rebuild).
- `docs/dev/releasing.md` — release process (any pushed `v*` tag triggers a full
  multi-platform CI build — never push casual `v*` tags).
- `docs/plans/` — internal design/investigation docs. **Gitignored, local-only —
  may be absent on a fresh clone.**

## Release layout checks

The prereq script checks each CI platform and both local assemblers through `release/check-layout.mjs`, then runs `server/scripts/release-layout.test.ts` with isolated profile/data fixtures. The Windows fixture runs the launcher against a stub server from another cwd. The Linux and macOS cases check default profile paths only, and none of these fixtures starts a native engine. Report a Cache Warm gate failure separately from layout/Hugging Face checks.
