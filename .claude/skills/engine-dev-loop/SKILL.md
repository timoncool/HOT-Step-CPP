---
name: engine-dev-loop
description: Safe edit-rebuild-test cycle for the HOT-Step C++ engine (dev-rebuild.bat, build script selection, stale-.obj recovery, smoke tests). Use when editing any file under engine/src/ or engine/tools/, when a build fails with linker/file-lock errors, or when ace-server won't restart cleanly.
---

# C++ Engine Development Loop

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

How to safely edit, rebuild, and smoke-test the C++ engine (`ace-server.exe` and friends)
without triggering the respawn/file-lock loop that this repo learned about the hard way.

**Context for readers with zero prior exposure:** the app has three tiers. A Node/Express
server (port 3001) spawns the C++ inference engine `ace-server.exe` (port 8085) as a managed
child process and serves the React UI. Agents run it in dev mode with `dev.bat` detached and
work through the Vite dev server at `http://localhost:3000`; `LAUNCH.bat`/`:3001` is the
end-user prod path. The engine runs the generation pipeline
LM → DiT → VAE (LM = language model that plans the song; DiT = Diffusion Transformer that
denoises latent audio; VAE = decoder that turns latents into 48 kHz stereo audio).
All commands below are Windows PowerShell (use `;` to chain, never `&&` in old
Windows PowerShell 5; this repo's convention is `;`).

## When to use this skill

- You edited (or are about to edit) anything under `engine/src/` or `engine/tools/`.
- A build fails with `LNK1104` / "cannot open ace-server.exe".
- ace-server keeps respawning every ~3 seconds, or the linker can't replace the exe.
- A header change seems to be ignored by the build (stale object files).
- A build after an upstream sync fails with the `hotstep_sampler_linked_` sentinel linker error. (For the full sync + hook-verification process itself, use the `upstream-sync` skill — it owns that procedure.)
- You need to confirm a rebuild actually worked and the engine is healthy.

## Golden rules (hard constraints — each prevented real damage)

1. **Recompile immediately after editing ANY `engine/src/` or `engine/tools/` file — do not
   batch up edits and wait.** (Verbatim instruction from the departing lead engineer.)
   WHY: uncompiled edits pile up, and when the eventual build breaks you can't tell which
   edit did it; the engine also silently keeps serving stale behavior in the meantime.

2. **Rebuild via `dev-rebuild.bat` at repo root, NEVER `engine/build.cmd` directly.**
   WHY: you cannot reliably tell whether the app is running (a Node server or orphaned
   tsx-watch tree may be alive in another terminal). Node auto-respawns ace-server on
   abnormal exit (the exit handler in `server/src/services/aceEngineProcess.ts:280-339`,
   retry after 3 s): if the linker
   deletes/replaces the exe while Node is alive, the respawned process re-locks
   `ace-server.exe` and the link fails with a file-lock error — an infinite respawn +
   file-lock loop. `dev-rebuild.bat` shuts everything down first, and is safe even when
   the app is already stopped (its shutdown request is a suppressed no-op,
   dev-rebuild.bat:31) — so there is no scenario where calling build.cmd directly is
   the right move.

3. **NEVER `cmake --build . --clean-first`.** WHY: CUDA kernel recompilation takes 20+
   minutes (documented in AGENTS.md; not independently timed). Only justified if the
   GGML/CUDA layer itself changed. For stale-object problems use the surgical recovery
   in Procedure 2 instead.

4. **Don't switch between build scripts casually.** WHY: `buildcuda.cmd`, `buildvulkan.cmd`,
   and `buildall.cmd` all run `cmake ..` unconditionally against the shared `engine/build/`
   directory with *different* flags — reconfiguring the cache can invalidate large parts of
   the incremental build (CUDA-recompile territory). `build.cmd` only configures when
   `CMakeCache.txt` is absent (`engine/build.cmd:83-92`). Stick to `build.cmd` (via
   `dev-rebuild.bat`) unless you deliberately need a different backend set.

5. **`dev-rebuild.bat` does NOT restart the app, and exit 0 does not prove it ran.**
   It checks `build.cmd`'s exit code: a failed build prints `[dev-rebuild] BUILD FAILED -
   the binaries on disk are STALE.` and exits 1 (dev-rebuild.bat:52-60). A good build prints
   "Done. Start the app with dev.bat to pick up changes." But `cmd.exe /c` can return 0
   having run nothing (AGENTS.md), so check the output for its first line,
   `[dev-rebuild] Requesting graceful shutdown...`, and the exe timestamp. Then relaunch
   with `dev.bat` (detached) and check `http://localhost:3000`.

## Procedure 1 — Standard rebuild after a C++ edit

```powershell
# 1. Rebuild (shuts down app gracefully, waits for ace-server to die, builds):
#    Its PowerShell wait loop also works with redirected stdin.
& "D:\Ace-Step-Latest\hot-step-cpp\dev-rebuild.bat"

# 2. VERIFY the build yourself. A failed build ends with "[dev-rebuild] BUILD FAILED" and
#    exit 1, but exit 0 is not proof the script ran at all (AGENTS.md). Check:
#    (a) "[dev-rebuild] Requesting graceful shutdown..." at the top and no "error C..." or
#        "LNK..." lines in the scrolled output, and
#    (b) a fresh LastWriteTime on the exe:
Get-Item "D:\Ace-Step-Latest\hot-step-cpp\engine\build\Release\ace-server.exe" | Select-Object LastWriteTime

# 3. Relaunch the app (dev-rebuild does NOT do this) — detached, then check http://localhost:3000:
cd D:\Ace-Step-Latest\hot-step-cpp; pwsh -NoProfile -File tools\detach.ps1 dev.bat   # dev mode (Vite :3000 HMR + tsx watch :3001); outlives your run
# LAUNCH.bat is the end-user prod launcher — agents don't run it
```

Note: dev-rebuild's shutdown also kills the Vite dev server on port 3000
(`killVite()`, `server/src/routes/shutdown.ts:59-87`; only when `dev.bat` set `HOT_STEP_DEV`,
and only a pid that really is a Node process running Vite), so in dev mode you must restart
with `dev.bat`, not just relaunch the engine.

### What dev-rebuild.bat actually does (`dev-rebuild.bat:1-60`)

0. Work-channel guard (lines 1-22). Opt-in: only when `HOTSTEP_WORK_CHANNEL=1` and
   `data\collaboration.db` exists does it rerun itself under
   `tools\mcp-lyricstudio\work-run.ps1`, reserving `app-server`, `gpu` and `engine-build`.
   Otherwise it goes straight to the steps below. (AGENTS.md: the work channel is used only
   when the user explicitly asks for it.)
1. Resolve `SERVER_PORT` from the active `.env` using the read-only data-path resolver,
   with the inherited environment taking precedence and 3001 as the default. A pending
   data move, unreadable pointer, invalid port or missing resolver dependencies aborts.
   POST `/api/shutdown` to that port with a 10-second request timeout. This asks the Node
   server to stop its children and respawn logic before exiting.
2. Poll `Get-CimInstance Win32_Process` for `ace-server.exe` whose `ExecutablePath`
   starts with this checkout's absolute directory plus a trailing separator. Other
   checkouts, including siblings with the same name prefix, are excluded. Sleep with
   `Start-Sleep`; after 10 seconds stop only those PIDs, and abort after 15 seconds if
   any remain. `update.bat` uses the same filter. `update.sh` resolves each candidate's
   executable via `/proc/<pid>/exe` on Linux or `lsof -d txt` on macOS before killing it.
   The update scripts must also repair installs with missing or stale dependencies:
   if the port probe fails, including an unresolved data root with `children: null`,
   they warn and use inherited `SERVER_PORT` or 3001, then retain the ownership checks
   and path-filtered engine wait. `dev-rebuild.bat` still aborts on probe failure.
   The Windows implementation lives in `tools/prod/stop-checkout.ps1`. Before
   requesting shutdown it verifies that the configured port's listener is a
   Node process with a command line inside this checkout. An unknown listener
   aborts the rebuild. `dev-rebuild.bat --dry-run` prints the scope without
   stopping anything or building. `update.sh` verifies the listener's working
   directory with `lsof`, and refuses when ownership cannot be established.
3. Once ace-server is gone: `call "%~dp0engine\build.cmd"`.
4. Checks the build's exit code: on failure prints `BUILD FAILED - the binaries on disk are
   STALE.` and exits 1; otherwise prints "Done." **No restart either way.**

Why the graceful route matters: the kill is initiated by the server that *owns* the respawn
logic, and then that server kills itself. Killing ace-server externally while Node lives
triggers the respawn handler instead (`server/src/services/aceEngineProcess.ts:280-339`; the
crash limiter at `aceEngineProcess.ts:80-86` only gives up after 3 crashes within a
30-second window).

There is also `POST /api/restart` (`shutdown.ts:166-184`), which writes a
`.restart-requested` marker in the repo root (dev) that `LAUNCH.bat` /
`server/restart-loop.cmd` loop on. That is for in-app restarts only — it does not rebuild
anything.

### Inspecting the shutdown scope without stopping the app

The rebuild script uses PowerShell `Start-Sleep`, so its wait loop works in agent
shells with redirected stdin. Run `dev-rebuild.bat` for the full shutdown and build
sequence. To inspect its engine scope without stopping anything:

```powershell
$checkout = (Resolve-Path 'D:\Ace-Step-Latest\hot-step-cpp').Path.TrimEnd('\') + '\'
function OwnEngine {
    Get-CimInstance Win32_Process -Filter "Name = 'ace-server.exe'" |
        Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($checkout, [StringComparison]::OrdinalIgnoreCase) }
}
OwnEngine | Select-Object ProcessId, ExecutablePath
# Only after graceful shutdown, if a checkout-owned engine remains locked:
# OwnEngine | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

Never stop an engine by process name. Another checkout may be serving a separate
instance on different ports. A path that cannot be inspected is excluded.

Windows rebuilds, updates and production promotion hold the same exclusive
`engine-artifact.lock` handle under the checkout's `.tmp` folder. If another
operation holds it, the command refuses immediately. Do not bypass the lock
or set `HOTSTEP_ENGINE_LOCK_ROOT` manually. A stable sibling clone on ports
3101/8186 uses its own engine copy. See the stable production procedure in
[building.md](../../../docs/dev/building.md#stable-production-beside-development).

### Errors in a header you never edited = broken literal in the file included *before* it

A single unterminated string literal or comment makes the parser run on into the next
header, so MSVC reports the damage at the wrong file entirely. Symptoms are surreal:
`(double) some_func() / 1048576.0` failing with *"cannot convert from `size_t (__cdecl *)(void)`
to `double`"* plus *"Context does not allow for disambiguation of overloaded function"* — a
call the compiler cannot see the `()` on, in code that is provably correct and unmodified.

Do not debug the reported file. Check the previous one in the include chain, at your edit.

The usual cause when an agent made the edit: a `\n` inside a C string written as a **real
newline**. Shell heredocs and some scripting layers collapse `\\n` → `\n` → an actual line
break, splitting the literal across two lines. Verify after any scripted C++ edit:

```bash
sed -n '<start>,<end>p' <file> | cat -A     # look for a line ending mid-string
```

Prefer the Edit tool over heredocs for C++ containing escapes. (Same class of bug bit a
Windows path in a Markdown file the same day: `\\2026` → octal escape → mangled text.)

### How long builds take

- Incremental `build.cmd` after a single .cpp/.h edit: minutes-scale — configure is skipped
  when `engine/build/CMakeCache.txt` exists (unverified exact duration; no timing data in repo).
- Full clean CUDA build / `--clean-first`: **20+ minutes** (AGENTS.md claim). Avoid.
- First generation after a rebuild is slow: LoKr adapter precompute takes ~17 s per
  DiT+adapter combo unless `--keep-loaded` is set (`server/src/services/aceEngineProcess.ts:175-182`).
  That slowness is normal, not a regression.

## Procedure 2 — Stale-.obj recovery (surgical, NOT clean)

Use when a header change isn't picked up, or you see weird ODR/linker behavior. This
recompiles only the core lib, leaving the expensive GGML/CUDA objects intact:

```powershell
Remove-Item -Recurse -Force "D:\Ace-Step-Latest\hot-step-cpp\engine\build\acestep-core.dir"
Remove-Item -Force "D:\Ace-Step-Latest\hot-step-cpp\engine\build\Release\acestep-core.lib"
& "D:\Ace-Step-Latest\hot-step-cpp\dev-rebuild.bat"
```

(Both paths verified to exist in the current tree.)

## Procedure 3 — After an upstream sync: verify fork hooks

The engine is a patched fork of acestep.cpp. Four upstream files carry HOT-Step `#include`
hooks that break if a sync overwrites them:

| Upstream file | Hook | If lost |
|---|---|---|
| `engine/src/pipeline-synth-ops.cpp` | `hot-step-sampler.h` (replaces `dit-sampler.h`) | **SILENT** — compiles, but all solvers/guidance/schedulers go dead |
| `engine/src/model-store.h` | `hot-step-params.h` | compile error |
| `engine/src/dit.h` | `adapter-merge.h` + `adapter-runtime.h` | compile error |
| `engine/src/fsq-detok.h` (with `fsq-tok.h`) | `fsq-quant.h`, the reference-conformant FSQ quantizer; neither header may carry upstream's own `FSQ_LEVELS` / `fsq_decode_index` / `fsq_encode_index` | one file reverted: compile error; both reverted: upstream's non-conformant FSQ (about 40% index match) without a build error |

```powershell
powershell -File "D:\Ace-Step-Latest\hot-step-cpp\engine\verify-hooks.ps1"   # exit 0 = all hooks intact
```

It checks more than the four hooks above (`engine/verify-hooks.ps1`):
`hot-step-server.cpp` → `hot-step-params.h`, the MM3 and YuE2 include/registration pairs in
`hot-step-server.cpp`, the family boot gate (`hot-step-families.h` and two calls to
`hot_step_any_family_weights_present()`), the FSQ hook (Hook 5), the linker sentinel below,
and hooks 7 to 18 for the HOT-ggml fork (each HOT-Step ggml change present, and
`engine/ggml` clean at the pinned commit). `engine/build.cmd` runs it before every compile
and stops the build when it fails (`build.cmd:94-109`).

**Linker sentinel (silent-hook tripwire):** `engine/src/hot-step-sampler.h:1540-1552`
defines `hotstep_sampler_linked_` (selectany/weak, external linkage);
`engine/tools/hot-step-server.cpp:52-53` references it via `extern`. If a sync reverts
pipeline-synth-ops.cpp to `dit-sampler.h`, the symbol vanishes and the link fails with
`unresolved external symbol hotstep_sampler_linked_`. **That error means "re-hook
hot-step-sampler.h", not a real linker problem.** It converts the otherwise-silent hook
loss into a build failure.

## Procedure 4 — Smoke-testing after a rebuild

**Required receipt for every engine hand-off: the headless render check.** It needs
no app and no UI, only a built `ace-synth` and the models folder:

```powershell
node tools\engine-smoke.mjs              # exit 0 = PASS, 1 = FAIL
node tools\engine-smoke.mjs --self-test  # proves each check can fail; run after touching the checker
```

It renders two fixed-seed instrumentals through `ace-synth --check`: "round" (10 s,
seed 42, 250 latent frames) and "odd" (7.33 s, seed 43: 183 frames padded to 184 for
the DiT patch, decoded in three ragged 96-frame VAE tiles), so padding and tile-edge
indexing get exercised. `--check` measures the raw float buffer **before** the WAV
writer's NaN-to-zero sanitiser (`audio-io.h` `wav_sanitize`) can hide a broken
render. Pass needs zero non-finite and zero subnormal samples, peak above -60 dBFS,
|DC| at most 0.02 per channel, and at most 0.1% of samples past full scale. The
wrapper also deletes the old WAV first, checks the new one exists and holds exactly
frames x 1920 samples, and prints per-stage wall times (model loads, text encode,
DiT, VAE) from the engine log, so a denormal stall or other slowdown shows up as a
time regression. ace-synth exits 1 if a WAV fails to write, 2 if a check fails.

Paste both `[Check Batch0] ...` lines, the `stages:` and `wav:` lines and the
`ENGINE SMOKE: PASS|FAIL` verdict into the hand-off. Each run writes its requests, WAVs
and logs to its own `logs/engine-smoke/<timestamp>-<pid>/` folder under the repo (not the
data root), so overlapping runs never share files; the next run prunes folders older than
a day. Without `--models` it asks `server/scripts/data-paths.ts` for the active models
folder. Run it with the GPU idle (check `GET :8085/jobs` for a `running` job first); it
loads its own models, so the app can stay up.

Measured 2026-10-20, RTX 5090, turbo BF16 DiT: about 3.7 s per case; round peak
-1.1 dBFS, rms 0.102, DC about 3e-4; DiT about 100 ms, VAE about 90 ms, model loads
about 2.8 s together. Normal values sit far inside every threshold.

Mac Metal works the same (finds `engine/build/ace-synth`). The Mac server holds only
the Q4_K_M turbo DiT, so pass it: `node tools/engine-smoke.mjs --models ~/hot-step-models
--dit acestep-v15-turbo-Q4_K_M.gguf`. Measured there 2026-10-20: about 2.5 s per case,
DiT about 600 ms, VAE about 550 ms; both cases and the self-test PASS.

**Block-based DSP: `monitor_run_chain`** (`engine/tools/monitor-chain.h`, used by
vst-host's live monitor, which feeds it device-sized blocks) is the only block-based
path. After touching it or vst-host's processing loop, run its test. It is
header-only, so compile it directly rather than turning on `BUILD_TESTING` in the
shared build cache:

```powershell
# VS developer prompt (vcvars64), in engine\build\test-standalone
cl /nologo /std:c++17 /O2 /EHsc ..\..\tests\test-monitor-chain.cpp /Fe:test-monitor-chain.exe; .\test-monitor-chain.exe
# Mac, from the repo root:
# clang++ -std=c++17 -O2 engine/tests/test-monitor-chain.cpp -o engine/build/test-standalone/test-monitor-chain && engine/build/test-standalone/test-monitor-chain
```

It runs a stateful two-stage chain at blocks 1, 37, 441, 1000 and 4096 over a burst
that decays into silence. Every size must match a one-block reference bit for bit,
stay finite, hold no subnormals and end in exact silence, with ns/sample printed per
size. A flush-off run must produce subnormals, which proves the counter. Nothing in
the host sets FTZ/DAZ, so a plugin without its own denormal guard leaves a tail stuck
at subnormal values: on the 5090 box that costs about 1.5x per sample (3.7 vs 2.4
ns); on Apple silicon there is no penalty. Timings are reported, not asserted.

If you only changed `ace-synth.cpp` or a header it uses, building just that target
leaves a running app alone: `cmake --build engine/build --config Release --target ace-synth`.
Anything ace-server links still goes through the full rebuild above.

Then the live-app checks:

Relaunch with `pwsh -NoProfile -File tools\detach.ps1 dev.bat` (a plain `dev.bat` or `Start-Process` launch dies when your agent run ends; see AGENTS.md), confirm the app answers at `http://localhost:3000/api/health`,
then probe the engine directly. **ace-server takes a few seconds to come up after `dev.bat` returns** (the script just `start`s detached windows) — retry `/health`
for up to ~30 s; connection refused immediately after launch is normal, not a build failure.
(Use `curl.exe` to be safe — in legacy Windows PowerShell 5.1 bare `curl` is an alias for
`Invoke-WebRequest`; in PowerShell 7 it resolves to the real curl.exe.)

```powershell
# 1. Engine alive? Returns {"status":"ok"}  (engine/tools/hot-step-server.cpp:3444-3446)
curl.exe -s http://localhost:8085/health

# 2. Models discovered + config sane?
curl.exe -s http://localhost:8085/props

# 3. Lua plugin registry loaded? (solvers/schedulers/guidance — the thing silent hook loss kills)
curl.exe -s http://localhost:8085/plugins

# 4. CUDA context up? Returns used/total/free VRAM MB; HTTP 500 if CUDA is broken
#    (hot-step-server.cpp:3480-3501). Non-CUDA builds always answer zeros.
curl.exe -s http://localhost:8085/vram
```

Full endpoint surface, if you need deeper probes (`hot-step-server.cpp:3438-5176`):
`POST /lm /synth /understand /vae /codes-decode /warm /job /models/unload
/models/restore-policy /pp-vae-reencode /sa3-refine /sa3-synth`,
`GET /health /worker-drain /props /logs /jobs /job /plugins /vram /models/loaded /sa3/props`,
plus `/supersep/*`, `/spectral-lifter`, and the `/mm3/*` and `/yue2/*` routes that
`mm3_register_routes`, `mm3_register_job_routes` and `yue2_register_routes` add.

**Real smoke test:** kick a short generation from the UI — **ask the human user to do this
and report back; do NOT use a browser agent for visual verification** (project rule).
Watch the newest session folder under `<data root>/logs/` (folders are named
`YYYY-MM-DD_HH-MM-SS`, so name-sorted = time-sorted): `ace_engine.log` (C++
stdout/stderr), `node_console.log`, and `generations/gen_<uuid>_<task>.log`.
**Never delete generated test audio or other generation outputs — even ones you predict
are bad.** The user verifies results by ear; leave all artifacts in place.

**Standalone engine (without the Node server):** `engine/server.cmd` runs
`ace-server.exe --host 0.0.0.0 --port 8085 --models .\models --adapters .\adapters --max-batch 1`.
Caveat: its `--models .\models` is relative to the current directory, but the real model
directory is `<data root>/models/` (the Node server passes the absolute path at
`server/src/services/aceEngineProcess.ts:161-170`). Treat server.cmd as a template, not a
turnkey smoke test.

## The build scripts — when each applies

All in `engine/`, all build into the shared `engine/build/` with
`--config Release -j %NUMBER_OF_PROCESSORS%`, all auto-locate `vcvars64.bat` via vswhere
and skip re-sourcing when `VSCMD_VER` is already set (prevents PATH overflow on repeated runs).

| Script | CMake config | Use when |
|---|---|---|
| `engine/build.cmd` | Configures **only if `CMakeCache.txt` absent** (`build.cmd:83-92`). Defaults: `-DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;120a" -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON`. Honors `HOT_STEP_CMAKE_FLAGS` (set by `update.bat:323-345` for auto-detected backends). Bootstraps the cuDNN 9 DLLs via pip into `build/Release/` (`build.cmd:40-76`), then runs `verify-hooks.ps1` and stops before compiling if any hook fails (`build.cmd:94-109`). Exits with the build's own exit code. | **The default dev build.** This is what dev-rebuild calls. Incremental. |
| `engine/buildcuda.cmd` | `cmake .. -DGGML_CUDA=ON` unconditionally (`buildcuda.cmd:36`). No cuDNN bootstrap, no hook check, no CPU variants. | Rare — `build.cmd` supersedes it for day-to-day dev. |
| `engine/buildvulkan.cmd` | `cmake .. -DGGML_VULKAN=ON` unconditionally (`buildvulkan.cmd:36`). | Testing the Vulkan backend (AMD/Intel) or Vulkan-specific fixes. |
| `engine/buildall.cmd` | Optional argument `all` (default), `cuda`, `vulkan` or `cpu`; then `cmake .. -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=%BUILD_CUDA% -DGGML_VULKAN=%BUILD_VULKAN% -DGGML_BACKEND_DL=ON %RELEASE_CMAKE_EXTRA%` unconditionally (`buildall.cmd:62`). | Release-style all-backends build (what CI resembles). Slowest. |

Linux/mac equivalents (`buildcuda.sh`, `buildvulkan.sh`, `buildcpu.sh`, `buildall.sh`,
`build-mac.sh`) exist but are not part of the Windows dev loop.

### Build outputs & which exe the server picks

Binaries land in `engine/build/Release/`: the tools `ace-server.exe`, `ace-lm.exe`,
`ace-synth.exe`, `ace-understand.exe`, `ace-train.exe`, `ace-caption.exe`, `ace-midi.exe`,
`neural-codec.exe`, `mp3-codec.exe`, `quantize.exe`, `mastering.exe` and `vst-host.exe`,
plus parity tests and probes (`sa3-ggml-test`, `fattn-train-test`, `yue2-probe`, the
`yue2-*-test` family and more), ggml's `vulkan-shaders-gen.exe`, and the `BUILD_TESTING`
targets when that option is on. The full table is
[docs/dev/engine.md#binaries](../../../docs/dev/engine.md#binaries).

The Node server auto-discovers the exe in priority order (`server/src/config.ts:133-139`):
`engine/ace-server.exe` (portable, flat) → `engine/build/Release/ace-server.exe` (VS
multi-config — **the dev path**) → `engine/build/ace-server.exe` (Ninja) →
`engine/build/Debug/ace-server.exe`. Override with `ACESTEPCPP_EXE` in `.env`.
**Gotcha:** a stale flat `engine/ace-server.exe` (e.g. from unpacking a portable release
into the repo) wins over your fresh Release build. None exists in the tree today, but
check if the server seems to run old code.

## What does NOT need a C++ rebuild

- **Solvers, schedulers, guidance modes and postprocess steps are hot-loadable Lua
  plugins.** The engine scans `engine/plugins/`, then the repo-root `plugins/`, then a
  data folder's `plugins/` overlay, each with `solvers/`, `schedulers/`, `guidance/` and
  `postprocess/` subdirs (`lua-plugin-registry.h`). Drop a `.lua` in the right subdir; it
  appears in the UI next launch. Native
  bridge is `apg()`; advanced plugins use `post_step()` for extra forward passes. Editing
  `dit-sampler.h` to add a solver is obsolete practice — routing goes through
  `hot-step-sampler.h`. Guide: `docs/dev/plugins-authoring.md`.
- **Node/TypeScript changes:** type-check with `npx tsc --noEmit` in `server/` (add
  `-p scripts/tsconfig.json` for `server/scripts/`) and `npx tsc --noEmit -p
  tsconfig.app.json` in `ui/` (a bare `npx tsc --noEmit` there checks nothing); `tsx watch`
  auto-restarts the server in dev mode. Don't `npm run build` during dev.

## Key files

| Path | Role |
|---|---|
| `dev-rebuild.bat` (repo root) | The safe rebuild entry point: graceful shutdown → wait/kill → `engine/build.cmd` → build-result check |
| `engine/build.cmd` | Default incremental CUDA dev build + cuDNN bootstrap + hook check before compiling |
| `engine/buildcuda.cmd` / `buildvulkan.cmd` / `buildall.cmd` | Alternate backend builds (reconfigure unconditionally — see Golden rule 4) |
| `engine/verify-hooks.ps1` | Hook and HOT-ggml checker, run by `build.cmd` and after any upstream sync (exit 0 = intact) |
| `tools/engine-smoke.mjs` | Headless render receipt (Procedure 4) |
| `engine/src/hot-step-sampler.h` | Fork sampler (solvers/schedulers/guidance routing) + linker sentinel at :1540-1552 |
| `engine/tools/hot-step-server.cpp` | ace-server main; sentinel extern at :52-53; HTTP routes at :3438+ |
| `engine/build/Release/` | Build output dir (the exe Node runs in dev) |
| `engine/build/acestep-core.dir/` + `engine/build/Release/acestep-core.lib` | The ONLY things to delete for stale-.obj recovery |
| `server/src/services/aceEngineProcess.ts` | Spawns ace-server (`startAceServer`, :149); respawn-on-crash handler (:280-339); crash limiter (:80-86); `stopAceServer`/`restartAceServer` |
| `server/src/routes/shutdown.ts` | `/api/shutdown` (stops ace-server, Vite, self) and `/api/restart` |
| `server/src/config.ts` | Exe discovery order (:133-139), `ACESTEPCPP_EXE` override |
| `engine/server.cmd` | Standalone ace-server launcher (template — model paths need adjustment) |
| `<data root>/logs/<session>/ace_engine.log` | C++ engine output for the newest run |

## Failure signatures

| Symptom | Cause | Fix |
|---|---|---|
| `LNK1104` / cannot open `ace-server.exe` during link | ace-server still running (respawn loop or manual start) | Run `dev-rebuild.bat` (its shutdown phase), or check `tasklist /FI "IMAGENAME eq ace-server.exe"` |
| `unresolved external symbol hotstep_sampler_linked_` | Upstream sync clobbered the `hot-step-sampler.h` include in `pipeline-synth-ops.cpp` | Re-add the include; run `verify-hooks.ps1` |
| Builds fine, but solver/scheduler/guidance selections have no effect | Silent hook loss predating the sentinel, or sentinel removed | Run `verify-hooks.ps1`; check `GET :8085/plugins` |
| Compile error: missing `hot-step-params.h` / `adapter-merge.h` / `adapter-runtime.h` | Sync clobbered `model-store.h` / `dit.h` hooks (these fail loudly) | Re-add the includes; run `verify-hooks.ps1` |
| Redefinition of `fsq_encode_index` / `fsq_decode_index` / `FSQ_LEVELS`, or `fsq_encode_index` undeclared | One FSQ header reverted to upstream's copy (`fsq-quant.h` is meant to fail loudly here) | Delete the upstream copies again, keep `#include "fsq-quant.h"` in `fsq-detok.h`; run `verify-hooks.ps1` |
| `[ace-server] Crashed 3 times within 30s — giving up` in node_console.log | An incomplete model set (`ERROR: no usable pipeline, synth missing: ...`), a missing DLL next to the exe (cuBLAS/cuDNN), or a startup crash. Not an empty models folder: that logs `No models installed — waiting for a download in the Model Manager.` and waits without spending the crash budget (`aceEngineProcess.ts:298-307`) | Check `ace_engine.log` in the newest `<data root>/logs/` session |
| `[build] ABORTED: engine hooks or the ggml fork are missing` | `verify-hooks.ps1` failed inside `build.cmd`: a lost hook, or `engine/ggml` dirty, off its pin, or still the stock ggml-org checkout | Read the `[FAIL]` lines above it; for the ggml hooks run `git submodule sync -- engine/ggml` then `git submodule update --init engine/ggml` |
| Engine respawns endlessly every ~3 s | Something external kills ace-server while Node lives (respawn handler fires); crashes spaced >30 s apart reset the limiter | Stop killing it externally; use `dev-rebuild.bat` / `/api/shutdown` |
| Node runs an old binary despite a successful build | Stale flat `engine/ace-server.exe` shadowing `build/Release/` (config.ts:133-139), or `ACESTEPCPP_EXE` set in `.env` | Delete the flat exe / unset the override |
| Header edit seemingly ignored | Stale .obj | Procedure 2 (surgical delete of `acestep-core.dir` + `acestep-core.lib`) |
| `dev-rebuild.bat` says `BUILD FAILED - the binaries on disk are STALE.` | The compile or link failed (or the hook check stopped it) | Scroll up for the first `error C` / `LNK` / `[FAIL]` line; the old exe is still the one on disk |
| `dev-rebuild.bat` exits 0 in ~2 s, prints nothing, app still running | The script never ran (AGENTS.md: `cmd.exe /c` can exit 0 having run nothing) | Run it by absolute path and verify its first output line (Procedure 1) |
| `ERROR: Input redirection is not supported` in an older rebuild script | Its `timeout` wait requires a console | Use the current script, which waits with PowerShell `Start-Sleep` |
| Compile errors in a header you never touched | Unterminated string/comment in the file included *before* it — usually a `\n` written as a real newline by a heredoc | `sed -n 'A,Bp' <the file you edited> \| cat -A`; use the Edit tool for C++ with escapes |
| Vite dead after rebuild in dev mode | dev-rebuild's shutdown kills Vite on port 3000 too (`killVite()`, `shutdown.ts:59-87`) | Restart with `dev.bat`, not LAUNCH.bat |
| Node server won't start / weird npm dep errors after an otherwise-good build | Wrong Node version or native addon ABI mismatch — **Node 20 to 24 LTS; 24 recommended** (`engines` enforces `<25`) | `node --version`; install dependencies for that Node version, don't touch the engine or build cache |
| Connection refused on :8085 right after relaunch | Engine still starting — dev.bat returns before services listen | Retry `/health` up to ~30 s, then check `ace_engine.log` |
| PATH weirdness after many rebuilds in one shell | Shouldn't happen — scripts skip vcvars when `VSCMD_VER` is set | Open a fresh shell |

## Institutional knowledge

- **VALIDATED (lead-engineer rule, verbatim in meaning):** Recompile immediately after
  editing any `engine/src/` or `engine/tools/` file — do not batch up edits and wait.
- **VALIDATED (code-verified):** the respawn/file-lock loop mechanism — respawn handler in
  `server/src/services/aceEngineProcess.ts:280-339`, 3 s delay, crash limiter
  `MAX_CRASHES = 3` within `CRASH_WINDOW_MS = 30_000`.
- **VALIDATED (code-verified):** the linker sentinel exists specifically to turn the silent
  `pipeline-synth-ops.cpp` hook loss into a hard link error.
- **VALIDATED (learned the hard way, per AGENTS.md):** `--clean-first` costs 20+ minutes of
  CUDA kernel recompilation. Duration is a doc claim, not independently timed.
- **UNVERIFIED:** exact incremental build duration ("minutes-scale") — no timing data in repo.
- **UNVERIFIED:** whether `engine/server.cmd` finds models when run standalone from
  `engine/` — its `--models .\models` path likely needs adjusting to `<data root>/models/`.

## Deeper reading

- `docs/dev/engine.md` — engine guide: binaries, per-backend pipeline, hook files, plugins,
  TensorRT, training subcommands.
- `docs/dev/building.md` — build scripts, dev loop, type-checks, portable packaging.
- `engine/docs/ARCHITECTURE.md` — request JSON, generation modes, CLI flags and HTTP endpoints.
- `docs/dev/plugins-authoring.md` — Lua plugin authoring (the no-rebuild path for solvers/schedulers/guidance).
- `docs/dev/releasing.md` — release builds; pushing any `v*` tag triggers a full multi-platform CI build.
- `AGENTS.md` — repo-wide rules (git discipline, environment, log layout); `CLAUDE.md` only imports it.
- **Full sync + hook-repair process: the `upstream-sync` skill** (`.claude/skills/upstream-sync/`,
  always present) — it carries the concrete re-hook repair steps this skill's failure table
  abbreviates. `docs/plans/upstream-sync-workflow.md` is the local long-form doc, **gitignored**
  — may be absent on a fresh clone.
