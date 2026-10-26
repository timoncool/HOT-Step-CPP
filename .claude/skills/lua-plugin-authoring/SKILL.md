---
name: lua-plugin-authoring
description: Explains how to write, test, and debug HOT-Step Lua plugins (solvers, schedulers, guidance modes, postprocess) including the apg()/post_step() API and UI param flow. Use when adding or modifying a sampling solver, noise scheduler, CFG/guidance mode, or VAE-decode postprocess plugin, or when a plugin fails to load, has no effect, or produces noise.
---

# Writing Lua plugins for HOT-Step CPP

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

Solvers, schedulers, guidance modes, and postprocess (VAE-decode replacement) are **Lua plugins (LuaJIT 2.1)** loaded by the C++ engine at startup. Adding one = drop a `.lua` file in the right directory and restart the app. **No C++ rebuild.**

Glossary (used throughout):
- **DiT**: the diffusion transformer that denoises audio latents over N steps. A **solver** decides how the latent `xt` advances each step given the model's predicted velocity `vt`. A **scheduler** decides the timestep values. A **guidance** mode combines the conditional and unconditional model predictions (classifier-free guidance, CFG). A **postprocess** plugin replaces the built-in tiled VAE (variational autoencoder) latent-to-audio decode.
- **FloatArray**: a zero-copy, bounds-checked view over a raw C++ `float*` (a table whose metamethods index an FFI pointer, built by the host prelude in `engine/src/lua-plugin.h:36-94`). It is **0-indexed** (`xt[0] … xt[n-1]`), unlike normal Lua tables. `#xt` returns its length.
- **`params`**: a Lua global table injected before every plugin call, holding the values the user set in the UI for this plugin's declared parameters.
- **`model_context`**: a Lua global injected before every plugin call describing the backend that is running (`native_sr`, `latent_fps`, `latent_channels`, `model_id`, `latent_rms`, `engine_renoise`, and `yue2` on the YuE2 NAR path); `lua-plugin.h:652-721`. Defaults match ACE, so a plugin that never reads it behaves as before.

## Golden rules

1. **The old approach of editing `engine/src/dit-sampler.h` is OBSOLETE.** All ACE sampling routes through `engine/src/hot-step-sampler.h` (included by upstream `pipeline-synth-ops.cpp`). Adding a solver/scheduler/guidance = write a `.lua` plugin. WHY: C++ edits to the old sampler are dead code and waste a rebuild cycle; losing the `hot-step-sampler.h` include during an upstream sync kills every plugin (a linker sentinel `hotstep_sampler_linked_` at `hot-step-sampler.h:1540-1552` turns that into a link error; `engine/verify-hooks.ps1` also checks it).
2. **"Hot-loadable" means no rebuild, NOT live reload.** Plugins are scanned once at ace-server startup (`PluginRegistry::init`, `engine/tools/hot-step-server.cpp:3409`). After editing a `.lua` file you must restart the app. `POST /api/plugins/reload` only clears the Node server's 60-second cache of the plugin list (`server/src/routes/plugins.ts:32-35`); it does not re-read files.
3. **Never kill `ace-server.exe` directly.** The Node server auto-respawns it on crash, causing an infinite respawn + file-lock loop. The working restart is `Invoke-RestMethod -Method Post http://localhost:3000/api/shutdown/restart` (the shutdown router is mounted at `/api/shutdown` in `server/src/index.ts:111`, so plain `/api/restart` 404s; `dev.bat` runs Node under `server/restart-loop.cmd`, which relaunches it, and Node respawns ace-server, which rescans plugins). Alternatively `dev-rebuild.bat` (clean shutdown; its compile is a no-op for Lua-only changes) **then** start again with `pwsh -NoProfile -File tools\detach.ps1 dev.bat`: `dev-rebuild.bat` does NOT relaunch, and `detach.ps1` refuses to start a second app while :3001 is listening. If plugin work escalates into editing any `engine/src/` C++ file (e.g. a new bridge function in `lua-plugin.h`), rebuild via `dev-rebuild.bat` immediately, never `engine/build.cmd` directly, and never `cmake --clean-first` (20+ min CUDA recompile; for stale `.obj` issues delete only `engine/build/acestep-core.dir/` and `engine/build/Release/acestep-core.lib`).
4. **FloatArrays are 0-indexed; Lua tables are 1-indexed.** An out-of-range index raises a Lua error that aborts the call. For a per-step `step()` the engine prints the error and continues, so `xt` never advances and the "successful" generation is pure noise. A full-loop `sample()` error fails the job instead (`[DiT] FATAL: full-loop solver … failed`). WHY: this is the single most common plugin bug and the per-step form fails silently from the user's perspective.
5. **Ship plugins in repo-root `plugins/<type>/`, not `engine/plugins/<type>/`.** The engine scans `engine/plugins/` (built-ins), then the repo-root `plugins/` (community), then the data folder's `plugins/` when Node passes `--user-plugins <data root>/plugins` (any install with a data folder): `engine/src/lua-plugin-registry.h:39-63`. Between the first two tiers, a duplicate `name` keeps the **first loaded**, so a community file cannot shadow a built-in. A plugin in the data-folder tier **replaces** a shipped plugin of the same name (`[Plugins] User plugin '<name>' from <path> overrides <path>`, :231-241); two user files with one name keep the first.
6. **In guidance plugins, always route the base combine through `apg()`.** Raw `uncond + w*(cond-uncond)` produces audible artifacts; the native `apg()` bridge adds momentum smoothing, perpendicular projection, and norm thresholding. Customize the *scale* or post-process `result` instead (see `engine/plugins/guidance/cfg_pp.lua`).
7. **Write Lua 5.1 as LuaJIT runs it.** There is no `//` integer division, no `math.tointeger` and no integer subtype; a file using Lua 5.3+ syntax fails to load (`[Plugins] ERROR loading …`) and never registers. Use `math.floor(a / b)` (commit `1830d89b` fixed exactly this in two shipped plugins).
8. **Do not visually verify the UI with a browser agent**: ask the user. API checks (`/api/plugins`) are fine.
9. **Never delete generated test audio, even output you believe is noise or broken**: the user verifies plugin results by ear and compares control vs. plugin runs. Leave every test generation in place and ask the user to listen.

## Where plugins live and how they load

Scanned at startup, in order (`lua-plugin-registry.h:39-63`, directories resolved from the ace-server executable path at `hot-step-server.cpp:3382-3411`):

1. `engine/plugins/{solvers,schedulers,guidance,postprocess}/`: built-ins
2. `<repo-root>/plugins/{solvers,schedulers,guidance,postprocess}/`: community plugins shipped with the app
3. `<data root>/plugins/{solvers,schedulers,guidance,postprocess}/`: user drop-ins, only when a data folder is set (`--user-plugins`, `server/src/services/aceEngineProcess.ts:173`). Release installs keep user plugins here so updates leave them alone.

Missing directories are skipped silently (e.g. `engine/plugins/postprocess/` does not exist; the shipped postprocess plugins are `plugins/postprocess/md_audio_tiled.lua`, `md_audio_tiled_V2.lua` and `md_audio_tiled_V2.1.lua`, all on `md_audio_tiled_core.lua`).

Loader rules (`lua-plugin-registry.h:172-253`):
- Only `.lua` files; sorted for deterministic order.
- **Companion-file exclusion**: filename stems containing `_constants`, `_math`, or `_data` are never loaded as plugins; they are `require()` targets (e.g. `beta_math.lua`, `stork4_constants.lua`). Stems ending `_core` are skipped only if a sibling without `_core` exists in the same dir (`md_audio_tiled_core` is skipped; `storm_sampler_core` loads as a plugin because no `storm_sampler.lua` exists). A shared library with any other name, such as `plugins/solvers/md_solver_commons.lua`, is loaded, defines no plugin table and is skipped with a harmless `has no solver/scheduler/guidance/postprocess table` warning.
- Plugin **type is detected by which global table the file defines**: `solver`, `scheduler`, `guidance`, or `postprocess` (`lua-plugin.h:518-563`). Wrong table for the directory → "declares wrong type, skipping". Empty/missing `name` → skipped. Lua syntax error → `[Plugins] ERROR loading <path>: <message>` on stderr, plugin absent.
- Startup log (in `<data root>/logs/<session>/ace_engine.log`): one line per plugin, `[Plugins] User plugins: <dir>` when the third tier is scanned, then `[Plugins] Loaded N solvers, N schedulers, N guidance, N postprocess`.

Sandbox (`lua-plugin.h:289-333`, `package.path` at :495-505): `math`, `string`, `table`, `print`, `type`, `pairs`, `ipairs`, `tonumber`, `tostring`, `require` are available. `os`, `io`, `debug`, `dofile`, `loadfile`, `ffi`, `jit` and `package.loadlib` are removed, including their `package.loaded`/`package.preload` copies. `require()` searches ONLY the plugin's own directory; C modules are blocked. `math.random`/`math.randomseed` are a host port of Lua 5.4's xoshiro256**, and `math.randomstream(seed)` returns an independent generator. Each plugin gets its **own `lua_State` that lives for the whole process**: file-level `local` variables persist across steps AND across generations (that is how stateful solvers work, and why they must self-reset; see the stateful-solver rule below).

## Procedure: write and test a plugin

1. Create `plugins\<type>\my_thing.lua` (repo root). Start by copying the closest template (see the table in "Worked templates" below).
2. Restart the app. If it is running: `Invoke-RestMethod -Method Post http://localhost:3000/api/shutdown/restart` (relaunches Node → respawns ace-server → rescans plugins). If it is fully stopped: start it with `pwsh -NoProfile -File tools\detach.ps1 dev.bat`. Never kill `ace-server.exe` yourself.
3. Confirm it loaded (resolve `<data root>` first with `cd server; npx tsx scripts/data-paths.ts`):
   ```powershell
   $s = Get-ChildItem "<data root>\logs" -Directory | Sort-Object Name -Descending | Select-Object -First 1
   Select-String -Path "$($s.FullName)\ace_engine.log" -Pattern '\[Plugins\]'
   ```
   Expect a per-plugin line naming yours. A load failure prints `[Plugins] ERROR loading <path>: <lua error>` here.
4. Confirm the API sees it:
   ```powershell
   Invoke-RestMethod http://localhost:3000/api/plugins | ConvertTo-Json -Depth 6
   ```
   (or engine-direct `http://localhost:8085/plugins`). If you edited metadata and the list looks stale, bust the Node cache: `Invoke-RestMethod -Method Post http://localhost:3000/api/plugins/reload` (cache only; file changes still need an app restart).
5. Generate once with known-good settings (solver `euler` + scheduler `linear`) as a control, then swap in your plugin. A generation selects plugins via request JSON fields `infer_method` (solver; note the non-obvious name), `scheduler`, `guidance_mode`, and `postprocess_plugin` (UI params `inferMethod` / `scheduler` / `guidanceMode` / `postprocessPlugin`, mapped in `translateParams.ts:150-152` and `:341-343`). Runtime confirmation lines in `ace_engine.log`:
   - `[DiT] Solver: <display> (<name>, N NFE/step, order K)` (`hot-step-sampler.h:712`)
   - `[DiT] Guidance: <display> (<name>) [native APG] [post_step]` (`hot-step-sampler.h:734`)
   - `[DiT] Custom schedule: <display> (<name>), shift=X` (`engine/src/sampler-schedule.h:152`; composites log at :132)
   - `[Postprocess] Using plugin '<name>' for VAE decode` (`pipeline-synth-ops.cpp:1742`)
6. Debug with `print()`: it goes to engine stdout, captured in `ace_engine.log` and the matching `<data root>/logs/<session>/generations/gen_*.log`. Runtime Lua errors surface as `[Plugins] ERROR in solver '<name>' step(): <traceback>` (also `schedule()`/`guide()`/`post_step()`/`process()` and `full-loop solver '<name>' sample()` variants):
   ```powershell
   Select-String -Path "$($s.FullName)\ace_engine.log" -Pattern '\[Plugins\] ERROR'
   ```

## The four plugin types: contracts

All contracts below are verified against `engine/src/lua-plugin.h` (the single source of truth). Full detail including full-loop solvers, `post_step()`, and postprocess internals: [reference.md](reference.md).

### 1. Solver: `plugins/solvers/`

```lua
solver = {
    name = "my_solver", display = "My Solver", description = "...",
    nfe = 1, order = 1,          -- informational (shown in logs/UI)
    needs_model = false,          -- true => step() gets model_fn + vt_buf
    stateful = false, stochastic = false,  -- informational
    -- owns_loop = true          -- advanced: define sample() instead of step(); see reference.md
    params = { ... },             -- optional; see "Declared UI params"
}

-- single-eval (needs_model = false): 5 args
function step(xt, vt, t_curr, t_prev, n)
    -- xt: mutable FloatArray (modify IN PLACE); vt: READ-ONLY FloatArray
    -- t_curr = current t; t_prev = the NEXT (lower) t despite the name
    local dt = t_curr - t_prev   -- positive
    for i = 0, n - 1 do xt[i] = xt[i] - vt[i] * dt end
end
```

Multi-eval (`needs_model = true`) gets 7 args: `step(xt, vt, t_curr, t_prev, n, model_fn, vt_buf)`. `model_fn(xt_arr, t_val)` runs a full CFG'd forward pass and writes guided velocity into `vt_buf` (a live-memory FloatArray). The `vt` arg is a **snapshot** taken before your step: read the original velocity from `vt`, read fresh model results from `vt_buf` (`hot-step-sampler.h:1414-1430`; this separation fixed the historical "Heun silently becomes Euler" bug).

Globals injected per call (`lua-plugin.h:757-778`): `step_index` (0-based), `batch_n`, `n_per` (elements per batch item), `seed` (job seed of batch item 0), `seed_b`, `seed_blend`, `params`, `model_context`. `n` = whole flattened batch (`batch_n * n_per`). The plugin's `math.random` is re-seeded from the job seed when `step_index == 0` (`lua_seed_plugin_rng`, :734-743), so seeded plugins reproduce.

Engine invariants: on ACE and MM3 the **final step never calls `step()`**: the engine computes `output = xt - vt * t_curr` itself (`hot-step-sampler.h:1405-1410`). SA3 and the YuE2 NAR call `step()` for every step, the last one with `t_prev = 0`, so a solver must handle that step too. DCW correction and repaint injection are applied after your step by the engine; do not reimplement them.

**Stateful solvers must self-reset**: `lua_State` persists across generations, so reset file-locals when `step_index == 0` (see `engine/plugins/solvers/unipc.lua:155`). Checking only "did `n` change" is insufficient: two same-length generations back-to-back will bleed state (`plugins/solvers/md_pingpong_simple.lua:263-266` documents the explosion this causes).

### 2. Scheduler: `plugins/schedulers/`

```lua
scheduler = { name = "my_sched", display = "My Sched", description = "...", params = {...} }

function schedule(output, num_steps, shift)
    -- write num_steps DESCENDING t values (1.0 -> ~0.0) into output (FloatArray)
    -- do NOT append a trailing 0: the engine handles the final x0 step
    for i = 0, num_steps - 1 do output[i] = 1.0 - i / num_steps end
    -- apply the standard shift warp yourself (every shipped scheduler does):
    if shift ~= 1.0 then
        for i = 0, num_steps - 1 do
            local t = output[i]
            output[i] = shift * t / (1.0 + (shift - 1.0) * t)
        end
    end
end
```

`shift` is NOT taken from the UI: it is **back-calculated** from the upstream schedule's second timestep and clamped (`<0.5 → 1.0`, `>10 → 3.0`) at `sampler-schedule.h:62-75`. Your plugin only runs when the request names a scheduler; empty = upstream default shifted-linear, and an unknown name falls back to `linear` with a warning (:139-142). On ACE the schedule is built once in `ops_build_schedule` (`pipeline-synth-ops.cpp:356-367`), before the cover blend picks its start sigma; a `custom_timesteps` CSV in the request overrides all schedulers there. A cover with `cover_noise_method = "rescale"` calls the scheduler again for the shortened tail (:1109-1113). Composite syntax `composite:A+B:crossover:split` blends two scheduler plugins engine-side (`sampler-schedule.h:80-134`). Name aliases: `karras` → `sgm_uniform`; `power:4.00` falls back to prefix `power` (`lua-plugin-registry.h:85-104`).

### 3. Guidance: `plugins/guidance/`

```lua
guidance = { name = "my_guide", display = "My Guide", description = "...", params = {...} }

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    -- pred_cond / pred_uncond: READ-ONLY FloatArrays [Oc*T]; result: mutable [Oc*T]
    -- customize the scale, then ALWAYS combine via apg():
    apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
end
```

Called per batch element per model eval. Globals: `step_idx` (note: guidance uses `step_idx`; solvers use `step_index`), `total_steps`, `dt`, `t_curr`, `params`, `model_context`. A guidance plugin may define only `post_step()` and no `guide()` (`has_guide`, `lua-plugin.h:544-556`).

`apg()` is a C bridge over native `apg_forward()` (`lua-plugin.h:978-1006`), registered lazily on the first `guide()` call (:1034-1041). Calling it at file top level therefore fails at load with Lua's `attempt to call a nil value (global 'apg')` and the plugin is skipped. Calling it from `post_step()` does NOT error: the `_apg_mbuf` global set before each `guide()` call (:1025-1026) is never cleared, so it silently reuses the momentum buffer of whichever batch element was guided last (wrong/stale state). Either way: only call `apg()` inside `guide()`.

**Native bypass gotcha**: when the selected guidance is named exactly `"apg"`, the engine takes the native C++ path and never calls Lua `guide()` (`use_apg_native`, `hot-step-sampler.h:733`). Editing `apg.lua`'s body does nothing; it is a documented fallback (`apg.lua:4-5`). To experiment, copy it under a new `name`.

Advanced: define a global `post_step(xt, t, n, eval_cond, eval_uncond, vt_cond, vt_uncond)` and the engine calls it after every solver step except the last, only while CFG is active. Each `eval_cond`/`eval_uncond` call is a **full model forward pass**: expensive. Details + gating conditions: [reference.md](reference.md). Real user: `engine/plugins/guidance/cfg_mp.lua:47`.

### 4. Postprocess: `plugins/postprocess/` (replaces tiled VAE decode)

```lua
postprocess = { name = "my_pp", display = "My PP", params = {...} }

function process(latents, B, C_lat, W, C_aud, final_samples, upscale_factor, vae_decode_fn)
    -- latents: plain Lua TABLE (1-indexed, NOT a FloatArray), channel-major [C_lat, W]
    -- B is always 1 (engine iterates batch items); upscale_factor = audio samples per
    --   latent frame of the running backend (1920 on ACE and YuE2, 4096 on SA3)
    -- vae_decode_fn(latent_table, T_latent) -> audio_table, T_audio
    -- MUST return: audio_table (1-indexed, [2 * T_audio]), T_audio
end
```

Selected per request via JSON field `postprocess_plugin` (`engine/src/request.cpp:214-215`; UI param `postprocessPlugin` → `server/src/services/generation/translateParams.ts:341-343`). The UI sends it only when the master post-processing toggle and the postprocess toggle are both on (`ui/src/stores/globalParamsStore.ts:990`). Unknown name or a failed `process()` → warning + automatic fallback to the built-in tiled decoder (`pipeline-synth-ops.cpp:1708-1712, 1778-1780`). This path uses Lua tables with transposes on both sides: deliberately not zero-copy. Reference implementation: `plugins/postprocess/md_audio_tiled.lua` + `md_audio_tiled_core.lua` (the `require()`-a-`_core`-module pattern). Full contract: [reference.md](reference.md).

## Declared UI params and how values reach your plugin

Schema (extracted at `lua-plugin.h:391-461`): four types: `slider` (`default`/`min`/`max`/`step`), `select` (`default` + `options` as `{value=,label=}` tables or bare strings), `toggle` (bool `default`), `text` (string `default`). Common fields: `key`, `label`, `hint`, `visible_when = { key = "...", equals = "..." }` (string-compared against a sibling param's current value; a boolean or number `equals` is formatted like JavaScript's `String()`, :442-457). Field reference + JSON shape: [reference.md](reference.md).

Flow: engine `GET /plugins` → Node proxy `GET /api/plugins` (60 s cache, empty lists on engine failure) → UI `PluginControls` (`ui/src/components/global-bar/PluginControls.tsx`) stores flat strings in `{ "pluginName:paramKey": "value" }`, persisted under localStorage key `hs-pluginParams` for ACE and `hs-pluginParams@<backend>` for other backends (`BACKEND_SCOPED_FIELDS` and `scopedKey()`, `ui/src/stores/globalParamsStore.ts:225, :246-249`) → request field `plugin_params` (`translateParams.ts:336-338`) → engine parses (`hot-step-server.cpp:1021-1046`, `[DIAG] Parsed N plugin_params` in the log) → before every Lua call, `lua_inject_params` (`lua-plugin.h:587-613`) filters by `"<plugin.name>:"` prefix and sets a fresh `params` global. Coercion: numeric string → number, `"true"`/`"false"` → boolean, else string.

Param traps:
- **`params` and any key may be nil** (untouched params are absent from the map). Always default: `local x = (params and params.x) or 0.5`.
- **The `or` idiom is WRONG for toggles defaulting true**: `(params and params.rms_servo) or true` is `true` even when the user turned it OFF (`false or true == true`). `md_pingpong_simple.lua:288` (`rms_servo_on`) carries exactly this latent bug; do not copy it. Correct:
  ```lua
  local v = true
  if params and params.rms_servo ~= nil then v = params.rms_servo end
  ```
- Keys are namespaced by plugin **`name`**, not filename. Renaming the plugin silently orphans users' stored values.
- **Do not rely on the `transform` schema field**: it is extracted and serialized but `PluginControls.tsx` never applies it; values are sent verbatim.
- A value sent as a JSON number rather than a string is stringified with `std::to_string` (six decimal places) before injection, so anything below 0.000001 arrives as 0 (`hot-step-server.cpp:1034-1037`). The UI always sends strings.
- `accent` (UI colorway) must be one of: amber, cyan (default), blue, teal, green, emerald, purple, indigo, orange, pink, rose, sky, violet (`PluginControls.tsx:27-41`).
- Fields like `stork_substeps`, `beat_stability`, `apg_momentum` are a **separate legacy sideband** channel (`hot-step-params.h:150-158`, `translateParams.ts:302-309`), not `plugin_params`. New plugins must use declared `params` only.

## Worked templates (all real, in-repo)

| Want to write… | Copy from |
|---|---|
| Solver, single-eval | `engine/plugins/solvers/euler.lua` (21 lines, canonical) |
| Solver, multi-eval / stateful | `engine/plugins/solvers/unipc.lua` (`needs_model=true`, history reset on `step_index==0`) |
| Solver, full-loop (`owns_loop`) | `plugins/solvers/md_dual_time_v2.lua` (`sample()` + `on_step()`, shared helpers from `md_solver_commons.lua`) |
| Solver, stochastic / heavy params (per-step) | `plugins/solvers/md_pingpong_simple.lua` (pure-Lua seeded RNG, hoisted scratch buffers; but see the toggle-bug note above) |
| Scheduler, simple | `engine/plugins/schedulers/linear.lua` |
| Scheduler with companion `require()` | `engine/plugins/schedulers/beta57.lua` + `beta_math.lua` |
| Guidance, scale-modifying | `engine/plugins/guidance/cfg_pp.lua` (21 lines) |
| Guidance with `post_step()` | `engine/plugins/guidance/cfg_mp.lua` |
| Postprocess | `plugins/postprocess/md_audio_tiled.lua` + `_core` |

## Key files

| Path | Role |
|---|---|
| `engine/src/lua-plugin.h` | The plugin API source of truth: FloatArray views, sandbox, RNG, `model_context`, all call contracts, param extraction |
| `engine/src/lua-plugin-registry.h` | Scan tiers, user-root override, companion exclusion, name lookup + aliases, JSON for `GET /plugins` |
| `engine/src/hot-step-sampler.h` | ACE sampling loop: solver/guidance dispatch, final-step x0, vt snapshot, post_step gating, full-loop `on_step`, linker sentinel |
| `engine/src/hot-step-sampler-trt.h` | ACE TensorRT sampler: solver and guidance dispatch on the TensorRT DiT |
| `engine/src/sampler-schedule.h` | Custom timesteps, scheduler dispatch, shift back-calculation, composite schedulers |
| `engine/src/pipeline-synth-ops.cpp` | `ops_build_schedule` (:334) and the postprocess plugin caller + fallback (`ops_vae_decode_postprocess`, :1699) |
| `engine/src/minimax/mm3-plugins.h`, `engine/src/sa3-refine.h`, `engine/src/yue2/yue2-nar-graph.h` + `yue2-nar-loop.h` | The MM3, SA3 and YuE2 NAR bridges (see "Four samplers" below) |
| `engine/src/hot-step-params.h` | `plugin_params` map + legacy sideband params |
| `engine/tools/hot-step-server.cpp` | Registry init (:3409), `GET /plugins` (:3458), `plugin_params` JSON parse (:1021) |
| `server/src/routes/plugins.ts` | Node proxy `/api/plugins` + 60 s cache + `/reload` |
| `server/src/services/generation/translateParams.ts` | UI params → engine request JSON (`plugin_params`, `postprocess_plugin`) |
| `ui/src/components/global-bar/PluginControls.tsx` | Renders declared params, per-param and per-plugin reset; accent map |
| `ui/src/stores/globalParamsStore.ts` | `hs-pluginParams` localStorage persistence (per backend) |
| `engine/plugins/` + `plugins/` | Built-in and community plugin tiers |
| `docs/dev/plugins-authoring.md` | Committed authoring guide, all plugin types and backends |
| `plugins/README.md` | Community plugins folder README: points at the authoring guide; its full-loop solver section is accurate |

## Failure signatures

| Symptom | Cause → fix |
|---|---|
| Plugin absent from UI dropdown | Lua syntax error at load (grep `[Plugins] ERROR loading` in `ace_engine.log`), including Lua 5.3+ syntax such as `//` that LuaJIT rejects; wrong global table for its directory; empty `name`; duplicate `name` between shipped tiers (first wins); or filename matched companion exclusion (`_constants`/`_math`/`_data`, or `_core` with a non-core sibling) |
| A shipped plugin behaves like someone else's | A data-folder plugin with the same `name` replaced it: grep `User plugin '<name>' … overrides` |
| All dropdowns empty / fallback lists | Node couldn't reach the engine: `/api/plugins` returned empty lists (`plugins.ts:24-28`). Check engine is up on :8085 |
| Output pure noise, generation "succeeds" | `step()` raised (index out of range, nil arithmetic): error printed each step, `xt` never advanced. Grep `[Plugins] ERROR in` |
| Generation fails with `full-loop solver '<name>' failed` | `sample()` raised or is missing (`hot-step-sampler.h:1197-1202`); the `[Plugins] ERROR in full-loop solver` line before it has the traceback |
| `float array is read-only` | Wrote to `vt` / `pred_cond` / `pred_uncond`. Write to `xt` / `result` / scratch tables |
| `attempt to call a nil value (global 'apg')` at load | Called `apg()` at file top level: it is only registered lazily inside `guide()` dispatch |
| `apg()` from `post_step()` behaves oddly (no error) | Silently reuses the stale momentum buffer from the last `guide()` call: never call `apg()` outside `guide()` |
| Multi-eval solver quietly acts like Euler | Expected the original velocity after calling `model_fn`: read original from the snapshot arg `vt`, fresh results from `vt_buf` |
| First step of 2nd generation explodes | Stateful solver didn't reset file-locals on `step_index == 0` (an n-change check alone misses same-length runs) |
| Toggle "can't be turned off" | `(params and params.key) or true` idiom: see Param traps |
| Param changes do nothing | Key mismatch vs schema `key`; edited `apg.lua` (native bypass); or values persisted under an old plugin `name` or another backend's `hs-pluginParams@…` slot |
| Guidance `post_step` never fires | `guidance_scale <= 1` (no CFG), final step, past the `cfg_cutoff_ratio` step (CFG turned off), or the DiT runs on the TensorRT sampler, which has no `post_step` call |
| YuE2 request fails with `unsupported YuE2 NAR solver or scheduler plugin` | The name is not on the YuE2 NAR allowlist (`engine/src/yue2/yue2-request.h:937-949`) |
| Every solver/scheduler/guidance dead after upstream sync | `pipeline-synth-ops.cpp` lost the `hot-step-sampler.h` include: now a link error via the `hotstep_sampler_linked_` sentinel. Run `engine/verify-hooks.ps1` |
| Generation very slow with custom guidance | Each `eval_cond`/`eval_uncond` in `post_step` is a full forward pass: budget them |

## Four samplers run these plugins, not one

The plugin layer never had an ACE dependency: every `lua_call_*` takes raw
`float*`, element counts and a param map. What was ACE-specific was the
*sampler*. Four families now dispatch into it, and a plugin change affects all of them:

| Sampler | Call sites | Notes |
|---|---|---|
| ACE-Step DiT | `hot-step-sampler.h`, `hot-step-sampler-trt.h` | the original; time-major `[T][Oc]`, `t` descends 1→0 |
| MiniMax-Music3 flow DiT | `minimax/mm3-plugins.h` | sigma ascends, latents channel-major: the bridge flips both |
| StableStep / SA3 refine and `/sa3-synth` | `sa3-refine.h` | `t` already descends like ACE; latents channel-major |
| YuE2 NAR decoder | `yue2/yue2-nar-graph.h`, `yue2/yue2-nar-loop.h` | solvers and schedulers only, from an allowlist; time-major, `n = frames * 64` |

**The layout trap, twice learned:** `apg_project()` (`guidance/apg-core.h`)
normalises per channel over time and indexes `[t*Oc + c]`: it is hard-wired to
ACE's **time-major** memory. MM3 and SA3 both store latents **channel-major**,
so both bridges transpose into the ACE view before any guidance plugin sees a
buffer, and back after. Skip that and APG silently normalises scrambled
mixtures rather than channels: it compiles, it runs, it sounds subtly wrong.
Solvers are exempt: `lua_call_solver_step` is handed only a flat `n`, and
`lua_call_solver_loop` uses `T`/`Oc` solely to publish `n_per`, so no solver
can index across channels.

Per-backend caveats worth knowing before you assume a plugin "works everywhere":
- **`owns_loop` solvers**: fine on ACE and the YuE2 NAR, **refused on MM3** (a
  full-loop solver bypasses MM3's per-step window-overlap blend and breaks every
  seam, so the run falls back to native Euler). On SA3 they are refused while the
  pingpong renoise is on, which is the production default, and native pingpong
  runs instead (`sa3-refine.h:504-514`).
- **SA3 wraps per-step Lua solvers in its pingpong renoise** by default
  (`sa3_lua_renoise`), because the distilled model runs hot without it. A solver
  that declares `stochastic = true` is left to inject its own noise instead
  (stacking both once diverged a pingpong solver), and
  `model_context.engine_renoise` tells the plugin which case it is in
  (`sa3-refine.h:670-684`).
- **Guidance on SA3 is near-decorative.** SA3 was trained at cfg=1 and has no
  unconditional branch, so the engine passes the cond velocity as *both*
  predictions. APG-family modes see `diff = 0` and pass through unchanged; only
  plugins doing cond-side work have any effect.
- **Schedulers on SA3 get rescaled.** Schedulers emit `sigma_max = 1.0` for full
  denoising; the SA3 refine is SDEdit and starts at `strength` (~0.3), so the
  returned curve is linearly rescaled onto `[strength → 0]`: spacing preserved,
  starting point not. Note the override array is `steps` long while the loop
  indexes `sigmas[i+1]`, so the terminal `0.0` must be appended (getting this
  wrong is an out-of-bounds read, not a compile error).
- **YuE2 NAR**: only the names in `engine/src/yue2/yue2-request.h:937-949` run;
  `step()` solvers that need `model_fn` are refused, guidance plugins are not
  called, and `model_context.yue2` carries the song plan (see
  `docs/dev/plugins-authoring.md`).
- **Postprocess** runs on ACE, SA3 and YuE2 (each with its own samples per
  frame), not on MM3, whose vocoder has a different contract.
- **`sampler_build_scheduler_override()` reads process-global
  `g_hotstep_params`**, not its arguments. `/sa3-refine` and `/sa3-synth`
  therefore save and restore `solver_name`/`scheduler`/`plugin_params` around
  the call (`hot-step-server.cpp:4064-4085`, :4637-4647): without that, a refine
  leaks its picks into the next ACE generation on the same worker.

## Institutional knowledge

- **VALIDATED**: `hot-step-sampler.h` replaced `dit-sampler.h` as the sampling path; the include lives in upstream `pipeline-synth-ops.cpp` and its loss during a sync used to be silent (everything compiled, all plugins dead). The linker sentinel now makes it a link error. Always run `engine/verify-hooks.ps1` after touching upstream files.
- **VALIDATED**: the engine snapshots `vt` before multi-eval solver steps (`hot-step-sampler.h:1414-1430`) because sharing one buffer between "original velocity" and "model_fn output" silently degraded Heun to Euler.
- **VALIDATED**: stateful plugins must reset on `step_index == 0`; persisting `lua_State`s bleed state across generations (documented in-code at `md_pingpong_simple.lua:263-266`: "explosive velocity on step 1").
- **VALIDATED**: `philox_randn` is NOT exposed to Lua. `sde.lua` says it needs that C helper and that "the C++ wrapper" adds the noise; nothing registers the helper and the engine always passes `use_sde = false` (`pipeline-synth-ops.cpp:1340, :1470`), so `sde` runs its deterministic Euler fallback. For seeded randomness use `math.random` (re-seeded from the job seed each generation) or `math.randomstream(seed)`; `md_pingpong_simple.lua` rolls its own LCG + Box-Muller instead.
- **VALIDATED**: the move to LuaJIT (`ff54dd8d`) kept outputs identical across all bundled plugins in an old-vs-new harness and ran element loops 8-47x faster; Linux and macOS builds were untested at the time.
- The outdated module-return examples that used to live in `plugins/README.md` were removed on 2026-09-25; the README now points at `docs/dev/plugins-authoring.md`. Trust that guide and `engine/src/lua-plugin.h`.
- **VERIFIED IN CODE, NOT BY EAR**: the TensorRT sampler (`hot-step-sampler-trt.h`) dispatches solver `step()`/`sample()` and guidance `guide()` (with the same native-APG bypass, :217) but never calls `post_step()`. It is not a line-for-line copy of the GGML loop, so check a plugin's output on the TensorRT backend before relying on it there.

## Deeper reading

- [reference.md](reference.md) (this folder): full-loop solver contract, `post_step()` details, postprocess internals, param schema JSON shape, docs-vs-code notes.
- `docs/dev/plugins-authoring.md`: committed authoring guide covering every plugin type (full-loop `sample()` and postprocess `process()` included), the scan roots, the `apg()` bridge, `post_step()`, the param schema, globals, `model_context`, RNG seeding and the YuE2 NAR contract, per backend. It was re-checked against the code on 2026-10-23; where it and this skill differ, the code wins.
- `docs/dev/engine.md`: the YuE2 NAR plugin allowlist and plan-conditioning keys.
- `engine/docs/ARCHITECTURE.md`: engine internals, request JSON.
- `docs/plans/`: internal design docs, **gitignored and local-only** (may be absent on a fresh clone).
