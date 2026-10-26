# Lua plugin authoring — deep reference

Companion to [SKILL.md](SKILL.md). All line references verified 2026-10-23 against `master` (HEAD `dda5f6b6`). Where any doc disagrees with `engine/src/lua-plugin.h`, the code wins.

## 1. Full-loop solvers (`owns_loop = true`)

For solvers that need to control the entire sampling iteration (adaptive dispatch, velocity caching, SDE restarts, ping-pong ancestral sampling). Set `owns_loop = true` in the `solver` table and define `sample()` instead of `step()`. Called via `lua_call_solver_loop` (`engine/src/lua-plugin.h:853-937`) from the ACE full-loop branch at `hot-step-sampler.h:1014-1205`; the SA3 and YuE2 NAR bridges call it too (MM3 refuses it).

```lua
function sample(xt, vt_buf, schedule, n, model_fn)
  -- xt:       FloatArray [n], mutable. Contains noise initially.
  -- vt_buf:   FloatArray [n], mutable. model_fn writes velocity here.
  -- schedule: plain Lua table, 1-INDEXED, num_steps descending t values, no trailing 0.
  -- n:        total element count (batch_n * n_per).
  -- model_fn(xt_array, t_val): full CFG'd forward pass; result lands in vt_buf.
end
```

Globals injected (`lua-plugin.h:872-887`): `num_steps`, `batch_n`, `n_per`, `seed`, `params`, `model_context`, and the function `on_step(step_idx, t_curr, t_next) -> bool` (true = cancelled, you should return immediately). The plugin's `math.random` is re-seeded from the job seed before every `sample()` call.

Contract:
1. Call `model_fn(xt, t)` to evaluate the model at any timestep; read the velocity from `vt_buf` afterwards.
2. After completing each step's `xt` update, call `on_step(step_idx, t_curr, t_next)`. **`on_step` is where the engine applies**: cancel checks, cover-switch, DCW correction, guidance `post_step`, repaint injection, CFG cutoff, and progress logging (`hot-step-sampler.h:1037-1184`). Skip it and all of those silently stop working for your solver. It also advances the guidance context to the next step, because a full-loop solver's `model_fn` (which runs guidance) fires before `on_step` (:1079-1085).
3. On the final step leave the x0 prediction in `xt` (`xt[i] = xt[i] - vt_buf[i] * t_curr`); there is no engine-side final step for full-loop solvers. The engine copies `xt` to the output as-is (:1204).
4. A Lua error or a missing `sample()` fails the job (`[DiT] FATAL: full-loop solver '<name>' failed`, :1197-1202), unlike a per-step `step()` error, which is logged and skipped.

Worked example with full commentary: `plugins/README.md` (its "Full-Loop Solvers" section). Real production plugins: `plugins/solvers/md_dual_time_v2.lua`, `md_hamiltonian_v2.lua`, `md_storm_core_V4.lua`.

Known engine-side TODO: DCW for full-loop multi-eval solvers uses the live `vt` rather than a pre-solver snapshot (`hot-step-sampler.h:1089` TODO comment): if your full-loop solver leaves stale velocities in `vt_buf` when calling `on_step`, DCW sees those.

## 2. Guidance `post_step()`: exact semantics

If a guidance plugin defines a global `post_step` function, this is detected at load time (`lua-plugin.h:548-551`, sets `has_post_step`, shown as `[post_step]` in the `[DiT] Guidance:` log line). A plugin may define `post_step()` without `guide()` (`has_guide`, :553-555).

The engine calls it after each solver step under ALL of these conditions (`hot-step-sampler.h:1450` per-step, :1091 inside the full-loop `on_step`):
- `has_post_step` is true, AND
- `do_cfg` is true (guidance_scale > 1 and CFG not yet cut off by `cfg_cutoff_ratio`), AND
- `step < num_steps - 1` (never on the final step).

MM3 (`minimax/mm3-plugins.h:495`) and SA3 (`sa3-refine.h:758`) call it as well; the TensorRT ACE sampler does not.

Signature (7 args, `lua-plugin.h:1073-1142`):

```lua
function post_step(xt, t, n, eval_cond, eval_uncond, vt_cond, vt_uncond)
  -- xt: mutable FloatArray, the latent AFTER the solver step
  -- t:  the timestep just stepped TO
  -- eval_cond(xt_arr, t_val) / eval_uncond(xt_arr, t_val): each is a FULL model
  --   forward pass (expensive). Results land in vt_cond / vt_uncond (FloatArrays).
end
```

Globals: same as `guide()`: `step_idx`, `total_steps`, `dt`, `t_curr`, `params`, `model_context`. `apg()` must NOT be used here: it does not raise; the `_apg_mbuf` global left over from the last `guide()` call (`lua-plugin.h:1025-1026`, never cleared) makes it silently run with the momentum buffer of whichever batch element was guided last. Wrong state, no error.

Real user: `engine/plugins/guidance/cfg_mp.lua` (manifold-projection CFG), `post_step` at line 47.

## 3. Postprocess plugin: full contract

Caller on ACE: `ops_vae_decode_postprocess` in `engine/src/pipeline-synth-ops.cpp:1699`; SA3 (`sa3-refine.h:1029`) and YuE2 (`yue2/yue2-pipeline.h:1644`) call the same bridge. Bridge: `lua_call_postprocess` in `lua-plugin.h:1162-1327`.

```lua
postprocess = { name = "...", display = "...", params = {...} }

function process(latents, B, C_lat, W, C_aud, final_samples, upscale_factor, vae_decode_fn)
  -- latents:        plain Lua TABLE (1-indexed), CHANNEL-major [C_lat, W]
  --                 (C++ holds time-major [T, C_lat]; the bridge transposes for you)
  -- B:              always 1: the engine iterates batch items and calls per item
  -- C_lat:          latent channels of the running backend (64 on ACE)
  -- W:              latent frame count (T_latent)
  -- C_aud:          audio channels (2)
  -- final_samples:  W * upscale_factor (expected audio length per channel)
  -- upscale_factor: audio samples per latent frame, passed by the caller
  --                 (1920 on ACE and YuE2, 4096 on SA3; lua-plugin.h:1173)
  -- vae_decode_fn(latent_table, T_latent) -> audio_table, T_audio
  --                 latent_table must be 1-indexed channel-major [C_lat, T_latent];
  --                 returned audio_table is 1-indexed [2 * T_audio], ch0 then ch1
  -- MUST return: audio_table, T_audio
end
```

Failure handling on ACE: unknown plugin name → `[Postprocess] WARNING: plugin '<name>' not found, falling back to built-in decoder`; `process()` erroring or returning `T_audio <= 0` → `[Postprocess Batch<b>] ERROR: plugin decode failed, falling back to built-in`. SA3 and YuE2 keep their native decode the same way. So a broken postprocess plugin degrades gracefully: check the log to know which path actually ran (`[Postprocess] Using plugin '<name>' for VAE decode (T_latent=N)`, or `[SA3]`/`[YuE2] Postprocess '<name>' decode: …`).

Performance note: this path marshals through Lua tables with a transpose on each side of every `vae_decode_fn` call: it is the deliberately-slow flexible path, not zero-copy like the solver/guidance FloatArrays.

Selection plumbing: UI `postprocessPlugin` (sent only with post-processing and the postprocess toggle on) → `req.postprocess_plugin` (`server/src/services/generation/translateParams.ts:341-343`) → parsed in `engine/src/request.cpp:214-215`, logged as `[Request] postprocess_plugin: <name>` (`request.cpp:892-893`).

Reference implementation: `plugins/postprocess/md_audio_tiled.lua` (and its `V2`/`V2.1` variants), which `require()` `md_audio_tiled_core.lua` (excluded from the plugin scan by the `_core`-with-sibling rule).

## 4. Param schema: full field reference

Extraction: `lua_extract_param` (`lua-plugin.h:391-461`). JSON serialization for `GET /plugins`: `param_to_json` (`lua-plugin-registry.h:270-315`).

```lua
params = {
  { key = "strength",              -- REQUIRED; how you read it: params.strength
    type = "slider",               -- "slider" (default) | "select" | "toggle" | "text"
    label = "Strength",            -- UI label (defaults to key)
    hint = "How hard to push",     -- tooltip
    default = 0.5, min = 0, max = 1, step = 0.01,   -- slider fields
    -- select fields:
    -- default = "a", options = { {value="a", label="Option A"}, "b" },  -- bare strings ok
    -- toggle: default = true|false
    -- text:   default = "some string"
    visible_when = { key = "mode", equals = "advanced" },  -- show only when sibling
                                    -- param "mode" currently equals "advanced" (string compare;
                                    -- a boolean or number equals is formatted like JS String())
    -- transform = "value * 0.05"  -- DO NOT USE: extracted & serialized but the UI
                                    -- never applies it (PluginControls.tsx); values sent verbatim
  },
}
```

Value round trip:
1. UI stores flat strings: `{ "<plugin.name>:<key>": "value" }` in Zustand + localStorage `hs-pluginParams` (ACE) or `hs-pluginParams@<backend>` (`ui/src/stores/globalParamsStore.ts:225`, `setPluginParam`/`resetPluginParams` at :718-732).
2. Sent as `plugin_params` on the generation request (`translateParams.ts:336-338`); parsed engine-side (`engine/tools/hot-step-server.cpp:1021-1046`, log `[DIAG] Parsed N plugin_params`); copied to `g_hotstep_params.plugin_params` (`hot-step-server.cpp:1524`; declared `engine/src/hot-step-params.h:255-256`).
3. `lua_inject_params` (`lua-plugin.h:587-613`) filters by `"<plugin.name>:"` prefix, strips it, sets a fresh global `params` table before EVERY Lua call. Coercion: fully-numeric string → Lua number; `"true"`/`"false"` → boolean; anything else → string.

Consequences worth internalizing:
- A param the user never touched is simply **absent**: `params.key` is nil, and `params` itself can be an empty table. Defaults in the schema are UI-side only; your Lua code must re-default.
- Because values transit as strings and are re-coerced, a `text` param containing `"2.5"` arrives as a *number*. If you need a literal numeric-looking string, that is not expressible.
- The 60 s Node cache (`server/src/routes/plugins.ts:13-14`) only caches the *schema list*, not values: `POST /api/plugins/reload` is only needed after metadata changes, and an engine restart is still required for the engine to see file edits.

## 5. Name lookup, aliases, defaults

`lua-plugin-registry.h:72-112`:
- Solver: alias `"ode"` → `euler`. Empty name → first entry of an `unordered_map` (nondeterministic; the sampler always passes a concrete name in practice). Unknown at generation time → `[DiT] ERROR: unknown solver '<name>', falling back to euler` (`hot-step-sampler.h:709`).
- Scheduler: alias `"karras"` → `sgm_uniform`; parameterized names fall back on the prefix before `:` (`"power:4.00"` → `power`). An empty scheduler never reaches lookup: `ops_build_schedule` keeps the upstream shifted-linear default (`pipeline-synth-ops.cpp:364`, `hot-step-params.h:133`). Unknown → `[DiT] WARNING: unknown scheduler '<name>', using linear` (`sampler-schedule.h:140`).
- Guidance: unknown → fallback to `apg` with an ERROR log (`hot-step-sampler.h:730`).
- Duplicate `name` across files: between the engine and repo-root tiers, first loaded wins, `[Plugins] WARNING: duplicate plugin '<name>' from <path> (keeping first: <path>)`. A data-folder (`--user-plugins`) plugin replaces the shipped one: `[Plugins] User plugin '<name>' from <path> overrides <path>`. Load order is engine tier, repo-root tier, then data-folder tier, alphabetical within a directory.

## 6. Docs-vs-code notes (code wins)

1. **Old module-return examples** (`return { ..., step = function(...) }`, `x:get(i)` / `x:set(i,v)`, a `schedule()` returning a table) still circulate in community files and were in `plugins/README.md` until 2026-09-25. The real API is a global `solver = {...}` table plus a global `function step(xt, vt, t_curr, t_prev, n)` using `xt[i]` indexing. Every shipped plugin uses the real style.
2. **`transform` is dead in the UI**: extracted (`lua-plugin.h:396`), serialized (`lua-plugin-registry.h:281`), typed (`ui/src/types/pluginTypes.ts`), never applied by `PluginControls.tsx`.
3. **`shift` is not a UI value**: ACE and SA3 back-calculate it from the upstream schedule's second timestep and clamp it (`<0.5 → 1.0`, `>10 → 3.0`, `engine/src/sampler-schedule.h:62-75`); MM3 passes the request's `flow_shift`, YuE2 passes 1.0.
4. **`docs/dev/plugins-authoring.md` was rewritten against the code on 2026-10-23** and now documents the full-loop `sample()`/`on_step` contract (`lua-plugin.h:853`), the postprocess `process()` contract (`lua-plugin.h:1162`), the user-root override and the per-backend differences. Earlier gaps listed here (shift "from UI", no full-loop or postprocess section, a Quick Start that said a matching user plugin is skipped) are fixed there.
5. **Global naming split**: solvers get `step_index`; guidance (`guide()` and `post_step()`) gets `step_idx`. Two names, same concept: do not mix them up across plugin types.

## 7. Handy debug snippets (PowerShell)

```powershell
# Newest session's plugin lines (<data root> from: cd server; npx tsx scripts/data-paths.ts)
$s = Get-ChildItem "<data root>\logs" -Directory | Sort-Object Name -Descending | Select-Object -First 1
Select-String -Path "$($s.FullName)\ace_engine.log" -Pattern '\[Plugins\]'

# Runtime errors from any plugin type
Select-String -Path "$($s.FullName)\ace_engine.log" -Pattern '\[Plugins\] ERROR'

# Which solver/scheduler/guidance actually ran in the last generation
Select-String -Path "$($s.FullName)\ace_engine.log" -Pattern '\[DiT\] (Solver|Guidance|Custom schedule)|\[Postprocess\]'

# What params the engine actually received
Select-String -Path "$($s.FullName)\ace_engine.log" -Pattern 'Parsed \d+ plugin_params'

# Dump your plugin's schema as the UI sees it
(Invoke-RestMethod http://localhost:3000/api/plugins).solvers |
  Where-Object { $_.name -eq 'my_solver' } | ConvertTo-Json -Depth 6
```
