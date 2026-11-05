# Plugin authoring guide

How to create custom solvers, schedulers, guidance modes and VAE-decode
postprocessors for the HOT-Step CPP engine, and how to test them. The host side
lives in `engine/src/lua-plugin.h` (calls, sandbox, float arrays) and
`engine/src/lua-plugin-registry.h` (scanning and `GET /plugins`).

---

## Quick start

1. Create a `.lua` file in the folder for its type:
   - `solvers/`: ODE/SDE solvers
   - `schedulers/`: noise schedules
   - `guidance/`: CFG guidance modes
   - `postprocess/`: replacements for the built-in tiled VAE decode

   In a checkout, the shipped plugins live under `engine/plugins/` and the project
   root's `plugins/`, and either works for your own.
2. Declare a metadata table named after the type: `solver`, `scheduler`, `guidance`
   or `postprocess`.
3. Implement the function the type needs: `step` (or `sample` for a full-loop
   solver), `schedule`, `guide` or `process`.
4. Restart the app. The engine loads plugins once at startup, and the plugin then
   appears in the UI with its parameters.

In a release, put your own plugins in the data folder's `plugins/` (same
`solvers/`, `schedulers/`, `guidance/` and `postprocess/` subfolders), so they
survive an update. The engine scans it after the shipped plugins; a plugin whose
name matches a shipped one replaces it, and the engine log names both files.

No C++ rebuild required. The engine loads all `.lua` files at startup.

---

## Where the engine looks

`PluginRegistry::init` scans these roots in order, each with `solvers/`,
`schedulers/`, `guidance/` and `postprocess/` subfolders:

1. `engine/plugins/`
2. `plugins/` at the project root
3. the root `ace-server` gets with `--user-plugins`, which the app sets to
   `<data folder>/plugins` whenever it runs with a data folder

Files load in name order within a folder. If two files in the first two roots
declare the same `name`, the first one loaded wins and the log warns about the
other. A file in the user root replaces the shipped plugin of the same name.

A file is skipped, with a warning in the engine log, when it has no `solver`,
`scheduler`, `guidance` or `postprocess` table, when its `name` is empty, or when
its table does not match its folder. Some files are never loaded as plugins
because they are companion data for `require()`: any file whose name contains
`_constants`, `_math` or `_data`, and `<name>_core.lua` when `<name>.lua` sits
beside it. Keep those strings out of a plugin's own file name.

---

## Plugin types

### Solver

Solvers advance the latent state `xt` by one step along the ODE/SDE trajectory.

The metadata goes in a global `solver` table:

```lua
solver = {
    name        = "my_solver",           -- unique ID, what infer_method sends (required)
    display     = "My Solver (2 NFE)",   -- name in the UI dropdown (defaults to name)
    description = "A custom solver",     -- hover text on the picker
    accent      = "cyan",                -- colour of the plugin's parameter panel
    nfe         = 2,                     -- function evaluations per step (default 1)
    order       = 2,                     -- solver order, informational (default 1)
    needs_model = true,                  -- step() receives model_fn and vt_buf (default false)
    stateful    = false,                 -- informational (default false)
    stochastic  = false,                 -- injects its own noise (default false)
    owns_loop   = false,                 -- full-loop solver, sample() instead of step() (default false)
    params      = {},                    -- UI parameters, see Parameter schema
}
```

`accent` is one of `amber`, `cyan`, `blue`, `teal`, `green`, `emerald`, `purple`,
`indigo`, `orange`, `pink`, `rose`, `sky` or `violet` (cyan when absent), for every
plugin type. The UI lists 1-NFE solvers first, then the multi-evaluation ones with
"(N NFE)" after the name. `stochastic` matters on SA3, which skips its own pingpong
re-noise for a solver that declares it.

A single-evaluation solver defines `step()`:

```lua
function step(xt, vt, t_curr, t_prev, n)
    -- xt:     mutable float array, the current latent (modify in place)
    -- vt:     read-only float array, the velocity at (xt, t_curr)
    -- t_curr: current timestep
    -- t_prev: next timestep (the step goes from t_curr to t_prev)
    -- n:      total elements in xt and vt

    local dt = t_curr - t_prev
    for i = 0, n - 1 do
        xt[i] = xt[i] - vt[i] * dt
    end
end
```

A multi-evaluation solver sets `needs_model = true` and gets two more arguments:

```lua
function step(xt, vt, t_curr, t_prev, n, model_fn, vt_buf)
    -- model_fn(x, t): evaluates the model at (x, t) and writes the velocity to vt_buf.
    --                 x must be one of the float arrays this call received.
    -- vt_buf:         mutable float array that receives model_fn's output

    local dt = t_curr - t_prev
    local t_mid = t_curr - 0.5 * dt

    -- Save state
    local k1 = {}
    local xt_orig = {}
    for i = 0, n - 1 do
        k1[i] = vt[i]
        xt_orig[i] = xt[i]
    end

    -- Evaluate at midpoint
    for i = 0, n - 1 do
        xt[i] = xt_orig[i] - 0.5 * k1[i] * dt
    end
    model_fn(xt, t_mid)  -- result appears in vt_buf

    -- Final update using midpoint velocity
    for i = 0, n - 1 do
        xt[i] = xt_orig[i] - vt_buf[i] * dt
    end
end
```

`model_fn` writes the velocity with CFG and the selected guidance applied, the same
velocity the main loop computes. `vt` is a snapshot taken before the step, so it
still holds the original velocity after `model_fn` overwrites `vt_buf`.

ACE and MM3 never call `step()` for the final step: the engine finishes with its own
update to `t = 0` (`x0 = xt - vt * t_curr`). SA3 and YuE2 call `step()` for every
step, the last one with `t_prev = 0`. YuE2 passes no `model_fn`, so it refuses
`needs_model` solvers unless they own the loop.

---

### Full-loop solvers

A solver with `owns_loop = true` defines `sample()` instead of `step()` and runs the
whole denoising loop itself:

```lua
function sample(xt, vt_buf, schedule, n, model_fn)
    for i = 1, #schedule do
        local t = schedule[i]
        local t_next = schedule[i + 1] or 0.0
        model_fn(xt, t)                    -- guided velocity at (xt, t) lands in vt_buf
        for j = 0, n - 1 do
            xt[j] = xt[j] - vt_buf[j] * (t - t_next)
        end
        if on_step(i - 1, t, t_next) then return end
    end
end
```

- `schedule` is a Lua table indexed from 1, holding the `num_steps` descending `t`
  values with no trailing 0. The update after the last entry goes to `t = 0`.
- `model_fn(x, t)` writes the guided velocity for `(x, t)` into `vt_buf`. `x` must
  be one of the float arrays the call received.
- Call `on_step(i, t, t_next)` after each step, with `i` counted from 0, and return
  as soon as it answers `true`: that is a cancel. On ACE the engine does its
  per-step work inside `on_step` (DCW, the guidance plugin's `post_step()`, repaint
  injection, the cover context switch and the CFG cutoff), so a solver that never
  calls it skips all of them.
- The result is whatever `xt` holds when `sample()` returns. A Lua error or a missing
  `sample()` fails the job.

Besides `params` and `model_context`, `sample()` sees the globals `num_steps`,
`batch_n`, `n_per`, `seed` and the `on_step` function. The engine reseeds the
plugin's RNG from the job seed before every `sample()` call.

ACE runs full-loop solvers. SA3 runs them only with `sampler=euler`; with the
default pingpong sampler it uses its native pingpong step instead. MM3 refuses them
and uses its native Euler update. YuE2 runs the ones on its allowlist (see
[Full-loop solvers on YuE2](#full-loop-solvers-on-yue2)).

---

### Scheduler

Schedulers produce a timestep sequence for the denoising trajectory.

The metadata goes in a global `scheduler` table:

```lua
scheduler = {
    name        = "my_schedule",
    display     = "My Schedule",
    description = "Custom noise schedule",
}
```

The plugin defines `schedule()`:

```lua
function schedule(output, num_steps, shift)
    -- output:    mutable float array of num_steps entries; write every one
    -- num_steps: number of timesteps to generate
    -- shift:     timestep shift, see below

    for i = 0, num_steps - 1 do
        output[i] = 1.0 - i / num_steps
    end
    apply_shift(output, num_steps, shift)
end
```

Timesteps go from `1.0` (pure noise) toward `0.0` (clean signal). The engine adds
the final step to `0.0` itself, so produce `num_steps` values, not `num_steps + 1`.

`shift` is not a UI value on every backend. ACE and SA3 back-calculate it from
their own default schedule (a value below 0.5 becomes 1.0 and one above 10 becomes
3.0), MM3 passes the request's `flow_shift`, and YuE2 passes 1.0.

On ACE a request may also name `power:4.00`: a name with a colon falls back to the
plugin named before the colon. `karras` is an alias for `sgm_uniform`, and
`composite:<a>+<b>:<crossover>:<split>` blends two schedulers. An unknown name uses
`linear`. YuE2 checks the result: the first value must be 1, and each later value
must be lower than the one before it and above 0, or the request fails. It fills
`output` with NaN first, so an entry the plugin skips fails the request too.

A common helper is the shift warp:

```lua
function apply_shift(ts, n, shift)
    if shift == 1.0 then return end
    for i = 0, n - 1 do
        local t = ts[i]
        ts[i] = shift * t / (1.0 + (shift - 1.0) * t)
    end
end
```

---

### Guidance

Guidance modes control how the conditional and unconditional model predictions are combined.

The metadata goes in a global `guidance` table:

```lua
guidance = {
    name        = "my_guidance",
    display     = "My Guidance",
    description = "Custom guidance mode",
}
```

The plugin defines `guide()`:

```lua
function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    -- pred_cond:      read-only float array, conditional velocity prediction
    -- pred_uncond:    read-only float array, unconditional velocity prediction
    -- guidance_scale: the guidance scale (w) from the UI
    -- result:         mutable float array, write the guided velocity here
    -- Oc:             channels per timestep frame
    -- T:              number of timestep frames (n = Oc * T)
    -- norm_threshold: APG norm threshold from the UI

    -- Route through APG for stability (strongly recommended)
    apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
end
```

> Always route through `apg()` for the base guidance computation. Raw linear interpolation (`result = uncond + w * (cond - uncond)`) produces severe audio artifacts (static, underwater sound, frequency distortion). The `apg()` function provides momentum smoothing, perpendicular projection, and norm thresholding that are essential for stable audio output. If your guidance mode needs custom math, apply it as a correction on top of the APG result.

`guide()` runs once per batch item for every model evaluation that uses CFG, the
evaluations a multi-eval solver makes through `model_fn` included. ACE calls it only
while CFG is on (`guidance_scale` above 1, and before any CFG cutoff). MM3 calls it on
every step once a guidance plugin is selected. SA3 calls it only with
`guidance_scale` above 1, and passes its single conditional prediction as both
`pred_cond` and `pred_uncond` because SA3 has no unconditional branch, which makes
APG-style modes a pass-through there. YuE2 has no guidance plugin slot.

A request that names `apg` runs the native C++ APG on ACE, MM3 and SA3;
`engine/plugins/guidance/apg.lua` gives the picker its entry, and its body never
runs.

The globals available in guidance plugins:

| Global | Type | Description |
|--------|------|-------------|
| `step_idx` | int | Current step index (0-based) |
| `total_steps` | int | Total number of denoising steps |
| `dt` | float | Current timestep delta (t_curr - t_next) |
| `t_curr` | float | Current timestep value |
| `params` | table | User-configured parameter values |
| `model_context` | table | The running backend, see [`model_context`](#model_context-all-plugin-types) |

---

### Postprocess

A postprocess plugin replaces the built-in tiled VAE decode. The metadata goes in a
global `postprocess` table (`name`, `display`, `description`, `accent`, `params`),
and the plugin defines `process()`:

```lua
postprocess = {
    name        = "my_decoder",
    display     = "My Decoder",
    description = "Plain decode through the engine's VAE",
}

function process(latents, B, C_lat, W, C_aud, final_samples, upscale_factor, vae_decode_fn)
    -- latents:        Lua table indexed from 1, channel-major: latents[c * W + t + 1]
    -- B:              always 1 (process() runs once per batch item)
    -- C_lat:          latent channels
    -- W:              latent frames
    -- C_aud:          audio channels (2)
    -- final_samples:  W * upscale_factor, the expected samples per channel
    -- upscale_factor: audio samples per latent frame
    -- vae_decode_fn(lat, frames): decodes `frames` frames from a table laid out like
    --                 latents. Returns audio, n: audio[1..n] is channel 0 and
    --                 audio[n+1..2n] channel 1. Returns nil, 0 on failure.
    local audio, n = vae_decode_fn(latents, W)
    return audio, n
end
```

Return `audio, n` in the same layout. A count of 0 or less, a value that is not a
table, or a Lua error counts as a failure, and the engine falls back to its built-in
decoder. `upscale_factor` is 1920 on ACE and YuE2 and 4096 on SA3. Unlike the other
types, `process()` works on plain Lua tables, not float arrays, so the latent and the
audio are copied in and out.

ACE picks the plugin with the request's `postprocess_plugin` field, SA3 with the
`postprocess_plugin` query parameter on `/sa3-refine` and `/sa3-synth`, and YuE2 with
its own `postprocess_plugin` field. MM3 has no postprocess slot: its vocoder is a
different decoder. `plugins/postprocess/md_audio_tiled.lua` is a full example; it
loads its core module with `require("md_audio_tiled_core")`.

---

## The `apg()` bridge

`apg()` runs the native C++ APG (Adaptive Projected Guidance,
`engine/src/guidance/apg-core.h`) from Lua:

1. Momentum: the difference `cond - uncond` is folded into a running average kept
   per batch item, with momentum -0.75 by default (on ACE, minus the sideband
   `apg_momentum`).
2. Norm thresholding: each channel's difference is scaled down so that its L2 norm
   over time is at most `norm_threshold` (0 turns this off).
3. Projection: the part of the difference parallel to `pred_cond` is dropped, and
   the result is `pred_cond + (guidance_scale - 1) * orthogonal part`.

The signature:

```lua
apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
```

`norm_threshold` is optional (2.5). The engine registers `apg()` on the first
`guide()` call. Call it from `guide()`, where it uses the momentum buffer of the
batch item that call is for.

---

## Advanced: the `post_step()` hook

For guidance modes that need to run extra model evaluations *after* the solver step (e.g., manifold projection), guidance plugins can declare a `post_step()` function. The engine detects this at load time and calls it after each solver step.

When to use it:
- Your guidance technique requires iterative refinement of the latent state
- You need to evaluate the model at positions different from the main solver trajectory
- The technique calls the model with conditioning and unconditioning separately

Each call to `eval_cond` or `eval_uncond` runs a full model forward pass, so use them sparingly.

```lua
function post_step(xt, t, n, eval_cond, eval_uncond, vt_cond, vt_uncond)
    -- xt:          mutable float array, the latent after the solver step (modify in place)
    -- t:           the timestep just stepped to (t_next)
    -- n:           total elements in xt
    -- eval_cond:   function(x, t), runs the model with conditioning, writes to vt_cond
    -- eval_uncond: function(x, t), runs the model without conditioning, writes to vt_uncond
    -- vt_cond:     mutable float array, output buffer for the conditional velocity
    -- vt_uncond:   mutable float array, output buffer for the unconditional velocity

    -- Example: one iteration of manifold projection
    local a = math.abs(dt) / 2.0

    eval_uncond(xt, t)                        -- fills vt_uncond
    for i = 0, n - 1 do
        xt[i] = xt[i] - a * vt_uncond[i]     -- push away from uncond manifold
    end

    eval_cond(xt, t)                          -- fills vt_cond
    for i = 0, n - 1 do
        xt[i] = xt[i] + a * vt_cond[i]       -- pull toward cond manifold
    end
end
```

The `post_step` hook has access to the same globals as `guide()` (`step_idx`, `total_steps`, `dt`, `t_curr`, `params`).

The hook is not called on the final step, where the latent is about to be decoded.
ACE calls it only while CFG is on, and SA3 only when its guidance runs
(`guidance_scale` above 1). MM3 calls it whenever the plugin is selected. On SA3
both eval functions run the same conditional model and write the same buffer.

A guidance plugin that only needs `post_step()` still has to declare `guide()`. ACE and MM3 call `guide()` every step and its `result` buffer holds the previous step's velocity on entry, so a missing `guide()` leaves stale or zero velocity whenever CFG is on. Route it through native APG:

```lua
function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
end
```

---

## Parameter schema

Plugins can declare user-facing parameters that appear in the UI. Parameters are
defined in the `params` array of the metadata table. Each entry needs a `key`;
`label` defaults to the key and `type` to `slider`. `hint` becomes the hover
explanation beside the label. The UI shows the parameters in a collapsible panel
under the plugin's picker, with a Reset button.

### Slider

```lua
{ key = "strength", type = "slider", label = "Strength",
  default = 0.5, min = 0.0, max = 1.0, step = 0.01,
  hint = "Controls the effect intensity" }
```

Missing slider fields default to `default = 0`, `min = 0`, `max = 1` and
`step = 0.01`.

### Select (dropdown)

```lua
{ key = "mode", type = "select", label = "Mode",
  default = "fast",
  options = {
      { value = "fast",    label = "Fast" },
      { value = "quality", label = "Quality" },
  },
  hint = "Choose between speed and quality" }
```

An option may also be a plain string, which serves as both value and label.

### Toggle

```lua
{ key = "enabled", type = "toggle", label = "Enable Feature",
  default = false,
  hint = "Turn this feature on or off" }
```

### Text

```lua
{ key = "pattern", type = "text", label = "Pattern",
  default = "",
  hint = "x.x.x." }
```

A text box. Its `hint` is also the placeholder text.

### Conditional visibility

Parameters can be shown/hidden based on another parameter's value:

```lua
{ key = "sub_param", type = "slider", label = "Sub-Parameter",
  default = 1.0, min = 0.0, max = 5.0, step = 0.1,
  visible_when = { key = "mode", equals = "quality" },
  hint = "Only visible when Mode is set to Quality" }
```

> Use string values for `equals` (e.g. `"true"`, `"2"`) for maximum compatibility. The engine also accepts Lua booleans (`true`/`false`) and numbers directly, so `equals = true` now works. Numbers are written the way JavaScript's `String()` writes them (`2`, `2.5`, `0.00001`, `1e+21`), since that is the form the UI compares against. Older builds only accept strings, so prefer strings if you distribute plugins for use with unpatched releases.

### Transform expressions

The schema accepts a `transform` field (for example `transform = "value * 0.05"`).
The engine reads it and returns it in `GET /plugins`, but the UI does not apply it:
the plugin receives the value exactly as the control shows it. Do any scaling inside
the plugin.

### Reading parameters

Parameters are available via the `params` global table, keyed by their `key` field:

```lua
local strength = (params and params.strength) or 0.5
local mode     = (params and params.mode) or "fast"
local enabled  = (params and params.enabled) or false
```

The engine builds `params` before every call from the request's `plugin_params`
entries named `<plugin name>:<key>`. Values travel as strings and are converted on
the way in: a string that parses fully as a number becomes a number, `"true"` and
`"false"` become booleans, and anything else stays a string. A select option or text
value such as `"2"` therefore arrives as the number 2.

The UI sends only the values the user has changed, so a parameter left at its
default is absent (nil) even though the control shows the default. Always fall back
to the declared default. `params` itself is always a table during a call. For a
toggle whose default is `true`, test for nil instead of using `or`, since
`params.flag or true` turns a user's `false` back into `true`.

---

## Float arrays

Every array argument except postprocess's tables is a float array: a view the host
builds over the engine's own buffer, so nothing is copied. Lua reads and writes the
same memory the C++ engine uses.

Indexing is 0-based, matching C++ rather than Lua's usual 1-based convention.

```lua
-- Read
local val = xt[i]

-- Write (only on mutable arrays)
xt[i] = val

-- Length
local n = #xt
```

Read-only arrays (`vt` in `step()`, `pred_cond` and `pred_uncond` in `guide()`)
raise an error on a write, and any array raises one for an index outside
`0 .. #xt - 1`. `model_fn`, `eval_cond`, `eval_uncond` and `apg()` accept only these
arrays, not Lua tables.

---

## Available globals

### Solver globals

These are set before every `step()` call:

| Global | Type | Description |
|--------|------|-------------|
| `step_index` | int | Current step index (0-based) |
| `batch_n` | int | Number of batch elements |
| `n_per` | int | Elements per batch element |
| `seed` | int | This generation's job seed (batch item 0) |
| `seed_b`, `seed_blend` | int, float | The second seed and blend of SA3 seed breeding; -1 and 0 everywhere else |
| `params` | table | Plugin parameters |
| `model_context` | table | The running backend, see below |

A full-loop solver's `sample()` gets `num_steps`, `batch_n`, `n_per`, `seed` and
`on_step` instead (see [Full-loop solvers](#full-loop-solvers)).

**Stochastic solvers and reproducibility.** Before the first `step()` of a
generation (and before every `sample()` of a full-loop solver) the engine seeds
the plugin's host RNG directly, so `math.random()` is a pure
function of the job seed and identical payloads render identically. Do **not**
re-seed from a clock or leave the RNG unseeded: the engine seeds `math.random`
from the clock when the state is created, and plugin states live for the whole
ace-server process, so an unseeded plugin makes every render irreproducible.
For an independent stream, call `local random_b = math.randomstream(seed_b)`.
The returned function accepts the same arguments as `math.random` and uses a
separate host-backed xoshiro256** state, initially equivalent to
`math.randomseed(seed_b, 0)`. Seeds must be finite integers within JavaScript's
safe integer range. Creating or drawing from this stream leaves the main RNG
unchanged. Create it again at `step_index == 0` to replay a generation, and reset
any cached Gaussian spare there too; a Lua wrapper around `math.randomseed`
does not observe the host's direct reseed. LuaJIT's Lua bit operators are
32-bit and `math.tointeger` is unavailable, so do not use them to emulate this
64-bit generator.

### Guidance globals

| Global | Type | Description |
|--------|------|-------------|
| `step_idx` | int | Current step index (0-based) |
| `total_steps` | int | Total denoising steps |
| `dt` | float | Timestep delta |
| `t_curr` | float | Current timestep |
| `params` | table | Plugin parameters |

Note the two spellings: solvers get `step_index`, guidance gets `step_idx`.

---

### `model_context` (all plugin types)

Every call (solver step, scheduler, guidance, postprocess) also gets a
`model_context` global with info about which backend is currently running:

| Field | Type | ACE | MM3 | SA3 | YuE2 | HeartMuLa |
|-------|------|-----|-----|-----|------|-----------|
| `native_sr` | int | 48000 | 44100 | 44100 | 48000 | 48000 |
| `latent_fps` | int | 25 | the flow latent rate, read from the model (about 86) | 10 | 25 | 25 |
| `latent_channels` | int | 64 | the DiT's input channels, read from the model | 256 | 64 | 256 |
| `model_id` | string | `"ace"` | `"mm3"` | `"sa3"` | `"yue2"` | `"heartmula"` |
| `latent_rms` | float | 2.0 | 2.35 (range 1.76-3.14) | 0.685 | 0.935 for postprocess, 2.0 on the NAR path | 2.0 (default, not measured) |
| `engine_renoise` | bool | false | false | true while SA3's pingpong re-noise wraps a per-step Lua solver | false | false |
| `yue2` | table | nil | nil | nil | the song plan, NAR solvers and schedulers only (below) | nil |

`latent_rms` is roughly how "loud" that model's native latent space runs.
A rms-target slider tuned against ACE's ~2.0 and pointed at SA3 unchanged
just keeps inflating the signal past where SA3 actually settles, like a
thermostat built for a big house bolted onto a studio apartment. Plugins
that expose an rms-target param should scale it by `model_context.latent_rms
/ 2.0` instead of hardcoding the ACE number. `md_solver_commons.lua` has
the helpers: `C.rms_scale()` returns that ratio, and `C.rms_scaled(p, key,
ace_default)` resolves a param with the convention that **`0` means "auto"**
(scale `ace_default` by the ratio) and any nonzero value is the user's
explicit override, used as-is.

Watch the trap: in Lua, `0` is truthy, only `nil` and `false` are falsy. A
naive `params.rms_target or default` will keep a user's `0` as `0` instead
of treating it as "unset". Compare against `0` explicitly, the way
`C.rms_scaled` does, rather than relying on Lua's truthiness.

`engine_renoise` tells a stochastic solver that the engine already re-noises after
each step; when it is false on SA3, the solver injects its own noise.

`model_context` is always present with sane ACE-matching defaults, so a
plugin that never reads it behaves exactly as before.

### `model_context.yue2` (YuE2 NAR solvers and schedulers only)

On the YuE2 NAR path (a Lua `infer_method` or `scheduler` on `/yue2/synth`),
`model_context.yue2` carries the song plan (the ABC score the LM wrote or the
user supplied) plus the chunk being solved. It is `nil` on every other path
(ACE, MM3, SA3, YuE2 postprocess), so guard with
`local y = model_context and model_context.yue2`.

| Field | Type | Meaning |
|-------|------|---------|
| `tempo` | number | quarter-note bpm from the plan's `Q:` line, 0 if missing |
| `meter`, `beats_per_bar` | string, int | `M:` text (e.g. `"4/4"`) and its numerator |
| `bars` | int | vocal-voice bars (`Z4` counts as 4) |
| `vocal_bars`, `vocal_pct` | int, number | bars with at least one sung note, and that as a percent of `bars` |
| `total_s` | number | `bars * beats_per_bar * 60 / tempo` (0 without a tempo) |
| `sections` | array | `{name, start_s, end_s, start_bar, end_bar}` from the plan's `% name` comment lines (lowercased, e.g. `"chorus"`, `"pre-chorus"`), empty if the plan has none |
| `frame_rate` | number | NAR latent frames per second (25) |
| `song`, `song_frames` | int | batch song index and its total latent frames |
| `chunk` | table | `{index, count, start_frame, end_frame, start_s, end_s}`; index is 0-based, end is exclusive |

Section times are plan time at a constant tempo; the rendered audio can run a
little long or short of them, and a preview render (`preview_max_frames`)
only covers the start. Schedulers run once per chunk, so a scheduler can shape
each chunk differently:

```lua
local y = model_context and model_context.yue2
local dense = false
if y then
    for _, s in ipairs(y.sections) do
        if s.name == "chorus" and s.start_s < y.chunk.end_s and s.end_s > y.chunk.start_s then
            dense = true
        end
    end
end
local gamma = dense and 2.5 or 1.0
for i = 0, num_steps - 1 do output[i] = 1.0 - (i / num_steps) ^ gamma end
```

`plugins/schedulers/md_plan_aware_yue2.lua` is the full version, and it logs
the table once per chunk.

#### `plan_strength` (YuE2 NAR schedulers only)

A YuE2 NAR scheduler may also set a global array `plan_strength` with one
number per step (1-indexed, length `num_steps`). After `schedule()` returns,
the engine reads it, clears it, and scales the plan conditioning (the AR-prefix
V the NAR decoder attends to) by `plan_strength[i + 1]` during step `i`.
1 = stock, below 1 loosens the plan, above 1 leans on it harder (clamped to
0..4). If it is missing or the wrong length, the engine uses the
`yue2:plan_w_start` / `yue2:plan_w_end` / `yue2:plan_w_curve` (linear or
cosine) params instead, which default to 1 (off, bit-identical).

```lua
local w = {}
for i = 0, num_steps - 1 do
  output[i] = 1.0 - i / num_steps
  w[i + 1] = math.min(1.0, 0.5 + i / num_steps)  -- loose early, locked later
end
plan_strength = w
```

`plugins/schedulers/md_plan_loose_yue2.lua` is the full version.

#### Full-loop solvers on YuE2

The YuE2 NAR runs `owns_loop = true` solvers through the same
`sample(xt, vt_buf, schedule, n, model_fn)` contract as ACE, one chunk per
call. `schedule` holds the chunk's descending `t` values (1 = noise) with no
trailing 0, and the update after the last entry goes to `t = 0`.
`model_fn(x, t)` writes the plan-conditioned velocity into `vt_buf`, with any
plan strength, plan CFG and CADS applied, and `t` must lie in [0, 1]. Call
`on_step(i, t, t_next)` after each step and return when it answers `true`: that
is a cancel, a model error or non-finite latents, and every later `model_fn`
call writes zeros without running the model. The latent is time-major,
`n = frames * 64`. Multi-evaluation `step()` solvers (`needs_model` without
`owns_loop`) are still refused on YuE2. Only names on the engine's YuE2
allowlist run there; the
[YuE2 request reference](../../engine/docs/ARCHITECTURE.md#yue2-request-post-yue2synth)
lists them.
`plugins/solvers/md_tlie_yue2.lua` and `md_latent_eq_yue2.lua` are examples.

---

## Where each plugin type runs

| Backend | Solvers | Schedulers | Guidance | Postprocess |
|---|---|---|---|---|
| ACE | `step()` and full-loop | yes | `guide()` and `post_step()` | yes |
| MM3 flow stage | `step()` only; full-loop falls back to native Euler | yes | `guide()` and `post_step()` | no |
| SA3 (`/sa3-refine`, `/sa3-synth`) | `step()`; full-loop only with `sampler=euler` | yes | `guide()` and `post_step()`, conditional prediction only | yes |
| YuE2 NAR | allowlisted `step()` (single-eval) and full-loop solvers | allowlisted | no | yes (VAE stage) |

MM3 denoises a song as a run of independent windows and starts each one at
`step_index` 0, so a stateful plugin resets per window. The MM3 solver picker in
the UI hides full-loop solvers.

---

## Sandbox

Each plugin runs in an isolated LuaJIT 2.1 VM (Lua 5.1 with LuaJIT's Lua 5.2 extensions, including `goto`). The standard Lua 5.1 base library is available (`print`, `type`, `pairs`, `ipairs`, `tonumber`, `tostring`, `pcall`, `error`, `setmetatable`, `select`, `unpack` and the rest), together with `math`, `string`, `table`, `bit`, `coroutine` and `require` for companion data files.

`os`, `io`, `debug`, `dofile`, `loadfile`, `ffi`, `jit` and `package.loadlib` are removed, so a plugin has no filesystem access, no shell commands, no process control and no raw memory.

**Float arrays** (`xt`, `vt` and the rest) are indexed from 0 and support `#xt`. Out-of-range indexes and writes to read-only inputs raise an error. Plain `for i = 0, n - 1 do ... end` loops over them are compiled by the JIT, so there is no need to batch work into helper calls. There is no integer type: numbers are doubles, and `string.format("%d", x)` truncates a non-integer `x` rather than raising an error.

The `require()` function loads companion Lua files from the plugin's own folder (e.g., precomputed constants in a separate `.lua` file), but cannot load C modules.

---

## Testing a plugin

1. Restart the app and read the engine log, `ace_engine.log` in the newest session
   folder under `<data root>/logs/`. Each loaded plugin gets a `[Plugins]` line with
   its type, display name and name in parentheses. A file that failed to load shows
   `[Plugins] ERROR loading <file>: <Lua error>`, and a skipped file says why.
2. `GET /api/plugins` returns the metadata and parameter schema exactly as the UI
   sees them. It proxies the engine's `GET /plugins` and caches the answer for 60 s;
   `POST /api/plugins/reload` drops the cache.
3. Render with it. Pick it in the UI, or send the engine a `/synth` request with
   `"infer_method": "my_solver"` (or `scheduler`, `guidance_mode`,
   `postprocess_plugin`) and `"plugin_params": {"my_solver:strength": 0.5}`. Keep
   the seed fixed and compare against the same render with Euler and the linear
   scheduler.
4. A Lua error during a call prints `[Plugins] ERROR in solver '<name>' step(): ...`
   (the same for `schedule()`, `guide()`, `post_step()` and `process()`). The render
   carries on with whatever the failed call left in its buffers, so check the log
   even when audio comes out. A full-loop solver's error, or a missing `sample()`,
   fails the job instead, and a failed `process()` falls back to the built-in
   decoder.

---

## Complete examples

### Minimal solver

```lua
-- my_solver.lua
solver = {
    name        = "my_solver",
    display     = "My Solver",
    description = "Simple Euler variant",
    nfe         = 1,
    order       = 1,
    needs_model = false,
}

function step(xt, vt, t_curr, t_prev, n)
    local dt = t_curr - t_prev
    for i = 0, n - 1 do
        xt[i] = xt[i] - vt[i] * dt
    end
end
```

### Scheduler with a custom curve

```lua
-- my_schedule.lua
scheduler = {
    name        = "my_schedule",
    display     = "Quadratic",
    description = "Quadratic timestep spacing",
    params      = {
        { key = "power", type = "slider", label = "Power",
          default = 2.0, min = 1.0, max = 4.0, step = 0.1 },
    },
}

function schedule(output, num_steps, shift)
    local p = (params and params.power) or 2.0
    for i = 0, num_steps - 1 do
        local frac = i / num_steps
        output[i] = (1.0 - frac) ^ p
    end
    -- Apply shift warp
    if shift ~= 1.0 then
        for i = 0, num_steps - 1 do
            local t = output[i]
            output[i] = shift * t / (1.0 + (shift - 1.0) * t)
        end
    end
end
```

### Guidance with APG and custom logic

```lua
-- my_guidance.lua
guidance = {
    name        = "my_guidance",
    display     = "My Guidance",
    description = "Warm-up guidance with linear ramp",
    params      = {
        { key = "warmup_steps", type = "slider", label = "Warm-Up Steps",
          default = 3, min = 0, max = 10, step = 1 },
    },
}

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    local warmup = (params and params.warmup_steps) or 3
    local progress = math.min((step_idx or 0) / math.max(warmup, 1), 1.0)
    local effective_scale = 1.0 + (guidance_scale - 1.0) * progress

    apg(pred_cond, pred_uncond, effective_scale, result, Oc, T, norm_threshold)
end
```

---

## Tips

- **Sigma schedule has no trailing 0.** The HOT-Step engine passes the schedule to full-loop solvers without a trailing 0 entry (unlike ComfyUI). This means the schedule length equals the step count: use `n_steps = #sigmas`, not `#sigmas - 1`. With `#sigmas - 1` the loop exits one step early, the terminal x0 projection never runs, and the output keeps residual noise at the final sigma level.
- **Test with Euler + Linear first.** The simplest solver/scheduler combination isolates your plugin's behaviour.
- **Use `print()` for debugging.** It writes to the engine's stdout, which the app copies into `ace_engine.log` and the terminal panel.
- **Lua tables for scratch space.** If you need temporary arrays, use Lua tables: `local tmp = {}; for i = 0, n-1 do tmp[i] = 0 end`. <!-- TODO(verify): this tip used to say tables are slower than float arrays. That was written for the Lua 5.4 userdata bridge and has not been measured against the LuaJIT views. -->
- **Be careful with the loop range.** Float arrays are 0-indexed: `for i = 0, n - 1 do ... end`.
- **Guidance plugins: always use `apg()`.** Raw math without APG produces audio artifacts. Apply your custom logic as a delta on top.
- **Stateful plugins** can keep state across steps in file-level `local` variables (e.g., previous velocity buffers, error accumulators). The plugin's Lua state lives as long as the engine process and serves every generation, so reset that state when `step_index == 0`.

## Release data paths

In releases, install user Lua plugins under `<data folder>/plugins`, retaining category subdirectories. The engine scans shipped plugins first and the user overlay last, so matching plugin names in the overlay take precedence. Migration preserves plugins that the new build does not ship. See [Data folder and upgrades](../user/data-folder.md).
