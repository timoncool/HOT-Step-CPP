# HOT-Step community plugins

Drop Lua plugin files here to extend the engine without rebuilding. The engine loads
`engine/plugins/` (built in) first, then this folder. Restart the app and the plugin appears
in its dropdown. On a duplicate name the first one loaded wins, so a built-in plugin cannot
be replaced from here; give yours a new `name`.

In a release build, put your own plugins in the data folder's `plugins/` instead, under the
same subfolders. The engine scans it last, a plugin there replaces a shipped one of the same
name, and it is kept when the app is upgraded. See
[Data folder and upgrades](../docs/user/data-folder.md).

```
plugins/
  solvers/        ODE and SDE solvers
  schedulers/     noise schedules
  guidance/       guidance modes
  postprocess/    VAE decode replacements (tiled decode plus a DSP chain)
  docs/           user manuals for the plugins shipped here
```

A file whose name contains `_constants`, `_math` or `_data`, or a `<name>_core.lua` next to
a `<name>.lua`, is treated as a companion file and is not loaded as a plugin. Plugins
`require` companion files by name, and `require` only looks in the plugin's own folder.

How to write one: [docs/dev/plugins-authoring.md](../docs/dev/plugins-authoring.md) has the
API (`solver = {...}` metadata table plus a global `step()`, the `apg()` bridge, `post_step()`,
the parameter schema, the sandbox). The list of every plugin the app currently ships, built in
and community, is generated into [docs/user/plugins.md](../docs/user/plugins.md).

## Full-Loop Solvers

For advanced solvers that need to control the entire sampling iteration
(e.g., adaptive dispatch, velocity caching, SDE restarts), set
`owns_loop = true` and define a `sample()` function instead of `step()`.

ACE-Step and the YuE2 NAR loop run full-loop solvers. MiniMax-Music3 and Stable Audio 3
generation leave them out of their pickers, and the StableStep refine runs its native
ping-pong step in their place.

### Full-Loop Solver Example

```lua
solver = {
  name       = "my_sampler",
  display    = "My Sampler",
  nfe        = 0,         -- varies per step
  order      = 1,
  stateful   = true,
  owns_loop  = true,      -- takes over the sampling loop

  params = {
    { key = "my_param", type = "slider", label = "My Param",
      default = 0.5, min = 0, max = 1, step = 0.01 },
  },
}

function sample(xt, vt_buf, schedule, n, model_fn)
  -- xt:        FloatArray [n], mutable. Contains noise initially.
  -- vt_buf:    FloatArray [n], mutable. model_fn writes velocity here.
  -- schedule:  Lua table {t_1, t_2, ..., t_N}  (1-indexed, N = num_steps)
  -- n:         total element count
  -- model_fn:  function(xt_array, t_val) → writes velocity to vt_buf
  --
  -- Globals: on_step, num_steps, batch_n, n_per, seed, params, model_context
  --
  -- Contract:
  --   1. Call model_fn(xt, t) to evaluate the model at any timestep
  --   2. Read velocity from vt_buf after model_fn returns
  --   3. After each step: call on_step(step_idx, t_curr, t_next) → bool
  --      Returns true if generation was cancelled (you should return)
  --   4. When done, xt must contain the denoised output (x0)

  local ns = #schedule
  for i = 1, ns do
    local t_curr = schedule[i]

    -- Evaluate model
    model_fn(xt, t_curr)

    if i < ns then
      -- Euler step (replace with your solver logic)
      local t_next = schedule[i + 1]
      local dt = t_curr - t_next
      for j = 0, n - 1 do
        xt[j] = xt[j] - vt_buf[j] * dt
      end
      -- Report step completion (engine applies DCW, repaint, etc.)
      if on_step(i - 1, t_curr, t_next) then return end
    else
      -- Final step: predict x0
      for j = 0, n - 1 do
        xt[j] = xt[j] - vt_buf[j] * t_curr
      end
    end
  end
end
```

> **Note:** `on_step()` applies engine corrections (DCW, repaint, guidance
> post-step) automatically. You don't need to handle these yourself.


## Sandbox

Plugins run in a restricted Lua environment: no `os`, `io`, `debug`, `dofile` or `loadfile`.
`math`, `string`, `table` and `require` (for companion data files in the same folder) are
available, along with the `FloatArray` API for zero-copy access to the latent buffers.

## Sharing

A plugin is a single `.lua` file. Share it as is; the recipient drops it in the matching
subfolder here. If it loads a companion file with `require` (the `md_*` solvers here share
`solvers/md_solver_commons.lua`), share that file too and put it in the same subfolder.
