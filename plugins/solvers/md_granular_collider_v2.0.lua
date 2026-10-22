-- ============================================================================
-- SPDX-License-Identifier: GPL-3.0-or-later
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
-- ============================================================================

-- MD Granular Collider Solver v2.0 -- Kinetic Intermodulation
-- MDMAchine | A&E Concepts (C) 2026
--
-- v2.0 CHANGES (ENHANCEMENT/OPTIMIZATION PASS):
--   PERFORMANCE:
--   - eliminated the full-size effective_vt copy (was allocating an
--     n-entry Lua table EVERY step -- 335k entries per step at 50-step
--     runs). New approach: base Euler update runs directly from vt,
--     then collided grains get their sideband contribution added to xt
--     as dt*mod afterward. Mathematically identical
--     (xt += dt*(vt+mod) == xt += dt*vt; xt += dt*mod), zero large
--     allocations per step.
--   - grain_energies scratch table hoisted to module level and reused
--     across steps (Lua tables grow but never shrink -- reuse is free).
--   NEW FEATURE:
--   - threshold_mode param: "absolute" (v1.x behavior, threshold is a
--     raw energy delta) or "relative" (threshold is a fraction of the
--     mean grain energy this step). Relative mode keeps collision rate
--     consistent across CFG / LoRA / sigma conditions -- directly
--     addresses the observed behavior where default absolute threshold
--     0.5 went completely dark at CFG 6 with LoRA scale 0.85 but fired
--     normally with LoRA off. Default remains "absolute" for exact
--     v1.x parity.
--   CARRIED FROM v1.5:
--   - NaN/inf guard on sidebands, stateful=true, N+1 engine call-count
--     documentation, summary/fingerprint/debug logging.
--
-- WHAT THIS DOES:
--   Shatters the velocity field into micro-grains and calculates their
--   kinetic energy. If adjacent grains have a massive energy difference
--   (a transient collision), it forces them to intermodulate and spawns
--   harmonic sidebands directly into the velocity field before integration.
--   Turns the ODE solver into a granular synthesis engine.
--
-- ENGINE NOTE: HS-CPP invokes the solver N+1 times per N-step
-- generation (confirmed in logs: steps=25 at 24 Inference Steps).
-- Set total_steps to the UI Inference Steps value; the boundary check
-- lands correctly because of this. The last generation's summary only
-- prints when the next generation starts (no end-of-run hook exists).

solver = {
    name        = "md_granular_collider",
    display     = "MD Granular Collider (Kinetic Intermodulation)",
    description = "Shatters velocity into grains. High energy deltas between grains spawn intermodulation sidebands. True granular physics inside the ODE.",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = true,
    stochastic  = false,
    params      = {
        {
            key     = "grain_size",
            type    = "slider",
            label   = "Grain Size",
            default = 64,
            min     = 16,
            max     = 256,
            step    = 16,
            hint    = "Size of the velocity chunks. Smaller grains = higher frequency splatter. 64 is a good starting point.",
        },
        {
            key     = "collision_threshold",
            type    = "slider",
            label   = "Collision Threshold",
            default = 0.5,
            min     = 0.05,
            max     = 2.0,
            step    = 0.05,
            hint    = "Energy delta required to spawn sidebands. In absolute mode: raw delta (v1.x behavior). In relative mode: fraction of this step's mean grain energy -- stays consistent across CFG/LoRA/sigma conditions.",
        },
        {
            key     = "threshold_mode",
            type    = "combo",
            label   = "Threshold Mode",
            default = "absolute",
            options = { "absolute", "relative" },
            hint    = "absolute = v1.x parity (raw energy delta, sensitive to CFG/LoRA level). relative = threshold scales with mean grain energy, collision rate stays consistent across conditions.",
        },
        {
            key     = "splatter_gain",
            type    = "slider",
            label   = "Splatter Gain",
            default = 0.2,
            min     = 0.0,
            max     = 1.0,
            step    = 0.05,
            hint    = "Volume of the intermodulation sidebands. 0=off (strict reverse parity with Euler).",
        },
        {
            key     = "total_steps",
            type    = "slider",
            label   = "Total Steps (for summary)",
            default = 24,
            min     = 4,
            max     = 128,
            step    = 1,
            hint    = "MUST match your Inference Steps setting. Used only to detect generation boundaries for the per-generation summary log -- does not affect solver math.",
        },
        {
            key     = "debug_log",
            type    = "toggle",
            label   = "Debug Log",
            default = true,
            hint    = "Print per-step grain collision diagnostics to console.",
        },
        {
            key     = "summary_log",
            type    = "toggle",
            label   = "Summary Log",
            default = true,
            hint    = "Print one rollup line per generation (total collisions, avg/peak magnitude) at the start of the next generation.",
        },
    },
}

local EPSILON = 1e-8
local MOD_CLAMP_MULT = 2.0  -- max sideband magnitude as multiple of mean grain energy

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

-- self-tracked generation state (see ENGINE NOTE in header)
local _internal_step = -1
local _last_n = 0
local _gen_collisions = 0
local _gen_possible_pairs = 0
local _gen_mag_sum = 0.0
local _gen_mag_max = 0.0
local _gen_clamp_hits = 0
local _gen_active_steps = 0
local _gen_total_steps_seen = 0

-- reusable scratch (Lua tables grow but never shrink; reuse avoids
-- per-step allocation entirely)
local _grain_energies = {}

local function print_summary(reason)
    if _gen_total_steps_seen <= 0 then return end
    local mag_mean = (_gen_active_steps > 0) and (_gen_mag_sum / _gen_active_steps) or 0.0
    local pair_pct = (_gen_possible_pairs > 0)
        and (100.0 * _gen_collisions / _gen_possible_pairs) or 0.0
    print(string.format(
        "[MD Granular Collider] SUMMARY (reset=%s) steps=%d active_steps=%d/%d collisions_total=%d (%.3f%% of pairs) mag_mean=%.6f mag_peak=%.6f clamp_hits_total=%d",
        reason, _gen_total_steps_seen, _gen_active_steps, _gen_total_steps_seen,
        _gen_collisions, pair_pct, mag_mean, _gen_mag_max, _gen_clamp_hits))
end

local function reset_gen_stats()
    _gen_collisions = 0
    _gen_possible_pairs = 0
    _gen_mag_sum = 0.0
    _gen_mag_max = 0.0
    _gen_clamp_hits = 0
    _gen_active_steps = 0
    _gen_total_steps_seen = 0
end

function step(xt, vt, t_curr, t_prev, n)
    local p = params or {}
    local g_size    = math.floor((p.grain_size) or 64)
    local threshold = clamp((p.collision_threshold) or 0.5, 0.05, 2.0)
    local thr_mode  = (p.threshold_mode) or "absolute"
    local splat_gain = clamp((p.splatter_gain) or 0.2, 0.0, 1.0)
    local total_steps = math.floor((p.total_steps) or 24)
    local debug_log  = (p.debug_log ~= false)
    local summary_log = (p.summary_log ~= false)

    local dt = t_prev - t_curr

    -- generation boundary detection
    local reset_reason = nil
    if n ~= _last_n then
        reset_reason = "n_changed"
    elseif _internal_step >= total_steps then
        reset_reason = "step_count_wraparound"
    end

    if reset_reason ~= nil then
        if summary_log then
            print_summary(reset_reason)
        end
        reset_gen_stats()
        _last_n = n
        _internal_step = -1
    end

    _internal_step = _internal_step + 1
    local cur_step = _internal_step
    _gen_total_steps_seen = _gen_total_steps_seen + 1

    if debug_log and cur_step == 0 then
        local xt_sum, vt_sum = 0.0, 0.0
        for i = 0, n - 1 do
            xt_sum = xt_sum + xt[i]
            vt_sum = vt_sum + vt[i]
        end
        print(string.format(
            "[MD Granular Collider] FINGERPRINT step=0 n=%d xt_sum=%.6f vt_sum=%.6f xt[0]=%.6f xt[n-1]=%.6f vt[0]=%.6f vt[n-1]=%.6f",
            n, xt_sum, vt_sum, xt[0], xt[n-1], vt[0], vt[n-1]))
    end

    -- base Euler update always runs first, directly from vt -- no copy.
    -- collided grains get their sideband contribution added to xt
    -- afterward as dt*mod, which is mathematically identical to the
    -- v1.x effective_vt approach with zero large allocations.
    for i = 0, n - 1 do
        xt[i] = xt[i] + dt * vt[i]
    end

    -- bypass: base Euler already applied, nothing more to do
    if splat_gain <= 0.0 or n <= g_size then
        if debug_log then
            print(string.format(
                "[MD Granular Collider] step=%d BYPASS (splat_gain=%.3f, n=%d, g_size=%d)",
                cur_step, splat_gain, n, g_size))
        end
        return
    end

    local num_grains = math.floor(n / g_size)

    -- grain energies into reusable scratch + running mean for relative mode
    local energy_total = 0.0
    for g = 0, num_grains - 1 do
        local start_idx = g * g_size
        local sum_sq = 0.0
        for i = 0, g_size - 1 do
            local val = vt[start_idx + i]
            sum_sq = sum_sq + val * val
        end
        local e = math.sqrt(sum_sq / g_size + EPSILON)
        _grain_energies[g] = e
        energy_total = energy_total + e
    end
    local energy_mean = energy_total / num_grains

    -- effective threshold: absolute (raw delta) or relative
    -- (fraction of this step's mean grain energy)
    local eff_threshold = threshold
    if thr_mode == "relative" then
        eff_threshold = threshold * energy_mean
    end

    local collision_count = 0
    local mag_sum = 0.0
    local mag_max = 0.0
    local clamp_hit_count = 0

    for g = 0, num_grains - 2 do
        local e1 = _grain_energies[g]
        local e2 = _grain_energies[g + 1]
        local delta = math.abs(e1 - e2)

        if delta > eff_threshold then
            collision_count = collision_count + 1
            local start1 = g * g_size
            local start2 = (g + 1) * g_size
            local mean_energy = (e1 + e2) * 0.5 + EPSILON
            local sideband_mag = delta * splat_gain
            local clamp_ceiling = mean_energy * MOD_CLAMP_MULT

            for i = 0, g_size - 1 do
                local v1 = vt[start1 + i]
                local v2 = vt[start2 + i]
                local mod = (v1 * v2 / mean_energy) * sideband_mag

                -- NaN/inf guard: drop non-finite sidebands
                if mod ~= mod or mod > 1e30 or mod < -1e30 then
                    mod = 0.0
                end

                local mod_abs = math.abs(mod)
                if mod_abs > clamp_ceiling then
                    clamp_hit_count = clamp_hit_count + 1
                end
                mod = clamp(mod, -clamp_ceiling, clamp_ceiling)

                local m_abs = math.abs(mod)
                mag_sum = mag_sum + m_abs
                if m_abs > mag_max then mag_max = m_abs end

                -- sideband contribution applied directly to xt
                xt[start1 + i] = xt[start1 + i] + dt * mod
                xt[start2 + i] = xt[start2 + i] - dt * mod
            end
        end
    end

    -- accumulate into generation totals
    _gen_collisions = _gen_collisions + collision_count
    _gen_possible_pairs = _gen_possible_pairs + (num_grains - 1)
    _gen_clamp_hits = _gen_clamp_hits + clamp_hit_count
    if collision_count > 0 then
        _gen_active_steps = _gen_active_steps + 1
        local sample_count = collision_count * g_size
        _gen_mag_sum = _gen_mag_sum + (mag_sum / sample_count)
        if mag_max > _gen_mag_max then _gen_mag_max = mag_max end
    end

    if telemetry ~= nil then
        telemetry.gc_collision_count  = collision_count
        telemetry.gc_num_grains       = num_grains
        telemetry.gc_mag_max          = mag_max
        telemetry.gc_clamp_hit_count  = clamp_hit_count
        telemetry.gc_energy_mean      = energy_mean
        telemetry.gc_eff_threshold    = eff_threshold
    end

    if debug_log then
        local sample_count = collision_count * g_size
        local mag_mean = (sample_count > 0) and (mag_sum / sample_count) or 0.0
        print(string.format(
            "[MD Granular Collider] step=%d grains=%d collisions=%d/%d eff_thr=%.4f mag_mean=%.6f mag_max=%.6f clamp_hits=%d",
            cur_step, num_grains, collision_count, num_grains - 1,
            eff_threshold, mag_mean, mag_max, clamp_hit_count))
    end
end
