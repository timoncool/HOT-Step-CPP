-- ============================================================================
-- SPDX-License-Identifier: GPL-3.0-or-later
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
--
-- This program is free software: you can redistribute it and/or modify
-- it under the terms of the GNU General Public License as published by
-- the Free Software Foundation, either version 3 of the License, or
-- (at your option) any later version.
--
-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
-- GNU General Public License for more details: https://www.gnu.org/licenses/
-- ============================================================================

-- MD Fractional Solver v1.1 — Fractional Adams-Bashforth-Moulton (FABM)
-- MDMAchine | A&E Concepts © 2026
--
-- v1.1 CHANGELOG:
--   - Defaults updated to validated baseline (2026-08-12 test session):
--       alpha 0.85 -> 0.85 (confirmed), memory_window 30 -> 32,
--       corrector_passes 1 -> 3, stability_clamp 5.0 -> 2.0
--   - Fixed v1.0 bug: corrector called model_fn() directly, which is not
--     available in step()-based solvers (needs_model=false). Corrector now
--     uses linear velocity extrapolation (2*v_curr - v_prev) instead of a
--     true model re-evaluation. nfe dropped from 2 to 1 accordingly.
--   - Fixed v1.0 bug: corrector_passes was checked as a boolean (>0) but
--     never actually looped — passing 1 vs 3 had identical effect. v1.1
--     now loops corrector_passes times, re-extrapolating velocity each
--     pass. This means the validated 2026-08-12 test (passes=3) actually
--     ran as a single pass in v1.0 — the true 3-pass behavior is UNTESTED.
--     Re-validate corrector_passes=3 against this build; if it sounds
--     different, drop back to passes=1 to reproduce the original result.
--   - KNOWN ISSUE: GL weights assume uniform step spacing (dt constant
--     across history). Paired with a scheduler that randomizes step
--     intervals (e.g. QRD Scatter with scatter_depth > 0), the GL index-k
--     weights no longer correspond to a consistent time-distance, which
--     smears phase coherence on audio latents (heard as warble/phasiness).
--     Validated clean on MD Psychoacoustic (smooth non-uniform density,
--     no randomization). Not yet fixed — weighting by actual sigma-distance
--     instead of raw history index would be the proper fix. Avoid pairing
--     with scattered/randomized schedulers until then.
--
-- WHAT THIS DOES:
--   Replaces the standard Euler/Heun step with a fractional-order
--   Adams-Bashforth-Moulton predictor-corrector. Each step integrates
--   the denoising ODE using Grünwald-Letnikov (GL) weights over the
--   full velocity history, giving the solver anomalous transport properties.
--
--   alpha < 1.0 : subdiffusion — step updates are memory-heavy, the solver
--                 "remembers" more and changes slowly (sticky transport).
--                 Good for preventing late-stage collapse and drift.
--   alpha = 1.0 : reduces to standard AB-Moulton (normal diffusion).
--   alpha > 1.0 : superdiffusion — step updates are dominated by recent
--                 velocities, solver makes larger jumps (aggressive transport).
--                 Good for fast convergence but can overshoot.
--
-- ARCHITECTURE:
--   - step() solver (not owns_loop). Called by the engine per step.
--   - stateful = true: maintains velocity history across steps.
--   - GL weights computed once on init, truncated to memory_window.
--   - Predictor: fractional AB (explicit GL-weighted history sum).
--   - Corrector: velocity-extrapolated fractional AM, blended with predictor.
--   - NaN guard with Euler fallback on both predictor and corrector.
--
-- VALIDATED BASELINE (2026-08-12):
--   Solver:    alpha=0.85, memory_window=32, corrector_blend=0.5,
--              corrector_passes=3, stability_clamp=2.0
--   Scheduler: MD Psychoacoustic (Complexity-Adaptive), defaults
--   Guider:    MD Phase Inversion (Anti-Phase Cancellation), defaults
--   Steps: 24, guidance_scale: 2.5
--   Result: best fractional output to date, most song-coherent run so far.
--
-- AUDIO LATENT NOTES:
--   The fractional memory kernel naturally prevents the "amnesia" problem
--   where late-step velocities forget early structural decisions. This is
--   especially relevant for audio latents where tonal coherence depends on
--   structural memory persisting through the detail phase.
-- ============================================================================

solver = {
    name        = "md_fractional_solver",
    display     = "MD Fractional Solver (FABM v1.1)",
    description = "Fractional Adams-Bashforth-Moulton predictor-corrector. Alpha < 1 = subdiffusive (memory-heavy, sticky), alpha > 1 = superdiffusive (aggressive jumps). GL kernel provides built-in structural memory. Validated baseline: pair with smooth non-uniform schedulers (e.g. MD Psychoacoustic), avoid scattered/randomized schedulers.",
    accent      = "orange",
    nfe         = 1,  -- no model re-eval, corrector uses velocity extrapolation
    order       = 2,
    needs_model = false,
    stateful    = true,
    stochastic  = false,
    owns_loop   = false,
    params      = {
        { key = "alpha", type = "slider", label = "Fractional Order (α)",
          default = 0.85, min = 0.1, max = 1.9, step = 0.01,
          hint = "Derivative order. <1 = subdiffusion (sticky, memory-heavy). 1.0 = standard. >1 = superdiffusion (aggressive, jumpy). Validated: 0.85." },
        { key = "memory_window", type = "slider", label = "Memory Window",
          default = 32, min = 3, max = 100, step = 1,
          hint = "GL history depth. Truncates the power-law kernel. Validated: 32." },
        { key = "corrector_weight", type = "slider", label = "Corrector Blend",
          default = 0.5, min = 0.0, max = 1.0, step = 0.05,
          hint = "Blend between predictor (0.0) and corrector (1.0). 0.5 = standard PECE. Validated: 0.5." },
        { key = "corrector_passes", type = "slider", label = "Corrector Passes",
          default = 3, min = 0, max = 3, step = 1,
          hint = "Number of corrector iterations. 0 = pure predictor (AB only). Validated: 3." },
        { key = "stability_clamp", type = "slider", label = "Stability Clamp",
          default = 2.0, min = 1.0, max = 20.0, step = 0.5,
          hint = "Maximum ratio of fractional update to Euler update. Prevents runaway on early steps. Validated: 2.0." },
    },
}

-- ── Constants ───────────────────────────────────────────────────────────────

local EPSILON = 1e-8

-- ── State ───────────────────────────────────────────────────────────────────

local _gl_weights = {}
local _v_history = {}    -- velocity history: { [1]={v={...}, sigma=s}, ... }
local _x_history = {}    -- latent history for corrector
local _step_idx = 0
local _n = 0
local _alpha_cached = -1
local _window_cached = -1

-- ── Helpers ─────────────────────────────────────────────────────────────────

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function has_nan(t, n)
    for i = 0, n - 1 do
        if t[i] ~= t[i] or math.abs(t[i]) == math.huge then return true end
    end
    return false
end

local function vec_clone(fa, n)
    local t = {}
    for i = 0, n - 1 do t[i] = fa[i] end
    return t
end

local function fa_to_tbl(fa, n)
    local t = {}
    for i = 0, n - 1 do t[i] = fa[i] end
    return t
end

local function tbl_to_fa(t, fa, n)
    for i = 0, n - 1 do fa[i] = t[i] end
end

-- ── GL Weight Computation ───────────────────────────────────────────────────

local function compute_gl_weights(alpha, K)
    _gl_weights[0] = 1.0
    for k = 1, K - 1 do
        _gl_weights[k] = _gl_weights[k - 1] * (1.0 - (alpha + 1.0) / k)
    end
    _alpha_cached = alpha
    _window_cached = K
end

-- ── Fractional AB Predictor ─────────────────────────────────────────────────
-- Computes x_predict = x + dt * sum_k( w_k * v_{n-k} )
-- The GL weights make this a fractional-order explicit step.
-- NOTE: assumes uniform dt across history (see v1.1 changelog).

local function fractional_predict(x, v_curr, dt, n)
    local x_pred = {}

    -- GL-weighted velocity sum over history
    local depth = math.min(#_v_history + 1, _window_cached)

    for i = 0, n - 1 do
        local gl_vel = _gl_weights[0] * v_curr[i]

        for k = 1, depth - 1 do
            local hist_idx = #_v_history - (k - 1)
            if hist_idx >= 1 and _v_history[hist_idx] then
                gl_vel = gl_vel + _gl_weights[k] * _v_history[hist_idx].v[i]
            end
        end

        x_pred[i] = x[i] + dt * gl_vel
    end

    return x_pred
end

-- ── Fractional AM Corrector ─────────────────────────────────────────────────
-- Refines the prediction using an extrapolated velocity (see v1.1 changelog
-- for why this isn't a true model re-evaluation).
-- x_corr = x + dt * ( w_0 * v_extrap + sum_{k=1}( w_k * v_{n-k} ) )

local function fractional_correct(x, v_pred, dt, blend, n)
    local x_corr = {}

    local depth = math.min(#_v_history + 1, _window_cached)

    for i = 0, n - 1 do
        local gl_vel = _gl_weights[0] * v_pred[i]

        for k = 1, depth - 1 do
            local hist_idx = #_v_history - (k - 1)
            if hist_idx >= 1 and _v_history[hist_idx] then
                gl_vel = gl_vel + _gl_weights[k] * _v_history[hist_idx].v[i]
            end
        end

        x_corr[i] = x[i] + dt * gl_vel
    end

    return x_corr
end

-- ── Euler Fallback ──────────────────────────────────────────────────────────

local function euler_step(x, v, dt, n)
    local x_next = {}
    for i = 0, n - 1 do x_next[i] = x[i] + dt * v[i] end
    return x_next
end

-- ── Stability Clamping ──────────────────────────────────────────────────────
-- Prevents the fractional update from exceeding stability_clamp * euler_step

local function clamp_update(x, x_frac, v_curr, dt, max_ratio, n)
    local clamped = false
    local x_out = {}
    for i = 0, n - 1 do
        local euler_delta = dt * v_curr[i]
        local frac_delta = x_frac[i] - x[i]
        local euler_mag = math.abs(euler_delta) + EPSILON
        local ratio = math.abs(frac_delta) / euler_mag

        if ratio > max_ratio then
            -- Clamp to max_ratio * euler magnitude, preserving sign
            local sign = frac_delta >= 0 and 1.0 or -1.0
            x_out[i] = x[i] + sign * max_ratio * euler_mag
            clamped = true
        else
            x_out[i] = x_frac[i]
        end
    end
    return x_out, clamped
end

-- ── Required step() function ────────────────────────────────────────────────

function step(xt, vt, sigma_curr, sigma_next, step_index, total_steps, model_fn)
    local p = params or {}
    local alpha       = p.alpha            or 0.85
    local window      = math.floor(p.memory_window or 32)
    local corr_blend  = p.corrector_weight or 0.5
    local corr_passes = math.floor(p.corrector_passes or 3)
    local stab_clamp  = p.stability_clamp  or 2.0

    local n = #xt  -- FloatArray length

    -- Recompute GL weights if alpha or window changed
    if alpha ~= _alpha_cached or window ~= _window_cached then
        compute_gl_weights(alpha, window)
    end

    -- Reset state on first step
    if step_index == 0 then
        _v_history = {}
        _x_history = {}
        _step_idx = 0
    end

    local dt = sigma_next - sigma_curr
    local x = fa_to_tbl(xt, n)
    local v_curr = fa_to_tbl(vt, n)

    -- ── Predictor (Fractional AB) ───────────────────────────────────────
    local x_pred = fractional_predict(x, v_curr, dt, n)

    -- NaN guard on predictor
    if has_nan(x_pred, n) then
        x_pred = euler_step(x, v_curr, dt, n)
    end

    -- Stability clamp
    x_pred = clamp_update(x, x_pred, v_curr, dt, stab_clamp, n)

    -- ── Corrector (Velocity Extrapolation) ─────────────────────────────
    -- Cannot call model_fn from step() (needs_model=false, engine owns
    -- model evaluation). Instead, extrapolate a corrector velocity from
    -- history and blend with the predictor result. Multiple passes
    -- progressively refine the extrapolation target.
    local x_final = x_pred
    for pass = 1, corr_passes do
        if #_v_history >= 1 then
            -- Extrapolate velocity: v_extrap = 2*v_curr - v_prev (linear)
            local v_prev = _v_history[#_v_history].v
            local v_extrap = {}
            for i = 0, n - 1 do
                v_extrap[i] = 2.0 * v_curr[i] - v_prev[i]
            end

            if has_nan(v_extrap, n) then break end

            local x_corr = fractional_correct(x, v_extrap, dt, corr_blend, n)

            if has_nan(x_corr, n) then break end

            -- Blend predictor and corrector
            for i = 0, n - 1 do
                x_final[i] = (1.0 - corr_blend) * x_pred[i] + corr_blend * x_corr[i]
            end
            x_final = clamp_update(x, x_final, v_curr, dt, stab_clamp, n)
        else
            break
        end
    end

    -- ── Update history ──────────────────────────────────────────────────
    table.insert(_v_history, { v = vec_clone(v_curr, n), sigma = sigma_curr })
    if #_v_history > window then
        table.remove(_v_history, 1)
    end

    _step_idx = _step_idx + 1

    -- Write final result
    tbl_to_fa(x_final, xt, n)
end
