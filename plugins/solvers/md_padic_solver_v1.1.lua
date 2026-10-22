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

-- MD p-Adic Solver v1.1 — Ultrametric Hierarchical Integration
-- MDMAchine | A&E Concepts © 2026
--
-- v1.1 CHANGELOG:
--   - Fixed v1.0 bug: used math.pow() which does not exist in standard Lua.
--     Every step errored with "attempt to call a nil value (field 'pow')"
--     and silently fell through to whatever engine fallback applied.
--     Replaced both call sites with the `^` exponentiation operator.
--   - Validated at stock defaults (2026-08-12): no further tuning needed
--     to get a clean, distinct result — landed close in character to
--     Wasserstein Flow (both are reference-regularized rather than pure
--     trajectory-following, hence the resemblance despite very different
--     math).
--
-- WHAT THIS DOES:
--   Uses p-adic (ultrametric) distance to weight velocity history instead
--   of the standard Euclidean norm. In a p-adic metric space, the distance
--   satisfies the strong triangle inequality:
--     d(x,z) <= max(d(x,y), d(y,z))   (not just d(x,y) + d(y,z))
--
--   This makes every triangle isoceles. "Close" has completely different
--   meaning: two velocities are p-adically close if their difference is
--   divisible by a high power of p. This naturally captures HIERARCHICAL
--   similarity rather than geometric similarity.
--
--   For audio latents, this maps onto the hierarchical structure of music:
--     - Rhythm (coarsest level, lowest p-adic valuation)
--     - Harmony (mid level)
--     - Timbre (finer level)
--     - Texture (finest level, highest p-adic valuation)
--
--   The solver maintains a velocity cache and computes p-adic distances
--   between current and cached velocities. Cache entries that are p-adically
--   close (hierarchically similar) get higher weight in the multistep update,
--   regardless of how Euclidean-far they are.
--
--   PRACTICAL IMPLEMENTATION:
--   True p-adic norms require integer arithmetic. For floating-point latents,
--   we approximate the p-adic valuation via:
--     v_p(x) = -log_p(|x|)  truncated to integer
--   This gives a proxy that captures the "order of magnitude" hierarchy.
--
--   The p-adic weighted multistep update:
--     x_next = x + dt * Σ_k w_k * v_k
--   Where w_k = p^(-v_p(||v_curr - v_k||)) / Z
--   Z is the normalization constant.
--
--   Velocities with hierarchically similar structure to the current velocity
--   get exponentially more weight, regardless of when they occurred in the
--   history. This is FUNDAMENTALLY different from recency-weighted caches
--   like DPM++ or STORM.
--
-- THIS IS COMPLETELY EXPERIMENTAL. The p-adic metric has never been applied
-- to diffusion model integration. The ultrametric property should preserve
-- hierarchical structure (rhythm survives even when timbre changes) but
-- could also produce bizarre artifacts from the non-Euclidean geometry.
--
-- VALIDATED BASELINE (2026-08-12):
--   All defaults (prime_base=3, cache_depth=8, hierarchy_sharpness=1.0,
--   euler_blend=0.5, valuation_epsilon=0.001). No tuning required.
--   Scheduler: MD Psychoacoustic (defaults). Guider: MD Phase Inversion (defaults).
--   Result: clean, distinct, similar character to Wasserstein Flow.
-- ============================================================================

solver = {
    name        = "md_padic_solver",
    display     = "MD p-Adic Solver (Ultrametric v1.1)",
    description = "Ultrametric velocity weighting via p-adic distance. Hierarchically similar velocities dominate regardless of recency. Validated clean at stock defaults.",
    accent      = "magenta",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = true,
    stochastic  = false,
    owns_loop   = false,
    params      = {
        { key = "prime_base", type = "slider", label = "Prime Base (p)",
          default = 3, min = 2, max = 11, step = 1,
          hint = "The prime number for p-adic valuation. 2 = binary hierarchy. 3 = ternary. Primes only (2,3,5,7,11). Non-prime values get rounded to nearest prime." },
        { key = "cache_depth", type = "slider", label = "Cache Depth",
          default = 8, min = 2, max = 20, step = 1,
          hint = "Number of velocity history entries to keep. p-adic weighting makes old entries potentially relevant." },
        { key = "hierarchy_sharpness", type = "slider", label = "Hierarchy Sharpness",
          default = 1.0, min = 0.1, max = 5.0, step = 0.1,
          hint = "Exponent on p-adic weights. Higher = sharper hierarchy (similar velocities dominate more). 1.0 = standard." },
        { key = "euler_blend", type = "slider", label = "Euler Blend",
          default = 0.5, min = 0.0, max = 1.0, step = 0.05,
          hint = "Blend between p-adic multistep (0.0) and standard Euler (1.0). Safety knob. Start at 0.5." },
        { key = "valuation_epsilon", type = "slider", label = "Valuation Floor",
          default = 0.001, min = 0.0001, max = 0.1, step = 0.0005,
          hint = "Minimum magnitude for p-adic valuation. Prevents division by zero and log(0)." },
    },
}

-- ── Constants ───────────────────────────────────────────────────────────────

local EPSILON = 1e-8

-- Nearest prime lookup for the slider
local PRIMES = { [2]=true, [3]=true, [5]=true, [7]=true, [11]=true }
local PRIME_LIST = { 2, 3, 5, 7, 11 }

local function nearest_prime(x)
    local best = 2
    local best_dist = math.abs(x - 2)
    for _, p in ipairs(PRIME_LIST) do
        local d = math.abs(x - p)
        if d < best_dist then
            best = p
            best_dist = d
        end
    end
    return best
end

-- ── State ───────────────────────────────────────────────────────────────────

local _v_cache = {}  -- { {v={...}, sigma=s}, ... }
local _n = 0

-- ── Helpers ─────────────────────────────────────────────────────────────────

local function fa_to_tbl(fa, n)
    local t = {}
    for i = 0, n - 1 do t[i] = fa[i] end
    return t
end

local function tbl_to_fa(t, fa, n)
    for i = 0, n - 1 do fa[i] = t[i] end
end

local function has_nan(t, n)
    for i = 0, n - 1 do
        if t[i] ~= t[i] or math.abs(t[i]) == math.huge then return true end
    end
    return false
end

local function vec_diff_norm(a, b, n)
    local s = 0.0
    for i = 0, n - 1 do
        local d = a[i] - b[i]
        s = s + d * d
    end
    return math.sqrt(s)
end

local function vec_clone(t, n)
    local c = {}
    for i = 0, n - 1 do c[i] = t[i] end
    return c
end

-- ── p-Adic Valuation ────────────────────────────────────────────────────────
-- Approximate p-adic valuation for floating-point:
--   v_p(x) = floor(-log_p(|x|))
-- This gives the "hierarchical depth" of the value.
-- Higher valuation = closer to zero in p-adic metric = FARTHER in p-adic distance.
-- p-adic distance: d_p(a,b) = p^(-v_p(|a-b|))
-- So small Euclidean difference with high valuation = small p-adic distance.

local function padic_valuation(x, p, val_eps)
    local ax = math.abs(x)
    if ax < val_eps then return 10 end  -- cap valuation for near-zero
    return math.floor(-math.log(ax) / math.log(p))
end

-- p-adic distance between two velocity vectors:
-- Use the MINIMUM valuation across all elements (ultrametric: distance is
-- determined by the "coarsest" disagreement)
local function padic_distance(a, b, p, val_eps, n)
    local min_val = 100

    -- Sample a subset for performance (every 4th element)
    local stride = math.max(1, math.floor(n / 64))
    for i = 0, n - 1, stride do
        local diff = a[i] - b[i]
        local val = padic_valuation(diff, p, val_eps)
        if val < min_val then min_val = val end
    end

    -- p-adic distance = p^(-min_valuation)
    return p ^ (-min_val)
end

-- ── Required step() function ────────────────────────────────────────────────

function step(xt, vt, sigma_curr, sigma_next, step_index, total_steps, model_fn)
    local params_ = params or {}
    local p_raw      = params_.prime_base         or 3
    local depth      = math.floor(params_.cache_depth or 8)
    local sharpness  = params_.hierarchy_sharpness or 1.0
    local euler_b    = params_.euler_blend         or 0.5
    local val_eps    = params_.valuation_epsilon   or 0.001

    local p = nearest_prime(math.floor(p_raw))
    local n = #xt
    local dt = sigma_next - sigma_curr

    -- Reset on first step
    if step_index == 0 then
        _v_cache = {}
        _n = n
    end

    local x = fa_to_tbl(xt, n)
    local v_curr = fa_to_tbl(vt, n)

    -- Standard Euler step (always computed as fallback/blend target)
    local x_euler = {}
    for i = 0, n - 1 do
        x_euler[i] = x[i] + dt * v_curr[i]
    end

    -- p-adic multistep (only if we have history)
    local x_padic
    if #_v_cache >= 1 then
        -- Compute p-adic distances from current velocity to all cached
        local weights = {}
        local weight_sum = 0.0

        for k = 1, #_v_cache do
            local d_p = padic_distance(v_curr, _v_cache[k].v, p, val_eps, n)
            -- Invert: close in p-adic = high weight
            -- w_k = (1 / d_p)^sharpness
            local w = (1.0 / (d_p + EPSILON)) ^ sharpness
            weights[k] = w
            weight_sum = weight_sum + w
        end

        -- Normalize weights
        if weight_sum > EPSILON then
            for k = 1, #_v_cache do
                weights[k] = weights[k] / weight_sum
            end
        end

        -- p-adic weighted velocity
        x_padic = {}
        for i = 0, n - 1 do
            local v_weighted = 0.0
            for k = 1, #_v_cache do
                v_weighted = v_weighted + weights[k] * _v_cache[k].v[i]
            end
            -- Blend current velocity with p-adic weighted history
            local v_final = 0.5 * v_curr[i] + 0.5 * v_weighted
            x_padic[i] = x[i] + dt * v_final
        end

        -- NaN guard
        if has_nan(x_padic, n) then
            x_padic = nil
        end
    end

    -- Final result: blend Euler and p-adic
    local x_final = {}
    if x_padic then
        for i = 0, n - 1 do
            x_final[i] = euler_b * x_euler[i] + (1.0 - euler_b) * x_padic[i]
        end
    else
        x_final = x_euler
    end

    -- Update cache
    table.insert(_v_cache, { v = vec_clone(v_curr, n), sigma = sigma_curr })
    if #_v_cache > depth then
        table.remove(_v_cache, 1)
    end

    tbl_to_fa(x_final, xt, n)
end
