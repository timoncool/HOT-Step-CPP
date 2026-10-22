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

-- MD Causal Scheduler v2.2 — LINA Time Warping + Multi-Mode Base Curves
-- MDMAchine | A&E Concepts © 2026
--
-- v2.2: Schema/code default sync — mode fallback "karras" -> "polynomial",
--       lina_shift fallback 1.0 -> 1.2 (matching UI schema defaults).
-- MDMAchine | A&E Concepts © 2026
--
-- Port of md_causal_scheduler_core.py to HOT-Step-CPP Lua.
--
-- 14 BASE SCHEDULE MODES:
--   karras     — power-law rho spacing (rho=7 default, Karras et al.)
--   simple     — smoothstep (cubic hermite: t*t*(3-2t))
--   linear     — uniform spacing
--   exponential — exp decay: sigma_max * (sigma_min/sigma_max)^t
--   polynomial  — power curve: linspace of sigma^(1/power)
--   beta        — beta distribution curve (alpha, beta params)
--   ays         — AYS adaptive schedule (sigmoid + concentration blend)
--   bong        — tangent-based 2-phase schedule (pivot point)
--   linear_quadratic — linear phase then quadratic phase
--   ddim_uniform — DDIM-style uniform timestep mapping
--   sgm_uniform  — SGM uniform (linear 999→0 mapped to sigma range)
--   blended     — karras + linear blend by blend_factor
--   variance_preserving — log-space interpolation
--   kl_optimal  — arctan-based KL-optimal spacing
--
-- LINA WARP:
--   Post-processes any base schedule by warping the time axis:
--   t_warped = t^shift (power warp on the CDF index)
--   shift < 1: front-loads steps (more at high sigma)
--   shift > 1: back-loads steps (more at low sigma)
--   shift = 1: no warp (identity)
-- ============================================================================

scheduler = {
    name        = "md_causal",
    display     = "MD Causal (LINA + 14 Modes)",
    description = "14 base schedule modes with LINA time-axis warp. Karras, smoothstep, beta, AYS, bong, DDIM, SGM, blended, variance-preserving, KL-optimal and more. Port of md_causal_scheduler_core v2.2.",
    params      = {
        {
            key     = "mode",
            type    = "select",
            label   = "Schedule Mode",
            default = "polynomial",
            options = {
                { value = "karras",              label = "Karras (rho)"          },
                { value = "simple",              label = "Simple (Smoothstep)"   },
                { value = "linear",              label = "Linear"                },
                { value = "exponential",         label = "Exponential"           },
                { value = "polynomial",          label = "Polynomial"            },
                { value = "beta",                label = "Beta"                  },
                { value = "ays",                 label = "AYS"                   },
                { value = "bong",                label = "Bong (Tangent)"        },
                { value = "linear_quadratic",    label = "Linear-Quadratic"      },
                { value = "ddim_uniform",        label = "DDIM Uniform"          },
                { value = "sgm_uniform",         label = "SGM Uniform"           },
                { value = "blended",             label = "Blended (Karras+Lin)"  },
                { value = "variance_preserving", label = "Variance Preserving"   },
                { value = "kl_optimal",          label = "KL Optimal"            },
            },
            hint = "Base schedule curve before LINA warp is applied.",
        },
        {
            key     = "lina_shift",
            type    = "slider",
            label   = "LINA Shift",
            default = 1.2,
            min     = 0.1,
            max     = 3.0,
            step    = 0.05,
            hint    = "Time-axis warp. 1.0=none. <1=front-load (more high-sigma steps). >1=back-load (more low-sigma steps).",
        },
        {
            key     = "rho",
            type    = "slider",
            label   = "Rho (Karras)",
            default = 7.0,
            min     = 1.0,
            max     = 15.0,
            step    = 0.5,
            hint    = "Karras rho parameter. 7=default. Higher=more steps at low sigma.",
            visible_when = { key = "mode", equals = "karras" },
        },
        {
            key     = "power",
            type    = "slider",
            label   = "Power (Polynomial)",
            default = 2.0,
            min     = 0.5,
            max     = 5.0,
            step    = 0.1,
            hint    = "Polynomial exponent. 2=quadratic, 1=linear.",
            visible_when = { key = "mode", equals = "polynomial" },
        },
        {
            key     = "beta_alpha",
            type    = "slider",
            label   = "Beta Alpha",
            default = 0.6,
            min     = 0.1,
            max     = 3.0,
            step    = 0.1,
            hint    = "Beta distribution alpha parameter.",
            visible_when = { key = "mode", equals = "beta" },
        },
        {
            key     = "beta_beta",
            type    = "slider",
            label   = "Beta Beta",
            default = 0.6,
            min     = 0.1,
            max     = 3.0,
            step    = 0.1,
            hint    = "Beta distribution beta parameter.",
            visible_when = { key = "mode", equals = "beta" },
        },
        {
            key     = "blend_factor",
            type    = "slider",
            label   = "Blend Factor",
            default = 0.5,
            min     = 0.0,
            max     = 1.0,
            step    = 0.05,
            hint    = "Blend between Karras (0) and Linear (1).",
            visible_when = { key = "mode", equals = "blended" },
        },
        {
            key     = "bong_pivot",
            type    = "slider",
            label   = "Bong Pivot",
            default = 0.5,
            min     = 0.1,
            max     = 0.9,
            step    = 0.05,
            hint    = "Bong: fraction of steps in compression phase.",
            visible_when = { key = "mode", equals = "bong" },
        },
        {
            key     = "bong_slope_comp",
            type    = "slider",
            label   = "Bong Slope Comp",
            default = 1.2,
            min     = 0.1,
            max     = 3.0,
            step    = 0.1,
            hint    = "Bong: tangent slope in compression phase.",
            visible_when = { key = "mode", equals = "bong" },
        },
        {
            key     = "bong_slope_detail",
            type    = "slider",
            label   = "Bong Slope Detail",
            default = 0.8,
            min     = 0.1,
            max     = 3.0,
            step    = 0.1,
            hint    = "Bong: tangent slope in detail phase.",
            visible_when = { key = "mode", equals = "bong" },
        },
    },
}

local EPSILON = 1e-6
local MONOTONIC_DECAY = 0.99

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

-- ── Base schedule generators ─────────────────────────────────────────────────

local function karras(n, s_min, s_max, rho)
    -- sigma[i] = (s_max^(1/rho) + i/(n-1) * (s_min^(1/rho) - s_max^(1/rho)))^rho
    local inv_rho = 1.0 / rho
    local max_inv = s_max ^ inv_rho
    local min_inv = s_min ^ inv_rho
    local s = {}
    for i = 0, n do
        local t = i / n
        s[i] = (max_inv + t * (min_inv - max_inv)) ^ rho
    end
    s[0] = s_max; s[n] = s_min
    return s
end

local function simple(n, s_min, s_max)
    local s = {}
    for i = 0, n do
        local t = i / n
        local smooth = t * t * (3.0 - 2.0 * t)
        s[i] = s_max - (s_max - s_min) * smooth
    end
    s[0] = s_max; s[n] = s_min
    return s
end

local function linear(n, s_min, s_max)
    local s = {}
    for i = 0, n do
        s[i] = s_max - (s_max - s_min) * (i / n)
    end
    s[0] = s_max; s[n] = s_min
    return s
end

local function exponential(n, s_min, s_max)
    local s = {}
    local safe_max = math.max(s_max, 1e-9)
    for i = 0, n do
        local t = i / n
        s[i] = safe_max * (s_min / safe_max) ^ t
    end
    s[0] = s_max; s[n] = s_min
    return s
end

local function polynomial(n, s_min, s_max, power)
    local s = {}
    local inv_p = 1.0 / math.max(power, 0.1)
    local lo = s_min ^ inv_p
    local hi = s_max ^ inv_p
    for i = 0, n do
        local t = i / n
        s[i] = (hi + t * (lo - hi)) ^ power
    end
    s[0] = s_max; s[n] = s_min
    return s
end

local function beta_sched(n, s_min, s_max, alpha, beta_)
    local s = {}
    for i = 0, n do
        local t = i / n
        local alpha_ = math.max(alpha, 0.1)
        local beta__ = math.max(beta_, 0.1)
        local beta_curve = clamp(1.0 - (1.0 - t ^ alpha_) ^ beta__, 0.0, 1.0)
        s[i] = s_max * (1.0 - beta_curve) + s_min * beta_curve
    end
    s[0] = s_max; s[n] = s_min
    return s
end

local function ays_sched(n, s_min, s_max)
    local s = {}
    for i = 0, n do
        local t = i / n
        -- sigmoid centered at 0.5, steepness 10
        local sig = 1.0 / (1.0 + math.exp(-10.0 * (t - 0.5)))
        -- AYS blend: sigmoid 0.7 + concentration (exp decay) 0.3
        local conc = math.exp(-2.0 * t)
        local ays = sig * 0.7 + conc * 0.3
        -- normalize and invert: high sigma at start
        s[i] = s_min + (s_max - s_min) * (1.0 - ays)
    end
    s[0] = s_max; s[n] = s_min
    return s
end

local function bong_sched(n, s_min, s_max, pivot, slope_comp, slope_det)
    local comp_steps = math.max(1, math.floor(n * pivot))
    local det_steps  = math.max(1, n - comp_steps)

    local sigmas = {}
    local pi_half = math.pi / 2.0 - 0.1

    -- Compression phase
    for i = 0, comp_steps - 1 do
        local t = i / math.max(comp_steps - 1, 1)
        local angle = t * pi_half * slope_comp
        local warped = math.tan(angle) / math.tan(pi_half * slope_comp)
        sigmas[i] = s_max * (1.0 - warped * pivot)
    end

    -- Detail phase
    for i = 0, det_steps - 1 do
        local t = i / math.max(det_steps - 1, 1)
        local angle = t * pi_half * slope_det
        local warped = math.tan(angle) / math.tan(pi_half * slope_det)
        local start = s_max * (1.0 - pivot)
        sigmas[comp_steps + i] = start * (1.0 - warped) + s_min * warped
    end

    sigmas[n] = s_min

    -- Enforce monotonic
    for i = 0, n - 1 do
        if sigmas[i] ~= nil and sigmas[i + 1] ~= nil then
            if sigmas[i] <= sigmas[i + 1] then
                sigmas[i + 1] = math.max(sigmas[i] * MONOTONIC_DECAY, sigmas[i] - EPSILON)
            end
        end
    end

    sigmas[0] = s_max; sigmas[n] = s_min
    return sigmas
end

local function ddim_uniform(n, s_min, s_max)
    local max_ts = 1000
    local s = {}
    for i = 0, n do
        local ts = max_ts - i * (max_ts / n)
        s[i] = s_min + (s_max - s_min) * ((ts / max_ts) ^ 0.5)
    end
    s[0] = s_max; s[n] = s_min
    return s
end

local function sgm_uniform(n, s_min, s_max)
    local s = {}
    for i = 0, n do
        local t = i / n
        s[i] = s_min + (s_max - s_min) * (1.0 - t)
    end
    s[0] = s_max; s[n] = s_min
    return s
end

local function blended(n, s_min, s_max, rho, blend)
    local k = karras(n, s_min, s_max, rho)
    local l = linear(n, s_min, s_max)
    local s = {}
    for i = 0, n do
        s[i] = (1.0 - blend) * k[i] + blend * l[i]
    end
    s[0] = s_max; s[n] = s_min
    return s
end

local function variance_preserving(n, s_min, s_max)
    local s = {}
    local log_min = math.log(math.max(s_min, 1e-9))
    local log_max = math.log(math.max(s_max, 1e-9))
    for i = 0, n do
        local t = i / n
        s[i] = math.exp((1.0 - t) * log_max + t * log_min)
    end
    s[0] = s_max; s[n] = s_min
    return s
end

local function kl_optimal(n, s_min, s_max)
    local s = {}
    local atan_min = math.atan(s_min)
    local atan_max = math.atan(s_max)
    for i = 0, n do
        local t = i / n
        s[i] = math.tan((1.0 - t) * atan_max + t * atan_min)
    end
    s[0] = s_max; s[n] = s_min
    return s
end

-- ── LINA warp ─────────────────────────────────────────────────────────────────

local function apply_lina_warp(sigmas, n, shift)
    if shift == 1.0 then return sigmas end
    local warped = {}
    for i = 0, n do
        local t = i / n
        -- Warp: t_warped = t^shift → index into sigma array
        local t_w = t ^ shift
        local raw_idx = t_w * n
        local idx_lo = clamp(math.floor(raw_idx), 0, n)
        local idx_hi = clamp(idx_lo + 1, 0, n)
        local frac   = raw_idx - idx_lo
        local s_lo   = sigmas[idx_lo] or sigmas[n]
        local s_hi   = sigmas[idx_hi] or sigmas[n]
        warped[i] = s_lo * (1.0 - frac) + s_hi * frac
    end
    warped[0] = sigmas[0]
    warped[n] = sigmas[n]
    return warped
end

-- ── Required schedule() function ─────────────────────────────────────────────

function schedule(output, num_steps, shift)
    local mode        = (params and params.mode)         or "polynomial"
    local lina_shift  = (params and params.lina_shift)   or 1.2
    local rho         = (params and params.rho)          or 7.0
    local power       = (params and params.power)        or 2.0
    local ba          = (params and params.beta_alpha)   or 0.6
    local bb          = (params and params.beta_beta)    or 0.6
    local blend       = (params and params.blend_factor) or 0.5
    local b_pivot     = (params and params.bong_pivot)   or 0.5
    local b_comp      = (params and params.bong_slope_comp)   or 1.2
    local b_det       = (params and params.bong_slope_detail) or 0.8

    local s_max = 1.0
    local s_min = 0.0

    local sigmas
    if     mode == "karras"              then sigmas = karras(num_steps, s_min, s_max, rho)
    elseif mode == "simple"              then sigmas = simple(num_steps, s_min, s_max)
    elseif mode == "linear"              then sigmas = linear(num_steps, s_min, s_max)
    elseif mode == "exponential"         then sigmas = exponential(num_steps, s_min, s_max)
    elseif mode == "polynomial"          then sigmas = polynomial(num_steps, s_min, s_max, power)
    elseif mode == "beta"                then sigmas = beta_sched(num_steps, s_min, s_max, ba, bb)
    elseif mode == "ays"                 then sigmas = ays_sched(num_steps, s_min, s_max)
    elseif mode == "bong"                then sigmas = bong_sched(num_steps, s_min, s_max, b_pivot, b_comp, b_det)
    elseif mode == "ddim_uniform"        then sigmas = ddim_uniform(num_steps, s_min, s_max)
    elseif mode == "sgm_uniform"         then sigmas = sgm_uniform(num_steps, s_min, s_max)
    elseif mode == "blended"             then sigmas = blended(num_steps, s_min, s_max, rho, blend)
    elseif mode == "variance_preserving" then sigmas = variance_preserving(num_steps, s_min, s_max)
    elseif mode == "kl_optimal"          then sigmas = kl_optimal(num_steps, s_min, s_max)
    else                                      sigmas = karras(num_steps, s_min, s_max, rho)
    end

    -- LINA warp
    if lina_shift ~= 1.0 then
        sigmas = apply_lina_warp(sigmas, num_steps, lina_shift)
    end

    -- Native shift warp
    if shift ~= 1.0 then
        for i = 0, num_steps do
            local t = sigmas[i]
            sigmas[i] = shift * t / (1.0 + (shift - 1.0) * t)
        end
    end

    for i = 0, num_steps - 1 do
        output[i] = sigmas[i] or 1.0 - i / num_steps
    end
end
