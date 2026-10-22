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

-- MD Noise Decay Scheduler v1.9 — Multi-Algorithm Decay Curves (Optimized)
-- MDMAchine | A&E Concepts © 2026
--
-- Port of noise_decay_core_V02.py NoiseDecayObject to HOT-Step-CPP Lua.
--
-- v1.9: Engine contract fix — generates num_steps+1 points, writes only
--       num_steps to output (no trailing zero). Users now get all N
--       effective steps instead of N-1. Matches HAP/HT/Causal pattern.
--
-- 6 ALGORITHMS:
--   polynomial   — classic power-law (1-x)^exp. exp=1 linear, exp=2 quadratic.
--   sigmoidal    — S-curve. Slow start, fast middle, slow end. Exp=steepness.
--   piecewise    — user-defined control anchors, linearly interpolated.
--   fourier      — cosine envelope. Exp=frequency (oscillation count).
--   exponential  — pure exp(-k*x) decay. Exp=fall rate.
--   gaussian     — smooth Gaussian flank (high-start, bell-shaped fall).
--
-- OPTIMIZATION NOTES:
--   - Zero-allocation generation (uses hoisted pre-allocated buffers).
--   - Endpoints are strictly enforced to [1.0 → 0.0] AFTER smoothing to 
--     prevent boundary drift.
--   - Invert_curve mirrors the shape (convex↔concave) while preserving
--     the strictly descending 1.0 → 0.0 flow-matching trajectory.
-- ============================================================================

scheduler = {
    name        = "md_noise_decay V2",
    display     = "MD Noise Decay (Multi-Algorithm) V2",
    description = "6-algorithm noise decay scheduler: polynomial, sigmoidal, piecewise, fourier, exponential, gaussian. Memory-optimized.",
    params      = {
        {
            key     = "algorithm",
            type    = "select",
            label   = "Algorithm",
            default = "polynomial",
            options = {
                { value = "polynomial",  label = "Polynomial"  },
                { value = "sigmoidal",   label = "Sigmoidal"   },
                { value = "piecewise",   label = "Piecewise"   },
                { value = "fourier",     label = "Fourier"     },
                { value = "exponential", label = "Exponential" },
                { value = "gaussian",    label = "Gaussian"    },
            },
            hint = "Decay curve shape algorithm",
        },
        {
            key     = "decay_exponent",
            type    = "slider",
            label   = "Exponent / Rate",
            default = 2.0,
            min     = 0.1,
            max     = 8.0,
            step    = 0.1,
            hint    = "Poly: curve power. Sigmoid: steepness. Fourier: frequency. Exp: fall rate. Gaussian: sharpness.",
        },
        {
            key     = "invert_curve",
            type    = "toggle",
            label   = "Invert Curve Shape",
            default = false,
            hint    = "Inverts the bulge of the curve (concave ↔ convex) while maintaining the 1.0 → 0.0 trajectory.",
        },
        {
            key     = "smooth_window",
            type    = "slider",
            label   = "Smooth Window",
            default = 1,
            min     = 1,
            max     = 9,
            step    = 1,
            hint    = "Moving average window size. 1=off, 3-5=mild smoothing, 7+=heavy",
        },
        {
            key     = "piecewise_p0",
            type    = "slider",
            label   = "Piecewise P0",
            default = 1.0,
            min     = 0.0,
            max     = 1.0,
            step    = 0.05,
            hint    = "Piecewise mode: value at start (t=0). Other modes: ignored.",
            visible_when = { key = "algorithm", equals = "piecewise" },
        },
        {
            key     = "piecewise_p1",
            type    = "slider",
            label   = "Piecewise P1",
            default = 0.5,
            min     = 0.0,
            max     = 1.0,
            step    = 0.05,
            hint    = "Piecewise mode: value at midpoint (t=0.5). Other modes: ignored.",
            visible_when = { key = "algorithm", equals = "piecewise" },
        },
        {
            key     = "piecewise_p2",
            type    = "slider",
            label   = "Piecewise P2",
            default = 0.0,
            min     = 0.0,
            max     = 1.0,
            step    = 0.05,
            hint    = "Piecewise mode: value at end (t=1). Other modes: ignored.",
            visible_when = { key = "algorithm", equals = "piecewise" },
        },
        {
            key     = "shift",
            type    = "slider",
            label   = "Shift Warp",
            default = 1.0,
            min     = 0.5,
            max     = 8.0,
            step    = 0.1,
            hint    = "HOT-Step-CPP native shift warp applied after decay curve",
        },
    },
}

-- ── Hoisted Buffers (Zero Allocation Loop) ────────────────────────────────────
local EPSILON = 1e-6
local _last_n = 0
local _sched_buf = {}
local _pad_buf = {}

-- ── Helpers ───────────────────────────────────────────────────────────────────

local function smooth_inplace(arr, n, w)
    if w < 2 or n < w then return end
    local half = math.floor(w / 2)
    
    -- Fill edge-padded buffer into _pad_buf
    for i = 1, half do _pad_buf[i] = arr[0] end
    for i = 0, n - 1 do _pad_buf[half + i + 1] = arr[i] end
    for i = 1, half do _pad_buf[half + n + i] = arr[n - 1] end

    -- Convolve
    for i = 0, n - 1 do
        local sum = 0.0
        for j = 1, w do
            sum = sum + _pad_buf[i + j]
        end
        arr[i] = sum / w
    end
end

-- Strictly anchors the array to [1.0 -> 0.0] mapping
local function normalize_endpoints_inplace(arr, n)
    local v_start = arr[0]
    local v_end   = arr[n - 1]
    local rng = v_start - v_end

    if math.abs(rng) < EPSILON then
        -- Fallback to linear if start and end collapsed
        for i = 0, n - 1 do
            arr[i] = 1.0 - (i / (n - 1))
        end
    else
        for i = 0, n - 1 do
            arr[i] = (arr[i] - v_end) / rng
        end
    end
end

-- ── Required schedule() function ─────────────────────────────────────────────

function schedule(output, num_steps, shift_val)
    local algo   = (params and params.algorithm)      or "polynomial"
    local exp    = (params and params.decay_exponent) or 2.0
    local invert = (params and params.invert_curve)   or false
    local win    = math.floor((params and params.smooth_window) or 1)
    local p0     = (params and params.piecewise_p0)   or 1.0
    local p1     = (params and params.piecewise_p1)   or 0.5
    local p2     = (params and params.piecewise_p2)   or 0.0
    local sh     = (params and params.shift)          or shift_val

    -- Pre-allocate buffers if length changes
    if num_steps + 1 > _last_n then
        for i = 0, num_steps + 22 do
            _sched_buf[i] = 0.0
            _pad_buf[i] = 0.0
        end
        _last_n = num_steps + 22
    end

    -- 1. Generate Raw Curve (num_steps+1 points, matching HAP/HT pattern)
    --    The trailing zero lives in _sched_buf[num_steps] but is NOT written
    --    to output — the engine contract requires no trailing zero in the
    --    schedule. The solver's own fallback handles terminal x0 projection.
    for i = 0, num_steps do
        local x = i / num_steps
        local v = 0.0

        if algo == "polynomial" then
            v = (1.0 - x) ^ exp
        elseif algo == "sigmoidal" then
            local scale = (exp / 2.0) * 2.5
            local sx = (x * 2.0 - 1.0) * scale
            v = 1.0 / (1.0 + math.exp(-sx))
        elseif algo == "piecewise" then
            if x <= 0.5 then
                v = p0 + (p1 - p0) * (x * 2.0)
            else
                v = p1 + (p2 - p1) * ((x - 0.5) * 2.0)
            end
        elseif algo == "fourier" then
            v = (math.cos(exp * math.pi * x) + 1.0) / 2.0
        elseif algo == "exponential" then
            v = math.exp(-exp * x)
        elseif algo == "gaussian" then
            local sigma = 1.0 / math.max(EPSILON, exp * 0.5)
            local gx = x * 3.0
            v = math.exp(-(gx * gx) / (2.0 * sigma * sigma))
        else
            v = (1.0 - x) ^ exp
        end
        
        _sched_buf[i] = v
    end

    -- 2. Normalize to strict [1.0 -> 0.0] trajectory (over all num_steps+1 points)
    normalize_endpoints_inplace(_sched_buf, num_steps + 1)

    -- 3. Temporal Smoothing (re-normalize after to prevent endpoint drift)
    if win >= 2 then
        smooth_inplace(_sched_buf, num_steps + 1, win)
        normalize_endpoints_inplace(_sched_buf, num_steps + 1)
    end

    -- 4. Shape Inversion
    -- Flips the curve's bulge (concave ↔ convex) while maintaining 1.0 -> 0.0 flow
    if invert then
        for i = 0, num_steps do
            _pad_buf[i] = 1.0 - _sched_buf[num_steps - i]
        end
        for i = 0, num_steps do
            _sched_buf[i] = _pad_buf[i]
        end
    end

    -- 5. Shift Warp
    if sh ~= 1.0 then
        for i = 0, num_steps do
            local t = _sched_buf[i]
            _sched_buf[i] = sh * t / (1.0 + (sh - 1.0) * t)
        end
    end

    -- 6. Safety Clamping and Write Output
    --    Write only num_steps entries (0 to num_steps-1). The trailing zero
    --    at _sched_buf[num_steps] stays in the local buffer, not in output.
    for i = 0, num_steps - 1 do
        local val = _sched_buf[i]
        if val < 0.0 then val = 0.0 end
        if val > 1.0 then val = 1.0 end
        output[i] = val
    end

    -- Enforce first entry is exactly 1.0 (engine contract)
    output[0] = 1.0
end