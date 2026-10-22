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

-- MD Psychoacoustic Scheduler v1.2 -- Complexity-Adaptive Sigma Decay
-- MDMAchine | A&E Concepts (C) 2026
--
-- WHAT THIS DOES:
--   Standard schedulers (Karras, cosine, HAP) generate a fixed sigma curve
--   before generation starts. This scheduler generates a BASE curve, then
--   the solver can query a post-step hook to WARP remaining sigmas based
--   on the latent's local complexity at each step.
--
--   Since the scheduler API only runs once (before the loop), we generate
--   an adaptive-ready schedule that front-loads structure steps and
--   clusters resolution around estimated phase transition zones.
--
--   TWO MODES:
--
--   MODE 1: COMPLEXITY-WEIGHTED DECAY
--     Generates a base linear schedule, then redistributes step density
--     using a psychoacoustic masking curve. High-complexity sigma regions
--     (mid-range where structure forms) get denser steps. Low-complexity
--     regions (early noise, late refinement) get sparser steps.
--     The complexity profile is estimated from the sigma value itself:
--       - High sigma (>0.7): mostly noise, low perceptual importance
--       - Mid sigma (0.3-0.7): structure formation, HIGH importance
--       - Low sigma (<0.3): detail refinement, moderate importance
--     The transition_center and transition_width params control where
--     the "phase transition" zone sits.
--
--   MODE 2: FIBONACCI CLUSTERING
--     Places step density at golden-ratio intervals around the phase
--     transition point. Fibonacci spacing naturally avoids integer
--     resonance artifacts and clusters resolution at structurally
--     important sigma values without requiring even spacing.
--
--   Both modes can be blended via the fibonacci_blend parameter.
--
-- v1.2 CHANGELOG:
--   + Added Harmonic Density: Injects secondary, proportional density peaks 
--     at octaves of the transition center to support high-frequency detail formation.
-- v1.1 CHANGELOG:
--   + Defaults updated from ear testing with MD Standing Wave solver:
--     transition_width 0.25->0.35 (wider dense zone, less harsh starts),
--     density_contrast 2.0->1.5 (gentler clustering),
--     tail_rush 0.3->0.15 (more steps for tail, cleaner endings).
--
-- AUDIO LATENT RELEVANCE:
--   Audio generation makes its core structural decisions (pitch, key,
--   rhythm) in a narrow sigma band. Wasting steps on pure noise or
--   micro-refinement is compute left on the table. This scheduler
--   focuses step budget where perceptual impact is highest and maps 
--   density to harmonic intervals.
-- ============================================================================

scheduler = {
    name        = "md_psychoacoustic",
    display     = "MD Psychoacoustic (Complexity-Adaptive)",
    description = "Clusters step density around phase transition zones using psychoacoustic masking curves. Optional Fibonacci golden-ratio spacing. Focuses compute where perceptual impact is highest.",
    params      = {
        {
            key     = "transition_center",
            type    = "slider",
            label   = "Phase Transition Center",
            default = 0.45,
            min     = 0.15,
            max     = 0.75,
            step    = 0.05,
            hint    = "Sigma value where the model makes its core structural decisions. Steps cluster here. 0.45=default for ACE-Step. Lower=earlier structure.",
        },
        {
            key     = "transition_width",
            type    = "slider",
            label   = "Transition Width",
            default = 0.35,
            min     = 0.05,
            max     = 0.50,
            step    = 0.05,
            hint    = "How wide the high-density zone is. Wider=smoother transitions, less harsh starts. 0.35 tested best for audio.",
        },
        {
            key     = "density_contrast",
            type    = "slider",
            label   = "Density Contrast",
            default = 1.5,
            min     = 1.0,
            max     = 5.0,
            step    = 0.25,
            hint    = "Step density ratio: peak vs background. 1=uniform. 1.5=gentle clustering (recommended). 2+=aggressive, may cause harsh starts.",
        },
        {
            key     = "harmonic_weight",
            type    = "slider",
            label   = "Harmonic Density",
            default = 0.0,
            min     = 0.0,
            max     = 1.0,
            step    = 0.1,
            hint    = "Adds secondary density peaks at octaves (center/2, center/4) for high-frequency detail. Counteracts tail_rush--balance them carefully. 0=off.",
        },
        {
            key     = "fibonacci_blend",
            type    = "slider",
            label   = "Fibonacci Blend",
            default = 0.0,
            min     = 0.0,
            max     = 1.0,
            step    = 0.1,
            hint    = "Blend with golden-ratio Fibonacci spacing. 0=pure complexity curve. 1=pure Fibonacci. 0.3-0.5=hybrid.",
        },
        {
            key     = "tail_rush",
            type    = "slider",
            label   = "Tail Rush",
            default = 0.15,
            min     = 0.0,
            max     = 0.8,
            step    = 0.05,
            hint    = "How aggressively to skip low-sigma detail steps. 0=even tail. 0.15=gentle (cleaner endings). Higher=faster tail, risks cutting off endings.",
        },
    },
}

local EPSILON = 1e-8
local PHI = (1.0 + math.sqrt(5.0)) / 2.0  -- golden ratio 1.618...
local PHI_INV = 1.0 / PHI                   -- 0.618...

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

-- Psychoacoustic masking weight: Gaussian bump centered on transition + optional harmonics
local function masking_weight(sigma, center, width, contrast, harmonic_w)
    local dist = (sigma - center) / math.max(width, EPSILON)
    local w = math.exp(-0.5 * dist * dist)
    
    if harmonic_w and harmonic_w > 0.0 then
        -- First harmonic (octave down in sigma = finer detail resonance)
        local h1_center = center * 0.5
        local h1_width = width * 0.5
        local dist_h1 = (sigma - h1_center) / math.max(h1_width, EPSILON)
        w = w + (harmonic_w * 0.5 * math.exp(-0.5 * dist_h1 * dist_h1))
        
        -- Second harmonic 
        local h2_center = center * 0.25
        local h2_width = width * 0.25
        local dist_h2 = (sigma - h2_center) / math.max(h2_width, EPSILON)
        w = w + (harmonic_w * 0.25 * math.exp(-0.5 * dist_h2 * dist_h2))
    end

    -- Scale: background=1, peak=contrast
    return 1.0 + (contrast - 1.0) * w
end

-- Generate Fibonacci-spaced points in [0, 1]
-- Uses golden-ratio subdivision: each point at PHI_INV * prev mod 1
local function fibonacci_points(n)
    local pts = {}
    for i = 0, n - 1 do
        pts[i] = (i * PHI_INV) % 1.0
    end
    -- Sort ascending
    local sorted = {}
    for i = 0, n - 1 do sorted[i + 1] = pts[i] end
    table.sort(sorted)
    local out = {}
    for i = 1, #sorted do out[i - 1] = sorted[i] end
    return out
end

-- CDF-based searchsorted interpolation (same pattern as HT scheduler)
local function cdf_to_sigmas(cdf, num_steps)
    -- cdf is a table of cumulative density values at fine resolution
    -- Map num_steps+1 evenly spaced CDF targets to sigma values
    local fine_n = #cdf
    local sigmas = {}

    for step = 0, num_steps do
        local target = step / math.max(num_steps, 1)
        -- Binary search in CDF
        local lo, hi = 1, fine_n
        while lo < hi do
            local mid = math.floor((lo + hi) / 2)
            if cdf[mid] < target then lo = mid + 1 else hi = mid end
        end
        -- Linear interpolation
        if lo > 1 and lo <= fine_n then
            local t_lo = cdf[lo - 1]
            local t_hi = cdf[lo]
            local dt = t_hi - t_lo
            local frac = 0.0
            if dt > EPSILON then frac = (target - t_lo) / dt end
            sigmas[step] = 1.0 - ((lo - 1 + frac - 1) / (fine_n - 1))
        else
            sigmas[step] = 1.0 - (step / math.max(num_steps, 1))
        end
    end

    return sigmas
end

function schedule(output, num_steps, shift)
    local p = params or {}
    local center   = clamp((p.transition_center)  or 0.45, 0.15, 0.75)
    local width    = clamp((p.transition_width)    or 0.35, 0.05, 0.50)
    local contrast = clamp((p.density_contrast)    or 1.5, 1.0, 5.0)
    local harm_w   = clamp((p.harmonic_weight)     or 0.0, 0.0, 1.0)
    local fib_blend = clamp((p.fibonacci_blend)    or 0.0, 0.0, 1.0)
    local tail     = clamp((p.tail_rush)           or 0.15, 0.0, 0.8)

    -- STEP 1: Generate complexity-weighted density at fine resolution
    local fine_n = 1024
    local density = {}
    local density_sum = 0.0

    for i = 1, fine_n do
        local sigma = 1.0 - ((i - 1) / (fine_n - 1))
        local w = masking_weight(sigma, center, width, contrast, harm_w)

        -- Tail rush: reduce density at low sigma
        if sigma < center - width then
            local tail_factor = 1.0 - tail * (1.0 - sigma / math.max(center - width, EPSILON))
            tail_factor = clamp(tail_factor, 0.2, 1.0)
            w = w * tail_factor
        end

        density[i] = w
        density_sum = density_sum + w
    end

    -- Build CDF
    local cdf = {}
    local running = 0.0
    for i = 1, fine_n do
        running = running + density[i] / density_sum
        cdf[i] = running
    end
    cdf[fine_n] = 1.0  -- clamp endpoint

    -- STEP 2: Map CDF to sigma schedule
    local sigmas_complexity = cdf_to_sigmas(cdf, num_steps)

    -- STEP 3: Fibonacci schedule (if blending)
    local sigmas_fib = nil
    if fib_blend > 0.0 then
        local fib_pts = fibonacci_points(num_steps + 1)
        sigmas_fib = {}
        -- Fibonacci points are in [0,1], map to sigma [1,0]
        -- Cluster around transition center
        for i = 0, num_steps do
            local fp = fib_pts[i] or (i / math.max(num_steps, 1))
            -- Warp fibonacci points toward transition center
            -- Use a soft attractor: pull points toward center
            local pulled = fp + 0.3 * math.sin(math.pi * fp) * (center - 0.5)
            pulled = clamp(pulled, 0.0, 1.0)
            sigmas_fib[i] = 1.0 - pulled
        end
        -- Sort descending (sigma should go 1.0 -> 0.0)
        local sorted = {}
        for i = 0, num_steps do sorted[i + 1] = sigmas_fib[i] end
        table.sort(sorted, function(a, b) return a > b end)
        for i = 0, num_steps do sigmas_fib[i] = sorted[i + 1] end
    end

    -- STEP 4: Blend
    local sigmas = {}
    for i = 0, num_steps do
        if fib_blend > 0.0 and sigmas_fib then
            sigmas[i] = (1.0 - fib_blend) * sigmas_complexity[i] + fib_blend * sigmas_fib[i]
        else
            sigmas[i] = sigmas_complexity[i]
        end
    end

    -- Force endpoints
    sigmas[0] = 1.0
    sigmas[num_steps] = 0.0

    -- Ensure monotonically decreasing
    for i = 1, num_steps do
        if sigmas[i] >= sigmas[i - 1] then
            sigmas[i] = sigmas[i - 1] - EPSILON
        end
    end

    -- Shift warp
    if shift ~= 1.0 then
        for i = 0, num_steps do
            local t = sigmas[i]
            sigmas[i] = shift * t / (1.0 + (shift - 1.0) * t)
        end
    end

    -- Write output (num_steps values, not num_steps+1)
    for i = 0, num_steps - 1 do
        output[i] = sigmas[i]
    end
end
