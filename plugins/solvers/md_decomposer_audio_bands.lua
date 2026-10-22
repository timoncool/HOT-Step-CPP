-- ============================================================================
-- INTERNAL / CONFIDENTIAL -- NOT FOR PUBLIC DISTRIBUTION
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
-- All Rights Reserved
-- ============================================================================

-- MD Decomposer: Audio Bands
-- Frequency-band decomposition for audio latents via cheap DCT proxy.
--
-- Strategy: compute per-element local gradient magnitude as a frequency
-- proxy, then partition indices into K bands by gradient quantile.
-- Low gradient = sub-bass/fundamental content. High gradient = transient/
-- harmonic detail. Each band gets equalized independently by Omni Core.
--
-- Sigma-adaptive band count:
--   High sigma (structure phase): K=2 (coarse: low vs high)
--   Mid sigma: K=4 (sub, low-mid, high-mid, high)
--   Low sigma (detail phase): K=6 (finer separation for detail EQ)
--   User can override K via decomposer_K param in core.
--
-- Registration:
--   Registers as _G.md_decomposers.audio_bands
--   Core selects via decomposer = "audio_bands"

local EPSILON = 1e-8

-- Sigma-adaptive K selection
local function adaptive_K(context)
    if not context or not context.sigma_ratio then return 4 end
    local sr = context.sigma_ratio
    if sr > 0.7 then return 2 end      -- early structure: coarse
    if sr > 0.3 then return 4 end      -- mid schedule: standard
    return 6                            -- detail phase: fine
end

-- Compute local gradient magnitude for each element (frequency proxy)
-- Uses central difference where possible, forward/backward at edges
local function compute_gradients(v, n)
    local grad = {}
    if n < 2 then
        grad[0] = 0.0
        return grad
    end

    grad[0] = math.abs(v[1] - v[0])
    for i = 1, n - 2 do
        grad[i] = math.abs(v[i + 1] - v[i - 1]) * 0.5
    end
    grad[n - 1] = math.abs(v[n - 1] - v[n - 2])

    return grad
end

-- Partition indices into K bands by gradient quantile
-- Returns: band_assignments[i] = band index (0..K-1), sorted thresholds
local function partition_by_gradient(grad, n, K)
    -- Collect and sort gradient values to find quantile thresholds
    local sorted = {}
    for i = 0, n - 1 do sorted[#sorted + 1] = grad[i] end
    table.sort(sorted)

    -- Compute K-1 quantile thresholds
    local thresholds = {}
    for b = 1, K - 1 do
        local idx = math.floor(b * #sorted / K)
        if idx < 1 then idx = 1 end
        if idx > #sorted then idx = #sorted end
        thresholds[b] = sorted[idx]
    end

    -- Assign each index to a band
    local assignments = {}
    for i = 0, n - 1 do
        local g = grad[i]
        local band = 0
        for b = 1, K - 1 do
            if g >= thresholds[b] then band = b end
        end
        assignments[i] = band
    end

    return assignments
end

-- Main partition function (Omni Core plugin contract)
local function partition(velocity, n, context)
    local K = adaptive_K(context)

    -- Override from core param if set
    if context and context.K_override and context.K_override > 0 then
        K = context.K_override
    end

    -- Compute gradients as frequency proxy
    local grad = compute_gradients(velocity, n)

    -- Partition indices by gradient quantile
    local assignments = partition_by_gradient(grad, n, K)

    -- Compute per-band L2 norms
    local norms = {}
    local counts = {}
    for b = 0, K - 1 do
        norms[b] = 0.0
        counts[b] = 0
    end

    for i = 0, n - 1 do
        local b = assignments[i]
        norms[b] = norms[b] + velocity[i] * velocity[i]
        counts[b] = counts[b] + 1
    end

    local norm_sum = 0.0
    local active_bands = 0
    for b = 0, K - 1 do
        norms[b] = math.sqrt(norms[b] + EPSILON)
        if counts[b] > 0 then
            norm_sum = norm_sum + norms[b]
            active_bands = active_bands + 1
        end
    end

    if active_bands < 1 then active_bands = 1 end
    local mean_norm = norm_sum / active_bands

    -- Build equalized velocity and per-band scales
    local equalized = {}
    local scales = {}
    for b = 0, K - 1 do
        scales[b] = (norms[b] > EPSILON and counts[b] > 0)
                    and (mean_norm / norms[b]) or 1.0
    end

    for i = 0, n - 1 do
        local b = assignments[i]
        equalized[i] = velocity[i] * scales[b]
    end

    return {
        equalized = equalized,
        scales    = scales,
        norms     = norms,
        K         = K,
    }
end

-- Register in global decomposer registry
_G.md_decomposers = _G.md_decomposers or {}
_G.md_decomposers.audio_bands = { partition = partition }
