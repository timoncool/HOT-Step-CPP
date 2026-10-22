-- ============================================================================
-- INTERNAL / CONFIDENTIAL -- NOT FOR PUBLIC DISTRIBUTION
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
-- All Rights Reserved
-- ============================================================================

-- MD Decomposer: Image Frequency
-- 2D spatial-frequency decomposition via cheap Haar wavelet proxy.
--
-- Strategy: recursive pairwise average/difference on the flat velocity
-- vector produces a multi-scale decomposition. Level 0 differences =
-- highest frequency, level 0 averages recursed to level 1 differences =
-- mid frequency, etc. Assigns each original index to the band where its
-- contribution lands in the wavelet tree.
--
-- Without spatial dimension info (H,W unknown), this operates as a 1D
-- multi-resolution decomposition. Still captures scale-dependent energy
-- imbalance which is the primary failure mode in image latents.
--
-- Sigma-adaptive K:
--   High sigma: K=2 (structure vs detail)
--   Mid sigma: K=4 (standard wavelet levels)
--   Low sigma: K=4 (detail phase, keep granularity)
--
-- Registration:
--   _G.md_decomposers.image_freq
--   Core selects via decomposer = "image_freq"

local EPSILON = 1e-8

local function adaptive_K(context)
    if not context or not context.sigma_ratio then return 4 end
    local sr = context.sigma_ratio
    if sr > 0.7 then return 2 end
    return 4
end

-- Haar-like band assignment via recursive pairing
-- For n elements, K=4 bands:
--   band 0: indices contributing to level-2 averages (lowest freq / DC-like)
--   band 1: indices contributing to level-2 differences (low-mid freq)
--   band 2: indices contributing to level-1 differences (high-mid freq)
--   band 3: indices contributing to level-0 differences (highest freq)
--
-- Simplified: assign by position modulo at increasing strides
-- stride 1 pairs -> odd indices = band 3 (finest detail)
-- stride 2 pairs of remaining -> next set = band 2
-- stride 4 pairs of remaining -> next set = band 1
-- everything left = band 0

local function assign_haar_bands(n, K)
    local assignments = {}
    local assigned = {}

    -- Initialize all to lowest band
    for i = 0, n - 1 do
        assignments[i] = 0
        assigned[i] = false
    end

    -- Number of levels = K-1 (each level peels off one band from the top)
    local levels = K - 1
    local stride = 1

    for level = 0, levels - 1 do
        local band = K - 1 - level   -- highest band first
        -- Mark every other unassigned position at this stride as this band
        local count = 0
        for i = 0, n - 1 do
            if not assigned[i] then
                if count % 2 == 1 then
                    assignments[i] = band
                    assigned[i] = true
                end
                count = count + 1
            end
        end
    end

    -- Remaining unassigned = band 0 (lowest frequency)
    for i = 0, n - 1 do
        if not assigned[i] then
            assignments[i] = 0
        end
    end

    return assignments
end

local function partition(velocity, n, context)
    local K = adaptive_K(context)

    if context and context.K_override and context.K_override > 0 then
        K = context.K_override
    end

    local assignments = assign_haar_bands(n, K)

    -- Per-band norms
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
    local active = 0
    for b = 0, K - 1 do
        norms[b] = math.sqrt(norms[b] + EPSILON)
        if counts[b] > 0 then
            norm_sum = norm_sum + norms[b]
            active = active + 1
        end
    end

    if active < 1 then active = 1 end
    local mean_norm = norm_sum / active

    local equalized = {}
    local scales = {}
    for b = 0, K - 1 do
        scales[b] = (norms[b] > EPSILON and counts[b] > 0)
                    and (mean_norm / norms[b]) or 1.0
    end

    for i = 0, n - 1 do
        equalized[i] = velocity[i] * scales[assignments[i]]
    end

    return {
        equalized = equalized,
        scales    = scales,
        norms     = norms,
        K         = K,
    }
end

_G.md_decomposers = _G.md_decomposers or {}
_G.md_decomposers.image_freq = { partition = partition }
