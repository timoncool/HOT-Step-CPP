-- ============================================================================
-- INTERNAL / CONFIDENTIAL -- NOT FOR PUBLIC DISTRIBUTION
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
-- All Rights Reserved
-- ============================================================================

-- MD Decomposer: [MODALITY NAME]
-- [Description of decomposition strategy]
--
-- Registration:
--   _G.md_decomposers.[name]
--   Core selects via decomposer = "[name]"
--
-- ============================================================================
-- PLUGIN CONTRACT
-- ============================================================================
--
-- A decomposer plugin must register a table with a partition() function:
--
--   _G.md_decomposers = _G.md_decomposers or {}
--   _G.md_decomposers.my_decomposer = { partition = partition }
--
-- partition(velocity, n, context) must return a table with:
--
--   equalized  : table[0..n-1]   -- velocity with per-group normalization applied
--   scales     : table[0..K-1]   -- per-group scale factor (mean_norm / group_norm)
--   norms      : table[0..K-1]   -- per-group raw L2 norms before equalization
--   K          : int             -- number of groups used
--
-- context fields available:
--   sigma_curr       : current sigma value
--   sigma_max        : maximum sigma (first step)
--   sigma_ratio      : sigma_curr / sigma_max (0..1, high = early)
--   step_idx         : current step index
--   K_override       : user override for group count (0 = plugin decides)
--   measurements     : table with {variance, stability, entropy, snr_proxy}
--                      (only present if adaptive scheduling is enabled)
--
-- GUIDELINES:
--   - Plugin decides default K based on modality and sigma_ratio
--   - If context.K_override > 0, use that K instead
--   - Handle degenerate cases (empty bands, n < K) gracefully
--   - Return equalized velocity same length as input
--   - scales and norms 0-indexed to match core's memory system
--   - Use EPSILON = 1e-8 for numerical safety
--
-- ============================================================================

local EPSILON = 1e-8

local function adaptive_K(context)
    -- Override from core param
    if context and context.K_override and context.K_override > 0 then
        return context.K_override
    end
    -- Default: adapt by sigma phase
    if not context or not context.sigma_ratio then return 4 end
    local sr = context.sigma_ratio
    if sr > 0.7 then return 2 end
    return 4
end

local function partition(velocity, n, context)
    local K = adaptive_K(context)

    -- ================================================================
    -- TODO: implement modality-specific index partitioning here
    -- assign each index 0..n-1 to a band 0..K-1
    -- ================================================================
    local assignments = {}
    local bsize = math.floor(n / K)
    for i = 0, n - 1 do
        local b = math.floor(i / bsize)
        if b >= K then b = K - 1 end
        assignments[i] = b
    end

    -- Per-band norms
    local norms = {}
    local counts = {}
    for b = 0, K - 1 do norms[b] = 0.0; counts[b] = 0 end

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

-- Change "template" to your modality name
_G.md_decomposers = _G.md_decomposers or {}
_G.md_decomposers.template = { partition = partition }
