-- ============================================================================
-- SPDX-License-Identifier: GPL-3.0-or-later
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
-- ============================================================================

-- MD OmniRelational Solver V4 -- Barbour Best Matching (Per-Block Fix)
-- MDMAchine | A&E Concepts 2026
--
-- V4 CRITICAL FIX:
--   V3's shape/scale decomposition was a mathematical no-op. Computing
--   shape = v/||v|| (unit direction) and scale = ||v||, then recombining
--   as shape * scale, reproduces v EXACTLY. So rw*rel + (1-rw)*v = v
--   for any rw value. V3 was silently running pure Euler at all times.
--
--   V4 uses PER-BLOCK decomposition: the velocity is split into K=4 blocks,
--   each block is normalized by its own L2 norm, then all blocks are
--   rescaled by the mean norm across blocks. This actually equalizes
--   component magnitudes (which was the design intent) instead of
--   collapsing to identity.
--
-- WHAT WAS NEW IN V3:
--
--   V1/V2 applied the same relational weight (rw) at every step regardless
--   of where you are in the denoising schedule. This is suboptimal:
--
--   High sigma (early steps): structure formation phase. The latent is still
--   mostly noise. Shape-preserving relational geometry matters most here --
--   normalizing direction prevents any single component dominating.
--
--   Low sigma (late steps): detail refinement phase. The latent is close to
--   x0. Raw velocity is more accurate. Relational re-injection here over-
--   smooths fine detail.
--
--   SIGMA-ADAPTIVE RELATIONAL WEIGHT:
--     rw_eff = rw * (t_curr / sigma_max) ^ sigma_power
--   At t=sigma_max (first step): rw_eff = rw (full effect)
--   At t=0 (final step): rw_eff = 0 (pure Euler)
--   sigma_power controls the decay curve. 1.0 = linear, 2.0 = quadratic.
--
--   LOOK-BACK SNR SMOOTHER (arXiv:2602.09449):
--     lambda_eff = lb_lambda * (t_curr / sigma_max) ^ lb_snr_power
--     x_smooth   = (1 - lambda_eff) * x_next + lambda_eff * x_prev
--   Heavy at high sigma (blends structure), fades at low sigma (preserves
--   detail). Same mechanism as STORM and PingPong. Off by default.
--
--   GENERATION STATE RESET:
--   Module-level state (sigma_max, x_prev for look-back) resets on
--   step_idx_==0 so same-size consecutive generations don't bleed.
--
--   DRIFT GUARD kept from V1 -- simple, clean, no hoisted buffer.

solver = {
    name        = "md_omni_relational_V4",
    display     = "MD OmniRelational V4 (Per-Block Equalization)",
    description = "Barbour Best Matching with sigma-adaptive relational weight and look-back smoother. rw fades toward pure Euler at low sigma. Proper state reset between generations.",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = true,
    stochastic  = true,
    params      = {
        {
            key     = "relational_weight",
            type    = "slider",
            label   = "Relational Weight",
            default = 0.5,
            min     = 0.0,
            max     = 1.0,
            step    = 0.05,
            hint    = "Base relational weight at high sigma. Fades toward 0 at low sigma. 0=pure Euler always. 0.5=balanced at structure phase.",
        },
        {
            key     = "sigma_power",
            type    = "slider",
            label   = "Sigma Decay Power",
            default = 1.0,
            min     = 0.25,
            max     = 4.0,
            step    = 0.25,
            hint    = "Controls how fast rw fades with sigma. 1.0=linear decay. 2.0=quadratic (faster fade). 0.5=slow fade. Higher = relational effect concentrated earlier.",
        },
        {
            key     = "look_back_lambda",
            type    = "slider",
            label   = "Look-Back Lambda",
            default = 0.0,
            min     = 0.0,
            max     = 0.5,
            step    = 0.01,
            hint    = "Look-back coherence smoother. 0=off. Blends current step with previous, fading out at low sigma. Suppresses trajectory shear. Start at 0.05-0.15.",
        },
        {
            key     = "look_back_snr_power",
            type    = "slider",
            label   = "Look-Back SNR Power",
            default = 1.5,
            min     = 0.5,
            max     = 3.0,
            step    = 0.1,
            hint    = "Controls how fast look-back fades with sigma. Higher = smoothing concentrated on early structure steps only.",
        },
        {
            key     = "drift_guard",
            type    = "toggle",
            label   = "Drift Guard (AOS)",
            default = false,
            hint    = "Project shape_vec onto orthogonal complement of x when cos_sim exceeds threshold. Prevents update reinforcing existing latent structure.",
        },
        {
            key     = "drift_threshold",
            type    = "slider",
            label   = "Drift Threshold",
            default = 0.85,
            min     = 0.1,
            max     = 1.0,
            step    = 0.05,
            hint    = "Cosine similarity ceiling before Gram-Schmidt projection fires. 0.85=standard. Only active when Drift Guard is on.",
        },
        {
            key     = "eta",
            type    = "slider",
            label   = "Eta (SDE Noise)",
            default = 0.0,
            min     = 0.0,
            max     = 1.0,
            step    = 0.05,
            hint    = "Ancestral noise injection. 0=deterministic ODE. Scales with t_prev.",
        },
        {
            key     = "seed",
            type    = "slider",
            label   = "Seed",
            default = 42,
            min     = 0,
            max     = 999999,
            step    = 1,
            hint    = "RNG seed for SDE noise.",
        },
    },
}

local EPSILON = 1e-8

-- Module state -- reset on step_idx_==0
local _sigma_max  = nil
local _x_prev_lb  = nil   -- look-back previous x (before update)

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function make_rng(seed)
    local state = math.floor(seed) % 2147483647
    if state <= 0 then state = state + 2147483646 end
    return function()
        state = (state * 1664525 + 1013904223) % 2147483648
        return state / 2147483648.0
    end
end

local function normal(u1, u2)
    return math.sqrt(-2.0 * math.log(math.max(u1, EPSILON))) * math.cos(2.0 * math.pi * u2)
end

local function l2_norm(arr, n)
    local s = 0.0
    for i = 0, n - 1 do s = s + arr[i] * arr[i] end
    return math.sqrt(s + EPSILON)
end

-- Per-block shape decomposition: splits velocity into K blocks, normalizes
-- each block by its own L2 norm, rescales all blocks by the mean norm.
-- This actually equalizes component magnitudes (single-global decompose
-- was identity: shape * scale = v exactly, a mathematical no-op).
local RELATIONAL_BLOCKS = 4

local function decompose_shape_blocks(vt, n)
    local K = RELATIONAL_BLOCKS
    local bsize = math.floor(n / K)
    local norms = {}
    local norm_sum = 0.0

    -- Compute per-block L2 norms
    for b = 0, K - 1 do
        local s = 0.0
        local lo = b * bsize
        local hi = (b == K - 1) and (n - 1) or (lo + bsize - 1)
        for i = lo, hi do s = s + vt[i] * vt[i] end
        norms[b] = math.sqrt(s + EPSILON)
        norm_sum = norm_sum + norms[b]
    end

    local mean_norm = norm_sum / K

    -- Build equalized velocity: scale each block by mean_norm/block_norm
    local equalized = {}
    for b = 0, K - 1 do
        local inv = (norms[b] > EPSILON) and (mean_norm / norms[b]) or 0.0
        local lo = b * bsize
        local hi = (b == K - 1) and (n - 1) or (lo + bsize - 1)
        for i = lo, hi do equalized[i] = vt[i] * inv end
    end

    return equalized
end

-- Gram-Schmidt drift guard: projects shape onto orthogonal complement of xt
local function apply_drift_guard(shape, xt, n, threshold)
    local norm_x = l2_norm(xt, n)
    if norm_x < EPSILON then return shape end

    local inv_x   = 1.0 / norm_x
    local dot_sx  = 0.0
    for i = 0, n - 1 do dot_sx = dot_sx + shape[i] * (xt[i] * inv_x) end

    if math.abs(dot_sx) <= threshold then return shape end

    local proj = {}
    for i = 0, n - 1 do
        proj[i] = shape[i] - dot_sx * (xt[i] * inv_x)
    end

    local proj_norm = l2_norm(proj, n)
    if proj_norm < EPSILON then return shape end

    local inv_proj = 1.0 / proj_norm
    for i = 0, n - 1 do proj[i] = proj[i] * inv_proj end
    return proj
end

-- Look-back SNR smoother: blend x_curr toward x_prev, lambda fades with sigma
local function look_back_smooth(x_curr, x_prev, t_curr, sigma_max, lb_lambda, snr_power, n)
    if x_prev == nil or lb_lambda < EPSILON then return x_curr, 0.0 end
    local ratio  = math.min(t_curr / math.max(sigma_max, EPSILON), 1.0)
    local lam    = lb_lambda * (ratio ^ snr_power)
    local out    = {}
    for i = 0, n - 1 do
        out[i] = (1.0 - lam) * x_curr[i] + lam * x_prev[i]
    end
    return out, lam
end

-- ── step() function ───────────────────────────────────────────────────────────

function step(xt, vt, t_curr, t_prev, n)
    local rw          = clamp((params and params.relational_weight)   or 0.5,  0.0, 1.0)
    local sig_power   = clamp((params and params.sigma_power)         or 1.0,  0.25, 4.0)
    local lb_lambda   = clamp((params and params.look_back_lambda)    or 0.0,  0.0, 0.5)
    local lb_snr_pow  = clamp((params and params.look_back_snr_power) or 1.5,  0.5, 3.0)
    local drift_on    = (params and params.drift_guard)               or false
    local drift_thr   = clamp((params and params.drift_threshold)     or 0.85, 0.1, 1.0)
    local eta         = clamp((params and params.eta)                 or 0.0,  0.0, 1.0)
    local seed        = math.floor((params and params.seed) or 42)

    local step_idx_ = step_index or 0

    -- Reset state at start of each generation
    if step_idx_ == 0 then
        _sigma_max = t_curr
        _x_prev_lb = nil
    end
    if _sigma_max == nil then _sigma_max = t_curr end

    -- Snapshot xt before update for look-back (copy to plain table)
    local x_curr_snapshot = nil
    if lb_lambda > EPSILON then
        x_curr_snapshot = {}
        for i = 0, n - 1 do x_curr_snapshot[i] = xt[i] end
    end

    -- ── 1. Per-block shape decomposition (equalized velocity) ────────────────
    local equalized = decompose_shape_blocks(vt, n)

    -- ── 2. Optional drift guard on equalized velocity ─────────────────────────
    if drift_on and drift_thr < 1.0 then
        local xt_tbl = {}
        for i = 0, n - 1 do xt_tbl[i] = xt[i] end
        -- Normalize equalized to unit, apply guard, rescale back
        local eq_norm = l2_norm(equalized, n)
        if eq_norm > EPSILON then
            local unit = {}
            for i = 0, n - 1 do unit[i] = equalized[i] / eq_norm end
            unit = apply_drift_guard(unit, xt_tbl, n, drift_thr)
            for i = 0, n - 1 do equalized[i] = unit[i] * eq_norm end
        end
    end

    -- ── 3. Sigma-adaptive relational weight ───────────────────────────────────
    -- rw_eff = rw * (t_curr / sigma_max) ^ sigma_power
    -- At high sigma: rw_eff = rw (full relational). At low sigma: fades to 0.
    local sigma_ratio = math.min(t_curr / math.max(_sigma_max, EPSILON), 1.0)
    local rw_eff      = rw * (sigma_ratio ^ sig_power)

    -- ── 4. Blend + Euler update ───────────────────────────────────────────────
    local dt = t_prev - t_curr   -- negative in flow-matching
    local x_next = {}
    for i = 0, n - 1 do
        local eff_vt_i = rw_eff * equalized[i] + (1.0 - rw_eff) * vt[i]
        x_next[i]      = xt[i] + dt * eff_vt_i
    end

    -- ── 5. Look-back smoother ─────────────────────────────────────────────────
    if lb_lambda > EPSILON then
        local lam
        x_next, lam = look_back_smooth(x_next, _x_prev_lb, t_curr, _sigma_max, lb_lambda, lb_snr_pow, n)
        _x_prev_lb = x_curr_snapshot
    end

    -- ── 6. Write x_next back to xt ───────────────────────────────────────────
    for i = 0, n - 1 do xt[i] = x_next[i] end

    -- ── 7. Optional SDE noise ─────────────────────────────────────────────────
    if eta > 0.0 and t_prev > EPSILON then
        local rng   = make_rng(seed + step_idx_ * 7919)
        local scale = t_prev * eta
        for i = 0, n - 1 do
            local u1 = math.max(rng(), EPSILON)
            local u2 = rng()
            xt[i]    = xt[i] + normal(u1, u2) * scale
        end
    end
end
