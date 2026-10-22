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

-- MD Phase Inversion Guidance v1.2 -- Anti-Phase Noise Cancellation
-- MDMAchine | A&E Concepts (C) 2026
--
-- WHAT THIS DOES:
--   Live sound noise cancellation logic applied to latent space guidance.
--   Instead of standard CFG (push away from uncond with uniform gain),
--   this decomposes the uncond prediction into:
--     - GENERIC component: the part of uncond that is "smooth" / low-info
--       (the audio equivalent of room rumble / phase smear)
--     - SPECIFIC component: the part that carries actual structural info
--
--   The generic component gets SUBTRACTED with gain (anti-phase cancellation),
--   while the specific component is left alone or reduced gently. This
--   selectively removes the "AI smooth" quality without the oversaturation
--   artifacts of high CFG.
--
--   DECOMPOSITION METHOD:
--   The "generic smoothness" is estimated as the low-frequency content of
--   the uncond prediction. We approximate this by computing a local moving
--   average of uncond (the smooth/DC component) and treating the remainder
--   as the "specific" high-frequency content.
--
--   anti_phase = uncond_smooth - proj_onto_cond(uncond_smooth)
--
--   This is the orthogonal smooth component -- the generic low-info content
--   that has nothing to do with the prompt. Subtracting it is equivalent to
--   noise cancellation in the latent domain.
--
--   SIGMA-ADAPTIVE CANCELLATION:
--   Early steps (high sigma): less cancellation, structure is still forming
--   Mid steps: maximum cancellation, this is where smoothness gets baked in
--   Late steps: reduced cancellation, preserve fine detail
--
-- RELATIONSHIP TO APG:
--   APG projects diff=(cond-uncond) onto cond and removes the parallel part.
--   Phase Inversion works on uncond directly: it finds the "smooth garbage"
--   in uncond and generates an anti-phase vector to cancel it from the
--   guided output. Different geometry, complementary effect.
--   Can be stacked with APG (use APG as base via apg() call, then apply
--   phase inversion as a post-correction).
--
-- v1.2 CHANGELOG:
--   + Added Adaptive Window toggle. Dynamically scales the smooth window with 
--     sigma to target broad rumble early and tight hiss late. Keeps strict 
--     parity when disabled.
-- v1.1 CHANGELOG:
--   + cancellation_strength default 0.5->0.4 from ear testing. Original
--     default stripped too much smooth content when stacked with the
--     standing wave solver's resonance boosting.
--
-- POST_STEP LAYER:
--   Optional post_step applies residual smoothness detection on the latent
--   itself (not just the guidance). If local variance drops below a floor
--   (indicating over-smoothed regions), it injects micro-texture noise
--   proportional to the deficit. This is the "detail recovery" stage.
-- ============================================================================

guidance = {
    name        = "md_phase_inversion",
    display     = "MD Phase Inversion (Anti-Phase Cancellation)",
    description = "Noise cancellation for latent space: decomposes uncond into smooth (generic) + specific components, generates anti-phase vector to cancel AI smoothness. Sigma-adaptive. Optional detail recovery via post_step.",
    params      = {
        {
            key     = "cancellation_strength",
            type    = "slider",
            label   = "Cancellation Strength",
            default = 0.4,
            min     = 0.0,
            max     = 1.5,
            step    = 0.05,
            hint    = "Anti-phase gain on the generic smooth component. 0=off (standard CFG). 0.4=recommended. >1.0=aggressive, may introduce artifacts.",
        },
        {
            key     = "smooth_window",
            type    = "slider",
            label   = "Smooth Window",
            default = 8,
            min     = 2,
            max     = 32,
            step    = 2,
            hint    = "Moving average window for extracting the smooth/DC component of uncond. Larger=catches broader smoothness. 8=good default for audio latents.",
        },
        {
            key     = "adaptive_window",
            type    = "toggle",
            label   = "Adaptive Window",
            default = false,
            hint    = "Dynamically scales the smooth window with sigma. High sigma = full window for broad rumble, low sigma = tight window for high-frequency hiss. Keeps strict parity when off.",
        },
        {
            key     = "window_floor",
            type    = "slider",
            label   = "Adaptive Window Floor",
            default = 0.25,
            min     = 0.05,
            max     = 0.75,
            step    = 0.05,
            hint    = "Minimum sigma multiplier for adaptive window. 0.25=window never drops below 25% of base size. Lower=tighter filter at low sigma, higher=more conservative scaling. Only active when Adaptive Window is on.",
        },
        {
            type    = "slider",
            label   = "Sigma Focus",
            default = 0.5,
            min     = 0.1,
            max     = 0.9,
            step    = 0.05,
            hint    = "Sigma value where cancellation peaks. Lower=focus on detail steps, higher=focus on structure steps.",
        },
        {
            key     = "sigma_bandwidth",
            type    = "slider",
            label   = "Sigma Bandwidth",
            default = 0.3,
            min     = 0.1,
            max     = 0.6,
            step    = 0.05,
            hint    = "How narrow the sigma-adaptive window is. Smaller=more focused cancellation, larger=broader application.",
        },
        {
            key     = "detail_recovery",
            type    = "toggle",
            label   = "Detail Recovery (post_step)",
            default = false,
            hint    = "Enable post_step micro-texture injection in over-smoothed latent regions.",
        },
        {
            key     = "recovery_strength",
            type    = "slider",
            label   = "Recovery Strength",
            default = 0.1,
            min     = 0.0,
            max     = 0.5,
            step    = 0.01,
            hint    = "How much micro-texture to inject in smoothed regions. Keep low (0.05-0.15). Too high=grainy.",
        },
        {
            key     = "recovery_floor",
            type    = "slider",
            label   = "Variance Floor",
            default = 0.3,
            min     = 0.05,
            max     = 1.0,
            step    = 0.05,
            hint    = "Local variance below this triggers detail recovery. Relative to global variance. Lower=more sensitive.",
        },
        {
            key     = "use_apg_base",
            type    = "toggle",
            label   = "APG Base",
            default = true,
            hint    = "Use APG as the base guidance (recommended). Off=standard CFG base.",
        },
        {
            key     = "momentum",
            type    = "slider",
            label   = "Anti-Phase Momentum",
            default = 0.0,
            min     = -0.8,
            max     = 0.0,
            step    = 0.05,
            hint    = "EMA on the anti-phase vector across steps. 0=off. Negative=damping (prevents cancellation drift). -0.3 to -0.5 recommended if enabled.",
        },
    },
}

-- State
local _anti_phase_ema = nil
local _last_n = 0

local EPSILON = 1e-8

-- Helpers

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function dot_product(a, b, n)
    local s = 0.0
    for i = 0, n - 1 do s = s + a[i] * b[i] end
    return s
end

local function l2_norm_sq(a, n)
    return dot_product(a, a, n)
end

-- Moving average (circular, wrapping at boundaries)
local function moving_average(arr, n, window)
    local out = {}
    local half_w = math.floor(window / 2)
    for i = 0, n - 1 do
        local sum = 0.0
        local count = 0
        for j = -half_w, half_w do
            local idx = i + j
            -- Clamp to boundaries (not circular, edge extension)
            if idx < 0 then idx = 0 end
            if idx >= n then idx = n - 1 end
            sum = sum + arr[idx]
            count = count + 1
        end
        out[i] = sum / count
    end
    return out
end

-- Sigma-adaptive weight: Gaussian centered on focus sigma
local function sigma_weight(sigma, focus, bandwidth)
    local dist = (sigma - focus) / math.max(bandwidth, EPSILON)
    return math.exp(-0.5 * dist * dist)
end

-- guide() -- main guidance function

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    local p = params or {}
    local cancel_str   = clamp((p.cancellation_strength) or 0.4, 0.0, 1.5)
    local base_window  = math.floor(clamp((p.smooth_window) or 8, 2, 32))
    local adaptive_win = p.adaptive_window or false
    local win_floor    = clamp((p.window_floor) or 0.25, 0.05, 0.75)
    local focus        = clamp((p.sigma_focus)      or 0.5, 0.1, 0.9)
    local bandwidth    = clamp((p.sigma_bandwidth)  or 0.3, 0.1, 0.6)
    local use_apg      = (p.use_apg_base == nil) and true or p.use_apg_base
    local mom          = clamp((p.momentum)         or 0.0, -0.8, 0.0)

    local n = Oc * T

    -- Reset on new generation
    if n ~= _last_n then
        _anti_phase_ema = nil
        _last_n = n
    end

    -- Get current sigma from step context
    local sigma = sigma_curr or 0.5
    
    -- Dynamic window scaling based on sigma
    local window = base_window
    if adaptive_win then
        -- Scale down as sigma decays to target higher frequency hiss at the tail
        local sig_mult = clamp(sigma, win_floor, 1.0)
        window = math.floor(clamp(base_window * sig_mult, 2, 32))
    end

    -- STEP 1: Base guidance (APG or standard CFG)
    if use_apg then
        apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    else
        -- Standard CFG: result = uncond + scale * (cond - uncond)
        for i = 0, n - 1 do
            result[i] = pred_uncond[i] + guidance_scale * (pred_cond[i] - pred_uncond[i])
        end
    end

    -- If cancellation is off, we're done
    if cancel_str < EPSILON then return end

    -- STEP 2: Extract smooth component of uncond
    local uncond_tbl = {}
    for i = 0, n - 1 do uncond_tbl[i] = pred_uncond[i] end

    local uncond_smooth = moving_average(uncond_tbl, n, window)

    -- STEP 3: Compute anti-phase vector
    -- anti_phase = uncond_smooth - proj_onto_cond(uncond_smooth)
    -- This is the orthogonal smooth garbage
    local cond_tbl = {}
    for i = 0, n - 1 do cond_tbl[i] = pred_cond[i] end

    local dot_sc = dot_product(uncond_smooth, cond_tbl, n)
    local dot_cc = l2_norm_sq(cond_tbl, n) + EPSILON
    local proj_scale = dot_sc / dot_cc

    local anti_phase = {}
    for i = 0, n - 1 do
        anti_phase[i] = uncond_smooth[i] - proj_scale * cond_tbl[i]
    end

    -- STEP 4: Momentum EMA on anti-phase
    if mom ~= 0.0 then
        if _anti_phase_ema == nil then
            _anti_phase_ema = {}
            for i = 0, n - 1 do _anti_phase_ema[i] = anti_phase[i] end
        else
            for i = 0, n - 1 do
                _anti_phase_ema[i] = anti_phase[i] + mom * _anti_phase_ema[i]
            end
            for i = 0, n - 1 do anti_phase[i] = _anti_phase_ema[i] end
        end
    end

    -- STEP 5: Sigma-adaptive cancellation strength
    local sigma_w = sigma_weight(sigma, focus, bandwidth)
    local effective_cancel = cancel_str * sigma_w

    -- STEP 6: Apply anti-phase cancellation to result
    -- Subtract the anti-phase vector (cancels generic smoothness)
    for i = 0, n - 1 do
        result[i] = result[i] - effective_cancel * anti_phase[i]
    end
end

-- post_step: detail recovery (optional)
-- Detects over-smoothed regions and injects micro-texture

function post_step(xt, t, n, eval_cond, eval_uncond, vt_cond, vt_uncond)
    local p = params or {}
    local recovery_on  = p.detail_recovery or false
    local rec_str      = clamp((p.recovery_strength) or 0.1, 0.0, 0.5)
    local var_floor    = clamp((p.recovery_floor)    or 0.3, 0.05, 1.0)

    if not recovery_on or rec_str < EPSILON then return end

    -- Compute global variance
    local mean = 0.0
    for i = 0, n - 1 do mean = mean + xt[i] end
    mean = mean / n

    local global_var = 0.0
    for i = 0, n - 1 do
        local d = xt[i] - mean
        global_var = global_var + d * d
    end
    global_var = global_var / n

    if global_var < EPSILON then return end

    -- Scan for over-smoothed regions using local variance
    -- Window size: sqrt(n) clamped to reasonable range
    local win = math.floor(clamp(math.sqrt(n), 4, 64))
    local half_w = math.floor(win / 2)

    local step_idx_ = step_index or 0
    -- Simple LCG for deterministic texture noise
    local rng_state = (42 + step_idx_ * 7919) % 2147483647
    if rng_state <= 0 then rng_state = rng_state + 2147483646 end
    local function rng()
        rng_state = (rng_state * 1664525 + 1013904223) % 2147483648
        return rng_state / 2147483648.0
    end

    for i = 0, n - 1 do
        -- Local variance around i
        local local_mean = 0.0
        local count = 0
        for j = math.max(0, i - half_w), math.min(n - 1, i + half_w) do
            local_mean = local_mean + xt[j]
            count = count + 1
        end
        local_mean = local_mean / count

        local local_var = 0.0
        for j = math.max(0, i - half_w), math.min(n - 1, i + half_w) do
            local d = xt[j] - local_mean
            local_var = local_var + d * d
        end
        local_var = local_var / count

        -- Compare local vs global
        local ratio = local_var / (global_var + EPSILON)

        if ratio < var_floor then
            -- Over-smoothed region: inject micro-texture
            local deficit = var_floor - ratio
            local noise_scale = rec_str * deficit * math.sqrt(global_var)
            -- Box-Muller for gaussian texture
            local u1 = math.max(rng(), EPSILON)
            local u2 = rng()
            local noise = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            xt[i] = xt[i] + noise_scale * noise
        end
    end
end
