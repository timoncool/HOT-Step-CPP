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

-- MD Firefly Ringdown Guidance v2.0 — QNM Decay Damping
-- MDMAchine | A&E Concepts © 2026
--
-- v2.0: Schema/code default sync (ringdown_start 0.5->0.65,
--       damping_mass 1.0->1.5).
--
-- Port of firefly_ringdown_core.py execute_ringdown_physics() to HOT-Step-CPP Lua.
--
-- WHAT THIS DOES:
--   Models the final denoising steps as a gravitational wave ringdown —
--   the damped oscillation of a perturbed black hole returning to equilibrium.
--   In QNM (quasi-normal mode) physics, the ringdown decays as:
--
--     envelope(t) = exp(-damping_mass * qnm_frequency * (1 - relative_t))
--
--   Applied to guidance: as sigma drops below ringdown_start, the guidance
--   velocity is progressively damped by this envelope. Early steps (high sigma)
--   are unaffected; late steps get exponential guidance reduction.
--
--   MARGINALIZE PHASE mode: instead of scaling the full velocity vector,
--   compute the L2 magnitude, scale the magnitude by the envelope, then
--   re-apply the original direction. This preserves steering direction
--   while reducing energy — "turn down the volume, keep the direction."
--
-- PARAMS:
--   ringdown_start    — sigma below which ringdown engages (fraction 0-1)
--   damping_mass      — how heavy the ringdown damping is
--   qnm_frequency     — oscillation frequency of the QNM mode
--   marginalize_phase — preserve direction, damp magnitude only
--   smooth_transition — sigmoid entry vs hard switch
-- ============================================================================

guidance = {
    name        = "md_firefly_ringdown V2",
    display     = "MD Firefly Ringdown (QNM Decay) V2",
    description = "Quasi-normal mode exponential decay damping on guidance velocity. Marginalize-phase preserves direction. Port of firefly_ringdown_core v1.0.",
    params      = {
        {
            key     = "ringdown_start",
            type    = "slider",
            label   = "Ringdown Start",
            default = 0.65,
            min     = 0.1,
            max     = 0.9,
            step    = 0.05,
            hint    = "Sigma below which ringdown engages. 0.5=mid-run. 0.3=late only.",
        },
        {
            key     = "damping_mass",
            type    = "slider",
            label   = "Damping Mass",
            default = 1.5,
            min     = 0.1,
            max     = 5.0,
            step    = 0.1,
            hint    = "Damping heaviness. Higher=faster decay. 1.0=standard.",
        },
        {
            key     = "qnm_frequency",
            type    = "slider",
            label   = "QNM Frequency",
            default = 2.0,
            min     = 0.5,
            max     = 10.0,
            step    = 0.25,
            hint    = "Oscillation frequency. Combined with mass for decay rate: rate = mass * freq.",
        },
        {
            key     = "marginalize_phase",
            type    = "toggle",
            label   = "Marginalize Phase",
            default = false,
            hint    = "Damp magnitude only, preserve steering direction (Gram-Schmidt magnitude projection).",
        },
        {
            key     = "smooth_transition",
            type    = "toggle",
            label   = "Smooth Entry",
            default = true,
            hint    = "Sigmoid fade into ringdown. Off=hard switch.",
        },
    },
}

local EPSILON = 1e-6

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

-- Smooth sigmoid entry (mirrors calculate_ringdown_strength)
local function ringdown_strength(t, threshold, width)
    if t > threshold + width then return 0.0 end
    if t < threshold - width then return 1.0 end
    local x = (threshold - t) / width
    local ok, val = pcall(function() return 1.0 / (1.0 + math.exp(-x * 5.0)) end)
    if not ok then return x > 0 and 1.0 or 0.0 end
    return val
end

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    local n       = Oc * T
    local rd_start = (params and params.ringdown_start)  or 0.65
    local d_mass   = (params and params.damping_mass)    or 1.5
    local qnm_freq = (params and params.qnm_frequency)  or 2.0
    local marg     = (params and params.marginalize_phase) or false
    local smooth   = (params and params.smooth_transition)
    if smooth == nil then smooth = true end

    -- Base APG
    apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)

    local sigma = t_curr or 0.5

    -- Convert sigma to t ∈ [0,1] via sigma/(sigma+1) (matches Python)
    local t = sigma / (sigma + 1.0)
    local threshold = 1.0 - rd_start

    -- Calculate ringdown strength (1.0=full effect, 0.0=no effect)
    local rs
    if smooth then
        rs = ringdown_strength(t, threshold, 0.05)
    else
        rs = (t <= threshold) and 1.0 or 0.0
    end

    -- No effect above ringdown zone
    if rs <= 0.001 then return end

    -- Compute damping envelope
    local relative_t = t / (threshold + EPSILON)
    local decay_rate = d_mass * qnm_freq
    local base_decay = math.exp(-decay_rate * (1.0 - relative_t))
    local damping_envelope = (1.0 - rs) + (rs * base_decay)

    if marg then
        -- Marginalize phase: scale magnitude, keep direction
        -- velocity = result - uncond (the guided component)
        -- compute magnitude, scale it, normalize direction
        local vel = {}
        local mag_sq = 0.0
        for i = 0, n - 1 do
            vel[i] = result[i] - pred_uncond[i]
            mag_sq = mag_sq + vel[i] * vel[i]
        end
        local mag = math.sqrt(mag_sq + EPSILON)
        local target_mag = mag * damping_envelope

        for i = 0, n - 1 do
            local dir = vel[i] / mag
            result[i] = pred_uncond[i] + dir * target_mag
        end
    else
        -- Direct scaling: result = uncond + velocity * envelope
        for i = 0, n - 1 do
            local vel = result[i] - pred_uncond[i]
            result[i] = pred_uncond[i] + vel * damping_envelope
        end
    end
end
