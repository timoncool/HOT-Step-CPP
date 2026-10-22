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

-- MD APT Guidance v1.0 — Temporal Phase Locker
-- MDMAchine | A&E Concepts © 2026
--
-- Port of apt_symmetry_core.py execute_apt_physics() to HOT-Step-CPP Lua.
--
-- WHAT THIS DOES:
--   Implements a two-phase guidance system based on phase transition physics.
--   At any given sigma, the latent is in one of two regimes:
--
--   TRANSPORT PHASE (high sigma):
--     The latent is still noisy — push harder. Velocity is scaled by
--     transport_velocity (>1.0 = enhanced flow, faster structure formation).
--
--   TRAPPING PHASE (low sigma):
--     Structure has crystallized — damp the guidance to prevent overshoot.
--     Velocity is scaled by (1 - trapping_damping).
--
--   The exceptional_point is the sigma threshold between phases.
--   Smooth transition uses a sigmoid to blend between the two regimes
--   rather than a hard switch. Transition width controls the blend zone.
--
-- PARAMS:
--   exceptional_point  — sigma where transport→trapping transition occurs
--   transport_velocity — velocity multiplier during transport (high sigma)
--   trapping_damping   — velocity reduction during trapping (low sigma)
--   smooth_transition  — sigmoid blend vs hard switch
--   transition_width   — sigmoid steepness (lower=sharper)
-- ============================================================================

guidance = {
    name        = "md_apt",
    display     = "MD APT (Temporal Phase Locker)",
    description = "Two-phase guidance: transport (enhanced flow) above exceptional point, trapping (damped) below. Sigmoid blend between phases. Port of apt_symmetry_core v1.0.",
    params      = {
        {
            key     = "exceptional_point",
            type    = "slider",
            label   = "Exceptional Point",
            default = 0.5,
            min     = 0.05,
            max     = 0.95,
            step    = 0.05,
            hint    = "Sigma threshold between transport and trapping phases. 0.5=midpoint.",
        },
        {
            key     = "transport_velocity",
            type    = "slider",
            label   = "Transport Velocity",
            default = 1.3,
            min     = 0.5,
            max     = 3.0,
            step    = 0.05,
            hint    = "Velocity multiplier during transport phase (high sigma). >1=enhanced flow.",
        },
        {
            key     = "trapping_damping",
            type    = "slider",
            label   = "Trapping Damping",
            default = 0.3,
            min     = 0.0,
            max     = 0.95,
            step    = 0.05,
            hint    = "Velocity reduction during trapping phase (low sigma). 0.3=30% damped. 0=no damping.",
        },
        {
            key     = "smooth_transition",
            type    = "toggle",
            label   = "Smooth Transition",
            default = true,
            hint    = "Sigmoid blend between phases. Off=hard switch at exceptional point.",
        },
        {
            key     = "transition_width",
            type    = "slider",
            label   = "Transition Width",
            default = 0.05,
            min     = 0.01,
            max     = 0.3,
            step    = 0.01,
            hint    = "Sigmoid blend zone width. Lower=sharper transition. 0.05=standard.",
            visible_when = { key = "smooth_transition", equals = true },
        },
    },
}

local EPSILON = 1e-6

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

-- Sigmoid phase blend
local function phase_blend(current_t, ep, tw)
    if tw <= 0 then
        return current_t > ep and 1.0 or 0.0
    end
    local x = (current_t - ep) / tw
    local ok, val = pcall(function() return 1.0 / (1.0 + math.exp(-x * 10.0)) end)
    if not ok then return x > 0 and 1.0 or 0.0 end
    return val
end

-- Log sigma to normalized timestep [0,1]
local function sigma_to_t(sigma)
    -- HOT-Step-CPP: sigma IS already in [0,1] (flow-matching normalized)
    -- Just clamp and return
    return clamp(sigma, 0.0, 1.0)
end

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    local n   = Oc * T
    local ep  = (params and params.exceptional_point)  or 0.5
    local tv  = (params and params.transport_velocity) or 1.3
    local td  = (params and params.trapping_damping)   or 0.3
    local sm  = (params and params.smooth_transition)
    if sm == nil then sm = true end
    local tw  = (params and params.transition_width)   or 0.05

    local sigma = t_curr or 0.5
    local current_t = sigma_to_t(sigma)

    -- Base APG (mandatory)
    apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)

    -- Compute velocity vector (cond - uncond) * scale — same as result from APG
    -- but we need to re-apply with phase-specific scaling
    -- Strategy: compute the transport and trapping results, blend, write to result

    if sm then
        local blend = phase_blend(current_t, ep, tw)
        -- blend=1 → transport, blend=0 → trapping
        -- transport: velocity * transport_velocity
        -- trapping: velocity * (1 - trapping_damping)
        -- effective scale = blend * tv + (1-blend) * (1-td)
        local eff_scale = blend * tv + (1.0 - blend) * (1.0 - td)

        for i = 0, n - 1 do
            local vel = pred_cond[i] - pred_uncond[i]
            result[i] = pred_uncond[i] + vel * guidance_scale * eff_scale
        end
    else
        local eff_scale
        if current_t > ep then
            eff_scale = tv  -- transport
        else
            eff_scale = 1.0 - td  -- trapping
        end

        for i = 0, n - 1 do
            local vel = pred_cond[i] - pred_uncond[i]
            result[i] = pred_uncond[i] + vel * guidance_scale * eff_scale
        end
    end
end
