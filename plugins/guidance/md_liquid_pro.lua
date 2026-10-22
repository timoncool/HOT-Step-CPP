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

-- MD Liquid PRO Guidance v1.0 — Non-Newtonian Fluid Physics
-- MDMAchine | A&E Concepts © 2026
--
-- Port of liquid_guider_pro_core.py calculate_liquid_guidance_pro() to HOT-Step-CPP Lua.
--
-- WHAT THIS DOES:
--   Models guidance as flow through a non-Newtonian fluid medium. The flow
--   state is determined by tanh(sigma - transition) / viscosity:
--
--   BELT MODE (flow_state > 0, high sigma):
--     Advection / constructive. The guidance acts like a conveyor belt
--     carrying the latent toward the target. CFG = 1 + 5 * belt_strength * |flow|.
--
--   GEAR MODE (flow_state <= 0, low sigma):
--     Shear / destructive friction. The guidance vector is compressed by
--     gear_friction. drag = friction * |flow|. CFG = 1 + 3 * (1 - drag).
--
--   OOBLECK (shear hardening):
--     When impact_velocity exceeds hardening_threshold, viscosity spikes
--     cubically — simulating oobleck (cornstarch/water) going solid on impact.
--     In gear mode with hardening active, drag approaches 0.98 (concrete).
--
-- PARAMS:
--   viscosity             — base fluid viscosity. Higher=smoother transitions.
--   transition            — sigma at belt→gear crossover
--   belt_strength         — advection scale in belt mode
--   gear_friction         — drag scale in gear mode
--   shear_hardening       — enable oobleck physics
--   hardening_threshold   — impact velocity that triggers hardening
-- ============================================================================

guidance = {
    name        = "md_liquid_pro",
    display     = "MD Liquid PRO (Non-Newtonian Fluid)",
    description = "Belt/Gear fluid guidance with oobleck shear-thickening. tanh flow state splits advective vs frictional regimes. Port of liquid_guider_pro_core v1.0.",
    params      = {
        {
            key     = "viscosity",
            type    = "slider",
            label   = "Viscosity",
            default = 2.0,
            min     = 0.1,
            max     = 10.0,
            step    = 0.1,
            hint    = "Base fluid viscosity. Higher=smoother belt/gear transition. 2.0=standard.",
        },
        {
            key     = "transition",
            type    = "slider",
            label   = "Transition Sigma",
            default = 0.5,
            min     = 0.05,
            max     = 0.95,
            step    = 0.05,
            hint    = "Sigma at belt→gear crossover. Above=belt (advection), below=gear (friction).",
        },
        {
            key     = "belt_strength",
            type    = "slider",
            label   = "Belt Strength",
            default = 0.5,
            min     = 0.0,
            max     = 1.0,
            step    = 0.05,
            hint    = "Advection intensity during belt mode. 0.5=moderate constructive push.",
        },
        {
            key     = "gear_friction",
            type    = "slider",
            label   = "Gear Friction",
            default = 0.3,
            min     = 0.0,
            max     = 0.95,
            step    = 0.05,
            hint    = "Drag during gear mode. Higher=more velocity compression. 0.3=moderate.",
        },
        {
            key     = "shear_hardening",
            type    = "toggle",
            label   = "Shear Hardening (Oobleck)",
            default = false,
            hint    = "Enable non-Newtonian shear-thickening. Impact velocity^3 drives viscosity toward concrete.",
        },
        {
            key     = "hardening_threshold",
            type    = "slider",
            label   = "Hardening Threshold",
            default = 0.1,
            min     = 0.01,
            max     = 1.0,
            step    = 0.01,
            hint    = "Impact velocity above which oobleck hardens. Lower=more sensitive.",
            visible_when = { key = "shear_hardening", equals = true },
        },
    },
}

local EPSILON       = 1e-6
local MAX_VISCOSITY = 20.0

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    local n     = Oc * T
    local visc  = math.max(0.1, (params and params.viscosity)          or 2.0)
    local trans = (params and params.transition)                        or 0.5
    local belt  = (params and params.belt_strength)                     or 0.5
    local gear  = (params and params.gear_friction)                     or 0.3
    local sh_on = (params and params.shear_hardening)                   or false
    local sh_th = math.max(0.01, (params and params.hardening_threshold) or 0.1)

    -- Base APG
    apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)

    local sigma = t_curr or 0.5

    -- Drive vector = cond - uncond
    -- Impact velocity = L2 norm of drive / element count
    local impact_vel = 0.0
    local drive_sq = 0.0
    for i = 0, n - 1 do
        local d = pred_cond[i] - pred_uncond[i]
        drive_sq = drive_sq + d * d
    end
    impact_vel = math.sqrt(drive_sq + EPSILON) / n

    -- Shear hardening: boost viscosity on high impact
    local current_visc = visc
    local is_hardened = false
    if sh_on and impact_vel > sh_th then
        local factor = (impact_vel / sh_th) ^ 3
        current_visc = math.min(visc * factor, MAX_VISCOSITY)
        is_hardened = true
    end

    -- Flow state: tanh((sigma - transition) / viscosity)
    local flow = math.tanh((sigma - trans) / (current_visc + EPSILON))

    if flow > 0 then
        -- BELT MODE (advection / constructive)
        local g_scale = belt * math.abs(flow)
        local eff_cfg = 1.0 + (5.0 * g_scale)
        for i = 0, n - 1 do
            local drive = pred_cond[i] - pred_uncond[i]
            result[i] = pred_uncond[i] + drive * eff_cfg
        end
    else
        -- GEAR MODE (shear / destructive)
        local drag = gear * math.abs(flow)
        if is_hardened then
            drag = math.min(drag * 2.0, 0.98)  -- 98% resistance (concrete)
        end
        local eff_cfg = 1.0 + (3.0 * (1.0 - drag))
        for i = 0, n - 1 do
            local drive = pred_cond[i] - pred_uncond[i]
            result[i] = pred_uncond[i] + drive * (1.0 - drag) * eff_cfg
        end
    end

    -- Safety clamp
    for i = 0, n - 1 do
        result[i] = clamp(result[i], -4.0, 4.0)
    end
end
