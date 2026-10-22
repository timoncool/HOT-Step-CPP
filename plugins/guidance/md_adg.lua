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

-- MD ADG Guidance v1.4.2 — Angle-based Dynamic Guidance
-- MDMAchine | A&E Concepts © 2026
--
-- Port of pingpong_trilogy_core.py adg_guidance() to HOT-Step-CPP Lua.
--
-- WHAT THIS DOES:
--   While APG projects the guidance diff, ADG rotates cond TOWARD the
--   (cond - uncond) direction by an angle that scales with sigma.
--
--   Per-step math:
--     gamma     = acos(cosine_similarity(cond, uncond))
--     gamma_tgt = clamp(gamma * (scale - 1), max=pi/3)
--     sigma_sc  = clamp(sigma / sigma_max, 0, 1)
--     gamma_eff = gamma_tgt * sigma_sc
--     orth      = cond - proj(cond, uncond)
--     out       = cos(gamma_eff)*cond + (sin(gamma_eff)/sin(gamma))*orth
--
--   The rotation is aggressive at high sigma (structure phase), settling
--   to near-identity as sigma → 0 (refinement phase). This is more
--   stylistically deforming than APG — it steers harder but avoids the
--   additive overshoot of standard CFG.
--
--   SIGMA_MAX NORMALIZATION (v1.4.2 fix):
--   ADG was hardcoded to 14.6 (ACE-Step 2.6B unnormalized sigma range).
--   HOT-Step-CPP uses flow-matching normalized sigmas 0→1, so sigma_scale
--   was computing as ~0.068 — ADG was essentially inactive.
--   Set sigma_max=1.0 (default) for HOT-Step-CPP. The UI param overrides.
--
-- IMPORTANT: Always routes through apg() for the base computation first,
--   then ADG applies its rotation as a post-correction. This follows the
--   HOT-Step-CPP guidance contract (never raw CFG).
-- ============================================================================

guidance = {
    name        = "md_adg",
    display     = "MD ADG (Angle-based Dynamic Guidance)",
    description = "Rotates cond toward the cond-uncond direction. Angle scales with sigma — aggressive at high sigma, settles to identity. Port of pingpong_trilogy v1.4.2.",
    params      = {
        {
            key     = "sigma_max",
            type    = "slider",
            label   = "Sigma Max",
            default = 1.0,
            min     = 0.1,
            max     = 20.0,
            step    = 0.1,
            hint    = "Maximum sigma for normalization. 1.0=HOT-Step-CPP flow-matching. 14.6=ACE-Step 2.6B unnormalized.",
        },
        {
            key     = "max_angle_deg",
            type    = "slider",
            label   = "Max Rotation (deg)",
            default = 60.0,
            min     = 10.0,
            max     = 90.0,
            step    = 5.0,
            hint    = "Hard cap on rotation angle in degrees. 60=pi/3 (Python default). Lower=safer, less deformation.",
        },
        {
            key     = "apg_blend",
            type    = "slider",
            label   = "APG Blend",
            default = 0.5,
            min     = 0.0,
            max     = 1.0,
            step    = 0.05,
            hint    = "Blend between native APG (0) and ADG rotation (1). 0.5=balanced. Start here.",
        },
        {
            key     = "warmup_steps",
            type    = "slider",
            label   = "Warmup Steps",
            default = 2,
            min     = 0,
            max     = 8,
            step    = 1,
            hint    = "Ramp guidance from 1→scale over N steps. ADG is aggressive — warmup recommended.",
        },
    },
}

-- ── Helpers ──────────────────────────────────────────────────────────────────

local EPSILON = 1e-6

local function dot_product(a, b, n)
    local s = 0.0
    for i = 0, n - 1 do s = s + a[i] * b[i] end
    return s
end

local function l2_norm_sq(a, n)
    return dot_product(a, a, n)
end

local function l2_norm(a, n)
    return math.sqrt(l2_norm_sq(a, n))
end

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

-- ── Required guide() function ─────────────────────────────────────────────────

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    local n          = Oc * T
    local sigma_max_ = (params and params.sigma_max)     or 1.0
    local max_ang    = (params and params.max_angle_deg) or 60.0
    local apg_blend_ = (params and params.apg_blend)     or 0.5
    local wu         = math.floor((params and params.warmup_steps) or 2)

    -- Get native APG result first (always route through apg per contract)
    apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)

    -- Warmup: early steps just use APG
    local cur_step = step_idx or 0
    if wu > 0 and cur_step < wu then
        return  -- result already has APG output, done
    end

    -- Effective guidance scale with warmup
    local eff_scale = guidance_scale
    if wu > 0 and cur_step < wu then
        eff_scale = 1.0 + (guidance_scale - 1.0) * (cur_step / wu)
    end

    -- Current sigma from globals
    local sigma = t_curr or 0.5

    -- Sigma scale: how much rotation to apply (0 at sigma=0, 1 at sigma=sigma_max)
    local sigma_scale = clamp(sigma / math.max(sigma_max_, EPSILON), 0.0, 1.0)

    if sigma_scale < EPSILON then
        return  -- at sigma=0, rotation is zero → APG output unchanged
    end

    -- Compute cosine similarity between cond and uncond (flat vectors)
    local norm_c = l2_norm(pred_cond,   n)
    local norm_u = l2_norm(pred_uncond, n)

    if norm_c < EPSILON or norm_u < EPSILON then
        return  -- degenerate — keep APG result
    end

    local dot_cu    = dot_product(pred_cond, pred_uncond, n)
    local cos_gamma = clamp(dot_cu / (norm_c * norm_u), -1.0 + EPSILON, 1.0 - EPSILON)
    local gamma     = math.acos(cos_gamma)
    local sin_gamma = math.max(math.sin(gamma), EPSILON)

    -- Rotation angle
    local hard_max   = max_ang * math.pi / 180.0   -- degrees → radians
    local gamma_tgt  = clamp((eff_scale - 1.0) * gamma, 0.0, hard_max)
    local gamma_eff  = gamma_tgt * sigma_scale

    if gamma_eff < EPSILON then
        return  -- negligible rotation → keep APG
    end

    -- Orthogonal component: cond minus proj(cond, uncond)
    -- proj_scale = dot(cond, uncond) / ||uncond||^2
    local norm_u_sq   = l2_norm_sq(pred_uncond, n) + EPSILON
    local proj_scale  = dot_cu / norm_u_sq

    -- ADG output: cos(gamma_eff)*cond + (sin(gamma_eff)/sin(gamma))*orth
    local cos_eff  = math.cos(gamma_eff)
    local sin_ratio = math.sin(gamma_eff) / sin_gamma

    -- Blend APG result with ADG rotation
    local adg_blend = apg_blend_
    local apg_blend_inv = 1.0 - adg_blend

    for i = 0, n - 1 do
        local orth_i   = pred_cond[i] - proj_scale * pred_uncond[i]
        local adg_out  = cos_eff * pred_cond[i] + sin_ratio * orth_i
        -- result[i] already contains APG output; blend with ADG
        result[i] = apg_blend_inv * result[i] + adg_blend * adg_out
    end
end
