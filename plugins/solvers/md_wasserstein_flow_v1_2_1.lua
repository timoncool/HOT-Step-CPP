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

-- MD Wasserstein Flow Solver v1.2.1 -- Proximal Optimal Transport Steps
-- MDMAchine | A&E Concepts (C) 2026
--
-- WHAT THIS DOES:
--   Treats each denoising step as a proximal step on an energy landscape.
--   The velocity field provides the base transport trajectory (Euler step),
--   while a proximal operator step minimizes regularized latent energy R(x):
--
--     x_next = x_curr + dt * v - (dt / tau) * grad(R(x))
--
--   This is a discrete split-operator realization of the JKO
--   (Jordan-Kinderlehrer-Otto) scheme in Wasserstein space.
--
--   The regularization energy R(x) combines:
--     - Spectral flatness penalty: pushes latent power distribution toward
--       uniformity, preventing energy concentration in narrow bands
--       (anti-metallic). Applied TRANSVERSE to velocity when ortho is on.
--     - RMS regularization: stabilizes latent norm around the ACE-Step
--       target. Always applied RADIALLY (not projected) because norm drift
--       accumulates along the trajectory, so norm correction must too.
--
--   The proximal parameter tau controls transport stiffness:
--     Small tau: stiff transport penalty, conservative regularization.
--     Large tau: relaxed transport cost, stronger regularization authority.
--
-- AUDIO LATENT NOTES:
--   - Split Projection: Spectral flatness gradient is projected orthogonal
--     to velocity v (transient-safe smoothing). RMS gradient bypasses the
--     projection (norm-safe stabilization). This fixes the v1.1 issue
--     where projected RMS could not correct drift along the trajectory.
--   - Trust Region: Total regularizer displacement bounded to
--     alpha * ||dt * v||, catching gradient explosions.
--   - Grad Clamp: Per-element spectral gradient clamped to +/- 5.0 as a
--     numerical safety belt independent of the trust region.
--
-- CHANGES vs v1.1:
--   - stateful flag corrected to false (no semantic state exists)
--   - ensure_scratch resizes in place instead of reallocating
--   - zero-fill removed from scratch init (buffers get overwritten anyway)
--   - split projection: spectral projected, RMS acts radially
--   - trust region alpha tightened from 1.5 to 1.0 (real ceiling)
--   - pass count reduced from 6 to 4 per prox iteration
--
-- CHANGES vs v1.2:
--   - base_disp_sq hoisted out of the prox iteration loop. dt * v[i] is
--     invariant across inner iterations, so compute the base displacement
--     norm once during the seed pass and reuse it. Saves N mults + N adds
--     per prox iteration.
-- ============================================================================

solver = {
    name        = "md_wasserstein_flow_v121",
    display     = "MD Wasserstein Flow (Proximal OT v1.2.1)",
    description = "Proximal optimal transport solver. Split projection: spectral flatness transverse to velocity, RMS radial. Trust region bounded, zero-alloc scratch buffers.",
    accent      = "teal",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = false,
    stochastic  = false,
    owns_loop   = false,
    params      = {
        { key = "tau", type = "slider", label = "Transport Cost (tau)",
          default = 1.0, min = 0.1, max = 10.0, step = 0.1,
          hint = "Proximal parameter. Low = conservative (small steps). High = permissive (big steps OK)." },
        { key = "spectral_weight", type = "slider", label = "Spectral Flatness Weight",
          default = 0.1, min = 0.0, max = 1.0, step = 0.01,
          hint = "Penalty for non-uniform energy distribution. Higher = flatter spectrum (anti-metallic). Projected transverse to velocity when ortho is on." },
        { key = "rms_weight", type = "slider", label = "RMS Regularization",
          default = 0.05, min = 0.0, max = 0.5, step = 0.01,
          hint = "Penalty for RMS deviation from target. Always acts radially (never projected) so it can correct drift along the trajectory." },
        { key = "rms_target", type = "slider", label = "RMS Target",
          default = 0, min = 0.0, max = 5.0, step = 0.1,
          hint = "Target RMS for audio latents. 0 = auto (2.0, scaled to the active model's latent RMS -- ACE ~2.0, SA3 ~0.685)." },
        { key = "sigma_gate", type = "slider", label = "Regularization Gate",
          default = 0.3, min = 0.0, max = 0.99, step = 0.05,
          hint = "Sigma level below which regularization activates fully. Above this, regularization fades (let structure form freely)." },
        { key = "prox_iterations", type = "slider", label = "Proximal Iterations",
          default = 1, min = 1, max = 5, step = 1,
          hint = "Inner proximal gradient steps per solver step. More = closer to true Wasserstein minimum. 1 is usually enough." },
        { key = "orthogonal_proj", type = "toggle", label = "Transverse Projection",
          default = true,
          hint = "Project SPECTRAL gradient orthogonal to velocity to preserve attack transients. RMS is unaffected (always radial)." },
    },
}

-- ----------------------------------------------------------------------------
-- Constants
-- ----------------------------------------------------------------------------

local EPSILON = 1e-8
local GRAD_CLAMP_MAX = 5.0
local TRUST_REGION_ALPHA = 1.0

-- ----------------------------------------------------------------------------
-- Reusable Zero-Allocation Scratch Buffers
--
-- No semantic state is stored across steps. Buffers are pure scratch,
-- resized only when the input dimension changes. No explicit boundary
-- reset is needed because there is no cross-step / cross-generation state
-- to bleed.
-- ----------------------------------------------------------------------------

local _scratch = {
    v        = {},
    x_next   = {},
    grad     = {},
    energies = {},
    size     = 0,
}

local function ensure_scratch(n)
    if _scratch.size < n then
        -- Grow in place. New entries stay uninitialized because the first
        -- pass in step() unconditionally writes every index.
        for i = _scratch.size, n - 1 do
            _scratch.v[i]        = 0.0
            _scratch.x_next[i]   = 0.0
            _scratch.grad[i]     = 0.0
            _scratch.energies[i] = 0.0
        end
    end
    _scratch.size = n
end

-- ----------------------------------------------------------------------------
-- Math Helpers
-- ----------------------------------------------------------------------------

-- ACE latents measure ~2.0 RMS; SA3 ~0.685; MM3 TBD. model_context is
-- injected by the engine (lua-plugin.h) before every step() call.
-- (Named model_rms_scale, not rms_scale, to avoid colliding with the
-- local `rms_scale` gradient variable used later in step().)
local function model_latent_rms_base()
    local v = model_context and tonumber(model_context.latent_rms)
    if not v or v <= 0 then v = 2.0 end
    return v
end

local function model_rms_scale()
    return model_latent_rms_base() / 2.0
end

-- 0/nil means "auto": ace_default scaled by model_rms_scale(). Nonzero user
-- value wins unchanged.
local function rms_scaled(p, key, ace_default)
    local v = tonumber(p and p[key]) or 0
    if v == 0 then return ace_default * model_rms_scale() end
    return v
end

local function has_nan(t, n)
    for i = 0, n - 1 do
        local val = t[i]
        if val ~= val or math.abs(val) == math.huge then
            return true
        end
    end
    return false
end

-- ----------------------------------------------------------------------------
-- Required step() function
-- ----------------------------------------------------------------------------

function step(xt, vt, sigma_curr, sigma_next, step_index, total_steps, model_fn)
    local p = params or {}
    local tau         = p.tau             or 1.0
    local spec_w      = p.spectral_weight or 0.1
    local rms_w       = p.rms_weight      or 0.05
    local rms_tgt     = rms_scaled(p, "rms_target", 2.0)
    local gate        = math.min(p.sigma_gate or 0.3, 0.99)
    local prox_iters  = math.floor(p.prox_iterations or 1)
    local use_ortho   = p.orthogonal_proj
    if use_ortho == nil then use_ortho = true end

    local n = #xt
    local dt = sigma_next - sigma_curr

    ensure_scratch(n)

    local v        = _scratch.v
    local x_next   = _scratch.x_next
    local grad     = _scratch.grad
    local energies = _scratch.energies

    -- Cache velocity (treated as immutable), seed x_next with base Euler
    -- step, and accumulate base_disp_sq = ||dt * v||^2 in the same pass.
    -- base_disp_sq is invariant across prox iterations (dt and v don't
    -- change), so hoist it out of the inner loop.
    local base_disp_sq = 0.0
    for i = 0, n - 1 do
        local vi = vt[i]
        v[i]      = vi
        local db  = dt * vi
        x_next[i] = xt[i] + db
        base_disp_sq = base_disp_sq + db * db
    end

    -- Regularization gating: fade in as sigma drops below gate
    local reg_strength = 1.0
    if sigma_curr > gate then
        reg_strength = math.max(0.0, 1.0 - (sigma_curr - gate) / (1.0 - gate + EPSILON))
    end

    -- Proximal regularization loop
    if reg_strength > 0.01 and (spec_w > 0.0 or rms_w > 0.0) then
        local step_size = math.abs(dt) / (tau + EPSILON)
        local effective_step = step_size * reg_strength
        local max_disp = TRUST_REGION_ALPHA * math.sqrt(base_disp_sq)

        for iter = 1, prox_iters do
            -- ---- PASS 1: energies + sum_e ----
            local sum_e = 0.0
            for i = 0, n - 1 do
                local xi = x_next[i]
                local e = xi * xi + EPSILON
                energies[i] = e
                sum_e = sum_e + e
            end

            local mean_e = sum_e / n
            local inv_mean_e = 1.0 / (mean_e + EPSILON)

            -- RMS scale (clamped) for radial component
            local current_rms = math.sqrt(sum_e / n)
            local diff_rms = current_rms - rms_tgt
            local rms_scale = 2.0 * diff_rms / (n * current_rms + EPSILON)
            rms_scale = math.min(math.max(rms_scale, -1.0), 1.0)

            -- ---- PASS 2: spectral gradient into grad[] (clamped) ----
            if spec_w > 0.0 then
                for i = 0, n - 1 do
                    local xi = x_next[i]
                    local raw = 2.0 * xi * (inv_mean_e - 1.0 / energies[i]) / n
                    if raw > GRAD_CLAMP_MAX then raw = GRAD_CLAMP_MAX
                    elseif raw < -GRAD_CLAMP_MAX then raw = -GRAD_CLAMP_MAX end
                    grad[i] = spec_w * raw
                end
            else
                for i = 0, n - 1 do grad[i] = 0.0 end
            end

            -- ---- Optional transverse projection of SPECTRAL only ----
            if use_ortho and spec_w > 0.0 then
                local dot_gv = 0.0
                local norm_v_sq = 0.0
                for i = 0, n - 1 do
                    local vi = v[i]
                    dot_gv = dot_gv + grad[i] * vi
                    norm_v_sq = norm_v_sq + vi * vi
                end
                local proj_scale = dot_gv / (norm_v_sq + EPSILON)
                for i = 0, n - 1 do
                    grad[i] = grad[i] - proj_scale * v[i]
                end
            end

            -- ---- PASS 3: add radial RMS component + reg_disp_sq accumulate ----
            local reg_disp_sq = 0.0
            if rms_w > 0.0 then
                for i = 0, n - 1 do
                    local xi = x_next[i]
                    local g = grad[i] + rms_w * rms_scale * xi
                    grad[i] = g
                    local dr = effective_step * g
                    reg_disp_sq = reg_disp_sq + dr * dr
                end
            else
                for i = 0, n - 1 do
                    local dr = effective_step * grad[i]
                    reg_disp_sq = reg_disp_sq + dr * dr
                end
            end

            -- Trust region: bound total regularizer displacement
            local actual_disp = math.sqrt(reg_disp_sq)
            local trust_scale = 1.0
            if actual_disp > max_disp and actual_disp > EPSILON then
                trust_scale = max_disp / actual_disp
            end

            -- ---- PASS 4: apply update ----
            local final_step = effective_step * trust_scale
            for i = 0, n - 1 do
                x_next[i] = x_next[i] - final_step * grad[i]
            end
        end
    end

    -- NaN fallback: restore clean Euler step if numerical breakdown occurs
    if has_nan(x_next, n) then
        for i = 0, n - 1 do
            xt[i] = xt[i] + dt * v[i]
        end
    else
        for i = 0, n - 1 do
            xt[i] = x_next[i]
        end
    end
end
