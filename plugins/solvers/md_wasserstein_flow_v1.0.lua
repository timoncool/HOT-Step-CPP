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

-- MD Wasserstein Flow Solver v1.0 — Proximal Optimal Transport Steps
-- MDMAchine | A&E Concepts © 2026
--
-- WHAT THIS DOES:
--   Treats each denoising step as an optimal transport problem. Instead of
--   following the velocity field directly (Euler), computes the step that
--   minimizes a Wasserstein-regularized energy:
--
--     x_next = argmin_x { (1/2τ)||x - x_curr||² + E(x) }
--
--   This is the JKO (Jordan-Kinderlehrer-Otto) scheme, a proximal operator
--   approach to gradient flows in Wasserstein space.
--
--   The energy E(x) combines:
--     - Model velocity alignment: -<v, x-x_curr>  (follow the score)
--     - Spectral flatness penalty: encourages uniform energy distribution
--     - RMS regularization: keeps latent norms in expected range
--
--   The proximal parameter τ (tau) controls the transport cost:
--     Small τ: expensive transport, conservative steps (stay close)
--     Large τ: cheap transport, aggressive steps (big moves OK)
--
--   PRACTICAL IMPLEMENTATION:
--   Full Wasserstein OT is intractable in latent dimension. We approximate
--   with the proximal gradient descent interpretation:
--     x_next = x_curr + dt * v - (dt/τ) * ∇R(x_curr)
--   Where R is the regularization energy (spectral flatness + RMS).
--   This reduces to a regularized Euler step with the regularizer acting
--   as a transport cost penalty.
--
-- WHY: Standard Euler follows the velocity field blindly. Wasserstein flow
--   adds a "transportation cost" that penalizes large, inefficient moves
--   in latent space. The result is smoother transport with fewer artifacts
--   from overshooting. The spectral regularizer additionally prevents the
--   energy distribution from collapsing into narrow bands (anti-metallic).
--
-- AUDIO LATENT NOTES:
--   The spectral flatness penalty directly addresses metallic shearing by
--   preventing energy from concentrating in upper latent bands. The RMS
--   regularizer keeps the latent norm near the ACE-Step target (~2.0).
-- ============================================================================

solver = {
    name        = "md_wasserstein_flow",
    display     = "MD Wasserstein Flow (Proximal OT v1.0)",
    description = "Proximal optimal transport solver. Wasserstein-regularized steps with spectral flatness and RMS penalties. Smoother transport than Euler with anti-metallic properties.",
    accent      = "teal",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = true,
    stochastic  = false,
    owns_loop   = false,
    params      = {
        { key = "tau", type = "slider", label = "Transport Cost (τ)",
          default = 1.0, min = 0.1, max = 10.0, step = 0.1,
          hint = "Proximal parameter. Low = conservative (small steps). High = permissive (big steps OK)." },
        { key = "spectral_weight", type = "slider", label = "Spectral Flatness Weight",
          default = 0.1, min = 0.0, max = 1.0, step = 0.01,
          hint = "Penalty for non-uniform energy distribution. Higher = flatter spectrum (anti-metallic)." },
        { key = "rms_weight", type = "slider", label = "RMS Regularization",
          default = 0.05, min = 0.0, max = 0.5, step = 0.01,
          hint = "Penalty for RMS deviation from target. Keeps latent norms stable." },
        { key = "rms_target", type = "slider", label = "RMS Target",
          default = 0, min = 0.0, max = 5.0, step = 0.1,
          hint = "Target RMS for audio latents. 0 = auto (2.0, scaled to the active model's latent RMS -- ACE ~2.0, SA3 ~0.685)." },
        { key = "sigma_gate", type = "slider", label = "Regularization Gate",
          default = 0.3, min = 0.0, max = 1.0, step = 0.05,
          hint = "Sigma level below which regularization activates fully. Above this, regularization fades (let structure form freely)." },
        { key = "prox_iterations", type = "slider", label = "Proximal Iterations",
          default = 1, min = 1, max = 5, step = 1,
          hint = "Inner proximal gradient steps per solver step. More = closer to true Wasserstein minimum. 1 is usually enough." },
    },
}

-- ── Constants ───────────────────────────────────────────────────────────────

local EPSILON = 1e-8

-- ── State ───────────────────────────────────────────────────────────────────

local _prev_x = nil
local _n = 0

-- ── Helpers ─────────────────────────────────────────────────────────────────

local function fa_to_tbl(fa, n)
    local t = {}
    for i = 0, n - 1 do t[i] = fa[i] end
    return t
end

local function tbl_to_fa(t, fa, n)
    for i = 0, n - 1 do fa[i] = t[i] end
end

local function vec_rms(t, n)
    local s = 0.0
    for i = 0, n - 1 do s = s + t[i] * t[i] end
    return math.sqrt(s / n)
end

local function has_nan(t, n)
    for i = 0, n - 1 do
        if t[i] ~= t[i] or math.abs(t[i]) == math.huge then return true end
    end
    return false
end

-- ── Spectral Flatness Gradient ──────────────────────────────────────────────
-- Spectral flatness = geometric_mean / arithmetic_mean of |x_i|²
-- Gradient of -log(flatness) w.r.t. x_i pushes toward uniform energy.
--
-- Approximation: treat x as a 1D signal, compute energy per element,
-- gradient points from high-energy elements toward mean energy.

local function spectral_flatness_grad(x, n)
    local grad = {}
    local energies = {}
    local sum_e = 0.0
    local sum_log_e = 0.0

    for i = 0, n - 1 do
        local e = x[i] * x[i] + EPSILON
        energies[i] = e
        sum_e = sum_e + e
        sum_log_e = sum_log_e + math.log(e)
    end

    local mean_e = sum_e / n
    local log_geo_mean = sum_log_e / n

    -- Gradient: push energy toward uniformity
    -- d(-log flatness)/dx_i ≈ 2*x_i * (1/mean_e - 1/e_i) / n
    for i = 0, n - 1 do
        grad[i] = 2.0 * x[i] * (1.0 / mean_e - 1.0 / energies[i]) / n
    end

    return grad
end

-- ── RMS Gradient ────────────────────────────────────────────────────────────
-- Gradient of (rms - target)² w.r.t. x_i

local function rms_grad(x, rms_val, target, n)
    local grad = {}
    local diff = rms_val - target
    local scale = 2.0 * diff / (n * rms_val + EPSILON)
    for i = 0, n - 1 do
        grad[i] = scale * x[i]
    end
    return grad
end

-- ACE latents measure ~2.0 RMS; SA3 ~0.685; MM3 TBD. model_context is
-- injected by the engine (lua-plugin.h) before every step() call.
local function latent_rms_base()
    local v = model_context and tonumber(model_context.latent_rms)
    if not v or v <= 0 then v = 2.0 end
    return v
end

local function rms_scale()
    return latent_rms_base() / 2.0
end

-- 0/nil means "auto": ace_default scaled by rms_scale(). Nonzero user value
-- wins unchanged.
local function rms_scaled(p, key, ace_default)
    local v = tonumber(p and p[key]) or 0
    if v == 0 then return ace_default * rms_scale() end
    return v
end

-- ── Required step() function ────────────────────────────────────────────────

function step(xt, vt, sigma_curr, sigma_next, step_index, total_steps, model_fn)
    local p = params or {}
    local tau         = p.tau             or 1.0
    local spec_w      = p.spectral_weight or 0.1
    local rms_w       = p.rms_weight      or 0.05
    local rms_tgt     = rms_scaled(p, "rms_target", 2.0)
    local gate        = p.sigma_gate      or 0.3
    local prox_iters  = math.floor(p.prox_iterations or 1)

    local n = #xt
    local dt = sigma_next - sigma_curr

    -- Reset on first step
    if step_index == 0 then
        _prev_x = nil
        _n = n
    end

    local x = fa_to_tbl(xt, n)
    local v = fa_to_tbl(vt, n)

    -- Base Euler step
    local x_next = {}
    for i = 0, n - 1 do
        x_next[i] = x[i] + dt * v[i]
    end

    -- Regularization gating: fade in as sigma drops below gate
    local reg_strength = 1.0
    if sigma_curr > gate then
        reg_strength = math.max(0.0, 1.0 - (sigma_curr - gate) / (1.0 - gate + EPSILON))
    end

    -- Proximal gradient iterations
    if reg_strength > 0.01 then
        local step_size = math.abs(dt) / (tau + EPSILON)

        for iter = 1, prox_iters do
            -- Spectral flatness gradient
            if spec_w > 0 then
                local sg = spectral_flatness_grad(x_next, n)
                for i = 0, n - 1 do
                    x_next[i] = x_next[i] - step_size * reg_strength * spec_w * sg[i]
                end
            end

            -- RMS regularization gradient
            if rms_w > 0 then
                local rms_val = vec_rms(x_next, n)
                local rg = rms_grad(x_next, rms_val, rms_tgt, n)
                for i = 0, n - 1 do
                    x_next[i] = x_next[i] - step_size * reg_strength * rms_w * rg[i]
                end
            end
        end
    end

    -- NaN guard: fall back to pure Euler
    if has_nan(x_next, n) then
        for i = 0, n - 1 do
            x_next[i] = x[i] + dt * v[i]
        end
    end

    -- Store for potential future use
    _prev_x = x

    tbl_to_fa(x_next, xt, n)
end
