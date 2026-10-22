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

-- MD Lévy Flight Solver v1.1 — Alpha-Stable SDE Noise
-- MDMAchine | A&E Concepts © 2026
--
-- v1.1 CHANGELOG:
--   - Defaults updated to validated baseline (2026-08-12 test session):
--       levy_alpha 1.5 -> 1.65, eta 1.0 -> 0.8,
--       truncation_sigma 5.0 -> 10.0, sigma_floor 0.05 -> 0.1
--   - Ran clean on first test, no code bugs found.
--
-- WHAT THIS DOES:
--   Replaces the Gaussian noise injection in SDE solvers with alpha-stable
--   Lévy noise generated via the Chambers-Mallows-Stuck (CMS) algorithm.
--
--   Lévy noise has heavy tails controlled by the stability index (levy_alpha):
--     levy_alpha = 2.0 : Gaussian (standard SDE behavior)
--     levy_alpha = 1.5 : mild heavy tails, occasional moderate jumps
--     levy_alpha = 1.0 : Cauchy distribution, frequent large jumps
--     levy_alpha < 1.0 : extremely heavy tails, rare massive jumps
--
--   WHY: Gaussian SDE steps can get trapped in local minima in latent space.
--   Lévy flights produce a mix of small local steps and occasional large
--   jumps that allow the solver to explore distant regions. This is the
--   same mechanism that makes Lévy flights optimal foraging strategies in
--   biology — maximize coverage of a search space.
--
--   The noise is truncated at truncation_sigma standard deviations to
--   prevent catastrophic single-step jumps. The truncation preserves the
--   heavy-tail character while bounding the maximum displacement.
--
--   Skewness (beta) controls asymmetry: beta=0 is symmetric, beta>0 biases
--   toward positive jumps, beta<0 biases negative. For audio latents,
--   slight positive skew can bias toward brighter harmonics.
--
-- ARCHITECTURE:
--   - step() solver, stochastic=true
--   - Euler base step + Lévy noise injection scaled by sqrt(|dt|)
--   - CMS algorithm: generates Lévy samples from two uniform random variables
--   - Pure Lua, no external dependencies
--
-- AUDIO LATENT NOTES:
--   The heavy-tail property means the solver occasionally makes large
--   creative jumps in latent space that Gaussian noise would never produce.
--   This can break the solver out of muddy or collapsed states but can also
--   introduce artifacts if levy_alpha is too low. Start at 1.5 and go lower.
--
-- VALIDATED BASELINE (2026-08-12):
--   levy_alpha=1.65, eta=0.8, truncation_sigma=10.0, sigma_floor=0.1
--   Scheduler: MD Psychoacoustic (defaults). Guider: MD Phase Inversion (defaults).
--   Result: clean, no artifacts, distinct character from other solvers.
-- ============================================================================

solver = {
    name        = "md_levy_flight",
    display     = "MD Lévy Flight (Alpha-Stable SDE v1.1)",
    description = "SDE solver with Lévy alpha-stable noise. Heavy tails enable latent space exploration beyond Gaussian reach. CMS algorithm, truncated for stability. Validated baseline: alpha=1.65, eta=0.8.",
    accent      = "violet",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = false,
    stochastic  = true,
    owns_loop   = false,
    params      = {
        { key = "levy_alpha", type = "slider", label = "Stability Index (α)",
          default = 1.65, min = 0.3, max = 2.0, step = 0.05,
          hint = "2.0 = Gaussian. 1.5 = mild heavy tails. 1.0 = Cauchy. <1 = extreme jumps. Controls tail weight. Validated: 1.65." },
        { key = "beta", type = "slider", label = "Skewness (β)",
          default = 0.0, min = -1.0, max = 1.0, step = 0.05,
          hint = "0 = symmetric. Positive = biased toward positive jumps. Negative = biased negative." },
        { key = "eta", type = "slider", label = "Noise Scale (η)",
          default = 0.8, min = 0.0, max = 3.0, step = 0.05,
          hint = "Multiplier on the Lévy noise amplitude. 0 = deterministic (pure Euler). 1 = standard SDE scale. Validated: 0.8." },
        { key = "truncation_sigma", type = "slider", label = "Truncation (σ)",
          default = 10.0, min = 1.0, max = 20.0, step = 0.5,
          hint = "Clamp Lévy samples beyond this many scale units. Lower = safer but less exploration." },
        { key = "sigma_floor", type = "slider", label = "Noise Floor (sigma)",
          default = 0.1, min = 0.0, max = 0.5, step = 0.01,
          hint = "Stop injecting noise below this sigma level. Protects detail phase from Lévy jumps." },
    },
}

-- ── Constants ───────────────────────────────────────────────────────────────

local EPSILON = 1e-8
local PI = math.pi

-- ── PRNG state (xoshiro128+ for reproducibility) ───────────────────────────
-- We need uniform [0,1) samples. math.random works but is implementation-
-- dependent. Using it here for portability; replace with xoshiro if needed.

local function uniform()
    return math.random()
end

-- ── Chambers-Mallows-Stuck Algorithm ────────────────────────────────────────
-- Generates a single alpha-stable Lévy random variable.
-- Reference: Chambers, Mallows, Stuck (1976) JASA 71(354):340-344
--
-- For alpha != 1:
--   X = S(alpha, beta) * sin(alpha * (V + C)) / (cos(V))^(1/alpha)
--       * (cos(V - alpha*(V + C)) / W)^((1-alpha)/alpha)
--
-- Where V ~ Uniform(-pi/2, pi/2), W ~ Exponential(1), C = atan(beta * tan(pi*alpha/2))

local function levy_sample(alpha, beta)
    -- Uniform on (-pi/2, pi/2)
    local V = (uniform() - 0.5) * PI

    -- Exponential(1) via inverse CDF
    local u2 = uniform()
    if u2 < 1e-15 then u2 = 1e-15 end
    local W = -math.log(u2)

    if math.abs(alpha - 1.0) < 1e-6 then
        -- Special case: Cauchy-like (alpha = 1)
        local half_pi_plus_bV = PI / 2 + beta * V
        local X = (1.0 / (PI / 2)) * (half_pi_plus_bV * math.tan(V)
                  - beta * math.log((PI / 2) * W * math.cos(V) / half_pi_plus_bV + EPSILON))
        return X
    end

    -- General case
    local C = math.atan(beta * math.tan(PI * alpha / 2.0))
    local S = (1.0 + beta * beta * math.tan(PI * alpha / 2.0) ^ 2) ^ (1.0 / (2.0 * alpha))

    local alpha_VC = alpha * (V + C)
    local cos_V = math.cos(V)

    if math.abs(cos_V) < 1e-10 then
        return 0.0  -- degenerate, return zero
    end

    local term1 = math.sin(alpha_VC) / (cos_V ^ (1.0 / alpha))
    local cos_remainder = math.cos(V - alpha_VC)

    if cos_remainder <= 0 then
        return 0.0  -- protect against negative base in fractional exponent
    end

    local term2 = (cos_remainder / W) ^ ((1.0 - alpha) / alpha)
    local X = S * term1 * term2

    -- NaN/Inf guard
    if X ~= X or math.abs(X) == math.huge then
        return 0.0
    end

    return X
end

-- ── Helpers ─────────────────────────────────────────────────────────────────

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function fa_to_tbl(fa, n)
    local t = {}
    for i = 0, n - 1 do t[i] = fa[i] end
    return t
end

local function tbl_to_fa(t, fa, n)
    for i = 0, n - 1 do fa[i] = t[i] end
end

-- ── Required step() function ────────────────────────────────────────────────

function step(xt, vt, sigma_curr, sigma_next, step_index, total_steps, model_fn)
    local p = params or {}
    local alpha     = p.levy_alpha       or 1.5
    local beta      = p.beta             or 0.0
    local eta       = p.eta              or 1.0
    local trunc     = p.truncation_sigma or 5.0
    local floor     = p.sigma_floor      or 0.05

    local n = #xt
    local dt = sigma_next - sigma_curr

    local x = fa_to_tbl(xt, n)
    local v = fa_to_tbl(vt, n)

    -- Euler base step
    for i = 0, n - 1 do
        x[i] = x[i] + dt * v[i]
    end

    -- Lévy noise injection (only above sigma floor)
    if eta > 0.0 and sigma_curr > floor then
        local noise_scale = eta * math.sqrt(math.abs(dt))

        -- Scale noise by sigma position (more noise at high sigma, less at low)
        local sigma_weight = sigma_curr  -- linear scaling with current noise level

        for i = 0, n - 1 do
            local sample = levy_sample(alpha, beta)

            -- Truncate
            sample = clamp(sample, -trunc, trunc)

            x[i] = x[i] + noise_scale * sigma_weight * sample
        end
    end

    tbl_to_fa(x, xt, n)
end
