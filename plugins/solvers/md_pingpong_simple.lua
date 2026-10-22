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

-- MD PingPong Simple v1.1 — Ancestral Euler + Momentum + Look-Back Smoother
-- MDMAchine | A&E Concepts © 2026
--
-- Port of MD_PingPong_Samplers.py (single-branch path) to HOT-Step-CPP Lua solver.
--
-- WHAT THIS DOES:
--   Standard ODE solvers (Euler, Heun, STORM) advance the latent deterministically.
--   PingPong injects ancestral (stochastic) noise at each step — the "ping" is
--   the clean denoising step, the "pong" is the noise re-injection that keeps
--   the trajectory alive and stochastic.
--
--   CORE STEP MATH:
--     dt     = t_prev - t_curr               (negative — stepping down)
--     x_euler = xt + dt * vt                 (standard Euler, matches STORM/OmniRelational)
--     x_next  = x_euler + noise * |dt| * ancestral_strength
--
--   MOMENTUM:
--   Latent velocity (x - x_prev) carried forward at each step. Maintains
--   flow continuity across the ODE trajectory — reduces erratic jumps between
--   steps, especially at low step counts.
--
--   NOISE COHERENCE:
--   Blends fresh Gaussian noise with the previous step's noise at ratio
--   noise_coherence. 0=fully fresh, 1=fully carried. Useful for temporal
--   smoothness in audio; keep low (0-0.2) to avoid spectral smearing.
--
--   LOOK-BACK SNR SMOOTHER:
--   λ(σ) = lambda_base * (σ/σ_max)^snr_power — heavy at high sigma, fades to
--   zero at low sigma. Suppresses ODE manifold shearing and harmonic hum.
--   Reference: arXiv:2602.09449
--
--   RMS SERVO:
--   Downward-only energy ceiling that follows a smooth curve from rms_max
--   (high sigma) to rms_min (low sigma). Domain-tunable: image latents
--   typically sit around 0.75-0.97; audio latents around 0.3-0.7.
--   Servo is DOWNWARD ONLY — never boosts energy, only clamps excess.
--
-- SOLVER API NOTE:
--   HOT-Step-CPP passes velocity vt where dt = t_prev - t_curr is NEGATIVE
--   (stepping from high sigma to low sigma). Euler update is:
--     x_next = xt + dt * vt   (same convention as STORM and OmniRelational)
--   Ancestral noise is added as: noise * |dt| * strength
--   Do NOT use xt - t_curr * vt for denoised — sign convention mismatch.
--
-- PARAMS:
--   ancestral_strength — noise injection scale. 1.0=standard ancestral, 0=pure ODE
--   noise_coherence    — step-to-step noise carry. 0=fresh, 0.2=subtle temporal link
--   momentum_strength  — latent velocity carry-over. 0.15=subtle, 0.3=strong
--   look_back_enabled  — SNR smoother toggle
--   look_back_lambda   — max smoothing weight (0.55=25-step, 0.35=35-step)
--   look_back_snr_power — falloff exponent (1.3=25-step, 1.5=35-step)
--   rms_servo          — energy ceiling toggle
--   rms_target_min     — servo floor at low sigma. 0 = auto (1.0, scaled to the
--                         active model's latent RMS -- ACE ~2.0, SA3 ~0.685)
--   rms_target_max     — servo ceiling at high sigma. 0 = auto (2.2, scaled)
--   rms_servo_gain     — correction aggressiveness (0.6=default, 1.0=hard snap)
--   seed               — RNG seed
-- ============================================================================

solver = {
    name        = "md_pingpong_simple",
    display     = "MD PingPong Simple (Ancestral)",
    description = "Ancestral Euler with momentum, noise coherence, look-back SNR smoother, and domain-tunable RMS servo. Single-branch stochastic sampler. Port of MD_PingPong_Samplers v3.5.",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = true,
    stochastic  = true,
    params      = {
        {
            key     = "ancestral_strength",
            type    = "slider",
            label   = "Ancestral Strength",
            default = 0.2,
            min     = 0.0,
            max     = 1.5,
            step    = 0.05,
            hint    = "Noise injection strength. 1.0=standard ancestral. 0=pure ODE (no noise).",
        },
        {
            key     = "noise_coherence",
            type    = "slider",
            label   = "Noise Coherence",
            default = 0.0,
            min     = 0.0,
            max     = 1.0,
            step    = 0.05,
            hint    = "Step-to-step noise correlation. 0=fresh noise each step. 0.2=subtle temporal link. Keep low for audio to avoid smearing.",
        },
        {
            key     = "momentum_strength",
            type    = "slider",
            label   = "Momentum",
            default = 0.1,
            min     = 0.0,
            max     = 0.5,
            step    = 0.01,
            hint    = "Latent velocity carry-over. 0.1=subtle flow continuity. 0.3=strong.",
        },
        {
            key     = "look_back_enabled",
            type    = "toggle",
            label   = "Look-Back Smoother",
            default = true,
            hint    = "SNR-adaptive latent EMA. Suppresses ODE manifold shearing and harmonic hum. arXiv:2602.09449.",
        },
        {
            key     = "look_back_lambda",
            type    = "slider",
            label   = "Look-Back Lambda",
            default = 0.55,
            min     = 0.1,
            max     = 1.0,
            step    = 0.05,
            hint    = "Max smoothing weight. Active when Look-Back Smoother is on. 0.55=25-step, 0.35=35-step.",
        },
        {
            key     = "look_back_snr_power",
            type    = "slider",
            label   = "SNR Power",
            default = 1.3,
            min     = 0.5,
            max     = 3.0,
            step    = 0.1,
            hint    = "Falloff exponent. Active when Look-Back Smoother is on. Higher=smoother fade at low sigma.",
        },
        {
            key     = "rms_servo",
            type    = "toggle",
            label   = "RMS Servo",
            default = true,
            hint    = "Downward-only energy ceiling. Prevents latent energy accumulation. Off by default — tune min/max for your domain before enabling.",
        },
        {
            key     = "rms_target_min",
            type    = "slider",
            label   = "RMS Target Min",
            default = 0,
            min     = 0.0,
            max     = 3.0,
            step    = 0.05,
            hint    = "RMS ceiling at low sigma (late steps). Active when RMS Servo is on. 0 = auto (1.0, scaled to the active model's latent RMS -- ACE ~2.0, SA3 ~0.685).",
        },
        {
            key     = "rms_target_max",
            type    = "slider",
            label   = "RMS Target Max",
            default = 0,
            min     = 0.0,
            max     = 4.0,
            step    = 0.05,
            hint    = "RMS ceiling at high sigma (early steps). Active when RMS Servo is on. 0 = auto (2.2, scaled to the active model's latent RMS).",
        },
        {
            key     = "rms_servo_gain",
            type    = "slider",
            label   = "Servo Gain",
            default = 0.75,
            min     = 0.1,
            max     = 1.0,
            step    = 0.05,
            hint    = "Servo correction aggressiveness. Active when RMS Servo is on. 0.6=soft, 1.0=hard snap.",
        },
        {
            key     = "seed",
            type    = "slider",
            label   = "Seed",
            default = 42,
            min     = 0,
            max     = 999999,
            step    = 1,
            hint    = "RNG seed for noise generation.",
        },
    },
}

-- ── State (file-level locals, reset when n changes) ──────────────────────────

local _prev_x       = nil   -- for momentum
local _prev_noise   = nil   -- for noise coherence
local _look_back_xp = nil   -- for look-back smoother
local _sigma_max    = nil   -- captured at step 0
local _last_n       = 0

-- Hoisted scratch tables — reused every step to avoid GC pressure
-- Initialized on first step or when n changes
local _noise_buf    = {}    -- reusable noise array
local _x_next_buf   = {}    -- reusable output array
local _x_copy_buf   = {}    -- reusable momentum copy

local EPSILON = 1e-8

-- ── Helpers ──────────────────────────────────────────────────────────────────

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
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

-- Seeded LCG RNG — deterministic, no dependency on math.random state
local function make_rng(seed)
    local state = math.floor(seed) % 2147483647
    if state <= 0 then state = state + 2147483646 end
    return function()
        state = (state * 1664525 + 1013904223) % 2147483648
        return state / 2147483648.0
    end
end

-- Box-Muller: two uniform [0,1] → one standard normal sample
local function normal(u1, u2)
    return math.sqrt(-2.0 * math.log(math.max(u1, EPSILON))) * math.cos(2.0 * math.pi * u2)
end

-- Array RMS
local function rms(arr, n)
    local s = 0.0
    for i = 0, n - 1 do s = s + arr[i] * arr[i] end
    return math.sqrt(s / n + EPSILON)
end

-- ── Required step() function ──────────────────────────────────────────────────

function step(xt, vt, t_curr, t_prev, n)
    local step_idx_ = step_index or 0

    -- Reset state on new generation: n change OR step 0 of any new run.
    -- Must check step_idx_==0 because same-length generations won't trigger n change,
    -- causing momentum/look-back to bleed finished audio from the previous run into
    -- the noise of the new one — explosive velocity on step 1.
    if n ~= _last_n or step_idx_ == 0 then
        _prev_x       = nil
        _prev_noise   = nil
        _look_back_xp = nil
        _sigma_max    = nil
        _last_n       = n
        -- Pre-size scratch tables for this n
        for i = 0, n - 1 do
            _noise_buf[i]  = 0.0
            _x_next_buf[i] = 0.0
            _x_copy_buf[i] = 0.0
        end
    end

    -- Read params with safe fallbacks
    local anc_strength   = (params and params.ancestral_strength)   or 1.0
    local noise_coh      = (params and params.noise_coherence)       or 0.0
    local mom_str        = (params and params.momentum_strength)     or 0.1
    local lb_enabled     = (params and params.look_back_enabled)     or false
    local lb_lambda      = (params and params.look_back_lambda)      or 0.55
    local lb_snr_power   = (params and params.look_back_snr_power)   or 1.3
    local rms_servo_on   = (params and params.rms_servo)             or true
    local rms_tgt_min    = rms_scaled(params, "rms_target_min", 1.0)
    local rms_tgt_max    = rms_scaled(params, "rms_target_max", 2.2)
    local rms_servo_gain = (params and params.rms_servo_gain)        or 0.6
    local seed           = math.floor((params and params.seed)       or 42)

    -- Capture sigma_max on first step for ratio computation
    if _sigma_max == nil then _sigma_max = t_curr end
    local sigma_max   = _sigma_max
    local sigma_ratio = clamp(t_curr / math.max(sigma_max, EPSILON), 0.0, 1.0)

    -- dt = t_prev - t_curr. In flow-matching, t steps DOWN (1→0),
    -- HOT-Step API: t_curr=high sigma, t_prev=lower target. t_curr > t_prev. dt=t_prev-t_curr is NEGATIVE.
    local dt = t_prev - t_curr

    -- Save current xt for momentum (reuse hoisted buffer)
    for i = 0, n - 1 do _x_copy_buf[i] = xt[i] end

    -- ── NOISE GENERATION ──────────────────────────────────────────────────────
    -- Write into hoisted buffer — no table allocation per step
    local rng = make_rng(seed + step_idx_ * 7919)
    for i = 0, n - 1 do
        local u1 = math.max(rng(), EPSILON)
        local u2 = rng()
        _noise_buf[i] = normal(u1, u2)
    end

    -- Noise coherence: blend with carried noise from previous step
    if noise_coh > 0.0 and _prev_noise ~= nil then
        for i = 0, n - 1 do
            _noise_buf[i] = _noise_buf[i] * (1.0 - noise_coh) + _prev_noise[i] * noise_coh
        end
    end
    -- Store for next step — reuse _prev_noise table if same size
    if _prev_noise == nil then _prev_noise = {} end
    for i = 0, n - 1 do _prev_noise[i] = _noise_buf[i] end

    -- ── ANCESTRAL STEP ────────────────────────────────────────────────────────
    -- Variance-preserving SDE noise for flow matching:
    --   noise_scale = sqrt(t_prev^2 - t_curr^2) * anc_strength
    -- t_curr > t_prev, so t_curr^2 - t_prev^2 > 0. Confirmed numerically.
    local noise_scale = math.sqrt(math.max(t_curr * t_curr - t_prev * t_prev, 0.0)) * anc_strength
    if noise_scale > EPSILON then
        for i = 0, n - 1 do
            _x_next_buf[i] = xt[i] + dt * vt[i] + _noise_buf[i] * noise_scale
        end
    else
        for i = 0, n - 1 do _x_next_buf[i] = xt[i] + dt * vt[i] end
    end

    -- ── MOMENTUM ──────────────────────────────────────────────────────────────
    if mom_str > 0.0 and _prev_x ~= nil then
        for i = 0, n - 1 do
            local vel = _x_copy_buf[i] - _prev_x[i]
            _x_next_buf[i] = _x_next_buf[i] + vel * mom_str
        end
    end

    -- ── LOOK-BACK SNR SMOOTHER ────────────────────────────────────────────────
    -- λ(σ) = lb_lambda * (σ/σ_max)^lb_snr_power — heavy early, fades late.
    if lb_enabled then
        local lb_w = lb_lambda * (sigma_ratio ^ lb_snr_power)
        if _look_back_xp == nil then
            _look_back_xp = {}
            for i = 0, n - 1 do
                local u1 = math.max(rng(), EPSILON)
                local u2 = rng()
                _look_back_xp[i] = _x_next_buf[i] + normal(u1, u2) * sigma_max * 0.1
            end
        end
        for i = 0, n - 1 do
            _x_next_buf[i] = _x_next_buf[i] * (1.0 - lb_w) + _look_back_xp[i] * lb_w
        end
        -- Update look-back buffer in-place
        for i = 0, n - 1 do _look_back_xp[i] = _x_next_buf[i] end
    end

    -- ── RMS SERVO (DOWNWARD ONLY) ─────────────────────────────────────────────
    if rms_servo_on then
        local rms_target = rms_tgt_min + (sigma_ratio ^ 0.6) * (rms_tgt_max - rms_tgt_min)
        local cur_rms = rms(_x_next_buf, n)
        if cur_rms > rms_target then
            local servo_rms = cur_rms + rms_servo_gain * (rms_target - cur_rms)
            local scale = servo_rms / cur_rms
            for i = 0, n - 1 do _x_next_buf[i] = _x_next_buf[i] * scale end
        end
    end

    -- ── UPDATE STATE & WRITE OUTPUT ───────────────────────────────────────────
    -- Store momentum reference — reuse table, copy values
    if _prev_x == nil then _prev_x = {} end
    for i = 0, n - 1 do _prev_x[i] = _x_copy_buf[i] end
    for i = 0, n - 1 do xt[i] = _x_next_buf[i] end
end
