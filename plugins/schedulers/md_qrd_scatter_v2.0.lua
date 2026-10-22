-- ============================================================================
-- SPDX-License-Identifier: GPL-3.0-or-later
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
-- ============================================================================

-- MD QRD Scatter Scheduler v2.0 -- Quadratic Residue Diffusion
-- MDMAchine | A&E Concepts (C) 2026
--
-- v2.0 CHANGES (ENHANCEMENT PASS):
--   - auto_prime mode (default ON): picks the smallest prime STRICTLY
--     GREATER than num_steps (still respecting prime_root as a floor),
--     so the residue sequence (i*i) mod p never repeats within a
--     single schedule. Fixes the mathematical flaw documented in v1.4:
--     with a fixed small prime (e.g. 11) at high step counts (e.g. 50),
--     the scatter pattern repeated ~4.5 times across the schedule --
--     periodic structure in an anti-periodicity scheduler. Physical
--     QRD acoustic panels use one period across the whole panel; this
--     now matches that design at any step count.
--   - with auto_prime OFF, behavior is exact v1.x parity (prime_root
--     snapped up to nearest prime, repetition and all) for anyone who
--     liked the old character at high step counts.
--   - log line now reports both requested and effective prime plus
--     whether auto mode engaged.
--   - telemetry gains qrd_weight_spread as a single value.
--
-- v1.x lineage: prime snapping (v1.1), build-time diagnostics (v1.2),
--   input fingerprint (v1.3), periodicity limitation documented (v1.4).
--
-- WHAT THIS DOES:
--   Applies physical acoustic treatment logic to the ODE schedule.
--   Uses a prime number sequence to calculate quadratic residues (n^2 mod p)
--   which dictate the exact distance between sigma steps. Completely
--   scatters the compute density to eliminate any harmonic ODE ringing.

scheduler = {
    name        = "md_qrd_scatter",
    display     = "MD QRD Scatter (Acoustic Diffusion)",
    description = "Uses prime number quadratic residues to scatter sigma step intervals. Eliminates harmonic step-ringing by destroying uniform grids.",
    params      = {
        {
            key     = "prime_root",
            type    = "slider",
            label   = "Prime Root",
            default = 11,
            min     = 3,
            max     = 131,
            step    = 1,
            hint    = "Prime floor for the sequence. With Auto Prime on, the effective prime is the smallest prime > max(this, num_steps) so the pattern never repeats within a schedule. With Auto Prime off, v1.x behavior: this value snapped up to the nearest prime.",
        },
        {
            key     = "auto_prime",
            type    = "toggle",
            label   = "Auto Prime",
            default = true,
            hint    = "ON: effective prime always exceeds the step count, guaranteeing a non-repeating scatter pattern across the whole schedule (recommended, especially above ~24 steps). OFF: exact v1.x parity.",
        },
        {
            key     = "scatter_depth",
            type    = "slider",
            label   = "Scatter Depth",
            default = 0.4,
            min     = 0.0,
            max     = 1.0,
            step    = 0.05,
            hint    = "How deeply the sequence perturbs the timeline. 0=off (strict linear parity). 0.4=heavy scattering.",
        },
        {
            key     = "debug_log",
            type    = "toggle",
            label   = "Debug Log",
            default = true,
            hint    = "Print schedule-build diagnostics to console (prime used, scatter stats, endpoint anomaly).",
        },
    },
}

local EPSILON = 1e-8

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function is_prime(x)
    if x < 2 then return false end
    if x == 2 then return true end
    if x % 2 == 0 then return false end
    local i = 3
    while i * i <= x do
        if x % i == 0 then return false end
        i = i + 2
    end
    return true
end

local function next_prime(x)
    local c = x
    if c < 2 then c = 2 end
    while not is_prime(c) do
        c = c + 1
    end
    return c
end

function schedule(output, num_steps, shift)
    local p = params or {}
    local prime_req = math.floor((p.prime_root) or 11)
    local auto_prime = (p.auto_prime ~= false)
    local depth = clamp((p.scatter_depth) or 0.4, 0.0, 1.0)
    local debug_log = (p.debug_log ~= false)

    if debug_log then
        print(string.format(
            "[MD QRD Scatter] FINGERPRINT call_num_steps=%d call_shift=%.6f",
            num_steps, shift))
    end

    -- effective prime selection:
    --   auto ON : smallest prime > max(prime_req, num_steps)
    --             -> residues never repeat within the schedule
    --   auto OFF: v1.x parity (prime_req snapped up to nearest prime)
    local prime
    if auto_prime then
        local floor_val = prime_req
        if num_steps + 1 > floor_val then floor_val = num_steps + 1 end
        prime = next_prime(floor_val)
    else
        prime = next_prime(prime_req)
    end

    local intervals = {}
    local sum_intervals = 0.0
    local weight_min = math.huge
    local weight_max = -math.huge

    for i = 1, num_steps do
        local residue = (i * i) % prime
        local weight = 1.0 + depth * (residue / prime - 0.5)
        weight = math.max(weight, 0.1)
        intervals[i] = weight
        sum_intervals = sum_intervals + weight
        if weight < weight_min then weight_min = weight end
        if weight > weight_max then weight_max = weight end
    end

    local sigmas = {}
    sigmas[0] = 1.0
    local current = 1.0

    for i = 1, num_steps do
        local step_drop = intervals[i] / sum_intervals
        current = current - step_drop
        sigmas[i] = clamp(current, 0.0, 1.0)
    end

    local pre_clamp_final = sigmas[num_steps]
    local endpoint_anomaly = math.abs(pre_clamp_final - 0.0)

    if telemetry ~= nil then
        telemetry.qrd_endpoint_anomaly = endpoint_anomaly
        telemetry.qrd_prime_used = prime
        telemetry.qrd_weight_min = weight_min
        telemetry.qrd_weight_max = weight_max
        telemetry.qrd_weight_spread = weight_max - weight_min
        telemetry.qrd_auto_prime = auto_prime
    end

    if debug_log then
        print(string.format(
            "[MD QRD Scatter] prime_requested=%d prime_used=%d auto_prime=%s depth=%.3f num_steps=%d",
            prime_req, prime, tostring(auto_prime), depth, num_steps))
        print(string.format(
            "[MD QRD Scatter] weight_min=%.4f weight_max=%.4f weight_spread=%.4f (1.0=uniform/no-scatter)",
            weight_min, weight_max, weight_max - weight_min))
        print(string.format(
            "[MD QRD Scatter] pre-clamp final sigma=%.6f endpoint_anomaly=%.6f",
            pre_clamp_final, endpoint_anomaly))
    end

    sigmas[num_steps] = 0.0

    if shift ~= 1.0 then
        for i = 0, num_steps do
            local t = sigmas[i]
            sigmas[i] = shift * t / (1.0 + (shift - 1.0) * t)
        end
    end

    for i = 0, num_steps - 1 do
        output[i] = sigmas[i]
    end
end
