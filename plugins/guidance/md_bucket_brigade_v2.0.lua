-- ============================================================================
-- SPDX-License-Identifier: GPL-3.0-or-later
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
-- ============================================================================

-- MD Bucket Brigade Guider v2.0 -- Analog Temporal Feedback
-- MDMAchine | A&E Concepts (C) 2026
--
-- v2.0 CHANGES (ENHANCEMENT/OPTIMIZATION PASS):
--   PERFORMANCE:
--   - true ring buffer: slots are indexed cur_step % ring_size and
--     REUSED in place. v1.x allocated a fresh n-entry Lua table every
--     call (335k entries per call at 50-step runs) and scanned the
--     whole buffer with pairs() to prune; v2.0 does zero per-call
--     allocation after warmup and pruning is free (overwrite is the
--     prune). buffer_len is tracked arithmetically, no scan.
--   - degradation output goes into a reusable module-level scratch
--     buffer instead of a fresh table per echo.
--   NEW FEATURE:
--   - regeneration toggle (default ON = exact v1.x parity): controls
--     whether the buffer stores the POST-feedback result (regenerative
--     -- echoes contain earlier echoes, feedback compounds gradually
--     like a real BBD pedal's regen knob; this is why echo_mag climbs
--     over a run) or the PRE-feedback result (clean single-tap echo,
--     each echo contains only base APG output, no compounding).
--     OFF gives a tighter, more transparent slap-back character;
--     ON gives the thicker self-reinforcing smear v1.x always had.
--   CARRIED FROM v1.7:
--   - NaN/inf guard on echo injection, N+1 engine call-count
--     documentation, summary/fingerprint/debug logging, v1.5 reset
--     boundary fix.
--
-- WHAT THIS DOES:
--   Mimics an analog BBD delay circuit. Stores the guided latent result
--   in a ring buffer, degrades the high frequencies (voltage loss), and
--   feeds it back into the trajectory N steps later. Glues the transients
--   and smears the temporal field in a highly musical way.
--
-- ENGINE NOTE: HS-CPP invokes the guider N+1 times per N-step
-- generation (confirmed in logs: steps=25 at 24 Inference Steps).
-- Set steps_per_generation to the UI Inference Steps value; the
-- boundary check lands correctly because of this. The last
-- generation's summary only prints when the next generation starts.

guidance = {
    name        = "md_bucket_brigade",
    display     = "MD Bucket Brigade (Analog Feedback)",
    description = "Captures the latent, degrades it, and feeds it back into the generation steps later like a physical analog delay pedal. Creates intense transient glue.",
    stateful    = true,
    params      = {
        {
            key     = "delay_steps",
            type    = "slider",
            label   = "Delay Steps",
            default = 3,
            min     = 1,
            max     = 8,
            step    = 1,
            hint    = "How many inference steps to delay the signal before feedback.",
        },
        {
            key     = "feedback_gain",
            type    = "slider",
            label   = "Feedback Gain",
            default = 0.15,
            min     = 0.0,
            max     = 0.5,
            step    = 0.01,
            hint    = "Volume of the delayed signal mixed back into the latent. 0=off (strict parity).",
        },
        {
            key     = "bbd_degradation",
            type    = "slider",
            label   = "BBD Degradation",
            default = 0.6,
            min     = 0.0,
            max     = 1.0,
            step    = 0.05,
            hint    = "Simulates high-frequency signal loss in analog bucket brigade chips via moving average.",
        },
        {
            key     = "regeneration",
            type    = "toggle",
            label   = "Regeneration",
            default = true,
            hint    = "ON (v1.x parity): buffer stores the post-feedback result -- echoes contain earlier echoes, feedback compounds like a real BBD regen knob. OFF: buffer stores pre-feedback result -- clean single-tap echo, tighter and more transparent.",
        },
        {
            key     = "steps_per_generation",
            type    = "slider",
            label   = "Steps Per Generation",
            default = 24,
            min     = 4,
            max     = 128,
            step    = 1,
            hint    = "MUST match your Inference Steps setting. Used to force a buffer reset at generation boundaries, since the engine doesn't reliably expose a step_index or generation ID to guiders. Mismatched value can cause stale-echo bleed across generations.",
        },
        {
            key     = "debug_log",
            type    = "toggle",
            label   = "Debug Log",
            default = true,
            hint    = "Print per-step buffer state and echo hit/miss diagnostics to console.",
        },
        {
            key     = "summary_log",
            type    = "toggle",
            label   = "Summary Log",
            default = true,
            hint    = "Print one rollup line per generation (HIT/MISS counts, avg/peak echo magnitude) at the start of the next generation.",
        },
    },
}

-- ring buffer: slot tables keyed 0..ring_size-1, reused in place.
-- _ring_step[slot] records which internal_step the slot currently holds,
-- so a lookup can verify the slot actually contains the wanted step.
local _ring = {}
local _ring_step = {}
local _ring_size = 0
local _stored_count = 0   -- how many slots hold valid data this generation

local _last_n = 0
local _internal_step = -1   -- -1 so first call increments to 0

-- reusable degradation scratch
local _deg_scratch = {}

-- generation-level accumulators for the summary line
local _gen_hits = 0
local _gen_misses_warmup = 0
local _gen_misses_unexpected = 0
local _gen_mag_sum = 0.0
local _gen_mag_max = 0.0
local _gen_buffer_len_min = math.huge
local _gen_buffer_len_max = -math.huge
local _gen_steps_seen = 0

local EPSILON = 1e-8

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

-- 3-point moving average into the reusable scratch buffer
local function apply_bbd_degradation(arr, n, strength)
    if strength <= 0.0 or n < 3 then return arr end
    _deg_scratch[0] = arr[0]
    _deg_scratch[n-1] = arr[n-1]
    for i = 1, n - 2 do
        local blurred = (arr[i-1] + arr[i] + arr[i+1]) / 3.0
        _deg_scratch[i] = arr[i] * (1.0 - strength) + blurred * strength
    end
    return _deg_scratch
end

local function print_summary(reason)
    if _gen_steps_seen <= 0 then return end
    local mag_mean = (_gen_hits > 0) and (_gen_mag_sum / _gen_hits) or 0.0
    local blen_min = (_gen_buffer_len_min == math.huge) and 0 or _gen_buffer_len_min
    local blen_max = (_gen_buffer_len_max == -math.huge) and 0 or _gen_buffer_len_max
    print(string.format(
        "[MD Bucket Brigade] SUMMARY (reset=%s) steps=%d hits=%d miss_warmup=%d miss_unexpected=%d echo_mag_mean=%.6f echo_mag_peak=%.6f buffer_len_range=%d-%d",
        reason, _gen_steps_seen, _gen_hits, _gen_misses_warmup, _gen_misses_unexpected,
        mag_mean, _gen_mag_max, blen_min, blen_max))
end

local function reset_gen_stats()
    _gen_hits = 0
    _gen_misses_warmup = 0
    _gen_misses_unexpected = 0
    _gen_mag_sum = 0.0
    _gen_mag_max = 0.0
    _gen_buffer_len_min = math.huge
    _gen_buffer_len_max = -math.huge
    _gen_steps_seen = 0
end

local function reset_ring(ring_size)
    -- keep the slot tables themselves (they will be overwritten in
    -- place) but invalidate their step tags so stale data can never
    -- be served into a new generation
    _ring_size = ring_size
    for s = 0, ring_size - 1 do
        _ring_step[s] = -1
        if _ring[s] == nil then _ring[s] = {} end
    end
    _stored_count = 0
end

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    local p = params or {}
    local d_steps    = math.floor((p.delay_steps) or 3)
    local f_gain     = clamp((p.feedback_gain) or 0.15, 0.0, 0.5)
    local deg        = clamp((p.bbd_degradation) or 0.6, 0.0, 1.0)
    local regen      = (p.regeneration ~= false)
    local steps_per_gen = math.floor((p.steps_per_generation) or 24)
    local debug_log  = (p.debug_log ~= false)
    local summary_log = (p.summary_log ~= false)

    local n = Oc * T
    local ring_size = d_steps + 1

    local reset_reason = nil

    -- primary reset signal: latent size changed = definitely a new generation
    if n ~= _last_n then
        reset_reason = "n_changed"
    -- secondary reset signal: hit the expected step count for a full
    -- generation (catches back-to-back jobs with identical latent size)
    elseif _internal_step >= steps_per_gen then
        reset_reason = "step_count_wraparound"
    end

    -- ring size can also change mid-session if the user moves the
    -- delay_steps slider between generations
    if ring_size ~= _ring_size then
        if reset_reason == nil then reset_reason = "ring_size_changed" end
    end

    if reset_reason ~= nil then
        if summary_log then
            print_summary(reset_reason)
        end
        reset_ring(ring_size)
        _last_n = n
        _internal_step = -1
        reset_gen_stats()
        if debug_log then
            print(string.format(
                "[MD Bucket Brigade] RESET reason=%s (n=%d, steps_per_generation=%d, ring_size=%d)",
                reset_reason, n, steps_per_gen, ring_size))
        end
    end

    -- self-tracking counter, independent of (unreliable) step_index
    _internal_step = _internal_step + 1
    local cur_step = _internal_step

    if debug_log and cur_step == 0 then
        local cond_sum, uncond_sum = 0.0, 0.0
        for i = 0, n - 1 do
            cond_sum = cond_sum + pred_cond[i]
            uncond_sum = uncond_sum + pred_uncond[i]
        end
        print(string.format(
            "[MD Bucket Brigade] FINGERPRINT step=0 n=%d guidance_scale=%.3f cond_sum=%.6f uncond_sum=%.6f cond[0]=%.6f cond[n-1]=%.6f uncond[0]=%.6f uncond[n-1]=%.6f",
            n, guidance_scale, cond_sum, uncond_sum, pred_cond[0], pred_cond[n-1], pred_uncond[0], pred_uncond[n-1]))
    end

    apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)

    -- with regeneration OFF, capture the pre-feedback (clean) result
    -- into this step's ring slot BEFORE injecting the echo
    local store_slot = cur_step % ring_size
    if not regen then
        local slot = _ring[store_slot]
        for i = 0, n - 1 do
            slot[i] = result[i]
        end
    end

    local echo_status = "OFF (feedback_gain=0)"
    local echo_mag = 0.0

    if f_gain > 0.0 then
        local target_step = cur_step - d_steps
        local target_slot = (target_step >= 0) and (target_step % ring_size) or -1
        if target_step >= 0 and _ring_step[target_slot] == target_step then
            local echo = _ring[target_slot]
            local degraded_echo = apply_bbd_degradation(echo, n, deg)

            local mag_sum = 0.0
            for i = 0, n - 1 do
                local injected = degraded_echo[i] * f_gain
                -- NaN/inf guard: drop non-finite echo samples
                if injected ~= injected or injected > 1e30 or injected < -1e30 then
                    injected = 0.0
                end
                result[i] = result[i] + injected
                mag_sum = mag_sum + math.abs(injected)
            end
            echo_mag = mag_sum / n
            echo_status = string.format("HIT (from step %d)", target_step)
            _gen_hits = _gen_hits + 1
            _gen_mag_sum = _gen_mag_sum + echo_mag
            if echo_mag > _gen_mag_max then _gen_mag_max = echo_mag end
        elseif target_step < 0 then
            echo_status = string.format("MISS (target_step=%d, warming up)", target_step)
            _gen_misses_warmup = _gen_misses_warmup + 1
        else
            echo_status = string.format(
                "MISS (target_step=%d not in buffer -- unexpected)", target_step)
            _gen_misses_unexpected = _gen_misses_unexpected + 1
        end
    end

    -- with regeneration ON (v1.x parity), capture the post-feedback
    -- result -- echoes will contain earlier echoes
    if regen then
        local slot = _ring[store_slot]
        for i = 0, n - 1 do
            slot[i] = result[i]
        end
    end
    _ring_step[store_slot] = cur_step
    if _stored_count < ring_size then _stored_count = _stored_count + 1 end
    local blen = _stored_count

    _gen_steps_seen = _gen_steps_seen + 1
    if blen < _gen_buffer_len_min then _gen_buffer_len_min = blen end
    if blen > _gen_buffer_len_max then _gen_buffer_len_max = blen end

    if telemetry ~= nil then
        telemetry.bbd_buffer_len = blen
        telemetry.bbd_echo_status = echo_status
        telemetry.bbd_echo_mag = echo_mag
        telemetry.bbd_internal_step = cur_step
        telemetry.bbd_regeneration = regen
    end

    if debug_log then
        print(string.format(
            "[MD Bucket Brigade] internal_step=%d raw_step_index=%s buffer_len=%d regen=%s echo=%s echo_mag=%.6f",
            cur_step, tostring(step_index), blen, tostring(regen), echo_status, echo_mag))
    end
end
