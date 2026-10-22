-- ============================================================================
-- MD APG Guidance v4.0 — Adaptive Projected Guidance
-- MDMAchine | A&E Concepts © 2026
-- Optimized: Zero-Allocation Buffers + Strict Boundary State Reset
--
-- v4.0: Added norm_threshold to guide() signature for HOT-Step contract
--       consistency. APG doesn't call apg() (it IS APG) but accepts the
--       arg so the engine doesn't choke on arity mismatch.
-- ============================================================================

guidance = {
    name        = "md_apg V4",
    display     = "MD APG (Adaptive Projected Guidance) V4",
    description = "APG: decomposes cond-uncond diff. Momentum buffer + norm threshold. Memory-optimized.",
    params      = {
        { key = "eta", type = "slider", label = "Eta (Parallel Bleed)", default = 0.0, min = 0.0, max = 1.0, step = 0.05, hint = "0=pure orthogonal. 0.3=soft parallel bleed." },
        { key = "momentum", type = "slider", label = "Momentum", default = -0.5, min = -1.0, max = 0.0, step = 0.05, hint = "EMA momentum across steps. Prevents drift." },
        { key = "norm_threshold", type = "slider", label = "Norm Threshold", default = 0.0, min = 0.0, max = 10.0, step = 0.25, hint = "L2 norm clip on diff. 0=off." },
        { key = "warmup_steps", type = "slider", label = "Warmup Steps", default = 0, min = 0, max = 8, step = 1, hint = "Linearly ramp guidance from 1→scale." },
    },
}

local EPSILON = 1e-8
local _last_n = 0
local _diff_buf = {}
local _momentum_avg = {}

local function dot_product(a, b, n)
    local s = 0.0
    for i = 0, n - 1 do s = s + a[i] * b[i] end
    return s
end

local function norm_sq(a, n) return dot_product(a, a, n) end
local function l2_norm(a, n) return math.sqrt(norm_sq(a, n)) end

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    local n   = Oc * T
    local eta = (params and params.eta)            or 0.0
    local mom = (params and params.momentum)       or -0.5
    local nth = (params and params.norm_threshold) or 0.0
    local wu  = math.floor((params and params.warmup_steps) or 0)
    
    -- Catch engine step via global injection
    local cur_step = step_index or step_idx or 0

    -- Strict reset on new generation
    if n ~= _last_n or cur_step == 0 then
        _last_n = n
        for i = 0, n - 1 do
            _momentum_avg[i] = 0.0
            _diff_buf[i] = 0.0
        end
    end

    local eff_scale = guidance_scale
    if wu > 0 and cur_step < wu then
        eff_scale = 1.0 + (guidance_scale - 1.0) * (cur_step / wu)
    end

    for i = 0, n - 1 do _diff_buf[i] = pred_cond[i] - pred_uncond[i] end

    if nth > 0.0 then
        local dn = l2_norm(_diff_buf, n)
        if dn > nth then
            local scale = nth / (dn + EPSILON)
            for i = 0, n - 1 do _diff_buf[i] = _diff_buf[i] * scale end
        end
    end

    if mom ~= 0.0 then
        if cur_step == 0 then
            for i = 0, n - 1 do _momentum_avg[i] = _diff_buf[i] end
        else
            for i = 0, n - 1 do
                _momentum_avg[i] = _diff_buf[i] + mom * _momentum_avg[i]
                _diff_buf[i] = _momentum_avg[i]
            end
        end
    end

    local dot_dc   = dot_product(_diff_buf, pred_cond, n)
    local dot_cc   = norm_sq(pred_cond, n) + EPSILON
    local proj_scale = dot_dc / dot_cc

    local gs_minus1 = eff_scale - 1.0
    for i = 0, n - 1 do
        local parallel_i  = proj_scale * pred_cond[i]
        local orthogonal_i = _diff_buf[i] - parallel_i
        local update_i    = orthogonal_i + eta * parallel_i
        result[i] = pred_cond[i] + gs_minus1 * update_i
    end
end