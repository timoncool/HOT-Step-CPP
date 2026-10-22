-- ============================================================================
-- MD Custom Noise Guidance v1.2 — Multi-Algorithm Structured Noise Injection
-- MDMAchine | A&E Concepts © 2026
-- Optimized: Zero-allocation algorithms, strict boundary resetting.
-- ============================================================================

guidance = {
    name        = "md_custom_noise V3",
    display     = "MD Custom Noise (Structured Injection) V3",
    description = "10-algorithm structured noise injection via post_step. Zero-allocation optimized.",
    params      = {
        {
            key     = "algorithm",
            type    = "select",
            label   = "Algorithm",
            default = "gaussian",
            options = {
                { value = "gaussian",     label = "Gaussian" },
                { value = "uniform",      label = "Uniform" },
                { value = "laplacian",    label = "Laplacian" },
                { value = "student_t",    label = "Student-t" },
                { value = "perlin",       label = "Perlin" },
                { value = "collatz",      label = "Collatz (Orbit)" },
                { value = "pink",         label = "Pink (1/f)" },
                { value = "pyramid",      label = "Pyramid" },
                { value = "power_normal", label = "Power Normal" },
                { value = "fourier_band", label = "Fourier Bandpass" },
            },
            hint = "Noise generation algorithm. Creates different latent exploration patterns.",
        },
        { key = "injection_strength", type = "slider", label = "Strength", default = 0.05, min = 0.0, max = 0.5, step = 0.005, hint = "Noise amplitude (scaled by sigma)." },
        { key = "scale", type = "slider", label = "Scale", default = 1.0, min = 0.1, max = 5.0, step = 0.1, hint = "Algorithm-specific control." },
        { key = "seed", type = "slider", label = "Seed", default = 42, min = 0, max = 999999, step = 1, hint = "Deterministic seed for reproducible noise." },
    },
}

local EPSILON = 1e-8
local _last_n = 0
local _noise_buf = {}
local _scratch_buf = {}

local function make_rng(seed)
    local state = math.floor(seed) % 2147483647
    if state <= 0 then state = state + 2147483646 end
    return function()
        state = (state * 1664525 + 1013904223) % 2147483648
        return state / 2147483648.0
    end
end

local function normal_sample(rng)
    local u1 = math.max(rng(), EPSILON)
    local u2 = rng()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
end

local function normalize_std_inplace(arr, n)
    local mean = 0.0
    for i = 0, n - 1 do mean = mean + arr[i] end
    mean = mean / n
    local var = 0.0
    for i = 0, n - 1 do
        local d = arr[i] - mean
        var = var + d * d
    end
    local std = math.sqrt(var / n + EPSILON)
    local inv_std = 1.0 / std
    for i = 0, n - 1 do arr[i] = (arr[i] - mean) * inv_std end
end

local function gen_gaussian(n, rng, _scale, out_buf)
    for i = 0, n - 1 do out_buf[i] = normal_sample(rng) end
end

local function gen_uniform(n, rng, _scale, out_buf)
    for i = 0, n - 1 do out_buf[i] = (rng() - 0.5) * 3.46 end 
end

local function gen_laplacian(n, rng, scale, out_buf)
    for i = 0, n - 1 do
        local u = rng() - 0.5
        local sign = (u >= 0) and 1.0 or -1.0
        out_buf[i] = -sign * math.log(1.0 - 2.0 * math.abs(u) + EPSILON) * scale
    end
end

local function gen_student_t(n, rng, df, out_buf)
    df = math.max(2.0, df)
    local df_int = math.floor(df)
    for i = 0, n - 1 do
        local z = normal_sample(rng)
        local chi = 0.0
        for _ = 1, df_int do
            local g = normal_sample(rng)
            chi = chi + g * g
        end
        out_buf[i] = z / math.sqrt(chi / df + EPSILON) * 0.7
    end
end

local function gen_perlin(n, rng, scale, out_buf)
    local grid_n = math.max(2, math.floor(n / (16 * scale)))
    for i = 0, grid_n - 1 do _scratch_buf[i] = normal_sample(rng) end
    local inv_n1 = 1.0 / (n - 1)
    for i = 0, n - 1 do
        local t = (i * inv_n1) * (grid_n - 1)
        local lo = math.floor(t)
        local hi = math.min(lo + 1, grid_n - 1)
        local frac = t - lo
        local s = frac * frac * (3.0 - 2.0 * frac)
        out_buf[i] = _scratch_buf[lo] * (1.0 - s) + _scratch_buf[hi] * s
    end
    normalize_std_inplace(out_buf, n)
end

local function gen_collatz(n, rng, scale, out_buf)
    local offset = math.floor(rng() * 1000)
    for i = 0, n - 1 do
        local val = math.max(1, math.floor(i * 0.01 * scale + offset))
        local steps = 0
        for _ = 1, 40 do
            if val <= 1 then break end
            if val % 2 == 0 then val = math.floor(val / 2)
            else val = 3 * val + 1 end
            steps = steps + 1
        end
        out_buf[i] = steps
    end
    normalize_std_inplace(out_buf, n)
end

local function gen_pink(n, rng, _scale, out_buf)
    local num_octaves = math.min(8, math.floor(math.log(n) / math.log(2)))
    for i = 0, n - 1 do out_buf[i] = 0.0 end
    local total_weight = 0.0
    local inv_n1 = 1.0 / (n - 1)

    for oct = 0, num_octaves - 1 do
        local scale_f = 2 ^ oct
        local small_n = math.max(2, math.floor(n / scale_f))
        for i = 0, small_n - 1 do _scratch_buf[i] = normal_sample(rng) end

        local weight = 1.0 / (oct + 1)
        total_weight = total_weight + weight

        for i = 0, n - 1 do
            local t = (i * inv_n1) * (small_n - 1)
            local lo = math.floor(t)
            local hi = math.min(lo + 1, small_n - 1)
            local frac = t - lo
            out_buf[i] = out_buf[i] + (_scratch_buf[lo] * (1.0 - frac) + _scratch_buf[hi] * frac) * weight
        end
    end
    if total_weight > 0 then
        local inv_total = 1.0 / total_weight
        for i = 0, n - 1 do out_buf[i] = out_buf[i] * inv_total end
    end
    normalize_std_inplace(out_buf, n)
end

local function gen_pyramid(n, rng, _scale, out_buf)
    local half_n = math.max(2, math.floor(n / 2))
    for i = 0, half_n - 1 do _scratch_buf[i] = normal_sample(rng) end

    local inv_n1 = 1.0 / (n - 1)
    for i = 0, n - 1 do
        local t = (i * inv_n1) * (half_n - 1)
        local lo = math.floor(t)
        local hi = math.min(lo + 1, half_n - 1)
        local frac = t - lo
        local upsampled = _scratch_buf[lo] * (1.0 - frac) + _scratch_buf[hi] * frac
        out_buf[i] = upsampled * 0.6 + normal_sample(rng) * 0.4
    end
end

local function gen_power_normal(n, rng, scale, out_buf)
    local exp = math.max(0.1, scale)
    for i = 0, n - 1 do
        local base = normal_sample(rng)
        local sign = (base >= 0) and 1.0 or -1.0
        out_buf[i] = sign * (math.abs(base) ^ exp)
    end
end

local function gen_fourier_band(n, rng, scale, out_buf)
    for i = 0, n - 1 do out_buf[i] = 0.0 end
    local center_freq = scale * 0.1 
    local bandwidth = 0.05
    local num_harmonics = math.min(16, math.max(4, math.floor(n / 8)))
    local inv_sqrt_h = 1.0 / math.sqrt(num_harmonics)

    for h = 1, num_harmonics do
        local freq = center_freq + (h - num_harmonics/2) * bandwidth / num_harmonics
        if freq > 0 then
            local amp = normal_sample(rng) * inv_sqrt_h
            local phase = rng() * 2.0 * math.pi
            local f_mult = 2.0 * math.pi * freq
            for i = 0, n - 1 do
                out_buf[i] = out_buf[i] + amp * math.sin(f_mult * i + phase)
            end
        end
    end
    normalize_std_inplace(out_buf, n)
end

local GENERATORS = {
    gaussian     = gen_gaussian,
    uniform      = gen_uniform,
    laplacian    = gen_laplacian,
    student_t    = gen_student_t,
    perlin       = gen_perlin,
    collatz      = gen_collatz,
    pink         = gen_pink,
    pyramid      = gen_pyramid,
    power_normal = gen_power_normal,
    fourier_band = gen_fourier_band,
}

-- result arrives holding stale velocity, so guide() must write the CFG combine.
-- This plugin only injects noise in post_step; the base combine is native APG.
function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
end

function post_step(xt, t, n, eval_cond, eval_uncond, vt_cond, vt_uncond)
    local cur_step = step_index or step_idx or 0

    if n ~= _last_n or cur_step == 0 then
        _last_n = n
        for i = 0, n - 1 do
            _noise_buf[i] = 0.0
            _scratch_buf[i] = 0.0
        end
    end

    local algo   = (params and params.algorithm)           or "gaussian"
    local str    = (params and params.injection_strength)  or 0.05
    local scale  = (params and params.scale)               or 1.0
    local seed   = math.floor((params and params.seed)     or 42)

    if str <= 0.0 then return end
    local eff_str = str * (t or 0.5)
    if eff_str < 1e-6 then return end

    local rng = make_rng(seed + cur_step * 7919)
    local gen_fn = GENERATORS[algo] or gen_gaussian
    
    gen_fn(n, rng, scale, _noise_buf)

    for i = 0, n - 1 do
        xt[i] = xt[i] + _noise_buf[i] * eff_str
    end
end