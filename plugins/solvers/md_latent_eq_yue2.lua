-- ============================================================================
-- SPDX-License-Identifier: GPL-3.0-or-later
-- Copyright (C) 2026 Alexander Allan (MDMAchine) | A&E Concepts
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

-- MD Latent Channel EQ (YuE2) v1.0
-- MDMAchine | A&E Concepts (C) 2026
--
-- Experimental full-loop solver for the YuE2 NAR (refiner). Plain Euler on the
-- true schedule (x -= v * (t - t_next), t 1 = noise -> 0 = clean), but the
-- velocity is scaled per latent channel group before each update. The 64
-- latent channels are split into 8 groups of 8 (group g = channels 8g..8g+7);
-- group = -1 scales all channels. `gains` overrides with several groups, e.g.
-- "0=1.2,3=0.7". Scaling only applies while eq_t1 <= t <= eq_t2 (solver t).
-- section_only limits scaling to frames inside sections whose name starts with
-- the given text (model_context.yue2.sections); empty = all frames.
-- Latent layout: flat time-major, n = frames * channels (frame f, channel c at
-- f * channels + c), same as md_wasserstein_yue2.

solver = {
    name        = "md_latent_eq_yue2",
    display     = "MD Latent Channel EQ (YuE2)",
    description = "YuE2 NAR: Euler with per-channel-group velocity gain (8 groups of 8 latent channels).",
    version     = "1.0",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = false,
    owns_loop   = true,
    params      = {
        { key = "group", type = "slider", label = "Channel Group", default = 0, min = -1, max = 7, step = 1,
          hint = "0..7 = channels 8g..8g+7, -1 = all channels." },
        { key = "gain", type = "slider", label = "Gain", default = 1.5, min = 0, max = 3, step = 0.01 },
        { key = "eq_t1", type = "slider", label = "Window t1", default = 0.0, min = 0, max = 1, step = 0.01 },
        { key = "eq_t2", type = "slider", label = "Window t2", default = 1.0, min = 0, max = 1, step = 0.01 },
        { key = "gains", type = "text", label = "Group Gains", default = "",
          hint = "Overrides Group/Gain, e.g. 0=1.2,3=0.7" },
        { key = "section_only", type = "text", label = "Section Only", default = "",
          hint = "Only frames in sections starting with this name (e.g. chorus). Empty = all." },
        { key = "debug_log", type = "toggle", label = "Debug Log", default = false },
    },
}

local CH_DEFAULT, GROUPS = 64, 8

local function setting(key, fallback)
    if params and params[key] ~= nil then return params[key] end
    return fallback
end

local function num(key, fallback)
    local v = tonumber(setting(key, fallback))
    if v == nil or v ~= v then return fallback end
    return v
end

local function clamp(x, a, b)
    if x < a then return a elseif x > b then return b end
    return x
end

-- Per-channel gain table (index 0..ch-1), or nil when all gains are 1.
local function channel_gains(ch)
    local gs = {}
    local any = false
    local per = math.max(1, math.floor(ch / GROUPS))
    local function set_group(g, gain)
        if gain ~= 1 then any = true end
        if g < 0 then
            for c = 0, ch - 1 do gs[c] = gain end
        else
            for c = g * per, math.min(ch, (g + 1) * per) - 1 do gs[c] = gain end
        end
    end
    local txt = tostring(setting("gains", "") or "")
    local parsed = false
    for item in txt:gmatch("[^,;]+") do
        local k, v = item:match("^%s*([-%d]+)%s*=%s*([-+%d%.eE]+)%s*$")
        local g, val = tonumber(k), tonumber(v)
        if g and val and g >= -1 and g < GROUPS then
            set_group(math.floor(g), clamp(val, 0, 3))
            parsed = true
        end
    end
    if not parsed then
        set_group(math.floor(clamp(num("group", 0), -1, GROUPS - 1)), clamp(num("gain", 1.5), 0, 3))
    end
    if not any then return nil end
    for c = 0, ch - 1 do if gs[c] == nil then gs[c] = 1 end end
    return gs
end

-- Frame mask (f -> true) for section_only, nil = all frames.
local function frame_mask(n)
    local want = tostring(setting("section_only", "") or ""):lower():gsub("^%s+", ""):gsub("%s+$", "")
    local y = model_context and model_context.yue2
    local frames, ch
    if y and y.chunk and y.chunk.end_frame and y.chunk.start_frame then
        frames = y.chunk.end_frame - y.chunk.start_frame
        if frames > 0 and n % frames == 0 then ch = math.floor(n / frames) end
    end
    if not ch then ch = CH_DEFAULT; frames = math.floor(n / ch) end
    if want == "" then return nil, frames, ch end
    if not (y and y.sections and y.chunk) then return {}, frames, ch end
    local fps = y.frame_rate or 25
    local mask = {}
    for f = 0, frames - 1 do
        local ts = (y.chunk.start_frame + f) / fps
        for _, s in ipairs(y.sections) do
            if s.start_s and s.end_s and ts >= s.start_s and ts < s.end_s then
                local nm = tostring(s.name or ""):lower()
                if nm:sub(1, #want) == want then mask[f] = true end
                break
            end
        end
    end
    return mask, frames, ch
end

function sample(xt, vt_buf, schedule, n, model_fn)
    local t1 = clamp(num("eq_t1", 0.0), 0, 1)
    local t2 = clamp(num("eq_t2", 1.0), 0, 1)
    if t2 < t1 then t1, t2 = t2, t1 end
    local dbgv = setting("debug_log", false)
    local dbg = dbgv == true or dbgv == "true"
    local mask, frames, ch = frame_mask(n)
    local gs = channel_gains(ch)
    if dbg then
        print(string.format("[md_latent_eq_yue2] frames=%d ch=%d gains=%s mask=%s",
            frames, ch, tostring(gs ~= nil), mask and "section" or "all"))
    end
    local ns = #schedule
    for i = 1, ns do
        local t = schedule[i]
        local t_next = (i < ns) and schedule[i + 1] or 0.0
        model_fn(xt, t)
        local dt = t - t_next
        if gs and t >= t1 and t <= t2 then
            for f = 0, frames - 1 do
                local b = f * ch
                if mask == nil or mask[f] then
                    for c = 0, ch - 1 do xt[b + c] = xt[b + c] - vt_buf[b + c] * gs[c] * dt end
                else
                    for c = 0, ch - 1 do xt[b + c] = xt[b + c] - vt_buf[b + c] * dt end
                end
            end
        else
            for k = 0, n - 1 do xt[k] = xt[k] - vt_buf[k] * dt end
        end
        if on_step and on_step(i - 1, t, t_next) then return end
    end
end
