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

-- MD Timestep Lie (YuE2) v1.0
-- MDMAchine | A&E Concepts (C) 2026
--
-- Experimental full-loop solver for the YuE2 NAR (refiner). It integrates with
-- the true schedule (plain Euler, x -= v * (t - t_next)) but asks the model for
-- its velocity at a falsified time t' = clamp(t + lie * w(t)).
--
-- Flow convention (engine/src/yue2/yue2-nar-graph.h, owns_loop branch and the
-- plan_w / CADS comments): t runs 1 = pure noise -> 0 = clean. The schedule
-- table holds N descending entries with no trailing 0; the step after the last
-- entry goes to t = 0. model_fn(x, t) writes velocity into vt_buf; the engine
-- applies the model's logit(t) conditioning (clamped to +-20) inside eval().
-- So lie > 0 tells the model the latent is noisier than it is (it invents more
-- detail), lie < 0 tells it the latent is cleaner (conservative refinement).
-- t' is clamped to [T_MIN, T_MAX].
--
-- lie_curve:
--   constant  lie applies at every step.
--   early     full lie for t >= lie_t2, none for t <= lie_t1, linear between.
--   late      full lie for t <= lie_t1, none for t >= lie_t2, linear between.
--   sections  per-frame lie from section_lies (e.g. "chorus=0.25,verse=-0.1"),
--             matched by name prefix against model_context.yue2.sections;
--             frames outside a listed section use `lie`. model_fn takes a
--             scalar t, so per-frame lies are done with one model_fn call per
--             distinct lie value in the chunk, picking each frame's velocity
--             from the call that matches its lie (a hard section mask).
--             NFE per step = number of distinct lie values in the chunk
--             (usually 2). Falls back to constant when the plan table or the
--             frame layout is missing.
-- Latent layout: flat time-major, n = frames * channels (frame f, channel c at
-- f * channels + c), same as md_wasserstein_yue2.

solver = {
    name        = "md_tlie_yue2",
    display     = "MD Timestep Lie (YuE2)",
    description = "YuE2 NAR: Euler on the true schedule, model queried at a falsified t. + lie = more invented detail, - lie = conservative.",
    version     = "1.0",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = false,
    owns_loop   = true,
    params      = {
        { key = "lie", type = "slider", label = "Lie (t offset)", default = 0.15, min = -0.5, max = 0.5, step = 0.01,
          hint = "Added to t when querying the model. + = noisier (more detail), - = cleaner (conservative)." },
        { key = "lie_curve", type = "select", label = "Lie Curve", default = "constant",
          options = {
              { value = "constant", label = "Constant" },
              { value = "early",    label = "Early (noisy end)" },
              { value = "late",     label = "Late (clean end)" },
              { value = "sections", label = "Per section" },
          } },
        { key = "lie_t1", type = "slider", label = "Window t1", default = 0.3, min = 0, max = 1, step = 0.01 },
        { key = "lie_t2", type = "slider", label = "Window t2", default = 0.7, min = 0, max = 1, step = 0.01 },
        { key = "section_lies", type = "text", label = "Section Lies", default = "",
          hint = "sections mode only, e.g. chorus=0.25,verse=-0.1. Unlisted frames use Lie." },
        { key = "debug_log", type = "toggle", label = "Debug Log", default = false },
    },
}

local T_MIN, T_MAX = 1e-4, 0.9999

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

-- Queried t for a lie; a zero lie keeps the true t (no clamp), so lie = 0 is plain Euler.
local function lied(t, l)
    if l == 0 then return t end
    return clamp(t + l, T_MIN, T_MAX)
end

local function curve_weight(curve, t, t1, t2)
    if t2 < t1 then t1, t2 = t2, t1 end
    local span = t2 - t1
    if curve == "early" then
        if t >= t2 then return 1 end
        if t <= t1 then return 0 end
        return (t - t1) / span
    elseif curve == "late" then
        if t <= t1 then return 1 end
        if t >= t2 then return 0 end
        return (t2 - t) / span
    end
    return 1
end

local function parse_section_lies(s)
    local out = {}
    for item in tostring(s or ""):gmatch("[^,;]+") do
        local k, v = item:match("^%s*([^=]-)%s*=%s*([-+%d%.eE]+)%s*$")
        local val = tonumber(v)
        if k and k ~= "" and val then
            out[#out + 1] = { name = k:lower(), lie = clamp(val, -0.5, 0.5) }
        end
    end
    return out
end

-- Per-frame lie table for this chunk, or nil (falls back to constant).
local function frame_lies(n, base_lie)
    local entries = parse_section_lies(setting("section_lies", ""))
    if #entries == 0 then return nil end
    local y = model_context and model_context.yue2
    if not y or not y.chunk or not y.sections then return nil end
    local fps = y.frame_rate or 25
    local frames = (y.chunk.end_frame or 0) - (y.chunk.start_frame or 0)
    if frames <= 0 or fps <= 0 or n % frames ~= 0 then return nil end
    local lies, distinct = {}, {}
    for f = 0, frames - 1 do
        local ts = (y.chunk.start_frame + f) / fps
        local l = base_lie
        for _, s in ipairs(y.sections) do
            if s.start_s and s.end_s and ts >= s.start_s and ts < s.end_s then
                local nm = tostring(s.name or ""):lower()
                for _, e in ipairs(entries) do
                    if nm:sub(1, #e.name) == e.name then l = e.lie break end
                end
                break
            end
        end
        lies[f] = l
        distinct[l] = true
    end
    local list = {}
    for l in pairs(distinct) do list[#list + 1] = l end
    table.sort(list)
    return lies, list, math.floor(n / frames), frames
end

function sample(xt, vt_buf, schedule, n, model_fn)
    local lie   = clamp(num("lie", 0.15), -0.5, 0.5)
    local curve = tostring(setting("lie_curve", "constant"))
    local t1    = clamp(num("lie_t1", 0.3), 0, 1)
    local t2    = clamp(num("lie_t2", 0.7), 0, 1)
    local dbgv  = setting("debug_log", false)
    local dbg   = dbgv == true or dbgv == "true"
    local ns    = #schedule

    local lies, list, ch, frames
    if curve == "sections" then
        lies, list, ch, frames = frame_lies(n, lie)
        if not lies then curve = "constant" end
        if dbg then
            print(string.format("[md_tlie_yue2] sections active=%s lies=%s", tostring(lies ~= nil),
                list and table.concat(list, ",") or "-"))
        end
    end

    local v = {}
    for i = 1, ns do
        local t = schedule[i]
        local t_next = (i < ns) and schedule[i + 1] or 0.0
        if lies then
            for _, l in ipairs(list) do
                model_fn(xt, lied(t, l))
                for f = 0, frames - 1 do
                    if lies[f] == l then
                        local b = f * ch
                        for c = 0, ch - 1 do v[b + c] = vt_buf[b + c] end
                    end
                end
            end
            for k = 0, n - 1 do vt_buf[k] = v[k] end
        else
            local tq = lied(t, lie * curve_weight(curve, t, t1, t2))
            model_fn(xt, tq)
            if dbg then print(string.format("[md_tlie_yue2] step %d t=%.4f t'=%.4f", i - 1, t, tq)) end
        end
        local dt = t - t_next
        for k = 0, n - 1 do xt[k] = xt[k] - vt_buf[k] * dt end
        if on_step and on_step(i - 1, t, t_next) then return end
    end
end
