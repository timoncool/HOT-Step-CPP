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

-- MD Plan Overdrive (YuE2) v1.0
-- MDMAchine | A&E Concepts (C) 2026
--
-- Experimental YuE2 NAR scheduler. Timesteps are the stock YuE2 grid,
-- t_i = 1 - i/N (t: 1 = noise, 0 = clean), identical to the engine's
-- no-scheduler path. It also sets the global `plan_strength` (N numbers),
-- which the engine reads once per chunk as the per-step plan weight w that
-- scales the AR-prefix V (plan conditioning). The engine clamps each w to
-- [0, 4] (yue2_plan_w_fill in engine/src/yue2/yue2-nar-graph.h), so values
-- above 1 are honoured up to 4; w = 1 everywhere is bit-identical to stock.
-- A table whose length is not N is ignored by the engine.
--
-- shape (q = i/(N-1): 0 at the noisy first step, 1 at the last):
--   flat   w = drive at every step
--   early  w = drive -> base (cosine fall)
--   late   w = base -> drive (cosine rise)
--   pulse  w = drive for the first half of each pulse_period steps, base after
-- Non-YuE2 models get the uniform grid; plan_strength is only read by YuE2.

scheduler = {
    name        = "md_plan_overdrive_yue2",
    display     = "MD Plan Overdrive (YuE2)",
    description = "YuE2 only: stock uniform timesteps plus a per-step plan strength curve (w up to 3; engine clamp 0..4).",
    version     = "1.0",
    params      = {
        { key = "drive", type = "slider", label = "Drive (peak plan strength)", default = 1.5, min = 0, max = 3, step = 0.05 },
        { key = "shape", type = "select", label = "Shape", default = "late",
          options = {
              { value = "flat",  label = "Flat"  },
              { value = "late",  label = "Late (rise)" },
              { value = "early", label = "Early (fall)" },
              { value = "pulse", label = "Pulse" },
          } },
        { key = "base", type = "slider", label = "Base Strength", default = 1.0, min = 0, max = 3, step = 0.05 },
        { key = "pulse_period", type = "slider", label = "Pulse Period (steps)", default = 4, min = 2, max = 32, step = 1 },
        { key = "debug_log", type = "toggle", label = "Debug Log", default = false },
    },
}

local function num(p, key, fallback)
    local v = tonumber(p[key])
    if v == nil or v ~= v then return fallback end
    return v
end

function schedule(output, num_steps, shift)
    local p = params or {}
    local drive  = math.max(0, math.min(3, num(p, "drive", 1.5)))
    local base   = math.max(0, math.min(3, num(p, "base", 1.0)))
    local period = math.max(2, math.floor(num(p, "pulse_period", 4)))
    local shape  = tostring(p.shape or "late")
    local dbg    = p.debug_log == true or p.debug_log == "true"

    for i = 0, num_steps - 1 do
        output[i] = 1.0 - i / num_steps
    end

    local w = {}
    for i = 0, num_steps - 1 do
        local q = num_steps > 1 and i / (num_steps - 1) or 0
        local c = 0.5 * (1 - math.cos(math.pi * q))
        local v
        if shape == "flat" then
            v = drive
        elseif shape == "early" then
            v = drive + (base - drive) * c
        elseif shape == "pulse" then
            v = ((i % period) < period / 2) and drive or base
        else
            v = base + (drive - base) * c
        end
        w[i + 1] = v
    end
    plan_strength = w

    if dbg then
        local parts = {}
        for i = 1, #w do parts[#parts + 1] = string.format("%.2f", w[i]) end
        print(string.format("[md_plan_overdrive_yue2] shape=%s drive=%.2f base=%.2f w=%s",
                            shape, drive, base, table.concat(parts, ",")))
    end
end
