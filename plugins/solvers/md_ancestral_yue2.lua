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

-- Ancestral Euler for rectified flow (x_t = (1-t) x0 + t eps), tuned for
-- YuE2 NAR. Each step splits the state into a clean estimate
-- x0 = x - t v and a noise estimate eps = x + (1-t) v, then rebuilds the
-- state at the next time from x0 and a noise mix:
--   eps' = sqrt(1 - c^2) eps + c z,  z ~ N(0, 1),  c = churn
-- churn = 0 reduces exactly to plain Euler and draws no random numbers.
-- z comes from math.random via Box-Muller; the engine seeds the Lua RNG from
-- the job seed at step 0, so a fixed seed gives a fixed render.

solver = {
    name = "md_ancestral_yue2",
    display = "MD Ancestral Euler (YuE2)",
    description = "One-evaluation Euler step with per-step noise re-injection (churn). churn 0 = plain Euler.",
    nfe = 1,
    order = 1,
    needs_model = false,
    stateful = false,
    params = {
        { key = "churn", type = "slider", label = "Churn", default = 0.0, min = 0.0, max = 1.0, step = 0.01 },
    },
}

local spare = nil

local function gauss()
    if spare then
        local s = spare
        spare = nil
        return s
    end
    local u1 = math.random()
    if u1 < 1e-12 then u1 = 1e-12 end
    local u2 = math.random()
    local r = math.sqrt(-2.0 * math.log(u1))
    local a = 2.0 * math.pi * u2
    spare = r * math.sin(a)
    return r * math.cos(a)
end

function step(xt, vt, t_curr, t_prev, n)
    if n <= 0 then return end
    local churn = 0.0
    if params and params.churn ~= nil then churn = tonumber(params.churn) or 0.0 end
    if churn < 0 then churn = 0 elseif churn > 1 then churn = 1 end

    local dt = t_prev - t_curr
    if churn == 0.0 or t_prev <= 0.0 then
        for i = 0, n - 1 do xt[i] = xt[i] + dt * vt[i] end
        return
    end

    -- Fresh stream per generation: drop any cached Box-Muller spare at step 0.
    if t_curr >= 0.999999 then spare = nil end
    local keep = math.sqrt(1.0 - churn * churn)
    for i = 0, n - 1 do
        local x, v = xt[i], vt[i]
        local x0 = x - t_curr * v
        local eps = x + (1.0 - t_curr) * v
        local e2 = keep * eps + churn * gauss()
        local y = (1.0 - t_prev) * x0 + t_prev * e2
        if y ~= y or y == math.huge or y == -math.huge then y = x + dt * v end
        xt[i] = y
    end
end
