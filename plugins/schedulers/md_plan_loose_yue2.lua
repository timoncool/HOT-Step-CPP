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

-- MD Plan-Loose Scheduler (YuE2) v1.0
-- MDMAchine | A&E Concepts (C) 2026
--
-- Demo of the plan_strength global: the YuE2 NAR engine reads an optional
-- array `plan_strength` (one number per step) after schedule() returns and
-- scales the plan conditioning (the AR-prefix V the NAR decoder attends to)
-- by it at each step. Timesteps are the stock uniform grid 1 - i/N.
-- The plan starts loose (w = 1 - loose) and ramps linearly to full strength
-- (w = 1) by the fraction `lock_at` of the steps, then stays at 1.
-- Other models ignore plan_strength.

scheduler = {
    name        = "md_plan_loose_yue2",
    display     = "MD Plan-Loose (YuE2)",
    description = "YuE2 only: uniform steps, plan conditioning loosened early and locked in later.",
    params      = {
        {
            key = "loose", type = "slider", label = "Early Looseness",
            default = 0.5, min = 0.0, max = 1.0, step = 0.05,
            hint = "Plan strength at step 0 is 1 - loose. 0 = stock behaviour.",
        },
        {
            key = "lock_at", type = "slider", label = "Lock At",
            default = 0.5, min = 0.05, max = 1.0, step = 0.05,
            hint = "Fraction of the steps by which the plan is back to full strength.",
        },
    },
}

function schedule(output, num_steps, shift)
    local p = params or {}
    local loose = tonumber(p.loose) or 0.5
    local lock_at = tonumber(p.lock_at) or 0.5
    if loose < 0 then loose = 0 elseif loose > 1 then loose = 1 end
    if lock_at < 0.05 then lock_at = 0.05 elseif lock_at > 1 then lock_at = 1 end
    local w = {}
    for i = 0, num_steps - 1 do
        output[i] = 1.0 - i / num_steps
        local frac = num_steps > 1 and i / ((num_steps - 1) * lock_at) or 1.0
        if frac > 1 then frac = 1 end
        w[i + 1] = (1.0 - loose) + loose * frac
    end
    plan_strength = w
end
