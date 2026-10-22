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

-- MD Plan-Aware Scheduler (YuE2) v1.0
-- MDMAchine | A&E Concepts (C) 2026
--
-- Demo of model_context.yue2, the plan metadata the YuE2 NAR path hands to
-- Lua plugins. The scheduler runs once per NAR chunk. When the chunk overlaps
-- a plan section whose name equals the `section` param (e.g. "chorus"), it
-- packs steps toward high noise, where large-scale structure is decided:
--     t_i = 1 - (i / N) ^ (1 + 3 * density)
-- Every other chunk (and every non-YuE2 model) gets the uniform 1 - i/N,
-- the same grid the stock YuE2 path uses.
--
-- It prints one line per chunk with the tempo, meter, bars, sections and chunk
-- span it saw, so the table is easy to inspect in the engine log.

scheduler = {
    name        = "md_plan_aware_yue2",
    display     = "MD Plan-Aware (YuE2)",
    description = "YuE2 only: denser high-noise steps in NAR chunks that overlap a chosen plan section; uniform elsewhere.",
    params      = {
        {
            key     = "section",
            type    = "select",
            label   = "Target Section",
            default = "chorus",
            options = {
                { value = "chorus",     label = "Chorus"     },
                { value = "verse",      label = "Verse"      },
                { value = "pre-chorus", label = "Pre-chorus" },
                { value = "bridge",     label = "Bridge"     },
                { value = "intro",      label = "Intro"      },
                { value = "interlude",  label = "Interlude"  },
                { value = "outro",      label = "Outro"      },
                { value = "none",       label = "None (uniform)" },
            },
            hint    = "Chunks overlapping a plan section with this name get the high-noise-dense schedule.",
        },
        {
            key     = "density",
            type    = "slider",
            label   = "High-Noise Density",
            default = 0.5,
            min     = 0.0,
            max     = 1.0,
            step    = 0.05,
            hint    = "0 = uniform. Higher = more steps near t=1 in matched chunks.",
        },
    },
}

local function describe(y)
    local secs = {}
    for i, s in ipairs(y.sections or {}) do
        secs[#secs + 1] = string.format("%s[%.1f-%.1f]", s.name, s.start_s, s.end_s)
    end
    local c = y.chunk or {}
    return string.format(
        "tempo=%.1f meter=%s bars=%d vocal_pct=%.1f total_s=%.1f fps=%.1f song=%d "
        .. "chunk=%d/%d frames=%d-%d s=%.2f-%.2f sections=%d %s",
        y.tempo or 0, y.meter or "", y.bars or 0, y.vocal_pct or 0, y.total_s or 0,
        y.frame_rate or 0, y.song or 0, c.index or -1, c.count or 0,
        c.start_frame or 0, c.end_frame or 0, c.start_s or 0, c.end_s or 0,
        #(y.sections or {}), table.concat(secs, " "))
end

local function chunk_matches(y, want)
    if not y or not y.chunk or want == "none" then return false end
    local a, b = y.chunk.start_s, y.chunk.end_s
    for _, s in ipairs(y.sections or {}) do
        if s.name == want and s.start_s < b and s.end_s > a then return true end
    end
    return false
end

function schedule(output, num_steps, shift)
    local p = params or {}
    local want = tostring(p.section or "chorus")
    local density = tonumber(p.density) or 0.5
    if density < 0 then density = 0 elseif density > 1 then density = 1 end
    local y = model_context and model_context.yue2
    if y then
        print("[md_plan_aware_yue2] yue2 " .. describe(y))
    else
        print("[md_plan_aware_yue2] no yue2 table (model " ..
              tostring(model_context and model_context.model_id) .. ")")
    end
    local match = chunk_matches(y, want)
    local gamma = match and (1.0 + 3.0 * density) or 1.0
    local parts = {}
    for i = 0, num_steps - 1 do
        local t = 1.0 - (i / num_steps) ^ gamma
        output[i] = t
        parts[#parts + 1] = string.format("%.4f", t)
    end
    print(string.format("[md_plan_aware_yue2] section=%s match=%s gamma=%.2f schedule=%s",
                        want, tostring(match), gamma, table.concat(parts, ",")))
end
