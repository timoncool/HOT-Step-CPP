--[[
md_audio_tiled.lua
Postprocess plugin adapter for MD Audio Tiled Core

Wraps md_audio_tiled_core.lua (v3.0.1) to conform to the HOT-Step
postprocess plugin contract. The core module is loaded via require()
and exposes execute_tiled_decode() as the entry point.

© 2026 Alexander Allan (MDMAchine) | A&E Concepts
GPL v3
--]]

local core = require("md_audio_tiled_core")

postprocess = {
    name        = "md_audio_tiled V2",
    display     = "MD Audio Tiled Decoder V2",
    description = "Tiled VAE decode with OLA crossfade, dual-pass merge, LSS, and DSP chain",
    accent      = "cyan",

    params = {
        -- Tiling Architecture
        { key = "tile_size", type = "slider", label = "Tile Size (Frames)",
          default = 1024, min = 256, max = 4096, step = 128,
          hint = "Latent frames per VAE decode chunk. Lower = less VRAM, higher = faster." },
        { key = "overlap", type = "slider", label = "Overlap (Frames)",
          default = 64, min = 16, max = 256, step = 16,
          hint = "Overlap between tiles for crossfading. Higher = smoother seams but more compute." },
        { key = "context_prefix", type = "slider", label = "Context Prefix",
          default = 512, min = 128, max = 1024, step = 64,
          hint = "Causal warm-up frames discarded after decode. Crucial for Oobleck VAE." },
        { key = "dual_pass", type = "toggle", label = "Dual Pass",
          default = false,
          hint = "Two staggered decode passes merged with trapezoidal weights. Eliminates seam artifacts but doubles VAE decode time." },
        
        -- Latent Processing
        { key = "lss_enabled", type = "toggle", label = "Enable LSS",
          default = true,
          hint = "Latent Spectral Suppressor. Suppresses low-variance VAE bias channels." },
        { key = "lss_strength", type = "slider", label = "LSS Strength",
          default = 0.25, min = 0, max = 1, step = 0.01,
          hint = "Latent channel suppression strength. 0.25 is gold standard." },
        { key = "lss_var_thresh", type = "slider", label = "LSS Variance Threshold",
          default = 0.12, min = 0.01, max = 0.5, step = 0.01,
          hint = "Normalized variance threshold below which channels are suppressed." },
        { key = "rms_leveling", type = "toggle", label = "RMS Leveling",
          default = true,
          hint = "Balances RMS energy between tiles before crossfading." },

        -- DSP Chain
        { key = "stereo_width", type = "slider", label = "Stereo Width",
          default = 0.8, min = 0, max = 2, step = 0.01,
          hint = "M/S stereo width. 0=mono, 1=unity, 2=doubled side." },
        { key = "highpass_hz", type = "slider", label = "High-Pass Filter (Hz)",
          default = 20.0, min = 0.0, max = 100.0, step = 5.0,
          hint = "Butterworth 2nd-order high-pass. Removes sub-audible DC rumble. 0 to disable." },
        
        -- Hum Notch Filter
        { key = "hum_notch", type = "toggle", label = "Hum Notch Filter",
          default = true,
          hint = "Multi-band surgical cuts at 74/94/654Hz to remove Oobleck VAE hum." },
        { key = "hum_bass_shelf_db", type = "slider", label = "Hum: 120Hz Shelf Cut (dB)",
          default = -2.0, min = -12.0, max = 0.0, step = 0.5,
          hint = "Bass shelf reduction below 120Hz.",
          visible_when = { key = "hum_notch", equals = true } },
        { key = "hum_74_db", type = "slider", label = "Hum: 74Hz Cut (dB)",
          default = -1.43, min = -12.0, max = 0.0, step = 0.1,
          hint = "Surgical notch depth at 74.4Hz.",
          visible_when = { key = "hum_notch", equals = true } },
        { key = "hum_94_db", type = "slider", label = "Hum: 94Hz Cut (dB)",
          default = -1.86, min = -12.0, max = 0.0, step = 0.1,
          hint = "Surgical notch depth at 94.0Hz.",
          visible_when = { key = "hum_notch", equals = true } },
        { key = "hum_656_db", type = "slider", label = "Hum: 654Hz Cut (dB)",
          default = -15.0, min = -24.0, max = 0.0, step = 1.0,
          hint = "Deep surgical notch depth at 654.0Hz.",
          visible_when = { key = "hum_notch", equals = true } },

        -- Mastering
        { key = "peak_normalize_db", type = "slider", label = "Peak Normalize",
          default = -1, min = -12, max = 0, step = 0.5,
          hint = "Transparent peak normalization (pure gain reduction, no distortion). Set to 0 to disable." },
        { key = "soft_clip_db", type = "slider", label = "Soft Clip Ceiling",
          default = -3.0, min = -12, max = 0, step = 0.5,
          hint = "tanh saturation ceiling in dB. Acts as safety net after peak normalize. Set to 0 to disable." },
    },
}

-- Entry point called by the engine via lua_call_postprocess()
function process(latents, B, C_lat, W, C_aud, final_samples, upscale_factor, vae_decode_fn)
    -- Build params table from UI-injected globals
    local p = {}
    for k, v in pairs(core.DEFAULT_PARAMS) do
        p[k] = v
    end

    -- Override from UI params (injected by lua_inject_params)
    if params then
        -- Tiling
        if params.tile_size then p.tile_size = params.tile_size end
        if params.overlap then p.overlap = params.overlap end
        if params.context_prefix then p.context_prefix = params.context_prefix end
        if params.dual_pass ~= nil then p.dual_pass = params.dual_pass end
        
        -- LSS & RMS
        if params.lss_enabled ~= nil then p.lss_enabled = params.lss_enabled end
        if params.lss_strength then p.lss_strength = params.lss_strength end
        if params.lss_var_thresh then p.lss_var_thresh = params.lss_var_thresh end
        if params.rms_leveling ~= nil then p.rms_leveling = params.rms_leveling end
        
        -- DSP
        if params.stereo_width then p.stereo_width = params.stereo_width end
        if params.highpass_hz then p.highpass_hz = params.highpass_hz end
        
        -- Hum Notch
        if params.hum_notch ~= nil then p.hum_notch_enabled = params.hum_notch end
        if params.hum_bass_shelf_db then p.hum_bass_shelf_db = params.hum_bass_shelf_db end
        if params.hum_74_db then p.hum_74_db = params.hum_74_db end
        if params.hum_94_db then p.hum_94_db = params.hum_94_db end
        if params.hum_656_db then p.hum_656_db = params.hum_656_db end

        -- Mastering
        if params.peak_normalize_db then
            if params.peak_normalize_db < 0 then
                p.peak_normalize_db = params.peak_normalize_db
            else
                p.peak_normalize_db = nil  -- disable
            end
        end
        if params.soft_clip_db then p.soft_clip_db = params.soft_clip_db end
    end

    local audio = core.execute_tiled_decode(
        vae_decode_fn, latents, B, C_lat, W,
        C_aud, final_samples, upscale_factor, p)

    -- Bridge expects two return values: audio_table, T_audio
    return audio, final_samples
end