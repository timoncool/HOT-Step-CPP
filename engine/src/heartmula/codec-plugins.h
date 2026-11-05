#pragma once
// heartmula/codec-plugins.h: the Lua solver, scheduler and guidance plugins on the
// HeartCodec flow decode.
//
// HOT-Step file. The bridge is minimax/mm3-plugins.h's, and its proof carries over
// unchanged: HeartCodec's sampler has MM3's shape (t ascends 0 -> 1, x += dt v, the
// same per-step overlap blend `(1 - (1 - 1e-6) t) noise + t context`), so with
// t_ace = 1 - t and v_ace = -v the plugins run in the ACE convention they were
// written against, and the final step stays the engine's.
//
// What differs from MM3: the latents are already time-major [frames][256], ACE's
// layout, so the boundary only negates, never transposes. CFG is one batch-2
// forward (row 0 uncond with zero mu, row 1 cond), and only when codec guidance
// is above 1; at 1 there is no uncond branch, so guidance plugins are skipped.
//
// Only hot-step-server includes this (it links LuaJIT). pipeline.h and the
// HeartMuLa tools see a plain heartcodec::Solve, empty unless a plugin is named,
// and an empty Solve runs solve_euler untouched, which is what heartcodec-probe
// holds to the reference. Owns-loop solvers are refused, as on MM3: they would
// skip the per-step overlap blend.

#include "heartmula/codec-decode.h"
#include "minimax/mm3-plugins.h"

#include <map>
#include <memory>

namespace heartcodec {

// Look the names up once per job. Unknown names warn and fall back to native, as on MM3.
inline MM3PluginRun plugins_resolve(const MM3PluginSel & sel) {
    MM3PluginRun run;
    run.sel = sel;
    if (!sel.any()) return run;
    auto & reg = PluginRegistry::instance();
    if (!sel.solver.empty()) {
        run.solver = reg.solver_lookup(sel.solver.c_str());
        if (!run.solver) {
            fprintf(stderr, "[HeartCodec-Plugins] WARNING: unknown solver '%s', using native Euler\n", sel.solver.c_str());
        } else if (run.solver->owns_loop) {
            fprintf(stderr, "[HeartCodec-Plugins] WARNING: solver '%s' owns its loop, which would skip the overlap "
                            "blend; using native Euler\n", sel.solver.c_str());
            run.solver = nullptr;
        }
    }
    if (!sel.scheduler.empty()) {
        run.scheduler = reg.scheduler_lookup(sel.scheduler.c_str());
        if (!run.scheduler) {
            fprintf(stderr, "[HeartCodec-Plugins] WARNING: unknown scheduler '%s', using the native schedule\n",
                    sel.scheduler.c_str());
        }
    }
    if (sel.guidance == "apg") {
        run.guidance_native_apg = true;
    } else if (!sel.guidance.empty()) {
        run.guidance = reg.guidance_lookup(sel.guidance.c_str());
        if (!run.guidance) {
            fprintf(stderr, "[HeartCodec-Plugins] WARNING: unknown guidance '%s', using native CFG\n", sel.guidance.c_str());
        }
    }
    run.active = run.solver || run.scheduler || run.guidance || run.guidance_native_apg;
    run.lua_ctx.native_sr       = 48000;
    run.lua_ctx.latent_fps      = 25;
    run.lua_ctx.latent_channels = 256;
    run.lua_ctx.model_id        = "heartmula";
    if (run.active) {
        fprintf(stderr, "[HeartCodec-Plugins] solver=%s scheduler=%s guidance=%s\n",
                run.solver ? run.solver->name.c_str() : "(native euler)",
                run.scheduler ? run.scheduler->name.c_str() : "(native linspace)",
                run.guidance_native_apg ? "apg (native)" : run.guidance ? run.guidance->name.c_str() : "(native cfg)");
    }
    return run;
}

// t ascending, steps + 1 entries ending at 1: the scheduler's (1 - t_ace, clamped), or the
// corrected linspace solve_euler walks.
inline std::vector<float> plugins_times(MM3PluginRun & run, int steps) {
    std::vector<float> ts((size_t) steps + 1, 1.0f);
    if (run.scheduler) {
        std::vector<float> t_ace((size_t) steps, 0.0f);
        lua_call_scheduler(*run.scheduler, t_ace.data(), steps, run.sel.shift, run.sel.params, run.lua_ctx);
        for (int i = 0; i < steps; i++) {
            float t = t_ace[(size_t) i];
            if (!(t == t)) t = 0.0f;  // third-party Lua: clamp rather than trust
            ts[(size_t) i] = 1.0f - std::min(1.0f, std::max(0.0f, t));
        }
    } else {
        for (int i = 0; i < steps; i++)
            ts[(size_t) i] = i <= steps / 2 ? float(i) / float(steps) : 1.0f - float(steps - i) / float(steps);
    }
    return ts;
}

// One window's flow loop through the plugins. Arguments as solve_euler.
inline std::vector<float> solve_plugins(MM3PluginRun & run, const Velocity & velocity, const std::vector<float> & noise,
                                        const std::vector<float> & context, const std::vector<float> & mu,
                                        int64_t frames, int64_t overlap, int steps, float guidance,
                                        const Cancel & cancelled) {
    require(frames > 0 && steps > 0 && overlap >= 0 && overlap <= frames && std::isfinite(guidance), "invalid solver parameters");
    require(noise.size() == size_t(frames * 256) && context.size() == noise.size() &&
            mu.size() == size_t(frames * 512), "invalid solver input shape");
    const int64_t N     = frames * 256;
    const int     batch = guidance > 1 ? 2 : 1;
    run.begin_window(N);  // fresh momentum and solver state per window, never across a seam
    const auto ts = plugins_times(run, steps);

    // rows 2: [uncond, cond]; rows 1: cond when `cond`, else uncond (zero mu).
    auto forward = [&](const float * x, float t, int rows, bool cond) {
        std::vector<float> input(size_t(rows * frames * 1024), 0);
        for (int b = 0; b < rows; ++b) for (int64_t f = 0; f < frames; ++f) {
            auto dst = input.data() + (b * frames + f) * 1024;
            std::copy_n(x + f * 256, 256, dst);
            if (f < overlap) std::copy_n(context.data() + f * 256, 256, dst + 256);
            if (rows == 2 ? b == 1 : cond) std::copy_n(mu.data() + f * 512, 512, dst + 512);
        }
        auto v = velocity(input, frames, std::vector<float>(size_t(rows), t));
        require(v.size() == size_t(rows * N), "invalid velocity output shape");
        return v;
    };
    // Guided velocity into run.tm_v, ACE convention (negated).
    int   cur_step = 0;
    float cur_dt   = 0.0f;
    auto  guide    = [&](const std::vector<float> & v, float t_ace) {
        if (batch == 2 && (run.guidance || run.guidance_native_apg)) {
            for (int64_t j = 0; j < N; ++j) {
                run.tm_u[(size_t) j] = -v[(size_t) j];
                run.tm_c[(size_t) j] = -v[(size_t) (N + j)];
            }
            GuidanceCtx ctx{ cur_step, steps, cur_dt, t_ace };
            if (run.guidance_native_apg) {
                apg_forward(run.tm_c.data(), run.tm_u.data(), guidance, run.mbuf, run.tm_v.data(), 256, (int) frames,
                            run.sel.apg_norm_threshold, run.apg_ws);
            } else {
                lua_call_guidance(*run.guidance, run.tm_c.data(), run.tm_u.data(), guidance, run.mbuf, run.tm_v.data(),
                                  256, (int) frames, ctx, run.sel.apg_norm_threshold, run.sel.params, run.lua_ctx);
            }
        } else {
            for (int64_t j = 0; j < N; ++j) {
                const float u = v[(size_t) j];
                run.tm_v[(size_t) j] = -(batch == 2 ? u + guidance * (v[(size_t) (N + j)] - u) : u);
            }
        }
    };

    // Callbacks run inside Lua, so a failure latches here and is rethrown after the step.
    std::string failed;
    auto        guarded = [&](auto && body) {
        try { body(); } catch (const std::exception & e) { if (failed.empty()) failed = e.what(); }
    };
    SolverModelFn model_fn;
    if (run.solver && run.solver->needs_model) {
        model_fn = [&](const float * xt_ace, float t_val) {
            guarded([&] { guide(forward(xt_ace, 1.0f - t_val, batch, true), t_val); });
        };
    }
    const bool      post = run.guidance && run.guidance->has_post_step && batch == 2;
    PostStepModelFn eval_cond, eval_uncond;
    if (post) {
        eval_cond = [&](const float * xt_ace, float t_val) {
            guarded([&] {
                auto v = forward(xt_ace, 1.0f - t_val, 1, true);
                for (int64_t j = 0; j < N; ++j) run.tm_c[(size_t) j] = -v[(size_t) j];
            });
        };
        eval_uncond = [&](const float * xt_ace, float t_val) {
            guarded([&] {
                auto v = forward(xt_ace, 1.0f - t_val, 1, false);
                for (int64_t j = 0; j < N; ++j) run.tm_u[(size_t) j] = -v[(size_t) j];
            });
        };
    }

    auto x = noise;
    for (int i = 0; i < steps; ++i) {
        check_cancel(cancelled);
        const float t = ts[(size_t) i];
        for (int64_t j = 0; j < overlap * 256; ++j)
            x[(size_t) j] = (1.0f - (1.0f - 1e-6f) * t) * noise[(size_t) j] + t * context[(size_t) j];
        const float t_ace = 1.0f - t, t_next = 1.0f - ts[(size_t) i + 1];
        cur_step = i;
        cur_dt   = t_ace - t_next;
        guide(forward(x.data(), t, batch, true), t_ace);
        if (run.solver && i < steps - 1) {
            // A multi-eval solver's model_fn overwrites tm_v, so it reads a snapshot (mm3-plugins.h).
            const float * vt = run.tm_v.data();
            if (run.solver->needs_model) {
                run.tm_vpre = run.tm_v;
                vt          = run.tm_vpre.data();
            }
            run.solver_state.step_index = i;
            lua_call_solver_step(*run.solver, x.data(), vt, t_ace, t_next, (int) N, run.solver_state, model_fn,
                                 run.tm_v.data(), run.sel.params, run.lua_ctx);
        } else {
            // Native Euler, and at the last step the engine-owned x0 = x - t_ace v_ace.
            for (int64_t j = 0; j < N; ++j) x[(size_t) j] -= cur_dt * run.tm_v[(size_t) j];
        }
        if (post && i < steps - 1) {
            GuidanceCtx ctx{ i, steps, cur_dt, t_next };
            lua_call_post_step(*run.guidance, x.data(), t_next, (int) N, eval_cond, eval_uncond, run.tm_c.data(),
                               run.tm_u.data(), ctx, run.sel.params, run.lua_ctx);
        }
        if (!failed.empty()) throw std::runtime_error("plugin sub-evaluation failed: " + failed);
    }
    std::copy_n(context.begin(), size_t(overlap * 256), x.begin());
    return x;
}

// The decode hook for a request's picks; empty when nothing resolves to a plugin.
inline Solve plugin_solver(const MM3PluginSel & sel) {
    auto run = std::make_shared<MM3PluginRun>(plugins_resolve(sel));
    if (!run->active) return {};
    return [run](const Velocity & velocity, const std::vector<float> & noise, const std::vector<float> & context,
                 const std::vector<float> & mu, int64_t frames, int64_t overlap, int steps, float guidance,
                 const Cancel & cancelled) {
        return solve_plugins(*run, velocity, noise, context, mu, frames, overlap, steps, guidance, cancelled);
    };
}

}  // namespace heartcodec
