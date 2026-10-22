#pragma once
// Full-loop (owns_loop) Lua solver bridge for one YuE2 NAR chunk.
//
// Kept apart from yue2-nar-graph.h so tests/test-yue2-nar-loop.cpp can drive it
// with a fake velocity model and no GGML. The plugin's sample() gets the whole
// schedule and calls model_fn(x, t) / on_step(i, t, t_next) itself, so every
// failure has to be carried out of Lua by flag: a model error, a cancel or
// non-finite latents set `fail`, after which model_fn writes zeros without
// touching the model and on_step answers "stop".

#include "lua-plugin.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// velocity(x, t, v_out): x and v_out hold n floats. Returns false on a model
// error, having set the caller's error string itself.
using Yue2NarVelocityFn = std::function<bool(const float * x, float t, float * v_out)>;

// `times` holds steps + 1 descending entries ending at 0; sample() sees the
// first `steps` and owns the final update to t = 0. on_step(step, steps)
// returning false cancels. Returns false with *err set on a Lua error, a model
// error, a cancel ("cancelled") or non-finite latents.
static bool yue2_nar_run_loop_solver(LuaPlugin & solver, std::vector<float> & state,
                                     const std::vector<float> & times, int steps, int T, int C,
                                     const Yue2NarVelocityFn & velocity,
                                     const std::function<bool(int step, int steps)> & on_step,
                                     const std::unordered_map<std::string, std::string> & params,
                                     const LuaModelContext & ctx, int64_t seed, std::string * err) {
    const int n = (int) state.size();
    std::vector<float> vt((size_t) n, 0.0f);
    std::string fail;
    auto all_finite = [](const float * p, int k) {
        for (int i = 0; i < k; i++) if (!std::isfinite(p[i])) return false;
        return true;
    };

    LoopModelFn model_fn = [&](const float * x, float t) {
        if (fail.empty() && !(t >= 0.0f && t <= 1.0f)) fail = "YuE2 NAR full-loop solver asked the model for t outside [0, 1]";
        if (fail.empty() && !velocity(x, t, vt.data())) {
            fail = (err && !err->empty()) ? *err : "YuE2 NAR model evaluation failed";
        }
        if (fail.empty() && !all_finite(vt.data(), n)) fail = "YuE2 NAR model produced non-finite velocity";
        if (!fail.empty()) std::fill(vt.begin(), vt.end(), 0.0f);
    };
    LoopOnStepFn loop_step = [&](int step_idx, float, float) -> bool {
        if (fail.empty() && !all_finite(state.data(), n)) fail = "YuE2 NAR plugin produced non-finite latents";
        if (fail.empty() && on_step && !on_step(step_idx + 1, steps)) fail = "cancelled";
        return !fail.empty();
    };

    const bool lua_ok = lua_call_solver_loop(solver, state.data(), vt.data(), times.data(), steps, n,
                                             /*N=*/1, T, C, model_fn, loop_step, params, ctx, seed);
    if (!fail.empty()) {
        if (err) *err = fail;  // a model error's own message is already in fail
        return false;
    }
    if (!lua_ok) {
        if (err) *err = "YuE2 NAR full-loop solver '" + solver.name + "' failed (Lua error or missing sample())";
        return false;
    }
    if (!all_finite(state.data(), n)) {
        if (err) *err = "YuE2 NAR plugin produced non-finite latents";
        return false;
    }
    return true;
}
