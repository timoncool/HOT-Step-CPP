// Self-check: the YuE2 NAR full-loop (owns_loop) bridge runs the shipped
// md_tlie_yue2 / md_latent_eq_yue2 plugins against a fake velocity model, and
// carries model errors, cancels, Lua errors and non-finite values out of
// sample(). Run from the repo root (loads plugins/solvers/*.lua). Needs the
// LuaJIT lib from an engine build:
//   cl /std:c++17 /EHsc /utf-8 /Iengine\src /Iengine\src\yue2 /Iengine\vendor\luajit\src engine\tests\test-yue2-nar-loop.cpp engine\build\luajit\src\lua51.lib
#include "yue2-nar-loop.h"
#include <cassert>
#include <cstdio>

static const int FRAMES = 4, CH = 64, N = FRAMES * CH, STEPS = 8;

static std::vector<float> uniform_times() {
    std::vector<float> t(STEPS + 1);
    for (int i = 0; i <= STEPS; i++) t[i] = 1.0f - (float) i / STEPS;
    return t;
}

static std::vector<float> start_state() {
    std::vector<float> x(N);
    for (int i = 0; i < N; i++) x[i] = 0.5f + 0.01f * i;
    return x;
}

static LuaPlugin load_file(const char * path) {
    LuaPlugin p;
    bool ok = lua_load_plugin(p, path, "plugins/solvers");
    if (!ok || !p.owns_loop) {
        std::fprintf(stderr, "failed to load %s as a full-loop solver\n", path);
        assert(false);
    }
    return p;
}

static LuaPlugin load_src(const char * src) {
    LuaPlugin p;
    p.name = "inline_loop";
    p.L    = luaL_newstate();
    lua_setup_sandbox(p.L);
    int rc = luaL_dostring(p.L, src);
    assert(rc == 0);
    return p;
}

struct Run {
    bool        ok = false;
    std::string err;
    int         calls = 0, last_step = 0;
    std::vector<float> x = start_state();
};

// velocity = x unless `bad_call` (1-based) fails (model error) or poisons (NaN).
static Run run(LuaPlugin & p, const std::unordered_map<std::string, std::string> & params,
               int fail_call = 0, int nan_call = 0, int cancel_step = 0,
               const LuaModelContext & ctx = LuaModelContext{}, Yue2NarVelocityFn vel = {}) {
    Run r;
    std::string * errp = &r.err;
    if (!vel) {
        vel = [&](const float * x, float, float * v) {
            r.calls++;
            if (r.calls == fail_call) { *errp = "fake model failure"; return false; }
            for (int i = 0; i < N; i++) v[i] = r.calls == nan_call ? NAN : x[i];
            return true;
        };
    }
    auto on_step = [&](int step, int) {
        r.last_step = step;
        return step != cancel_step;
    };
    r.ok = yue2_nar_run_loop_solver(p, r.x, uniform_times(), STEPS, FRAMES, CH, vel, on_step, params, ctx,
                                    1234, &r.err);
    return r;
}

static void expect_euler(const Run & r, double rate_scale = 1.0) {
    std::vector<float> want = start_state();
    const double dt = 1.0 / STEPS;
    for (int s = 0; s < STEPS; s++)
        for (int i = 0; i < N; i++) want[i] = (float) (want[i] - want[i] * rate_scale * dt);
    for (int i = 0; i < N; i++) {
        if (std::fabs(r.x[i] - want[i]) > 1e-5f) {
            std::fprintf(stderr, "x[%d] = %g, want %g\n", i, r.x[i], want[i]);
            assert(false);
        }
    }
}

int main() {
    LuaPlugin tlie = load_file("plugins/solvers/md_tlie_yue2.lua");
    LuaPlugin eq   = load_file("plugins/solvers/md_latent_eq_yue2.lua");

    // lie = 0 and gain = 1 are plain Euler on the true schedule.
    Run a = run(tlie, { { "md_tlie_yue2:lie", "0" } });
    assert(a.ok && a.calls == STEPS && a.last_step == STEPS);
    expect_euler(a);
    // lie = 0 queries the true t, including t = 1 (no clamp to T_MAX): velocity = t.
    std::vector<float> lie0_t;
    Run a2 = run(tlie, { { "md_tlie_yue2:lie", "0" } }, 0, 0, 0, {}, [&](const float *, float t, float * v) {
        lie0_t.push_back(t);
        for (int i = 0; i < N; i++) v[i] = t;
        return true;
    });
    assert(a2.ok && lie0_t.size() == STEPS);
    {
        std::vector<float> times = uniform_times();
        double want = start_state()[0];
        for (int s = 0; s < STEPS; s++) {
            assert(lie0_t[s] == times[s]);
            want -= times[s] * (times[s] - times[s + 1]);
        }
        assert(std::fabs(a2.x[0] - want) < 1e-6);
    }
    Run b = run(eq, { { "md_latent_eq_yue2:gain", "1" } });
    assert(b.ok && b.calls == STEPS);
    expect_euler(b);
    // group -1, gain 2 scales every channel's velocity.
    Run c = run(eq, { { "md_latent_eq_yue2:group", "-1" }, { "md_latent_eq_yue2:gain", "2" } });
    assert(c.ok);
    expect_euler(c, 2.0);
    // A later unity entry resets its group: gains "-1=2,3=1" leaves channels 24..31 at 1.
    Run c2 = run(eq, { { "md_latent_eq_yue2:gains", "-1=2,3=1" } });
    assert(c2.ok);
    for (int i = 0; i < N; i++) {
        const int ch = i % CH;
        const double rate = (ch >= 24 && ch < 32) ? 1.0 : 2.0;
        double want = start_state()[i];
        for (int s = 0; s < STEPS; s++) want -= want * rate / STEPS;
        if (std::fabs(c2.x[i] - want) > 1e-5) {
            std::fprintf(stderr, "gains x[%d] = %g, want %g\n", i, c2.x[i], want);
            assert(false);
        }
    }

    // Model error on the 3rd call: stops, keeps the model's message, no more calls.
    Run d = run(tlie, {}, /*fail_call=*/3);
    assert(!d.ok && d.err == "fake model failure" && d.calls == 3);
    // Cancel after step 2.
    Run e = run(eq, {}, 0, 0, /*cancel_step=*/2);
    assert(!e.ok && e.err == "cancelled" && e.calls == 2);
    // Non-finite velocity.
    Run f = run(tlie, {}, 0, /*nan_call=*/2);
    assert(!f.ok && f.err.find("non-finite velocity") != std::string::npos && f.calls == 2);

    // Plan context reaches the plugin: per-section lies make one model call per
    // distinct lie each step, and each frame takes its own section's velocity.
    LuaYue2Plan plan;
    plan.frame_rate = 1.0;
    plan.chunk_start_frame = 0;
    plan.chunk_end_frame   = FRAMES;
    plan.sections = { { "verse", 0, 1, 0.0, 2.0 }, { "chorus", 1, 2, 2.0, 4.0 } };
    LuaModelContext ctx;
    ctx.model_id = "yue2";
    ctx.yue2     = &plan;
    int sec_calls = 0;
    Run g = run(tlie, { { "md_tlie_yue2:lie", "0" }, { "md_tlie_yue2:lie_curve", "sections" },
                        { "md_tlie_yue2:section_lies", "chorus=0.25" } },
                0, 0, 0, ctx, [&](const float *, float t, float * v) {
                    sec_calls++;
                    for (int i = 0; i < N; i++) v[i] = t;  // velocity = queried t
                    return true;
                });
    assert(g.ok && sec_calls == 2 * STEPS);
    {
        std::vector<float> times = uniform_times();
        for (int f = 0; f < FRAMES; f++) {
            double want = start_state()[f * CH];
            for (int s = 0; s < STEPS; s++) {
                double tq = f >= 2 ? std::min(0.9999, (double) times[s] + 0.25) : (double) times[s];
                want -= tq * (times[s] - times[s + 1]);
            }
            if (std::fabs(g.x[f * CH] - want) > 1e-5) {
                std::fprintf(stderr, "sections frame %d: %g, want %g\n", f, g.x[f * CH], want);
                assert(false);
            }
        }
    }

    // Lua error inside sample().
    LuaPlugin boom = load_src("function sample(xt, vt, sched, n, model_fn) error('boom') end");
    Run h = run(boom, {});
    assert(!h.ok && h.err.find("Lua error") != std::string::npos);
    // Plugin writes NaN into the latents.
    LuaPlugin nan_plugin = load_src(
        "function sample(xt, vt, sched, n, model_fn)\n"
        "  for i = 1, #sched do model_fn(xt, sched[i]); xt[0] = 0/0\n"
        "    if on_step(i - 1, sched[i], 0) then return end end\n"
        "end");
    Run k = run(nan_plugin, {});
    assert(!k.ok && k.err.find("non-finite latents") != std::string::npos && k.calls == 1);
    // Model asked for t outside [0, 1]: never reaches the model.
    LuaPlugin bad_t = load_src("function sample(xt, vt, sched, n, model_fn) model_fn(xt, 1.5) end");
    Run m = run(bad_t, {});
    assert(!m.ok && m.err.find("outside [0, 1]") != std::string::npos && m.calls == 0);

    std::printf("ok\n");
    return 0;
}
