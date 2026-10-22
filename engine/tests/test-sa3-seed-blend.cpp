// Run from the repo root in a VS developer shell:
// cl /std:c++17 /EHsc /O2 /utf-8 /Iengine\src /Iengine\vendor\luajit\src engine\tests\test-sa3-seed-blend.cpp engine\build\luajit\src\lua51.lib /Felogs\test-sa3-seed-blend.exe /Fologs\test-sa3-seed-blend.obj
#include "lua-plugin.h"
#include <stdexcept>

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static std::vector<float> run(LuaPlugin & plugin, int64_t seed, int64_t seed_b,
                              float blend, int n, int steps = 4) {
    std::vector<float> x(n, 0.25f), v(n, 0.125f);
    SolverState state;
    state.seeds = &seed;
    state.seed_b = seed_b;
    state.seed_blend = blend;
    state.n_per = n;
    for (int i = 0; i < steps; ++i) {
        state.step_index = i;
        lua_call_solver_step(plugin, x.data(), v.data(), 1.0f - i * 0.125f,
                             0.875f - i * 0.125f, n, state, {}, nullptr, {});
    }
    return x;
}

int main() {
    try {
        LuaPlugin plugin;
        check(lua_load_plugin(plugin, "plugins/solvers/md_pingpong_blend.lua", "plugins/solvers"),
              "plugin failed to load");
        check(luaL_dostring(plugin.L, R"LUA(
            math.randomseed(123)
            local expected = math.random()
            math.randomseed(123)
            local a, b = math.randomstream(456), math.randomstream(456)
            for i = 1, 100 do assert(a() == b()) end
            assert(math.random() == expected, 'independent stream changed the main RNG')
            for _, invalid in ipairs({0.5, math.huge, 0/0, 9007199254740992}) do
                assert(not pcall(math.randomstream, invalid), 'invalid seed accepted')
            end
        )LUA") == 0, "independent RNG isolation or seed validation failed");
        for (int n : {5, 64}) {
            const auto a = run(plugin, 123, -1, 0, n);
            const auto b = run(plugin, 456, -1, 0, n);
            check(a == run(plugin, 123, 456, 0, n), "blend zero differs from plain seed A");
            const auto half = run(plugin, 123, 456, 0.5f, n);
            check(half != a && half != b, "default blend did not mix both streams");
            check(half == run(plugin, 123, 456, 0.5f, n), "default blend replay differs");
            check(b == run(plugin, 123, 456, 1, n), "blend one differs from plain seed B");
            check(a == run(plugin, 123, -1, 0, n), "seed B leaked into a later unbred run");
            check(run(plugin, 1099511628232LL, -1, 0, n) ==
                  run(plugin, 123, 1099511628232LL, 1, n), "seed B lost its high bits");

            // Independent analytic first-step oracle for the default 0.5 blend.
            LuaRandState ra, rb;
            lua_rand_seed(&ra, 123, 0);
            lua_rand_seed(&rb, 456, 0);
            const auto uniform = [](LuaRandState * r) {
                return (double) (lua_rand_next(r) >> 11) / 9007199254740992.0;
            };
            const auto gaussian = [&](LuaRandState * r) {
                const double u = std::max(1e-12, uniform(r));
                const double angle = 2.0 * std::acos(-1.0) * uniform(r);
                return std::sqrt(-2.0 * std::log(u)) * std::cos(angle);
            };
            const double want = 0.125 * (0.25 - 0.125) + 0.875 *
                (std::cos(0.5 * 1.5707963) * gaussian(&ra) +
                 std::sin(0.5 * 1.5707963) * gaussian(&rb));
            const auto first = run(plugin, 123, 456, 0.5f, n, 1);
            check(std::abs(first[0] - want) < 1e-6, "default blend disagrees with 64-bit RNG oracle");
        }
        std::puts("PASS: LuaJIT blend 0/default/1, replay, odd/even lengths, seed-B equivalence and high seed bits");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
