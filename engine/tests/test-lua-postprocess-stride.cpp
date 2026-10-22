// Self-check: lua_call_postprocess moves short stereo output through both
// transfers at the right strides. The decode callback hands back audio at
// capacity stride; process() returns a table packed at the count it returns;
// audio_out comes back at capacity stride. Needs the LuaJIT lib from an
// engine build:
//   cl /std:c++17 /EHsc /utf-8 /I..\src /I..\vendor\luajit\src test-lua-postprocess-stride.cpp ..\build\luajit\src\lua51.lib
#include "lua-plugin.h"
#include <cassert>
#include <cstdio>

static const int SPF   = 8;   // samples per latent frame
static const int SHORT = 3;   // decoder returns this many samples fewer than capacity

// Fake decoder: ch0[i] = i + 1, ch1[i] = -(i + 1), capacity stride, junk in the gap.
static int fake_decode(const float *, int T_lat, float * out, int max_T) {
    int n = T_lat * SPF - SHORT;
    for (int i = 0; i < 2 * max_T; i++) out[i] = 999.0f;
    for (int i = 0; i < n; i++) {
        out[i]         = (float) (i + 1);
        out[max_T + i] = (float) -(i + 1);
    }
    return n;
}

static void run(const char * process_src, int want_T) {
    LuaPlugin p;
    p.name = "stride_test";
    p.L    = luaL_newstate();
    luaL_openlibs(p.L);
    int rc = luaL_dostring(p.L, process_src);
    assert(rc == 0);

    const int T_lat = 4, C_lat = 2, max_T = T_lat * SPF;
    std::vector<float> lat((size_t) T_lat * C_lat, 0.0f);
    std::vector<float> out((size_t) 2 * max_T, 0.0f);
    int T = lua_call_postprocess(p, lat.data(), T_lat, C_lat, 2, out.data(), max_T, fake_decode, {},
                                 LuaModelContext{}, SPF);
    if (T != want_T) {
        std::fprintf(stderr, "T = %d, want %d\n", T, want_T);
        assert(false);
    }
    const int n = T_lat * SPF - SHORT;
    for (int i = 0; i < n; i++) {
        if (out[i] != (float) (i + 1) || out[(size_t) max_T + i] != (float) -(i + 1)) {
            std::fprintf(stderr, "sample %d: ch0 %g ch1 %g\n", i, out[i], out[(size_t) max_T + i]);
            assert(false);
        }
    }
    lua_close(p.L);
    p.L = nullptr;
}

int main() {
    // Pass-through: returns the decoder's short table and count.
    run("function process(lat, B, C, W, Ca, final, up, dec) return dec(lat, W) end", 4 * SPF - SHORT);
    // Shipped-plugin shape: repacks into a final_samples-stride table and returns final_samples.
    run("function process(lat, B, C, W, Ca, final, up, dec)\n"
        "  local a, n = dec(lat, W)\n"
        "  local o = {}\n"
        "  for c = 0, 1 do for i = 1, final do\n"
        "    o[c * final + i] = i <= n and a[c * n + i] or 0\n"
        "  end end\n"
        "  return o, final\n"
        "end", 4 * SPF);
    std::puts("lua postprocess stride: ok");
    return 0;
}
