#pragma once

// YUE2_AITK_HOST_PROFILE=1: host wall time per trainer phase, summed over a
// step and printed with it as {"stage":"host_profile",...}. Off (the default)
// it costs one cached flag test per scope. Phases that end in a blocking
// copy include the wait for the GPU work queued before them.

#include <chrono>
#include <cstdlib>
#include <map>
#include <string>

namespace yue2_aitk_prof {
inline bool on() { static const bool v = std::getenv("YUE2_AITK_HOST_PROFILE") != nullptr; return v; }
inline std::map<std::string, double> & totals() { static std::map<std::string, double> m; return m; }
struct Scope {
    const char * name;
    std::chrono::steady_clock::time_point start;
    bool live;
    explicit Scope(const char * n) : name(n), live(on()) { if (live) start = std::chrono::steady_clock::now(); }
    ~Scope() {
        if (live) totals()[name] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    Scope(const Scope &) = delete;
    Scope & operator=(const Scope &) = delete;
};
// Sequential phases in one function: mark("x") charges the time since the
// previous mark (or construction) to "x".
struct Lap {
    std::chrono::steady_clock::time_point last;
    bool live;
    Lap() : live(on()) { if (live) last = std::chrono::steady_clock::now(); }
    void mark(const char * name) {
        if (!live) return;
        const auto now = std::chrono::steady_clock::now();
        totals()[name] += std::chrono::duration<double, std::milli>(now - last).count();
        last = now;
    }
};
} // namespace yue2_aitk_prof

#define YUE2_PROF_CAT2(a, b) a##b
#define YUE2_PROF_CAT(a, b) YUE2_PROF_CAT2(a, b)
#define YUE2_PROF(name) yue2_aitk_prof::Scope YUE2_PROF_CAT(yue2_prof_scope_, __LINE__)(name)
