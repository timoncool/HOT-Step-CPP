#pragma once
// lm-cfg-extras.h: opt-in refinements to classifier-free guidance on LM logits,
// shared by the ACE-Step 5Hz LM (pipeline-lm.cpp) and the YuE2 AR decode loop.
//
// Ported in spirit from comfyui_acetricks' CFGExtLogitsProcessor. Plain CFG is
// u + s * (c - u) on raw logits; the three knobs below change that blend and
// all default to off, so a request that sets none of them takes the caller's
// existing CFG path untouched (YuE2's is fixture-pinned BF16 arithmetic).
//
//   min_p    plausibility mask: blend only the entries whose cond probability
//            is at least min_p times the top cond probability; every other
//            entry keeps its cond logit. Stops CFG from promoting tokens the
//            conditional model itself considers implausible. 0 = off.
//   logprob  blend log-softmax(c) and log-softmax(u) instead of raw logits.
//            The two rows are normalised before the difference is taken, so a
//            row whose logits sit at a different offset no longer biases it.
//   rescale  pull the blended row's spread back toward the cond row's (std
//            over the finite entries), lerped by this strength in [0, 1].
//            High CFG scales flatten or sharpen the distribution; this undoes
//            the change in spread while keeping the change in ranking.
//
// Entries where cond is not finite (masked) stay as cond. An uncond entry that
// is not finite where cond is falls back to cond, so the blend is a no-op there.

#include <cmath>
#include <cstddef>
#include <vector>

struct LmCfgExtras {
    float min_p   = 0.0f;
    bool  logprob = false;
    float rescale = 0.0f;

    bool any() const { return min_p > 0.0f || logprob || rescale > 0.0f; }
};

namespace lm_cfg_extras_detail {

// In-place log-softmax over the finite entries; non-finite entries untouched.
static inline void log_softmax(std::vector<double> & x) {
    double mx = -INFINITY;
    for (double v : x) {
        if (std::isfinite(v) && v > mx) mx = v;
    }
    if (!std::isfinite(mx)) return;
    double z = 0.0;
    for (double v : x) {
        if (std::isfinite(v)) z += std::exp(v - mx);
    }
    const double lse = mx + std::log(z);
    for (double & v : x) {
        if (std::isfinite(v)) v -= lse;
    }
}

static inline double finite_std(const std::vector<double> & x) {
    double  s = 0.0, s2 = 0.0;
    int64_t n = 0;
    for (double v : x) {
        if (!std::isfinite(v)) continue;
        s += v;
        s2 += v * v;
        n++;
    }
    if (n < 2) return 0.0;
    const double mean = s / (double) n;
    return std::sqrt(std::max(0.0, s2 / (double) n - mean * mean));
}

}  // namespace lm_cfg_extras_detail

// out[i] for i in [0, V). cond/uncond/out may not alias each other except
// out == cond, which is allowed (the decode loops blend in place).
static inline void lm_cfg_blend_extras(const float * cond, const float * uncond, size_t V, float scale,
                                       const LmCfgExtras & x, float * out) {
    using namespace lm_cfg_extras_detail;
    std::vector<double> c(V), u(V);
    for (size_t i = 0; i < V; i++) {
        c[i] = cond[i];
        u[i] = std::isfinite(uncond[i]) ? uncond[i] : cond[i];
    }
    std::vector<char> blend(V, 1);
    if (x.min_p > 0.0f) {
        double mx = -INFINITY;
        for (double v : c) {
            if (std::isfinite(v) && v > mx) mx = v;
        }
        // p_i >= min_p * p_max  <=>  c_i >= c_max + log(min_p)
        const double floor = mx + std::log((double) x.min_p);
        for (size_t i = 0; i < V; i++) blend[i] = std::isfinite(c[i]) && c[i] >= floor;
    }
    if (x.logprob) {
        log_softmax(c);
        log_softmax(u);
    }
    std::vector<double> r(V);
    for (size_t i = 0; i < V; i++) {
        r[i] = (std::isfinite(c[i]) && blend[i]) ? u[i] + (double) scale * (c[i] - u[i]) : c[i];
    }
    if (x.rescale > 0.0f) {
        const double sc = finite_std(c), sr = finite_std(r);
        if (sc > 0.0 && sr > 0.0) {
            const double k = 1.0 + (double) x.rescale * (sc / sr - 1.0);
            for (double & v : r) {
                if (std::isfinite(v)) v *= k;
            }
        }
    }
    for (size_t i = 0; i < V; i++) out[i] = (float) r[i];
}
