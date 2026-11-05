#pragma once
// heartmula/sample.h: heartlib's CFG blend and sample_topk, with injectable draws.
//
// HOT-Step file. Reference (modeling_heartmula.py):
//
//   guided = uncond + (cond - uncond) * cfg_scale           (when B == 2)
//   logits = guided / temperature
//   logits[logits < topk(logits, k).values[-1]] = -inf      (strict <: ties survive)
//   probs  = softmax(log_softmax(logits))
//   q      = empty_like(probs).exponential_(1)
//   code   = argmax(probs / q)                              (first index on ties)
//
// The exponential race is a multinomial draw. Filtered entries have probability
// 0, so only candidates need a draw; the stream visits them in ascending vocab
// order, so a seed reproduces a song within C++. torch's Philox stream cannot be
// reproduced, so parity uses injected draws (one per vocab entry, as torch does)
// or teacher codes, and same-seed listening compares across runtimes.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

struct HeartMuLaDraws {
    std::mt19937_64 rng;
    const float *   injected = nullptr;  // [vocab] per call when set: q values torch would draw

    explicit HeartMuLaDraws(uint64_t seed = 0) : rng(seed) {}

    // Exp(1) for vocab entry i.
    float next(int64_t i) {
        if (injected) {
            return injected[i];
        }
        const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        return (float) -std::log1p(-u);
    }
};

static void heartmula_cfg_blend(const float * cond, const float * uncond, int64_t n, float scale, float * out) {
    for (int64_t i = 0; i < n; i++) {
        out[i] = uncond[i] + (cond[i] - uncond[i]) * scale;
    }
}

static int32_t heartmula_sample_topk(const float * logits, int64_t n, int topk, float temperature,
                                     HeartMuLaDraws & draws) {
    std::vector<float> l(logits, logits + n);
    for (auto & x : l) {
        x /= temperature;
    }
    const int64_t      k = std::clamp<int64_t>(topk, 1, n);
    std::vector<float> sorted(l);
    std::nth_element(sorted.begin(), sorted.begin() + (k - 1), sorted.end(), std::greater<float>());
    const float threshold = sorted[(size_t) (k - 1)];
    const float top       = *std::max_element(l.begin(), l.end());
    double      sum       = 0.0;
    for (int64_t i = 0; i < n; i++) {
        if (l[(size_t) i] >= threshold) {
            sum += std::exp((double) l[(size_t) i] - top);
        }
    }
    int32_t best  = 0;
    double  score = -1.0;
    for (int64_t i = 0; i < n; i++) {
        if (l[(size_t) i] < threshold) {
            continue;
        }
        const double p = std::exp((double) l[(size_t) i] - top) / sum;
        const double s = p / (double) draws.next(i);
        if (s > score) {
            score = s;
            best  = (int32_t) i;
        }
    }
    return best;
}
