#pragma once
// yue2/yue2-guide.h: source-audio guidance for the YuE2 semantic stage.
//
// Idea from comfyui_acetricks' TokenizerGuideLogitsProcessor. The real-audio
// tokenizer (MERT + Mothersuperior's head, yue2-mert.h / yue2-tok-head.h)
// reads a recording and scores every one of the 32768 semantic codes for each
// 40 ms frame. Its single best code is right only 16-19% of the time on
// YuE2's own songs, so forcing argmax codes into the LM (codec_ids) drags it
// off-distribution. The right code is usually somewhere in the head's top few
// dozen, though. A guide file keeps those top-K candidates per frame, and the
// decode loop uses them two ways:
//
//   scale    in [0, 1]: mix the two distributions, p = (1 - scale) * p_LM +
//            scale * p_tok, where p_tok is the tokenizer's distribution
//            renormalised over the candidates. Roughly the share of frames
//            that follow the source: 0 = the LM alone, 1 = the tokenizer alone
//            (non-candidates then get zero mass). Measured on a 20 s cot=off
//            render (2026-10-22), a log-space lerp instead did almost nothing
//            at 0.5 (0.6% top-1 agreement vs 0.0% unguided) and everything at
//            1.0, which made it a switch rather than a dial.
//   outside  additive penalty (nats, <= 0) on every legal code NOT in the
//            candidate set. 0 = no restriction; YUE2_GUIDE_HARD or below =
//            the candidates are the only codes the sampler can draw.
//
// Both act on the CFG'd logits row before the reference sampler chain, the
// same seam end_bias uses, so penalty/temperature/top-k/top-p run unchanged on
// the guided row. MUSIC_END is blocked while a guide frame is active unless
// the request allows it, so the song cannot stop before the source does.
//
// Frames line up one to one: the tokenizer and the semantic stage both run at
// 25 Hz, and the decode loop's step IS the frame index (a continued prefix
// carries its own frames into the count). offset_sec shifts the source
// against the output.
//
// The guide file is written by `ace-train yue2-guide` (train/yue2-guide-run.h),
// which owns the MERT/TF32 setup ace-server deliberately never touches.
//
// File layout, little-endian:
//   char[4] "Y2GD", u32 version (1), u32 T, u32 K,
//   i32 ids[T*K]  raw codes in [0, 32768), best first per frame
//   f32 lp[T*K]   the head's log-softmax over all 32768 codes for those ids

#include "hot-step-fsutf8.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static constexpr float YUE2_GUIDE_HARD = -100.0f;  // outside at or below this = mask

struct Yue2Guide {
    int64_t              T = 0, K = 0;
    std::vector<int32_t> ids;  // [T*K]
    std::vector<float>   lp;   // [T*K]
};

struct Yue2GuideParams {
    std::string path;                 // empty = off
    float       scale          = 0.5f;
    int         k              = 0;   // candidates used per frame; 0 = all the file holds
    float       outside        = 0.0f;
    float       offset_sec     = 0.0f;
    bool        allow_end      = false;
    // Scale envelope over the guided frames, then an LFO on top. Both multiply
    // `scale`; neither touches `outside`.
    std::string schedule;             // "" | up | down | sin | inv_sin
    float       schedule_power = 1.0f;
    std::string lfo;                  // "" | sine | triangle | saw | square
    float       lfo_period_sec = 4.0f;
    float       lfo_depth      = 0.0f;  // 0 = no modulation, 1 = full swing to 0
    float       lfo_phase      = 0.0f;  // fraction of a period
};

static bool yue2_guide_write(const std::string & path, const Yue2Guide & g, std::string * err) {
    FILE * f = hs_fopen(path, "wb");
    if (!f) {
        if (err) *err = "cannot write " + path;
        return false;
    }
    const uint32_t hdr[3] = { 1u, (uint32_t) g.T, (uint32_t) g.K };
    bool           ok     = fwrite("Y2GD", 1, 4, f) == 4 && fwrite(hdr, 4, 3, f) == 3 &&
              fwrite(g.ids.data(), 4, g.ids.size(), f) == g.ids.size() &&
              fwrite(g.lp.data(), 4, g.lp.size(), f) == g.lp.size();
    ok = (fclose(f) == 0) && ok;
    if (!ok && err) *err = "short write on " + path;
    return ok;
}

static bool yue2_guide_read(const std::string & path, Yue2Guide * g, std::string * err) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        if (err) *err = "guide file not found: " + path;
        return false;
    }
    char     magic[4] = {};
    uint32_t hdr[3]   = {};
    bool     ok       = fread(magic, 1, 4, f) == 4 && !memcmp(magic, "Y2GD", 4) && fread(hdr, 4, 3, f) == 3 &&
              hdr[0] == 1u && hdr[1] > 0 && hdr[2] > 0 && hdr[2] <= 32768;
    if (ok) {
        g->T = hdr[1];
        g->K = hdr[2];
        const size_t n = (size_t) g->T * (size_t) g->K;
        g->ids.resize(n);
        g->lp.resize(n);
        ok = fread(g->ids.data(), 4, n, f) == n && fread(g->lp.data(), 4, n, f) == n;
    }
    fclose(f);
    if (!ok && err) *err = "not a valid YuE2 guide file (Y2GD v1): " + path;
    return ok;
}

// The effective blend scale at guide frame `f` of `T`.
static float yue2_guide_scale_at(const Yue2GuideParams & p, int64_t f, int64_t T) {
    float s = p.scale;
    if (!p.schedule.empty()) {
        const double x = T > 1 ? (double) f / (double) (T - 1) : 1.0;
        double       e = 1.0;
        if (p.schedule == "up") e = x;
        else if (p.schedule == "down") e = 1.0 - x;
        else if (p.schedule == "sin") e = std::sin(x * 3.14159265358979323846);
        else if (p.schedule == "inv_sin") e = 1.0 - std::sin(x * 3.14159265358979323846);
        s *= (float) std::pow(std::max(0.0, e), (double) p.schedule_power);
    }
    if (!p.lfo.empty() && p.lfo_depth > 0.0f && p.lfo_period_sec > 0.0f) {
        const double t = std::fmod((double) f * 0.04 / (double) p.lfo_period_sec + (double) p.lfo_phase, 1.0);
        double       w = 1.0;  // waveform in [0, 1]
        if (p.lfo == "sine") w = 0.5 + 0.5 * std::sin(t * 6.28318530717958647692);
        else if (p.lfo == "triangle") w = 1.0 - 2.0 * std::fabs(t - 0.5);
        else if (p.lfo == "saw") w = t;
        else if (p.lfo == "square") w = t < 0.5 ? 1.0 : 0.0;
        s *= (float) (1.0 - (double) p.lfo_depth + (double) p.lfo_depth * w);
    }
    return s;
}

// Guide one CFG'd logits row in place. `row` holds vocab ids [base, base+W);
// legal codes are [legal_lo, legal_hi) (absolute), raw code c is token
// legal_lo + c. `step` is the semantic frame about to be drawn. Returns true
// when a guide frame was applied.
static bool yue2_guide_apply(float * row, int64_t W, int64_t base, int64_t legal_lo, int64_t legal_hi, int64_t end_id,
                             const Yue2Guide & g, const Yue2GuideParams & p, int64_t step) {
    const int64_t f = step + (int64_t) std::llround((double) p.offset_sec * 25.0);
    if (f < 0 || f >= g.T) return false;
    const int64_t lo = legal_lo - base, hi = legal_hi - base;
    if (lo < 0 || hi > W || lo >= hi) return false;
    const int64_t k     = p.k > 0 ? std::min<int64_t>(p.k, g.K) : g.K;
    const float   scale = yue2_guide_scale_at(p, f, g.T);

    // The LM's log-partition over the legal codes: lp_model = row - lse.
    float mx = -INFINITY;
    for (int64_t v = lo; v < hi; v++) mx = std::max(mx, row[v]);
    if (!std::isfinite(mx)) return false;
    double z = 0.0;
    for (int64_t v = lo; v < hi; v++) {
        if (std::isfinite(row[v])) z += std::exp((double) (row[v] - mx));
    }
    const float lse = mx + (float) std::log(z);

    const int32_t * ids = g.ids.data() + (size_t) f * (size_t) g.K;
    const float *   glp = g.lp.data() + (size_t) f * (size_t) g.K;

    // The tokenizer's distribution renormalised over the k candidates.
    double gmx = -INFINITY, gz = 0.0;
    for (int64_t j = 0; j < k; j++) gmx = std::max(gmx, (double) glp[j]);
    for (int64_t j = 0; j < k; j++) gz += std::exp((double) glp[j] - gmx);

    // Candidates: log((1-s) p_LM + s p_tok) + lse, from the untouched row.
    const double s = std::min(1.0, std::max(0.0, (double) scale));
    std::vector<float> guided((size_t) k, NAN);
    for (int64_t j = 0; j < k; j++) {
        const int64_t v = lo + ids[j];
        if (v < lo || v >= hi) continue;
        const double p_lm  = std::isfinite(row[v]) ? std::exp((double) (row[v] - lse)) : 0.0;
        const double p_tok = std::exp((double) glp[j] - gmx) / gz;
        const double mix   = (1.0 - s) * p_lm + s * p_tok;
        guided[(size_t) j] = mix > 0.0 ? (float) std::log(mix) + lse : -INFINITY;
    }
    // Everything else keeps (1-s) of its mass, minus the outside penalty.
    const float keep = s < 1.0 ? (float) std::log(1.0 - s) : -INFINITY;
    const float pen  = p.outside <= YUE2_GUIDE_HARD ? -INFINITY : std::min(0.0f, p.outside);
    if (keep != 0.0f || pen != 0.0f) {
        for (int64_t v = lo; v < hi; v++) row[v] += keep + pen;
    }
    for (int64_t j = 0; j < k; j++) {
        if (!std::isnan(guided[(size_t) j])) row[lo + ids[j]] = guided[(size_t) j];
    }
    const int64_t e = end_id - base;
    if (!p.allow_end && e >= 0 && e < W) row[e] = -INFINITY;
    return true;
}
