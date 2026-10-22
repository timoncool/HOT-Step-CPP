// HOTSTEP-347 range audit for fix 2 (half-staged BF16 GEMM). Probe only.
//
// Force-included (-include) when the probe compiles an unchanged copy of
// engine/src/train/yue2-aitk-runtime.cpp; never part of a production build.
// It redirects the trainer's ggml_backend_graph_compute(_async) calls. With
// HS347_AUDIT_OUT set and a Metal backend, each graph runs in chunks that end
// before every MUL_MAT the Metal simdgroup kernel (mul_mm) would take, so the
// operands that kernel stages into threadgroup tiles can be read once their
// producers have finished. Every element is classified as the tile would see it:
//   BF16 kernel, src1: F32 -> BF16 (RNE, the kernel's (bfloat) cast) -> half
//   BF16 kernel, src0: BF16 -> half
//   F32 kernel (today's companion path): F32 -> half directly
// Host conversions use arm64 _Float16 (RNE, FZ16 off). Whether the GPU keeps
// half subnormals is the separate metal-half-probe.swift question.
//
// Chunking hides later consumers from ggml-metal's single-use fusion check, so
// run with GGML_METAL_FUSION_DISABLE=1 GGML_METAL_NO_BF16_ROUND_PRODUCER=1
// (bit-identical to the fused path, HOTSTEP-345) and check the losses against
// the frozen baseline bits.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"  // ggml_cgraph::grads: the "training graph" signal F would use

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace hs347 {

struct Counts {
    uint64_t elems = 0, nonfinite = 0, bf16_inf = 0, zero = 0, not_bf16 = 0;
    uint64_t normal = 0, sub_exact = 0, sub_rounded = 0, flushed = 0, rounded_normal = 0, overflow = 0;
    double max_abs = 0, min_nonzero = INFINITY, sum_abs = 0, max_delta = 0, sum_delta = 0;
    uint64_t hist[256] = {};  // biased F32 exponent of the staged (pre-half) value; 0 = zero/subnormal F32
    void add(const Counts & o) {
        elems += o.elems; nonfinite += o.nonfinite; bf16_inf += o.bf16_inf; zero += o.zero; not_bf16 += o.not_bf16;
        normal += o.normal; sub_exact += o.sub_exact; sub_rounded += o.sub_rounded; flushed += o.flushed;
        rounded_normal += o.rounded_normal; overflow += o.overflow;
        max_abs = std::max(max_abs, o.max_abs); min_nonzero = std::min(min_nonzero, o.min_nonzero);
        sum_abs += o.sum_abs; max_delta = std::max(max_delta, o.max_delta); sum_delta += o.sum_delta;
        for (int i = 0; i < 256; ++i) hist[i] += o.hist[i];
    }
    uint64_t changed() const { return sub_rounded + flushed + rounded_normal + overflow; }
};

// Tile steps: one threadgroup's 32-wide K block, staging a 64-row src0 tile and a
// 32-row src1 tile (kernel_mul_mm, NR0 = 64, NR1 = 32, NK = 32). A guard that
// stays bit-exact must fall back to bfloat for a step if either tile holds a
// value that half cannot represent exactly.
struct Family {
    uint64_t mm_nodes = 0, mv_nodes = 0, src0_scans = 0;
    double flops = 0, worst_node_changed_frac = 0;
    double steps = 0, steps_bad_any = 0, steps_bad_src0 = 0, steps_bad_src1 = 0;
    Counts src0, src1;
    std::vector<std::string> outliers;
};

struct State {
    std::mutex mu;
    std::map<std::string, Family> families;
    struct Tiles { std::vector<uint32_t> bad_per_k; int64_t row_blocks = 0; };
    // Frozen weights (and their transposed copies) by root, view, orientation and shape.
    std::map<std::tuple<const void *, const void *, bool, int64_t, int64_t>, Tiles> src0_tiles;
    uint64_t calls = 0, metal_calls = 0, chunks = 0, audited = 0, skipped = 0;
    std::chrono::steady_clock::time_point last_dump = std::chrono::steady_clock::now();
};
inline State & state() { static State s; return s; }
inline const char * out_path() { static const char * p = std::getenv("HS347_AUDIT_OUT"); return p && *p ? p : nullptr; }

inline uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float from_bits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
// ggml_compute_fp32_to_bf16 (RNE; NaN stays NaN).
inline float bf16_rne(float x) {
    uint32_t u = bits(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;
    u += 0x7fffu + ((u >> 16) & 1u);
    return from_bits(u & 0xffff0000u);
}

// Returns true when the staged value is not exactly representable in half.
inline bool classify(float x, bool round_to_bf16, Counts & c) {
    ++c.elems;
    float v = x;
    if (round_to_bf16) {
        v = bf16_rne(x);
        if (std::isfinite(x) && bits(v) != bits(x)) ++c.not_bf16;
        if (std::isfinite(x) && std::isinf(v)) ++c.bf16_inf;
    }
    if (!std::isfinite(v)) { ++c.nonfinite; return true; }
    ++c.hist[(bits(v) >> 23) & 0xffu];
    if (v == 0.0f) { ++c.zero; return false; }
    const double a = std::fabs(v);
    c.max_abs = std::max(c.max_abs, a); c.min_nonzero = std::min(c.min_nonzero, a); c.sum_abs += a;
    const float back = static_cast<float>(static_cast<_Float16>(v));
    if (std::isinf(back)) { ++c.overflow; return true; }
    if (back == v) { a >= 0x1p-14 ? ++c.normal : ++c.sub_exact; return false; }
    const double d = std::fabs(double(back) - double(v));
    c.max_delta = std::max(c.max_delta, d); c.sum_delta += d;
    if (back == 0.0f) ++c.flushed;
    else if (a < 0x1p-14 || std::fabs(back) < 0x1p-14f) ++c.sub_rounded;
    else ++c.rounded_normal;
    return true;
}

// Rows of ne0 contiguous elements, walked by stride; threads take blocks of
// tile_rows rows (per ne2/ne3 batch). tiles->bad_per_k[kb] counts the row blocks
// whose 32-wide K block kb holds at least one inexact value.
inline bool scan(const ggml_tensor * t, bool round_to_bf16, Counts & out, int64_t tile_rows, State::Tiles * tiles) {
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_BF16) return false;
    if (t->nb[0] != ggml_type_size(t->type)) return false;
    const size_t span = ggml_nbytes(t);
    std::vector<char> copy;
    const char * base = static_cast<const char *>(t->data);
    if (!t->buffer || !ggml_backend_buffer_is_host(t->buffer)) {
        copy.resize(span);
        ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), copy.data(), 0, span);
        base = copy.data();
    }
    const int64_t nk = (t->ne[0] + 31) / 32, rb1 = (t->ne[1] + tile_rows - 1) / tile_rows;
    const int64_t blocks = rb1 * t->ne[2] * t->ne[3];
    const int nt = std::max(1, std::min<int>(int(std::thread::hardware_concurrency()), int(blocks)));
    std::vector<Counts> part(nt);
    std::vector<std::vector<uint32_t>> bad(nt, std::vector<uint32_t>(nk, 0));
    std::vector<std::thread> pool;
    for (int w = 0; w < nt; ++w) pool.emplace_back([&, w] {
        std::vector<char> flag(nk);
        for (int64_t b = w; b < blocks; b += nt) {
            std::fill(flag.begin(), flag.end(), 0);
            const int64_t ib = b % rb1, i2 = (b / rb1) % t->ne[2], i3 = b / (rb1 * t->ne[2]);
            for (int64_t i1 = ib * tile_rows; i1 < std::min(t->ne[1], (ib + 1) * tile_rows); ++i1) {
                const char * row = base + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                if (t->type == GGML_TYPE_F32) {
                    const float * p = reinterpret_cast<const float *>(row);
                    for (int64_t i = 0; i < t->ne[0]; ++i) if (classify(p[i], round_to_bf16, part[w])) flag[i / 32] = 1;
                } else {
                    const uint16_t * p = reinterpret_cast<const uint16_t *>(row);
                    for (int64_t i = 0; i < t->ne[0]; ++i) if (classify(from_bits(uint32_t(p[i]) << 16), false, part[w])) flag[i / 32] = 1;
                }
            }
            for (int64_t k = 0; k < nk; ++k) bad[w][k] += flag[k];
        }
    });
    for (auto & th : pool) th.join();
    for (auto & p : part) out.add(p);
    if (tiles) {
        tiles->row_blocks = blocks;
        tiles->bad_per_k.assign(nk, 0);
        for (int w = 0; w < nt; ++w) for (int64_t k = 0; k < nk; ++k) tiles->bad_per_k[k] += bad[w][k];
    }
    return true;
}

inline std::string normalize(const char * name) {
    std::string s;
    for (const char * p = name; *p; ++p) {
        if (*p >= '0' && *p <= '9') { if (s.empty() || s.back() != '#') s += '#'; }
        else s += *p;
    }
    return s.empty() ? "(unnamed)" : s;
}

// ggml_metal_op_mul_mat_use_mm on a device with simdgroup mm (M2: yes).
inline bool metal_use_mm(const ggml_tensor * n) {
    return !ggml_is_transposed(n->src[0]) && !ggml_is_transposed(n->src[1]) && n->src[0]->ne[0] >= 64 && n->src[1]->ne[1] > 8;
}

inline void audit(const ggml_tensor * n, bool train_graph, uint64_t call) {
    const ggml_tensor * s0 = n->src[0], * s1 = n->src[1];
    const ggml_tensor * root = s0;
    const void * view = nullptr;
    bool transposed = false;
    while (root->src[0] && (root->op == GGML_OP_VIEW || root->op == GGML_OP_RESHAPE || root->op == GGML_OP_PERMUTE ||
                            root->op == GGML_OP_TRANSPOSE || root->op == GGML_OP_CONT || root->op == GGML_OP_CPY)) {
        transposed |= root->op == GGML_OP_TRANSPOSE;
        if (root->op == GGML_OP_VIEW) view = root->data;
        root = root->src[0];
    }
    const bool frozen = root->op == GGML_OP_NONE && !(root->flags & GGML_TENSOR_FLAG_PARAM);
    const bool bf16 = s0->type == GGML_TYPE_BF16;
    std::string name = normalize(root->name);
    if (name == "(unnamed)" || name.rfind("leaf_", 0) == 0 || name.rfind("node_", 0) == 0)
        name = "unnamed [" + std::to_string(root->ne[0]) + "," + std::to_string(root->ne[1]) + "]";
    const std::string key = std::string(bf16 ? "bf16" : "f32") + "|" + (transposed ? "dX" : "fwd") + "|" + name +
                            (frozen ? "" : " (trainable)") + "|" + (train_graph ? "grads" : "no-grads");
    const bool mm = metal_use_mm(n);
    Counts c0, c1;
    State::Tiles t0, t1;
    bool counted0 = false;
    if (mm) {
        // Frozen weights: element counts once (a transposed copy holds the same values),
        // tile counts once per orientation and view.
        const auto tkey = std::make_tuple(static_cast<const void *>(root->data), view, transposed, s0->ne[0], s0->ne[1]);
        bool scan0 = true;
        {
            auto & st = state();
            std::lock_guard<std::mutex> lock(st.mu);
            auto it = frozen ? st.src0_tiles.find(tkey) : st.src0_tiles.end();
            if (it != st.src0_tiles.end()) { t0 = it->second; scan0 = false; }
        }
        if (scan0) {
            if (!scan(s0, false, c0, 64, &t0)) { std::lock_guard<std::mutex> lock(state().mu); ++state().skipped; return; }
            counted0 = !transposed;
            if (frozen) { std::lock_guard<std::mutex> lock(state().mu); state().src0_tiles[tkey] = t0; }
        }
        if (!scan(s1, bf16, c1, 32, &t1)) { std::lock_guard<std::mutex> lock(state().mu); ++state().skipped; return; }
    }
    auto & st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    Family & f = st.families[key];
    if (!mm) { ++f.mv_nodes; return; }
    ++st.audited; ++f.mm_nodes;
    if (t0.bad_per_k.size() == t1.bad_per_k.size()) {
        const double n0 = double(t0.row_blocks), n1 = double(t1.row_blocks);
        for (size_t k = 0; k < t0.bad_per_k.size(); ++k) {
            const double a = t0.bad_per_k[k], b = t1.bad_per_k[k];
            f.steps += n0 * n1;
            f.steps_bad_any += n0 * n1 - (n0 - a) * (n1 - b);
            f.steps_bad_src0 += a * n1;
            f.steps_bad_src1 += b * n0;
        }
    } else ++st.skipped;
    f.flops += 2.0 * double(s0->ne[0]) * double(s0->ne[1]) * double(s1->ne[1]) * double(s1->ne[2]) * double(s1->ne[3]);
    if (counted0) { ++f.src0_scans; f.src0.add(c0); }
    f.src1.add(c1);
    const double frac = c1.elems ? double(c1.changed()) / double(c1.elems) : 0.0;
    f.worst_node_changed_frac = std::max(f.worst_node_changed_frac, frac);
    if ((c1.overflow || c1.nonfinite || c1.bf16_inf || c0.overflow || c0.nonfinite) && f.outliers.size() < 32) {
        char buf[512];
        std::snprintf(buf, sizeof buf, "call %llu node '%s' src1 [%lld,%lld] overflow %llu nonfinite %llu bf16_inf %llu max_abs %.6g; src0 overflow %llu nonfinite %llu",
                      (unsigned long long) call, n->name, (long long) s1->ne[0], (long long) s1->ne[1],
                      (unsigned long long) c1.overflow, (unsigned long long) c1.nonfinite, (unsigned long long) c1.bf16_inf, c1.max_abs,
                      (unsigned long long) c0.overflow, (unsigned long long) c0.nonfinite);
        f.outliers.push_back(buf);
    }
}

inline void write_counts(FILE * o, const Counts & c) {
    std::fprintf(o, "{\"elems\":%llu,\"nonfinite\":%llu,\"bf16_inf\":%llu,\"zero\":%llu,\"not_bf16\":%llu,\"normal\":%llu,"
                    "\"sub_exact\":%llu,\"sub_rounded\":%llu,\"flushed\":%llu,\"rounded_normal\":%llu,\"overflow\":%llu,"
                    "\"max_abs\":%.9g,\"min_nonzero\":%.9g,\"sum_abs\":%.9g,\"max_delta\":%.9g,\"sum_delta\":%.9g,\"log2_hist\":{",
                 (unsigned long long) c.elems, (unsigned long long) c.nonfinite, (unsigned long long) c.bf16_inf,
                 (unsigned long long) c.zero, (unsigned long long) c.not_bf16, (unsigned long long) c.normal,
                 (unsigned long long) c.sub_exact, (unsigned long long) c.sub_rounded, (unsigned long long) c.flushed,
                 (unsigned long long) c.rounded_normal, (unsigned long long) c.overflow,
                 c.max_abs, std::isinf(c.min_nonzero) ? 0.0 : c.min_nonzero, c.sum_abs, c.max_delta, c.sum_delta);
    bool first = true;
    for (int e = 0; e < 256; ++e) if (c.hist[e]) {
        // key: floor(log2|v|) for normal F32 values; "zero" for 0 (no F32 subnormals reach here in practice)
        if (e == 0) std::fprintf(o, "%s\"zero_or_f32_subnormal\":%llu", first ? "" : ",", (unsigned long long) c.hist[e]);
        else std::fprintf(o, "%s\"%d\":%llu", first ? "" : ",", e - 127, (unsigned long long) c.hist[e]);
        first = false;
    }
    std::fprintf(o, "}}");
}

inline void dump() {
    const char * path = out_path();
    if (!path) return;
    auto & st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    const std::string tmp = std::string(path) + ".tmp";
    FILE * o = std::fopen(tmp.c_str(), "w");
    if (!o) return;
    std::fprintf(o, "{\"calls\":%llu,\"metal_calls\":%llu,\"chunks\":%llu,\"audited_mm_nodes\":%llu,\"skipped\":%llu,\"families\":{",
                 (unsigned long long) st.calls, (unsigned long long) st.metal_calls, (unsigned long long) st.chunks,
                 (unsigned long long) st.audited, (unsigned long long) st.skipped);
    bool first = true;
    for (const auto & [key, f] : st.families) {
        std::fprintf(o, "%s\n\"%s\":{\"mm_nodes\":%llu,\"mv_nodes\":%llu,\"src0_scans\":%llu,\"flops\":%.6g,\"worst_node_changed_frac\":%.9g,"
                        "\"steps\":%.6g,\"steps_bad_any\":%.6g,\"steps_bad_src0\":%.6g,\"steps_bad_src1\":%.6g,\"src0\":",
                     first ? "" : ",", key.c_str(), (unsigned long long) f.mm_nodes, (unsigned long long) f.mv_nodes,
                     (unsigned long long) f.src0_scans, f.flops, f.worst_node_changed_frac,
                     f.steps, f.steps_bad_any, f.steps_bad_src0, f.steps_bad_src1);
        write_counts(o, f.src0);
        std::fprintf(o, ",\"src1\":");
        write_counts(o, f.src1);
        std::fprintf(o, ",\"outliers\":[");
        for (size_t i = 0; i < f.outliers.size(); ++i) std::fprintf(o, "%s\"%s\"", i ? "," : "", f.outliers[i].c_str());
        std::fprintf(o, "]}");
        first = false;
    }
    std::fprintf(o, "\n}}\n");
    std::fclose(o);
    std::rename(tmp.c_str(), path);
    st.last_dump = std::chrono::steady_clock::now();
}

inline bool is_metal(ggml_backend_t b) {
    ggml_backend_dev_t dev = b ? ggml_backend_get_device(b) : nullptr;
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg && std::strcmp(ggml_backend_reg_name(reg), "MTL") == 0;
}

// Both trainer entry points land here; the async one runs synchronously.
inline ggml_status compute(ggml_backend_t b, ggml_cgraph * g) {
    auto & st = state();
    uint64_t call;
    { std::lock_guard<std::mutex> lock(st.mu); call = ++st.calls; }
    if (!out_path() || !is_metal(b)) return ggml_backend_graph_compute(b, g);
    { std::lock_guard<std::mutex> lock(st.mu); ++st.metal_calls; }
    const bool train_graph = g->grads != nullptr;
    int start = 0;
    for (int i = 0; i < g->n_nodes; ++i) {
        const ggml_tensor * n = g->nodes[i];
        if (n->op != GGML_OP_MUL_MAT || n->src[1]->type != GGML_TYPE_F32 ||
            (n->src[0]->type != GGML_TYPE_BF16 && n->src[0]->type != GGML_TYPE_F32)) continue;
        if (metal_use_mm(n) && i > start) {
            ggml_cgraph view = ggml_graph_view(g, start, i);
            const ggml_status s = ggml_backend_graph_compute(b, &view);
            if (s != GGML_STATUS_SUCCESS) return s;
            { std::lock_guard<std::mutex> lock(st.mu); ++st.chunks; }
            start = i;
        }
        audit(n, train_graph, call);
    }
    ggml_cgraph view = ggml_graph_view(g, start, g->n_nodes);
    const ggml_status s = ggml_backend_graph_compute(b, &view);
    bool due;
    { std::lock_guard<std::mutex> lock(st.mu); ++st.chunks; due = std::chrono::steady_clock::now() - st.last_dump > std::chrono::seconds(60); }
    if (due) dump();
    return s;
}

}  // namespace hs347

#define ggml_backend_graph_compute(b, g)       ::hs347::compute((b), (g))
#define ggml_backend_graph_compute_async(b, g) ::hs347::compute((b), (g))
