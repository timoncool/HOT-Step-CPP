// yue2-train-gemm-precision-test.cpp -- the S2a training-GEMM precision gate.
//
// Builds yue2_aitk_graph::linear() (and the head-loss GEMMs) at YuE2 shapes,
// forward and autodiff backward, once on the BF16 tensor-core path and once
// with the strict-F32 rollback, on the first GPU. Every tensor must be
// finite; the BF16 path must stay within the frozen S2a limits of strict
// (RESEARCH/YUE2_S2A_TOLERANCES.md gate 3): forward 0.5% rel-L2, input
// gradient 1%, each trainable factor's gradient 2%. It also checks which
// path each build resolved to (yue2_aitk_graph::gemm_counts()).
//
// Exit codes: 0 pass, 1 a check failed, 2 setup error.

#include "../src/train/yue2-aitk-graph.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int64_t kHidden = 2048, kQkv = 4096, kRank = 32, kSeq = 384, kVocab = 8192, kLatent = 64;

std::vector<float> rand_vec(size_t n, float scale, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (float & x : v) x = d(rng);
    return v;
}

// One case: builds its tensors and output in `ctx`; returns the scalar loss
// input `y` and lists the leaves to fill and the gradients to compare.
struct Leaf { ggml_tensor * t; float scale; bool trainable; std::string name; };
struct Built { ggml_tensor * y = nullptr; ggml_tensor * x = nullptr; std::vector<Leaf> leaves; bool backward = true; };
using Builder = std::function<Built(ggml_context *)>;

ggml_tensor * leaf(ggml_context * ctx, Built & b, ggml_type type, int64_t n0, int64_t n1, float scale,
                   bool trainable, const char * name) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, n0, n1);
    ggml_set_input(t);
    if (trainable) ggml_set_param(t);
    b.leaves.push_back({t, scale, trainable, name});
    return t;
}

struct Run {
    std::vector<float> y;
    std::vector<std::vector<float>> grads;  // x first, then each trainable leaf
    yue2_aitk_graph::GemmCounts counts;
};

bool run(ggml_backend_t backend, const Builder & build, bool strict, Run * out) {
    yue2_aitk_graph::strict_f32() = strict;
    const yue2_aitk_graph::GemmCounts before = yue2_aitk_graph::gemm_counts();
    ggml_init_params ip{}; ip.mem_size = 64u * 1024u * 1024u; ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    Built b = build(ctx);
    out->counts.bf16 = yue2_aitk_graph::gemm_counts().bf16 - before.bf16;
    out->counts.f32 = yue2_aitk_graph::gemm_counts().f32 - before.f32;
    for (const Leaf & l : b.leaves) ggml_set_input(l.t);
    ggml_set_output(b.y);
    ggml_tensor * target = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, b.y->ne[0], b.y->ne[1]);
    ggml_set_input(target);
    ggml_tensor * loss = ggml_sum(ctx, ggml_mul(ctx, b.y, target));
    ggml_cgraph * g = ggml_new_graph_custom(ctx, 4096, true);
    ggml_build_forward_expand(g, loss);
    if (b.backward) { ggml_set_loss(loss); ggml_build_backward_expand(ctx, g, nullptr); }
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = alloc && ggml_gallocr_alloc_graph(alloc, g);
    uint32_t seed = 1234;
    if (ok) {
        for (const Leaf & l : b.leaves) {
            std::vector<float> v = rand_vec(size_t(ggml_nelements(l.t)), l.scale, seed++);
            if (l.t->type == GGML_TYPE_BF16) {
                std::vector<ggml_bf16_t> h(v.size());
                ggml_fp32_to_bf16_row(v.data(), h.data(), int64_t(v.size()));
                ggml_backend_tensor_set(l.t, h.data(), 0, ggml_nbytes(l.t));
            } else {
                ggml_backend_tensor_set(l.t, v.data(), 0, ggml_nbytes(l.t));
            }
        }
        std::vector<float> tv = rand_vec(size_t(ggml_nelements(target)), 1.0f, 999);
        ggml_backend_tensor_set(target, tv.data(), 0, ggml_nbytes(target));
        if (b.backward) ggml_graph_reset(g);
        ok = ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        out->y.resize(size_t(ggml_nelements(b.y)));
        ggml_backend_tensor_get(b.y, out->y.data(), 0, ggml_nbytes(b.y));
        if (b.backward) {
            std::vector<ggml_tensor *> want{b.x};
            for (const Leaf & l : b.leaves) if (l.trainable && l.t != b.x) want.push_back(l.t);
            for (ggml_tensor * t : want) {
                ggml_tensor * gt = ggml_graph_get_grad(g, t);
                if (!gt) { ok = false; break; }
                out->grads.emplace_back(size_t(ggml_nelements(gt)));
                ggml_backend_tensor_get(gt, out->grads.back().data(), 0, ggml_nbytes(gt));
            }
        }
    }
    if (alloc) ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}

bool finite(const std::vector<float> & v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}
double rel_l2(const std::vector<float> & a, const std::vector<float> & ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double d = double(a[i]) - ref[i]; num += d * d; den += double(ref[i]) * ref[i]; }
    return den > 0 ? std::sqrt(num / den) : (num > 0 ? INFINITY : 0.0);
}

Yue2AitkConvRotLinear bf16_base(ggml_context * ctx, Built & b, int64_t in, int64_t out) {
    Yue2AitkConvRotLinear w; w.cols = in; w.rows = out;
    w.parts.push_back(leaf(ctx, b, GGML_TYPE_BF16, in, out, 0.02f, false, "base"));
    return w;
}
ggml_tensor * input(ggml_context * ctx, Built & b, int64_t in) {
    // The real x is the output of a BF16 round node; a trainable leaf stands
    // in for whatever produced it so its gradient can be compared.
    ggml_tensor * p = leaf(ctx, b, GGML_TYPE_F32, in, kSeq, 1.0f, true, "x");
    b.x = p;
    return yue2_aitk_graph::round(ctx, p);
}
Yue2AitkFusedLora lora(ggml_context * ctx, Built & b, int64_t in, int64_t out) {
    Yue2AitkFusedLora s; s.input_width = in; s.output_width = out; s.rank = kRank; s.scale = 1.0f;
    s.a = leaf(ctx, b, GGML_TYPE_F32, in, kRank, 0.02f, true, "lora.A");
    s.b = leaf(ctx, b, GGML_TYPE_F32, kRank, out, 0.02f, true, "lora.B");
    return s;
}

} // namespace

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { std::puts("yue2-train-gemm-precision-test: no GPU device"); return 2; }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { std::puts("yue2-train-gemm-precision-test: backend init failed"); return 2; }
    std::printf("device %s\n", ggml_backend_dev_name(dev));

    struct Case { const char * name; Builder build; int bf16_gemms; int f32_gemms; };  // fast-path counts
    std::vector<Case> cases;
    cases.push_back({"bf16 base + LoRA (qkv)", [](ggml_context * ctx) {
        Built b; ggml_tensor * x = input(ctx, b, kHidden);
        Yue2AitkConvRotLinear w = bf16_base(ctx, b, kHidden, kQkv);
        Yue2AitkFusedLora s = lora(ctx, b, kHidden, kQkv);
        b.y = yue2_aitk_graph::linear(ctx, w, x, &s); return b; }, 3, 0});
    cases.push_back({"bf16 base + LoKr dim 128 f4 (qkv)", [](ggml_context * ctx) {
        Built b; ggml_tensor * x = input(ctx, b, kHidden);
        Yue2AitkConvRotLinear w = bf16_base(ctx, b, kHidden, kQkv);
        Yue2AitkFusedLora s; std::vector<ggml_tensor *> params;
        if (!yue2_aitk_detail::make_lokr_site(ctx, kHidden, kQkv, 128, 4, 64.0f, "lokr", &s, &params)) std::abort();
        for (ggml_tensor * t : params) b.leaves.push_back({t, 0.05f, true, ggml_get_name(t)});
        b.y = yue2_aitk_graph::linear(ctx, w, x, &s); return b; }, 2, 0});
    cases.push_back({"bf16 base + companion + LoRA (qkv)", [](ggml_context * ctx) {
        Built b; ggml_tensor * x = input(ctx, b, kHidden);
        Yue2AitkConvRotLinear w = bf16_base(ctx, b, kHidden, kQkv);
        w.frozen_a = leaf(ctx, b, GGML_TYPE_F32, kHidden, 64, 0.02f, false, "comp.A");
        w.frozen_b = leaf(ctx, b, GGML_TYPE_F32, 64, kQkv, 0.02f, false, "comp.B");
        Yue2AitkFusedLora s = lora(ctx, b, kHidden, kQkv);
        b.y = yue2_aitk_graph::linear(ctx, w, x, &s); return b; }, 3, 2});
    cases.push_back({"companion dense llm2vae + bias", [](ggml_context * ctx) {
        Built b; ggml_tensor * x = input(ctx, b, kHidden);
        Yue2AitkConvRotLinear w; w.cols = kHidden; w.rows = kLatent;
        w.dense = leaf(ctx, b, GGML_TYPE_F32, kHidden, kLatent, 0.02f, false, "dense");
        ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kLatent);
        b.leaves.push_back({bias, 0.1f, false, "bias"});
        b.y = yue2_aitk_graph::linear(ctx, w, x, nullptr, bias); return b; }, 0, 1});
    cases.push_back({"head logits (GGUF head, forward)", [](ggml_context * ctx) {
        Built b; b.backward = false; ggml_tensor * x = input(ctx, b, kHidden);
        ggml_tensor * head = leaf(ctx, b, GGML_TYPE_BF16, kHidden, kVocab, 0.02f, false, "head");
        b.y = ggml_bf16_round(ctx, yue2_aitk_graph::mm(ctx, head, x)); return b; }, 1, 0});
    cases.push_back({"head hidden gradient (dense_t, forward)", [](ggml_context * ctx) {
        Built b; b.backward = false;
        ggml_tensor * g = leaf(ctx, b, GGML_TYPE_F32, kVocab, kSeq, 1e-3f, false, "logit_grad");
        b.x = g;
        ggml_tensor * head_t = leaf(ctx, b, GGML_TYPE_BF16, kVocab, kHidden, 0.02f, false, "head_t");
        b.y = yue2_aitk_graph::mm(ctx, head_t, ggml_bf16_round(ctx, g)); return b; }, 1, 0});

    bool all = true;
    for (const Case & c : cases) {
        Run strict, fast, again;
        if (!run(backend, c.build, true, &strict) || !run(backend, c.build, false, &fast) ||
            !run(backend, c.build, true, &again)) {
            std::printf("[FAIL] %s: graph build or compute failed\n", c.name); all = false; continue;
        }
        bool ok = true;
        std::string why;
        // LoKr counts one site per linear; every other GEMM counts itself.
        // The frozen companion stays F32 on both paths (mm_f32).
        if (fast.counts.bf16 != c.bf16_gemms || fast.counts.f32 != c.f32_gemms) { ok = false; why += " fast-path-counts"; }
        if (strict.counts.f32 != c.bf16_gemms + c.f32_gemms || strict.counts.bf16 != 0) { ok = false; why += " strict-path-not-f32"; }
        // No BF16 GEMM at all: the two paths must build the same graph.
        if (c.bf16_gemms == 0 && rel_l2(fast.y, strict.y) != 0.0) { ok = false; why += " f32-only-case-differs"; }
        if (!finite(strict.y) || !finite(fast.y)) { ok = false; why += " non-finite-y"; }
        // Control: strict twice must agree, or the harness itself is broken.
        if (rel_l2(again.y, strict.y) != 0.0) { ok = false; why += " strict-not-repeatable"; }
        const double ry = rel_l2(fast.y, strict.y);
        if (!(ry <= 0.005)) { ok = false; why += " y"; }
        std::printf("       %s: fast bf16 %lld f32 %lld / strict f32 %lld | y rel-L2 %.3e",
                    c.name, (long long) fast.counts.bf16, (long long) fast.counts.f32, (long long) strict.counts.f32, ry);
        for (size_t i = 0; i < strict.grads.size() && i < fast.grads.size(); ++i) {
            const double r = rel_l2(fast.grads[i], strict.grads[i]);
            const double lim = i == 0 ? 0.01 : 0.02;
            if (!finite(strict.grads[i]) || !finite(fast.grads[i])) { ok = false; why += " non-finite-grad"; }
            if (!(r <= lim)) { ok = false; why += " grad" + std::to_string(i); }
            std::printf(" | %s %.3e", i == 0 ? "dx" : "dW", r);
        }
        if (strict.grads.size() != fast.grads.size()) { ok = false; why += " grad-count"; }
        std::printf("\n%s %s%s\n", ok ? "[PASS]" : "[FAIL]", c.name, why.c_str());
        all &= ok;
    }
    ggml_backend_free(backend);
    std::puts(all ? "yue2-train-gemm-precision-test: ALL PASS" : "yue2-train-gemm-precision-test: FAIL");
    return all ? 0 : 1;
}
