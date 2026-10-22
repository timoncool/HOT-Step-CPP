// test-yue2-aitk-dora.cpp — YuE2 joint/AITK DoRA graph fixture.
//
// Drives the real train/yue2-aitk-graph.h builders (linear, linear_split,
// dora_norm_exact, dora_norm_expanded) on small shapes, in both precision
// modes (bf16 default and strict-f32):
//   1. m == nrm: the DoRA site's forward and every gradient (x, A, B) are bit
//      for bit the LoRA site's, fused and split. This is the init state
//      (m starts at the exact norm), so a fresh DoRA run starts as LoRA.
//   2. m != nrm: the output is the LoRA output times m / nrm (one BF16 round),
//      dL/dm matches the host formula, and A/B take the LoRA gradients of the
//      rescaled upstream. The norm is a constant: no gradient reaches it.
//   3. Row norms over a three-part base (q|k|v) plus a frozen companion
//      delta: exact (identity pass) and expanded (rank-r) agree with a double
//      host reference, and an all-zero row lands on the floor, not 0/0.
//   4. yue2-adapter.h's loader, on CPU over a 2x2 identity base: a v1 DoRA
//      file merges W' = m (W0 + s BA) / ||.||_row at the alpha/rank the merge
//      uses (AR and NAR halves); an unknown DoRA version, and a numeric
//      dora_scaling that disagrees with the resolved alpha over the tensor
//      rank, are refused with the base untouched; plain LoRA still merges.
// The ConvRot8 base is not built here (no small fixture for its weights); the
// trainer's YUE2_AITK_DORA_CHECK=1 compares both norm paths on it. CPU and
// Vulkan run case 3 only (see main).
//
// Build: configure engine/ with -DBUILD_TESTING=ON, target yue2-aitk-dora-test.
// Backend: GGML_BACKEND=CUDA0 / Vulkan0 / MTL0 / CPU, else ggml's best.

#include <string>  // backend.h relies on its includer for std::string

#include "backend.h"
#include "train/yue2-aitk-graph.h"
#include "yue2/yue2-adapter.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...)                           \
    do {                                           \
        if (!(cond)) {                             \
            fprintf(stderr, "FAIL: " __VA_ARGS__); \
            fprintf(stderr, "\n");                 \
            g_fail++;                              \
        }                                          \
    } while (0)

static uint32_t g_rng = 0x5EEDD02Au;
static float frand() {  // deterministic, roughly [-1, 1)
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return (float) (g_rng & 0xFFFFFF) / 8388608.0f - 1.0f;
}
static float bf16(float v) {  // round to nearest even, as ggml_bf16_round
    uint32_t u;
    memcpy(&u, &v, 4);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    memcpy(&v, &u, 4);
    return v;
}
static std::vector<float> rnd(size_t n, float amp, bool round_bf16 = false) {
    std::vector<float> v(n);
    for (float & x : v) x = round_bf16 ? bf16(amp * frand()) : amp * frand();
    return v;
}

static ggml_backend_t g_backend = nullptr;

static void put(ggml_tensor * t, const std::vector<float> & v) {
    if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> h(v.size());
        for (size_t i = 0; i < v.size(); i++) h[i] = ggml_fp32_to_bf16(v[i]);
        ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
    } else {
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    }
}
static std::vector<float> get(ggml_tensor * t) {
    std::vector<float> v(size_t(ggml_nelements(t)));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}
static bool same_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
static double max_abs(const std::vector<float> & v) {
    double m = 0.0;
    for (float x : v) m = std::fmax(m, std::fabs((double) x));
    return m;
}
// max |a - b| / max |b|, NaN-proof (a NaN anywhere returns NaN, and NaN < tol is false).
static double rel_err(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return NAN;
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        if (std::isnan(d)) return NAN;
        worst = std::fmax(worst, d);
    }
    const double s = max_abs(b);
    return s > 0.0 ? worst / s : worst;
}

// ── Cases 1 and 2: one fused site, q|k|v widths ──────────────────────────────
constexpr int64_t kIn = 64, kQ = 48, kK = 24, kOut = kQ + 2 * kK, kRank = 8, kSeq = 7;
constexpr float   kScale = 0.5f;

struct Site {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    Yue2AitkConvRotLinear w;
    Yue2AitkFusedLora     lora, dora;
    ggml_tensor *         x = nullptr, * t = nullptr;
    std::vector<float>    xv, tv, wv, av, bv;

    bool init() {
        ggml_init_params ip{ 32 * ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, kIn, kOut);
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kIn, kSeq);
        t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kOut, kSeq);
        lora.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kIn, kRank);
        lora.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kRank, kOut);
        lora.input_width = kIn; lora.output_width = kOut; lora.rank = kRank; lora.scale = kScale;
        dora = lora;
        dora.m = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kOut);
        dora.nrm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kOut);
        ggml_set_param(x); ggml_set_param(lora.a); ggml_set_param(lora.b); ggml_set_param(dora.m);
        w.parts = { weight }; w.rows = kOut; w.cols = kIn;
        buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
        if (!buf) return false;
        // Activations and upstream gradients are BF16 values in the trainer.
        xv = rnd(size_t(kIn * kSeq), 1.0f, true);
        tv = rnd(size_t(kOut * kSeq), 1.0f, true);
        wv = rnd(size_t(kIn * kOut), 0.2f);
        av = rnd(size_t(kIn * kRank), 0.1f);
        bv = rnd(size_t(kRank * kOut), 0.1f);
        put(weight, wv); put(x, xv); put(t, tv); put(lora.a, av); put(lora.b, bv);
        return true;
    }
    void release() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

struct Pass {
    std::vector<float> y, gx, ga, gb, gm;
    bool               nrm_grad = false;  // nrm took a gradient (must not)
};

// loss = sum(y * t); split runs linear_split over q|k|v and joins on the host.
static bool run_site(Site & s, const Yue2AitkFusedLora & ad, bool split, ggml_tensor * upstream, Pass * out) {
    const size_t nodes = 2048;
    std::vector<uint8_t> arena(ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, true));
    ggml_init_params ip{ arena.size(), arena.data(), true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> ys;
    if (split) {
        ys = yue2_aitk_graph::linear_split(ctx, s.w, s.x, &ad, { kQ, kK, kK });
    } else {
        ys = { yue2_aitk_graph::linear(ctx, s.w, s.x, &ad) };
    }
    ggml_tensor * loss = nullptr;
    int64_t first = 0;
    for (ggml_tensor * y : ys) {
        ggml_tensor * tp = ggml_cont(ctx, ggml_view_2d(ctx, upstream, y->ne[0], kSeq, upstream->nb[1], size_t(first) * sizeof(float)));
        ggml_tensor * l = ggml_sum(ctx, ggml_mul(ctx, y, tp));
        loss = loss ? ggml_add(ctx, loss, l) : l;
        first += y->ne[0];
    }
    for (ggml_tensor * y : ys) ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, nodes, true);
    ggml_build_forward_expand(gf, loss);
    ggml_set_loss(loss);
    ggml_build_backward_expand(ctx, gf, nullptr);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
    bool ok = alloc && ggml_gallocr_alloc_graph(alloc, gf);
    if (ok) {
        ggml_graph_reset(gf);
        ok = ggml_backend_graph_compute(g_backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        out->y.assign(size_t(kOut * kSeq), 0.0f);
        first = 0;
        for (ggml_tensor * y : ys) {
            const std::vector<float> part = get(y);
            for (int64_t c = 0; c < kSeq; c++)
                for (int64_t r = 0; r < y->ne[0]; r++) out->y[size_t(c * kOut + first + r)] = part[size_t(c * y->ne[0] + r)];
            first += y->ne[0];
        }
        auto grad = [&](ggml_tensor * p, std::vector<float> & v) {
            ggml_tensor * g = p ? ggml_graph_get_grad(gf, p) : nullptr;
            if (g) v = get(g); else v.clear();
        };
        grad(s.x, out->gx); grad(ad.a, out->ga); grad(ad.b, out->gb); grad(ad.m, out->gm);
        out->nrm_grad = ad.nrm && ggml_graph_get_grad(gf, ad.nrm) != nullptr;
    }
    if (alloc) ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}

static void case_identity_and_rescale(bool split) {
    const char * mode = yue2_aitk_graph::strict_f32() ? "strict-f32" : "bf16";
    const char * path = split ? "split" : "fused";
    Site s;
    if (!s.init()) { CHECK(false, "site allocation failed"); return; }
    Pass lora, dora;
    CHECK(run_site(s, s.lora, split, s.t, &lora), "LoRA pass failed (%s, %s)", mode, path);

    // 1. m == nrm (any positive values): factor exactly 1.
    std::vector<float> nv(static_cast<size_t>(kOut));
    for (float & v : nv) v = 0.5f + std::fabs(frand());
    put(s.dora.nrm, nv); put(s.dora.m, nv);
    CHECK(run_site(s, s.dora, split, s.t, &dora), "DoRA pass failed (%s, %s)", mode, path);
    CHECK(same_bits(dora.y, lora.y), "m == nrm forward differs from LoRA (%s, %s)", mode, path);
    CHECK(same_bits(dora.gx, lora.gx), "m == nrm dx differs from LoRA (%s, %s)", mode, path);
    CHECK(same_bits(dora.ga, lora.ga), "m == nrm dA differs from LoRA (%s, %s)", mode, path);
    CHECK(same_bits(dora.gb, lora.gb), "m == nrm dB differs from LoRA (%s, %s)", mode, path);
    CHECK(!dora.nrm_grad, "the detached norm took a gradient (%s, %s)", mode, path);

    // 2. m != nrm: y = bf16(y_lora * m / nrm) per row.
    std::vector<float> mv(static_cast<size_t>(kOut)), f(static_cast<size_t>(kOut));
    for (int64_t o = 0; o < kOut; o++) { mv[size_t(o)] = nv[size_t(o)] * (1.0f + 0.25f * frand()); f[size_t(o)] = mv[size_t(o)] / nv[size_t(o)]; }
    put(s.dora.m, mv);
    CHECK(run_site(s, s.dora, split, s.t, &dora), "DoRA pass failed (%s, %s)", mode, path);
    std::vector<float> want(lora.y.size()), gm_want(size_t(kOut), 0.0f);
    for (int64_t c = 0; c < kSeq; c++)
        for (int64_t o = 0; o < kOut; o++) {
            const size_t i = size_t(c * kOut + o);
            want[i] = bf16(lora.y[i] * f[size_t(o)]);
            gm_want[size_t(o)] += float((double) s.tv[i] * lora.y[i] / nv[size_t(o)]);
        }
    // A bf16 ulp at most: the backend's m / nrm may differ from the host's in
    // its last F32 bit (CUDA builds with fast math).
    const double ey = rel_err(dora.y, want), em = rel_err(dora.gm, gm_want);
    CHECK(ey < 8e-3, "rescaled forward off by %.3g (%s, %s)", ey, mode, path);
    CHECK(em < 1e-4, "dL/dm off by %.3g (%s, %s)", em, mode, path);
    CHECK(!dora.nrm_grad, "the detached norm took a gradient (%s, %s)", mode, path);

    // A/B and x see the upstream rescaled by the factor: a LoRA pass over
    // t * f must give the same gradients, up to that pass's own roundings.
    ggml_init_params ip{ 2 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * tctx = ggml_init(ip);
    ggml_tensor * tf = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, kOut, kSeq);
    ggml_backend_buffer_t tbuf = ggml_backend_alloc_ctx_tensors(tctx, g_backend);
    std::vector<float> tfv(s.tv.size());
    for (int64_t c = 0; c < kSeq; c++)
        for (int64_t o = 0; o < kOut; o++) tfv[size_t(c * kOut + o)] = s.tv[size_t(c * kOut + o)] * f[size_t(o)];
    put(tf, tfv);
    Pass scaled;
    CHECK(run_site(s, s.lora, split, tf, &scaled), "scaled LoRA pass failed (%s, %s)", mode, path);
    const double ea = rel_err(dora.ga, scaled.ga), eb = rel_err(dora.gb, scaled.gb), ex = rel_err(dora.gx, scaled.gx);
    CHECK(ea < 2e-2 && eb < 2e-2 && ex < 2e-2, "A/B/x gradients off the rescaled LoRA ones: %.3g %.3g %.3g (%s, %s)",
          ea, eb, ex, mode, path);
    printf("  %-10s %-5s  m != nrm: forward %.2e%s, dm %.2e, dA %.2e, dB %.2e, dx %.2e\n", mode, path,
           ey, same_bits(dora.y, want) ? " (bit-exact)" : "", em, ea, eb, ex);
    ggml_backend_buffer_free(tbuf);
    ggml_free(tctx);
    s.release();
}

// ── Case 3: row norms over a split base plus companion ───────────────────────
static void case_norms() {
    constexpr int64_t in = 96, q = 64, k = 32, out = q + 2 * k, r = 4, rc = 3;
    constexpr float   scale = 0.75f;
    constexpr int64_t zero_row = 5;
    ggml_init_params ip{ 32 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    Yue2AitkConvRotLinear w;
    w.parts = { ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, in, q), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, k),
                ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, in, k) };
    w.frozen_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, rc);
    w.frozen_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rc, out);
    w.rows = out; w.cols = in;
    Yue2AitkFusedLora site;
    site.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, r);
    site.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, out);
    site.m = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out);
    site.nrm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out);
    site.input_width = in; site.output_width = out; site.rank = r; site.scale = scale;
    ggml_tensor * eye = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, in);
    ggml_tensor * w0sq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
    if (!buf) { CHECK(false, "norm fixture allocation failed"); ggml_free(ctx); return; }

    // Host copies in [out][in] row order. Every value is BF16-representable,
    // F32 tensors too: CUDA runs F32 GEMMs as TF32 (cuBLAS TF32 math mode),
    // which would round arbitrary F32 inputs to 10 mantissa bits and put a
    // ~1e-4 backend error into a check meant for the graph's maths.
    std::vector<double> w0(static_cast<size_t>(out * in));
    int64_t row0 = 0;
    for (ggml_tensor * p : w.parts) {
        std::vector<float> v = rnd(size_t(in * p->ne[1]), 0.2f, true);
        if (zero_row >= row0 && zero_row < row0 + p->ne[1])
            for (int64_t i = 0; i < in; i++) v[size_t((zero_row - row0) * in + i)] = 0.0f;
        put(p, v);
        for (int64_t o = 0; o < p->ne[1]; o++)
            for (int64_t i = 0; i < in; i++) w0[size_t((row0 + o) * in + i)] = v[size_t(o * in + i)];
        row0 += p->ne[1];
    }
    std::vector<float> fa = rnd(size_t(in * rc), 0.1f, true), fb = rnd(size_t(rc * out), 0.1f, true);
    std::vector<float> av = rnd(size_t(in * r), 0.1f, true), bv = rnd(size_t(r * out), 0.1f, true);
    for (int64_t c = 0; c < rc; c++) fb[size_t(zero_row * rc + c)] = 0.0f;
    for (int64_t c = 0; c < r; c++) bv[size_t(zero_row * r + c)] = 0.0f;
    put(w.frozen_a, fa); put(w.frozen_b, fb); put(site.a, av);
    std::vector<float> ev(size_t(in * in), 0.0f);
    for (int64_t i = 0; i < in; i++) ev[size_t(i * in + i)] = 1.0f;
    put(eye, ev);
    for (int64_t o = 0; o < out; o++)
        for (int64_t i = 0; i < in; i++) {
            double d = 0.0;
            for (int64_t c = 0; c < rc; c++) d += (double) fb[size_t(o * rc + c)] * fa[size_t(c * in + i)];
            w0[size_t(o * in + i)] += d;
        }
    auto host_norm = [&](const std::vector<float> & b) {
        std::vector<float> n(static_cast<size_t>(out));
        for (int64_t o = 0; o < out; o++) {
            double sq = 0.0;
            for (int64_t i = 0; i < in; i++) {
                double d = 0.0;
                for (int64_t c = 0; c < r; c++) d += (double) b[size_t(o * r + c)] * av[size_t(c * in + i)];
                const double v = w0[size_t(o * in + i)] + scale * d;
                sq += v * v;
            }
            n[size_t(o)] = float(std::fmax(std::sqrt(sq), (double) kYue2AitkDoraNormFloor));
        }
        return n;
    };
    auto pass = [&](bool exact) {
        const size_t nodes = 256;
        std::vector<uint8_t> arena(ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false));
        ggml_init_params gp{ arena.size(), arena.data(), true };
        ggml_context * g = ggml_init(gp);
        ggml_tensor * n = exact ? yue2_aitk_graph::dora_norm_exact(g, w, site, eye)
                                : yue2_aitk_graph::dora_norm_expanded(g, w, site, w0sq);
        ggml_cgraph * gf = ggml_new_graph_custom(g, nodes, false);
        ggml_build_forward_expand(gf, ggml_cpy(g, n, ggml_reshape_2d(g, site.nrm, 1, out)));
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        const bool ok = alloc && ggml_gallocr_alloc_graph(alloc, gf) && ggml_backend_graph_compute(g_backend, gf) == GGML_STATUS_SUCCESS;
        if (alloc) ggml_gallocr_free(alloc);
        ggml_free(g);
        CHECK(ok, "%s norm graph failed", exact ? "exact" : "expanded");
        return get(site.nrm);
    };

    // init_dora's order: the exact pass at B = 0 gives ||W0||, squared into w0sq.
    std::vector<float> zero_b(bv.size(), 0.0f);
    put(site.b, zero_b);
    const std::vector<float> n0 = pass(true);
    const double e0 = rel_err(n0, host_norm(zero_b));
    CHECK(e0 < 1e-5, "exact ||W0|| off by %.3g", e0);
    std::vector<float> sq(n0.size());
    for (size_t i = 0; i < n0.size(); i++) sq[i] = n0[i] * n0[i];
    put(w0sq, sq);

    put(site.b, bv);
    const std::vector<float> want = host_norm(bv), exact = pass(true), expanded = pass(false);
    const double ee = rel_err(exact, want), ex = rel_err(expanded, want), exe = rel_err(expanded, exact);
    CHECK(ee < 1e-5, "exact norm off the host by %.3g", ee);
    CHECK(ex < 1e-5, "expanded norm off the host by %.3g", ex);
    // The zero row: both paths at the floor, so m / nrm stays finite.
    const float zf = kYue2AitkDoraNormFloor;
    CHECK(std::fabs(exact[size_t(zero_row)] - zf) <= 1e-3f * zf && std::fabs(expanded[size_t(zero_row)] - zf) <= 1e-3f * zf,
          "zero row norm: exact %g expanded %g, want the floor %g", exact[size_t(zero_row)], expanded[size_t(zero_row)], zf);
    printf("  norms  %-10s exact %.2e, expanded %.2e, expanded vs exact %.2e, zero row %g / %g\n",
           yue2_aitk_graph::strict_f32() ? "strict-f32" : "bf16", ee, ex, exe, exact[size_t(zero_row)], expanded[size_t(zero_row)]);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// Case 4. One 2x2 identity base `blk.0.attn_q.weight` and a rank-1 module
// with A = [1, 1], B = [1, 1]^T, m = [1, 1], so W0 + s BA = [[1+s, s], [s, 1+s]].
// `nar` runs the same module on the decoder half (blk.0.nar_attn_q), so the
// NAR tags are checked against keys their family accepts.
static void case_loader(ggml_backend_t cpu, bool nar) {
    const char * fam = nar ? "nar" : "ar";
    ggml_init_params ip{ 64 * ggml_tensor_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    ggml_set_name(t, nar ? "blk.0.nar_attn_q.weight" : "blk.0.attn_q.weight");
    GGUFModel gf{};
    gf.gguf = gguf_init_empty();
    gf.meta = ctx;
    gguf_add_tensor(gf.gguf, t);
    gguf_set_val_u32(gf.gguf, "yue2.block_count", 1);
    float base[4] = { 1, 0, 0, 1 };
    gf.mapping = reinterpret_cast<uint8_t *>(base);
    float data[6] = { 1, 1, 1, 1, 1, 1 };
    const std::string mod = nar ? "yue2.blk.0.nar_attn_q" : "yue2.blk.0.attn_q";
    const std::string v1 = std::string("yue2-") + fam + "-dora-v1", v99 = std::string("yue2-") + fam + "-dora-v99",
                      lora_v1 = std::string("yue2-") + fam + "-lora-v1";
    auto make_st = [&](bool magnitude) {
        STFile st{};
        st.mapping   = reinterpret_cast<uint8_t *>(data);
        st.file_size = sizeof(data);
        st.entries.push_back({ mod + ".lora_A.weight", "F32", { 1, 2 }, 2, 0, 8 });
        st.entries.push_back({ mod + ".lora_B.weight", "F32", { 2, 1 }, 2, 8, 16 });
        if (magnitude) st.entries.push_back({ mod + ".lora_magnitude_vector.weight", "F32", { 2 }, 1, 16, 24 });
        return st;
    };
    const STFile dora = make_st(true), lora = make_st(false);
    // Returns the merge's result; `out` gets the patched (or untouched) base.
    auto run = [&](const STFile & st, const char * fmt, int rank, float alpha, const char * scaling, float out[4]) {
        WeightCtx wc{};
        wctx_init(&wc, 2);
        ggml_tensor * wt = ggml_dup_tensor(wc.ctx, t);
        ggml_set_name(wt, ggml_get_name(t));
        wc.pending.push_back({ wt, base, sizeof(base), 0 });
        Yue2AdapterMeta md;
        md.format = fmt; md.rank = rank; md.alpha = alpha;
        if (&st == &dora) { md.use_dora = "true"; md.dora_scaling = scaling; }
        std::string error;
        const int n = yue2_adapter_merge_st(&wc, gf, st, "", Yue2LmAdapterScales{}, cpu, &error, nullptr, &md);
        memcpy(out, wc.pending[0].src, 4 * sizeof(float));
        wctx_free(&wc);
        return n;
    };
    // W' rows: [1+s, s] / sqrt((1+s)^2 + s^2), m = 1. The merge rounds its
    // delta through BF16, hence the tolerance.
    auto expect = [&](const char * tag, const float got[4], double sc) {
        const double n = std::sqrt((1 + sc) * (1 + sc) + sc * sc);
        const double w[4] = { (1 + sc) / n, sc / n, sc / n, (1 + sc) / n };
        for (int i = 0; i < 4; i++) CHECK(std::fabs(got[i] - w[i]) < 4e-3, "loader %s %s: [%d] %g, want %g", fam, tag, i, got[i], w[i]);
    };
    auto untouched = [&](const char * tag, int n, const float got[4]) {
        CHECK(n < 0, "loader %s %s: merged (%d), want a refusal", fam, tag, n);
        CHECK(!memcmp(got, base, sizeof(base)), "loader %s %s: base changed on a refusal", fam, tag);
    };
    float out[4];
    CHECK(run(dora, v1.c_str(), 1, 1, "1.000000", out) == 1, "loader %s v1: no merge", fam);
    expect("v1 scale 1", out, 1.0);
    CHECK(run(dora, v1.c_str(), 1, 2, "alpha/rank", out) == 1, "loader %s alpha/rank: no merge", fam);
    expect("alpha/rank scale 2", out, 2.0);
    // Metadata rank 2 / alpha 2 reads as 1, but the tensor rank is 1: the
    // merge would apply 2. Refused.
    untouched("metadata-rank scaling", run(dora, v1.c_str(), 2, 2, "1.000000", out), out);
    untouched("v99", run(dora, v99.c_str(), 1, 1, "1.000000", out), out);
    untouched("magnitude in lora-v1", run(dora, lora_v1.c_str(), 1, 1, "", out), out);
    // Plain LoRA keeps its merge: W0 + s BA at s = 1.
    CHECK(run(lora, lora_v1.c_str(), 1, 1, "", out) == 1, "loader %s lora: no merge", fam);
    const float want[4] = { 2, 1, 1, 2 };
    for (int i = 0; i < 4; i++) CHECK(std::fabs(out[i] - want[i]) < 1e-2, "loader %s lora: [%d] %g, want %g", fam, i, out[i], want[i]);
    gguf_free(gf.gguf);
    ggml_free(ctx);
}

int main() {
    BackendPair bp = backend_init("yue2-aitk-dora");
    g_backend = bp.backend;
    printf("yue2-aitk-dora-test on %s\n", ggml_backend_name(g_backend));
    // Every backward through a BF16 base weight (LoRA's too) needs OUT_PROD
    // over a BF16 source. CPU and Vulkan have none, so they run the
    // forward-only norm case.
    bool backward = false;
    {
        ggml_init_params ip{ 4 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * pc = ggml_init(ip);
        ggml_tensor * probe = ggml_out_prod(pc, ggml_new_tensor_2d(pc, GGML_TYPE_BF16, 8, 4), ggml_new_tensor_2d(pc, GGML_TYPE_F32, 8, 4));
        backward = ggml_backend_supports_op(g_backend, probe);
        ggml_free(pc);
    }
    if (!backward) printf("  backward cases skipped: no OUT_PROD over BF16 on this backend\n");
    for (bool strict : { false, true }) {
        yue2_aitk_graph::strict_f32() = strict;
        if (backward) {
            case_identity_and_rescale(false);
            case_identity_and_rescale(true);
        }
        case_norms();
    }
    case_loader(bp.cpu_backend, false);
    case_loader(bp.cpu_backend, true);
    backend_release(bp.backend, bp.cpu_backend);
    if (g_fail) {
        fprintf(stderr, "yue2-aitk-dora-test: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("yue2-aitk-dora-test: all checks passed\n");
    return 0;
}
