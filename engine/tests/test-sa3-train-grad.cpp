// test-sa3-train-grad.cpp: SA3 backward feasibility gate (one DiT block).
//
// Drives sa3_train_build_layer (train/sa3-train-gate.h) with DoRA-rows on all
// nine adapted Linears and every block input trainable:
//   1. Placement: forward + backward of one full-size medium block. Every node
//      must be supported by the backend under test, so nothing can fall back
//      to the CPU; the op histogram is printed.
//   2. Finite differences on a small random block (CPU, f32): the directional
//      derivative of loss = sum(y * R) along a random direction, for A, B and
//      magnitude of every site and for x, gcond, cross and local, matches the
//      analytic gradient. Padding and local conditioning are exercised.
//      Negative controls: the DoRA norm wired as a constant (a real bug class)
//      and one gradient scaled by 1.05 must both be caught.
//   3. Replay against the official trainer: block 0 of a captured step
//      (_experiments/sa3-training/gate/block0-step2.safetensors; regenerate
//      with tools/sa3-training-block-fixture.py --step 2 and --step 1, which
//      asserts the replay reproduces the stage-4 reference capture before it
//      writes anything; see its header for provenance). Output and every gradient
//      are compared with the official block recomputed in fp32 (tight), and
//      with the bf16 training capture itself (the official fp32 recompute
//      sits up to 2.5% from it). Negative control: the gradients must not
//      match a different step's capture.
// Exit 77 (skip) when the replay fixture is missing and everything else passed.
// Fixture dir: SA3_GATE_FIXTURES, else found by walking up from the cwd.
// Backend: GGML_BACKEND=CUDA0 / Vulkan0 / MTL0 / CPU, else ggml's best.

#include <string>  // backend.h relies on its includer for std::string

#include "backend.h"
#include "safetensors.h"
#include "train/sa3-train-gate.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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

static uint32_t g_rng = 0x5A3D0A11u;
static float frand() {  // deterministic, roughly [-1, 1)
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return (float) (g_rng & 0xFFFFFF) / 8388608.0f - 1.0f;
}
static std::vector<float> rnd(size_t n, float amp, float bias = 0.0f) {
    std::vector<float> v(n);
    for (float & x : v) x = bias + amp * frand();
    return v;
}
static void put(ggml_tensor * t, const void * p) { ggml_backend_tensor_set(t, p, 0, ggml_nbytes(t)); }
static void put(ggml_tensor * t, const std::vector<float> & v) {
    if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> h(v.size());
        ggml_fp32_to_bf16_row(v.data(), h.data(), (int64_t) v.size());
        put(t, h.data());
    } else {
        put(t, v.data());
    }
}
static std::vector<float> get(ggml_tensor * t) {
    std::vector<float> v(size_t(ggml_nelements(t)));
    if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> h(v.size());
        ggml_backend_tensor_get(t, h.data(), 0, ggml_nbytes(t));
        ggml_bf16_to_fp32_row(h.data(), v.data(), (int64_t) v.size());
    } else {
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    }
    return v;
}
// BF16 base weights (how the trainer keeps the frozen DiT) need OUT_PROD over a
// BF16 source for the activation gradient.
static bool bf16_backward_ok(ggml_backend_t backend) {
    ggml_init_params ip{ 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context *   pc = ggml_init(ip);
    ggml_tensor *    pr = ggml_out_prod(pc, ggml_new_tensor_2d(pc, GGML_TYPE_BF16, 8, 4), ggml_new_tensor_2d(pc, GGML_TYPE_F32, 8, 4));
    const bool       ok = ggml_backend_supports_op(backend, pr);
    ggml_free(pc);
    return ok;
}
static double dot(const std::vector<float> & a, const std::vector<float> & b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i++) s += (double) a[i] * b[i];
    return s;
}
static double norm(const std::vector<float> & a) { return std::sqrt(dot(a, a)); }
// ||a - b|| / ||b||; with ||b|| == 0 it is ||a|| (both must be zero). NaN-proof.
static double rel_l2(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return NAN;
    double d = 0.0;
    for (size_t i = 0; i < a.size(); i++) d += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
    const double nb = norm(b);
    return nb > 0.0 ? std::sqrt(d) / nb : std::sqrt(d);
}

// ── One block: frozen base, DoRA-rows sites, inputs ────────────────────────────
struct Block {
    SA3DiTConfig          c{};
    SA3DiTLayer           ly{};
    Sa3LayerAdapters      ad{};
    ggml_tensor *         a_copy[SA3_SITE_COUNT]{}, * b_copy[SA3_SITE_COUNT]{};
    ggml_tensor *         x = nullptr, * gcond = nullptr, * cross = nullptr, * local = nullptr;
    ggml_tensor *         positions = nullptr, * vmask = nullptr, * upstream = nullptr;
    int64_t               T = 0, S = 0, S_c = 0, rank = 0;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    ggml_tensor ** weight(int s) {
        ggml_tensor ** w[SA3_SITE_COUNT] = { &ly.attn_qkv, &ly.attn_out, &ly.cross_q, &ly.cross_kv, &ly.cross_out,
                                             &ly.ff0_w,    &ly.ff2_w,    &ly.loc0_w,  &ly.loc2_w };
        return w[s];
    }
    void site_dims(int s, int64_t * in, int64_t * out) const {
        const int64_t d = c.dim, f = c.ff_inner;
        const int64_t io[SA3_SITE_COUNT][2] = { { d, 5 * d }, { d, d },     { d, 2 * d },          { d, 3 * d }, { d, d },
                                                { d, 2 * f }, { f, d },     { c.local_cond_dim, d }, { d, d } };
        *in  = io[s][0];
        *out = io[s][1];
    }

    bool init(const SA3DiTConfig & cfg, int64_t t, int64_t s_c, int64_t r, ggml_backend_t backend,
              ggml_type wtype = GGML_TYPE_F32) {
        c = cfg; T = t; S = c.n_memory_tokens + t; S_c = s_c; rank = r;
        ggml_init_params ip{ 96 * ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        auto f32 = [&](int64_t a, int64_t b = 1) { return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a, b); };
        auto f1  = [&](int64_t a) { return ggml_new_tensor_1d(ctx, GGML_TYPE_F32, a); };
        const int64_t d = c.dim, D = c.dim_heads, f = c.ff_inner;
        ly.pre_norm_g = f1(d); ly.ca_norm_g = f1(d); ly.ff_norm_g = f1(d); ly.ssg = f1(6 * d);
        ly.attn_qn_g = f1(D); ly.attn_kn_g = f1(D); ly.cross_qn_g = f1(D); ly.cross_kn_g = f1(D);
        ly.ff0_b = f1(2 * f); ly.ff2_b = f1(d); ly.loc0_b = f1(d); ly.loc2_b = f1(d);
        for (int s = 0; s < SA3_SITE_COUNT; s++) {
            int64_t in, out;
            site_dims(s, &in, &out);
            *weight(s) = ggml_new_tensor_2d(ctx, wtype, in, out);
            Sa3DoraRows & p = ad.site[s];
            p.a = f32(in, r); p.b = f32(r, out); p.m = f1(out); p.base_nsq = f1(out);
            a_copy[s] = f32(in, r); b_copy[s] = f32(r, out);
            ggml_set_param(p.a); ggml_set_param(p.b); ggml_set_param(p.m);
        }
        x = f32(d, S); gcond = f32(6 * d, 1); cross = f32(d, S_c); local = f32(c.local_cond_dim, T);
        upstream = f32(d, S);
        ggml_set_param(x); ggml_set_param(gcond); ggml_set_param(cross); ggml_set_param(local);
        positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
        vmask     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, S);
        buf       = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) return false;
        std::vector<int32_t> pos((size_t) S);
        for (int64_t i = 0; i < S; i++) pos[size_t(i)] = (int32_t) i;
        put(positions, pos.data());
        return true;
    }
    // base_nsq from the base weight, in double; copies for the detached-norm control.
    void finish_sites() {
        for (int s = 0; s < SA3_SITE_COUNT; s++) {
            int64_t in, out;
            site_dims(s, &in, &out);
            const std::vector<float> w = get(*weight(s));
            std::vector<float>       nsq((size_t) out);
            for (int64_t o = 0; o < out; o++) {
                double acc = 0.0;
                for (int64_t i = 0; i < in; i++) acc += (double) w[size_t(o * in + i)] * w[size_t(o * in + i)];
                nsq[size_t(o)] = (float) acc;
            }
            put(ad.site[s].base_nsq, nsq);
            sync_copies(s);
        }
    }
    void sync_copies(int s) {
        put(a_copy[s], get(ad.site[s].a));
        put(b_copy[s], get(ad.site[s].b));
    }
    void release() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

// Forward + backward of loss = sum(y * upstream) on one backend.
struct Run {
    std::vector<uint8_t> arena;
    ggml_context *       ctx   = nullptr;
    ggml_cgraph *        gf    = nullptr;
    ggml_gallocr_t       alloc = nullptr;
    ggml_tensor *        y = nullptr, * loss = nullptr;
    ggml_backend_t       backend = nullptr;
    int                  refused = 0;
    std::string          refused_ops;
    std::map<std::string, int> ops;

    bool build(Block & b, ggml_backend_t be, bool detached_norm) {
        backend = be;
        const size_t nodes = 16384;
        arena.resize(ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, true));
        ggml_init_params ip{ arena.size(), arena.data(), true };
        ctx = ggml_init(ip);
        Sa3LayerAdapters ad = b.ad;
        for (int s = 0; s < SA3_SITE_COUNT && detached_norm; s++) {
            ad.site[s].a_norm = b.a_copy[s];
            ad.site[s].b_norm = b.b_copy[s];
        }
        y = sa3_train_build_layer(ctx, b.c, &b.ly, &ad, b.x, b.gcond, b.positions, b.vmask, b.cross, b.local, b.S, b.S_c);
        ggml_set_output(y);
        loss = ggml_sum(ctx, ggml_mul(ctx, y, b.upstream));
        ggml_set_output(loss);
        ggml_set_loss(loss);
        gf = ggml_new_graph_custom(ctx, nodes, true);
        ggml_build_forward_expand(gf, loss);
        ggml_build_backward_expand(ctx, gf, nullptr);
        refused = sa3_train_audit_graph(backend, gf, &ops, &refused_ops);
        if (refused) return false;
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        return alloc && ggml_gallocr_alloc_graph(alloc, gf);
    }
    double compute() {
        ggml_graph_reset(gf);
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) return NAN;
        float l = 0.0f;
        ggml_backend_tensor_get(loss, &l, 0, sizeof(float));
        return l;
    }
    std::vector<float> grad(ggml_tensor * t) {
        ggml_tensor * g = ggml_graph_get_grad(gf, t);
        return g ? get(g) : std::vector<float>();
    }
    void release() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
        alloc = nullptr; ctx = nullptr;
    }
};

static SA3DiTConfig medium_config() {
    SA3DiTConfig c{};
    c.io_channels = 256; c.dim = 1536; c.depth = 24; c.n_heads = 24; c.dim_heads = 64;
    c.cond_token_dim = 768; c.global_cond_dim = 768; c.local_cond_dim = 257; c.n_memory_tokens = 64;
    c.patch_size = 1; c.ts_feat_dim = 256; c.ff_inner = 6144; c.rope_dims = 32; c.rope_theta = 10000.0f;
    c.norm_eps = 1e-5f; c.qk_norm_eps = 1e-6f; c.ts_min_freq = 0.5f; c.ts_max_freq = 10000.0f;
    return c;
}

// Random block values. B is nonzero so every A gradient is live (at the
// official init B = 0 and every A gradient is exactly zero).
static void fill_random(Block & b, int pad_last) {
    const int64_t d = b.c.dim, D = b.c.dim_heads, f = b.c.ff_inner;
    put(b.ly.pre_norm_g, rnd(size_t(d), 0.1f, 1.0f)); put(b.ly.ca_norm_g, rnd(size_t(d), 0.1f, 1.0f));
    put(b.ly.ff_norm_g, rnd(size_t(d), 0.1f, 1.0f));  put(b.ly.ssg, rnd(size_t(6 * d), 0.3f));
    for (ggml_tensor * g : { b.ly.attn_qn_g, b.ly.attn_kn_g, b.ly.cross_qn_g, b.ly.cross_kn_g }) put(g, rnd(size_t(D), 0.1f, 1.0f));
    put(b.ly.ff0_b, rnd(size_t(2 * f), 0.1f)); put(b.ly.ff2_b, rnd(size_t(d), 0.1f));
    put(b.ly.loc0_b, rnd(size_t(d), 0.1f));    put(b.ly.loc2_b, rnd(size_t(d), 0.1f));
    for (int s = 0; s < SA3_SITE_COUNT; s++) {
        int64_t in, out;
        b.site_dims(s, &in, &out);
        const float ws = 1.0f / std::sqrt((float) in);
        put(*b.weight(s), rnd(size_t(in * out), 1.7f * ws));
        put(b.ad.site[s].a, rnd(size_t(in * b.rank), 1.7f * ws));
        put(b.ad.site[s].b, rnd(size_t(b.rank * out), 0.3f));
        put(b.ad.site[s].m, rnd(size_t(out), 0.3f, 1.0f));
        b.ad.site[s].scale = 1.0f;
    }
    put(b.x, rnd(size_t(d * b.S), 1.0f)); put(b.gcond, rnd(size_t(6 * d), 0.3f));
    put(b.cross, rnd(size_t(d * b.S_c), 1.0f)); put(b.local, rnd(size_t(b.c.local_cond_dim * b.T), 1.0f));
    put(b.upstream, rnd(size_t(d * b.S), 1.0f));
    std::vector<float> vm((size_t) b.S, 1.0f);
    for (int i = 0; i < pad_last; i++) vm[size_t(b.S - 1 - i)] = 0.0f;
    put(b.vmask, vm);
    b.finish_sites();
}

// ── 1. Placement on the backend under test, full medium block ─────────────────
static void case_placement(ggml_backend_t backend, ggml_type wtype) {
    Block b;
    if (!b.init(medium_config(), 32, 257, 16, backend, wtype)) {
        CHECK(false, "placement: cannot allocate a medium block on %s", ggml_backend_name(backend));
        return;
    }
    fill_random(b, 0);
    Run r;
    const bool ok = r.build(b, backend, false);
    std::string hist;
    for (auto & kv : r.ops) hist += " " + kv.first + "=" + std::to_string(kv.second);
    printf("  placement %s, %s base: %d nodes, %d refused%s%s\n  ops:%s\n", ggml_backend_name(backend),
           ggml_type_name(wtype), ggml_graph_n_nodes(r.gf), r.refused, r.refused ? " -> " : "", r.refused_ops.c_str(),
           hist.c_str());
    CHECK(r.refused == 0, "placement: %s refuses %d node(s): %s", ggml_backend_name(backend), r.refused, r.refused_ops.c_str());
    // Every op placed but the graph still failed to allocate: that is a failure, not a skip.
    CHECK(ok || r.refused != 0, "placement: graph allocation failed on %s", ggml_backend_name(backend));
    if (ok) {
        const double l = r.compute();
        CHECK(std::isfinite(l), "placement: non-finite loss on %s", ggml_backend_name(backend));
        bool all_grads = true;
        for (int s = 0; s < SA3_SITE_COUNT; s++)
            for (ggml_tensor * t : { b.ad.site[s].a, b.ad.site[s].b, b.ad.site[s].m }) {
                const std::vector<float> g = r.grad(t);
                all_grads = all_grads && !g.empty() && norm(g) > 0.0 && std::isfinite(norm(g));
            }
        CHECK(all_grads, "placement: a site gradient is missing, zero or non-finite");
    }
    r.release();
    b.release();
}

// ── 2. Finite differences, small block on CPU ─────────────────────────────────
struct FdTensor {
    std::string   name;
    ggml_tensor * t;
};

// Directional derivative along a random unit-RMS direction scaled to 4e-2 of
// the tensor's RMS (SA3_GATE_FD_H overrides). Counts failures; worst_out gets
// the worst |fd - an| / (tol*|an| + 1e-5), which passes at <= 1.
// `scale_one` scales one analytic gradient (control).
static int fd_check(Block & b, Run & r, const std::vector<FdTensor> & ts, double tol, const char * tag, int scale_one,
                    bool verbose, double * worst_out) {
    if (!std::isfinite(r.compute())) return (int) ts.size();
    std::vector<std::vector<float>> grads;
    for (const FdTensor & ft : ts) grads.push_back(r.grad(ft.t));
    int    fails = 0;
    double worst = 0.0;
    for (size_t k = 0; k < ts.size(); k++) {
        ggml_tensor *            t  = ts[k].t;
        const std::vector<float> p0 = get(t);
        const double             rms = std::max(norm(p0) / std::sqrt((double) p0.size()), 1e-2);
        std::vector<float>       dir = rnd(p0.size(), 1.0f);
        const double             dn  = norm(dir) / std::sqrt((double) dir.size());
        static const double fd_h = getenv("SA3_GATE_FD_H") ? atof(getenv("SA3_GATE_FD_H")) : 4e-2;
        for (float & v : dir) v = (float) (v / dn * fd_h * rms);
        auto sync = [&]() { for (int s = 0; s < SA3_SITE_COUNT; s++) if (t == b.ad.site[s].a || t == b.ad.site[s].b) b.sync_copies(s); };
        // central difference along h*dir, per unit dir
        auto central = [&](double h) {
            std::vector<float> pp(p0), pm(p0);
            for (size_t i = 0; i < p0.size(); i++) { pp[i] += (float) (h * dir[i]); pm[i] -= (float) (h * dir[i]); }
            put(t, pp); sync();
            const double lp = r.compute();
            put(t, pm); sync();
            const double lm = r.compute();
            return (lp - lm) / (2.0 * h);
        };
        // Richardson: cancels the h^2 term, so a step large enough to keep
        // f32 rounding out of the difference stays accurate.
        const double fd = (4.0 * central(0.5) - central(1.0)) / 3.0;
        put(t, p0); sync();
        double       an  = dot(grads[k], dir);
        if ((int) k == scale_one) an *= 1.05;
        // Pass: |fd - an| <= tol*|an| + 1e-5. The loss is one f32 sum, so each
        // difference is quantised to ~1 ULP of L (2-4e-6 here, the same
        // residual on every tensor); 1e-5 covers that after Richardson.
        const double err = std::fabs(fd - an) / std::fabs(an);
        const double fit = std::fabs(fd - an) / (tol * std::fabs(an) + 1e-5);  // <= 1 passes
        worst = std::max(worst, fit);
        const bool bad = !(fit <= 1.0);
        fails += bad;
        if (verbose || bad) printf("    %s %-26s fd %+.6e an %+.6e rel %.2e%s\n", tag, ts[k].name.c_str(), fd, an, err, bad ? "  <-" : "");
    }
    if (worst_out) *worst_out = worst;
    return fails;
}

static void case_finite_differences(ggml_backend_t cpu) {
    SA3DiTConfig c = medium_config();
    c.dim = 128; c.n_heads = 2; c.dim_heads = 64; c.rope_dims = 32; c.ff_inner = 256; c.local_cond_dim = 9;
    c.n_memory_tokens = 4;
    Block b;
    if (!b.init(c, 6, 5, 4, cpu)) {
        CHECK(false, "fd: cannot allocate the small block");
        return;
    }
    fill_random(b, 2);  // last two data tokens padded
    std::vector<FdTensor> ts;
    for (int s = 0; s < SA3_SITE_COUNT; s++) {
        const std::string n = sa3_train_site_names[s];
        ts.push_back({ n + ".A", b.ad.site[s].a });
        ts.push_back({ n + ".B", b.ad.site[s].b });
        ts.push_back({ n + ".m", b.ad.site[s].m });
    }
    ts.push_back({ "x", b.x }); ts.push_back({ "gcond", b.gcond });
    ts.push_back({ "cross", b.cross }); ts.push_back({ "local", b.local });

    const double tol = 2e-3;
    Run r;
    if (!r.build(b, cpu, false)) {
        CHECK(false, "fd: graph build/alloc failed (refused: %s)", r.refused_ops.c_str());
        return;
    }
    double worst = 0.0;
    const int fails = fd_check(b, r, ts, tol, "fd", -1, getenv("SA3_GATE_VERBOSE") != nullptr, &worst);
    printf("  finite differences: %zu tensors, worst %.2f of the bound |fd-an| <= %.0e*|an| + 1e-5, %d over\n", ts.size(),
           worst, tol, fails);
    CHECK(fails == 0, "fd: %d tensor(s) disagree with finite differences", fails);

    // Control A: a 5% error on one gradient (cross_attn.to_kv.A) must be caught.
    const int scaled = fd_check(b, r, ts, tol, "ctl-scale", 9, false, nullptr);
    printf("  control (one gradient x1.05): %d caught\n", scaled);
    CHECK(scaled >= 1, "fd control: a 5%% gradient error went unnoticed");
    r.release();

    // Control B: the DoRA norm treated as a constant must break A and B.
    Run rd;
    if (!rd.build(b, cpu, true)) {
        CHECK(false, "fd control: detached graph build failed");
        return;
    }
    int caught = 0;
    for (int s = 0; s < SA3_SITE_COUNT; s++) {
        std::vector<FdTensor> ab = { ts[size_t(3 * s)], ts[size_t(3 * s + 1)] };
        caught += fd_check(b, rd, ab, tol, "ctl-detach", -1, false, nullptr);
    }
    printf("  control (DoRA norm detached): %d of %d A/B tensors caught\n", caught, 2 * SA3_SITE_COUNT);
    CHECK(caught == 2 * SA3_SITE_COUNT, "fd control: a detached DoRA norm passed for %d A/B tensor(s)", 2 * SA3_SITE_COUNT - caught);
    rd.release();
    b.release();
}

// ── 3. Replay against the official trainer ─────────────────────────────────────
static std::string find_fixture_dir() {
    if (const char * e = getenv("SA3_GATE_FIXTURES")) return e;
    std::string up;
    for (int i = 0; i < 6; i++) {
        const std::string dir = up + "_experiments/sa3-training/gate";
        if (FILE * f = fopen((dir + "/block0-step2.safetensors").c_str(), "rb")) {
            fclose(f);
            return dir;
        }
        up += "../";
    }
    return "";
}

static std::vector<float> st_f32(const STFile & st, const std::string & name, size_t want) {
    const STEntry * e = st_find(st, name.c_str());
    if (!e || e->dtype != "F32" || (e->data_end - e->data_start) != want * sizeof(float)) {
        CHECK(false, "replay: fixture tensor %s missing or wrong size", name.c_str());
        return std::vector<float>(want, 0.0f);
    }
    std::vector<float> v(want);
    memcpy(v.data(), st_data(st, *e), want * sizeof(float));
    return v;
}

// Returns false when the fixture is absent.
static bool case_replay(ggml_backend_t backend, const std::string & dir, ggml_type wtype = GGML_TYPE_F32) {
    STFile st{}, st_other{};
    if (dir.empty() || !st_open(&st, (dir + "/block0-step2.safetensors").c_str())) return false;
    const bool have_other = st_open(&st_other, (dir + "/block0-step1.safetensors").c_str());

    const SA3DiTConfig c = medium_config();
    Block              b;
    const int64_t      T = 32, S_c = 257, rank = 16;
    if (!b.init(c, T, S_c, rank, backend, wtype)) {
        CHECK(false, "replay: cannot allocate the medium block");
        st_close(&st);
        return true;
    }
    const int64_t S = b.S;
    auto ld = [&](ggml_tensor * t, const std::string & n) { put(t, st_f32(st, n, size_t(ggml_nelements(t)))); };
    ld(b.ly.pre_norm_g, "base.pre_norm.gamma"); ld(b.ly.ca_norm_g, "base.cross_attend_norm.gamma");
    ld(b.ly.ff_norm_g, "base.ff_norm.gamma");   ld(b.ly.ssg, "base.to_scale_shift_gate");
    ld(b.ly.attn_qn_g, "base.self_attn.q_norm.gamma");   ld(b.ly.attn_kn_g, "base.self_attn.k_norm.gamma");
    ld(b.ly.cross_qn_g, "base.cross_attn.q_norm.gamma"); ld(b.ly.cross_kn_g, "base.cross_attn.k_norm.gamma");
    ld(b.ly.ff0_b, "base.ff.ff.0.proj.bias"); ld(b.ly.ff2_b, "base.ff.ff.2.bias");
    ld(b.ly.loc0_b, "base.to_local_embed.0.bias"); ld(b.ly.loc2_b, "base.to_local_embed.2.bias");
    const std::string pz = ".parametrizations.weight.0.";
    for (int s = 0; s < SA3_SITE_COUNT; s++) {
        const std::string n = sa3_train_site_names[s];
        ld(*b.weight(s), "base." + n + ".weight");
        ld(b.ad.site[s].a, "init." + n + pz + "lora_A");
        ld(b.ad.site[s].b, "init." + n + pz + "lora_B");
        ld(b.ad.site[s].m, "init." + n + pz + "magnitude");
        b.ad.site[s].scale = 16.0f / 16.0f;  // alpha / rank of the reference run
    }
    ld(b.x, "x"); ld(b.gcond, "global_cond"); ld(b.cross, "context"); ld(b.local, "local_add_cond");
    ld(b.upstream, "dy");
    put(b.vmask, st_f32(st, "padding_mask", size_t(S)));
    b.finish_sites();

    Run r;
    if (!r.build(b, backend, false)) {
        CHECK(false, "replay: graph build/alloc failed (refused: %s)", r.refused_ops.c_str());
        b.release();
        st_close(&st);
        return true;
    }
    const double l = r.compute();
    CHECK(std::isfinite(l), "replay: non-finite loss");
    const std::vector<float> y = get(r.y);

    // Tolerances (rel L2 per tensor). fp32: the same block in torch fp32 (no
    // TF32); CPU lands at 3.6e-4, a GPU backend's f32 GEMMs at 2.8e-3 (CUDA),
    // hence 1e-3 / 5e-3. capture: the bf16 training run itself, from which the
    // official fp32 recompute sits up to 2.5% away.
    const bool   on_cpu   = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    const double tol_fp32 = on_cpu ? 1e-3 : 5e-3, tol_capture = 5e-2;
    double worst_fp32 = 0.0, worst_capture = 0.0;
    int    n_cmp = 0, other_close = 0, other_n = 0;
    auto cmp = [&](const std::string & what, const std::vector<float> & got, const std::string & fp32_name,
                   const std::string & capture_name) {
        const double e32 = rel_l2(got, st_f32(st, fp32_name, got.size()));
        worst_fp32 = std::max(worst_fp32, e32);
        CHECK(e32 <= tol_fp32, "replay %s vs official fp32: rel %.3e > %.0e", what.c_str(), e32, tol_fp32);
        double ecap = NAN;
        if (!capture_name.empty()) {
            const std::vector<float> ref = st_f32(st, capture_name, got.size());
            ecap = rel_l2(got, ref);
            worst_capture = std::max(worst_capture, ecap);
            CHECK(ecap <= tol_capture, "replay %s vs bf16 capture: rel %.3e > %.0e", what.c_str(), ecap, tol_capture);
            if (have_other && norm(ref) > 0.0) {
                const std::string other_name = capture_name == "y_bf16" ? "" : capture_name;
                if (!other_name.empty()) {
                    other_n++;
                    other_close += rel_l2(got, st_f32(st_other, other_name, got.size())) <= tol_capture;
                }
            }
        }
        n_cmp++;
        if (getenv("SA3_GATE_VERBOSE")) printf("    %-46s fp32 %.2e  capture %.2e\n", what.c_str(), e32, ecap);
    };
    // y is token-major [dim, S] == torch (1, S, dim) contiguous.
    cmp("y", y, "y_fp32", "y_bf16");
    const char * kinds[3] = { "lora_A", "lora_B", "magnitude" };
    for (int s = 0; s < SA3_SITE_COUNT; s++) {
        ggml_tensor * ps[3] = { b.ad.site[s].a, b.ad.site[s].b, b.ad.site[s].m };
        for (int k = 0; k < 3; k++) {
            const std::string n = std::string(sa3_train_site_names[s]) + pz + kinds[k];
            cmp(n, r.grad(ps[k]), "grad_fp32." + n, "grad_fixture." + n);
        }
    }
    cmp("dx", r.grad(b.x), "dx_fp32", "");
    cmp("dcontext", r.grad(b.cross), "dcontext_fp32", "");
    cmp("dglobal_cond", r.grad(b.gcond), "dglobal_cond_fp32", "");
    cmp("dlocal_add_cond", r.grad(b.local), "dlocal_add_cond_fp32", "");
    printf("  replay %s, %s base (step 2, block 0, S=%lld): %d tensors, worst rel vs official fp32 %.2e (tol %.0e), "
           "vs bf16 capture %.2e (tol %.0e)\n",
           ggml_backend_name(backend), ggml_type_name(wtype), (long long) S, n_cmp, worst_fp32, tol_fp32, worst_capture,
           tol_capture);
    if (have_other) {
        printf("  control (step-1 capture as reference): %d of %d nonzero gradients pass\n", other_close, other_n);
        CHECK(other_close == 0, "replay control: %d gradient(s) also match a different step", other_close);
        st_close(&st_other);
    } else {
        CHECK(false, "replay control: block0-step1.safetensors missing");
    }
    r.release();
    b.release();
    st_close(&st);
    return true;
}

int main() {
    BackendPair bp = backend_init("sa3-train-grad");
    printf("sa3-train-grad-test on %s\n", ggml_backend_name(bp.backend));
    const bool bf16 = bf16_backward_ok(bp.backend);
    case_placement(bp.backend, GGML_TYPE_F32);
    if (bf16) case_placement(bp.backend, GGML_TYPE_BF16);
    else printf("  BF16-base cases skipped: no OUT_PROD over BF16 on %s\n", ggml_backend_name(bp.backend));
    case_finite_differences(bp.cpu_backend);
    const std::string dir    = find_fixture_dir();
    bool              replay = case_replay(bp.backend, dir);
    if (replay && bf16) case_replay(bp.backend, dir, GGML_TYPE_BF16);
    if (replay && bp.cpu_backend != bp.backend && getenv("SA3_GATE_REPLAY_CPU")) case_replay(bp.cpu_backend, dir);
    backend_release(bp.backend, bp.cpu_backend);
    if (g_fail) {
        fprintf(stderr, "sa3-train-grad-test: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    if (!replay) {
        printf("sa3-train-grad-test: replay SKIPPED (no fixture: run tools/sa3-training-block-fixture.py, or set SA3_GATE_FIXTURES)\n");
        return 77;
    }
    printf("sa3-train-grad-test: all checks passed\n");
    return 0;
}
