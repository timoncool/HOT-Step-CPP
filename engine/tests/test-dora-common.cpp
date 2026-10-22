// test-dora-common.cpp — freeze fixture for dora-common.h's dora_col_norm.
//
// The two detached-refresh callers (train/lm-graph.h lm_lora_dora_refresh and
// minimax/mm3-lm-dora.h mm3_lm_dora_prepare) built this graph inline before it
// was extracted. Their pre-extraction sequences are frozen below with the
// production GGML_PREC_F32 delta precision contract, and every case requires the helper to match them
// BIT FOR BIT on the selected backend, plus a double-precision host reference
// within tolerance. Covers orientation (in != out both ways), zero norm, bias
// outside the rescale, and the cast variants each caller uses (F32/F16/Q8_0
// base, F32 vs F16 A/B).
//
// Build: configure engine/ with -DBUILD_TESTING=ON, target dora-common-test.
// Backend: GGML_BACKEND=CUDA0 / Vulkan0 / MTL0 / CPU, else ggml's best.

#include <string>  // backend.h relies on its includer for std::string

#include "backend.h"
#include "dora-common.h"
#include "qwen3-lora.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;

// Max that cannot be fooled by NaN: `NaN > worst` is false, so a plain max
// would drop it and pass. Once worst is NaN it stays NaN and `worst < tol`
// fails.
static void track_worst(double & worst, double v) {
    if (!std::isnan(worst) && !(v <= worst)) {
        worst = v;
    }
}

#define CHECK(cond, ...)                         \
    do {                                         \
        if (!(cond)) {                           \
            fprintf(stderr, "FAIL: " __VA_ARGS__); \
            fprintf(stderr, "\n");               \
            g_fail++;                            \
        }                                        \
    } while (0)

static uint32_t g_rng = 0x2545F491u;

static float frand() {  // deterministic, roughly [-1, 1)
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return (float) (g_rng & 0xFFFFFF) / 8388608.0f - 1.0f;
}

static BackendPair          g_bp;
static ggml_backend_sched_t g_sched;

// Host tensor -> device tensor of `type`, returning the values the device
// actually holds (after F16 rounding or Q8_0 quantization) for the reference.
static std::vector<float> upload(ggml_tensor * t, const std::vector<float> & v) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> held(v);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, v.data(), 0, n * sizeof(float));
    } else {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        if (t->type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(v.data(), (ggml_fp16_t *) raw.data(), n);
        } else {
            ggml_quantize_chunk(t->type, v.data(), raw.data(), 0, t->ne[1], t->ne[0], nullptr);
        }
        ggml_get_type_traits(t->type)->to_float(raw.data(), held.data(), n);
        ggml_backend_tensor_set(t, raw.data(), 0, raw.size());
    }
    return held;
}

struct Site {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *w = nullptr, *a = nullptr, *b = nullptr, *nrm = nullptr;
    std::vector<float> hw, ha, hb;  // device-held values
    int64_t in = 0, out = 0, r = 0;

    ~Site() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

// w [in, out], a [in, r], b [r, out] (ggml ne order). zero_col >= 0 zeroes that
// output channel of both W and B, so its norm must come out exactly 0.
static void make_site(Site & s, int64_t in, int64_t out, int64_t r, ggml_type tw, ggml_type tab, int zero_col) {
    s.in = in, s.out = out, s.r = r;
    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
    s.ctx = ggml_init(ip);
    s.w   = ggml_new_tensor_2d(s.ctx, tw, in, out);
    s.a   = ggml_new_tensor_2d(s.ctx, tab, in, r);
    s.b   = ggml_new_tensor_2d(s.ctx, tab, r, out);
    s.nrm = ggml_new_tensor_1d(s.ctx, GGML_TYPE_F32, out);
    s.buf = ggml_backend_alloc_ctx_tensors(s.ctx, g_bp.backend);

    std::vector<float> w(in * out), a(in * r), b(r * out);
    for (int64_t o = 0; o < out; o++) {
        // Column scale varies with o so a wrong reduction axis cannot pass.
        const float cs = 0.25f + 0.05f * (float) o;
        for (int64_t i = 0; i < in; i++) {
            w[o * in + i] = (o == zero_col) ? 0.0f : cs * frand();
        }
    }
    for (auto & x : a) x = 0.5f * frand();
    for (int64_t o = 0; o < out; o++) {
        for (int64_t k = 0; k < r; k++) {
            b[o * r + k] = (o == zero_col) ? 0.0f : 0.5f * frand();
        }
    }
    s.hw = upload(s.w, w);
    s.ha = upload(s.a, a);
    s.hb = upload(s.b, b);
}

static std::vector<double> ref_norm(const Site & s, float scale) {
    std::vector<double> nr(s.out);
    for (int64_t o = 0; o < s.out; o++) {
        double acc = 0.0;
        for (int64_t i = 0; i < s.in; i++) {
            double d = 0.0;
            for (int64_t k = 0; k < s.r; k++) {
                d += (double) s.ha[k * s.in + i] * (double) s.hb[o * s.r + k];
            }
            const double v = (double) s.hw[o * s.in + i] + (double) scale * d;
            acc += v * v;
        }
        nr[o] = std::sqrt(acc);
    }
    return nr;
}

static ggml_context * graph_ctx(std::vector<uint8_t> & arena) {
    const size_t need = ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false);
    arena.assign(need, 0);
    ggml_init_params ip = { need, arena.data(), true };
    return ggml_init(ip);
}

// ── LM trainer plumbing: cpy into the persistent nrm, compute without alloc ──
static std::vector<float> run_lm(Site & s, float scale, bool legacy) {
    std::vector<uint8_t> arena;
    ggml_context *       ctx = graph_ctx(arena);
    ggml_cgraph *        gf  = ggml_new_graph_custom(ctx, 64, false);
    ggml_tensor *        nr;
    ggml_tensor *        wf = s.w->type == GGML_TYPE_F32 ? s.w : ggml_cast(ctx, s.w, GGML_TYPE_F32);  // qwen3_f32
    if (legacy) {
        // Frozen lm-graph.h lm_lora_dora_refresh with production delta precision.
        ggml_tensor * delta = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, s.a)), s.b);
        ggml_mul_mat_set_prec(delta, GGML_PREC_F32);
        ggml_tensor * wd    = ggml_add(ctx, wf, ggml_scale(ctx, delta, scale));
        nr                  = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, wd)));  // [1, out]
    } else {
        nr = dora_col_norm(ctx, wf, s.a, s.b, scale);
    }
    ggml_build_forward_expand(gf, ggml_cpy(ctx, nr, ggml_reshape_2d(ctx, s.nrm, 1, s.nrm->ne[0])));
    ggml_backend_sched_reset(g_sched);
    const bool ok = ggml_backend_sched_graph_compute(g_sched, gf) == GGML_STATUS_SUCCESS;
    ggml_free(ctx);
    CHECK(ok, "lm graph compute");
    std::vector<float> host(s.out);
    ggml_backend_tensor_get(s.nrm, host.data(), 0, host.size() * sizeof(float));
    return host;
}

// ── MM3 runtime plumbing: unconditional casts, set_output, tensor_get ──
static std::vector<float> run_mm3(Site & s, float scale, bool legacy) {
    std::vector<uint8_t> arena;
    ggml_context *       ctx = graph_ctx(arena);
    ggml_tensor *        outt;
    if (legacy) {
        // Frozen mm3-lm-dora.h mm3_lm_dora_prepare with production delta precision.
        ggml_tensor * a_t   = ggml_cont(ctx, ggml_transpose(ctx, s.a));   // [r, in] f16
        ggml_tensor * b32   = ggml_cast(ctx, s.b, GGML_TYPE_F32);         // [r, out]
        ggml_tensor * delta = ggml_mul_mat(ctx, a_t, b32);                // [in, out] f32
        ggml_mul_mat_set_prec(delta, GGML_PREC_F32);
        ggml_tensor * basef = ggml_cast(ctx, s.w, GGML_TYPE_F32);
        ggml_tensor * wd    = ggml_add(ctx, basef, ggml_scale(ctx, delta, scale));
        ggml_tensor * nr    = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, wd)));  // [1, out]
        outt                = ggml_reshape_1d(ctx, nr, nr->ne[1]);
    } else {
        ggml_tensor * nr = dora_col_norm(ctx, ggml_cast(ctx, s.w, GGML_TYPE_F32), s.a,
                                         ggml_cast(ctx, s.b, GGML_TYPE_F32), scale);
        outt             = ggml_reshape_1d(ctx, nr, nr->ne[1]);
    }
    ggml_set_output(outt);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, outt);
    ggml_backend_sched_reset(g_sched);
    const bool ok = ggml_backend_sched_alloc_graph(g_sched, gf) &&
                    ggml_backend_sched_graph_compute(g_sched, gf) == GGML_STATUS_SUCCESS;
    CHECK(ok, "mm3 graph compute");
    std::vector<float> host((size_t) ggml_nelements(outt));
    if (ok) {
        ggml_backend_tensor_get(outt, host.data(), 0, host.size() * sizeof(float));
    }
    ggml_free(ctx);
    return host;
}

static void check_norm(const char * tag, const Site & s, float scale, int zero_col,
                       const std::vector<float> & legacy, const std::vector<float> & now) {
    CHECK(legacy.size() == (size_t) s.out && now.size() == (size_t) s.out, "%s: size", tag);
    CHECK(memcmp(legacy.data(), now.data(), now.size() * sizeof(float)) == 0, "%s: helper differs from legacy graph", tag);
    const std::vector<double> ref = ref_norm(s, scale);
    double worst = 0.0;
    int    bad   = 0;
    for (int64_t o = 0; o < s.out; o++) {
        bad += !std::isfinite(now[o]) || !std::isfinite(ref[o]);
        if (o == zero_col) {
            CHECK(now[o] == 0.0f, "%s: zero column %lld gave %g, want exactly 0", tag, (long long) o, (double) now[o]);
            continue;
        }
        const double rel = std::fabs((double) now[o] - ref[o]) / ref[o];
        track_worst(worst, rel);
    }
    // Matching NaNs pass the memcmp above, so finiteness is checked on its own.
    CHECK(bad == 0, "%s: %d non-finite norm or reference value(s)", tag, bad);
    // Bit-exactness against the legacy graph is the gate; this only catches a
    // gross error (axis, scale). 1e-3 because CUDA and Vulkan mul_mat land
    // near 1e-4 here (CUDA's small F32 x F32 likely on TF32 tensor cores), with
    // the legacy graph identical; CPU and Metal sit near 1e-7. The printed
    // max_rel shows it per backend.
    CHECK(worst < 1e-3, "%s: max rel err %.3g vs host reference", tag, worst);
    printf("  %-34s in=%-3lld out=%-3lld bit-exact=%s  max_rel=%.2e\n", tag, (long long) s.in, (long long) s.out,
           memcmp(legacy.data(), now.data(), now.size() * sizeof(float)) == 0 ? "yes" : "NO", worst);
}

// Forward through the shared apply (qwen3_linear_lora -> qwen3_lora_dora) with
// a bias added after: the rescale must cover W x + s B A x and never the bias.
static void check_bias(const Site & s, float scale, const std::vector<float> & nrm) {
    const int64_t seq = 3;
    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context *   wc = ggml_init(ip);
    ggml_tensor *    m  = ggml_new_tensor_1d(wc, GGML_TYPE_F32, s.out);
    ggml_tensor *    bi = ggml_new_tensor_1d(wc, GGML_TYPE_F32, s.out);
    ggml_tensor *    x  = ggml_new_tensor_2d(wc, GGML_TYPE_F32, s.in, seq);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(wc, g_bp.backend);

    std::vector<float> hm(s.out), hbias(s.out), hx(s.in * seq);
    for (int64_t o = 0; o < s.out; o++) {
        hm[o]    = (float) nrm[o] * (1.0f + 0.1f * frand());  // m near its init so the rescale is visible
        hbias[o] = 2.0f * frand();
    }
    for (auto & v : hx) v = frand();
    ggml_backend_tensor_set(m, hm.data(), 0, hm.size() * sizeof(float));
    ggml_backend_tensor_set(bi, hbias.data(), 0, hbias.size() * sizeof(float));
    ggml_backend_tensor_set(x, hx.data(), 0, hx.size() * sizeof(float));
    ggml_backend_tensor_set(s.nrm, nrm.data(), 0, nrm.size() * sizeof(float));

    QwLoraPair p;
    p.A = s.a, p.B = s.b, p.scale = scale, p.m = m, p.nrm = s.nrm;

    std::vector<uint8_t> arena;
    ggml_context *       ctx = graph_ctx(arena);
    ggml_tensor *        y   = ggml_add(ctx, qwen3_linear_lora(ctx, s.w, &p, x), bi);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, y);
    ggml_backend_sched_reset(g_sched);
    const bool ok = ggml_backend_sched_alloc_graph(g_sched, gf) &&
                    ggml_backend_sched_graph_compute(g_sched, gf) == GGML_STATUS_SUCCESS;
    CHECK(ok, "bias graph compute");
    std::vector<float> hy(s.out * seq);
    if (ok) {
        ggml_backend_tensor_get(y, hy.data(), 0, hy.size() * sizeof(float));
    }
    ggml_free(ctx);

    double worst = 0.0;
    int    bad   = 0;
    for (int64_t t = 0; t < seq; t++) {
        for (int64_t o = 0; o < s.out; o++) {
            double acc = 0.0;
            for (int64_t i = 0; i < s.in; i++) {
                double d = 0.0;
                for (int64_t k = 0; k < s.r; k++) {
                    d += (double) s.ha[k * s.in + i] * (double) s.hb[o * s.r + k];
                }
                acc += ((double) s.hw[o * s.in + i] + (double) scale * d) * (double) hx[t * s.in + i];
            }
            const double want = (double) hm[o] * acc / (double) nrm[o] + (double) hbias[o];
            const double err  = std::fabs((double) hy[t * s.out + o] - want) / (1.0 + std::fabs(want));
            bad += !std::isfinite(hy[t * s.out + o]) || !std::isfinite(want);
            track_worst(worst, err);
        }
    }
    CHECK(bad == 0, "bias: %d non-finite output or reference value(s)", bad);
    CHECK(worst < 1e-4, "bias: max err %.3g, rescale must exclude the bias", worst);
    printf("  %-34s max_err=%.2e\n", "bias outside rescale", worst);

    ggml_backend_buffer_free(buf);
    ggml_free(wc);
}

// Freezes today's zero-norm behaviour of the shared rescale: m / 0 is not
// guarded, so a dead output channel yields a non-finite activation. Any change
// here is a deliberate compatibility fix, not an extraction side effect.
static void check_zero_rescale(const Site & s, float scale, const std::vector<float> & nrm, int zero_col) {
    ggml_init_params ip = { 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context *   wc = ggml_init(ip);
    ggml_tensor *    m  = ggml_new_tensor_1d(wc, GGML_TYPE_F32, s.out);
    ggml_tensor *    x  = ggml_new_tensor_2d(wc, GGML_TYPE_F32, s.in, 1);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(wc, g_bp.backend);
    std::vector<float> ones(s.out, 1.0f), hx(s.in, 1.0f);
    ggml_backend_tensor_set(m, ones.data(), 0, ones.size() * sizeof(float));
    ggml_backend_tensor_set(x, hx.data(), 0, hx.size() * sizeof(float));
    ggml_backend_tensor_set(s.nrm, nrm.data(), 0, nrm.size() * sizeof(float));

    QwLoraPair p;
    p.A = s.a, p.B = s.b, p.scale = scale, p.m = m, p.nrm = s.nrm;
    std::vector<uint8_t> arena;
    ggml_context *       ctx = graph_ctx(arena);
    ggml_tensor *        y   = qwen3_linear_lora(ctx, s.w, &p, x);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, y);
    ggml_backend_sched_reset(g_sched);
    const bool ok = ggml_backend_sched_alloc_graph(g_sched, gf) &&
                    ggml_backend_sched_graph_compute(g_sched, gf) == GGML_STATUS_SUCCESS;
    std::vector<float> hy(s.out, 0.0f);
    if (ok) {
        ggml_backend_tensor_get(y, hy.data(), 0, hy.size() * sizeof(float));
    }
    ggml_free(ctx);
    CHECK(ok && !std::isfinite(hy[zero_col]), "zero-norm rescale: column %d gave %g, frozen behaviour is non-finite",
          zero_col, (double) hy[zero_col]);
    CHECK(ok && std::isfinite(hy[zero_col == 0 ? 1 : 0]), "zero-norm rescale leaked into a live column");
    printf("  %-34s col %d -> %g (frozen, unguarded)\n", "zero-norm rescale", zero_col, (double) hy[zero_col]);
    ggml_backend_buffer_free(buf);
    ggml_free(wc);
}

int main() {
    g_bp    = backend_init("DoRA-fixture");
    g_sched = backend_sched_new(g_bp, 64);
    printf("dora-common-test on %s\n", ggml_backend_name(g_bp.backend));

    struct Shape { int64_t in, out, r; };
    // Orientation both ways; Q8_0 needs in % 32 == 0.
    const Shape shapes[] = { { 64, 40, 4 }, { 32, 96, 8 } };
    const float scale    = 0.75f;

    for (const Shape & sh : shapes) {
        // LM trainer: F32 A/B (params), base F32 / BF16-style F16 / Q8_0.
        for (ggml_type tw : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
            Site s;
            make_site(s, sh.in, sh.out, sh.r, tw, GGML_TYPE_F32, 5);
            char tag[64];
            snprintf(tag, sizeof(tag), "lm  base=%s ab=f32", ggml_type_name(tw));
            check_norm(tag, s, scale, 5, run_lm(s, scale, true), run_lm(s, scale, false));
        }
        // MM3 runtime: F16 A/B off disk, base F32 / F16 / Q8_0.
        for (ggml_type tw : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
            Site s;
            make_site(s, sh.in, sh.out, sh.r, tw, GGML_TYPE_F16, 7);
            char tag[64];
            snprintf(tag, sizeof(tag), "mm3 base=%s ab=f16", ggml_type_name(tw));
            check_norm(tag, s, scale, 7, run_mm3(s, scale, true), run_mm3(s, scale, false));
        }
    }

    {   // Bias and zero-norm rescale through the shared apply path.
        Site s;
        make_site(s, 64, 40, 4, GGML_TYPE_F32, GGML_TYPE_F32, 3);
        const std::vector<float> nrm = run_lm(s, scale, false);
        check_zero_rescale(s, scale, nrm, 3);
        Site live;
        make_site(live, 64, 40, 4, GGML_TYPE_F32, GGML_TYPE_F32, -1);
        check_bias(live, scale, run_lm(live, scale, false));
    }

    ggml_backend_sched_free(g_sched);
    backend_release(g_bp.backend, g_bp.cpu_backend);
    printf(g_fail ? "dora-common-test: %d FAILURE(S)\n" : "dora-common-test: PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
