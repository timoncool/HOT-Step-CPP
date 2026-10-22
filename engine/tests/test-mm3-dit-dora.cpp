// test-mm3-dit-dora.cpp — MM3 flow-DiT DoRA: trainer forward vs merged weights.
//
// One fake DiT block (tiny widths, F16 base, the real tensor names) runs
// through the REAL trainer code: mm3_train_make_adapters (--target mlpv, so the
// fused-qkv V-only mask is live), mm3_train_dora_refresh, mm3_dt_linear and
// mm3_train_export. The exported file then goes through the REAL loader,
// mm3_adapter_merge over a GGUF written here and opened with gf_load. Checks:
//
//   1. merged W' . x matches the trainer's forward at strength 1, per site;
//   2. masked q,k rows of the fused qkv come back as the base;
//   3. an all-zero row (base and B) stays finite and zero;
//   4. strength 0 is the base and 0.5 is W0 + 0.5*(W' - W0);
//   5. dL/dm is zero on masked rows, matches a finite difference elsewhere;
//   6. a malformed DoRA file (missing / NaN / mis-sized magnitude) merges
//      nothing at all;
//   7. a plain LoRA export still has no magnitude and merges as W0 + s*BA;
//  10. adapter_config.json's use_dora must agree with the magnitudes;
//  11. Q8_0 / Q4_0 bases merge in place within one quant step, or are
//      refused whole on a backend that cannot dequantize them.
//
// And the MM3 LM runtime (mm3-lm-adapter.h / mm3-lm-dora.h), through the real
// loader and norm pass on a fake one-layer LM:
//
//   8. with dora_renorm off, nrm is the legacy bytes at every dial; on, it is
//      ||W + s*BA||_col for the effective s (the norm mm3_lm_merge_apply
//      uses), and at the trained scale the pass's own value bit for bit;
//   9. a use_dora file with a module missing its magnitude does not load.
//
// Build: configure engine/ with -DBUILD_TESTING=ON, target mm3-dit-dora-test.
// Backend: GGML_BACKEND=CUDA0 / Vulkan0 / MTL0 / CPU, else ggml's best.

#include <string>

#include "backend.h"
#include "gguf-weights.h"
#include "minimax/mm3-model.h"
#include "minimax/mm3-lm-dora.h"
#include "train/mm3-dit-train-run.h"

#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

// train/lm-common.h declares the trainer's JSONL sink and ace-train.cpp defines
// it; MSVC rejects a declared-but-undefined static, so the test supplies one.
static void jl(const char *, ...) {}

static int g_fail = 0;

#define CHECK(cond, ...)                           \
    do {                                           \
        if (!(cond)) {                             \
            fprintf(stderr, "FAIL: " __VA_ARGS__); \
            fprintf(stderr, "\n");                 \
            g_fail++;                              \
        }                                          \
    } while (0)

static uint32_t g_rng = 0x9E3779B9u;

static float frand() {  // deterministic, roughly [-1, 1)
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return (float) (g_rng & 0xFFFFFF) / 8388608.0f - 1.0f;
}

// NaN-proof: once worst is NaN it stays NaN and `worst < tol` fails.
static void track_worst(double & worst, double v) {
    if (!std::isnan(worst) && !(v <= worst)) {
        worst = v;
    }
}

static BackendPair          g_bp;
static ggml_backend_sched_t g_sched;

static const int64_t E = 24, FI = 32, R = 4, S = 5;

// ── fixtures ────────────────────────────────────────────────────────────────

struct Site {
    const char *         gguf;
    int64_t              in, out;
    MM3TrainLora *       lo;
    std::vector<float>   w;   // F16-rounded base, ggml [in, out]
};

static void put(ggml_tensor * t, const std::vector<float> & v) {
    GGML_ASSERT(ggml_nelements(t) == (int64_t) v.size());
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
}

static std::vector<float> get(const ggml_tensor * t) {
    std::vector<float> v((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

// y[out, S] = W x for a host F32 W [in, out] (ggml order) and x [in, S].
static std::vector<double> host_matvec(const std::vector<float> & w, const std::vector<float> & x, int64_t in,
                                       int64_t out) {
    std::vector<double> y((size_t) (out * S), 0.0);
    for (int64_t s = 0; s < S; s++) {
        for (int64_t o = 0; o < out; o++) {
            double acc = 0.0;
            for (int64_t i = 0; i < in; i++) {
                acc += (double) w[(size_t) (o * in + i)] * x[(size_t) (s * in + i)];
            }
            y[(size_t) (s * out + o)] = acc;
        }
    }
    return y;
}

// Forward one site through mm3_dt_linear on the backend: returns y [out, S].
static std::vector<float> trainer_forward(ggml_tensor * w, const MM3TrainLora & lo, const std::vector<float> & x) {
    std::vector<uint8_t> arena(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(1024, false));
    ggml_init_params     ip  = { arena.size(), arena.data(), true };
    ggml_context *       ctx = ggml_init(ip);
    ggml_tensor *        tx  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w->ne[0], S);
    ggml_set_input(tx);
    ggml_tensor * y = mm3_dt_linear(ctx, w, tx, lo);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);
    ggml_build_forward_expand(gf, y);
    ggml_backend_sched_reset(g_sched);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(g_sched, gf));
    put(tx, x);
    GGML_ASSERT(ggml_backend_sched_graph_compute(g_sched, gf) == GGML_STATUS_SUCCESS);
    std::vector<float> out = get(y);
    ggml_free(ctx);
    return out;
}

// L = sum(y^2) for one site. With `opt`, the backward runs exactly as the
// trainer's does (lm_optim_fill_gacc into the optimizer's accumulators) and
// dL/dm is read back from m's accumulator.
static double trainer_loss_grad(ggml_tensor * w, const MM3TrainLora & lo, const std::vector<float> & x,
                                LmOptim * opt, std::vector<float> * gm) {
    std::vector<uint8_t> arena(ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(4096, true));
    ggml_init_params     ip  = { arena.size(), arena.data(), true };
    ggml_context *       ctx = ggml_init(ip);
    ggml_tensor *        tx  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w->ne[0], S);
    ggml_set_input(tx);
    ggml_tensor * y    = mm3_dt_linear(ctx, w, tx, lo);
    ggml_tensor * loss = ggml_sum(ctx, ggml_sqr(ctx, y));
    if (opt) {
        ggml_set_loss(loss);
    }
    ggml_set_output(loss);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, opt != nullptr);
    ggml_build_forward_expand(gf, loss);
    if (opt) {
        std::vector<ggml_tensor *> gacc;
        lm_optim_fill_gacc(opt, gf, &gacc);
        ggml_build_backward_expand(ctx, gf, gacc.data());
        lm_optim_zero_grad(opt);
    }
    ggml_backend_sched_reset(g_sched);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(g_sched, gf));
    put(tx, x);
    GGML_ASSERT(ggml_backend_sched_graph_compute(g_sched, gf) == GGML_STATUS_SUCCESS);
    float lv = 0.0f;
    ggml_backend_tensor_get(loss, &lv, 0, sizeof(float));
    if (opt && gm) {
        *gm = get(opt->acc[(size_t) opt->param_slot.at(lo.m)]);
    }
    ggml_free(ctx);
    return lv;
}

// Minimal safetensors writer for the malformed-file cases.
struct StEnt {
    std::string          name;
    std::vector<int64_t> shape;
    std::vector<float>   data;
};

static void st_write(const std::string & path, const std::vector<StEnt> & ents, const std::string & meta) {
    std::string hdr = "{";
    size_t      off = 0;
    for (const StEnt & e : ents) {
        std::string shp;
        for (int64_t d : e.shape) {
            shp += (shp.empty() ? "" : ",") + std::to_string(d);
        }
        const size_t nb = e.data.size() * sizeof(float);
        hdr += (off || hdr.size() > 1 ? "," : "") + std::string("\"") + e.name + "\":{\"dtype\":\"F32\",\"shape\":[" +
               shp + "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + nb) + "]}";
        off += nb;
    }
    hdr += ",\"__metadata__\":{" + meta + "}}";
    while ((8 + hdr.size()) % 8) {
        hdr += " ";
    }
    FILE * f = fopen(path.c_str(), "wb");
    GGML_ASSERT(f);
    const uint64_t hl = hdr.size();
    fwrite(&hl, sizeof(hl), 1, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    for (const StEnt & e : ents) {
        fwrite(e.data.data(), sizeof(float), e.data.size(), f);
    }
    fclose(f);
}

// Read every entry of a safetensors file back as F32 (the export writes F32).
static std::vector<StEnt> st_read_all(const std::string & path, std::string * meta) {
    STFile st = {};
    GGML_ASSERT(st_open(&st, path.c_str()));
    std::vector<StEnt> out;
    for (const STEntry & e : st.entries) {
        StEnt s;
        s.name = e.name;
        int64_t n = 1;
        for (int d = 0; d < e.n_dims; d++) {
            s.shape.push_back(e.shape[d]);
            n *= e.shape[d];
        }
        s.data.resize((size_t) n);
        memcpy(s.data.data(), st_data(st, e), (size_t) n * sizeof(float));
        out.push_back(std::move(s));
    }
    if (meta) {
        *meta = mm3_read_meta_str(st, "use_dora");
    }
    st_close(&st);
    return out;
}

// Open the base GGUF into a fresh WeightCtx (the loader's real input) and
// merge `path` at `scale`. Returns the merge count; `merged` gets each site's
// resulting weight as F32, in `names` order.
static int merge_into(const std::string & gguf_path, const std::vector<std::string> & names, const std::string & path,
                      float scale, std::vector<std::vector<float>> * merged, bool * untouched,
                      std::vector<ggml_type> * types = nullptr) {
    GGUFModel gf = {};
    GGML_ASSERT(gf_load(&gf, gguf_path.c_str()));
    WeightCtx wctx;
    wctx_init(&wctx, 16);
    std::vector<const void *> orig;
    for (const std::string & n : names) {
        gf_load_tensor(&wctx, gf, n);
        orig.push_back(wctx.pending.back().src);
    }
    const int n = mm3_adapter_merge(&wctx, gf, path.c_str(), scale, g_bp.backend);
    if (untouched) {
        *untouched = true;
        for (size_t i = 0; i < names.size(); i++) {
            *untouched = *untouched && wctx.pending[i].src == orig[i];
        }
    }
    if (merged) {
        merged->clear();
        for (size_t i = 0; i < names.size(); i++) {
            const WeightCtx::PendingCopy & pc = wctx.pending[i];
            const int64_t                  ne = ggml_nelements(pc.tensor);
            std::vector<float>             v((size_t) ne);
            // Whatever type the merge wrote back, read it through that type's
            // own dequantizer. The size must still be the native encoding.
            GGML_ASSERT(pc.nbytes == ggml_row_size(pc.tensor->type, pc.tensor->ne[0]) * (size_t) pc.tensor->ne[1]);
            ggml_get_type_traits(pc.tensor->type)->to_float(pc.src, v.data(), ne);
            merged->push_back(std::move(v));
            if (types) {
                types->push_back(pc.tensor->type);
            }
        }
    }
    wctx_free(&wctx);
    gf_close(&gf);
    return n;
}

// ── 8 + 9: the MM3 LM runtime's dials ──────────────────────────────────────

static void lm_dial_checks() {
    const int64_t LIN = 48, LOUT = 40, LR = 4;
    const float   alpha = 8.0f;   // base_scale = alpha / r = 2

    // Fake LM: 36 layers so the loader's layer bound is real, but only layer 0's
    // q and k exist. Base F16, host copy F16-rounded.
    static MM3Model m;
    m.lm.blk.assign(MM3_LM_ADAPTER_LAYERS, MM3LmLayer{});
    ggml_init_params wip = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context *   wc  = ggml_init(wip);
    ggml_tensor *    wq  = ggml_new_tensor_2d(wc, GGML_TYPE_F16, LIN, LOUT);
    ggml_tensor *    wk  = ggml_new_tensor_2d(wc, GGML_TYPE_F16, LIN, LOUT);
    ggml_backend_buffer_t wb = ggml_backend_alloc_ctx_tensors(wc, g_bp.backend);
    std::vector<float>    whost[2];
    for (int k = 0; k < 2; k++) {
        std::vector<ggml_fp16_t> h((size_t) (LIN * LOUT));
        whost[k].resize(h.size());
        for (size_t i = 0; i < h.size(); i++) {
            h[i]        = ggml_fp32_to_fp16(0.2f * frand());
            whost[k][i] = ggml_fp16_to_fp32(h[i]);
        }
        ggml_backend_tensor_set(k ? wk : wq, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
    }
    m.lm.blk[0].attn_q = wq;
    m.lm.blk[0].attn_k = wk;
    m.lm_resident      = true;

    // PEFT file: torch shapes A [r, in], B [out, r], magnitude [out].
    auto rnd = [](size_t n, float sc) { std::vector<float> v(n); for (float & x : v) x = sc * frand(); return v; };
    const std::string pre[2] = { "base_model.model.model.layers.0.self_attn.q_proj.",
                                 "base_model.model.model.layers.0.self_attn.k_proj." };
    std::vector<StEnt> ents;
    for (int k = 0; k < 2; k++) {
        ents.push_back({ pre[k] + "lora_A.weight", { LR, LIN }, rnd((size_t) (LR * LIN), 0.3f) });
        ents.push_back({ pre[k] + "lora_B.weight", { LOUT, LR }, rnd((size_t) (LOUT * LR), 0.3f) });
        std::vector<float> mag = rnd((size_t) LOUT, 0.2f);
        for (float & x : mag) x += 1.0f;
        ents.push_back({ pre[k] + "lora_magnitude_vector.weight", { LOUT }, mag });
    }
    const std::string dir = "mm3-lm-dora-test";
    std::filesystem::create_directories(dir);
    {
        FILE * f = fopen((dir + "/adapter_config.json").c_str(), "wb");
        GGML_ASSERT(f);
        fprintf(f, "{\"peft_type\":\"LORA\",\"r\":%lld,\"lora_alpha\":%.1f,\"use_dora\":true}", (long long) LR,
                (double) alpha);
        fclose(f);
    }
    const std::string apath = dir + "/adapter_model.safetensors";

    // 9 first: drop k's magnitude.
    {
        std::vector<StEnt> part(ents.begin(), ents.end() - 1);
        st_write(apath, part, "\"format\":\"pt\"");
        std::string    lerr;
        MM3LmAdapter * bad = mm3_lm_adapter_load(apath.c_str(), &lerr);
        CHECK(!bad, "LM: a use_dora file missing one magnitude loaded");
        fprintf(stderr, "[mm3-dit-dora] LM partial-m refused: %s\n", lerr.c_str());
        mm3_lm_adapter_free(bad);
    }

    st_write(apath, ents, "\"format\":\"pt\"");
    std::string    err;
    MM3LmAdapter * ad = mm3_lm_adapter_load(apath.c_str(), &err);
    CHECK(ad, "LM: DoRA adapter did not load: %s", err.c_str());
    if (!ad) {
        return;
    }
    CHECK(mm3_lm_dora_prepare(m, ad, &err), "LM: norm pass: %s", err.c_str());

    const MM3LmAdapterPair * ps[2] = { &ad->mods[0][MM3_LM_ADAPTER_Q], &ad->mods[0][MM3_LM_ADAPTER_K] };
    const std::vector<float> base_q = get(ps[0]->nrm);

    // Host ||W + s*BA|| from the adapter's own (F16) A and B, as the merge reads them.
    auto host_nrm = [&](int k, double s) {
        const MM3LmAdapterPair & p = *ps[k];
        std::vector<ggml_fp16_t> ah((size_t) ggml_nelements(p.a)), bh((size_t) ggml_nelements(p.b));
        ggml_backend_tensor_get(p.a, ah.data(), 0, ggml_nbytes(p.a));
        ggml_backend_tensor_get(p.b, bh.data(), 0, ggml_nbytes(p.b));
        std::vector<double> out((size_t) LOUT);
        for (int64_t o = 0; o < LOUT; o++) {
            double ss = 0.0;
            for (int64_t i = 0; i < LIN; i++) {
                double d = 0.0;
                for (int64_t r = 0; r < LR; r++) {
                    d += (double) ggml_fp16_to_fp32(ah[(size_t) (r * LIN + i)]) * ggml_fp16_to_fp32(bh[(size_t) (o * LR + r)]);
                }
                const double v = whost[k][(size_t) (o * LIN + i)] + s * d;
                ss += v * v;
            }
            out[(size_t) o] = std::sqrt(ss);
        }
        return out;
    };

    const std::vector<float> base_k = get(ps[1]->nrm);
    const float              dials[][2] = { { 0.5f, 0.7f }, { 1.3f, 1.0f }, { 1.0f, 0.25f } };  // global, attn

    // Switch off (the default): the legacy render at every dial, byte for byte.
    MM3LmAdapterScales sc;
    for (const auto & d : dials) {
        sc.global = d[0];
        sc.attn   = d[1];
        mm3_lm_dora_set_scales(ad, sc);
        CHECK(get(ps[0]->nrm) == base_q && get(ps[1]->nrm) == base_k,
              "LM: switch off at dial %.2f/%.2f moved nrm off the legacy bytes", d[0], d[1]);
    }

    // Switch on: nrm follows the dial, matching merge mode's ||W + s*BA||.
    sc.dora_renorm = true;
    double worst   = 0.0;
    for (const auto & d : dials) {
        sc.global = d[0];
        sc.attn   = d[1];
        mm3_lm_dora_set_scales(ad, sc);
        for (int k = 0; k < 2; k++) {
            const float               s   = ad->effective(0, k == 0 ? MM3_LM_ADAPTER_Q : MM3_LM_ADAPTER_K, sc);
            const std::vector<float>  got = get(ps[k]->nrm);
            const std::vector<double> ref = host_nrm(k, s);
            for (size_t o = 0; o < got.size(); o++) track_worst(worst, std::fabs(got[o] - ref[o]) / ref[o]);
        }
    }
    CHECK(worst < 1e-5, "LM: dialled nrm vs ||W + s*BA|| worst rel err %.3g", worst);

    // Switching off again from a dialled state restores the legacy bytes.
    sc.dora_renorm = false;
    mm3_lm_dora_set_scales(ad, sc);
    CHECK(get(ps[0]->nrm) == base_q && get(ps[1]->nrm) == base_k,
          "LM: switch off after a dialled request did not restore the legacy bytes");
    // And on at dial 1 is the pass's own nrm too.
    sc = MM3LmAdapterScales{};
    sc.dora_renorm = true;
    mm3_lm_dora_set_scales(ad, sc);
    CHECK(get(ps[0]->nrm) == base_q, "LM: switch on at dial 1, nrm is not the norm pass's own bytes");
    fprintf(stderr, "[mm3-dit-dora] LM switch off: legacy bytes at every dial; on: nrm vs ||W + s*BA|| worst "
                    "rel err %.3g, dial 1 bit-identical\n", worst);

    mm3_lm_adapter_free(ad);
    ggml_backend_buffer_free(wb);
    ggml_free(wc);
    std::filesystem::remove_all(dir);
}

int main() {
    g_bp    = backend_init("mm3-dit-dora-test");
    g_sched = backend_sched_new(g_bp, 8192);
    fprintf(stderr, "[mm3-dit-dora] backend %s\n", ggml_backend_name(g_bp.backend));

    // ── fake model: one block, the real shapes' structure at tiny widths ──
    static MM3Model m;
    m.synth_cfg.dit.embedding_length = (uint32_t) E;
    m.synth_cfg.dit.ff_inner         = (uint32_t) FI;
    m.synth_cfg.dit.concat_channels  = (uint32_t) E;
    m.synth_cfg.dit.in_channels      = (uint32_t) E;
    m.synth.dit.blk.assign(1, MM3DitBlock{});

    ggml_init_params wip  = { ggml_tensor_overhead() * 16, nullptr, true };
    ggml_context *   wctx = ggml_init(wip);
    MM3DitBlock &    blk  = m.synth.dit.blk[0];
    blk.attn_qkv          = ggml_new_tensor_2d(wctx, GGML_TYPE_F16, E, 3 * E);
    blk.attn_output       = ggml_new_tensor_2d(wctx, GGML_TYPE_F16, E, E);
    blk.ffn_in_w          = ggml_new_tensor_2d(wctx, GGML_TYPE_F16, E, 2 * FI);
    blk.ffn_out_w         = ggml_new_tensor_2d(wctx, GGML_TYPE_F16, FI, E);
    ggml_set_name(blk.attn_qkv, "dit.blk.0.attn_qkv.weight");
    ggml_set_name(blk.attn_output, "dit.blk.0.attn_output.weight");
    ggml_set_name(blk.ffn_in_w, "dit.blk.0.ffn_in.weight");
    ggml_set_name(blk.ffn_out_w, "dit.blk.0.ffn_out.weight");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, g_bp.backend);
    GGML_ASSERT(wbuf);

    MM3TrainAdapters           ad;
    std::vector<ggml_tensor *> params;
    ggml_init_params           aip  = { ggml_tensor_overhead() * 64, nullptr, true };
    ggml_context *             actx = ggml_init(aip);
    mm3_train_make_adapters(actx, m, R, 8.0f, /*mlpv*/ true, /*dora*/ true, &ad, &params);
    GGML_ASSERT(params.size() == 12);  // 4 sites x (A, B, m)
    ggml_backend_buffer_t abuf = ggml_backend_alloc_ctx_tensors(actx, g_bp.backend);
    GGML_ASSERT(abuf);

    std::vector<Site> sites = {
        { "dit.blk.0.attn_qkv.weight", E, 3 * E, &ad.blk[0].qkv, {} },
        { "dit.blk.0.attn_output.weight", E, E, &ad.blk[0].attn_out, {} },
        { "dit.blk.0.ffn_in.weight", E, 2 * FI, &ad.blk[0].ff_in, {} },
        { "dit.blk.0.ffn_out.weight", FI, E, &ad.blk[0].ff_out, {} },
    };
    ggml_tensor * wts[4] = { blk.attn_qkv, blk.attn_output, blk.ffn_in_w, blk.ffn_out_w };

    // Base: random, F16-rounded on the host so the host reference sees the
    // exact bytes. attn_output row 0 is all zero (the zero-norm case).
    for (size_t k = 0; k < sites.size(); k++) {
        Site &                   st = sites[k];
        std::vector<ggml_fp16_t> h((size_t) (st.in * st.out));
        st.w.resize(h.size());
        for (size_t i = 0; i < h.size(); i++) {
            const bool zero_row = (k == 1) && (i < (size_t) st.in);
            h[i]                = ggml_fp32_to_fp16(zero_row ? 0.0f : 0.2f * frand());
            st.w[i]             = ggml_fp16_to_fp32(h[i]);
        }
        ggml_backend_tensor_set(wts[k], h.data(), 0, h.size() * sizeof(ggml_fp16_t));
    }

    // Mask and keep, exactly as mm3_train_dit_run fills them.
    {
        std::vector<float> mv((size_t) (3 * E), 0.0f);
        for (int64_t i = 2 * E; i < 3 * E; i++) {
            mv[(size_t) i] = 1.0f;
        }
        put(ad.qkv_vmask, mv);
        for (float & x : mv) {
            x = 1.0f - x;
        }
        put(ad.qkv_vkeep, mv);
    }

    // B = 0 first: the refresh must give ||W||, so m = nrm is the neutral init.
    for (ggml_tensor * t : params) {
        put(t, std::vector<float>((size_t) ggml_nelements(t), 0.0f));
    }
    for (Site & st : sites) {
        put(st.lo->a, [&] { std::vector<float> v((size_t) (st.in * R)); for (float & x : v) x = 0.3f * frand(); return v; }());
    }
    std::string        err;
    std::vector<float> x((size_t) (FI * S));
    for (float & v : x) v = frand();
    CHECK(mm3_train_dora_refresh(m, ad, g_sched, &err), "refresh at init: %s", err.c_str());
    for (size_t k = 0; k < sites.size(); k++) {
        const Site &              st = sites[k];
        std::vector<float>        xs(x.begin(), x.begin() + st.in * S);
        put(st.lo->m, get(st.lo->nrm));
        // Against the same site with DoRA switched off: m/nrm is exactly 1,
        // so the outputs must be bit-identical, not merely close.
        MM3TrainLora plain = *st.lo;
        plain.m = plain.nrm = plain.keep = nullptr;
        const std::vector<float> y  = trainer_forward(wts[k], *st.lo, xs);
        const std::vector<float> yp = trainer_forward(wts[k], plain, xs);
        double                   wd = 0.0, sy = 0.0;
        for (size_t i = 0; i < y.size(); i++) {
            track_worst(wd, std::fabs(y[i] - yp[i]));
            sy = std::max(sy, (double) std::fabs(yp[i]));
        }
        // CPU: m/nrm is exactly 1 and the outputs are bit-identical. CUDA
        // builds with fast-math division, so the ratio can sit an ulp off 1;
        // allow a few float ulps of the output, and say which it was.
        CHECK(wd <= 4.0 * FLT_EPSILON * sy, "%s: B=0, m=||W|| vs plain LoRA max diff %.3g (|y| max %.3g)", st.gguf,
              wd, sy);
        fprintf(stderr, "[mm3-dit-dora] neutral  %-30s max |y_dora - y_lora| %.3g (%s)\n", st.gguf, wd,
                wd == 0.0 ? "bit-identical" : "within float ulps");
    }

    // Now a trained-looking adapter: random B, m = nrm * (1 + 0.2*rand). B is
    // zero where training leaves it zero (the masked q,k rows: zero init, zero
    // gradient) and on attn_output row 0, so that row of W + s*BA is all zero.
    // B is ggml [R, out]: element (r, o) sits at o*R + r.
    for (size_t k = 0; k < sites.size(); k++) {
        std::vector<float> b((size_t) (R * sites[k].out));
        for (size_t i = 0; i < b.size(); i++) {
            const bool zero = (k == 0 && i < (size_t) (2 * E * R)) || (k == 1 && i < (size_t) R);
            b[i]            = zero ? 0.0f : 0.3f * frand();
        }
        put(sites[k].lo->b, b);
    }
    CHECK(mm3_train_dora_refresh(m, ad, g_sched, &err), "refresh: %s", err.c_str());
    for (Site & st : sites) {
        std::vector<float> mv = get(st.lo->nrm);
        for (float & v : mv) v *= 1.0f + 0.2f * frand();
        put(st.lo->m, mv);
    }

    // Host reference W' = m * (W + s*BA) / max(||.||, floor), per output row,
    // with keep rows held at W + s*BA (they are W there: B's q,k rows matter
    // only through the mask, which zeroes them).
    const float sc = 8.0f / (float) R;
    // bf16_ab: round A and B through BF16 first, as the merge does.
    // base: W0 to build on (default the site's F16 base; the quantized
    // round trip passes its dequantized one).
    auto host_wprime = [&](const Site & st, bool bf16_ab, const std::vector<float> * base = nullptr) {
        const std::vector<float> & w0 = base ? *base : st.w;
        std::vector<float> a = get(st.lo->a), b = get(st.lo->b);
        const std::vector<float> mm = get(st.lo->m);
        if (bf16_ab) {
            for (float & v : a) v = ggml_bf16_to_fp32(ggml_fp32_to_bf16(v));
            for (float & v : b) v = ggml_bf16_to_fp32(ggml_fp32_to_bf16(v));
        }
        std::vector<float> w1((size_t) (st.in * st.out));
        for (int64_t o = 0; o < st.out; o++) {
            const bool masked = st.lo->keep && o < 2 * E;
            double     ss     = 0.0;
            for (int64_t i = 0; i < st.in; i++) {
                double d = 0.0;
                for (int64_t r = 0; r < R; r++) d += (double) a[(size_t) (r * st.in + i)] * b[(size_t) (o * R + r)];
                const double v           = w0[(size_t) (o * st.in + i)] + (masked ? 0.0 : sc * d);
                w1[(size_t) (o * st.in + i)] = (float) v;
                ss += v * v;
            }
            const double f = masked ? 1.0 : mm[(size_t) o] / std::max(std::sqrt(ss), 1e-12);
            for (int64_t i = 0; i < st.in; i++) w1[(size_t) (o * st.in + i)] = (float) (w1[(size_t) (o * st.in + i)] * f);
        }
        return w1;
    };

    // 1+2+3: trainer forward vs host W' (the trainer side of the parity).
    for (size_t k = 0; k < sites.size(); k++) {
        const Site &              st  = sites[k];
        std::vector<float>        xs(x.begin(), x.begin() + st.in * S);
        const std::vector<float>  y   = trainer_forward(wts[k], *st.lo, xs);
        const std::vector<double> ref = host_matvec(host_wprime(st, false), xs, st.in, st.out);
        double                    worst = 0.0;
        for (size_t i = 0; i < y.size(); i++) track_worst(worst, std::fabs(y[i] - ref[i]));
        CHECK(worst < 1e-3, "%s: trainer forward vs host W' max err %.3g", st.gguf, worst);
        if (k == 1) {
            for (int64_t s = 0; s < S; s++) {
                CHECK(std::isfinite(y[(size_t) (s * st.out)]) && y[(size_t) (s * st.out)] == 0.0f,
                      "zero row: output %g, want exactly 0", y[(size_t) (s * st.out)]);
            }
        }
        fprintf(stderr, "[mm3-dit-dora] trainer  %-30s max |y - y_ref| %.3g\n", st.gguf, worst);
    }

    // 5: gradient of m. Masked rows exactly zero; elsewhere vs central FD.
    {
        // The trainer's own optimizer wiring, minus the step itself.
        ggml_init_params oip = { ggml_tensor_overhead() * 4, nullptr, true };
        ggml_context *   octx = ggml_init(oip);
        LmOptim          opt;
        opt.t_lossgrad              = ggml_new_tensor_1d(octx, GGML_TYPE_F32, 1);
        ggml_backend_buffer_t obuf = ggml_backend_alloc_ctx_tensors(octx, g_bp.backend);
        const float           one  = 1.0f;
        ggml_backend_tensor_set(opt.t_lossgrad, &one, 0, sizeof(float));
        CHECK(lm_optim_init(&opt, params, g_bp.backend, &err), "lm_optim_init: %s", err.c_str());

        const Site &       st = sites[0];  // fused qkv, V-only
        std::vector<float> xs(x.begin(), x.begin() + st.in * S), gm;
        trainer_loss_grad(wts[0], *st.lo, xs, &opt, &gm);
        for (int64_t o = 0; o < 2 * E; o++) {
            CHECK(gm[(size_t) o] == 0.0f, "masked qkv row %lld got dL/dm %g", (long long) o, gm[(size_t) o]);
        }
        std::vector<float> m0 = get(st.lo->m);
        double             worst_rel = 0.0;
        for (int64_t o : { 2 * E, 2 * E + 7, 3 * E - 1 }) {
            const float h = 1e-2f * std::max(1e-3f, std::fabs(m0[(size_t) o]));
            std::vector<float> mp = m0, mn = m0;
            mp[(size_t) o] += h;
            mn[(size_t) o] -= h;
            put(st.lo->m, mp);
            const double lp = trainer_loss_grad(wts[0], *st.lo, xs, nullptr, nullptr);
            put(st.lo->m, mn);
            const double ln = trainer_loss_grad(wts[0], *st.lo, xs, nullptr, nullptr);
            put(st.lo->m, m0);
            const double fd = (lp - ln) / (2.0 * h);
            track_worst(worst_rel, std::fabs(fd - gm[(size_t) o]) / std::max(1e-6, std::fabs(fd)));
        }
        CHECK(worst_rel < 2e-2, "dL/dm vs finite difference: worst rel err %.3g", worst_rel);
        fprintf(stderr, "[mm3-dit-dora] dL/dm vs FD worst rel err %.3g, masked rows zero\n", worst_rel);
        ggml_backend_buffer_free(obuf);
        ggml_free(octx);
    }

    // Base GGUF for the loader, written from the same F16 bytes.
    const std::string dir  = ".";
    const std::string gpath = dir + "/mm3-dit-dora-test-base.gguf";
    // Every site F16, except ffn_out (site 3) in `qt` when it is a quant type.
    auto write_base = [&](const std::string & path, ggml_type qt) {
        gguf_context *   g   = gguf_init_empty();
        ggml_init_params hip = { ggml_tensor_overhead() * 8 + (size_t) 64 * 1024, nullptr, false };
        ggml_context *   hc  = ggml_init(hip);
        for (size_t k = 0; k < sites.size(); k++) {
            const ggml_type ty = (k == 3) ? qt : GGML_TYPE_F16;
            ggml_tensor *   t  = ggml_new_tensor_2d(hc, ty, sites[k].in, sites[k].out);
            ggml_set_name(t, sites[k].gguf);
            if (ty == GGML_TYPE_F16) {
                ggml_fp32_to_fp16_row(sites[k].w.data(), (ggml_fp16_t *) t->data, ggml_nelements(t));
            } else {
                ggml_quantize_chunk(ty, sites[k].w.data(), t->data, 0, sites[k].out, sites[k].in, nullptr);
            }
            gguf_add_tensor(g, t);
        }
        GGML_ASSERT(gguf_write_to_file(g, path.c_str(), false));
        gguf_free(g);
        ggml_free(hc);
    };
    write_base(gpath, GGML_TYPE_F16);
    std::vector<std::string> names;
    for (const Site & st : sites) names.push_back(st.gguf);

    // Export through the real writer.
    const std::string apath = dir + "/mm3-dit-dora-test-adapter.safetensors";
    CHECK(mm3_train_export(ad, m, R, 8.0f, apath, &err), "export: %s", err.c_str());
    std::string               use_dora;
    const std::vector<StEnt> exported = st_read_all(apath, &use_dora);
    CHECK(use_dora == "true", "export metadata use_dora = '%s'", use_dora.c_str());
    {
        int n_mag = 0;
        for (const StEnt & e : exported) n_mag += e.name.find(".lora_magnitude_vector.weight") != std::string::npos;
        CHECK(n_mag == 4, "export wrote %d magnitudes, want 4", n_mag);
    }

    // 1+2+3 through the loader: merged W' vs host W' and vs the trainer.
    std::vector<std::vector<float>> merged, base0, half;
    CHECK(merge_into(gpath, names, apath, 1.0f, &merged, nullptr) == 4, "DoRA merge did not patch 4 tensors");
    for (size_t k = 0; k < sites.size() && merged.size() == sites.size(); k++) {
        const Site &             st  = sites[k];
        const std::vector<float> ref = host_wprime(st, true);   // BF16 A/B, as merged
        double                   worst = 0.0, scale = 0.0, dmax = 0.0;
        for (size_t i = 0; i < ref.size(); i++) {
            track_worst(worst, std::fabs(merged[k][i] - ref[i]));
            scale = std::max(scale, (double) std::fabs(ref[i]));
            dmax  = std::max(dmax, (double) std::fabs(ref[i] - st.w[i]));
        }
        // Reference rounds A and B through BF16 like the merge (PEFT parity,
        // as for plain LoRA). What is left is the BF16 round of W' - W0 (at
        // most 2^-8 of its largest element) and the F16 store (2^-11 of |W'|).
        const double tol = dmax / 256.0 + scale / 2048.0 + 1e-6;
        CHECK(worst < tol, "%s: merged W' vs host max err %.3g, bound %.3g", st.gguf, worst, tol);
        if (k == 0) {
            double qk = 0.0;
            for (size_t i = 0; i < (size_t) (2 * E * st.in); i++) track_worst(qk, std::fabs(merged[k][i] - st.w[i]));
            CHECK(qk < 2e-3, "masked q,k rows moved in the merge: max %.3g", qk);
        }
        if (k == 1) {
            for (int64_t i = 0; i < st.in; i++) CHECK(merged[k][(size_t) i] == 0.0f, "zero row not zero after merge");
        }
        std::vector<float>        xs(x.begin(), x.begin() + st.in * S);
        const std::vector<double> ym = host_matvec(merged[k], xs, st.in, st.out);
        const std::vector<float>  yt = trainer_forward(wts[k], *st.lo, xs);
        double                    wy = 0.0, sy = 0.0;
        for (size_t i = 0; i < yt.size(); i++) {
            track_worst(wy, std::fabs(ym[i] - yt[i]));
            sy = std::max(sy, (double) std::fabs(yt[i]));
        }
        CHECK(wy < 1e-2 * sy + 1e-5, "%s: merged vs trainer forward max err %.3g (|y| max %.3g)", st.gguf, wy, sy);
        fprintf(stderr, "[mm3-dit-dora] merge    %-30s max |W'-ref| %.3g, max |y_merge - y_train| %.3g of %.3g\n",
                st.gguf, worst, wy, sy);
    }

    // 4: strength.
    CHECK(merge_into(gpath, names, apath, 0.0f, &base0, nullptr) == 4, "scale 0 merge count");
    CHECK(merge_into(gpath, names, apath, 0.5f, &half, nullptr) == 4, "scale 0.5 merge count");
    for (size_t k = 0; k < sites.size() && base0.size() == sites.size() && half.size() == sites.size(); k++) {
        double w0 = 0.0, wh = 0.0;
        for (size_t i = 0; i < sites[k].w.size(); i++) {
            track_worst(w0, std::fabs(base0[k][i] - sites[k].w[i]));
            const double want = sites[k].w[i] + 0.5 * ((double) merged[k][i] - sites[k].w[i]);
            track_worst(wh, std::fabs(half[k][i] - want));
        }
        CHECK(w0 == 0.0, "%s: strength 0 is not the base (max %.3g)", sites[k].gguf, w0);
        CHECK(wh < 2e-3, "%s: strength 0.5 is not W0 + 0.5*(W'-W0) (max %.3g)", sites[k].gguf, wh);
    }

    // 6: malformed DoRA files merge nothing.
    auto variant = [&](const char * tag, auto edit) {
        std::vector<StEnt> ents = exported;
        edit(ents);
        const std::string p = dir + "/mm3-dit-dora-test-bad.safetensors";
        st_write(p, ents, "\"format\":\"pt\",\"use_dora\":\"true\"");
        bool      untouched = false;
        const int n         = merge_into(gpath, names, p, 1.0f, nullptr, &untouched);
        CHECK(n == -1 && untouched, "%s: merged %d, untouched %d (want -1, 1)", tag, n, (int) untouched);
        remove(p.c_str());
    };
    variant("missing magnitude", [](std::vector<StEnt> & e) {
        for (size_t i = 0; i < e.size(); i++) {
            if (e[i].name.find("ff.ff.2.lora_magnitude") != std::string::npos) {
                e.erase(e.begin() + (long) i);
                return;
            }
        }
    });
    variant("NaN magnitude", [](std::vector<StEnt> & e) {
        for (StEnt & s : e) {
            if (s.name.find("to_out.lora_magnitude") != std::string::npos) s.data[3] = NAN;
        }
    });
    variant("short magnitude", [](std::vector<StEnt> & e) {
        for (StEnt & s : e) {
            if (s.name.find("to_qkv.lora_magnitude") != std::string::npos) {
                s.data.pop_back();
                s.shape[0]--;
            }
        }
    });
    variant("magnitude without factors", [](std::vector<StEnt> & e) {
        e.push_back({ "diffusion_model.diffusion_transformer.transformer.layers.0.ff.ff.9.lora_magnitude_vector.weight",
                      { 4 }, { 1, 1, 1, 1 } });
    });

    // 7: plain LoRA. Same factors, no magnitudes, so the merge is W0 + s*BA.
    {
        std::vector<StEnt> ents;
        for (const StEnt & e : exported) {
            if (e.name.find("magnitude") == std::string::npos) ents.push_back(e);
        }
        const std::string p = dir + "/mm3-dit-dora-test-lora.safetensors";
        st_write(p, ents, "\"format\":\"pt\"");
        std::vector<std::vector<float>> lm;
        CHECK(merge_into(gpath, names, p, 1.0f, &lm, nullptr) == 4, "plain LoRA merge count");
        for (size_t k = 0; k < sites.size() && lm.size() == sites.size(); k++) {
            const std::vector<float> a = get(sites[k].lo->a), b = get(sites[k].lo->b);
            double                   worst = 0.0;
            for (int64_t o = 0; o < sites[k].out; o++) {
                for (int64_t i = 0; i < sites[k].in; i++) {
                    double d = 0.0;
                    for (int64_t r = 0; r < R; r++) d += (double) a[(size_t) (r * sites[k].in + i)] * b[(size_t) (o * R + r)];
                    track_worst(worst, std::fabs(lm[k][(size_t) (o * sites[k].in + i)] -
                                                 (sites[k].w[(size_t) (o * sites[k].in + i)] + sc * d)));
                }
            }
            CHECK(worst < 4e-3, "%s: plain LoRA merge is not W0 + s*BA (max %.3g)", sites[k].gguf, worst);
        }
        remove(p.c_str());
    }


    // 10: adapter_config.json declares DoRA. A PEFT directory saying
    // use_dora:true with no magnitudes, or use_dora:false over a file that has
    // them, merges nothing; a config that agrees with the tensors merges.
    {
        const std::string pd = dir + "/mm3-dit-dora-test-peft";
        std::filesystem::create_directories(pd);
        auto cfg = [&](const char * v) {
            FILE * f = fopen((pd + "/adapter_config.json").c_str(), "wb");
            GGML_ASSERT(f);
            fprintf(f, "{\"peft_type\":\"LORA\",\"r\":%d,\"lora_alpha\":8,\"use_dora\":%s}", (int) R, v);
            fclose(f);
        };
        std::vector<StEnt> lora_only;
        for (const StEnt & e : exported) {
            if (e.name.find("magnitude") == std::string::npos) lora_only.push_back(e);
        }
        bool untouched = false;
        st_write(pd + "/adapter_model.safetensors", lora_only, "\"format\":\"pt\"");
        cfg("true");
        int n = merge_into(gpath, names, pd, 1.0f, nullptr, &untouched);
        CHECK(n == -1 && untouched, "config use_dora=true, no magnitudes: merged %d, untouched %d", n, (int) untouched);
        st_write(pd + "/adapter_model.safetensors", exported, "\"format\":\"pt\"");
        cfg("false");
        n = merge_into(gpath, names, pd, 1.0f, nullptr, &untouched);
        CHECK(n == -1 && untouched, "config use_dora=false over magnitudes: merged %d, untouched %d", n,
              (int) untouched);
        cfg("true");
        n = merge_into(gpath, names, pd, 1.0f, nullptr, nullptr);
        CHECK(n == 4, "config use_dora=true with magnitudes: merged %d, want 4", n);
        fprintf(stderr, "[mm3-dit-dora] config   use_dora declaration checked against the tensors (3 cases)\n");
        std::filesystem::remove_all(pd);
    }

    // 11: quantized bases. ffn_out (in = FI = 32, one block per row) is stored
    // Q8_0 and then Q4_0, the rest F16. The merge reads the quantized W0,
    // writes W' back in the same type, and lands within one quantization step
    // of DoRA(dequant(W0)). A backend that cannot dequantize the type in-graph
    // must refuse the whole adapter instead.
    for (ggml_type qt : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 }) {
        const Site &         st = sites[3];
        std::vector<uint8_t> q(ggml_row_size(qt, st.in) * (size_t) st.out);
        ggml_quantize_chunk(qt, st.w.data(), q.data(), 0, st.out, st.in, nullptr);
        std::vector<float> w0q(st.w.size());
        ggml_get_type_traits(qt)->to_float(q.data(), w0q.data(), (int64_t) w0q.size());

        const std::string qpath = dir + "/mm3-dit-dora-test-base-q.gguf";
        write_base(qpath, qt);
        std::vector<std::vector<float>> qm;
        std::vector<ggml_type>          qtypes;
        bool                            untouched = false;
        const int n = merge_into(qpath, names, apath, 1.0f, &qm, &untouched, &qtypes);
        if (!mm3_backend_can_cast_f32(g_bp.backend, qt)) {
            CHECK(n == -1 && untouched, "%s base on a backend that cannot cast it: merged %d", ggml_type_name(qt), n);
            fprintf(stderr, "[mm3-dit-dora] quant    %-6s refused (backend cannot dequantize it in-graph)\n",
                    ggml_type_name(qt));
        } else {
            CHECK(n == 4, "%s base: merged %d, want 4", ggml_type_name(qt), n);
            if (qm.size() == sites.size()) {
                CHECK(qtypes[3] == qt, "%s base came back as %s", ggml_type_name(qt), ggml_type_name(qtypes[3]));
                const std::vector<float> ref = host_wprime(st, true, &w0q);
                double                   worst = 0.0, scale = 0.0, dmax = 0.0;
                for (size_t i = 0; i < ref.size(); i++) {
                    track_worst(worst, std::fabs(qm[3][i] - ref[i]));
                    scale = std::max(scale, (double) std::fabs(ref[i]));
                    dmax  = std::max(dmax, (double) std::fabs(ref[i] - w0q[i]));
                }
                // One quant step of the largest block (Q8_0: max/127, Q4_0:
                // max/8) plus the BF16 round of W' - W0.
                const double tol = scale / (qt == GGML_TYPE_Q8_0 ? 127.0 : 8.0) + dmax / 256.0 + 1e-6;
                CHECK(worst < tol, "%s base: merged W' vs DoRA(W0q) max err %.3g, bound %.3g", ggml_type_name(qt),
                      worst, tol);
                fprintf(stderr, "[mm3-dit-dora] quant    %-6s ffn_out merged in place, max |W'-ref| %.3g (bound %.3g)\n",
                        ggml_type_name(qt), worst, tol);
            }
        }
        remove(qpath.c_str());
    }

    remove(apath.c_str());
    remove(gpath.c_str());

    lm_dial_checks();
    ggml_backend_buffer_free(abuf);
    ggml_backend_buffer_free(wbuf);
    ggml_free(actx);
    ggml_free(wctx);
    ggml_backend_sched_free(g_sched);
    backend_release(g_bp.backend, g_bp.cpu_backend);

    if (g_fail) {
        fprintf(stderr, "[mm3-dit-dora] FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    fprintf(stderr, "[mm3-dit-dora] PASS\n");
    return 0;
}
