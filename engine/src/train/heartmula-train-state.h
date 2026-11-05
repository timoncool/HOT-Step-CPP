#pragma once
// train/heartmula-train-state.h — what the HeartMuLa LoRA trainer keeps:
// the frozen model (optionally cast to BF16), the trainable factors, the
// prepared dataset and its deterministic order, the resumable checkpoint and
// the adapter GGUF that heartmula/adapter.h loads.
//
// HOT-Step file, TRAINING-SIDE (ace-train only).
//
// ── Determinism (contract §5.4) ────────────────────────────────────────────
//
// Nothing random is carried as state. The sample order of epoch e is a
// seeded shuffle of (seed, e); a window start is a pure function of (seed,
// step, slot); the LoRA init is a function of the seed. So "data cursor and
// RNG" are the number of optimizer steps done, and a resumed run sees the
// stream the uninterrupted one would have seen.
//
// ── Checkpoint ─────────────────────────────────────────────────────────────
//
// <out>/hm_train_state.bin: magic, version, the run identity (every setting
// that changes WHAT is trained), the step counters, every A/B factor and
// every AdamW moment. Written to .tmp and renamed, so an interrupted write
// leaves the previous checkpoint intact. Single machine, host-endian, not a
// distribution format.
//
// ── Adapter ────────────────────────────────────────────────────────────────
//
// <out>/adapters/step-NNNNNN.gguf, the layout of convert-heartmula-lora.py
// (engine/docs/HEARTMULA_CONTRACT.md "LoRA adapter"): F32 A [in, r] and
// B [r, out] per backbone q/k/v/output projection (plus mlp w1/w2/w3 with
// --lora-mlp), keyed to the base model's
// variant and fingerprint. Immutable: written once through .tmp + rename and
// never overwritten.

#include "heartmula/adapter.h"
#include "train/heartmula-prepare.h"
#include "train/heartmula-train-graph.h"
#include "train/yue2-train-optim.h"

#include <filesystem>
#include <fstream>
#include <random>
#include <unordered_map>

#define HMT_STATE_FILE    "hm_train_state.bin"
#define HMT_STATE_VERSION 1
static const char HMT_STATE_MAGIC[8] = { 'H', 'M', 'T', 'R', 'A', 'I', 'N', '1' };

// ── Model ──────────────────────────────────────────────────────────────────
//
// heartmula_model_load's checks and layout, with one difference: `bf16` loads
// every matrix the backward multiplies through (block projections, the heads,
// the projection) as BF16, converted on the host from the F16 file. Embedding
// tables, norms and the MuQ bias load as inference loads them.
//
// F16 is the default because it measured closer to the FP32 oracle at the
// same speed (selftest, crop 1: gradients rel L2 0.004 against 0.011, logits
// 3e-4 against 3e-3; clip A 466 against 468 ms per micro-batch). BF16 stays as
// the arm to try if long clips ever underflow the F16 copy of the gradient
// operand.
static bool hmt_model_load(HeartMuLaModel * m, const char * gpath, const char * lpath, bool bf16, std::string * err) {
    if (!bf16) return heartmula_model_load(m, gpath, lpath, err);
    using namespace heartmula_model_detail;
    heartmula_model_free(m);
    GGUFModel gg = {}, gl = {};
    if (!gf_load(&gg, gpath)) return fail(err, std::string("cannot open ") + gpath);
    if (!gf_load(&gl, lpath)) {
        gf_close(&gg);
        return fail(err, std::string("cannot open ") + lpath);
    }
    auto close = [&]() { gf_close(&gg); gf_close(&gl); };
    if (!check_header(gg, gpath, "heartmula-global", err) || !check_header(gl, lpath, "heartmula-local", err)) {
        close();
        return false;
    }
    m->variant          = gf_get_str(gg, "heartmula.variant");
    m->base_fingerprint = gf_get_str(gg, "heartmula.base_fingerprint");
    if (m->variant != gf_get_str(gl, "heartmula.variant") || m->base_fingerprint != gf_get_str(gl, "heartmula.base_fingerprint")) {
        close();
        return fail(err, std::string(lpath) + ": base_fingerprint differs from " + gpath);
    }
    m->global.cfg = heartmula_global_config();
    m->local.cfg  = heartmula_local_config();
    auto mat = [&](WeightCtx * w, const GGUFModel & gf, const std::string & name) -> ggml_tensor * {
        ggml_tensor * src = ggml_get_tensor(gf.meta, name.c_str());
        if (!src || src->type != GGML_TYPE_F16) return gf_load_tensor(w, gf, name);
        ggml_tensor * t = ggml_new_tensor(w->ctx, GGML_TYPE_BF16, ggml_n_dims(src), src->ne);
        ggml_set_name(t, name.c_str());
        const size_t n   = (size_t) ggml_nelements(src);
        auto         buf = std::make_unique<float[]>(n / 2 + 1);
        const auto * in  = (const ggml_fp16_t *) (gf.mapping + gf.data_offset + gguf_get_tensor_offset(gf.gguf, gguf_find_tensor(gf.gguf, name.c_str())));
        auto *       out = (ggml_bf16_t *) buf.get();
        for (size_t i = 0; i < n; i++) out[i] = ggml_fp32_to_bf16(ggml_fp16_to_fp32(in[i]));
        w->pending.push_back({ t, buf.get(), n * sizeof(ggml_bf16_t), 0 });
        w->staging.push_back(std::move(buf));
        return t;
    };
    auto stack = [&](WeightCtx * w, const GGUFModel & gf, const std::string & prefix, HeartMuLaStack * s) {
        s->layers.resize((size_t) s->cfg.n_layer);
        for (int i = 0; i < s->cfg.n_layer; i++) {
            const std::string p = prefix + ".layers." + std::to_string(i) + ".";
            HeartMuLaLayer &  l = s->layers[(size_t) i];
            l.sa_norm  = gf_load_tensor_f32(w, gf, p + "sa_norm.scale");
            l.q        = mat(w, gf, p + "attn.q_proj.weight");
            l.k        = mat(w, gf, p + "attn.k_proj.weight");
            l.v        = mat(w, gf, p + "attn.v_proj.weight");
            l.o        = mat(w, gf, p + "attn.output_proj.weight");
            l.mlp_norm = gf_load_tensor_f32(w, gf, p + "mlp_norm.scale");
            l.w1       = mat(w, gf, p + "mlp.w1.weight");
            l.w2       = mat(w, gf, p + "mlp.w2.weight");
            l.w3       = mat(w, gf, p + "mlp.w3.weight");
        }
        s->norm = gf_load_tensor_f32(w, gf, prefix + ".norm.scale");
        const std::vector<float> ff = heartmula_rope_freq_factors(s->cfg.head_dim, s->cfg.rope_base);
        s->freq_factors = ggml_new_tensor_1d(w->ctx, GGML_TYPE_F32, (int64_t) ff.size());
        ggml_set_name(s->freq_factors, (prefix + ".rope_freq_factors").c_str());
        auto buf = std::make_unique<float[]>(ff.size());
        std::copy(ff.begin(), ff.end(), buf.get());
        w->pending.push_back({ s->freq_factors, buf.get(), ff.size() * sizeof(float), 0 });
        w->staging.push_back(std::move(buf));
    };
    std::vector<std::string> gnames = stack_names("backbone", m->global.cfg.n_layer);
    for (const char * n : { "text_embeddings.weight", "audio_embeddings.weight", "unconditional_text_embedding.weight",
                            "muq_linear.bias", "codebook0_head.weight" }) gnames.push_back(n);
    std::vector<std::string> lnames = stack_names("decoder", m->local.cfg.n_layer);
    lnames.push_back("projection.weight");
    for (int i = 0; i < 7; i++) lnames.push_back("audio_head." + std::to_string(i) + ".weight");
    if (!has_all(gg, gnames, gpath, err) || !has_all(gl, lnames, lpath, err)) {
        close();
        return false;
    }
    m->bp          = backend_init("HeartMuLa train");
    m->backend_ref = true;
    wctx_init(&m->wctx_global, (int) gnames.size() + 1);
    stack(&m->wctx_global, gg, "backbone", &m->global);
    m->text_embd   = gf_load_tensor(&m->wctx_global, gg, "text_embeddings.weight");
    m->audio_embd  = gf_load_tensor(&m->wctx_global, gg, "audio_embeddings.weight");
    m->uncond_embd = gf_load_tensor_f32(&m->wctx_global, gg, "unconditional_text_embedding.weight");
    m->muq_bias    = gf_load_tensor_f32(&m->wctx_global, gg, "muq_linear.bias");
    m->c0_head     = mat(&m->wctx_global, gg, "codebook0_head.weight");
    wctx_init(&m->wctx_local, (int) lnames.size() + 1);
    stack(&m->wctx_local, gl, "decoder", &m->local);
    m->projection = mat(&m->wctx_local, gl, "projection.weight");
    for (int i = 0; i < 7; i++) m->audio_head[i] = mat(&m->wctx_local, gl, "audio_head." + std::to_string(i) + ".weight");
    if (!check_shapes(m->global, gpath, err) || !check_shapes(m->local, lpath, err)) {
        close();
        heartmula_model_free(m);
        return false;
    }
    const bool ok = wctx_alloc(&m->wctx_global, m->bp.backend) && wctx_alloc(&m->wctx_local, m->bp.backend);
    close();
    if (!ok) {
        heartmula_model_free(m);
        return fail(err, "HeartMuLa weight allocation failed (out of VRAM?)");
    }
    m->loaded = true;
    return true;
}

// ── Trainable factors ──────────────────────────────────────────────────────

static std::string hmt_target(int layer, int s) {
    static const char * n[HMT_SITES] = { "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.output_proj",
                                         "mlp.w1", "mlp.w2", "mlp.w3" };
    return "backbone.layers." + std::to_string(layer) + "." + n[s] + ".weight";
}

struct HmtLora {
    ggml_context *             ctx = nullptr;
    ggml_backend_buffer_t      buf = nullptr;
    std::vector<HmtLayerLora>  layers;
    std::vector<ggml_tensor *> params;  // A, B per site, layer-major (q, k, v, o[, w1, w2, w3])
    ggml_tensor * t_adamw = nullptr, * t_lossgrad = nullptr, * t_clip = nullptr, * t_eps = nullptr, * t_gnorm2 = nullptr;
    int   rank  = 0;
    float alpha = 0.0f;
};

static void hmt_lora_free(HmtLora * l) {
    if (l->buf) ggml_backend_buffer_free(l->buf);
    if (l->ctx) ggml_free(l->ctx);
    *l = HmtLora{};
}

// Kaiming-uniform A (bound 1/sqrt(in), torchtune's a = sqrt 5), zero B (contract §1).
static bool hmt_lora_init(HmtLora * L, const HeartMuLaModel & m, int rank, float alpha, uint64_t seed, bool mlp,
                          std::string * err) {
    hmt_lora_free(L);
    const int n_layer = m.global.cfg.n_layer;
    const int n_sites = mlp ? HMT_SITES : 4;
    ggml_init_params ip = { (size_t) (2 * HMT_SITES * n_layer + 16) * ggml_tensor_overhead(), nullptr, true };
    L->ctx   = ggml_init(ip);
    L->rank  = rank;
    L->alpha = alpha;
    L->layers.resize((size_t) n_layer);
    for (int i = 0; i < n_layer; i++) {
        const HeartMuLaLayer & b    = m.global.layers[(size_t) i];
        ggml_tensor * const    w[HMT_SITES] = { b.q, b.k, b.v, b.o, b.w1, b.w2, b.w3 };
        for (int s = 0; s < n_sites; s++) {
            HmtSite & site = L->layers[(size_t) i].s[s];
            site.A = ggml_new_tensor_2d(L->ctx, GGML_TYPE_F32, w[s]->ne[0], rank);
            site.B = ggml_new_tensor_2d(L->ctx, GGML_TYPE_F32, rank, w[s]->ne[1]);
            ggml_set_name(site.A, (hmt_target(i, s) + ".lora_a").c_str());
            ggml_set_name(site.B, (hmt_target(i, s) + ".lora_b").c_str());
            ggml_set_param(site.A);
            ggml_set_param(site.B);
            L->params.push_back(site.A);
            L->params.push_back(site.B);
        }
    }
    L->t_adamw    = ggml_new_tensor_1d(L->ctx, GGML_TYPE_F32, 7);
    L->t_lossgrad = ggml_new_tensor_1d(L->ctx, GGML_TYPE_F32, 1);
    L->t_clip     = ggml_new_tensor_1d(L->ctx, GGML_TYPE_F32, 1);
    L->t_eps      = ggml_new_tensor_1d(L->ctx, GGML_TYPE_F32, 1);
    L->t_gnorm2   = ggml_new_tensor_1d(L->ctx, GGML_TYPE_F32, 1);
    L->buf = ggml_backend_alloc_ctx_tensors(L->ctx, m.bp.backend);
    if (!L->buf) {
        *err = "LoRA factor allocation failed";
        return false;
    }
    // Without the WEIGHTS hint the scheduler splits the graph at every factor.
    ggml_backend_buffer_set_usage(L->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(L->buf, 0);
    std::mt19937_64 rng(seed ^ 0x48656172744d754cull);
    std::vector<float> a;
    for (auto & layer : L->layers) {
        for (auto & site : layer.s) {
            if (!site.A) continue;
            const float bound = 1.0f / sqrtf((float) site.A->ne[0]);
            std::uniform_real_distribution<float> u(-bound, bound);
            a.resize((size_t) ggml_nelements(site.A));
            for (auto & x : a) x = u(rng);
            ggml_backend_tensor_set(site.A, a.data(), 0, a.size() * sizeof(float));
        }
    }
    const float one = 1.0f, eps = 1e-6f;
    ggml_backend_tensor_set(L->t_lossgrad, &one, 0, sizeof(float));
    ggml_backend_tensor_set(L->t_eps, &eps, 0, sizeof(float));
    return true;
}

static int64_t hmt_lora_param_count(const HmtLora & L) {
    int64_t n = 0;
    for (auto * p : L.params) n += ggml_nelements(p);
    return n;
}

// ── Dataset ────────────────────────────────────────────────────────────────

struct HmtSample {
    std::string          id, key, codes_path;
    std::vector<int32_t> prompt;
    int64_t              muq_idx = 0, frames = 0;
};

struct HmtData {
    std::vector<HmtSample> samples;
    uint64_t               fingerprint = 0;  // FNV-1a over what each sample trains on
};

static uint64_t hmt_fnv(uint64_t h, const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// Reads <dir>/heartmula_prepare.json (heartmula-prepare.h). Refuses another version.
static bool hmt_data_load(const std::string & dir, HmtData * d, std::string * err) {
    namespace fs = std::filesystem;
    const std::string path = (fs::u8path(dir) / HEARTMULA_PREPARE_MANIFEST).u8string();
    std::vector<uint8_t> buf;
    if (!hp_read_file(path, &buf)) { *err = "cannot read " + path + " (run heartmula-prepare first)"; return false; }
    yyjson_doc * doc = yyjson_read((const char *) buf.data(), buf.size(), 0);
    if (!doc) { *err = path + " is not valid JSON"; return false; }
    yyjson_val * root = yyjson_doc_get_root(doc);
    const bool   ok   = std::string(pm_js_str(root, "format")) == "heartmula-prepare" &&
                        yyjson_get_int(yyjson_obj_get(root, "version")) == HEARTMULA_PREPARE_VERSION;
    if (!ok) {
        yyjson_doc_free(doc);
        *err = path + ": not a heartmula-prepare v" + std::to_string(HEARTMULA_PREPARE_VERSION) + " manifest; prepare again";
        return false;
    }
    d->samples.clear();
    d->fingerprint = 1469598103934665603ull;
    size_t i, n;
    yyjson_val * v;
    yyjson_arr_foreach(yyjson_obj_get(root, "samples"), i, n, v) {
        HmtSample s;
        s.id         = pm_js_str(v, "id");
        s.key        = pm_js_str(v, "key");
        s.codes_path = (fs::u8path(dir) / fs::u8path(pm_js_str(v, "codes"))).u8string();
        s.frames     = yyjson_get_int(yyjson_obj_get(v, "frames"));
        s.muq_idx    = yyjson_get_int(yyjson_obj_get(v, "muq_idx"));
        size_t j, m;
        yyjson_val * p;
        yyjson_arr_foreach(yyjson_obj_get(v, "prompt_ids"), j, m, p) s.prompt.push_back((int32_t) yyjson_get_int(p));
        if (s.frames < 2 || s.prompt.empty() || s.muq_idx < 0 || s.muq_idx >= (int64_t) s.prompt.size()) {
            yyjson_doc_free(doc);
            *err = path + ": sample " + s.id + " is malformed";
            return false;
        }
        d->fingerprint = hmt_fnv(d->fingerprint, s.key.data(), s.key.size());
        d->fingerprint = hmt_fnv(d->fingerprint, &s.frames, sizeof(s.frames));
        d->fingerprint = hmt_fnv(d->fingerprint, &s.muq_idx, sizeof(s.muq_idx));
        d->fingerprint = hmt_fnv(d->fingerprint, s.prompt.data(), s.prompt.size() * sizeof(int32_t));
        d->samples.push_back(std::move(s));
    }
    yyjson_doc_free(doc);
    if (d->samples.empty()) { *err = path + " holds no prepared samples (all were skipped?); nothing to train on"; return false; }
    return true;
}

// Frames [w0, w0 + T) of a sample's codes, frame-major int32.
static bool hmt_read_codes(const HmtSample & s, int64_t w0, int64_t T, std::vector<int32_t> * out, std::string * err) {
    FILE * f = hs_fopen(s.codes_path, "rb");
    if (!f) { *err = "cannot open " + s.codes_path + " (prepare again)"; return false; }
    out->resize((size_t) (T * 8));
    const bool ok = fseek(f, (long) (w0 * 32), SEEK_SET) == 0 && fread(out->data(), 32, (size_t) T, f) == (size_t) T;
    fclose(f);
    if (!ok) { *err = s.codes_path + " is shorter than its manifest row says (prepare again)"; return false; }
    for (int32_t c : *out) {
        if (c < 0 || c >= 8192) { *err = s.codes_path + " holds a code outside 0..8191"; return false; }
    }
    return true;
}

static uint64_t hmt_mix(uint64_t seed, uint64_t a, uint64_t b) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ull * (a + 1) + 0xD1B54A32D192ED03ull * (b + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// One micro-step of an optimizer step: which sample, which window, and whether an
// end-of-song frame follows it.
struct HmtPick {
    int     sample = 0;
    int64_t w0 = 0, T = 0;
    bool    eos = false;
};

// Song structure in training (--eos, --crop-frames):
//
// EOS. Inference ends a song at the first frame holding a code >= 8193 (pipeline.h).
// A window that reaches the real end of its song gets one more frame of 8193 in all
// eight codebooks, so the last real frame becomes an input that predicts the end. The
// labels are 8197 wide already, so nothing else changes.
//
// Crop. The model is causal: the loss on frames [0, N) of a song is the same whether
// the rest of the song follows or not. A window anchored at the start is therefore the
// full sequence with its tail's loss left out, and teaches nothing false. A window from
// the middle would teach that songs start mid-song, since the model can only know where
// it is from what came before. So crop windows start at frame 0, and one sample in
// HMT_FULL_EVERY trains the whole song (with its EOS frame), so later sections and
// endings are still learned. --max-frames keeps the older random windows (selftest).
#define HMT_FULL_EVERY 4
#define HMT_EOS_CODE   8193

// Whether a window of T frames that reaches its song's end can carry the end frame:
// the backbone then holds S + (T + 1) - 1 rows, which must fit the context. Training
// (hmt_schedule) and scoring (run.h hmt_eval) both decide with this.
static bool hmt_eos_fits(int64_t S, int64_t T, int64_t ctx_limit) {
    return S + T <= ctx_limit;
}

// Per-codebook loss factors, graph.h HmtGraphCtx::cbw. Default: every factor 1,
// so the loss is the plain mean (1/8) sum_k CE_k. --cb0-weighted: the SFT
// weighting of the HeartMuLa paper, CB0 2.0 and CB1..7 0.7 down to 0.1,
// normalised to sum to 1 (8 w_k / sum w, since wce already holds the 1/8):
// loss = sum_k (w_k / 4.8) CE_k, so a run whose CEs are all equal reports the
// same loss either way and the LR means the same thing.
static void hmt_cb_weights(bool weighted, float out[8]) {
    static const float w[8] = { 2.0f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f };
    float sum = 0.0f;
    for (float x : w) sum += x;
    for (int k = 0; k < 8; k++) out[k] = weighted ? 8.0f * w[k] / sum : 1.0f;
}

// Optimizer step `step` (0-based). Epochs are a seeded shuffle; an epoch has
// ceil(N / accum) steps and its last one may be short (contract §1: each
// micro-loss is divided by the micro-steps actually in the step).
static std::vector<HmtPick> hmt_schedule(const HmtData & d, int accum, int64_t max_frames, uint64_t seed, int64_t step,
                                         int64_t crop_frames = 0, bool eos = false, int64_t ctx_limit = 8192) {
    const int64_t N = (int64_t) d.samples.size(), per = (N + accum - 1) / accum;
    const int64_t epoch = step / per, j = step % per;
    std::vector<int> order((size_t) N);
    for (int i = 0; i < (int) N; i++) order[(size_t) i] = i;
    std::mt19937_64 rng(hmt_mix(seed, (uint64_t) epoch, 1));
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<HmtPick> out;
    for (int64_t k = j * accum; k < std::min(N, (j + 1) * accum); k++) {
        HmtPick p;
        p.sample = order[(size_t) k];
        const int64_t F = d.samples[(size_t) p.sample].frames;
        if (crop_frames > 0) {
            const bool full = F <= crop_frames || hmt_mix(seed, (uint64_t) step, (uint64_t) (k + 2)) % HMT_FULL_EVERY == 0;
            p.T  = full ? F : crop_frames;
            p.w0 = 0;
        } else {
            p.T  = max_frames > 0 ? std::min(F, max_frames) : F;
            p.w0 = F > p.T ? (int64_t) (hmt_mix(seed, (uint64_t) step, (uint64_t) (k + 2)) % (uint64_t) (F - p.T + 1)) : 0;
        }
        p.eos = eos && p.w0 + p.T == F && hmt_eos_fits((int64_t) d.samples[(size_t) p.sample].prompt.size(), p.T, ctx_limit);
        out.push_back(p);
    }
    return out;
}

// ── Checkpoint ─────────────────────────────────────────────────────────────

// Everything that decides WHAT is trained. A resume into anything else refuses.
struct HmtIdentity {
    int32_t  rank = 0, accum = 0, bf16 = 0;
    float    alpha = 0, lr = 0, wd = 0, beta1 = 0, beta2 = 0, eps = 0, clip = 0, dropout = 0;
    uint64_t seed = 0, data = 0;
    int64_t  max_frames = 0;
    int32_t  mlp = 0;  // --lora-mlp
    int32_t  eos = 0;  // --eos
    int64_t  crop_frames = 0;  // --crop-frames
    int32_t  cb0_weighted = 0;  // --cb0-weighted
    std::string base_fingerprint;
    std::string weights;  // hmt_weights_id of the global and local GGUFs actually loaded
};

// What the frozen weights are, beyond base_fingerprint (which F16 and every quant
// of one base share): file size plus each tensor's name, type and shape. Telling
// F16 from Q8_0, or two different quant recipes, needs no 7 GB content hash.
static std::string hmt_weights_id(const std::string & path) {
    ggml_context * meta = nullptr;
    gguf_context * g    = gguf_init_from_file(path.c_str(), { true, &meta });
    if (!g) return "unreadable:" + path;
    std::error_code ec;
    uint64_t h = hmt_fnv(1469598103934665603ull, nullptr, 0);
    const uint64_t bytes = (uint64_t) std::filesystem::file_size(std::filesystem::u8path(path), ec);
    h = hmt_fnv(h, &bytes, sizeof(bytes));
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        const ggml_tensor * t = ggml_get_tensor(meta, gguf_get_tensor_name(g, i));
        h = hmt_fnv(h, t->name, strlen(t->name));
        h = hmt_fnv(h, &t->type, sizeof(t->type));
        h = hmt_fnv(h, t->ne, sizeof(t->ne));
    }
    gguf_free(g);
    ggml_free(meta);
    char b[24];
    snprintf(b, sizeof(b), "%016llx", (unsigned long long) h);
    return b;
}

static uint64_t hmt_identity_hash(const HmtIdentity & id) {
    uint64_t h = 1469598103934665603ull;
    h = hmt_fnv(h, &id.rank, sizeof(id.rank));
    h = hmt_fnv(h, &id.accum, sizeof(id.accum));
    h = hmt_fnv(h, &id.bf16, sizeof(id.bf16));
    for (float f : { id.alpha, id.lr, id.wd, id.beta1, id.beta2, id.eps, id.clip }) h = hmt_fnv(h, &f, sizeof(f));
    h = hmt_fnv(h, &id.seed, sizeof(id.seed));
    h = hmt_fnv(h, &id.data, sizeof(id.data));
    h = hmt_fnv(h, &id.max_frames, sizeof(id.max_frames));
    if (id.dropout != 0.0f) h = hmt_fnv(h, &id.dropout, sizeof(id.dropout));  // p = 0 keeps older states resumable
    if (id.mlp) h = hmt_fnv(h, "mlp", 3);  // attention-only keeps older states resumable
    if (id.eos) h = hmt_fnv(h, "eos", 3);  // off keeps older states resumable
    if (id.crop_frames) h = hmt_fnv(h, &id.crop_frames, sizeof(id.crop_frames));
    if (id.cb0_weighted) h = hmt_fnv(h, "cbw", 3);  // off keeps older states resumable
    h = hmt_fnv(h, id.weights.data(), id.weights.size());
    return hmt_fnv(h, id.base_fingerprint.data(), id.base_fingerprint.size());
}

struct HmtCounters {
    int64_t steps_done = 0, skipped = 0;
    int32_t opt_step = 0, opt_iter = 0;
};

template <typename T> static bool hmt_put(FILE * f, const T & v) { return fwrite(&v, sizeof(T), 1, f) == 1; }
template <typename T> static bool hmt_get(FILE * f, T * v) { return fread(v, sizeof(T), 1, f) == 1; }

static std::vector<ggml_tensor *> hmt_state_tensors(const HmtLora & L, const LmOptim & o) {
    std::vector<ggml_tensor *> t = L.params;
    for (auto * x : yue2_optim_state_tensors(o)) t.push_back(x);
    return t;
}

static bool hmt_state_save(const std::string & path, uint64_t ident, const HmtCounters & c, const HmtLora & L,
                           const LmOptim & o, std::string * err) {
    const std::string tmp = path + ".tmp";
    FILE * f = hs_fopen(tmp, "wb");
    if (!f) { *err = "cannot open " + tmp; return false; }
    bool ok = fwrite(HMT_STATE_MAGIC, 1, 8, f) == 8 && hmt_put(f, (uint32_t) HMT_STATE_VERSION) && hmt_put(f, ident) &&
              hmt_put(f, c.steps_done) && hmt_put(f, c.skipped) && hmt_put(f, c.opt_step) && hmt_put(f, c.opt_iter) &&
              yue2_optim_save(f, o);
    const auto tensors = hmt_state_tensors(L, o);
    ok = ok && hmt_put(f, (uint32_t) tensors.size());
    std::vector<uint8_t> scratch;
    for (auto * t : tensors) {
        if (!ok) break;
        const uint64_t n = ggml_nbytes(t);
        const uint32_t len = (uint32_t) strlen(t->name);
        scratch.resize((size_t) n);
        ggml_backend_tensor_get(t, scratch.data(), 0, (size_t) n);
        ok = hmt_put(f, len) && fwrite(t->name, 1, len, f) == len && hmt_put(f, n) && fwrite(scratch.data(), 1, (size_t) n, f) == n;
    }
    ok = ok && hmt_put(f, (uint32_t) 0xD09EF00Du);
    if (fclose(f) != 0) ok = false;
    if (!ok) { hs_remove(tmp); *err = "write failed (disk full?): " + tmp; return false; }
    // Replace in one step (MoveFileExW REPLACE_EXISTING on Windows, rename(2) elsewhere): the old
    // checkpoint stays whole until the new one is in place, and stays put if the replace fails.
    std::error_code ec;
    std::filesystem::rename(std::filesystem::u8path(tmp), std::filesystem::u8path(path), ec);
    if (ec) {
        hs_remove(tmp);
        *err = "cannot replace " + path + " (" + ec.message() + "); the previous checkpoint is unchanged";
        return false;
    }
    return true;
}

static bool hmt_state_load(const std::string & path, uint64_t ident, HmtCounters * c, HmtLora * L, LmOptim * o,
                           std::string * err) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) { *err = "cannot open " + path + " (no checkpoint to resume from)"; return false; }
    auto fail = [&](const std::string & why) { fclose(f); *err = path + ": " + why; return false; };
    char     magic[8];
    uint32_t ver = 0;
    uint64_t got = 0;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, HMT_STATE_MAGIC, 8) != 0) return fail("not a heartmula-train checkpoint");
    if (!hmt_get(f, &ver) || ver != HMT_STATE_VERSION) return fail("checkpoint version " + std::to_string(ver) + ", this build reads " + std::to_string(HMT_STATE_VERSION));
    if (!hmt_get(f, &got)) return fail("truncated header");
    if (got != ident) {
        return fail("was written by a different run (rank, alpha, lr, weight decay, betas, eps, clip, grad-accum, seed, "
                    "max-frames, weights, base model or the prepared data changed). Restore them, or start a new --out");
    }
    HmtCounters k;
    if (!hmt_get(f, &k.steps_done) || !hmt_get(f, &k.skipped) || !hmt_get(f, &k.opt_step) || !hmt_get(f, &k.opt_iter)) return fail("truncated header");
    std::string oerr;
    if (!yue2_optim_load(f, 3, o, &oerr)) return fail(oerr);
    std::unordered_map<std::string, ggml_tensor *> live;
    for (auto * t : hmt_state_tensors(*L, *o)) live[t->name] = t;
    uint32_t n = 0;
    if (!hmt_get(f, &n) || n != live.size()) return fail("tensor count differs from this run's");
    std::vector<uint8_t> scratch;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t len = 0;
        uint64_t bytes = 0;
        std::string name;
        if (!hmt_get(f, &len) || len > 256) return fail("truncated tensor record");
        name.resize(len);
        if (fread(&name[0], 1, len, f) != len || !hmt_get(f, &bytes)) return fail("truncated tensor record");
        auto it = live.find(name);
        if (it == live.end() || ggml_nbytes(it->second) != bytes) return fail("tensor '" + name + "' does not match this run");
        scratch.resize((size_t) bytes);
        if (fread(scratch.data(), 1, (size_t) bytes, f) != bytes) return fail("truncated data for '" + name + "'");
        ggml_backend_tensor_set(it->second, scratch.data(), 0, (size_t) bytes);
        live.erase(it);
    }
    uint32_t eof = 0;
    if (!hmt_get(f, &eof) || eof != 0xD09EF00Du) return fail("no end marker (truncated)");
    fclose(f);
    o->opt_step = k.opt_step;
    o->opt_iter = k.opt_iter;
    *c = k;
    return true;
}

// ── Adapter export ─────────────────────────────────────────────────────────

// `trigger` is the word the dataset's tags led with, recorded so generation can lead with it too.
static bool hmt_export(const std::string & path, const std::string & name, int64_t step, const HmtLora & L,
                       const HeartMuLaModel & m, std::string * err, const std::string & trigger = "") {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::exists(fs::u8path(path), ec)) return true;  // immutable: a step is written once
    ggml_init_params ip = { (L.params.size() + 8) * ggml_tensor_overhead(), nullptr, true };
    ggml_context *   ctx = ggml_init(ip);
    gguf_context *   g   = gguf_init_empty();
    std::vector<std::vector<float>> data;
    data.reserve(L.params.size());
    std::string targets = "[";
    for (size_t i = 0; i < L.layers.size(); i++) {
        for (int s = 0; s < HMT_SITES; s++) {
            if (L.layers[i].s[s].A) targets += std::string(targets.size() > 1 ? ", " : "") + "\"" + hmt_target((int) i, s) + "\"";
        }
    }
    targets += "]";
    const std::string source = "{\"producer\": \"ace-train " + std::string(ACE_VERSION) + " heartmula-train\", \"step\": " + std::to_string((long long) step) + "}";
    gguf_set_val_str(g, "general.architecture", "heartmula-lora");
    gguf_set_val_str(g, "general.name", name.c_str());
    gguf_set_val_u32(g, "general.file_type", 0);
    gguf_set_val_u32(g, "heartmula.format_version", HEARTMULA_FORMAT_VERSION);
    gguf_set_val_str(g, "heartmula.component", "lora");
    gguf_set_val_str(g, "heartmula.variant", m.variant.c_str());
    gguf_set_val_str(g, "heartmula.base_fingerprint", m.base_fingerprint.c_str());
    gguf_set_val_str(g, "heartmula.source", source.c_str());
    gguf_set_val_str(g, "heartmula.lora.target_component", "global");
    gguf_set_val_u32(g, "heartmula.lora.rank", (uint32_t) L.rank);
    gguf_set_val_f32(g, "heartmula.lora.alpha", L.alpha);
    gguf_set_val_str(g, "heartmula.lora.targets", targets.c_str());
    if (!trigger.empty()) gguf_set_val_str(g, "heartmula.lora.trigger", trigger.c_str());
    for (ggml_tensor * p : L.params) {
        data.emplace_back((size_t) ggml_nelements(p));
        ggml_backend_tensor_get(p, data.back().data(), 0, ggml_nbytes(p));
        for (float x : data.back()) {
            if (!std::isfinite(x)) {
                gguf_free(g);
                ggml_free(ctx);
                *err = std::string("non-finite value in ") + p->name + "; not exporting";
                return false;
            }
        }
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p->ne[0], p->ne[1]);
        ggml_set_name(t, p->name);
        t->data = data.back().data();
        gguf_add_tensor(g, t);
    }
    fs::create_directories(fs::u8path(path).parent_path(), ec);
    const std::string tmp = path + ".tmp";
    const bool        ok  = gguf_write_to_file(g, tmp.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    if (!ok || hs_rename(tmp, path) != 0) {
        hs_remove(tmp);
        *err = "cannot write " + path;
        return false;
    }
    return true;
}

// ── .npy (selftest fixtures) ───────────────────────────────────────────────

struct HmtNpy {
    std::vector<int64_t> shape;
    std::vector<float>   f;  // float data, or integers widened
};

static bool hmt_npy(const std::string & path, HmtNpy * out, std::string * err) {
    std::ifstream in(std::filesystem::u8path(path), std::ios::binary);
    char magic[8];
    if (!in || !in.read(magic, 8) || std::string(magic + 1, 5) != "NUMPY") { *err = "cannot read " + path; return false; }
    uint32_t hl = 0;
    if (magic[6] == 1) { uint16_t h = 0; in.read((char *) &h, 2); hl = h; } else { in.read((char *) &hl, 4); }
    std::string h(hl, '\0');
    in.read(&h[0], hl);
    const size_t d0 = h.find('\'', h.find("'descr':") + 8) + 1;
    const std::string descr = h.substr(d0, h.find('\'', d0) - d0);
    const size_t s0 = h.find('(', h.find("'shape':")) + 1;
    std::string dims = h.substr(s0, h.find(')', s0) - s0);
    out->shape.clear();
    int64_t n = 1;
    for (size_t p = 0; p < dims.size();) {
        const size_t c = dims.find(',', p);
        const std::string tok = dims.substr(p, c == std::string::npos ? std::string::npos : c - p);
        if (tok.find_first_of("0123456789") != std::string::npos) { out->shape.push_back(std::stoll(tok)); n *= out->shape.back(); }
        if (c == std::string::npos) break;
        p = c + 1;
    }
    out->f.resize((size_t) n);
    if (descr == "<f4") {
        in.read((char *) out->f.data(), n * 4);
    } else if (descr == "<i8") {
        std::vector<int64_t> v((size_t) n);
        in.read((char *) v.data(), n * 8);
        for (int64_t i = 0; i < n; i++) out->f[(size_t) i] = (float) v[(size_t) i];
    } else {
        *err = path + ": unsupported dtype " + descr;
        return false;
    }
    if (!in) { *err = path + ": truncated"; return false; }
    return true;
}
