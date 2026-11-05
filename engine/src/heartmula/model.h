#pragma once
// heartmula/model.h: HeartMuLa global (3B) and local (300M) LM weights.
//
// HOT-Step file. Loads the two LM GGUFs written by engine/tools/convert-heartmula.py
// (engine/docs/HEARTMULA_CONTRACT.md, "GGUF metadata"). Both are torchtune
// llama3_2 decoders with Identity embeddings/outputs (heartlib modeling_heartmula.py):
//
//   global  28 layers, 24 q heads / 8 kv heads x 128, k/v width 1024, ctx 8192
//   local    3 layers,  8 q heads / 4 kv heads x 384, k/v width 1536, ctx 8
//
// Both use SwiGLU 8192, RMSNorm eps 1e-5, rope base 5e5 with Llama-3 scaling
// (factor 32, low 1, high 4, original context 8192), and torchtune's interleaved
// RoPE (pairs 2i, 2i+1), which is ggml's NORMAL mode. The global file also holds
// text_embeddings, audio_embeddings (8 x 8197 rows, codebook k at k * 8197),
// codebook0_head, muq_linear and unconditional_text_embedding; the local file holds
// projection and audio_head.0..6 (already split and transposed to [8197, 3072]).
// The local decode reads the global's audio_embeddings, so both load together.

#include <string>  // backend.h uses std::wstring without including it

#include "backend.h"
#include "gguf-weights.h"
#include "weight-ctx.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#define HEARTMULA_FORMAT_VERSION 1
#define HEARTMULA_CODEBOOKS      8
#define HEARTMULA_AUDIO_VOCAB    8197
#define HEARTMULA_COLUMNS        9  // 8 audio codebooks + text, heartlib's _parallel_number

struct HeartMuLaLmConfig {
    int   n_layer   = 0;
    int   n_embd    = 3072;
    int   n_head    = 0;
    int   n_head_kv = 0;
    int   head_dim  = 0;
    int   n_ff      = 8192;
    int   n_ctx     = 0;
    float rms_eps   = 1e-5f;
    float rope_base = 500000.0f;
};

static HeartMuLaLmConfig heartmula_global_config() {
    HeartMuLaLmConfig c;
    c.n_layer   = 28;
    c.n_head    = 24;
    c.n_head_kv = 8;
    c.head_dim  = 128;
    c.n_ctx     = 8192;
    return c;
}

static HeartMuLaLmConfig heartmula_local_config() {
    HeartMuLaLmConfig c;
    c.n_layer   = 3;
    c.n_head    = 8;
    c.n_head_kv = 4;
    c.head_dim  = 384;
    c.n_ctx     = HEARTMULA_CODEBOOKS;
    return c;
}

// torchtune Llama3ScaledRoPE.apply_scaling as ggml freq factors: ggml divides
// each pair's base frequency by ff[i], so ff = freq / scaled_freq.
static std::vector<float> heartmula_rope_freq_factors(int head_dim, float base) {
    const double scale = 32.0, low = 1.0, high = 4.0, old_ctx = 8192.0;
    const double low_wavelen = old_ctx / low, high_wavelen = old_ctx / high;
    std::vector<float> ff((size_t) head_dim / 2);
    for (int i = 0; i < head_dim / 2; i++) {
        const float  freq    = 1.0f / powf(base, (float) (2 * i) / (float) head_dim);
        const double wavelen = 2.0 * 3.14159265358979323846 / freq;
        double       scaled  = freq;
        if (wavelen > low_wavelen) {
            scaled = freq / scale;
        } else if (wavelen >= high_wavelen) {
            const double smooth = (old_ctx / wavelen - low) / (high - low);
            scaled              = (1.0 - smooth) * freq / scale + smooth * freq;
        }
        ff[(size_t) i] = (float) ((double) freq / (double) (float) scaled);
    }
    return ff;
}

struct HeartMuLaLayer {
    ggml_tensor * sa_norm  = nullptr;
    ggml_tensor * q        = nullptr;
    ggml_tensor * k        = nullptr;
    ggml_tensor * v        = nullptr;
    ggml_tensor * o        = nullptr;
    ggml_tensor * mlp_norm = nullptr;
    ggml_tensor * w1       = nullptr;  // gate
    ggml_tensor * w2       = nullptr;  // down
    ggml_tensor * w3       = nullptr;  // up
};

struct HeartMuLaStack {
    HeartMuLaLmConfig           cfg;
    std::vector<HeartMuLaLayer> layers;
    ggml_tensor *               norm         = nullptr;
    ggml_tensor *               freq_factors = nullptr;  // F32 [head_dim / 2]
};

struct HeartMuLaModel {
    BackendPair bp          = {};
    bool        backend_ref = false;

    HeartMuLaStack global;
    HeartMuLaStack local;

    ggml_tensor * text_embd    = nullptr;  // [3072, 128256]
    ggml_tensor * audio_embd   = nullptr;  // [3072, 8 * 8197]
    ggml_tensor * uncond_embd  = nullptr;  // [3072, 1]
    ggml_tensor * muq_bias     = nullptr;  // [3072]; zero MuQ input projects to this
    ggml_tensor * c0_head      = nullptr;  // [3072, 8197]
    ggml_tensor * projection   = nullptr;  // [3072, 3072]
    ggml_tensor * audio_head[HEARTMULA_CODEBOOKS - 1] = {};

    WeightCtx   wctx_global = {};
    WeightCtx   wctx_local  = {};
    std::string variant;
    std::string base_fingerprint;
    bool        loaded = false;
};

static void heartmula_model_free(HeartMuLaModel * m) {
    wctx_free(&m->wctx_global);
    wctx_free(&m->wctx_local);
    if (m->backend_ref) {
        backend_release(m->bp.backend, m->bp.cpu_backend);
    }
    *m = HeartMuLaModel{};
}

namespace heartmula_model_detail {

inline bool fail(std::string * err, const std::string & msg) {
    if (err) {
        *err = msg;
    }
    return false;
}

// Header identity checks for one LM file; the full catalogue checks are S6's.
inline bool check_header(const GGUFModel & gf, const char * path, const char * arch, std::string * err) {
    const std::string got = gf_get_str(gf, "general.architecture");
    if (got != arch) {
        return fail(err, std::string(path) + ": architecture is '" + got + "', expected '" + arch + "'");
    }
    if (gf_get_u32(gf, "heartmula.format_version") != HEARTMULA_FORMAT_VERSION) {
        return fail(err, std::string(path) + ": unsupported heartmula.format_version");
    }
    if (!*gf_get_str(gf, "heartmula.base_fingerprint")) {
        return fail(err, std::string(path) + ": heartmula.base_fingerprint is missing");
    }
    return true;
}

inline bool has_all(const GGUFModel & gf, const std::vector<std::string> & names, const char * path,
                    std::string * err) {
    for (const auto & n : names) {
        if (gguf_find_tensor(gf.gguf, n.c_str()) < 0) {
            return fail(err, std::string(path) + ": tensor '" + n + "' is missing");
        }
    }
    return true;
}

inline std::vector<std::string> stack_names(const std::string & prefix, int n_layer) {
    std::vector<std::string> out;
    for (int i = 0; i < n_layer; i++) {
        const std::string p = prefix + ".layers." + std::to_string(i) + ".";
        for (const char * s : { "sa_norm.scale", "attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight",
                                "attn.output_proj.weight", "mlp_norm.scale", "mlp.w1.weight", "mlp.w2.weight",
                                "mlp.w3.weight" }) {
            out.push_back(p + s);
        }
    }
    out.push_back(prefix + ".norm.scale");
    return out;
}

inline void load_stack(WeightCtx * w, const GGUFModel & gf, const std::string & prefix, HeartMuLaStack * s) {
    s->layers.resize((size_t) s->cfg.n_layer);
    for (int i = 0; i < s->cfg.n_layer; i++) {
        const std::string p = prefix + ".layers." + std::to_string(i) + ".";
        HeartMuLaLayer &  l = s->layers[(size_t) i];
        l.sa_norm           = gf_load_tensor_f32(w, gf, p + "sa_norm.scale");
        l.q                 = gf_load_tensor(w, gf, p + "attn.q_proj.weight");
        l.k                 = gf_load_tensor(w, gf, p + "attn.k_proj.weight");
        l.v                 = gf_load_tensor(w, gf, p + "attn.v_proj.weight");
        l.o                 = gf_load_tensor(w, gf, p + "attn.output_proj.weight");
        l.mlp_norm          = gf_load_tensor_f32(w, gf, p + "mlp_norm.scale");
        l.w1                = gf_load_tensor(w, gf, p + "mlp.w1.weight");
        l.w2                = gf_load_tensor(w, gf, p + "mlp.w2.weight");
        l.w3                = gf_load_tensor(w, gf, p + "mlp.w3.weight");
    }
    s->norm = gf_load_tensor_f32(w, gf, prefix + ".norm.scale");

    // Computed, not stored: staged like a converted weight so it rides the same buffer.
    const std::vector<float> ff = heartmula_rope_freq_factors(s->cfg.head_dim, s->cfg.rope_base);
    s->freq_factors              = ggml_new_tensor_1d(w->ctx, GGML_TYPE_F32, (int64_t) ff.size());
    ggml_set_name(s->freq_factors, (prefix + ".rope_freq_factors").c_str());
    auto buf = std::make_unique<float[]>(ff.size());
    std::copy(ff.begin(), ff.end(), buf.get());
    w->pending.push_back({ s->freq_factors, buf.get(), ff.size() * sizeof(float), 0 });
    w->staging.push_back(std::move(buf));
}

inline bool check_shapes(const HeartMuLaStack & s, const char * path, std::string * err) {
    const int64_t H = s.cfg.n_embd, q = (int64_t) s.cfg.n_head * s.cfg.head_dim,
                  kv = (int64_t) s.cfg.n_head_kv * s.cfg.head_dim;
    for (const auto & l : s.layers) {
        if (l.q->ne[0] != H || l.q->ne[1] != q || l.k->ne[1] != kv || l.v->ne[1] != kv || l.o->ne[0] != q ||
            l.w1->ne[1] != s.cfg.n_ff || l.w2->ne[0] != s.cfg.n_ff) {
            return fail(err, std::string(path) + ": layer shapes do not match the expected HeartMuLa config");
        }
    }
    return true;
}

}  // namespace heartmula_model_detail

// Load global + local. Both must share the variant and base fingerprint.
static bool heartmula_model_load(HeartMuLaModel * m, const char * global_path, const char * local_path,
                                 std::string * err) {
    using namespace heartmula_model_detail;
    heartmula_model_free(m);
    GGUFModel gg = {}, gl = {};
    if (!gf_load(&gg, global_path)) {
        return fail(err, std::string("cannot open ") + global_path);
    }
    if (!gf_load(&gl, local_path)) {
        gf_close(&gg);
        return fail(err, std::string("cannot open ") + local_path);
    }
    auto close = [&]() {
        gf_close(&gg);
        gf_close(&gl);
    };
    if (!check_header(gg, global_path, "heartmula-global", err) ||
        !check_header(gl, local_path, "heartmula-local", err)) {
        close();
        return false;
    }
    m->variant          = gf_get_str(gg, "heartmula.variant");
    m->base_fingerprint = gf_get_str(gg, "heartmula.base_fingerprint");
    if (m->variant != gf_get_str(gl, "heartmula.variant") ||
        m->base_fingerprint != gf_get_str(gl, "heartmula.base_fingerprint")) {
        close();
        return fail(err, std::string(local_path) + ": base_fingerprint differs from " + global_path);
    }

    m->global.cfg = heartmula_global_config();
    m->local.cfg  = heartmula_local_config();

    std::vector<std::string> gnames = stack_names("backbone", m->global.cfg.n_layer);
    for (const char * n : { "text_embeddings.weight", "audio_embeddings.weight", "unconditional_text_embedding.weight",
                            "muq_linear.bias", "codebook0_head.weight" }) {
        gnames.push_back(n);
    }
    std::vector<std::string> lnames = stack_names("decoder", m->local.cfg.n_layer);
    lnames.push_back("projection.weight");
    for (int i = 0; i < HEARTMULA_CODEBOOKS - 1; i++) {
        lnames.push_back("audio_head." + std::to_string(i) + ".weight");
    }
    if (!has_all(gg, gnames, global_path, err) || !has_all(gl, lnames, local_path, err)) {
        close();
        return false;
    }

    m->bp          = backend_init("HeartMuLa");
    m->backend_ref = true;

    wctx_init(&m->wctx_global, (int) gnames.size() + 1);
    load_stack(&m->wctx_global, gg, "backbone", &m->global);
    m->text_embd   = gf_load_tensor(&m->wctx_global, gg, "text_embeddings.weight");
    m->audio_embd  = gf_load_tensor(&m->wctx_global, gg, "audio_embeddings.weight");
    m->uncond_embd = gf_load_tensor_f32(&m->wctx_global, gg, "unconditional_text_embedding.weight");
    m->muq_bias    = gf_load_tensor_f32(&m->wctx_global, gg, "muq_linear.bias");
    m->c0_head     = gf_load_tensor(&m->wctx_global, gg, "codebook0_head.weight");

    wctx_init(&m->wctx_local, (int) lnames.size() + 1);
    load_stack(&m->wctx_local, gl, "decoder", &m->local);
    m->projection = gf_load_tensor(&m->wctx_local, gl, "projection.weight");
    for (int i = 0; i < HEARTMULA_CODEBOOKS - 1; i++) {
        m->audio_head[i] = gf_load_tensor(&m->wctx_local, gl, "audio_head." + std::to_string(i) + ".weight");
    }

    const bool shapes = check_shapes(m->global, global_path, err) && check_shapes(m->local, local_path, err) &&
                        m->audio_embd->ne[1] == (int64_t) HEARTMULA_CODEBOOKS * HEARTMULA_AUDIO_VOCAB &&
                        m->c0_head->ne[1] == HEARTMULA_AUDIO_VOCAB &&
                        m->audio_head[0]->ne[1] == HEARTMULA_AUDIO_VOCAB;
    if (!shapes) {
        close();
        heartmula_model_free(m);
        return err && !err->empty() ? false : fail(err, "HeartMuLa embedding/head shapes are unexpected");
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
