// moss-lm-graph.h: MOSS-Music LM (Qwen3-8B) prefill, with audio splice + deepstack.
//
// SCOPE: a single PREFILL pass over T positions, batch 1, no KV cache. That is
// exactly what the audio->LM seam needs to be proven against the reference's
// `logits_last` fixture, and it is the riskiest remaining join in the port
// (deepstack uses two different layer indexings that both start at 0). The
// cached decode loop is a separate, later step -- model it on
// minimax/mm3-lm-graph.h, whose LM is architecturally IDENTICAL to this one.
//
// ── The two audio joins ──────────────────────────────────────────────────────
//
// 1. SPLICE. inputs_embeds = token_embd[ids], with the rows at <|AUDIO|>
//    (id 151654) REPLACED by the audio adapter output. Done here as
//    emb*text_mask + audio_vals rather than a scatter, because token_embd is
//    q8_0 and dequantising host-side to patch rows would be worse in every way.
//
// 2. DEEPSTACK. The three merger outputs are ADDED into the first three LM
//    layers -- at the audio rows only -- after each block's output. Note the two
//    indexings: mergers come from ENCODER layers 8/16/24, and land in LM layers
//    0/1/2. Both start at 0 and neither is the other.
//
// Both join tensors are built host-side (zeros at text rows) and uploaded, which
// keeps the graph free of scatter ops and makes the masking obvious on
// inspection. At 4096 x ~500 f32 that is ~8 MB per tensor -- irrelevant next to
// the 8.7 GB of weights.

#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "moss/moss-model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace moss {

namespace lm_detail {

static ggml_tensor * rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

}  // namespace lm_detail

// One KV cache: per layer a [D, n_ctx, Nkv] F16 tensor, all layers in one backend
// buffer. Zeroed once -- a padded attention window must read finite values, never
// uninitialised F16 bit patterns that decode to NaN.
struct LmKv {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;
    int64_t n_ctx  = 0;
    int64_t n_past = 0;      // positions already resident
};

inline void moss_lm_kv_free(LmKv * kv) {
    if (!kv) {
        return;
    }
    if (kv->buf) {
        ggml_backend_buffer_free(kv->buf);
    }
    if (kv->ctx) {
        ggml_free(kv->ctx);
    }
    *kv = {};
}

inline bool moss_lm_kv_init(LmKv * kv, const LmModel & m, int64_t n_ctx,
                            ggml_backend_t backend) {
    *kv = {};
    const LmHParams & hp = m.hp;
    const int64_t D = (int64_t) hp.head_dim, Nkv = (int64_t) hp.n_head_kv;

    ggml_init_params ip = { ggml_tensor_overhead() * (2 * hp.n_layer + 8), nullptr, true };
    kv->ctx = ggml_init(ip);
    if (!kv->ctx) {
        return false;
    }
    kv->k.resize(hp.n_layer);
    kv->v.resize(hp.n_layer);
    for (uint32_t i = 0; i < hp.n_layer; ++i) {
        kv->k[i] = ggml_new_tensor_3d(kv->ctx, GGML_TYPE_F16, D, n_ctx, Nkv);
        kv->v[i] = ggml_new_tensor_3d(kv->ctx, GGML_TYPE_F16, D, n_ctx, Nkv);
    }
    kv->buf = ggml_backend_alloc_ctx_tensors(kv->ctx, backend);
    if (!kv->buf) {
        moss_lm_kv_free(kv);
        return false;
    }
    ggml_backend_buffer_clear(kv->buf, 0);
    kv->n_ctx  = n_ctx;
    kv->n_past = 0;
    return true;
}

// Evaluates T tokens at absolute positions [kv.n_past, kv.n_past + T), writing
// their K/V into the cache and attending over everything resident. Returns the
// logits at the LAST position and advances kv.n_past.
//
//   ids          [T]       token ids, hp.audio_token_id at every audio slot
//   audio_vals   [H*T]     adapter output at audio rows, 0 elsewhere. EMPTY to skip
//                          the splice entirely (the decode path).
//   merger_vals  K x [H*T] deepstack merger output at audio rows, 0 elsewhere
//   text_mask    [T]       1.0 at text rows, 0.0 at audio rows. EMPTY = all text.
inline bool moss_lm_eval(const LmModel & m, LmKv & kv, const std::vector<int32_t> & ids,
                         const std::vector<float> & audio_vals,
                         const std::vector<std::vector<float>> & merger_vals,
                         const std::vector<float> & text_mask,
                         ggml_backend_t backend, std::vector<float> * out_logits) {
    using namespace lm_detail;
    const LmHParams & hp = m.hp;
    const int64_t T   = (int64_t) ids.size();
    const int64_t H   = (int64_t) hp.n_embd;
    const int64_t D   = (int64_t) hp.head_dim;
    const int64_t Nh  = (int64_t) hp.n_head;
    const int64_t Nkv = (int64_t) hp.n_head_kv;
    const int64_t n_inject = (int64_t) merger_vals.size();
    const int64_t n_past = kv.n_past;
    const int64_t n_kv = n_past + T;
    const bool has_audio = !audio_vals.empty();

    if (n_kv > kv.n_ctx) {
        fprintf(stderr, "[MOSS] eval: n_kv %lld exceeds cache n_ctx %lld\n",
                (long long) n_kv, (long long) kv.n_ctx);
        return false;
    }
    if (has_audio && ((int64_t) audio_vals.size() != H * T ||
                      (int64_t) text_mask.size() != T)) {
        fprintf(stderr, "[MOSS] eval: join tensors have the wrong size\n");
        return false;
    }

    // ~60 nodes per block once reshapes/permutes/conts are counted, plus the
    // splice, the injections and the head. Overshoot: see the encoder graph note.
    const size_t n_nodes = (size_t) (512 + 128 * hp.n_layer);
    const size_t ctx_size = ggml_tensor_overhead() * (n_nodes + 64) +
                            ggml_graph_overhead_custom(n_nodes, false);
    ggml_init_params ip = { ctx_size, nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_nodes, false);

    ggml_tensor * in_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(in_ids);
    ggml_tensor * in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(in_pos);
    ggml_tensor * in_mask = nullptr;
    ggml_tensor * in_audio = nullptr;
    std::vector<ggml_tensor *> in_merge((size_t) n_inject, nullptr);
    if (has_audio) {
        in_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, T);
        ggml_set_input(in_mask);
        in_audio = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
        ggml_set_input(in_audio);
        for (int64_t k = 0; k < n_inject; ++k) {
            in_merge[(size_t) k] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
            ggml_set_input(in_merge[(size_t) k]);
        }
    }
    // Causal mask over the whole resident window: [n_kv, T].
    ggml_tensor * in_kq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, T);
    ggml_set_input(in_kq);
    // KV destination rows, I64 as ggml_set_rows requires.
    ggml_tensor * in_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
    ggml_set_input(in_rows);

    // ---- splice ----------------------------------------------------------
    ggml_tensor * h = ggml_get_rows(ctx, m.token_embd, in_ids);   // [H, T]
    if (has_audio) {
        h = ggml_add(ctx, ggml_mul(ctx, h, in_mask), in_audio);
    }

    const float scale = 1.0f / sqrtf((float) D);
    for (uint32_t il = 0; il < hp.n_layer; ++il) {
        const LmLayer & w = m.layers[il];

        ggml_tensor * n = rms(ctx, h, w.attn_norm, hp.rms_eps);
        ggml_tensor * q = ggml_mul_mat(ctx, w.attn_q, n);
        ggml_tensor * k = ggml_mul_mat(ctx, w.attn_k, n);
        ggml_tensor * v = ggml_mul_mat(ctx, w.attn_v, n);

        q = ggml_reshape_3d(ctx, q, D, Nh, T);
        k = ggml_reshape_3d(ctx, k, D, Nkv, T);
        v = ggml_reshape_3d(ctx, v, D, Nkv, T);

        // Qwen3 per-head RMSNorm over the head dim, BEFORE RoPE.
        q = ggml_mul(ctx, ggml_rms_norm(ctx, q, hp.rms_eps), w.attn_q_norm);
        k = ggml_mul(ctx, ggml_rms_norm(ctx, k, hp.rms_eps), w.attn_k_norm);

        q = ggml_rope_ext(ctx, q, in_pos, NULL, (int) D, GGML_ROPE_TYPE_NEOX, 0,
                          hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, in_pos, NULL, (int) D, GGML_ROPE_TYPE_NEOX, 0,
                          hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // [D, head, pos] -> [D, pos, head] for the cache layout and the contraction.
        ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [D, T, Nh]
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [D, T, Nkv]
        ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));  // [D, T, Nkv]

        // Cache writes are expanded into the graph EAGERLY so they are ordered
        // before this layer's read -- ggml executes nodes in list order, and a
        // cache view carries no data dependency on the write that filled it.
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, kv.k[il], kp, in_rows));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, kv.v[il], vp, in_rows));

        ggml_tensor * kw = ggml_view_3d(ctx, kv.k[il], D, n_kv, Nkv,
                                        kv.k[il]->nb[1], kv.k[il]->nb[2], 0);
        ggml_tensor * vw = ggml_view_3d(ctx, kv.v[il], D, n_kv, Nkv,
                                        kv.v[il]->nb[1], kv.v[il]->nb[2], 0);

        // GQA falls out of ggml's mul_mat broadcast: query head h reads KV head
        // h / (Nh/Nkv). No repeat needed.
        ggml_tensor * kq = ggml_mul_mat(ctx, kw, qp);                 // [n_kv, T, Nh]
        kq = ggml_soft_max_ext(ctx, kq, in_kq, scale, 0.0f);
        ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, vw));   // [n_kv, D, Nkv]
        ggml_tensor * o = ggml_mul_mat(ctx, vt, kq);                  // [D, T, Nh]
        o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));         // [D, Nh, T]
        o = ggml_reshape_2d(ctx, o, H, T);
        h = ggml_add(ctx, h, ggml_mul_mat(ctx, w.attn_output, o));

        ggml_tensor * n2 = rms(ctx, h, w.ffn_norm, hp.rms_eps);
        ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, w.ffn_gate, n2));
        ggml_tensor * up = ggml_mul_mat(ctx, w.ffn_up, n2);
        h = ggml_add(ctx, h, ggml_mul_mat(ctx, w.ffn_down, ggml_mul(ctx, gate, up)));

        // Deepstack: merger k lands in LM layer k. The values are already zero
        // at text rows, so a plain add is the masking. Prefill only -- on the
        // decode path there are no audio rows in the window being evaluated.
        if (has_audio && (int64_t) il < n_inject) {
            h = ggml_add(ctx, h, in_merge[(size_t) il]);
        }
    }

    h = rms(ctx, h, m.output_norm, hp.rms_eps);
    // Only the final position matters for the fixture check.
    ggml_tensor * last = ggml_view_2d(ctx, h, H, 1, h->nb[1], (size_t) (T - 1) * h->nb[1]);
    ggml_tensor * logits = ggml_mul_mat(ctx, m.output, ggml_cont(ctx, last));  // [n_vocab, 1]
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "[MOSS] LM graph allocation failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(in_ids, ids.data(), 0, ids.size() * sizeof(int32_t));
    {
        std::vector<int32_t> pos((size_t) T);
        std::vector<int64_t> rows((size_t) T);
        for (int64_t i = 0; i < T; ++i) {
            pos[(size_t) i]  = (int32_t) (n_past + i);
            rows[(size_t) i] = n_past + i;
        }
        ggml_backend_tensor_set(in_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
        ggml_backend_tensor_set(in_rows, rows.data(), 0, rows.size() * sizeof(int64_t));
    }
    if (has_audio) {
        ggml_backend_tensor_set(in_mask, text_mask.data(), 0, text_mask.size() * sizeof(float));
        ggml_backend_tensor_set(in_audio, audio_vals.data(), 0, audio_vals.size() * sizeof(float));
        for (int64_t k = 0; k < n_inject; ++k) {
            ggml_backend_tensor_set(in_merge[(size_t) k], merger_vals[(size_t) k].data(), 0,
                                    merger_vals[(size_t) k].size() * sizeof(float));
        }
    }
    {
        // Query i sits at absolute position n_past + i; key j is visible iff
        // j <= n_past + i.
        std::vector<float> kq((size_t) (n_kv * T), 0.0f);
        for (int64_t i = 0; i < T; ++i) {
            for (int64_t j = 0; j < n_kv; ++j) {
                kq[(size_t) (i * n_kv + j)] = (j <= n_past + i) ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(in_kq, kq.data(), 0, kq.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[MOSS] LM graph compute failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    out_logits->resize((size_t) hp.n_vocab);
    ggml_backend_tensor_get(logits, out_logits->data(), 0, out_logits->size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    kv.n_past = n_kv;
    return true;
}

// Convenience wrapper: one-shot prefill into a throwaway cache. Kept because the
// parity test drives this signature, and it doubles as the check that the CACHED
// path reproduces the no-cache result it was validated at.
inline bool moss_lm_prefill_logits(const LmModel & m, const std::vector<int32_t> & ids,
                                   const std::vector<float> & audio_vals,
                                   const std::vector<std::vector<float>> & merger_vals,
                                   const std::vector<float> & text_mask,
                                   ggml_backend_t backend, std::vector<float> * out_logits) {
    LmKv kv;
    if (!moss_lm_kv_init(&kv, m, (int64_t) ids.size(), backend)) {
        fprintf(stderr, "[MOSS] could not allocate a KV cache for prefill\n");
        return false;
    }
    const bool ok = moss_lm_eval(m, kv, ids, audio_vals, merger_vals, text_mask,
                                 backend, out_logits);
    moss_lm_kv_free(&kv);
    return ok;
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

struct SamplerParams {
    // Defaults are the configuration measured on the reference: greedy, with a
    // LIGHT repetition penalty and a real frequency penalty.
    //
    // These are not interchangeable knobs. repetition_penalty fires once a token
    // has appeared AT ALL, so any value strong enough to break a decode loop also
    // crushes legitimate repetition -- a repeated chorus, a riff described twice.
    // frequency_penalty scales with HOW OFTEN a token has appeared, so a hook sung
    // three times is barely touched while a runaway loop is. Measured on lyric
    // transcription: rep 1.15 alone scored 0.53 grounded and produced a 954-token
    // loop; rep 1.05 + freq 0.3 scored 0.75. Shipping without freq looks like
    // model incompetence rather than a config gap.
    float temperature        = 0.0f;   // 0 = greedy
    float repetition_penalty = 1.05f;
    float frequency_penalty  = 0.30f;
    float top_p              = 1.0f;   // nucleus, only when temperature > 0
    uint32_t seed            = 0;      // the caller reseeds `rng` per generation
    std::mt19937 rng{ 0 };
};

// Applies the penalties in place and returns the chosen token. `counts` maps
// token id -> how many times it has already been emitted. Greedy unless
// temperature > 0, then nucleus sampling (top_p) from the seeded rng.
inline int32_t moss_sample(std::vector<float> & logits, const std::vector<int32_t> & counts,
                           SamplerParams & sp) {
    const size_t n = logits.size();
    for (size_t i = 0; i < n && i < counts.size(); ++i) {
        const int32_t c = counts[i];
        if (c <= 0) {
            continue;
        }
        if (sp.repetition_penalty != 1.0f) {
            // Standard asymmetric form: divide positive logits, multiply negative.
            logits[i] = (logits[i] > 0.0f) ? logits[i] / sp.repetition_penalty
                                           : logits[i] * sp.repetition_penalty;
        }
        logits[i] -= sp.frequency_penalty * (float) c;
    }
    if (sp.temperature > 0.0f) {
        // Top 256 candidates hold all the nucleus mass in practice; sorting the
        // whole 150k vocabulary every token would cost more than the decode.
        const size_t k = std::min<size_t>(256, n);
        std::vector<int32_t> idx(n);
        for (size_t i = 0; i < n; ++i) idx[i] = (int32_t) i;
        std::partial_sort(idx.begin(), idx.begin() + (long) k, idx.end(),
                          [&](int32_t a, int32_t b) { return logits[(size_t) a] > logits[(size_t) b]; });
        std::vector<double> p(k);
        const double mx = logits[(size_t) idx[0]];
        double sum = 0.0;
        for (size_t i = 0; i < k; ++i) {
            p[i] = std::exp(((double) logits[(size_t) idx[i]] - mx) / sp.temperature);
            sum += p[i];
        }
        size_t keep = k;
        double acc = 0.0;
        for (size_t i = 0; i < k; ++i) {
            acc += p[i] / sum;
            if (acc >= sp.top_p) { keep = i + 1; break; }
        }
        std::discrete_distribution<size_t> pick(p.begin(), p.begin() + (long) keep);
        return idx[pick(sp.rng)];
    }
    size_t best = 0;
    for (size_t i = 1; i < n; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return (int32_t) best;
}

}  // namespace moss
