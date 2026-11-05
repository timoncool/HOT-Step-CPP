#pragma once
// heartmula/local-graph.h: the HeartMuLa local decoder (codebooks 1..7 of a frame).
//
// HOT-Step file. Port of the local half of heartlib HeartMuLa.generate_frame:
//
//   decoder.reset_caches()
//   h = [last_h, audio_embd(c0)]                 positions 0, 1
//   for i in 1..7:
//       d    = decoder(projection(h), pos)       3 layers, 8 q / 4 kv heads x 384
//       c_i  = sample(d[-1] @ audio_head[i-1])   CFG blended like codebook 0
//       h    = audio_embd(c_i + i * 8197)        position i + 1
//
// Seven cached graphs, one per codebook: graph 1 takes two positions, graphs 2..7
// one, and each reads a different head. Every input is data, so a frame is seven
// computes and no rebuilds. The eight-position F32 KV cache needs no clearing at
// a frame boundary: graph i only sees columns <= its position, and every one of
// those was rewritten earlier in the same frame. That is heartlib's per-frame
// reset_caches. Head dim 384 has no flash kernel, so attention is manual.

#include "heartmula/lm-graph.h"
#include "heartmula/sample.h"

#include <functional>

struct HeartMuLaLocalGraph {
    ggml_context *             kv_ctx = nullptr;
    ggml_backend_buffer_t      kv_buf = nullptr;
    std::vector<ggml_tensor *> kv_k, kv_v;
    int                        B = 0;

    HeartMuLaSlot slot[HEARTMULA_CODEBOOKS];  // slot[i] predicts codebook i; slot[0] unused
    ggml_tensor * in_last[HEARTMULA_CODEBOOKS] = {};  // F32 [H, 1, B], slot 1 only
    ggml_tensor * in_code[HEARTMULA_CODEBOOKS] = {};  // I32 [B] flat audio_embd rows
};

static void heartmula_local_free(HeartMuLaLocalGraph * g) {
    for (auto & s : g->slot) {
        heartmula_slot_free(&s);
    }
    if (g->kv_buf) {
        ggml_backend_buffer_free(g->kv_buf);
    }
    if (g->kv_ctx) {
        ggml_free(g->kv_ctx);
    }
    *g = HeartMuLaLocalGraph{};
}

static bool heartmula_local_build(const HeartMuLaModel & m, HeartMuLaLocalGraph * g, int i, std::string * err) {
    const HeartMuLaStack & st  = m.local;
    const int64_t          H   = st.cfg.n_embd;
    const int64_t          B   = g->B;
    const int64_t          T   = i == 1 ? 2 : 1;
    const int64_t          K   = HEARTMULA_CODEBOOKS;
    HeartMuLaSlot *        s   = &g->slot[i];
    ggml_cgraph *          gf  = nullptr;
    ggml_context *         ctx = heartmula_slot_begin(s, m.bp, &gf);

    g->in_code[i] = heartmula_input(ctx, GGML_TYPE_I32, B, 1, 1, "hm_local_code");
    s->in_pos     = heartmula_input(ctx, GGML_TYPE_I32, T, 1, 1, "hm_local_pos");
    s->in_rows    = heartmula_input(ctx, GGML_TYPE_I64, T, 1, 1, "hm_local_rows");
    s->in_mask    = heartmula_input(ctx, GGML_TYPE_F16, K, T, 1, "hm_local_mask");

    ggml_tensor * h = ggml_reshape_3d(ctx, ggml_get_rows(ctx, m.audio_embd, g->in_code[i]), H, 1, B);
    if (i == 1) {
        g->in_last[i] = heartmula_input(ctx, GGML_TYPE_F32, H, 1, B, "hm_local_last");
        h             = ggml_concat(ctx, g->in_last[i], h, 1);  // [H, 2, B]
    }
    h = ggml_mul_mat(ctx, m.projection, h);
    for (int l = 0; l < st.cfg.n_layer; l++) {
        h = heartmula_block(ctx, gf, st, st.layers[(size_t) l], h, s->in_pos, s->in_mask, s->in_rows, g->kv_k[l],
                            g->kv_v[l], K, false);
    }
    ggml_tensor * last = ggml_view_3d(ctx, h, H, 1, B, h->nb[1], h->nb[2], (size_t) (T - 1) * h->nb[1]);
    s->out_hidden      = heartmula_rms(ctx, ggml_cont(ctx, last), st.norm, st.cfg.rms_eps);
    s->out_logits      = ggml_mul_mat(ctx, m.audio_head[i - 1], s->out_hidden);
    ggml_set_output(s->out_logits);
    ggml_build_forward_expand(gf, s->out_logits);
    s->T    = T;
    s->n_kv = K;
    return heartmula_slot_alloc(s, gf, err, "local");
}

static bool heartmula_local_prepare(const HeartMuLaModel & m, HeartMuLaLocalGraph * g, int cfg_rows,
                                    std::string * err) {
    if (g->B == cfg_rows && g->kv_buf) {
        return true;
    }
    heartmula_local_free(g);
    const HeartMuLaLmConfig & c = m.local.cfg;
    g->B                        = cfg_rows;
    ggml_init_params ip         = { (size_t) (2 * c.n_layer) * ggml_tensor_overhead() + 1024, nullptr, true };
    g->kv_ctx                   = ggml_init(ip);
    for (int l = 0; l < c.n_layer; l++) {
        g->kv_k.push_back(ggml_new_tensor_4d(g->kv_ctx, GGML_TYPE_F32, c.head_dim, c.n_ctx, c.n_head_kv, g->B));
        g->kv_v.push_back(ggml_new_tensor_4d(g->kv_ctx, GGML_TYPE_F32, c.head_dim, c.n_ctx, c.n_head_kv, g->B));
    }
    g->kv_buf = ggml_backend_alloc_ctx_tensors(g->kv_ctx, m.bp.backend);
    if (!g->kv_buf) {
        heartmula_local_free(g);
        if (err) {
            *err = "HeartMuLa local KV allocation failed";
        }
        return false;
    }
    ggml_backend_buffer_clear(g->kv_buf, 0);
    for (int i = 1; i < HEARTMULA_CODEBOOKS; i++) {
        if (!heartmula_local_build(m, g, i, err)) {
            heartmula_local_free(g);
            return false;
        }
    }
    return true;
}

// Picks codebook k's code from its raw logits [B][8197] (row 0 conditioned).
using HeartMuLaChooser = std::function<int32_t(int k, const float * logits, int B)>;

// heartlib's CFG + sample_topk as a chooser.
static HeartMuLaChooser heartmula_sampling_chooser(float cfg_scale, int topk, float temperature,
                                                   HeartMuLaDraws * draws) {
    return [=](int, const float * logits, int B) {
        const int64_t V = HEARTMULA_AUDIO_VOCAB;
        if (B == 2) {
            std::vector<float> guided((size_t) V);
            heartmula_cfg_blend(logits, logits + V, V, cfg_scale, guided.data());
            return heartmula_sample_topk(guided.data(), V, topk, temperature, *draws);
        }
        return heartmula_sample_topk(logits, V, topk, temperature, *draws);
    };
}

// One frame: codebook 0 from the global's logits, then the local decoder.
// `last_h` is the global's normed last hidden [B][H]. Optional `raw` receives
// every codebook's logits [8][B][8197].
static bool heartmula_local_frame(const HeartMuLaModel & m, HeartMuLaLocalGraph * g, const float * last_h,
                                  const float * c0_logits, const HeartMuLaChooser & choose,
                                  int32_t codes[HEARTMULA_CODEBOOKS], float * raw, std::string * err) {
    const int     B = g->B;
    const int64_t V = HEARTMULA_AUDIO_VOCAB;
    if (raw) {
        std::copy(c0_logits, c0_logits + B * V, raw);
    }
    codes[0] = choose(0, c0_logits, B);
    std::vector<float> logits((size_t) (B * V));
    for (int i = 1; i < HEARTMULA_CODEBOOKS; i++) {
        HeartMuLaSlot & s  = g->slot[i];
        const int64_t   T  = s.T;
        const int64_t   p0 = i == 1 ? 0 : i;
        int32_t         pos[2];
        int64_t         rows[2];
        uint16_t        mask[2 * HEARTMULA_CODEBOOKS];
        for (int64_t t = 0; t < T; t++) {
            pos[t]  = (int32_t) (p0 + t);
            rows[t] = p0 + t;
            for (int j = 0; j < HEARTMULA_CODEBOOKS; j++) {
                mask[t * HEARTMULA_CODEBOOKS + j] = ggml_fp32_to_fp16(j <= p0 + t ? 0.0f : -INFINITY);
            }
        }
        // Both CFG rows share the sampled code (heartlib repeats it).
        const int32_t code = codes[i - 1] + (i - 1) * HEARTMULA_AUDIO_VOCAB;
        int32_t       ids[2] = { code, code };
        ggml_backend_tensor_set(g->in_code[i], ids, 0, (size_t) B * sizeof(int32_t));
        ggml_backend_tensor_set(s.in_pos, pos, 0, (size_t) T * sizeof(int32_t));
        ggml_backend_tensor_set(s.in_rows, rows, 0, (size_t) T * sizeof(int64_t));
        ggml_backend_tensor_set(s.in_mask, mask, 0, (size_t) T * HEARTMULA_CODEBOOKS * sizeof(uint16_t));
        if (i == 1) {
            ggml_backend_tensor_set(g->in_last[i], last_h, 0, ggml_nbytes(g->in_last[i]));
        }
        if (ggml_backend_sched_graph_compute(s.sched, s.graph) != GGML_STATUS_SUCCESS) {
            if (err) {
                *err = "HeartMuLa local graph compute failed";
            }
            return false;
        }
        ggml_backend_tensor_get(s.out_logits, logits.data(), 0, logits.size() * sizeof(float));
        if (raw) {
            std::copy(logits.begin(), logits.end(), raw + (size_t) i * B * V);
        }
        codes[i] = choose(i, logits.data(), B);
    }
    return true;
}
