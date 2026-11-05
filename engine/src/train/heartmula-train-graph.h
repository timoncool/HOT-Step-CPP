#pragma once
// train/heartmula-train-graph.h — the HeartMuLa LoRA training graphs.
//
// HOT-Step file, TRAINING-SIDE (ace-train only). Builds the loss of
// engine/docs/HEARTMULA_TRAINING_CONTRACT.md §2 as a chain of SEGMENTS with
// per-layer recompute, and obeys the graph-construction rules of §6.2 (G1-G7).
//
// ── Segments ───────────────────────────────────────────────────────────────
//
// A micro-step is one sample: S prompt rows, T frames, L = S + T - 1 backbone
// positions. Checkpoints are persistent F32 tensors:
//
//   C[0..28]   [H, L]   backbone residual stream: C[0] the input embedding,
//                       C[l+1] = global block l (C[l])
//   Q[0..3]    [H, 8T]  local residual stream, frame-major (column p + 8t):
//                       Q[0] = projection([norm(C[28]) at S-1+t, emb(c[t][0..6])]),
//                       Q[l+1] = local block l (Q[l])
//
// Forward (no autodiff): embed -> C[0], 28 global blocks, head -> Q[0], three
// local blocks. Backward, last segment first, each one recomputed from its
// checkpoint and differentiated with the surrogate sum(Y * dY) (lm-ckpt.h
// explains why that is exact):
//
//   tail      Q[3] -> norm -> audio_head[0..6] -> CE 1..7    dQ[3]
//   local l   Q[l] -> Q[l+1]                                  dQ[l]
//   head      C[28] -> norm -> codebook0_head -> CE 0, and -> Q[0]   dC[28]
//   global l  C[l] -> C[l+1], LoRA grads into the optimizer   dC[l] (l > 0)
//
// The local decoder treats every frame independently, so the local and tail
// segments run in frame chunks: a chunk is a view of Q[l], and its dX lands
// in the matching columns of the gradient buffer.
//
// That is the full-recompute mode (--keep-layers -1). The default kept mode
// (--keep-layers N, N >= 0) builds blocks 28-N..27, the head, the local
// decoder and the tail as ONE graph with its backward (hmt_g_kept_bwd), so
// none of that is recomputed; only blocks below 28-N keep the per-layer
// segments, and Q / Gl are not allocated. Contract §8 has the measured
// ms / VRAM per N.
//
// The loss of a micro-step is (1/8) sum_k mean_t CE_k, divided by the number
// of micro-steps in its optimizer step (contract §1, "Batch"). That factor is
// baked into the CE scale (`wce`), so every loss node is seeded with 1.
// `cbw[k]` multiplies codebook k's term on top: all 1 by default, the
// --cb0-weighted factors otherwise (state.h hmt_cb_weights).
//
// ── G-rules (contract §6.2) as they appear here ────────────────────────────
//
//   G1  frozen weights are F16 or BF16; the activation gradient goes through
//       GGML_BACKWARD_MM=1 (set by ace-train before the first graph), and the
//       init probe fails on any node the GPU refuses.
//   G2  no CONCAT: the local input is ACC of the backbone rows into row 0 of a
//       gated embedding canvas.
//   G3  local GQA folds (heads, frames) into ne2, heads fastest; ne3 stays 1.
//   G4  local q/k are laid out [D, heads*frames, 8] before RoPE.
//   G5  v is transposed with one CONT over the combined permute.
//   G6  local attention is exact (head dim 384); global attention is
//       FLASH_ATTN_TRAIN (BF16 q/k/v where the device has the kernels, else
//       TF32) or exact, per --attn.
//   G7  activations stay F32.

#include "heartmula/lm-graph.h"
#include "train/lm-ckpt.h"  // lm_ckpt_fill_gacc: checkpoint input -> dX, LoRA -> acc, loss -> 1

#include <functional>
#include <string>
#include <vector>

#define HMT_MAX_NODES 32768

// One LoRA site: y = W x + scale * B (A x). A [in, r], B [r, out], F32 params.
struct HmtSite {
    ggml_tensor * A = nullptr;
    ggml_tensor * B = nullptr;
};

// Per backbone layer: q, k, v, output, then the MLP's w1 (gate), w2 (down), w3 (up),
// the order adapter.h's proj_name uses. An attention-only run leaves 4..6 empty.
#define HMT_SITES 7
struct HmtLayerLora {
    HmtSite s[HMT_SITES];
};

// Sample inputs, one buffer per micro-step shape (rebuilt when L or T change).
struct HmtSampleBuf {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_context *        lctx = nullptr;  // labels live alone so one clear zeroes them
    ggml_backend_buffer_t lbuf = nullptr;
    int64_t               S = 0, T = 0, L = 0;
    bool                  consts = false;  // pos and the masks depend on L only: uploaded once per shape

    ggml_tensor * pos   = nullptr;  // I32 [L]
    ggml_tensor * mask  = nullptr;  // F32 [L, L] causal
    ggml_tensor * mask16 = nullptr; // F16 [L, L] causal, for FLASH_ATTN_TRAIN
    ggml_tensor * tids  = nullptr;  // I32 [L] text ids (prompt rows)
    ggml_tensor * aids  = nullptr;  // I32 [8L] flat audio_embeddings rows
    ggml_tensor * wt    = nullptr;  // F32 [1, L] text gate
    ggml_tensor * wm    = nullptr;  // F32 [1, L] MuQ-bias gate
    ggml_tensor * wa    = nullptr;  // F32 [1, 8, L] audio gates
    ggml_tensor * lids  = nullptr;  // I32 [8T] local canvas rows
    ggml_tensor * lgate = nullptr;  // F32 [1, 8T] 0 at row 0 of every frame
    ggml_tensor * ones  = nullptr;  // F32 [1, 8T] for the label scatter
    ggml_tensor * lidx  = nullptr;  // I64 [8T] flat one-hot positions
    ggml_tensor * labels = nullptr; // F32 [V, T, 8] one-hot, codebook-major
};

// Checkpoints, gradient ping-pongs and constants, sized for the longest sample.
struct HmtCkpt {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    int64_t               Lcap = 0, Tcap = 0;
    std::vector<ggml_tensor *> C;  // 29 x [H, Lcap]
    std::vector<ggml_tensor *> Q;  // 4 x [H, 8 Tcap]
    ggml_tensor * Gg[2] = { nullptr, nullptr };  // [H, Lcap]
    ggml_tensor * Gl[2] = { nullptr, nullptr };  // [H, 8 Tcap]
    ggml_tensor * lpos  = nullptr;  // I32 [8] 0..7
    ggml_tensor * lmask = nullptr;  // F32 [8, 8] causal
    ggml_tensor * t_one = nullptr;  // F32 [1] = 1, the seed of every loss node
    ggml_tensor * gn2   = nullptr;  // F32 [1] readback of the finite-grad check
};

// Everything a segment build needs.
struct HmtGraphCtx {
    const HeartMuLaModel *            m     = nullptr;
    const std::vector<HmtLayerLora> * lora  = nullptr;
    float                             scale = 1.0f;  // alpha / rank
    int                               attn  = 0;     // global attention: 0 exact, 1 flash (TF32), 2 flash BF16 q/k/v
    ggml_tensor *                     drop_pool = nullptr;  // F32 dropout mask pool, null at p = 0
    uint64_t                          drop_key  = 0;        // per micro-step: picks every site's mask window
    float                             cbw[8]    = { 1, 1, 1, 1, 1, 1, 1, 1 };  // per-codebook loss factor
    const LmOptim *                   opt   = nullptr;
    HmtCkpt *                         ck    = nullptr;
    HmtSampleBuf *                    sb    = nullptr;
    ggml_backend_sched_t              sched = nullptr;
    ggml_context *                    ctx   = nullptr;  // the last graph's; freed by the next build
    std::vector<uint8_t>              arena;
    std::vector<ggml_tensor *>        gacc;
};

static void hmt_sample_free(HmtSampleBuf * s) {
    if (s->buf) ggml_backend_buffer_free(s->buf);
    if (s->ctx) ggml_free(s->ctx);
    if (s->lbuf) ggml_backend_buffer_free(s->lbuf);
    if (s->lctx) ggml_free(s->lctx);
    *s = HmtSampleBuf{};
}

static void hmt_ckpt_free(HmtCkpt * c) {
    if (c->buf) ggml_backend_buffer_free(c->buf);
    if (c->ctx) ggml_free(c->ctx);
    *c = HmtCkpt{};
}

static ggml_tensor * hmt_new(ggml_context * ctx, ggml_type t, int64_t n0, int64_t n1, int64_t n2, const char * name) {
    ggml_tensor * x = n2 > 1 ? ggml_new_tensor_3d(ctx, t, n0, n1, n2)
                    : n1 > 1 ? ggml_new_tensor_2d(ctx, t, n0, n1)
                             : ggml_new_tensor_1d(ctx, t, n0);
    ggml_set_name(x, name);
    return x;
}

// Persistent buffers. `Lcap`/`Tcap` are the longest sample's L and T. Only
// C[0..nC) exist (the rest stay null); the local checkpoints Q and Gl only with
// `local`, and Gg only when a global block is recomputed (nC > 1).
static bool hmt_ckpt_alloc(HmtCkpt * c, ggml_backend_t backend, int64_t Lcap, int64_t Tcap, int nC, bool local,
                           std::string * err) {
    hmt_ckpt_free(c);
    const int64_t    H  = 3072;
    ggml_init_params ip = { 64 * ggml_tensor_overhead(), nullptr, true };
    c->ctx              = ggml_init(ip);
    c->Lcap             = Lcap;
    c->Tcap             = Tcap;
    char nm[32];
    for (int l = 0; l <= 28; l++) {
        if (l >= nC) { c->C.push_back(nullptr); continue; }
        snprintf(nm, sizeof(nm), "hmt.C%d", l);
        c->C.push_back(hmt_new(c->ctx, GGML_TYPE_F32, H, Lcap, 1, nm));
        // ggml_set_param only asserts op == NONE, so a pre-allocated leaf is
        // legal; the flag is what lets the gacc table give it a dX buffer.
        // C[0] has nothing trainable upstream, so it never needs one.
        if (l > 0) ggml_set_param(c->C.back());
    }
    for (int l = 0; l <= 3 && local; l++) {
        snprintf(nm, sizeof(nm), "hmt.Q%d", l);
        c->Q.push_back(hmt_new(c->ctx, GGML_TYPE_F32, H, 8 * Tcap, 1, nm));
        ggml_set_param(c->Q.back());
    }
    if (nC > 1) {
        c->Gg[0] = hmt_new(c->ctx, GGML_TYPE_F32, H, Lcap, 1, "hmt.Gg0");
        c->Gg[1] = hmt_new(c->ctx, GGML_TYPE_F32, H, Lcap, 1, "hmt.Gg1");
    }
    if (local) {
        c->Gl[0] = hmt_new(c->ctx, GGML_TYPE_F32, H, 8 * Tcap, 1, "hmt.Gl0");
        c->Gl[1] = hmt_new(c->ctx, GGML_TYPE_F32, H, 8 * Tcap, 1, "hmt.Gl1");
    }
    c->lpos  = hmt_new(c->ctx, GGML_TYPE_I32, 8, 1, 1, "hmt.lpos");
    c->lmask = hmt_new(c->ctx, GGML_TYPE_F32, 8, 8, 1, "hmt.lmask");
    c->t_one = hmt_new(c->ctx, GGML_TYPE_F32, 1, 1, 1, "hmt.one");
    c->gn2   = hmt_new(c->ctx, GGML_TYPE_F32, 1, 1, 1, "hmt.gn2");
    c->buf   = ggml_backend_alloc_ctx_tensors(c->ctx, backend);
    if (!c->buf) {
        *err = "checkpoint buffers do not fit (" + std::to_string((long long) Lcap) + " positions)";
        return false;
    }
    ggml_backend_buffer_clear(c->buf, 0);
    int32_t pos[8];
    float   mask[64];
    for (int i = 0; i < 8; i++) {
        pos[i] = i;
        for (int j = 0; j < 8; j++) mask[i * 8 + j] = j <= i ? 0.0f : -INFINITY;
    }
    const float one = 1.0f;
    ggml_backend_tensor_set(c->lpos, pos, 0, sizeof(pos));
    ggml_backend_tensor_set(c->lmask, mask, 0, sizeof(mask));
    ggml_backend_tensor_set(c->t_one, &one, 0, sizeof(one));
    return true;
}

static bool hmt_sample_alloc(HmtSampleBuf * s, ggml_backend_t backend, int64_t S, int64_t T, std::string * err) {
    if (s->buf && s->S == S && s->T == T) return true;
    hmt_sample_free(s);
    const int64_t    L  = S + T - 1;
    ggml_init_params ip = { 32 * ggml_tensor_overhead(), nullptr, true };
    s->ctx              = ggml_init(ip);
    s->lctx             = ggml_init(ip);
    s->S = S, s->T = T, s->L = L;
    s->pos    = hmt_new(s->ctx, GGML_TYPE_I32, L, 1, 1, "hmt.pos");
    s->mask   = hmt_new(s->ctx, GGML_TYPE_F32, L, L, 1, "hmt.mask");
    s->mask16 = hmt_new(s->ctx, GGML_TYPE_F16, L, L, 1, "hmt.mask16");
    s->tids   = hmt_new(s->ctx, GGML_TYPE_I32, L, 1, 1, "hmt.tids");
    s->aids   = hmt_new(s->ctx, GGML_TYPE_I32, 8 * L, 1, 1, "hmt.aids");
    s->wt     = hmt_new(s->ctx, GGML_TYPE_F32, 1, L, 1, "hmt.wt");
    s->wm     = hmt_new(s->ctx, GGML_TYPE_F32, 1, L, 1, "hmt.wm");
    s->wa     = hmt_new(s->ctx, GGML_TYPE_F32, 1, 8, L, "hmt.wa");
    s->lids   = hmt_new(s->ctx, GGML_TYPE_I32, 8 * T, 1, 1, "hmt.lids");
    s->lgate  = hmt_new(s->ctx, GGML_TYPE_F32, 1, 8 * T, 1, "hmt.lgate");
    s->ones   = hmt_new(s->ctx, GGML_TYPE_F32, 1, 8 * T, 1, "hmt.ones");
    s->lidx   = hmt_new(s->ctx, GGML_TYPE_I64, 8 * T, 1, 1, "hmt.lidx");
    s->labels = hmt_new(s->lctx, GGML_TYPE_F32, HEARTMULA_AUDIO_VOCAB, T, 8, "hmt.labels");
    s->buf    = ggml_backend_alloc_ctx_tensors(s->ctx, backend);
    s->lbuf   = ggml_backend_alloc_ctx_tensors(s->lctx, backend);
    if (!s->buf || !s->lbuf) {
        *err = "sample buffers do not fit (S=" + std::to_string((long long) S) + " T=" + std::to_string((long long) T) + ")";
        hmt_sample_free(s);
        return false;
    }
    return true;
}

// Upload one sample. `codes` is frame-major [T][8] for the window being trained.
static void hmt_sample_upload(HmtSampleBuf * s, const int32_t * prompt, int64_t muq_idx, const int32_t * codes) {
    const int64_t S = s->S, T = s->T, L = s->L, V = HEARTMULA_AUDIO_VOCAB;
    std::vector<int32_t> pos((size_t) L), tids((size_t) L, 0), aids((size_t) (8 * L), 0);
    std::vector<float>   mask(s->consts ? 0 : (size_t) (L * L)), wt((size_t) L, 0.0f), wm((size_t) L, 0.0f), wa((size_t) (8 * L), 0.0f);
    for (int64_t n = 0; n < L; n++) {
        if (!s->consts) {
            pos[(size_t) n] = (int32_t) n;
            for (int64_t j = 0; j < L; j++) mask[(size_t) (n * L + j)] = j <= n ? 0.0f : -INFINITY;
        }
        if (n < S) {
            tids[(size_t) n] = prompt[n];
            (n == muq_idx ? wm : wt)[(size_t) n] = 1.0f;  // §2.1: the MuQ row is muq_linear(0) = its bias
        } else {
            const int64_t t = n - S;  // frame t is fed at row S + t (frames 0..T-2)
            for (int k = 0; k < 8; k++) {
                aids[(size_t) (n * 8 + k)] = codes[t * 8 + k] + k * (int32_t) V;
                wa[(size_t) (n * 8 + k)]   = 1.0f;
            }
        }
    }
    std::vector<int32_t> lids((size_t) (8 * T), 0);
    std::vector<float>   lgate((size_t) (8 * T), 1.0f), ones((size_t) (8 * T), 1.0f);
    std::vector<int64_t> lidx((size_t) (8 * T));
    for (int64_t t = 0; t < T; t++) {
        lgate[(size_t) (t * 8)] = 0.0f;  // row 0 is the backbone row, ACC'd in (G2)
        for (int p = 1; p < 8; p++) lids[(size_t) (t * 8 + p)] = codes[t * 8 + p - 1] + (p - 1) * (int32_t) V;
        for (int k = 0; k < 8; k++) lidx[(size_t) (k * T + t)] = (k * T + t) * V + codes[t * 8 + k];
    }
    if (!s->consts) {
        ggml_backend_tensor_set(s->pos, pos.data(), 0, ggml_nbytes(s->pos));
        ggml_backend_tensor_set(s->mask, mask.data(), 0, ggml_nbytes(s->mask));
        std::vector<ggml_fp16_t> mask16(mask.size());
        ggml_fp32_to_fp16_row(mask.data(), mask16.data(), (int64_t) mask.size());
        ggml_backend_tensor_set(s->mask16, mask16.data(), 0, ggml_nbytes(s->mask16));
        s->consts = true;
    }
    ggml_backend_tensor_set(s->tids, tids.data(), 0, ggml_nbytes(s->tids));
    ggml_backend_tensor_set(s->aids, aids.data(), 0, ggml_nbytes(s->aids));
    ggml_backend_tensor_set(s->wt, wt.data(), 0, ggml_nbytes(s->wt));
    ggml_backend_tensor_set(s->wm, wm.data(), 0, ggml_nbytes(s->wm));
    ggml_backend_tensor_set(s->wa, wa.data(), 0, ggml_nbytes(s->wa));
    ggml_backend_tensor_set(s->lids, lids.data(), 0, ggml_nbytes(s->lids));
    ggml_backend_tensor_set(s->lgate, lgate.data(), 0, ggml_nbytes(s->lgate));
    ggml_backend_tensor_set(s->ones, ones.data(), 0, ggml_nbytes(s->ones));
    ggml_backend_tensor_set(s->lidx, lidx.data(), 0, ggml_nbytes(s->lidx));
}

// ── Blocks ─────────────────────────────────────────────────────────────────

// LoRA dropout (contract §1, lever 6): the A branch reads x times a mask. Each
// mask is a window of the persistent pool (Bernoulli(1 - p) values of 0 or
// 1 / (1 - p), drawn once from the seed) at an offset hashed from the micro-
// step key, the layer and the site. A recompute reads the same window, so the
// masks reproduce exactly with no per-step upload and nothing retained.
static ggml_tensor * hmt_drop(ggml_context * ctx, const HmtGraphCtx & g, ggml_tensor * x, int l, int site) {
    if (!g.drop_pool) return x;
    const int64_t n     = ggml_nelements(x);
    const int64_t slots = (ggml_nelements(g.drop_pool) - n) / 16;
    // Attention sites keep their original index so older dropout runs resume unchanged.
    const uint64_t id   = site < 4 ? (uint64_t) (l * 4 + site + 1) : (uint64_t) (28 * 4 + l * 3 + site - 3);
    uint64_t      z     = g.drop_key + 0x9E3779B97F4A7C15ull * id;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const size_t  off   = (size_t) (z % (uint64_t) slots) * 16 * sizeof(float);
    return ggml_mul(ctx, x, ggml_view_2d(ctx, g.drop_pool, x->ne[0], x->ne[1], x->ne[0] * sizeof(float), off));
}

static ggml_tensor * hmt_linear(ggml_context * ctx, ggml_tensor * w, const HmtSite * site, float scale,
                                ggml_tensor * x, ggml_tensor * xa = nullptr) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (site && site->A) {
        ggml_tensor * t = ggml_mul_mat(ctx, site->A, xa ? xa : x);
        if (scale != 1.0f) t = ggml_scale(ctx, t, scale);
        y = ggml_add(ctx, y, ggml_mul_mat(ctx, site->B, t));
    }
    return y;
}

// `lin(W, site, in)` applies the site's LoRA when the run trains the MLP; null = frozen.
static ggml_tensor * hmt_mlp(ggml_context * ctx, const HeartMuLaLayer & w, ggml_tensor * x, float eps,
                             const std::function<ggml_tensor *(ggml_tensor *, int, ggml_tensor *)> & lin = nullptr) {
    ggml_tensor * n = heartmula_rms(ctx, x, w.mlp_norm, eps);
    if (lin) {
        ggml_tensor * g = ggml_swiglu_split(ctx, lin(w.w1, 4, n), lin(w.w3, 6, n));
        return ggml_add(ctx, x, lin(w.w2, 5, g));
    }
    // Split SwiGLU: its backward keeps only the two GEMM outputs and recomputes the
    // silu, so the [ffn, rows] silu output is never retained.
    ggml_tensor * g = ggml_swiglu_split(ctx, ggml_mul_mat(ctx, w.w1, n), ggml_mul_mat(ctx, w.w3, n));
    return ggml_add(ctx, x, ggml_mul_mat(ctx, w.w2, g));
}

// Global block on x [H, L] (batch 1, so every ne3 is 1).
static ggml_tensor * hmt_global_block(ggml_context * ctx, const HmtGraphCtx & g, int l, ggml_tensor * x) {
    const HeartMuLaStack &    st = g.m->global;
    const HeartMuLaLmConfig & c  = st.cfg;
    const HeartMuLaLayer &    w  = st.layers[(size_t) l];
    const HmtLayerLora *      lr = g.lora ? &(*g.lora)[(size_t) l] : nullptr;
    const int64_t             D = c.head_dim, L = x->ne[1];
    ggml_tensor * n = heartmula_rms(ctx, x, w.sa_norm, c.rms_eps);
    auto lin = [&](ggml_tensor * W, int s, ggml_tensor * in) {
        return hmt_linear(ctx, W, lr ? &lr->s[s] : nullptr, g.scale, in, lr ? hmt_drop(ctx, g, in, l, s) : nullptr);
    };
    ggml_tensor * q = ggml_reshape_3d(ctx, lin(w.q, 0, n), D, c.n_head, L);
    ggml_tensor * k = ggml_reshape_3d(ctx, lin(w.k, 1, n), D, c.n_head_kv, L);
    ggml_tensor * v = ggml_reshape_3d(ctx, lin(w.v, 2, n), D, c.n_head_kv, L);
    q = heartmula_rope(ctx, q, g.sb->pos, st);
    k = heartmula_rope(ctx, k, g.sb->pos, st);
    ggml_tensor * o;
    if (g.attn) {
        // FLASH_ATTN_TRAIN reads q/k/v as permuted views (only nb[0] == 4 needed) and
        // keeps O + LSE instead of the [L, L, Nh] softmax. BF16 casts q/k/v only (G7).
        ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3), * kp = ggml_permute(ctx, k, 0, 2, 1, 3);
        ggml_tensor * vp = ggml_permute(ctx, v, 0, 2, 1, 3);
        if (g.attn == 2) {
            qp = ggml_cast(ctx, qp, GGML_TYPE_BF16);
            kp = ggml_cast(ctx, kp, GGML_TYPE_BF16);
            vp = ggml_cast(ctx, vp, GGML_TYPE_BF16);
        }
        ggml_tensor * packed = ggml_flash_attn_train(ctx, qp, kp, vp, g.sb->mask16, 1.0f / sqrtf((float) D));
        ggml_flash_attn_train_set_prec(packed, g.attn == 2 ? GGML_PREC_BF16 : GGML_PREC_DEFAULT);
        ggml_flash_attn_train_set_causal(packed, 0);
        o = ggml_reshape_2d(ctx, ggml_flash_attn_train_get_o(ctx, packed), D * c.n_head, L);  // [D, Nh, L]
    } else {
        ggml_tensor * q3 = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [D, L, Nh]
        ggml_tensor * k3 = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [D, L, Nkv]
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [L, D, Nkv] (G5)
        ggml_tensor * s  = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, k3, q3), g.sb->mask, 1.0f / sqrtf((float) D), 0.0f);
        o = ggml_cont(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, vt, s), 0, 2, 1, 3));  // [D, Nh, L]
        o = ggml_reshape_2d(ctx, o, D * c.n_head, L);
    }
    x = ggml_add(ctx, x, lin(w.o, 3, o));
    if (lr && lr->s[4].A) return hmt_mlp(ctx, w, x, c.rms_eps, lin);
    return hmt_mlp(ctx, w, x, c.rms_eps);
}

// Local block on x [H, 8 Tc], frame-major. No LoRA (the local decoder is frozen).
static ggml_tensor * hmt_local_block(ggml_context * ctx, const HmtGraphCtx & g, int l, ggml_tensor * x) {
    const HeartMuLaStack &    st = g.m->local;
    const HeartMuLaLmConfig & c  = st.cfg;
    const HeartMuLaLayer &    w  = st.layers[(size_t) l];
    const int64_t             D = c.head_dim, Nh = c.n_head, Nkv = c.n_head_kv, Tc = x->ne[1] / 8;
    ggml_tensor * n = heartmula_rms(ctx, x, w.sa_norm, c.rms_eps);
    // G4: [D, heads, 8, Tc] -> [D, heads*Tc, 8] (heads fastest), one CONT, then RoPE over the 8 positions.
    auto lay = [&](ggml_tensor * y, int64_t heads) {
        y = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, y, D, heads, 8, Tc), 0, 1, 3, 2));
        return heartmula_rope(ctx, ggml_reshape_3d(ctx, y, D, heads * Tc, 8), g.ck->lpos, st);
    };
    ggml_tensor * q = lay(ggml_mul_mat(ctx, w.q, n), Nh);
    ggml_tensor * k = lay(ggml_mul_mat(ctx, w.k, n), Nkv);
    // G3: [D, 8, heads*Tc]; q index t*Nh + h reads kv index t*Nkv + h/2 through the ne2 broadcast.
    ggml_tensor * q3 = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * k3 = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    // G5: v [D, Nkv, 8, Tc] -> [8, D, Nkv, Tc] in one CONT, then [8, D, Nkv*Tc].
    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, ggml_mul_mat(ctx, w.v, n), D, Nkv, 8, Tc), 1, 2, 0, 3));
    vt = ggml_reshape_3d(ctx, vt, 8, D, Nkv * Tc);
    ggml_tensor * s = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, k3, q3), g.ck->lmask, 1.0f / sqrtf((float) D), 0.0f);
    ggml_tensor * o = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, vt, s), D, 8, Nh, Tc);  // [D, 8, Nh, Tc]
    o = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)), D * Nh, 8 * Tc);
    x = ggml_add(ctx, x, ggml_mul_mat(ctx, w.o, o));
    return hmt_mlp(ctx, w, x, c.rms_eps);
}

// ── Graph plumbing ─────────────────────────────────────────────────────────

static ggml_context * hmt_begin(HmtGraphCtx & g, ggml_cgraph ** gf, bool grads) {
    const size_t bytes = ggml_tensor_overhead() * (HMT_MAX_NODES + 256) + ggml_graph_overhead_custom(HMT_MAX_NODES, grads);
    if (g.ctx) ggml_free(g.ctx);  // the arena holds the tensors; only the context struct is owned
    if (g.arena.size() < bytes) g.arena.resize(bytes);
    ggml_init_params ip  = { g.arena.size(), g.arena.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    g.ctx                = ctx;
    *gf                  = ggml_new_graph_custom(ctx, HMT_MAX_NODES, grads);
    return ctx;
}

// [H, n] columns [c0, c0 + n) of a checkpoint or gradient buffer.
static ggml_tensor * hmt_cols(ggml_context * ctx, ggml_tensor * t, int64_t c0, int64_t n) {
    return ggml_view_2d(ctx, t, t->ne[0], n, t->nb[1], (size_t) c0 * t->nb[1]);
}

// Surrogate for a segment whose output Y continues into the gradient dY.
static ggml_tensor * hmt_surrogate(ggml_context * ctx, ggml_tensor * Y, ggml_tensor * dY) {
    return ggml_sum(ctx, ggml_mul(ctx, Y, dY));
}

// Finish a backward graph: loss root, gacc table (`X` -> `dX`), autodiff.
static void hmt_backward(HmtGraphCtx & g, ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * loss, ggml_tensor * X,
                         ggml_tensor * dX) {
    ggml_set_loss(loss);
    ggml_build_forward_expand(gf, loss);
    lm_ckpt_fill_gacc(g.opt, gf, X, dX, g.ck->t_one, &g.gacc);
    ggml_build_backward_expand(ctx, gf, g.gacc.data());
}

// ── Segment builders. Each returns the graph, built in g.arena. ─────────────

// C[0] = sum_k wa_k audio_k + wt text + wm muq_bias (the inference input of lm-graph.h, note B).
static ggml_cgraph * hmt_g_embed(HmtGraphCtx & g) {
    const HeartMuLaModel & m = *g.m;
    const int64_t          H = m.global.cfg.n_embd, L = g.sb->L;
    ggml_cgraph *          gf;
    ggml_context *         ctx = hmt_begin(g, &gf, false);
    ggml_tensor * a = ggml_mul(ctx, ggml_reshape_3d(ctx, ggml_get_rows(ctx, m.audio_embd, g.sb->aids), H, 8, L), g.sb->wa);
    ggml_tensor * h = ggml_view_2d(ctx, a, H, L, a->nb[2], 0);
    for (int k = 1; k < 8; k++) h = ggml_add(ctx, h, ggml_view_2d(ctx, a, H, L, a->nb[2], (size_t) k * a->nb[1]));
    h = ggml_add(ctx, h, ggml_mul(ctx, ggml_get_rows(ctx, m.text_embd, g.sb->tids), g.sb->wt));
    h = ggml_add(ctx, h, ggml_mul(ctx, ggml_repeat(ctx, m.muq_bias, h), g.sb->wm));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, h, hmt_cols(ctx, g.ck->C[0], 0, L)));
    return gf;
}

static ggml_cgraph * hmt_g_global_fwd(HmtGraphCtx & g, int l) {
    const int64_t  L = g.sb->L;
    ggml_cgraph *  gf;
    ggml_context * ctx = hmt_begin(g, &gf, false);
    ggml_tensor *  Y   = hmt_global_block(ctx, g, l, hmt_cols(ctx, g.ck->C[(size_t) l], 0, L));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, Y, hmt_cols(ctx, g.ck->C[(size_t) l + 1], 0, L)));
    return gf;
}

static ggml_cgraph * hmt_g_global_bwd(HmtGraphCtx & g, int l, int cur) {
    const int64_t  L = g.sb->L;
    ggml_cgraph *  gf;
    ggml_context * ctx = hmt_begin(g, &gf, true);
    ggml_tensor *  Y   = hmt_global_block(ctx, g, l, hmt_cols(ctx, g.ck->C[(size_t) l], 0, L));
    ggml_tensor *  Ls  = hmt_surrogate(ctx, Y, hmt_cols(ctx, g.ck->Gg[cur], 0, L));
    hmt_backward(g, ctx, gf, Ls, l > 0 ? g.ck->C[(size_t) l] : nullptr, l > 0 ? g.ck->Gg[1 - cur] : nullptr);
    return gf;
}

// The head: norm(C[28]) rows S-1.. -> codebook 0 CE, and the local input Q[0].
struct HmtHeadOut {
    ggml_tensor * ce0    = nullptr;
    ggml_tensor * last   = nullptr;  // [H, T] normed backbone rows (tap)
    ggml_tensor * logits = nullptr;  // [V, T] codebook 0 (tap)
};

static ggml_tensor * hmt_head(ggml_context * ctx, HmtGraphCtx & g, ggml_tensor * X, HmtHeadOut * out) {
    const HeartMuLaModel & m = *g.m;
    const int64_t          H = m.global.cfg.n_embd, S = g.sb->S, T = g.sb->T;
    ggml_tensor * hn   = heartmula_rms(ctx, X, m.global.norm, m.global.cfg.rms_eps);
    ggml_tensor * last = ggml_cont(ctx, ggml_view_2d(ctx, hn, H, T, hn->nb[1], (size_t) (S - 1) * hn->nb[1]));
    if (out) {
        out->last   = last;
        out->logits = ggml_mul_mat(ctx, m.c0_head, last);
        out->ce0    = ggml_cross_entropy_loss(ctx, out->logits, ggml_view_2d(ctx, g.sb->labels, HEARTMULA_AUDIO_VOCAB, T,
                                                                                  g.sb->labels->nb[1], 0));
    }
    // G2: [last_h, emb_0(c0), ..., emb_6(c6)] as ACC of `last` into row 0 of a gated canvas.
    ggml_tensor * canvas = ggml_mul(ctx, ggml_get_rows(ctx, m.audio_embd, g.sb->lids), g.sb->lgate);  // [H, 8T]
    canvas               = ggml_reshape_3d(ctx, canvas, H, 8, T);
    ggml_tensor * in     = ggml_acc(ctx, canvas, ggml_reshape_3d(ctx, last, H, 1, T), canvas->nb[1], canvas->nb[2],
                                    canvas->nb[3], 0);
    return ggml_mul_mat(ctx, m.projection, ggml_reshape_2d(ctx, in, H, 8 * T));
}

static ggml_cgraph * hmt_g_head_fwd(HmtGraphCtx & g) {
    ggml_cgraph *  gf;
    ggml_context * ctx = hmt_begin(g, &gf, false);
    ggml_tensor *  P   = hmt_head(ctx, g, hmt_cols(ctx, g.ck->C[28], 0, g.sb->L), nullptr);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, P, hmt_cols(ctx, g.ck->Q[0], 0, 8 * g.sb->T)));
    return gf;
}

static ggml_cgraph * hmt_g_head_bwd(HmtGraphCtx & g, int cur, float wce, HmtHeadOut * out, bool taps) {
    ggml_cgraph *  gf;
    ggml_context * ctx = hmt_begin(g, &gf, true);
    ggml_tensor *  P   = hmt_head(ctx, g, hmt_cols(ctx, g.ck->C[28], 0, g.sb->L), out);
    ggml_tensor *  Ls  = ggml_add(ctx, ggml_scale(ctx, out->ce0, wce * g.cbw[0]),
                                  hmt_surrogate(ctx, P, hmt_cols(ctx, g.ck->Gl[cur], 0, 8 * g.sb->T)));
    ggml_set_output(out->ce0);
    if (taps) {
        ggml_set_output(out->last);
        ggml_set_output(out->logits);
    }
    hmt_backward(g, ctx, gf, Ls, g.ck->C[28], g.ck->Gg[0]);
    return gf;
}

static ggml_cgraph * hmt_g_local_fwd(HmtGraphCtx & g, int l, int64_t f0, int64_t Tc) {
    ggml_cgraph *  gf;
    ggml_context * ctx = hmt_begin(g, &gf, false);
    ggml_tensor *  Y   = hmt_local_block(ctx, g, l, hmt_cols(ctx, g.ck->Q[(size_t) l], 8 * f0, 8 * Tc));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, Y, hmt_cols(ctx, g.ck->Q[(size_t) l + 1], 8 * f0, 8 * Tc)));
    return gf;
}

static ggml_cgraph * hmt_g_local_bwd(HmtGraphCtx & g, int l, int64_t f0, int64_t Tc, int cur) {
    ggml_cgraph *  gf;
    ggml_context * ctx = hmt_begin(g, &gf, true);
    ggml_tensor *  Y   = hmt_local_block(ctx, g, l, hmt_cols(ctx, g.ck->Q[(size_t) l], 8 * f0, 8 * Tc));
    ggml_tensor *  Ls  = hmt_surrogate(ctx, Y, hmt_cols(ctx, g.ck->Gl[cur], 8 * f0, 8 * Tc));
    hmt_backward(g, ctx, gf, Ls, g.ck->Q[(size_t) l], g.ck->Gl[1 - cur]);
    return gf;
}

// Codebooks 1..7 of frames [f0, f0 + Tc). `ce[i]` is CE_i over the chunk (mean
// over its rows); the loss weights each by Tc / T so the chunks sum to the mean.
struct HmtTailOut {
    ggml_tensor * ce[8]     = {};
    ggml_tensor * dec       = nullptr;  // [H, 8, Tc] normed local output (tap)
    ggml_tensor * logits[8] = {};       // [V, Tc] (tap)
};

static ggml_cgraph * hmt_g_tail_bwd(HmtGraphCtx & g, int64_t f0, int64_t Tc, float wce, int dst, HmtTailOut * out,
                                    bool taps) {
    const HeartMuLaModel & m = *g.m;
    const int64_t          H = m.local.cfg.n_embd, T = g.sb->T, V = HEARTMULA_AUDIO_VOCAB;
    ggml_cgraph *          gf;
    ggml_context *         ctx = hmt_begin(g, &gf, true);
    ggml_tensor * X  = hmt_cols(ctx, g.ck->Q[3], 8 * f0, 8 * Tc);
    ggml_tensor * y  = ggml_reshape_3d(ctx, heartmula_rms(ctx, X, m.local.norm, m.local.cfg.rms_eps), H, 8, Tc);
    ggml_tensor * Ls = nullptr;
    out->dec         = y;
    for (int i = 1; i < 8; i++) {
        ggml_tensor * yi  = ggml_cont(ctx, ggml_view_2d(ctx, y, H, Tc, y->nb[2], (size_t) i * y->nb[1]));
        ggml_tensor * lg  = ggml_mul_mat(ctx, m.audio_head[i - 1], yi);
        ggml_tensor * lab = ggml_view_2d(ctx, g.sb->labels, V, Tc, g.sb->labels->nb[1],
                                         (size_t) i * g.sb->labels->nb[2] + (size_t) f0 * g.sb->labels->nb[1]);
        out->ce[i]     = ggml_cross_entropy_loss(ctx, lg, lab);
        out->logits[i] = lg;
        ggml_set_output(out->ce[i]);
        if (taps) ggml_set_output(lg);
        ggml_tensor * term = ggml_scale(ctx, out->ce[i], wce * g.cbw[i] * (float) Tc / (float) T);
        Ls                 = Ls ? ggml_add(ctx, Ls, term) : term;
    }
    if (taps) ggml_set_output(y);
    hmt_backward(g, ctx, gf, Ls, g.ck->Q[3], g.ck->Gl[dst]);
    return gf;
}

// Most frames one local segment may hold. The local GQA backward sums the
// Nh * Tc broadcast heads with REPEAT_BACK, and CUDA takes at most 1 << 15 in
// its source's ne2 * ne3, so 4096 frames at 8 heads.
static int64_t hmt_local_frames_max(const HmtGraphCtx & g) {
    return (int64_t) (1 << 15) / g.m->local.cfg.n_head;
}

// Kept tail: global blocks k0..27, the head, the whole local decoder and all
// eight CEs in ONE graph with its backward, so none of it is recomputed. The
// graph's dX for C[k0] lands in Gg[0]; blocks below k0 keep the per-layer
// recompute. Outputs and taps as hmt_g_head_bwd / hmt_g_tail_bwd (Tc = T).
// Past hmt_local_frames_max the local decoder and tail run as frame chunks
// inside the graph, each CE weighted Tc / T; the taps then hold the last chunk.
static ggml_cgraph * hmt_g_kept_bwd(HmtGraphCtx & g, int k0, float wce, HmtHeadOut * head, HmtTailOut * tail, bool taps) {
    const HeartMuLaModel & m = *g.m;
    const int64_t          H = m.local.cfg.n_embd, T = g.sb->T, V = HEARTMULA_AUDIO_VOCAB;
    const int64_t          cap = hmt_local_frames_max(g);
    ggml_cgraph *          gf;
    ggml_context *         ctx = hmt_begin(g, &gf, true);
    ggml_tensor *          X   = hmt_cols(ctx, g.ck->C[(size_t) k0], 0, g.sb->L);
    for (int l = k0; l < 28; l++) X = hmt_global_block(ctx, g, l, X);
    ggml_tensor * Q  = hmt_head(ctx, g, X, head);
    ggml_tensor * Ls = ggml_scale(ctx, head->ce0, wce * g.cbw[0]);
    ggml_set_output(head->ce0);
    ggml_tensor * y = nullptr, * ce[8] = {};
    for (int64_t f0 = 0; f0 < T; f0 += cap) {
        const int64_t Tc = std::min(cap, T - f0);
        ggml_tensor * Qc = hmt_cols(ctx, Q, 8 * f0, 8 * Tc);
        for (int l = 0; l < 3; l++) Qc = hmt_local_block(ctx, g, l, Qc);
        y = ggml_reshape_3d(ctx, heartmula_rms(ctx, Qc, m.local.norm, m.local.cfg.rms_eps), H, 8, Tc);
        for (int i = 1; i < 8; i++) {
            ggml_tensor * yi  = ggml_cont(ctx, ggml_view_2d(ctx, y, H, Tc, y->nb[2], (size_t) i * y->nb[1]));
            ggml_tensor * lg  = ggml_mul_mat(ctx, m.audio_head[i - 1], yi);
            ggml_tensor * lab = ggml_view_2d(ctx, g.sb->labels, V, Tc, g.sb->labels->nb[1],
                                             (size_t) i * g.sb->labels->nb[2] + (size_t) f0 * g.sb->labels->nb[1]);
            ggml_tensor * c   = ggml_cross_entropy_loss(ctx, lg, lab);
            if (Tc < T) c = ggml_scale(ctx, c, (float) Tc / (float) T);
            ce[i]           = ce[i] ? ggml_add(ctx, ce[i], c) : c;
            tail->logits[i] = lg;
            if (taps) ggml_set_output(lg);
        }
    }
    tail->dec = y;
    for (int i = 1; i < 8; i++) {
        tail->ce[i] = ce[i];
        ggml_set_output(ce[i]);
        Ls = ggml_add(ctx, Ls, ggml_scale(ctx, ce[i], wce * g.cbw[i]));
    }
    if (taps) {
        ggml_set_output(y);
        ggml_set_output(head->last);
        ggml_set_output(head->logits);
    }
    hmt_backward(g, ctx, gf, Ls, k0 > 0 ? g.ck->C[(size_t) k0] : nullptr, k0 > 0 ? g.ck->Gg[0] : nullptr);
    return gf;
}

// Forward-only scoring (held-out CE, run.h hmt_eval): the head's CE 0 and Q[0],
// then CE 1..7 of frames [f0, f0 + Tc) from Q[3]. No autodiff, no LoRA dropout.
static ggml_cgraph * hmt_g_head_eval(HmtGraphCtx & g, HmtHeadOut * out) {
    ggml_cgraph *  gf;
    ggml_context * ctx = hmt_begin(g, &gf, false);
    ggml_tensor *  P   = hmt_head(ctx, g, hmt_cols(ctx, g.ck->C[28], 0, g.sb->L), out);
    ggml_set_output(out->ce0);
    ggml_build_forward_expand(gf, out->ce0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, P, hmt_cols(ctx, g.ck->Q[0], 0, 8 * g.sb->T)));
    return gf;
}

static ggml_cgraph * hmt_g_tail_eval(HmtGraphCtx & g, int64_t f0, int64_t Tc, HmtTailOut * out) {
    const HeartMuLaModel & m = *g.m;
    const int64_t          H = m.local.cfg.n_embd, V = HEARTMULA_AUDIO_VOCAB;
    ggml_cgraph *          gf;
    ggml_context *         ctx = hmt_begin(g, &gf, false);
    ggml_tensor * X = hmt_cols(ctx, g.ck->Q[3], 8 * f0, 8 * Tc);
    ggml_tensor * y = ggml_reshape_3d(ctx, heartmula_rms(ctx, X, m.local.norm, m.local.cfg.rms_eps), H, 8, Tc);
    for (int i = 1; i < 8; i++) {
        ggml_tensor * yi  = ggml_cont(ctx, ggml_view_2d(ctx, y, H, Tc, y->nb[2], (size_t) i * y->nb[1]));
        ggml_tensor * lab = ggml_view_2d(ctx, g.sb->labels, V, Tc, g.sb->labels->nb[1],
                                         (size_t) i * g.sb->labels->nb[2] + (size_t) f0 * g.sb->labels->nb[1]);
        out->ce[i] = ggml_cross_entropy_loss(ctx, ggml_mul_mat(ctx, m.audio_head[i - 1], yi), lab);
        ggml_set_output(out->ce[i]);
        ggml_build_forward_expand(gf, out->ce[i]);
    }
    return gf;
}

// One-hot labels: scatter 1.0 at (k, t, code) into the zeroed [V, T, 8] buffer.
static ggml_cgraph * hmt_g_labels(HmtGraphCtx & g) {
    ggml_cgraph *  gf;
    ggml_context * ctx  = hmt_begin(g, &gf, false);
    ggml_tensor *  flat = ggml_reshape_2d(ctx, g.sb->labels, 1, ggml_nelements(g.sb->labels));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, flat, g.sb->ones, g.sb->lidx));
    return gf;
}

// Sum of squares of every gradient accumulator, for the finite check before a step.
static ggml_cgraph * hmt_g_gradnorm(HmtGraphCtx & g) {
    ggml_cgraph *  gf;
    ggml_context * ctx = hmt_begin(g, &gf, false);
    ggml_tensor *  acc = nullptr;
    for (ggml_tensor * a : g.opt->acc) {
        ggml_tensor * s = ggml_sum(ctx, ggml_sqr(ctx, a));
        acc             = acc ? ggml_add(ctx, acc, s) : s;
    }
    ggml_build_forward_expand(gf, ggml_cpy(ctx, acc, g.ck->gn2));
    return gf;
}

// Every node of `gf` the training device runs natively? Names the refused ops (§6.3).
static bool hmt_probe_graph(ggml_backend_t gpu, ggml_cgraph * gf, const char * what, std::string * err) {
    std::string bad;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(gpu, t)) {
            char b[160];
            snprintf(b, sizeof(b), " %s(%s, src0 %s)", ggml_op_desc(t), ggml_type_name(t->type),
                     t->src[0] ? ggml_type_name(t->src[0]->type) : "-");
            if (bad.find(b) == std::string::npos) bad += b;
        }
    }
    if (!bad.empty()) {
        *err = std::string("the ") + what + " graph has nodes the GPU refuses, which would run on the CPU:" + bad;
        return false;
    }
    return true;
}
