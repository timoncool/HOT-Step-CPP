#pragma once
// train/mm3-dit-train-graph.h — trainable MiniMax-Music3 flow DiT.
//
// HOT-Step file. TRAINING-SIDE: built into ace-train only. Step 4 of the MM3
// training path (docs/plans/2026-08-15-mm3-dit-trainer-design.md).
//
// ── Why this is much smaller than dit-train-graph.h (the ACE one) ────────────
//
// The MM3 inference forward is already nearly differentiable, which was checked
// by reading mm3-dit-graph.h rather than assumed:
//
//   * A MANUAL ATTENTION PATH ALREADY EXISTS. mm3_dit_attn_f32() — the same
//     mul_mat -> soft_max_ext -> mul_mat -> permute shape ACE's trainer uses.
//     Flash sits behind MM3DitGraph::use_flash_attn. We simply never take it:
//     GGML_OP_FLASH_ATTN_EXT has no backward. No new attention code.
//   * THE GLU IS ALREADY EXPLICIT — ggml_mul(val, ggml_silu(gate)) over two
//     view_2d halves, not the fused ggml_swiglu (which has no backward).
//   * NO CROSS-ATTENTION AND NO AdaLN. Conditioning enters by channel concat at
//     the input. So none of the ACE trainer's cross-attn masking, encoder
//     padding, or 6-way AdaLN split exists here. This is the single biggest
//     simplification.
//   * Norms are LayerNorm (ggml_norm), not rms_norm — so ACE's "rms_norm
//     backward is wrong on permuted inputs" trap does not apply.
//
// Everything else in the block has a backward: mul_mat, norm, silu, rope_ext,
// soft_max_ext, view/cont, add, mul.
//
// ── What this file adds over the inference forward ──────────────────────────
//
//   1. LoRA sites at the six projections mm3-adapter.h merges, so a trained
//      adapter loads back through the EXISTING path with no new format:
//      attn_qkv, attn_output, ffn_in, ffn_out, proj_in, proj_out.
//   2. The rectified-flow loss.
//
// ── The trap carried over from the ACE trainer ──────────────────────────────
//
// **Never let the token axis reach ne2 of a mul_mat whose other operand is a 2-D
// trainable factor.** ggml then emits the weight gradient as out_prod with
// dst->ne[2] == S and ggml-cuda takes its per-token fallback: one cublasSgemm
// per token. On the ACE trainer that was 9.7x. Activations here are [E, S] 2-D
// throughout and the LoRA factors are 2-D, so ne2 == 1 everywhere — but if a
// batch axis is ever added, fold it into the column count with a reshape rather
// than letting it land in ne2.

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "minimax/mm3-dit-graph.h"   // mm3_dit_ln, mm3_dit_attn_f32, weights, config

// ── LoRA sites ──────────────────────────────────────────────────────────────

// delta = (alpha/rank) * B @ A, added to the base projection's output.
// a: [in, rank]   b: [rank, out]   — ggml order, so both are 2-D with ne2 == 1.
struct MM3TrainLora {
    ggml_tensor * a     = nullptr;
    ggml_tensor * b     = nullptr;
    // Optional [out, 1] constant 0/1 mask on the delta's OUTPUT channels.
    // Used by --target mlpv to train the fused qkv's V rows only: masking the
    // delta also masks the backward (chain rule), so the q,k rows of B receive
    // zero gradient forever and stay at their zero init — the optimizer
    // CANNOT spend budget on attention routing. Measured motivation (run 09
    // group ablation, Rob's ears, 2026-08-16): q,k deltas scramble structure,
    // proj_in/out inject seed-dependent "radio tuning" fuzz, while MLP+V+out
    // carry timbre with clear lyrics.
    ggml_tensor * out_mask = nullptr;
    // DoRA (both null for plain LoRA). m [out] is a PARAM; nrm [out] is
    // ||W + s*BA||_col, refreshed once per optimizer window from frozen A/B and
    // never differentiated (paper section 4.3). keep [out,1] is 1 - out_mask:
    // masked rows get factor exactly 1, so their m takes no gradient and the
    // mlpv q,k rows stay the base.
    ggml_tensor * m        = nullptr;
    ggml_tensor * nrm      = nullptr;
    ggml_tensor * keep     = nullptr;
    float         scale = 1.0f;   // alpha / rank
    bool          on() const { return a && b; }
};

struct MM3TrainBlockAdapters {
    MM3TrainLora qkv, attn_out, ff_in, ff_out;
};

struct MM3TrainAdapters {
    std::vector<MM3TrainBlockAdapters> blk;
    MM3TrainLora                       proj_in, proj_out;
    // Shared V-only qkv mask when --target mlpv; owned by the adapter ctx,
    // filled once after buffer allocation (it is not a param, so the param
    // init loop never touches it).
    ggml_tensor *                      qkv_vmask = nullptr;
    ggml_tensor *                      qkv_vkeep = nullptr;   // 1 - qkv_vmask, DoRA only
};

// y = W x  (+ scale * B(A x) when the site is active).
//
// W is the frozen base in its GGUF type; x is [in, S]. The LoRA branch runs in
// F32. Kept as one helper so every site is identical and a site cannot silently
// be left un-adapted.
static ggml_tensor * mm3_dt_linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x,
                                   const MM3TrainLora & lo) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (!lo.on()) {
        return y;
    }
    ggml_tensor * ax = ggml_mul_mat(ctx, lo.a, x);                    // [rank, S]
    ggml_tensor * bx = ggml_mul_mat(ctx, lo.b, ax);                   // [out,  S]
    if (lo.out_mask) {
        bx = ggml_mul(ctx, bx, lo.out_mask);   // broadcast [out,1] over S
    }
    y = ggml_add(ctx, y, ggml_scale(ctx, bx, lo.scale));
    if (lo.m) {
        // DoRA: (W + s*BA) x * m / nrm, per output row. Callers add any bias
        // after this, so bias never enters the rescale.
        ggml_tensor * f = ggml_div(ctx, lo.m, lo.nrm);
        if (lo.keep) {
            f = ggml_add(ctx, ggml_mul(ctx, ggml_reshape_2d(ctx, f, f->ne[0], 1), lo.out_mask), lo.keep);
        }
        y = ggml_mul(ctx, y, f);
    }
    return y;
}

// ── LayerNorm, rebuilt from ops that HAVE a backward ────────────────────────
//
// **GGML_OP_NORM HAS NO BACKWARD.** ggml_compute_backward supports RMS_NORM but
// not NORM, so mm3_dit_ln() — which inference uses — cannot appear in a
// trainable graph. It aborts at ggml.c with "unsupported ggml op for backward
// pass: NORM" only once the backward is expanded, i.e. after the whole forward
// has built and looked fine.
//
// This is the real reason ACE's DiT trains and a naive MM3 mirror does not: ACE
// uses rms_norm, MM3 uses LayerNorm. The design note that said "LayerNorm, so
// ACE's rms_norm trap does not apply" was right about the trap and wrong about
// the consequence.
//
// The identity that avoids a vendored ggml patch:
//
//     LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps)
//     RMSNorm(y)   = y / sqrt(mean(y^2) + eps)
//     with y = x - mean(x),  mean(y^2) == var(x)
//     =>  LayerNorm(x) == RMSNorm(x - mean(x))
//
// SUB, MEAN and RMS_NORM all have backward, so the composite is differentiable.
// ggml_mean reduces over ne0 to [1, ne1], which broadcasts back through ggml_sub.
static ggml_tensor * mm3_dt_ln(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b,
                               float eps) {
    // ggml_repeat is NOT cosmetic. ggml_mean gives [1, S]; subtracting it from
    // [E, S] relies on broadcast, and SUB's backward does not reduce the
    // gradient back down, so it aborts on
    //   GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, grads[isrc1]))
    // Materialising the mean to [E, S] makes the shapes match and hands the
    // reduction to REPEAT's backward (REPEAT_BACK), which does it correctly.
    //
    // The norm's gamma/beta do NOT need this: they are frozen, so src1 never
    // needs a gradient and the broadcast is harmless there.
    // sum_rows/E, NOT ggml_mean. GGML_OP_MEAN's backward is
    //   ggml_add1_or_set(..., scale(grad, 1/ne0))
    // and ggml_add1 asserts its operand is a SCALAR — so MEAN is only
    // differentiable when it reduces to one value, never row-wise. SUM_ROWS'
    // backward is a plain ggml_repeat and is correct for our [E,S] -> [1,S].
    const int64_t E_       = x->ne[0];
    ggml_tensor * mean     = ggml_repeat(ctx, ggml_scale(ctx, ggml_sum_rows(ctx, x), 1.0f / (float) E_), x);
    ggml_tensor * centered = ggml_sub(ctx, x, mean);
    ggml_tensor * n        = ggml_rms_norm(ctx, centered, eps);
    n                      = ggml_mul(ctx, n, w);
    return ggml_add(ctx, n, b);
}

// ── One trainable block: mirrors mm3_dit_block() op for op ──────────────────

static ggml_tensor * mm3_dt_block(ggml_context * ctx, const MM3DitConfig & c, const MM3DitBlock & w,
                                  const MM3TrainBlockAdapters & ad, ggml_tensor * h,
                                  ggml_tensor * positions) {
    const int64_t E  = (int64_t) c.embedding_length;
    const int64_t D  = (int64_t) c.head_dim;
    const int64_t Nh = (int64_t) c.head_count;
    const int64_t FI = (int64_t) c.ff_inner;
    const int64_t S  = h->ne[1];

    // ── self-attention ──
    ggml_tensor * n   = mm3_dt_ln(ctx, h, w.attn_norm_w, w.attn_norm_b, c.layer_norm_eps);
    ggml_tensor * qkv = mm3_dt_linear(ctx, w.attn_qkv, n, ad.qkv);   // [3E, S]

    ggml_tensor * q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, E, S, qkv->nb[1], 0));
    ggml_tensor * k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, E, S, qkv->nb[1], (size_t) E * qkv->nb[0]));
    ggml_tensor * v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, E, S, qkv->nb[1], (size_t) (2 * E) * qkv->nb[0]));

    q = ggml_reshape_3d(ctx, q, D, Nh, S);
    k = ggml_reshape_3d(ctx, k, D, Nh, S);
    v = ggml_reshape_3d(ctx, v, D, Nh, S);

    // Partial NeoX RoPE over rope_dim of head_dim, exactly as inference.
    q = ggml_rope_ext(ctx, q, positions, NULL, (int) c.rope_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f,
                      0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, (int) c.rope_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f,
                      0.0f, 1.0f, 0.0f, 0.0f);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);  // [D, S, Nh]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // ALWAYS the manual path: flash-attn has no backward. The retained [S,S,Nh]
    // softmax output is the S^2 term in the VRAM model.
    ggml_tensor * attn = mm3_dit_attn_f32(ctx, q, k, v, 1.0f / sqrtf((float) D));
    attn               = ggml_reshape_2d(ctx, attn, Nh * D, S);
    h                  = ggml_add(ctx, h, mm3_dt_linear(ctx, w.attn_output, attn, ad.attn_out));

    // ── feed-forward: GLU, VALUE half first (mm3.dit.glu_order = value_gate) ──
    ggml_tensor * n2 = mm3_dt_ln(ctx, h, w.ffn_norm_w, w.ffn_norm_b, c.layer_norm_eps);
    ggml_tensor * f  = ggml_add(ctx, mm3_dt_linear(ctx, w.ffn_in_w, n2, ad.ff_in), w.ffn_in_b);

    ggml_tensor * val  = ggml_cont(ctx, ggml_view_2d(ctx, f, FI, S, f->nb[1], 0));
    ggml_tensor * gate = ggml_cont(ctx, ggml_view_2d(ctx, f, FI, S, f->nb[1], (size_t) FI * f->nb[0]));
    ggml_tensor * y    = ggml_mul(ctx, val, ggml_silu(ctx, gate));

    y = ggml_add(ctx, mm3_dt_linear(ctx, w.ffn_out_w, y, ad.ff_out), w.ffn_out_b);
    return ggml_add(ctx, h, y);
}

// ── Inputs ──────────────────────────────────────────────────────────────────

struct MM3TrainInputs {
    // LAYOUTS ARE THE INFERENCE GRAPH'S, NOT THE OBVIOUS ONES. mm3-dit-graph.h
    // takes latents as [S, 128] (torch memory order) and TRANSPOSES them
    // in-graph, while conditioning arrives already channel-first as [2048, S].
    // Getting either backwards asserts inside ggml_concat rather than training
    // something subtly wrong, which is the one mercy here.
    ggml_tensor * xt        = nullptr;  // [S, 128]   noised latents, torch order
    ggml_tensor * cond      = nullptr;  // [2048, S]  conditioning, channel-first
    // [fourier_dim, 1] host-computed Fourier features, cos-first — NOT the final
    // embedding. The two-layer MLP that turns these into [E,1] runs IN-GRAPH,
    // exactly as mm3_dit_build does. Passing a host-computed [E,1] instead would
    // duplicate that MLP on the host and silently diverge from inference the
    // moment either changed.
    ggml_tensor * fourier   = nullptr;
    ggml_tensor * positions = nullptr;  // [S+1] i32, 0..S  (time token occupies 0)
    // [E, S+1] of zeros, uploaded each micro-step. GGML_OP_CONCAT HAS NO
    // BACKWARD (docs/TRAINING.md flags the same thing for the ACE trainer:
    // "CONCAT has no backward (use ACC)"), and the time-token prepend sits
    // downstream of proj_in's LoRA, so it genuinely needs one. Two ggml_acc
    // writes into this canvas replace it; ACC has a proper backward.
    ggml_tensor * seq_zeros = nullptr;
    ggml_tensor * vtarget   = nullptr;  // [S, 128]   velocity target, torch order
};

// ── Full trainable forward: returns predicted velocity [128, S] ─────────────
//
// Mirrors mm3_dit_build(). Two conventions that are silent if broken and are
// therefore taken from the inference graph rather than re-derived:
//   * the time embedding is prepended as SEQUENCE token 0, so the latent frames
//     sit at rope positions 1..S;
//   * Fourier is cos-first, computed on the host in double by the caller and
//     handed in as `in.fourier`; the MLP over it runs here.
// ── Split points for gradient checkpointing ─────────────────────────────────
//
// mm3_dt_forward below is prologue -> block loop -> epilogue. Gradient
// checkpointing (mm3-dit-train-ckpt.h) needs to rebuild ONE RANGE of blocks at
// a time starting from a saved boundary, so the three parts are exposed
// separately here and mm3_dt_forward is written in terms of them. Keeping one
// definition of each part is the point: a checkpointed run that quietly
// disagreed with the monolithic one would be worse than no checkpointing.
//
// The boundary between segments is the block-loop hidden state h [E, S+1] --
// a single tensor, because MM3 has no cross-attention, no AdaLN and no
// per-layer conditioning. Everything else a segment needs (positions, and the
// conditioning that entered at proj_in) is either a graph input or already
// folded into h.
//
// NOTE: unlike ACE's trainer, MM3's PROLOGUE CONTAINS A TRAINED PARAMETER
// (ad.proj_in), so segment 0 must recompute the prologue inside its own
// gradient graph rather than reading a frozen boundary.
static ggml_tensor * mm3_dt_prologue(ggml_context * ctx, const MM3Model & m, const MM3TrainAdapters & ad,
                                     const MM3TrainInputs & in);
static ggml_tensor * mm3_dt_stack(ggml_context * ctx, const MM3Model & m, const MM3TrainAdapters & ad,
                                  const MM3TrainInputs & in, ggml_tensor * h, int lo, int hi);
static ggml_tensor * mm3_dt_epilogue(ggml_context * ctx, const MM3Model & m, const MM3TrainAdapters & ad,
                                     ggml_tensor * h, int64_t S);

static ggml_tensor * mm3_dt_forward(ggml_context * ctx, const MM3Model & m, const MM3TrainAdapters & ad,
                                    const MM3TrainInputs & in) {
    // ne[0], NOT ne[1]: xt arrives in torch order [S, 128], so the sequence
    // length is the FASTEST axis. Reading ne[1] silently yields 128 and the
    // whole graph builds at the wrong length — it fails later, in the loss,
    // looking like a target-shape problem. mm3_dit_build takes ne[0] too.
    const int64_t S = in.xt->ne[0];
    ggml_tensor * h = mm3_dt_prologue(ctx, m, ad, in);
    h               = mm3_dt_stack(ctx, m, ad, in, h, 0, (int) m.synth.dit.blk.size());
    return mm3_dt_epilogue(ctx, m, ad, h, S);
}

static ggml_tensor * mm3_dt_prologue(ggml_context * ctx, const MM3Model & m, const MM3TrainAdapters & ad,
                                     const MM3TrainInputs & in) {
    const MM3DitWeights & w = m.synth.dit;

    // [S, 128] -> [128, S], then cat(x, zeros_like(x), cond) -> [2304, S].
    ggml_tensor * x     = ggml_cont(ctx, ggml_transpose(ctx, in.xt));
    ggml_tensor * zeros = ggml_scale(ctx, x, 0.0f);
    ggml_tensor * full  = ggml_concat(ctx, ggml_concat(ctx, x, zeros, 0), in.cond, 0);

    // preprocess_conv is Conv1d k=1 with a residual — i.e. a matmul plus x.
    full = ggml_add(ctx, ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w.preprocess_conv,
                                                           w.preprocess_conv->ne[1], w.preprocess_conv->ne[2]),
                                      full),
                    full);

    ggml_tensor * h = mm3_dt_linear(ctx, w.proj_in, full, ad.proj_in);   // [E, S]

    // Timestep embedding: two-layer MLP with a SiLU between, over the host's
    // Fourier features. Mirrors mm3_dit_build.
    ggml_tensor * temb = ggml_add(ctx, ggml_mul_mat(ctx, w.time_embd_w[0], in.fourier), w.time_embd_b[0]);
    temb               = ggml_silu(ctx, temb);
    temb               = ggml_add(ctx, ggml_mul_mat(ctx, w.time_embd_w[1], temb), w.time_embd_b[1]);

    // Time token at sequence position 0 — this is what shifts every latent
    // frame's RoPE position by one. Built with ACC rather than CONCAT so it
    // is differentiable; see MM3TrainInputs::seq_zeros.
    {
        ggml_tensor * canvas = in.seq_zeros;
        canvas = ggml_acc(ctx, canvas, temb, canvas->nb[1], canvas->nb[2], canvas->nb[3], 0);
        canvas = ggml_acc(ctx, canvas, h,    canvas->nb[1], canvas->nb[2], canvas->nb[3],
                          canvas->nb[1]);   // one row in: latents start at position 1
        h = canvas;                                                      // [E, S+1]
    }
    return h;
}

// Blocks [lo, hi). The whole point of the split: a checkpoint segment rebuilds
// exactly this range, starting from a saved boundary instead of from block 0.
static ggml_tensor * mm3_dt_stack(ggml_context * ctx, const MM3Model & m, const MM3TrainAdapters & ad,
                                  const MM3TrainInputs & in, ggml_tensor * h, int lo, int hi) {
    const MM3DitConfig &  c = m.synth_cfg.dit;
    const MM3DitWeights & w = m.synth.dit;
    for (int i = lo; i < hi; i++) {
        h = mm3_dt_block(ctx, c, w.blk[(size_t) i], ad.blk[(size_t) i], h, in.positions);
    }
    return h;
}

static ggml_tensor * mm3_dt_epilogue(ggml_context * ctx, const MM3Model & m, const MM3TrainAdapters & ad,
                                     ggml_tensor * h, int64_t S) {
    const MM3DitConfig &  c = m.synth_cfg.dit;
    const MM3DitWeights & w = m.synth.dit;
    const int64_t         IC = (int64_t) c.in_channels;

    // Strip the time token, project out, postprocess residual.
    h = ggml_cont(ctx, ggml_view_2d(ctx, h, h->ne[0], S, h->nb[1], h->nb[1]));
    h = mm3_dt_linear(ctx, w.proj_out, h, ad.proj_out);                  // [IC, S]
    h = ggml_add(ctx, ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w.postprocess_conv,
                                                        w.postprocess_conv->ne[1], w.postprocess_conv->ne[2]),
                                   h),
                 h);
    GGML_ASSERT(h->ne[0] == IC);
    // Back to torch memory order, matching both inference and vtarget.
    return ggml_cont(ctx, ggml_transpose(ctx, h));   // [S, 128]
}

// ── Loss ────────────────────────────────────────────────────────────────────
//
// Plain rectified-flow MSE against a caller-supplied velocity target:
//
//   sigma ~ 1 - sigmoid(logit_normal(mean, std))
//   x_t   = sigma * noise + (1 - sigma) * x0
//   loss  = mean((v_pred - v_target)^2)
//
// THE SIGN OF v_target IS NOT SETTLED IN THIS FILE, deliberately. Take it from
// mm3-dit-graph.h's own Euler step (`x + (sigma_next - sigma) * v`, sigma RISING
// 0->1) rather than from SimpleTuner, whose flow_matching_target_direction = -1
// is stated against a different convention — and note that the DiT
// output-negation trap (`mm3.dit.output_negated`, ComfyUI negates, diffusers and
// this port do not) sits exactly here. Gate it before training anything: at high
// sigma a frozen-model prediction must point from noise toward data. If the loss
// is flat or rises, the sign is inverted.
//
// Deliberately NOT ported from the ACE trainer: the product loss and channel
// balancing. Those were fitted to ACE's DCAE latents; MM3's DAV latents are a
// different distribution (measured on real music: std ~2.15, absmax ~20).
static ggml_tensor * mm3_dt_loss(ggml_context * ctx, ggml_tensor * pred, ggml_tensor * target) {
    ggml_tensor * d = ggml_sub(ctx, pred, target);
    return ggml_scale(ctx, ggml_sum(ctx, ggml_sqr(ctx, d)), 1.0f / (float) ggml_nelements(d));
}
