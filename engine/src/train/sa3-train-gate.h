// sa3-train-gate.h: autodiff-safe SA3 DiT block with functional DoRA-rows adapters.
//
// The training counterpart of sa3_dit_build_layer (sa3-dit-ggml.h), for the
// official recipe (stable_audio_3 scripts/train_lora.py, adapter_type
// "dora-rows"): every adapted Linear computes
//   W' = m ⊙ (W + s·B·A) / ||W + s·B·A||_row ,  s = alpha / rank
// (stable_audio_3/models/lora/model.py dora_forward). The base W stays frozen.
//
// Same math as the inference block, three graph changes so ggml's backward
// can run through it on the pinned HOT-ggml without a fork change:
//   - DoRA is applied functionally: y = (W·x + s·B·(A·x)) ⊙ m / sqrt(nsq),
//     nsq = ||W||²_row + 2s·rowdot(W, BA) + s²·||BA||²_row. W' is never
//     materialised, so no full-size weight gradient exists anywhere.
//   - sigmoid has no backward in ggml: sigmoid(z) = exp(-softplus(-z)).
//   - concat has no backward in ggml: the local-conditioning left pad is an
//     acc into the data rows.
//
// The functional DoRA-rows norm is ported from betweentwomidnights/sa3.cpp
// (src/dit.h dit_lin, commit b17fb3d4), MIT License,
// Copyright (c) 2026 betweentwomidnights:
//   Permission is hereby granted, free of charge, to any person obtaining a
//   copy of this software and associated documentation files (the
//   "Software"), to deal in the Software without restriction, including
//   without limitation the rights to use, copy, modify, merge, publish,
//   distribute, sublicense, and/or sell copies of the Software, and to permit
//   persons to whom the Software is furnished to do so, subject to the
//   following conditions: The above copyright notice and this permission
//   notice shall be included in all copies or substantial portions of the
//   Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
//   NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//   USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "../sa3-dit-ggml.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <map>
#include <string>

// Adapted Linears of one block, in upstream module-name order.
enum Sa3TrainSite {
    SA3_SITE_QKV,   // self_attn.to_qkv
    SA3_SITE_OUT,   // self_attn.to_out
    SA3_SITE_CQ,    // cross_attn.to_q
    SA3_SITE_CKV,   // cross_attn.to_kv
    SA3_SITE_COUT,  // cross_attn.to_out
    SA3_SITE_FF0,   // ff.ff.0.proj
    SA3_SITE_FF2,   // ff.ff.2
    SA3_SITE_LOC0,  // to_local_embed.0
    SA3_SITE_LOC2,  // to_local_embed.2
    SA3_SITE_COUNT
};

static const char * const sa3_train_site_names[SA3_SITE_COUNT] = {
    "self_attn.to_qkv", "self_attn.to_out", "cross_attn.to_q", "cross_attn.to_kv", "cross_attn.to_out",
    "ff.ff.0.proj",     "ff.ff.2",          "to_local_embed.0", "to_local_embed.2",
};

// One DoRA-rows site. Layouts are ggml (ne0 innermost) over torch row-major:
// torch lora_A (r, in) is [in, r], lora_B (out, r) is [r, out].
struct Sa3DoraRows {
    ggml_tensor * a        = nullptr;  // [in, r]  trainable
    ggml_tensor * b        = nullptr;  // [r, out] trainable
    ggml_tensor * m        = nullptr;  // [out]    trainable magnitude
    ggml_tensor * base_nsq = nullptr;  // [out]    constant ||W_row||², host-computed once
    // Norm-path factors. Null = a/b. A test sets non-trainable copies here to
    // build the "norm treated as a constant" bug on purpose.
    ggml_tensor * a_norm   = nullptr;
    ggml_tensor * b_norm   = nullptr;
    float         scale    = 1.0f;     // alpha / rank
};

struct Sa3LayerAdapters {
    Sa3DoraRows site[SA3_SITE_COUNT];
};

// y = x·W'ᵀ for W' above, without forming W'. w [in, out] frozen, x [in, S].
static ggml_tensor * sa3_train_lin(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, const Sa3DoraRows * d) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);  // [out, S]
    if (!d || !d->a) {
        return y;
    }
    const float   s   = d->scale;
    const int64_t out = y->ne[0];
    ggml_tensor * ax  = ggml_mul_mat(ctx, d->a, x);   // [r, S]
    ggml_tensor * bax = ggml_mul_mat(ctx, d->b, ax);  // [out, S]
    y                 = ggml_add(ctx, y, ggml_scale(ctx, bax, s));

    // nsq[o] = ||W_o||² + 2s·Σ_r (W·Aᵀ)[o,r]·B[o,r] + s²·Σ_r B[o,r]·(A·Aᵀ·Bᵀ)[r,o]
    ggml_tensor * A   = d->a_norm ? d->a_norm : d->a;
    ggml_tensor * B   = d->b_norm ? d->b_norm : d->b;
    ggml_tensor * wa  = ggml_mul_mat(ctx, w, A);                                          // [out, r]
    ggml_tensor * t2  = ggml_sum_rows(ctx, ggml_mul(ctx, B, ggml_cont(ctx, ggml_transpose(ctx, wa))));  // [1, out]
    ggml_tensor * aa  = ggml_mul_mat(ctx, A, A);                                          // [r, r]
    ggml_tensor * t3  = ggml_sum_rows(ctx, ggml_mul(ctx, B, ggml_mul_mat(ctx, aa, B)));    // [1, out]
    ggml_tensor * nsq = ggml_add(ctx, d->base_nsq, ggml_scale(ctx, ggml_reshape_1d(ctx, t2, out), 2.0f * s));
    nsq               = ggml_add(ctx, nsq, ggml_scale(ctx, ggml_reshape_1d(ctx, t3, out), s * s));
    return ggml_mul(ctx, y, ggml_div(ctx, d->m, ggml_sqrt(ctx, nsq)));  // row scale, broadcast over S
}

// sigmoid(1 - g) = exp(-softplus(g - 1)): every op here has a ggml backward.
static ggml_tensor * sa3_train_gate_sigmoid(ggml_context * ctx, ggml_tensor * g) {
    return ggml_exp(ctx, ggml_neg(ctx, ggml_softplus(ctx, ggml_scale_bias(ctx, g, 1.0f, -1.0f))));
}

// One TransformerBlock (adaLN branch), arguments as sa3_dit_build_layer.
// ad may be null (frozen block). Inputs may be trainable or depend on
// trainable tensors; gradients flow to all of them.
static ggml_tensor * sa3_train_build_layer(ggml_context * ctx, const SA3DiTConfig & c, const SA3DiTLayer * ly,
                                           const Sa3LayerAdapters * ad, ggml_tensor * x, ggml_tensor * gcond,
                                           ggml_tensor * positions, ggml_tensor * vmask, ggml_tensor * cross,
                                           ggml_tensor * local_in, int64_t S, int64_t S_c) {
    const int     D     = c.dim_heads;
    const int     H     = c.n_heads;
    const int64_t dim   = c.dim;
    const int64_t n_mem = c.n_memory_tokens;
    const float   scale = 1.0f / sqrtf((float) D);
    auto site = [&](Sa3TrainSite s) { return ad ? &ad->site[s] : nullptr; };

    ggml_tensor * ssg   = ggml_add(ctx, ly->ssg, gcond);  // [6*dim, 1]
    auto          chunk = [&](int i) {
        return ggml_cont(ctx, ggml_view_1d(ctx, ssg, dim, (size_t) i * dim * sizeof(float)));
    };
    ggml_tensor * scale_self = chunk(0);
    ggml_tensor * shift_self = chunk(1);
    ggml_tensor * sg_self    = sa3_train_gate_sigmoid(ctx, chunk(2));
    ggml_tensor * scale_ff   = chunk(3);
    ggml_tensor * shift_ff   = chunk(4);
    ggml_tensor * sg_ff      = sa3_train_gate_sigmoid(ctx, chunk(5));

    auto mod = [&](ggml_tensor * t, ggml_tensor * s, ggml_tensor * b) {
        return ggml_add(ctx, ggml_add(ctx, t, ggml_mul(ctx, t, s)), b);
    };
    auto slice_heads = [&](ggml_tensor * fused, int i, int64_t n) {
        ggml_tensor * v = ggml_view_2d(ctx, fused, dim, n, fused->nb[1], (size_t) i * dim * sizeof(float));
        return ggml_reshape_3d(ctx, ggml_cont(ctx, v), D, H, n);
    };
    auto heads_mid = [&](ggml_tensor * t) { return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3)); };
    // [D, H, n] -> [n, D, H] in one cont: a cont of a cont'd transpose would
    // hand the inner cont a strided gradient, which ggml's CONT backward refuses.
    auto v_t = [&](ggml_tensor * t) { return ggml_cont(ctx, ggml_permute(ctx, t, 1, 2, 0, 3)); };
    // attn(q,k,v) - attn(qd,kd,v) as [dim, n]. Each map is brought to [D, H, n]
    // before the subtraction, so SUB's backward (a NEG on CUDA, which wants a
    // contiguous source) sees a contiguous gradient.
    auto diff_attn = [&](ggml_tensor * q, ggml_tensor * k, ggml_tensor * qd, ggml_tensor * kd, ggml_tensor * vt, int64_t n) {
        ggml_tensor * o1 = heads_mid(sa3_dit_attn_map(ctx, q, k, vt, scale));
        ggml_tensor * o2 = heads_mid(sa3_dit_attn_map(ctx, qd, kd, vt, scale));
        return ggml_reshape_2d(ctx, ggml_sub(ctx, o1, o2), dim, n);
    };

    // self-attention (adaLN-modulated, gated)
    {
        ggml_tensor * xn  = mod(sa3_dit_rms(ctx, x, ly->pre_norm_g, c.norm_eps), scale_self, shift_self);
        ggml_tensor * qkv = sa3_train_lin(ctx, ly->attn_qkv, xn, site(SA3_SITE_QKV));  // [5*dim, S]
        ggml_tensor * q   = sa3_dit_rms(ctx, slice_heads(qkv, 0, S), ly->attn_qn_g, c.qk_norm_eps);
        ggml_tensor * k   = sa3_dit_rms(ctx, slice_heads(qkv, 1, S), ly->attn_kn_g, c.qk_norm_eps);
        ggml_tensor * v   = slice_heads(qkv, 2, S);
        ggml_tensor * qd  = sa3_dit_rms(ctx, slice_heads(qkv, 3, S), ly->attn_qn_g, c.qk_norm_eps);
        ggml_tensor * kd  = sa3_dit_rms(ctx, slice_heads(qkv, 4, S), ly->attn_kn_g, c.qk_norm_eps);
        auto rope = [&](ggml_tensor * t) {
            return ggml_rope_ext(ctx, t, positions, NULL, c.rope_dims, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f,
                                 0.0f, 1.0f, 0.0f, 0.0f);
        };
        q  = heads_mid(rope(q));
        k  = heads_mid(rope(k));
        qd = heads_mid(rope(qd));
        kd = heads_mid(rope(kd));
        // padding: V-zeroing only, applied after the transpose so the CPU
        // backward's MUL never reads a gradient strided along ne0
        ggml_tensor * vt  = ggml_mul(ctx, v_t(v), ggml_reshape_3d(ctx, vmask, S, 1, 1));  // [S, D, H]
        ggml_tensor * out = diff_attn(q, k, qd, kd, vt, S);
        out = ggml_mul(ctx, sa3_train_lin(ctx, ly->attn_out, out, site(SA3_SITE_OUT)), sg_self);
        x   = ggml_add(ctx, x, out);
    }

    // cross-attention (ungated, unmodulated, no RoPE, no mask)
    {
        ggml_tensor * xn = sa3_dit_rms(ctx, x, ly->ca_norm_g, c.norm_eps);
        ggml_tensor * qq = sa3_train_lin(ctx, ly->cross_q, xn, site(SA3_SITE_CQ));       // [2*dim, S]
        ggml_tensor * kv = sa3_train_lin(ctx, ly->cross_kv, cross, site(SA3_SITE_CKV));  // [3*dim, S_c]
        ggml_tensor * q  = heads_mid(sa3_dit_rms(ctx, slice_heads(qq, 0, S), ly->cross_qn_g, c.qk_norm_eps));
        ggml_tensor * qd = heads_mid(sa3_dit_rms(ctx, slice_heads(qq, 1, S), ly->cross_qn_g, c.qk_norm_eps));
        ggml_tensor * k  = heads_mid(sa3_dit_rms(ctx, slice_heads(kv, 0, S_c), ly->cross_kn_g, c.qk_norm_eps));
        ggml_tensor * kd = heads_mid(sa3_dit_rms(ctx, slice_heads(kv, 1, S_c), ly->cross_kn_g, c.qk_norm_eps));
        ggml_tensor * vt = v_t(slice_heads(kv, 2, S_c));
        x = ggml_add(ctx, x, sa3_train_lin(ctx, ly->cross_out, diff_attn(q, k, qd, kd, vt, S), site(SA3_SITE_COUT)));
    }

    // local additive conditioning on the data tokens only (memory rows untouched)
    {
        ggml_tensor * le = ggml_add(ctx, sa3_train_lin(ctx, ly->loc0_w, local_in, site(SA3_SITE_LOC0)), ly->loc0_b);
        le = ggml_add(ctx, sa3_train_lin(ctx, ly->loc2_w, ggml_silu(ctx, le), site(SA3_SITE_LOC2)), ly->loc2_b);
        x  = ggml_acc(ctx, x, le, x->nb[1], x->nb[2], x->nb[3], (size_t) n_mem * x->nb[1]);
    }

    // feed-forward (adaLN-modulated, gated): GLU, x * SiLU(gate)
    {
        ggml_tensor * y     = mod(sa3_dit_rms(ctx, x, ly->ff_norm_g, c.norm_eps), scale_ff, shift_ff);
        ggml_tensor * p     = ggml_add(ctx, sa3_train_lin(ctx, ly->ff0_w, y, site(SA3_SITE_FF0)), ly->ff0_b);
        const int64_t inner = c.ff_inner;
        ggml_tensor * a     = ggml_cont(ctx, ggml_view_2d(ctx, p, inner, S, p->nb[1], 0));
        ggml_tensor * g     = ggml_cont(ctx, ggml_view_2d(ctx, p, inner, S, p->nb[1], (size_t) inner * sizeof(float)));
        ggml_tensor * h     = ggml_add(ctx, sa3_train_lin(ctx, ly->ff2_w, ggml_mul(ctx, a, ggml_silu(ctx, g)), site(SA3_SITE_FF2)),
                                       ly->ff2_b);
        x                   = ggml_add(ctx, x, ggml_mul(ctx, h, sg_ff));
    }
    return x;
}

// Backend placement receipt: every node of `gf` must run on `backend`, so a
// single-backend compute can't fall back to CPU. Returns the number of nodes
// the backend refuses; fills an op-name histogram of the whole graph.
static int sa3_train_audit_graph(ggml_backend_t backend, ggml_cgraph * gf, std::map<std::string, int> * ops,
                                 std::string * refused) {
    int bad = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * n    = ggml_graph_node(gf, i);
        std::string   name = ggml_op_desc(n);
        if (ops) {
            (*ops)[name]++;
        }
        if (!ggml_backend_supports_op(backend, n)) {
            bad++;
            if (refused && refused->find(name) == std::string::npos) {
                *refused += (refused->empty() ? "" : ",") + name;
            }
        }
    }
    return bad;
}
