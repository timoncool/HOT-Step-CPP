#pragma once

// BF16 activation storage (on by default on CUDA, the only backend with the
// BF16-source kernels, except in strict F32, the reference;
// YUE2_AITK_BF16_STORAGE=0 is the rollback). The block
// graph is built and differentiated exactly as before, in F32. This pass then
// stores every BF16_ROUND output as BF16, along with the views and conts that
// only reread it. Every other node keeps its F32 output, so each op does the
// same F32 maths on the same values; only the bytes it reads shrink. Gradient
// sums stay F32 because ADD/ACC outputs are never retyped.
//
// Construction-time byte metadata (nb, view offsets) follows each retyped
// tensor's element size. Tensors read back or handed on as F32 (block output,
// K/V captures, input gradient) are passed as `keep_f32` and keep their root.

#include "ggml.h"
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace yue2_aitk_bf16_storage {

inline bool enabled() {
    static const bool on = [] { const char * v = std::getenv("YUE2_AITK_BF16_STORAGE"); return !(v && *v == '0'); }();
    return on;
}

// YUE2_AITK_BF16_DEFER=0 retypes deferrable rounds too (for A/B timing).
inline bool defer_rounds() {
    static const bool on = [] { const char * v = std::getenv("YUE2_AITK_BF16_DEFER"); return !(v && *v == '0'); }();
    return on;
}

// CUDA writes a BF16 GEMM's result straight into its BF16 round store, so such
// a round is stored rather than deferred. GGML_CUDA_NO_GEMM_BF16_STORE=1 turns
// both off.
inline bool gemm_store() {
    static const bool on = std::getenv("GGML_CUDA_NO_GEMM_BF16_STORE") == nullptr;
    return on;
}

// YUE2_AITK_ATTENTION_COPY=1 keeps flash attention's BF16 q/k/v copies.
inline bool attention_views() {
    static const bool on = std::getenv("YUE2_AITK_ATTENTION_COPY") == nullptr;
    return on;
}

inline const ggml_tensor * owner(const ggml_tensor * t) { return t->view_src ? t->view_src : t; }

inline void retype(ggml_tensor * t, ggml_type type) {
    const size_t from = ggml_type_size(t->type), to = ggml_type_size(type);
    for (int k = 0; k < GGML_MAX_DIMS; ++k) t->nb[k] = t->nb[k] / from * to;
    t->view_offs = t->view_offs / from * to;
    if (t->op == GGML_OP_VIEW) {
        size_t offset; std::memcpy(&offset, t->op_params, sizeof(offset));
        offset = offset / from * to; std::memcpy(t->op_params, &offset, sizeof(offset));
    }
    t->type = type;
}

// Can `reader` take a BF16 tensor as source `idx`? Only ops with a BF16-input
// CUDA path whose F32 result is bit-identical to reading the F32 copy.
inline bool reads_bf16(const ggml_tensor * reader, int idx) {
    switch (reader->op) {
        case GGML_OP_ADD: case GGML_OP_SUB: case GGML_OP_MUL:
        case GGML_OP_SCALE: case GGML_OP_RMS_NORM: case GGML_OP_RMS_NORM_BACK: case GGML_OP_SILU_BACK:
        case GGML_OP_ACC: case GGML_OP_OUT_PROD: case GGML_OP_CPY: case GGML_OP_CONT: case GGML_OP_DUP:
            return true;
        case GGML_OP_REPEAT: return idx == 0;
        // The trainer's RoPE is plain NeoX, which CUDA reads from BF16.
        case GGML_OP_ROPE: return idx == 0;
        case GGML_OP_SET: return idx == 1;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(reader)) {
                case GGML_UNARY_OP_SILU: case GGML_UNARY_OP_NEG: case GGML_UNARY_OP_BF16_ROUND: return true;
                default: return false;
            }
        case GGML_OP_MUL_MAT:
            // cuBLAS takes the BF16 src1 as is for a BF16 weight and widens it
            // for an F32 one; quantized weights would leave their MMQ/MMVQ path.
            return idx == 1 && (reader->src[0]->type == GGML_TYPE_BF16 || reader->src[0]->type == GGML_TYPE_F32);
        default:
            return false;
    }
}

// Returns the number of tensors retyped.
inline int apply(ggml_cgraph * graph, const std::vector<const ggml_tensor *> & keep_f32) {
    std::unordered_set<const ggml_tensor *> keep;
    for (const ggml_tensor * t : keep_f32) if (t) keep.insert(owner(t));
    const int n = ggml_graph_n_nodes(graph);
    // An owner stays F32 if any non-view node reads it (or a view of it) and
    // has no BF16 path for that source.
    // An in-place node other than a pure view or a pass-through round writes
    // its owner's bytes: leave that owner alone too.
    std::unordered_set<const ggml_tensor *> blocked;
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * t = ggml_graph_node(graph, i);
        const bool pure_view = t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE ||
            t->op == GGML_OP_TRANSPOSE || (t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_BF16_ROUND && t->view_src);
        if (pure_view) continue;
        if (t->view_src) blocked.insert(t->view_src);
        for (int j = 0; j < GGML_MAX_SRC; ++j)
            if (t->src[j] && !reads_bf16(t, j)) blocked.insert(owner(t->src[j]));
    }
    // A round the CUDA backend can defer stays F32: its only reader is an
    // ADD/SUB/MUL that rounds it on load, and its source, which nothing else
    // reads, is not an ADD/SUB/MUL (those store BF16 directly). The GEMM
    // output then makes one F32 round trip instead of F32 + round + BF16.
    std::unordered_map<const ggml_tensor *, int> uses;
    std::unordered_map<const ggml_tensor *, const ggml_tensor *> reader;
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * t = ggml_graph_node(graph, i);
        for (int j = 0; j < GGML_MAX_SRC; ++j) if (t->src[j]) { ++uses[t->src[j]]; reader[t->src[j]] = t; }
    }
    auto deferrable = [&](const ggml_tensor * r) {
        const ggml_tensor * x = r->src[0];
        if (x->view_src || x->op == GGML_OP_NONE || x->op == GGML_OP_ADD || x->op == GGML_OP_SUB || x->op == GGML_OP_MUL) return false;
        if (gemm_store() && x->op == GGML_OP_MUL_MAT && x->src[0]->type == GGML_TYPE_BF16 && x->op_params[0] != GGML_PREC_F32) return false;
        if (uses[x] != 1 || uses[r] != 1) return false;
        const ggml_tensor * u = reader[r];
        return u->op == GGML_OP_ADD || u->op == GGML_OP_SUB || u->op == GGML_OP_MUL;
    };
    std::unordered_set<const ggml_tensor *> bf16;
    int changed = 0;
    for (int i = 0; i < n; ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        if (t->type != GGML_TYPE_F32) continue;
        bool to_bf16 = false;
        if (t->view_src) {
            to_bf16 = bf16.count(t->view_src) != 0;
        } else if (keep.count(t) || blocked.count(t) || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            continue;
        } else if (t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_BF16_ROUND) {
            to_bf16 = !(defer_rounds() && deferrable(t));
        } else if (t->op == GGML_OP_CONT || t->op == GGML_OP_DUP) {
            to_bf16 = t->src[0]->type == GGML_TYPE_BF16;
        }
        if (!to_bf16) continue;
        // Every element is BF16-exact, so the cast loses nothing; whatever the
        // offsets were in F32 bytes, they are halved for BF16.
        GGML_ASSERT(t->view_offs % 4 == 0);
        retype(t, GGML_TYPE_BF16);
        if (!t->view_src) bf16.insert(t);
        ++changed;
    }
    // A cast of a now-BF16 tensor into BF16 that only flash attention reads
    // is the same bytes in another layout, and the BF16 attention kernels
    // take strided q/k/v: make it a view instead of a copy.
    if (attention_views()) {
        std::unordered_set<const ggml_tensor *> other_reader;
        for (int i = 0; i < n; ++i) {
            const ggml_tensor * t = ggml_graph_node(graph, i);
            if (t->op == GGML_OP_FLASH_ATTN_TRAIN || t->op == GGML_OP_FLASH_ATTN_TRAIN_BACK) continue;
            for (int j = 0; j < GGML_MAX_SRC; ++j) if (t->src[j] && t->src[j] != t) other_reader.insert(t->src[j]);
        }
        for (int i = 0; i < n; ++i) {
            ggml_tensor * t = ggml_graph_node(graph, i);
            ggml_tensor * s = t->src[0];
            if (t->op != GGML_OP_CPY || t->src[1] != t || t->type != GGML_TYPE_BF16 || !s || s->type != GGML_TYPE_BF16 ||
                !uses.count(t) || other_reader.count(t) || (t->flags & GGML_TENSOR_FLAG_OUTPUT) || t->view_src ||
                !ggml_are_same_shape(t, s) || s->nb[0] != ggml_type_size(GGML_TYPE_BF16)) continue;
            const size_t offset = s->view_offs;
            t->op = GGML_OP_VIEW;
            t->src[1] = nullptr;
            t->view_src = const_cast<ggml_tensor *>(owner(s));
            t->view_offs = offset;
            std::memcpy(t->op_params, &offset, sizeof(offset));
            for (int k = 0; k < GGML_MAX_DIMS; ++k) t->nb[k] = s->nb[k];
            ++changed;
        }
    }
    return changed;
}

} // namespace yue2_aitk_bf16_storage
