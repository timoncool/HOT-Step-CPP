#pragma once
// dora-common.h — the DoRA column-norm graph shared by every detached refresh.
// HOT-Step file (does not exist upstream), 2026-10-20.
//
// DoRA is  W' = m * (W + s*BA) / ||W + s*BA||_col  (Liu et al. 2024). The
// denominator is DETACHED (paper section 4.3, PEFT's dora.py): it is refreshed
// from frozen A/B and never differentiated, so it is a plain forward graph.
//
// Orientation: ggml stores the base as [in, out], so the paper's per-column
// norm is a reduction over ne0, one value per output channel. Bias is not part
// of W and never enters the norm.
//
// Casts stay with the caller on purpose. The LM trainer passes its F32 B as-is
// and casts the base only when it is not already F32; the MM3 runtime casts
// both unconditionally. The two are numerically identical but not the same
// graph, and this helper exists to share the maths, not to unify them.
//
// Zero norm is passed through: a column of W + s*BA that is all zero gives
// exactly 0 here, and the m / nrm rescale downstream is left to the caller.

#include "ggml.h"

// nrm[1, out] = ||w_f32 + scale * A.B||_col.
//   w_f32  [in, out] F32 base
//   a      [in, r]   PEFT lora_A layout
//   b_f32  [r, out]  PEFT lora_B layout, F32
static inline ggml_tensor * dora_col_norm(ggml_context * ctx,
                                          ggml_tensor *  w_f32,
                                          ggml_tensor *  a,
                                          ggml_tensor *  b_f32,
                                          float          scale) {
    // delta[in, out] = A[in, r] . B[r, out]: mul_mat contracts ne0, so the left
    // operand is A transposed to [r, in].
    ggml_tensor * delta = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, a)), b_f32);
    // F16 inputs must still accumulate in F32, including on Vulkan.
    ggml_mul_mat_set_prec(delta, GGML_PREC_F32);
    ggml_tensor * wd    = ggml_add(ctx, w_f32, ggml_scale(ctx, delta, scale));
    return ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, wd)));
}
