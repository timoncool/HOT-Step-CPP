# Fix 2 guarded half staging: microbenchmark stop (HOTSTEP-348)

**Result.** The guarded kernel is bit-identical to today's BF16 kernel on every
input tried, but it is slower, not faster: 1.18x the BF16 kernel's time on the
base-forward shapes at the audit's fallback mix. The time box was a net saving
of at least 0.5 s on the A metric. This comes out at roughly +2 s, so F stops
here. No production code was committed and the fork is unchanged.

Measured on MacServer (M2 Max, 30-core GPU, Apple8, no other GPU work).
Parent `0000631f` (on `e56a0231` plus E's probes), ggml `cf9db4c0` with
`guarded-kernel.patch` applied to a bench-only copy.

## What was built

`guarded-kernel.patch` adds function constant `FC_MUL_MM + 8` to
`kernel_mul_mm` (simdgroup path). For each 32-wide K step, every thread loads
its 16 BF16 weights and 8 F32 activations, rounds the activations to BF16 with
the kernel's usual cast, and tests each value for exactness in half
(`(float)(half)v == v`). `simd_all` plus one byte per simdgroup in threadgroup
memory gives a threadgroup vote, read after the barrier the kernel already has.
The step then stages and multiplies in half if all values are exact, and in
bfloat otherwise. Accumulation stays F32 and the BF16_ROUND epilogue is shared.

`bench.mm` compiles the fork's embedded `mul_mm` source the way ggml-metal does
(`run-bench.sh` repeats the CMake concatenation) and times three kernels with
the same function constants: today's `kernel_mul_mm_bf16_f32`, the guarded one,
and `kernel_mul_mm_f16_f32` on the same weights stored as F16 (plain half
staging, the ceiling). Times are medians of five interleaved runs of ten
dispatches each, measured on the GPU timeline. Weights and activations are
synthetic. Normal-range values are exact in half. To match the audit's
per-family fallback rates, the bench plants a 1e-9 in the chosen fraction of
weight tiles (64 rows x 32 K) and activation tiles (32 rows x 32 K).

## Numbers (ms per GEMM, N = 7668 unless noted)

| Shape (K x M) | Mix | BF16 today | Guarded | F16 ceiling | Guarded / BF16 |
|---|---|---:|---:|---:|---:|
| gate_up 2048 x 12288 | audit | 52.53 | 61.49 | 41.11 | 1.171 |
| gate_up | all half | 52.53 | 59.30 | 41.11 | 1.129 |
| gate_up | all bfloat | 52.53 | 66.32 | 41.11 | 1.263 |
| qkv 2048 x 4096 | audit | 17.57 | 20.44 | 13.75 | 1.164 |
| down 6144 x 2048 | audit | 26.36 | 31.54 | 20.61 | 1.196 |
| o 2048 x 2048 | audit | 8.83 | 10.37 | 6.91 | 1.174 |
| gate_up, N = 7680 | audit | 52.27 | 58.86 | 40.90 | 1.126 |

Weighted by the audit's base-forward FLOP shares: guarded/BF16 **1.176**,
F16/BF16 0.783. Applied to the estimated ~11.3 s of base-forward BF16 GEMM per
update, that is about +2.0 s (guarded) against -2.5 s (the unguarded F16
ceiling, which the exact gate does not allow).

**Bits.** Guarded and today's output match byte for byte on every shape and mix
(audit, all half, all bfloat, and a mix that is half exact half-subnormals), both
with the plain store and with the BF16_ROUND epilogue: `BITS PASS`.

## Where the time goes

Two bench-only variants with the exactness test removed:

| Variant | Guarded / BF16 (audit shapes, weighted) |
|---|---:|
| Vote forced true: always half, no test (bits differ, as expected) | 1.038 |
| Vote forced false: always bfloat, no test | 1.059 |
| Real guard | 1.176 |

Staging in half without any test still runs slower than today's kernel. The
F16 kernel gets its 22% because its weights are already half in device memory
and its activations need one conversion. Here each step converts every BF16
weight to half (two conversions) and every activation F32 -> BF16 -> half
(three). Each staged value feeds only 32 or 64 multiply-adds, so those
conversions cost about what the half MAC saves. The vote and the two code paths
add about 4-6%, and the per-value test about another 12%.

So no in-kernel guard on BF16 weights can pay on this tiling. Making it cheaper
would not help: even a free test leaves the half path above 1.0.

## What could still pay (not built, estimate only)

Move the conversion and the test out of the GEMM. One pre-pass would write a
half copy of the weights and of the BF16-rounded activations, plus one flag per
tile. The GEMM would then read the flags and, on passing steps, load half
directly as the F16 kernel does. Each pre-pass reads its operand once, while the
GEMM reads it about N/32 or M/64 times. My estimate: roughly 0.69 x 0.83 +
0.31 x 1.06 = 0.90 of today's time at the audit mix, so about 1 s per update
after pre-pass cost. Unmeasured. It needs per-GEMM scratch of about 80 MB at
gate_up (half weights plus half activations), held through the dispatch, and a
scratch mechanism in ggml-metal. Using `get_alloc_size` would also grow
inference allocations, so this is a scope question, not a tweak.

## Reproduce

Apply `guarded-kernel.patch` to a copy of the fork's `src/` (never to
`engine/ggml`). Copy `ggml-common.h`, `ggml-metal/ggml-metal-impl.h` and
`kernels/{common.h,dequantize.h,mul_mm.metal}` to a directory with the same
layout, then from the Mac mirror root:

```sh
zsh tools/yue2-aitk-reference/hs348-half-stage/run-bench.sh <ggml-src-copy> .probe-logs/hs348/bench-N 10
```

Exit 0 means every bit check passed. Raw logs are in `.scratch/hs348/` on
RobWorkPC (`bench-1` is the guard, `bench-nochkT` and `bench-nochkF` are the
forced variants) and in `.probe-logs/hs348/` on the Mac mirror.
