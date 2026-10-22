# Fix 2 half pre-pass: result (HOTSTEP-353)

**Result.** The base-forward BF16 GEMMs of training graphs now run on a half pre-pass. It is
bit-identical: all 28 frozen scalars match over seven updates in every arm, and every arm
exports the same adapter hashes. It is faster, but by less than the target. The median paired
saving on the warm-song metric is 0.629 s (1.97%), and over the seven updates 7.35 s
(about 1.05 s per update, 2.5%). The candidate's warm median is 31.173 s, so sub-30 s is not
reached, and the warm saving is under the default 3% kernel-change bar.

Measured on MacServer (M2 Max, 30-core GPU, Apple8). Parent `d81dfe7b`, fork `d5831e92`
(on `cf9db4c0`).

## What was built

Three dispatches replace `kernel_mul_mm_bf16_f32` for an eligible GEMM:

1. `kernel_mul_mm_hp_cvt_w` writes a half copy of the BF16 weights and one flag per 8-wide K
   block of each 64-row tile, set when every value in it is exact in half.
2. `kernel_mul_mm_hp_cvt_a` writes `(bfloat) x` and `half((bfloat) x)` copies of the F32
   activations, with the same flags per 32-row tile. The bfloat cast is the GEMM's own.
3. `kernel_mul_mm_bf16_f32_hp` runs each 32-wide K step on the half MAC when all eight flags
   of the step (four per operand) are set, and on the bfloat MAC otherwise. Staging copies 16
   bits from whichever copy the step picks, so only the MAC depends on the type. Accumulation
   stays F32 and the store is the existing one, BF16_ROUND epilogue included.

HOTSTEP-347 showed that on Apple8 a half product equals the bfloat product when both inputs
are exact in half, also with the type mixed per K block, so the output matches today's kernel
bit for bit for any input.

Eligible: graphs built with grads, Apple8 without the tensor API, BF16 weights times F32
activations, 2-D, contiguous, `K % 32 == 0`, and a weight operand that reaches a frozen leaf
without a transpose (the audit's base-forward families). dX GEMMs, BF16 casts of trainable
LoRA factors, the head graphs (no grads) and inference keep today's kernel.
`GGML_METAL_HALF_PREPASS_DISABLE=1` turns it off.

The copies live in one private scratch buffer shared by all GEMMs and grown on demand, in
64 MB steps. Command buffers are ordered on it by Metal's hazard tracking; inside an encoder
the op puts a barrier before and after the pre-pass. Each use holds a reference until its
command buffer completes (a completed handler drops it), and the current buffer holds one more.
A replaced buffer is freed once no command buffer uses it, on the next scratch request or in
`ggml_metal_free`, never on Metal's completion thread. (Fork `d5831e92` leaked replaced
buffers; the review fix replaced that.) Weights are converted on every
GEMM, not cached: the base weights reach the GEMM as unnamed tensors, and nothing proves a
cache keyed on their address stays valid.

## Microbenchmark (step 1)

`bench.mm` with `hp.metal` appended to the fork's embedded `mul_mm` source, as HOTSTEP-348.
Synthetic operands with the audit's fallback mix per tile, ms per GEMM at N = 7668, median of
five interleaved runs of ten dispatches. `hp+both` includes both pre-passes. Weighted by the
audit's base-forward FLOP shares. Every variant passed the bit check against today's kernel on
all shapes and mixes, plain and with the BF16_ROUND epilogue.

| Variant | GEMM only | GEMM + both pre-passes |
|---|---:|---:|
| 1: branch per step, bfloat path reads the F32 activations | 0.911 | 0.931 |
| 2: runs of same-type steps | 0.962 | 0.982 |
| **4: 16-bit staging from a bfloat or half copy (built)** | **0.877** | **0.903** |
| 5: as 4, type picked per 8-wide K block | 0.983 | 1.009 |
| Half path only, branch compiled out (bits differ) | 0.788 | 0.808 |
| F16 weights (HOTSTEP-348 ceiling) | 0.782 | n/a |

Variant 4 per shape (audit mix, with pre-passes): gate_up 0.880, qkv 0.894, down 0.947, o 0.921; an earlier run of the same build gave 0.898 overall. Its all-bfloat
case runs at 0.98 to 1.01 of today's kernel, because it reads 2-byte activations. Per-8-block
decisions (variant 5) would raise the half share from about 74% to 93% of steps, but the
per-block branch costs more than that gains. Reading operands straight from device memory
with `simdgroup_load` was 1.53x and was dropped.

The projection was about 1.1 s per warm update. The measured saving is about half of that.
The difference was not isolated; candidates are the two barriers per GEMM, the pre-passes
running on smaller real N (audit mean about 5,300 rows against 7,668 in the bench), and the
lower occupancy of the hp kernel (832 threads per threadgroup against 1,024).

## Three paired runs

H's protocol (HOTSTEP-350): `metal-baseline/run.py` unchanged, fresh seed-42 state, one
identical warmup update, then seven measured updates, `--runs 1 --steps 7`. Each pair ran
predecessor then candidate, serially. The predecessor arm is H's `warm-final` binary pointed
at the `d81dfe7b` libraries (`libggml-metal` hash matches H's record). Nothing else trained or
rendered; `host-load.txt` has the snapshot.

| Pair | Predecessor warm s | Candidate warm s | Saving s | Predecessor seven-step s | Candidate seven-step s |
|---|---:|---:|---:|---:|---:|
| 1 | 31.765 | 31.146 | 0.619 | 297.114 | 291.765 |
| 2 | 32.259 | 31.173 | 1.086 | 296.460 | 287.905 |
| 3 | 32.032 | 31.402 | 0.629 | 295.812 | 288.465 |

Warm median/spread: predecessor 32.032/0.494 s, candidate 31.173/0.256 s. Today's
predecessor median is 0.80 s above H's 31.228 s for the same build, so compare within this
session. Against H's figures the candidate is 31.173 s versus 31.228 s (H's final) and
35.491 s (H's baseline); those were measured on another day.

| Update | Predecessor median s | Candidate median s |
|---|---:|---:|
| 1 | 32.032 | 31.173 |
| 2 | 39.441 | 38.263 |
| 3 | 72.689 | 71.217 |
| 4 | 45.844 | 44.674 |
| 5 | 37.224 | 36.237 |
| 6 | 23.034 | 22.365 |
| 7 | 45.685 | 44.366 |

## Bits and controls

Every arm matches the frozen warm baseline (`.probe-logs/hs342/warm-baseline-full`) in all 28
scalars (AR CE, NAR MSE, KL, gradient norm over seven updates), byte for byte, and all of
them export the same combined, AR and NAR adapter hashes (`compare.py`: GATE PASS over 7 arms):
the three predecessor arms, the three candidate arms, and a switched-off control (candidate
binary with `GGML_METAL_HALF_PREPASS_DISABLE=1`, run by `runarm.py` because `run.py` refuses
`GGML_` variables). The candidate logs show the hp pipelines compiling; the predecessor and
switched-off logs do not. An earlier candidate trial on the uncommitted tree also matched.

After the scratch-lifetime fix, two more arms matched the same way (`fix-results.json`):

- the fixed candidate (31.278 s warm, 291.764 s over seven updates, peak footprint 16.499 GB);
- the isolated control: the original baseline fork `6d57c78a` plus this change alone, on the
  same trainer (34.863 s warm). That fork has no BF16_ROUND epilogue, so the port passes
  `round_epilogue = false` and declares the function constant itself
  (`.probe-logs/hs353/iso-applied.patch` on the mirror).

Both logs show every scratch buffer freed by teardown: the trainer grew it five times, to
384 MB, and freed three of them mid-run.

`scratch-life.cpp` checks the lifetime directly. It launches 25 base-forward GEMMs in graphs
with grads, without synchronizing, with N rising from 2,048 to 24,576, then falling, then
partial tiles. Each output matches the same GEMM in a graph without grads (today's kernel) bit
for bit. The scratch was replaced 3 times while earlier command buffers were in flight, 2 were
freed during the run, and all 4 by `ggml_metal_free`: RESULT PASS.

## Memory

`/usr/bin/time -l` over the whole process (load, warmup, seven updates, export), decimal GB.

| Arm | Max RSS (pairs 1-3) | Peak footprint (pairs 1-3) |
|---|---|---|
| Predecessor | 17.430 / 17.382 / 17.418 | 15.562 / 16.159 / 15.890 |
| Candidate | 17.379 / 17.380 / 17.416 | 16.416 / 16.417 / 16.346 |

Peak footprint is 0.53 GB higher at the median (0.26 to 0.85 GB by pair): the scratch, whose
largest need here is the down projection (about 25 MB of weights plus 4 bytes per activation
element), plus the smaller buffers d5831e92 leaked as it grew. RSS is unchanged. With the
lifetime fix, the peak is the largest scratch (384 MB here); the one fixed-candidate run's
footprint was 16.499 GB.

## Non-Metal proof

`git -C engine/ggml diff --stat cf9db4c0..hot-step-neutral`: only files under `src/ggml-metal/`
(`ggml-metal-ops.cpp`/`.h`, `ggml-metal-device.h`/`.m`, `ggml-metal-context.m`, `kernels/mul_mm.metal`). The kernels are compiled only without the tensor
API and with BF16; no shared op, ABI, graph or CPU, CUDA or Vulkan code changes.

## Reproduce

Microbenchmark, from the Mac mirror root (exit 0 means every bit check passed;
`HP_VARIANT=1|2|4|5` and `HP_FORCE=1|2` select the variants above):

```sh
HP_VARIANT=4 zsh tools/yue2-aitk-reference/hs353-half-prepass/run-bench.sh engine/ggml/src .probe-logs/hs353/bench-N 10
```

Pairs: build the warm binary as in `metal-baseline/README.md`, then
`python3 .probe-logs/hs353/batch.py` and `python3 tools/yue2-aitk-reference/hs353-half-prepass/compare.py <arm dirs>`.
Raw logs are under `.probe-logs/hs353/` on the Mac mirror; `results.json`, the bench logs and
the scripts are copied to `.probe-logs/hs353/` on the development checkout. Training logs stay
private because they contain dataset names.
