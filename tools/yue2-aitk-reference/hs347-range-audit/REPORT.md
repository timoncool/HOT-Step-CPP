# Fix 2 range audit: verdict (HOTSTEP-347)

**Verdict.** Plain half staging is not eligible for any GEMM. The BF16 forward
GEMMs on the base weights are eligible only behind a per-tile exactness guard
that falls back to bfloat, and with that guard the output is bit-identical to
today's kernel. That guard would keep about 69% of base-forward tile steps on
half. No dX GEMM is eligible: nearly all of their dy values sit below the range
half holds exactly. No loss scaling or recipe change is proposed. The training
context can come from a mechanism the graphs already carry,
`ggml_cgraph::grads`, so no scope question is open for F.

Measured on MacServer (M2 Max, 30-core GPU, Apple8). Parent `9ade6553` (the
probe commit on `e56a0231`), ggml `cf9db4c0`. The run is the frozen
HOTSTEP-342 argv: seven fresh seed-42 updates over all six songs.

## What was measured

`range-audit.h` reads both operands of every Metal `mul_mm` GEMM in the
training run (32,751 GEMMs), just before the kernel runs, and classifies each
element the way a half tile would stage it:

- `src1` (activations or dy): F32, then BF16 by RNE (today's `(bfloat)` cast), then half.
- `src0` (weights): BF16, then half.

A value counts as changed if half cannot hold it exactly. That covers overflow
to inf, rounding inside the subnormal range, and flushing to zero. The audit
also counts tile steps: one threadgroup's 32-wide K block over a 64-row `src0`
tile and a 32-row `src1` tile, as `kernel_mul_mm` stages them. A step must fall
back if either tile holds a changed value.

The audit runs each graph in chunks with fusion off. All 28 frozen scalars
(AR CE, NAR MSE, KL and gradient norm over seven updates) are bit-identical
to the frozen warm baseline (`bits.py`: GATE PASS), so the audit saw the real
training numerics. Peak footprint was 16.09 GB.

## GPU behaviour (`metal-half-probe.swift`)

Each test ran with ggml-metal's compile options (the defaults) and again with
fast math off. Both gave the same results:

| Test | Result |
|---|---|
| BF16 to half, all 65,536 bit patterns | 0 mismatches against host RNE; all 2,814 half-subnormal results kept, none flushed |
| F32 to half, 1.57 M values (random, ties, the 2^-27 to 2^-14 range) | 0 mismatches, 0 flushed |
| F32 to bfloat, the same values, including 262,144 exact ties | 0 mismatches against RNE |
| simdgroup MAC, half tiles against bfloat tiles, values exact in both | bit-identical: 0 differing outputs in 1.18 M (K 256 and 2048; narrow, wide, half-subnormal and subnormal-only operands) |
| Same, alternating half and bfloat per 8-wide K block | bit-identical to all-bfloat |

So this GPU keeps half subnormals through conversion, threadgroup storage and
the matrix unit, and a half product equals the bfloat product whenever both
inputs are exact in half. The subnormal-only case would show a flush as zero
outputs, and none appeared. The mixed-block result is the basis for the
guard: per-step switching between half and bfloat does not change a single
bit. This was measured on Apple8 (M2) only.

## Ranges by GEMM group

FLOP share is of all audited `mul_mm` FLOPs over the seven updates. Full
per-family table: `summarize.py audit.json`.

| Group | FLOP share | src1 max abs | src1 changed | src0 changed | Overflow / non-finite | Tile steps that must fall back (any / weight tile / input tile) |
|---|---:|---:|---:|---:|---:|---|
| Base forward (AR+NAR qkv, o, gate/up, down) | 63.6% | 1.71e3 | 0.023% | 0.016% | 0 / 0 | 31.5% / 25.6% / 8.1% |
| Base dX | 20.1% | 3.3e-3 | 97.5% | (same weights) | 0 / 0 | 99.9% / 25.8% / 99.9% |
| Head logits forward (no-grads graph) | 8.8% | 46.8 | 4e-6% | 0.045% | 0 / 0 | 58.0% / 58.0% / 0.004% |
| Head dX (no-grads graph) | 4.4% | 2.0e-4 | 100% | 0.045% | 0 / 0 | 100% / 58.0% / 100% |
| LoRA forward (BF16 casts of A/B) | 1.9% | 1.71e3 | 0.032% | 1.64% | 0 / 0 | 75.3% / 72.3% / 7.8% |
| LoRA dX | 0.8% | 3.3e-3 | 97.3% | (same) | 0 / 0 | 97.4% / 72.0% / 95.0% |
| F32 companion forward and dX (not fix 2) | 0.3% | 142 | see below | see below | 0 / 0 | not applicable |

- **Overflow is not the problem.** The largest staged value is 1.71e3, at the
  `ffn_down` input, 38 times below 65,504. No GEMM saw an inf or NaN.
- **Underflow is.** In the dX GEMMs, 99% of `ffn_down` dy values are below
  2^-17. Below 2^-17, a BF16 value is exact in half only if its low bits
  happen to be zero; below 2^-24, it never is.
- **The forward rates are tiny but scattered.** About 0.02% of base weights
  are below 2^-17 (the smallest nonzero is 1.5e-11). Spread across
  2,048-element tiles, that leaves a changed value in 25.6% of weight tiles.
  Activation tiles fail on 8.1% of steps, and the two together on 31.5%.

## Per-GEMM eligibility

| GEMM | Forward | dX |
|---|---|---|
| Base q/k/v, o, gate/up, down (AR and NAR), `vae2llm` | Guarded only: about 69% of tile steps stay half | Not eligible |
| LoRA A/B (BF16 casts) | Guarded only, but 75% fall back on 1.9% of FLOPs: leave on BF16 | Not eligible |
| Head logits / head dX | Not reached: these run in no-grads graphs. Logits would fall back 58% anyway | Not eligible |
| F32 companion | Not part of fix 2 | Not part of fix 2 |

**Why unguarded half staging fails.** F is frozen as an exact-bits slice
(`metal-baseline/gates.json`, `exact_slices`). Unguarded, 31% of base-forward
tile steps would stage a value that half rounds or flushes. Each change is at
most 2^-25 (about 3e-8, the measured maximum) in absolute terms, but it lands
in the GEMM output. I
expect the loss bits to move, but this is unmeasured: no half kernel was run.
Under the frozen gate, that rules out unguarded staging. Relaxing the gate
after seeing results is not allowed (HOTSTEP-342).

**Why dX is out.** Making dy exact in half would need loss scaling, which is a
recipe change, and these values span 2^-82 to 2^-8. Even a scaled dy would not
pass an exact-bits gate. Not recommended.

## What F needs

1. **A bit-exact runtime guard, not a range argument.** For each 32-wide K
   block, the threadgroup checks whether every staged value it loads is exact
   in half. If so, it stages half and runs the half MAC; if not, it stages
   bfloat. The probe shows mixed steps are bit-identical to today's kernel, so
   this holds for any input, including future songs, adapters or larger
   activations. The finite sample here then only predicts speed, not safety.
   The exactness test for a BF16 value: finite, and either zero, or magnitude
   in 2^-14 to 65,504, or on the 2^-24 grid below that. Weights are frozen,
   so their per-tile mask can be computed once. Activations need the check in
   the kernel or a pre-pass.
2. **Training context: `ctx->gf->grads != NULL`.** `ggml_metal_op` already
   holds the graph. Every trainer block graph and endpoint graph is built with
   `grads = true` (`yue2-aitk-block-executor.h:225,333`,
   `yue2-aitk-endpoints.h`), and this covers all base-forward GEMMs above.
   No inference graph in `engine/src` is built that way (they use `false` or
   `ggml_new_graph`). Scheduler splits are graph views, which have
   `grads = NULL`. It is a property of the graph, not a process setting, so
   it cannot leak into inference. The LM, MM3 and DiT trainers on Metal also
   build `grads` graphs. With the guard their bits do not change either. An
   unguarded version would put them on ranges nobody audited.
3. **Device gate.** Only Apple8 (M2) was probed. M1 (Apple7) needs the same
   probe before it is enabled, and devices with the tensor API take the other
   `mul_mm` path. Unprobed devices keep today's kernel.
4. **Kill switch.** An environment variable that can only disable the path,
   like the fork's other `_DISABLE` switches.

**Expected benefit, an estimate.** The research measured 7.3 TFLOPS for the
BF16 kernel and 9.4 for the half-staged one on the gate/up shape, and BF16
GEMMs were about 17.7 s of the baseline update. Base forward is 63.6% of
those FLOPs, and 68.5% of its tile steps stay half. That gives at most about
17.7 × 0.636 × 0.685 × (1 − 7.3/9.4) ≈ 1.7 s, before the guard's own cost.
The guard adds a vote, and possibly a barrier, to every K step, which is
unmeasured. The 3% bar on G's 31.3 s is about 0.94 s. So F could clear it, but
only if the guard costs well under 0.8 s.

**Recommendation.** Run F time-boxed, with a microbenchmark gate before any
integration: the guarded kernel on the real gate/up and down shapes, with the
real fallback mix. Stop if the projection is under the 3% bar.

## Side finding (existing behaviour, out of scope)

On Metal, `GGML_PREC_F32` does not change the kernel. `kernel_mul_mm_f32_f32`
stages both operands as half, so every `mm_f32` GEMM is half-staged on Metal
today, with no BF16 rounding first. That covers the frozen F32 companion
factors. The audit saw no overflow there (max 142), but 63% of the companion
weight values and up to 99.98% of some of its inputs are rounded to half's
11-bit significand. CUDA runs these GEMMs in F32. This batch is gated Metal
against Metal, so it changes nothing here. It is a Metal-against-CUDA
precision difference for Lead to note.

## Limits

- One seven-update run on one seed. The guard makes F's correctness
  independent of this sample; the speed estimate is not.
- The tile counts assume `kernel_mul_mm`'s 64 × 32 × 32 blocking and 2-D
  GEMMs (every audited GEMM here is 2-D).
- The hook makes all of the trainer's graph launches synchronous and runs
  with fusion off. Neither changes values (the gate passed), but its timings
  mean nothing.

Raw logs: `.scratch/hs347/` on RobWorkPC (`audit.json`, `summary.md`,
`bits.txt`, `metal-half-probe.log`, `run-1.log`), and `.probe-logs/hs347/` on
the Mac mirror.
