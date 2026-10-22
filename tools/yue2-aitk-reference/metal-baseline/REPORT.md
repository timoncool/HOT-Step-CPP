# HOTSTEP-342 baseline evidence

Timing and logged scalar repeatability are measured. The warm metric is accepted.
The gates below are frozen for Reviewer sign-off before B starts. The approved
batch scope is Metal versus Metal; A4 recovery and replay are a separate backlog
job and do not gate A or this batch. No A4 pass is claimed.

## Revisions and protocol

Baseline parent `1ca695cc73cb5570d99052cd5f00830d01dab15e`; comparison parent
`7652d959ca1f17ddd33a26784548e404085d0a30`, captured as local master at task start.
Both use clean ggml `6d57c78a2c749cd7d9d58c2f7bd3178729171584`.
Later concurrent master commits, including attention reuse, are not measured here.
MacServer only. No production edits, Windows GPU work, pushes or releases.

Original HOTSTEP-338 argv and model/input hashes are archived. Each arm uses a
fresh seed-42 init and the same six-song dataset, full AR context, rank 64,
companion, optimizer and KL settings. Only output directory, step cap, save
cadence and preview pause change. Baseline ordinary seven-step runs passed three
times. Each revision then has three matched warm-song measurements: one full
seven-step sequence and two one-step repeats. Runs are grouped by revision,
not interleaved. See `results.json` for every timing and full-precision loss.

## Metric and timing

The ordinary first step includes cache-fill work. A diagnostic wrapper calls
the unchanged runtime twice in one process, first for one warmup update and then
with fresh model, adapter, optimizer and RNG state. Pipeline and frozen-teacher
caches survive. The warm baseline compiles zero additional pipelines and uses
2,528 BF16 GEMMs rather than 2,724. This is the matched warm-song metric required
by the parent plan, not an optimization saving. All seven warm-run losses and
gradient norms equal the ordinary baseline exactly.

Baseline warm seconds: **36.133547, 35.453133, 35.366556**. Median **35.453133 s**.
Spread is maximum minus minimum, **0.766990417 s**. Three times spread is
**2.300971251 s**. Reviewer accepted this metric and spread on HOTSTEP-342.

Captured-master warm seconds: **35.846569, 35.650270, 35.615361**. Median
**35.650270 s**, spread **0.231208291 s**, median delta **+0.197138 s**.
This is a control with unchanged production sources. No speed gain or regression
is established by a difference this small.

The ordinary baseline first steps were 41.180014, 40.716133 and 40.724110 s,
median 40.724110 s and spread 0.463880876 s. The historical 40.8 s figure
reproduces, but is not the selected warm-song metric.

The full measured warm sequences follow. Losses and gradient norms are identical
between revisions; full-precision values are in `results.json`.

| Update | Baseline warm s | Captured master warm s | AR CE | NAR MSE | KL | Gradient norm |
|---|---|---|---|---|---|---|
| 1 | 36.134 | 35.847 | 5.772480878 | 1.217101485 | 0 | 0.123408541 |
| 2 | 45.047 | 45.109 | 5.863287221 | 1.237981920 | 0.000442632 | 0.125031038 |
| 3 | 82.390 | 82.295 | 4.376926734 | 0.877772738 | 0.000431848 | 0.090175342 |
| 4 | 52.656 | 52.622 | 4.203646537 | 0.902860446 | 0.000670208 | 0.089750939 |
| 5 | 42.698 | 42.784 | 4.017974430 | 1.242763772 | 0.000763822 | 0.091958349 |
| 6 | 26.367 | 26.369 | 5.699079020 | 1.041303273 | 0.001736564 | 0.105855706 |
| 7 | 51.999 | 51.933 | 5.920067774 | 1.173319216 | 0.004319192 | 0.103834331 |

Peak process RSS in the full warm runs was 16.285 GiB for both revisions,
including retained caches. Ordinary baseline peak was 15.060 GiB. This is not
isolated GPU allocation telemetry.

## Frozen numeric gates

Every gate applies separately to AR CE, NAR MSE, KL and the logged gradient norm
at **each of updates 1 through 7**, using the archived recipe and full-precision
JSON values in `results.json`. Missing, duplicated or reordered updates fail.
Reject nonfinite values before comparison. Rounded console summaries, means and
final-update-only comparisons cannot establish a pass.

| Slice | Per-update scalar gate |
|---|---|
| B, split-K | `abs(candidate-reference) <= 1.1920928955078125e-7 + 7.62939453125e-6 * abs(reference)` |
| C, tiled copy | Bit-identical |
| D, rounding fusion | Bit-identical, for each enabled producer family |
| G, retained attention O/LSE | Bit-identical |
| F, conditional half staging | Bit-identical |

Exact means identical IEEE-754 binary64 bit patterns after parsing the original
full-precision JSON values, including signed zero. Do not cast to F32 or apply
an epsilon. This checks the logged scalars, not every underlying tensor.
The independent kernel/output/gradient probes in each slice remain required.

B's allowance is fixed before its implementation is measured. F32 spacing at
one is `epsilon = 2^-23 = 1.1920928955078125e-7`. Allocate one epsilon as the
absolute floor and 64 epsilons as the relative budget:
`atol = epsilon`, `rtol = 64 * epsilon = 2^-17`.
The baseline's measured noise is zero, so there is no added repeatability margin.
This choice allows small F32 regrouping differences while keeping a strict
training acceptance limit. The factor 64 is an engineering budget, not a
measured reduction depth or a theorem about training stability. Cancellation,
BF16 rounding boundaries and optimizer propagation prevent inferring an
end-to-end scalar error bound from F32 precision alone. B must separately justify
its kernel reduction error with its real shapes and gradient probes. A scalar
pass alone does not prove that an arbitrary reduction algorithm is correct.

At baseline update 1 this allows approximately 4.4160e-5 AR CE, 9.4050e-6 NAR
MSE, 1.1921e-7 KL and 1.0607e-6 gradient norm absolute difference. The zero KL
reference uses the absolute floor, avoiding division by zero. The formula,
not these rounded examples, decides pass/fail. B's fixed-order reduction must
also reproduce its own scalars bit-for-bit across fresh-state repeats.

Test each exact slice alone against the original baseline and incrementally
against its immediate predecessor with the earlier optimizations held fixed.
For cumulative comparisons against the original baseline, use the single B
budget when B is enabled, otherwise exact equality. This keeps a prior accepted
B difference from being attributed to C, D, G or F. No later slice earns another
error allowance. Apply the same limit at every update; never multiply it by the
update count, song length or repeat count. Report all signed differences and
per-field maxima. An unexplained accumulating drift remains a rejection even
inside the numeric envelope; investigate rather than raise a threshold.

G gets no tolerance because retaining the correct adapted-forward O/LSE should
reuse the same values as recomputation. Changes in precision, mask, layout,
lifetime or training state are failures. F gets no tolerance: eligibility must
preserve every staged BF16 value, and F32 accumulation order/results must remain
unchanged. A range audit alone does not prove that. If half staging changes
accumulation or hardware arithmetic and fails exact output/update comparisons,
leave that path disabled or return it for a scope decision; do not borrow B's
budget. D likewise must preserve the existing F32 result and BF16 rounding.

Historical A4 thresholds remain archived in `gates.json`, explicitly inactive
for this batch. The missing fixture has not been replayed. Scalar gradient norms
do not establish sampled tensor-gradient parity or audio quality.

## Verification and evidence

For this gate freeze, a Python JSON/`struct.pack('>d', value)` check verified
all 180 scalars across the 45 archived updates are finite and bit-identical
to their matching baseline update. It also checked the JSON constants against
the powers-of-two derivation. `git diff --check` and
`node tools/docs/check-docs.mjs` passed, with the same 11 screenshot warnings.
No new GPU run or product build was needed for these documentation/JSON changes.

The Mac mirror was synced from a local git bundle. `engine/build-mac.sh` passed,
then `cmake --build engine/build --target ace-train` passed for the comparison.
The warm wrapper compiled with C++17/O3 against the selected trainer libraries.
All nine probe invocations exited 0, covering 45 updates including warmups.
Python compilation and `node tools/docs/check-docs.mjs` passed; the latter reports
11 existing missing screenshot warnings. No package suites were run for this
probe-only change.

At completion no trainer process remained; the worker was idle and the existing
dev stack reported engine ready. That stack retained its baseline startup
revision; the standalone comparison trainer used the rebuilt captured revision.

Non-Metal proof: this slice changes only tooling/report files. The two measured
revisions differ only in Metal test device selection, with identical production
sources and the same ggml pin. CUDA/CPU/Vulkan runtime checks were not run.

Private raw logs, exact argv JSON, source worker log, binary hashes and input
hashes are preserved under `.scratch/hs342/` locally and `.probe-logs/hs342/` on
MacServer. The latter also retains all generated adapters. Public numeric evidence
is `results.json`; the frozen gates awaiting sign-off are `gates.json`. Public
argv substitutes private paths only; the archived argv retains the originals.
