# Metal training baseline probe

This probe replays the archived HOTSTEP-338 worker command for HOTSTEP-342.
It changes only output directory, step count, save cadence and preview pause.
The six-song order, seed 42, model, companion and training settings stay fixed.
Every invocation uses a new output directory and fresh adapter/optimizer state.

Keep original worker logs, argv and input manifests under a private repo-owned
probe directory. They contain dataset names and must not be committed. `run.py`
records SHA-256 hashes of the model, companion, prepared files and executable.
It requires an idle worker and generation queue and refuses inherited GGML/YuE2
environment overrides. Run it in the foreground and collect its exit code.

```sh
python3 run.py original-hs338.log .probe-logs/hs342/baseline --runs 3
python3 summarize.py .probe-logs/hs342/baseline --output summary.json
```

The baseline parent is `1ca695cc73cb5570d99052cd5f00830d01dab15e`, with ggml
`6d57c78a2c749cd7d9d58c2f7bd3178729171584`. The comparison parent captured at
task start is `7652d959ca1f17ddd33a26784548e404085d0a30`, with the same ggml.
Their production sources are identical; only Metal test device selection differs.
Later concurrent changes to master are outside this pair.

## Warm cache diagnostic

`warm.cpp` calls the existing runtime twice in one process: one warmup update,
then the requested measurement. Each call constructs a new model, training
state, optimizer, sampler and song order. The Metal device's library and pipeline
cache survive between calls, as does the existing frozen-teacher hidden-state
cache in `yue2-aitk-joint-step.h`. Thus this is a warm-song measurement, including
the existing teacher cache hit, not just a pipeline compilation experiment.
Never change model or inputs between the two calls: the teacher cache keys on
input token IDs. No trainer implementation is changed or timed hooks
inserted. Require identical step-one losses and gradient norm to the ordinary
executable before using this probe's warm metric.

Compile on the Mac against the exact selected revision's existing Release build:

```sh
c++ -std=c++17 -O3 -DNDEBUG -arch arm64 -Iengine/src \
  tools/yue2-aitk-reference/metal-baseline/warm.cpp -o .probe-logs/hs342/warm \
  -Lengine/build -Wl,-rpath,"$PWD/engine/build" \
  -lyue2-aitk-kernels -lyyjson -lggml -lggml-cpu -lggml-blas -lggml-metal -lggml-base
python3 run.py original-hs338.log .probe-logs/hs342/warm-baseline \
  --binary .probe-logs/hs342/warm --runs 3
```

Use the first update after `HS342_PHASE measured_fresh_state` as the warm-song
metric. Preserve all seven updates separately. The sum or mean over different
songs is not the optimization metric. Spread is maximum minus minimum of the
matched metric in seconds; the plan's repeat trigger is three times that spread.
Process RSS includes warmup; it is not isolated GPU allocation telemetry.

## Numeric gates

The approved batch compares Metal against Metal. A4 recovery and replay are a
separate backlog job; the archived A4 limits are not active batch gates.

`gates.json` freezes per-update AR CE, NAR MSE, KL and gradient-norm limits.
C, D, G and F require exact scalar bits. B uses
`abs(candidate-reference) <= 2^-23 + 2^-17 * abs(reference)`.
Apply every gate to all seven updates and reject nonfinite or missing values.
See `REPORT.md` for the F32 budget derivation, isolated/incremental controls,
cumulative budget and why G/F have no tolerance. Kernel and tensor probes remain
required; scalar norms do not prove tensor equality. Reviewer must sign off
before B starts. Do not relax limits after observing an optimization's results.
