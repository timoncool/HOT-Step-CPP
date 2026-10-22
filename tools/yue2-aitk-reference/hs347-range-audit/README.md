# Fix 2 range audit (HOTSTEP-347)

Decides whether a half-staged BF16 GEMM (fix 2, HOTSTEP-341) can be built for
Metal training, and for which GEMMs. Probe only: nothing here enters a
production build, and nothing here is a half-staged kernel. Results and the
verdict are in [REPORT.md](REPORT.md).

## Files

- `range-audit.h` is force-included (`-include`) when the probe binary
  compiles an unchanged copy of `engine/src/train/yue2-aitk-runtime.cpp`. It
  redirects the trainer's `ggml_backend_graph_compute(_async)` calls. With
  `HS347_AUDIT_OUT` set and a Metal backend, each graph runs in chunks that end
  before every `MUL_MAT` that Metal would send to `kernel_mul_mm`. The hook
  then reads that GEMM's real operands and classifies every element as a half
  tile would stage it: F32 to BF16 (RNE) to half for `src1`, BF16 to half for
  `src0`. It also counts tile steps (one threadgroup's 32-wide K block over a
  64-row `src0` tile and a 32-row `src1` tile) that hold any value half cannot
  represent exactly.
- `audit-main.cpp` runs one fresh training run and writes the audit JSON.
- `metal-half-probe.swift` checks this GPU's half and bfloat conversions and
  whether half and bfloat simdgroup products match bit for bit.
- `run-audit.sh` runs the seven frozen HOTSTEP-342 updates with fusion off,
  then checks the exact-bits gate against the frozen warm baseline.

## Running on the Mac mirror

Run this from the mirror root, with the Release build in `engine/build`. The
flags are the `yue2-aitk-kernels` target's own (`build.ninja`), plus
`-Iengine/ggml/src` for `ggml_cgraph`.

```sh
T=tools/yue2-aitk-reference/hs347-range-audit
c++ -O3 -DNDEBUG -std=gnu++17 -arch arm64 -DGGML_BACKEND_SHARED -DGGML_MAX_NAME=128 \
  -DGGML_SHARED -DGGML_USE_BLAS -DGGML_USE_CPU -DGGML_USE_METAL -DYUE2_AITK_CUDA_RUNTIME=0 \
  -D_FORTIFY_SOURCE=2 -Iengine/src/train -Iengine/src -Iengine/vendor/yyjson \
  -Iengine/ggml/include -Iengine/ggml/src -I$T -include $T/range-audit.h \
  engine/src/train/yue2-aitk-runtime.cpp $T/audit-main.cpp -o .probe-logs/hs347/range-audit \
  -Lengine/build -Wl,-rpath,$PWD/engine/build \
  -lyyjson -lggml -lggml-cpu -lggml-blas -lggml-metal -lggml-base
zsh $T/run-audit.sh .probe-logs/hs347/range-audit .probe-logs/hs347/audit7
swiftc -O $T/metal-half-probe.swift -o .probe-logs/hs347/metal-half-probe && .probe-logs/hs347/metal-half-probe
```

`run-audit.sh` reads A's archived argv (`.probe-logs/hs343/b-full-1/run-1.argv.json`)
and G's bit checker (`.probe-logs/hs346/bits.py`) from the mirror's probe logs.
Those logs hold dataset paths and stay out of the repository.

Chunked execution hides later consumers from ggml-metal's single-use fusion
check, so the script sets `GGML_METAL_FUSION_DISABLE=1` and
`GGML_METAL_NO_BF16_ROUND_PRODUCER=1`. HOTSTEP-345 showed the unfused path is
bit-identical, and the run must still match all 28 frozen scalars.
