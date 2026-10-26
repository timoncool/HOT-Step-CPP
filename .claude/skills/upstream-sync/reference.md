# upstream-sync — reference

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

Deep detail supporting [SKILL.md](SKILL.md). Everything here was verified against the working tree on 2026-10-23.

## 1. Linker sentinel — exact mechanism

The silent-clobber defense is a symbol-dependency chain:

`engine/src/hot-step-sampler.h:1540-1552` (comment at 1546 names the expected error):

```cpp
//     "unresolved external symbol hotstep_sampler_linked_"
#if defined(_MSC_VER)
__declspec(selectany) int hotstep_sampler_linked_ = 1;
#else
__attribute__((weak)) int hotstep_sampler_linked_ = 1;
#endif
```

`engine/tools/hot-step-server.cpp:47-53`:

```cpp
// hot-step-sampler.h defines hotstep_sampler_linked_ with external linkage.
...
extern int hotstep_sampler_linked_;
static volatile int * _hotstep_guard_ = &hotstep_sampler_linked_;
```

Chain: `pipeline-synth-ops.cpp` includes `hot-step-sampler.h` → the symbol is compiled into `acestep-core.lib` → `hot-step-server.cpp` (the `ace-server` executable) references it at link time. If the include in `pipeline-synth-ops.cpp` reverts to `dit-sampler.h`, the symbol is never emitted and the link of `ace-server` fails with exactly `unresolved external symbol hotstep_sampler_linked_`.

Limits of the sentinel: it only guards the `pipeline-synth-ops.cpp` → `hot-step-sampler.h` edge. If `hot-step-sampler.h` itself is damaged/replaced (sentinel definition lost), both the definition and the failure signal disappear together — that is why `verify-hooks.ps1` Hook 6 greps for the literal string `hotstep_sampler_linked_` inside `hot-step-sampler.h`.

## 2. Runtime proof: log-line comparison

Both samplers print a `[DiT] Solver:` line, so its mere presence proves nothing. Formats differ:

| Sampler | Line | Source |
|---|---|---|
| HOT-Step (active) | `[DiT] Solver: <name> (<scheduler>, <N> NFE/step, order <N>)` | `engine/src/hot-step-sampler.h:712` |
| HOT-Step (active) | `[DiT] Guidance: <display> (<name>)...` | `engine/src/hot-step-sampler.h:734` |
| Upstream vanilla | `[DiT] Solver: <name> (<N> NFE/step, order <N>)` — no scheduler field | upstream `src/dit-sampler.h:404` (the fork's uncompiled copy is identical) |
| Upstream vanilla | *(no Guidance line at all)* | — |

So: **`[DiT] Guidance:` present = HOT-Step sampler is live.** Check the newest session folder: `<data root>/logs/YYYY-MM-DD_HH-MM-SS/ace_engine.log`.

(NFE = number of function evaluations, i.e. DiT forward passes per solver step.)

## 3. verify-hooks.ps1 — check-by-check

Source: `engine/verify-hooks.ps1` (305 lines). Regex-greps plus one git check, exit 0/1. Usage: `powershell -File engine\verify-hooks.ps1` (Windows) or `pwsh -NoProfile -File engine/verify-hooks.ps1` (macOS). `engine/build.cmd` runs it before every compile (`build.cmd:94-109`) and `engine/build-mac.sh` before CMake.

| # | File | Requires | Notes |
|---|---|---|---|
| 1 | `engine/src/pipeline-synth-ops.cpp` | `#include "hot-step-sampler.h"` | Explicit FAIL message if `dit-sampler.h` found instead; WARN+error if neither |
| 2 | `engine/src/model-store.h` | `#include "hot-step-params.h"` | |
| 3 | `engine/src/dit.h` | `#include "adapter-merge.h"` and `#include "adapter-runtime.h"` | Upstream's `dit.h` has the `adapter-merge.h` include too, so only the second half catches a direct copy. Does NOT check `convrot.h`, `hot-step-build-flags.h`, `concept-steer.h`, `config-json.h` or `weight-source.h` — a merge must still keep them or the build breaks |
| 4 | `engine/tools/hot-step-server.cpp` | `#include "hot-step-params.h"` | HOT-Step-zone file, can't be sync-clobbered, checked anyway |
| 4-family | `engine/tools/hot-step-server.cpp` | `minimax/mm3-server.h` + `mm3_register_routes(`, `minimax/mm3-job.h` + `mm3_register_job_routes(`, `yue2/yue2-server.h` + `yue2_register_routes(` | Each family's include and registration call; `mm3-job.h` is a mid-file include, after the job system |
| 4-gate | `engine/tools/hot-step-server.cpp` | `#include "hot-step-families.h"` and at least two `hot_step_any_family_weights_present(` calls | Keeps MM3-only or YuE2-only installs booting (#118) |
| 5 | `engine/src/fsq-detok.h`, `engine/src/fsq-tok.h` | `fsq-detok.h` includes `fsq-quant.h`; neither re-declares `FSQ_LEVELS`/`fsq_decode_index`/`fsq_encode_index` | Upstream's FSQ copies are not reference-conformant |
| 6 | `engine/src/hot-step-sampler.h` | string `hotstep_sampler_linked_` | Sentinel presence |
| 7–16, 18 | `engine/ggml/...` | HOT-Step patch markers and op names (BF16 out_prod, mm-backward, quant copies, F16→F32 GEMM accumulate, flash-attn-train, alloc free blocks, Vulkan training ops, ConvRot8, BF16_ROUND, BF16 MUL_MAT gradient) | `engine/ggml` must be the HOT-ggml fork; fix is `git submodule sync -- engine/ggml` then `git submodule update --init engine/ggml` |
| 17 | `engine/ggml` | HEAD equals the pinned gitlink, no local changes or untracked files | Fails closed on any git error; warns (does not fail) outside a git checkout |

Not checked but fork-only and load-bearing in `pipeline-synth-ops.cpp` (verified against the upstream clone — these headers do not exist upstream): `hot-step-sampler-trt.h` (line 10), `adapter-trt.h` (line 11), `trt-runtime-probe.h` (line 16). A manual merge must preserve these too; losing them is a compile error (self-catching), not silent.

## 4. `engine/UPSTREAM_SYNC` — actual contents (last sync)

```
UPSTREAM_REPO=acestep.cpp
UPSTREAM_COMMIT=9d38f00267d71f8a1d7a6d3e325476472ce09179
SYNC_DATE=2026-07-14
SYNC_NOTES=LM perf overhaul (graph arenas, static decode graph replay, set_rows KV, padded attn windows), snake autofuse + fused CUDA/Vulkan kernels via ggml bump to ServeurpersoCom b677b63c, K/V F16 cast before flash_attn_ext, dead full-attention padding mask removed (incl. hot-step-sampler.h), GGML_CUDA_GRAPHS default ON, cpp-httplib 0.44, WAV PCM24 fixes, locale-immune float parser. Upstream src/solvers/ NOT copied (fork has own).
```

The local-only workflow doc's Phase 1 ("first non-comment line = hash") and Phase 6 (heredoc writing a comment header + bare hash) predate this format and are both wrong against it. SKILL.md Phases 1 and 6 carry the corrected snippets.

## 5. `engine/src/_backup_pre_sync/`

- Flat snapshot of the fork's `engine/src/` (including `guidance/`, `schedulers/`, `solvers/` subdirs and an `ace-server.cpp.bak`) taken before the 2026-04-26 sync; file timestamps 2026-04-07 through 2026-04-21.
- **Git-tracked** (not gitignored, 59 files) — so `git add engine/src/` in Phase 6 will stage it if you touch it.
- Purpose: manual rollback safety net for that sync. Treat as read-only historical reference ("what did our pipeline-synth-ops.cpp look like before the last merge"). Do not copy from it during a new sync except to recover a botched merge.
- Optionally refresh it with a fresh snapshot at the start of a new sync — but the pre-sync git commit (Golden rule 3) is the primary rollback mechanism.

## 6. Hardening ledger — DONE / NOT DONE / PENDING

From the local-only design doc `docs/plans/2026-05-05-harden-upstream-sync-implementation.md`, updated against the tree:

**DONE (implemented, in the tree):**
- Linker sentinel (`hot-step-sampler.h:1540-1552` + `hot-step-server.cpp:52-53`)
- `engine/verify-hooks.ps1`, grown from 5 checks to hooks 1–18 (MM3/YuE2 families, boot gates, FSQ, the HOT-ggml fork)
- `build.cmd` and `build-mac.sh` run `verify-hooks.ps1` before compiling, so a lost hook or a stock ggml checkout stops the build in seconds
- Modified-Upstream-Zone exclusion workflow
- `engine/UPSTREAM_SYNC` marker in `KEY=VALUE` format
- Two full syncs completed: upstream `31cc9ea` (2026-04-26, local tag `v1.5-upstream-sync-31cc9ea`) and `9d38f00` (2026-07-14, fork commit `a178b57`, local tag `v1.5-upstream-sync-9d38f00`)

**NOT IMPLEMENTED (explored, rejected — do not "restore"):**
- `hot-step-hooks-verify.h` compile-time guard header
- `static_assert` inside the clobber-prone file (pointless — a direct copy deletes the assert too)
- Rewriting `dit-sampler.h` as a 2-line shim forwarding to `hot-step-sampler.h`
- `knowledge/hot-step-hooks/` knowledge item; `.agents/workflows/upstream-sync.md` (does not exist — the live doc is `docs/plans/upstream-sync-workflow.md`)

**PENDING (as of 2026-10-23):**
- The upstream clone at `D:\Ace-Step-Latest\acestepcpp\acestep.cpp` has local `master` at `9d38f00` and a fetched `origin/master` 34 commits ahead (`de858d8`, "ggml: fork rebase", 2026-10-16). 21 commits only move upstream's ggml gitlink. The rest change 9 files in `src/` (`backend.h`, `dit-graph.h`, `dit.h`, `metadata-fsm.h`, `pipeline-lm.cpp`, `pipeline-lm.h`, `qwen3-enc.h`, `qwen3-lm.h`, `request.cpp`) and 7 in `tools/` (`ace-lm.cpp`, `ace-server.cpp`, `ace-synth.cpp`, `ace-understand.cpp`, `mp3-codec.cpp`, `neural-codec.cpp`, `quantize.cpp`), plus build scripts. Only `metadata-fsm.h` is pristine in the fork; the next sync is mostly manual merging, with `pipeline-lm.cpp` (+229/−179, LM request batching) the largest.

## 7. Caveats / could-not-verify

- The claim that swapping only the include line back to `dit-sampler.h` (without copying the whole file) compiles: **not empirically tested**. The fork's call site (`pipeline-synth-ops.cpp:1465`) passes an extended argument list, so a partial clobber is expected to be a compile error; the *whole-file* copy is the truly silent case.
- Upstream clone path `D:\Ace-Step-Latest\acestepcpp\acestep.cpp` is machine-specific. On a fresh machine, clone upstream anywhere and substitute the path throughout. The fork repo itself has no remote for acestep.cpp (`git remote -v` lists only HOT-Step remotes), so the separate clone is the only source.
- `docs/plans/*` docs are gitignored — absent on other machines. This skill folder is the committed, portable source of truth for the sync procedure.
