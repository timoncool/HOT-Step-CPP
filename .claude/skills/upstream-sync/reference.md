# upstream-sync — reference

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

Deep detail supporting [SKILL.md](SKILL.md). Everything here was verified against the working tree on 2026-07-02.

## 1. Linker sentinel — exact mechanism

The silent-clobber defense is a symbol-dependency chain:

`engine/src/hot-step-sampler.h:1308-1315` (comment at 1309 names the expected error):

```cpp
//     "unresolved external symbol hotstep_sampler_linked_"
#if defined(_MSC_VER)
__declspec(selectany) int hotstep_sampler_linked_ = 1;
#else
__attribute__((weak)) int hotstep_sampler_linked_ = 1;
#endif
```

`engine/tools/hot-step-server.cpp:47-52`:

```cpp
// hot-step-sampler.h defines hotstep_sampler_linked_ with external linkage.
...
extern int hotstep_sampler_linked_;
static volatile int * _hotstep_guard_ = &hotstep_sampler_linked_;
```

Chain: `pipeline-synth-ops.cpp` includes `hot-step-sampler.h` → the symbol is compiled into `acestep-core.lib` → `hot-step-server.cpp` (the `ace-server` executable) references it at link time. If the include in `pipeline-synth-ops.cpp` reverts to `dit-sampler.h`, the symbol is never emitted and the link of `ace-server` fails with exactly `unresolved external symbol hotstep_sampler_linked_`.

Limits of the sentinel: it only guards the `pipeline-synth-ops.cpp` → `hot-step-sampler.h` edge. If `hot-step-sampler.h` itself is damaged/replaced (sentinel definition lost), both the definition and the failure signal disappear together — that is why `verify-hooks.ps1` check 5 greps for the literal string `hotstep_sampler_linked_` inside `hot-step-sampler.h`.

## 2. Runtime proof: log-line comparison

Both samplers print a `[DiT] Solver:` line, so its mere presence proves nothing. Formats differ:

| Sampler | Line | Source |
|---|---|---|
| HOT-Step (active) | `[DiT] Solver: <name> (<scheduler>, <N> NFE/step, order <N>)` | `engine/src/hot-step-sampler.h:503` |
| HOT-Step (active) | `[DiT] Guidance: <display> (<name>)...` | `engine/src/hot-step-sampler.h:525` |
| Upstream vanilla | `[DiT] Solver: <name> (<N> NFE/step, order <N>)` — no scheduler field | upstream `src/dit-sampler.h:397` |
| Upstream vanilla | *(no Guidance line at all)* | — |

So: **`[DiT] Guidance:` present = HOT-Step sampler is live.** Check the newest session folder: `<data root>/logs/YYYY-MM-DD_HH-MM-SS/ace_engine.log`.

(NFE = number of function evaluations, i.e. DiT forward passes per solver step.)

## 3. verify-hooks.ps1 — check-by-check

Source: `engine/verify-hooks.ps1` (80 lines). Regex-greps, exit 0/1:

| # | File | Requires | Notes |
|---|---|---|---|
| 1 | `engine/src/pipeline-synth-ops.cpp` | `#include "hot-step-sampler.h"` | Explicit FAIL message if `dit-sampler.h` found instead (lines 19-22); WARN+error if neither |
| 2 | `engine/src/model-store.h` | `#include "hot-step-params.h"` | |
| 3a | `engine/src/dit.h` | `#include "adapter-merge.h"` | |
| 3b | `engine/src/dit.h` | `#include "adapter-runtime.h"` | Does NOT check `hot-step-build-flags.h` (dit.h:12) — a merge must still keep it or the build breaks |
| 4 | `engine/tools/hot-step-server.cpp` | `#include "hot-step-params.h"` | HOT-Step-zone file, can't be sync-clobbered, checked anyway |
| 5 | `engine/src/hot-step-sampler.h` | string `hotstep_sampler_linked_` | Sentinel presence |

Not checked but fork-only and load-bearing in `pipeline-synth-ops.cpp` (verified against the upstream clone — these headers do not exist upstream): `hot-step-sampler-trt.h` (line 10), `adapter-trt.h` (line 11), `stream-pipeline.h` (line 12). A manual merge must preserve these too; losing them is a compile error (self-catching), not silent.

## 4. `engine/UPSTREAM_SYNC` — actual contents (last sync)

```
UPSTREAM_REPO=acestep.cpp
UPSTREAM_COMMIT=31cc9ea3a4b21030249ce9e4794ea2df8de46627
SYNC_DATE=2026-04-26
SYNC_NOTES=Full sync: lm_seed, VAE latent I/O, repaint/splice, caption lock, LM CFG fix, DCW, custom scheduler, mp3_bitrate, vae selection, JobStatus enum.
```

The local-only workflow doc's Phase 1 ("first non-comment line = hash") and Phase 6 (heredoc writing a comment header + bare hash) predate this format and are both wrong against it. SKILL.md Phases 1 and 6 carry the corrected snippets.

## 5. `engine/src/_backup_pre_sync/`

- Flat snapshot of the fork's `engine/src/` (including `guidance/`, `schedulers/`, `solvers/` subdirs and an `ace-server.cpp.bak`) taken before the 2026-04-26 sync; file timestamps 2026-04-07 through 2026-04-21.
- **Git-tracked** (not gitignored) — so `git add engine/src/` in Phase 6 will stage it if you touch it.
- Purpose: manual rollback safety net for that sync. Treat as read-only historical reference ("what did our pipeline-synth-ops.cpp look like before the last merge"). Do not copy from it during a new sync except to recover a botched merge.
- Optionally refresh it with a fresh snapshot at the start of a new sync — but the pre-sync git commit (Golden rule 3) is the primary rollback mechanism.

## 6. Hardening ledger — DONE / NOT DONE / PENDING

From the local-only design doc `docs/plans/2026-05-05-harden-upstream-sync-implementation.md`:

**DONE (implemented, in the tree):**
- Linker sentinel (`hot-step-sampler.h:1311-1315` + `hot-step-server.cpp:51-52`)
- `engine/verify-hooks.ps1` (5 checks)
- Modified-Upstream-Zone exclusion workflow
- `engine/UPSTREAM_SYNC` marker in `KEY=VALUE` format
- One full sync completed at upstream `31cc9ea` (2026-04-26), local tag `v1.5-upstream-sync-31cc9ea`

**NOT IMPLEMENTED (explored, rejected — do not "restore"):**
- `hot-step-hooks-verify.h` compile-time guard header
- `static_assert` inside the clobber-prone file (pointless — a direct copy deletes the assert too)
- Rewriting `dit-sampler.h` as a 2-line shim forwarding to `hot-step-sampler.h`
- `knowledge/hot-step-hooks/` knowledge item; `.agents/workflows/upstream-sync.md` (does not exist — the live doc is `docs/plans/upstream-sync-workflow.md`)

**PENDING (as of 2026-07-02):**
- Upstream clone HEAD is `4922ed1` ("cmake: add Intel Arc GPU support via SYCL backend"), ahead of last-synced `31cc9ea` — includes SYCL/Intel Arc, GGML syncs, Vulkan snake-activation work. Next sync has real Phase 2/3 work.

## 7. Caveats / could-not-verify

- The claim that swapping only the include line back to `dit-sampler.h` (without copying the whole file) compiles: **not empirically tested**. The fork's call site (`pipeline-synth-ops.cpp:1503`) passes an extended argument list, so a partial clobber is expected to be a compile error; the *whole-file* copy is the truly silent case.
- Upstream clone path `D:\Ace-Step-Latest\acestepcpp\acestep.cpp` is machine-specific. On a fresh machine, clone upstream anywhere and substitute the path throughout.
- `docs/plans/*` docs are gitignored — absent on other machines. This skill folder is the committed, portable source of truth for the sync procedure.
