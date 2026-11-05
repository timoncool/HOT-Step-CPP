---
name: release-process
description: Runbook for cutting and publishing a HOT-Step CPP release via a v* git tag that triggers the multi-platform CI build and drafts a GitHub Release. Use when asked to cut a release, publish a release, bump the version, push a version tag, run a CI compile test, verify release assets, or debug a failed Release/Cache Warm workflow run.
---

# Cutting & publishing a release — HOT-Step CPP

HOT-Step CPP ships as portable archives for Windows (CUDA/Vulkan/CPU), Linux, and
macOS. A release is produced entirely by CI: you push a git tag matching `v*`,
the `Release` workflow (`.github/workflows/release.yml`) builds every platform
variant and creates a **draft** GitHub Release on `scragnog/HOT-Step-CPP`. You
then arrange and verify the draft; publishing it with the `gh` CLI needs the
user's explicit approval (Golden rule 6). There are **no version
numbers to edit in any file** — the git tag is the sole version source.

All commands below are Windows PowerShell (use `;` to chain, never `&&` in
older shells — this repo's convention). `gh` is already authenticated as
`scragnog`.

## When to use this skill

- The user asks to cut, tag, or publish a release (`vX.Y.Z`).
- The user wants a throwaway CI compile check of the full build matrix, or a
  Windows preflight (manual `release.yml` dispatch) before tagging.
- A `Release` or `Cache Warm` workflow run failed and needs diagnosing.
- You are about to push ANY tag starting with `v` (read Golden rules first).
- Someone asks "where do I bump the version?" (answer: nowhere — see step 2).

## Golden rules (hard constraints)

1. **ANY pushed tag matching `v*` triggers the full multi-platform CI build.**
   The trigger is `on: push: tags: ['v*']` (release.yml:20-22) with no other
   filter; the only other trigger is the manual Windows preflight
   (`workflow_dispatch`, release.yml:14-19), which creates no tag or release.
   A tag push runs 21 GitHub-hosted jobs including multiple CUDA toolkit
   installs, and it also starts the compile-only `rocm-build.yml`, whose push
   trigger has no branch or tag filter (it ran for v1.4.2). Never push a local
   marker/feature tag that starts with `v`.
   WHY: old marker tags like `v1.5-pre-ggml-migration` show how easily one gets
   made; each new one pushed fires the whole pipeline and drafts a release.
2. **Real releases are plain semver `vX.Y.Z`: no hyphen, exactly three numbers.
   Throwaway builds MUST be hyphenated**, conventionally `vX.Y.Z-CI-Test`. WHY:
   release-notes anchoring finds the previous release with
   `git describe --tags --abbrev=0 --exclude '*-*' "${TAG}^"` (release.yml:2036),
   so hyphenated tags are skipped by the changelog. A stray *non-hyphenated*
   tag between releases would silently truncate the next release's notes. The
   in-app update check (`server/src/services/updates/checker.ts`, `STABLE`)
   parses only `v<major>.<minor>.<patch>`: while a release like `v1.4.1.1` or
   `v1.3` is GitHub's latest, every install's check fails with "latest release
   is not a stable version tag".
3. **Pushing a tag requires explicit user approval** (repo git rule: every push
   needs approval). Ask before `git push origin vX.Y.Z`.
4. **Do not bump any `package.json` version.** `server/package.json` and
   `ui/package.json` both sit at `"version": "1.0.2"` while v1.4.2 has shipped —
   these fields are stale and read by nothing in the release pipeline or the
   app. There is no root `package.json`. Editing them achieves nothing.
5. **Everything must be committed AND pushed to `origin/master` before
   tagging.** CI checks out the tag's commit, not your working tree —
   UNCOMMITTED work will not be in the release. (Committed-but-unpushed work
   *would* technically ship via the tag push itself, but push master first
   anyway so `origin/master` is never behind a published release tag.) Stage
   explicit paths only — NEVER `git add -A` or `git add -f`. The tree often
   holds untracked scratch output (build logs under `release/`, `.tmp/`
   folders) that a blanket add would sweep in.
6. **The release is created as a DRAFT** (`gh release create --draft`,
   release.yml:2124-2132). Nothing auto-publishes. **Publishing
   (`--draft=false`) requires explicit user approval, same as a push** —
   present the asset count and notes to the user and wait for their
   go-ahead. Shipped binaries deserve at least the scrutiny of a push, and
   publishing also starts the `RunPod image` workflow and puts the release in
   front of every install's update check.
7. **Never change the Windows runner from `windows-2022`** (release.yml:73).
   `windows-latest` is windows-2025 whose MSVC (`_MSC_VER >= 1950`) is rejected
   by CUDA 12.8/13.1 `nvcc` (`host_config.h`: "Only the versions between 2017
   and 2022 are supported"). The runner image is also baked into the build
   cache key on purpose.
8. **Never tag until `check-release-prereqs.mjs` passes.** A packaged build
   ships code only; weights and data files reach the user by download or by
   being inside the archive, and both are easy to forget because the failure is
   invisible on the dev machine. `node server/scripts/check-release-prereqs.mjs`
   (exit 0 required) verifies every catalogue entry exists on Hugging Face at
   the claimed size in a public repo, that packs reference real file ids, and
   that every `server/src/data/` file is packaged. It also checks the engine
   binaries the server spawns, the portable layout and its fixtures, the
   TensorRT files CI downloads, the latest Cache Warm run on `master`, and
   `tools/docs/check-docs.mjs` (full list: `docs/dev/releasing.md` section 0).
   WHY: v1.3 shipped MM3 training gated on two GGUFs nobody could download
   (#137) and an MM3 caption corpus CI never copied into the archives (#139).
   Green CI proves neither.
9. **Clean up hyphenated test tags after use** (release draft + remote tag +
   local tag). WHY: leftover drafts clutter the release page, and tag hygiene
   protects the changelog logic.

## Procedure

### 0. Preconditions

- On `master` (the only branch used in this repo), working tree clean.
- `git status` clean; `git push` of master already done (with user approval).
- **If master recently absorbed an upstream acestep.cpp sync, run
  `engine/verify-hooks.ps1` before tagging** — the sampler hook can be lost
  silently (compiles, but all solvers/schedulers/guidance go dead) and
  nothing in CI or the asset check catches it. Locally, `engine/build.cmd`
  runs it before every compile and release gate tier 0 runs it too.
- **`node server/scripts/check-release-prereqs.mjs` exits 0** (golden rule 8).
  If this cycle added a model, its weights must already be uploaded and in
  `server/src/data/model-registry.json` — uploading after the release is live
  does not help anyone who already downloaded it.
- **The release gate passes on a packaged build** and the user has listened to
  the renders it staged: `node tools/release-gate/run.mjs --zip <zip>`
  (`docs/dev/releasing.md` section 0b, `tools/release-gate/README.md`). Exit 0
  is GO, 1 is NO-GO, 2 means it could not run.
- Pick the version: look at the latest release
  (`gh release list --limit 3`) and bump semver appropriately.

### 1. Windows preflight, and an optional full compile test

The Windows preflight builds and tests all five Windows archives without a tag
(`docs/dev/releasing.md` section 1). Push master first (dispatch builds the
remote branch), and ask the user before both the push and the dispatch:

```powershell
gh workflow run release.yml --ref master
# optional, non-gating CPU generation measurement:
gh workflow run release.yml --ref master -f cpu_generation_trial=true
```

The archives are versioned `0.0.0-preflight-<12-char sha>` and only uploaded as
artifacts. Check all five `test-windows-artifacts` jobs and their
`windows-smoke-<variant>` evidence; the `cpu-generation-trial` job is
`continue-on-error`, so read its evidence artifact rather than the run colour.

The throwaway tag below verifies CI compiles across all platforms without
cutting a real release. Hyphenated tags build everything but are excluded from
changelog anchoring. Never *publish* a `-CI-Test` draft: publishing a release
also publishes the RunPod container image (`runpod-image.yml`); delete the draft
instead.

```powershell
git tag -a vX.Y.Z-CI-Test -m "compile test"
git push origin vX.Y.Z-CI-Test        # ask the user first — this is a push
# ... monitor (step 4) ...
# cleanup when done:
gh release delete vX.Y.Z-CI-Test --cleanup-tag --yes   # deletes draft + remote tag
git tag -d vX.Y.Z-CI-Test
```

If the compile test FAILED, no draft exists (the `release` job needs
`build-windows`, `test-windows-artifacts`, `build-macos` and `build-linux` all
green), so `gh release delete` errors with "release not found".
Delete the tag directly instead:

```powershell
git push origin --delete vX.Y.Z-CI-Test; git tag -d vX.Y.Z-CI-Test
```

Re-pushing the same `-CI-Test` name after deletion is fine. Tags cannot be
renamed — delete and recreate.

### 2. Version bump — there is none

This is the biggest doc/reality trap. "Cutting a release" is: commit + push
master, create tag, push tag. **No file edits.**

- The tag is the version. The `prepare` job strips the `v` from
  `GITHUB_REF_NAME` for the Windows jobs (release.yml:47-64; a manual dispatch
  uses `0.0.0-preflight-<sha>` instead), and the Linux and macOS jobs use
  `${GITHUB_REF_NAME#v}`. That string names the archives
  (`HOT-Step-CPP-vX.Y.Z-win-x64-<variant>.zip`) and is written to each
  package's `VERSION` file, which the server reads as `APP_VERSION` (printed at
  startup, `version` in `/api/health`). `build-info.json` records the tag too.
- Engine binaries embed the **git short hash + commit date**, not semver:
  `engine/tools/version.cmake:29` writes
  `#define ACE_VERSION "<short-hash> (<date>)"` into a generated `version.h`,
  wired as an always-run CMake target in `engine/CMakeLists.txt:13-21`.
  Automatic at build time — never edit it.
- The in-app update check (`server/src/services/updates/`) compares the
  installed `build-info.json` tag, or `VERSION`, with GitHub's latest published
  release. Nothing to edit there either, but mind the three-number rule
  (Golden rule 2).

### 3. Cut the release

```powershell
git tag -a vX.Y.Z -m "vX.Y.Z — <one-line summary>"
git push origin vX.Y.Z                # requires explicit user approval
```

To rebuild against a different commit: delete the tag remotely and locally,
re-create it on the new commit, re-push (the workflow re-runs).

### 4. Monitor the build

```powershell
gh run list --limit 5                 # find the Release run for your tag
gh run view <run-id>                  # per-job status + timings
```

Expected jobs for a tag push: `prepare` (picks the archive version),
`build-windows` × 5 variants (`cuda13.1`, `cuda12.8`, `cuda12-volta`, `vulkan`,
`cpu`), `test-windows-artifacts` × 5 (each archive on a fresh `windows-2022`
VM), `essentia / linux` and `essentia / macos`, `build-linux` × 6 (the same 5
plus `rocm`), `build-macos` (Apple Silicon Metal, `macos-15`), then a final
`release` job that collects the `release-*` artifacts, generates notes, and
creates the draft. Warm-cache CUDA jobs run
roughly 7–13 min; a cold cache means ~1.5 h per CUDA job (timings from workflow
comments, not re-measured — see Failure signatures for the fix).

To read a failed/cancelled job's log **while the run is still in progress**
(`gh run view --log` won't show it yet), pull it from the API:

```powershell
gh api repos/scragnog/HOT-Step-CPP/actions/jobs/<job-id>/logs > log.txt
```

(If you happen to be in Git-Bash instead of PowerShell, prefix with
`MSYS_NO_PATHCONV=1` so the leading-slash API path isn't mangled.)

Note `fail-fast: true` on the `build-windows` and `build-linux` matrices: one
variant failing cancels its siblings. Cancelled siblings are not the root
cause — find the one that *failed*. `test-windows-artifacts` is
`fail-fast: false`, so every archive gets tested.

### 5. Verify the draft, then publish

```powershell
gh release view vX.Y.Z --json assets --jq '.assets | length'   # expect 24
gh release view vX.Y.Z --json body --jq .body                  # eyeball notes + download table
gh release edit vX.Y.Z --draft=false --latest                  # publish — ONLY after explicit user approval (Golden rule 6)
```

Before handing the draft over, arrange the body the way `docs/dev/releasing.md`
section 4 lists it: curated notes under `Highlights`, then GitHub's generated
`What's Changed` list and `Full Changelog` link against the previous release
tag, then the Downloads table unchanged. Publishing is the user's call. After
it, the `RunPod image` workflow runs on its own; check it with
`gh run list --workflow runpod-image.yml --limit 1`, because neither workflow
reports the other's result.

**Expect 24 assets** (12 archives + 12 `.sha256`; v1.4.2 had exactly these).
Asset names:

- `HOT-Step-CPP-vX.Y.Z-win-x64-{cuda13.1,cuda12.8,cuda12-volta,vulkan,cpu}.zip`
- `HOT-Step-CPP-vX.Y.Z-linux-x64-{same 5, plus rocm}.tar.gz`
- `HOT-Step-CPP-vX.Y.Z-macOS-arm64.tar.gz`
- one `.sha256` per archive

If the count is short, do NOT publish — a build job failed or an artifact
upload was missed; go back to step 4.

### 6. Cleanup

Delete any leftover hyphenated test tags and their draft releases (step 1
cleanup commands). Leave nothing matching `v*-*` on the remote.

## Release notes — shaped by commit messages

The `release` job (release.yml:2030-2122) writes the draft body under
`## What's Changed`: it buckets commits since the previous non-hyphenated tag
by conventional-commit prefix on the first line, `feat*` → "🎵 Features",
`fix*` → "🔧 Fixes", everything else → "📝 Other", then appends a per-file
Downloads table, a ROCm install note when a ROCm archive is present, and a
SHA256 note. **Commit message discipline on master directly becomes the
draft's notes** — write `feat(...)`/`fix(...)` first lines that read well in a
changelog. The published body follows step 5, and the Downloads table carries
over from this draft.

## Key files

| Path | Role |
|---|---|
| `.github/workflows/release.yml` | The whole pipeline. Tag push: `prepare`, 5 Windows builds and 5 archive tests, Essentia, 6 Linux and 1 macOS builds, notes, draft. Manual dispatch: the Windows builds and archive tests only (preflight), plus the opt-in CPU trial. The only workflow that drafts a release. |
| `.github/workflows/cache-warm.yml` | Builds the engine on master under the same cache keys release.yml uses, so tag runs can restore it (GitHub caches are ref-scoped; only master caches are visible to tag runs). Windows and Linux variants only, no macOS job. Triggers: manual dispatch, or master push touching `engine/ggml` (submodule gitlink), `engine/CMakeLists.txt`, or itself; a newer run on the same ref cancels one in progress. |
| `.github/workflows/essentia.yml` | Reusable: release.yml's `essentia` job, also dispatchable alone. Builds the Linux and macOS Essentia bundles. |
| `.github/workflows/runpod-image.yml` | Runs on `release: published` (and manual dispatch on a `v*` tag); pushes `ghcr.io/scragnog/hot-step-cpp:<tag>`, plus `:latest` for a plain non-prerelease tag. |
| `docs/dev/releasing.md` | Committed runbook: release-body layout, Windows preflight, release gate, caching, TensorRT, build identity, data layout. |
| `server/scripts/check-release-prereqs.mjs` | The prerequisites gate (Golden rule 8). `--offline` skips the Hugging Face and TensorRT checks. |
| `tools/release-gate/run.mjs` | API-driven go/no-go on a packaged build; stages renders for the ear test. |
| `engine/tools/version.cmake` | Generates `version.h` with `ACE_VERSION "<git-hash> (<date>)"` at build time. Never hand-edit versions. |
| `engine/CMakeLists.txt:13-21` | `version` custom target wiring for the above. |
| `release/` | Packaging inputs and gates: `esbuild.config.mjs` (bundles the server to `server.mjs` plus `chunks/`), `smoke-boot.mjs` (boot test before packaging), `build-info.mjs` and `check-layout.mjs` (package identity), `smoke-artifact.mjs` (Windows archive test), `HOT-Step.bat`/`HOT-Step.sh` launchers, `profile-path.mjs`, `.env.example`, `README.txt`, and the local `build-release.ps1`. |
| `server/package.json`, `ui/package.json` | `version` fields are stale (1.0.2) and unused by the pipeline. Do not bump. |

## Failure signatures

| Symptom | Cause → fix |
|---|---|
| CUDA jobs take ~1.5 h instead of ~7–13 min | Master build cache missing/stale. Re-warm: `gh workflow run cache-warm.yml` (or GitHub → Actions → Cache Warm → Run workflow on master), wait for it, then re-run the release (delete + re-push the tag). |
| `nvcc` fatal / `host_config.h` "Only the versions between 2017 and 2022 are supported" | Someone switched the runner to `windows-latest` (= windows-2025). Restore `runs-on: windows-2022` (release.yml:73). |
| CMake configure fails after cache restore (stale `cl.exe`/ninja path) | Cache from a different runner image or dead ephemeral tool path. The build step's self-repair deletes every `CMakeCache.txt` (release.yml:375-377); if it recurs, check the runner-image part of the cache key wasn't changed (release.yml:294). |
| ninja loops `build.ninja still dirty ... system time is not set` | Future-dated mtimes from a cache written by a clock-ahead runner; handled by the mtime clamp (release.yml:383-386). |
| Vulkan job loops `build.ninja still dirty after 100 tries` | Restored `vulkan-shaders-gen-prefix` ExternalProject; handled by the prefix nuke (release.yml:392-393). |
| `test-windows-artifacts` failed, so no draft | The extracted archive did not boot, or a backend or runtime DLL check failed. Download the `windows-smoke-<variant>` evidence (logs and `result.json`); acceptance rules are in `docs/dev/building.md` (Windows archive gate) and `release/smoke-artifact.md`. |
| `server.mjs failed its boot test` | `release/smoke-boot.mjs` saw the bundle exit or never answer `/api/health` within 60 s; it prints the last 40 lines of server output. Exit 13 at startup was the 1.4.1 deadlock that esbuild code splitting fixed. |
| `FATAL: required engine binary missing from package: <name>` | The engine build did not produce `ace-server`, `ace-train` or `ace-caption`. A new engine tool must join both the copy list and this guard in all three assemblers. |
| `build-info.json check failed` / `check-layout:` errors | `release/check-layout.mjs --package` rejected the package identity: a contract field, the os/variant, the ggml backend library, `engine/.variant`, `.cuda-version` or `VERSION` disagrees with the asset. |
| Release notes show only 2–3 commits | Changelog anchored to a stray **non-hyphenated** tag pushed between releases (the `--exclude '*-*'` guard only protects against hyphenated ones). Delete the stray tag, delete the draft, re-push the release tag. |
| Draft has fewer than 24 assets | A build job failed (fail-fast cancelled siblings) or an artifact upload was missed. Don't publish; `gh run view <run-id>` and pull the failed job's log via the API command in step 4. |
| Whole matrix cancels when one variant fails | `fail-fast: true` on the `build-windows` and `build-linux` matrices — expected. Diagnose the variant that actually failed. |
| `throw "Build failed: ace-server.exe not found"` in a Windows job | The engine build produced no binary — a real compile failure earlier in that job's log. To reproduce/fix locally, follow AGENTS.md's build rules: `dev-rebuild.bat` (never `engine/build.cmd` directly) and never `cmake --clean-first` (20+ min CUDA recompile). |
| Released binary crashes instantly (Windows `0xC0000409` / Linux SIGSEGV) right after `[Server] Models: ...`, but builds green and local build is fine | **Stale-object mixed-ABI binary from the build cache** (bit v1.1.3, 2026-07-16, issues #82/#83). The git-mtime-restore stamps sources with their *commit* time; any commit authored while the last Cache Warm was still running (or otherwise not in the warm build but committed before its cache-save time) gets an mtime OLDER than the cached `.obj` files, so ninja never rebuilds its dependents. If that commit changed a struct (e.g. `AceRequest` in `request.h`), TUs disagree on layout → memory corruption on first use (`/props` is the first endpoint to touch `AceRequest`, hit by the UI on load — hence "crashes on startup"). Only variants whose warm job SAVED after the stray commit's timestamp are affected (slow CUDA jobs), which is why cpu/vulkan variants work — check warm job `completedAt` vs `git log --pretty=%cI` of struct-touching commits. Since 2026-07-16 both workflows carry a `.built-commit` stamp guard (re-touches files changed since the cached build's commit; no stamp → touches all engine sources) and release.yml smoke-tests every packaged engine (`/health` + `/props` + `/plugins` against stub GGUFs) — so a recurrence should fail the build instead of shipping. The guard also compares the cached and current `engine/ggml` pins and touches what changed between them. If the guard itself is suspect, bumping the cache-generation comment atop `engine/CMakeLists.txt` changes the cache key but does not force a cold build: `restore-keys` still restores the newest cache with the same `cmake-<os>-<variant>-` prefix. For a cold build, list the caches with `gh cache list` and, with the user's approval, delete the suspect ones (`gh cache delete <key>`) before re-running Cache Warm. |

## Institutional knowledge

Facts from the departing lead engineer, verified against the workflows on
2026-07-02 unless noted; line references re-checked 2026-10-23:

- **VALIDATED — the tag is the only version.** No root `package.json`; the
  server/ui `version` fields are dead (stale at 1.0.2 vs v1.4.2 shipped, both
  checked). Engine binaries self-version from git via `version.cmake`.
- **VALIDATED — cache scoping is why cache-warm exists.** GitHub Actions caches
  are ref-scoped: a cache saved by tag run A is invisible to tag run B; only
  default-branch (master) caches are visible to all runs. `cache-warm.yml`
  builds on master under the same keys
  (`cmake-windows-2022-<variant>-<hashFiles('engine/ggml/**','engine/CMakeLists.txt')>`
  and the Linux `cmake-linux-<variant>-...` equivalent) so tag runs get warm
  restores. It has no macOS job, so `build-macos` (`cmake-macos-...`) never
  restores a master cache.
- **VALIDATED — timestamp restore is load-bearing.** `actions/checkout` sets
  all mtimes to "now", which makes Ninja rebuild everything despite a cache
  hit. The workflow restores git commit mtimes for `engine/` AND separately
  for the ggml **submodule from its own history** (release.yml:298-362, which
  also holds the `.built-commit` guard) — the CUDA kernels, the long pole, live
  in the submodule, and the superproject only tracks the gitlink.
- **VALIDATED — selective submodule init is intentional.** `submodules: false`
  on checkout; jobs manually init `engine/ggml` and `engine/vendor/vst3sdk`
  with only the `base cmake pluginterfaces public.sdk` sub-submodules
  (release.yml:129-138) — vstgui4 is "fragile + unneeded".
- **The Linux `rocm` variant ships no ROCm runtime.** rocBLAS unpacks to
  ~3.9 GB and hipBLASLt to ~2.7 GB (per-architecture Tensile kernels), and
  AMD's packages track the user's kernel driver, so the archive carries only
  our binaries and `libggml-hip.so`. Users install ROCm themselves;
  `engine/src/backend.h` dlopen()s each ROCm library on startup and prints the
  per-distro install command when the HIP backend fails to load. `GPU_TARGETS`
  is fat-binaried for RDNA2–RDNA4 (gfx1030, gfx1100/1101/1102, gfx1200/1201)
  so one archive covers them; CDNA is deliberately excluded. Note ROCm needs
  an apt pin (`Pin-Priority: 600` on `o=repo.radeon.com`) or Ubuntu jammy's
  own 5.0.0-1 ROCm packages break the install, and ROCm must be ≥ 6.3 —
  ggml's `vendors/hip.h` uses `__hip_fp8_e4m3`, which 6.2 does not have.
- **VALIDATED — cuda12-volta exists for Tesla V100 (sm_70).** ggml's mma
  flash-attention and MMQ have no sm_70 device code, so that variant builds
  with `-DGGML_CUDA_FORCE_CUBLAS=ON -DHOT_STEP_DISABLE_FA=ON
  -DCMAKE_CUDA_ARCHITECTURES=70-real` (release.yml:95-104).
- **VALIDATED (2026-10-23) — the warm and release flags match; the TensorRT
  step does not.** Every Windows and Linux variant carries the same CMake flags
  in both files. The `cpu` variants build with `-DGGML_BACKEND_DL=ON
  -DGGML_CPU_ALL_VARIANTS=ON` (4df3bee3): without the runtime-selected CPU
  variants, GGML_NATIVE copied the runner's AVX512 and the archive died with
  `0xC000001D` on other CPUs. Cache Warm has no TensorRT SDK step, so its
  `cuda13.1` cache is configured without `HOT_STEP_TRT`; the release job turns
  it on and recompiles the engine targets (`acestep-core` defines it
  `PUBLIC`), while the ggml objects stay cached. The cache *key* hashes only
  ggml + CMakeLists, not the flags.
- **VALIDATED — releasing never touches the local dev build.** No
  `dev-rebuild.bat` involvement; CI builds from scratch/cache on runners.
- **VALIDATED — auth:** the workflow's top-level `permissions: contents: read`
  (release.yml:24-25) gives way to `contents: write` on the `release` job only
  (release.yml:2008-2009). That plus `GH_TOKEN: ${{ github.token }}` is all
  `gh release create` needs; no PAT.
- **UNVALIDATED — exact timings.** The 7–13 min warm / ~1.5 h cold CUDA figures
  come from workflow and doc comments, not re-measurement.
- **VALIDATED (2026-09-03) — Essentia is built in CI for Linux and macOS.**
  The repo's `Essentia/` folder holds only the Windows `.exe`, and copying it
  wholesale is what put a Windows binary in every v1.3 Linux archive (#144).
  `essentia.yml` (reusable, called by `release.yml` as job `essentia`, and
  dispatchable on its own: `gh workflow run essentia.yml`) builds
  `essentia_streaming_extractor_music` from source via `tools/essentia/`
  (cached on the recipe hash), uploads it as a tarball, and the packaging
  steps fail if the bundle is missing. Windows still ships the committed
  `.exe`. Test a recipe change with the standalone workflow (~5 min), not a
  `-CI-Test` tag. Verified 2026-09-04 by a fully green v1.3.1-CI-Test run
  (33867682332): every archive built, and the macOS and linux-cpu tarballs
  were opened and contain `Essentia/essentia_streaming_extractor_music`
  (Linux with 14 libraries under `lib/`, macOS with its dylibs beside it).

Deeper detail (packaging contents, cache keys per OS, pinned tool versions,
recent test-tag naming history): see [reference.md](reference.md).

## Deeper reading

- `docs/dev/releasing.md` — committed human runbook, including the release-body
  layout, the Windows preflight, the release gate and build identity.
- `.github/workflows/release.yml` and `cache-warm.yml` — ground truth; when
  the doc and the workflow disagree, the workflow wins.
- `docs/plans/2026-05-11-release-automation-design.md` — design doc referenced
  at release.yml:9. `docs/plans/` is **gitignored/local-only and may be
  absent** on your checkout.
- `AGENTS.md` (which `CLAUDE.md` imports) — repo-wide git rules (master only,
  explicit-path staging, push needs approval) that still apply during
  releasing.

## Shared data layout

Check all three CI assemblers and both local packagers when changing release layout. Packages use `release/.env.example`, leave managed paths unset and copy `server/src/data/` wholesale into app-owned `server/data/`. Include `profile-path.mjs` with each launcher. The three CI assemblers also copy `docs/user/presets/yue2-plan-play/` to `presets/yue2-plan-play/` and fail without `plan-play-sections.json`. Release restart state is `<profile>/restart-requested`, matching `dataroot/paths.ts`; dev keeps its checkout marker. Run the prereq script for layout fixtures and distribution checks. See [Data folder and upgrades](../../../docs/user/data-folder.md).
