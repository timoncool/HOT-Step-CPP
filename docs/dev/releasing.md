# Releasing HOT-Step CPP — agent runbook

How to cut and publish a release, plus the non-obvious gotchas. Written for an
agent (or human) driving the process with the `gh` CLI on Windows/Git-Bash.

## TL;DR

Releases are cut by **pushing a `vX.Y.Z` tag**. The `Release` workflow builds
every platform and creates a **draft** GitHub Release; you review and publish it.
A separate `Cache Warm` workflow keeps the engine build cache on `master` so
release builds take ~10–15 min instead of ~1.5h for the CUDA jobs.

Starting with the next release containing `.github/workflows/runpod-image.yml`,
the separate `RunPod image` workflow runs when you **publish** the GitHub Release
(not on tag push, so the draft stays private) and publishes
`ghcr.io/scragnog/hot-step-cpp:<release-tag>`, plus `:latest` unless the release is
a prerelease or the tag is hyphenated (`-CI-Test`). It uses `GITHUB_TOKEN`
with `packages: write`, builds the `runpod` Docker target for Linux x86_64 and
has its own cache and concurrency group. It runs independently of the portable
release jobs, so inspect its result separately; neither workflow gates the other.
Ordinary `master` pushes do not build images. See the [RunPod guide](runpod.md)
for pulling the image or building your own.

Manual `RunPod image` dispatch must select an existing `v*` release tag; branch
dispatches fail before checkout or publishing, and the same `:latest` rule applies
(a manual run of a plain `vX.Y.Z` tag does update `latest`, including reruns of older
tags). Publishing a throwaway `-CI-Test` release builds and pushes its own image tag,
so delete drafts instead of publishing them. On the first GHCR
publication, check the package visibility and make it public for anonymous pulls,
or configure registry credentials in RunPod for a private package.

## Prerequisites

- `gh` authenticated (as `scragnog`).
- On `master`, working tree clean, everything committed **and pushed**.
- Pick a semver version **without a hyphen**: `vX.Y.Z` (hyphens are reserved for
  test/pre-release tags — see gotchas).
- **Prerequisites check passes** (below). This one is not optional.

## 0. Check that users can actually get everything the build needs

```bash
node server/scripts/check-release-prereqs.mjs
```

Exit 0 or do not tag. It verifies that every model in the catalogue really
exists in its Hugging Face repo at the size claimed, that the repos are public,
that no pack points at a missing file id, and that every runtime data file is
packaged by `release.yml`.

**Why this is a hard gate.** A packaged build only ships code. Everything else —
weights, catalogues, corpora — has to be either inside the archive or
downloadable, and both routes are easy to forget because the failure is
invisible here: the file is on the dev machine, so the feature works, tsc is
clean, and CI is green. v1.3 shipped MM3 training gated on two GGUFs that had
never been uploaded (#137) and an MM3 caption corpus that CI never copied into
the archives (#139). Neither was catchable by any other check.

If you added a model this cycle, uploading the weights is a separate deliberate
step — see the `model-management` skill — and it must happen **before** the tag,
not after the release goes live.

## 0b. Run the release gate on a packaged build

```bash
node tools/release-gate/run.mjs --zip release/out/HOT-Step-CPP-vX.Y.Z-win-x64-cuda.zip
```

Exit 0 or do not tag. The gate boots the extracted zip against this checkout's
models, then walks the app through its API: every ACE generation mode, a check
that solvers, schedulers, guidance modes and adapters actually change the
output, MM3 and YuE2 renders, the audio tools, a few steps of every trainer,
and fixed-seed fingerprints. Budget about an hour on the 5090 with training
included, or `--tiers 0-3` for the half-hour subset. The run ends by staging
every render in `_experiments/_LISTENING/<stamp>-release-gate/` with a score
sheet: listen before you tag, because nothing mechanical judges quality.
Details and flags: [tools/release-gate/README.md](../../tools/release-gate/README.md).

Every assembler (the local `release/build-release.ps1` and each CI platform job) also runs `release/smoke-boot.mjs` on the freshly bundled `server.mjs`, once the server and its native deps are staged, before anything is packaged. It boots the bundle from the staged release root, with a throwaway data folder and profile, no engine, on a free port, and fails the build if it exits or never answers `/api/health` (60 s). Run it with the Node the release ships, because the staged `better-sqlite3` is built for that version. A bundle that dies at startup cannot be zipped.

The server bundle is `server/server.mjs` plus `server/chunks/`. esbuild code splitting is on deliberately: without it, the top-level await in `dataroot/bootstrap.ts` plus the server's import cycles deadlock the bundle at startup (exit 13, the 1.4.1 bug).

## 1. Windows preflight before releasing

After the intended changes are committed and pushed to `master`, manually dispatch the
Release workflow. Both the push and dispatch require the maintainer's approval:

```bash
gh workflow run release.yml --ref master
```

Manual dispatch uses the same five Windows build jobs and fresh `windows-2022` archive
test jobs as a tag release. It derives `0.0.0-preflight-<12-character commit SHA>` from
the dispatched commit for both the packaged `VERSION` and archive names. There is no
version input or branch name in the archive path. It uploads artifacts only, creates no
tag or release, and skips Linux, macOS, Essentia and all release notes/creation work.

Inspect all five `test-windows-artifacts` results and their `windows-smoke-<variant>`
evidence artifacts. Each job verifies the shipping zip's SHA256, extracts it under
`HOT Step smoke` outside the checkout, boots the packaged server with its bundled Node,
checks engine endpoints with stub models, and runs the model-free backend diagnostic.
The fresh test VM installs no toolkit and receives no build tree. Child process `PATH`
contains Windows system directories plus the isolated registry-provisioned CUDA runtime.
Missing backend and runtime DLL negative checks must fail as expected. CUDA runtime
downloads are cached by their registry entries; the logs record bytes and download time.
See [Windows archive gate](building.md#windows-archive-gate) for acceptance and limits.

A tag release depends on every Windows archive test passing before creating its draft.
It collects the original checksummed `release-*` artifacts, excluding smoke evidence.
The Windows preflight proves startup and backend loading on hosted runners; the local
release gate and listening check above still cover model execution and audio quality.

### Optional CPU generation measurement

For a measured CPU trial, obtain approval for both pushing the reviewed commits and
dispatching the workflow. Push first: `--ref master` uses the remote branch, so local
commits are not included until pushed. After approval and a successful push, run:

```bash
gh workflow run release.yml --ref master -f cpu_generation_trial=true
```

The input defaults to `false`. This still builds and tests all five Windows variants;
only the added generation job uses the CPU archive. It downloads approximately
3.3 GB of models without caching, with a ten-minute download budget, then attempts
one ten-second LM generation and one-step synthesis within five minutes total,
including engine startup. It starts the archive's `ace-server.exe`, submits `/lm`,
polls `/job`, then submits `/synth?format=wav16` and fetches the completed WAV,
using the same HTTP sequence as the app. The trial does not need standalone
LM or synthesis executables.
It creates no tag or release and does not gate release creation.

Inspect the `cpu-generation-trial` job and its evidence artifact directly. The job
uses `continue-on-error`, so a green workflow alone does not prove generation passed.
Report the run link, `download.json` download seconds, `inference.json` inference
seconds and peak working set bytes, and whether WAV validation passed. If it times
out or fails, report that outcome rather than a successful generation time.
The artifact includes logs and any generated audio for inspection.
If the packaged server cannot start or become healthy, the step fails and records
null inference time and peak RAM, rather than reporting a measurement.

The reduced-step trial establishes execution only, not audio quality or normal
generation speed. Keep it non-gating until hosted measurements support a decision.
See [trial details and validation limits](building.md#optional-hosted-cpu-generation-trial).

### Optional full-platform compile test

To verify CI compiles without cutting a real release, push a throwaway
**hyphenated** tag — it triggers the same build pipeline but is ignored by the
changelog logic:

```bash
git tag -a vX.Y.Z-CI-Test -m "compile test" && git push origin vX.Y.Z-CI-Test
# ...watch it (section 3)... then delete when done:
gh release delete vX.Y.Z-CI-Test --cleanup-tag --yes   # removes draft + remote tag
git tag -d vX.Y.Z-CI-Test
```

Re-pushing the **same** `-CI-Test` name is free (delete remote+local, recreate,
push). Tags cannot be renamed.

## 2. Cut the release

```bash
git tag -a vX.Y.Z -m "vX.Y.Z — <one-line summary>"
git push origin vX.Y.Z
```

This triggers `Release`: builds Windows (`cuda13.1`, `cuda12.8`, `cuda12-volta`,
`vulkan`, `cpu`), Linux (those five plus `rocm`) and macOS (Metal), then tests all
five Windows archives before creating a **draft** release with **24 assets**
(12 archives + 12 `.sha256`).

To change the commit or re-run: delete + re-push the tag (it rebuilds).

## 3. Monitor

```bash
gh run list --limit 5
gh run view <run-id>                 # per-job status + timings
```

To read a **failed/cancelled job's** log while the run is still in progress
(`gh run view --log` won't show it yet), pull it from the API:

```bash
MSYS_NO_PATHCONV=1 gh api repos/scragnog/HOT-Step-CPP/actions/jobs/<job-id>/logs > log.txt
```

`MSYS_NO_PATHCONV=1` stops Git-Bash rewriting the leading-slash API path into a
filesystem path.

## 4. Publish

The workflow leaves the release as a **draft**. Before publishing, arrange the
release body in this order:

1. Curated release notes under a `Highlights` heading.
2. GitHub's generated `What's Changed` list and `Full Changelog` link, generated
   against the previous release tag. Keep both; never replace them with the
   curated notes.
3. The Downloads table, unchanged.

Verify the asset count and notes, then leave the draft for the maintainer to
publish. The maintainer uses:

```bash
gh release view vX.Y.Z --json assets --jq '.assets | length'   # expect 24
gh release edit vX.Y.Z --draft=false --latest
```

## 5. Cleanup

Delete any leftover test tags and their drafts (see section 1).

## Build caching — why releases are fast (and how it breaks)

- **GitHub Actions caches are ref-scoped.** A cache saved by one tag run is NOT
  visible to a different tag run — only **default-branch (`master`) caches** are
  visible to every run, including release tags. So releases can only reuse a
  cache that was created on `master`.
- **`.github/workflows/cache-warm.yml`** builds the engine on `master` (when
  `engine/ggml` or `engine/CMakeLists.txt` change, or via manual dispatch) and
  saves the build dir under the **same cache keys** `release.yml` uses. Release
  runs restore it and skip the CUDA compile (the long part).
- **Timings:** cold (no master cache) CUDA jobs ≈ 1.5h each; warm ≈ 7–13 min.
- **Tag only after Cache Warm has finished green.** A release job restores the
  cache once, at its start; a warm run still in flight is a miss and that job
  compiles cold. `check-release-prereqs.mjs` fails while the latest Cache Warm
  on `master` is running or failed (v1.3.5 hit both: Cache Warm had been
  failing for four days, then the Windows CUDA 13.1 cache landed seven minutes
  after the release job looked for it).
- **If CUDA suddenly rebuilds slow:** the master cache is missing/stale. Re-warm
  it: GitHub → Actions → **Cache Warm** → *Run workflow* (on `master`). It also
  auto-runs when `engine/ggml`/`CMakeLists.txt` change.
- Cache reuse depends on git-restored source mtimes (incl. the **ggml submodule**
  — its `.cu` files live in the submodule's own history, not the superproject).

## TensorRT SDK for CI

The Windows `cuda13.1` variant is the only build that compiles the TensorRT
DiT renderer (`HOT_STEP_TRT`, MM3's TensorRT DiT path). CMake only enables it
when `engine/deps/tensorrt/{include,lib}` exist at configure time, and that
directory is gitignored and never checked out — so `release.yml` fetches it
itself, for that one matrix entry only.

- **Headers + license zip** (`tensorrt-<version>-headers.zip`, hosted at HF
  `scragnog/HOT-Step-CPP-TensorRT`) unpacks to `engine/deps/tensorrt/include`.
  Public TensorRT headers are Apache-2.0, so this zip is small and freely
  redistributable — built by `tools/tensorrt-sdk/make-sdk-zip.ps1`.
- **Import libraries** (`nvinfer_10.lib`, `nvonnxparser_10.lib`) are **not**
  hosted anywhere — NVIDIA's SLA doesn't name the SDK's own `.lib` files as
  distributable, only the runtime DLLs and headers. `release.yml` instead
  downloads the (distributable) `nvinfer_10.dll` / `nvonnxparser_10.dll` from
  the same HF repo and regenerates equivalent import libs from their own
  export tables at build time, via `tools/tensorrt-sdk/make-import-libs.ps1`
  (`dumpbin /exports` → a `.def` → `lib.exe /def`). This is purely mechanical
  — nothing is copied out of NVIDIA's SDK archive.
- Every filename and sha256 the workflow fetches lives in `release.yml`'s
  top-level `env:` block (`TRT_HF_REPO`, `TRT_SDK_ZIP`, `TRT_SDK_SHA256`,
  `TRT_NVINFER_DLL`, `TRT_NVINFER_DLL_SHA256`, `TRT_NVONNXPARSER_DLL`,
  `TRT_NVONNXPARSER_DLL_SHA256`). The download step verifies each sha256
  before use and fails the job on a mismatch.
- The assembled `engine/deps/tensorrt/` is cached (`actions/cache`, keyed on
  `TRT_SDK_ZIP` + the nvinfer DLL sha256) so a normal release re-run doesn't
  re-download or re-derive anything.
- A step right after the CMake configure asserts the configure log contains
  `[TRT] Found vendored SDK` — if TRT silently failed to enable, the job fails
  loudly instead of shipping a cuda13.1 build with a stubbed-out renderer.
- The runtime DLLs themselves are **never** packaged into the release archive
  — a user gets them through the Model Manager, same as any other large model
  file (see `docs/plans/2026-09-11-mm3-trt-dit-shipping.md`). CI downloads
  them only to derive import libs; `check-release-prereqs.mjs` checks that all
  three files (headers zip + both DLLs) actually exist on HF before a tag.

**Bumping the TensorRT version:**

1. Update the vendored dev copy at `engine/deps/tensorrt/` (used for local
   builds) to the new SDK.
2. `./tools/tensorrt-sdk/make-sdk-zip.ps1` — regenerates the headers zip and
   prints its name, size, and sha256.
3. Upload the new zip to `scragnog/HOT-Step-CPP-TensorRT` on Hugging Face,
   alongside the new `nvinfer_10.dll` / `nvonnxparser_10.dll` (get their
   sha256 with `Get-FileHash`).
4. Update `release.yml`'s `env:` block: `TRT_SDK_ZIP`, `TRT_SDK_SHA256`, and
   the DLL sha256 values (the DLL filenames themselves rarely change).
5. `node server/scripts/check-release-prereqs.mjs` — confirms all three files
   are reachable on HF before you tag.

## Gotchas / lessons learned

- **Windows runner is pinned to `windows-2022`.** Do NOT switch to
  `windows-latest` — that's windows-2025, whose MSVC (`_MSC_VER >= 1950`) is
  rejected by CUDA 12.8/13.1 `nvcc` (`host_config.h`: VS 2017–2022 only).
- **The Vulkan SDK is pinned (1.4.357.0), not `latest`.** LunarG can list a new
  version before its `config.json` exists, and `latest` then 404s in
  `setup-vulkan-sdk`. To bump it, check
  `https://vulkan.lunarg.com/sdk/config/<ver>/windows/config.json` returns 200.
- **Any pushed `v*` tag triggers the Release pipeline.** Use `vX.Y.Z` for
  releases and `-CI-Test` (or other hyphenated) tags for throwaway checks; delete
  them afterward. Don't push local feature tags that match `v*`.
- **Changelog range** = commits since the previous **non-hyphenated** tag
  (`git describe ... --exclude '*-*'`). This is why a stray `vX-CI-Test` tag must
  not be treated as a release; the exclude guard handles it, but still clean up.
- **The release is a draft** — it does not auto-publish. Review before going live.
- **Cache key** = `cmake-<runner>-<variant>-<hash(engine/ggml, CMakeLists)>`.
  Changing the runner image invalidates it (compiler abs-paths bake into
  `CMakeCache.txt`); the key includes the runner image to prevent stale restores.

## Reference

- Workflows: [`.github/workflows/release.yml`](../../.github/workflows/release.yml),
  [`.github/workflows/cache-warm.yml`](../../.github/workflows/cache-warm.yml)

## Data layout acceptance

Archives contain app binaries and immutable `server/data/` resources. User data is initialized in sibling `HOT-Step-Data` or reused through the profile pointer. Package `release/.env.example`, `profile-path.mjs`, the platform launcher and `README.txt`, plus `open-browser-if-needed.mjs` on Windows (a Node helper; the launcher must never start PowerShell, which antivirus quarantines); do not create mutable app-root models or adapters. Windows uses `runtime/node.exe`, Linux and macOS use `runtime/bin/node`. Local packagers follow the same layout.

The prereq script checks three workflow platform sections and both local assemblers, required engine targets and wholesale resource copies, unset managed defaults, profile marker agreement and disposable release/dev fixtures. Run it without `--offline` to retain Hugging Face checks. Actual native platform startup remains a release smoke-test requirement. See [Data folder and upgrades](../user/data-folder.md).
