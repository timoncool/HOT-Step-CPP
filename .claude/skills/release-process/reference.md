# release-process — reference

Depth beyond the runbook in [SKILL.md](SKILL.md). All line references are into
`.github/workflows/release.yml` unless stated. First verified against the
workflow files and the live v1.1.2 release on 2026-07-02; re-checked against
the workflows and the live v1.4.2 release on 2026-10-23. Six workflows exist:
`release.yml`, `cache-warm.yml`, `essentia.yml` (called by `release.yml`),
`runpod-image.yml`, `rocm-build.yml` (compile-only) and `docs.yml`.

## Build matrix — exact variants and flags

Pinned tool versions (release.yml:27-41): Node `24.18.0` (`NODE_VERSION`), and
the TensorRT SDK files the Windows `cuda13.1` job downloads (`TRT_*`).

### prepare (release.yml:47-64), runner `windows-2022`

Picks the archive version: the tag without its `v`, or
`0.0.0-preflight-<first 12 characters of the sha>` on a manual dispatch. It
throws on anything outside `[0-9A-Za-z._-]`.

### build-windows (release.yml:66-767), runner `windows-2022` (pinned)

| Variant | CUDA toolkit | Extra CMake flags beyond the common set |
|---|---|---|
| `cuda13.1` | 13.1.0 | `-DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON`, plus the TensorRT SDK (the only variant with it) |
| `cuda12.8` | 12.8.1 | same as cuda13.1 without TensorRT (compatibility build) |
| `cuda12-volta` | 12.8.1 | cuda flags + `-DGGML_CUDA_FORCE_CUBLAS=ON -DHOT_STEP_DISABLE_FA=ON -DCMAKE_CUDA_ARCHITECTURES=70-real` (Tesla V100 sm_70; ggml mma flash-attn/MMQ have no sm_70 device code) |
| `vulkan` | — | `-DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON` (Vulkan SDK pinned to 1.4.357.0) |
| `cpu` | — | `-DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON` (CPU variants picked at runtime; a native build copied the runner's AVX512 and died with `0xC000001D` elsewhere) |

All Windows variants also set `-DHOT_STEP_STATIC_RUNTIME=ON`.

### cpu-generation-trial (release.yml:770-1024), manual dispatch only

Runs when `cpu_generation_trial=true`, after `build-windows`, on
`windows-2022` with `continue-on-error: true`. It downloads four registry models
(10-minute budget), starts the CPU archive's `ace-server.exe`, runs `/lm` then
`/synth?format=wav16` in a five-minute budget and uploads the
`cpu-generation-trial` evidence artifact. Nothing depends on it.

### test-windows-artifacts (release.yml:1027-1109), runner `windows-2022`

One fresh VM per Windows variant, `fail-fast: false`, 30-minute timeout. It
checks the zip against its `.sha256`, restores the CUDA runtime cache for CUDA
variants, runs `release/smoke-artifact.mjs` and uploads `windows-smoke-<variant>`
evidence. Runs for both tag pushes and manual dispatches.

### build-linux (release.yml:1470-1991), runner `ubuntu-22.04`

Pinned to 22.04 for wider glibc compatibility across distros. Six variants: the
five Windows ones with the same flags minus `-DHOT_STEP_STATIC_RUNTIME=ON`, plus
`rocm` (`-DGGML_HIP=ON -DGGML_STATIC=OFF
-DGPU_TARGETS="gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"`, ROCm 6.4.4
`rocm-hip-sdk` from repo.radeon.com behind an apt pin). Tag pushes only, after
`essentia`.

### build-macos (release.yml:1114-1465), runner `macos-15`

Single Apple Silicon / Metal build (`-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
-DGGML_BACKEND_DL=OFF`). `macos-15` because Xcode 16+ is needed for
`std::from_chars<float>`. Tag pushes only, after `essentia`.

### essentia (release.yml:2000-2002)

Calls `essentia.yml`, whose `linux` (`ubuntu-22.04`) and `macos` (`macos-15`)
jobs build or restore the Essentia bundles. Tag pushes only.

### release job (release.yml:2004-2132), runner `ubuntu-latest`

`needs: [build-windows, test-windows-artifacts, build-macos, build-linux]`, tag
pushes only, `permissions: contents: write`. Downloads the `release-*`
artifacts (never the smoke evidence or the CPU trial), collects `*.zip` /
`*.tar.gz` / `*.sha256` into `release-assets/` (skipping `essentia-*`),
generates notes, then `gh release create "<tag>" --draft --title "<tag>"
--notes-file release-notes.md release-assets/*`.

`fail-fast: true` on `build-windows` and `build-linux` (release.yml:75, 1475) —
one variant failure cancels its siblings. `test-windows-artifacts` and both
`cache-warm.yml` matrices use `fail-fast: false`, so they finish as many
variants as they can.

## What's inside each archive

Assembled in "Assemble release" (Windows: release.yml:480-675; Linux:
1787-1915; macOS: 1269-1390). Each portable package bundles:

- `VERSION` (the archive version, release.yml:491) and `build-info.json`
  (`release/build-info.mjs`, checked by `release/check-layout.mjs --package`,
  release.yml:664-668).
- Portable Node 24.18.0 runtime (`runtime/node.exe` on Windows, release.yml:495;
  `runtime/bin/node` plus ICU data on Linux and macOS). The server and UI accept
  Node 20 to 24; the release pin needs a matching better-sqlite3 prebuild because
  it ships inside every archive.
- Engine binaries (`ace-server`, `mastering`, `mp3-codec`, `neural-codec`,
  `vst-host`, `quantize`, `ace-midi`, `ace-train`, `ace-caption`) and the ggml
  libraries (release.yml:510-548). The build fails when `ace-server`,
  `ace-train` or `ace-caption` is missing (release.yml:539-543). Windows also
  gets the newest MSVC CRT and OpenMP DLLs from the runner (release.yml:565-589).
- Lua plugins: `engine/plugins` into `engine/plugins`, and the top-level
  `plugins/` folder when present (release.yml:550-556).
- `presets/yue2-plan-play/` from `docs/user/presets/`; the job fails without
  `plan-play-sections.json` (release.yml:558-563).
- Variant marker files (release.yml:653-662):
  - `engine/.variant` — `cuda` / `vulkan` / `cpu` (every CUDA build is `cuda`
    from the server's perspective; Linux adds `rocm`)
  - `engine/.cuda-version` — CUDA major version, only for cuda variants
    (runtime DLL version selection)
- esbuild-bundled server: `node release/esbuild.config.mjs` producing
  `server/server.mjs` plus `server/chunks/`; the job throws if `server.mjs` is
  missing (release.yml:595-598), and `release/smoke-boot.mjs` boots it once its
  dependencies are staged (release.yml:631-633).
- `better-sqlite3` prebuilt for the portable Node (release.yml:455-461 Windows;
  `prebuild-install --target 24.18.0 --arch arm64` on macOS, release.yml:1235),
  plus its `bindings` and `file-uri-to-path` dependencies.
- ffmpeg from `ffmpeg-static`'s install script (release.yml:437-453, 618-622).
- `server/data/`: all of `server/src/data/`, copied wholesale
  (release.yml:624-630).
- Prebuilt `ui/dist` (release.yml:635-639). No `models/` or `adapters/`
  folders: the mutable data folders are created beside the app at first boot.
- `.env.example`, `profile-path.mjs`, `HOT-Step.bat` (or `.sh`), `README.txt`,
  and on Windows `open-browser-if-needed.mjs` (release.yml:670-675).
- `Essentia/`: Windows copies the committed folder (the `.exe`) when present;
  Linux and macOS unpack the `essentia` job's bundle and fail without it.
  `noise_samples/` is copied when present.

Windows packages are `.zip` (compressed via .NET `ZipFile`, release.yml:744-749)
with a sidecar `.sha256` (release.yml:756-757). Linux/macOS are `.tar.gz` with
the same sidecar scheme.

## Cache machinery (why releases take ~10-15 min, not ~1.5 h)

- **Ref scoping:** GitHub Actions caches saved on `refs/tags/vX` are invisible
  to other tag runs. Only caches saved on the default branch (master) are
  visible to every run. Hence `cache-warm.yml`: it builds the engine on master
  under the exact keys release.yml restores.
- **Cache keys** (release.yml:294, 1148, 1651):
  - `cmake-windows-2022-<variant>-<hashFiles('engine/ggml/**','engine/CMakeLists.txt')>`
  - `cmake-linux-<variant>-<same hash>`
  - `cmake-macos-<same hash>`
  The runner image is baked into the Windows key deliberately — compiler
  absolute paths live in `CMakeCache.txt`, so a cache from a different image is
  poison. The key hashes only ggml + `engine/CMakeLists.txt`, **not** the CMake
  flags: flag edits do NOT invalidate the cache. Each key also has a
  `restore-keys` prefix (the key without its hash), so a changed hash, a
  `CMakeLists.txt` edit included, restores the newest older cache and builds
  incrementally on it.
- **cache-warm.yml triggers** (cache-warm.yml:21-28): `workflow_dispatch`, or a
  master push touching `engine/ggml` (the submodule **gitlink**, i.e. a
  submodule pointer bump), `engine/CMakeLists.txt`, or the workflow file
  itself. It warms the five Windows and six Linux variants (no macOS job) and
  stops after the engine build — no packaging, no release. Concurrency is
  `cache-warm-${{ github.ref }}` with `cancel-in-progress: true`
  (cache-warm.yml:30-32), so a newer qualifying push cancels a warm run in
  progress.
- **Timestamp restore** (release.yml:298-362; Linux 1655-1718; macOS 1152-1215;
  mirrored in cache-warm): `actions/checkout` sets every mtime to "now", which
  makes Ninja treat all cached objects as stale. The step rewrites mtimes from
  git commit history for `engine/`, and separately for the ggml submodule
  **from the submodule's own history** — the superproject only tracks the
  gitlink, and the CUDA kernels (the long pole) live inside the submodule.
- **Stale-cache guard** (same step): each build writes
  `engine/build/.built-commit`. After a restore, the step touches every
  `engine/` file changed since that commit (all of `engine/src` and
  `engine/tools` when the stamp is missing or unknown), then compares the cached
  and current `engine/ggml` pins and touches what changed (all of `engine/ggml`
  when the cached pin cannot be resolved).
- **Cache-restore self-repair** in the build step (release.yml:364-393):
  1. Delete every `CMakeCache.txt` recursively (top-level and nested
     ExternalProject ones) — dead ephemeral ninja / Vulkan-SDK absolute paths
     (release.yml:375-377).
  2. Clamp future-dated mtimes to now — a cache written by a clock-ahead
     runner otherwise makes ninja loop `build.ninja still dirty ... system
     time is not set` (release.yml:383-386).
  3. Nuke `vulkan-shaders-gen-prefix` — that ExternalProject loops
     `build.ninja still dirty after 100 tries` when restored; it rebuilds in
     seconds (release.yml:392-393).
- **Known warm/release drift:** the CMake flags match in every variant, but
  cache-warm has no TensorRT SDK step, so its Windows `cuda13.1` cache is
  configured without `HOT_STEP_TRT`. The release job's configure turns it on;
  `acestep-core` defines it `PUBLIC`, so the engine targets recompile while the
  ggml objects stay cached. Keep the matrices in sync when editing either file —
  the file headers say "must match".

## Release notes internals (release.yml:2030-2122)

- Previous tag: `git describe --tags --abbrev=0 --exclude '*-*' "${TAG}^"`.
  Empty result (first release) falls back to `git log --oneline "$TAG"`.
- The body starts with `## What's Changed`. Bucketing is a case-insensitive grep
  on the first line of each commit: `^<hash> feat` → Features, `^<hash> fix` →
  Fixes, everything else → Other. Scopes like `feat(adapter):` match fine; a
  body-only `feat:` does not.
- Downloads table rows are matched by filename regex (`win.*cuda13`,
  `win.*cuda12-volta` before `win.*cuda12`, `linux.*rocm`, `macOS`, etc.) —
  renaming the archive pattern breaks table classification silently. A ROCm
  install note follows the table when a `*rocm*` asset exists.
- The maintainer's final body puts curated notes and GitHub's own generated
  `What's Changed` list ahead of this table (`docs/dev/releasing.md` section 4).

## Tag hygiene and history

- Real releases: `v1.4.2` (latest, published 2026-10-22, 24 assets — verified
  live 2026-10-23), `v1.4.1.1`, `v1.4.1`, `v1.4.0`, `v1.3.6`, ... Two of them,
  `v1.3` and `v1.4.1.1`, do not fit the `vX.Y.Z` form the in-app update check
  parses.
- Throwaway convention: any hyphenated `v*` tag. `docs/dev/releasing.md` uses
  `vX.Y.Z-CI-Test`; the actual throwaways are lowercase `v0.0.0-ci-test-N`.
  Case/format of the suffix doesn't matter to the changelog guard — **only the
  hyphen does**. On 2026-10-23 twelve `v0.0.0-ci-test-*` tags were still on the
  remote, and the `v0.0.0-ci-test-13` draft still existed.
- Marker tags such as `v1.5-pre-ggml-migration`, `v1.0.7a-pre-docker` and
  `v1.5-upstream-sync-*` are on the remote too. They are hyphenated, so the
  changelog skips them; pushing a new marker tag fires the full pipeline.
- Tags cannot be renamed; delete (remote + local) and recreate:
  `gh release delete <tag> --cleanup-tag --yes; git tag -d <tag>`.

## gh CLI cheat sheet (PowerShell)

```powershell
gh release list --limit 5
gh run list --limit 5
gh run view <run-id>
gh run view <run-id> --log                                   # finished runs only
gh api repos/scragnog/HOT-Step-CPP/actions/jobs/<job-id>/logs > log.txt   # in-progress runs
gh release view vX.Y.Z --json assets --jq '.assets | length'
gh release view vX.Y.Z --json body --jq .body
gh release edit vX.Y.Z --draft=false --latest
gh release delete vX.Y.Z-CI-Test --cleanup-tag --yes
gh workflow run release.yml --ref master                      # Windows preflight
gh workflow run cache-warm.yml                                # re-warm master cache
gh run list --workflow runpod-image.yml --limit 1             # after publishing
gh cache list                                                 # build caches by key
```

Quoting note: the `--jq` expressions above are single-quoted; in PowerShell
keep them single-quoted so `$` and `|` inside are passed literally.
