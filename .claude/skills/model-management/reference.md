# Model Management — Reference

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

Deep detail backing [SKILL.md](SKILL.md). All line numbers verified against the working tree on 2026-07-02; re-verify after large refactors.

## 1. Engine startup scan — exact order and rules (`engine/src/model-registry.h`)

`registry_scan()` (line 234) runs once at ace-server startup on the `--models` dir:

1. **Root `.gguf` files** — classified by GGUF header key `general.architecture` (`registry_classify_gguf`, lines 99-130):
   `acestep-lm` → LM, `acestep-dit` → DiT, `acestep-text-enc` → Text-Enc, `acestep-vae` → VAE, `pp-vae` → PP-VAE.
   Unknown arch → `[Registry] WARNING: skipping <f> (unknown architecture)` and ignored (line 248).
2. **Root `.safetensors` files** — classified by lowercase filename **prefix only** (lines 269-291): `pp-vae`/`pp_vae` → PP-VAE; else `vae` or `scragvae` → VAE. No other single-file safetensors roles exist — a safetensors DiT/LM must be a checkpoint **directory** (step 4).
3. **Root `.onnx` files** (lines 293-326) — prefix `dit_`/`dit-` → DiT, `vae_`/`vae-`/`scragvae_`/`scragvae-` → VAE, `lm_`/`lm-` → LM, `text_enc`/`text-enc` → Text-Enc; anything else skipped with `(unrecognized ONNX prefix)`.
4. **Subdirectories with `config.json` + `model.safetensors` (or `model.safetensors.index.json` for sharded)** (lines 328-375) — classified by **`config.json` content** via `config_json_classify()`, not by directory name. `ModelEntry.path` is the directory; the loader finds files inside.
5. **Subdirectories containing at least one `.onnx`** (lines 377-455) — classified by the ONNX filename prefix, falling back to the directory-name prefix (`lm*`, `dit*`, `vae*`, `text*`/`enc*`). Dirs already claimed as safetensors checkpoints are skipped.

Returns false (→ `[Server] ERROR: no GGUF models found`, exit 1) only if the total classified count is zero.

`registry_scan_adapters()` (lines 464-497): root `.safetensors` files logged as `(ComfyUI)`; subdirs containing `adapter_model.safetensors` logged as `(PEFT)`.

`registry_find_non_onnx()` (lines 62-85): returns the first (or named) non-`.onnx` entry in a bucket. Exists because **ONNX VAE files are decoder-only** — VAE *encode* (cover/repaint/extend) must use a non-ONNX VAE. With only an ONNX VAE installed, encode paths have no VAE at all.

## 2. Startup validation (`engine/tools/ace-server.cpp`)

- `--models <dir>` required (lines 1662-1667).
- Scan finds nothing → `[Server] ERROR: no GGUF models found in <dir>`, exit 1 (lines 1673-1677).
- Pipeline check (lines 1686-1711): `have_synth = DiT && Text-Enc && VAE`. Incomplete + LM present → `WARNING: /synth unavailable, missing: <list>`, server stays up LM-only. Incomplete + no LM → `ERROR: no usable pipeline`, exit 1.
- `/props` (lines 1500-1564): `models` object has buckets **`lm`, `embedding`, `dit`, `vae`** (note: text-enc bucket is named `embedding` here); plus `adapters` array, `cli.max_batch`, `default` (full AceRequest defaults), and `presets` (`turbo`: 8 steps / guidance 1.0 / shift 3.0; `sft`: 50 steps / guidance 1.0 / shift 1.0).

Node side (`server/src/index.ts`): on abnormal exit, respawn after 3 s; max 3 crashes per 30 s window (lines 152-156, 284-308), then `setEngineReady(false, ...)` with the missing-DLL hint. With the engine down, `GET /api/models` returns the fallback `{ models: { dit: [], lm: [], vae: [], understand: [] }, adapters: [], config: {}, defaults: {}, aceServerDown: true, error }` (`routes/models.ts:28-36`) — note the shape mismatch vs live `/props` (no `embedding`, spurious `understand`).

First-launch bootstrap (`index.ts:318-380`): before spawning ace-server, checks the 3 required cuBLAS/cudart DLLs (ids `cuda-rt-cublas[,-lt,-cudart]`, or `-12` variants when `.cuda-version` beside the exe says CUDA ≤ 12) and auto-downloads any missing via the download service.

## 3. GGUF loading behavior (`engine/src/gguf-weights.h`)

- `gf_load()` mmaps the file and verifies **every tensor's byte range fits inside the file** before use (lines 132-150). Truncated download → `[GGUF] FATAL: '<path>' is truncated or corrupt ... Re-download the file` and a clean load failure — added to avoid a segfault deep in `cuMemcpyHtoDAsync`.
- `gf_load_tensor()` on a missing tensor name → `[GGUF] FATAL: tensor '<n>' not found` then **`exit(1)`** (lines 164-168; a second differently-worded FATAL for meta-context misses is at :172-175). The whole ace-server process dies; Node respawns it.
- Small tensors (norms, biases, scale_shift_table, etc.) are converted to F32 at load time (`gf_load_tensor_f32`).

## 4. Safetensors handling

- `engine/src/safetensors.h` — minimal read-only mmap parser (8-byte LE header length + JSON header). Dtypes F32/BF16/F16.
- `engine/src/weight-source.h` — format-agnostic layer over GGUF/safetensors. GGUF tensor names are canonical; safetensors may need a `name_prefix` (e.g. `model.` for the text encoder). **Shape order differs**: GGUF stores ggml order (ne[0] innermost), safetensors stores PyTorch order; `ws_shape()` normalizes to ggml order. `st_multi_open()` tries `model.safetensors`, then sharded `model.safetensors.index.json`, then diffusers `diffusion_pytorch_model.safetensors` (mirrored by `convert.py:find_sf_files`).
- `README.md` (Status table) confirms: safetensors DiT/LM/Text-Enc/VAE all loadable; BF16 safetensors bit-perfect vs BF16 GGUF; adapters work with both base formats.

## 5. ModelStore / VRAM policy (`engine/src/model-store.h`)

- `EVICT_STRICT` (default): at most one GPU module resident. `require`/`release` refcounting; a conflicting `require` while another module has refcount > 0 **asserts** (documented at lines 37-41; the EvictPolicy enum is at :105-108) — that is a caller programming error, not a recoverable state.
- `EVICT_NEVER` (`--keep-loaded`): nothing evicted, everything accumulates. Node passes the flag when `ACESTEPCPP_KEEP_LOADED=1` (`config.ts:132`, `index.ts:181-184`).
- Invariant under both policies: exactly ONE LM instance process-wide, shared between generate and understand via identical ModelKey (lines 17-22, 34-35).
- `ModelKey` (lines 83-103): `(kind, path)` plus LM extras (`max_seq`, `n_kv_sets`) and DiT extras: `adapter_path`, `adapter_scale`, `adapter_group_scales`, basin re-base `rebase_source`/`rebase_beta`, and multi-adapter `adapter_stack` signature. Any difference = a distinct cached DiT.
- `DiTMeta` (lines 112-118): CPU-cached config + `silence_full` (from `silence_latent`) + `null_cond_cpu` + `is_turbo`/`is_merge` flags — available before the DiT hits the GPU.
- Manual-unload UI hooks: `store_evict_lm` (line 155, used by Song Builder), `store_list_loaded` (line 161), `store_evict_label` (line 167 — labels like `"LM"`, `"DiT"`, `"VAE-Dec"`; skips in-use modules).
- Warm-on-startup: `ACESTEPCPP_WARM_DIT`/`_VAE`/`_ADAPTER`(+`_SCALE`) post `/warm` after boot, gated on keepLoaded (`config.ts:133-149`) — under EVICT_STRICT the warm would be dropped instantly.

## 6. quantize.cpp policy detail (`engine/tools/quantize.cpp`)

Usage: `quantize <input.gguf> <output.gguf> <type>` (line 7). Variant table (lines 40-54):

| Type | Base | Bump type | embed_tokens | Bump mode |
|---|---|---|---|---|
| Q2_K | Q2_K | Q4_K | Q6_K | first 4 layers |
| Q3_K_S | Q3_K | — | Q6_K | none |
| Q3_K_M | Q3_K | Q5_K | Q6_K | first + last + every 3rd |
| Q3_K_L | Q3_K | Q5_K | Q6_K | all important |
| Q4_K_S | Q4_K | Q5_K | Q6_K | first 4 layers |
| Q4_K_M | Q4_K | Q6_K | Q6_K | first + last + every 3rd |
| Q5_K_S | Q5_K | — | Q6_K | none |
| Q5_K_M | Q5_K | Q6_K | Q6_K | first + last + every 3rd |
| Q6_K | Q6_K | — | Q6_K | none |
| Q8_0 / NVFP4 / MXFP4 | as named | — | Q8_0 | none |

- "Important" tensors: `v_proj.weight` + `down_proj.weight` (S/M); L adds `o_proj.weight` (lines 74-82).
- Never quantized (`should_quantize`, lines 89-109): any tensor in a `vae` arch, 1-D tensors (promoted to F32 instead), text-enc `embed_tokens`, `silence_latent`, `scale_shift_table`, `null_condition_emb`.
- Streaming one-tensor-at-a-time write; low memory. Non-block-aligned tensors kept as-is.

## 7. Node download service internals (`server/src/services/modelDownloadService.ts`)

Data shapes (lines 60-92):

```ts
DownloadStatus = 'queued'|'downloading'|'paused'|'completed'|'failed'|'cancelled'
DownloadJob    = { jobId, fileId, filename, status, bytesDownloaded, totalBytes, speed, error? }
RegistryFile   = { id, filename, role, subdir?, repoPath?, displayName, scale, variant, quant,
                   sizeBytes, repo, description, tags[] }
```

- Target dir (lines 110-118): `role === 'runtime'` → engine dir (beside ace-server.exe); else `modelsDir[/subdir]`.
- Installed check (lines 177-208): filename presence with extensions `.gguf|.onnx|.safetensors|.bin` in models root + one subdir level, plus `.dll` in the engine dir. **Name-only — no size/hash.**
- `startDownload` dedupes: an existing queued/downloading job for the same fileId returns its jobId (lines 225-234).
- URL: `https://huggingface.co/{repo}/resolve/main/{repoPath || filename}` (lines 392-393). Relative redirects resolved against the original URL; max 5 redirects. HTTP 416 on a Range request treated as "already complete" (lines 500-505).
- Retries: 3 attempts with 0 / 2 s / 5 s delays (lines 352-353). Resume via `.part` file size → `Range: bytes=N-`.
- Validation before rename (`_validateDownload`, lines 429-473): size within ±5% of `sizeBytes`; `MZ` PE header for `.dll`; `GGUF` magic for `.gguf`. Failure deletes the file. **No content validation for `.onnx`/`.bin`/`.safetensors` beyond size.**
- Deletion (`deleteFile`, lines 300-339): extension allowlist `.gguf|.onnx|.safetensors|.bin|.dll`, path-traversal guards; DLLs deleted from engine dir, models from root + one subdir level.
- Variant filtering (getRegistry, lines 122-173): `.variant` and `.cuda-version` marker files beside ace-server.exe hide CUDA-only files/packs on vulkan/cpu builds; CUDA ≤ 12 shows the `cuda12-runtime` pack and remaps `cuda-rt-*` ids to `-12` variants inside packs.
- `cleanupJobs()` (lines 341-348): comment says "older than 60s" but it deletes ALL terminal jobs immediately — and nothing in `server/src` calls it.

## 8. Catalogue statistics (`server/src/data/model-registry.json`, verified 2026-07-29)

- **152 files** by role: 86 dit, 18 runtime, 13 stablestep, 13 vae, 8 lm, 5 supersep, 4 whisper, 3 pp-vae, 2 embedding.
- **9 packs**: `quick-start`, `minimal`, `xl-quality`, `blackwell`, `cuda-runtime`, `cuda12-runtime`, `stem-separation`, `stablestep-separation`, `supersep-runtime`.
- Repos (10 unique): `Serveurperso/ACE-Step-1.5-GGUF`, `ACE-Step/Ace-Step1.5`, `scragnog/ace-step-1.5-gguf-merge-models`, `scragnog/Ace-Step-1.5-MXFP4-Quants`, `scragnog/HOT-Step-CPP-PP-VAE`, `scragnog/Ace-Step-1.5-ScragVAE`, `scragnog/HOT-Step-CPP-SuperSep` (also hosts runtime DLLs), `scragnog/HOT-Step-CPP-StableStep`, `mdmachine/ACEStep-XL-Regrind-V1` (Regrind VAEs, `vae/` subfolder), `ggerganov/whisper.cpp`.
- Entry with **no `repo` field** (download would hit `https://huggingface.co/undefined/...` and fail): `vae-dreamvae-onnx`. Local/display-only. (The Regrind entries used to be in this bucket; since 2026-07-29 all 8 Regrind VAEs download from `mdmachine/ACEStep-XL-Regrind-V1` via `repoPath`.)

## 9. Model Manager API summary (`server/src/routes/modelManager.ts`, mount `/api/model-manager`)

| Route | Behavior |
|---|---|
| `GET /registry` | `{ packs, files: (RegistryFile & {installed})[], modelsDir, variant, cudaMajor }` |
| `POST /download` `{fileId}` | `{ jobId }` (dedupes active jobs) |
| `GET /downloads` | SSE: `data: {"jobs": DownloadJob[]}` on every progress event + 1 s interval |
| `POST /download/:jobId/cancel` | `{ ok }` |
| `POST /download/:jobId/resume` | `{ jobId }` |
| `DELETE /files/:filename` | `{ ok }` (extension allowlist, traversal guard) |

UI: `ui/src/components/model-manager/ModelCatalogueTab.tsx:18-28` defines 7 tabs — `dit | lm | embedding | vae | pp-vae | supersep | whisper`. Entry point is "Get More Models" in the Models dropdown (`ui/src/components/global-bar/ModelsDropdown.tsx`); on first launch with no models the Model Manager opens automatically.

## 10. convert.py specifics (`engine/convert.py`)

- Hardcoded I/O: `engine/checkpoints/` → `engine/models/` (lines 14-16); no CLI args. Neither dir exists in this working tree — create them before running, then move output to repo-root `<data root>/models/`.
- Classification by checkpoint directory name (lines 55-64): `acestep-5Hz-lm*` → `acestep-lm`, `acestep-v15*` → `acestep-dit`, `Qwen3-Embedding*` → `acestep-text-enc`, and the directory named **exactly `vae`** → `acestep-vae`. Anything else is skipped.
- Handles single-file, sharded-index, and diffusers safetensors layouts (`find_sf_files`, lines 31-45). Output GGUFs are self-contained (weights + config + tokenizer + silence_latent). Skips outputs that already exist.

## 11. Design-doc drift (code wins)

The original design doc (`docs/plans/2026-05-03-model-manager-design.md`, gitignored — may be absent) describes 5 roles / 5 repos / ~104 files / 5 UI tabs, pause-resume ETA fields, and `recommendedSettings` per entry. The shipped code has 9 roles, 10 repos, 152 files, 7 tabs, `role: runtime` + `subdir` + `repoPath` fields the doc lacks, and no `recommendedSettings` in actual entries. Always trust the code and the JSON over that doc.
