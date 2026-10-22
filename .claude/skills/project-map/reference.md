# Project Map — Deep Reference

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

Companion to [SKILL.md](SKILL.md). All paths repo-relative to the HOT-Step CPP root. Line numbers verified against `master` on 2026-07-02; they drift over time — treat as anchors, re-grep if a ref misses.

## 1. Engine source map (`engine/src/`, headers unless noted)

| Subsystem | Files |
|---|---|
| HTTP server + job queue | `engine/tools/hot-step-server.cpp` (the real ace-server binary); embedded webui built from `engine/tools/webui` + `tools/public/index.html.gz` via `xxd.cmake` |
| LM phase (Qwen3 → audio codes) | pipeline-lm.cpp/.h, qwen3-lm.h, qwen3-enc.h, bpe.h, prompt.h, metadata-fsm.h, sampling.h |
| DiT synth phase | pipeline-synth.cpp, **pipeline-synth-ops.cpp** (fork hook at line 9: `#include "hot-step-sampler.h"`), pipeline-synth-ops.h, pipeline-synth-impl.h, dit.h (fork hooks at lines 11/13: adapter-merge.h + adapter-runtime.h), dit-graph.h, dit-sampler.h (upstream, **bypassed**), denoiser.h |
| HOT-Step sampler layer | **hot-step-sampler.h** (routes ALL solver/scheduler/guidance selection), hot-step-params.h (server→sampler sideband param channel, hooked into model-store.h:53), sampler-schedule.h, sampler-repaint.h, sampler-dcw.h, dcw.h, dwt-haar.h |
| Native solver/scheduler/guidance registries (LEGACY) | `engine/src/solvers/` (solver-euler.h, solver-heun.h, solver-dpm.h, solver-unipc.h, solver-rk4.h, solver-sde.h, solver-stork.h, solver-dopri.h, solver-jkass.h, solver-gl2s.h, solver-aflops.h, solver-rfsolver.h, solver-interface.h, solver-registry.h, …), `engine/src/schedulers/`, `engine/src/guidance/` — new work goes in Lua plugins instead |
| Lua plugin system | lua-plugin.h, lua-plugin-registry.h (`init()` at :35 scans `<engine_dir>/plugins/{solvers,schedulers,guidance,postprocess}` AND `<project_dir>/plugins/...` — project overlay can add/shadow) |
| Adapters (LoRA/LoKr) | adapter-merge.h, adapter-runtime.h, adapter-cancel.h, adapter-trt.h, weight-source.h, weight-ctx.h |
| VAE | vae.h, vae-enc.h, vae-ort.h, vae-enc-ort.h (ONNX Runtime / TensorRT path) |
| Model loading/eviction | model-store.cpp/.h (EVICT_STRICT default vs EVICT_NEVER under `--keep-loaded`), model-registry.h, gguf-weights.h, safetensors.h |
| Conditioning / tokenizers | cond-enc.h, cond-enc-ort.h, text-enc-ort.h, fsq-tok.h, fsq-detok.h |
| Stem separation (ONNX) | supersep.cpp/.h, supersep-stft.h |
| Understand / analysis | pipeline-understand.cpp/.h |
| Streaming preview | stream-pipeline.h (ring-buffer preview WAVs; request fields `stream_mode/depth/chunk_dir`) |
| Post-DSP | mastering.h, spectral-lifter.h, silence-latent.h |
| Lyrics timing (LRC) | lrc-alignment.h, dit-alignment-graph.h, alignment-config.h |
| Request parsing | request.cpp/.h, config-json.h, task-types.h |
| TensorRT paths | dit-trt.h, lm-trt.h, lm-trtllm.h, hot-step-sampler-trt.h, adapter-trt.h |
| Misc infra | backend.h, audio-io.h, audio-resample.h, wav.h, philox.h, timer.h, debug.h, hot-step-build-flags.h |
| Upstream-sync backup | `engine/src/_backup_pre_sync/` — snapshot copies, not compiled |

**Other binaries** (`engine/tools/`): ace-lm.cpp, ace-synth.cpp, ace-understand.cpp, neural-codec.cpp, mp3-codec.cpp, quantize.cpp, vst-host.cpp, mastering.cpp, plus synth-batch-runner.h. `ace-server.cpp` = uncompiled upstream reference (`engine/CMakeLists.txt:416-419`).

## 2. Engine HTTP API (hot-step-server.cpp, registration lines)

| Endpoint | Line | Purpose |
|---|---|---|
| `POST /lm` | 2704 | LM phase — returns job id; result is array of enriched AceRequests with `audio_codes` |
| `POST /synth` | 2705 | DiT + VAE synth (multipart variant carries source/ref/seed latents & audio) |
| `POST /understand` | 2706 | Audio analysis (pipeline-understand) |
| `POST /vae` | 2707 | Raw VAE encode/decode |
| `POST /warm` | 2708 | Preload DiT+VAE+adapter (used by warm-on-startup, index.ts ~487+) |
| `GET /health` | 2709 | Liveness (unanswerable during compute — single-threaded) |
| `GET /props` | 2712 | Model/adapter lists + CLI defaults (feeds `/api/models`) |
| `GET /logs` | 2713 | Engine log tail |
| `GET /jobs` | 2714 | Job list |
| `GET /plugins` | 2716 | Lua plugin registry dump (feeds `/api/plugins`) |
| `GET /vram` | 2722 | cudaMemGetInfo (feeds `/api/logs/vram`) |
| `GET /models/loaded` | 2744 | Resident model list |
| `POST /models/unload` | 2762 | Evict models |
| `GET /job?id=` | 2783 | Poll job status (`status`, `phase` e.g. `adapter_precompute`, `phase_step/total`) |
| `POST /job` | 2828 | Fetch job result (large payload — 300 s timeout in aceClient) |
| `POST /pp-vae-reencode` | 2852 | Post-processing VAE re-encode |
| `POST /supersep/separate` | 3149 | ONNX stem separation |
| `GET /supersep/progress` | 3271 | Separation progress |
| `GET /supersep/result` | 3313 | Separation result |
| `GET /supersep/serve` | 3357 | Serve stem file |
| `POST /supersep/recombine` | 3397 | Recombine stems |
| `POST /spectral-lifter` | 3494 | Spectral lifter DSP |
| `GET /` | 3535 | Embedded webui (gzip HTML) |

## 3. AceRequest field catalogue (aceClient.ts:38-151)

Grouped; see the interface for exact optionality and comments.

- **Core**: `caption, lyrics, bpm, duration, keyscale, timesignature, vocal_language, seed`
- **LM sampling**: `lm_batch_size, lm_temperature, lm_cfg_scale, lm_cfg_cutoff_ratio, lm_top_p, lm_top_k, lm_negative_prompt, use_cot_caption, audio_codes` (LM output)
- **DiT**: `inference_steps, guidance_scale, shift, infer_method` (solver), `scheduler, guidance_mode, batch_cfg, synth_batch_size, peak_clip, negative_prompt`
- **Task routing**: `task_type, track, repainting_start/end, audio_cover_strength, cover_noise_strength/method, seed_strength, evict_lm`
- **Model routing**: `synth_model, lm_model, vae_model, emb_model`
- **Adapters**: `adapter, adapter_scale` (single); `adapters: [{name, scale}]` (multi-stack, supersedes single); `adapter_sections: [{weights, size}]` + `adapter_section_align_at` + `adapter_section_isolation` (per-section masking — isolation reverted in practice); `adapter_group_scales` (self_attn/cross_attn/mlp/cond_embed/time_embed/proj_in); `adapter_mode: "merge"|"runtime"`, `adapter_runtime_quant: "bf16"|"q8_0"|"q4_k"`; `rebase_source, rebase_beta` (basin re-base, UNVALIDATED)
- **Solver/guidance sub-params**: `stork_substeps, beat_stability, frequency_damping, temporal_smoothing, apg_momentum, apg_norm_threshold`
- **DCW** (Differential Correction in Wavelet domain): `dcw_enabled, dcw_mode, dcw_scaler, dcw_high_scaler`
- **Latent post**: `latent_shift, latent_rescale, custom_timesteps, cfg_cutoff_ratio, cache_ratio`
- **Post-VAE denoise**: `denoise_strength, denoise_smoothing, denoise_mix`; `pp_vae_reencode`
- **Misc**: `get_lrc` (synced lyrics), `vae_chunk`, `use_ort_vae`
- **Plugins**: `plugin_params: Record<string, string|number|boolean>`, `postprocess_plugin`
- **Streaming**: `stream_mode, stream_depth, stream_chunk_dir`

Fields the C++ struct does not know about (the **sideband**) are dropped by the `/lm` echo — see SKILL.md golden gotcha and `generate.ts:312-328`.

## 4. Server internals worth knowing

- **Route mounts**: index.ts:72-95. Static: `/audio` (98), `/references` (111), `ui/dist` + SPA fallback (123-140).
- **Engine spawn**: `startAceServer()` index.ts:158; args 166-213; env injection (CUDA_VISIBLE_DEVICES + TensorRT/TRT-LLM lib dirs prepended to PATH with case-insensitive Windows PATH-key handling) 219-258; stdout/stderr piped to console + `ace_engine.log` + `pushLog` with GGML-noise filtering 262-282; crash limiter 284-308 (3 crashes / 30 s window).
- **Portable CUDA bootstrap**: index.ts:318 onward. `.cuda-version` marker next to the exe selects CUDA 12 vs 13 DLL set; downloads via `modelDownloadService`. Offline → CPU-only start.
- **Warm-on-startup**: gated on `keepLoaded && warmDit` (config.ts:132-141); polls `/health` then POSTs `/warm`.
- **generate.ts anatomy** (~1550 lines): job type + `aceJobId` mapping :47; `jobs` Map :79 (TTL 1 h :83); `pollUntilDone` :102 (500 ms poll, 2-min stall, 5–120 min wall clamp default 45); `runGeneration` :173; LM submit :303; LM-echo rebuild :312-328; LM cache write :331-343; source audio/latent/seed-latent load :449-478; deferred parallel tasks (whisper, cover art) :591+; per-track synth loop with `synthReq` :647+; multipart submit :814-826; whisper transcription :940+; auto-trim :1057-1094; serial queue + retry (MAX_RETRIES=1) :1291+.
- **aceClient.ts**: `BASE = config.aceServer.url`; timeouts :17-19 (15 s / 30 s / 300 s); `AceProps` :22; `AceRequest` :38; job status types :154+.
- **Post-processing** (`services/generation/postProcessing.ts`): `runPostProcessingChain` :70; stage order QE(unmastered) :105 → PP-VAE :135 → Spectral Lifter :151 → Vocal Naturalizer :171 → gain offset :198 → VST chain :248 → Mastering :264 → LUFS :282 → QE(mastered) :303. `parallelQualityEval` runs pre-QE concurrently with PP-VAE.
- **logs.ts**: `pushLog` :34, `subscribeLines` :49 (in-memory ring feeding terminal panel), `GET /` :56, `GET /vram` :100, `GET /models-loaded` :120, `POST /models-unload` :133.
- **stemStudio.ts**: two modes documented in header :1-8 — `extract` (generative, sequential DiT `/synth` per track) and `supersep` (ONNX NN). Both persist to `<data root>/library/stems/<jobId>/`.

## 5. Config / env (`server/src/config.ts`)

- Portable mode: `HOT_STEP_ROOT` env → `PROJECT_ROOT` (config.ts:13-16). Dev: two levels up from `server/src/`.
- `.env` bootstrapped from `.env.example` on first launch (:20-30).
- ace-server exe auto-located: `engine/ace-server.exe` (portable) → `engine/build/Release/` (VS) → `engine/build/` (Ninja) → `engine/build/Debug/` (:46-52). Override: `ACESTEPCPP_EXE`.
- Defaults: models `<data root>/models/`, adapters `<data root>/adapters/`, port 8085, host 127.0.0.1, vaeChunk 1024, vaeOverlap 64 (:100-105). Node server: port 3001, host 0.0.0.0 (:186-187).
- ffmpeg: portable `server/ffmpeg.exe` else `ffmpeg-static` npm package (:68-93). Essentia: `Essentia/essentia_streaming_extractor_music.exe` or `ESSENTIA_BIN` (:181). Whisper: `tools/whisper/whisper-cli.exe` + `<data root>/models/whisper/` (:254-255). vst-host.exe lives next to ace-server.exe (:236-240).
- Lireek LLM providers (:202-231): gemini (default), openai, anthropic, ollama, lmstudio, unsloth, llamacpp, openai-compat — impls in `server/src/services/lireek/llm/`.
- Settings UI whitelist `EXPOSED_ENV_KEYS` :265-286; restart-required set :289-294; hot-reload `reloadEnvConfig()` :300.

## 6. UI structure

- Component folders (`ui/src/components/`): assistant, cover-studio, create, details, global-bar, insta-gen, library, lyric-studio, model-manager, player, playlist, repaint-studio, settings, shared, sidebar, song-builder, stem-builder, stem-studio, terminal.
- API wrappers (`ui/src/services/`): api.ts (generic `/api` fetch), assistantApi.ts, inspireApi.ts, lireekApi.ts, stemStudioApi.ts, supersepApi.ts.
- Stores (`ui/src/stores/`): globalParamsStore.ts (all generation params — the big one), playbackStore.ts (+playbackConverters.ts), audioGenQueueStore.ts, streamingStore.ts, vstChainStore.ts, abCompareStore.ts, discoStore.ts.
- View routing: `ui/src/App.tsx:99-114` maps view state → URL paths.
- Dev proxy: `ui/vite.config.ts` — port 3000, host 0.0.0.0, proxies `/api`, `/audio`, `/references` → `http://127.0.0.1:3001`.

## 7. Status ledger (as of 2026-07-02)

- **DONE**: everything in the feature table; multi-adapter stacking (`adapters[]`, fixes in commit `168dcb5`); per-section adapter masking P1 (proportional map) + P2 (alignment-anchored token map); Lua plugin system (21 solvers / 9 schedulers / 7 guidance shipped in `engine/plugins/`, postprocess in root `plugins/postprocess/`); safetensors + GGUF dual-format loading; streaming preview pipeline; ONNX/TensorRT VAE path.
- **DISABLED (deliberate)**: draft-LM speculative decoding (config.ts:117-122).
- **REVERTED**: regional self-attn isolation for per-section masking (commit `ee041e1` — broke musical continuity).
- **UNVALIDATED**: LoKr cross-base basin nudge β·(S−T) (`rebase_source`/`rebase_beta`, aceClient.ts:112-113).
- **Unverified claims** (check before relying): whether root `plugins/` ships in release zips vs being a purely local overlay; contents of `docs/plans/upstream-sync-workflow.md` (gitignored, may be absent).
