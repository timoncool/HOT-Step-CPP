# Reference: Generation Request Flow — deep detail

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

Companion to [SKILL.md](SKILL.md). All line numbers verified against `master` (HEAD `168dcb5`, 2026-07-02) — re-verify after major refactors.

## 1. UI tier — where the request is born

- **Create tab:** `ui/src/components/create/CreatePanel.tsx` collects content (caption/lyrics/instrumental/metadata) → `onGenerate(partialParams)` → `App.tsx:502` `handleGenerate` merges `useGlobalParamsStore.getState().getGlobalParams()` (engine knobs) with content params and settings flags, then calls `enqueueSimpleGen()` (`ui/src/stores/audioGenQueueStore.ts`) → `generateApi.submit()` → `POST /generate` (`ui/src/services/api.ts:148-153`).
- **Zustand store** (`ui/src/stores/globalParamsStore.ts`): every knob persisted per-field to localStorage under `hs-*` keys (e.g. `hs-apgMomentum` line 96). `getGlobalParams()` (line 324) is the single assembly point. Notable conditional gates (a gated param is simply `undefined` in the payload):
  - `apgMomentum` / `apgNormThreshold` only when `guidanceMode === 'apg'` (:396).
  - `storkSubsteps` only for `inferMethod === 'stork2' | 'stork4'` (:392).
  - `rebaseSource`/`rebaseBeta` only with a primary adapter AND `adapterMode === 'merge'` (:381-382).
  - Adapter stack only when `advancedAdapters` is true (:334) — a stale `false` here suppresses a persisted stack.
  - `triggerWords` derived from adapter filenames when `triggerUseFilename` (:357-362).
- Other studios submit their own shapes to the same `/api/generate` endpoint: `CoverStudio.tsx`, `RepaintStudio.tsx`, `SongBuilder.tsx`, `StemBuilder.tsx`, `InstaGenPanel.tsx`, plus the lyric-studio queue in `audioGenQueueStore.ts`.
- UI polls `GET /api/generate/status/:id`; streaming mode uses SSE `GET /api/generate/stream/:id` (generate.ts:1508).
- Canonical UI shape: `GenerationParams` (`ui/src/types.ts:80`) — camelCase; includes post-processing-only fields (whisper, mastering, gain, …) that never reach the engine.

## 2. Node tier — routes/generate.ts walkthrough

Endpoints (all under `/api/generate`):

| Route | Line | Purpose |
|---|---|---|
| `POST /` | :1355 | Submit; 503 if engine not ready; stores `req.body` verbatim as `job.params` (untyped `any`) |
| `GET /status/:id` | :1391 | Poll job (adds `ace_phase` / phase progress from engine) |
| `POST /cancel/:id` | :1409 | Cancel one |
| `POST /cancel-all` | :1425 | Cancel all active |
| `GET /queue` | :1443 | Queue health |
| `POST /reset-queue` | :1469 | Force-reset: cancel everything, drain queue |
| `GET /stream/:id` | :1508 | SSE for streaming mode (`status`/preview/done/error events) |

**Queue:** strictly serial — one `runGeneration` at a time, because `subscribeLines()` is a global engine-log pub/sub with no job tagging. One automatic retry with a fresh random seed on transient failure (job status reset to `'pending'` at :1317).

**`runGeneration` sequence** (starting ~:173):

1. `translateParams(job.params)` → `aceReq` (:191). Resolved seed written **back** into `job.params.seed` (:195-197) so the DB row is reproducible even with `randomSeed`.
2. LM decision (:207-210): `needsLm = !skipLm && !aceReq.audio_codes && !isCoverTask`, where `isCoverTask = ['cover','cover-nofsq','repaint','lego','extract'].includes(task_type)` (:208). If `skipLm` on a non-cover task, defaults filled: bpm 120, duration 120, `'C major'`, timesig `'4'` (:221-224).
3. LM phase: cache check (`computeLmCacheKey`, default on, disable with `cacheLmCodes:false`) → `aceClient.submitLm()` (:303) → `pollUntilDone` → fetch JSON array of enriched AceRequests (:309-310) → **sideband rebuild** (:319-328 fresh, :237-246 cache-hit — see SKILL.md gotcha) → cache store of the 7 LM fields only (:332-344).
4. **Trigger-word re-injection** (:366-403): the LM's CoT caption replaces the original caption, losing the trigger word `translateParams` injected. Re-injected here on both cache-hit and fresh paths. `replace` placement overwrites the caption; `prepend`/`append` inject only triggers not already present (multi-adapter dedup, :384-397). Requires `triggerPlacement` AND `loraPath`; otherwise a `[Trigger]` WARNING is logged and injection is skipped (:399-403).
5. Batch expansion (:411-422): when LM was skipped/cached-as-1, clones the template with fresh random seeds up to `batchSize`. The clone is a spread of the template, so sideband survives.
6. **Per-track synth loop** (one `/synth` call per track — deliberately avoids the engine's `multipart/mixed` multi-track response):
   - Source prep (~:446-530): cover tasks load source audio, apply tempo/pitch, VAE-encode via engine `/vae` with a disk latent cache (`sourceLatentCache.ts`, keyed on path+tempo+pitch+vae_model, :491-497); timbre reference likewise.
   - Format selection (:549-566): `wav32` if any post-processing active, `source === 'builder'`, or mastering-with-reference; else `wav16`. wav16 peak-normalizes to 0 dBFS in the engine; wav32 skips normalization (headroom for PP; stable levels across Song Builder appends).
   - Multipart decision (:814-826): multipart iff any of `srcAudioBuf | refAudioBuf | srcLatentBuf | refLatentBuf | seedLatentBuf` present → `submitSynthMultipart`; else plain-JSON `submitSynth`.
7. After synth: audio saved to `<data root>/library/audio/<uuid>.wav|mp3` (`config.data.audioDir`, :840); LRC lyrics from base64 `x-lrc-text` response header (:924); post-DiT latent fetched and stored alongside (~:1023).
8. Auto-trim (before PP; needs `autoTrimEnabled` + `durationBuffer`, corrects duration metadata, :1057-1094) → post-processing chain (`postProcessing.ts`; raw WAV never modified, PP runs on a copy) → **SQLite insert** per track (:1152): `INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url, duration, bpm, key_scale, time_signature, tags, dit_model, generation_params, ...)` — `generation_params` = `JSON.stringify(job.params)` (full UI params incl. resolved seed), `style` = the user's original style input, `caption` = the LM CoT caption.

**Watchdogs** (`pollUntilDone`, ~:100-140):
- Stall: 2 min (`STALE_TIMEOUT_MS = 120_000`, :107) without stage/progress change → cancel + `"Generation stalled — no progress for Ns"`.
- Wall clock: `generationTimeoutMinutes` clamped [5,120], default 45 → `"Generation timed out (N min limit)"` (:139).
- Transient poll HTTP errors are swallowed and retried.

## 3. translateParams special cases (translateParams.ts)

- Caption fallback chain: `params.prompt || params.songDescription || params.caption || params.style || ''` (:13).
- `instrumental: true` → `lyrics = '[Instrumental]'` (:17-18) — there is no instrumental flag on the wire.
- `duration` gets `durationBuffer` added **only to `req.duration`, not `job.params.duration`** (:25-28) — so the DB stores the user's requested duration; auto-trim later removes the buffer from the audio.
- `timeSignature '4/4'` → `'4'` (:30-33).
- `loraStack` → `adapters[]` via `mapPath()`; supersedes single `adapter` (:88-95).
- Per-section `[Section]{k=v}` lyric directives (`adapterSections.ts`): parsed into `adapter_sections` + forced `adapter_mode='runtime'` **only with a 2+ adapter stack** (:104-117); otherwise directives are **stripped** so they don't reach the LM as garbage tokens (:118-123).
- Rebase only when `loraPath && rebaseSource && rebaseBeta` all truthy (:126-129).
- Falsy-guard fields where 0 intentionally means "let the LM fill it": `bpm`, `duration`, `inferenceSteps`, `batchSize`. Fields where 0/false matter use `!== undefined` (e.g. `guidanceScale`, `shift`, `seedStrength`, `dcwEnabled`).

## 4. aceClient.ts — the wire

- `AceRequest` interface (:38) = Node-side canonical superset (C++ struct fields + all sideband). This, plus `ServerFields`, is the authoritative request schema — ARCHITECTURE.md's JSON reference is stale.
- Engine is **single-threaded httplib**: during heavy compute it cannot respond. Timeouts: `TIMEOUT_QUICK` 15 s (submit/health), `TIMEOUT_POLL` 30 s, `TIMEOUT_RESULT` 300 s (:17-19).
- `submitLm(request, mode?, keepLoaded)` (:319) — mode `inspire`/`format` optional; `keepLoaded` → `?keep_loaded=1`.
- `submitSynth(request | request[], format, keepLoaded)` (:332); `submitSynthMultipart(...)` (:347) — part names `request` (JSON), `audio`, `ref_audio`, `src_latents`, `ref_latents`, `seed_latents`. **Multipart sends only `request[0]` when given an array** (:376).
- Fine-grained engine phases (`AceJobPhase`, :163-180, e.g. `adapter_precompute`, `dit_inference`) mirror `job_phase_str()` in hot-step-server.cpp and surface via `/api/generate/status/:id`.
- Engine job status ints: 0=running, 1=done, 2=failed, 3=cancelled.
- Note: the engine `/vae` handler and CLI tools use the C++ struct's `vae` field (request.h:142), while server-mode selection uses the sideband `vae_model` — different names for adjacent concepts.

## 5. Engine tier — hot-step-server.cpp

- `POST /synth` (`handle_synth`, :1442): parses the SAME JSON body twice — `parse_server_fields()` for sideband, `request_parse_json[_array]()` for the upstream struct. Diagnostics: prints first 300 chars of the body as `[DIAG] /synth body` (:1568). Query params: `?format=wav16|wav24|wav32|mp3` (:1429,:1586), `?keep_loaded=1` (co-resident; one-way flip of the model store to never-evict, :1115-1116). Returns `{"id":"N"}` immediately; work runs on the single GPU work queue.
- `POST /lm` (`handle_lm`, :888): modes `generate` (metadata + lyrics + audio codes) / `inspire` / `format`. Result body: JSON array of enriched AceRequests via `request_to_json` (:869) — serialization is **sparse** (defaults omitted).
- **`ServerFields`** (:557-604) — the definitive sideband-only field list: `vae_model`, `emb_model`, `solver_name` (JSON key `infer_method`), `scheduler`, `guidance_mode`, `apg_momentum`, `apg_norm_threshold`, `stork_substeps`, `beat_stability`, `frequency_damping`, `temporal_smoothing`, `group_scales` (`adapter_group_scales`), `adapter_mode`, `adapter_runtime_quant`, `adapter_section_align_at`, `adapter_section_isolation` (dead knob — see SKILL.md), `rebase_source`, `rebase_beta`, `dcw_*`, `latent_shift`, `latent_rescale`, `cfg_cutoff_ratio`, `cache_ratio`, `custom_timesteps`, `denoise_strength/smoothing/mix`, `plugin_params`, `seed_strength`, `seed_latents` (multipart part, not JSON), `evict_lm`, `vae_chunk`, `batch_cfg`.
- `parse_server_fields` (:606) resets defaults first: solver `euler`, guidance `apg`, adapter_mode `merge` (:607-618). **yyjson trap** (:700-705): bare `yyjson_get_real()` yields 0 for JSON ints; `get_num` lambda handles both — used for `adapter_group_scales` (:709-719) and `denoise_strength`/`denoise_smoothing`/`denoise_mix` (:763-769, with the in-code NOTE at :761); `rebase_beta` uses the same pattern inline at :663. Fields still on bare `get_real`: `apg_momentum` (:667), `apg_norm_threshold` (:670), `seed_strength` (:674), `beat_stability`, `frequency_damping`, `temporal_smoothing`, and the DCW/latent floats further down.
- **`synth_worker`** (:949):
  - Resolves DiT/VAE/text-enc from the model registry; unknown `vae_model`/`emb_model` **fall back to default with only a stderr warning**.
  - Adapter stack: `adapters[]` supersedes single `adapter` (single is folded into a 1-element stack); unknown adapter name tries absolute path, else the **job fails**. Resolved stack lives in `g_hotstep_params.adapters` (:1067,:1097).
  - Copies every ServerFields value into **`g_hotstep_params`** (:1132-1194) **before** `ace_synth_load()` — adapter merge reads group scales at merge time.
  - Per-section masking forces runtime mode engine-side too (needs `adapter_sections` non-empty AND stack ≥ 2, :1150-1162).
  - `rebase_source` name resolved through the registry; unknown → cleared with a warning (:1169-1177).
  - Auto-shift when `shift == -1`: adaptive from duration + step count (:1244-1250).
  - Batch: `synth_batch_size` expands per-seed variants; total clamped to DiT max 9 (:964-973).
  - Post-decode: peak-normalize (skipped for wav32), spectral denoiser, WAV/MP3 encode; multi-request results return `multipart/mixed` — Node sidesteps this with one track per call.
- **`hot-step-params.h`**: `HotStepParams` global (`g_hotstep_params`) is safe as a global because the GPU worker is single-threaded (header comment, :9-11). Also defines group classification `adapter_determine_group()` — **`cross_attn` must be checked before `self_attn`** (:60-63, substring matching) — and the adapter stack cache-key signature (gap-fixes landed in `168dcb5`).

## 6. LM cache internals (lmCache.ts)

- `LmCacheEntry` (:9-17): exactly the 7 LM output fields — `audio_codes`, `caption`, `lyrics`, `bpm`, `duration`, `keyscale`, `timesignature`.
- Key (`computeLmCacheKey`, :23-46): sha256/16 over seed, caption, lyrics, bpm, duration, keyscale, timesignature, vocal_language, lm_model, lm_batch_size, lm_temperature, lm_cfg_scale, lm_top_p, lm_top_k, lm_negative_prompt, use_cot_caption. **DiT/adapter/DCW params deliberately excluded** from both key and value.
- In-memory `Map`, LRU max 20 (`LM_CACHE_MAX`, :19); timestamp refreshed on hit.
- Consequence: changing DiT-only params reuses cached codes (intended). Adding a new **LM-affecting** param without extending this key serves stale codes — checklist item #6 in SKILL.md.

## 7. Status / history ledger

- **DONE:** sideband rebuild on both LM paths (hardened in `8ea519b`/`168dcb5`); multi-adapter stack + per-section masking (P1 proportional map, P2 alignment via `adapter_section_align_at`); adapter stack cache-key signature; LM cache; latent capture; streaming SSE + engine ring buffer; serialized queue + retry + stall/wall-clock watchdogs; forced wav32 for builder/PP.
- **REVERTED:** regional self-attn isolation (`0f3bf6d` → `ee041e1`, broke musical continuity). `adapter_section_isolation` remains as zombie plumbing (parsed :573/:656, copied :1145, consumed nowhere in `engine/src/`).
- **UNVALIDATED:** basin re-base β·(S−T) nudge for cross-base LoKR adapters — fully plumbed, effectiveness unproven.

## 8. Known doc drift (code wins)

- ARCHITECTURE.md:355-362 model-compat table says `lego` uses the LM; Node's `isCoverTask` list (generate.ts:208) skips the LM for lego. `complete` is not in that list and would run the LM via `/api/generate`.
- ARCHITECTURE.md "shift 0 = auto" describes upstream; this fork's auto path is `shift == -1` (hot-step-server.cpp:1244-1250).
- ARCHITECTURE.md request-JSON reference predates `scheduler`, `guidance_mode`, `adapters[]`, `plugin_params`, and other sideband fields — authoritative schema is `AceRequest` in aceClient.ts + `ServerFields` in hot-step-server.cpp.
