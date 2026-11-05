# Reference: Generation Request Flow — deep detail

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

Companion to [SKILL.md](SKILL.md). All line numbers verified against `master` (HEAD `dda5f6b6`, 2026-10-23); re-verify after major refactors.

## 1. UI tier: where the request is born

- **Create tab:** `ui/src/components/create/CreatePanel.tsx` collects content (caption/lyrics/instrumental/metadata) → `onGenerate(contentParams, contentIntent)` → `App.tsx:705` `handleGenerate` merges `useGlobalParamsStore.getState().getGlobalParams()` (engine knobs) with content params and app settings (`coResident`, `cacheLmCodes`, `parallel*`), then calls `enqueueSimpleGen()` (`ui/src/stores/audioGenQueueStore.ts:1424`).
- **Resolve preview:** `generateApi.resolveIfSelected(intent)` (`ui/src/services/api.ts:198`) reads `GET /api/resolve/path` (`'resolved'` unless the server sets `GENERATION_INTENT_PATH=old`; a 404 from an older server means the old path) and posts the `CreateIntent` to `POST /api/resolve/preview` (`server/src/routes/resolve.ts`). It returns `{ request, provenance, warnings, uiEffects, version }`; `request` is the exact body to submit and carries `expectedBackend`.
- **Submit:** `queueOwner()` (`audioGenQueueStore.ts:259`, localStorage `lireek-audio-queue-owner-v1`) picks the path. `'server'`: `audioQueueApi.enqueue()` → `POST /api/audio-queue/items`, and the durable intent queue (`server/src/services/audioQueue/intentQueue.ts`) submits it through `submitGeneration()`. `'browser'`: `generateApi.submit()` → `POST /api/generate`, then the store polls status itself.
- **Zustand store** (`ui/src/stores/globalParamsStore.ts`): every knob persisted per-field to localStorage under `hs-*` keys (e.g. `hs-apgMomentum`, :397); `inferMethod`, `scheduler`, `guidanceMode`, `pluginParams` and `backendParams` are stored per backend (`BACKEND_SCOPED_FIELDS`, :219). `getGlobalParams()` (:741) is the single assembly point. Notable conditional gates (a gated param is simply `undefined` in the payload):
  - `apgMomentum` / `apgNormThreshold` only when `guidanceMode === 'apg'` (:878-879).
  - `storkSubsteps` only for `inferMethod === 'stork2' | 'stork4'` (:874).
  - `rebaseSource`/`rebaseBeta` only with a primary adapter AND a chosen source (:854-855), in merge or runtime mode.
  - Adapter stack only when `advancedAdapters` is true (:751): a stale `false` here suppresses a persisted stack.
  - `triggerSpecs` derived from adapter filenames when `triggerUseFilename` (:787-797); the server prefers a trigger embedded in the adapter's metadata.
- The same gating exists server-side in `resolveGenerationParams()` (`server/src/services/generation/defaults.ts:155`) for compact `generation-intent/1` bodies. Keep the two in step.
- Other entry points reach the same `submitGeneration()`: Cover Studio submits with `generateApi.submit()` and through its cover workflow (`server/src/services/workflows/coverWorkflow.ts:298`); Song Builder sections (`server/src/services/songBuilder/sectionOps.ts:328`), Insta-Gen (`instaGenWorkflow.ts:166`), Repaint Studio and Stem Builder (`repaintLayerWorkflow.ts:105`) enqueue captured bodies on the audio intent queue; Lyric Studio renders go through the queue store's written-song path.
- UI polls `GET /api/generate/status/:id`. The old ACE SSE stream (`/stream/:id`) was removed (`1b9d6510`); MM3 "play while rendering" pipes audio from `GET /api/generate/mm3/stream/:id`.
- Canonical UI shape: `GenerationParams` (`ui/src/types.ts:108`): camelCase; includes post-processing-only fields (whisper, mastering, gain, …) that never reach the engine.

## 2. Node tier: routes/generate.ts and the ACE backend

Endpoints (all under `/api/generate`, `server/src/routes/generate.ts`):

| Route | Line | Purpose |
|---|---|---|
| `POST /` | :476 | Submit through `submitGeneration` (:344); stores `structuredClone(submission)` as `job.params` (untyped `any`) |
| `GET /status/:id` | :483 | Poll job (adds `attempts`, `ace_phase` / `ace_phase_progress`, YuE2 `batch`, `mm3_*` fields); falls back to the SQLite job ledger once the job is gone from memory |
| `POST /cancel/:id` | :627 | Cancel one (`cancelGenerationJob`, :612) |
| `POST /cancel-all` | :633 | Cancel all active |
| `GET /queue` | :651 | Queue health (depth, lane owner, current job) |
| `POST /reset-queue` | :675 | Force-reset: fail every active job, drain the GPU lane |
| `POST /yue2/plan` | :281 | YuE2 score preview on the GPU lane |
| `GET /mm3/stream/:id` | :541 | Pipe of a running MM3 render's audio |
| `POST /storm/stream`, `/storm/stop`, `GET`/`POST /storm/control` | :785, :986, :994, :1048 | STORM continuous slot streams (no LM phase, no song row) |

**`submitGeneration` checks, in order** (:344-464): 503 while a training preprocess has suspended the engine or the engine is still booting; 401 without a user; 400 for a malformed `backendOverride`; 409 when `expectedBackend` differs from the backend the job would run on; a `generation-intent/1` body is expanded by `resolveGenerationIntent()` (`services/generation/intent.ts`); 400 when a timbre reference file is missing (ACE only, #162); a YuE2 Prepare document is checked and stripped; `buildEnvelope()` (`services/generation/envelope.ts:99`) freezes the backend (active, or `backendOverride`), the resolved request and the retry policy; the job goes into the in-memory map and the ledger, then `enqueueGeneration()`.

**Queue:** strictly serial on the shared GPU lane (`services/generation/gpuLane.ts`, `runOnGpuLane`), because `subscribeLines()` is a global engine-log pub/sub with no job tagging and post-processing re-runs use the same lane. `runGeneration` (:156) evicts every other backend family's models (`evictOtherFamilies`) before calling `backend.generate()`. Retry (:203-263): up to the envelope's `policy.retry.maxAttempts` (ACE: 2, reseeding on retry), but only for errors thrown out of `backend.generate()`; a returned outcome, including a failure the backend recorded itself, ends the retry scope. `runAceGeneration` catches its own pipeline failures, so an ACE stall, timeout or engine failure is not retried.

**`runAceGeneration` sequence** (`server/src/services/backends/ace/generate.ts:26`):

1. `releaseMinimaxVramForAce()` (:49), then `translateParams(job.params)` → `aceReq` (:51). Resolved seed written **back** into `job.params.seed` (:58-60), and `job.params.lmSeed` set to the sent `lm_seed` or, when tied, the DiT seed (:61-65), so the DB row is reproducible even with `randomSeed`.
2. LM decision (:70-73): `needsLm = !skipLm && !aceReq.audio_codes && !isCoverTask`, where `isCoverTask = ['cover','cover-nofsq','repaint','lego','extract'].includes(task_type)` (:71). If `skipLm` on a non-cover task, defaults filled: bpm 120, duration 120, `'C major'`, timesig `'4'` (:84-87).
3. LM phase: cache check (`computeLmCacheKey`, default on, disable with `cacheLmCodes:false`) → `aceClient.submitLm()` (:170) → `pollUntilDone` → fetch the JSON array of enriched AceRequests (:176-177); an empty array is a failure (:185-190) → **sideband rebuild** (:199-213 fresh, :100-112 cache hit; see SKILL.md gotcha) → plan guard for LM-adapter runs (:225-274: degenerate plans are resampled up to twice with seed + 1009·n and repetition penalty + 0.05·n, rebuilt with the same pattern at :251-261, then rendered anyway with a warning) → a result without codes fails the job (:280-285) → cache store of the LM fields only (:288-301).
4. **Trigger-word re-injection** (:331-353): the LM's CoT caption replaces the original caption, losing the trigger word `translateParams` injected. Re-injected here on both cache-hit and fresh paths, resolved by the same `resolveAdapterTriggers()` call `translateParams` makes, with `skipPresent` so a trigger the planner kept is not doubled. Needs a resolved trigger and at least one adapter path (`loraStack`, `loraPath`, `lmAdapter`); otherwise a `[Trigger]` WARNING is logged (:350-353).
5. Batch expansion (:361-372): when the LM was skipped/cached-as-1, clones the template with fresh random seeds up to `batchSize`. The clone is a spread of the template, so sideband survives.
6. **Per-track synth loop** (:656-998, one `/synth` call per track; deliberately avoids the engine's `multipart/mixed` multi-track response):
   - Source prep (:393-497): cover-family tasks load source audio, apply tempo/pitch, VAE-encode via engine `/vae` with a disk latent cache (`sourceLatentCache.ts`, keyed on path+tempo+pitch+vae_model); Song Builder seed latents (:420-430); timbre reference likewise.
   - Format selection (:503-518): `wav32` if any post-processing is active, `source === 'builder'`, or mastering-with-reference; else `wav16`. wav16 peak-normalizes to 0 dBFS in the engine; wav32 skips normalization (headroom for PP; stable levels across Song Builder appends).
   - `get_lrc` set on every non-instrumental result unless `skipLrc` (:520-533).
   - Multipart decision (:815-828): multipart iff any of `srcAudioBuf | refAudioBuf | srcLatentBuf | refLatentBuf | seedLatentBuf` present → `submitSynthMultipart`; else plain-JSON `submitSynth`.
7. After each track: audio saved to `<data root>/library/audio/<uuid>.wav|mp3` (`config.data.audioDir`, :842-845); LRC lyrics from the base64 `x-lrc-text` response header (:927-938); post-DiT latent fetched and stored as `.latent` HSLAT (:960-994).
8. Optional no-adapter reference render (:1000-1076): with `noAdapterRender` and a DiT adapter, each track is re-rendered at 20 steps with every adapter field deleted, saved as `<name>_noadapter.<ext>`.
9. Auto-trim (before PP; needs `autoTrimEnabled` + `durationBuffer`, corrects duration metadata, :1098-1135) → post-processing chain (`postProcessing.ts`; raw WAV never modified, PP runs on a copy, :1145-1187) → **SQLite insert** per track (:1248-1260): `INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url, duration, bpm, key_scale, time_signature, tags, dit_model, generation_params, mastered_audio_url, latent_url, quality_scores, noadapter_audio_url, backend)`. `generation_params` = the job params with that track's own `seed`/`lmSeed` (:1241-1245); `duration` = the measured WAV length when readable (:1222-1223); `style` = the user's original style input; `caption` = the LM CoT caption.

**Watchdogs** (`server/src/services/generation/pollUntilDone.ts`):
- Stall (:107-144): a ticking stage (`… Step n/total`) gets 2 min (`STALE_TIMEOUT_MS`, :20); quiet stages and VAE decode get 15 min (:26, :34); YuE2 NAR gets a frame-scaled allowance (:39-42). When the window passes, the watchdog polls the engine first: `done` keeps the result, still running on a ticking stage waits out the quiet window, otherwise cancel + `"Generation stalled — no progress for Ns (last stage: …)"`. Engine heartbeats (phase, step, `adapter_progress`) count as progress (:165-178).
- Wall clock (:147-154): `generationTimeoutMinutes` clamped [5,360], default 45 → `"Generation timed out (N min limit)"`, after one last poll that keeps a finished render.
- Engine crash (crash counter changed, :95-97) or three `/job` 404s in a row (:188-190) → `"Generation lost: …"`.
- Transient poll HTTP errors are logged and retried; a poll blind for over 10 s pauses the stall clock (`BLIND_GRACE_MS`, :50).

## 3. translateParams special cases (translateParams.ts)

- Caption fallback chain: `params.prompt || params.songDescription || params.caption || params.style || ''` (:49).
- `instrumental: true` → `lyrics = '[Instrumental]'` (:53-54): there is no instrumental flag on the wire.
- `duration` is sent only when `> 0` (the UI's Auto is -1), with `durationBuffer` added **only when `autoTrimEnabled`** and **only to `req.duration`, not `job.params.duration`** (:65-68), so the DB stores the user's requested duration; auto-trim later removes the buffer from the audio.
- `timeSignature '4/4'` → `'4'` (:70-73).
- `vocal_language` is always sent, normalized to an ISO code (:87); send `'unknown'` to let the LM decide.
- `lm_seed` is left unset when `lmSeedFollowsDit` (default), so the engine ties it to the resolved DiT seed (:103-106). Seed breeding `seedB`/`seedBlend`/`seedBlendEnd` → `seed_b`/`seed_blend`/`seed_blend_end` (:109-119).
- An LM adapter with no explicit `lmRepPenalty` gets `lm_rep_penalty = 1.05` (:185). An `.onnx` `vaeModel` is dropped (:187-189).
- `loraStack` → `adapters[]` via `mapPath()`; supersedes single `adapter` (:196-226). A per-adapter `stepStart`/`stepEnd` window compiles to a step-domain gain curve; an explicit `gainCurve` defaults to the `t` domain.
- Per-section `[Section]{k=v}` lyric directives (`adapterSections.ts`): parsed into `adapter_sections` + forced `adapter_mode='runtime'` **only with a 2+ adapter stack** (:237-250); otherwise directives are **stripped** so they don't reach the LM as garbage tokens (:251-256). Gain curves without directives synthesize one whole-song section and force runtime mode (:263-270).
- Rebase only when `loraPath && rebaseSource && rebaseBeta` all truthy (:273-276).
- Trigger words are applied to the caption when a trigger resolves and an adapter path is present (:287-299).
- Falsy-guard fields where 0 intentionally means "let the LM fill it": `bpm`, `inferenceSteps`, `batchSize`. Fields where 0/false matter use `!== undefined` (e.g. `guidanceScale`, `shift`, `seedStrength`, `dcwEnabled`).

## 4. aceClient.ts: the wire

- `AceRequest` interface (:84) = Node-side canonical superset (C++ struct fields + all sideband). This, plus `ServerFields`, is the authoritative request schema.
- Engine is **single-threaded httplib**: during heavy compute it cannot respond. Timeouts: `TIMEOUT_QUICK` 15 s (submit/health), `TIMEOUT_POLL` 30 s, `TIMEOUT_RESULT` 300 s, `TIMEOUT_SA3` 30 min (:18-21).
- `submitLm(request, mode?, keepLoaded)` (:470): mode `inspire`/`format` optional (sent as `lm_mode` in the body); `keepLoaded` → `?keep_loaded=1`.
- `submitSynth(request | request[], format, keepLoaded)` (:483); `submitSynthMultipart(...)` (:498): part names `request` (JSON), `audio`, `ref_audio`, `src_latents`, `ref_latents`, `seed_latents`. **Multipart sends only `request[0]` when given an array** (:527).
- Fine-grained engine phases (`AceJobPhase`, :258-273, e.g. `adapter_precompute`, `dit_inference`) mirror `job_phase_str()` in hot-step-server.cpp (:334) and surface via `/api/generate/status/:id`.
- Engine job status: stored as ints in the engine (0=running, 1=done, 2=failed, 3=cancelled) and reported by `GET /job` as the strings `running`/`done`/`failed`/`cancelled` (`job_status_str`, hot-step-server.cpp:470).
- Note: the engine `/vae` handler and CLI tools use the C++ struct's `vae` field (request.h:221), while server-mode selection uses the sideband `vae_model`: different names for adjacent concepts.

## 5. Engine tier: hot-step-server.cpp

- `POST /synth` (`handle_synth`, :1960): parses the SAME JSON body twice: `parse_server_fields()` for sideband, `request_parse_json[_array]()` for the upstream struct. A request still asking for `stream_mode` or an `.onnx` VAE gets a 400 (`synth_removed_feature`, :1949). Diagnostics: prints first 300 chars of a plain-JSON body as `[DIAG] /synth body` (:2090). Query params: `?format=wav16|wav24|wav32|mp3` (:2112-2129), `?keep_loaded=1` (co-resident; one-way flip of the model store to never-evict, :1440-1443). Returns `{"id":"N"}` immediately; work runs on the single GPU work queue.
- `POST /lm` (`handle_lm`, :1198): parses only the C++ struct (no sideband); empty caption → 400; `lm_mode` `generate` (metadata + lyrics + audio codes) / `inspire` / `format`; `lm_batch_size` clamped to `[1, --max-batch]` (itself clamped to 9). `lm_worker` (:1090) resolves the LM and any `lm_adapter` (unresolvable → job fails), defaults `lm_seed` to `seed`, and stores a JSON array of enriched AceRequests via `request_to_json` (:1179). Serialization is **sparse** (defaults omitted).
- **`ServerFields`** (:697-772): the definitive sideband-only field list: `vae_model`, `emb_model`, `solver_name` (JSON key `infer_method`), `scheduler`, `guidance_mode`, `apg_momentum`, `apg_norm_threshold`, `stork_substeps` (also in the struct), `beat_stability`, `frequency_damping`, `temporal_smoothing`, `group_scales` (`adapter_group_scales`), `adapter_mode`, `adapter_runtime_quant`, `adapter_runtime_dora`, `adapter_merge_lowvram`, `adapter_section_align_at`, `adapter_section_isolation` (dead knob, see SKILL.md), `rebase_source`, `rebase_beta`, `concepts`, `concept_extract` (+ its sub-fields), `dcw_*`, `latent_shift`, `latent_rescale`, `cfg_cutoff_ratio`, `cache_ratio`, `custom_timesteps`, `denoise_strength/smoothing/mix`, `plugin_params`, `seed_strength`, `seed_latents` (multipart part, not JSON), `evict_lm`, `vae_chunk`, `batch_cfg`, `stream_mode` (parsed only to refuse it).
- `parse_server_fields` (:774) resets defaults first: solver `euler`, scheduler empty, guidance `apg`, adapter_mode `merge` (:775-788). Numeric fields are read with `yyjson_get_num` under `yyjson_is_num` (note at :880-885); `vae_chunk`, `batch_cfg` and `stork_substeps` are read only when the JSON value is an integer. For an array body it reads the first object only (:795-797).
- **`synth_worker`** (:1259):
  - Resolves DiT/VAE/text-enc from the model registry; unknown `vae_model`/`emb_model` **fall back to default with only a stderr warning** (:1323-1342).
  - Adapter stack: `adapters[]` supersedes single `adapter` (single is folded into a 1-element stack); a name not in the registry is tried as a file path, then as a directory holding `adapter_model.safetensors` or `lokr_weights.safetensors`, else the **job fails** (:1365-1420). Resolved stack lives in `g_hotstep_params.adapters`.
  - Copies every ServerFields value into **`g_hotstep_params`** (:1453-1528) **before** `ace_synth_load()` (:1572): adapter merge reads group scales at merge time. `dit_sliding_window` is read from the struct, not the sideband (:1468).
  - Per-section masking forces runtime mode engine-side too: needs `adapter_sections` non-empty AND either a stack of 2+ or any adapter with a gain curve (:1476-1497).
  - `rebase_source` name resolved through the DiT registry; unknown → cleared with a warning (:1498-1514).
  - Auto-shift when `shift == -1`: adaptive from duration + step count (:1586-1606).
  - Batch: `synth_batch_size` expands per-seed variants; total clamped to DiT max 9 (:1274-1285, expansion :1608-1637).
  - Post-decode: peak-normalize (skipped for wav32), spectral denoiser, WAV/MP3 encode; multi-request results return `multipart/mixed`; Node sidesteps this with one track per call.
- **`hot-step-params.h`**: `HotStepParams` global (`g_hotstep_params`, :288) is safe as a global because the GPU worker is single-threaded (header comment, :9-11). Also defines group classification `adapter_determine_group()` (:107): **`cross_attn` must be checked before `self_attn`** (note at :105, substring matching).

## 6. LM cache internals (lmCache.ts)

- `LmCacheEntry` (:9-20): the 7 LM output fields `audio_codes`, `caption`, `lyrics`, `bpm`, `duration`, `keyscale`, `timesignature`, plus the resolved per-output `lm_seed`.
- Key (`computeLmCacheKey`, :26-70): sha256/16 over the effective LM seed (`lm_seed`, or `seed` when tied), caption, lyrics, bpm, duration, keyscale, timesignature, vocal_language, lm_model, lm_batch_size, lm_temperature, lm_cfg_scale, lm_top_p, lm_top_k, lm_negative_prompt, use_cot_caption, the repetition and sampling extras (`lm_rep_penalty`, `lm_rep_window`, `lm_rep_mode`, `lm_dry_base`, `lm_dry_min_len`, `lm_min_p`, `lm_cfg_min_p`, `lm_cfg_logprob`, `lm_cfg_rescale`) and `lm_adapter`/`lm_adapter_scale`. **DiT/adapter/DCW params deliberately excluded** from both key and value.
- In-memory `Map`, LRU max 20 (`LM_CACHE_MAX`, :22); timestamp refreshed on hit.
- Consequence: changing DiT-only params reuses cached codes (intended). Adding a new **LM-affecting** param without extending this key serves stale codes: checklist item #6 in SKILL.md.

## 7. Status / history ledger

- **DONE:** sideband rebuild on all LM paths (hardened in `8ea519b`/`168dcb5`, plan-guard path since); multi-adapter stack + per-section masking (P1 proportional map, P2 alignment via `adapter_section_align_at`) + timestep gain curves; adapter stack cache-key signature; LM cache; latent capture; GPU lane + envelope + attempt records; durable audio intent queue; server-resolved Create intents; stall/wall-clock/crash watchdogs; forced wav32 for builder/PP; every numeric sideband field on `yyjson_get_num` (`7cb013f5`).
- **REMOVED:** the ACE stream mode above the engine (`1b9d6510`), including `/api/generate/stream/:id`; the engine refuses `stream_mode`.
- **REVERTED:** regional self-attn isolation (`0f3bf6d` → `ee041e1`, broke musical continuity). `adapter_section_isolation` remains as zombie plumbing (parsed :836, copied :1475, consumed nowhere in `engine/src/`).
- **UNVALIDATED:** basin re-base β·(S−T) nudge for cross-base LoKR adapters: fully plumbed, effectiveness unproven.

## 8. Doc cross-checks (code wins)

- engine/docs/ARCHITECTURE.md now documents the ace-server sideband fields and the `shift == -1` auto path. Where its tables and `ServerFields` / the `AceRequest` interface disagree, the code is the schema.
- The Node-side `isCoverTask` list (ace/generate.ts:71) is policy, not engine capability: `complete` runs the LM through `/api/generate`, the other cover-family tasks never do.
