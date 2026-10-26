---
name: generation-request-flow
description: Traces the full life of a music generation request (UI form → Node job queue → engine LM → synth → SQLite song row) and every place params can silently drop, especially the LM echo sideband gotcha. Use when adding/plumbing a generation parameter, debugging "param set in UI but engine uses default", touching /api/generate, translateParams, aceClient, hot-step-server.cpp request parsing, the LM/synth request rebuild, or understanding task modes (text2music/cover/repaint/lego/extract/complete), the serial job queue, and stall/timeout watchdogs.
---

# Generation Request Lifecycle & Param Plumbing

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

This skill maps the end-to-end path of an ACE-Step generation request in HOT-Step CPP and catalogs every point where a parameter can be silently lost. Most historical bugs in this area are **param-plumbing bugs**: a value set in the UI never reaches the C++ sampler, and nothing errors. The other backends (MiniMax-Music3, YuE2, SA3) share the Node queue described here but have their own `generate()` under `server/src/services/backends/<id>/`.

Terms used below:
- **DiT**: the diffusion transformer that turns audio codes + text embeddings into audio latents (the "synth" phase). Pipeline is LM → DiT → VAE.
- **LM**: the language model phase that generates metadata (bpm, duration, key), a chain-of-thought (CoT) caption, lyrics, and `audio_codes` for the DiT.
- **aceReq**: the Node-side request object (interface `AceRequest`, [server/src/services/aceClient.ts:84](../../../server/src/services/aceClient.ts)) sent to the C++ engine. Snake_case, produced from camelCase UI params by `translateParams()`.
- **Sideband**: HOT-Step-only fields (solver, scheduler, guidance, adapter knobs, …) that travel *alongside* the upstream C++ `AceRequest` struct in the same JSON body but are parsed separately into `ServerFields` (engine) because they don't exist in the upstream struct.
- **Envelope**: the frozen snapshot `buildEnvelope()` takes at enqueue: the backend the job runs on, the resolved request and the retry policy.

## When to use this skill

- Adding a new generation parameter (UI knob → engine behavior).
- Debugging: a param is set in the UI but the engine logs/behavior show the default.
- Modifying `server/src/routes/generate.ts`, `server/src/services/backends/ace/generate.ts`, `translateParams.ts`, `aceClient.ts`, or `engine/tools/hot-step-server.cpp` request handling.
- Understanding generation modes (`text2music`, `cover`, `repaint`, `lego`, …) or the job queue/watchdogs.

## Golden rules

1. **NEVER rebuild the post-LM synth request by copying the LM echo or by whitelisting fields to carry across.** The engine's `/lm` response only contains fields in the C++ `AceRequest` struct, so every sideband field vanishes. Always rebuild as `{ ...aceReq, <the LM output fields> }`. WHY: every field-copy whitelist ever written here went stale and silently dropped params; commits `8ea519b` and `168dcb5` fixed real bugs of exactly this class. Full detail in the next section.
2. **Read every numeric engine JSON field with `yyjson_get_num()` under a `yyjson_is_num()` guard, never `yyjson_get_real()`.** WHY: `yyjson_get_real()` returns **0 for JSON integers**, and JavaScript serializes `2.0` as `2`, so a whole-number value silently becomes `0.0` (`latent_rescale: 1` arrived as 0 and zeroed the DiT output). Commit `7cb013f5` moved every numeric field in hot-step-server.cpp to `yyjson_get_num`; the in-code note is at [hot-step-server.cpp:880-885](../../../engine/tools/hot-step-server.cpp). Keep new fields on the same pattern.
3. **C++ changes rebuild ONLY via `dev-rebuild.bat` at the repo root** (call it by absolute path, see AGENTS.md), never `engine\build.cmd` directly, under any circumstances (you cannot reliably tell whether the app is running; Node auto-respawns ace-server on crash → infinite respawn + file-lock loop). Never `cmake --build . --clean-first` (20+ min CUDA recompile). After the rebuild, run `node tools/engine-smoke.mjs` with the GPU idle and paste its `[Check Batch0]` line and `ENGINE SMOKE:` verdict (AGENTS.md render receipt).
4. **Type-check TypeScript with `npx tsc --noEmit` in `server/` and `npx tsc --noEmit -p tsconfig.app.json` in `ui/`; do not `npm run build` during dev.** A bare `npx tsc --noEmit` in `ui/` checks nothing and exits 0.
5. **After any upstream sync, run `powershell -File engine\verify-hooks.ps1`.** WHY: `pipeline-synth-ops.cpp` losing its `hot-step-sampler.h` include is **SILENT**: everything compiles but all solvers/schedulers/guidance/sideband go dead (they read `g_hotstep_params`, which nothing consumes anymore). `build.cmd` also runs the check before every compile.
6. **Git: all work on `master`, stage explicit paths (never `git add -A`), push only with explicit user approval.** Any pushed `v*` tag triggers a full multi-platform CI release build.
7. **A new GPU input tensor inside `hot-step-sampler.h` (masks, encoder states, constants) needs both execution paths handled.** The DiT prefers a direct static graph where inputs flagged in `mark_resident_inputs` stay resident across steps ([hot-step-sampler.h:152-158, :196](../../../engine/src/hot-step-sampler.h)); when any op is unsupported it falls back to the GGML scheduler, which aliases input buffers as scratch space, so every constant must be re-uploaded on every evaluation (the `!dit_graph.direct` block in `evaluate_velocity`, :810-832) and forced onto the backend in `force_inputs_to_backend` (:167). Skipping the fallback re-upload produced blank output and the "nil adapter" bug. Anything that overwrites a resident input mid-sampling must set `enc_dirty` or re-upload it. Never hoist a fallback-path upload out of the step loop as an "optimization".
8. **Never delete generated audio artifacts (`<data root>/library/audio/*`, test outputs) based on your own judgment of quality**: the user verifies results by ear.
9. **Node 20 to 24 LTS, 24 recommended**: `engines` enforces `<25`.

## THE LM ECHO SIDEBAND GOTCHA (read this before touching the ACE backend's generate.ts)

**Ground truth from the departing lead engineer, verified line-by-line in code:**

Server-side params (`ServerFields`) do **not** survive the `/lm` round trip: the engine echoes back only what it knows. When building the synth request after the LM step, rebuild it from the **original `aceReq` plus the LM-returned fields ONLY**. Never maintain a whitelist of fields to copy across: every whitelist has gone stale and silently dropped params. Commits `8ea519b` and `168dcb5` fixed bugs of exactly this class.

**Mechanics of why:**
- `/lm`'s result body is `request_to_json()` serialized from the **C++ `AceRequest` struct** (`lm_worker`, [hot-step-server.cpp:1179](../../../engine/tools/hot-step-server.cpp); struct at [engine/src/request.h:36](../../../engine/src/request.h)).
- Fields that exist only in `ServerFields` ([hot-step-server.cpp:697-772](../../../engine/tools/hot-step-server.cpp)), such as `vae_model`, `emb_model`, `infer_method`, `scheduler`, `guidance_mode`, `adapter_group_scales`, `adapter_mode`, `adapter_runtime_quant`, `rebase_source`/`rebase_beta`, `apg_*`, `denoise_*`, `plugin_params`, `concepts`, `seed_strength`, `evict_lm`, `vae_chunk` and `batch_cfg`, are **not in the C++ struct**, so the LM echo cannot contain them. Copy the echo forward and they all silently drop.

**The correct pattern: it exists in three places in [server/src/services/backends/ace/generate.ts](../../../server/src/services/backends/ace/generate.ts), keep them identical:**

```ts
// ace/generate.ts:199-213 (fresh LM), :100-112 (LM-cache hit), :251-261 (plan-guard resample)
lmResults = lmResults.map(lmOut => ({
  ...aceReq,                       // CURRENT request = every sideband field survives
  audio_codes: lmOut.audio_codes,  // then ONLY the LM-generated fields:
  caption: lmOut.caption,
  lyrics: lmOut.lyrics,
  bpm: lmOut.bpm,
  duration: lmOut.duration,
  keyscale: lmOut.keyscale,
  timesignature: lmOut.timesignature,
  lm_seed: lmOut.lm_seed,          // resolved per-output LM seed (base + batch index)
}));
```

The **only** whitelist allowed is the LM-output list (those 8 fields, which are also exactly what `LmCacheEntry` stores). The base must always be a spread of the live `aceReq`. If you add a new server-side param you do **not** touch this rebuild code: that is the whole point of the pattern.

## The pipeline at a glance

```
UI: CreatePanel → App.tsx handleGenerate               [ui/src/App.tsx:705]
  getGlobalParams() + content + app settings, camelCase [ui/src/stores/globalParamsStore.ts:741, types.ts:108]
  → enqueueSimpleGen()                                   [ui/src/stores/audioGenQueueStore.ts:1424]
      POST /api/resolve/preview (CreateIntent → exact body) unless GENERATION_INTENT_PATH=old [server/src/routes/resolve.ts]
      then POST /api/audio-queue/items (server queue) or POST /api/generate (browser queue)
  → submitGeneration(): 503/401/409/400 checks, intent resolve [server/src/routes/generate.ts:344]
  → buildEnvelope(): backend + resolved request frozen   [server/src/services/generation/envelope.ts:99]
  → enqueueGeneration() → runOnGpuLane(), one GPU owner  [generate.ts:191, services/generation/gpuLane.ts:78]
  → runGeneration() → evictOtherFamilies() → backend.generate() [generate.ts:156]
  → ACE: runAceGeneration()                              [server/src/services/backends/ace/generate.ts:26]
      translateParams(): camelCase → snake_case          [services/generation/translateParams.ts:47]
      POST :8085/lm (unless skipped or cached)           [aceClient.ts:470; engine handle_lm hot-step-server.cpp:1198]
      REBUILD synth req: {...aceReq, LM fields}          [ace/generate.ts:199-213]  ← THE SIDEBAND GOTCHA
      POST :8085/synth, one track per call (JSON or multipart) [aceClient.ts:483 / :498; engine handle_synth :1960]
  → engine: parse_server_fields() + request_parse_json[_array]() on the SAME body [hot-step-server.cpp:774]
  → synth_worker: ServerFields → g_hotstep_params before ace_synth_load() [:1259, copy :1453-1528, load :1572]
  → synth pipeline: TextEnc → DiT (hot-step-sampler.h reads g_hotstep_params) → VAE → WAV/MP3
  → Node saves <data root>/library/audio/<uuid>.wav|mp3 [ace/generate.ts:842-845]
  → auto-trim → post-processing chain → INSERT INTO songs [ace/generate.ts:1098-1135, :1145-1187, :1248-1260]
```

Engine port 8085 is set in [server/src/config.ts:217](../../../server/src/config.ts) (`ACESTEPCPP_PORT`). Node job statuses: `pending | running | lm_running | synth_running | saving | succeeded | failed | cancelled` ([server/src/services/generation/jobTypes.ts:16](../../../server/src/services/generation/jobTypes.ts)).

## Procedure: adding a new generation parameter end-to-end

A new param must be threaded through every layer below. Missing any one = silent default.

1. **UI state:** add to `ui/src/stores/globalParamsStore.ts` (persisted per-field to localStorage under an `hs-*` key; solver/scheduler/guidance/plugin/backend picks are per backend through `BACKEND_SCOPED_FIELDS` and `scopedKey()`, :219-249) and include it in `getGlobalParams()` (:741). Watch the conditional gating there: many params are sent only when their mode is active (e.g. `apgMomentum` only when `guidanceMode === 'apg'`, :878). Add the field to `GenerationParams` in `ui/src/types.ts:108`. The visible controls for engine knobs live in `ui/src/components/global-bar/` (`GlobalParamBar.tsx`, `GenerationDropdown.tsx`); grep an existing knob's setter to find the right spot.
2. **Node defaults mirror:** `POST /api/generate` also accepts a compact `generation-intent/1` body, which Node expands with `GENERATION_DEFAULTS` + `resolveGenerationParams()` in `server/src/services/generation/defaults.ts` (:4, :155), a copy of the `getGlobalParams()` gating. Add the default and the same gate there, and list the key in `server/src/contracts/generationControls.ts` (documented in [docs/dev/frontend-create-controls.md](../../../docs/dev/frontend-create-controls.md)). `intent.golden.test.ts` replays captured bodies against it.
3. **Node translate:** map camelCase → snake_case in `server/src/services/generation/translateParams.ts`. Use `!== undefined` guards for fields where `0`/`false` is meaningful: a truthy guard (`if (params.foo)`) silently drops legitimate zeros.
4. **Node wire type:** add the snake_case field to the `AceRequest` interface in `server/src/services/aceClient.ts:84` (this interface is the Node-side canonical superset: C++ struct fields + sideband).
5. **Engine parse:** add to `ServerFields` (hot-step-server.cpp:697) and `parse_server_fields()` (:774): `if ((v = yyjson_obj_get(obj, "my_field")) && yyjson_is_num(v)) sf->my_field = (float) yyjson_get_num(v);` (Golden rule 2). A field that belongs in the upstream C++ struct instead goes into `request.h` + `request_parse_json()` / `request_build_doc()` in `request.cpp`; struct fields survive the LM echo.
6. **Engine global:** copy into `g_hotstep_params` in the block at hot-step-server.cpp:1453-1528 (note: this copy happens **before** `ace_synth_load()` at :1572 because adapter merge reads group scales at merge time), and add the field to `HotStepParams` in `engine/src/hot-step-params.h` (:130). Consume it in `hot-step-sampler.h` (or wherever).
7. **Do NOT touch** the LM-echo rebuild in ace/generate.ts: the `...aceReq` spread carries your field automatically. If the field changes what the LM emits, add it to `computeLmCacheKey()` (`lmCache.ts:26`) or stale codes are served.
8. Rebuild: `dev-rebuild.bat` + `node tools/engine-smoke.mjs` (C++), the two tsc checks from Golden rule 4 (TS). Verify arrival, **but beware the `[DIAG]` trap**: the `[DIAG] /synth body (first 300 chars)` line (hot-step-server.cpp:2090, plain-JSON bodies only) truncates at 300 chars, and the body starts with caption then lyrics, so with any realistic caption your sideband field will NOT appear in the visible prefix. **A field absent from the DIAG line is NOT evidence it was dropped.** Reliable checks: add a temporary `fprintf` of your parsed value in `parse_server_fields` (or next to the `[Server] HOT-Step params:` summary at hot-step-server.cpp:1537), or log the full `aceReq` server-side before `submitSynth`. For any new UI control, ask the user to visually confirm it; do not use the browser agent for visual verification (API checks are fine).

## Key files

| Path | Role |
|---|---|
| `ui/src/stores/globalParamsStore.ts` | Zustand store for all engine knobs; `getGlobalParams()` (:741) is the single assembly point with mode-conditional gating |
| `ui/src/types.ts:108` | `GenerationParams`: canonical camelCase UI shape (~150 fields, incl. post-processing-only fields that never reach the engine) |
| `ui/src/App.tsx:705` | `handleGenerate`: merges global params + Create-tab content + app settings, enqueues via `enqueueSimpleGen` |
| `ui/src/stores/audioGenQueueStore.ts:1424` | `enqueueSimpleGen`: resolve preview, then the server audio queue or a direct submit plus status polling |
| `ui/src/services/api.ts:181` | `generateApi`: resolve preview/path, POST `/generate`, status, cancel, queue, reset |
| `server/src/routes/resolve.ts` | `GET /api/resolve/path`, `POST /api/resolve/preview` (Create and written-song intents → exact `/api/generate` body) |
| `server/src/routes/generate.ts` | Queue and endpoints: `submitGeneration` (:344), `enqueueGeneration` retry loop (:191), `runGeneration` (:156), status/cancel/queue/reset, MM3 stream pipe, STORM streams |
| `server/src/services/generation/envelope.ts` | `buildEnvelope`: captures the backend (active or `backendOverride`) and resolved request once, deep-frozen |
| `server/src/services/generation/gpuLane.ts` | `runOnGpuLane`: one GPU owner at a time, shared with post-processing re-runs |
| `server/src/services/generation/pollUntilDone.ts` | Engine job polling + stall/wall-clock/crash watchdogs |
| `server/src/services/backends/ace/generate.ts` | `runAceGeneration`: the whole ACE lifecycle (LM phase, sideband rebuild, plan guard, trigger re-injection, synth loop, PP, DB insert) |
| `server/src/services/backends/ace/index.ts` | ACE manifest: capabilities, operations list (:118), retry policy (:170) |
| `server/src/services/generation/defaults.ts` | `GENERATION_DEFAULTS` + `resolveGenerationParams()` for `generation-intent/1` bodies |
| `server/src/services/generation/translateParams.ts` | camelCase → `AceRequest` snake_case; pure function |
| `server/src/services/generation/lmCache.ts` | LM output cache: stores only the LM fields, keyed by LM-affecting params only (sha256/16, LRU max 20) |
| `server/src/services/audioQueue/intentQueue.ts` | Durable audio intent queue: captured bodies in SQLite, submitted through `submitGeneration` |
| `server/src/services/aceClient.ts` | Wire types + HTTP client to engine :8085; timeouts 15 s submit / 30 s poll / 300 s result (:18-20) |
| `engine/tools/hot-step-server.cpp` | Engine HTTP server: `handle_lm` (:1198), `lm_worker` (:1090), `handle_synth` (:1960), `ServerFields` (:697), `parse_server_fields` (:774), `synth_worker` (:1259) |
| `engine/src/hot-step-params.h` | `HotStepParams` sideband global (`g_hotstep_params`, :288) read by `hot-step-sampler.h` inside the DiT; also `AdapterGroupScales`/`AdapterSpec`/`AdapterSection` |
| `engine/src/request.h:36` | Upstream C++ `AceRequest` struct: fields here DO survive the `/lm` echo |
| `server/src/config.ts:217` | Engine port (8085 default) |

## Every place params can silently drop (checklist)

1. **The LM echo** (see gotcha section above): THE classic.
2. **yyjson int-vs-float trap**: fixed for every current field (`7cb013f5`), but any new numeric field read with `yyjson_get_real()` turns whole-number JSON values into `0.0`. Use `yyjson_get_num`.
3. **UI-side gating** in `getGlobalParams()` (globalParamsStore.ts:741-1006) and its server mirror `resolveGenerationParams()` (defaults.ts:155): params are conditionally `undefined` based on mode/toggles. A stale `advancedAdapters=false` suppresses a persisted adapter stack (globalParamsStore.ts:751). If a param "never arrives," check the gate here first, in both copies.
4. **translateParams falsy guards**: `if (params.bpm)` / `if (params.inferenceSteps)` / `if (params.batchSize)` intentionally drop 0 ("LM fills it in"), and `duration` is sent only when `> 0` (the UI's Auto is -1). A new field copied with a truthy guard drops legitimate `0`/`false`. Use `!== undefined` where 0 matters.
5. **Trigger word destroyed by the LM's CoT caption**: `translateParams` injects it (:287-299), the LM's chain-of-thought caption replaces the whole caption, and ace/generate.ts re-injects it (:331-353) on both cache-hit and fresh paths. Both passes resolve triggers the same way (`resolveAdapterTriggers` in `triggerWords.ts`: a manual override, then the trigger embedded in the adapter's safetensors metadata, then the filename fallback the UI sends when "use filename" is on) and skip a trigger already present. Re-injection needs a resolved trigger AND at least one adapter path (DiT stack, `loraPath` or `lmAdapter`); triggers with no adapter log `[Trigger] "…" configured but no adapter is loaded — skipping re-injection` (:352). Debugging "adapter trigger missing from caption" starts here.
6. **LM cache staleness**: only the LM fields are cached/restored, keyed by `computeLmCacheKey` (lmCache.ts:26-70). If you add an LM-affecting param, it MUST go into that key or stale audio codes get served for new settings. DiT/adapter params are deliberately excluded from key and value.
7. **Multipart array truncation**: `submitSynthMultipart` sends only `request[0]` if given an array (aceClient.ts:527).
8. **Engine registry fallbacks**: unknown `vae_model`/`emb_model` fall back to the default with only a stderr warning, not an error (hot-step-server.cpp:1326, :1336); an `.onnx` VAE name is a 400 (`synth_removed_feature`, :1949) and `translateParams` drops it before it gets there (:187); unknown `rebase_source` is skipped with a warning (:1511). Unknown DiT adapter name (after the absolute-path and adapter-directory fallbacks) DOES fail the job (:1406), and so does an unresolvable `lm_adapter` on `/lm` (`resolve_lm_adapter_path`, :1056).
9. **Duplicated fields**: `dcw_scaler`/`dcw_high_scaler`/`dcw_mode`, `latent_shift`/`latent_rescale`, `custom_timesteps` and `stork_substeps` exist in BOTH the C++ `AceRequest` struct (request.h) and `ServerFields`; `dcw_enabled` is sideband-only. The server sampler reads the **`g_hotstep_params` (ServerFields) copy** for DCW, the solver and its substeps; keep both parse paths in sync or CLI-tool vs server behavior diverges. Also note naming: the solver is `solver` in the C++ struct but `infer_method` on the sideband, and the struct has `vae` (request.h:221, used by CLI/`/vae`) while the server sideband uses `vae_model`.
10. **Multi-request `/synth` arrays**: `parse_server_fields` reads only the first object of an array (:795-797) and the adapter stack comes from `ace_reqs[0]`, so per-request sideband differences inside one array are ignored. Node avoids this by sending one track per call.
11. **Upstream sync hook loss**: silent solver/guidance/sideband death; run `engine\verify-hooks.ps1`.
12. **LM-skip paths** (cover-family tasks, `skipLm`, pre-filled `audio_codes`) bypass the LM entirely: anything depending on LM-filled metadata must handle 0/empty. When `skipLm` on non-cover: defaults filled (bpm 120, duration 120, 'C major', '4') at ace/generate.ts:84-87.

## Generation modes (code truth)

Engine `task_type` (request.h:180, docs in engine/docs/ARCHITECTURE.md): `text2music` (default), `cover` (FSQ roundtrip, free reinterpretation), `cover-nofsq` (clean latents, faithful remix), `repaint` (region regenerate via `repainting_start`/`repainting_end`; negative start = outpaint before), `lego` (new layered track; base model only), `extract` (stem isolation; base only), `complete` (fill mix around a stem; base only). LM modes: `generate` / `inspire` / `format`.

**Node-side LM policy:** `isCoverTask` (`ace/generate.ts:71`) is `['cover','cover-nofsq','repaint','lego','extract']`, so the Node pipeline **never runs the LM for those**. `complete` is absent from that list, so it runs the LM via `/api/generate` unless `skipLm` is set or `audio_codes` are supplied. That list is Node-side policy, not the engine's task registry. `buildEnvelope()` also rejects a `taskType` missing from the ACE operations list (`ace/index.ts:118-120`).

## Failure signatures

| Symptom | Cause / where to look |
|---|---|
| Param set in UI, engine behaves as default | Sideband dropped post-LM (gotcha), UI gate (checklist #3), or a new float read with `yyjson_get_real` (checklist #2). Do NOT trust the 300-char `[DIAG] /synth body` prefix as proof of absence (procedure step 8); add a temporary parse-site fprintf instead |
| "Generation stalled — no progress for Ns (last stage: …)" | No-progress watchdog in `pollUntilDone.ts`: 2 min on a ticking "Step n/total" stage, 15 min on quiet stages and VAE decode; it asks the engine first and keeps the job if the engine says done (:107-144). Engine wedged |
| "Generation timed out (N min limit)" | Wall-clock watchdog, `generationTimeoutMinutes` clamped [5,360], default 45 (pollUntilDone.ts:18, :147-154) |
| "Generation lost: the engine crashed mid-render" / "…no longer knows this job" | Engine process died or restarted under the job (crash counter, or three 404s from `/job`, pollUntilDone.ts:95-97, :188-190). Read `ace_engine.log` |
| "Generation failed on ace-server" | Engine job status `failed`: read `<data root>/logs\<session>\generations\gen_<uuid>_*.log`, then `ace_engine.log` for `[Server] Adapter not found` / synth load failure |
| "LM job completed but returned no parseable results" / "LM planner returned no audio codes" | `/lm` result empty or a result without codes (ace/generate.ts:185-190, :280-285); see `ace_engine.log` |
| `[Plan Guard] Degenerate plan …` warnings | LM-adapter runs only: a looping plan is resampled up to twice with a shifted seed and firmer repetition penalty, then rendered anyway (ace/generate.ts:225-274) |
| All solvers/schedulers behave like euler after an upstream sync | Lost `hot-step-sampler.h` hook in `pipeline-synth-ops.cpp`: SILENT; run `engine\verify-hooks.ps1` |
| Trigger word absent from final caption | CoT caption replaced it and re-injection found no adapter or no trigger: grep gen log for `[Trigger]` |
| Stale LM output despite changed DiT/adapter params | Correct behavior (LM cache keys only LM params); set `cacheLmCodes: false` to force a fresh LM run |
| 503 "Engine not ready" / "Engine is paused for training preprocessing" on submit | Engine bootstrap unfinished, or a training preprocess owns the GPU (`submitGeneration`, generate.ts:347-357) |
| 409 "Expected backend 'x' but the active backend is 'y'" | A resolved body or MCP call names `expectedBackend` and the active backend changed (generate.ts:373-389) |
| 400 "Timbre reference file is missing" | A profile points at a deleted upload (generate.ts:420-425) |
| Song Builder tracks get quieter each append | wav16 peak-renormalization: fixed by forced `wav32` for builder/PP (ace/generate.ts:503-518) |

## Institutional knowledge

- **VALIDATED (LM echo sideband gotcha):** described above; the rebuild pattern at ace/generate.ts:100-112, :199-213 and :251-261 is the fix for real shipped bugs (commits `8ea519b`, `168dcb5`).
- **VALIDATED (yyjson int trap):** fixed for every field in `7cb013f5`; in-code note at hot-step-server.cpp:880-885.
- **VALIDATED (copy-before-load ordering):** `g_hotstep_params` must be populated **before** `ace_synth_load()` because adapter merge reads group scales at merge time (comment at hot-step-server.cpp:1453-1456).
- **VALIDATED (zombie plumbing):** regional self-attn isolation was reverted (`0f3bf6d` → `ee041e1`, "broke musical continuity"), but `adapter_section_isolation` is still sent by the UI with 2+ adapter stacks, parsed (:836) and copied to `g_hotstep_params` (:1475) while nothing in `engine/src/` consumes it. It is a **dead knob**: don't mistake its plumbing for a working feature, and do NOT re-implement the consumer: per-section self-attn masking breaks musical continuity across section boundaries; any retry needs a cross-section-coherence design first.
- **VALIDATED (auto-shift fork behavior):** when `shift == -1`, this fork computes adaptive shift from duration + steps (hot-step-server.cpp:1586-1606). `shift == 0` keeps upstream's auto (3.0 turbo, 1.0 otherwise).
- **VALIDATED (DiT batch clamp):** `synth_batch_size` expands per-seed variants; total clamped to DiT max **9** (hot-step-server.cpp:1274-1285). Node avoids engine `multipart/mixed` results by submitting one track per `/synth` call.
- **VALIDATED (retry scope):** the queue retries only errors thrown out of `backend.generate()` (generate.ts:203-263). `runAceGeneration` catches its own pipeline failures (stalls, timeouts, engine failures) and marks the job failed, so those are not retried even though the ACE policy says `maxAttempts: 2`.
- **UNVALIDATED / HYPOTHESIS (basin re-base):** the β·(S−T) nudge for cross-base LoKR adapters is plumbed end-to-end (globalParamsStore.ts:854 → translateParams.ts:273 → ServerFields :717 → g_hotstep_params :1502-1514) and works in merge and runtime mode, but its effectiveness is unproven. **Invariant (VALIDATED): the nudge applies to the FIRST adapter of a stack only** (`dit.h:791-792`: `rb_src = (si == 0) ? rebase_source : nullptr`; runtime mode `adapter-runtime.h:1091`); one nudge per stack, never per adapter. Any plumbing change that makes rebase per-adapter is a bug (issue #72).

## Useful commands (PowerShell)

```powershell
# Each line starts from the repo root.
cd server; npx tsc --noEmit                         # type-check server (never npm run build in dev)
cd ui; npx tsc --noEmit -p tsconfig.app.json        # type-check UI (a bare tsc --noEmit checks nothing)
& "$PWD\dev-rebuild.bat"                            # the ONLY sanctioned way to rebuild C++, by absolute path
node tools/engine-smoke.mjs                         # render receipt after an engine rebuild (GPU idle)
powershell -File engine\verify-hooks.ps1            # after any upstream sync
cd server; npx tsx scripts/data-paths.ts            # resolve <data root>/logs before reading session logs
Invoke-RestMethod http://localhost:3000/api/generate/queue                   # queue health (dev app; :3001 is the prod port)
Invoke-RestMethod -Method Post http://localhost:3000/api/generate/reset-queue # unwedge the queue
```

## Deeper reading

- [reference.md](reference.md) (same folder): full node-tier walkthrough, ServerFields field list, endpoints/watchdogs, LM cache internals, translateParams special cases.
- [docs/dev/api-contracts.md](../../../docs/dev/api-contracts.md): the `/api/generate` request, response, error and job contracts a client relies on.
- [docs/dev/frontend-create-controls.md](../../../docs/dev/frontend-create-controls.md): the flat body vs `generation-intent/1`, and every Create control's default and gate.
- [engine/docs/ARCHITECTURE.md](../../../engine/docs/ARCHITECTURE.md): engine internals, task types, request JSON and an ace-server sideband table. **Caveat:** where it and the code disagree, the code (`aceClient.ts` `AceRequest` interface + `ServerFields`) is the authoritative schema.
- [docs/dev/plugins-authoring.md](../../../docs/dev/plugins-authoring.md): Lua solver/scheduler/guidance plugins (how `plugin_params` are consumed).
- `docs/plans/`: internal design docs (per-section masking, upstream sync workflow). **Gitignored, local-only: may be absent on a fresh clone.**
- [AGENTS.md](../../../AGENTS.md): build/git rules and log layout.
