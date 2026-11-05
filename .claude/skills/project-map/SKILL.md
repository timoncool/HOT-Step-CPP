---
name: project-map
description: Maps every HOT-Step CPP feature to its route file, service, UI folder, and engine subsystem, including port topology and the browser-to-engine request path. Use when you need to find where a feature or bug lives, which tier owns a symptom, or how a request flows through the system.
---

# Project Map & Navigation — HOT-Step CPP

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

HOT-Step CPP is a desktop app for local AI music generation (caption + lyrics in, stereo audio out). It is a heavily-extended fork of acestep.cpp, a C++/GGML port of ACE-Step 1.5, and it now hosts four generation backends: ACE-Step 1.5 (`ace`, the default), MiniMax-Music3 (`minimax-m3`), YuE2 (`yue2`) and Stable Audio 3 (`sa3`). Three tiers:

1. **Engine**: C++17/GGML (CUDA, Vulkan, Metal or CPU) in `engine/`. The ACE-Step pipeline is **LM** (Qwen3 language model that emits `audio_codes`) → **DiT** (Diffusion Transformer, the flow-matching synth stage) → **VAE** (decodes latents to audio). The same `ace-server` process also serves the MM3 (`/mm3/*`), YuE2 (`/yue2/*`) and SA3 (`/sa3-*`) pipelines.
2. **Server**: Node/TypeScript/Express/better-sqlite3 in `server/src/`. Orchestrates the engine, owns SQLite and files, serves the UI.
3. **UI**: React 19/Vite/TanStack Query/Zustand/Tailwind in `ui/src/`. One component folder per "studio" (feature area).

## When to use this skill

- "Where is the code for X?" / "Which file handles Y?"
- Triaging a bug report: deciding whether the fault is UI, Node server, or C++ engine.
- Tracing a request end-to-end (browser → Express → ace-server → audio file).
- Onboarding to any unfamiliar feature before editing it.

## Golden rules (hard constraints — each prevents expensive damage)

1. **C++ changes: rebuild with `.\dev-rebuild.bat` at repo root, NEVER `engine\build.cmd` directly, under any circumstances.** WHY: you cannot reliably tell whether the app is running; the Node server auto-respawns `ace-server.exe` on crash (`server/src/services/aceEngineProcess.ts`, `onExited` in `startAceServer`); killing the engine mid-run causes an infinite respawn + file-lock loop that blocks the build. `dev-rebuild.bat` POSTs `/api/shutdown`, waits for `ace-server.exe` to exit (force-kills after 10 s), then builds. It does NOT relaunch. Start the app again yourself with `dev.bat`, detached (`pwsh -NoProfile -File tools\detach.ps1 dev.bat`), and work at `http://localhost:3000`. `LAUNCH.bat` is the end-user prod path; agents do not run it. Recompile immediately after editing any `engine/src/` or `engine/tools/` file, then run `node tools/engine-smoke.mjs` with the GPU idle.
2. **NEVER `cmake --build . --clean-first`** unless the GGML/CUDA layer itself changed. WHY: CUDA kernel recompilation takes 20+ minutes. For stale `.obj` issues delete only `engine\build\acestep-core.dir\` and `engine\build\Release\acestep-core.lib`.
3. **Engine HTTP/API changes go in `engine/tools/hot-step-server.cpp` (or the family route files under `engine/src/minimax/` and `engine/src/yue2/`), NOT `engine/tools/ace-server.cpp`.** WHY: upstream `ace-server.cpp` is kept as reference and is NOT compiled. The `ace-server` binary is built from `hot-step-server.cpp` (`engine/CMakeLists.txt:482-484`). Edits to `ace-server.cpp` silently do nothing.
4. **Don't `npm run build` during dev.** Type-check instead: `npx tsc --noEmit` in `server\` (covers `server/src/` only), `npx tsc --noEmit -p scripts/tsconfig.json` in `server\` for `server/scripts/`, and `npx tsc --noEmit -p tsconfig.app.json` (or `npx tsc -b`) in `ui\`. A bare `npx tsc --noEmit` in `ui\` checks nothing and exits 0. Only build before user testing.
5. **Git: all work on `master`, no branches. Never `git add -A` or `git add -f`** (re-adds gitignored `.agents/`, `node_modules/`, `/data/`, `logs/`). Stage explicit paths. **Push requires explicit user approval.** Commit locally often. **Any pushed `v*` tag triggers a full multi-platform CI release build**; use a `-CI-Test` suffix for throwaway tags.
6. **Three fork-hook files break on upstream sync** (see Failure signatures). After any sync run `powershell -File engine\verify-hooks.ps1` (`engine\build.cmd` also runs it before every compile). The `pipeline-synth-ops.cpp` hook loss is SILENT: it compiles but all solvers/schedulers/guidance go dead.
7. **Don't visually verify UI with a browser agent; ask the human user.** Browser agent is fine for API-only checks.
8. **Node 20 to 24 LTS, 24 recommended.** The `engines` field enforces `<25`.
9. **PowerShell is the shell: use `;` not `&&`** (unless pwsh 7 chaining). Windows 11 environment.

## Process / port topology

```
dev.bat (agents: run detached, app at http://localhost:3000)
  (LAUNCH.bat = end-user prod launcher; agents don't use it)
  └─ Node server  server/src/index.ts → Express :3001 (host 0.0.0.0, config.ts:312-315)
       ├─ serves prebuilt ui/dist/ with SPA fallback, when ui/dist exists (index.ts:182-206)
       ├─ /api/* routes → SQLite <data root>/library/hotstep.db
       ├─ /audio → <data dir>/audio/, /references → <data dir>/references/ (index.ts:153-181)
       ├─ /listening → _experiments/_LISTENING/<folder>/ score sheets, loopback only (index.ts:164-166)
       └─ spawns child: ace-server (C++ engine) on 127.0.0.1:8085 (config.ts:217-218),
          owned by services/aceEngineProcess.ts
Dev mode adds Vite on :3000 (HMR); ui/vite.config.ts proxies /api, /audio, /references → 127.0.0.1:3001.
/listening is NOT proxied: score sheets open on :3001 directly.
tsx watch (server/restart-loop.cmd) auto-restarts the Node server on server-code changes.
```

- Engine spawn args are built in `startAceServer()` (`server/src/services/aceEngineProcess.ts:149-202`): `--models --host --port`, plus conditional `--adapters`, `--user-plugins` (release data folder only), `--keep-loaded`, `--noise-profile`, `--draft-lm` (deliberately disabled by default, `config.ts:232-237`), `--vae-chunk`, `--vae-overlap`. `index.ts` only calls it from the bootstrap block.
- Crash respawn limiter (`aceEngineProcess.ts:295-339`): abnormal exit → restart after 3 s; 3 crashes inside 30 s → give up, engine marked not-ready. An empty models folder is a first run, not a crash (`aceEngineProcess.ts:298-307`): the engine prints `ERROR: no models found` and exits 1, no restart budget is spent, `/api/health` reports `engine.noModels: true`, and the engine starts by itself when a Model Manager download finishes (`retryIfGaveUp()`, `index.ts:389-395`) or when the 5 s models-folder check finds a top-level `.gguf` that has stopped growing (`watchForModels`, `aceEngineProcess.ts:44-68`).
- A deliberate stop (`stopAceServer`, used by training jobs that borrow the GPU) never respawns; `/api/generate` answers 503 "paused for training preprocessing" until `restartAceServer()` runs. MM3 and YuE2 keep their model selection in engine memory, so `index.ts` re-applies it after every restart through `onEngineRestarted` (`index.ts:397-422`).
- Portable Windows CUDA builds download the cuBLAS/cudart DLLs from Hugging Face before the first engine start (`ensureRequiredRuntime`, `index.ts:246-349`; bootstrap `index.ts:354-384`).
- Config comes from `.env`: repo root in a checkout without a data folder, `<data root>/settings/.env` with one (`config.ts:36-40`). It is bootstrapped from `.env.example` on first launch (`config.ts:95-109`). The Settings UI can only edit whitelisted `EXPOSED_ENV_KEYS` (`config.ts:442-470`); keys in `RESTART_REQUIRED_KEYS` (`config.ts:483-488`) need a restart; the rest hot-reload via `reloadEnvConfig()` (`config.ts:494-636`).

## Request path: browser → engine (main generation)

Key term: **`AceRequest`**: the JSON request object sent to the engine's ACE-Step routes, typed at `server/src/services/aceClient.ts:84-233` (caption/lyrics/bpm/seed, LM sampling params, DiT params like `inference_steps`/`guidance_scale`/`scheduler`, model routing, adapter fields, `plugin_params`). **`aceReq`** in code = the instance of this object for the current job.

1. **UI**: `ui/src/components/create/CreatePanel.tsx` + `ui/src/stores/globalParamsStore.ts` (Zustand store holding all generation params; `getGlobalParams()` assembles the body). `App.tsx`'s `handleGenerate` hands it to `enqueueSimpleGen()` in `ui/src/stores/audioGenQueueStore.ts` (the client queue executor), which calls `generateApi.submit` in `ui/src/services/api.ts` → `POST /api/generate`. When the server owns the queue, the same body goes to the durable audio queue (`/api/audio-queue`) instead, which submits it through the same code. Other studios submit through the same route, or through `/api/workflows` jobs that queue `/api/generate` bodies on the durable audio queue.
2. **Route**: `server/src/routes/generate.ts` (~1060 lines). `submitGeneration()` (`generate.ts:344-464`) checks engine readiness, auth and the backend, builds the job envelope (`services/generation/envelope.ts`) and stores the job in the in-memory `jobs` Map (`generate.ts:60`); the outcome is also written to the `generation_jobs` ledger (`services/generation/jobLedger.ts`) so `/status` can answer after a restart. `enqueueGeneration()` (`generate.ts:191-273`) runs jobs one at a time on the shared GPU lane (`services/generation/gpuLane.ts`), with the retry policy from the envelope. `runGeneration()` (`generate.ts:156-184`) evicts other model families and calls the backend's `generate()`.
3. **ACE-Step backend** (`server/src/services/backends/ace/generate.ts`; MM3, YuE2 and SA3 have their own `generate.ts` under `services/backends/<id>/`):
   - `translateParams()` (`services/generation/translateParams.ts:47`) converts UI params → `AceRequest` (`ace/generate.ts:51`).
   - **LM phase**: `aceClient.submitLm` (`ace/generate.ts:170`); results cached in `services/generation/lmCache.ts`, keyed by the LM seed (the DiT seed when the two are tied) plus the LM-affecting params.
   - **CRITICAL "sideband" gotcha** (`ace/generate.ts:192-213`, cache-hit twin at `:96-112`): server-only fields (`adapter_runtime_quant`, alignment timing, rebase, `plugin_params`, …) are not part of the C++ LM struct and do NOT survive the `/lm` round trip. Synth requests are rebuilt as `{ ...aceReq, <LM output fields> }`, taking only `audio_codes`, caption, lyrics, bpm, duration, keyscale, timesignature and the per-output `lm_seed` from the echo. Never whitelist-echo the LM response.
   - Source audio/latent loading for cover/repaint/seed tasks (`ace/generate.ts:398-428`, `services/generation/sourceAudio.ts`, `sourceLatentCache.ts`).
   - **Per-track synth**: `aceClient.submitSynth`, or `submitSynthMultipart` when source/reference audio or latents are attached (`ace/generate.ts:815-827`).
   - Optional Whisper lyric transcription (`services/whisperTranscribe.ts`), auto-trim (`services/autoTrim.ts`), then the **post-processing chain** `runPostProcessingChain` (`services/generation/postProcessing.ts:359`), in order: quality check (unmastered) → shared vocal separation → StableStep SA3 refine → PP-VAE re-encode → Spectral Lifter → Vocal Naturalizer → pre-VST gain offset → VST chain → Mastering → final LUFS normalizer → final peak guard → quality check (mastered). Each song row is inserted into `songs` at the end (`ace/generate.ts:1247-1259`).
4. **Engine client**: `services/aceClient.ts` wraps every engine endpoint. **The engine answers HTTP on one thread**: during DiT/VAE/adapter compute it cannot answer at all. Timeouts are deliberately generous: 15 s quick / 30 s poll / 300 s result / 30 min SA3 (`aceClient.ts:18-21`). A "hung" health check mid-generation is normal.
5. **Watchdogs** in `pollUntilDone` (`services/generation/pollUntilDone.ts`): 500 ms poll; a stage that ticks ("Step n/total") and goes quiet for 2 min triggers a probe of the engine, and the job is cancelled as stalled only if the engine does not report it still running; non-ticking stages (model loads, adapter merge, VAE decode) get a 15 min quiet window; YuE2 NAR gets a frame-scaled window. Wall-clock timeout clamped to 5 to 360 min, default 45. An engine crash or three 404s on the job id fail the job at once.
6. **Progress**: the UI polls `GET /api/generate/status/:id` (through TanStack Query, `ui/src/queries/shell-queue/queue.ts`); live log lines come from `routes/logs.ts` (`pushLog`/`subscribeLines`, `logs.ts:35,50`).
7. **Engine job model**: `POST /lm`, `/synth`, `/understand`, `/vae`, `/codes-decode` or `/warm` returns a job id; poll `GET /job?id=`; fetch the result with `GET /job?id=&result=1` (`&latent=1` for the post-DiT latent); cancel with `POST /job?id=&cancel=1`. Node job ids ≠ engine job ids (mapped by the `aceJobId` field, `services/generation/jobTypes.ts`).

**Engine HTTP endpoints** (`engine/tools/hot-step-server.cpp:3438-5176`): `POST /lm /synth /understand /vae /codes-decode /warm /job /models/unload /models/restore-policy /pp-vae-reencode /sa3-refine /sa3-synth /supersep/separate /supersep/release /supersep/recombine /spectral-lifter`; `GET /health /worker-drain /props /logs /jobs /job /plugins /vram /models/loaded /sa3/props /supersep/progress /supersep/result /supersep/serve /` (embedded webui). Family routes are registered from the same file: `mm3_register_routes` (`engine/src/minimax/mm3-server.h`) and `mm3_register_job_routes` (`mm3-job.h`) for `/mm3/*`, `yue2_register_routes` (`engine/src/yue2/yue2-server.h`) for `/yue2/*`.

## Feature → files map (route ⇄ service ⇄ UI ⇄ engine)

API mounts are all registered in `server/src/index.ts:98-151`. Route files in `server/src/routes/`, services in `server/src/services/`, UI folders in `ui/src/components/` (paths starting `ui/src/` or `services/…Api.ts` are outside that folder). The generated route list is [docs/dev/api.md](../../../docs/dev/api.md).

| Feature / Studio | API mount | Route file | Key service(s) | UI folder / file | Engine piece |
|---|---|---|---|---|---|
| Create (main generation) | `/api/generate` | generate.ts | generation/* (envelope, gpuLane, pollUntilDone, translateParams, lmCache, postProcessing), backends/<id>/generate.ts, aceClient, autoTrim, whisperTranscribe | create/ + global-bar/ | `/lm` `/synth` → pipeline-lm.cpp, pipeline-synth*.cpp; `/mm3/synth`, `/yue2/synth`, `/sa3-synth` |
| Backend toggle + capabilities | `/api` (`/backends`, `/backends/active`, `/backends/models`, `/capabilities`, `/mm3/*`, `/yue2/import-adapter`) | backends.ts | backends/registry.ts, backends/<id>/index.ts | global-bar/BackendToggle.tsx, `ui/src/stores/backendStore.ts`, `ui/src/hooks/useCapabilities.ts`, shared/BackendCapabilityGate.tsx | `/mm3/props`, `/yue2/props`, `/sa3/props` |
| MiniMax-Music3 backend | via `/api/generate` (`/mm3/stream/:id` live audio), `/api` | generate.ts, backends.ts | backends/minimax/ | global-bar/Mm3LmAdapterDropdown.tsx, Mm3RendererRow.tsx, `ui/src/stores/mm3StreamStore.ts` | engine/src/minimax/ (`/mm3/*`) |
| YuE2 backend | via `/api/generate` (`/yue2/plan` score preview) | generate.ts | backends/yue2/ | global-bar/Yue2LmAdapterDropdown.tsx | engine/src/yue2/ (`/yue2/*`) |
| Stable Audio 3 backend | via `/api/generate` | generate.ts | backends/sa3/, sa3Tokenizer.ts | global-bar/Sa3SamplerControls.tsx | sa3-*.h; `/sa3-synth`, `/sa3-refine`, `/sa3/props` |
| Insta-Gen (genre-first quick gen) | `/api/inspire`, `/api/workflows` (`insta-*` kinds) | inspire.ts | lireek/llm/registry, lireek/prompts, workflows/instaGenWorkflow | insta-gen/ (`ui/src/services/inspireApi.ts`) | `/lm` in inspire mode |
| Lyric Studio ("Lireek") | `/api/lireek` | lireek.ts + lireek/{crudRoutes,llmRoutes,mm3Routes}.ts | lireek/{llmService, geniusService, profilerService, prompts, slopDetector, exportService, lyricWorkflow, mm3Compose, llm/} | lyric-studio/ (`ui/src/services/lireekApi.ts`, `lyricWorkflowApi.ts`) | none (external LLMs) |
| Song library | `/api/songs` | songs.ts | audioCrop, disco-analyzer, library/{importTrack,exportZip}, generation/rePostProcess | library/, player/, details/, playlist/ | none |
| Player waveform peaks | `/api/audio` | audio.ts | audio/peaks.ts | `ui/src/audio/peaksClient.ts` | none |
| Models dropdown | `/api/models` | models.ts | aceClient (`GET /props`), training/adapterLayout | global-bar/ModelsDropdown.tsx, BackendModelsDropdown.tsx | model-registry.h, model-store.cpp |
| Model Manager (downloads) | `/api/model-manager` | modelManager.ts | modelDownloadService.ts + `server/src/data/model-registry.json` | model-manager/ (ModelManagerModal, opened from the global bar's model dropdowns) | none |
| Adapters (LoRA/LoKr fine-tunes) | `/api/adapters` | adapters.ts | adapters/stMetadata.ts, training/adapterLayout, generation/adapterSections.ts | create/AdaptersAccordion.tsx, global-bar/AdaptersDropdown.tsx | adapter-merge.h, adapter-runtime.h, adapter-cancel.h, lokr-*.h |
| Stem Studio (separation) | `/api/stem-studio` | stemStudio.ts (2 modes: `extract` = DiT per-track `/synth`; `supersep` = GGML separation) | aceClient, audioConvert | stem-studio/ (`ui/src/services/stemStudioApi.ts`) | supersep.cpp, bs-roformer-ggml.h, mdx23c-ggml.h, supersep-stft.h; `/supersep/*` |
| SuperSep raw proxy | `/api/supersep` | supersep.ts | audioConvert | `ui/src/services/supersepApi.ts` (used by stem-studio/, cover-studio/, `stores/coverStemsStore.ts`) | `/supersep/*` |
| Stem Builder (layering) | `/api/workflows` (`layer-render` kind) | workflows.ts | workflows/repaintLayerWorkflow.ts | stem-builder/ (`ui/src/services/repaintLayerWorkflowApi.ts`) | `/synth` (lego task) |
| Repaint Studio | `/api/workflows` (`repaint-render` kind), renders through the generate queue | workflows.ts | workflows/repaintLayerWorkflow.ts, generation/sourceAudio.ts | repaint-studio/ | sampler-repaint.h |
| Song Builder | `/api/builder` | songBuilder.ts | songBuilder/sectionOps.ts, workflows/workflowJobs (section kind) | song-builder/ | seed-latent path of `/synth` (`ace/generate.ts:420-428`) |
| Cover Studio (audio covers) | `/api/generate`, `/api/analyze`, `/api/yue2-cover`, `/api/workflows` (`cover-*` kinds) | generate.ts, analyze.ts, yue2Cover.ts | generation/sourceAudio.ts, workflows/coverWorkflow.ts, yue2Cover.ts, yue2Takes.ts, backends/yue2/coverArrangement.ts | cover-studio/ (`ui/src/services/coverWorkflowApi.ts`, `yue2CoverApi.ts`) | cover task types in request.h; `/yue2/*` for YuE2 covers |
| STORM (continuous stream) | `/api/generate/storm/*` | generate.ts | storm/{slot,drift}.ts, streamSessions/ | storm/ | ACE `/synth`, `/yue2/synth` or `/sa3-synth` slots (`storm/slot.ts`) |
| MIDI Studio | `/api/midi-studio` | midiStudio.ts | muscriptor.ts, midiParser.ts, midiLyrics.ts | midi-studio/ (`ui/src/services/midiStudioApi.ts`) | `ace-midi` binary (engine/tools/ace-midi.cpp) |
| Training Studio | `/api/training` | training.ts | training/* | training-studio/ (`ui/src/services/trainingApi.ts` and siblings, `stores/trainingStore.ts`) | `ace-train` (engine/tools/ace-train.cpp, engine/src/train/), `ace-caption` (engine/src/moss/), `/understand` |
| Training on another PC | `/api/workers`, `/api/training/worker` | workers.ts | training/trainingWorkers.ts, training/workerUpdate.ts, training/yue2Mirror.ts | training-studio/ (worker selection) | none |
| Compare Studio | `/api/compare` | compare.ts | compare/{analyzer,store,decode} | compare-studio/ (`ui/src/services/compareApi.ts`, `stores/compareStore.ts`) | none |
| Cover Art (images) | `/api/cover-art` | coverArt.ts | coverArt/{coverArtService, coverArtDownloader, promptBuilder, sceneLlm} | library/CoverArtPromptModal.tsx, global-bar/CoverArtDropdown.tsx | external sd-cli (not the music engine) |
| Mastering | `/api/mastering` | mastering.ts | generation/postProcessing.ts, audio/peaks.ts | create/MasteringSection.tsx, global-bar/MasteringDropdown.tsx | mastering.h, tools/mastering.cpp |
| Post-processing re-run | `/api/songs/:id/postprocess` | songs.ts | generation/rePostProcess.ts | global-bar/PostProcessingDropdown.tsx, `stores/postProcessStore.ts`, shared/PostProcessActivity.tsx | `/pp-vae-reencode`, `/spectral-lifter`, `/sa3-refine` |
| VST3 chain + monitor | `/api/vst` | vst.ts (+ vstControl.ts helpers) | spawns vst-host (`config.ts:379-386`) | global-bar/VstChainDropdown.tsx, `ui/src/stores/vstChainStore.ts` | tools/vst-host.cpp |
| Lua plugins (solvers etc.) | `/api/plugins` | plugins.ts | aceClient (`GET /plugins`) | global-bar/PluginControls.tsx, SamplerPluginControls.tsx, `ui/src/hooks/usePluginRegistry.ts` | lua-plugin.h, lua-plugin-registry.h, hot-step-sampler.h |
| Assistant (in-app help chat) | `/api/assistant` | assistant.ts | lireek/llm/registry, `server/src/data/assistant-knowledge.md` | assistant/ (`ui/src/services/assistantApi.ts`) | none |
| Settings (.env editor, GPUs, data folder) | `/api/settings` (`/env`, `/gpus`, `/data-folder/*`) | settings.ts, settingsDataFolder.ts | config.ts `reloadEnvConfig`, gpuDevices.ts, `server/src/dataroot/` | settings/ (`ui/src/services/dataFolderApi.ts`) | none |
| Parameter profiles | `/api/profiles` | profiles.ts | none (JSON files under `<data dir>/profiles/`, one subfolder per non-ACE backend) | global-bar/ProfilesModal.tsx | none |
| Seeds manager | `/api/seeds` | seeds.ts | none (JSON files under `<data dir>/seeds/`) | global-bar/SeedManagerDrawer.tsx | none |
| Presets and settings documents | `/api/preferences` | preferences.ts | preferences/, workflows/revisions.ts | `ui/src/services/preferencesApi.ts`, `presetCollection.ts`, `settingDocument.ts` | none |
| Studio drafts | `/api/studio-drafts` | studioDrafts.ts | studioDrafts/ | shared/StudioDraftPicker.tsx, `ui/src/services/studioDraftMirror.ts` | none |
| Durable audio queue | `/api/audio-queue` | audioQueue.ts | audioQueue/intentQueue.ts (submits through `submitGeneration`) | `ui/src/services/audioQueueApi.ts`, `stores/audioGenQueueStore.ts` | none |
| Workflow jobs + documents | `/api/workflows` | workflows.ts | workflows/{workflowJobs,revisions}.ts; kinds registered by studio modules | `ui/src/services/workflowApi.ts`, `ui/src/queries/studios/workflows.ts` | none |
| Stream sessions (analysis, recording) | `/api/stream-sessions` | streamSessions.ts | streamSessions/ | `ui/src/services/streamSessionsApi.ts` | none |
| Intent preview | `/api/resolve` | resolve.ts | generation/resolve/ | `generateApi.preview` in `ui/src/services/api.ts` | none |
| Export / import | `/api/export-import` | exportImport.ts | exportImport/, assets/audioAssets.ts, library/importTrack.ts | `ui/src/services/exportImportApi.ts` | none |
| DAW client facade | `/api/v1/daw` | dawV1.ts | daw/ | none (external DAW clients; `HOTSTEP_DAW_KEY`) | none |
| Update notices | `/api/updates` | updates.ts | updates/ | `ui/src/services/updatesApi.ts` | none |
| Uploads (audio/latent/cover) | `/api/upload` | upload.ts | latentFormat.ts, assets/audioAssets.ts | shared/LatentImport.tsx, studio upload fields | none |
| Downloads / export | `/api/download` | download.ts | audioMetadata.ts, downloadSave.ts, ffmpeg | player/, library/ | tools/mp3-codec.cpp |
| Logs / VRAM / terminal | `/api/logs` (`/`, `/vram`, `/models-loaded`, `/models-unload`) | logs.ts | logger.ts | terminal/TerminalPanel.tsx (inside shared/ActivitySidebar.tsx), shared/VramIndicator.tsx | `/vram`, `/models/loaded`, `/models/unload` |
| Health / lifecycle | `/api/health`, `/api/shutdown` (+ `/restart`) | health.ts, shutdown.ts | engineState.ts (at `server/src/engineState.ts`, NOT services/), aceEngineProcess.ts | App.tsx health probes | `/health` |
| Auth (local single-user) | `/api/auth` | auth.ts | db/database.ts | `ui/src/context/AuthContext.tsx` | none |
| Analyze (BPM/key, tags of an upload) | `/api/analyze` (`/`, `/metadata`) | analyze.ts | Essentia binary (`config.ts:294-296`), training/essentiaClient, training/audioMeta | cover-studio/ | none |
| Ear-test score sheets | `/listening` (not under `/api`) | listening.ts | none | none (static pages, see the ear-test-scoresheet skill) | none |

**UI views** (`viewFromUrl`/`urlForView`, `ui/src/App.tsx:103-150`): default `create`, plus `insta-gen`, `lyric-studio`, `cover-studio`, `stem-studio`, `stem-builder`, `song-builder`, `storm`, `midi-studio`, `training-studio`, `compare-studio` (URL `/compare`), `repaint`, `library`, `settings`. The assistant is an overlay panel (`hs-showAssistant`); the terminal lives in the activity sidebar. Engine-specific studios are hidden from the sidebar by `STUDIO_NEEDS` in `ui/src/components/sidebar/Sidebar.tsx`. Client stores in `ui/src/stores/`: Zustand (`globalParamsStore`, `backendStore`, `trainingStore`, `vstChainStore`, `coverStemsStore`) and hand-rolled `useSyncExternalStore` stores (`playbackStore`, `audioGenQueueStore`, `streamingStore`, `mm3StreamStore`, `abCompareStore`, `compareStore`, `discoStore`, `postProcessStore`). Server snapshots live in TanStack Query (`ui/src/queries/`).

**Engine source map**: full per-subsystem table in [reference.md](reference.md). Fast rules: LM phase → `engine/src/pipeline-lm.cpp` / `qwen3-lm.h`; DiT synth → `pipeline-synth*.cpp` / `dit.h`; all solver/scheduler/guidance routing → `hot-step-sampler.h` (upstream `dit-sampler.h` is bypassed); adapters → `adapter-merge.h` / `adapter-runtime.h`; VAE → `vae.h` / `vae-enc.h` (GGML only; ONNX Runtime was removed in `a44753f1`); MM3 → `engine/src/minimax/`; YuE2 → `engine/src/yue2/`; SA3 → `sa3-*.h`; training → `engine/tools/ace-train.cpp` + `engine/src/train/`; HTTP server → `engine/tools/hot-step-server.cpp`.

**Plugins**: solvers / schedulers / guidance are hot-loadable Lua files in `engine/plugins/{solvers,schedulers,guidance}/`. The registry (`init()`, `engine/src/lua-plugin-registry.h:39-63`) also scans the repo-root `plugins/` overlay, which adds more `md_*` solvers, schedulers and guidance plus `postprocess/` (`md_audio_tiled*`), and in a release the data folder's `plugins/` (passed as `--user-plugins`). A plugin in the user folder overrides a shipped one of the same name; a duplicate name in the repo-root overlay is skipped with a warning. The release workflow packages both `engine/plugins` and root `plugins/`. Drop a `.lua` in the right subdir → appears in UI next launch, no C++ rebuild. Adding a solver/scheduler/guidance = write a Lua plugin, never edit `dit-sampler.h`. Guide: `docs/dev/plugins-authoring.md`. Native headers in `engine/src/solvers|schedulers|guidance/` are legacy; new work goes in Lua.

## Data layer

- **Find the live data folder with `cd server; npx tsx scripts/data-paths.ts`** (read-only). With a data folder, `config.data.dir` is `<data root>/library/` and a `DATA_DIR` value resolves against the data root; a checkout without one resolves `DATA_DIR` against `server/` (`resolveDataDir()`, `config.ts:27-30`). A stale legacy `data/` directory may exist at repo root with a plausible-looking but outdated `hotstep.db`. Do not trust it, and do not delete it without asking.
- SQLite: `<data dir>/hotstep.db` (better-sqlite3). Core tables are created in `server/src/db/database.ts:37-316`: users, songs, playlists, playlist_songs, artists, lyrics_sets, profiles, generations, settings, album_presets, audio_generations, builder_projects, builder_sections, training_datasets, yue2_rung_scores, yue2_album_scores, compare_sessions, compare_takes, compare_scores, audio_metrics. Feature services create their own tables on first use: `audio_intents`/`audio_queue_state`/`audio_queue_imports` (`services/audioQueue/intentQueue.ts`), `generation_jobs` (`services/generation/jobLedger.ts`), `workflow_documents`/`document_import_receipts` (`services/workflows/revisions.ts`), `workflow_jobs`/`workflow_events` (`services/workflows/workflowJobs.ts`), `audio_assets` (`services/assets/audioAssets.ts`), `lyric_workflow_items` (`services/lireek/lyricWorkflow.ts`). **Lireek tables are unified into hotstep.db** (`index.ts:18`). `config.lireek.dbPath` (`config.ts:372-374`) only names the legacy `lireek.db` that `migrateLireekData()` imports once (`database.ts:532-625`).
- Files under `<data dir>`: `audio/` (outputs; `audio/takes/<id>/` holds YuE2 build sessions), `references/` (mastering and timbre refs), `stems/<jobId>/`, `midi/<jobId>/`, `seeds/`, `profiles/`, `vst/{states,chain.json}`, `lyrics/` (exports), `training/`, `muscriptor/`. **Never delete generated audio outputs or experiment artifacts on your own judgment; the user verifies results by ear. Ask first.**
- Logs: `<data root>/logs/YYYY-MM-DD_HH-MM-SS/{ace_engine.log, node_console.log, generations/gen_<uuid>_<task>.log, training/<kind>_<jobId>.log}` (`LOGS_DIR`, `config.ts:22`; checkout `logs/` without a data folder). Newest folder = current session. Generation failure → matching `gen_*.log` first, then cross-ref the other two.

## Exact commands (PowerShell)

```powershell
pwsh -NoProfile -File tools\detach.ps1 dev.bat   # How agents run dev.bat: Vite :3000 HMR + Node :3001 tsx watch; check http://localhost:3000
.\LAUNCH.bat                    # Prod launcher for end users; agents never run it
.\dev-rebuild.bat               # C++ rebuild: graceful shutdown -> wait -> build (then start dev.bat yourself)
node tools/engine-smoke.mjs     # Render receipt after an engine rebuild, GPU idle
npx tsc --noEmit                            # from server\  (server/src only; do NOT npm run build in dev)
npx tsc --noEmit -p scripts/tsconfig.json   # from server\  (server/scripts)
npx tsc --noEmit -p tsconfig.app.json       # from ui\  (a bare tsc --noEmit in ui\ checks nothing)
npx tsx scripts/data-paths.ts               # from server\  (active data root and children)
# Stale .obj without the 20-min CUDA recompile:
Remove-Item -Recurse -Force engine\build\acestep-core.dir; Remove-Item -Force engine\build\Release\acestep-core.lib
powershell -File engine\verify-hooks.ps1   # After any upstream sync
```

## Failure signatures (symptom → cause → fix)

| Symptom | Cause | Fix / first file |
|---|---|---|
| `[ace-server] Restarting in 3 seconds... (crash N/3)` then `Crashed 3 times within 30s — giving up.` | Missing DLL next to ace-server.exe or a model that fails to load (an empty models folder is NOT this, see the next row) | `<data root>/logs/<newest>/ace_engine.log`; respawn logic `aceEngineProcess.ts:295-339` |
| `No models installed — waiting for a download in the Model Manager.`; `/api/health` shows `engine.noModels: true` | Empty models folder (first run), not a crash | Download a model in the Model Manager or copy a `.gguf` into the models folder; the engine starts by itself (download hook, or the 5 s folder check) |
| Solver/scheduler/guidance UI options silently do nothing | Upstream sync overwrote the `hot-step-sampler.h` include in `engine/src/pipeline-synth-ops.cpp:9`. **Compiles fine, silent.** | Run `engine\verify-hooks.ps1`; restore the include |
| Compile error in model-store.h / dit.h after sync | Lost `hot-step-params.h` include (`model-store.h:52`) or `adapter-merge.h`+`adapter-runtime.h` (`dit.h:11,14`) | Restore includes; `verify-hooks.ps1` confirms |
| Generation fails "stalled" | `pollUntilDone` watchdog: a ticking stage went quiet for 2 min and the engine stopped reporting it running, or a quiet stage passed its 15 min window | Check `gen_*.log` for the last engine phase; `adapter_precompute` ~17 s LoKr cold-start is normal |
| `/api/generate` answers 503 "paused for training preprocessing" | A training job stopped the engine to borrow the GPU (`isEngineSuspended()`) | Wait for the job; the engine restarts afterwards |
| `/api/health` or engine health flaky mid-generation | Engine answers HTTP on one thread; can't answer during compute (`aceClient.ts:6-8`) | Not a bug unless it persists after the job ends |
| Synth params vanish after LM phase (adapter/solver settings ignored) | Sideband fields don't survive the `/lm` round trip | Rebuild synth req from original `aceReq` + LM fields only (`backends/ace/generate.ts:192-213`) |
| Infinite ace-server respawn + file locks during rebuild | You ran `engine\build.cmd` with the app up | Use `.\dev-rebuild.bat` |
| npm install fails / weird dep errors | Node outside 20 to 24, or native addon ABI mismatch | Use Node 24 and install matching dependencies |
| Vulkan pinned-memory alloc failure in VAE | VAE chunk too large | Tune `ACESTEPCPP_VAE_CHUNK` / `ACESTEPCPP_VAE_OVERLAP` in `.env` |
| First-launch hang "Downloading CUDA runtime..." | Portable CUDA DLL bootstrap (`index.ts:354-384`) | Wait; if the download fails the engine starts on CPU only |
| `/synth` answers 400 for an `.onnx` VAE or `stream_mode` | ONNX Runtime and the streaming preview pipeline were removed from the engine (`a44753f1`) | Use a GGUF/safetensors VAE; there is no streaming preview on ACE |
| Accidental CI release build | Pushed a `v*` tag | Use `-CI-Test` suffix for throwaways; see `docs/dev/releasing.md` |

## Institutional knowledge

- **VALIDATED: the real server binary**: `ace-server.exe` builds from `engine/tools/hot-step-server.cpp`; upstream `ace-server.cpp` is uncompiled reference (`engine/CMakeLists.txt:482-484`).
- **VALIDATED: LM echo sideband**: server-only `AceRequest` fields never survive the `/lm` round trip; synth requests must be rebuilt from the original request + LM output fields only, never echoed back (`backends/ace/generate.ts:192-213`).
- **VALIDATED: GGML clobbers input buffers** in the per-section adapter-masking path, so masks must be re-uploaded every step.
- **VALIDATED (by revert): regional self-attn isolation broke musical continuity** and was reverted in commit `ee041e1`. The `adapter_section_isolation` field still exists in `AceRequest` (`aceClient.ts:176`) and the engine still parses it into `g_hotstep_params` (`hot-step-server.cpp:836-837`), but nothing reads it. Do NOT re-attempt without a design that preserves cross-section musical coherence; naive per-section attention masking is known to break continuity.
- **VALIDATED: draft-LM speculative decoding is deliberately disabled** (`config.ts:232-237`): GGML per-call overhead (~10 ms) negates the speedup.
- **VALIDATED: `--keep-loaded` is spawn-time** (restart required); flips the engine ModelStore to EVICT_NEVER so the ~17 s LoKr precompute happens once. A separate per-request `?keep_loaded=1` co-resident toggle also exists (`config.ts:238-246`). Default OFF (VRAM trade-off).
- **VALIDATED: LM results cached by LM seed + LM params** (`lmCache.ts`): changing only synth params reuses cached codes; changing the LM seed (or the DiT seed while the two are tied) re-runs the LM.
- **UNVALIDATED / HYPOTHESIS: LoKr cross-base basin nudge**: `rebase_source`/`rebase_beta` fields exist (`aceClient.ts:196-197`) implementing β·(S−T) toward the adapter's training base; efficacy unproven. Non-turbo XL bases still fail despite ~99% weight identity (basin sensitivity, not drift). **When multiple adapters are stacked, the basin re-base applies once per STACK, never once per adapter** (issue #72; regression class fixed in 168dcb5).
- Repo root has stray build artifacts (`*.obj`, `nul`, `rebuild_engine.*`): untracked experiment leftovers. Do NOT delete them without asking; they may be the user's test artifacts.

## Deeper reading

- [reference.md](reference.md) (this folder): full engine source map, complete engine endpoint list, `AceRequest` field catalogue, data/config details.
- `docs/dev/architecture.md` (tiers, request path, backends), `docs/dev/api.md` (generated route list), `docs/dev/api-contracts.md` (request, job, audio queue and workflow contracts).
- `FEATURES.md`: full feature catalogue. `engine/docs/ARCHITECTURE.md`: engine internals, CLI, request JSON. `docs/dev/plugins-authoring.md`: Lua plugin authoring. `docs/dev/releasing.md`: release runbook. `server/src/data/assistant-knowledge.md`: in-app assistant KB.
- `docs/plans/` is **gitignored (local-only)**: internal design/investigation docs (perf, adapters, `upstream-sync-workflow.md`); may be absent on a fresh clone.
