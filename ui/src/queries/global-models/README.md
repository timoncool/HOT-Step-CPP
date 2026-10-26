# Global model server state

HOTSTEP-98 owns this area's query options and compatibility projections. The
canonical key factory, retry policy and SSE bridge come from the foundation in
`../keys.ts`, `../contract.ts` and `../sseBridge.ts`. Public local resources omit
auth, worker and backend scope. Backend resources capture the backend in the key
and request URL. Reads consume Query's AbortSignal and throw on HTTP failures.

## Consumer disposition

| Consumer | Query resource or retained client work |
| --- | --- |
| `hooks/usePluginRegistry.ts` and all sampler/global-bar callers | Shared `plugins`; reload invalidates that key for every observer |
| `hooks/useCapabilities.ts`, `BackendCapabilityGate` and global-bar gates | Shared backend-scoped `capabilities`; backend store is a scoped compatibility projection |
| `shared/VramIndicator.tsx`, `terminal/TerminalPanel.tsx` | Shared `vram`; loaded-model dropdown uses shared `loaded-models` only while open |
| `hooks/useEventSource.ts`, `terminal/TerminalPanel.tsx` | Shared log SSE connection and bounded log reducer; batches cleared on teardown |
| `model-manager/useModelRegistry.ts` | `global-models/registry`, retained across modal remount |
| `model-manager/useDownloadStream.ts` | Validated download frames update `global-models/downloads`; one underlying shared SSE connection |
| `model-manager/ModelManagerModal.tsx` | Explicit download/cancel/resume/delete commands; completion, reconnect and deletion invalidate affected model snapshots |
| `GlobalParamBar.tsx`, `ModelsDropdown.tsx`, `AdaptersDropdown.tsx` | `global-models/ace-models`; selections and first-launch timer stay client-owned |
| `BackendToggle.tsx`, `backendStore.ts` | `global-models/backends`; active backend mirror remains for synchronous form hydration |
| `BackendModelsDropdown.tsx`, `Yue2LmAdapterDropdown.tsx` | Backend-scoped `global-models/backend-models`; model selection/import remain explicit commands |
| `Mm3LmAdapterDropdown.tsx` | `global-models/mm3-adapters` |
| `PostProcessingDropdown.tsx` | Shared plugins plus `global-models/pp-vae`, `stablestep`, `sa3-adapters` |
| `training-studio/Sa3TrainCard.tsx` | The same `global-models/sa3-adapters` list for its audition picker, read only while training targets this PC |
| `GenerationDropdown.tsx`, `MasteringDropdown.tsx`, `PostProcessingDropdown.tsx` | One `global-models/references` snapshot; upload/delete invalidates all consumers |
| `CoverArtDropdown.tsx` | `global-models/cover-art`; download/cancel are explicit commands |
| `AdaptersDropdown.tsx` | Folder-filtered `global-models/lm-adapters` and shared `adapter-scan`; scan is a read-only POST owned by Query, fresh per click and shared with shell-queue |
| `ProfilesModal.tsx` | `global-models/profiles`, one list per backend (`backend` filter); save/rename/delete/import invalidates it |
| `SeedManagerDrawer.tsx` | `global-models/seeds`; save/delete/favorite invalidates it; random GET remains an explicit selection command |
| `MonitorBar.tsx`, `vstChainStore.ts`, `VstChainDropdown.tsx` | `global-models/monitor`, `vst-chain`, `vst-scan`; exact-key projections preserve existing store selectors |
| `vstChainStore.ts` preset collection | Family-filtered `global-models/vst-presets` supplies authoritative reads through the existing injectable API; the collection retains unsaved edits, migration decisions and revision conflicts |
| `globalParamsStore.ts` | Form state and local persistence; active-model PUT is a client-to-server mirror command, with no independent read |

Mutations run once as explicit transport calls; they do not use query functions
or automatic retry. Query cache updates and invalidations follow successful
commands. Preset conflicts also invalidate the read-back snapshot. Optimistic
VST chain edits and monitor commands remain client intent until the authoritative
snapshot arrives. Compatibility projections never initiate a network request,
match captured keys, hydrate from existing cache on HMR and unsubscribe on HMR
disposal. Capability projections retain a previous up manifest over a down one,
while the raw cached down manifest keeps recovery polling active. A first down
manifest remains visible so missing models can be explained.

## Endpoints and cadence for Tester

All requests go through `http://localhost:3000` during the assigned runtime job.
No app run belongs to this slice.

| Endpoint | Expected cadence |
| --- | --- |
| `GET /api/plugins` | One shared read on first use; reload invalidates all subscribers |
| `GET /api/capabilities?backend=<id>` | One shared read per backend; 10 seconds while absent/down, stops when up |
| `GET /api/logs/vram` | Shared 1500 ms poll while observed; the only reader is the Terminal box's VRAM badge (`shared/VramIndicator.tsx`) |
| `GET /api/logs/models-loaded` | 2000 ms only while unload dropdown is open |
| `GET /api/model-manager/registry` | First use/stale remount; download completion, reconnect or deletion invalidates |
| `GET /api/model-manager/downloads` | One shared SSE connection; no polling |
| `GET /api/logs` | Existing shared log SSE connection; no polling |
| `GET /api/models` | Shared read; 3000 ms while the DiT list is empty/unavailable, stops once populated |
| `GET /api/backends`, `/api/backends/models?backend=<id>` | First use/stale remount; backend switch/model selection invalidates captured target |
| `GET /api/models/pp-vae`, `/api/models/stablestep`, `/api/models/stablestep/adapters`, `/api/mm3/lm-adapters` | First use/stale remount; model installation/deletion invalidates |
| `GET /api/mastering/references`, `/api/profiles?backend=<id>`, `/api/seeds`, `/api/adapters/lm?folder=...` | First use/stale remount; own writes or manual refresh invalidate |
| `POST /api/adapters/scan` | Read-only scan on explicit request, deduplicated by folder |
| `GET /api/cover-art/status` | First use; 1500 ms while downloading, stops on terminal phase |
| `GET /api/vst/monitor/status` | 300 ms while running; 2000 ms while idle to discover an already-running monitor |
| `GET /api/vst/scan`, `/api/vst/chain` | First use/stale remount; rescan and chain writes invalidate/update |
| `GET /api/preferences/presets/vst-chain` | Collection load, explicit refresh or conflict read-back; no polling |

Polling uses foundation's capped error backoff, at most 30 seconds. Default
freshness is 30 seconds and unused caches remain for five minutes. Monitor
status and preset conflict read-back are immediately stale. Query retries apply
only to network/5xx failures; submit commands never retry automatically.

## Acceptance steps

1. Open global generation/sampler/postprocessing consumers together. Count one
   plugin read and one capability read per backend key. Invalidate/reload plugins
   and check every subscriber sees the replacement registry.
2. Open the Terminal box. Its VRAM badge polls at 1500 ms. Hover the unload
   dropdown and confirm its separate 2000 ms list stops when closed. Unload
   once and verify VRAM/list refresh.
3. Capture an initial down capability response, then restore the engine. Verify
   the 10-second probe recovers and stops. Repeat after an up manifest was seen:
   the previous controls stay available while the raw down response is polled.
4. Switch backend A to B with A's capability/model reply delayed. A's response
   must only enter A's cache; B remains selected and shows B's catalogue.
5. Close and reopen Model Manager within 30 seconds. The registry should reuse
   the fresh cache. Complete a download or delete a model and confirm registry,
   backend catalogues, capabilities and availability probes refresh. Disconnect
   and reconnect download SSE, then replay a completed frame: snapshots refresh
   on reconnect, completion callbacks do not repeat for the same job.
6. Exercise reference upload/delete, profile changes, seed favorites, VST
   rescan/chain edits and cover-art download/cancel. Confirm shared lists refresh,
   terminal cover-art polling stops and no command repeats on remount.
7. Switch tabs and return, then repeat HMR while observing network traffic.
   Check that shared streams, query projections and polls do not accumulate.

Focused fake-fetch tests cover shared requests, plugin invalidation, backend
isolation, down/up recovery, cancellation, HTTP failures, registry reuse,
download completion/replay/reconnect/teardown and exact-key projections. Existing
VST preset conflict/import tests also run through the Query-backed API adapter.
Runtime request counts, actual downloads, GPU unload, engine outage and human
visual feedback remain for the assigned Tester job. No reduction is claimed
without Tester's pre-change baseline.
