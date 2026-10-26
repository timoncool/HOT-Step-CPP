# TanStack Query — integration audit and migration plan

Audited 2026-10-20 against `ui/`. Pinned to `@tanstack/react-query` ^5.104.3.
This page is a **worklist**: it records where the integration uses deprecated
APIs, bypasses the cache, or lacks a mutation lifecycle, and what fixes it. It
does not re-decide what is a query and what is a command — that ownership stays
with [ui-server-state.md](ui-server-state.md) (policy contract + consumer
manifest) and the
[integration audit in architecture.md](architecture.md#ui-server-state-integration-audit)
(per-path dispositions, area owners). The phases below are the work items for
the named areas, following the R1–R6 request model used by that audit. P1 and
P2 have landed on `master`; P3 to P6 are open and P7 is deferred.

**Status (re-checked 2026-10-23 at `be3c85f8`):**

| Phase | Status | Evidence |
|---|---|---|
| P1 | Done | `63023472`, merged as `e2396255`. No non-test `fetchQuery` in `ui/src`; the 73 sites call `queryClient.query()`; the `no-restricted-syntax` rule is in `ui/eslint.config.js` |
| P2 | Done, with the residuals below | `18897e6c` and `c7498e67`. JSON traffic runs through one axios instance (`services/axiosClient.ts`): the request interceptor adds the bearer from `setAuthToken`, which AuthProvider calls beside `startAuthSession`, and the response interceptor is the global catch (`handleFailure`). `<ToastContainer />` is mounted in `main.tsx`. `queries/writes.ts` holds `writeAndInvalidate` and `useUpload`, and `queries/invalidation.ts` holds `queryManager`. Uploads stayed on XHR in `httpClient.ts` `upload()`, because only XHR reports upload progress; their failures go through the same `handleFailure`. Known-slow routes (adapter scan and browse, Whisper retranscribe, mastering run, external LLM calls) pass `SLOW_ROUTE_TIMEOUT_MS` (300 s) instead of the 60 s default |
| P3 | Not started | The SeedManagerDrawer, CoverArtPromptModal and Yue2LmAdapterDropdown writes are still raw `fetch`; the nuke handlers still name no keys at the write site (F4#11); component error toasts remain (residual 1). MetadataEditorModal's upload is `useUpload` and its save is `writeAndInvalidate`, both since P2, but neither is a `useMutation` with contextual success |
| P4 | Not started | Every F3 and F4 item tagged P4 is unchanged |
| P5 | Not started | No `useEnvSettingsForm`; `SettingsPanel.tsx:200–218` still seeds the form with `queryClient.query()` in an effect |
| P6 | Not started | `queries/` still holds area folders; F4 #3 to #13 and #17 are open, and only the doc-only notes (#14 to #16) are recorded. The one exception is `queries/sa3-previews/` (added in `2367aefc`), which already has the target shape: one `keys.ts` composer whose `list` key registers and whose `dataset` prefix invalidates |
| P7 | Deferred | Unchanged |

At P2 the verification was `tsc -b` clean, the full UI suite at the 354-test
baseline (2 pre-existing `.css` import failures), lint at or below baseline
with zero new issues, and the P2 exit-criteria greps passing. The Scale
table's first column and the F1 ledger are a pre-sweep snapshot.

**P2 residuals (close in P3):** (1) component-level error toasts on command
failures double-toast with the interceptor until they are removed (guardrail
1: the interceptor is the single notifier). At `be3c85f8` they are in 13
files: `App.tsx`, `CoverRecentSongs.tsx`, `CoverStudio.tsx`,
`CuratedProfileModal.tsx`, `LyricStudioV2.tsx`, `PresetSettingsModal.tsx`,
`ProfilesTab.tsx`, `RecordingsTab.tsx`, `WrittenSongsTab.tsx`,
`RepaintStudio.tsx`, `UnifiedRecentSongs.tsx`, `SongBuilder.tsx` and
`StemBuilder.tsx`. Only one CoverStudio catch already skips an `ApiError`.
(2) Sidecar and peaks reads stay on raw fetch, outside the JSON/upload scope
and the `src/services` grep: `utils/lrcUtils.ts` (`fetchLrc`),
`utils/wordLrcUtils.ts` (`fetchLyricsJson`), `queries/shell-queue/lyrics.ts`
(`fetchScoreAbc`) and `audio/peaksClient.ts`. (3) XHR uploads have no timeout,
so a stalled upload hangs (pre-existing). (4) A regression after P2:
`components/training-studio/Sa3TrainCard.tsx:114` (added in `8c0a6951`) calls
`client.invalidateQueries` directly, so `queries/invalidation.ts` is no longer
the only call site. The same card builds its `training-data/sa3-status`
options inline (`Sa3TrainCard.tsx:28–33`), against hook-folder rule 1.

## Foundation in place

| Piece | File | What it pins down |
|---|---|---|
| Client + policy | `queries/client.ts`, `queries/contract.ts` | staleTime 30 s, gcTime 5 min, window-focus refetch off, retry ≤ 2 (network/5xx only), mutation retry off, `pollWhile` at 1.5 s active / 30 s error cap |
| Provider | `main.tsx:21` | One HMR-stable client outside auth and every view |
| Key schema | `queries/keys.ts` | 5-slot `['hs', resource, scope, id, filters]`; session epoch via `startAuthSession`; `SONG_VIEWS` invalidation set |
| Area slices | `queries/<area>/` | cover-lyrics, global-models, shell-queue, studios, training-data, training-runs: options factories, read helpers, per-area hooks. `queries/sa3-previews/` is a folder of its own whose keys sit in the training-runs area |
| SSE → cache | `queries/sseBridge.ts` | `bridgeEvents`: coalesced invalidate, validated in-place update, once-ledger for replay-safe side effects |
| Invalidation | `queries/invalidation.ts` | `queryManager` and the `invalidateKeys` / `invalidateQuery` primitives; the bridge flushes through `invalidateKeys` (P2) |
| Write layer | `queries/writes.ts` | `writeAndInvalidate(client, fn, plan)` with `events(...)` or `keysPlan(...)` plans, and the `useUpload()` mutation (P2) |
| Transport | `services/axiosClient.ts`, `services/httpClient.ts` | One axios instance: bearer interceptor, global error catch (`handleFailure`), 60 s default timeout. `ApiClient.upload()` stays on XHR and `streamEvents()` on fetch, both reporting failures through `handleFailure` (P2) |

Full contract: [ui-server-state.md](ui-server-state.md). This page audits
against it.

## Scale (non-test grep)

| API | 2026-10-20 (pre-sweep) | 2026-10-23 (`be3c85f8`) | Note |
|---|---|---|---|
| `useQuery` | 116 sites, 75 files | 123 sites, 78 files | Declarative reads, the established norm |
| `useMutation` | **1** (`components/midi-studio/MidiStudio.tsx:47`) | 8: `MidiStudio.tsx:51`, `queries/writes.ts:66` (`useUpload`), `settings/UpdatesPanel.tsx:26`, `training-studio/Sa3TrainCard.tsx` ×5 | Most writes are still an imperative call plus a cache follow-up |
| `queryClient.fetchQuery` | **73** | 0 | Deprecated in 5.104.x (F1); the 73 sites now call `queryClient.query()` |
| `prefetchQuery`, `ensureQueryData` | 0, 0 | 0, 0 | Not used |
| `getQueryData` | 1 (`queries/sseBridge.ts:96`) | 3: `queries/sseBridge.ts:97`, `audio/peaksClient.ts:42`, `queries/global-models/projection.ts:7` | Synchronous cache reads: the bridge's update guard, cached peaks for the first paint, store projections. Not deprecated |
| `invalidateQueries` | ~40 files | 2 files: `queries/invalidation.ts` and `Sa3TrainCard.tsx:114` | P2 moved every call into the manager; the Sa3TrainCard call is residual 4 |
| `useQueryClient` | 39 files | 42 files | Mostly cache follow-up after writes |

## Findings

### F1 — `fetchQuery` is deprecated; 73 production sites

**Status: fixed in P1** (`63023472`). Every site in the ledger below now calls
`queryClient.query()`, and ESLint rejects a new `fetchQuery`. The ledger keeps
its pre-sweep line numbers. Its other dispositions (F4, P4, P5) are tracked in
their own sections.

Verified against the installed `@tanstack/query-core` 5.104.x source
(`build/modern/queryClient.js`):

| Deprecated method | Replacement | Deprecation text (abridged) |
|---|---|---|
| `fetchQuery(options)` | `queryClient.query(options)` | "Use queryClient.query(options) instead. This method will be removed in the next major version." (`queryClient.js:422`) |
| `prefetchQuery(options)` | `queryClient.query(options).catch(noop)` | same, plus "You can swallow errors with `.catch(noop)`" (`queryClient.js:436`) |
| `ensureQueryData(options)` | `queryClient.query({ ...options, staleTime: 'static' })` | (`queryClient.js:137`) |

Semantics are identical for the sweep: `query()` builds the query, fetches only
when `isStaleByTime(staleTime)`, and forces `retry: false` — byte-for-byte the
body of today's `fetchQuery`. The rename is a **pure mechanical change with no
behavior delta**; each site keeps its existing `staleTime`.

Site ledger (grouped; lines are non-test call sites):

| File | Lines | Key / factory | Disposition |
|---|---|---|---|
| `queries/shell-queue/queue.ts` | 55, 60, 63, 68, 71, 76 | `read*` helpers: audioQueueItems, generationStatus, workflowJob, song, songIds, selectedPath | P1 rename — one-shot by design (contract) |
| `queries/shell-queue/preferences.ts` | 43, 57 | presets / setting options | P1 rename |
| `queries/global-models/presets.ts` | 21 | preset family | P1 rename |
| `queries/cover-lyrics/reads.ts` | 24 | `snapshotOptions` (`readSnapshot`) | P1 rename |
| `queries/cover-lyrics/workflows.ts`, `queries/studios/workflows.ts` | 25, 26 each | workflow job / document | P1 rename |
| `audio/peaksClient.ts` | 52 | `peaksOptions(url)` | P1 rename |
| `utils/yue2CaptionSource.ts` | 226 | inline `'shell-queue/yue2-captions'` | P1 rename + F4#12 (key to a factory, stop swallowing errors) |
| `services/stemStudioApi.ts` | 196 | inline duplicate of `extractResultOptions` | P1 rename + F4#7 (call the factory) |
| `services/supersepApi.ts` | 134 | inline duplicate of `separationResultOptions` | P1 rename + F4#7 |
| `services/yue2CoverApi.ts` | 199 | request body in key (`filters: body`) | P1 rename; cardinality note F4#15 |
| `components/settings/SettingsPanel.tsx` | 202, 203 | `envOptions()` / `gpusOptions()` | **P5** — becomes `useQuery` inside `useEnvSettingsForm` (scoped refactor below) |
| `components/shared/FileBrowserModal.tsx`, `training-studio/FolderPicker.tsx`, `BatchImportWizard.tsx`, `NewDatasetWizard.tsx` | 70 / 49 / 166 / 81 | browse / scanPreview | P1 rename; NewDatasetWizard also P4 (useQuery) |
| `components/global-bar/AdaptersDropdown.tsx`, `create/AdaptersAccordion.tsx` | 125 / 140 | `adapterScanOptions(dir)` | P1 rename |
| `components/insta-gen/InstaGenPanel.tsx` | 211, 384 | inline song duplicate / inspirePrompt | P1 rename + F4#4 (use the factory) |
| `components/lyric-studio/RecordingsTab.tsx` | 123 | inline generation-status duplicate | P1 rename + F4#3 (use the factory) |
| `components/song-builder/SongBuilder.tsx` | 210 | project options | P1 rename |
| `components/stem-studio/StemStudio.tsx` | 239 | `extractResultOptions(jobId)` | P1 rename (already uses the factory — correct) |
| `stores/trainingStore.ts` | 549, 563, 581, 611, 797, 837, 939, 963, 1034, 1063, 1594, 1640, 1644 | `tq.*` imperative training reads | P1 rename — stores cannot use hooks; keep imperative |
| `stores/vstChainStore.ts` | 133, 142, 329 | vst-scan / vst-chain / monitor | P1 rename |
| `stores/backendStore.ts` | 267, 288, 298 | backends / capabilities / backendModels | P1 rename |
| `components/training-studio/Yue2AitkTrainCard.tsx` | 440, 444, 681, 687, 695, 916, 929, 940 | job / jobs / preparation reads | P1 rename |
| `components/training-studio/RefinePanel.tsx` | 140, 171, 191, 204, 336 | job / ladder / jointRuns (staleTime 0) | P1 rename |
| `components/training-studio/ReviewPanel.tsx`, `Yue2LadderReview.tsx` | 76 / 141 | ladder staleTime 0 (batch) | P1 rename |
| `components/training-studio/Yue2OptimisePanel.tsx` | 36 | job | P1 rename |
| `components/training-studio/sendAuditionToCustomGen.ts`, `SendToLyricStudio.tsx` | 14 / 54 | dataset / lyricStudioPreview hand-offs | P1 rename |
| `components/training-studio/useTrainingStream.ts`, `PipelineStageProgress.tsx` | 38 / 67 | job (SSE-event-driven) | P1 rename |
| `App.tsx` | 414 | `songOptions(id)` inside the disco-extraction loop | **P4** — the whole hand-rolled loop becomes a `pollWhile` query; this fetch disappears |

### F2 — Writes have no mutation lifecycle

At the audit the single `useMutation` in the app was `MidiStudio.tsx:47`
(remove MIDI job), and even there all cache work lives inside the helper
(`queries/studios/midi.ts:32–45` today). Every other write was a raw API call
followed by ad-hoc `invalidateQueries` / `setQueryData` scattered across
components and zustand stores. Since P2 those follow-ups go through
`queryManager` (`writeAndInvalidate` in `stores/trainingStore.ts`,
`stores/backendStore.ts` and `stores/vstChainStore.ts`;
`queryManager.keys(...)` in `SettingsPanel.tsx:161` and `:256`), but most
sites still name their own keys and hold their own busy and error state. The
mutations added since are `useUpload` (`queries/writes.ts:66`), Check now in
`settings/UpdatesPanel.tsx:26`, and cancel, start, render and audition in
`training-studio/Sa3TrainCard.tsx`. Consequences:

- no shared pending/error/settled state: each site hand-rolls
  `setSaving` / `setError` useState pairs;
- correctness depends on every call site remembering its own invalidation:
  the three destructive handlers in SettingsPanel name no keys and lean on the
  lireek transport and a parent callback, which miss the Lyric Studio
  audio-generation snapshots after a songs nuke (F4#11);
- the same multipart upload block was copy-pasted 7× (F3). Fixed in P2.

Note: the architecture.md audit deliberately classifies POST/PUT/PATCH/DELETE
as explicit commands ("commands stay explicit"). `useMutation` does not change
that classification — it gives commands a consistent lifecycle (pending,
error, settled, invalidation in one place) without moving them out of the
command layer. The recommendation is scoped accordingly:

- **In React components**: adopt `useMutation` where a write drives visible
  per-write UI state (busy flag, error text, success note). Candidates:
  SeedManagerDrawer ×3 (`SeedManagerDrawer.tsx:46–64`, read side already
  `useQuery`), CoverArtPromptModal submit (`CoverArtPromptModal.tsx:127–163`),
  Yue2LmAdapterDropdown importComfy (`Yue2LmAdapterDropdown.tsx:452–477`),
  MetadataEditorModal save (its upload is `useUpload` since P2), SettingsPanel
  nuke handlers + env save.
- **In zustand stores** (trainingStore, vstChainStore, backendStore,
  audioGenQueueStore): hooks cannot be used. The `runCommand` pattern
  (`queries/training-runs/queries.ts:138–149`: execute, then invalidate on
  success, no retry) was the right shape. P2 standardized it as
  `writeAndInvalidate` (`queries/writes.ts:34–38`), which trainingStore,
  backendStore and vstChainStore now use. audioGenQueueStore (P7) and
  postProcessStore (F3) do not.

### F3 — Reads and timers bypassing the cache

| Site | Pattern | Recommendation | Phase |
|---|---|---|---|
| `components/storm/DriftPanel.tsx:303–318` | raw `fetch` of `/api/generate/storm/control?streamId=` on a 2 s `setInterval` into `useState`; no error state (silent `catch {}`) | `useQuery({ queryKey: stormControl(streamId), refetchInterval: playing ? 2000 : false })`. **Not covered by the architecture.md disposition table; add a row when fixing** | P4, open |
| `components/library/CoverArtPromptModal.tsx:73–95` | preview POST on modal open with hand-rolled `setLoadingPreview`; errors swallowed to empty prompt | `useQuery({ enabled: !!song })` for the preview; submit → mutation (job follow is already a QueryObserver + `pollWhile`) | P4 / P3, open |
| `stores/vstChainStore.ts:215–225` | module-level 1.5 s `setInterval` polling `vstApi.openGuis()` into store state, self-clearing when empty | query with the existing `pollWhile` helper (proven pattern: `capabilitiesOptions`, `queries/global-models/options.ts:35–38`) | P4, open |
| `components/training-studio/NewDatasetWizard.tsx:70–97` | 400 ms debounced scan with a monotonic-seq race guard on top of `queryClient.query(tq.scanPreview(...))` | `useQuery` + debounce + `keepPreviousData` deletes the seq-guard machinery | P4, open |
| Upload cluster: `LatentImport.tsx`, `MetadataEditorModal.tsx`, `MidiStudio.tsx`, `StemBuilder.tsx`, `SourceSelector.tsx`, `RepaintStudio.tsx`, `CoverStudio.tsx` | seven near-identical hand-rolled multipart blocks (token header, error parse, `setUploading`/`setError`) duplicating `apiClient.upload()` | Done in P2: the first six use `useUpload`; CoverStudio calls `apiClient.upload()` (`services/httpClient.ts:121–158`) directly at its four upload sites | P2, done |
| `services/api.ts:492–548` | post-processing functions (`runSongPostProcessing`, `getSongPostProcessingStatus`, `revertSongPostProcessing`, `retranscribeLyrics`), on `apiClient` since P2 (retranscribe passes `SLOW_ROUTE_TIMEOUT_MS`) | remaining: callers (`stores/postProcessStore.ts:190,229`) become mutations. The status read is already a query with `pollWhile` (`postProcessStatusOptions`, `queries/shell-queue/songs.ts:32–37`, observed at `postProcessStore.ts:128`) | P3, open |
| `components/training-studio/useTrainingStream.ts:27–47` + `stores/trainingStore.ts:1110+` | training SSE writes into zustand state that duplicates the job resource in `queries/training-runs`; terminal frames trigger a cache refresh anyway | re-point the reducer at `setQueryData`/invalidate (sseBridge style) to remove the second owner. **Structural, needs owner design** | P7, deferred |
| `stores/audioGenQueueStore.ts:628–690, 1178–1204` | ~100 lines of hand-rolled executor state around reads that already share the query cache (`queries/shell-queue/queue.ts:54–60`) | structural refactor (dynamic `refetchInterval` + separated executor). **Needs owner design** | P7, deferred |
| `stores/compareStore.ts` (Compare studio, added `551b48eb` after the audit's scan) | session list, session, take analysis and diff are read with direct `compareApi` calls into a `useSyncExternalStore` store, and `searchCompareSongs` runs from `CompareStudio.tsx`; nothing goes through Query | options factories in an area slice, observed with `useQuery`; writes through `writeAndInvalidate` | not assigned |

Keep as-is (verified against the architecture.md dispositions): media blobs
(`StemMixer.tsx:125`, `useAbEngine.ts:316`, `useStreamAudio.ts`), the App health
probes (`useServerHealthReload`, `App.tsx:152–181`, and the offline check at
`App.tsx:458–473`, liveness transport), log streams (`useEventSource.ts`),
live note SSE (`MidiPlayer.tsx:103–139`), and the download-progress bridge
(`useDownloadStream.ts`, already SSE-fed, F4#14).

### F4 — Anti-pattern ledger

| # | Site | Issue | Fix | Phase |
|---|---|---|---|---|
| 1 | app-wide | near-total absence of `useMutation` (1 site at the audit, 8 now; see Scale) | F2: component mutations + shared store write helper | P2 done (`useUpload`, `writeAndInvalidate`); P3 open |
| 2 | `App.tsx:388–429` | hand-rolled disco-extraction loop: raw POST then 360×5 s `setTimeout` polling `queryClient.query(songOptions, staleTime 0)`; bypasses retry policy and never invalidates `SONG_VIEWS` | `useQuery` + `refetchInterval: pollWhile(...)`; invalidate song views on completion | P4, open |
| 3 | `components/lyric-studio/RecordingsTab.tsx:123` | inline ad-hoc key + fn byte-equivalent to `generationStatusOptions` (`queries/shell-queue/queue.ts:27–31`); bypasses the factory and its `retry: false` | call the factory | P6, open |
| 4 | `components/insta-gen/InstaGenPanel.tsx:212` | inline duplicate of `songOptions` instead of `readSong` (`queries/shell-queue/queue.ts:67–68`) | call the factory | P6, open |
| 5 | `components/stem-studio/SourceSelector.tsx:48–51` | second resource for one endpoint: inline `studios/recent-songs` duplicating `recentSongsOptions` (`shell-queue/songs.ts:22–25`); `SONG_VIEWS` lists both, but the divergence (resource name, extra `limit` filter) is a maintenance trap | single canonical read; if `limit` matters, add it to the factory's filters | P6, open |
| 6 | `components/lyric-studio/QueuePanel.tsx:181–187` | third spelling of the LM list (vs `trainingQueries.lmAdapters` and `AdaptersDropdown.tsx:66–69` local options); `lmFolder` read from localStorage **during render**, so a storage change mid-mount does not re-key | one factory; folder into state, key derives from it | P6, open |
| 7 | `services/stemStudioApi.ts:179–182`, `services/supersepApi.ts:121–122` | service-layer inline duplicates of `extractResultOptions` / `separationResultOptions` (`queries/studios/resources.ts:22–35`) | call the factories | P6, open |
| 8 | `components/cover-studio/Yue2Build.tsx:192` | positional key surgery in `setQueryData`: splices the old session's scope slot into a freshly built tuple to graft cache identity across a session-id change; any 5-slot schema change silently corrupts this write | add a `keys.ts` helper for "same resource, new session scope" and use it | P6, open |
| 9 | `components/stem-studio/RecentExtractions.tsx:14,36–38`, `components/stem-builder/RecentBuilds.tsx:82–84` | refetch-via-prop-bump: parent increments a numeric prop, child `useEffect` invalidates (through `queryManager.keys` since P2) | invalidate directly at the mutation site (parents already have the client) | P6, open |
| 10 | `components/lyric-studio/ProviderSelector.tsx:25–28` | invalidate-then-bypass-cache: forced refresh invalidates the providers key, then calls `lireekApi.getProviders()` directly and returns data that never enters the cache it just invalidated | return the query's data (the two `useQuery` observers at L134/L238 already hold it) | P6, open |
| 11 | `SettingsPanel.tsx:109–153` | destructive writes that name no keys at the write site. `handleNukeLyrics` / `handleNukeProfiles` (`:127`, `:141`) are covered implicitly: `lireekApi`'s `api()` runs `changed('lireek')` after every command (`services/lireekApi.ts:24`), which refreshes every `cover-lyrics/lireek` snapshot. The songs nuke (`:109`) relies on App's `onNukeComplete` (empties the library, invalidates song views), but the server also deletes Lyric Studio audio-generation entries, and their `cover-lyrics/lireek` snapshots (e.g. `RecordingsTab.tsx:101`) are not invalidated | mutations with invalidation at the write site (cover-lyrics prefixes via `changed()`) | P3, open |
| 12 | `utils/yue2CaptionSource.ts:226–241` | ad-hoc inline key (no factory) and catches **all** errors into an empty result: a failed request is indistinguishable from "no caption source" | key to a factory; surface the error to the caller | P6, open |
| 13 | `queries/studios/shared.ts:10–13` + `adapterScanOptions` ×2 (`global-models/options.ts:11–15`, `shell-queue/settings.ts:45–49`) | duplicate factories with identical key+fn; consumers split between the copies | one owner per resource; re-point imports | P6, open |
| 14 | `components/model-manager/useDownloadStream.ts:16–19` | disabled dummy query as SSE cache slot (`enabled: false`, never-running queryFn, `initialData: []`): non-standard but intentional (the SSE bridge writes the whole frame) | **keep**; recorded as a sanctioned exception in [ui-server-state.md](ui-server-state.md#sanctioned-exceptions) | P6 (doc only), done |
| 15 | `services/yue2CoverApi.ts:199–203` | request body used as query key ("queryable POST"): deliberate, but unbounded key cardinality (every distinct body is a new cache entry gcTime must reap) | **keep** the pattern (noted in [ui-server-state.md](ui-server-state.md#sanctioned-exceptions)); consider bounding `filters` to stable fields | P6, note done; bounding open |
| 16 | `components/global-bar/BackendToggle.tsx:32`, `MonitorBar.tsx:29` | side-effect-only `useQuery` subscriptions: liveness of those cache entries coupled to component mount | **keep**; coupling noted in [ui-server-state.md](ui-server-state.md#sanctioned-exceptions) | P6 (doc only), done |
| 17 | `components/training-studio/Sa3TrainCard.tsx:28–33, 114` (added `8c0a6951`, after P2) | component-local `training-data/sa3-status` options, and a direct `client.invalidateQueries` after a start (P2 residual 4) | move the options into `queries/training-data/`; invalidate through `queryManager` | P6, open |

### F5 — HTTP transport has no global error handling

**Status: fixed in P2** (`18897e6c`), except that uploads stayed on XHR.

At the audit, `services/httpClient.ts` was a fetch-based `ApiClient`: every
JSON method threw a well-shaped `ApiError` (status, body, `currentRevision`,
`reason`), but there was no single place where failures were caught. About 75
catch/console sites across `services/` and `stores/` each decided for
themselves between console-log, silent swallow, or a local `setError`
useState. Uploads were a second transport (raw XHR) only because fetch has no
upload-progress event. Neither `axios` nor a toast library was installed.

Fix (P2): one shared axios instance for all JSON traffic, with one response
interceptor as the global catch: method/URL/header filters, `ApiError`
normalization, centralized logging, `react-toastify` (target pipeline section
below). `axios` ^1.20.0 and `react-toastify` ^11.1.0 are now in
`ui/package.json`. The instance runs on a `globalThis.fetch` adapter
(`axiosClient.ts:167–224`), so the Node test stubs that replace
`globalThis.fetch` keep intercepting requests. Multipart upload stayed on XHR
(`httpClient.ts:121–158`) for its progress event; its failures call the same
`handleFailure`. SSE stays on fetch (`streamEvents`, `sharedEventSource.ts`),
because axios does not stream response bodies. A non-OK response on
`streamEvents` or lireekApi's `consumeSSE` goes through `sseFailure`, which
calls `handleFailure` too; `workflowApi` `followJob` and assistantApi
`chatStream` still handle their own non-OK responses.

## Target pipeline: TanStack Query over Axios with global toast interception

The architectural standard the phases converge to — three layers, one
notification path:

```
[ UI components ]      consume query state, trigger mutations
       ↕
[ TanStack Query ]     caching, dedup, retry, isPending/isError; queryManager for invalidation
       ↕  (named fetcher/mutator functions in services/)
[ Axios transport ]    one instance; request interceptor (auth), response interceptor (global catch)
       ↕
[ REST API ]

[ react-toastify ]     fired only by the axios response interceptor (errors) and mutation onSuccess (successes)
```

Layer ownership:

| Concern | Owner | Rule |
|---|---|---|
| HTTP config, auth headers, timeouts | Axios instance + request interceptor | The bearer comes from a module-level session accessor that AuthProvider updates (the same site that calls `startAuthSession` — the token lives in auth state, not localStorage); fetchers stop threading `token` per call |
| Caching, dedup, retry, component status | TanStack Query (`contract.ts` policy) | Unchanged by the transport swap |
| Error notification | Axios response interceptor — one place | Filters below; query-layer `onError` never toasts |
| Success notification | Mutation `onSuccess` — contextual, per operation | `toast.success` where a write is user-visible; silent for background work |
| Cache coherence after writes | `queryManager` (central invalidation section) | Writes declare events; the manager resolves keys |

### Global error filtering in the response interceptor

The interceptor is the global try/catch. It normalizes every failure to
`ApiError` (keeping `currentRevision`/`reason`), logs centrally, and toasts —
except when a filter says the request is silent:

1. **Method filter** — GET errors never toast. Reads retry per the contract's
   policy and surface inline state (`isError`) in the component; only non-GET
   (commands/mutations) notify globally. This is the primary storm guard — a
   failing 1.5 s queue poll toasts zero times.
2. **URL patterns** — an `IGNORED_URL_PATTERNS` list for endpoints that are
   never user-facing even as commands (health probes, analytics). Today it
   holds only `/api/health` (`axiosClient.ts:61–63`).
3. **Per-request escape** — a request may opt out with an
   `X-Skip-Global-Error` header, for silent background operations that don't
   fit a pattern. `ApiClient` callers set it with `ClientOptions.silent`.

Status mapping on toast (`messageFor`, `handleFailure` in `axiosClient.ts`):
no response, a network failure or a timeout shows "Server unreachable". Any
other failure shows the server's `error` field when the body has one, and
otherwise a short status message (403, 404, or "Server error (HTTP n)"). A 401
of any method, GET included, logs, calls AuthProvider's re-auth callback
(`setOnUnauthorized`) and shows one `toast.warn`, all at most once per 30 s. Aborts
are never errors: no log, no toast. The axios path rejects with a
`CanceledError`, which the interceptor rethrows before `handleFailure`
(`axiosClient.ts:142`); the upload and SSE paths reject with an `AbortError`
without calling it. `contract.ts`'s `isAbortError` recognizes both, so the
retry policy never retries an abort.

### Operational guardrails

1. **No duplicate toasts.** Never `toast.error` in a query-layer callback
   (`useMutation.onError`, `MutationCache`) unless the request explicitly
   bypassed the interceptor (filters 2–3). Errors have exactly one notifier:
   the axios interceptor.
2. **Contextual success, global error.** Success feedback is operation-specific
   and lives in `onSuccess`; server errors are system-wide and live in the
   interceptor. A mutation that needs a custom error *message* bypasses via
   filter 3 and handles its own toast — it does not stack on the global one.
3. **Selective exclusion.** Silent background operations either match an ignore
   pattern or pass the skip header — never a per-site `try/catch` that swallows
   the error (the F4#12 class).

Timeouts: the axios instance has a 60 s default (`DEFAULT_TIMEOUT_MS`), and
known-slow routes pass `SLOW_ROUTE_TIMEOUT_MS` (300 s) through
`ClientOptions.timeout` (`axiosClient.ts:27–32`). A blanket 10 s would break
adapter scans and folder browse.

## Custom hook folders

The repo already has the convention; the audit confirms it and pins what goes
where.

| Location | Contents | Examples today / target |
|---|---|---|
| `queries/<entity>/` (target, P6) — today `queries/<area>/` | One folder per entity type: the key composer (`keys.ts`), options factories, and TanStack hooks (`useQuery`/`useMutation` wrappers); all invalidation lives in the global `queryManager` (central invalidation section) | Today: `global-models/hooks.ts` (`useVram`, `useReferences`), `training-runs/hooks.ts` (`useRunJob`, `useReload`, …), `studios/shared.ts` (`useStudioSongs`). Target: those hooks move into the entity folders they serve |
| `hooks/` | Generic React hooks with **no** server state | `useTheme`, `useStreamAudio`, `useGlobalHotkeys`, `usePersistedState` |

Rules:

1. A hook that takes a `QueryClient`, returns query data, or owns a
   `queryKey` lives under `queries/`, in the entity folder that owns the
   resource (today: the owning area slice), never inline in a component.
   (Violations today: F4#3/#4/#5/#6/#17, and the SettingsPanel env form, see
   the scoped refactor below.)
2. Cross-area shared write helpers live in two files added in P2:
   `queries/writes.ts` holds `writeAndInvalidate(client, fn, plan)` for stores
   (the standardized `runCommand` semantics; the plan comes from
   `events(...)`, resolved by `queryManager`'s event map, or from
   `keysPlan(...)` for keys the write already knows) and `useUpload()` (the
   shared upload mutation over `apiClient.upload`, which stayed on XHR).
   `queries/invalidation.ts` holds the global `queryManager` (central
   invalidation section below).
3. `hooks/` should be free of TanStack imports; if a generic hook needs server
   state, it belongs to an area slice instead. Today `hooks/useCapabilities.ts`
   and `hooks/usePluginRegistry.ts` break this: both observe global-models
   options with `useQuery`.

Moves implied by the audit:

- `useEnvSettingsForm` → `queries/shell-queue/settings.ts`, next to the
  `envOptions` / `gpusOptions` factories it consumes (new file section or new
  `settings-forms.ts` in that slice), in P5; P6 moves it with the slice to
  `queries/env/`.
- Inline query definitions (F4#3/#4/#5/#6/#17, `utils/yue2CaptionSource.ts`) →
  their area factories.
- `hooks/useCapabilities.ts`, `hooks/usePluginRegistry.ts` → the
  global-models slice.
- The upload mutation → `queries/writes.ts`. Done in P2 (`useUpload`).

## Scoped refactor: `useEnvSettingsForm` (SettingsPanel env form)

**Problem statement.** `QueryClient.fetchQuery` is deprecated and slated for
removal in the next major version (replaced by `queryClient.query()` or
declarative hooks — verified against the installed 5.104.x source, F1). P1
renamed the env form's calls to `queryClient.query()`, but the form in
`SettingsPanel.tsx` still fetches imperatively inside a `useCallback` + mount
`useEffect`, manually toggles loading state (`setEnvLoading`), and duplicates
server data into local React state (`envValues` / `envOriginal`). This defeats
TanStack Query's caching, background refetching, and lifecycle management.

**Verified current shape** (`SettingsPanel.tsx` at `be3c85f8`):

- L172–174: draft state — `envValues`, `envOriginal` both
  `useState<Record<string,string>>({})`; `envLoading` a manual flag not derived
  from any query.
- L200–216: `loadEnvSettings = useCallback(async () => { setEnvLoading(true);
  const [data, gpuData] = await Promise.all([queryClient.query(envOptions()),
  queryClient.query(gpusOptions()).catch(() => ({ gpus: [] }))]);
  setEnvValues(data.values); setEnvOriginal(data.values); … } finally {
  setEnvLoading(false); } }, [queryClient])`, run once on mount (L218).
  `envOptions` has `staleTime: 0` (`queries/shell-queue/settings.ts:19–23`), so
  it always hits the network.
- **No sync with refetches.** The form never observes the `shell-queue/env`
  query; if that entry is invalidated while the panel is open, the draft
  silently diverges from the cache until remount. The dirty check (L221)
  compares two local copies, so an external change is invisible until it
  conflicts at save time.
- L241–277: save diffs `envValues` vs `envOriginal`, calls
  `settingsApi.updateEnv(diff)` directly, then `setEnvOriginal({...envValues})`
  + `queryManager.keys(queryClient, envOptions().queryKey)` (L255–256).

**Target hook** (`queries/shell-queue/settings.ts`):

```ts
export function useEnvSettingsForm() {
  const queryClient = useQueryClient();
  // staleTime is already 0 on envOptions — every observation is a fresh read.
  const env = useQuery(envOptions());
  const gpus = useQuery({ ...gpusOptions(), retry: false });

  const [draft, setDraft] = useState<Record<string, string> | null>(null);
  const [original, setOriginal] = useState<Record<string, string>>({});
  const dirty = draft !== null && Object.keys(draft).some(k => draft[k] !== original[k]);
  const dirtyRef = useRef(false);
  dirtyRef.current = dirty;

  // Draft safety: seed on first data; re-seed on background refetch only while
  // clean. While dirty, a refetch updates `original` to the new server values —
  // external changes then surface in the diff (the user sees what a save would
  // overwrite) instead of clobbering the edits in place.
  useEffect(() => {
    if (!env.data) return;
    if (dirtyRef.current) setOriginal(env.data.values);
    else { setDraft(env.data.values); setOriginal(env.data.values); }
  }, [env.data]);

  const save = async () => {
    const diff: Record<string, string> = {};
    for (const [k, v] of Object.entries(draft ?? {})) if (v !== original[k]) diff[k] = v;
    const res = await settingsApi.updateEnv(diff);
    // Invalidate via the manager (P2 has landed) instead of hand-syncing:
    // the refetch re-seeds `original` (and `draft`, since it is now clean)
    // through the effect above. queryManager has no `env` group yet (P6), so
    // name the key, as SettingsPanel does today.
    await queryManager.keys(queryClient, envOptions().queryKey);
    return res; // panel renders the restart-required warning from res
  };

  return { draft, setDraft, dirty, save, isPending: env.isPending, gpus: gpus.data?.gpus ?? [] };
}
```

Design decisions:

| Decision | Rationale |
|---|---|
| `useQuery` instead of `query()` in an effect | The form now observes the cache: any invalidation (other tab, a future write path, the save itself) refetches and re-syncs automatically. Removes 2 of the 73 imperative `query()` sites (deprecated `fetchQuery` calls until P1) *and* fixes the divergence bug at once. |
| No `setEnvLoading` | `isPending` / `isFetching` come from the query. The manual flag (L174) is deleted. |
| Draft seeded only while clean (`dirtyRef`) | Background refetches never clobber in-progress edits. Refetch-while-dirty updates `original`, so external changes appear as new diffs rather than vanishing. Known edge: if the server normalizes a saved value (trim, default), the post-save reseed can mark one field dirty that the user did not touch — acceptable; compare against `res.updated` keys to suppress if it proves noisy. |
| Save stays an explicit command + invalidate | Matches the contract's mutation rule ("on success, invalidate the related lists for the captured scope"). Invalidation goes through `queryManager.keys` with the env key, as the save already does (P2 has landed); an `env` group can replace it in P6. A `useMutation` wrapper is optional polish (P3), not required for correctness. |
| GPUs: `retry: false`, error → empty list | Replaces `.catch(() => ({ gpus: [] }))`; a GPU-detection failure must not fail the env form. |
| `envStatus` / `pendingRestartKeys` / `restarting` stay in the component | UI state, not server state — the hook returns the save result and the panel renders the warning (existing logic L257–271 moves unchanged). |

Steps:

1. Add `useEnvSettingsForm` to `queries/shell-queue/settings.ts`.
2. `SettingsPanel.tsx`: delete L172–174 draft/loading state and L200–218
   `loadEnvSettings` + effect; consume the hook; `envLoading` → `isPending`;
   `handleEnvSave` body → `save()` (restart-warning logic stays).
3. Add `queries/shell-queue/settings-form.test.ts`: seed-on-first-load,
   refetch-while-clean re-seeds, refetch-while-dirty keeps draft + updates
   original, save invalidates and re-seeds.
4. Update the consumer manifest row for SettingsPanel in ui-server-state.md
   (disposition changes from "one-shot fetchQuery" to "useQuery observer +
   clean-only draft seeding").

## Central invalidation: entity folders, one key composer, closure invalidation

**Question.** Can a global singleton handle all invalidation in one place?
**Answer — yes, and it is one object: the global `queryManager`
(`queries/invalidation.ts`) is the only place in the app that invalidates.
Each entity type gets its own folder under `queries/` owning exactly one key
composer; the manager imports those composers and exposes one method group per
entity — with as many methods as that entity has distinct invalidation needs,
because different writes dirty different request sets. Registering a query and
invalidating it use the same key shape by construction, not by convention.**

### Target layout

One folder per entity type, nested under `queries/`. Shared entities (read by
more than one area, or dirtied by writes from elsewhere) sit at the top level;
area-local entities nest one level down. Each folder holds up to three files —
the composer, options and hooks; invalidation is never in an entity folder:

```
queries/
  client.ts, contract.ts        # unchanged — policy + HMR-stable client
  keys.ts                       # low-level 5-slot primitives (serverKey, scopeOf)
  invalidation.ts               # the one global queryManager — all invalidation lives here
  songs/                        # shared entity
    keys.ts                     # songKeys — the composer (below)
    options.ts                  # songOptions() built on songKeys
    hooks.ts                    # useSongLibrary(), useSong(id)
  env/  gpus/  sidecars/  datasets/ ...     # shared entities
  training-runs/jobs/           # area-local entity, nested
```

Moves are code organization only: the 5-slot tuple is the cache identity, so
re-folding files changes no request and no cache entry.

### One composer per entity — registration and invalidation share it

Today there are two composers in `keys.ts`: `serverKey()` builds full keys for
registration (options factories) and `resourceKey()` builds prefixes for
invalidation — two places to keep the same shape, which is how inline key
duplicates (F4#3–7, #12, #13) and string-prefix predicates (`workerReads`,
training-data/queries.ts:160) keep appearing. The target gives each entity one
composer object exposing both sides:

```ts
// queries/songs/keys.ts — the same object is called at registration time
// (queryOptions) and at invalidation time; they cannot drift apart.
import { serverKey, type KeyParts } from '../keys';

export const songKeys = {
  // full keys — registration: queryOptions({ queryKey: songKeys.list() })
  list: (p?: KeyParts) => serverKey('songs', { auth: true, ...p }),
  detail: (id: string, p?: KeyParts) => serverKey('song', { auth: true, id, ...p }),
  // prefixes — invalidation: c.invalidateQueries({ queryKey: songKeys.all() })
  all: () => ['hs', 'songs'] as const,
  details: () => ['hs', 'song'] as const,
  /** One song's detail entry — the exact key registration builds (composer reuse). */
  detailOne: (id: string) => serverKey('song', { auth: true, id }),
  /** Every cached view of song rows across areas (today's SONG_VIEWS, keys.ts:184–187). */
  views: () => [songKeys.all(), songKeys.details(), /* recent-feed prefixes via their composers */],
};
```

Because both sides call the same object, a scope change or rename updates
registration and invalidation together. What this dissolves:

- inline key+fn duplicates in components (F4#3–7, #12, #13) — "call the factory" becomes "call the composer";
- string-prefix predicates like `workerReads` (`.startsWith('training-data/')`) — become scope-narrowed prefixes from the composer;
- positional key surgery (F4#8, Yue2Build splicing a scope slot into a tuple) — becomes `songKeys.detail(id)` at the new session epoch.

### One global queryManager — all invalidation in one object

There is exactly one place in the app that invalidates cache entries: the
global `queryManager` in `queries/invalidation.ts` (one direct call has crept
back in since P2, F4#17). Entity folders keep their
composers; the manager imports them and exposes **one method group per entity,
with as many methods as that entity has distinct invalidation needs** — a song
write dirties every view of song rows, while an in-place patch dirties only one
detail entry, so `songs` gets both:

```ts
// queries/invalidation.ts — the single global queryManager. It is the only
// file in the app that calls invalidateQueries; everything else (mutations'
// onSuccess, store writeAndInvalidate, the SSE bridge) calls into it.
import { songKeys } from './songs/keys';
import { sidecarKeys } from './sidecars/keys';

export const queryManager = {
  // songs — multiple methods: different writes dirty different request sets
  songs: {
    /** Library + every detail + every recent feed (today's SONG_VIEWS). */
    views: (c: QueryClient) =>
      Promise.all(songKeys.views().map(k => c.invalidateQueries({ queryKey: k }))),
    /** One song's detail entry, after an in-place patch. */
    one: (c: QueryClient, id: string) =>
      c.invalidateQueries({ queryKey: songKeys.detailOne(id) }),
  },
  sidecars: {
    /** All lrc / lyrics-json entries, after a crop or retranscription. */
    all: (c: QueryClient) =>
      Promise.all(sidecarKeys.resources().map(r => c.invalidateQueries({ queryKey: r }))),
  },
  env: {
    values: (c: QueryClient) => c.invalidateQueries({ queryKey: envKeys.all() }),
  },
  /* …one group per entity */

  /** Event map: writes declare what happened; the manager resolves it to the
   *  methods above. Parameterized methods are called directly when the write
   *  knows the id (e.g. a patch). */
  events: {
    'song:written': (c: QueryClient) => queryManager.songs.views(c),
    'sidecars:changed': (c: QueryClient) => queryManager.sidecars.all(c),
    'env:updated': (c: QueryClient) => queryManager.env.values(c),
  },
};

/** The single entry point for writes — mutations, stores and the SSE bridge
 *  all call this; none of them names a key or a prefix. */
export function invalidateEvents(c: QueryClient, events: EventRef[]): Promise<void>;
```

What P2 shipped (`queries/invalidation.ts`) is smaller than this sketch. The
groups are `songs.views` (`SONG_VIEWS`) and `sidecars.all` (`lrc` and
`lyrics-json`), plus `keys(c, ...keys)` and `filter(c, filter)` for writes
that already know their keys or need a predicate. The event map has
`song:written` and `sidecars:changed`, resolved by `invalidateEvents`. Under
them sit the two primitives that call `invalidateQueries`: `invalidateKeys`
and `invalidateQuery`. There is no `songs.one` or `env` group yet, and no
entity composers (P6). Most writes call `queryManager.keys(...)` with keys
built by their area's factories.

What this buys:

- one import surface for all cache coherence. At the audit that knowledge was
  scattered across ~40 files with `invalidateQueries`; after P2, grepping
  `invalidateQueries` found exactly one file (two at `be3c85f8`, F4#17);
- a store write **cannot forget** invalidation: `writeAndInvalidate` takes its
  plan as a required argument (the F4#11 class, writes that name no keys,
  becomes unexpressible);
- the SSE bridge's mapping is derived from the manager's event map instead of a
  second hand-maintained copy (`sseBridge.ts`). Not done: P2 only routes the
  bridge's flush through `invalidateKeys`; each caller of `bridgeEvents` still
  names its own keys.

What stays out of the manager: in-place patches (`setQueryData`) and targeted
removals tied to a lifecycle moment (the session-epoch `removeQueries` in
`startAuthSession`, `forgetPeaks` after a file rewrite) — those stay where the
moment happens, built on the same composers. The manager owns coherence
invalidation only.

### Migration mapping (area folders → entity folders)

| Today | Target |
|---|---|
| `queries/shell-queue/songs.ts` + `SONG_VIEWS` (`keys.ts:184–187`) / `invalidateSongViews` (`invalidation.ts:63`, re-exported at `keys.ts:191`) | `queries/songs/` — composer, options, hooks; SONG_VIEWS dissolves into `songKeys.views()`, invalidated via `queryManager.songs` |
| `queries/shell-queue/settings.ts` (`envOptions`, `gpusOptions`, stem stats, data folder, update notice) | `queries/env/` (with `useEnvSettingsForm`), `queries/gpus/`, … |
| `queries/shell-queue/lyrics.ts` + `invalidateSidecars` (`invalidation.ts:64`, re-exported at `keys.ts:191`) | `queries/sidecars/` — composers; invalidation via `queryManager.sidecars` |
| `queries/training-data/queries.ts` (datasets, jobs, statuses) | `queries/datasets/`, `training-runs/jobs/`; `workerReads` predicate → composer scope-narrowed prefix |
| `queries/studios/resources.ts` (extract / separation results) | `queries/extractions/`, `queries/separations/` |
| Area hooks (`training-runs/hooks.ts`, `global-models/hooks.ts`, `studios/shared.ts`) | Move into the entity folders they serve |

The full per-resource mapping is produced during P6 execution, one area at a
time; the rows above are the pattern. `queryManager` shipped in **P2**
(`18897e6c`). The old per-area helpers became thin re-exports of its groups
(`invalidation.ts:63–64`, re-exported through `keys.ts:191`), the reverse of
the original plan. As P6 re-orgs each entity, its method group closes over the
entity's composer and the old helper is deleted. Consumed by P3 mutations and
the sseBridge re-pointing throughout.

## useQuery vs useMutation strategy

Decision rules:

| Situation | Tool |
|---|---|
| Data is displayed, polled, shared across components, or must survive remount | `useQuery` — already the norm (123 sites at `be3c85f8`) |
| One-shot read that seeds a form/wizard and must never be overwritten by background refetch | imperative `query()` with `staleTime: 0` — the existing pattern (training-data/queries.ts:113–116); the F1 ledger keeps these, renamed only |
| A write drives visible per-write UI state (pending / error / success) and dirties cached resources | `useMutation` in the component (P3 list) |
| A write from a zustand store or other non-React code | `writeAndInvalidate` (`queries/writes.ts:34–38`) — hooks cannot be used outside components; the `runCommand` precedent (training-runs/queries.ts:138–149) standardized in P2 |
| Queryable POST (idempotent, body in the key) | stays a query — sanctioned exception F4#15, never a mutation |

Structural rules:

1. **One owner per resource.** Every read goes through the owning entity's
   options factory (today: the area slice); no inline key+fn in components
   (F4#3–7, #12, #13).
2. **Writes declare effects, not mechanics.** A mutation or store write names
   the events it dirties (`invalidateEvents(c, [...])`); `queryManager`
   resolves them to key prefixes. Call sites stop enumerating invalidation
   prefixes directly.
3. **Optimistic updates: none justified today.** Every visible write is either
   slow (generations, training) or already cache-patched by design (song writes
   patch rows, then invalidate). Revisit only if a sub-100 ms write feels laggy.
4. **SSE and mutations share one path.** `bridgeEvents` maps to the same
   event map as writes (`queryManager.events`), so a server-side change and a
   local write can never drift apart in what they invalidate. Target: today
   the bridge only flushes its callers' keys through `invalidateKeys`.
5. **Errors have one notifier: the axios interceptor.** Error toasts live in
   the transport layer's response interceptor (target pipeline section);
   success toasts are contextual and live in mutation `onSuccess`; query-layer
   `onError` never toasts — it exists only for non-notification reactions.

## v5.104 features reviewed — pivots considered

Checked against the installed `@tanstack/query-core` 5.104.x before deciding:

| Feature | Verdict | Note |
|---|---|---|
| `QueryCache` / `MutationCache` callbacks (`onError`, `onSettled`) | **Reject (superseded)** | The axios response interceptor is the global catch (target pipeline section); a second error path in `MutationCache` would double-toast. Query-layer callbacks stay for non-notification reactions only |
| `staleTime: 'static'` (5.104.x, `queryClient.js:137`) | Optional (P6) | Entries that never go stale; `peaks` uses `Infinity` today (equivalent). Switch for self-documenting intent, no behavior change |
| `useSuspenseQuery` | **Reject** | The app renders explicit loading states; draft forms and the HMR-stable client do not play well with suspense boundaries. No benefit over `useQuery` here |
| Infinite queries | N/A | No server-side pagination exists (recent songs is `limit=50`, no cursor) |
| Query tags | **Reject** | The 5-slot resource-prefix key schema already scopes invalidation (`SONG_VIEWS`); tags would be a parallel second system |
| React Query Devtools | Defer | Not installed; add as a dev dependency only if cache debugging becomes routine |

## Migration plan

Phases are independent after P1 and each ships as one reviewable change per
area. Owner areas follow the architecture.md audit. P1 and P2 are done; the
status table at the top records what each phase left behind.

| Phase | Scope | Owner area(s) | Risk | Exit criteria |
|---|---|---|---|---|
| **P1 — deprecation sweep** (done) | Rename all 73 `fetchQuery` → `queryClient.query()` (F1 ledger); update the one doc reference (`ui-server-state.md:133`); add an ESLint `no-restricted-syntax` rule banning `queryClient.fetchQuery` so the sweep is permanent | foundation | Low — mechanical, semantics identical (verified in installed source) | `npx tsc -b` + full UI suite green; zero non-test `fetchQuery` in `ui/src` (grep) |
| **P2 — HTTP foundation + shared write layer** (done; uploads stayed on XHR, residuals at the top) | Ordered steps: **(1)** migrate `ApiClient`'s JSON methods and `upload()` from fetch/XHR onto one shared axios instance (F5) — upload progress and abort come from the same API, deleting the XHR special case; SSE stays on fetch. **(2)** request interceptor injects the bearer from the auth session's module-level accessor (ending per-call token threading); one response interceptor = the global catch with the target pipeline's filters — GET errors silent by default, `IGNORED_URL_PATTERNS`, `X-Skip-Global-Error` escape header, status-mapped toasts (401 deduped warn + re-auth flow), aborts never toast; `<ToastContainer />` (react-toastify) in `main.tsx`; default timeout with per-request overrides for slow routes. **(3)** new `queries/writes.ts`: `writeAndInvalidate` for stores (standardized `runCommand` semantics) + `useUpload()` mutation over axios upload. **(4)** new `queries/invalidation.ts` — the global `queryManager` (central invalidation section above; methods wrap today's per-area helpers until P6 closes them over entity composers; SSE bridge mapping derived from it). **(5)** collapse the 7 upload blocks (F3); re-point store writes | foundation + studios | Medium–High — every request path touches the new transport; steps 1–2 land first as their own reviewable change, before 3–5 build on it | `grep "fetch(" src/services` empty outside SSE paths; one interceptor owns error normalization and toasts; no hand-rolled multipart block left; uploads keep progress + abort; store writes share one helper and a required event list; one object owns all invalidation |
| **P3 — component mutations** | `useMutation` for: SeedManagerDrawer ×3, CoverArtPromptModal submit, Yue2LmAdapterDropdown importComfy, MetadataEditorModal save+upload (the upload is `useUpload` since P2), SettingsPanel nuke handlers (fixes F4#11) + env save — contextual `toast.success` in `onSuccess`, redundant per-site `setError` pairs deleted (F2), error toasts stay with the axios interceptor; **also remove now-redundant component-level error toasts on command failures app-wide** (13 files at `be3c85f8`, listed under P2 residual 1; until this lands, a failed command double-toasts: interceptor + component catch); the post-processing callers in `stores/postProcessStore.ts` become mutations (the `api.ts` functions themselves moved onto `apiClient` in P2) | per-area (global-bar, library, settings/shell-queue) | Medium — error UX moves to the transport layer; invalidation moves to the write site | No raw fetch POST/DELETE in those components; nuke handlers invalidate cover-lyrics prefixes via `queryManager`; no `toast.error` in any query-layer callback (grep) |
| **P4 — useQuery conversions** | DriftPanel 2 s poll → `refetchInterval` (add its disposition row to architecture.md), CoverArtPromptModal preview → `useQuery(enabled)`, vstChainStore openGuis interval → query + `pollWhile`, NewDatasetWizard scan → `useQuery` + `keepPreviousData` (delete seq guard), App.tsx disco loop → `pollWhile` query | storm, library, global-models, training-data, shell-queue | Medium — polling semantics change; verify intervals match the old cadence | No `setInterval`/`setTimeout` network loops left in those files |
| **P5 — env form refactor** | `useEnvSettingsForm` (scoped section above) | shell-queue | Low–Medium — form now tracks external edits (behavior change, intended) | Draft-safety tests green; manifest row updated |
| **P6 — entity folders, composer unification, dedup** | Re-org `queries/` into per-entity folders (central invalidation section): extract one key composer per entity shared by registration and invalidation, move hooks and invalidation closures alongside; then duplicate factories (F4#13), inline key duplicates (#3/#4/#5/#6/#7/#12) — trivial once composers exist, Yue2Build key surgery → composer (#8), refreshTrigger props (#9), ProviderSelector bypass (#10); doc-only notes for sanctioned exceptions (#14/#15/#16) | per-area | Medium — large mechanical diff, no behavior change (cache identity is the tuple, not the folder) | Every `queryKey` in an options factory comes from an entity composer; no string-prefix invalidation left (`grep startsWith src/queries` empty); SONG_VIEWS and `workerReads` dissolved into composers; no inline `queryKey`/`queryFn` in components (grep) |
| **P7 — structural (deferred)** | audioGenQueueStore projection loop; useTrainingStream → cache re-targeting (F3) | shell-queue / training-runs | High — executor redesign | Owner design doc first; no code before review |

Ordering: P1 went first because it is mechanical, unblocks nothing else's
*code* but removes the deprecation warnings and locks in the new API. Within
P2 the HTTP foundation (steps 1–2: axios + global interceptor + toasts) was
planned to land first, since `useUpload`, `writeAndInvalidate` and
`queryManager` all build on it. In practice all of P2 landed in one commit,
`18897e6c`, with the slow-route timeouts following in `c7498e67`. With P2 in, P3–P6 can
proceed in parallel across areas. P7 stays deferred until an owner designs it;
its reads already share the cache, so deferral costs only duplicated executor
state, not extra requests.

## Verification

```bash
cd ui && npx tsc -b
cd ui && node --import ../server/node_modules/tsx/dist/loader.mjs --test --test-force-exit --test-timeout=60000 "src/**/*.test.ts"
node tools/docs/check-docs.mjs   # from repo root; expect only missing-screenshot warnings
make dev                          # manual: hot UI on :3000, API on :3001
```

Per-phase greps (run from `ui/`):

```bash
grep -rn "fetchQuery" src --include="*.ts" --include="*.tsx" | grep -v "\.test\.ts:"   # P1: empty
grep -rn "fetch(" src/services --include="*.ts"                                        # P2: only transport internals and SSE — axiosClient's fetch adapter, httpClient streamEvents, workflowApi followJob, assistantApi chatStream, lireekApi consumeSSE
grep -rn "\.invalidateQueries(" src --include="*.ts" --include="*.tsx" | grep -v "\.test\.ts:"   # P2: only queries/invalidation.ts (at be3c85f8 also Sa3TrainCard.tsx:114, F4#17)
grep -rn "new FormData(" src/services --include="*.ts"                                      # P2: only httpClient upload()
grep -rn "queryKey:" src/components --include="*.tsx" | grep -v "Options()"            # P6: no inline keys
```

## Cross-references

- [ui-server-state.md](ui-server-state.md) — policy contract, key schema, SSE
  bridge docs, consumer manifest (the ownership documents this page audits against).
- [architecture.md — UI server-state integration audit](architecture.md#ui-server-state-integration-audit) — per-path dispositions and area owners; P4 must add the DriftPanel row.
- [config.md](config.md) — the env keys the settings form edits, with defaults and restart-required flags.
