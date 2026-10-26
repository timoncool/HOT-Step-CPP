# ui-feature-dev — reference

Deep-dive companion to [SKILL.md](SKILL.md). All paths/line refs verified against `master` at `dda5f6b6` on 2026-10-23.

## 1. Worked example: Stem Studio end-to-end

Files: `ui/src/components/stem-studio/{StemStudio,SourceSelector,TrackSelector,RecentExtractions}.tsx`, `ui/src/services/stemStudioApi.ts`, `server/src/routes/stemStudio.ts` (mounted `app.use('/api/stem-studio', stemStudioRoutes)` at `server/src/index.ts:123`).

### 1.1 Registration (the six edits)

- Import: `App.tsx:54`, `import { StemStudio } from './components/stem-studio/StemStudio';`
- URL → view: `App.tsx:107`, `if (path.startsWith('/stem-studio')) return 'stem-studio';`
- View → URL: `App.tsx:132`, `if (view === 'stem-studio') return '/stem-studio';`
- Render: `App.tsx:946-952`, `<DiscoPulseWrapper hue={DISCO.assistant} className="flex-1 overflow-hidden"><StemStudio /></DiscoPulseWrapper>`
- Nav: `Sidebar.tsx:144-150`, a `NavItem` inside `SidebarNav` with `icon={<Scissors size={20} />}`, `label={t('sidebar.stemStudio')}`, `active={activeView === 'stem-studio'}`, `onClick={() => onNav('stem-studio')}`, `isExpanded={isExpanded}`. Stem Studio works on every engine, so it has no `STUDIO_NEEDS` entry and no `available(...)` guard.
- Palette: `CommandPalette.tsx:22`, `{ view: 'stem-studio', labelKey: 'sidebar.stemStudio' }` in `PALETTE_VIEWS`.

### 1.2 Orchestrator state (StemStudio.tsx:44-122)

All local `useState`; no dedicated store. Persistence is piecemeal:

- `hs-stem-sourceUrl` / `hs-stem-sourceFile`: lazy `useState(() => localStorage.getItem(...))` init + setter wrappers that write-through (lines 64-67).
- `hs-stem-sepLevel`: read in a lazy initialiser (line 75) and persisted in an effect (lines 119-122); `hs-stem-extractModel` is restored when the model list arrives and persisted in an effect (lines 105-117).
- Cross-studio state touched: shared queue count via `useAudioGenQueueSelector(s => s.items.filter(i => i.status === 'pending' || i.status === 'loading-adapter' || i.status === 'generating').length)` (lines 101-103) and shared sidebar width via `usePersistedState('hs-activitySidebarWidth', 320)` (line 100).

### 1.3 Server data (lines 84-112)

```tsx
const modelsQuery = useQuery(modelsOptions());            // queries/studios/shared.ts re-exports aceModelsOptions
const baseModels = getBaseModels(modelsQuery.data?.models.dit || []);   // name-prefix filter, lines 36-42

useEffect(() => {
  const base = getBaseModels(modelsQuery.data?.models.dit || []);
  if (base.length && !extractModel) {
    const stored = localStorage.getItem('hs-stem-extractModel');
    setExtractModel(stored && base.includes(stored) ? stored : base[0]);
  }
}, [modelsQuery.data, extractModel]);
```

The model list is a Query snapshot shared with every other reader of the same key; the effect only seeds the local selection from it. Error handling convention: a failed read renders an inline `role="alert"` line from `modelsQuery.error` (line 285), and an empty list shows a warning banner (`styles.noModelsWarning`); no toast spam, no crash.

### 1.4 Job submit + wait (`handleExtract`, lines 168-221)

1. Guard clauses (`!sourceAudioUrl`; the `supersep` mode branches to `handleSupersep`; empty tracks or no model) → early return.
2. `setIsExtracting(true)` → `submitExtraction({ sourceAudioUrl, sourceFileName, tracks, style?, lyrics?, ditSettings })` returns `jobId` (POST `/api/stem-studio/extract` through `apiClient`, `stemStudioApi.ts:115-118`; server responds `{ id }`).
3. `waitForExtraction(jobId, setExtractProgress)` (`stemStudioApi.ts:170-187`) observes `serverKey('studios/extract-progress', { id })` through `waitForSnapshot` (`queries/studios/wait.ts`) at `ACTIVE_POLL_MS` (1.5 s) until `status ∈ {done, failed, cancelled}`; on `done` it reads `GET /:id/result` with `queryClient.query()` → `{ id, stems: [{ trackName, audioUrl, durationSec, index, sizeBytes? }] }` and invalidates the `studios/extractions` list.
4. `loadResultIntoMixer` (line 223 onward) maps stems into shared `StemMixer` props.
5. **Failures are folded back into the same `ExtractProgress` shape** with `status: 'failed'` + `error` (lines 207-216) so one render path handles progress AND errors.
6. `finally { setIsExtracting(false); }`.

Progress shape (`stemStudioApi.ts:30-39`):

```ts
interface ExtractProgress {
  status: 'pending' | 'extracting' | 'separating' | 'saving' | 'done' | 'failed' | 'cancelled';
  progress: number;          // 0-100
  currentTrack: string;
  completedStems: string[];
  totalTracks: number;
  warning?: string; error?: string; sepMessage?: string;
}
```

### 1.5 Layout skeleton (lines 284-470)

```
div.flex.flex-col.w-full.h-full.overflow-hidden             ← root (284)
  pane switcher: Source / Tracks / Recent                    ← below 1024 px and portrait-wide only (287-303)
  div.flex-1.flex.overflow-hidden.min-h-0                    ← main row (306)
    div (min-[1024px]:w-[300px], border-r, flex-shrink-0)    ← left: source + optional fields
    div.flex-1 (overflow-y-auto, border-r)                   ← center: selectors, progress, mixer
    div.w-1.5.cursor-col-resize (onMouseDown=resize)         ← drag handle (434)
    div (style={{width: sidebarWidth}}, border-l)            ← right: Section×2 (Recent + Queue)
```

Right sidebar reuses `Section` from `shared/ActivitySidebar.tsx` and `InlineAudioQueue` from `lyric-studio/` (lines 445-465), the standard "Recent X + Queue" pattern. Resize handler clamps `Math.min(700, Math.max(240, ...))` (line 268) and sets/clears `document.body.style.cursor/userSelect`.

### 1.6 Downloads (lines 246-260)

Synthesized `<a download>` clicks against direct GET URLs (`getStemUrl`, `getDownloadAllUrl`); no blob juggling.

## 2. Store templates

### 2.1 Hand-rolled external store (copy `abCompareStore.ts`)

```ts
import { useSyncExternalStore, useRef, useCallback } from 'react';

export interface MyState { value: number }
let _state: MyState = { value: 0 };

const _listeners = new Set<() => void>();
function notify() { _listeners.forEach(cb => cb()); }
function setState(updates: Partial<MyState>) { _state = { ..._state, ...updates }; notify(); }
function subscribe(cb: () => void) { _listeners.add(cb); return () => _listeners.delete(cb); }
function getSnapshot() { return _state; }

// Plain action functions — callable from non-React code
export function setValue(v: number) { setState({ value: v }); }

// Hooks
export function useMyStore(): MyState { return useSyncExternalStore(subscribe, getSnapshot); }
export function useMyStoreSelector<T>(selector: (s: MyState) => T): T {
  const selectorRef = useRef(selector); selectorRef.current = selector;
  const selectedRef = useRef<T>(selector(_state));
  const getSelected = useCallback(() => {
    const next = selectorRef.current(_state);
    if (Object.is(selectedRef.current, next)) return selectedRef.current;
    selectedRef.current = next; return next;
  }, []);
  return useSyncExternalStore(subscribe, getSelected);
}
```

This is exactly `abCompareStore.ts:23-101`. The selector hook's `Object.is` guard prevents render loops from fresh-object selectors. Keep server snapshots out of a store like this; they belong in TanStack Query (SKILL.md, "Server state and client state").

### 2.2 Adding a field to globalParamsStore (zustand, per-key localStorage)

Touch points in `ui/src/stores/globalParamsStore.ts`:

1. Default: add the value to `GLOBAL_PARAM_DEFAULTS` (line 44), so reset-to-default controls can read it.
2. State init: `myKnob: readKey("hs-myKnob", GLOBAL_PARAM_DEFAULTS.myKnob),` inside `create<any>()(...)` (line 313 onward).
3. Setter: `setMyKnob: (v: number) => { set({ myKnob: v }); writeKey('hs-myKnob', v); },`
4. **`getGlobalParams()`** (starts line 741): add the field to the returned `Partial<GenerationParams>`, gated `undefined` unless its feature toggle is on (match the existing conditional-gating style). Also add the field to `GenerationParams` in `ui/src/types.ts`.
5. If the value must differ per backend (sampler picks and their params do), add it to `BACKEND_SCOPED_FIELDS` (line 219) instead of step 2; the hydrate, the backend switch and the preset writer read that table.
6. **Server side**: map the field in `server/src/services/generation/translateParams.ts` (entry point `translateParams()` at :47, called from `server/src/services/backends/ace/generate.ts:51`) so it lands in `aceReq`. Mirror the emission in `resolveGenerationParams()` (`server/src/services/generation/defaults.ts`, with its default in `GENERATION_DEFAULTS`) and list the key in `server/src/contracts/generationControls.ts`, or intent-based submissions drop it. **Sideband rule**: ServerFields-only params are not part of the C++ `AceRequest` struct and never survive the `/lm` round trip; the ACE backend rebuilds each LM result from the current `aceReq`, taking only LM-generated fields from the echo (comment at ace/generate.ts:192-198). Add your param to `aceReq` before the LM phase; never extend the LM-echo pick-list into a copy-across whitelist.

Consume with a selector: `const myKnob = useGlobalParamsStore(s => s.myKnob);`. Show it with the shared primitives: `<Slider ... defaultValue={GLOBAL_PARAM_DEFAULTS.myKnob} />` or a `Toggle`/`StyledSelect` with a `ParamLabel` (docs/dev/ui-design.md).

### 2.3 usePersistedState / writePersistedState

`usePersistedState(key, default)` (`hooks/usePersistedState.ts:9-39`) = useState + localStorage + storage-event listener. To update it from OUTSIDE React (store code, another component tree), you MUST use `writePersistedState(key, value)` (lines 43-49); it dispatches a synthetic `StorageEvent` because native ones only fire cross-tab.

## 3. Per-studio API client template (from stemStudioApi.ts)

```ts
// myStudioApi.ts — API client for My Studio
import { queryClient } from '../queries/client';
import { serverKey } from '../queries/keys';
import { ACTIVE_POLL_MS } from '../queries/contract';
import { waitForSnapshot } from '../queries/studios/wait';
import { apiClient, type ReadOptions } from './httpClient';

const P = '/my-studio';   // apiClient already adds /api

export interface MyJobProgress { status: 'pending' | 'running' | 'done' | 'failed' | 'cancelled'; progress: number; error?: string }

// Command: no retry, no signal; errors arrive as ApiError.
export async function submitJob(params: MyParams): Promise<string> {
  const data = await apiClient.post<{ id: string }>(`${P}/start`, params);
  return data.id;
}

// Reads take ReadOptions so Query can pass its AbortSignal.
export const getProgress = (jobId: string, opts?: ReadOptions) =>
  apiClient.get<MyJobProgress>(`${P}/${jobId}/progress`, opts);
export const getResult = (jobId: string, opts?: ReadOptions) =>
  apiClient.get<MyResult>(`${P}/${jobId}/result`, opts);

// Waiter: a Query-observed snapshot, never a hand-rolled setTimeout loop.
export async function waitForJob(jobId: string, onProgress?: (p: MyJobProgress) => void): Promise<MyResult> {
  const p = await waitForSnapshot(serverKey('studios/my-progress', { id: jobId }),
    signal => getProgress(jobId, { signal }),
    p => !['done', 'failed', 'cancelled'].includes(p.status), onProgress, { pollMs: ACTIVE_POLL_MS });
  if (p.status !== 'done') throw new Error(p.error || `Job ${p.status}`);
  return queryClient.query({
    queryKey: serverKey('studios/my-result', { id: jobId }),
    queryFn: ({ signal }) => getResult(jobId, { signal }),
  });
}
```

Conventions: one shared transport (`apiClient`; no raw `fetch` for JSON), the bearer is added by the axios interceptor (no token argument), errors are `ApiError` carrying the server's status and `error` field, the area key prefix belongs to the owning `queries/<area>/` README, and a finished job invalidates the list it appears in (`stemStudioApi.ts` calls `queryManager.keys(queryClient, resourceKey('studios/extractions'))`). Long multi-step work on the server is better modelled as a workflow job (`ui/src/services/workflowApi.ts`, `ui/src/queries/studios/workflows.ts`).

For the main generation pipeline instead, use `generateApi` (`api.ts:181-222`): `submit(params)` → `{ jobId, status }`; `status(jobId, opts?)` → `GenerationJob` (`types.ts:380`, statuses `pending|lm_running|synth_running|saving|succeeded|failed|cancelled`); also `cancel`, `cancelAll`, `queueStatus`, `resetQueue`. Enqueue via `audioGenQueueStore` (`enqueueSimpleGen`, `enqueueAudioGen`) rather than reimplementing the loop.

## 4. DISCO hue table (App.tsx:351-363)

| Key | Hue | Colour | Typical use |
|---|---|---|---|
| `sidebar` | 120 | green | left nav |
| `createPanel` | 330 | hot pink | create/insta-gen/repaint/song-builder panels |
| `songGrid` | 195 | deep sky blue | library grids, lyric studio |
| `activity` | 270 | purple | settings, cover studio, activity sidebar |
| `playlist` | 20 | orange | playlist sidebar, MIDI Studio, Compare Studio |
| `assistant` | 175 | cyan | assistant, stem studio, stem builder |
| `terminal` | 135 | green | terminal panel |
| `player` | 45 | gold | player bar |
| `rightSidebar` | 300 | magenta | right sidebars |
| `storm` | 210 | electric blue | STORM |
| `trainingStudio` | 60 | amber | Training Studio |

`DISCO` is a local const inside the `App` component; new render branches reference it directly. Pick the hue matching your studio's family.

## 5. localStorage / persistence namespaces

| Prefix / key | Owner |
|---|---|
| `hs-*` | global params (per-field, globalParamsStore), UI layout (`hs-activitySidebarWidth`, shared across studios; `hs-sidebar-open`; `hs-showAssistant`), studio-local (`hs-stem-*`), deep links `hs-lastLyricStudioUrl` / `hs-lastTrainingStudioUrl` |
| `hs-<name>@<backendId>` | per-backend copies of the fields in `BACKEND_SCOPED_FIELDS` (ACE keeps the bare key) |
| `hs-studioDraft:<studio>` | per-browser pointer to the studio draft an editor opens with (`services/studioDraftMirror.ts`) |
| `ace-settings` | AppSettings blob (SettingsPanel; read back inside `getGlobalParams()` at globalParamsStore.ts:743) |
| `lireek-audio-gen-queue` | LEGACY queue localStorage key; migration shim only (audioGenQueueStore.ts:251) |
| `lireek-audio-queue-owner-v1`, `server-migration-backup-v1*` | queue ownership and server-migration state (audioGenQueueStore.ts:252-256) |
| IndexedDB `lireek-queue-store` | current queue persistence (store `queue`, key `state`; audioGenQueueStore.ts:248-250) |
| `i18nextLng` | language detector cache (i18n/index.ts:51) |

## 6. Shared components inventory (`ui/src/components/shared/`)

`ABCompareModal`, `Accordion`, `ActivitySidebar` (exports **`Section`**, props `title?`, `icon?`, `count?`, `countColor?`, `defaultOpen?`, `tabs?`/`activeTab?`/`onTabChange?`, controlled `open?`/`onToggle?`, `boxed?`, `children`; verified ActivitySidebar.tsx:35-51), `AudioQueueMigrationBanner`, `BackendCapabilityGate`, `BottomSheet`, `CommandPalette`, `ConfirmDialog`, `CoverArtSubjectSection`, `CoverImage`, `DiscoPulseWrapper`, `EditableSlider`, `ErrorBoundary`, `FileBrowserModal`, `Fold`, `HiHatParticles`, `HotkeyHelpModal`, `HoverFullText`, `LatentImport`, `NoModelsBanner`, `ParamLabel`, `PostProcessActivity`, `PresetSyncNotices`, `RestoredQueueBanner`, `Slider`, `SnareFlashOverlay`, `SongActionsMenu`, `StemMixer`, `StudioDraftPicker`, `StyledSelect`, `Toast` (`ToastType = 'success' | 'error' | 'info' | 'warning'`), `Toggle`, `UnifiedRecentSongs`, `VramIndicator`, plus `noModelsPoll.ts`.

Hooks (`ui/src/hooks/`): `usePersistedState`, `useEventSource` (SSE), `useCapabilities`, `useJobProgress`, `useStreamAudio`, `useTheme`, `usePluginRegistry`, `useGlobalHotkeys`, `useDisguiseMode`, `useAssistantActions`.

## 7. Status ledger (what's current vs legacy)

- **Current**: TanStack Query for server snapshots, axios transport with one global error interceptor, hand-rolled external stores (majority) and zustand stores for params/backend/training/VST/cover stems, `hs-*` per-key persistence with per-backend scoping, IndexedDB queue persistence, pushState URL routing, per-studio API client files over `apiClient`, `getGlobalParams()` as sole UI request assembler.
- **Legacy, do not extend**: `GlobalParamsContext.tsx` (compat wrapper, header says so), `lireek-audio-gen-queue` localStorage persistence (migration shim), `Toggle` in `settings/SettingsPrimitives.tsx` and `ToggleSwitch` in `global-bar/BarSection.tsx` (thin wrappers over `shared/Toggle`).
- **Known discrepancies**: docs say "Zustand" but 9 of 14 stores (counting `lyric-studio/playlistStore`) are hand-rolled; the i18n rule coexists with hard-coded English strings (StemStudio.tsx:300,330,341); `ui/README.md` is the untouched Vite template.
- **Unverified**: full `NavItem` prop surface beyond `icon/label/active/onClick/isExpanded/badge` is all there is today (Sidebar.tsx:417-425); re-check if the component grows.
