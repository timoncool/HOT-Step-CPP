# STORM

A live streaming performance mode. Instead of queuing one song and waiting for it, STORM opens a continuous audio stream and renders it in back-to-back "slots" that crossfade into each other, while you retune the style, lyrics, seed and sampler settings for the next slot without stopping playback. It also has a one-off Sequential mode that queues a single song the normal way, and a DJ mode that runs two independent streams on a crossfader.

![STORM](../../images/studio-storm.webp)
<!-- screenshot: needed -->

## Where to find it

The "STORM" entry in the sidebar (`/storm`). The engine must be ready first.

STORM runs on ACE-Step 1.5 and YuE2, and the sidebar lists it only while one of them is active. MiniMax-Music3 does not support STORM. The page and server also have Stable Audio 3 slot and drift code, described below, but the Stable Audio 3 backend reports no streaming support, so with it active the sidebar hides STORM and opening `/storm` moves the app to Custom-Gen.

<!-- TODO(verify): the Stable Audio 3 capability manifest reports streaming: false (server/src/services/backends/sa3/index.ts), so with SA3 active the sidebar hides STORM, the app moves an open /storm to Custom-Gen, and the page shows "Not supported by the active backend". The STORM page and server have SA3 slot and drift code, so the flag looks wrong; confirm before documenting SA3 access. -->

Starting a stream while any job is active in the generation queue is refused ("N generation job(s) active — wait for the queue to drain before streaming"). Once a stream is running, other generations can still be queued: each slot takes the shared GPU lane in turn, so queued work runs between slots.

## Workflow

1. Pick a mode tab: **Sequential**, **Continuous**, **DJ**, or **Drift** for automatic changes toward caption and parameter targets.
2. In the Compose column, write a **Style** caption and, unless **Instrumental** is on, **Lyrics**. Optionally add a LoRA trigger word or a percussive beat intro/outro for DJ mixing.
3. In the Sampler column, tune the extra params for whichever solver you're using, and optionally pick a specific Solver, Scheduler or Guider for the stream (leave them on "inherit global" to use the ones set in the global bar). Guidance, steps, duration and BPM sit in the Live Controls column on the right.
4. Click **Generate** (Sequential) or **Start** (Continuous). Sequential queues one song through the normal job queue, the same as any other studio, using the Live Controls' guidance, steps, duration and BPM at the moment you click; nothing STORM does afterwards changes it. Continuous opens the audio stream and starts playing slot 1 as soon as it renders.
5. While a Continuous stream plays, use the Live Controls column on the right to change the style, lyrics, seed, key, BPM or per-slot sliders for the *next* slot, then click **Send** (or drag a slider, which sends after you stop moving it). Changes land on the slot after the one currently rendering.
6. Click **REC** to start capturing the master output to a downloadable `.webm` file; click it again to stop and download. Recording only happens in the browser, nothing is saved to the library automatically.
7. Click **Stop** (or navigate away from the page) to end playback. No next slot starts. An in-flight ACE-Step or YuE2 slot is cancelled; Stable Audio 3 finishes the current slot, then stops. Output is discarded either way, and the GPU lane stays held until the engine finishes.

For DJ mode: start Deck A and Deck B independently from their own panels, each with its own caption, lyrics and Start/Stop. The center column shows each deck's detected key on the Camelot wheel with a compatibility read-out, a crossfader, cut buttons, per-deck nudge buttons, and Quantize A→B to line up Deck B's next slot to Deck A's next beat.

## Controls

### Mode tabs

| Control | What it does |
|---|---|
| Sequential | Builds one song from the Compose/Sampler panels and sends it to the normal generation queue. No live stream, no crossfading. |
| Continuous | Opens a persistent audio stream: slots render and crossfade back-to-back until you stop it. Live Controls become active. |
| DJ | Two independent Continuous streams (Deck A, Deck B) mixed through a crossfader, with key-compatibility and beat-sync helpers. |
| Drift | Continuous streaming with Directed Drift. Caption weights, parameter values and seeds evolve between slots under your control. |

### Compose column

| Control | What it does |
|---|---|
| Style | The style caption for the stream. Supports `{A|B|C}` wildcard syntax, expandable in place with the wildcard button; with **auto** on, wildcards resolve when the stream starts. Lyrics and Negative Prompt take wildcards the same way. |
| LoRA | A trigger word prepended to the style caption. |
| Beat I/O | Appends a note asking for a clean 1/2/4/8-bar percussive intro and outro, for cleaner DJ transitions. |
| Instrumental | Skips lyrics and forces an instrumental generation. |
| Lyrics | Lyrics for the stream. The loop/cycle/shuffle buttons control how sections advance automatically as slots play: loop repeats the same lyrics, cycle steps through sections in order, shuffle picks a random unseen section each slot. |
| Negative Prompt | Free-text negative prompt, sent with every slot. |

### Sampler column

| Control | What it does |
|---|---|
| Solver extra params | Sliders specific to the selected solver. While Solver is STORM or "— inherit global —", the STORM solver's Detail Sens, Coherence, Early Focus and Cache Depth show; they only affect the STORM solver. Other solvers show their own sliders, or a note that they have none. |
| Precision | RK solver order (auto, 2 to 5). Applies from the next Start, not live. |
| Cache Ratio / CFG Cutoff | Skip redundant DiT steps / skip the unconditional guidance pass after a ratio of steps, to speed up rendering. Applies from the next Start. |
| LSS | Latent self-similarity strength, live-adjustable while streaming. |
| Solver / Scheduler / Guider | Overrides the stream's solver, noise scheduler or guidance mode. "— inherit global —" uses whatever is set in the global bar. Changing these while a stream plays sends the change live. |
| Scheduler params / Guider params | Extra sliders and dropdowns specific to the chosen Scheduler or Guider, shown only when one is picked that has extra params. |
| XFade | Crossfade length between slots, in beats. |
| MaxBuf | How far ahead of playback the client is allowed to buffer, in minutes or in slot multiples (toggle with the clock/note icon). The engine pauses rendering once the buffer fills and resumes once playback catches up. |

A reset arrow appears beside Precision, Cache Ratio, CFG Cutoff, LSS, XFade and any
solver/scheduler/guider param whenever the value has moved from the default; click
it to snap that one field back.

Everything above XFade is for ACE-Step. On YuE2 the column is headed "YuE2 NAR" and
holds that backend's ODE steps, NAR cache ratio, NAR solver and NAR scheduler, plus the
extra sliders of the chosen NAR solver or scheduler. On Stable Audio 3 only XFade and
MaxBuf appear.

### Live Controls (Continuous, right column)

| Control | What it does |
|---|---|
| Guidance / Steps / Duration | Sliders for the next slot's CFG scale, inference steps and length. Click the value to type a number directly. Steps is hidden on YuE2 and Stable Audio 3. |
| Seed | The base seed. Each slot advances the seed by one unless Seed Lock is on, in which case every slot reuses the same seed. Includes randomize, a save/load drawer, and the lock toggle. |
| Key | ACE-Step only. Auto shows the key detected client-side from the audio. Manual lets you type a key, but this does not currently change generation, it's a display-only field the server ignores. |
| BPM | ACE-Step only. Auto shows the BPM detected client-side from the audio. Manual, the starting mode, sends a fixed BPM for the next slot from a 60 to 200 slider. |
| AI ↻ | Enables automatic AI lyric continuation every 1, 2 or 4 slots, using the external LLM provider and model last picked in Custom-Gen's "Generate with AI" dialog. The direction field is a persistent hint; the gear icon opens a preset manager and the prompt template used for continuation. |
| Style / Lyrics tabs | Fields for the next slot's style or lyrics. The pin icon next to each makes it "stick" (stays applied to every following slot) instead of clearing after Send. The Style tab has its own LoRA trigger word and Beat I/O for the next slot. |
| Send | Queues the typed style/lyrics for the slot after the one now rendering; the button briefly shows that slot's number. |
| Mini player | Continuous only, while playing: the slot number, detected key and BPM, played and buffered time, a volume slider and **REC**. |
| Record (REC) | Captures the mixed output to a `.webm` file, downloaded when you stop recording. |

Timeline boxes above the panel show each slot's state (queued, rendering, rendered, playing, past). Clicking a box shows that slot's stamped metadata (seed, detected key/BPM, guidance, steps, duration, solver, scheduler, guidance mode, style, lyrics) once it has played; clicking a box never seeks playback.

### Directed Drift

Choose Drift, set the starting caption and sampler settings, then Start. Adjust the
Drift panel while listening to steer later slots. Pressure settings persist in this
browser and apply again when you start a Drift stream.

<!-- TODO(verify): the Drift tab shows only the Drift panel. The header button (Generate/Start) renders for Sequential and Continuous only, although StormPage.handleStart has a drift branch and the panel says "Start the stream above", so there is no Start on this tab. Drift changes do reach a stream started from the Continuous tab, live or as a pending pressure set while idle. -->

| Control | What it does |
|---|---|
| Target Caption | Comma-separated tags describing where the stream should head. Commit with Enter or by leaving the field. Live tag bars show each tag's weight; tags above 50% enter the caption. |
| Pull | Attraction toward the target caption and parameter values, from 0 to 1. Zero removes attraction; higher values pull harder. |
| Rate | How far drift advances per slot, from 0 to 1. Zero stops drift movement. |
| Jitter | Random variation in the path, from 0 to 1. Higher values make it less predictable. |
| Param targets | Sliders set destinations within each parameter's range. The lock toggle excludes that parameter from drift so it uses the stream setting. Live readouts show current and target values. |
| Pull now | Sets Pull to 1 for one slot, then restores the previous value. Per-slot movement limits still apply. |
| Anchor | Copies the live parameter positions and active caption tags into the targets. Available after a live snapshot arrives. |
| Energy boost | Raises guidance and, where supported, BPM targets and adds energetic caption tags. |
| Mood / Genre | Mood buttons replace the target caption with a preset; Genre applies your comma-separated genre tags. |
| Custom tag sets | Save a named group of tags locally, apply it as a target caption, or remove it. |
| Seed evolution | Lock leaves the stream seed unchanged. Walk advances it each slot. Jump occasionally chooses a random seed, with higher jitter increasing the chance. |
| Drift plugin choice | Allows occasional solver, scheduler or guidance changes when candidate plugins are available. The chance per slot is `0.1 × jitter × rate`. |

ACE-Step targets include guidance, LSS strength, steps, BPM and duration. YuE2
targets are CFG scale, ODE steps and NAR cache ratio. Stable Audio 3 targets are
guidance, steps and duration. Backend-specific targets do not add capabilities
that the selected model lacks.

## Tips and limits

Nothing STORM renders is written to SQLite or the library. Continuous and DJ streams exist only as audio in the browser tab; the only way to keep output is the **REC** button, which Continuous mode has and the DJ decks do not. Sequential mode is the exception, since it goes through the normal generation queue and lands in the library like any other song.

Leaving the STORM page stops whatever is streaming.

AI Continuation needs an LLM provider already picked in Custom-Gen's "Generate with AI" dialog; with none picked it shows a warning instead of generating, and disables itself after three consecutive failures.

The STORM solver plugin and this streaming UI are a contribution from MDMAchine (Alexander Allan), shipped under GPLv3 alongside the rest of the plugin stack.

## Related

- [Create (Custom-Gen)](create.md)
- [Library](library.md)
- [Generation](../generation.md)
- [Backends](../backends.md)
