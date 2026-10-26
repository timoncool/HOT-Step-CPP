# Compare

Side-by-side listening for takes you already generated. Put two or more songs in a session, play them against one shared playhead, flip between them instantly, and score each one. Use it for seed sweeps, plugin A/B tests and checkpoint ladders, anywhere you need to hear small differences fairly.

![Compare](../../images/studio-compare-studio.webp)
<!-- screenshot: needed -->

## Where to find it

The "Compare" entry in the sidebar (`/compare`). From the Library you can also select two or more songs and press **Compare** (the button shows the count, for example Compare 3), which opens the New comparison panel with those songs already picked.

## Workflow

1. Press **New comparison**, search the library by title or style (results come 25 at a time, newest first) and tap songs to pick them. The first pick is the baseline.
2. Give it a name and, optionally, a group. A blank name uses the date and time. Sessions with the same group are listed together.
3. Press **Compare N songs**. The session opens and each take is analysed in the background.
4. Press play (or Space). Tap a take, or press 1 to 9, to switch which one you hear; the playhead does not move. The A/B button (or B / Tab) flips between the takes in slot A and slot B. Use Set A and Set B on a take to choose the pair.
5. Drag across any waveform to loop a region, or use **In** and **Out** while playing. **Clear loop** removes it. A short tap on a waveform switches to that take and moves the playhead to the tapped point.
6. Turn on **Level match** so every take plays at the loudness of the quietest one.
7. Score each take: Character and Quality from 1 to 5, a Keep switch and a Note. Any one of these marks the take scored, and a session archives itself once every take is scored.

## Controls

| Control | What it does |
|---|---|
| Show archived | Lists fully scored sessions too. |
| Unscored chips | Quick links to sessions that still need scores. The dot is green for scored, amber for partial, grey for unscored. |
| Archive / Unarchive | Icon in a session's header. Moves the session out of the default list, or back into it. |
| Delete | Icon in a session's header. Asks first, then deletes the session and its scores. |
| Seek bar | Shared playhead for all takes, with the loop region shaded. |
| A/B button | Flips between the take in slot A and the take in slot B, at the same playhead. The button shows which takes are in the slots (for example A: 1 / B: 2). A and B are always two different takes; at the start A is the baseline (or the first take) and B is the next take. Keys: B or Tab. Hidden when the session has only one take. |
| Set A / Set B | Buttons on each take row that put that take in slot A or slot B. The active slot is highlighted. Setting a take that already sits in the other slot swaps the two. Tapping a take that is in neither slot plays it and puts it in the slot you are not currently hearing, so the A/B button flips between the take you were on and the one you tapped. If a take in a slot is removed, the slot picks another take. |
| Level match | Off at the start. Gains every take down to the quietest integrated loudness, so loudness alone does not win. Each take then shows the gain applied. |
| Take details | Under each title: baseline marker, backend, seed, and for a seed blend its second seed and blend amount. |
| Remove take (bin icon) | Asks first, then takes that song and its scores out of the session. |
| Metric chips | Integrated LUFS, short-term max, true peak (red above 0 dBTP), RMS, crest, spectral centroid, flatness, stereo correlation. |
| Difference trace | Active take against the baseline over time: loudness difference in LU and log-spectral distance in dB. |

## Tips and limits

- Score with level match on. A louder take nearly always sounds better at first.
- Analysis runs the first time a song is opened here and is cached after that; it runs again if the audio file changes.
- Removing a take or deleting a session never deletes the song from the library.
- The app's global shortcuts are off on this page, which has its own keys: Space, 1 to 9, B and Tab.
- Pressing play here pauses the main player bar, so only one player is heard.

## Related

- [Library](library.md)
- [Generation](../generation.md)
- [Getting higher quality output](../quality.md)
