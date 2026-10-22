# YuE2 Plan Play presets

Four profile presets for the **YuE2** backend: Loose, Sections, Subtle and Wander. They set
the YuE2 NAR solver, CADS, plan-strength and plan CFG knobs (`backendParams`) plus the
matching YuE2 plugin values (`pluginParams`). They only do something useful with YuE2
active; on ACE-Step 1.5 or SA3 the YuE2 knobs are stored but unused.

## Import

1. Switch the active backend to YuE2.
2. Open Profiles at the right of the bar and click the import icon in the header.
3. Pick one of the `plan-play-*.json` files. In a downloaded release they are in
   `presets/yue2-plan-play/` inside the HOT-Step folder; in a source checkout, this folder.
   The profile takes the file name.
4. Apply it. YuE2 knobs the preset names are merged over your current ones, so model paths
   and knobs it does not list are kept.
