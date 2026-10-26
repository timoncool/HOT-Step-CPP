# YuE2 Plan Play presets

Four profile presets for the **YuE2** backend: Loose, Sections, Subtle and Wander. They set
the YuE2 NAR solver, CADS, plan-strength and plan CFG knobs (`backendParams`) plus the
matching YuE2 plugin values (`pluginParams`). They only do something useful with YuE2
active. Profiles are kept per engine, so a preset imported while another engine is active
lands in that engine's list, and its YuE2 knobs are stored there but unused.

| File | What it sets |
|---|---|
| `plan-play-subtle.json` | Plan Strength (start) 0.9 and a touch of CADS (0.01), on the stock Midpoint solver and Uniform scheduler |
| `plan-play-loose.json` | Plan Strength (start) 0.8, CADS 0.05, and the Ancestral Euler solver with churn 0.25 |
| `plan-play-sections.json` | The Plan-Aware scheduler with the Wasserstein Flow solver |
| `plan-play-wander.json` | The Plan-Loose scheduler (loose 0.35), CADS 0.1, and Ancestral Euler with churn 0.5 |

## Import

1. Switch the active backend to YuE2.
2. Open Profiles at the right of the bar and click the import icon in the header.
3. Pick one of the `plan-play-*.json` files. In a downloaded release they are in
   `presets/yue2-plan-play/` inside the HOT-Step folder; in a source checkout, this folder.
   The profile takes the file name.
4. Apply it. YuE2 knobs the preset names are merged over your current ones, so model paths
   and knobs it does not list are kept.
