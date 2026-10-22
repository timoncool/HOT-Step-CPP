# Data folder and upgrades

Release builds keep your library, downloaded models, adapters, settings and tools in one data folder outside the app. By default this is `HOT-Step-Data` beside the extracted app folder. Your user profile records its location, so a new version uses the same data without downloading the models again.

## What lives there

| Folder | Contents |
|---|---|
| `library/` | Song database, audio, references, stems and caches, VST state, lyrics, training datasets and runs |
| `library/muscriptor/` | MuScriptor tokens and preserved optional-install files |
| `library/models/muscriptor/` | MuScriptor weights, unless overridden |
| `models/`, `adapters/` | Downloaded model weights and adapters |
| `settings/.env` | Saved settings and service credentials |
| `logs/` | Session, generation and training logs |
| `tools/whisper/`, `tools/runtime/` | Downloaded Whisper executable and runtime libraries |
| `plugins/` | Your Lua plugins, overriding shipped plugins with the same name |
| `tmp/`, `cache/` | Reserved working folders |

Engine binaries, the bundled Node runtime, UI, shipped plugins, noise samples and `server/data/` catalogues stay in the app folder. MuScriptor uses the shipped `ace-midi` executable with weights from the data folder. Preserved Python environments may need rebuilding if an optional integration uses them after relocation.

## Upgrade to a new version

1. Stop HOT-Step and back up your data folder, including `settings/.env`.
2. Extract the complete new archive into a separate folder.
3. Start `HOT-Step.bat` on Windows or `./HOT-Step.sh` on Linux and macOS.
4. Check **Settings > Environment > Data folder**. The saved location should be unchanged.

With no saved location, starting the new launcher inside an older install carries that install's data into the sibling folder. A fresh extraction elsewhere cannot find an arbitrary older app folder. Use **Import previous install** in Settings and provide the older app folder. Import needs an empty current library and preserves the older install. Conflicting populated destinations stop startup instead of merging libraries.

## Choose or move a folder

Open the Data folder panel in Settings, choose an action and enter an absolute path:

- **Move current data** transfers managed data into an empty destination.
- **Use existing folder** selects a compatible existing data folder. It copies and deletes nothing.
- **Create empty data folder** starts an empty library while retaining your current folder.
- **Import previous install** copies an older install into the current empty library.

Click **Preview change** to see the folders, size and space required. Finish queued and active work and downloads, then click **Confirm and restart**. Transfers run before the library opens; the launch console shows progress. Same-volume moves rename folders. Across volumes, HOT-Step copies and checks file sizes before removing verified source folders. External path overrides remain outside the transfer.

If restart fails, the change remains scheduled. Use **Restart to apply**, or **Cancel scheduled change** before transfer begins. A process `HOT_STEP_DATA_ROOT` override must be cleared before Settings can change the saved location.

## Development checkouts

A development checkout started with `dev.bat` uses its own folders until you change them in this panel. Nothing moves at startup. **Move current data** carries the checkout's library, models, adapters, logs, Whisper and `.env` into the destination and leaves files tracked by git in place. The checkout's choice is saved in its own `data-location.dev-<hash>.json` in the profile folder below, separate from the release app's `data-location.json`. Settings will not let a checkout and a release share a data folder. Deleting that file points the checkout back at its own folders; data that already moved stays where it is.

## Recover a missing drive or interrupted transfer

The profile pointer is `data-location.json` in:

| Platform | Profile folder |
|---|---|
| Windows | `%LOCALAPPDATA%\HOT-Step` |
| Linux | `$XDG_CONFIG_HOME/hot-step`, or `~/.config/hot-step` |
| macOS | `~/Library/Application Support/HOT-Step` |

Restart state and the transfer journal stay in this profile even while data moves. If the saved data folder is missing or inaccessible, HOT-Step starts with the default `HOT-Step-Data` beside the app. The launch console warns with the saved and fallback paths. This may show an empty library. The saved location stays unchanged, and no data is moved or deleted; reconnect the drive and restart to use it again. A development checkout falls back to its original checkout folders. An explicit `HOT_STEP_DATA_ROOT` override still takes precedence; an invalid override stops startup.

A malformed pointer, failed transfer, or unsuitable fallback folder still shows an explanation page and keeps the saved state. Fix the reported problem and restart. An interrupted transfer resumes. Follow the error page's instructions to abandon an unfinished same-volume move. Keep the pointer and data intact until recovery is complete.

Browser preferences stay in browser localStorage for that address. They do not move with the data folder. Back up the profile pointer separately if you need to restore the saved location on another machine.

## Related

- [Settings](studios/settings.md)
- [Model files](models.md)
- [Configuration for contributors](../dev/config.md)
