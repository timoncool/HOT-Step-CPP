# Library, playlist, drafts and Song Builder for frontend clients

This page covers the library's songs, the playlist, saved studio drafts and Song Builder projects. The shared types are in [`contracts/songs.ts`](../../server/src/contracts/songs.ts), [`contracts/studioDrafts.ts`](../../server/src/contracts/studioDrafts.ts), [`contracts/studioDraftFields.ts`](../../server/src/contracts/studioDraftFields.ts) and [`contracts/songBuilder.ts`](../../server/src/contracts/songBuilder.ts). Every path is listed in the [route index](api.md); the asset-based import and the export resolver are in [frontend-media.md](frontend-media.md).

Routes marked "token" need `Authorization: Bearer <token>` from `GET /api/auth/auto` and answer `401 { error: 'Unauthorized' }` without it.

## Songs

A song on the wire is its database row ([`Song`](../../server/src/contracts/songs.ts), fields in [`SONG_FIELDS`](../../server/src/contracts/songs.ts)) with two columns decoded: `tags` is an array and `is_public` a boolean. Everything else is sent as stored:

- `generation_params` and `metadata_overrides` stay JSON strings;
- `created_at` is SQLite UTC text (`YYYY-MM-DD HH:MM:SS`);
- empty media fields are `''`.

Display rules (camelCase, fallback titles, artist names) are each client's own.

Media identity: a song is its `id`. Its audio is the relative URL in `audio_url`, with variants in `mastered_audio_url` and `noadapter_audio_url`, and `latent_url` and the stem and disco URLs alongside. Prefix the server origin to fetch any of them.

| Call | Returns |
|---|---|
| `GET /api/songs[?source=]` (token) | `{ songs }`, the user's songs, newest first (`created_at` descending), not paginated. `source` matches `generation_params.source`, for example `create`, `lyric-studio` or `cover-studio`. |
| `GET /api/songs/ids` (token) | `{ ids }`, to prune local references to deleted songs. |
| `GET /api/songs/recent[?source=&limit=50]` (token) | `{ songs }` in the normalised [`RecentSong`](../../server/src/contracts/songs.ts) shape: newest first, at most `limit`. `source=all` or none means every source. Adds `source`, `artist_name`, `artist_image`, `album` and `generation_id` from Lyric Studio where known. |
| `GET /api/songs/:id` | `{ song }`; 404 when not found. This route does not check the token. |
| `POST /api/songs` (token) | `{ song }`. Creates a row from the given fields (`id` optional). |
| `PATCH /api/songs/:id` (token) | `{ song }`. Writes only [`SONG_EDITABLE_FIELDS`](../../server/src/contracts/songs.ts), plus `tags` (an array) and `metadata_overrides` (an object; `null` or `''` clears it). Any other field is ignored. 404 for a song that is not the user's. There is no revision check: the last write wins. |
| `DELETE /api/songs/:id` (token) | `{ success: true }`; also deletes the song's files. 404 when not found. |
| `DELETE /api/songs` (token) | `{ success: true, deletedCount }`: deletes every one of the user's songs and their audio files. |
| `POST /api/songs/bulk-delete` (token) | Takes `{ ids }` and returns `{ success: true, deletedCount }`. An empty `ids` answers 400. |
| `POST /api/songs/import` (token) | Multipart `audio` (up to 50 files) and an optional `description`. Returns `{ songs, errors, accepted }`: 200 when at least one file imported, 400 with the same `errors` when none did. |
| `POST /api/songs/export-zip` (token) | Takes `{ ids }` (1 to 500 song ids) and streams one zip (`hotstep-library-<time>.zip`) of the user's selected songs' `audio_url` files, in the order of `ids`. Each entry is named after the song's title, with ` (2)` and so on for repeats. 400 for a bad `ids`, 404 when none of the songs has an audio file, 413 when the files total more than 256 MiB. A file that fails while streaming ends the download early. |

Other song actions:

| Call | Effect |
|---|---|
| `POST /api/songs/:id/crop` (token) | `{ inPoint, outPoint, audioUrl? }` in seconds (swapped when reversed) crops the audio, its mastered copy and its LRC in place, and answers `{ cropped: true, newDuration }`. The song is found by id, then by `audioUrl`. 400 without both points, 404 for an unknown song or a missing audio file. |
| `POST /api/songs/:id/postprocess` | Runs the post-processing chain on a raw WAV render with the settings in the body (the master switch is forced on), and answers `{ jobId, status, stage, stages }`. 400 when no stage is on or the song is not a WAV; 409 when it was already post-processed (with `alreadyProcessed` and `masteredAudioUrl`) or a pass is running. |
| `GET /api/songs/:id/postprocess[?jobId=]` | That pass, or the song's current one: `{ jobId, songId, status, stage, error, masteredAudioUrl }`; 404 when there is none. |
| `DELETE /api/songs/:id/postprocess` | Deletes the post-processed copy and answers `{ removed, songId }`; the raw render stays. 409 while a pass is running. |
| `POST /api/songs/:id/retranscribe` | `{ model?, language?, beamSize? }` re-runs Whisper against the song's source lyrics, writes its `.lyrics.json`, and answers `{ success, lineCount, wordCount }`. |
| `POST /api/songs/:id/extract-kick`, `POST /api/songs/:id/analyze-disco` | Drum stems and the beat data behind `disco_data_url`. Extraction answers `{ status: 'started' \| 'in-progress' \| 'exists', ... }` and runs in the background; analysis of existing stems answers `{ status: 'created' \| 'exists', discoDataUrl }`. |
| `POST /api/songs/nuke-generations` (token) | Deletes all of the user's songs, every Lyric Studio audio link and every file in the audio folder. Answers `{ success, songsDeleted, filesDeleted, lireekAudioGensDeleted }`. |

## Playlist

The playlist is one revisioned document per user, under `/api/studio-drafts`, with a token on every route. An item needs a string `id`, `title` and `audioUrl`; the other fields of [`playlistItemSchema`](../../server/src/contracts/studioDrafts.ts) are optional, and items keep any extra fields a client stores (the schema passes them through). A playlist holds at most 2000 items. An item `id` may point at a song that no longer exists, or at a queue-only render: the playlist never checks.

- `GET /api/studio-drafts/playlist` returns `{ document }`. It is `null` before the first change, which means revision 0.
- `POST /api/studio-drafts/playlist/commands` takes `{ expectedRevision, command }` and returns `{ document }` at the next revision. The command is one of ([`playlistCommandSchema`](../../server/src/contracts/studioDrafts.ts)):

| `operation` | Fields | Effect |
|---|---|---|
| `add` | `item` | Appends it. An id already present changes nothing, but the revision still moves. |
| `remove` | `id` | Removes it. An unknown id is ignored. |
| `clear` | none | Empties the playlist. |
| `reorder` | `ids` | The new order. It must name every current item exactly once, or the command answers `400 { error: 'Reorder must name every current item exactly once' }`. |
| `update` | `id`, `patch` | Merges `patch` into the item, keeping its `id`. An unknown id is ignored. |

A stale `expectedRevision` answers `409 { error, currentRevision }` and changes nothing. To retry, the client reads the playlist again, reapplies its edit to the current items, and sends it with the new revision. Nothing is retried automatically. A malformed command answers 400.

## Studio drafts

A draft saves a studio's form so another client, or the same one later, can pick it up. Drafts cover the studios `create`, `cover`, `repaint`, `storm`, `stem-studio` and `stem-builder`. Draft documents are at schema version 1.

A draft body is `{ studio, fields, backendId?, sourceAssetId?, sourceSongId?, sourceRevision? }`, plus any extra top-level fields a client keeps. These are stored as given.

**Fields.** `fields` keeps the browser storage names the bundled UI uses, so every client reads the same draft. [`STUDIO_DRAFT_FIELDS`](../../server/src/contracts/studioDraftFields.ts) lists every key by studio, with the value type the server accepts: `string`, `boolean`, finite `number`, `number|null`, `string|null`, `object|null` or `array`. Cover includes the YuE2 score, the Prepare document (`cover-studio-prepareV2`), the Build page's mode, session and lyrics draft, render settings (Follow source and the lyric schedule among them), adapter and artist selections, caption controls, and optional `draftName`. Create also accepts the caption-source keys matching `CREATE_CAPTION_SOURCE_KEY`.

A save answers 400 when:

- a key belongs to another studio or to none;
- a value has the wrong type, or `cover-studio-prepareV2` is not a valid Prepare document ([`contracts/yue2Prepare.ts`](../../server/src/contracts/yue2Prepare.ts));
- `sourceAssetId` or `sourceSongId` disagrees with the matching field;
- the body is larger than 1 MiB.

The server never interprets a value. Selection pointers, playback state and device handles are never draft fields.

| Call | Returns |
|---|---|
| `GET /api/studio-drafts/drafts[?studio=]` | `{ documents }` |
| `POST /api/studio-drafts/drafts` with `{ body }` | `201 { document }` at revision 1. Sending `expectedRevision` here answers 400. |
| `GET /api/studio-drafts/drafts/:id` | `{ document, sourceError }`. `sourceError` names a source asset that no longer exists; the draft still loads, and the user reuploads the source. An unknown id answers 404 (`This saved studio draft no longer exists…`), here and on `PUT` and `DELETE`. |
| `PUT /api/studio-drafts/drafts/:id` with `{ body, expectedRevision }` | `{ document }` at the next revision. A missing `expectedRevision` or a different `studio` answers 400; a stale one answers `409 { error, currentRevision }`. |
| `DELETE /api/studio-drafts/drafts/:id?expectedRevision=N` | `{ removed: true }`. A missing revision answers 400; a stale one answers 409. |

**Source staleness.** A draft's source is its `sourceAssetId`, `sourceSongId` and `sourceRevision`, plus the source fields in [`DRAFT_SOURCE_FIELDS`](../../server/src/contracts/studioDraftFields.ts). When a save changes any of them, the server drops the results computed from the old source, listed in `DRAFT_SOURCE_RESULT_FIELDS`:

- Cover's analysis, metadata, artist caption and lyrics source;
- Create's caption sources.

The user's own text and settings are kept.

**Importing browser storage.** `POST /api/studio-drafts/import` takes `{ storageKey, raw, sourceHash: 'sha256:<hex of raw>', schemaVersion: 1, expectedRevision, resolution? }` and moves one browser storage value into a draft or the playlist. It answers `{ receipt, document, created }`.

- Repeating the same `storageKey` and hash returns the first import.
- Different content under an imported key answers 409 until `resolution` is `keep-both` or `replace`.
- A new draft import needs `expectedRevision: 0`.
- The playlist key `lireek-playQueue` needs `replace` when a playlist already exists.

**Handoffs between studios.** `GET /api/studio-drafts/handoffs/:id` reads a Cover draft or a training audition sent to Create, by the id the sending studio returned. The original revisioned document stays the workflow's input. Any other document id answers 404.

## Song Builder

A project is one song built from an ordered chain of sections ([`contracts/songBuilder.ts`](../../server/src/contracts/songBuilder.ts)). Each section renders `variant_count` candidate songs. Choosing one makes it the source that the next section extends or leads into. All routes are under `/api/builder` and need a token.

**Projects.**

- `POST /projects` and `PATCH /projects/:id` take [`BuilderProjectFields`](../../server/src/contracts/songBuilder.ts) in camelCase and answer the [project view](../../server/src/contracts/songBuilder.ts) `{ project, sections }`. Project fields are stored snake_case.
- A project edit takes no revision: the last write wins per field. It still moves `project.revision`, so section commands based on the old settings are refused.
- `GET /projects` returns `{ projects }`, newest-updated first, each with `section_count`.
- `GET /projects/:id` returns the view.
- `DELETE /projects/:id` returns `{ ok: true }`. Candidate songs stay in the library.

**Sections.** The view lists sections by `position`, then creation time. Each carries its candidate ids, its resolved `candidates` (songs deleted since are dropped from the list) and its `chosen` song.

`status` is:

- `generating` while its job runs;
- then `ready` (some candidates) or `failed` (none);
- `chosen` once a candidate is picked.

A section whose job ended elsewhere (cancelled, or interrupted by a server restart) is settled on the next read. Nothing reruns by itself.

**Commands.** Every section command carries `expectedRevision`, the project revision the client last saw. A stale one answers `409 { error, currentRevision }`; reload the project and decide again. Each accepted command moves the revision by one and answers the whole project view.

| Command | Body | Notes |
|---|---|---|
| `POST /projects/:id/sections/generate` | [`generateSectionSchema`](../../server/src/contracts/songBuilder.ts) | Adds `{ jobId, sectionId }` to the view. Details below. |
| `POST /sections/:id/choose` | `{ songId, expectedRevision }` | `songId` must be one of the section's candidates (409 otherwise). Stops its remaining renders, and fills the project's BPM and key from the chosen song when they are unset. |
| `POST /sections/:id/stop` | `{ expectedRevision }` | Keeps the candidates that landed and stops the rest. |
| `PATCH /sections/:id` | `{ label?, lyrics?, expectedRevision }` | Non-string values answer 400. |
| `DELETE /sections/:id?expectedRevision=N` | none | Also stops its renders. Any later command on it answers 404. |

**Generating a section.** `direction` is `first`, `append` or `prepend`:

- `length` is `{ bars }`, which needs the project BPM, or `{ seconds }`.
- `overlap` is the seconds of existing audio the new section overwrites at the seam.
- `clipPoint` is where an append extends from, or where a prepend connects. `null` means the end, or the start.

Node works out the geometry from the stored project and its chosen sections, so the client sends no positions.

`idempotencyKey` makes a submit safe to repeat:

- The same key and body return the original job and section, even after the revision has moved on.
- The same key with a different body answers 409.

`expectedBackend` must still be the active engine, or the command answers 409 before anything is created. Follow the job with the workflow job routes, or by reading the project; candidates appear as they land.

Errors from Song Builder are `{ error, currentRevision?, issues? }`. A malformed generate body answers 400 with `issues`.
