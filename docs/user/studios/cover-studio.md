# Cover Studio

Make a new version of a recording. ACE-Step conditions its audio renderer on the
source; YuE2 transcribes the source into a lead sheet and composes from the
reviewed score. Both use your caption and lyrics or an Instrumental choice.

![Cover Studio](../../images/hot-step-cover-studio.webp)

## Where to find it

Sidebar: **Cover Studio**. You need to be signed in to upload a source track, and
the active backend must support covers. ACE and YuE2 offer different cover workflows;
MiniMax-Music3 shows a "not supported" notice. The [backend switcher](../backends.md) appears once a second
backend is registered, so on a single-backend install this is not a decision you
have to make.

## ACE workflow

1. Drop a source track into the upload zone (MP3, WAV, FLAC, OGG, M4A, Opus, AAC),
   or send one over from [Library](library.md) with "Send to Cover Studio". A
   `.latent`/`.hslat` file exported from an earlier HOT-Step generation can be
   imported instead of audio, and pre-fills lyrics, caption, BPM and key from its
   embedded metadata.
2. The upload runs metadata extraction (artist/title/album/duration) and Essentia
   BPM/key analysis automatically. Re-adding a file you already analysed in this
   browser reuses the cached result instead of re-analysing.
   An uploaded source is saved as a cover draft on the server. If you close and
   reopen Cover Studio, the draft keeps its source identity and analysis.
   The studio's fields also save to a server draft about a second after each
   change. This browser's copy is still the one you edit. The draft belongs
   to this browser: another browser or tool that edits the same draft never
   overwrites your fields. If that happens, your next change is saved as a
   new draft and a short note under the header says so. If a save fails, a
   note says the draft wasn't saved; your edits stay in this browser, and the
   next change tries again. To go back to a saved draft, open the **Drafts**
   fold under the Generate button (collapsed by default) and pick one from
   **Load a saved draft**; if the form has edits not yet saved
   to a draft, you are asked first. A draft whose uploaded source has gone
   missing is reported as unavailable, so you can reupload it. Loading a
   draft never starts a job.
   In YuE2 mode, the Drafts fold also has a name box. Enter a name
   and choose **Save named draft** to keep a separate snapshot of the current
   score and settings. **Load a saved draft** restores that snapshot. Leaving
   Cover Studio and returning also restores the current browser's score,
   adapter choices, caption, and other controls without a manual save.
3. Fix the analysis if Essentia got it wrong: the ÷2 / Detected / ×2 buttons and a
   free-text BPM box correct tempo halving or doubling, and a key dropdown
   overrides the detected key. The key override changes generation conditioning;
   it does not apply a DSP pitch shift to the recording. Use **Pitch Shift**
   in the cover settings to change the audio pitch.
   Turn off **Use detected BPM/key** to ignore detected, cached and dataset
   BPM/key results. Loading another recording then resets the BPM and key to
   120 BPM and C major and clears any manual overrides, so the previous song's
   values never carry over; type new values after the load. Source analysis may
   still run. The switch is remembered in this browser; turning it back on uses
   automatic values on the next load.
4. Optionally turn on Advanced Mode to split the source into stems, then mute or
   lower individual stems in the mixer that opens. Splitting alone changes
   nothing: an untouched split recombines at full volume, which is the original
   audio.
5. Fill in artist and title and search Genius for lyrics, paste your own, or turn
   on Instrumental to skip lyrics entirely.
6. Pick a target artist tile (optional). It auto-fills the Style Description from
   that artist's profile or past generations, and if a matching album preset
   carries an adapter, loads that adapter (and its reference track, if any) for
   the render. Skip this and write the Style Description yourself for an
   artist-free cover, using the Caption LLM button to draft one if you like.
7. Set Structure Fidelity, Source Preservation, Tempo Scale and Pitch Shift, and
   optionally a Timbre Reference track, then generate.

YuE2 saves the widths of the source, lyrics and score, and generation settings
columns in this browser. Drag the dividers to make room for the panel you are using.
Drag the divider below the lyrics to change the lead sheet height; that height is
saved too. Wide generation settings columns arrange controls side by side.

YuE2's **Render mode** offers Base YuE2 with a free-text Covering band label but no adapters, Existing Artist with a
trained Lyric Studio artist and album, and Custom with a covering band and hand-picked
AR composer and NAR renderer. Existing Artist lists only artists with at least one
album preset containing both YuE2 adapter paths. The selected mode and choices
are saved with the draft.

## Cover details in Library

Finished covers keep their Covering band label visible as a Tag on Library
cards and rows. YuE2's AR scale and VAE badges help compare takes. In the
track's **How it was made** panel, the Cover Studio group shows the stored
source label and identity, score voices, chord choice, tempo, key and cover
CFG when available. MusicXML sources are identified as scores, without a
source audio ID. A source identity can be a stored URL or ID; a recording
hash is not currently saved with the song.

Export Params retains the stored cover metadata and YuE2 score. Edit still
opens Create and does not reload Cover Studio or its source. Use a saved
Cover Studio draft to resume the workflow.

## YuE2 workflow

1. Choose **Source Audio** or **Import MusicXML** in the left panel. Source Audio
   accepts a recording of at most 100 MB and 10 minutes, or a song sent from
   Library. Detected BPM and key are shown for reference; they do not change
   the YuE2 cover request. In MusicXML mode, choose a `.musicxml`, `.xml` or `.mxl` file,
   select a part when prompted, and review conversion warnings. Switching modes
   clears the previous source and its Prepare confirmations.
2. For Source Audio, transcription starts by itself once the recording has
   loaded (after the upload, the analysis and the dataset lookup, which can supply
   a score of its own). SheetSage2 writes the melody and detected chords into ABC.
   The score panel shows queue and transcription progress, with Cancel and
   **Retranscribe**; **Transcribe full score** is there if the automatic run was
   cancelled or failed. A source restored with the page is not transcribed again. SheetSage2 must be registered for automatic
   transcription; if it is missing, the panel shows the model setup guidance.
   In Source Audio mode you can also paste ABC or import MusicXML in the score
   editor to replace the transcribed ABC while keeping the source recording.
   In MusicXML mode the imported score already supplies ABC, without SheetSage2
   or a source recording. The file picker and drop area are in the left panel.
   Import writes the score in YuE2's own layout: the chosen part's melody on the
   Vocal voice, an Ins voice, 32nd-note lengths with ties, and pickup bars
   padded with rests. Tuplets round to the nearest 32nd with a warning. A grand
   staff starts with its top staff; choose another staff in the import picker.
   You can also choose a lyric verse. If the chosen verse is missing from the
   part, import uses verse 1 when present, then the first available verse; a
   part without lyrics imports without them. The first voice on the chosen staff supplies
   the melody. Other staves and parts fold into the single Ins line, which takes
   the highest sounding accompaniment note at each point. Repeats, alternate
   endings, D.C. and D.S. instructions play out before import. Key and meter
   changes remain in the score, and transposing
   instruments are written at sounding pitch. A later tempo change is ignored
   with a warning because YuE2 uses only the opening tempo. Long scores warn
   where YuE2's 5,120-token limit may cut them.
   Chord symbols in the file become chord symbols on the Vocal voice, written
   the way SheetSage2 writes them: at each change, and again at the start of
   every bar the chord carries into. A note that a change falls inside is split
   and tied, and a bar with no melody keeps its chord rather than becoming a
   rest. Roots are respelled for the key, as SheetSage2 does. Chord types
   YuE2 doesn't know are simplified to the nearest one it does (a ninth becomes
   a seventh), with a warning. When the file has no chord symbols, as most piano
   scores don't, chords are inferred for each bar, or each half bar where they
   change, from the other staves and parts. The warning says so; check them
   before confirming the structure, because the guess is a simple pitch-class match and can
   misread passing notes or name a seventh that isn't there. In MusicXML mode the
   score's title fills an empty Title field and its lyrics fill an empty lyrics
   box; text you have already typed is kept. A score with lyrics also turns
   **Instrumental** off, since that would discard them.
   Importing over an existing score asks before replacing it. Neither path uses
   source audio as an inference conditioning signal after approval.
   A score with no sounding Vocal bar fails transcription or submission with "The transcriber heard no melody in this source"; add a melody note before confirming or generating.
3. Prepare the score in two steps, **1. Fix structure** then **2. Assign lyrics**
   (see [Prepare](#prepare-fix-the-structure-then-assign-lyrics)). Each step ends
   with a confirmation, and Generate stays disabled until both are current for the
   source. Editing the score or changing the source audio, stem mix or source mode
   withdraws them. The raw ABC editor is still there as an advanced editor in step
   1; a change made in it counts as a structure edit. **MIDI**, beside the ABC/Chart
   switch, downloads the score as a MIDI file: the vocal and instrumental lines as
   separate tracks and the chord symbols as a comping track, at the score's tempo
   and meter, for a DAW or notation app.
   A score whose bars are mostly 2/4 shows a **Convert 2/4 to 4/4** button in step 1. It pairs
   every two 2/4 bars into one 4/4 bar, in every voice, and shows the result as a
   chart before anything changes. Each note keeps its pitch and timing: where a
   removed bar line would carry an accidental into the second half of the new
   bar, that note gets its own accidental. Section labels stay on their bars, and
   a 2/4 bar with no partner before a new section, a meter change or the end
   stays in 2/4. Turn on **First bar is a pickup** when the preview puts the
   downbeats mid-bar: bar 1 then stays in 2/4 and pairing starts at bar 2.
   **Apply 4/4** replaces this cover's score (the dataset's `.abc` is not
   changed), so the structure needs confirming again; **Undo conversion** brings the
   2/4 score back. A score with repeats, endings, double bars, voice overlays,
   comments or lyric lines inside the music, or voices that disagree on their
   bars, is refused with the reason and left as it was. Transcription still
   writes the meter it detects.
   For uploaded sources, transcription and approval use the draft's revision;
   a result from an older source or a cancelled transcription cannot replace the
   current score.
4. Paste lyrics or search by artist and title; choose **Instrumental** to render
   without lyrics. An Instrumental cover moves the melody to Ins whether or not
   re-planning is on; the score is already in hand. An instrument plays the
   score's Vocal melody instead of a wordless voice. Where
   both voices play at once the melody wins and Ins keeps only the notes it
   doesn't cover. Vocal keeps its chord symbols. On Base YuE2 the style also
   gains "Instrumental" and "no vocals, no singing, no choir, no spoken words";
   with an adapter pair the style is left as the adapter was trained. The
   generation log reports how many bars moved. Write a YuE2 style caption; Generate
   refuses to start without one, instrumental or not. With an adapter pair, you can
   choose a caption from its training tracks or the nearest detected BPM. The
   chosen caption fills the editable textarea, so later edits go into the render.
   Section edits and lyric assignment happen on the Prepare tab, described under
   [Prepare](#prepare-fix-the-structure-then-assign-lyrics) below. The old chip
   editor and the **Match sections to score** and **Match score to lyrics** buttons
   are gone; lyric suggestions replace them and never rewrite the score.
   Prepare edits only this cover's draft. Dataset files are not touched.
   When a dataset song's tempo or key differs from the score's, Analysis
   shows the score's values with **Save to dataset**. After a confirmation it
   replaces the BPM and key in the song's `.txt`, keeping a backup the same way.
   If the transcriber counted half or double time, use **½×** or **2×** next
   to the score tempo first; the button saves the adjusted value. For uploaded
   sources, the server calculates that tempo and key from the current ABC and
   applies the confirmed dataset save.
5. Choose **Base YuE2** or an adapter pair. In pair mode, select a Lyric Studio
   artist and, if it has multiple albums with YuE2 adapters, an album. Cover
   Studio fills the matching AR and NAR adapters. Either manual selector can
   override that choice. The selected artist fills **Covering band**, which you
   can edit. Downloads use the filename prepend from Lyric Studio settings,
   followed by the covering band and the title with its source-band cover suffix.
   **Score to render** defaults to both the vocal and instrumental lines, no
   chord symbols, free tempo and source key. Choose vocal melody only if you
   want to omit the transcribed instrumental line; keep chords to preserve more
   of the original arrangement, retain or set tempo, or shift the key by up to
   twelve semitones. Keep chords also sets Chain of Thought: on, the render uses
   Full, so the chord symbols reach the composer; off, it uses Melody only. In
   MusicXML mode, Keep chords turns on by itself when the imported score has
   chord symbols, including inferred ones, and off when it has none; you can
   still change it. An Instrumental render keeps the chords on the Vocal voice
   when it moves the melody to Ins. Advanced **Condition strength** tightens style, lyrics and
   score together. Advanced **Lyric Timing (C6)** ties each lyric block to its score section's
   timing; Off by default, Soft and Hard stay selectable but condition strength 1 with either on
   ruins lyric adherence. Build by section ignores it: with a kept prefix it made each take
   replay the whole song so far one frame at a time. The reviewed full score stays available. Click
   **Generate cover** on the **Prepare** tab for a one-shot render; queued renders
   keep all five choices, selected pair, score and settings from that click
   even if the picker changes later.

### Prepare: fix the structure, then assign lyrics

The **Prepare** tab has two steps and a Ready state: **1. Fix structure**,
**2. Assign lyrics**, **Ready**. The order matters. Confirming the structure
freezes the score's sections, boundaries and labels, so pasting new words later
can never re-section the music. Lyrics then only choose which words go in which
section.

#### Step 1: Fix structure

A strip shows every section with its occurrence number, label, bar range and
duration, in one row that scrolls sideways when the song is long. Click a section
to select it; Shift-click, or Shift with the arrow keys, adds its neighbours. The
selected bars are highlighted in the chart and the notation. Clicking a section
also opens a small popup with the playback buttons (below); hovering the section
you last clicked brings it back, and clicking another section moves it there.
**Stop** is in the popup too. Lyrics are not needed in this step.

The edits sit in two collapsed panels side by side, **Edit the selection** and
**Fix bar grouping**. Open one to use it; the open panel spans the width beneath
them.

- **Merge**: select adjacent sections and press Merge. The merged name starts as
  the first label; you can type another. Only the boundaries go. Every note, rest,
  chord and meter stays as it was, so three verse fragments become one verse even
  if their bars have mixed meters.
- **Split**: pick a bar and press **Split here**, naming the new section. The first
  section always starts at bar 1. **Move boundary here** moves the start of a
  section instead; a boundary cannot cross its neighbours or leave a section empty.
  The lead sheet numbers every bar, and while you edit the structure you can also
  drag a section's ▶ label on the lead sheet onto the bar where it should start.
- **Rename**: changes the label in the score, never the music or the words.
  Consecutive choruses can stay separate sections.
- **Delete section and its music** is tucked away on purpose. It warns how many
  bars and how much time it removes, and it turns recording alignment off (see
  below). Merging junk markers is the normal fix; deleting music is a different job.
- **Undo** and **Redo** cover every structure edit in one history, including
  boundary moves and bar regrouping.

**Fix bar grouping** rewrites a run of bars as bars of one meter, in every voice.
The step flags short bars inside the score (a warning, not an error: a short bar
can be deliberate). Choose **From bar**, **To bar** and the **New meter**, press
**Preview**, and look at the result before **Apply**. A run such as 3/4 + 4/4 + 1/4
holds eight quarter-note beats and becomes two 4/4 bars. Notes that cross a new
barline are split and tied, chord symbols keep their positions, and nothing
changes in time. Simply changing the meter marking would shift the music, so that
is never what this does. Nothing is applied automatically. It refuses, with the
reason shown and the score untouched, when:

- a section boundary would fall inside a new bar (move or merge that boundary first);
- the run's beats do not make a whole number of bars in the new meter;
- the voices do not line up on the same bars or meters;
- the key or tempo changes inside the run (regroup either side of the change);
- the run is already in that meter;
- the score uses notation it cannot read exactly, or the check that the result
  sounds the same as the original fails.

In the section popup, **Hear score** plays the selection with the built-in synthesizer. **Hear
recording (approximate)** plays the source recording from one bar before the
selection to its end. Its position is worked out from the score's own tempo, so it
can drift from the real recording; that is why it is labelled approximate. **Loop**
repeats the selected bars. The recording and the score never play together.
Recording playback is unavailable for MusicXML sources, which have no recording,
and after any cut (**Delete section and its music**, or a cut in a draft from an
earlier version), because the score no longer follows the recording. Score
playback still works. Undoing every cut brings the recording back.

**Confirm structure & continue** stays pinned to the bottom of the column, so it is
always in view. It checks that the score has at least one section
and that its section names are usable as lyric tags (no square brackets), then
locks the structure. It does not need lyrics and does not start a transcription.
The confirmation is tied to the exact score: any later change to the ABC
withdraws it.

#### Step 2: Assign lyrics

The left side holds the full lyrics text and, in the collapsed **Lyric Blocks**
panel, your pasted lyrics as blocks, one per `[Tag]` line. The tag
is only a hint about where the words were meant to go; it never renames a section.
The right side has one row for each section of the confirmed score, in order. Each
row shows its words, **Instrumental**, or **Needs words**, and a **Hear** button.
The score itself is read-only here.

- Assign a block to a section from the block's **Assign to…** menu, or select a
  block and use **Put selected block here** on the row. Putting a block on a
  section that already has words asks whether to **Append** or **Swap**; it never
  silently replaces words. Several blocks can fill one section, in the order you
  set.
- **Split before line…** cuts a block at a line, **Join with next** merges two
  blocks. A block cannot span two sections until you split it.
- **Exclude** keeps a block in the draft without singing it. **Instrumental**
  marks a section as wordless.
- Suggestions from the tags and the order are made when you first arrive at this
  step, once per confirmed structure, as long as a section still needs words. They
  mark doubtful ones **not sure** and list blocks left unplaced. **Suggest from tags
  and order** makes them again at any time, for instance after discarding the
  recording suggestions. Pick which
  suggestions to take and press **Accept selected**, or **Discard**. Nothing
  changes until you accept, and suggestions are dropped if the score or lyrics
  change while they are being made.
- **Use the recording to improve suggestions** is a button for when the first
  suggestions are not good enough. It listens to the vocal in the source recording
  to see where each block is sung. It takes longer, and the first run may need to separate the vocals. If it
  fails, you get the tag and order suggestions with a note. It is not offered when
  there is no recording or after a cut. Manual assignment always works.
- **Reuse these words for Chorus 2 and 3** appears when you pasted one chorus and
  the score has several. It shows the previews, and accepting makes independent
  copies: changing one later does not change the others. Words your paste repeats
  stay separate occurrences. Nothing is repeated without your say-so.
- **Update blocks from this text** (re-paste): edit the pasted text and apply it.
  Blocks whose text is unchanged keep their assignment. Anything changed or
  ambiguous, such as identical blocks that cannot be told apart, comes in
  unassigned for review. The structure stays confirmed.

The collapsed **What will be sent, in score order** panel shows exactly the lyrics a
render receives. **Confirm lyrics & finish** is pinned to the bottom of the column
and needs every section to have words or
be Instrumental, and every block to be used or excluded. Changing any words,
assignments or Instrumental and Excluded choices withdraws the lyrics confirmation
only.

#### Ready, and going back

Ready means the structure and the lyrics are both confirmed. Generate and Build by
section use that exact score and these exact lyrics. The server checks both
confirmations again on submit: a render whose score or lyrics no longer match, or
whose Instrumental switch contradicts the lyrics (all sections Instrumental, or
none), is refused with the reason.

**Edit structure** unlocks step 1, and rendering stays off while it is open. A
section whose music, label (a rename included) or boundary
changed loses its assignment and is flagged for review; its words return to
Unassigned rather than disappearing. Unchanged sections keep theirs. Confirm the
structure again, then resolve the flags in step 2. Switching to a different source
starts a separate preparation. If a new score replaces your preparation (for
example a fresh transcription), a note offers **Restore the previous
preparation**.

#### Drafts from before this flow

An older draft opens in the new flow without transcribing anything. Its saved
score and lyrics carry over. When the lyric tags match the score's sections one
for one, each block is proposed for its section (a tag with no words proposes
Instrumental). If they do not match, every block starts Unassigned and none is
lost. Old drafts need the two confirmations once, even if their score was approved
before. Cuts from an earlier version are treated as cuts, so recording playback
and recording suggestions stay off for those scores.

### Build by section

For a one-shot cover, **Generate cover** renders the song in one go. Choose
**Build by section** on **Prepare** to compare takes a few sections at a time.
It opens the **Build** tab, using the approved score, lyrics and settings from
Prepare. Finish preparing those before generating the first takes.
Before each take or preview, and before a Training Studio refine preview, the app frees the VRAM of the other model families (ACE-Step, MiniMax-Music3) and waits for each to confirm, so YuE2 does not load beside a resident model. The title at
the top reads covering artist, then song, as the saved track is named.

1. Two bars show the song. The **Source** bar is the score as it is, and never
   changes. The **Song** bar is the output you are building. It starts empty:
   click a source section to add it to the end of the song, and click it again to
   repeat it (a repeated chorus is its own entry on the Song bar). Hovering a
   source section opens the same playback popup as on Prepare: **Hear score**,
   **Hear recording (approximate)**, **Stop** and **Loop**. The source section
   after the song's last one has a dashed outline and reads **next in source**,
   so you can see where the original goes next; you don't have to use it. Added sections
   arrive selected, so Generate is ready straight away; adding the score's last
   section also ticks **Ends the song**, which you can untick. Built sections
   have ticks and start times, the next one has an arrow, and the rest are planned.
   Hover a planned section to move it earlier or later, or remove it; removing the
   only section, with nothing kept, goes back to an empty Song bar. Built
   sections are locked until you undo back to them. Click a planned section to
   build up to it, and click the last selected one again to clear the selection
   (Generate then waits for one); its lyrics, and those of the planned sections before it, appear
   beside the takes. You don't have to decide what follows: until the song is
   marked as ending, each take runs a few seconds into the section that follows
   its last one in the source, so the next take has a seam to start from, and
   that section is not added to your song. Tick **Ends the song** before building
   the last section so its take composes an ending instead. A take of the score's
   own last section ends there either way. Only a kept take that
   really composed an ending locks the song; one that ran on into the next source
   section does not, so you can still add sections or untick **Ends the song**
   after removing what followed it. Editing the plan is refused while
   takes render, and takes made for an earlier plan can no longer be kept. The
   plan is saved with the build, so it survives a reload and is what **Render
   Final Song** renders.
2. Choose 1 to 10 takes (4 by default), then press **Generate N takes of A → B**, with the
   count and section names shown for your selection. Each take uses a fresh seed,
   continues everything kept so far, and stops a little beyond the chosen range.
   Takes render in groups of up to four, with one audio result per seed. Ten
   takes run as 4 + 4 + 2, and each group finishes before the next starts.
   The pending row shows the running group's shared stage and percentage, plus
   the total takes remaining. **Cancel** stops that group and skips later groups;
   earlier completed takes stay available. Wait for the pending row to clear
   before generating again.
   If a group fails or returns a missing or mismatched track, none of that
   group's takes are added and later groups stop. Earlier completed takes stay.
   Builds also work with **Condition strength** values other than 1.

   **Caption** (at the foot of the lyrics column, under the sections' lyrics)
   starts as the build's caption for each section. Edit it to steer this section's takes, for example "quiet,
   acoustic guitar, whispers" for a breakdown; the next section starts from the
   build's caption again, and the reset button beside the field puts it back. It
   changes what the composer writes (the AR) for those takes, and an adapter's
   trigger is still added. The sound is rendered (the NAR) from the whole song
   under one caption, so a take's preview plays everything, kept sections
   included, under its own caption, and **Render Final Song** renders the whole
   song under the build's caption. So a section can be written differently, but its
   timbre leans on that one caption, and joins between sections composed under
   different captions may be audible. Hover a take's
   name to see the prompt it was sent; ✎ marks a take with its own caption.

   The lyrics beside the takes are editable: one box per chosen section, showing
   the lines it sings without its tag. An instrumental section shows none. Edits
   apply to that section only, when you press Generate, so a second chorus can
   sing other words than the first without touching it, and neither the Prepare
   page nor the dataset's lyrics file changes. A section's lines cannot hold a
   tag line such as `[verse]`. **Reset** sings the build's lyrics for that
   section again. Edits are refused while takes are rendering (cancel or wait),
   built sections keep the lines they were built with, and takes made with the
   old lines stay listed with **Keep** disabled. **Use Prepare lyrics** puts the
   Prepare page's text back for every section not yet built.
3. Play each take to compare it. Preview starts four seconds before its first new
   section, where available, so you can hear the join. A take renders only that
   lead-in and its new part, not the whole song again, so late sections take
   about as long as early ones. The composer still hears everything kept. Takes covering the first
   section play from the beginning. Each card shows a waveform, playhead and the new part's length.
   The play button pauses and resumes from the same spot. Tap the waveform to
   play from that point, lead-in included; the playhead shows where you are.
   Playback stops when you keep, delete the take being played, undo, or leave
   the page. **Keep** adds that take to the build and advances to the next
   section. Use the trash button to delete an unwanted candidate.

   To crop a take, drag across its waveform over the part to keep, or play it
   and press **End at playhead** (and, on a take of the first section, **Start
   at playhead**). What is cut is dimmed, with a red line at the end and, on a
   first-section take, a green line at the start. The end stays in the part
   that take composed, and the start at least a second before it. An end inside
   a section asks what the next take does: compose that section again from the
   cut, or move on to the next section there. While the song is still open and
   nothing follows, moving on means the next section you add starts at the cut.
   **Keep** waits for the answer, then keeps the take and cuts it; **No crop**
   clears it. The next takes continue from exactly the cropped audio. The start
   only changes the saved song: **Render Final Song** starts there with a
   half-second fade in, and **Play build** starts there too. The audio before it
   still feeds the composer, so later takes are unaffected.
   A line under the Song bar shows a stored start or a cropped last keep, with
   **Start from the top** and **Undo crop** to take either back. To crop what is
   already kept, press **Crop previous section** at the top of the Takes box: it
   opens the last kept take, uncropped, with the same waveform, and **Apply**
   cuts it (choosing **No crop** and applying undoes a crop). Takes already made
   from it stop being keepable. Takes made before a crop can no longer be kept
   until the build they continued is back. Adding, removing or moving sections
   after the ones a take composed leaves it keepable.
4. Repeat for the remaining sections. Because takes hold only their own part,
   **Render the build so far** (on the Song bar) renders everything kept once,
   which is quicker than a take since nothing is composed; then **Play build**
   plays it, and clicking a built section plays from its start. After another
   keep the button comes back to render again. A failed render shows **Build
   render failed**; hover it for the reason. To undo a keep, open **Build
   actions** and choose **Undo last keep**. A built section's undo button rolls
   back far enough to build that section again.

5. When every section is built, press **Render Final Song** in the Takes box.
   It submits the kept audio codes as an ordinary generation job, with the usual
   post-processing, drift check and Library save. To render before the song is
   finished, open **Build actions** and choose **Save the song so far**. Once
   the song is saved, the build clears so the next Build starts fresh. If the
   render fails, is cancelled or times out, the build stays so you can try again.

**AR strength** (0 to 10, default 0.6) is Cover Studio's own setting, shown on
both Prepare and Build. It replaces the adapter picker's AR scale for every
Cover render, including 0. Values above 0.6 hurt adherence to the score: on
one seed, 1.0 matched 11% of chords against 89% at 0.6.

A row of settings sits at the top of the Build tab: **Voices**, **Tempo** (with
**Target BPM** when set), **Pitch Shift**, **Condition strength** and **AR
strength**, the same controls and explanations as on Prepare. Before a build
exists they are Prepare's settings and change at once. Once a build exists it
has its own copy: change the row, then choose **Apply**. Until the first keep,
Apply rewrites the build's score and request with the new settings, keeps the
planned sections, and Prepare follows. Takes made before it can no longer be
kept. After a keep, the kept audio was composed under the old settings, so the
row folds to one line saying the settings are locked and listing them. **Change**
opens it again, and its button becomes **Start a fresh build with these**: it sets
them on Prepare and leaves the current build. The row is locked while takes are
rendering, AR strength included.

Edits on Prepare do not change a build that exists. Choose **Start over** in
Build actions or beside **Build in progress** on Prepare to start a new build
from Prepare. **Open** on Prepare returns to the existing build.

A take's seed reproduces its composition only with the same group size. The
same seed rendered alone can differ from one rendered in a group of four.

YuE2 does not use ACE's fidelity, source-preservation/noise, NoFSQ, timbre or
latent controls. Its source BPM/key fields are metadata. The source **Tempo fix**
buttons halve or double the current BPM, including a manual override, within
20 to 300 BPM. **Use detected BPM/key** also applies to YuE2 source loads.
These controls do not change the reviewed ABC notes, key or tempo. Hover
**Source key behavior** for the active backend's explanation.
The YuE2 score tempo and pitch controls
change ABC before composition, rather than source audio. A completed YuE2 cover
uses the ordinary generation queue and appears in Library. Its saved job records
the source kind, a recording identity for Source Audio, the reviewed full score,
the five render choices and the ABC actually rendered. MusicXML mode has no
recording identity; its title comes from the score unless you edit the Title
field, while lyrics and style come from the fields you set. YuE2 ends duration
when its model finishes the song.

After the file upload finishes, Cover Studio shows **Analyzing** while it opens
the source. When its audio exactly matches one unambiguous dataset track and
that track's sidecar has valid BPM and key, it uses those values without running
Essentia. Cover Studio also reads eligible dataset lyrics, labels them **from
dataset**, skips the Genius lookup, and lets you edit them. An instrumental
sidecar selects Instrumental. If the sidecar is absent or incomplete, the
usual lookup and analysis path applies. Dataset lyrics are recorded as such in
the cover's generation parameters until you edit them. A saved `<stem>.abc`
beside the dataset audio opens in the score editor without a transcription job;
**Retranscribe** replaces that saved score after a successful run.

## ACE-Step controls

| Control | What it does |
|---|---|
| Source Audio | Upload zone for the reference track. Drag-drop or click to browse. |
| Latent import | Alternative to Source Audio: load a `.latent`/`.hslat` file and reuse its embedded lyrics/caption/BPM/key. |
| Timbre Reference (optional) | A second reference track (upload or pick from saved references) fed to the engine as a DiT timbre conditioner, independent of any adapter's own reference track. |
| Tempo fix: ÷2 / Detected / ×2 | Corrects Essentia's common tempo halving/doubling error. |
| Custom (BPM) | Free-text BPM override, 20 to 300, replaces the corrected detected value. |
| Key fix | Dropdown of all 24 major/minor keys, overrides the detected key. |
| Language | Vocal language used for the generation. |
| Advanced Mode | Splits the source into stems (SuperSep, selectable separation level), then opens a mixer for muting or attenuating individual stems before generation. |
| Target Artist (optional) | Grid of trained artists. Selecting one auto-fills Style Description and, when available, an album adapter preset. |
| Album | Appears when the selected artist has more than one album preset with an adapter; picks which one to apply. |
| Style Description | Free-text caption for genre, instruments, vocal style, production and mood. Auto-filled from the target artist, always editable. Its Generate button redrafts it with the selected Caption LLM provider on demand; the same provider fills it automatically on artist selection when no existing caption is found. |
| Instrumental | Skips the lyrics requirement and sends the track as instrumental. |
| Structure Fidelity | 0 to 1, default 0.5. How closely the output follows the source's arrangement. |
| Source Preservation | 0 to 1, default 0. "0 = fresh generation from noise, 1 = closely preserve the original." Above 0, a Noise Method choice appears. Classic (Truncate), the default, drops the early steps and runs the rest. Full Denoise (Rescale) runs the same number of steps, spaced by the selected scheduler across the remaining range. |
| NoFSQ Mode | Toggle. Skips FSQ quantization for a result closer to the source. |
| Tempo Scale | 0.5x to 2.0x, default 1.0. Scales the target BPM from the (corrected) source BPM. |
| Pitch Shift | -12 to +12 semitones, default 0. Shows the transposed target key live. |

Every slider's value is click-to-edit: click the number next to a slider to type
an exact figure instead of dragging. A reset arrow appears beside a label whenever
its value has moved from the default; click it to snap that one field back.

## Tips and limits

- Cover generation needs ACE or YuE2 active. MiniMax-Music3 replaces the studio
  with a not-supported notice.
- Splitting into stems is step one of two. The split by itself does not change the
  render; the mixer is where muting or lowering a stem actually removes it from
  the cover.
- For ACE-Step, BPM/key detection depends on the bundled Essentia analyzer. If it returns a
  wrong value, use the tempo-fix and key-fix controls above to correct it. If it
  fails outright, the source loads with no detected-BPM/key panel at all and
  generation falls back to a default of 120 BPM.
- Source filename, detected metadata and lyrics are cached in the browser, so
  re-uploading a track you already analysed here skips analysis and reuses the
  cache.
- Covers queue rather than block the UI: you can adjust settings and start another
  cover while one is still rendering. Uploaded-source covers are submitted to the
  server as soon as you start them, so a queued cover remains there if you close
  or reload the page.
- For ACE-Step, if Whisper lyric transcription is enabled in [generation](../generation.md)
  settings, generated covers get the same LRC-synced lyrics as any other track.
- Finished covers save to the library tagged with their source, alongside every
  other generation; open [Library](library.md) to find, play or export one.

Artist, lyrics-set and preset lists reuse recent results when you return.
Transcription and section-build status refresh while active and stop at the
end. Returning to Cover Studio does not start a new transcription or render;
your lyric edits and queued execution remain separate from list refreshes.

## Related

- [Create](create.md)
- [Repaint Studio](repaint-studio.md)
- [Library](library.md)
- [Generation](../generation.md)
- [Backends](../backends.md)
- [Adapters](../adapters.md)
