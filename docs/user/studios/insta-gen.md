# Auto-Gen

Genre-first song creation: pick one or more genres, choose how lyrics get written, and Auto-Gen fills in everything else, style caption, lyrics, BPM, key, time signature and title, then queues the song. Reach for it when you want a finished song fast and don't need to hand-tune every generation parameter; for that, use Custom-Gen instead.

![Auto-Gen](../../images/studio-insta-gen.webp)
<!-- screenshot: needed -->

## Where to find it

The "Auto-Gen" entry in the sidebar (`/insta-gen`). Songs render through the audio queue, and in Instrumental and Lyrics mode the built-in LM also writes the metadata (and in Lyrics mode the lyrics), so the engine has to be running.

The "Lyrics + AI" vocal mode has an external LLM write the lyrics, and usually the BPM, key, title and caption too; when its reply carries no BPM, the built-in LM fills in the metadata. It needs at least one provider set up under Settings > AI Services: Gemini, OpenAI, Anthropic, Ollama, LM Studio, Unsloth Studio, llama.cpp, or a custom OpenAI-compatible endpoint. The LLM Provider dropdown lists the providers the server reports as available: a cloud provider once its API key is set, the OpenAI-compatible entry once it has a base URL, Unsloth Studio once it has a username and password or "Keyless local Studio" is on. Ollama, LM Studio and llama.cpp are always listed, whether or not they are running.

Next to the panel, a Generations list shows songs made in Auto-Gen only, filtered out from the rest of the library. Progress shows in the Queue section of the right-hand Activity column.

## Workflow

1. Select one or more genres in **Select Genres**, or click **Random** to pick 2 to 4 at random.
2. Choose a **Vocal Mode**: Instrumental, Lyrics (the built-in LM writes lyrics on its own), or Lyrics + AI (an external LLM writes lyrics from a subject you give it).
3. For Lyrics + AI, enter a **Song Subject** or click its **Random** button to have the LLM invent one, then pick an **LLM Provider** and **Model**.
4. Optionally add **Additional style tags** and, unless the mode is Instrumental, a **Vocal Language**.
5. Leave **Preview lyrics first** on to review and edit before committing, or turn it off to generate straight through. The toggle is hidden in Instrumental mode, which always generates straight through.
6. Click the action button. With preview on, Auto-Gen saves the resolved lyrics and metadata as a preview; a **Cancel** button under the progress stops that step. Edit the preview's lyrics and caption, then choose **Generate Song** to approve that exact revision for rendering. **Refine in Custom-Gen** copies the preview into Custom-Gen, and **Back to Editing** drops it. Instrumental mode and preview off submit the sequence directly.
7. Watch progress in the Queue section of the Activity column; the finished song lands in the Generations list beside the panel. The server keeps jobs and progress events across browser reconnects. A server restart marks an active workflow interrupted instead of silently rerunning it. The audio queue controls render admission.

## Controls

| Control | What it does |
|---|---|
| Select Genres | Search box over a curated, categorised genre taxonomy (Pop, Rock, Electronic, Hip-Hop and more). Selected genres appear as removable chips and feed the style caption; Clear All removes them. Random picks 2 to 4 genres. |
| Vocal Mode | Instrumental (no lyrics), Lyrics (built-in LM writes lyrics on a random topic), or Lyrics + AI (an external LLM writes lyrics from your subject). |
| System Prompt (Lyrics + AI only) | Collapsible editor for the system prompt sent to the external LLM. Save stores a custom prompt; Reset, shown once a custom prompt exists, asks first, then discards it and restores the built-in default. |
| Song Subject (Lyrics + AI only) | What the song is about, in your own words. Required unless you click its Random button, which asks the LLM to invent a subject instead. Random needs a provider selected; typing in the box turns it off again. |
| LLM Provider / Model (Lyrics + AI only) | Which configured external LLM writes the lyrics, and which of its models. |
| Additional style tags | Free-text tags appended after the selected genres to build the style caption, for example "female vocals, melancholic, reverb-heavy guitar". |
| Vocal Language | Language the lyrics are written and sung in. Hidden when Vocal Mode is Instrumental. |
| Style Caption | Read-only preview of the caption built from the selected genres plus any additional style tags. In Lyrics + AI mode the external LLM writes the caption that is sent (the default system prompt asks it for a long style description); the preview screen shows that caption. |
| Cover Art | Optional override for the auto-generated cover art subject, keeping the automatic art direction. Only shown when cover art generation is enabled in post-processing settings. |
| Caption Rewrite (CoT) | When on, the built-in LM rewrites the caption into a richer style description before generation. When off, your typed caption is sent as-is. |
| Preview lyrics first | When on, the action button stops at a preview of the generated lyrics and metadata before you commit. When off, it queues generation immediately. Hidden in Instrumental mode. |
| Action button | Inspire (Lyrics mode with preview on), Generate Lyrics (Lyrics + AI with preview on), or Generate (Instrumental, or preview off). Runs the lyric/metadata step, or queues generation straight away. Needs at least one genre or style tag; Lyrics + AI also needs a provider and a subject or Random. |
| BPM / Key / Time / Duration (preview screen) | The metadata the lyric step resolved, shown as badges. |
| Edit Caption / Generated Lyrics (preview screen) | Editable text of the caption and lyrics Auto-Gen produced. Changes here are what actually gets generated. |
| Generate Song (preview screen) | Approves the saved preview revision with its edited caption and lyrics plus resolved metadata. A stale revision or changed backend cannot render. |
| Refine in Custom-Gen (preview screen) | Copies the current caption, lyrics, BPM, key, time signature, vocal language and, when the LLM wrote one, title into Custom-Gen's fields and switches to that studio, so you can adjust generation parameters by hand before rendering. The duration goes too, except on MiniMax-Music3 and YuE2. |

A reset arrow appears beside Vocal Mode, Vocal Language and Caption Rewrite whenever
the value has moved from the default; click it to snap that one field back.

## Tips and limits

The song title comes from the external LLM when it supplies one, otherwise Auto-Gen derives it from the lyrics: the first line of the chorus, then verse 1, then the first verse, then whatever lyric line comes first. With no lyrics to read, the style caption becomes the title.

Auto-Gen has no Duration control of its own. It sends the Duration set in Custom-Gen: a set length wins, and the length the lyric step estimated is used only while Custom-Gen's Duration is on Auto. MiniMax-Music3 and YuE2 report that duration is not editable, so their renders always run on Auto. Auto-Gen captures the backend at submission; a backend switch before rendering stops the workflow.

The server repeats the action button's checks (a genre or style tag; for Lyrics + AI, a provider and a subject or Random) and adds no minimum length.

## Related

- [Create (Custom-Gen)](create.md)
- [Lyric Studio](lyric-studio.md)
- [Library](library.md)
- [Generation](../generation.md)
- [Backends](../backends.md)
- [Getting started](../getting-started.md)
