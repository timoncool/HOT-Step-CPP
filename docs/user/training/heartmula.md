# HeartMuLa training

Train a LoRA adapter for the HeartMuLa 3B backbone from your own audio in
[Training Studio](../studios/training-studio.md). The base model, the local
decoder and the codec stay frozen. Only the adapter on the backbone trains. The
trainer is native: no Python runs at any point. The finished adapter appears in the
HeartMuLa adapter picker, see [Adapters](../adapters.md).

## Where to find it

Make HeartMuLa the active backend in the engine picker at the top of the sidebar.
Training Studio then shows three phase tabs after the dataset step:

| Tab | What it does |
|---|---|
| Prepare | Encodes each track to HeartCodec codes and builds its prompt. Nothing from ACE-Step preprocessing is used. |
| Train | Starts, pauses, resumes and cancels a run; shows loss, step time and memory. |
| Review | Lists the checkpoint previews of the run, so you can hear the adapter change. |

Each dataset card shows two chips: **Codes** (prepared samples that still match
the dataset) and **Adapter** (the newest published adapter).

## Requirements

- A CUDA build of the engine and the `ace-train` binary. Other backends (Metal,
  Vulkan, CPU) do not train HeartMuLa yet.
- A card with at least 11.9 GB of memory. The trainer measured 10.4 GiB at its
  smallest setting and 15.4 GiB at its fastest. It picks the fastest setting that
  fits the free memory on its own, so a smaller card trains slower, not worse.
- A HeartMuLa global and local model pair of the variant and quant you train on,
  the tokenizer and the HeartCodec RVQ file, installed as for generation. See
  [Models](../models.md). Prepare needs only the tokenizer and the RVQ file.
- The HeartCodec encoder pack for Prepare: `heartcodec-muencoder-F16.gguf`,
  `heartcodec-wavlm-F16.gguf`, `heartcodec-whisper-F16.gguf` and
  `heartcodec-query-F16.gguf`. Generation does not need them. The Prepare card
  names any that are missing and how to build them with
  `engine/tools/convert-heartmula.py --only encoder`, see
  [the engine guide](../../dev/engine.md#heartmula-training).

The app stops the engine while a preparation or training job runs, because the
trainer needs the GPU to itself, and restarts it afterwards.

## Prepare

Prepare reads the dataset's captions and lyrics. The prompt for each track is its
trigger word, caption and genre as comma-separated tags, then its lyrics (empty
for an instrumental). A track is trained whole, never cut. A track is skipped, with
the reason shown, when its prompt is invalid, when it is too long for the model's
8192-position context, when its audio is missing or unreadable, or when encoding
fails. Changing a track's audio, caption, genre or lyrics makes its sample stale;
run Prepare again before training. Prepared codes are cached, so a second run is
cheap. The Prepare card uses the same live progress panel as YuE2: completed and
failed track counts, current track, phase, an ETA once enough tracks finish, and
the recent log remain visible while the settings are folded away.

## Train

The starting recipe is rank 64, alpha 64, learning rate 1e-4, dropout 0.05 and four
micro-batches per step, with at most 30 epochs or 480 updates by default.
The Train page uses YuE2's amber training panel. Choose
**Resume a previous run** or enter an adapter name, then set the steps, save
interval, seed, rank, alpha and learning rate in the visible parameter row.
Dropout, songs per update, **Train MLP layers** and **Crop songs** sit under **Advanced**.
Checkpoint preview settings follow the recipe, and the start, pause and cancel
controls stay below the form:

| Control | Meaning |
|---|---|
| Rank, Alpha | Adapter size and scaling. Higher ranks use more memory. |
| Learning rate, Gradient accumulation, Training seed | Usual meanings. Keep the seed fixed to compare recipes. |
| LoRA dropout | 0.05 by default; 0 turns it off. Dropout adds about 8% to each step and 1.3 GiB of memory. |
| Train MLP layers | On by default: also trains each backbone block's MLP, not only the attention projections, which unlocks the MLP dial in the Adapters tab. At rank 64 the adapter is about 97M parameters (389 MB) instead of 37M, and steps need more memory and time. Runs started before it was the default resume as attention-only. |
| Crop songs | Off by default. On trains the opening stretch of each longer song (the given seconds, 10 to 600, 120 to start), so songs still start at their start, and trains every fourth song whole so later sections and endings are learned too. Steps get faster on long songs; peak memory still follows the longest song. Runs started before this change used random windows from the middle of songs, which taught previews to start mid-track. |
| Save every, steps | Resumable checkpoint interval. 0 saves only when stopping or finishing. |

**Save and pause** writes a checkpoint and stops; **Resume run** continues it with
the saved recipe and optimizer state. **Cancel** stops at the next micro-step and
keeps the last checkpoint. A server restart or a failed run also keeps its
checkpoints and can be resumed. When the run finishes, the adapter is copied to
the adapter folder, where generation can pick it up.

Measured on an RTX 5090 with nothing else on the GPU, four 94-second samples, rank
64: about 1.0 s per step (median 1001 ms, four micro-batches) and 15.8 GiB allocated.
Step time scales with clip length, and the memory figure is the fastest setting; a
card with less free memory keeps fewer blocks resident and takes longer per step.

### Presets and resume

The recipe cards follow YuE2's layout. **HOT-Approved** selects the current
server recipe, with dropout 0.05 and four songs per update. Its default budget is
`min(480, 30 * ceil(prepared songs / songs per update))`, so small sets stop at
30 epochs. The trainer's last update in each epoch may contain fewer songs.
Before preparation, the form shows 480; the budget is resolved when the prepared
set is known, separately for each dataset in a batch. This is a reversible
mitigation for overfitting; improved sound quality has not been established.
**Fast** uses 240 updates with four songs per update;
**Balanced** uses 480 with four; **Thorough** uses 720 with eight. These change
the training budget, not the model or loss function. Save your own settings in
**Your presets**, or import and export HeartMuLa preset JSON files. Browser
presets store only recipe values, never dataset, adapter name or resume paths.
The changed-settings row shows differences from the server defaults and resets
each field independently.

Explicit update counts, including Fast, Balanced, Thorough and saved presets,
keep their values. A resume keeps the original recipe, including older runs
with dropout off. Reset all to defaults or choose HOT-Approved to restore the
automatic budget.

Resume from the picker or paste a known run ID or run-folder path from this
dataset. The saved recipe is restored, including its base model and quant.

**Base model** and **Base quant** pick what the adapter trains on. The lists show only
the variants and quants with both a global and a local model installed. F16 is the
reference. Smaller quants use less memory (the frozen weights measured 8.4 GiB at F16,
4.9 GiB at Q8_0 and 3.3 GiB at Q4_K_M), and train against the rounding you will
generate with. The adapter records its variant: it plays natively on that variant and
is re-based onto the others (see [HeartMuLa](../backends.md#heartmula)). Checkpoint
previews render on the run's own base. GPU and adapter type are fixed to automatic
CUDA and LoRA by the current trainer.

Every song that is trained to its last frame also trains the end-of-song frame (the same frame the base model samples to end a song), so adapters keep learning where songs end. Runs started before this was added resume without it.

The dataset's trigger word leads every sample's tags, and the finished adapter stores
it, so generation can put it back in front of your tags. Attention mode and layer residency are automatic, with no per-run
controls exposed by this API.

**Perform all stages** on Train prepares missing or stale HeartMuLa audio,
then trains with the settings below. Valid preparation is reused. **Train
multiple** opens the dataset picker; configure the recipe, then use **Start
batch** to prepare and train each selected dataset in order. Each dataset uses
its own slug as the adapter name. The server saves the queue on every stage
transition, so closing the page does not stop it. After a server restart the
queue is paused. Resume it in the batch panel to keep completed stages and
continue an interrupted run from its saved checkpoint. Pause takes effect
after the current stage; Cancel stops the current job and the remaining queue.

On Prepare, **Perform all stages** runs the single encoding stage. **Clear
HeartMuLa caches** shows the folder count and size and deletes only this
dataset's HeartMuLa preparation. Source files, labels, adapters and YuE2, ACE
and MM3 caches remain. **Rebuild all samples** re-encodes HeartMuLa audio without
clearing other backends' data.

## Checkpoint previews

The checkpoint ladder uses the same preview cards and players as YuE2 on
Review. Train shows the live loss chart, step time, VRAM and an ETA for an active run
or a run that finished within 15 minutes of opening the page. Use its **Run** picker to compare checkpoints from an earlier
run. HeartMuLa omits YuE2 scoring, alignment and decoder controls.

With previews on, they render at every save (**Save every**, 100 steps by default;
with Save every at 0, only the base and final takes render). The server pauses training at
each preview step, renders a full song (up to 240 seconds, the engine's maximum) from
the saved checkpoint, then resumes. Training never runs while a preview renders.
Every take uses the same tags, lyrics, seed and sampler, so takes differ only by
their weights.

The ETA is the steps left at the run's mean step time plus the previews still to come
(each checkpoint take and the final one). A preview is costed from this run: the base
take's render time until the first mid-run take, then that take's full pause, which
also counts stopping the trainer, the engine restart and the model reload.

The prompt comes from the dataset, as in YuE2's previews. The tags are those of
the dataset's first sung track (the first track when none is sung), which is what
training saw. The lyrics are the newest Lyric Studio generation for the dataset's
artist, falling back to that track's own lyrics; set **Preview lyrics** to
**Dataset lyrics** to skip the generation. **Tags override** and **Lyrics
override** replace either. Checkpoint renders prepend the adapter's trigger to overridden tags too, using
the same handling as normal generation. A trigger already present as a comma-separated
tag is kept without duplication. The base take uses the saved tags as supplied.
The prompt is worked out once when the run starts and saved with the run, so
every take, retry and resume sings the same words. **Preview
maximum seconds** (5 to 240) caps the length; a song ends on its own before that.
On Review, each take lists the track and generation its tags and lyrics came from.

- **Base model** is rendered once before training starts.
- **Step N** is rendered at each preview step.
- **Final** is rendered when the run finishes, even if the last step is not a
  preview step.

Previews always render on the model the trainer used (3B, F16), whichever model
you generate with, because an adapter is bound to its base. The **Review** tab
lists the takes oldest first, each with its seed and sampler; **Retry preview** re-renders a failed or cancelled take. A render that fails
or is cancelled never stops the run, and the checkpoints are kept. Each pause costs
the render time plus the engine restart and the trainer reloading its models. On the
same card, a 30-second preview took about 13 s to render and paused training for 18
to 20 s in total. A full 240-second song took about 70 s to render and paused training
for about 80 s (a short instrumental clip dataset that ran to the cap; a song that ends
earlier is shorter).

A preview or a retry cannot start while a job runs on the dataset, and a job cannot
start while a preview renders.

If a preview retry is rejected, its error stays visible on that run's ladder even
after the preview list refreshes. Switching runs shows only the selected run's
retry error. Errors loading the list appear separately.
