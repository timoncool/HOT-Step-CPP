# YuE2 VAE decoder fine-tune

Fine-tunes the YuE2 VAE decoder on real audio and grows it from 3 to N residual units per
block (default 9, about 2x the parameters). The encoder stays frozen, so the latent space,
and therefore the LM, the NAR and every adapter, is unchanged. The added units start
zero-gated: at step 0 the grown decoder's output is bit-identical to the original's.

Runs in the ACE VAE training venv (`D:\Ace-Step-Latest\vae-training\venv`: torch,
stable-audio-tools for the losses and discriminator). The base model is the pinned
`m-a-p/YuE2-Vae` download; its own `modeling_vae.py` defines the architecture.

## Files

| File | Does |
|---|---|
| `common.py` | Builds encoder, original decoder and grown decoder from the base download |
| `data.py` | Train/eval split, random training chunks, the fixed held-out evaluation set |
| `train.py` | The training loop: checkpoints and an evaluation every `--eval-every` steps |
| `evaluate.py` | Round-trip measures against the source and the original decoder; also standalone |
| `export.py` | A checkpoint to an upstream-layout VAE directory for the converter |
| `run.bat` | The launcher with the settings used so far |

## Run

```
cd /d D:\Ace-Step-Latest\hot-step-cpp
tools\yue2-vae-train\run.bat            first start
tools\yue2-vae-train\run.bat --resume   continue after a stop
```

Agents start it with `pwsh -NoProfile -File tools\detach.ps1 tools\yue2-vae-train\run.bat`.
To pause, create an empty file named `STOP` in the run folder: it saves and exits within
a step. The trainer refuses to start without `--vram-gb` + 1 GB of free VRAM and caps
itself at `--vram-gb`, so it stops with an out-of-memory error rather than spilling into
shared memory. Nothing else should use the GPU while it trains.

## Reading the report

`<run>/evals/report.md` gets one row per checkpoint, round-tripping the held-out clips
through the trained decoder; the `original` row is the same clips through the original
decoder. `*` marks a measure that beats the original. `evals/audio/` holds the same 10 s
clip from the source, from the original decoder and from every checkpoint, for listening.

| Measure | Meaning |
|---|---|
| `mel` | Multi-scale log-mel distance to the source |
| `lsd_*` | Log-spectral distance per band, dB |
| `abslevel_*` | Level error per band against the source, dB |
| `flat_10-16k` | High-band spectral flatness minus the source's: above 0 sounds noisier |
| `buzz` | 150/300 Hz modulation of the 4-16 kHz envelope above the source's, dB: the ringing heard in early checkpoints |

The per-step loss lines in `train.log` move with each random batch and do not show
progress; the checkpoint measures do.

## Into the engine

```
python export.py --base <base> --ckpt <run>/ema_NNNNNN.pt --out <dir>
python engine/tools/convert-yue2.py --components vae --src-vae-dec2x <dir> --out "<data root>/models/yue2"
```

This writes `yue2-vae-dec2x-f32.gguf`. After an engine restart, YuE2's VAE Variant control
offers "2x decoder (experimental)". The converter needs the `gguf` package
(`pip install gguf`, or `--target` a folder on `PYTHONPATH`).
