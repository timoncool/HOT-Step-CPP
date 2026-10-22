"""Fine-tune a grown YuE2 VAE decoder; the encoder stays frozen, so latents
(and therefore the LM/NAR) are unchanged.

Losses follow the ACE decoder fine-tune (config_phase2_quality_v2.json):
sum/diff + L/R multi-resolution STFT, multi-scale mel, 0.05 x L1, and an
EnCodec STFT discriminator (adversarial 1.5, feature matching 3.0). The
discriminator is new, so for --disc-warmup steps it trains alone and the
decoder sees reconstruction losses only.

Every --eval-every steps: checkpoint + round-trip of the held-out set against
the original decoder (evals/report.md, evals/metrics.csv, evals/audio/).
Create a file named STOP in the run directory to save and exit; restart
with --resume to continue.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from ema_pytorch import EMA
from stable_audio_tools.models.discriminators import EncodecDiscriminator
from stable_audio_tools.training.losses import auraloss
from stable_audio_tools.training.losses.semantic import MelSpectrogramLoss

import common
import data
import evaluate

STFT = dict(fft_sizes=[8192, 4096, 2048, 1024, 512, 256, 128, 64, 32],
            hop_sizes=[2048, 1024, 512, 256, 128, 64, 32, 16, 8],
            win_lengths=[8192, 4096, 2048, 1024, 512, 256, 128, 64, 32], perceptual_weighting=False)


def inv_lr(step, inv_gamma=100000, power=0.5):
    return (1 + step / inv_gamma) ** -power


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="dir with config.json, model.safetensors, modeling_vae.py")
    ap.add_argument("--run", required=True)
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--n-res", type=int, default=9)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--frames", type=int, default=70, help="chunk length in latent frames (x1920 samples)")
    ap.add_argument("--lr", type=float, default=3e-5, help="pretrained decoder weights")
    ap.add_argument("--lr-new", type=float, default=1e-4, help="appended residual units")
    ap.add_argument("--lr-disc", type=float, default=1e-4)
    ap.add_argument("--disc-warmup", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--vram-gb", type=float, default=26.0,
                    help="hard cap for this process; OOM instead of spilling into shared memory")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    free, total = torch.cuda.mem_get_info()
    if free < (a.vram_gb + 1) * 2**30:
        raise SystemExit(f"only {free / 2**30:.1f} GB VRAM free, need {a.vram_gb + 1:.0f}; "
                         "close whatever else is on the GPU (Unreal?) or lower --vram-gb")
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 2**30 / total))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    log = (run / "train.log").open("a", encoding="utf-8")

    def say(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    enc, base_dec, dec = common.build(a.base, a.n_res)
    dec.train()
    new = common.new_param_names(dec)
    params = dict(dec.named_parameters())
    opt = torch.optim.AdamW([
        {"params": [p for n, p in params.items() if n not in new], "lr": a.lr},
        {"params": [p for n, p in params.items() if n in new], "lr": a.lr_new},
    ], betas=(0.8, 0.99), weight_decay=1e-3)
    disc = EncodecDiscriminator(in_channels=2, filters=64, n_ffts=[4096, 2048, 1024, 512, 256, 128],
                                hop_lengths=[1024, 512, 256, 128, 64, 32],
                                win_lengths=[4096, 2048, 1024, 512, 256, 128]).cuda()
    opt_d = torch.optim.AdamW(disc.parameters(), lr=a.lr_disc, betas=(0.8, 0.99), weight_decay=1e-3)
    ema_dec = common.build(a.base, a.n_res)[2]  # weight-norm modules can't be deep-copied
    ema_dec.load_state_dict(dec.state_dict())
    ema = EMA(dec, ema_model=ema_dec, beta=0.999, update_every=1, update_after_step=0,
              include_online_model=False)
    sdstft = auraloss.SumAndDifferenceSTFTLoss(sample_rate=data.SR, **STFT).cuda()
    lrstft = auraloss.MultiResolutionSTFTLoss(sample_rate=data.SR, **STFT).cuda()
    mel = MelSpectrogramLoss(data.SR, n_mels=[150, 100, 64, 32], window_lengths=[4096, 2048, 1024, 512],
                             pow=1.0, log_weight=1.0, mag_weight=0.0)
    base_lrs = [g["lr"] for g in opt.param_groups]

    step = 0
    latest = run / "latest.pt"
    if a.resume and latest.exists():
        ck = torch.load(latest, map_location="cuda", weights_only=False)
        dec.load_state_dict(ck["model"])
        ema.ema_model.load_state_dict(ck["ema"])
        disc.load_state_dict(ck["disc"])
        opt.load_state_dict(ck["opt"])
        opt_d.load_state_dict(ck["opt_d"])
        step = ck["step"]
        say(f"resumed at step {step}")
    (run / "args.json").write_text(json.dumps(vars(a), indent=1))

    clips = np.load(a.evalset)["clips"]
    evals = run / "evals"
    evals.mkdir(exist_ok=True)
    base = evaluate.base_metrics(enc, base_dec, clips, evals / "base_metrics.json")
    if not (evals / "audio" / "ref_source.flac").exists():
        (evals / "audio").mkdir(exist_ok=True)
        import soundfile as sf
        sf.write(str(evals / "audio" / "ref_source.flac"), clips[-1].T, data.SR)
        sf.write(str(evals / "audio" / "ref_original_decoder.flac"),
                 evaluate.roundtrip(enc, base_dec, clips[-1:])[0].cpu().numpy().T, data.SR)

    def save(tag=None):
        ck = {"step": step, "n_res": a.n_res, "model": dec.state_dict(), "ema": ema.ema_model.state_dict(),
              "disc": disc.state_dict(), "opt": opt.state_dict(), "opt_d": opt_d.state_dict()}
        torch.save(ck, run / "latest.tmp")
        (run / "latest.tmp").replace(latest)
        if tag:
            torch.save({"step": step, "n_res": a.n_res, "ema": ema.ema_model.state_dict()}, run / f"ema_{tag}.pt")

    def run_eval():
        m, _ = evaluate.summarise(enc, ema.ema_model, clips)
        ref = evaluate.roundtrip(enc, ema.ema_model, clips[-1:])[0].cpu().numpy()
        wins = evaluate.report(evals, step, base, m, audio={f"ref_step{step:06d}": ref})
        say(f"eval step {step}: {wins}/{len(evaluate.HEADLINE)} headline metrics better than original | "
            + " ".join(f"{k}={m[k]:.3f}({base[k]:.3f})" for k in evaluate.HEADLINE))

    loader = iter(torch.utils.data.DataLoader(data.Chunks(data.read_lines(a.train_list), a.frames * 1920),
                                              batch_size=a.batch, num_workers=a.workers, pin_memory=True,
                                              persistent_workers=True, prefetch_factor=4))
    say(f"start step {step}, n_res {a.n_res}, batch {a.batch}x{a.frames * 1920} samples")
    t0 = time.time()
    while step < a.steps:
        if (run / "STOP").exists():
            save()
            say(f"STOP file found, saved step {step}, exiting")
            return
        x = next(loader).cuda(non_blocking=True)
        z = common.encode_sample(enc, x)
        y = dec(z)
        n = min(x.shape[-1], y.shape[-1])
        x, y = x[..., :n], y[..., :n]

        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * inv_lr(step)
        for g in opt_d.param_groups:
            g["lr"] = a.lr_disc * inv_lr(step)

        # Discriminator step one clip at a time: its six full-length STFT scales
        # are the memory peak (~5 GB per clip-pass in FP32).
        opt_d.zero_grad(set_to_none=True)
        disc.requires_grad_(True)
        loss_d = 0.0
        yd = y.detach()
        for i in range(x.shape[0]):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ld, _, _ = disc.loss(reals=x[i:i + 1], fakes=yd[i:i + 1])
            (ld / x.shape[0]).backward()
            loss_d += ld.detach() / x.shape[0]
        torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
        opt_d.step()

        rec = (sdstft(y, x) + 0.5 * lrstft(y[:, :1], x[:, :1]) + 0.5 * lrstft(y[:, 1:], x[:, 1:])
               + mel(y, x) + 0.05 * (y - x).abs().mean())
        opt.zero_grad(set_to_none=True)
        adv = fm = torch.zeros((), device=x.device)
        if step >= a.disc_warmup:
            # Adversarial terms one clip at a time against a detached copy of the
            # output; the summed gradient then rides the single decoder backward.
            disc.requires_grad_(False)  # no graph for the reals pass, no stray disc grads
            yv = y.detach().requires_grad_(True)
            for i in range(x.shape[0]):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, adv_i, fm_i = disc.loss(reals=x[i:i + 1], fakes=yv[i:i + 1])
                ((1.5 * adv_i.float() + 3.0 * fm_i.float()) / x.shape[0]).backward()
                adv = adv + adv_i.detach().float() / x.shape[0]
                fm = fm + fm_i.detach().float() / x.shape[0]
            torch.autograd.backward([rec, y], [None, yv.grad])
        else:
            rec.backward()
        loss = rec.detach() + 1.5 * adv + 3.0 * fm
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        ema.update()
        step += 1

        if not math.isfinite(loss.item()):
            say(f"non-finite loss at step {step}, stopping without saving")
            return
        if step % 50 == 0:
            dt = (time.time() - t0) / 50
            t0 = time.time()
            say(f"step {step} rec {rec.item():.4f} adv {adv.item():.4f} fm {fm.item():.4f} "
                f"disc {loss_d.item():.4f} {dt:.2f}s/step vram {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
        if step % a.eval_every == 0:
            save(tag=f"{step:06d}")
            run_eval()
    if step % a.eval_every:
        save(tag=f"{step:06d}")
        run_eval()
    say("done")


if __name__ == "__main__":
    main()
