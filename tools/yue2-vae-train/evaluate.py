"""Round-trip metrics: source vs original decoder vs trained decoder.

Same bands and measures as the first diagnosis on HOTSTEP-263:
  level  dB of decoded energy relative to source, per band (0 is ideal)
  lsd    log-spectral distance per band, dB (lower is better)
  flat   spectral flatness 10-16 kHz minus the source's (0 is ideal; > 0 = noisier)
  mel    multi-scale log-mel L1 (lower is better)
  buzz   periodic modulation of the 4-16 kHz envelope at 150/300 Hz (frame rate
         x first upsample stride), dB above the source's; the ringing/buzz heard
         in early checkpoints (lower is better, 0 = like source)
Run standalone on a checkpoint:  python evaluate.py --base B --evalset E --ckpt C [--out DIR]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import common

SR = 48000
BANDS = [(0, 500), (500, 2000), (2000, 5000), (5000, 10000), (10000, 16000), (16000, 22000)]
BAND_NAMES = ["0-0.5k", "0.5-2k", "2-5k", "5-10k", "10-16k", "16-22k"]


def _spec(x, n_fft=2048, hop=512):
    w = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x.mean(0), n_fft, hop, window=w, return_complex=True).abs()


def _logmel(x):
    import torchaudio
    out = []
    for n_fft, n_mels in ((4096, 150), (2048, 100), (1024, 64), (512, 32)):
        m = torchaudio.transforms.MelSpectrogram(SR, n_fft=n_fft, hop_length=n_fft // 4, n_mels=n_mels,
                                                 power=1.0).to(x.device)
        out.append(torch.log10(m(x).clamp_min(1e-5)))
    return out


def _modpeaks(x, rates=(150, 300)):
    from scipy.signal import butter, sosfiltfilt, welch
    sos = butter(6, [4000, 16000], "bandpass", fs=SR, output="sos")
    e = np.abs(sosfiltfilt(sos, x.mean(0)))
    e = e[: len(e) // 48 * 48].reshape(-1, 48).mean(1)  # envelope at 1 kHz
    f, P = welch(e - e.mean(), 1000, nperseg=min(4000, len(e)))
    out = []
    for hz in rates:
        i = int(np.argmin(abs(f - hz)))
        nb = np.r_[P[i - 12:i - 3], P[i + 4:i + 13]]
        out.append(10 * np.log10(P[i - 1:i + 2].mean() / np.median(nb)))
    return np.array(out)


def clip_metrics(src, y):
    n = min(src.shape[-1], y.shape[-1])
    src, y = src[..., :n], y[..., :n]
    S, Y = _spec(src), _spec(y)
    f = torch.fft.rfftfreq(2048, 1 / SR, device=src.device)
    r = {}
    for (lo, hi), name in zip(BANDS, BAND_NAMES):
        m = (f >= lo) & (f < hi)
        r[f"level_{name}"] = (10 * torch.log10(Y[m].pow(2).sum() / S[m].pow(2).sum())).item()
        r[f"lsd_{name}"] = (20 * (torch.log10(Y[m] + 1e-7) - torch.log10(S[m] + 1e-7))).pow(2).mean().sqrt().item()

    def flat(M):
        m = (f >= 10000) & (f < 16000)
        P = M[m].pow(2) + 1e-12
        return (torch.exp(torch.log(P).mean(0)) / P.mean(0)).median().item()

    r["flat_10-16k"] = flat(Y) - flat(S)
    r["buzz"] = float((_modpeaks(y.cpu().numpy()) - _modpeaks(src.cpu().numpy())).mean())
    r["mel"] = float(sum((a - b).abs().mean().item() for a, b in zip(_logmel(src), _logmel(y))))
    return r


@torch.no_grad()
def roundtrip(enc, dec, clips, device="cuda"):
    was = dec.training
    dec.eval()
    outs = []
    for c in clips:
        x = torch.from_numpy(c).to(device)[None]
        outs.append(dec(common.encode_mean(enc, x))[0])
    dec.train(was)
    return outs


def summarise(enc, dec, clips):
    rows = []
    for c, y in zip(clips, roundtrip(enc, dec, clips)):
        rows.append(clip_metrics(torch.from_numpy(c).cuda(), y))
    keys = rows[0].keys()
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    # Level error is signed per clip; average its magnitude too, or +2 and -2 cancel.
    for name in BAND_NAMES:
        mean[f"abslevel_{name}"] = float(np.mean([abs(r[f"level_{name}"]) for r in rows]))
    return mean, rows


HEADLINE = ["mel", "lsd_5-10k", "lsd_10-16k", "lsd_16-22k", "abslevel_10-16k", "abslevel_16-22k", "flat_10-16k",
            "buzz"]


def base_metrics(enc, base_dec, clips, cache):
    cache = Path(cache)
    if cache.exists():
        m = json.loads(cache.read_text())
        if all(k in m for k in HEADLINE):  # older caches predate newer metrics
            return m
    m, _ = summarise(enc, base_dec, clips)
    cache.write_text(json.dumps(m, indent=1))
    return m


def report(out_dir, step, base, new, audio=None):
    """Append one checkpoint to metrics.csv and report.md; optional listening FLACs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["step"] + sorted(new)
    csv_path = out / "metrics.csv"
    if csv_path.exists() and csv_path.read_text().splitlines()[0].split(",") != fields:
        csv_path = out / f"metrics_{len(fields)}cols.csv"  # columns changed: new file, old one kept
    first = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if first:
            w.writeheader()
            w.writerow({"step": "base", **base})
        w.writerow({"step": step, **new})
    md = out / "report.md"
    header = "| step | " + " | ".join(HEADLINE) + " | better |"
    text = md.read_text(encoding="utf-8") if md.exists() else (
        "# YuE2 decoder checkpoints vs original decoder\n\n"
        "Round-trip of the fixed held-out set. Lower is better except `flat` (0 = like source).\n")
    if header not in text:  # first run, or the measures changed: start a new table
        text += ("\n" + header + "\n|" + "---|" * (len(HEADLINE) + 2) + "\n"
                 "| original | " + " | ".join(f"{base[k]:.3f}" for k in HEADLINE) + " | |\n")
        md.write_text(text, encoding="utf-8")

    def better(k):
        return abs(new[k]) < abs(base[k]) if k.startswith("flat") else new[k] < base[k]

    wins = sum(better(k) for k in HEADLINE)
    with md.open("a", encoding="utf-8") as fh:
        fh.write(f"| {step} | " + " | ".join(f"{new[k]:.3f}{'*' if better(k) else ''}" for k in HEADLINE)
                 + f" | {wins}/{len(HEADLINE)} |\n")
    if audio is not None:
        (out / "audio").mkdir(exist_ok=True)
        for name, y in audio.items():
            sf.write(str(out / "audio" / f"{name}.flac"), y.T, SR)
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    enc, base_dec, dec = common.build(a.base, ck["n_res"])
    dec.load_state_dict(ck["ema"])
    clips = np.load(a.evalset)["clips"]
    out = Path(a.out or Path(a.ckpt).parent / "evals")
    out.mkdir(parents=True, exist_ok=True)
    base = base_metrics(enc, base_dec, clips, out / "base_metrics.json")
    new, _ = summarise(enc, dec, clips)
    report(out, ck["step"], base, new)
    print(json.dumps({"base": {k: base[k] for k in HEADLINE}, "new": {k: new[k] for k in HEADLINE}}, indent=1))


if __name__ == "__main__":
    main()
