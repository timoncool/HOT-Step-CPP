"""Dataset split, random training chunks and the fixed evaluation set."""
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

SR = 48000
EXTS = {".flac", ".wav", ".mp3"}


def split(root, out_dir, n_eval=24, seed=1234):
    files = sorted(str(p) for p in Path(root).rglob("*") if p.suffix.lower() in EXTS)
    rng = random.Random(seed)
    rng.shuffle(files)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.txt").write_text("\n".join(files[:n_eval]) + "\n", encoding="utf-8")
    (out / "train.txt").write_text("\n".join(files[n_eval:]) + "\n", encoding="utf-8")
    return files[n_eval:], files[:n_eval]


def read_lines(p):
    return [l for l in Path(p).read_text(encoding="utf-8").splitlines() if l]


def read_clip(path, start_s, dur_s):
    """[2, dur_s*SR] float32 at 48 kHz, or None if the file is short/unreadable."""
    try:
        info = sf.info(path)
        sr = info.samplerate
        n = int(round(dur_s * sr)) + 64
        start = int(start_s * sr)
        if info.frames < start + n:
            return None
        x, _ = sf.read(path, frames=n, start=start, dtype="float32", always_2d=True)
    except Exception:
        return None
    if len(x) < n:  # header claimed more frames than the file delivers (seen: 0)
        return None
    x = torch.from_numpy(x.T.copy())
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    x = x[:2]
    if sr != SR:
        x = AF.resample(x, sr, SR)
    want = int(round(dur_s * SR))
    return x[:, :want] if x.shape[1] >= want else None


class Chunks(torch.utils.data.IterableDataset):
    """Endless random chunks; skips near-silent ones."""

    def __init__(self, files, samples):
        self.files, self.samples = files, samples

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random((wi.id if wi else 0) * 7919 + random.randrange(1 << 30))
        dur = self.samples / SR
        while True:
            f = rng.choice(self.files)
            try:
                length = sf.info(f).duration
            except Exception:
                continue
            if length < dur + 1:
                continue
            try:  # one bad file must not end an overnight run
                x = read_clip(f, rng.uniform(0, length - dur - 0.5), dur)
            except Exception:
                continue
            if x is None or x.pow(2).mean().sqrt() < 1e-3:
                continue
            peak = x.abs().max()
            if peak > 1:
                x = x / peak
            yield x


def build_evalset(eval_files, out_path, dur_s=10.0, extra=None):
    """Fixed clips: 10 s from 40% into each held-out track (+ optional planar f32 extras)."""
    clips, names = [], []
    for f in eval_files:
        try:
            length = sf.info(f).duration
        except Exception:
            continue
        x = read_clip(f, max(0.0, length * 0.4), dur_s)
        if x is None or x.pow(2).mean().sqrt() < 1e-3:
            continue
        clips.append(x.numpy())
        names.append(Path(f).stem)
    for name, p in (extra or {}).items():
        x = np.fromfile(p, "<f4").reshape(2, -1)[:, : int(dur_s * SR)]
        clips.append(x)
        names.append(name)
    np.savez(out_path, clips=np.stack(clips).astype(np.float32), names=np.array(names))
    return len(clips)
