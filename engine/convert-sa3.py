#!/usr/bin/env python3
# convert-sa3.py: Stable Audio 3 checkpoints -> GGUF for the StableStep GGML backend.
#
# Produces four GGUFs into models/:
#   sa3-dit-BF16.gguf       arch "sa3-dit"       from stable-audio-3-medium (model.* keys)
#   sa3-same-enc-F16.gguf   arch "sa3-same-enc"  from pretransform.model.* (encoder side)
#   sa3-same-dec-F16.gguf   arch "sa3-same-dec"  from pretransform.model.* (decoder side)
#   sa3-text-enc-BF16.gguf  arch "sa3-t5gemma"   from the t5gemma-b-b-ul2 subfolder
#
# The SAME pair is stored F16 (not BF16): the decoder's sinusoidal FF blocks
# amplify per-weight rounding noise across its 12 layers (see write_sa3_gguf).
#
# Tensor policy: >=2D weights -> BF16; 1D tensors (norms, biases, scales) -> F32
# (precision finding from the ONNX leg: this model's norm/timestep paths are
# fp32-sensitive — measured cosine 0.966 with blanket fp16 vs 0.9995 scoped).
# Tensor names are the source names minus the strip prefix; the C++ graph
# builders consume them as-is. The full model_config.json is embedded verbatim
# under metadata key "sa3.config_json" (the C++ side parses what it needs).
#
# Runs in the StableAudio3 uv venv:
#   cd d:/Ace-Step-Latest/StableAudio3
#   uv run --with gguf python d:/Ace-Step-Latest/hot-step-cpp/engine/convert-sa3.py
#
# Training base (ace-train sa3-train --base-dit), DiT only, never over the
# installed inference files:
#   uv run --with gguf python .../convert-sa3.py --dit-only --dit-name sa3-dit-base-BF16.gguf #     --ckpt <medium-base model.safetensors> --config <medium-base model_config.json> --out-dir <dir>
# Existing outputs are refused unless --force. Every file records the sha256 of
# its source checkpoint and config ("sa3.source_sha256", "sa3.source_config_sha256").

import hashlib
import json
import os
import sys

import numpy as np
import gguf

sys.path.insert(0, r"d:/Ace-Step-Latest/StableAudio3")
from huggingface_hub import hf_hub_download
from safetensors import safe_open

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
REPO = "stabilityai/stable-audio-3-medium"


def log(msg):
    print(f"[convert-sa3] {msg}", file=sys.stderr, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def to_np(t):
    import torch
    if t.dtype == torch.bfloat16:
        return t.float().numpy()
    return t.numpy()


def write_sa3_gguf(out_path, arch, tensors, config_json, extra_meta=None, half="bf16"):
    """tensors: list of (name, np.float32 array). >=2D stored in `half`
    ("bf16" or "f16"), 1D tensors (norms, biases, scales) stored F32.

    half="f16" is used for the SAME autoencoder halves: the decoder's
    sinusoidal FF layers amplify weight rounding noise layer over layer
    (bf16 ~0.4% rel error -> parity cosine 0.9987 < 0.999; f16 ~0.05%
    passes). Same file size either way; weight magnitudes are far inside
    f16 range."""
    w = gguf.GGUFWriter(out_path, arch)
    w.add_string("sa3.config_json", config_json)
    for k, v in (extra_meta or {}).items():
        w.add_string(k, v)
    import torch
    n_half = n_f32 = 0
    for name, arr in tensors:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.size == 0:
            # e.g. bottleneck.noise_scaling_factor (1, 0, 1) when
            # noise_augment_dim == 0. ggml's gguf reader hits an integer
            # divide-by-zero on ne==0 tensors, and the C++ side never reads
            # them — drop.
            log(f"  skipping zero-element tensor {name} {arr.shape}")
            continue
        if arr.ndim >= 2:
            # raw_dtype does NOT convert — it labels. Convert to 16-bit bytes
            # explicitly (uint16 view keeps the logical shape).
            if half == "f16":
                h = torch.from_numpy(arr).to(torch.float16).view(torch.uint16).numpy()
                w.add_tensor(name, h, raw_dtype=gguf.GGMLQuantizationType.F16)
            else:
                h = torch.from_numpy(arr).to(torch.bfloat16).view(torch.uint16).numpy()
                w.add_tensor(name, h, raw_dtype=gguf.GGMLQuantizationType.BF16)
            n_half += 1
        else:
            w.add_tensor(name, arr)  # F32
            n_f32 += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    size = os.path.getsize(out_path) / 1e9
    log(f"{os.path.basename(out_path)}: {n_half} {half.upper()} + {n_f32} F32 tensors, {size:.2f} GB")


def main():
    # Local-checkpoint support (e.g. LoRA-merged models): --ckpt/--config
    # override the HF download; --out-dir redirects output; --dit-only skips
    # the SAME + T5Gemma GGUFs (unchanged when only the DiT was fine-tuned).
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="Local model.safetensors (default: HF download)")
    ap.add_argument("--config", default=None, help="Local model_config.json (default: HF download)")
    ap.add_argument("--out-dir", default=OUTPUT_DIR)
    ap.add_argument("--dit-only", action="store_true")
    ap.add_argument("--dit-name", default="sa3-dit-BF16.gguf", help="DiT output file name")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()
    out_dir = args.out_dir
    names = [args.dit_name] + ([] if args.dit_only else
             ["sa3-same-enc-F16.gguf", "sa3-same-dec-F16.gguf", "sa3-text-enc-BF16.gguf"])
    existing = [n for n in names if os.path.exists(os.path.join(out_dir, n))]
    if existing and not args.force:
        sys.exit(f"[convert-sa3] refusing to overwrite {existing} in {out_dir} (pass --force)")

    os.makedirs(out_dir, exist_ok=True)
    cfg_path = args.config or hf_hub_download(REPO, "model_config.json")
    ckpt_path = args.ckpt or hf_hub_download(REPO, "model.safetensors")
    with open(cfg_path) as f:
        config_json = f.read()
    source_meta = {"sa3.source_sha256": sha256(ckpt_path), "sa3.source_config_sha256": sha256(cfg_path)}

    # ── Split the combined checkpoint by prefix ─────────────────────────
    dit_tensors, enc_tensors, dec_tensors = [], [], []
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("model."):
                dit_tensors.append((key[len("model."):], to_np(f.get_tensor(key))))
            elif key.startswith("pretransform.model."):
                sub = key[len("pretransform.model."):]
                # AudioAutoencoder members: encoder.*, decoder.*, bottleneck.*,
                # pretransform.* (patched — no weights). Bottleneck params go to BOTH
                # (encoder applies scale/bias+running_std, decoder inverts).
                if sub.startswith("encoder."):
                    enc_tensors.append((sub, to_np(f.get_tensor(key))))
                elif sub.startswith("decoder."):
                    dec_tensors.append((sub, to_np(f.get_tensor(key))))
                elif sub.startswith("bottleneck."):
                    t = to_np(f.get_tensor(key))
                    enc_tensors.append((sub, t))
                    dec_tensors.append((sub, t))
            # conditioner.* (learned padding, seconds embedder) rides with the DiT
            # gguf — small and needed by the same backend module.
            elif key.startswith("conditioner."):
                dit_tensors.append((key, to_np(f.get_tensor(key))))

    write_sa3_gguf(os.path.join(out_dir, args.dit_name),
                  "sa3-dit", dit_tensors, config_json, extra_meta=source_meta)
    if args.dit_only:
        log("Done (dit-only).")
        return
    write_sa3_gguf(os.path.join(out_dir, "sa3-same-enc-F16.gguf"),
                  "sa3-same-enc", enc_tensors, config_json, extra_meta=source_meta, half="f16")
    write_sa3_gguf(os.path.join(out_dir, "sa3-same-dec-F16.gguf"),
                  "sa3-same-dec", dec_tensors, config_json, extra_meta=source_meta, half="f16")

    # ── T5Gemma encoder (separate HF model in the repo subfolder) ───────
    t5_cfg = hf_hub_download(REPO, "config.json", subfolder="t5gemma-b-b-ul2")
    t5_ckpt = hf_hub_download(REPO, "model.safetensors", subfolder="t5gemma-b-b-ul2")
    with open(t5_cfg) as f:
        t5_config_json = f.read()
    t5_tensors = []
    with safe_open(t5_ckpt, framework="pt", device="cpu") as f:
        for key in f.keys():
            # Encoder-only: drop the decoder half (never used by SA3)
            if key.startswith("decoder."):
                continue
            t5_tensors.append((key, to_np(f.get_tensor(key))))
    # The SA3 conditioner's learned padding embedding is applied to the text
    # encoder's output (padded positions replaced) — it belongs to this module,
    # so duplicate it here (it also rides in the DiT gguf with the rest of
    # conditioner.*).
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        key = "conditioner.conditioners.prompt.padding_embedding"
        t5_tensors.append((key, to_np(f.get_tensor(key))))
    write_sa3_gguf(os.path.join(out_dir, "sa3-text-enc-BF16.gguf"),
                  "sa3-t5gemma", t5_tensors, t5_config_json,
                  extra_meta={"sa3.parent_config_json": config_json, **source_meta})

    log("Done.")


if __name__ == "__main__":
    main()
