#!/usr/bin/env python3
"""Adapter-on HeartMuLa codebook-0 logits from installed heartlib, offline.

  venv/Scripts/python.exe engine/tools/heartmula-adapter-reference.py \
      --src <checkpoint-root> --fixtures <heartmula-reference.py output> \
      --lora lora_r64_a64.pt --rank 64 --alpha 64 --out <new dir> [--variant 3B]

Applies the trainer's own torchtune LoRALinear (train_lora.py apply_lora) to the
backbone q/k/v/output projections, loads the .pt, then runs real generate_frame with
the fixed-input method of heartmula-reference.py: frame 0 is the prefill from the
frozen prompt, frame 1 one cached decode step; the teacher forces the local codes.
Saves each frame's raw codebook-0 logits [2, 8197] (row 0 conditioned, row 1 not)
for FP32 and F16 execution. CPU by default so it does not take the GPU slot.
Output directories are immutable.
"""

import argparse
import gc
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

PROJ = ("q_proj", "k_proj", "v_proj", "output_proj")


def load_npy(path, device, dtype=None):
    value = torch.from_numpy(np.load(path)).to(device)
    return value if dtype is None or not value.is_floating_point() else value.to(dtype)


def apply_lora(model, state, rank, alpha):
    from torchtune.modules.peft import LoRALinear
    for layer in model.backbone.layers:
        for name in PROJ:
            orig = getattr(layer.attn, name)
            lora = LoRALinear(in_dim=orig.in_features, out_dim=orig.out_features, rank=rank, alpha=alpha, dropout=0.0)
            lora = lora.to(orig.weight.dtype)
            lora.weight.data = orig.weight.data.clone()
            setattr(layer.attn, name, lora)
    own = model.state_dict()
    missing = [k for k in state if k not in own]
    # 4 projections x 2 factors per layer.
    if missing or len(state) != 8 * len(model.backbone.layers):
        raise ValueError(f"Adapter keys do not match the model: {missing[:3]} ({len(state)} tensors)")
    for key, value in state.items():
        own[key].copy_(value.to(own[key].dtype))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", default="3B")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    from heartlib.heartmula import modeling_heartmula as module
    state = torch.load(args.lora, map_location="cpu", weights_only=True)
    base = args.fixtures / "lm" / args.variant
    codes = torch.from_numpy(np.load(base / "fp32" / "teacher_codes.npy"))
    for dtype_name, dtype in (("fp32", torch.float32), ("f16", torch.float16)):
        print("START " + dtype_name, flush=True)
        model = module.HeartMuLa.from_pretrained(args.src / ("HeartMuLa-oss-" + args.variant), dtype=dtype,
                                                device_map=args.device, local_files_only=True).eval()
        model.setup_caches(2)
        apply_lora(model, state, args.rank, args.alpha)
        prompt = base / dtype_name / "prompt"
        muq_embed = load_npy(prompt / "muq_embed.npy", args.device, dtype)
        muq_idx = load_npy(prompt / "muq_idx.npy", args.device)
        raw_c0, book, frame = [], 0, 0
        handle = model.codebook0_head.register_forward_hook(lambda m, i, o: raw_c0.append(o.detach().float().cpu()))

        def teacher(logits, topk, temperature):
            nonlocal book
            result = codes[min(frame, 3), book].expand(logits.shape[0], 1).to(device=args.device, dtype=torch.int)
            book += 1
            return result

        autocast = (lambda: torch.autocast(args.device, dtype=dtype)) if dtype != torch.float32 else nullcontext
        try:
            with patch.object(module, "sample_topk", teacher):
                for frame in range(2):
                    book = 0
                    source = base / dtype_name / f"frame{frame}"
                    tokens = load_npy(source / "input_tokens.npy", args.device)
                    masks = load_npy(source / "input_mask.npy", args.device).bool()
                    pos = load_npy(source / "input_pos.npy", args.device)
                    with autocast():
                        model.generate_frame(tokens, masks, pos, 1.0, 50, 1.5,
                                             continuous_segments=muq_embed if frame == 0 else None,
                                             starts=muq_idx if frame == 0 else None)
                    if book != 8:
                        raise ValueError("Teacher sampler did not see all eight codebooks")
        finally:
            handle.remove()
        if len(raw_c0) != 2:
            raise ValueError(f"Expected one codebook-0 capture per frame, got {len(raw_c0)}")
        (args.out / dtype_name).mkdir(parents=True, exist_ok=True)
        for f, value in enumerate(raw_c0):
            if not torch.isfinite(value).all() or tuple(value.shape) != (2, 8197):
                raise ValueError(f"Bad reference logits: frame {f} {tuple(value.shape)}")
            np.save(args.out / dtype_name / f"frame{f}_c0.npy", value.numpy())
        print("PASS " + dtype_name, flush=True)
        del model, raw_c0
        gc.collect()


if __name__ == "__main__":
    main()
