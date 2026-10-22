"""Export a trained decoder checkpoint as an upstream-layout VAE directory that
engine/tools/convert-yue2.py reads with --src-vae-dec2x.

  python export.py --base B --ckpt runs/dec2x/ema_014000.pt --out DIR

Writes DIR/config.json (release_variant "dec2x", decoder_config.res_units, and
the halo the grown decoder's receptive field needs) and DIR/model.safetensors
(the base encoder unchanged, the checkpoint's EMA decoder, weight-norm g/v kept).
"""
import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    n_res = ck["n_res"]
    mv = common.load_modeling(a.base)
    cfg = common.load_config(a.base)
    dec = mv.OobleckDecoder(**cfg["decoder_config"])
    common.grow(mv, dec, n_res)
    dec.load_state_dict(ck["ema"], strict=True)

    core = cfg["decode_core_frames"]
    low, high = mv._dependency_interval(dec, 0, core * cfg["downsampling_ratio"] - 1)
    required = max(0, -low, high - core + 1)
    halo = max(cfg["decode_halo_frames"], 8 * math.ceil((required + 4) / 8))  # upstream keeps 4 frames spare

    state = {k: v for k, v in load_file(str(Path(a.base) / "model.safetensors")).items() if k.startswith("encoder.")}
    state.update({f"decoder.{k}": v.float().contiguous() for k, v in ck["ema"].items()})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out / "model.safetensors"))
    cfg["release_variant"] = "dec2x"
    cfg["decoder_config"]["res_units"] = n_res
    cfg["required_halo"] = required
    cfg["decode_halo_frames"] = halo
    cfg["trained_from"] = {"checkpoint": Path(a.ckpt).name, "step": ck["step"]}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"exported step {ck['step']}, {n_res} units/block, required_halo {required}, halo {halo} -> {out}")


if __name__ == "__main__":
    main()
