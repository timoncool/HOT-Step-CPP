"""Shared pieces for the YuE2 VAE decoder fine-tune.

The architecture comes from the upstream checkpoint's own modeling_vae.py
(loaded from --base, not vendored), so the trained decoder is the released
one plus extra residual units. Extra units are appended after the original
three in every decoder block, with their output conv's weight-norm gain at
zero: x + 0 = x, so the grown decoder starts bit-identical to the original.
"""
import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

DILATIONS = (1, 3, 9)


def load_modeling(base):
    spec = importlib.util.spec_from_file_location("yue2_modeling_vae", Path(base) / "modeling_vae.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_config(base):
    return json.loads((Path(base) / "config.json").read_text())


def build(base, n_res=3, device="cuda"):
    """Returns (encoder, base_decoder, decoder). Base pieces are frozen, eval, FP32."""
    mv = load_modeling(base)
    cfg = load_config(base)
    state = load_file(str(Path(base) / "model.safetensors"))
    enc = mv.OobleckEncoder(**cfg["encoder_config"])
    enc.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("encoder.")}, strict=True)
    dec_state = {k[8:]: v for k, v in state.items() if k.startswith("decoder.")}
    base_dec = mv.OobleckDecoder(**cfg["decoder_config"])
    base_dec.load_state_dict(dec_state, strict=True)
    dec = mv.OobleckDecoder(**cfg["decoder_config"])
    dec.load_state_dict(dec_state, strict=True)
    grow(mv, dec, n_res)
    for m in (enc, base_dec):
        m.to(device).eval().requires_grad_(False)
    return enc, base_dec, dec.to(device)


def grow(mv, dec, n_res):
    for blk in dec.layers:
        if not isinstance(blk, mv.DecoderBlock):
            continue
        ch = blk.layers[1].out_channels
        for i in range(3, n_res):
            ru = mv.ResidualUnit(ch, ch, DILATIONS[i % 3], "snake")
            with torch.no_grad():
                ru.layers[3].weight_g.zero_()
                ru.layers[3].bias.zero_()
            blk.layers.append(ru)


def new_param_names(dec):
    """Parameters of the appended units (block layer index >= 5)."""
    out = set()
    for name, _ in dec.named_parameters():
        p = name.split(".")
        if len(p) > 3 and p[0] == "layers" and p[2] == "layers" and int(p[3]) >= 5:
            out.add(name)
    return out


def n_res_of(dec):
    return len(dec.layers[1].layers) - 2


@torch.no_grad()
def encode_mean(enc, audio):
    mean, _ = enc(audio).chunk(2, dim=1)
    return mean


@torch.no_grad()
def encode_sample(enc, audio):
    mean, scale = enc(audio).chunk(2, dim=1)
    return mean + torch.randn_like(mean) * (torch.nn.functional.softplus(scale) + 1e-4)
