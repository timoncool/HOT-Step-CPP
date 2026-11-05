#!/usr/bin/env python3
"""Offline HeartMuLa LoRA (torchtune LoRALinear state dict) to an adapter GGUF.

Requires numpy, torch and gguf==0.17.1. No model code is imported.
Example:
  python engine/tools/convert-heartmula-lora.py --lora lora_r64_a64.pt --alpha 64 \
      --base heartmula-3b-global-F16.gguf --out heartmula-lora-r64.gguf

The .pt carries no alpha, so --alpha is required (the run name says r64_a64). Rank
comes from the factor shapes. --base is the global GGUF the adapter was trained on:
it supplies the variant and base_fingerprint, and every target must exist there with
matching in/out widths (engine/docs/HEARTMULA_CONTRACT.md, "LoRA adapter").

Source <proj>.lora_a.weight [rank, in] and .lora_b.weight [out, rank] become
<base tensor name>.lora_a / .lora_b, F32 (the source is bf16, so this is exact), with
no transpose. Only backbone q/k/v/output projections are accepted; anything else is
refused. --negatives DIR also writes wrong-variant, wrong-fingerprint, wrong-component
and wrong-target files there for the engine's rejection checks, plus a valid rank-1
adapter (ok-rank1.gguf) that must load. Existing outputs are refused.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import gguf
import numpy as np
import torch

VERSION = 1
PROJ = ("q_proj", "k_proj", "v_proj", "output_proj")
SOURCE = re.compile(r"^(backbone\.layers\.(\d+)\.attn\.(" + "|".join(PROJ) + r"))\.lora_(a|b)\.weight$")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_base(path):
    """Variant, fingerprint and the (out, in) of every global projection."""
    reader = gguf.GGUFReader(path)
    meta = {k: f.contents() for k, f in reader.fields.items() if k.startswith(("heartmula.", "general."))}
    if meta.get("general.architecture") != "heartmula-global":
        raise ValueError(f"{path}: not a heartmula-global GGUF")
    shapes = {t.name: tuple(int(n) for n in t.data.shape) for t in reader.tensors}
    return meta["heartmula.variant"], meta["heartmula.base_fingerprint"], shapes


def load_factors(path, base_shapes):
    """{base tensor name: (lora_a [r,in] f32, lora_b [out,r] f32)}, fully validated."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    parts = {}
    for key, value in state.items():
        match = SOURCE.match(key)
        if not match:
            raise ValueError(f"Unsupported adapter tensor (only backbone q/k/v/output_proj): {key}")
        base = match.group(1) + ".weight"
        parts.setdefault(base, {})[match.group(4)] = value.detach().to(torch.float32).numpy()
    rank = None
    for base, ab in sorted(parts.items()):
        if set(ab) != {"a", "b"}:
            raise ValueError(f"Unpaired factor: {base}")
        if base not in base_shapes:
            raise ValueError(f"Target not in base GGUF: {base}")
        out_dim, in_dim = base_shapes[base]
        a, b = ab["a"], ab["b"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != in_dim or b.shape[0] != out_dim or a.shape[0] != b.shape[1]:
            raise ValueError(f"{base}: lora_a {a.shape} / lora_b {b.shape} do not fit base [{out_dim}, {in_dim}]")
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError(f"{base}: nonfinite factor")
        rank = a.shape[0] if rank is None else rank
        if a.shape[0] != rank:
            raise ValueError(f"{base}: rank {a.shape[0]} differs from {rank}")
    layers = sorted({int(re.search(r"layers\.(\d+)\.", n).group(1)) for n in parts})
    if not parts or layers != list(range(len(layers))) or len(parts) != 4 * len(layers):
        raise ValueError("Adapter must cover q/k/v/output_proj of every backbone layer it touches")
    return rank, {base: (ab["a"], ab["b"]) for base, ab in parts.items()}


def write(path, factors, rank, alpha, variant, fingerprint, source, targets=None, component="lora"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial =path.with_suffix(".gguf.partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path)
    names = sorted(factors)
    targets = names if targets is None else targets
    manifest = {}
    writer = gguf.GGUFWriter(partial, "heartmula-lora", use_temp_file=True)
    writer.add_string("general.name", path.stem)
    writer.add_uint32("general.file_type", 0)
    writer.add_uint32("heartmula.format_version", VERSION)
    writer.add_string("heartmula.component", component)
    writer.add_string("heartmula.variant", variant)
    writer.add_string("heartmula.base_fingerprint", fingerprint)
    writer.add_string("heartmula.source", json.dumps(source, sort_keys=True))
    writer.add_string("heartmula.lora.target_component", "global")
    writer.add_uint32("heartmula.lora.rank", rank)
    writer.add_float32("heartmula.lora.alpha", alpha)
    writer.add_string("heartmula.lora.targets", json.dumps(targets))
    for name in names:
        for suffix, array in zip((".lora_a", ".lora_b"), factors[name]):
            array = np.ascontiguousarray(array, dtype=np.float32)
            writer.add_tensor(name + suffix, array)
            manifest[name + suffix] = hashlib.sha256(array.tobytes()).hexdigest()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    verify(partial, manifest, rank, alpha, variant, fingerprint, targets)
    partial.rename(path)
    return path


def verify(path, manifest, rank, alpha, variant, fingerprint, targets):
    reader = gguf.GGUFReader(path)
    want = {"heartmula.variant": variant, "heartmula.base_fingerprint": fingerprint,
            "heartmula.lora.rank": rank, "heartmula.lora.targets": json.dumps(targets)}
    for key, value in want.items():
        if reader.fields[key].contents() != value:
            raise ValueError(f"Metadata round-trip mismatch: {key}")
    if abs(float(reader.fields["heartmula.lora.alpha"].contents()) - alpha) > 0:
        raise ValueError("Metadata round-trip mismatch: alpha")
    if {t.name for t in reader.tensors} != set(manifest):
        raise ValueError("GGUF tensor set differs")
    for tensor in reader.tensors:
        if hashlib.sha256(tensor.data.tobytes()).hexdigest() != manifest[tensor.name]:
            raise ValueError(f"GGUF payload differs: {tensor.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--base", type=Path, required=True, help="the global GGUF the adapter was trained on")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--negatives", type=Path, help="also write wrong-variant / wrong-target files here")
    args = parser.parse_args()
    variant, fingerprint, shapes = read_base(args.base)
    rank, factors = load_factors(args.lora, shapes)
    source = {args.lora.name: dict(bytes=args.lora.stat().st_size, sha256=sha256(args.lora)),
              "base_gguf": args.base.name}
    out = write(args.out, factors, rank, args.alpha, variant, fingerprint, source)
    print(f"PASS {out.name}: variant {variant}, rank {rank}, alpha {args.alpha}, "
          f"{len(factors)} sites, {2 * len(factors)} tensors, {out.stat().st_size} bytes")
    if args.negatives:
        args.negatives.mkdir(parents=True, exist_ok=True)
        other = "3B-happy-new-year" if variant == "3B" else "3B"
        write(args.negatives / "bad-variant.gguf", factors, rank, args.alpha, other, fingerprint, source)
        write(args.negatives / "bad-fingerprint.gguf", factors, rank, args.alpha, variant, "0" * 64, source)
        write(args.negatives / "bad-component.gguf", factors, rank, args.alpha, variant, fingerprint, source,
              component="global")
        write(args.negatives / "ok-rank1.gguf", {k: (a[:1], b[:, :1]) for k, (a, b) in factors.items()}, 1, 1.0,
              variant, fingerprint, source)
        extra = sorted(factors)[:-1] + ["backbone.layers.0.mlp.w1.weight"]
        write(args.negatives / "bad-target.gguf", {k: factors[k] for k in sorted(factors)[:-1]}, rank, args.alpha,
              variant, fingerprint, source, targets=extra)
        print(f"PASS negatives in {args.negatives}")


if __name__ == "__main__":
    main()
