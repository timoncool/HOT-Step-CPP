#!/usr/bin/env python3
"""Tiny RoPE/GQA fixtures for the HeartMuLa LM port, offline.

  python engine/tools/heartmula-tiny-fixtures.py --out <dir>

Runs torchtune's own MultiHeadAttention with its Llama3ScaledRoPE at the two
HeartMuLa attention shapes (global 24/8 heads x 128, local 8/4 heads x 384) on
random inputs, sparse positions up to 8191 and a causal mask. Projections are
identity slices, so the fixture isolates RoPE, GQA head mapping and attention.
It also records heartlib's sample_topk on tie-heavy logits with the exact
exponential draws it used. heartmula-probe --tiny <dir> replays all of them.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
from torchtune.modules import MultiHeadAttention


class Slice(nn.Module):
    def __init__(self, start, stop):
        super().__init__()
        self.start, self.stop = start, stop

    def forward(self, x):
        return x[..., self.start:self.stop]


def case(out, name, heads, kv_heads, dim, positions, seed):
    torch.manual_seed(seed)
    kv = kv_heads * dim
    rope = Llama3ScaledRoPE(dim, max_seq_len=8192, base=500_000, scale_factor=32)
    attn = MultiHeadAttention(embed_dim=heads * dim, num_heads=heads, num_kv_heads=kv_heads, head_dim=dim,
                              q_proj=nn.Identity(), k_proj=Slice(0, kv), v_proj=Slice(kv, 2 * kv),
                              output_proj=nn.Identity(), pos_embeddings=rope, max_seq_len=8192, is_causal=True)
    s = len(positions)
    q = torch.randn(2, s, heads * dim)
    y = torch.randn(2, s, 2 * kv)
    pos = torch.tensor(positions).expand(2, -1)
    mask = torch.ones(s, s, dtype=torch.bool).tril().expand(2, -1, -1)
    with torch.no_grad():
        result = attn(q, y, mask=mask, input_pos=pos)
        q_rot = rope(q.view(2, s, heads, dim), input_pos=pos)
    for key, value in dict(q=q, kv=y, pos=pos[0], out=result, q_rope=q_rot).items():
        np.save(out / f"{name}_{key}.npy", value.numpy(), allow_pickle=False)
    print(f"PASS {name}: heads {heads}/{kv_heads} x {dim}, positions {positions}")


def sampler_case(out):
    from heartlib.heartmula.modeling_heartmula import sample_topk
    generator = torch.Generator().manual_seed(756)
    # A coarse grid makes ties at the top-k threshold common, as F16 logits do.
    logits = (torch.randn(64, 8197, generator=generator) * 3).div(0.25).round().mul(0.25)
    torch.manual_seed(757)
    draws = torch.empty_like(logits).exponential_(1)
    torch.manual_seed(757)
    codes = sample_topk(logits, 50, 0.9)
    for key, value in dict(logits=logits, draws=draws, codes=codes[:, 0].to(torch.int32)).items():
        np.save(out / f"sample_{key}.npy", value.numpy(), allow_pickle=False)
    print("PASS sampler: 64 rows, topk 50, temperature 0.9")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    positions = [0, 1, 2, 7, 25, 26, 2047, 4096, 8191]
    case(args.out, "global", 24, 8, 128, positions, 754)
    case(args.out, "local", 8, 4, 384, positions[:8], 755)
    sampler_case(args.out)


if __name__ == "__main__":
    main()
