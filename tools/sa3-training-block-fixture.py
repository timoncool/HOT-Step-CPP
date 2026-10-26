#!/usr/bin/env python3
"""Block-0 replay fixture for the SA3 backward gate (engine/tests/test-sa3-train-grad.cpp).

Input: a verified official-trainer capture from tools/sa3-training-reference.py
(the stage-4 run is _experiments/sa3-training/reference/run-02, manifest
SHA256 13d14ad8646f1748a92ce34f71a8461fbdad7d5f72929fb4ffbef2ed9b19afe7) and the
pinned upstream checkout it names. Uses the same .venv-sa3-reference.

It rebuilds the official medium-base DiT with the adapters the capture held
before --step, replays that step's DiT forward in the capture's precision
(bf16 autocast) and ASSERTS that output, loss and every DiT adapter gradient
reproduce the capture. Only then does it record block 0: inputs, output,
upstream gradient and adapter gradients, plus block 0 recomputed alone in
true fp32 (no TF32) with input gradients. Base weights are stored as f32
copies of the bf16 values, so they are exact.

The gate test reads step 2 (all adapter kinds have live gradients; at step 0
B = 0 and every A gradient is zero) and step 1 (negative control). From the
repo root, with a CUDA GPU idle:

  .venv-sa3-reference/Scripts/python.exe tools/sa3-training-block-fixture.py --step 2
  .venv-sa3-reference/Scripts/python.exe tools/sa3-training-block-fixture.py --step 1

Outputs: _experiments/sa3-training/gate/block0-step<N>.safetensors (the test
finds them there, or via SA3_GATE_FIXTURES). Metadata records the reference
manifest hash, upstream and base revisions, and the replay agreement.
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser(description="SA3 block-0 replay fixture")
ap.add_argument("--ref", type=Path, default=Path("_experiments/sa3-training/reference/run-02"))
ap.add_argument("--upstream", type=Path, default=Path("_experiments/sa3-training/reference/upstream"))
ap.add_argument("--step", type=int, required=True)
ap.add_argument("--out", type=Path, help="default _experiments/sa3-training/gate/block0-step<N>.safetensors")
args = ap.parse_args()
out_path = args.out or Path(f"_experiments/sa3-training/gate/block0-step{args.step}.safetensors")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


manifest_path = args.ref / "manifest.json"
manifest = json.loads(manifest_path.read_text())
if manifest.get("status") != "complete":
    sys.exit("reference run is not complete")
if not 0 <= args.step < len(manifest["steps"]):
    sys.exit(f"--step must be 0..{len(manifest['steps']) - 1}")
for name in ("effective-model-config.json", f"step-{args.step:02d}-events.pt", f"step-{args.step:02d}-gradients.pt",
             "initial-adapters.pt" if args.step == 0 else f"step-{args.step - 1:02d}-updated.pt"):
    if sha256(args.ref / name) != manifest["files"][name]["sha256"]:
        sys.exit("reference file hash mismatch: " + name)
rev = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
dirty = subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"], text=True).strip()
if rev != manifest["upstream_revision"] or dirty:
    sys.exit("upstream checkout must be clean at " + manifest["upstream_revision"])

os.environ.update(HF_HUB_OFFLINE="1", ENABLE_TORCH_COMPILE="0")
sys.path.insert(0, str(args.upstream.resolve()))
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from stable_audio_3.factory import create_diffusion_cond_from_config  # noqa: E402
from stable_audio_3.loading_utils import copy_state_dict  # noqa: E402
from stable_audio_3.models import transformer  # noqa: E402
from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper  # noqa: E402

if not torch.cuda.is_available():
    sys.exit("CUDA required: the capture ran in bf16 on CUDA")
# The capture ran without flash-attn or flex (manifest "attention"): SDPA with V-zeroing.
transformer.flex_attention_available = False
transformer.flex_attention_compiled = None
torch.set_float32_matmul_precision("high")

weights = hf_hub_download("stabilityai/stable-audio-3-medium-base", "model.safetensors",
                          revision=manifest["base_revision"])
print("Hashing pinned base weights", flush=True)
if sha256(weights) != manifest["base_sha256"]:
    sys.exit("base weight SHA256 mismatch")
config = json.loads((args.ref / "effective-model-config.json").read_text())
load = lambda n: torch.load(args.ref / n, map_location="cpu", weights_only=True)

dev = torch.device("cuda")
model = create_diffusion_cond_from_config(config)
copy_state_dict(model, load_file(weights))
model.to(device=dev, dtype=torch.bfloat16).eval().requires_grad_(False)
model.pretransform.enable_grad = False
a = manifest["arguments"]
lora_config = dict(rank=a["rank"], alpha=a["lora_alpha"], adapter_type=a["adapter_type"],
                   dropout=a["dropout"], include=a["include"], exclude=a["exclude"])
# Same wrapper arguments as scripts/train_lora.py at the pinned revision.
wrapper = DiffusionCondTrainingWrapper(
    model, mask_loss_weight=1.0, mask_padding_attention=True, silence_extension_scale_seconds=4.0,
    use_ema=False, log_loss_info=False, optimizer_configs=manifest["optimizer"], pre_encoded=False,
    timestep_sampler="trunc_logit_normal", timestep_sampler_options={},
    inpainting_config={"mask_kwargs": {"mask_type_probabilities": [0.1, 0.8, 0.1]}},
    use_effective_length_for_schedule=True, sample_rate=44100, sample_size=config.get("sample_size"),
    lora_config=lora_config, lora_state_dict=None, svd_bases_path=None, log_every_n_steps=1,
    ot_coupling=True, base_precision=a["base_precision"]).to(dev)

init = load("initial-adapters.pt") if args.step == 0 else load(f"step-{args.step - 1:02d}-updated.pt")
params = {n: p for n, p in wrapper.named_parameters() if p.requires_grad}
assert params.keys() == init.keys(), "adapter key mismatch"
with torch.no_grad():
    for n, p in params.items():
        p.copy_(init[n].to(dev))

events = load(f"step-{args.step:02d}-events.pt")
dit_ev = next(e for e in events if e["function"] == "forward" and e["source"].endswith("models/dit.py"))
ts_ev = next(e for e in events if e["function"] == "training_step")
L = dit_ev["locals"]
dit = wrapper.diffusion.model.model  # DiffusionTransformer
layer0 = dit.transformer.layers[0]
pre = "diffusion.model.model.transformer.layers.0."

cap = {}


def pre_hook(_mod, fargs, fkw):
    if "x" in cap:
        return
    cap["x"] = fargs[0].detach().clone()
    for k in ("context", "global_cond", "local_add_cond", "padding_mask"):
        v = fkw.get(k)
        cap[k] = v.detach().clone() if isinstance(v, torch.Tensor) else v
    cap["rope"] = fkw.get("rotary_pos_emb")  # blocks own no RoPE (add_rope off): must be passed


def post_hook(_mod, _fargs, _fkw, out):
    if "y" in cap:
        return
    cap["y"] = out.detach().clone()
    out.register_hook(lambda g: cap.setdefault("dy", g.detach().clone()))


h1 = layer0.register_forward_pre_hook(pre_hook, with_kwargs=True)
h2 = layer0.register_forward_hook(post_hook, with_kwargs=True)
cuda = lambda v: v.to(dev) if isinstance(v, torch.Tensor) else v
# The captured cross_attn_cond is already the post-CFG-dropout value, so no dropout here.
fwd = dict(x=L["x"], t=L["t"], cross_attn_cond=L["cross_attn_cond"], local_add_cond=L["local_add_cond"],
           global_embed=L["global_embed"], padding_mask=L["padding_mask"], cfg_dropout_prob=0.0)
wrapper.train()
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = dit(**{k: cuda(v) for k, v in fwd.items()})
h1.remove()
h2.remove()

# 1. Output: same kernels on the same inputs (observed bit-exact on an RTX 5090).
ref_out = dit_ev["result"].to(dev).float()
out_rel = ((out.float() - ref_out).norm() / ref_out.norm()).item()
print(f"REPLAY output rel={out_rel:.3g}")
assert out_rel <= 1e-3, f"replayed DiT output does not reproduce the capture (rel {out_rel:.3g})"

# 2. Loss, as diffusion.py training_step and tools/sa3-training-reference.py --verify.
T = {k: cuda(v) for k, v in ts_ev["locals"].items() if isinstance(v, torch.Tensor)}
mse = (out.float() - T["targets"]) ** 2


def masked_mean(m):
    return (torch.where(m.unsqueeze(1), mse, 0.0).sum((1, 2)) / (m.sum(1) * mse.shape[1] + 1e-8)).mean()


loss = masked_mean(T["loss_mask"]) + masked_mean(T["inpaint_mask"].squeeze(1).bool() & T["augmented_padding_mask"].bool())
ref_loss = manifest["steps"][args.step]["loss"]
loss_rel = abs(loss.item() - ref_loss) / abs(ref_loss)
print(f"REPLAY loss={loss.item():.9g} capture={ref_loss:.9g} rel={loss_rel:.3g}")
assert loss_rel <= 1e-4, f"replayed loss does not reproduce the capture (rel {loss_rel:.3g})"
loss.backward()

# 3. Gradients. Backward atomics are nondeterministic (two identical captures
# diverged), so per tensor the bar is direction (cosine) and, over all DiT
# adapters together, relative L2. Observed: aggregate <= 1.4e-2, cosine >= 0.99.
grads_ref = load(f"step-{args.step:02d}-gradients.pt")
outside = [n for n, p in params.items() if p.grad is None]
assert outside and all(".conditioner." in n for n in outside), f"unexpected gradient-free adapters: {outside}"
num = den = 0.0
worst_cos, worst_name = 1.0, ""
for n, p in params.items():
    if p.grad is None:
        continue
    g, r = p.grad.float().cpu(), grads_ref[n].float()
    num += float((g - r).pow(2).sum())
    den += float(r.pow(2).sum())
    if float(r.norm()) == 0.0:
        assert float(g.norm()) == 0.0, f"{n}: capture gradient is zero, replay is not"
        continue
    cos = torch.nn.functional.cosine_similarity(g.flatten(), r.flatten(), dim=0).item()
    if cos < worst_cos:
        worst_cos, worst_name = cos, n
grad_rel = (num / den) ** 0.5
print(f"REPLAY grads n={len(params) - len(outside)} (conditioner adapters outside the DiT: {len(outside)}) "
      f"aggregate rel={grad_rel:.3g} worst cos={worst_cos:.6f} ({worst_name})")
assert grad_rel <= 5e-2, f"replayed gradients do not reproduce the capture (aggregate rel {grad_rel:.3g})"
assert worst_cos >= 0.99, f"replayed gradient direction differs: {worst_name} cos {worst_cos:.6f}"

# 4. Block 0 alone under the same autocast must land on the in-model call.
blk16 = copy.deepcopy(layer0)
for p in blk16.parameters():
    p.grad = None
with torch.autocast("cuda", dtype=torch.bfloat16):
    y16 = blk16(cap["x"], rotary_pos_emb=cap["rope"],
                **{k: cap[k] for k in ("context", "global_cond", "local_add_cond", "padding_mask") if cap[k] is not None})
y16.backward(cap["dy"])
blk16_named = dict(blk16.named_parameters())
iso_y = ((y16.float() - cap["y"].float()).norm() / cap["y"].float().norm()).item()
iso_g = max(((blk16_named[n[len(pre):]].grad.float() - p.grad.float()).norm() / p.grad.float().norm()).item()
            for n, p in params.items() if n.startswith(pre) and float(p.grad.norm()) > 0.0)
print(f"ISOLATED block0 y rel={iso_y:.3g} grad worst rel={iso_g:.3g}")
assert iso_y <= 1e-3 and iso_g <= 1e-2, "block 0 called alone does not reproduce the in-model call"

# 5. True fp32 recompute of block 0 (no TF32): the tight oracle for an f32 port.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
blk = copy.deepcopy(layer0).float()
for p in blk.parameters():
    p.grad = None
xin = cap["x"].float().requires_grad_(True)
kw = {k: (cap[k].float() if isinstance(cap[k], torch.Tensor) and cap[k].is_floating_point() else cap[k])
      for k in ("context", "global_cond", "local_add_cond", "padding_mask")}
for k in ("context", "global_cond", "local_add_cond"):
    kw[k].requires_grad_(True)
rope = cap["rope"]
rope32 = tuple(r.float() if isinstance(r, torch.Tensor) else r for r in rope) if isinstance(rope, (tuple, list)) else rope.float()
y32 = blk(xin, rotary_pos_emb=rope32, **{k: v for k, v in kw.items() if v is not None})
y32.backward(cap["dy"].float())
fp32_y = ((y32 - cap["y"].float()).norm() / cap["y"].float().norm()).item()
print(f"FP32 block0 y rel vs bf16 capture={fp32_y:.3g}")

blk_named = dict(blk.named_parameters())
save = {
    "x": cap["x"].float(), "y_bf16": cap["y"].float(), "y_fp32": y32.detach(), "dy": cap["dy"].float(),
    "context": kw["context"], "global_cond": kw["global_cond"], "local_add_cond": kw["local_add_cond"],
    "dx_fp32": xin.grad, "dcontext_fp32": kw["context"].grad, "dglobal_cond_fp32": kw["global_cond"].grad,
    "dlocal_add_cond_fp32": kw["local_add_cond"].grad,
}
if kw["padding_mask"] is not None:
    save["padding_mask"] = kw["padding_mask"].float()
for n, p in params.items():
    if n.startswith(pre):
        short = n[len(pre):]
        save["init." + short] = init[n].float()
        save["grad_bf16." + short] = p.grad.float()
        save["grad_fixture." + short] = grads_ref[n].float()
        save["grad_fp32." + short] = blk_named[short].grad.float()
for n, t in layer0.state_dict().items():
    if "parametrizations" in n and n.endswith("original"):
        save["base." + n.replace(".parametrizations.weight.original", ".weight")] = t.float()
    elif "parametrizations" not in n and "lora_" not in n:
        save["base." + n] = t.float()
save = {k: v.detach().contiguous().cpu() for k, v in save.items()}
out_path.parent.mkdir(parents=True, exist_ok=True)
meta = {"step": str(args.step), "reference_manifest_sha256": sha256(manifest_path),
        "upstream_revision": rev, "base_revision": manifest["base_revision"], "base_sha256": manifest["base_sha256"],
        "replay_output_rel": repr(out_rel), "replay_loss": repr(loss.item()), "capture_loss": repr(ref_loss),
        "replay_grad_rel": repr(grad_rel), "replay_grad_worst_cos": repr(worst_cos),
        "isolated_y_rel": repr(iso_y), "isolated_grad_rel": repr(iso_g), "fp32_y_rel": repr(fp32_y)}
save_file(save, str(out_path), metadata=meta)
print(f"WROTE {out_path} ({len(save)} tensors)")
