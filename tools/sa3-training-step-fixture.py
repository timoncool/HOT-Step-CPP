#!/usr/bin/env python3
"""Full-step replay fixture for the SA3 trainer (ace-train sa3-train-check).

Input: a verified official-trainer capture from tools/sa3-training-reference.py
(the stage-4 run is _experiments/sa3-training/reference/run-02, manifest
SHA256 13d14ad8646f1748a92ce34f71a8461fbdad7d5f72929fb4ffbef2ed9b19afe7) and the
pinned upstream checkout it names. Uses the same .venv-sa3-reference.

For one captured step it rebuilds the official medium-base model with the
adapters the capture held before that step and:
  1. replays the step in the capture's precision (bf16 base, bf16 autocast) from
     the captured inputs and ASSERTS that the DiT output, the loss and every
     adapter gradient (seconds_total conditioner included) reproduce the capture;
  2. recomputes the same step in true fp32 (no TF32) on exactly the weights the
     GGML trainer holds: >=2D base tensors rounded to bf16 (what convert-sa3.py
     writes), 1D tensors exact f32. That is the tight oracle for the trainer.
It writes the step's inputs (crop audio, SAME latents, prompt embedding,
seconds_total, CFG drop, noised latents, t, inpaint conditioning, masks,
targets), the adapters before and after the optimizer step, and the capture
and fp32 gradients of all 687 adapter tensors.

From the repo root, with a CUDA GPU idle:

  .venv-sa3-reference/Scripts/python.exe tools/sa3-training-step-fixture.py --step 0
  .venv-sa3-reference/Scripts/python.exe tools/sa3-training-step-fixture.py --step 1
  .venv-sa3-reference/Scripts/python.exe tools/sa3-training-step-fixture.py --step 2

Outputs: _experiments/sa3-training/step/step<N>.safetensors. Metadata records the
reference manifest hash, revisions and the replay agreement.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser(description="SA3 full-step replay fixture")
ap.add_argument("--ref", type=Path, default=Path("_experiments/sa3-training/reference/run-02"))
ap.add_argument("--upstream", type=Path, default=Path("_experiments/sa3-training/reference/upstream"))
ap.add_argument("--step", type=int, required=True)
ap.add_argument("--force-drop", action="store_true",
                help="fp32 oracle with the CFG-dropout branch taken (the capture keeps its own draw)")
ap.add_argument("--same-long", action="store_true",
                help="also write same-long.safetensors: a 30 s signal through the official chunked SAME encode")
ap.add_argument("--out", type=Path, help="default _experiments/sa3-training/step/step<N>[-drop].safetensors")
args = ap.parse_args()
out_path = args.out or Path(f"_experiments/sa3-training/step/step{args.step}{'-drop' if args.force_drop else ''}.safetensors")


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
before = "initial-adapters.pt" if args.step == 0 else f"step-{args.step - 1:02d}-updated.pt"
for name in ("effective-model-config.json", f"step-{args.step:02d}-events.pt", f"step-{args.step:02d}-gradients.pt",
             f"step-{args.step:02d}-updated.pt", before):
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

weights = hf_hub_download("stabilityai/stable-audio-3-medium-base", "model.safetensors",
                          revision=manifest["base_revision"])
print("Hashing pinned base weights", flush=True)
if sha256(weights) != manifest["base_sha256"]:
    sys.exit("base weight SHA256 mismatch")
config = json.loads((args.ref / "effective-model-config.json").read_text())
load = lambda n: torch.load(args.ref / n, map_location="cpu", weights_only=True)
dev = torch.device("cuda")
a = manifest["arguments"]
lora_config = dict(rank=a["rank"], alpha=a["lora_alpha"], adapter_type=a["adapter_type"],
                   dropout=a["dropout"], include=a["include"], exclude=a["exclude"])
init = load(before)
updated = load(f"step-{args.step:02d}-updated.pt")
grads_ref = load(f"step-{args.step:02d}-gradients.pt")
events = load(f"step-{args.step:02d}-events.pt")
dit_ev = next(e for e in events if e["function"] == "forward" and e["source"].endswith("models/dit.py"))
ts_ev = next(e for e in events if e["function"] == "training_step")
crop_ev = next(e for e in events if "offset" in e["locals"])
enc_ev = next(e for e in events if e["function"] == "encode" and "kwargs" in e["locals"])
L, TS = dit_ev["locals"], ts_ev["locals"]
md = TS["metadata"][0]
drop = bool(L["dropout_mask"].flatten()[0])
state = load_file(weights)


def build(fp32):
    """Wrapper as scripts/train_lora.py; fp32=True keeps every tensor f32 with
    >=2D base weights rounded through bf16, like the GGML trainer's GGUF."""
    model = create_diffusion_cond_from_config(config)
    sd = state
    if fp32:
        sd = {k: (v.to(torch.bfloat16).float() if v.ndim >= 2 else v.float()) for k, v in state.items()}
    copy_state_dict(model, sd)
    model.to(device=dev, dtype=torch.float32 if fp32 else torch.bfloat16).eval().requires_grad_(False)
    model.pretransform.enable_grad = False
    w = DiffusionCondTrainingWrapper(
        model, mask_loss_weight=1.0, mask_padding_attention=True, silence_extension_scale_seconds=4.0,
        use_ema=False, log_loss_info=False, optimizer_configs=manifest["optimizer"], pre_encoded=False,
        timestep_sampler="trunc_logit_normal", timestep_sampler_options={},
        inpainting_config={"mask_kwargs": {"mask_type_probabilities": [0.1, 0.8, 0.1]}},
        use_effective_length_for_schedule=True, sample_rate=44100, sample_size=config.get("sample_size"),
        lora_config=lora_config, lora_state_dict=None, svd_bases_path=None, log_every_n_steps=1,
        ot_coupling=True, base_precision=None if fp32 else a["base_precision"]).to(dev)
    params = {n: p for n, p in w.named_parameters() if p.requires_grad}
    assert params.keys() == init.keys(), "adapter key mismatch"
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(init[n].to(dev))
    return w, params


cuda = lambda v: v.to(dev) if isinstance(v, torch.Tensor) else v
T = {k: cuda(v) for k, v in TS.items() if isinstance(v, torch.Tensor)}
prompt = TS["conditioning"]["prompt"][0].to(dev)  # T5Gemma output, frozen: an input here


def step(w, autocast):
    """One training step from the captured inputs: conditioning -> DiT -> loss."""
    dit = w.diffusion.model.model
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        sec = w.diffusion.conditioner.conditioners["seconds_total"]([md["seconds_total"]], dev)[0]  # (1,1,768)
        p = prompt if autocast else prompt.float()
        cross = torch.cat([p.to(sec.dtype), sec], dim=1)
        if drop:  # the capture's CFG-dropout draw (dit.py forward)
            cross = torch.zeros_like(cross)
        out = dit(x=T["noised_inputs"], t=T["t"], cross_attn_cond=cross, local_add_cond=L["local_add_cond"].to(dev),
                  global_embed=sec.squeeze(1), padding_mask=T["augmented_padding_mask"], cfg_dropout_prob=0.0)
    mse = (out.float() - T["targets"]) ** 2

    def masked_mean(m):
        return (torch.where(m.unsqueeze(1), mse, 0.0).sum((1, 2)) / (m.sum(1) * mse.shape[1] + 1e-8)).mean()

    ctx = T["inpaint_mask"].squeeze(1).bool() & T["augmented_padding_mask"].bool()
    loss = masked_mean(T["loss_mask"]) + (masked_mean(ctx) if ctx.sum() > 0 else 0.0)
    return out, loss


# 1. Capture precision: must reproduce the recorded step.
torch.set_float32_matmul_precision("high")
w16, p16 = build(False)
w16.train()
out16, loss16 = step(w16, True)
ref_out = dit_ev["result"].to(dev).float()
out_rel = ((out16.float() - ref_out).norm() / ref_out.norm()).item()
ref_loss = manifest["steps"][args.step]["loss"]
loss_rel = abs(loss16.item() - ref_loss) / abs(ref_loss)
print(f"REPLAY output rel={out_rel:.3g} loss={loss16.item():.9g} capture={ref_loss:.9g} rel={loss_rel:.3g}")
replay_loss = loss16.item()
assert out_rel <= 1e-3, f"replayed DiT output does not reproduce the capture (rel {out_rel:.3g})"
assert loss_rel <= 1e-4, f"replayed loss does not reproduce the capture (rel {loss_rel:.3g})"
loss16.backward()
num = den = 0.0
worst_cos, worst_name = 1.0, ""
for n, p in p16.items():
    assert p.grad is not None, f"{n}: no gradient in replay"
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
print(f"REPLAY grads n={len(p16)} aggregate rel={grad_rel:.3g} worst cos={worst_cos:.6f} ({worst_name})")
assert grad_rel <= 5e-2, f"replayed gradients do not reproduce the capture (aggregate rel {grad_rel:.3g})"
assert worst_cos >= 0.98, f"replayed gradient direction differs: {worst_name} cos {worst_cos:.6f}"
del w16, p16, out16, loss16
torch.cuda.empty_cache()

# 2. True fp32 on the GGML trainer's weights: the tight oracle.
drop = drop or args.force_drop
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
w32, p32 = build(True)
w32.train()
out32, loss32 = step(w32, False)
loss32.backward()
fp32_out = ((out32 - ref_out).norm() / ref_out.norm()).item()
print(f"FP32 loss={loss32.item():.9g} output rel vs bf16 capture={fp32_out:.3g}")

save = {
    "audio_crop": crop_ev["result"][0].float(),  # (2, n) after pad/crop, before SAME
    "latents_capture": enc_ev["result"][0].float(),  # (256, T) SAME output on that crop
    "diffusion_input": T["diffusion_input"][0].float(), "noise": T["noise"][0].float(),
    "noised_inputs": T["noised_inputs"][0].float(), "targets": T["targets"][0].float(),
    "t": T["t"].float(), "prompt": prompt[0].float(), "prompt_mask": TS["conditioning"]["prompt"][1][0].float(),
    "local_add_cond": L["local_add_cond"][0].float(), "padding_mask": T["padding_masks"][0].float(),
    "augmented_padding_mask": T["augmented_padding_mask"][0].float(), "loss_mask": T["loss_mask"][0].float(),
    "inpaint_mask": T["inpaint_mask"][0, 0].float(), "output_capture": ref_out[0], "output_fp32": out32.detach()[0],
    "loss_fp32": loss32.detach().reshape(1), "loss_capture": torch.tensor([ref_loss]),
}
tok = w32.diffusion.conditioner.conditioners["prompt"].tokenizer(
    [md["prompt"]], truncation=True, max_length=256, padding="max_length", return_tensors="pt")
assert torch.equal(tok["attention_mask"][0].bool(), TS["conditioning"]["prompt"][1][0].cpu().bool()), "prompt mask"
save["prompt_ids"] = tok["input_ids"][0].to(torch.int32)
# T5Gemma in true fp32 + the base padding embedding: the oracle for the GGML
# text encoder. Upstream loads T5Gemma in its config dtype (bf16), and on CUDA
# that output differs from fp32 by tens of percent, so "prompt" (the capture)
# is not a usable target for an f32 port.
t5 = w32.diffusion.conditioner.conditioners["prompt"]
t5m = t5.model.float().to(dev)
with torch.no_grad():
    emb = t5m(input_ids=tok["input_ids"].to(dev), attention_mask=tok["attention_mask"].to(dev).bool())["last_hidden_state"]
    emb = t5.apply_padding(emb.float(), tok["attention_mask"].to(dev).bool())
save["prompt_fp32"] = emb[0].float()
print(f"T5 capture vs fp32 rel={((prompt[0].float() - emb[0]).norm() / emb[0].norm()).item():.3g}")
for n, p in p32.items():
    save["before." + n] = init[n].float()
    save["after." + n] = updated[n].float()
    save["grad_capture." + n] = grads_ref[n].float()
    save["grad_fp32." + n] = p.grad.float()
save = {k: v.detach().contiguous().cpu() for k, v in save.items()}
out_path.parent.mkdir(parents=True, exist_ok=True)
meta = {"step": str(args.step), "reference_manifest_sha256": sha256(manifest_path),
        "upstream_revision": rev, "base_revision": manifest["base_revision"], "base_sha256": manifest["base_sha256"],
        "seconds_total": repr(md["seconds_total"]), "cfg_dropped": str(int(drop)), "forced_drop": str(int(args.force_drop)),
        "crop_offset": str(crop_ev["locals"]["offset"]), "crop_source_samples": str(crop_ev["locals"]["n_samples"]),
        "replay_output_rel": repr(out_rel), "replay_loss": repr(replay_loss),
        "replay_loss_rel": repr(loss_rel), "replay_grad_rel": repr(grad_rel), "replay_grad_worst_cos": repr(worst_cos),
        "fp32_loss": repr(loss32.item()), "fp32_output_rel_vs_capture": repr(fp32_out)}
save_file(save, str(out_path), metadata=meta)
print(f"WROTE {out_path} ({len(save)} tensors) dropped={int(drop)} seconds_total={md['seconds_total']}")

if args.same_long:
    # Official chunked encode (autoencoders.py encode_audio: 128-latent chunks,
    # 32 overlap) in true fp32 on f16-rounded weights, as convert-sa3.py stores SAME.
    ae = w32.diffusion.pretransform
    with torch.no_grad():
        for p in ae.parameters():
            p.copy_(p.half().float())
        n = 30 * 44100
        tt = torch.arange(n, dtype=torch.float64) / 44100
        g = torch.Generator().manual_seed(7)
        sig = torch.stack([0.2 * torch.sin(2 * torch.pi * 110 * tt) + 0.1 * torch.sin(2 * torch.pi * 0.5 * tt) * torch.sin(2 * torch.pi * 660 * tt),
                           0.2 * torch.sin(2 * torch.pi * 165 * tt + 1.0)]).float()
        sig = sig + 0.02 * torch.randn(sig.shape, generator=g)
        pad = (4096 - n % 4096) % 4096
        audio = torch.nn.functional.pad(sig, (0, pad))[None].to(dev)
        lat = ae.encode(audio)
        assert ae.chunked and audio.shape[-1] >= 128 * 4096, "expected the chunked path"
    p_long = out_path.parent / "same-long.safetensors"
    save_file({"audio": audio[0].cpu().contiguous(), "latents": lat[0].float().cpu().contiguous()}, str(p_long),
              metadata={"samples": str(n), "padded": str(audio.shape[-1]), "chunked": "1"})
    print(f"WROTE {p_long} latents {tuple(lat.shape)}")
