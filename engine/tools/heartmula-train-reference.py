#!/usr/bin/env python3
"""HeartMuLa LoRA training oracle and Python baseline, offline.

  python engine/tools/heartmula-train-reference.py --src <checkpoint-root> \
      --sample clip-a.pt --sample clip-b.pt --out <new dir> --device cuda [--sections fixtures,bench]

Written from heartlib's HeartMuLa (modeling_heartmula.py) and torchtune's LoRALinear;
it shares no code with any external training script. engine/docs/HEARTMULA_TRAINING_CONTRACT.md
is the spec this script freezes; the native trainer is checked against its output.

Samples are torch files {audio_tokens [T, 8], tags_ids, lyrics_ids} with the prompt ids
already built by heartlib's own preprocess (lower-case, <tag> wrap, BOS/EOS on each part).

fixtures  FP32 weights and maths (TF32 off). Two crops of the first sample. LoRA r64/a64 on
          the backbone q/k/v/output projections, dropout 0 (the contract says why). Saves the
          prompt, MuQ slot, input embedding, backbone and decoder taps, logits [T, 8, V], the
          eight CEs and the loss, LoRA A/B gradients at the torchtune init state and at a
          seeded non-zero B state, and three AdamW steps of two accumulated micro-batches,
          with each step's accumulated gradient and the optimizer moments after the last
          step. The same graph then runs in plain BF16, the Python recipe's precision, to
          size the drift that sets the native tolerances.
bench     The Python recipe as the overnight run had it: BF16 weights, BF16 LoRA factors,
          dropout 0.05, no backbone recompute (torchtune has no per-layer switch, so the
          recipe never had one), recompute on the local decoder only, AdamW lr 1e-4 wd 0.01,
          batch 1, four micro-batches per optimizer step. Times micro-batches and optimizer
          steps after warm-up and reads the CUDA allocator peaks, per sequence length.

No network access, no runtime integration. Output directories are immutable.
"""

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

spec = importlib.util.spec_from_file_location("heartmula_converter", Path(__file__).with_name("convert-heartmula.py"))
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)

PROJ = ("q_proj", "k_proj", "v_proj", "output_proj")
SHORT = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "output_proj": "o"}
RANK, ALPHA = 64, 64.0
LR, WD, BETAS, EPS = 1e-4, 0.01, (0.9, 0.999), 1e-8
SEED = 776
CROP = 64               # frames per fixture micro-batch
CROP_B_START = 256      # second micro-batch starts here in the same sample
ADAMW_STEPS = 3
PERTURB_STD = 2e-3      # seeded B for the non-zero gradient state


class Fixtures:
    def __init__(self, path):
        self.path = path
        self.arrays = {}

    def save(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy() if value.is_floating_point() else value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite reference: {name}")
        path = self.path / (name + ".npy")
        if name in self.arrays or path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, value, allow_pickle=False)
        self.arrays[name] = dict(file=path.relative_to(self.path).as_posix(), shape=list(value.shape),
                                 dtype=str(value.dtype), bytes=path.stat().st_size, sha256=converter.sha256(path))


def load_sample(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    codes = data["audio_tokens"].long()
    if codes.dim() != 2 or codes.shape[1] != 8 or int(codes.max()) >= 8192 or int(codes.min()) < 0:
        raise ValueError(f"{path}: audio_tokens must be [T, 8] codes in 0..8191")
    tags, lyrics = list(data["tags_ids"]), list(data["lyrics_ids"])
    return dict(codes=codes, tags=tags, lyrics=lyrics, prompt=torch.tensor(tags + [0] + lyrics), muq_idx=len(tags))


def load_model(src, variant, dtype, device):
    from heartlib.heartmula.modeling_heartmula import HeartMuLa
    model = HeartMuLa.from_pretrained(src / ("HeartMuLa-oss-" + variant), dtype=dtype, device_map=device,
                                      local_files_only=True)
    model.requires_grad_(False)
    return model


def apply_lora(model, dropout):
    """torchtune LoRALinear on every backbone attention projection; the base weight is shared."""
    from torchtune.modules.peft import LoRALinear
    for layer in model.backbone.layers:
        for name in PROJ:
            orig = getattr(layer.attn, name)
            if orig.bias is not None:
                raise ValueError("HeartMuLa attention projections carry no bias")
            lora = LoRALinear(in_dim=orig.in_features, out_dim=orig.out_features, rank=RANK, alpha=ALPHA,
                              dropout=dropout).to(device=orig.weight.device, dtype=orig.weight.dtype)
            lora.weight = orig.weight
            setattr(layer.attn, name, lora)


def lora_modules(model):
    return [(i, SHORT[name], getattr(layer.attn, name)) for i, layer in enumerate(model.backbone.layers) for name in PROJ]


def lora_params(model):
    out = []
    for _, _, m in lora_modules(model):
        out += [m.lora_a.weight, m.lora_b.weight]
    return out


def set_lora_state(model, state):
    """state: {"A_q": [L, r, in], "B_q": [L, out, r], ...} on CPU, FP32."""
    with torch.no_grad():
        for i, p, m in lora_modules(model):
            m.lora_a.weight.copy_(state["A_" + p][i])
            m.lora_b.weight.copy_(state["B_" + p][i])


def get_lora(model, attr=None):
    """Stacked FP32 A/B (or their grads, or AdamW moments via a callable) per projection."""
    out = {}
    for i, p, m in lora_modules(model):
        for key, w in (("A_" + p, m.lora_a.weight), ("B_" + p, m.lora_b.weight)):
            value = w if attr is None else attr(w)
            out.setdefault(key, []).append(value.detach().float().cpu())
    return {k: torch.stack(v) for k, v in out.items()}


def seeded_init(model):
    """torchtune's own init (kaiming-uniform A, zero B) under the fixture seed, plus a seeded non-zero B."""
    torch.manual_seed(SEED)
    for _, _, m in lora_modules(model):
        m.initialize_parameters()
    init = get_lora(model)
    g = torch.Generator().manual_seed(SEED + 1)
    perturbed = dict(init)
    for key in [k for k in init if k.startswith("B_")]:
        perturbed[key] = torch.randn(init[key].shape, generator=g) * PERTURB_STD
    return init, perturbed


def forward(model, sample, start, frames, taps=None):
    """The frozen loss (contract section 2). Returns (loss, per-codebook CE [8])."""
    device = next(model.parameters()).device
    prompt, muq_idx = sample["prompt"].to(device), sample["muq_idx"]
    codes = sample["codes"][start:start + frames].to(device)
    S, T = prompt.shape[0], codes.shape[0]
    L = S + T - 1  # the last frame is predicted, never fed

    # heartlib's nine-column input: text in column 8 for prompt rows, codebooks 0..7 for audio rows.
    tokens = torch.zeros(1, L, 9, dtype=torch.long, device=device)
    mask = torch.zeros(1, L, 9, dtype=torch.bool, device=device)
    tokens[0, :S, 8] = prompt
    mask[0, :S, 8] = True
    tokens[0, S:, :8] = codes[:-1]
    mask[0, S:, :8] = True
    embeds = model._embed_tokens(tokens, uncond_mask=None)
    h = (embeds * mask.unsqueeze(-1)).sum(dim=2, dtype=embeds.dtype)
    # No reference audio: generate_frame writes muq_linear(zeros), i.e. the bias, into the MuQ slot.
    muq = model.muq_linear(torch.zeros(model.config.muq_dim, device=device, dtype=h.dtype))
    h = torch.cat((h[:, :muq_idx], muq.view(1, 1, -1).to(h.dtype), h[:, muq_idx + 1:]), 1)

    causal = torch.ones(L, L, dtype=torch.bool, device=device).tril().unsqueeze(0)
    hidden = model.backbone(h, input_pos=torch.arange(L, device=device).unsqueeze(0), mask=causal)
    last = hidden[0, S - 1:].to(h.dtype)  # [T, H]: position S-1+t predicts frame t
    c0 = model.codebook0_head(last.to(model.codebook0_head.weight.dtype))

    # Local decoder over every frame at once: [last_h, emb(c0..c6)] at positions 0..7, causal.
    local = torch.cat((last.unsqueeze(1), torch.stack([model._embed_audio(k, codes[:, k]) for k in range(7)], 1)), 1)
    proj = model.projection(local.to(model.projection.weight.dtype))
    dec_pos = torch.arange(8, device=device).expand(T, 8)
    dec_mask = torch.ones(8, 8, dtype=torch.bool, device=device).tril().expand(T, 8, 8)
    run = lambda x: model.decoder(x, input_pos=dec_pos, mask=dec_mask)
    if taps is None and model.training_recompute_local:
        dec = torch.utils.checkpoint.checkpoint(run, proj, use_reentrant=False)
    else:
        dec = run(proj)
    head = model.audio_head
    ci = [torch.matmul(dec[:, i].to(head.dtype), head[i - 1]) for i in range(1, 8)]
    logits = [c0] + ci
    ces = torch.stack([F.cross_entropy(logits[k].float(), codes[:, k]) for k in range(8)])
    loss = ces.mean()
    if taps is not None:
        taps.update(input_embedding=h[0], muq_slot=muq, backbone_last=last, decoder_out=dec,
                    logits=torch.stack([x.float() for x in logits], 1))  # [T, 8, V]
    return loss, ces


def grads_at(model, sample, state, fixtures, prefix):
    set_lora_state(model, state)
    for p in lora_params(model):
        p.grad = None
    loss, ces = forward(model, sample, 0, CROP)
    loss.backward()
    for k, v in get_lora(model, lambda w: w.grad).items():
        fixtures.save(f"{prefix}/grad/{k}", v)
    fixtures.save(f"{prefix}/loss", loss.detach().float().cpu().reshape(1))
    fixtures.save(f"{prefix}/ce", ces.detach().float().cpu())
    return loss.item()


def adamw_run(model, sample, state, fixtures, prefix, save_grads):
    set_lora_state(model, state)
    params = lora_params(model)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD, betas=BETAS, eps=EPS, foreach=False)
    losses = []
    for step in range(1, ADAMW_STEPS + 1):
        opt.zero_grad(set_to_none=True)
        micro = []
        for start in (0, CROP_B_START):
            loss, _ = forward(model, sample, start, CROP)
            (loss / 2).backward()
            micro.append(loss.item())
        if save_grads:
            for k, v in get_lora(model, lambda w: w.grad).items():
                fixtures.save(f"{prefix}/step{step}/grad/{k}", v)
        opt.step()
        losses.append(micro)
        if step in (1, ADAMW_STEPS):
            for k, v in get_lora(model).items():
                fixtures.save(f"{prefix}/step{step}/{k}", v)
    for name, key in (("exp_avg", "exp_avg"), ("exp_avg_sq", "exp_avg_sq")):
        for k, v in get_lora(model, lambda w: opt.state[w][key]).items():
            fixtures.save(f"{prefix}/step{ADAMW_STEPS}/{name}/{k}", v)
    fixtures.save(f"{prefix}/micro_losses", np.asarray(losses, dtype=np.float32))
    return losses


def compare(ref, cand):
    ref, cand = ref.double().ravel(), cand.double().ravel()
    d = cand - ref
    return dict(max_abs=float(d.abs().max()), relative_l2=float(d.norm() / max(ref.norm().item(), 1e-30)),
                cosine=float(torch.dot(ref, cand) / max((ref.norm() * cand.norm()).item(), 1e-30)))


def fixture_section(args, fixtures, metrics):
    sample = load_sample(args.sample[0])
    if sample["codes"].shape[0] < CROP_B_START + CROP:
        raise ValueError("First sample is too short for the two fixture crops")
    fixtures.save("input/prompt_ids", sample["prompt"])
    fixtures.save("input/muq_idx", np.asarray([sample["muq_idx"]], dtype=np.int64))
    fixtures.save("input/codes", sample["codes"])
    fixtures.save("input/crops", np.asarray([[0, CROP], [CROP_B_START, CROP]], dtype=np.int64))
    # Tokenizer fixture for the native prepare: a neutral prompt through heartlib's own preprocess.
    from types import SimpleNamespace
    from tokenizers import Tokenizer
    from heartlib.pipelines.music_generation import HeartMuLaGenConfig, HeartMuLaGenPipeline
    prep = SimpleNamespace(text_tokenizer=Tokenizer.from_file(str(args.src / "tokenizer.json")),
                           config=HeartMuLaGenConfig.from_file(str(args.src / "gen_config.json")),
                           _muq_dim=512, _parallel_number=9, mula_dtype=torch.float32)
    text = dict(tags="Piano, Ambient, Calm, 80 BPM", lyrics="[Verse]\nA quiet morning\nA distant light\n\n[Chorus]\nHold on")
    inputs = HeartMuLaGenPipeline.preprocess(prep, text, cfg_scale=1.0)
    fixtures.save("tokenizer/prompt_ids", inputs["tokens"][0, :, -1])
    fixtures.save("tokenizer/muq_idx", np.asarray([inputs["muq_idx"][0]], dtype=np.int64))
    metrics["tokenizer_text"] = text

    results = {}
    for precision in ("fp32", "bf16"):
        print("START fixtures/" + precision, flush=True)
        dtype = torch.float32 if precision == "fp32" else torch.bfloat16
        model = load_model(args.src, args.variant, dtype, args.device)
        model.training_recompute_local = False
        model.train()
        apply_lora(model, dropout=0.0)
        if precision == "fp32":
            init, perturbed = seeded_init(model)
            for k in init:
                fixtures.save(f"lora/init/{k}", init[k])
                if k.startswith("B_"):
                    fixtures.save(f"lora/perturbed/{k}", perturbed[k])
            metrics["param_dtypes_fp32"] = sorted({str(p.dtype) for p in model.parameters()})
        else:
            metrics["param_dtypes_bf16"] = {n: str(p.dtype) for n, p in model.named_parameters()
                                            if "layers" not in n}
        # No autocast in either: the recipe runs plain BF16 (its Accelerator has no mixed precision).
        prefix = precision
        out = results.setdefault(precision, {})
        # Taps and logits at the perturbed state, first crop.
        set_lora_state(model, perturbed)
        taps = {}
        with torch.no_grad():
            loss, ces = forward(model, sample, 0, CROP, taps)
        for name, value in taps.items():
            fixtures.save(f"{prefix}/taps/{name}", value)
        fixtures.save(f"{prefix}/forward/ce", ces)
        fixtures.save(f"{prefix}/forward/loss", loss.reshape(1))
        out["taps"] = {k: v.detach().float().cpu() for k, v in taps.items()}
        # Base model loss (B = 0 makes the adapter an identity).
        set_lora_state(model, init)
        with torch.no_grad():
            base_loss, base_ces = forward(model, sample, 0, CROP)
        fixtures.save(f"{prefix}/forward_base/ce", base_ces)
        out["base_ces"] = base_ces.float().cpu()
        out["ces"] = ces.float().cpu()
        out["grad_init_loss"] = grads_at(model, sample, init, fixtures, prefix + "/init")
        out["grad_init"] = get_lora(model, lambda w: w.grad)
        out["grad_perturbed_loss"] = grads_at(model, sample, perturbed, fixtures, prefix + "/perturbed")
        out["grad_perturbed"] = get_lora(model, lambda w: w.grad)
        out["adamw_losses"] = adamw_run(model, sample, perturbed, fixtures, prefix + "/adamw", precision == "fp32")
        out["adamw_final"] = get_lora(model)
        print(f"PASS fixtures/{precision} loss={loss.item():.6f} base={base_loss.item():.6f} "
              f"adamw={out['adamw_losses']}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # BF16 recipe precision against the FP32 oracle: the measured drift class.
    a, b = results["fp32"], results["bf16"]
    drift = dict(ce_abs=[float(x) for x in (b["ces"] - a["ces"]).abs()],
                 base_ce_abs=[float(x) for x in (b["base_ces"] - a["base_ces"]).abs()],
                 taps={k: compare(a["taps"][k], b["taps"][k]) for k in a["taps"]})
    for state in ("grad_init", "grad_perturbed"):
        drift[state] = {k: compare(a[state][k], b[state][k]) for k in a[state] if a[state][k].abs().max() > 0}
    drift["adamw_update"] = {k: compare(a["adamw_final"][k] - perturbed[k], b["adamw_final"][k] - perturbed[k])
                             for k in a["adamw_final"]}
    drift["adamw_losses"] = dict(fp32=a["adamw_losses"], bf16=b["adamw_losses"])
    metrics["fp32"] = dict(ce=[float(x) for x in a["ces"]], loss=float(a["ces"].mean()),
                           base_ce=[float(x) for x in a["base_ces"]], base_loss=float(a["base_ces"].mean()),
                           grad_init_loss=a["grad_init_loss"], grad_perturbed_loss=a["grad_perturbed_loss"])
    metrics["bf16_vs_fp32"] = drift


def bench_section(args, metrics):
    print("START bench", flush=True)
    model = load_model(args.src, args.variant, torch.bfloat16, args.device)
    model.training_recompute_local = True
    model.train()
    apply_lora(model, dropout=0.05)
    torch.manual_seed(SEED)
    params = lora_params(model)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    samples = [load_sample(p) for p in args.sample]
    weights_gib = torch.cuda.memory_allocated() / 2**30
    rows = []
    # Fixture crop, then growing lengths of the first sample, then each whole sample (the real recipe).
    cases = [("fixture", 0, 0, CROP)] + [(f"T{t}", 0, 0, t) for t in (250, 500, 750)] + \
            [(f"sample{i}", i, 0, s["codes"].shape[0]) for i, s in enumerate(samples)]
    for name, index, start, frames in cases:
        sample = samples[index]
        frames = min(frames, sample["codes"].shape[0])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        micro_ms, step_ms = [], []
        for step in range(args.bench_steps + 1):  # step 0 is warm-up
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t_step = time.perf_counter()
            for _ in range(4):
                t0 = time.perf_counter()
                loss, _ = forward(model, sample, start, frames)
                (loss / 4).backward()
                torch.cuda.synchronize()
                if step:
                    micro_ms.append((time.perf_counter() - t0) * 1e3)
            opt.step()
            torch.cuda.synchronize()
            if step:
                step_ms.append((time.perf_counter() - t_step) * 1e3)
        row = dict(case=name, S=int(sample["prompt"].shape[0]), T=int(frames),
                   L=int(sample["prompt"].shape[0] + frames - 1),
                   micro_ms_median=float(np.median(micro_ms)), micro_ms_min=float(np.min(micro_ms)),
                   step_ms_median=float(np.median(step_ms)), steps=len(step_ms),
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, last_loss=loss.item())
        rows.append(row)
        print("BENCH " + json.dumps(row), flush=True)
    metrics["bench"] = dict(rows=rows, weights_allocated_gib=weights_gib,
                            recipe=dict(dtype="bf16 weights, bf16 LoRA factors, no autocast", rank=RANK,
                                        alpha=ALPHA, dropout=0.05, lr=LR, weight_decay=WD, grad_accum=4,
                                        backbone_recompute=False, local_recompute=True, batch=1),
                            gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                            sdpa_flash=torch.backends.cuda.flash_sdp_enabled())
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--sample", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", default="3B")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--sections", default="fixtures,bench")
    parser.add_argument("--bench-steps", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fixtures = Fixtures(args.out)
    versions = {}
    for package in ("torch", "torchtune", "transformers", "tokenizers", "heartlib", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unpackaged"
    metrics = dict(version=1, seed=SEED, rank=RANK, alpha=ALPHA, crop=CROP, crop_b_start=CROP_B_START,
                   adamw=dict(lr=LR, weight_decay=WD, betas=BETAS, eps=EPS, steps=ADAMW_STEPS, micro_batches=2),
                   perturb_std=PERTURB_STD, versions=versions,
                   samples={p.name: converter.sha256(p) for p in args.sample},
                   checkpoint=converter.source_identity(args.src / ("HeartMuLa-oss-" + args.variant)),
                   script=converter.sha256(__file__))
    sections = args.sections.split(",")
    try:
        if "fixtures" in sections:
            fixture_section(args, fixtures, metrics)
        if "bench" in sections:
            bench_section(args, metrics)
    finally:
        metrics["arrays"] = fixtures.arrays
        converter.write_json(args.out / "fixture-progress.json", metrics)
    converter.write_json(args.out / "fixture-manifest.json", metrics)
    print(f"PASS {len(fixtures.arrays)} arrays; manifest SHA256 {converter.sha256(args.out / 'fixture-manifest.json')}",
          flush=True)


if __name__ == "__main__":
    main()
