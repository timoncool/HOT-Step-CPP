#!/usr/bin/env python3
"""Capture a short CUDA run of the pinned official SA3 LoRA trainer.

Run with Python 3.12 in .venv-sa3-reference. Install torch/torchaudio 2.7.1
from the cu128 index, then upstream[lora], matplotlib, pillow, accelerate,
and wandb. The generated requirements.lock.txt records exact resolved versions.
Clone Stability-AI/stable-audio-3 at UPSTREAM_REV into --upstream first.
No network logger, demos, product code, or installed model files are changed.
Tensor .pt files contain CPU tensors and primitives, readable with weights_only=True.
The capture uses the official raw-audio dataset and training loop, not a reimplementation.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

UPSTREAM_REV = "77e0398c26f7d22d801ec1bb036307e8b231f741"
BASE_REV = "b32993f73c3bdc3864043a72d8032606bba737c8"
INFERENCE_REV = "27b5a21b791b1b033d193a9e1e3ce78493f102f9"
BASE_SHA = "c443fcc4d491475064cd0ff3eb92459b1e5f5060e86d96d016f048e528e24195"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(out):
    """Check file hashes, replay masked losses and AdamW updates on CPU."""
    import torch
    torch.set_num_threads(4)
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise RuntimeError("Incomplete fixture run")
    for name, receipt in manifest["files"].items():
        if sha256(out / name) != receipt["sha256"]:
            raise RuntimeError("Hash mismatch: " + name)
    load = lambda name: torch.load(out / name, map_location="cpu", weights_only=True)
    initial = load("initial-adapters.pt")
    params = {n: torch.nn.Parameter(p.clone()) for n, p in initial.items()}
    opt = torch.optim.AdamW(list(params.values()), **manifest["optimizer"]["diffusion"]["optimizer"]["config"])
    max_update_error = 0.0
    for step in manifest["steps"]:
        i = step["step"]
        events = load(f"step-{i:02d}-events.pt")
        local = next(e["locals"] for e in events if e["function"] == "training_step")
        crop = next(e["locals"] for e in events if "offset" in e["locals"])
        assert crop["offset"] >= 0
        assert any(e["function"] == "encode" and "kwargs" in e["locals"] for e in events)
        assert any("dropout_mask" in e["locals"] for e in events)
        torch.testing.assert_close(local["targets"], local["noise"] - local["diffusion_input"], rtol=0, atol=0)
        mse = (local["output"] - local["targets"]) ** 2
        mask = local["loss_mask"]
        context = local["inpaint_mask"].squeeze(1).bool() & local["augmented_padding_mask"].bool()
        def masked_mean(m):
            return (torch.where(m.unsqueeze(1), mse, 0.0).sum((1, 2)) / (m.sum(1) * mse.shape[1] + 1e-8)).mean()
        loss = masked_mean(mask) + masked_mean(context)
        torch.testing.assert_close(loss, torch.tensor(step["loss"], dtype=loss.dtype), rtol=2e-5, atol=1e-7)
        gradients = load(f"step-{i:02d}-gradients.pt")
        updated = load(f"step-{i:02d}-updated.pt")
        assert params.keys() == gradients.keys() == updated.keys()
        for name, param in params.items():
            grad = gradients[name]
            assert torch.isfinite(grad).all() and torch.isfinite(updated[name]).all(), name
            param.grad = grad
        opt.step()
        for name, param in params.items():
            err = float((param.detach() - updated[name]).abs().max())
            max_update_error = max(max_update_error, err)
            torch.testing.assert_close(param.detach(), updated[name], rtol=2e-5, atol=1e-6, msg=name)
        print(f"VERIFY step={i} loss={float(loss):.9g} gradients={len(gradients)} crop_offset={crop['offset']}", flush=True)
    assert any(not torch.equal(initial[n], p.detach()) for n, p in params.items()), "No adapter changed"
    print(f"VERIFY PASS files={len(manifest['files'])} steps={len(manifest['steps'])} max_adamw_abs_error={max_update_error:.9g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("_experiments/sa3-training/reference/upstream"))
    parser.add_argument("--output", type=Path, help="New directory; existing output is refused")
    parser.add_argument("--verify", type=Path, help="Verify an existing fixture run on CPU")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
        return
    if args.output is None:
        parser.error("--output is required unless verifying")
    if args.steps < 2 or args.duration <= 0:
        parser.error("need at least two steps and positive duration")
    upstream = args.upstream.resolve()
    rev = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    if rev != UPSTREAM_REV or dirty:
        raise RuntimeError("Upstream must be clean at " + UPSTREAM_REV)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    def timeout():
        (out / "TIMEOUT.txt").write_text("Reference exceeded wall-clock timeout; run is incomplete.\n", encoding="utf-8")
        os._exit(124)
    timer = threading.Timer(args.timeout_seconds, timeout)
    timer.daemon = True
    timer.start()
    os.environ.update(ENABLE_TORCH_COMPILE="0", WANDB_MODE="disabled", HF_HUB_DISABLE_PROGRESS_BARS="1")
    sys.path.insert(0, str(upstream))
    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from scripts import train_lora as official
    from stable_audio_3.models import transformer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fixtures are not accepted")
    # Windows has no Triton wheel in this environment. Use upstream's SDPA fallback.
    transformer.flex_attention_available = False
    transformer.flex_attention_compiled = None
    base_repo = "stabilityai/stable-audio-3-medium-base"
    config_path = hf_hub_download(base_repo, "model_config.json", revision=BASE_REV)
    weights = hf_hub_download(base_repo, "model.safetensors", revision=BASE_REV)
    print("Hashing pinned base weights", flush=True)
    if sha256(weights) != BASE_SHA:
        raise RuntimeError("Base weight SHA256 mismatch")
    text_snapshot = Path(snapshot_download("stabilityai/stable-audio-3-medium", revision=INFERENCE_REV,
                                         allow_patterns="t5gemma-b-b-ul2/*"))
    config = json.loads(Path(config_path).read_text())
    # Only replace the unpinned remote text-model locator with its pinned local path.
    text_config = config["model"]["conditioning"]["configs"][0]["config"]
    text_config["model_path"] = str(text_snapshot)
    effective_config = out / "effective-model-config.json"
    effective_config.write_text(json.dumps(config, indent=2), encoding="utf-8")

    class PinnedModel:
        def resolve(self):
            return str(effective_config), weights

    official.base_models["medium-base"] = PinnedModel()
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    (out / "requirements.lock.txt").write_text("\n".join(f"{n}=={v}" for n, v in sorted(packages.items())) + "\n", encoding="utf-8")
    manifest = dict(status="running", upstream_revision=rev, base_revision=BASE_REV,
                    collector_sha256=sha256(__file__),
                    base_sha256=BASE_SHA, text_model_revision=INFERENCE_REV,
                    fixture_model="medium-base", inference_model_tested=False,
                    python=sys.version, packages=packages, torch_cuda=torch.version.cuda,
                    cuda_device=torch.cuda.get_device_name(), gpu_before=subprocess.check_output(["nvidia-smi"], text=True),
                    source_model_config_sha256=sha256(config_path),
                    text_files={p.name: sha256(p) for p in (text_snapshot / "t5gemma-b-b-ul2").iterdir() if p.is_file()},
                    attention="upstream SDPA fallback; flash_attn absent; flex disabled", svd_bases_loaded=False,
                    instrumentation="Python return profiling, Lightning callbacks; no loss/optimizer changes",
                    steps=[])
    data = out / "audio"
    data.mkdir()
    # Four deterministic stereo signals satisfy upstream's four-item demo prefetch.
    # Short and long clips exercise padding and random cropping respectively.
    for i, seconds in enumerate([1.7, 2.3, 4.1, 5.2]):
        t = np.arange(int(seconds * 44100), dtype=np.float64) / 44100
        audio = np.stack([0.12 * np.sin(2 * np.pi * (220 + i * 35) * t),
                          0.1 * np.sin(2 * np.pi * (330 + i * 25) * t)], axis=1).astype(np.float32)
        sf.write(data / f"signal-{i}.wav", audio, 44100, subtype="FLOAT")
        (data / f"signal-{i}.txt").write_text("instrumental electronic tones, sustained stereo harmony", encoding="utf-8")

    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {str(k): cpu(v) for k, v in value.items() if isinstance(v, (torch.Tensor, dict, list, tuple, str, int, float, bool, type(None)))}
        if isinstance(value, (list, tuple)):
            return [cpu(v) for v in value]
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        return str(type(value))

    def save(name, value):
        torch.save(cpu(value), out / name)

    capture = {"step": None, "events": []}

    def profile(frame, event, result):
        if event != "return" or capture["step"] is None:
            return
        filename = frame.f_code.co_filename.replace("\\", "/")
        name = frame.f_code.co_name
        selected = ((filename.endswith("/training/diffusion.py") and name == "training_step")
                    or (filename.endswith("/models/dit.py") and name == "forward")
                    or (filename.endswith("/models/pretransforms.py") and name == "encode")
                    or (filename.endswith("/data/utils.py") and name == "__call__" and "offset" in frame.f_locals))
        if selected:
            values = {k: v for k, v in frame.f_locals.items() if k not in {"self", "p"}}
            capture["events"].append({"source": filename.split("/stable_audio_3/")[-1], "function": name,
                                      "locals": cpu(values), "result": cpu(result),
                                      "result_device": str(result.device) if isinstance(result, torch.Tensor) else None})

    class Receipt(official.pl.Callback):
        def on_train_start(self, trainer, module):
            params = {n: p for n, p in module.named_parameters() if p.requires_grad}
            if not params or any("parametrizations" not in n for n in params):
                raise RuntimeError("Unexpected trainable base parameters")
            if any(p.device.type != "cuda" for p in params.values()):
                raise RuntimeError("Adapters must execute on CUDA")
            manifest.update(diffusion_objective=module.diffusion_objective,
                            trainable_parameters={n: list(p.shape) for n, p in params.items()},
                            optimizer=module.optimizer_configs, lora=module.lora_config,
                            pretransform_scale=module.diffusion.pretransform.scale,
                            pretransform_chunked=module.diffusion.pretransform.chunked)
            save("initial-adapters.pt", params)
            torch.cuda.reset_peak_memory_stats()
            capture["step"] = 0
            sys.setprofile(profile)

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            self.started = time.perf_counter()
            self.step = trainer.global_step
            save(f"step-{self.step:02d}-batch.pt", {"batch": batch, "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()})

        def on_before_optimizer_step(self, trainer, module, optimizer):
            gradients = {n: p.grad for n, p in module.named_parameters() if p.requires_grad}
            bad = [n for n, g in gradients.items() if g is None or not torch.isfinite(g).all()]
            if bad:
                raise RuntimeError("Missing/nonfinite gradients: " + repr(bad))
            save(f"step-{self.step:02d}-gradients.pt", gradients)

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            torch.cuda.synchronize()
            loss = float(outputs["loss"])
            if not np.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            save(f"step-{self.step:02d}-events.pt", capture["events"])
            save(f"step-{self.step:02d}-updated.pt", {n: p for n, p in module.named_parameters() if p.requires_grad})
            save(f"step-{self.step:02d}-optimizer.pt", trainer.optimizers[0].state_dict())
            manifest["steps"].append(dict(step=self.step, loss=loss, seconds=time.perf_counter()-self.started,
                                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                          peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            print(f"REFERENCE step={self.step} loss={loss:.9g}", flush=True)
            capture["events"] = []
            capture["step"] = trainer.global_step

    real_trainer = official.pl.Trainer

    def trainer_factory(**kwargs):
        kwargs.update(callbacks=[Receipt()], logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False, accelerator="gpu", devices=1)
        return real_trainer(**kwargs)

    official.pl.Trainer = trainer_factory
    train_args = argparse.Namespace(model="medium-base", data_dir=str(data), encoded_dir=None,
        save_dir=str(out / "lightning"), seed=args.seed, duration=args.duration, batch_size=1,
        num_workers=0, rank=16, lora_alpha=16, adapter_type="dora-rows", dropout=0.0,
        include=None, exclude=None, lora_checkpoint=None, lr=1e-4, svd_bases_path=None,
        log_every=1, base_precision="bf16", logger="none", checkpoint_every=100000,
        demo_every=100000, steps=args.steps, name="sa3-reference", gradient_clip_val=0)
    manifest["arguments"] = vars(train_args).copy()
    try:
        official.train(train_args)
        if len(manifest["steps"]) != args.steps:
            raise RuntimeError("Incomplete step count")
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        timer.cancel()
        sys.setprofile(None)
        official.pl.Trainer = real_trainer
        manifest["files"] = {str(p.relative_to(out)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                             for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("REFERENCE COMPLETE " + str(out / "manifest.json"), flush=True)


if __name__ == "__main__":
    main()
