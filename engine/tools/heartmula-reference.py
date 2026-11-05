#!/usr/bin/env python3
"""Freeze installed heartlib fixed-input LM and codec parity fixtures, offline.

  python engine/tools/heartmula-reference.py --src <checkpoint-root> --out <fixtures> --device cuda

Uses heartlib's real generate_frame with sample_topk replaced by fixed teacher
codes, as in the cached teacher-forcing check in HOTSTEP-745. Raw conditioned,
unconditioned and CFG logits are saved for c0..c7, prefill and three incremental
frames. A separate sparse-position probe at 4096 checks long-position RoPE
without claiming to represent a full long-context prompt. Full-sequence logits
also establish cached-versus-full drift. Both checkpoint variants run FP32 and
F16. Codec fixtures drive real detokenize, record every supplied random tensor,
RVQ output, flow input/velocity, solved latent, scalar input/output and assembled
waveform. Short and multi-chunk cases use the default 29.76 s chunk geometry.
Codec F16 baseline rounds eligible weights to F16 but computes in FP32, keeping
the converter's protected buffers/state. Execution precision is explicit.

Requires the installed heartlib and conversion dependencies. No network access,
trainer import or runtime Python integration. Output directories are immutable;
use a new --out for a new reference. --sections lm / codec can produce separate
manifests to resume independent sections after a failure.
"""

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import shutil
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("heartmula_converter", Path(__file__).with_name("convert-heartmula.py"))
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class Fixtures:
    def __init__(self, path):
        self.path = path
        self.arrays = {}
        self.metrics = {}

    def save(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
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

    def compare(self, label, ref_name, candidate_name, targets=None, selection=None):
        ref = np.load(self.path / self.arrays[ref_name]["file"]).astype(np.float64)
        candidate = np.load(self.path / self.arrays[candidate_name]["file"]).astype(np.float64)
        if selection is not None:
            ref, candidate = ref[selection], candidate[selection]
        delta = candidate - ref
        result = dict(max_abs=float(np.abs(delta).max()),
                      relative_l2=float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(ref.ravel()), 1e-30)))
        if targets is not None:
            result["argmax_agreement"] = float((ref.argmax(-1) == candidate.argmax(-1)).mean())
            target = torch.as_tensor(targets, dtype=torch.long).reshape(-1)
            for key, logits in (("reference_ce", ref), ("candidate_ce", candidate)):
                result[key] = float(torch.nn.functional.cross_entropy(
                    torch.from_numpy(logits).reshape(-1, logits.shape[-1]), target))
        self.metrics[label] = result


def freeze_sources(src, out):
    import heartlib
    root = Path(heartlib.__file__).parent
    source_files = {p.relative_to(root).as_posix(): converter.sha256(p) for p in sorted(root.rglob("*.py"))}
    versions = {}
    for package in ("torch", "torchtune", "torchao", "transformers", "tokenizers", "vector-quantize-pytorch",
                    "heartlib", "numpy", "gguf", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unpackaged"
    # Attention, cache and RoPE implementations also affect LM numerical parity.
    import torchtune
    tune_root = Path(torchtune.__file__).parent
    tune_hashes = {p.relative_to(tune_root).as_posix(): converter.sha256(p)
                   for sub in ("modules", "models/llama3_2") for p in sorted((tune_root / sub).rglob("*.py"))}
    configs = {}
    for directory in ("HeartMuLa-oss-3B", "HeartMuLa-oss-3B-happy-new-year", "HeartCodec-oss", "HeartCodec-oss-encoder"):
        for path in sorted((src / directory).glob("*.json")):
            dest = out / "configs" / directory / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            configs[dest.relative_to(out).as_posix()] = converter.sha256(dest)
    for filename in ("gen_config.json", "tokenizer.json"):
        shutil.copyfile(src / filename, out / filename)
        configs[filename] = converter.sha256(out / filename)
    checkpoints = {name: converter.source_identity(src / name) for name in
                   ("HeartMuLa-oss-3B", "HeartMuLa-oss-3B-happy-new-year", "HeartCodec-oss", "HeartCodec-oss-encoder")}
    return dict(versions=versions, heartlib_root=str(root), heartlib_sources=source_files,
                torchtune_sources=tune_hashes, configs=configs, checkpoints=checkpoints,
                scripts={Path(__file__).name: converter.sha256(__file__),
                         "convert-heartmula.py": converter.sha256(converter.__file__)})


def make_prompt(src, device, dtype):
    from heartlib.pipelines.music_generation import HeartMuLaGenPipeline, HeartMuLaGenConfig
    from tokenizers import Tokenizer
    pipe = HeartMuLaGenPipeline(
        str(src / "HeartMuLa-oss-3B"), str(src / "HeartCodec-oss"), torch.device(device), torch.device(device),
        dtype, torch.float32, True, None, Tokenizer.from_file(str(src / "tokenizer.json")),
        HeartMuLaGenConfig.from_file(str(src / "gen_config.json")))
    prompt = pipe.preprocess(dict(tags="Piano, ambient", lyrics="[verse]\nA quiet morning\nA distant light"), cfg_scale=1.5)
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in prompt.items()}


def full_logits(model, prompt, codes, device, dtype):
    from torchtune.modules.common_utils import disable_kv_cache
    tokens = prompt["tokens"]
    batch, length, _ = tokens.shape
    audio = torch.zeros(batch, len(codes) - 1, 9, device=device, dtype=torch.long)
    audio[:, :, :8] = codes[:-1]
    tokens = torch.cat((tokens, audio), 1)
    masks = torch.cat((prompt["tokens_mask"], torch.ones_like(audio, dtype=torch.bool)), 1)
    masks[:, length:, 8] = False
    uncond = torch.tensor([False, True], device=device)
    h = (model._embed_tokens(tokens, uncond_mask=uncond) * masks.unsqueeze(-1)).sum(2, dtype=dtype)
    h[0, prompt["muq_idx"][0]] = model.muq_linear(prompt["muq_embed"][:1])[0]
    h[1, prompt["muq_idx"][1]] = model.unconditional_text_embedding.weight[0]
    causal = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=device).tril().unsqueeze(0)
    with disable_kv_cache(model.backbone), disable_kv_cache(model.decoder):
        output = model.backbone(h, input_pos=torch.arange(tokens.shape[1], device=device).expand(batch, -1), mask=causal)
        last = output[:, length - 1:].to(dtype)
        logits = [model.codebook0_head(last)]
        for frame in range(len(codes)):
            local = torch.cat((last[:, frame:frame+1], torch.stack(
                [model._embed_audio(k, codes[frame, k].expand(batch)) for k in range(7)], 1)), 1)
            local_h = model.decoder(model.projection(local), input_pos=torch.arange(8, device=device).expand(batch, -1),
                                    mask=torch.ones(8, 8, dtype=torch.bool, device=device).tril().expand(batch, -1, -1))
            if frame == 0:
                locals_by_book = [[] for _ in range(7)]
            for k in range(7):
                locals_by_book[k].append(torch.mm(local_h[:, k + 1].to(dtype), model.audio_head[k]))
        logits.extend(torch.stack(values, 1) for values in locals_by_book)
    return torch.stack(logits, 2)


@torch.inference_mode()
def lm_fixtures(args, fixtures):
    from heartlib.heartmula import modeling_heartmula as module
    codes = (torch.arange(4 * 8).reshape(4, 8) * 137 + 17) % 8192
    for variant in converter.VARIANTS:
        for dtype_name, dtype in (("fp32", torch.float32), ("f16", torch.float16)):
            prefix = f"lm/{variant}/{dtype_name}"
            print("START " + prefix, flush=True)
            model = module.HeartMuLa.from_pretrained(args.src / ("HeartMuLa-oss-" + variant),
                                                    dtype=dtype, device_map=args.device, local_files_only=True).eval()
            model.setup_caches(2)
            prompt = make_prompt(args.src, args.device, dtype)
            for name, value in prompt.items():
                fixtures.save(prefix + "/prompt/" + name, value)
            fixtures.save(prefix + "/teacher_codes", codes)
            raw_c0 = []
            decoder_h = []
            handles = [model.codebook0_head.register_forward_hook(lambda m, i, o: raw_c0.append(o.detach())),
                       model.decoder.register_forward_hook(lambda m, i, o: decoder_h.append(o[:, -1].detach()))]
            all_logits = []
            book = 0
            frame = 0
            def teacher(logits, topk, temperature):
                nonlocal book
                raw = raw_c0[-1] if book == 0 else torch.mm(decoder_h[-1], model.audio_head[book - 1])
                # Matmul stays inside the same inference autocast as generate_frame.
                all_logits.append(raw.detach().float().cpu())
                fixtures.save(prefix + f"/frame{frame}/c{book}/raw", raw.float())
                fixtures.save(prefix + f"/frame{frame}/c{book}/guided", logits.float())
                result = codes[min(frame, 3), book].expand(logits.shape[0], 1).to(device=args.device, dtype=torch.int)
                book += 1
                return result
            autocast = lambda: torch.autocast(torch.device(args.device).type, dtype=dtype) if dtype != torch.float32 else nullcontext()
            try:
                with patch.object(module, "sample_topk", teacher):
                    for frame in range(5):
                        book = 0
                        if frame == 0:
                            tokens, masks, pos = prompt["tokens"], prompt["tokens_mask"], prompt["pos"]
                        else:
                            tokens = torch.zeros(2, 1, 9, device=args.device, dtype=torch.long)
                            tokens[:, 0, :8] = codes[min(frame - 1, 3)].to(args.device)
                            masks = torch.ones_like(tokens, dtype=torch.bool)
                            masks[:, :, 8] = False
                            position = 4096 if frame == 4 else prompt["pos"].shape[1] + frame - 1
                            pos = torch.full((2, 1), position, device=args.device, dtype=torch.long)
                            if frame == 4:
                                # Empty cache slots must not become visible in this sparse RoPE probe.
                                model.backbone_causal_mask[4096, prompt["pos"].shape[1] + 3:4096] = False
                                for layer in model.backbone.layers:
                                    cache = layer.attn.kv_cache
                                    cache.cache_pos += 4096 - cache.size
                        fixtures.save(prefix + f"/frame{frame}/input_tokens", tokens)
                        fixtures.save(prefix + f"/frame{frame}/input_mask", masks)
                        fixtures.save(prefix + f"/frame{frame}/input_pos", pos)
                        with autocast():
                            model.generate_frame(tokens, masks, pos, 1.0, 50, 1.5,
                                                 continuous_segments=prompt["muq_embed"] if frame == 0 else None,
                                                 starts=prompt["muq_idx"] if frame == 0 else None)
                        if book != 8:
                            raise ValueError("Teacher sampler did not see all eight codebooks")
            finally:
                for handle in handles:
                    handle.remove()
            with autocast():
                full = full_logits(model, prompt, codes.to(args.device), args.device, dtype)
            cached = torch.stack(all_logits[:32]).reshape(4, 8, 2, 8197).permute(2, 0, 1, 3)
            fixtures.save(prefix + "/cached_logits", cached)
            fixtures.save(prefix + "/full_logits", full.float())
            targets = codes.unsqueeze(0).expand(2, -1, -1).numpy()
            fixtures.compare(prefix + "/cached_vs_full", prefix + "/full_logits", prefix + "/cached_logits", targets)
            for k in range(8):
                fixtures.compare(prefix + f"/cached_vs_full/c{k}", prefix + "/full_logits", prefix + "/cached_logits",
                                 targets[:, :, k], (slice(None), slice(None), k))
            if dtype_name == "f16":
                fixtures.compare(f"lm/{variant}/fp32_vs_f16", f"lm/{variant}/fp32/cached_logits", prefix + "/cached_logits", targets)
                for f in range(5):
                    for k in range(8):
                        fixtures.compare(f"lm/{variant}/fp32_vs_f16/frame{f}/c{k}",
                                         f"lm/{variant}/fp32/frame{f}/c{k}/raw", prefix + f"/frame{f}/c{k}/raw",
                                         [int(codes[min(f, 3), k])] * 2)
            del model, prompt, full, cached, all_logits, raw_c0, decoder_h
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("PASS " + prefix, flush=True)


@torch.inference_mode()
def codec_fixtures(args, fixtures):
    from heartlib.heartcodec.modeling_heartcodec import HeartCodec
    for precision in ("fp32", "f16-weights-fp32-compute"):
        print("START codec/" + precision, flush=True)
        model = HeartCodec.from_pretrained(args.src / "HeartCodec-oss", dtype=torch.float32,
                                           device_map=args.device, local_files_only=True).eval()
        if precision != "fp32":
            # The scalar converter folds first, then rounds the effective weight.
            from torch.nn.utils.parametrize import remove_parametrizations
            for scalar_module in model.scalar_model.modules():
                if hasattr(scalar_module, "parametrizations") and "weight" in scalar_module.parametrizations:
                    remove_parametrizations(scalar_module, "weight", leave_parametrized=True)
            for name, value in list(model.named_parameters()) + list(model.named_buffers()):
                original = name.removeprefix("decoder.")
                part = converter.component(original, "codec")
                if converter.storage_dtype(original, value, part) == np.float16:
                    value.copy_(value.half().float())
        for case, length in (("short", 8), ("multi-chunk", 400)):
            prefix = f"codec/{precision}/{case}"
            codes = (torch.arange(8 * length).reshape(8, length) * 97 + 11) % 8192
            fixtures.save(prefix + "/codes", codes)
            counters = {}
            def capture(name, value):
                index = counters.get(name, 0)
                counters[name] = index + 1
                fixtures.save(prefix + f"/{name}/{index}", value)
            flow = model.flow_matching
            handles = []
            # get_output_from_indices is not module.forward, so intercept the method itself.
            get_rvq = flow.vq_embed.get_output_from_indices
            def rvq(indices, *a, **kw):
                result = get_rvq(indices, *a, **kw)
                capture("rvq_indices", indices)
                capture("rvq_vectors", result)
                return result
            def velocity(m, inputs, kwargs, output):
                capture("flow_input", inputs[0])
                capture("flow_timestep", kwargs["timestep"])
                capture("flow_velocity", output)
            handles.append(flow.estimator.register_forward_hook(velocity, with_kwargs=True))
            inference_codes = flow.inference_codes
            def solve(*a, **kw):
                capture("true_latents", a[1])
                capture("incontext_length", np.asarray(a[3]))
                result = inference_codes(*a, **kw)
                capture("latent", result)
                return result
            decode = model.scalar_model.decode
            def scalar(value):
                capture("scalar_input", value)
                result = decode(value)
                capture("scalar_output", result)
                return result
            generator = torch.Generator(device="cpu").manual_seed(751)
            randn = torch.randn
            def supplied_noise(*shape, **kwargs):
                # Draw on CPU once and supply the exact saved values on either backend.
                target_device = kwargs.pop("device", "cpu")
                kwargs.pop("generator", None)
                value = randn(*shape, generator=generator, **kwargs)
                capture("noise", value)
                return value.to(target_device)
            try:
                with patch.object(flow.vq_embed, "get_output_from_indices", rvq), \
                     patch.object(flow, "inference_codes", solve), \
                     patch.object(model.scalar_model, "decode", scalar), patch.object(torch, "randn", supplied_noise):
                    waveform = model.detokenize(codes, num_steps=10, disable_progress=True)
                fixtures.save(prefix + "/waveform", waveform)
                if list(waveform.shape) != [2, length * 3840]:
                    raise ValueError("Unexpected codec frame count")
                if precision != "fp32":
                    fixtures.compare(prefix + "/weight_drift", f"codec/fp32/{case}/waveform", prefix + "/waveform")
                    for stage in ("rvq_vectors", "flow_velocity", "latent", "scalar_output"):
                        for index in range(counters[stage]):
                            name = f"/{stage}/{index}"
                            fixtures.compare(prefix + name, f"codec/fp32/{case}" + name, prefix + name)
            finally:
                for handle in handles:
                    handle.remove()
            print("PASS " + prefix, flush=True)
        del model, flow, handles, get_rvq, inference_codes, decode
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sections", choices=("all", "lm", "codec"), default="all")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(751)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.90)
    fixtures = Fixtures(args.out)
    manifest = dict(version=1, device=args.device, sources=freeze_sources(args.src, args.out),
                    lm=dict(cfg_scale=1.5, frames=4, long_position=4096, long_context="sparse_position_probe"),
                    codec=dict(chunk_seconds=29.76, steps=10, guidance_scale=1.25, sample_rate=48000,
                               f16_baseline="eligible source weights rounded; FP32 computation"))
    converter.write_json(args.out / "source-manifest.json", manifest)
    try:
        if args.sections in ("all", "lm"):
            lm_fixtures(args, fixtures)
        if args.sections in ("all", "codec"):
            codec_fixtures(args, fixtures)
    finally:
        manifest.update(arrays=fixtures.arrays, metrics=fixtures.metrics)
        converter.write_json(args.out / "fixture-progress.json", manifest)
    manifest["complete"] = args.sections
    converter.write_json(args.out / "fixture-manifest.json", manifest)
    print(f"PASS {len(fixtures.arrays)} arrays; manifest SHA256 {converter.sha256(args.out / 'fixture-manifest.json')}", flush=True)


if __name__ == "__main__":
    main()
