#!/usr/bin/env python3
"""Offline HeartMuLa/HeartCodec safetensors to independent GGUF components.

Requires numpy, torch, safetensors and gguf==0.17.1. No model code is imported.
Example:
  python engine/tools/convert-heartmula.py --src <checkpoint-root> --out <gguf-dir>
  python engine/tools/convert-heartmula.py --src <checkpoint-root> --out <gguf-dir> --only encoder

--only encoder writes just what training's Prepare needs: the four HeartCodec
encoder files (muencoder, wavlm, whisper, query) plus heartcodec-rvq-F32.gguf,
which they pin by hash. It needs only HeartCodec-oss and HeartCodec-oss-encoder
under --src, plus tokenizer.json and gen_config.json in the --src root (their
hashes go into the manifest). Copy the four encoder files next to the generation packs; the RVQ
file must be byte-identical to the one already installed, or the pack is refused.

Both 3B variants are mandatory and retain distinct identities. Source names are
kept except audio_head -> audio_head.0..6.weight, each transposed to [8197,3072],
and scalar weight-normalization pairs -> the effective .weight. GGUF reverses
the NumPy shape in its header, as usual; no other implicit transpose occurs.
RVQ, norms, biases, frontend buffers, and integer counters retain their source
precision. Other floating tensors become F16. The manifest describes every
source/output relation and records full source/output hashes, shapes and errors.
Existing outputs are refused. A failed run leaves .partial files for diagnosis.
"""

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open

VERSION = 1
VARIANTS = ("3B", "3B-happy-new-year")
TOTALS = {"3B": 3938113536, "3B-happy-new-year": 3938113536,
          "codec": 1659568725, "encoder": 537134946}
COMPONENTS = ("global", "local", "flow", "rvq", "scalar-decoder",
              "scalar-encoder", "muencoder", "wavlm", "whisper", "query")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_identity(directory):
    """Freeze downloaded revisions and actual payloads, including local changes."""
    directory = Path(directory)
    files = {}
    for path in sorted(directory.glob("*")):
        if path.suffix not in (".json", ".safetensors"):
            continue
        metadata = directory / ".cache/huggingface/download" / (path.name + ".metadata")
        revision = metadata.read_text(encoding="utf-8").splitlines()[0] if metadata.exists() else None
        files[path.name] = dict(bytes=path.stat().st_size, sha256=sha256(path), revision=revision)
    if not files or "config.json" not in files:
        raise ValueError(f"Missing checkpoint/config: {directory}")
    return files


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.tensors = {}
        self.headers = {}
        for file in sorted(self.path.glob("*.safetensors")):
            with file.open("rb") as stream:
                size = int.from_bytes(stream.read(8), "little")
                raw = stream.read(size)
            header = json.loads(raw)
            self.headers[file.name] = hashlib.sha256(raw).hexdigest()
            end = 0
            for name, info in sorted(((n, i) for n, i in header.items() if n != "__metadata__"),
                                     key=lambda item: item[1]["data_offsets"][0]):
                if name in self.tensors:
                    raise ValueError(f"Duplicate tensor: {name}")
                start, stop = info["data_offsets"]
                width = {"F32": 4, "I64": 8}[info["dtype"]]
                if start != end or stop - start != math.prod(info["shape"]) * width:
                    raise ValueError(f"Invalid tensor extent: {name}")
                self.tensors[name] = dict(info, file=file.name)
                end = stop
            if 8 + size + end != file.stat().st_size:
                raise ValueError(f"Invalid source byte count: {file}")
        if not self.tensors:
            raise ValueError(f"No tensors: {path}")
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            expected = json.loads(index.read_text())["weight_map"]
            if expected != {n: i["file"] for n, i in self.tensors.items()}:
                raise ValueError("Shard index differs from actual headers")

    def get(self, name):
        with safe_open(self.path / self.tensors[name]["file"], framework="pt", device="cpu") as source:
            return source.get_tensor(name)


def component(name, kind):
    if kind in VARIANTS:
        if name == "audio_head" or name.startswith(("decoder.", "projection.")):
            return "local"
        if name.startswith(("backbone.", "text_embeddings.", "audio_embeddings.",
                            "codebook0_head.", "muq_linear.", "unconditional_text_embedding.")):
            return "global"
    else:
        prefixes = {"flow_matching.vq_embed.": "rvq", "flow_matching.estimator.": "flow",
                    "flow_matching.cond_feature_emb.": "flow", "flow_matching.zero_cond_embedding1": "flow",
                    "scalar_model.decoder.": "scalar-decoder", "scalar_model.encoder.": "scalar-encoder",
                    "encoder.feature_extractors.muencoder.": "muencoder",
                    "encoder.feature_extractors.wavlm_encoder.": "wavlm",
                    "encoder.feature_extractors.whisper_encoder.": "whisper", "encoder.query.": "query"}
        for prefix, result in prefixes.items():
            if name.startswith(prefix):
                return result
    raise ValueError(f"Unaccounted source tensor: {name}")


def output_groups(checkpoint, names):
    consumed = set()
    for name in sorted(names):
        if name in consumed:
            continue
        value = checkpoint.get(name)
        if name == "audio_head":
            if tuple(value.shape) != (7, 3072, 8197):
                raise ValueError("Unexpected audio_head shape")
            consumed.add(name)
            outputs = [(f"audio_head.{i}.weight", value[i].T.contiguous(), f"slice[{i}],transpose(1,0)")
                       for i in range(7)]
            yield [name], outputs, None
        elif name.startswith("scalar_model.") and name.endswith(".parametrizations.weight.original0"):
            other = name[:-1] + "1"
            if other not in names:
                raise ValueError(f"Missing weight-norm partner: {name}")
            vector = checkpoint.get(other)
            dims = tuple(range(1, vector.ndim))
            weight = vector * (value / torch.linalg.vector_norm(vector, dim=dims, keepdim=True))
            reference = torch._weight_norm(vector, value, 0)
            error = float((weight - reference).abs().max())
            if not torch.allclose(weight, reference, atol=2e-6, rtol=2e-6):
                raise ValueError(f"Weight norm fold failed: {name}: {error}")
            consumed.update((name, other))
            target = name.replace(".parametrizations.weight.original0", ".weight")
            yield [name, other], [(target, weight, "weight_norm(dim=0),folded_once")], error
        else:
            if name.startswith("scalar_model.") and ".parametrizations.weight.original1" in name:
                raise ValueError(f"Unpaired weight norm: {name}")
            consumed.add(name)
            yield [name], [(name, value, "identity")], None
    if consumed != set(names):
        raise ValueError("Source accounting differs")


def storage_dtype(name, tensor, part):
    if not tensor.is_floating_point():
        return np.int64
    # Convolutions/linear matrices are F16; scalar state and frontend math stay F32.
    if part == "rvq" or tensor.ndim < 2 or any(word in name for word in
            ("norm", "running_", "mel", "filter", "window", "alpha", "freq", "position")):
        return np.float32
    return np.float16


def verify(path, rows, metadata):
    reader = gguf.GGUFReader(path)
    for key, value in metadata.items():
        if reader.fields[key].contents() != value:
            raise ValueError(f"Metadata round-trip mismatch: {key}")
    expected = {row["name"]: row for group in rows for row in group["outputs"]}
    if len(reader.tensors) != len(expected):
        raise ValueError("GGUF tensor count differs")
    end = int(reader.data_offset)
    for tensor in reader.tensors:
        row = expected.pop(tensor.name)
        if list(tensor.data.shape) != row["shape"] or tensor.n_bytes != row["bytes"]:
            raise ValueError(f"GGUF shape/bytes differ: {tensor.name}")
        if hashlib.sha256(tensor.data.tobytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"GGUF payload differs: {tensor.name}")
        end = max(end, tensor.data_offset + (tensor.n_bytes + 31) // 32 * 32)
    if expected or end != Path(path).stat().st_size:
        raise ValueError("GGUF file extent differs")


def convert(checkpoint, kind, part, names, identity, dependencies, out):
    stem = f"heartmula-{kind.lower()}-{part}" if kind in VARIANTS else f"heartcodec-{part}"
    path = out / (stem + ("-F32.gguf" if part == "rvq" else "-F16.gguf"))
    partial = path.with_suffix(".gguf.partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path)
    config = (checkpoint.path / "config.json").read_bytes().decode("utf-8")
    fingerprint = hashlib.sha256(json.dumps(
        {name: info["sha256"] for name, info in identity.items() if name.endswith(".safetensors")},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    metadata = {"general.architecture": "heartmula-" + part if kind in VARIANTS else "heartcodec-" + part,
                "general.name": stem, "general.file_type": 0 if part == "rvq" else 1,
                "heartmula.format_version": VERSION, "heartmula.base_fingerprint": fingerprint,
                "heartmula.component": part, "heartmula.variant": kind if kind in VARIANTS else "shared",
                "heartmula.source": json.dumps(identity, sort_keys=True),
                "heartmula.config_json": config, "heartmula.config_sha256": identity["config.json"]["sha256"],
                "heartmula.dependencies": json.dumps(dependencies, sort_keys=True),
                "heartmula.tensor_layout": "numpy-C; GGUF dimensions reversed; transforms in tensor_manifest"}
    writer = gguf.GGUFWriter(partial, metadata["general.architecture"], use_temp_file=True)
    rows = []
    for sources, outputs, fold_error in output_groups(checkpoint, names):
        row = dict(sources={name: checkpoint.tensors[name] for name in sources}, outputs=[])
        if fold_error is not None:
            row["weight_norm_max_abs"] = fold_error
        for name, tensor, transform in outputs:
            array = tensor.detach().numpy()
            dtype = storage_dtype(name, tensor, part)
            encoded = np.ascontiguousarray(array, dtype=dtype)
            if tensor.is_floating_point() and not np.isfinite(encoded).all():
                raise ValueError(f"Nonfinite output: {name}")
            delta = np.abs(array.astype(np.float32) - encoded.astype(np.float32))
            row["outputs"].append(dict(name=name, shape=list(encoded.shape), dtype=str(encoded.dtype),
                                       bytes=encoded.nbytes, sha256=hashlib.sha256(encoded.tobytes()).hexdigest(),
                                       transform=transform, max_abs_error=float(delta.max()),
                                       relative_l2=float(np.linalg.norm(delta.ravel()) /
                                                         max(float(np.linalg.norm(array.ravel())), 1e-30))))
            writer.add_tensor(name, encoded)
        rows.append(row)
    metadata["heartmula.tensor_manifest"] = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    for key, value in metadata.items():
        if key == "general.architecture":
            continue
        if isinstance(value, int):
            writer.add_uint32(key, value)
        else:
            writer.add_string(key, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    verify(partial, rows, metadata)
    partial.rename(path)
    result = dict(file=path.name, bytes=path.stat().st_size, sha256=sha256(path), tensors=rows,
                  source_tensors=len(names), source_elements=sum(math.prod(checkpoint.tensors[n]["shape"]) for n in names))
    print(f"PASS {path.name}: {len(names)} source tensors, {len([t for r in rows for t in r['outputs']])} outputs", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", choices=("encoder",), help="write the HeartCodec encoder pack (and its RVQ) only")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(args.out)
    manifest_path = args.out / "conversion-manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    tokenizer = {n: sha256(args.src / n) for n in ("tokenizer.json", "gen_config.json")}
    manifest = dict(version=VERSION, converter_sha256=sha256(__file__), tokenizer=tokenizer, components=[])
    rvq_hash = None
    # Shared codec first supplies an exact RVQ dependency to every other component.
    kinds = (("codec", "HeartCodec-oss"), ("encoder", "HeartCodec-oss-encoder"),
             *((v, "HeartMuLa-oss-" + v) for v in VARIANTS))
    if args.only == "encoder":
        kinds = kinds[:2]
    for kind, directory in kinds:
        checkpoint = Checkpoint(args.src / directory)
        count = sum(math.prod(t["shape"]) for t in checkpoint.tensors.values())
        if count != TOTALS[kind]:
            raise ValueError(f"Unexpected checkpoint element count: {kind}: {count}")
        identity = source_identity(checkpoint.path)
        groups = {}
        for name in checkpoint.tensors:
            groups.setdefault(component(name, kind), []).append(name)
        for part in sorted(groups, key=lambda name: (name != "rvq", name)):
            if args.only == "encoder" and kind == "codec" and part != "rvq":
                continue
            dependencies = dict(tokenizer=tokenizer, rvq_sha256=rvq_hash)
            result = convert(checkpoint, kind, part, groups[part], identity, dependencies, args.out)
            if part == "rvq":
                rvq_hash = result["sha256"]
            manifest["components"].append(result)
            write_json(args.out / "conversion-progress.json", manifest)
    write_json(manifest_path, manifest)
    print(f"PASS manifest {manifest_path}: {sha256(manifest_path)}", flush=True)


if __name__ == "__main__":
    main()
