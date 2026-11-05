#!/usr/bin/env python3
"""Freeze HeartCodec encoder fixtures from installed heartlib, offline, FP32 on CPU.

  python engine/tools/heartcodec-encoder-reference.py --src <checkpoint-root> --audio CLIP --out CASE
  python engine/tools/heartcodec-encoder-reference.py --src <checkpoint-root> --synthetic SECONDS --out CASE

CASE receives input.npy ([channels, samples] float32 at sample_rate.npy) and, for
the first 29.76 s chunk, every stage heartcodec-encode-probe compares: the 24 kHz
chunk with its 240-sample tail, MuEncoder mel and its two hidden states, 16 kHz
audio, WavLM conv features and layer average, Whisper log-mel and output, the four
stereo feature streams, query output and RVQ codes. codes.npy is the whole clip
through HeartCodec.tokenize, [8, frames] int64. The synthetic input is a fixed
seeded mix of tones and noise, so the case is reproducible without any audio file.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def save(out, name, value):
    value = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if value.dtype == np.float64:
        value = value.astype(np.float32)
    np.save(out / f"{name}.npy", np.ascontiguousarray(value))


def synthetic(seconds, rate=24000):
    t = np.arange(int(seconds * rate)) / rate
    rng = np.random.default_rng(1234)
    left = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 1760 * t * (1 + 0.1 * t / seconds))
    right = 0.25 * np.sin(2 * np.pi * 330 * t + 0.5) + 0.05 * rng.standard_normal(t.shape)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t) ** 2
    return np.stack([left * envelope, right]).astype(np.float32), rate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path)
    source.add_argument("--synthetic", type=float)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--f16-from", type=Path,
                        help="GGUF directory; round every weight stored there as F16 before running, "
                             "so the case measures PyTorch's own FP32-to-F16 weight drift")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=False)

    from heartlib.heartcodec.modeling_heartcodec import HeartCodec
    from torch.nn import functional as F
    from torchaudio.functional import resample

    if args.audio:
        import soundfile
        data, rate = soundfile.read(str(args.audio), dtype="float32", always_2d=True)
        waveform = data.T.copy()
    else:
        waveform, rate = synthetic(args.synthetic)
    save(args.out, "input", waveform)
    save(args.out, "sample_rate", np.array([rate], dtype=np.int64))

    codec = HeartCodec.from_encoder_decoder_pretrained(
        str(args.src / "HeartCodec-oss"), str(args.src / "HeartCodec-oss-encoder"), dtype=torch.float32).eval()
    rounded = 0
    if args.f16_from:
        import gguf
        names = set()
        for role in ("muencoder", "wavlm", "whisper", "query"):
            reader = gguf.GGUFReader(str(args.f16_from / f"heartcodec-{role}-F16.gguf"))
            names |= {t.name for t in reader.tensors if t.tensor_type == gguf.GGMLQuantizationType.F16}
        state = codec.state_dict()
        with torch.no_grad():
            for name in names:
                state[name].copy_(state[name].half().float())
                rounded += 1
    encoder, quantizer = codec.encoder, codec.decoder.quantizer
    fe = encoder.feature_extractors

    with torch.inference_mode():
        codes = codec.tokenize(torch.from_numpy(waveform), rate)
        save(args.out, "codes", codes)

        # Chunk 0 exactly as HeartCodecEncoder.tokenize prepares it.
        audio = torch.from_numpy(waveform).float()
        if audio.shape[0] == 1:
            audio = audio.expand(2, -1)
        if rate != 24000:
            audio = resample(audio, rate, 24000)
        save(args.out, "audio_24k", audio)
        chunk = F.pad(audio[:, :714240], (0, 714240 - min(714240, audio.shape[-1])))
        chunk = F.pad(chunk, (0, 240))[None]
        save(args.out, "chunk_24k", chunk[0])

        flat = chunk.flatten(0, 1)
        mu = fe.muencoder
        mel = mu.preprocessor_melspec_2048(flat[..., :flat.shape[-1] // 960 * 960])[..., :-1]
        save(args.out, "mu_mel", (mel - mu.mean) / mu.std)
        acoustic, semantic = mu(flat)
        save(args.out, "mu_acoustic", acoustic)
        save(args.out, "mu_semantic", semantic)

        wav16 = fe.resample_24k_to_16k(flat)
        save(args.out, "wav_16k", wav16)
        save(args.out, "wavlm_conv", fe.wavlm_encoder.feature_extractor(wav16).transpose(1, 2))
        states = fe.wavlm_encoder(wav16, output_hidden_states=True, return_dict=True).hidden_states
        save(args.out, "wavlm_layers", torch.stack(states[6:10], dim=1).mean(dim=1))

        whisper_mel = fe.whisper_processor(wav16.numpy(), sampling_rate=16000, return_tensors="pt",
                                           do_normalize=False).input_features
        save(args.out, "whisper_mel", whisper_mel)
        save(args.out, "whisper_out", fe.whisper_encoder(whisper_mel, return_dict=True).last_hidden_state)

        features = fe(chunk)
        for name, value in zip(("acoustic", "semantic", "whisper", "wavlm"), features):
            save(args.out, f"feature_{name}", value[0])
        query = encoder.query(*features)
        save(args.out, "query", query[0])
        _, indices, _ = quantizer(query)
        save(args.out, "chunk_codes", indices[0].T.contiguous().long())

    meta = dict(rate=rate, channels=int(waveform.shape[0]), samples=int(waveform.shape[1]),
                frames=int(codes.shape[1]), torch=torch.__version__, threads=args.threads, f16_rounded=rounded,
                source="audio" if args.audio else f"synthetic {args.synthetic} s")
    (args.out / "case.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("PASS", json.dumps(meta))


if __name__ == "__main__":
    main()
