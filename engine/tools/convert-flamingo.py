#!/usr/bin/env python3
# convert-flamingo.py: Music Flamingo (nvidia/music-flamingo-2601-hf) safetensors -> GGUF
# for the HOT-Step captioning backend (ace-caption --family flamingo).
#
# Produces two files into --out:
#
#   flamingo-lm-<quant>.gguf  arch "qwen2"    -- the 7B Qwen2 LM in llama.cpp naming
#                                              (token_embd, blk.N.attn_q, ... plus the
#                                              q/k/v BIASES Qwen2 has and Qwen3 dropped)
#                                              and the tokenizer.
#   flamingo-aud-f16.gguf     arch "mfl-aud"  -- Whisper-large encoder (AF-Whisper), its
#                                              LEARNED positions, and the 2-layer GELU
#                                              projector into the LM's 3584-d space.
#
# Same two-file split and quantisation policy as convert-moss.py: --quant applies to the
# LM only, the audio file is always f16 with norms/biases f32. It is ~1.4 GB.
#
# Offline tooling only: the app never runs Python. Needs numpy + gguf (llama.cpp's
# gguf-py); BF16 shards are read with the mmap reader borrowed from convert-moss.py.
#
# USAGE
#   python engine/tools/convert-flamingo.py \
#          --src "<hf-cache>/models--nvidia--music-flamingo-2601-hf/snapshots/<hash>" \
#          --out "<data root>/models/flamingo" --quant q8_0

import argparse
import importlib.util
import json
import os
import sys

import numpy  # noqa: F401  (needed by the borrowed reader)


try:
    import gguf
except ImportError:
    sys.exit("need gguf (llama.cpp's gguf-py): pip install gguf")

# Reuse the safetensors reader and the tensor-policy writer rather than copy them.
_spec = importlib.util.spec_from_file_location(
    "convert_moss", os.path.join(os.path.dirname(os.path.abspath(__file__)), "convert-moss.py"))
_cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cm)
Safetensors, Bundle, F32, HALF, QUANT = _cm.Safetensors, _cm.Bundle, _cm.F32, _cm.HALF, _cm.QUANT


def log(m):
    print(m, flush=True)


def die(m):
    sys.exit(f"convert-flamingo: {m}")


AUDIO_TOKEN = 151667   # <sound>
AUDIO_BOS = 151670     # <|sound_bos|>
AUDIO_EOS = 151671     # <|sound_eos|>


def build_tokenizer(b, src_dir, vocab_size):
    """tokenizer.json is the FAST Qwen2 tokenizer. Its pre-tokenizer regex splits digits
    one at a time (\\p{N}), which is what engine/src/bpe.h does, so the plain BPE path
    matches. Merges are stored as [a, b] pairs in newer tokenizer.json files."""
    with open(os.path.join(src_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
        tj = json.load(f)
    vocab = tj["model"]["vocab"]
    added = {a["id"]: a["content"] for a in tj["added_tokens"]}

    reverse = {i: s for s, i in vocab.items()}
    reverse.update(added)
    tokens, toktypes, n_pad = [], [], 0
    for i in range(vocab_size):
        if i in reverse:
            tokens.append(reverse[i])
            toktypes.append(gguf.TokenType.CONTROL if i in added else gguf.TokenType.NORMAL)
        else:
            tokens.append(f"[PAD{i}]")
            toktypes.append(gguf.TokenType.UNUSED)
            n_pad += 1
    merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]

    b.meta("add_tokenizer_model", "gpt2")
    b.meta("add_tokenizer_pre", "qwen2")
    b.meta("add_token_list", tokens)
    b.meta("add_token_types", toktypes)
    b.meta("add_token_merges", merges)
    b.meta("add_bos_token_id", 151643)
    b.meta("add_eos_token_id", 151645)     # <|im_end|>
    b.meta("add_pad_token_id", 151669)
    b.meta("add_add_bos_token", False)
    log(f"tokenizer: {len(tokens)} tokens ({len(vocab)} BPE + {len(added)} added + "
        f"{n_pad} unused), {len(merges)} merges")


def build_lm(st, cfg, src_dir, quant):
    tc = cfg["text_config"]
    n_layer = tc["num_hidden_layers"]
    b = Bundle("qwen2", quant)
    b.meta("add_name", "Music-Flamingo-2601 LM")
    b.meta("add_block_count", n_layer)
    b.meta("add_context_length", tc["max_position_embeddings"])
    b.meta("add_embedding_length", tc["hidden_size"])
    b.meta("add_feed_forward_length", tc["intermediate_size"])
    b.meta("add_head_count", tc["num_attention_heads"])
    b.meta("add_head_count_kv", tc["num_key_value_heads"])
    hd = tc["hidden_size"] // tc["num_attention_heads"]
    b.meta("add_key_length", hd)
    b.meta("add_value_length", hd)
    b.meta("add_layer_norm_rms_eps", tc["rms_norm_eps"])
    b.meta("add_rope_freq_base", tc["rope_parameters"]["rope_theta"])
    b.meta("add_vocab_size", tc["vocab_size"])
    b.meta("add_file_type", gguf.LlamaFileType.ALL_F32)

    p = "language_model.model."
    b.put("token_embd.weight", st.get(p + "embed_tokens.weight"), QUANT)
    b.put("output_norm.weight", st.get(p + "norm.weight"), F32)
    b.put("output.weight", st.get("language_model.lm_head.weight"), QUANT)   # untied
    for i in range(n_layer):
        s, d = f"{p}layers.{i}.", f"blk.{i}."
        b.put(d + "attn_norm.weight", st.get(s + "input_layernorm.weight"), F32)
        b.put(d + "ffn_norm.weight", st.get(s + "post_attention_layernorm.weight"), F32)
        for x in ("q", "k", "v"):
            b.put(d + f"attn_{x}.weight", st.get(s + f"self_attn.{x}_proj.weight"), QUANT)
            b.put(d + f"attn_{x}.bias", st.get(s + f"self_attn.{x}_proj.bias"), F32)
        b.put(d + "attn_output.weight", st.get(s + "self_attn.o_proj.weight"), QUANT)
        b.put(d + "ffn_gate.weight", st.get(s + "mlp.gate_proj.weight"), QUANT)
        b.put(d + "ffn_up.weight", st.get(s + "mlp.up_proj.weight"), QUANT)
        b.put(d + "ffn_down.weight", st.get(s + "mlp.down_proj.weight"), QUANT)
    build_tokenizer(b, src_dir, tc["vocab_size"])
    return b


def build_aud(st, cfg):
    ac, tc = cfg["audio_config"], cfg["text_config"]
    rp = cfg["rope_parameters"]
    n_layer = ac["num_hidden_layers"]
    b = Bundle("mfl-aud", "f16")
    b.meta("add_name", "Music-Flamingo-2601 audio tower")
    u32 = lambda k, v: b.meta("add_uint32", "mfl.audio." + k, int(v))   # noqa: E731
    f32 = lambda k, v: b.meta("add_float32", "mfl.audio." + k, float(v))   # noqa: E731
    u32("block_count", n_layer)
    u32("embedding_length", ac["hidden_size"])
    u32("head_count", ac["num_attention_heads"])
    u32("feed_forward_length", ac["intermediate_size"])
    u32("n_mels", ac["num_mel_bins"])
    u32("max_source_positions", ac["max_source_positions"])
    u32("lm_embedding_length", tc["hidden_size"])
    u32("sample_rate", 16000)
    u32("n_fft", 400)
    u32("hop_length", 160)
    u32("chunk_seconds", 30)
    u32("max_audio_seconds", 1200)
    f32("layer_norm_eps", 1e-5)
    # Rotary time embedding (MusicFlamingoRotaryEmbedding). head_dim here is the FULL
    # encoder width, 1280, so the rotated slice is int(1280 * 0.2) = 256 per axis.
    u32("rote_head_dim", cfg["head_dim"])
    f32("rote_partial", rp["partial_rotary_factor"])
    f32("rote_theta", rp["rope_theta"])
    u32("rote_max_positions", cfg["max_position_embeddings"])
    f32("frame_step", cfg["audio_frame_step"])
    u32("audio_token_id", cfg["audio_token_id"])
    u32("audio_bos_id", cfg["audio_bos_token_id"])
    u32("audio_eos_id", cfg["audio_eos_token_id"])
    b.meta("add_file_type", gguf.LlamaFileType.ALL_F32)

    a = "audio_tower."
    # Conv1d weight [OC, IC, K]: GGUF reverses dims, so ggml sees [K, IC, OC], which is
    # the kernel layout ggml_im2col wants. Nothing to transpose.
    for k in ("conv1", "conv2"):
        b.put(f"aud.{k}.weight", st.get(a + f"{k}.weight"), HALF)
        b.put(f"aud.{k}.bias", st.get(a + f"{k}.bias"), F32)
    # Learned (unlike MOSS's recomputed sinusoids): keep it exact.
    b.put("aud.pos.weight", st.get(a + "embed_positions.weight"), F32)
    b.put("aud.norm.weight", st.get(a + "layer_norm.weight"), F32)
    b.put("aud.norm.bias", st.get(a + "layer_norm.bias"), F32)
    for i in range(n_layer):
        s, d = f"{a}layers.{i}.", f"aud.blk.{i}."
        b.put(d + "attn_norm.weight", st.get(s + "self_attn_layer_norm.weight"), F32)
        b.put(d + "attn_norm.bias", st.get(s + "self_attn_layer_norm.bias"), F32)
        b.put(d + "attn_q.weight", st.get(s + "self_attn.q_proj.weight"), HALF)
        b.put(d + "attn_q.bias", st.get(s + "self_attn.q_proj.bias"), F32)
        b.put(d + "attn_k.weight", st.get(s + "self_attn.k_proj.weight"), HALF)   # no bias
        b.put(d + "attn_v.weight", st.get(s + "self_attn.v_proj.weight"), HALF)
        b.put(d + "attn_v.bias", st.get(s + "self_attn.v_proj.bias"), F32)
        b.put(d + "attn_out.weight", st.get(s + "self_attn.out_proj.weight"), HALF)
        b.put(d + "attn_out.bias", st.get(s + "self_attn.out_proj.bias"), F32)
        b.put(d + "ffn_norm.weight", st.get(s + "final_layer_norm.weight"), F32)
        b.put(d + "ffn_norm.bias", st.get(s + "final_layer_norm.bias"), F32)
        b.put(d + "ffn_up.weight", st.get(s + "fc1.weight"), HALF)
        b.put(d + "ffn_up.bias", st.get(s + "fc1.bias"), F32)
        b.put(d + "ffn_down.weight", st.get(s + "fc2.weight"), HALF)
        b.put(d + "ffn_down.bias", st.get(s + "fc2.bias"), F32)
    for k in ("linear_1", "linear_2"):
        b.put(f"aud.proj.{k}.weight", st.get(f"multi_modal_projector.{k}.weight"), HALF)
        b.put(f"aud.proj.{k}.bias", st.get(f"multi_modal_projector.{k}.bias"), F32)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF snapshot dir for music-flamingo-2601-hf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default="q8_0", choices=("f16", "q8_0"),
                    help="LM only; the audio tower is always f16")
    ap.add_argument("--components", default="lm,aud")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    with open(os.path.join(src, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("model_type") != "musicflamingo":
        die(f"{src}: model_type is {cfg.get('model_type')!r}, expected 'musicflamingo'")
    shards = sorted(p for p in os.listdir(src) if p.endswith(".safetensors"))
    st = Safetensors([os.path.join(src, p) for p in shards])
    log(f"source: {len(shards)} shard(s), {len(st.names())} tensors")

    os.makedirs(args.out, exist_ok=True)
    want = {c.strip() for c in args.components.split(",") if c.strip()}
    consumed = set()
    if "lm" in want:
        log("building LM ...")
        b = build_lm(st, cfg, src, args.quant)
        consumed |= {n for n in st.names() if n.startswith("language_model.")}
        b.write(os.path.join(args.out, f"flamingo-lm-{args.quant}.gguf"))
    if "aud" in want:
        log("building audio tower (always f16) ...")
        b = build_aud(st, cfg)
        consumed |= {n for n in st.names()
                     if n.startswith(("audio_tower.", "multi_modal_projector."))}
        b.write(os.path.join(args.out, "flamingo-aud-f16.gguf"))
    if want >= {"lm", "aud"}:
        leftover = st.names() - consumed
        if leftover:
            die(f"{len(leftover)} source tensors not consumed, e.g. {sorted(leftover)[:5]}")
        log("all source tensors accounted for")
    st.close()
    log("done")


if __name__ == "__main__":
    main()
