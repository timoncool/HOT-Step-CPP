#!/usr/bin/env python3
"""Convert tokenizer.json + gen_config.json to a tensor-free HeartMuLa GGUF.

CPU-only dependencies: gguf, tokenizers. No torch or heartlib import.
  python engine/tools/convert-heartmula-tokenizer.py --src /path/to/ckpt \
      --out /path/to/heartmula-tokenizer.gguf
  python engine/tools/convert-heartmula-tokenizer.py --src /path/to/ckpt \
      --out /path/to/heartmula-tokenizer.gguf --check engine/build/Release/heartmula-probe.exe

Schema v1 stores standard tokenizer.ggml tokens/merges/BOS/EOS plus
heartmula.tokenizer.* Unicode ranges and Python lowercase mappings. Unicode
L/N/space ranges come from HF's own regex engine, avoiding platform/locale
differences. Cased/Case_Ignorable and lower mappings follow this Python's
str.lower(), as used by heartlib. Source hashes and versions identify both.
Common component metadata follows engine/docs/HEARTMULA_CONTRACT.md. For this
tensor-free component, config_json is gen_config.json and base_fingerprint is
the canonical hash of the two source JSON hashes. heartmula.token.{text_bos,
text_eos,audio_eos,empty} are 128000, 128001, 8193, 0. All LM variants share it.
The C++ raw mode mirrors Tokenizer.encode; tags/lyrics modes mirror heartlib
preprocess, including wrapping tags and adding EOS. Explicit special tokens
in input retain HF's behavior, including a second BOS if one is supplied.
--fixtures saves the complete fixed parity corpus for the probe/reviewer.
--check also tests seeded Unicode mixtures and returns nonzero on any mismatch.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import unicodedata
from pathlib import Path

import gguf
import tokenizers
from tokenizers import Regex, Tokenizer, pre_tokenizers

PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
           r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
PREFIX = "heartmula.tokenizer."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ranges_for(predicate):
    result = []
    start = None
    for cp in range(0x110001):
        hit = cp < 0x110000 and not 0xD800 <= cp < 0xE000 and predicate(chr(cp))
        if hit and start is None:
            start = cp
        elif not hit and start is not None:
            result.extend((start, cp - 1))
            start = None
    return result


def unicode_tables():
    # Split offsets count Unicode scalar values, so restore the surrogate gap.
    scalar_text = "".join(chr(cp) for cp in range(0x110000) if not 0xD800 <= cp < 0xE000)
    tables = {}
    for name, pattern in (("letters", r"\p{L}+"), ("numbers", r"\p{N}+"), ("spaces", r"\s+")):
        split = pre_tokenizers.Split(Regex(pattern), "removed", invert=True)
        pairs = []
        for _, (a, b) in split.pre_tokenize_str(scalar_text):
            # No matching property spans the surrogate gap (both sides are Cn/Co).
            pairs.extend((a + (0x800 if a >= 0xD800 else 0),
                          b - 1 + (0x800 if b - 1 >= 0xD800 else 0)))
        tables[name] = pairs
    cased = lambda c: c.islower() or c.isupper() or c.istitle()
    tables["cased"] = ranges_for(cased)

    def case_ignorable(c):
        # Probe Python's Final_Sigma rule to recover its exact derived property,
        # including characters that are BOTH Cased and Case_Ignorable.
        if cased(c):
            return ("A\u03a3" + c).lower()[1] == "\u03c2"
        return ("A\u03a3" + c + "A").lower()[1] == "\u03c3"

    tables["case_ignorable"] = ranges_for(case_ignorable)
    lower = {cp: chr(cp).lower() for cp in range(0x110000)
             if not 0xD800 <= cp < 0xE000 and chr(cp).lower() != chr(cp)}
    return tables, lower


def validate(source, config):
    model = source["model"]
    require(model["type"] == "BPE" and model["ignore_merges"] is True,
            "expected HeartMuLa BPE with ignore_merges=true")
    require(not any(model.get(k) for k in ("dropout", "unk_token", "continuing_subword_prefix",
                                         "end_of_word_suffix", "fuse_unk", "byte_fallback")),
            "unsupported BPE options")
    require(source.get("normalizer") is None and source.get("padding") is None
            and source.get("truncation") is None, "unsupported normalization/padding/truncation")
    require(source["pre_tokenizer"] == {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": PATTERN}, "behavior": "Isolated", "invert": False},
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
        "unsupported pretokenizer")
    post = source["post_processor"]["processors"]
    require(source["post_processor"]["type"] == "Sequence" and len(post) == 2
            and post[0] == {"type": "ByteLevel", "add_prefix_space": True,
                           "trim_offsets": False, "use_regex": True}
            and post[1]["type"] == "TemplateProcessing"
            and post[1]["single"] == [
                {"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}}]
            and post[1]["special_tokens"]["<|begin_of_text|>"]["ids"] == [128000],
            "unsupported single-sequence postprocessor")
    require(config == {"text_bos_id": 128000, "text_eos_id": 128001,
                       "audio_eos_id": 8193, "empty_id": 0}, "unexpected generation config")
    vocab = model["vocab"]
    added = source["added_tokens"]
    require(len(vocab) == 128000 and sorted(vocab.values()) == list(range(128000)),
            "expected contiguous 128000-token base vocabulary")
    require(set(pre_tokenizers.ByteLevel.alphabet()) <= vocab.keys(), "missing base byte token")
    require(len(added) == 256 and sorted(t["id"] for t in added) == list(range(128000, 128256)),
            "expected contiguous 256 added tokens")
    tokens = [None] * 128256
    for text, index in vocab.items():
        tokens[index] = text
    for t in added:
        require(t == dict(id=t["id"], content=t["content"], single_word=False,
                          lstrip=False, rstrip=False, normalized=False, special=True),
                "unsupported added-token flags")
        require(t["content"] not in vocab and "\0" not in t["content"], "invalid added token")
        tokens[t["id"]] = t["content"]
    require(len(set(tokens)) == len(tokens) and all(tokens) and not any("\0" in t for t in tokens),
            "duplicate, empty or NUL-containing token")
    require(tokens[128000] == "<|begin_of_text|>" and tokens[128001] == "<|end_of_text|>",
            "BOS/EOS spellings disagree")
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
    require(len(merges) == 280147 and len(set(merges)) == len(merges), "unexpected merge count")
    for merge in merges:
        parts = merge.split(" ")
        require(len(parts) == 2 and all(p in vocab for p in parts) and "".join(parts) in vocab,
                "merge references a missing base token")
    return tokens, merges


def fixed_cases():
    texts = ["", "Hello WORLD!", "<tag>ROCK, GUITAR, 120 BPM</tag>",
             "[Verse]\nThe lantern lights the quiet street.\n[Chorus]\nWe sing again.",
             "[Verse]\n你好，世界\nこんにちは 世界\n안녕하세요 세상\nHola, corazón\n",
             "Привет мир! Ελληνικά العربية हिन्‍दी ไทย עברית", "İ I ı i STRASSE Straße ẞ",
             "ΟΣ ΟΣΑ Σ Σ\u0301 ΑΣ\u0345 ΑΣ\u0345Α", "é e\u0301 a\u0308\u0323",
             "👩🏽‍🚀🌙🎶\u200d✨", "'S 'T 'RE 'VE 'M 'LL 'D 'soup 'ſ 'ſoup",
             "don’t can’t we’re 'twas 'remainder", "1 12 123 1234 1234567890",
             "١٢٣٤ １２３４ ²Ⅻ½ 𝟙𝟚𝟛𝟜", "a\0b\0", "<tag>jazz</tag>", "<TAG>Jazz</TAG>",
             "<|begin_of_text|>", "<|end_of_text|>", "<|begin_of_text|>x<|end_of_text|>",
             "<|unknown_special|> <|start_header_id|>user<|end_header_id|>",
             "a\u200bb\u200cc\u200dd\u2060e\ufefff", "\u2028\u2029x", "a\r\nb\n\rc\rd\n",
             "  !?\r\n \tword", " \t\r\n  \n\r\t ", "\tword\vword\fword\u0085word\u00a0word"]
    texts += [space * n + tail for space in (" ", "\t", "\r", "\n", "\r\n", "\u00a0", "\u2003")
              for n in (1, 2, 3, 8, 31) for tail in ("", "x", "!?\n")]
    texts += [f"a{chr(cp)}B1234{chr(cp)}!?" for cp in (
        0x345, 0x5B0, 0x60C, 0x903, 0xE50, 0x1680, 0x180E, 0x1C89, 0x2007,
        0x200B, 0x202F, 0x205F, 0x2160, 0x3000, 0x1D400, 0x1F600, 0x20000, 0x10FFFF)]
    return [dict(text=text, mode=mode, add_special_tokens=add)
            for text in texts for mode, add in (("raw", True), ("raw", False),
                                                 ("tags", True), ("lyrics", True))]


def expected_ids(tok, config, case):
    text = case["text"]
    mode = case["mode"]
    if mode != "raw":
        text = text.lower()
        if mode == "tags":
            if not text.startswith("<tag>"):
                text = "<tag>" + text
            if not text.endswith("</tag>"):
                text += "</tag>"
    ids = tok.encode(text, add_special_tokens=case["add_special_tokens"]).ids
    if mode != "raw":
        if not ids or ids[0] != config["text_bos_id"]:
            ids.insert(0, config["text_bos_id"])
        if ids[-1] != config["text_eos_id"]:
            ids.append(config["text_eos_id"])
    return ids


def check(probe, out, tok, config, source, fixtures, prompt_cases, tables, lower):
    cases = fixed_cases()
    cases += [dict(text="x" + t["content"] + "Y", mode="raw", add_special_tokens=True)
              for t in source["added_tokens"]]
    if prompt_cases:
        for row in json.loads(prompt_cases.read_text(encoding="utf-8")):
            for key in ("tags", "lyrics"):
                require(isinstance(row[key], str), "prompt cases require tags and lyrics strings")
                cases.append(dict(text=row[key], mode=key, add_special_tokens=True))
    fixed = len(cases)
    boundaries = set(lower)
    for pairs in tables.values():
        boundaries.update(cp + delta for cp in pairs for delta in (-1, 0, 1)
                          if 0 <= cp + delta < 0x110000 and not 0xD800 <= cp + delta < 0xE000)
    for cp in sorted(boundaries):
        char = chr(cp)
        cases.append(dict(text="a" + char + "B1234" + char + "!?\r\n", mode="raw", add_special_tokens=True))
        cases.append(dict(text="AΣ" + char + "A " + char + "Σ", mode="lyrics", add_special_tokens=True))
    unicode_count = len(cases) - fixed
    if fixtures:
        fixtures.parent.mkdir(parents=True, exist_ok=True)
        fixtures.write_text(json.dumps({"cases": [dict(c, ids=expected_ids(tok, config, c))
                                                   for c in cases]}, ensure_ascii=True, indent=2),
                            encoding="utf-8")
    rng = random.Random(752)
    alphabet = list("aAbB'Sſſ0123456789 \t\r\n!?[]<>/éİΣΑ\u0301\u0345\u00a0\u200b你字🎶")
    for _ in range(1200):
        def random_char():
            if rng.random() < .8:
                return rng.choice(alphabet)
            cp = rng.randrange(0x110000)
            return chr(32 if 0xD800 <= cp < 0xE000 else cp)
        text = "".join(random_char() for _ in range(rng.randrange(1, 90)))
        cases.append(dict(text=text, mode=rng.choice(["raw", "tags", "lyrics"]),
                          add_special_tokens=True))
    result = subprocess.run([str(probe), "--tokenize", "--tokenizer", str(out), "--stdin"],
                            input="".join(json.dumps(c, ensure_ascii=True) + "\n" for c in cases),
                            encoding="utf-8", capture_output=True, timeout=180)
    require(result.returncode == 0, f"probe failed: {result.stderr[-4000:]}")
    rows = result.stdout.splitlines()
    require(len(rows) == len(cases), f"probe returned {len(rows)} rows for {len(cases)} cases")
    mismatches = []
    for i, (case, row) in enumerate(zip(cases, rows)):
        expected = expected_ids(tok, config, case)
        actual = json.loads(row)
        if actual.get("ids") != expected:
            mismatches.append(dict(index=i, case=case, expected=expected, actual=actual))
    for mismatch in mismatches[:10]:
        print(json.dumps(mismatch, ensure_ascii=True))
    print(f"TOKENIZER PARITY: {'FAIL' if mismatches else 'PASS'}; fixed={fixed}; "
          f"unicode={unicode_count}; seeded=1200; total={len(cases)}; mismatches={len(mismatches)}")
    require(not mismatches, "token IDs differ from HF tokenizers")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="directory with tokenizer.json and gen_config.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", type=Path, help="heartmula-probe executable for CPU parity")
    parser.add_argument("--fixtures", type=Path, help="save fixed HF cases as JSON")
    parser.add_argument("--prompt-cases", type=Path, help="also check a fixed JSON array of real tags/lyrics")
    parser.add_argument("--source-revision", default="local-unversioned", help="HF source revision, if known")
    args = parser.parse_args()
    if (args.fixtures or args.prompt_cases) and not args.check:
        parser.error("--fixtures and --prompt-cases require --check")
    paths = [args.src / name for name in ("tokenizer.json", "gen_config.json")]
    source, config = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    tokens, merges = validate(source, config)
    tables, lower = unicode_tables()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    require(args.out.resolve() not in [p.resolve() for p in paths], "output would overwrite source")
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    writer = gguf.GGUFWriter(partial, "heartmula-tokenizer")
    writer.add_name("HeartMuLa tokenizer")
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    writer.add_uint32("heartmula.format_version", 1)
    writer.add_string("heartmula.component", "tokenizer")
    writer.add_string("heartmula.variant", "shared")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    canonical_hashes = json.dumps(hashes, sort_keys=True, separators=(",", ":"))
    writer.add_string("heartmula.base_fingerprint", hashlib.sha256(canonical_hashes.encode()).hexdigest())
    writer.add_string("heartmula.source", json.dumps({p.name: dict(bytes=p.stat().st_size,
                      sha256=hashes[p.name], revision=args.source_revision) for p in paths}, sort_keys=True))
    writer.add_string("heartmula.config_json", paths[1].read_text(encoding="utf-8"))
    writer.add_string("heartmula.config_sha256", hashes["gen_config.json"])
    writer.add_string("heartmula.dependencies", json.dumps(dict(tokenizer=hashes, rvq_sha256=None), sort_keys=True))
    writer.add_string("heartmula.tensor_layout", "no tensors; standard byte-BPE arrays and Unicode tables")
    writer.add_string("heartmula.tensor_manifest", "[]")
    writer.add_string(PREFIX + "source_sha256", canonical_hashes)
    writer.add_uint32(PREFIX + "version", 1)
    writer.add_string(PREFIX + "pattern", PATTERN)
    writer.add_string(PREFIX + "tokenizers_version", tokenizers.__version__)
    writer.add_string(PREFIX + "python_unicode_version", unicodedata.unidata_version)
    for path in paths:
        writer.add_string(PREFIX + path.stem + "_sha256", hashlib.sha256(path.read_bytes()).hexdigest())
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("llama-bpe")
    writer.add_token_list(tokens)
    writer.add_token_types([1] * 128000 + [3] * 256)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(config["text_bos_id"])
    writer.add_eos_token_id(config["text_eos_id"])
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)
    for key, value in config.items():
        writer.add_uint32("heartmula.token." + key.removesuffix("_id"), value)
    for key, pairs in tables.items():
        writer.add_key_value(PREFIX + key, pairs, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32)
    writer.add_key_value(PREFIX + "lower_keys", list(lower), gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32)
    writer.add_array(PREFIX + "lower_values", list(lower.values()))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    os.replace(partial, args.out)
    print(f"Converted {len(tokens)} tokens, {len(merges)} merges, no tensors; {args.out.stat().st_size} bytes")
    if args.check:
        check(args.check.resolve(), args.out.resolve(), Tokenizer.from_file(str(paths[0])), config, source,
              args.fixtures, args.prompt_cases, tables, lower)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
