#pragma once

// HeartMuLa schema v1: byte-BPE storage is shared with bpe.h. Unicode splitting,
// ignore_merges, added tokens and heartlib preprocessing belong to this backend.
// No locale, Python runtime, GPU allocation or global tokenizer state.
#include "bpe.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

struct HeartMuLaTokenizer {
    BPETokenizer bpe{};
    std::vector<std::pair<std::string, int32_t>> specials;
    std::vector<uint32_t> letters, numbers, spaces, cased, case_ignorable;
    std::unordered_map<uint32_t, std::string> lower;
    int32_t bos_id = 128000, eos_id = 128001, audio_eos_id = 8193, empty_id = 0;
    bool loaded = false;
};

namespace heartmula_tokenizer_detail {

inline bool contains(const std::vector<uint32_t> & ranges, uint32_t cp) {
    auto it = std::upper_bound(ranges.begin(), ranges.end(), cp);
    // Inclusive [start,end] pairs; upper_bound skips an end equal to cp.
    return ((it - ranges.begin()) & 1) || (it != ranges.begin() && *(it - 1) == cp);
}

struct Character { uint32_t cp; size_t begin, end; };

inline std::vector<Character> decode(const std::string & text) {
    std::vector<Character> chars;
    for (size_t i = 0; i < text.size();) {
        const size_t begin = i;
        const uint32_t lead = (unsigned char) text[i++];
        uint32_t cp = lead;
        size_t extra = 0;
        if (lead >= 0xc2 && lead <= 0xdf) { cp &= 0x1f; extra = 1; }
        else if (lead >= 0xe0 && lead <= 0xef) { cp &= 0x0f; extra = 2; }
        else if (lead >= 0xf0 && lead <= 0xf4) { cp &= 7; extra = 3; }
        else if (lead >= 0x80) { throw std::runtime_error("invalid UTF-8 lead byte"); }
        if (i + extra > text.size()) { throw std::runtime_error("truncated UTF-8"); }
        for (size_t n = 0; n < extra; ++n) {
            const uint32_t next = (unsigned char) text[i++];
            if ((next & 0xc0) != 0x80) { throw std::runtime_error("invalid UTF-8 continuation"); }
            cp = (cp << 6) | (next & 0x3f);
        }
        if ((extra == 1 && cp < 0x80) || (extra == 2 && cp < 0x800) ||
            (extra == 3 && cp < 0x10000) || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) {
            throw std::runtime_error("invalid UTF-8 scalar value");
        }
        chars.push_back({ cp, begin, i });
    }
    return chars;
}

inline std::string lowercase(const HeartMuLaTokenizer & t, const std::string & text) {
    const auto chars = decode(text);
    std::string result;
    bool preceding_cased = false;
    for (size_t i = 0; i < chars.size(); ++i) {
        const uint32_t cp = chars[i].cp;
        bool final_sigma = false;
        if (cp == 0x3a3 && preceding_cased) {
            size_t next = i + 1;
            while (next < chars.size() && contains(t.case_ignorable, chars[next].cp)) { ++next; }
            final_sigma = next == chars.size() || !contains(t.cased, chars[next].cp);
        }
        const auto lower = t.lower.find(cp);
        if (final_sigma) { result += "\xcf\x82"; }
        else if (lower != t.lower.end()) { result += lower->second; }
        else { result.append(text, chars[i].begin, chars[i].end - chars[i].begin); }
        if (!contains(t.case_ignorable, cp)) { preceding_cased = contains(t.cased, cp); }
    }
    return result;
}

inline std::vector<std::string> split(const HeartMuLaTokenizer & t, const std::string & text) {
    const auto c = decode(text);
    const size_t n = c.size();
    const auto letter = [&](size_t i) { return i < n && contains(t.letters, c[i].cp); };
    const auto number = [&](size_t i) { return i < n && contains(t.numbers, c[i].cp); };
    const auto space = [&](size_t i) { return i < n && contains(t.spaces, c[i].cp); };
    const auto newline = [&](size_t i) { return i < n && (c[i].cp == '\r' || c[i].cp == '\n'); };
    const auto punct = [&](size_t i) { return i < n && !space(i) && !letter(i) && !number(i); };
    std::vector<std::string> chunks;
    for (size_t i = 0; i < n;) {
        size_t end = i;
        // (?i:'s|'t|'re|'ve|'m|'ll|'d). No word-boundary assertion.
        if (c[i].cp == '\'') {
            for (const char * suffix : { "s", "t", "re", "ve", "m", "ll", "d" }) {
                size_t j = i + 1, k = 0;
                while (suffix[k] && j < n) {
                    uint32_t cp = c[j].cp;
                    if (cp >= 'A' && cp <= 'Z') { cp += 'a' - 'A'; }
                    if (cp == 0x17f) { cp = 's'; } // Unicode case-fold of long s.
                    if (cp != (uint32_t) suffix[k]) { break; }
                    ++j; ++k;
                }
                if (!suffix[k]) { end = j; break; }
            }
        }
        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (end == i) {
            size_t j = i;
            if (!newline(j) && !letter(j) && !number(j) && letter(j + 1)) { ++j; }
            if (letter(j)) { while (letter(j)) { ++j; } end = j; }
        }
        if (end == i && number(i)) {
            end = i;
            do { ++end; } while (end < i + 3 && number(end));
        }
        // A literal ASCII space is the optional prefix of punctuation.
        if (end == i) {
            size_t j = i + (c[i].cp == ' ' ? 1 : 0);
            if (punct(j)) {
                while (punct(j)) { ++j; }
                while (newline(j)) { ++j; }
                end = j;
            }
        }
        if (end == i && space(i)) {
            size_t j = i, last_newline = i;
            while (space(j)) { if (newline(j)) { last_newline = j + 1; } ++j; }
            if (last_newline > i) { end = last_newline; } // \s*[\r\n]+
            else if (j == n || j == i + 1) { end = j; }
            else { end = j - 1; } // \s+(?!\S), then \s+
        }
        if (end == i) { throw std::runtime_error("pretokenizer failed to consume a scalar"); }
        chunks.push_back(text.substr(c[i].begin, c[end - 1].end - c[i].begin));
        i = end;
    }
    return chunks;
}

inline std::vector<uint32_t> read_u32_array(gguf_context * ctx, const char * key) {
    const int64_t index = gguf_find_key(ctx, key);
    if (index < 0 || gguf_get_kv_type(ctx, index) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(ctx, index) != GGUF_TYPE_UINT32) {
        throw std::runtime_error(std::string("missing/invalid UINT32 array: ") + key);
    }
    const auto count = gguf_get_arr_n(ctx, index);
    const auto * data = (const uint32_t *) gguf_get_arr_data(ctx, index);
    return std::vector<uint32_t>(data, data + count);
}

inline void read_ranges(gguf_context * ctx, const char * key, std::vector<uint32_t> & out) {
    out = read_u32_array(ctx, key);
    if (out.empty() || out.size() % 2) { throw std::runtime_error("invalid Unicode ranges"); }
    for (size_t i = 0; i < out.size(); i += 2) {
        if (out[i] > out[i + 1] || out[i + 1] > 0x10ffff || (i && out[i] <= out[i - 1])) {
            throw std::runtime_error("unsorted Unicode ranges");
        }
    }
}

inline uint32_t read_u32(gguf_context * ctx, const char * key) {
    const int64_t index = gguf_find_key(ctx, key);
    if (index < 0 || gguf_get_kv_type(ctx, index) != GGUF_TYPE_UINT32) {
        throw std::runtime_error(std::string("missing/invalid UINT32: ") + key);
    }
    return gguf_get_val_u32(ctx, index);
}

} // namespace heartmula_tokenizer_detail

inline bool heartmula_tokenizer_load(HeartMuLaTokenizer * out, const char * path, std::string * error) {
    using namespace heartmula_tokenizer_detail;
    gguf_context * ctx = gguf_init_from_file(path, { true, nullptr });
    if (!ctx) { if (error) { *error = "cannot open tokenizer GGUF"; } return false; }
    try {
        HeartMuLaTokenizer t;
        const int64_t arch = gguf_find_key(ctx, "general.architecture");
        if (arch < 0 || gguf_get_kv_type(ctx, arch) != GGUF_TYPE_STRING ||
            std::string(gguf_get_val_str(ctx, arch)) != "heartmula-tokenizer" ||
            gguf_get_n_tensors(ctx) != 0 || read_u32(ctx, "heartmula.tokenizer.version") != 1) {
            throw std::runtime_error("expected tensor-free HeartMuLa tokenizer schema v1");
        }
        if (read_u32(ctx, "heartmula.format_version") != 1) {
            throw std::runtime_error("unsupported HeartMuLa format version");
        }
        t.bos_id = (int32_t) read_u32(ctx, "heartmula.token.text_bos");
        t.eos_id = (int32_t) read_u32(ctx, "heartmula.token.text_eos");
        t.audio_eos_id = (int32_t) read_u32(ctx, "heartmula.token.audio_eos");
        t.empty_id = (int32_t) read_u32(ctx, "heartmula.token.empty");
        if (t.bos_id != 128000 || t.eos_id != 128001 || t.audio_eos_id != 8193 || t.empty_id != 0 ||
            read_u32(ctx, "tokenizer.ggml.bos_token_id") != (uint32_t) t.bos_id ||
            read_u32(ctx, "tokenizer.ggml.eos_token_id") != (uint32_t) t.eos_id) {
            throw std::runtime_error("incompatible HeartMuLa special IDs");
        }
        const auto array = [&](const char * name, size_t count) {
            const int64_t key = gguf_find_key(ctx, name);
            if (key < 0 || gguf_get_kv_type(ctx, key) != GGUF_TYPE_ARRAY ||
                gguf_get_arr_type(ctx, key) != GGUF_TYPE_STRING || gguf_get_arr_n(ctx, key) != count) {
                throw std::runtime_error(std::string("missing/invalid string array: ") + name);
            }
            return key;
        };
        const auto tokens = array("tokenizer.ggml.tokens", 128256);
        const auto merges = array("tokenizer.ggml.merges", 280147);
        t.bpe.id_to_str.reserve(128256);
        for (size_t id = 0; id < 128256; ++id) {
            std::string text = gguf_get_arr_str(ctx, tokens, id);
            if (text.empty() || !t.bpe.vocab.emplace(text, (int) id).second) {
                throw std::runtime_error("empty/duplicate vocabulary entry");
            }
            t.bpe.id_to_str.push_back(text);
            if (id >= 128000) { t.specials.emplace_back(text, (int32_t) id); }
        }
        if (t.bpe.id_to_str[t.bos_id] != "<|begin_of_text|>" ||
            t.bpe.id_to_str[t.eos_id] != "<|end_of_text|>") {
            throw std::runtime_error("BOS/EOS spelling mismatch");
        }
        for (size_t rank = 0; rank < 280147; ++rank) {
            const std::string merge = gguf_get_arr_str(ctx, merges, rank);
            const auto sep = merge.find(' ');
            if (sep == std::string::npos || merge.find(' ', sep + 1) != std::string::npos ||
                !t.bpe.merges.emplace(merge, (int) rank).second) {
                throw std::runtime_error("invalid/duplicate merge");
            }
            for (const auto & text : { merge.substr(0, sep), merge.substr(sep + 1),
                                      merge.substr(0, sep) + merge.substr(sep + 1) }) {
                const auto it = t.bpe.vocab.find(text);
                if (it == t.bpe.vocab.end() || it->second >= 128000) {
                    throw std::runtime_error("merge references a missing base token");
                }
            }
        }
        build_byte_encoder(t.bpe.byte2str);
        for (const auto & byte : t.bpe.byte2str) {
            const auto it = t.bpe.vocab.find(byte);
            if (it == t.bpe.vocab.end() || it->second >= 128000) {
                throw std::runtime_error("missing base byte token");
            }
        }
        t.bpe.n_vocab = 128256;
        t.bpe.eos_id = t.eos_id;
        read_ranges(ctx, "heartmula.tokenizer.letters", t.letters);
        read_ranges(ctx, "heartmula.tokenizer.numbers", t.numbers);
        read_ranges(ctx, "heartmula.tokenizer.spaces", t.spaces);
        read_ranges(ctx, "heartmula.tokenizer.cased", t.cased);
        read_ranges(ctx, "heartmula.tokenizer.case_ignorable", t.case_ignorable);
        const auto lower_keys = read_u32_array(ctx, "heartmula.tokenizer.lower_keys");
        const auto lower_values = array("heartmula.tokenizer.lower_values", lower_keys.size());
        for (size_t i = 0; i < lower_keys.size(); ++i) {
            const std::string value = gguf_get_arr_str(ctx, lower_values, i);
            if (lower_keys[i] > 0x10ffff || value.empty() ||
                !t.lower.emplace(lower_keys[i], value).second) {
                throw std::runtime_error("invalid lowercase table");
            }
            decode(value);
        }
        std::sort(t.specials.begin(), t.specials.end(), [](const auto & a, const auto & b) {
            return a.first.size() > b.first.size();
        });
        t.loaded = true;
        *out = std::move(t); // A failed load leaves the caller's previous tokenizer intact.
        gguf_free(ctx);
        return true;
    } catch (const std::exception & e) {
        if (error) { *error = e.what(); }
        gguf_free(ctx);
        return false;
    }
}

// Raw mode mirrors HF's single-sequence postprocessor, which adds BOS but no EOS.
inline std::vector<int32_t> heartmula_tokenizer_encode(const HeartMuLaTokenizer & t,
                                                      const std::string & text, bool add_bos = true) {
    using namespace heartmula_tokenizer_detail;
    if (!t.loaded) { throw std::runtime_error("tokenizer is not loaded"); }
    decode(text); // Validate before special matching or byte-BPE.
    std::vector<int32_t> ids;
    if (add_bos) { ids.push_back(t.bos_id); }
    const auto flush = [&](size_t start, size_t end) {
        for (const auto & chunk : split(t, text.substr(start, end - start))) {
            const std::string encoded = byte_level_encode(&t.bpe, chunk);
            const auto whole = t.bpe.vocab.find(encoded);
            if (whole != t.bpe.vocab.end() && whole->second < 128000) {
                ids.push_back(whole->second); // HF BPE ignore_merges=true.
            } else {
                encode_chunk(&t.bpe, chunk, ids);
            }
        }
    };
    size_t start = 0, i = 0;
    while (i < text.size()) {
        const std::pair<std::string, int32_t> * hit = nullptr;
        for (const auto & special : t.specials) {
            if (text[i] == special.first[0] && text.compare(i, special.first.size(), special.first) == 0) {
                hit = &special; break;
            }
        }
        if (!hit) { ++i; continue; }
        flush(start, i);
        ids.push_back(hit->second);
        i += hit->first.size(); start = i;
    }
    flush(start, text.size());
    return ids;
}

// heartlib preprocesses tags and lyrics independently; the pipeline inserts its
// reference-audio placeholder between the two resulting rows.
inline std::vector<int32_t> heartmula_tokenizer_prompt(const HeartMuLaTokenizer & t,
                                                      const std::string & input, bool tags) {
    std::string text = heartmula_tokenizer_detail::lowercase(t, input);
    if (tags) {
        if (text.compare(0, 5, "<tag>") != 0) { text = "<tag>" + text; }
        if (text.size() < 6 || text.compare(text.size() - 6, 6, "</tag>") != 0) { text += "</tag>"; }
    }
    auto ids = heartmula_tokenizer_encode(t, text);
    if (ids.empty() || ids.front() != t.bos_id) { ids.insert(ids.begin(), t.bos_id); }
    if (ids.back() != t.eos_id) { ids.push_back(t.eos_id); }
    return ids;
}
