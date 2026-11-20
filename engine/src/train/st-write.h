#pragma once
// st-write.h: minimal safetensors WRITER (the engine's safetensors.h is
// read-only). Emits [8B LE header_len][JSON header][raw data], data_offsets
// relative to the data section, no inter-tensor padding, header space-padded
// so (8 + header_len) % 8 == 0.
//
// Source data is ALWAYS f32; storage dtype is chosen per file.
//
// docs/plans/2026-07-27-preprocess-implementation.md §3.2

#include "hot-step-fsutf8.h"  // hs_fopen/hs_rename/hs_remove — UTF-8 paths
#include "yyjson.h"

#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

enum STWDType { STW_F32, STW_BF16, STW_F16 };

struct STWTensor {
    std::string          name;
    std::vector<int64_t> shape;  // row-major; shape[0] varies slowest
    const float *        data;   // prod(shape) f32 values, row-major
    // Per-tensor storage override (2026-09-09): -1 = the file's dtype. Small
    // metadata tensors (markers, meta vectors, soft-prompt rows) must stay F32
    // in an F16 file — a loader keys on their dtype, and a byte count above
    // 65504 (a base file's size split into two floats) is inf in F16.
    int                  dtype_override = -1;
};

// float -> IEEE half with round-to-nearest-even (2026-09-09: the PiSSA
// residual and delta exports). Same rounding as ggml_fp32_to_fp16, written out
// here so this header stays free of ggml. Overflow saturates to +-inf, which
// the LM loaders refuse by name rather than render as noise.
static inline uint16_t stw_f32_to_f16(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    const uint32_t sign = (b >> 16) & 0x8000u;
    const uint32_t exp  = (b >> 23) & 0xFFu;
    const uint32_t man  = b & 0x7FFFFFu;
    if (exp == 0xFFu) {  // inf / NaN
        return (uint16_t) (sign | 0x7C00u | (man ? 0x200u : 0u));
    }
    int e = (int) exp - 127 + 15;
    if (e >= 0x1F) {
        return (uint16_t) (sign | 0x7C00u);  // overflow -> inf
    }
    if (e <= 0) {
        if (e < -10) {
            return (uint16_t) sign;  // underflow -> signed zero
        }
        // subnormal: shift the implicit-1 mantissa right, round to nearest even
        uint32_t m     = man | 0x800000u;
        int      shift = 14 - e;
        uint32_t half  = m >> shift;
        uint32_t rem   = m & ((1u << shift) - 1u);
        uint32_t mid   = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1u))) {
            half++;
        }
        return (uint16_t) (sign | half);
    }
    uint32_t half = ((uint32_t) e << 10) | (man >> 13);
    uint32_t rem  = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) {
        half++;  // may carry into the exponent, which is the correct rounding
    }
    return (uint16_t) (sign | half);
}

// float -> bfloat16 with round-to-nearest-even.
static inline uint16_t stw_f32_to_bf16(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    if (((b >> 23) & 0xFFu) == 0xFFu && (b & 0x7FFFFFu)) {
        return (uint16_t) ((b >> 16) | 0x40u);  // keep NaN
    }
    b += 0x7FFFu + ((b >> 16) & 1u);
    return (uint16_t) (b >> 16);
}

static inline int64_t stw_numel(const std::vector<int64_t> & shape) {
    int64_t n = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        n *= shape[i];
    }
    return shape.empty() ? 0 : n;
}

// Publish a fully written sibling in one rename. A failed replacement leaves the previous file in place.
static bool stw_replace_file(const char * tmp, const char * path) {
    std::error_code error;
    std::filesystem::rename(std::filesystem::u8path(tmp), std::filesystem::u8path(path), error);
    if (error) {
        fprintf(stderr, "[STWrite] rename %s -> %s failed: %s\n", tmp, path, error.message().c_str());
    }
    return !error;
}

// Write tensors and metadata to a temporary sibling before publishing them together.
static bool st_write_file(const char *                                             path,
                          const std::vector<STWTensor> &                           tensors,
                          const std::vector<std::pair<std::string, std::string>> & metadata,
                          STWDType                                                 dtype) {
    auto dtype_of = [&](size_t i) -> STWDType {
        return tensors[i].dtype_override < 0 ? dtype : (STWDType) tensors[i].dtype_override;
    };
    auto esz_of = [&](size_t i) -> size_t { return dtype_of(i) == STW_F32 ? 4u : 2u; };
    auto dts_of = [&](size_t i) -> const char * {
        const STWDType d = dtype_of(i);
        return d == STW_F32 ? "F32" : d == STW_F16 ? "F16" : "BF16";
    };

    // 1. Offsets (relative to the data section, no padding between tensors).
    std::vector<uint64_t> starts(tensors.size()), ends(tensors.size());
    uint64_t              cursor = 0;
    for (size_t i = 0; i < tensors.size(); i++) {
        int64_t n = stw_numel(tensors[i].shape);
        if (n < 0 || tensors[i].data == NULL) {
            fprintf(stderr, "[STWrite] invalid tensor '%s'\n", tensors[i].name.c_str());
            return false;
        }
        starts[i] = cursor;
        cursor += (uint64_t) n * (uint64_t) esz_of(i);
        ends[i] = cursor;
    }

    // 2. JSON header (yyjson -- captions in __metadata__ carry quotes/newlines).
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    if (!metadata.empty()) {
        yyjson_mut_val * md = yyjson_mut_obj(doc);
        for (size_t i = 0; i < metadata.size(); i++) {
            yyjson_mut_obj_add(md, yyjson_mut_strncpy(doc, metadata[i].first.c_str(), metadata[i].first.size()),
                               yyjson_mut_strncpy(doc, metadata[i].second.c_str(), metadata[i].second.size()));
        }
        yyjson_mut_obj_add_val(doc, root, "__metadata__", md);
    }

    for (size_t i = 0; i < tensors.size(); i++) {
        yyjson_mut_val * e = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_str(doc, e, "dtype", dts_of(i));
        yyjson_mut_val * sh = yyjson_mut_obj_add_arr(doc, e, "shape");
        for (size_t d = 0; d < tensors[i].shape.size(); d++) {
            yyjson_mut_arr_add_sint(doc, sh, tensors[i].shape[d]);
        }
        yyjson_mut_val * off = yyjson_mut_obj_add_arr(doc, e, "data_offsets");
        yyjson_mut_arr_add_uint(doc, off, starts[i]);
        yyjson_mut_arr_add_uint(doc, off, ends[i]);
        yyjson_mut_obj_add(root, yyjson_mut_strncpy(doc, tensors[i].name.c_str(), tensors[i].name.size()), e);
    }

    size_t hdr_len = 0;
    char * hdr     = yyjson_mut_write(doc, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &hdr_len);
    yyjson_mut_doc_free(doc);
    if (!hdr) {
        fprintf(stderr, "[STWrite] header serialization failed for %s\n", path);
        return false;
    }

    // 3. Space-pad so (8 + header_len) % 8 == 0.
    std::string header(hdr, hdr_len);
    free(hdr);
    while ((8 + header.size()) % 8 != 0) {
        header.push_back(' ');
    }

    // 4. Write to a temp file, then rename.
    std::string tmp = std::string(path) + ".__writing__";
    FILE *      f   = hs_fopen(tmp, "wb");
    if (!f) {
        fprintf(stderr, "[STWrite] cannot open %s for writing: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }

    uint64_t hl = (uint64_t) header.size();
    uint8_t  hl_le[8];
    for (int i = 0; i < 8; i++) {
        hl_le[i] = (uint8_t) ((hl >> (8 * i)) & 0xFFu);  // explicit little-endian
    }
    bool ok = (fwrite(hl_le, 1, 8, f) == 8);
    ok      = ok && (fwrite(header.data(), 1, header.size(), f) == header.size());

    // 5. Stream tensor data in offset order.
    std::vector<uint16_t> scratch;
    for (size_t i = 0; ok && i < tensors.size(); i++) {
        const int64_t  n  = stw_numel(tensors[i].shape);
        const float *  p  = tensors[i].data;
        const STWDType dt = dtype_of(i);
        if (dt == STW_F32) {
            size_t left = (size_t) n;
            while (ok && left > 0) {
                size_t chunk = left > 65536 ? 65536 : left;
                ok           = (fwrite(p, sizeof(float), chunk, f) == chunk);
                p += chunk;
                left -= chunk;
            }
        } else {
            if (scratch.size() < 65536) {
                scratch.resize(65536);
            }
            size_t left = (size_t) n;
            while (ok && left > 0) {
                size_t chunk = left > 65536 ? 65536 : left;
                if (dt == STW_F16) {
                    for (size_t k = 0; k < chunk; k++) {
                        scratch[k] = stw_f32_to_f16(p[k]);
                    }
                } else {
                    for (size_t k = 0; k < chunk; k++) {
                        scratch[k] = stw_f32_to_bf16(p[k]);
                    }
                }
                ok = (fwrite(scratch.data(), sizeof(uint16_t), chunk, f) == chunk);
                p += chunk;
                left -= chunk;
            }
        }
    }

    if (ok) {
        ok = (fflush(f) == 0) && (ferror(f) == 0);
    }
    if (fclose(f) != 0) ok = false;

    if (!ok) {
        hs_remove(tmp);
        fprintf(stderr, "[STWrite] write failed for %s\n", path);
        return false;
    }

    if (!stw_replace_file(tmp.c_str(), path)) {
        hs_remove(tmp);
        return false;
    }
    return true;
}
