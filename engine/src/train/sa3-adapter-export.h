#pragma once

// SA3 DoRA-rows export contract. All arrays are row-major F32: A [rank,in],
// B [out,rank], magnitude [out]. `stem` uses the training `dit.*` namespace.
// Export is evaluated against the actual BF16 inference GGUF, never a training
// base approximation. Its immutable revision/hash belongs in inference_base_id.
// Raw adapter parameters are a separate sa3-lora file. A complete resume also
// needs the trainer's optimizer, RNG, step and dataset state, saved by the caller.
// No tensors or input parameters are modified. Existing output files are refused.

#include "../gguf-weights.h"
#include <cmath>
#include <filesystem>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sa3_export {

struct Rows {
    std::string stem;
    int64_t in = 0, out = 0, rank = 0;
    float alpha = 0;
    std::vector<float> a, b, magnitude;
};

// Reverse the training converter's weight names. Unknown modules fail closed.
inline std::string original_name(const std::string & stem) {
    const std::pair<const char *, const char *> direct[] = {
        {"pre_conv", "preprocess_conv"}, {"post_conv", "postprocess_conv"},
        {"proj_in", "transformer.project_in"}, {"proj_out", "transformer.project_out"},
        {"cond_embed.0", "to_cond_embed.0"}, {"cond_embed.2", "to_cond_embed.2"},
        {"global_embed.0", "to_global_embed.0"}, {"global_embed.2", "to_global_embed.2"},
        {"time_embed.0", "to_timestep_embed.0"}, {"time_embed.2", "to_timestep_embed.2"},
        {"gce.0", "transformer.global_cond_embedder.0"},
        {"gce.2", "transformer.global_cond_embedder.2"}
    };
    if (stem.compare(0, 4, "dit.") != 0) throw std::invalid_argument("expected dit.* training stem");
    const auto tail = stem.substr(4);
    for (const auto & item : direct)
        if (tail == item.first) return "model." + std::string(item.second) + ".weight";
    const auto dot = tail.find('.');
    if (dot == std::string::npos || dot == 0 ||
        tail.substr(0, dot).find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("unknown SA3 target: " + stem);
    const std::pair<const char *, const char *> layer[] = {
        {"self.qkv", "self_attn.to_qkv"}, {"self.out", "self_attn.to_out"},
        {"cross.q", "cross_attn.to_q"}, {"cross.kv", "cross_attn.to_kv"},
        {"cross.out", "cross_attn.to_out"}, {"ff.proj", "ff.ff.0.proj"},
        {"ff.out", "ff.ff.2"}, {"local.0", "to_local_embed.0"},
        {"local.2", "to_local_embed.2"}
    };
    for (const auto & item : layer)
        if (tail.substr(dot + 1) == item.first)
            return "model.transformer.layers." + tail.substr(0, dot) + "." + item.second + ".weight";
    throw std::invalid_argument("unknown SA3 target: " + stem);
}

inline void validate(const Rows & p) {
    original_name(p.stem);
    const auto max_count = (std::numeric_limits<size_t>::max)() / sizeof(float);
    if (p.in <= 0 || p.out <= 0 || p.rank <= 0 ||
        uint64_t(p.in) > max_count / uint64_t(p.rank) ||
        uint64_t(p.out) > max_count / uint64_t(p.rank) ||
        p.a.size() != size_t(p.in) * size_t(p.rank) ||
        p.b.size() != size_t(p.out) * size_t(p.rank) || p.magnitude.size() != size_t(p.out) ||
        !std::isfinite(p.alpha)) throw std::invalid_argument("invalid DoRA dimensions/alpha");
    for (const auto * v : {&p.a, &p.b, &p.magnitude})
        for (float f : *v) if (!std::isfinite(f)) throw std::invalid_argument("nonfinite DoRA parameter");
    for (float f : p.magnitude) if (f < 0) throw std::invalid_argument("negative DoRA magnitude");
}

struct File {
    ggml_context * ctx = nullptr;
    gguf_context * gguf = nullptr;
    std::vector<std::vector<float>> arrays;
    explicit File(size_t count) {
        ctx = ggml_init({count * (ggml_tensor_overhead() + 64) + 1024, nullptr, true});
        if (!ctx) throw std::runtime_error("export tensor context allocation failed");
        gguf = gguf_init_empty();
        if (!gguf) { ggml_free(ctx); throw std::runtime_error("export GGUF allocation failed"); }
    }
    File(const File &) = delete;
    File & operator=(const File &) = delete;
    ~File() { gguf_free(gguf); ggml_free(ctx); }
    void add(const std::string & name, int64_t n0, int64_t n1, std::vector<float> data) {
        if (name.size() >= GGML_MAX_NAME) throw std::invalid_argument("tensor name too long");
        arrays.push_back(std::move(data));
        auto * t = n1 == 0 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n0) :
                            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1);
        ggml_set_name(t, name.c_str());
        t->data = arrays.back().data();
        gguf_add_tensor(gguf, t);
    }
    void write(const std::string & path) {
        if (path.empty() || std::filesystem::exists(std::filesystem::u8path(path)))
            throw std::invalid_argument("output path empty or already exists");
        if (!gguf_write_to_file(gguf, path.c_str(), false))
            throw std::runtime_error("GGUF write failed: " + path);
    }
};

inline void validate_set(const std::vector<Rows> & params) {
    if (params.empty()) throw std::invalid_argument("empty DoRA adapter");
    std::set<std::string> targets;
    for (const auto & p : params) {
        validate(p);
        if (!targets.insert(original_name(p.stem)).second)
            throw std::invalid_argument("duplicate DoRA target");
    }
}

inline void write_inference(const std::string & path, const GGUFModel & base,
                            const std::string & inference_base_id,
                            const std::vector<Rows> & params, double eps = 1e-12) {
    validate_set(params);
    if (!base.meta || !base.gguf || inference_base_id.empty() || !std::isfinite(eps) || eps <= 0)
        throw std::invalid_argument("inference base identity and positive epsilon required");
    File file(params.size() * 3);
    gguf_set_val_str(file.gguf, "general.architecture", "sa3-adapter");
    gguf_set_val_str(file.gguf, "sa3.adapter_format", "dora-rows-v1");
    gguf_set_val_str(file.gguf, "sa3.inference_base_id", inference_base_id.c_str());
    for (const auto & p : params) {
        const auto name = original_name(p.stem);
        auto * w = ggml_get_tensor(base.meta, name.c_str());
        const bool matrix = w && w->ne[0] == p.in && w->ne[1] == p.out && w->ne[2] == 1;
        const bool conv = w && w->ne[0] == 1 && w->ne[1] == p.in && w->ne[2] == p.out;
        if (!w || w->type != GGML_TYPE_BF16 || w->ne[3] != 1 || !(matrix || conv))
            throw std::invalid_argument("incompatible BF16 inference weight: " + name);
        const auto * data = static_cast<const ggml_bf16_t *>(gf_get_data(base, name.c_str()));
        if (!data) throw std::invalid_argument("missing inference weight data");
        std::vector<float> folded = p.b, scale(size_t(p.out)), row(size_t(p.in));
        for (float & f : folded) {
            f = float(double(f) * double(p.alpha) / double(p.rank));
            if (!std::isfinite(f)) throw std::invalid_argument("folded B overflow");
        }
        for (int64_t i = 0; i < p.out; ++i) {
            ggml_bf16_to_fp32_row(data + i * p.in, row.data(), p.in);
            double norm2 = 0;
            for (int64_t j = 0; j < p.in; ++j) {
                double v = row[size_t(j)];
                for (int64_t k = 0; k < p.rank; ++k)
                    v += double(folded[size_t(i * p.rank + k)]) * p.a[size_t(k * p.in + j)];
                norm2 += v * v;
            }
            scale[size_t(i)] = float(p.magnitude[size_t(i)] / (std::sqrt(norm2) + eps));
            if (!std::isfinite(norm2) || !std::isfinite(scale[size_t(i)]))
                throw std::invalid_argument("nonfinite row normalization");
        }
        file.add(name + ".lora_a", p.in, p.rank, p.a);
        file.add(name + ".lora_b", p.rank, p.out, std::move(folded));
        file.add(name + ".row_scale", p.out, 0, std::move(scale));
    }
    file.write(path);
}

inline void write_raw(const std::string & path, const std::string & training_base_id,
                      const std::vector<Rows> & params) {
    validate_set(params);
    if (training_base_id.empty()) throw std::invalid_argument("training base identity required");
    if (uint64_t(params[0].rank) > (std::numeric_limits<uint32_t>::max)())
        throw std::invalid_argument("raw rank exceeds GGUF metadata range");
    // The community raw format has global rank/alpha metadata.
    for (const auto & p : params)
        if (p.rank != params[0].rank || p.alpha != params[0].alpha)
            throw std::invalid_argument("raw state requires uniform rank and alpha");
    File file(params.size() * 3);
    gguf_set_val_str(file.gguf, "general.architecture", "sa3-lora");
    gguf_set_val_str(file.gguf, "lora.adapter_type", "dora-rows");
    gguf_set_val_str(file.gguf, "lora.target", "dit");
    gguf_set_val_str(file.gguf, "lora.base_model", training_base_id.c_str());
    gguf_set_val_u32(file.gguf, "lora.rank", uint32_t(params[0].rank));
    gguf_set_val_f32(file.gguf, "lora.alpha", params[0].alpha);
    gguf_set_val_u32(file.gguf, "lora.n_targets", uint32_t(params.size()));
    for (const auto & p : params) {
        file.add(p.stem + ".lora_A", p.in, p.rank, p.a);
        file.add(p.stem + ".lora_B", p.rank, p.out, p.b);
        file.add(p.stem + ".magnitude", p.out, 0, p.magnitude);
    }
    file.write(path);
}

} // namespace sa3_export
