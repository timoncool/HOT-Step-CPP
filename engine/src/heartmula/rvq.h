#pragma once

// Shared HeartCodec RVQ. Codebook sums and projections remain F32.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "gguf-weights.h"
#include "ggml-alloc.h"
#include "train/yue2-aitk-sha256.h"
#include "yyjson.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace heartcodec {

inline void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error("HeartCodec: " + message);
}

// Thrown at every cancel point, so callers catch it by type.
struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("HeartCodec: cancelled") {}
};

inline void check_cancel(const std::function<bool()> & cancelled) {
    if (cancelled && cancelled()) throw Cancelled();
}

// Whether Weights::load keeps a tensor in its stored type (else F32).
inline bool keep_native(const std::string & role, const ggml_tensor * t, bool gpu) {
    const std::string name = ggml_get_name(t);
    const bool matrix = ggml_n_dims(t) == 2 && name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0;
    return role != "rvq" && matrix && (gpu || ggml_is_quantized(t->type));
}

struct Weights {
    WeightCtx storage{};
    std::map<std::string, ggml_tensor *> tensors;
    std::string fingerprint, dependencies, config, rvq_hash, file_hash;
    Weights() = default;
    Weights(const Weights &) = delete;
    Weights & operator=(const Weights &) = delete;
    ~Weights() { wctx_free(&storage); }

    void load(const std::string & path, const std::string & role, ggml_backend_t backend) {
        require(!storage.ctx, "weights already loaded");
        GGUFModel file{};
        require(gf_load(&file, path.c_str()), "cannot load " + path);
        struct Close { GGUFModel * f; ~Close() { gf_close(f); } } close{&file};
        auto str = [&](const char * key) -> std::string {
            int64_t i = gguf_find_key(file.gguf, key);
            require(i >= 0 && gguf_get_kv_type(file.gguf, i) == GGUF_TYPE_STRING, path + ": missing " + key);
            return gguf_get_val_str(file.gguf, i);
        };
        require(str("general.architecture") == "heartcodec-" + role, path + ": architecture mismatch");
        require(str("heartmula.component") == role, path + ": component mismatch");
        require(str("heartmula.variant") == "shared", path + ": codec variant must be shared");
        int64_t version = gguf_find_key(file.gguf, "heartmula.format_version");
        require(version >= 0 && gguf_get_kv_type(file.gguf, version) == GGUF_TYPE_UINT32 &&
                gguf_get_val_u32(file.gguf, version) == 1, path + ": unsupported format version");
        fingerprint = str("heartmula.base_fingerprint");
        dependencies = str("heartmula.dependencies");
        config = str("heartmula.config_json");
        require(yue2_aitk::sha256::bytes(config.data(), config.size()).hex() == str("heartmula.config_sha256"),
                path + ": config hash mismatch");
        using Json = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;
        Json dep(yyjson_read(dependencies.data(), dependencies.size(), 0), yyjson_doc_free);
        require(bool(dep), path + ": invalid dependencies JSON");
        auto hash = yyjson_obj_get(yyjson_doc_get_root(dep.get()), "rvq_sha256");
        if (role == "rvq") {
            require(yyjson_is_null(hash), path + ": RVQ must not depend on another codebook");
            yue2_aitk::sha256::digest digest;
            require(yue2_aitk::sha256::file(path, digest), path + ": cannot hash codebook");
            file_hash = digest.hex();
        } else {
            require(yyjson_is_str(hash) && yyjson_get_len(hash) == 64, path + ": missing RVQ hash");
            rvq_hash = yyjson_get_str(hash);
        }
        Json cfg(yyjson_read(config.data(), config.size(), 0), yyjson_doc_free);
        require(bool(cfg), path + ": invalid config JSON");
        auto root = yyjson_doc_get_root(cfg.get());
        auto integer = [&](const char * key, int expected) {
            auto val = yyjson_obj_get(root, key);
            require(yyjson_is_int(val) && yyjson_get_sint(val) == expected, path + ": unsupported config " + key);
        };
        integer("sample_rate", 48000); integer("dim", 512); integer("codebook_dim", 32);
        integer("codebook_size", 8192); integer("num_quantizers", 8);
        integer("num_attention_heads", 24); integer("attention_head_dim", 64);
        integer("num_layers", 24); integer("num_layers_2", 6);
        integer("in_channels", 1024); integer("out_channels", 256);
        integer("latent_hidden_dim", 128); integer("num_bands", 1); integer("init_channel", 64);
        integer("num_samples", 2); integer("default_kernel_size", 7);
        integer("delay_kernel_size", 5); integer("res_kernel_size", 7);
        require(yyjson_is_true(yyjson_obj_get(root, "causal")), path + ": expected causal scalar decoder");
        auto array = [&](const char * key, std::initializer_list<int> expected) {
            auto val = yyjson_obj_get(root, key);
            require(yyjson_is_arr(val) && yyjson_arr_size(val) == expected.size(), path + ": invalid " + key);
            size_t i = 0;
            for (int v : expected) {
                auto item = yyjson_arr_get(val, i++);
                require(yyjson_is_int(item) && yyjson_get_sint(item) == v, path + ": unsupported " + key);
            }
        };
        array("upsample_factors", {5, 4, 4, 4, 3}); array("upsample_kernel_sizes", {10, 8, 8, 8, 6});
        int64_t count = gguf_get_n_tensors(file.gguf);
        wctx_init(&storage, int(count));
        // F32 execution of rounded F16 weights is the S1 parity baseline on
        // CPU: it avoids F16 activation rounding in the CPU GEMM and im2col
        // kernels. On a GPU, linear matrices keep their stored type (F16 or a
        // quant) so the flow fits its F16/Q8_0 budget; conv kernels, norms and
        // tables are small and stay F32 everywhere.
        auto device = ggml_backend_get_device(backend);
        const bool gpu = device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU;
        for (int64_t i = 0; i < count; ++i) {
            const char * name = gguf_get_tensor_name(file.gguf, i);
            ggml_tensor * src = ggml_get_tensor(file.meta, name);
            require(role != "rvq" || src->type == GGML_TYPE_F32, path + ": RVQ state must remain F32");
            if (keep_native(role, src, gpu)) {
                tensors[name] = gf_load_tensor(&storage, file, name);
            } else if (src->type == GGML_TYPE_F16) {
                auto dst = ggml_new_tensor(storage.ctx, GGML_TYPE_F32, ggml_n_dims(src), src->ne);
                ggml_set_name(dst, name);
                auto data = std::make_unique<float[]>(size_t(ggml_nelements(src)));
                auto raw = reinterpret_cast<const ggml_fp16_t *>(file.mapping + file.data_offset + gguf_get_tensor_offset(file.gguf, i));
                ggml_fp16_to_fp32_row(raw, data.get(), ggml_nelements(src));
                storage.pending.push_back({dst, data.get(), ggml_nbytes(dst), 0});
                storage.staging.push_back(std::move(data));
                tensors[name] = dst;
            } else {
                require(src->type == GGML_TYPE_F32, path + ": unsupported tensor precision for " + name);
                tensors[name] = gf_load_tensor(&storage, file, name);
            }
        }
        require(wctx_alloc(&storage, backend), path + ": allocation failed");
    }
    ggml_tensor * get(const std::string & name) const {
        auto it = tensors.find(name);
        require(it != tensors.end(), "missing tensor " + name);
        return it->second;
    }
    ggml_tensor * optional(const std::string & name) const {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : it->second;
    }
};

struct Graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_backend_t backend;
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> inputs;
    // `nodes` bounds both the graph and its context; size it per graph.
    Graph(ggml_backend_t b, size_t nodes) : backend(b) {
        ctx = ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true});
        require(ctx != nullptr, "graph allocation failed");
        graph = ggml_new_graph_custom(ctx, nodes, false);
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    Graph(const Graph &) = delete;
    ~Graph() { if (alloc) ggml_gallocr_free(alloc); if (ctx) ggml_free(ctx); }
    ggml_tensor * input(ggml_type type, std::vector<int64_t> shape, const void * data) {
        auto t = ggml_new_tensor(ctx, type, int(shape.size()), shape.data());
        ggml_set_input(t);
        const auto p = static_cast<const uint8_t *>(data);
        inputs.push_back({t, std::vector<uint8_t>(p, p + ggml_nbytes(t))});
        return t;
    }
    std::vector<float> run(ggml_tensor * output) {
        output = ggml_cont(ctx, output);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        require(ggml_gallocr_alloc_graph(alloc, graph), "compute allocation failed");
        for (const auto & v : inputs) ggml_backend_tensor_set(v.first, v.second.data(), 0, v.second.size());
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph computation failed");
        std::vector<float> values(size_t(ggml_nelements(output)));
        ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
        for (float v : values) require(std::isfinite(v), "nonfinite output");
        return values;
    }
};

inline ggml_tensor * linear(ggml_context * c, const Weights & w, const std::string & p, ggml_tensor * x) {
    auto a = w.get(p + ".weight");
    require(a->ne[0] == x->ne[0] && a->ne[2] == 1 && a->ne[3] == 1, p + ": matrix shape mismatch");
    auto y = ggml_mul_mat(c, a, x);
    if (auto b = w.optional(p + ".bias")) y = ggml_add(c, y, b);
    return y;
}

inline std::vector<float> rvq_decode(const Weights & w, ggml_backend_t backend,
                                     const std::vector<int64_t> & codes, int64_t frames) {
    require(frames > 0 && codes.size() == size_t(frames * 8), "expected [8,frames] codes");
    Graph g(backend, 64);
    ggml_tensor * sum = nullptr;
    for (int q = 0; q < 8; ++q) {
        std::vector<int32_t> ids(size_t(frames), 0);
        for (int64_t t = 0; t < frames; ++t) {
            auto id = codes[size_t(q * frames + t)];
            require(id >= 0 && id < 8192, "code outside [0,8192)");
            ids[size_t(t)] = int32_t(id);
        }
        auto book = w.get("flow_matching.vq_embed.layers." + std::to_string(q) + "._codebook.embed");
        require(book->type == GGML_TYPE_F32 && book->ne[0] == 32 && book->ne[1] == 8192, "invalid codebook shape/type");
        auto rows = ggml_get_rows(g.ctx, ggml_reshape_2d(g.ctx, book, 32, 8192),
                                 g.input(GGML_TYPE_I32, {frames}, ids.data()));
        sum = sum ? ggml_add(g.ctx, sum, rows) : rows;
    }
    return g.run(linear(g.ctx, w, "flow_matching.vq_embed.project_out", sum));
}

} // namespace heartcodec
