#pragma once
// HeartCodec query encoder (feature fusion, six weight-normalized blocks with
// partial RoPE and Q/K LayerNorm, one query per two 25 Hz frames) and the shared
// RVQ's nearest-code search. The search runs on the host in double: F16 or a
// different summation order changes nearest-code choices more than anything else.
#include "heartmula/encoder-features.h"

namespace heartcodec {

// Stereo features [2048 | 2048 | 1536 | 1536, frames] -> queries [512, frames / 2].
inline std::vector<float> query_run(const Weights & w, ggml_backend_t backend, const std::vector<float> & acoustic,
                                    const std::vector<float> & semantic, const std::vector<float> & whisper,
                                    const std::vector<float> & wavlm, int64_t frames) {
    const std::string p = "encoder.query.";
    require(frames > 0 && frames % 2 == 0, "query frames must be even");
    require(acoustic.size() == size_t(2048 * frames) && semantic.size() == acoustic.size() &&
            whisper.size() == size_t(1536 * frames) && wavlm.size() == whisper.size(), "query feature shapes");
    Graph g(backend, 2048);
    auto c = g.ctx;
    auto fused = ggml_concat(c, ggml_concat(c,
        linear(c, w, p + "cond_fusion_layer_acoustic", g.input(GGML_TYPE_F32, {2048, frames}, acoustic.data())),
        linear(c, w, p + "cond_fusion_layer_semantic", g.input(GGML_TYPE_F32, {2048, frames}, semantic.data())), 0),
        ggml_concat(c, g.input(GGML_TYPE_F32, {1536, frames}, whisper.data()),
        linear(c, w, p + "cond_fusion_layer_phone", g.input(GGML_TYPE_F32, {1536, frames}, wavlm.data())), 0), 0);
    fused = linear(c, w, p + "feature_proj", fused);
    // Two frames then the class token, per group: [512, 3, frames / 2].
    const int64_t groups = frames / 2, length = groups * 3;
    auto cls = ggml_repeat(c, ggml_reshape_3d(c, w.get(p + "cls_token"), 512, 1, 1), ggml_new_tensor_3d(c, GGML_TYPE_F32, 512, 1, groups));
    auto x = ggml_reshape_2d(c, ggml_concat(c, ggml_reshape_3d(c, fused, 512, 2, groups), cls, 1), 512, length);
    auto pos = positions(g, length);
    for (int l = 0; l < 6; ++l) {
        const std::string q = p + "encoder_transformers." + std::to_string(l) + ".";
        auto qkv = ggml_mul_mat(c, w.get(q + "self_attn.to_qkv.weight"), x);
        auto head = [&](int i, const char * norm) {
            auto t = ggml_cont(c, ggml_view_3d(c, qkv, 128, 4, length, 128 * ggml_element_size(qkv), qkv->nb[1],
                                               size_t(i) * 512 * ggml_element_size(qkv)));
            if (norm) {
                t = layer_norm(c, w, q + "self_attn." + norm, t);
                t = ggml_rope_ext(c, t, pos, nullptr, 64, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            }
            return t;
        };
        auto attn = attend(c, head(0, "q_norm"), head(1, "k_norm"), head(2, nullptr), 1.0f / std::sqrt(128.0f));
        attn = ggml_mul_mat(c, w.get(q + "self_attn.to_out.weight"), attn);
        x = ggml_add(c, x, ggml_mul(c, attn, w.get(q + "self_attn_scale.scale")));
        auto h = linear(c, w, q + "ff.ff.0.proj", x);
        auto values = ggml_cont(c, ggml_view_2d(c, h, 2048, length, h->nb[1], 0));
        auto gate = ggml_cont(c, ggml_view_2d(c, h, 2048, length, h->nb[1], 2048 * ggml_element_size(h)));
        h = linear(c, w, q + "ff.ff.2", ggml_mul(c, values, ggml_sigmoid(c, gate)));
        x = ggml_add(c, x, ggml_mul(c, h, w.get(q + "ff_scale.scale")));
    }
    // Keep the class-token positions 2, 5, 8, ...
    return g.run(ggml_view_2d(c, x, 512, groups, 3 * x->nb[1], 2 * x->nb[1]));
}

// Shared RVQ encode: project_in, then per layer the nearest of 8192 codes by
// squared distance (vector-quantize-pytorch's clamped cdist, first index on ties)
// and subtraction from the residual. Queries [512, n] -> codes [n][8].
inline std::vector<int64_t> rvq_encode(const Weights & rvq, const std::vector<float> & queries, int64_t n) {
    require(n > 0 && queries.size() == size_t(512 * n), "RVQ encode input shape");
    auto fetch = [&](const std::string & name, int64_t expected) {
        auto t = rvq.get(name);
        require(t->type == GGML_TYPE_F32 && ggml_nelements(t) == expected, name + ": unexpected RVQ tensor");
        std::vector<float> v(static_cast<size_t>(expected));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        return v;
    };
    const std::string p = "flow_matching.vq_embed.";
    auto weight = fetch(p + "project_in.weight", 512 * 32), bias = fetch(p + "project_in.bias", 32);
    std::vector<std::vector<float>> books;
    std::vector<std::vector<double>> squares;
    for (int q = 0; q < 8; ++q) {
        books.push_back(fetch(p + "layers." + std::to_string(q) + "._codebook.embed", 8192 * 32));
        std::vector<double> sq(8192, 0.0);
        for (int e = 0; e < 8192; ++e) for (int d = 0; d < 32; ++d) sq[size_t(e)] += double(books.back()[size_t(e * 32 + d)]) * books.back()[size_t(e * 32 + d)];
        squares.push_back(std::move(sq));
    }
    std::vector<int64_t> codes(size_t(n * 8));
    for (int64_t t = 0; t < n; ++t) {
        float residual[32];
        for (int o = 0; o < 32; ++o) {
            double acc = bias[size_t(o)];
            for (int i = 0; i < 512; ++i) acc += double(weight[size_t(o * 512 + i)]) * queries[size_t(t * 512 + i)];
            residual[o] = float(acc);
        }
        for (int q = 0; q < 8; ++q) {
            double x2 = 0;
            for (int d = 0; d < 32; ++d) x2 += double(residual[d]) * residual[d];
            int best = 0;
            double best_distance = INFINITY;
            const float * book = books[size_t(q)].data();
            for (int e = 0; e < 8192; ++e) {
                double xy = 0;
                for (int d = 0; d < 32; ++d) xy += double(residual[d]) * book[e * 32 + d];
                double distance = std::max(x2 + squares[size_t(q)][size_t(e)] - 2.0 * xy, 1e-8);
                if (distance < best_distance) { best_distance = distance; best = e; }
            }
            codes[size_t(t * 8 + q)] = best;
            for (int d = 0; d < 32; ++d) residual[d] -= book[best * 32 + d];
        }
    }
    return codes;
}

} // namespace heartcodec
