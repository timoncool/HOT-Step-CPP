#pragma once

// yue2-joint-train --fixture-replay <dir>: one pinned joint step against a
// reference fixture, then exit. Nothing is trained or saved.
//
// The fixture directory holds the inputs a reference trainer used for one
// update, written as raw little-endian files beside a JSON record:
//   onestep.json        "nar_window": [start, end], "t": flow time in [0, 1]
//   replay_latent.f32   [frames, 64] latents (BF16 values), frame-major
//   replay_tokens.i32   [frames] raw semantic codes
//   replay_noise.f32    [end - start, 64] flow noise (BF16 values)
//   replay_prefix.i32   Direct-mode prefix ids, ending in ABC_START
// Every random draw of a normal step is replaced: the window, noise and t
// come from the files, the prompt is the off-mode prefix with no sheet, and
// each LoRA A is set from a fixed integer hash (below) with B = 0, so a test
// in any language can rebuild the same adapter.
//
// Reported, as <dir>/replay-native.json plus raw F32 dumps:
//   step 1   losses and the raw (pre-clip) B gradients; A gradients are 0
//            while B = 0, so their max |g| is reported as a check;
//   update   the run's own optimizer (--optimizer, --lr, --planner-lr-scale,
//            --adam-eps ...) after the usual clip at 1.0;
//   step 2   losses again on the same inputs.

#include "yue2-aitk-joint-step.h"
#include "yue2-aitk-sampler.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "yyjson.h"

namespace yue2_aitk_fixture {

// A[r, i] = bound * (2 * h / 2^32 - 1), bound = 1 / sqrt(width),
// h = (r*2654435761 + i*40503 + layer*7919 + site*104729 + expert*1299709) mod 2^32,
// expert 0 = AR, 1 = NAR; site 0..3 = qkv, o, gate_up, down. Computed in
// uint64 and double, then rounded once to float, as the reference does.
inline float a_value(uint64_t r, uint64_t i, int layer, int site, int expert, int64_t width) {
    const uint64_t h = (r * 2654435761ull + i * 40503ull +
                        (uint64_t) (layer * 7919 + site * 104729 + expert * 1299709)) % 4294967296ull;
    return (float) (((double) h / 4294967296.0 * 2.0 - 1.0) / std::sqrt((double) width));
}

template <class T> inline bool read_raw(const std::filesystem::path & path, std::vector<T> * out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    const std::streamsize bytes = in.tellg();
    if (bytes <= 0 || bytes % (std::streamsize) sizeof(T)) return false;
    out->resize((size_t) bytes / sizeof(T));
    in.seekg(0);
    return (bool) in.read(reinterpret_cast<char *>(out->data()), bytes);
}

inline bool write_raw(const std::filesystem::path & path, const std::vector<float> & v) {
    std::ofstream out(path, std::ios::binary);
    return out && out.write(reinterpret_cast<const char *>(v.data()), (std::streamsize) (v.size() * sizeof(float)));
}

inline bool fail(std::string * error, const std::string & message) { if (error) *error = message; return false; }

struct Args {
    std::filesystem::path dir;
    float kl_weight = 0.2f, ar_loss_weight = 1.0f, beta1 = .9f, beta2 = .999f, lr = 1e-4f, weight_decay = 1e-4f;
};

static const char * kSiteNames[4] = {"self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"};

inline bool run(const Args & a, ggml_backend_t backend, const Yue2AitkModel & model, Yue2AitkTrainState & state,
                yue2_aitk::Optimizer * adamw, LmOptim * lm, ggml_backend_sched_t osched, std::string * error) {
    // ── inputs ──
    int64_t start = 0, end = 0; double t = 0;
    {
        const std::string json = (a.dir / "onestep.json").u8string();
        yyjson_doc * doc = yyjson_read_file(json.c_str(), 0, nullptr, nullptr);
        yyjson_val * root = doc ? yyjson_doc_get_root(doc) : nullptr;
        yyjson_val * win = root ? yyjson_obj_get(root, "nar_window") : nullptr;
        yyjson_val * tv = root ? yyjson_obj_get(root, "t") : nullptr;
        const bool ok = win && yyjson_is_arr(win) && yyjson_arr_size(win) == 2 && tv && yyjson_is_num(tv);
        if (ok) { start = yyjson_get_sint(yyjson_arr_get(win, 0)); end = yyjson_get_sint(yyjson_arr_get(win, 1)); t = yyjson_get_num(tv); }
        if (doc) yyjson_doc_free(doc);
        if (!ok || start < 0 || end <= start || !(t > 0 && t < 1)) return fail(error, "fixture onestep.json needs nar_window [start, end] and t in (0, 1)");
    }
    yue2_aitk::SongInput song;
    std::vector<int32_t> prefix;
    std::vector<float> noise;
    if (!read_raw(a.dir / "replay_latent.f32", &song.latents) || !read_raw(a.dir / "replay_tokens.i32", &song.semantic_tokens) ||
        !read_raw(a.dir / "replay_noise.f32", &noise) || !read_raw(a.dir / "replay_prefix.i32", &prefix))
        return fail(error, "fixture replay_{latent,tokens,noise,prefix} files are missing or malformed");
    const size_t frames = (size_t) (end - start);
    if (song.latents.size() != song.semantic_tokens.size() * 64 || (size_t) end > song.semantic_tokens.size() || noise.size() != frames * 64)
        return fail(error, "fixture latent/token/noise sizes do not agree with the window");

    yue2_aitk::PromptInput prompt;
    prompt.retained_prefix_ids = prefix; prompt.dropped_prefix_ids = prefix;
    prompt.retain_abc = false; prompt.retain_caption = true;
    yue2_aitk::Batch batch;
    std::string why;
    if (!yue2_aitk::build(prompt, song, yue2_aitk::FrameRange{(size_t) start, (size_t) end}, 0, &batch, &why, false))
        return fail(error, "fixture batch: " + why);

    // The sampler's BF16 recipe (yue2-aitk-sampler.h), with the stored noise.
    const float t01 = yue2_aitk::bf16_round_f32((float) t);
    yue2_aitk_joint::Input input;
    input.batch = &batch; input.timestep = t01;
    input.noisy_latents.resize(frames * 64); input.flow_target.resize(frames * 64);
    for (size_t i = 0; i < frames * 64; ++i) {
        const float clean = yue2_aitk::bf16_round_f32(batch.nar.latents[i]);
        const float n = yue2_aitk::bf16_round_f32(noise[i]);
        input.flow_target[i] = yue2_aitk::bf16_round_f32(n - clean);
        input.noisy_latents[i] = yue2_aitk::bf16_round_f32((1.0f - t01) * clean + t01 * n);
    }
    input.kl_weight = a.kl_weight; input.ar_loss_weight = a.ar_loss_weight; input.beta1 = a.beta1; input.beta2 = a.beta2;
    input.adamw_lr = a.lr; input.adamw_weight_decay = a.weight_decay; input.measure_kl = true;
    if (lm) lm->base_lr = a.lr;  // constant rate; the replay is one update

    // ── adapter: A from the hash, B = 0 ──
    struct Site { std::string key; int expert, layer, site; ggml_tensor * a, * b, * gb, * ga; };
    std::vector<Site> sites;
    {
        const auto named = state.named_tensors();
        auto find = [&](const std::string & n) -> const Yue2AitkTrainState::NamedTensors * {
            for (const auto & x : named) if (x.name == n) return &x;
            return nullptr;
        };
        for (int expert = 0; expert < 2; ++expert) for (int layer = 0; layer < Yue2AitkTrainState::kLayers; ++layer) for (int s = 0; s < 4; ++s) {
            const std::string key = std::string(expert == 0 ? "text_encoders" : "diffusion_model") + ".model.layers." +
                                    std::to_string(layer) + "." + kSiteNames[s];
            const auto * na = find(key + ".lora_A.weight"); const auto * nb = find(key + ".lora_B.weight");
            if (!na || !nb) return fail(error, "fixture replay needs a LoRA adapter (" + key + " not found)");
            sites.push_back({key, expert, layer, s, na->parameter, nb->parameter, nb->gradient, na->gradient});
        }
    }
    for (const Site & s : sites) {
        const int64_t width = s.a->ne[0], rank = s.a->ne[1];
        std::vector<float> va((size_t) (width * rank));
        for (int64_t r = 0; r < rank; ++r) for (int64_t i = 0; i < width; ++i) va[(size_t) (r * width + i)] = a_value((uint64_t) r, (uint64_t) i, s.layer, s.site, s.expert, width);
        ggml_backend_tensor_set(s.a, va.data(), 0, ggml_nbytes(s.a));
        ggml_backend_tensor_memset(s.b, 0, 0, ggml_nbytes(s.b));
    }
    ggml_backend_synchronize(backend);

    auto get = [](ggml_tensor * x) { std::vector<float> v((size_t) ggml_nelements(x)); ggml_backend_tensor_get(x, v.data(), 0, ggml_nbytes(x)); return v; };
    auto norm = [](const std::vector<float> & v) { double s = 0; for (float x : v) s += (double) x * x; return std::sqrt(s); };
    const size_t samples[3] = {0, 3, 110};
    auto sample_name = [](const Site & s, size_t k) { return std::string(s.expert == 0 ? "ar." : "nar.") + std::to_string(k); };
    auto step = [&](bool finalize, yue2_aitk_joint::Metrics * m) {
        input.accumulate = false; input.finalize = finalize; input.accum_scale = 1.0f;
        return yue2_aitk_joint::run(backend, model, state, adamw, input, m, error, {}, lm, osched);
    };

    // ── step 1, gradients only ──
    yue2_aitk_joint::Metrics s1, upd, s2;
    if (!step(false, &s1)) return false;
    ggml_backend_synchronize(backend);
    std::vector<double> bnorm[2]; double amax = 0;
    for (size_t k = 0; k < sites.size(); ++k) {
        const Site & s = sites[k];
        const auto gb = get(s.gb); bnorm[s.expert].push_back(norm(gb));
        for (float x : get(s.ga)) amax = std::fmax(amax, std::fabs((double) x));
        const size_t local = k % (sites.size() / 2);
        for (size_t q : samples) if (q == local && !write_raw(a.dir / ("native_gradB_" + sample_name(s, q) + ".f32"), gb)) return fail(error, "cannot write gradient dump");
    }
    // ── one real update (recomputes step 1, then clips and steps) ──
    if (!step(true, &upd)) return false;
    ggml_backend_synchronize(backend);
    for (size_t k = 0; k < sites.size(); ++k) {
        const size_t local = k % (sites.size() / 2);
        for (size_t q : samples) if (q == local && !write_raw(a.dir / ("native_postB_" + sample_name(sites[k], q) + ".f32"), get(sites[k].b))) return fail(error, "cannot write parameter dump");
    }
    // ── step 2, same inputs ──
    if (!step(false, &s2)) return false;

    const std::string out_path = (a.dir / "replay-native.json").u8string();
    FILE * f = std::fopen(out_path.c_str(), "wb");
    if (!f) return fail(error, "cannot write replay-native.json");
    auto losses = [&](const char * name, const yue2_aitk_joint::Metrics & m) {
        const double loss = m.nar_mse + (double) a.ar_loss_weight * m.ar_ce + (double) a.kl_weight * m.ar_kl;
        std::fprintf(f, " \"%s\": {\"loss\": %.9g, \"flow\": %.9g, \"ar_ce\": %.9g, \"ar_kl\": %.9g, \"ar_targets\": %zu},\n",
                     name, loss, m.nar_mse, m.ar_ce, m.ar_kl, batch.ar.target_ids.size());
    };
    std::fprintf(f, "{\n");
    losses("step1", s1);
    losses("step1_repeat_in_update", upd);
    std::fprintf(f, " \"step1_grad_norm_preclip\": %.9g,\n \"A_grad_max_abs\": %.9g,\n", upd.gradient_norm, amax);
    for (int e = 0; e < 2; ++e) {
        std::fprintf(f, " \"%s_B_grad_norms\": [", e == 0 ? "ar" : "nar");
        for (size_t i = 0; i < bnorm[e].size(); ++i) std::fprintf(f, "%s%.9g", i ? ", " : "", bnorm[e][i]);
        std::fprintf(f, "],\n");
    }
    losses("step2_same_inputs", s2);
    std::fprintf(f, " \"timestep_bf16\": %.9g, \"nar_window\": [%lld, %lld]\n}\n", (double) t01, (long long) start, (long long) end);
    std::fclose(f);
    std::fprintf(stderr, "[yue2-aitk] fixture replay: step1 ar_ce %.6f ar_kl %.3g nar_mse %.6f | grad norm %.6f | step2 ar_ce %.6f ar_kl %.4g nar_mse %.6f -> %s\n",
                 s1.ar_ce, s1.ar_kl, s1.nar_mse, upd.gradient_norm, s2.ar_ce, s2.ar_kl, s2.nar_mse, out_path.c_str());
    return true;
}

} // namespace yue2_aitk_fixture
