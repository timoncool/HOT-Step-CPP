#pragma once
// heartmula/adapter.h: runtime LoRA for the HeartMuLa global (backbone) LM.
//
// HOT-Step file. Reads the GGUF written by engine/tools/convert-heartmula-lora.py
// (engine/docs/HEARTMULA_CONTRACT.md, "LoRA adapter"). The factors stay apart from the
// base weights, so one adapter works over any quant of its base:
//
//   y = W x + (alpha / rank) x scale x B (A x)
//
// where scale is the master dial x the attention or MLP dial x the dial of the
// layer's depth third (HeartMuLaAdapterScales, YuE2's grouping).
//
// The factor math is qwen3-lora.h's (qwen3_linear_lora); this file only loads and
// checks. Targets are backbone q/k/v/output projections and, for adapters trained
// with --lora-mlp, the backbone mlp w1/w2/w3; the local decoder is never adapted. Accepted only when the variant and base fingerprint equal the loaded
// model's, the component is a global-targeting LoRA, `heartmula.lora.targets` is
// exactly the set of factor tensors present, and every shape fits its base weight.
//
// Basin re-base (adapter-merge.h's, done at runtime). An adapter trained on another
// variant can attach when its home global is installed: the home weights S of the
// targeted projections load with it, and each targeted projection computes
//
//   (1 - beta) W x + beta S x + LoRA
//
// which is the adapter-merge.h nudge W + beta (S - W) applied to x without touching W,
// so any quant of either base works. beta is a per-request dial (HeartMuLaAdapterScales).
//
// Every load takes a new id. The LM graph (lm-graph.h) keys its built graphs and its
// KV rows on (id, scales), so switching, reloading or re-scaling the adapter drops them.

#include "heartmula/model.h"
#include "qwen3-lora.h"

#include <array>
#include <atomic>
#include <cmath>
#include <set>
#include <string>
#include <vector>

// Per-group dials. The effective scale of a site is global x (attn or mlp) x its third.
struct HeartMuLaAdapterScales {
    float global = 1.0f, attn = 1.0f, mlp = 1.0f, early = 1.0f, mid = 1.0f, late = 1.0f;
    float rebase = 0.75f;  // basin re-base beta; read only by an adapter with home weights

    bool operator==(const HeartMuLaAdapterScales & o) const {
        return global == o.global && attn == o.attn && mlp == o.mlp && early == o.early && mid == o.mid &&
               late == o.late && rebase == o.rebase;
    }
    bool operator!=(const HeartMuLaAdapterScales & o) const { return !(*this == o); }

    // Thirds as YuE2 splits them: [0, n/3), [n/3, 2n/3), the rest (28 layers: 0-8, 9-17, 18-27).
    float site(int layer, int n_layer, bool mlp_site) const {
        const float third = layer < n_layer / 3 ? early : layer < 2 * n_layer / 3 ? mid : late;
        return global * (mlp_site ? mlp : attn) * third;
    }
};

struct HeartMuLaAdapter {
    WeightCtx                wctx = {};
    std::vector<QwLoraLayer> layers;  // per backbone layer; slots Q/K/V/O[/GATE/DOWN/UP], scale = alpha / rank
    int                      rank  = 0;
    float                    alpha = 0.0f;
    uint64_t                 id    = 0;  // 0 = not loaded
    bool                     has_mlp = false;
    // Re-base: the home global's weights per layer, in site order (proj_name); empty when
    // the adapter was trained on the loaded base.
    std::vector<std::array<ggml_tensor *, 7>> home;
    std::string              home_path;
    std::string              attached;  // variant|fingerprint of the model it was loaded over
    std::string              path, variant, base_fingerprint;
    bool                     loaded() const { return id != 0; }
};

static void heartmula_adapter_free(HeartMuLaAdapter * a) {
    wctx_free(&a->wctx);
    *a = HeartMuLaAdapter{};
}

namespace heartmula_adapter_detail {

inline bool fail(std::string * err, const std::string & msg) {
    if (err) {
        *err = msg;
    }
    return false;
}

// The trainer's site order (train/heartmula-train-graph.h HmtLayerLora).
inline const char * proj_name(int slot) {
    static const char * names[7] = { "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.output_proj",
                                     "mlp.w1", "mlp.w2", "mlp.w3" };
    return names[slot];
}

inline QwLoraSlot proj_slot(int slot) {
    static const QwLoraSlot slots[7] = { QW_LORA_Q, QW_LORA_K, QW_LORA_V, QW_LORA_O, QW_LORA_GATE, QW_LORA_DOWN, QW_LORA_UP };
    return slots[slot];
}

// `["a", "b"]` of plain strings (the converter writes json.dumps of tensor names).
inline bool parse_targets(const std::string & s, std::vector<std::string> * out) {
    size_t i = s.find_first_not_of(" \t\r\n");
    if (i == std::string::npos || s[i] != '[') {
        return false;
    }
    for (++i;;) {
        i = s.find_first_not_of(" \t\r\n", i);
        if (i == std::string::npos) {
            return false;
        }
        if (s[i] == ']') {
            return s.find_first_not_of(" \t\r\n", i + 1) == std::string::npos;
        }
        if (!out->empty()) {
            if (s[i] != ',') {
                return false;
            }
            i = s.find_first_not_of(" \t\r\n", i + 1);
            if (i == std::string::npos) {
                return false;
            }
        }
        if (s[i] != '"') {
            return false;
        }
        const size_t e = s.find('"', i + 1);
        if (e == std::string::npos) {
            return false;
        }
        const std::string name = s.substr(i + 1, e - i - 1);
        if (name.find('\\') != std::string::npos) {
            return false;
        }
        out->push_back(name);
        i = e + 1;
    }
}

}  // namespace heartmula_adapter_detail

// Load `path` for model `m`. On failure `a` is left empty and `err` names the file and cause.
// `home` is the adapter's home global GGUF when it was trained on another base (re-base).
static bool heartmula_adapter_load(HeartMuLaAdapter * a, const HeartMuLaModel & m, const char * path,
                                   std::string * err, const char * home = nullptr) {
    using namespace heartmula_adapter_detail;
    static std::atomic<uint64_t> next_id{ 1 };
    heartmula_adapter_free(a);
    const std::string file = path;
    const auto        bad  = [&](const std::string & why) { return fail(err, file + ": " + why); };
    if (!m.loaded) {
        return bad("no HeartMuLa model is loaded to attach the adapter to");
    }
    GGUFModel gf = {};
    if (!gf_load(&gf, path)) {
        return bad("cannot open adapter");
    }
    const auto done = [&](bool ok) {
        gf_close(&gf);
        if (!ok) {
            heartmula_adapter_free(a);
        }
        return ok;
    };
    if (std::string(gf_get_str(gf, "general.architecture")) != "heartmula-lora" ||
        std::string(gf_get_str(gf, "heartmula.component")) != "lora") {
        done(false);
        return bad("not a HeartMuLa LoRA adapter (architecture/component)");
    }
    if (gf_get_u32(gf, "heartmula.format_version") != HEARTMULA_FORMAT_VERSION) {
        done(false);
        return bad("unsupported heartmula.format_version");
    }
    if (std::string(gf_get_str(gf, "heartmula.lora.target_component")) != "global") {
        done(false);
        return bad("heartmula.lora.target_component is not 'global'");
    }
    const std::string variant = gf_get_str(gf, "heartmula.variant"),
                      fp      = gf_get_str(gf, "heartmula.base_fingerprint");
    GGUFModel hg = {};  // the home global, open until the weights are allocated
    const bool rebase = home && home[0];
    if (rebase) {
        if (!gf_load(&hg, home)) {
            done(false);
            return bad(std::string("cannot open the home model ") + home);
        }
        const bool same = std::string(gf_get_str(hg, "heartmula.variant")) == variant &&
                          std::string(gf_get_str(hg, "heartmula.base_fingerprint")) == fp;
        if (!same) {
            gf_close(&hg);
            done(false);
            return bad(std::string(home) + " is not the global model this adapter was trained on");
        }
    } else if (variant != m.variant) {
        done(false);
        return bad("variant is '" + variant + "', the loaded model is '" + m.variant + "'");
    } else if (fp != m.base_fingerprint) {
        done(false);
        return bad("heartmula.base_fingerprint differs from the loaded global model's");
    }
    const auto close_home = [&](bool ok) {
        if (rebase) gf_close(&hg);
        return done(ok);
    };
    const int   rank  = (int) gf_get_u32(gf, "heartmula.lora.rank");
    const float alpha = gf_get_f32(gf, "heartmula.lora.alpha");
    if (rank < 1 || !std::isfinite(alpha) || alpha <= 0.0f) {
        close_home(false);
        return bad("heartmula.lora.rank/alpha are missing or invalid");
    }

    // Targets: exactly the projections that carry factors, all inside the backbone set.
    std::vector<std::string> targets;
    if (!parse_targets(gf_get_str(gf, "heartmula.lora.targets"), &targets) || targets.empty()) {
        close_home(false);
        return bad("heartmula.lora.targets is missing or not a JSON array of names");
    }
    const int                n_layer = m.global.cfg.n_layer;
    std::set<std::string>    allowed, listed(targets.begin(), targets.end());
    for (int i = 0; i < n_layer; i++) {
        for (int s = 0; s < 7; s++) {
            allowed.insert("backbone.layers." + std::to_string(i) + "." + proj_name(s) + ".weight");
        }
    }
    for (const auto & t : targets) {
        if (!allowed.count(t)) {
            close_home(false);
            return bad("target '" + t + "' is not a backbone attention or MLP projection of this model");
        }
    }
    if (listed.size() != targets.size()) {
        close_home(false);
        return bad("heartmula.lora.targets lists a tensor twice");
    }
    if ((size_t) gguf_get_n_tensors(gf.gguf) != 2 * targets.size()) {
        close_home(false);
        return bad("tensor count does not match heartmula.lora.targets");
    }

    a->layers.assign((size_t) n_layer, QwLoraLayer{});
    if (rebase) a->home.assign((size_t) n_layer, std::array<ggml_tensor *, 7>{});
    wctx_init(&a->wctx, (int) ((rebase ? 3 : 2) * targets.size()) + 1);
    const float scale = alpha / (float) rank;
    for (int i = 0; i < n_layer; i++) {
        const HeartMuLaLayer & base = m.global.layers[(size_t) i];
        ggml_tensor * const    w[7] = { base.q, base.k, base.v, base.o, base.w1, base.w2, base.w3 };
        for (int s = 0; s < 7; s++) {
            const std::string name = "backbone.layers." + std::to_string(i) + "." + proj_name(s) + ".weight";
            if (!listed.count(name)) {
                continue;
            }
            ggml_tensor * ta = ggml_get_tensor(gf.meta, (name + ".lora_a").c_str());
            ggml_tensor * tb = ggml_get_tensor(gf.meta, (name + ".lora_b").c_str());
            if (!ta || !tb) {
                close_home(false);
                return bad("target '" + name + "' lacks its .lora_a/.lora_b pair");
            }
            const bool type_ok = (ta->type == GGML_TYPE_F32 || ta->type == GGML_TYPE_F16) &&
                                 (tb->type == GGML_TYPE_F32 || tb->type == GGML_TYPE_F16);
            // ggml order: A [in, rank], B [rank, out]; the base weight is [in, out]. Rank 1 makes A
            // a 1-D tensor to ggml_n_dims, so only the trailing dimensions are checked.
            if (!type_ok || ta->ne[0] != w[s]->ne[0] || ta->ne[1] != rank || tb->ne[0] != rank ||
                tb->ne[1] != w[s]->ne[1] || ta->ne[2] != 1 || ta->ne[3] != 1 || tb->ne[2] != 1 || tb->ne[3] != 1) {
                close_home(false);
                return bad("'" + name + "' factors do not fit the base weight at rank " + std::to_string(rank));
            }
            QwLoraPair & p = a->layers[(size_t) i].p[proj_slot(s)];
            a->has_mlp |= s >= 4;
            p.A     = gf_load_tensor(&a->wctx, gf, name + ".lora_a");
            p.B     = gf_load_tensor(&a->wctx, gf, name + ".lora_b");
            p.scale = scale;
            if (rebase) {
                // The home weight must have the loaded base's shape: same family, another variant.
                ggml_tensor * th = ggml_get_tensor(hg.meta, name.c_str());
                if (!th || th->ne[0] != w[s]->ne[0] || th->ne[1] != w[s]->ne[1] || th->ne[2] != 1 || th->ne[3] != 1) {
                    close_home(false);
                    return bad("home model weight '" + name + "' is missing or shaped unlike the loaded base");
                }
                a->home[(size_t) i][(size_t) s] = gf_load_tensor(&a->wctx, hg, name);
            }
        }
    }
    if (!wctx_alloc(&a->wctx, m.bp.backend)) {
        close_home(false);
        return bad("adapter weight allocation failed (out of VRAM?)");
    }
    a->rank             = rank;
    a->alpha            = alpha;
    a->path             = file;
    a->variant          = variant;
    a->base_fingerprint = fp;
    a->home_path        = rebase ? home : "";
    a->attached         = m.variant + "|" + m.base_fingerprint;
    a->id               = next_id.fetch_add(1);
    if (rebase) {
        fprintf(stderr, "[HeartMuLa] adapter %s: trained on %s, re-based onto %s from %s (%zu targets)\n",
                file.c_str(), variant.c_str(), m.variant.c_str(), home, targets.size());
    }
    return close_home(true);
}
