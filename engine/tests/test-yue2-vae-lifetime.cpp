#include "yue2/yue2-vae-graph.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

static bool check(bool ok, const char * message) {
    if (!ok) fprintf(stderr, "FAIL: %s\n", message);
    return ok;
}

// Real decoder allocations, without the LM or a full song. Run with the
// installed models directory; GGML_BACKEND selects CPU/CUDA0/Vulkan0.
int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s MODELS_DIR\n", argv[0]);
        return 2;
    }

    Yue2Model empty;
    yue2_vae_decode_cache(empty);
    std::weak_ptr<Yue2VaeGraph> empty_cache = empty.vae_decode_cache;
    yue2_unload(&empty);
    if (!check(empty_cache.expired(), "unload must clear a cache even without resident weights")) return 1;

    Yue2Model m;
    yue2_discover(&m, argv[1]);
    std::string error;
    const int64_t frames = 4;
    std::vector<float> latent(64 * frames, 0.0f), audio;
    int64_t samples = 0;
    const int initial_refs = g_backend_refs;

    for (int cycle = 0; cycle < 2; ++cycle) {
        std::weak_ptr<Yue2VaeGraph> previous;
        for (auto variant : { YUE2_VAE_STANDARD, YUE2_VAE_LEGACY }) {
            if (!yue2_load_parts(&m, false, false, true, variant, false, &error)) {
                fprintf(stderr, "FAIL: load: %s\n", error.c_str());
                yue2_unload(&m);
                return 1;
            }
            if (!check(previous.expired(), "variant switch must destroy the previous decoder cache")) return 1;
            if (!yue2_vae_decode(m, latent.data(), frames, &audio, &samples, &error)) {
                fprintf(stderr, "FAIL: decode: %s\n", error.c_str());
                yue2_unload(&m);
                return 1;
            }
            if (!check(samples > 0 && audio.size() == (size_t) (2 * samples), "decoder output shape")) return 1;
            for (float value : audio) {
                if (!check(std::isfinite(value), "decoder output must be finite")) return 1;
            }
            previous = m.vae_decode_cache;
            auto * cached = m.vae_decode_cache.get();
            if (!check(cached && cached->sched && cached->prep.buffer, "decoder must allocate real compute resources")) return 1;
            if (!yue2_vae_decode_tiled(m, latent.data(), frames, 512, m.vae_cfg.decode_halo_frames,
                                      &audio, &samples, nullptr, &error)) {
                fprintf(stderr, "FAIL: tiled decode: %s\n", error.c_str());
                yue2_unload(&m);
                return 1;
            }
            if (!check(m.vae_decode_cache.get() == cached, "resident model should reuse its decoder cache")) return 1;
            fprintf(stderr, "PASS: cycle %d %s decode, tiled decode and cache reuse\n", cycle + 1,
                    YUE2_VAE_VARIANT_NAME[variant]);
        }
        yue2_unload(&m);
        if (!check(previous.expired() && !m.vae_decode_cache, "unload must destroy the decoder cache")) return 1;
        if (!check(g_backend_refs == initial_refs, "unload must release the model backend reference")) return 1;
        yue2_unload(&m);
        if (!check(g_backend_refs == initial_refs, "repeated unload must be safe")) return 1;
    }
    fprintf(stderr, "PASS: YuE2 VAE lifetime regression\n");
    return 0;
}
