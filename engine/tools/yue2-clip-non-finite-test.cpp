// yue2-clip-non-finite-test.cpp -- S2d-1 gate 3 (RESEARCH/YUE2_S2D_TOLERANCES.md):
//
// Case "inf": a gradient containing one non-finite value must still fail
// Yue2AitkTrainState::clip_gradients() with "non-finite gradient before
// clipping", on both the device path (default) and the host rollback
// (YUE2_AITK_CLIP_HOST=1).
//
// Case "huge": a gradient containing one *finite* but extreme value (1e20)
// must NOT be rejected -- squaring it in F32 (1e40) would overflow to inf,
// which is why clip_gradients_device() normalizes by the global max |value|
// before squaring instead of squaring raw values. Checks the device path's
// reported norm against the host path's (F64 accumulation, no overflow
// concern) within a tight relative tolerance.
//
// Case "clipped": max_norm is set below the gradient's actual norm (with
// pre_scale != 1), so the clip must actually engage (scale < 1), unlike the
// gate recipes' real runs where every observed gradient_norm stayed under
// their max_norm. Checks the post-clip gradient's norm lands at max_norm.
//
// Case "extreme": a single finite 1e26 spike -- well past "huge"'s 1e20,
// chosen so a *fixed* pre-square scale (rev 2 of this fix) still overflows
// (1e26 * 1e-6 = 1e20, squares to 1e40 > F32 max) while the true-max
// normalization (rev 3) does not, since it divides by whatever the max
// actually is, not a constant.
//
// Case "dense_large": every element at 1e19, not just one -- large enough
// that summing just two of their raw squares (1e38 each) overflows F32,
// so this catches a fix that only normalizes a single spike but still
// overflows the *sum* of many large-but-individually-safe squared terms.
//
// Case "near_limit": every element at 1e38, near the F32 limit (~3.4e38).
// Normalizing each term by the true max (rev 3) keeps every square <=1, so
// the sum-of-squares never overflows -- but reconstructing the final norm
// as max * sqrt(sum) in F32 does: max ~1e38 times sqrt(n) ~2e3 is ~2e41,
// over FLT_MAX. This catches a fix that normalizes correctly but still
// recombines the factored norm in F32 instead of host double. Its norm
// (~2e41) is itself above the 1e30 max_norm sentinel the other cases use,
// so unlike them it actually clips -- also exercising clip_gradients_host's
// max_norm/length division at that scale, where casting length to float
// before dividing saturates to inf and silently zeros the gradient instead
// of clipping it to max_norm.

#include "train/yue2-aitk-train-state.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

struct Result { bool ok; double norm; double post_norm; std::string error; };

// Puts the backend's own env var exactly where clip_gradients() reads it
// (a static local, latched on first call), so each (path, case) combination
// needs its own process.
Result run_one(bool host_path, const std::string & kase, bool gpu = false) {
#ifdef _WIN32
    _putenv_s("YUE2_AITK_CLIP_HOST", host_path ? "1" : "");
#else
    if (host_path) setenv("YUE2_AITK_CLIP_HOST", "1", 1); else unsetenv("YUE2_AITK_CLIP_HOST");
#endif
    ggml_backend_load_all();
    // gpu: the first GPU, which on a CUDA build runs the CUDA clip kernels.
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(gpu ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) return {false, 0.0, 0.0, "no device found"};
    ggml_backend_t backend = ggml_backend_dev_init(cpu_dev, nullptr);
    if (!backend) return {false, 0.0, 0.0, "backend init failed"};

    Yue2AitkTrainState state;
    std::string error;
    // cursor=true gives one plain 2048x2048 gradient slot via
    // cursor_gradient(), with no adapter plumbing needed to reach it.
    if (!state.initialize(backend, Yue2AitkTrainState::kDefaultSeed, &error, /*cursor=*/true)) {
        ggml_backend_free(backend);
        return {false, 0.0, 0.0, "initialize failed: " + error};
    }

    ggml_tensor * grad = state.cursor_gradient();
    const size_t n = ggml_nelements(grad);
    std::vector<float> values(n, 0.01f);
    float max_norm = 1e30f;   // huge: never clips, so the reported norm is the raw value
    float pre_scale = 1.0f;
    if (kase == "inf") {
        values[n / 2] = std::numeric_limits<float>::infinity();
    } else if (kase == "huge") {
        values[n / 2] = 1e20f;
    } else if (kase == "extreme") {
        values[n / 2] = 1e26f;
    } else if (kase == "dense_large") {
        std::fill(values.begin(), values.end(), 1e19f);
    } else if (kase == "near_limit") {
        std::fill(values.begin(), values.end(), 1e38f);
    } else if (kase == "clipped") {
        // Uniform 2.0 everywhere: length = pre_scale * 2.0 * sqrt(n), well
        // above max_norm, so the clip must actually engage.
        std::fill(values.begin(), values.end(), 2.0f);
        pre_scale = 0.5f;
        max_norm = 1.0f;
    }
    ggml_backend_tensor_set(grad, values.data(), 0, ggml_nbytes(grad));

    double norm = 0.0;
    std::string clip_error;
    const bool ok = state.clip_gradients(max_norm, &norm, &clip_error, pre_scale);

    double post_norm = 0.0;
    if (ok) {
        std::vector<float> after(n);
        ggml_backend_tensor_get(grad, after.data(), 0, ggml_nbytes(grad));
        double sq = 0.0;
        for (float v : after) sq += double(v) * double(v);
        post_norm = std::sqrt(sq);
    }
    ggml_backend_free(backend);
    return {ok, norm, post_norm, clip_error};
}

} // namespace

int main(int argc, char ** argv) {
    const char * which_path = std::getenv("YUE2_CLIP_TEST_PATH");
    const char * which_case = std::getenv("YUE2_CLIP_TEST_CASE");
    if (which_path && which_case) {
        const bool host_path = std::string(which_path) == "host";
        const bool cuda_path = std::string(which_path) == "cuda";
        const std::string kase = which_case;
        const Result r = run_one(host_path, kase, cuda_path);
        const char * path_name = host_path ? "host" : cuda_path ? "cuda" : "device";
        if (kase == "huge" || kase == "extreme" || kase == "dense_large" || kase == "near_limit") {
            // Expect success, norm close to a known-by-construction value:
            // "huge"/"extreme" are a single spike (the other 2048*2048-1
            // elements at 0.01 contribute negligibly); "dense_large" and
            // "near_limit" are uniform, so norm = sqrt(n)*value exactly.
            const double n_elems = 2048.0 * 2048.0;
            const double expected = kase == "huge" ? 1e20 : kase == "extreme" ? 1e26
                : kase == "dense_large" ? std::sqrt(n_elems) * 1e19 : std::sqrt(n_elems) * 1e38;
            if (!r.ok) { std::fprintf(stderr, "FAIL (%s/%s): rejected a finite gradient: %s\n", path_name, which_case, r.error.c_str()); return 1; }
            const double rel = std::fabs(r.norm - expected) / expected;
            if (!std::isfinite(r.norm) || rel > 1e-5) { std::fprintf(stderr, "FAIL (%s/%s): norm=%.6g (rel diff %.3g)\n", path_name, which_case, r.norm, rel); return 1; }
            if (kase == "near_limit") {
                // Unlike huge/extreme/dense_large, near_limit's norm
                // (~2e41) is itself above the 1e30 max_norm sentinel, so
                // the clip actually engages here. This is what catches the
                // host path's pre-fix float(length) cast: that saturated
                // to inf, so max_norm/inf rounded to a scale of exactly 0
                // instead of the correct ~4.9e-12, zeroing the gradient
                // instead of clipping it to max_norm.
                const double rel_to_max = std::fabs(r.post_norm - double(1e30f)) / double(1e30f);
                if (!std::isfinite(r.post_norm) || rel_to_max > 1e-4) { std::fprintf(stderr, "FAIL (%s/near_limit): post-clip norm=%.6g (expected ~1e30)\n", path_name, r.post_norm); return 1; }
            }
            std::printf("PASS (%s/%s): norm=%.6g (rel diff %.3g to %.6g)\n", path_name, which_case, r.norm, rel, expected);
            return 0;
        } else if (kase == "clipped") {
            // pre_scale=0.5, uniform 2.0 -> pre-clip length = 0.5*2.0*sqrt(n);
            // max_norm=1.0 is far below that, so the clip must engage and
            // the post-clip gradient's own norm must land at ~max_norm.
            if (!r.ok) { std::fprintf(stderr, "FAIL (%s/clipped): rejected a finite gradient: %s\n", path_name, r.error.c_str()); return 1; }
            const double rel_to_max = std::fabs(r.post_norm - 1.0) / 1.0;
            if (!std::isfinite(r.post_norm) || rel_to_max > 1e-4) { std::fprintf(stderr, "FAIL (%s/clipped): post-clip norm=%.6g (expected ~1.0)\n", path_name, r.post_norm); return 1; }
            std::printf("PASS (%s/clipped): pre-clip norm=%.6g, post-clip norm=%.6g (expected ~1.0)\n", path_name, r.norm, r.post_norm);
            return 0;
        } else {
            if (r.ok) { std::fprintf(stderr, "FAIL (%s/inf): accepted a non-finite gradient\n", path_name); return 1; }
            if (r.error != "non-finite gradient before clipping") { std::fprintf(stderr, "FAIL (%s/inf): wrong error message: %s\n", path_name, r.error.c_str()); return 1; }
            std::printf("PASS (%s/inf): clip_gradients rejected the non-finite gradient (\"%s\")\n", path_name, r.error.c_str());
            return 0;
        }
    }

    const std::string self = argv[0];
    bool all_ok = true;
    // "cuda": the device path on a CUDA GPU (the trainer's CUDA clip kernels);
    // only on a CUDA build with a CUDA device.
    bool cuda = false;
#if YUE2_AITK_CUDA_RUNTIME
    ggml_backend_load_all();
    if (ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU))
        cuda = std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(gpu)), "CUDA") == 0;
#endif
    if (!cuda) std::printf("SKIP (cuda): no CUDA build or device\n");
    for (const char * path : {"device", "host", "cuda"}) {
        if (!cuda && std::strcmp(path, "cuda") == 0) continue;
        for (const char * kase : {"inf", "huge", "extreme", "dense_large", "near_limit", "clipped"}) {
#ifdef _WIN32
            _putenv_s("YUE2_CLIP_TEST_PATH", path);
            _putenv_s("YUE2_CLIP_TEST_CASE", kase);
#else
            setenv("YUE2_CLIP_TEST_PATH", path, 1);
            setenv("YUE2_CLIP_TEST_CASE", kase, 1);
#endif
            const int rc = std::system(("\"" + self + "\"").c_str());
            if (rc != 0) { std::fprintf(stderr, "child (%s/%s) failed (rc=%d)\n", path, kase, rc); all_ok = false; }
        }
    }
    return all_ok ? 0 : 1;
}
