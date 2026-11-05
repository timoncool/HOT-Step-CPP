// Native dora-rows-v1 loader acceptance against an independent FP64 reference.
// Build target: sa3-dora-load-test (BUILD_TESTING=ON).
// Run with no arguments for synthetic export/load acceptance. External fixtures
// below add full-model validation after the synthetic cases pass.
// Run: sa3-dora-load-test BASE.gguf ADAPTER1.gguf ADAPTER2.gguf
// Or set SA3_DORA_BASE, SA3_DORA_ADAPTER1 and SA3_DORA_ADAPTER2 for CTest.
// Uses CPU buffers only. Tests every adapted element, including upload/readback.
// No model download, render, server restart, or GPU allocation is required.
// Existing files lack a base fingerprint: geometry is checked, not provenance.

#include "sa3-dit-ggml.h"

#include <stdexcept>
#include <set>

#define SA3_EXPORT_NO_MAIN
#include "test-sa3-adapter-export.cpp"

static void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

static double bf16_value(uint16_t bits) {
    uint32_t word = uint32_t(bits) << 16;
    float value;
    memcpy(&value, &word, sizeof(value));
    return value;
}

struct Files {
    GGUFModel base = {};
    Sa3AdapterSet adapters;
    ~Files() { adapters.close(); gf_close(&base); }
};

struct Weights {
    WeightCtx ctx = {};
    Weights() { wctx_init(&ctx, 1); }
    ~Weights() { wctx_free(&ctx); }
};

// Deliberately double precision and separate from the native merge. B already
// contains alpha/r; applying another scale here would violate dora-rows-v1.
static std::vector<double> reference_delta(const GGUFModel & base, const GGUFModel & ad,
                                          const std::string & name) {
    auto * w = ggml_get_tensor(base.meta, name.c_str());
    auto * d = ggml_get_tensor(ad.meta, (name + ".row_scale").c_str());
    auto * a = ggml_get_tensor(ad.meta, (name + ".lora_a").c_str());
    auto * b = ggml_get_tensor(ad.meta, (name + ".lora_b").c_str());
    require(w && d && a && b, name + ": missing tensor");
    const int64_t rows = d->ne[0], cols = a->ne[0], rank = a->ne[1];
    require(w->type == GGML_TYPE_BF16 && d->type == GGML_TYPE_F32 &&
            a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32, name + ": dtype");
    require(rows > 0 && cols > 0 && rank > 0 && b->ne[0] == rank && b->ne[1] == rows &&
            ggml_nelements(d) == rows && ggml_nelements(a) == cols * rank &&
            ggml_nelements(b) == rows * rank, name + ": factor geometry");
    const bool matrix = w->ne[0] == cols && w->ne[1] == rows && w->ne[2] == 1;
    const bool conv = w->ne[0] == 1 && w->ne[1] == cols && w->ne[2] == rows;
    require((matrix || conv) && ggml_nelements(w) == rows * cols, name + ": base geometry");
    const auto * wp = static_cast<const uint16_t *>(gf_get_data(base, name.c_str()));
    const auto * dp = static_cast<const float *>(gf_get_data(ad, (name + ".row_scale").c_str()));
    const auto * ap = static_cast<const float *>(gf_get_data(ad, (name + ".lora_a").c_str()));
    const auto * bp = static_cast<const float *>(gf_get_data(ad, (name + ".lora_b").c_str()));
    std::vector<double> result(size_t(rows * cols), 0.0);
    for (int64_t row = 0; row < rows; ++row) {
        double * dst = result.data() + row * cols;
        for (int64_t k = 0; k < rank; ++k) {
            for (int64_t col = 0; col < cols; ++col) {
                dst[col] += double(bp[row * rank + k]) * double(ap[k * cols + col]);
            }
        }
        for (int64_t col = 0; col < cols; ++col) {
            const double original = bf16_value(wp[row * cols + col]);
            dst[col] = double(dp[row]) * (original + dst[col]) - original;
            if (!std::isfinite(dst[col])) throw std::runtime_error(name + ": nonfinite reference");
        }
    }
    return result;
}

static std::vector<uint16_t> load(const GGUFModel & base, Sa3AdapterSet * adapters,
                                 const std::string & name, ggml_backend_t backend) {
    Weights weights;
    auto * meta = ggml_get_tensor(base.meta, name.c_str());
    const bool conv = meta->ne[0] == 1 && meta->ne[2] > 1;
    const int64_t shape[] = { meta->ne[1], meta->ne[2] };
    auto * tensor = sa3_load_tensor_adapted(&weights.ctx, base, adapters, name,
                                          conv ? shape : nullptr, conv ? 2 : 0);
    require(tensor && tensor->type == GGML_TYPE_BF16, name + ": loaded type");
    require(ggml_nelements(tensor) == ggml_nelements(meta), name + ": loaded size");
    if (conv) require(tensor->ne[0] == shape[0] && tensor->ne[1] == shape[1],
                      name + ": convolution reshape");
    require(wctx_alloc(&weights.ctx, backend), name + ": upload");
    std::vector<uint16_t> out(size_t(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(uint16_t));
    return out;
}

int main(int argc, char ** argv) {
    try { sa3_export_test::run(); }
    catch (const std::exception & e) {
        fprintf(stderr, "SA3 EXPORT FAIL: %s\n", e.what());
        return 1;
    }
    const char * base_path = argc == 4 ? argv[1] : std::getenv("SA3_DORA_BASE");
    const char * ad1 = argc == 4 ? argv[2] : std::getenv("SA3_DORA_ADAPTER1");
    const char * ad2 = argc == 4 ? argv[3] : std::getenv("SA3_DORA_ADAPTER2");
    if (argc != 1 && argc != 4) {
        fprintf(stderr, "Usage: sa3-dora-load-test BASE.gguf ADAPTER1.gguf ADAPTER2.gguf\n");
        return 1;
    }
    if (!base_path || !ad1 || !ad2) {
        printf("Synthetic export/load passed. Optional full-model checks need SA3_DORA_BASE, SA3_DORA_ADAPTER1, SA3_DORA_ADAPTER2\n");
        return 0;
    }
    ggml_backend_t backend = nullptr;
    try {
        require(std::string(ad1) != ad2, "Supply two distinct existing adapter files");
        Files files;
        require(gf_load(&files.base, base_path), "base load");
        require(files.adapters.open({{ad1, 1.0f}, {ad2, 1.0f}}), "adapter open");
        std::set<std::string> targets[2];
        for (size_t i = 0; i < 2; ++i) {
            const auto & ad = files.adapters.files[i];
            const char * format = gf_get_str(ad, "sa3.adapter_format");
            require(format && std::string(format) == "dora-rows-v1", "adapter format");
            for (int64_t t = 0; t < gguf_get_n_tensors(ad.gguf); ++t) {
                std::string name = gguf_get_tensor_name(ad.gguf, t);
                const std::string suffix = ".row_scale";
                if (name.size() > suffix.size() &&
                    name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
                    targets[i].insert(name.substr(0, name.size() - suffix.size()));
            }
            require(!targets[i].empty() && targets[i].size() * 3 ==
                    size_t(gguf_get_n_tensors(ad.gguf)), "incomplete factor triples");
        }
        require(targets[0] == targets[1], "fixture requires matching adapter target sets");
        ggml_backend_load_all();
        backend = cpu_backend_new(4);
        require(backend != nullptr, "CPU backend unavailable");
        const float strengths[][2] = {
            {0, 0}, {.25f, 0}, {.75f, 0}, {1, 0},
            {0, .25f}, {0, .75f}, {0, 1}, {.25f, .75f}
        };
        uint64_t comparisons = 0, zero_bytes = 0, changed[2] = {};
        double worst_ratio = 0;
        for (const auto & name : targets[0]) {
            const auto original = load(files.base, nullptr, name, backend);
            const auto delta1 = reference_delta(files.base, files.adapters.files[0], name);
            const auto delta2 = reference_delta(files.base, files.adapters.files[1], name);
            // Each single-adapter run actually opens only that adapter; an
            // inactive second file must not hide a broken single-file path.
            for (size_t case_id = 0; case_id < 9; ++case_id) {
                const size_t strength_id = case_id == 8 ? 0 : case_id;
                const float s1 = strengths[strength_id][0], s2 = strengths[strength_id][1];
                Sa3AdapterSet selected;
                const bool second_only = (case_id >= 4 && case_id <= 6) || case_id == 8;
                selected.files = case_id == 7 ? files.adapters.files :
                    std::vector<GGUFModel>{files.adapters.files[second_only ? 1 : 0]};
                selected.names = case_id == 7 ? files.adapters.names :
                    std::vector<std::string>{files.adapters.names[second_only ? 1 : 0]};
                selected.scales = case_id == 7 ? std::vector<float>{s1, s2} :
                    std::vector<float>{second_only ? s2 : s1};
                // selected borrows mmaps; only files.adapters owns/ closes them.
                const auto actual = load(files.base, &selected, name, backend);
                if (s1 == 0 && s2 == 0) {
                    require(actual == original, name + ": strength zero is not bit exact");
                    zero_bytes += actual.size() * sizeof(uint16_t);
                }
                for (size_t j = 0; j < actual.size(); ++j) {
                    const double expected = bf16_value(original[j]) + s1 * delta1[j] + s2 * delta2[j];
                    const double value = bf16_value(actual[j]);
                    // BF16 round-to-nearest is <= 1/256 relative. The absolute
                    // term admits FP32 cancellation near zero in the native merge.
                    const double tolerance = 2e-6 + .004 * std::abs(expected);
                    const double error = std::abs(value - expected);
                    if (!std::isfinite(value) || !std::isfinite(expected) || error > tolerance)
                        throw std::runtime_error(name + ": case " + std::to_string(case_id) +
                            " element " + std::to_string(j) + " reference mismatch");
                    worst_ratio = std::max(worst_ratio, error / tolerance);
                    ++comparisons;
                    if (case_id == 3 && actual[j] != original[j]) ++changed[0];
                    if (case_id == 6 && actual[j] != original[j]) ++changed[1];
                }
            }
            printf("PASS %s (9 cases, all elements)\n", name.c_str());
            fflush(stdout);
        }
        require(changed[0] > 0 && changed[1] > 0, "adapter silently ignored");
        printf("SA3 DORA PASS: targets=%zu cases=9 comparisons=%llu zero_exact_bytes=%llu "
               "changed=%llu/%llu max_error_over_tolerance=%.6g tolerance=2e-6+0.004*abs(reference)\n",
               targets[0].size(), (unsigned long long) comparisons, (unsigned long long) zero_bytes,
               (unsigned long long) changed[0], (unsigned long long) changed[1], worst_ratio);
        printf("LIMIT: geometry/finite weights only; no base fingerprint, cache or audio validation.\n");
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "SA3 DORA FAIL: %s\n", e.what());
        if (backend) ggml_backend_free(backend);
        return 1;
    }
}
