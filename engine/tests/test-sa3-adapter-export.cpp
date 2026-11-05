// Synthetic CPU-only writer/reader round trip. Also included by the existing
// sa3-dora-load-test target so CTest exercises export without model downloads.
#include "sa3-dit-ggml.h"
#include "train/sa3-adapter-export.h"
#include <chrono>
#include <stdexcept>

namespace sa3_export_test {
inline void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> void rejects(F fn) {
    bool failed = false;
    try { fn(); } catch (const std::invalid_argument &) { failed = true; }
    check(failed, "invalid export accepted");
}
struct Model {
    GGUFModel gf{};
    explicit Model(const std::string & path) { check(gf_load(&gf, path.c_str()), "fixture load"); }
    ~Model() { gf_close(&gf); }
};
struct Scratch {
    std::filesystem::path path;
    Scratch() {
        path = std::filesystem::current_path() / ("sa3-export-test-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        check(std::filesystem::create_directory(path), "fixture directory");
    }
    ~Scratch() { std::error_code ec; std::filesystem::remove_all(path, ec); }
    std::string file(const std::string & name) { return (path / name).u8string(); }
};
inline std::vector<float> merged(const GGUFModel & base, const std::string & path,
                                  const std::string & name, ggml_backend_t cpu) {
    Sa3AdapterSet adapters;
    check(adapters.open({{path, 1.f}}), "open exported adapter");
    WeightCtx weights{};
    wctx_init(&weights, 1);
    auto * t = sa3_load_tensor_adapted(&weights, base, &adapters, name);
    check(t && wctx_alloc(&weights, cpu), "native merge/upload");
    std::vector<ggml_bf16_t> data(size_t(ggml_nelements(t)));
    ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(ggml_bf16_t));
    std::vector<float> result(data.size());
    ggml_bf16_to_fp32_row(data.data(), result.data(), int64_t(data.size()));
    wctx_free(&weights);
    adapters.close();
    return result;
}
inline void run() {
    using namespace sa3_export;
    Scratch scratch;
    const std::string names[] = {"model.transformer.layers.0.self_attn.to_qkv.weight",
                                 "model.preprocess_conv.weight"};
    // Intentionally write BF16 base values distinct from any training base.
    const float base_values[] = {3, 4, 0, 0, 0, 2};
    std::vector<ggml_bf16_t> bf16(6);
    ggml_fp32_to_bf16_row_ref(base_values, bf16.data(), 6);
    auto * ctx = ggml_init({4096, nullptr, true});
    auto * g = gguf_init_empty();
    for (int i = 0; i < 2; ++i) {
        auto * t = i == 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 3, 2) :
                           ggml_new_tensor_3d(ctx, GGML_TYPE_BF16, 1, 3, 2);
        ggml_set_name(t, names[i].c_str());
        t->data = bf16.data();
        gguf_add_tensor(g, t);
    }
    gguf_set_val_str(g, "general.architecture", "sa3-dit");
    check(gguf_write_to_file(g, scratch.file("base.gguf").c_str(), false), "write base");
    gguf_free(g); ggml_free(ctx);
    Model base(scratch.file("base.gguf"));
    Rows p{"dit.0.self.qkv", 3, 2, 2, 6.f, {1, -2, .5f, 0, 1, 2}, {0, 0, 0, 0}, {5, 2}};
    Rows conv = p; conv.stem = "dit.pre_conv";
    std::vector<Rows> identity{p, conv};
    write_inference(scratch.file("identity.gguf"), base.gf, "inference-fixture-hash", identity);
    ggml_backend_load_all();
    auto * cpu = cpu_backend_new(2);
    check(cpu != nullptr, "CPU backend");
    try {
        for (const auto & name : names) {
            const auto actual = merged(base.gf, scratch.file("identity.gguf"), name, cpu);
            for (size_t j = 0; j < 6; ++j) check(actual[j] == base_values[j], "zero B identity");
        }
        p.b = {.25f, -.5f, 1.f, .75f};
        p.magnitude = {7, 3};
        conv = p; conv.stem = "dit.pre_conv";
        const auto saved = p.b;
        write_raw(scratch.file("raw.gguf"), "training-fixture-hash", {p, conv});
        write_inference(scratch.file("nonzero.gguf"), base.gf, "inference-fixture-hash", {p, conv});
        check(p.b == saved, "writer mutated raw B");
        Model raw(scratch.file("raw.gguf")), exported(scratch.file("nonzero.gguf"));
        check(std::string(gf_get_str(exported.gf, "sa3.adapter_format")) == "dora-rows-v1", "format");
        check(std::string(gf_get_str(exported.gf, "sa3.inference_base_id")) == "inference-fixture-hash", "base provenance");
        check(std::string(gf_get_str(raw.gf, "general.architecture")) == "sa3-lora", "raw format");
        check(std::string(gf_get_str(raw.gf, "lora.adapter_type")) == "dora-rows", "raw adapter type");
        check(!ggml_get_tensor(raw.gf.meta, (names[0] + ".row_scale").c_str()), "raw mixed with export");
        auto * raw_b = static_cast<const float *>(gf_get_data(raw.gf, "dit.0.self.qkv.lora_B"));
        auto * raw_m = static_cast<const float *>(gf_get_data(raw.gf, "dit.0.self.qkv.magnitude"));
        auto * raw_a = static_cast<const float *>(gf_get_data(raw.gf, "dit.0.self.qkv.lora_A"));
        check(raw_b && raw_m && raw_m[0] == 7 && raw_m[1] == 3, "raw magnitude");
        check(raw_a != nullptr, "raw A missing");
        for (size_t j = 0; j < p.a.size(); ++j) check(raw_a[j] == p.a[j], "raw A changed");
        check(gguf_get_val_f32(raw.gf.gguf, gguf_find_key(raw.gf.gguf, "lora.alpha")) == 6.f, "raw alpha");
        check(gguf_get_val_u32(raw.gf.gguf, gguf_find_key(raw.gf.gguf, "lora.rank")) == 2, "raw rank");
        for (size_t j = 0; j < saved.size(); ++j) check(raw_b[j] == saved[j], "raw B changed");
        for (const auto & name : names) {
            auto * b = static_cast<const float *>(gf_get_data(exported.gf, (name + ".lora_b").c_str()));
            check(b != nullptr, "original names/lowercase B");
            for (size_t j = 0; j < p.b.size(); ++j) check(b[j] == 3.f * p.b[j], "alpha/rank folding");
            const auto actual = merged(base.gf, scratch.file("nonzero.gguf"), name, cpu);
            bool changed = false;
            for (int i = 0; i < 2; ++i) {
                // Independent FP64 reference uses the RAW parameters and explicit
                // alpha/rank, rather than the writer's emitted row scales or B.
                double expected[3], norm2 = 0;
                for (int j = 0; j < 3; ++j) {
                    expected[j] = base_values[i * 3 + j];
                    for (int k = 0; k < 2; ++k)
                        expected[j] += double(p.b[i * 2 + k]) * p.a[k * 3 + j] * 6.0 / 2.0;
                    norm2 += expected[j] * expected[j];
                }
                for (int j = 0; j < 3; ++j) {
                    const double value = expected[j] * p.magnitude[i] / (std::sqrt(norm2) + 1e-12);
                    check(std::abs(actual[i * 3 + j] - value) <= 2e-6 + .004 * std::abs(value), "native/reference mismatch");
                    changed |= actual[i * 3 + j] != base_values[i * 3 + j];
                }
            }
            check(changed, "nonzero adapter ignored");
        }
        Rows q = p; q.alpha = 2.f;
        for (float & f : q.b) f *= 3;
        write_inference(scratch.file("equivalent.gguf"), base.gf, "inference-fixture-hash", {q});
        check(merged(base.gf, scratch.file("equivalent.gguf"), names[0], cpu) ==
              merged(base.gf, scratch.file("nonzero.gguf"), names[0], cpu), "alpha-equivalent merge");
        const std::pair<const char *, const char *> mappings[] = {
            {"dit.23.cross.kv", "model.transformer.layers.23.cross_attn.to_kv.weight"},
            {"dit.1.ff.proj", "model.transformer.layers.1.ff.ff.0.proj.weight"},
            {"dit.2.local.2", "model.transformer.layers.2.to_local_embed.2.weight"},
            {"dit.gce.0", "model.transformer.global_cond_embedder.0.weight"},
            {"dit.proj_out", "model.transformer.project_out.weight"}
        };
        for (const auto & item : mappings) check(original_name(item.first) == item.second, "name mapping");
        rejects([&] { original_name("dit.bad.self.qkv"); });
        rejects([&] { original_name("dit.0.unknown"); });
        rejects([&] { write_inference(scratch.file("bad.gguf"), base.gf, "id", {p, p}); });
        rejects([&] { write_inference(scratch.file("bad.gguf"), base.gf, "", {p}); });
        rejects([&] { write_inference(scratch.file("identity.gguf"), base.gf, "id", {p}); });
        q = p; q.stem = "dit.1.self.qkv";
        rejects([&] { write_inference(scratch.file("bad.gguf"), base.gf, "id", {q}); });
        q = p; q.in = 2;
        rejects([&] { write_inference(scratch.file("bad.gguf"), base.gf, "id", {q}); });
        q = p; q.a[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { write_inference(scratch.file("bad.gguf"), base.gf, "id", {q}); });
        q = p; q.b[0] = std::numeric_limits<float>::max();
        rejects([&] { write_inference(scratch.file("bad.gguf"), base.gf, "id", {q}); });
    } catch (...) { ggml_backend_free(cpu); throw; }
    ggml_backend_free(cpu);
    printf("SA3 EXPORT PASS: identity, nonzero FP64, alpha, names, raw state, invalid inputs, matrix/conv native round trip\n");
}
} // namespace sa3_export_test

#ifndef SA3_EXPORT_NO_MAIN
int main() {
    try { sa3_export_test::run(); return 0; }
    catch (const std::exception & e) { fprintf(stderr, "SA3 EXPORT FAIL: %s\n", e.what()); return 1; }
}
#endif
