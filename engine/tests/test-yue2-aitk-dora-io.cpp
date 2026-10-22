// Self-check for the YuE2 DoRA adapter export and the 672-slot resume round
// trip. Header-only, no CUDA, no model files:
//   cl /std:c++17 /EHsc /I..\src /I..\src\train test-yue2-aitk-dora-io.cpp
#include "yue2-aitk-native-adapter-io.h"
#include "yue2-aitk-resume.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>

namespace fs = std::filesystem;

static const int64_t kRank = 4;
static const char * kSites[4] = {"self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"};

struct Inv {
    std::vector<Yue2AitkF32Matrix> m;
    std::vector<std::vector<float>> store;
};

// Distinct values everywhere so a misplaced row cannot match by accident.
static Inv make_inventory(bool dora) {
    Inv inv;
    inv.store.reserve(700);
    float seed = 0.001f;
    auto add = [&](const std::string & name, int64_t rows, int64_t cols, bool positive) {
        inv.store.emplace_back((size_t) (rows * cols));
        for (float & v : inv.store.back()) { seed += 0.0003f; if (seed > 3.0f) seed = 0.001f; v = positive ? 0.5f + seed : seed - 1.5f; }
        inv.m.push_back({name, rows, cols, inv.store.back().data()});
    };
    for (const char * expert : {"diffusion_model", "text_encoders"}) {
        for (int layer = 0; layer < 28; ++layer) {
            for (const char * site : kSites) {
                int64_t in = 0, out = 0;
                yue2_aitk_adapter_io_detail::site_dims(site, &in, &out);
                const std::string base = std::string(expert) + ".model.layers." + std::to_string(layer) + "." + site;
                add(base + ".lora_A.weight", kRank, in, false);
                add(base + ".lora_B.weight", out, kRank, false);
                if (dora) add(base + ".lora_magnitude_vector.weight", 1, out, true);
            }
        }
    }
    return inv;
}

struct StTensor { std::string dtype; std::vector<int64_t> shape; std::vector<uint8_t> bytes; };
struct StFile { std::map<std::string, StTensor> t; std::string header; };

static std::string between(const std::string & s, size_t from, const char * open, const char * close, size_t * end = nullptr) {
    const size_t a = s.find(open, from); assert(a != std::string::npos);
    const size_t b = s.find(close, a + strlen(open)); assert(b != std::string::npos);
    if (end) *end = b;
    return s.substr(a + strlen(open), b - a - strlen(open));
}

static bool read_st(const std::string & path, StFile * out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint64_t n = 0;
    if (std::fread(&n, 8, 1, f) != 1) { std::fclose(f); return false; }
    out->header.resize((size_t) n);
    if (std::fread(&out->header[0], 1, (size_t) n, f) != n) { std::fclose(f); return false; }
    std::vector<uint8_t> data;
    uint8_t buf[65536]; size_t got;
    while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0) data.insert(data.end(), buf, buf + got);
    std::fclose(f);
    const std::string & h = out->header;
    size_t pos = 0;
    for (;;) {
        // Every entry is "name":{"dtype":"X","shape":[...],"data_offsets":[a,b]}
        const size_t d = h.find("\"dtype\"", pos);
        if (d == std::string::npos) break;
        const size_t nameEnd = h.rfind("\":{", d);
        const size_t nameBeg = h.rfind('"', nameEnd - 1);
        const std::string name = h.substr(nameBeg + 1, nameEnd - nameBeg - 1);
        StTensor t;
        t.dtype = between(h, d, "\"dtype\":\"", "\"");
        const std::string shape = between(h, d, "\"shape\":[", "]");
        size_t p = 0;
        while (p < shape.size()) { t.shape.push_back(std::stoll(shape.substr(p))); p = shape.find(',', p); if (p == std::string::npos) break; ++p; }
        size_t offEnd = 0;
        const std::string off = between(h, d, "\"data_offsets\":[", "]", &offEnd);
        const size_t comma = off.find(',');
        const size_t a = std::stoull(off.substr(0, comma)), b = std::stoull(off.substr(comma + 1));
        t.bytes.assign(data.begin() + a, data.begin() + b);
        out->t[name] = std::move(t);
        pos = offEnd;
    }
    return true;
}

static float bf16(const uint8_t * p) { uint32_t u = (uint32_t) p[0] << 16 | (uint32_t) p[1] << 24; float f; std::memcpy(&f, &u, 4); return f; }
static float f32(const uint8_t * p) { float f; std::memcpy(&f, p, 4); return f; }

static size_t file_count(const fs::path & dir) { size_t n = 0; for (auto & e : fs::directory_iterator(dir)) { (void) e; ++n; } return n; }

static std::vector<Yue2AitkF32Matrix> with(const Inv & inv, const std::function<void(std::vector<Yue2AitkF32Matrix> &)> & edit) {
    std::vector<Yue2AitkF32Matrix> v = inv.m; edit(v); return v;
}

int main() {
    const fs::path dir = fs::temp_directory_path() / "yue2-dora-io-selfcheck";
    fs::remove_all(dir); fs::create_directories(dir);
    auto P = [&](const char * n) { return (dir / n).string(); };

    const Inv dora = make_inventory(true), lora = make_inventory(false);
    assert(dora.m.size() == 672 && lora.m.size() == 448);

    // --- fused DoRA: 672 tensors, A/B BF16, m F32 1-D [out], metadata ---
    assert(yue2_aitk_write_fused_dora(dora.m, kRank, 8.0f, 7, P("fused.safetensors").c_str()));
    StFile fused; assert(read_st(P("fused.safetensors"), &fused));
    assert(fused.t.size() == 672);
    assert(fused.header.find("yue2-aitk-fused-dora-v1") != std::string::npos);
    assert(fused.header.find("\"use_dora\":\"true\"") != std::string::npos);
    assert(fused.header.find("\"dora_scaling\":\"2\"") != std::string::npos);
    const std::string qkvm = "diffusion_model.model.layers.3.self_attn.qkv_proj.lora_magnitude_vector.weight";
    const StTensor & fm = fused.t.at(qkvm);
    assert(fm.dtype == "F32" && fm.shape == std::vector<int64_t>{4096} && fm.bytes.size() == 4096 * 4);
    assert(fused.t.at("diffusion_model.model.layers.3.self_attn.qkv_proj.lora_A.weight").dtype == "BF16");
    // Refuses to overwrite.
    assert(!yue2_aitk_write_fused_dora(dora.m, kRank, 8.0f, 7, P("fused.safetensors").c_str()));

    // --- native split DoRA: split by row, join by concatenation ---
    assert(yue2_aitk_write_native_split_dora(dora.m, kRank, 8.0f, 7, P("ar.safetensors").c_str(), P("nar.safetensors").c_str()));
    StFile ar, nar; assert(read_st(P("ar.safetensors"), &ar) && read_st(P("nar.safetensors"), &nar));
    assert(ar.t.size() == 28 * 7 * 3 && nar.t.size() == 28 * 7 * 3);
    assert(ar.header.find("yue2-ar-dora-v1") != std::string::npos && nar.header.find("yue2-nar-dora-v1") != std::string::npos);
    assert(ar.header.find("\"use_dora\":\"true\"") != std::string::npos);
    // AR is text_encoders. Join q/k/v and gate/up and compare with the fused vector.
    const auto join = [&](StFile & f, const char * tag, std::initializer_list<const char *> parts, const std::string & fused_name) {
        std::vector<uint8_t> cat;
        for (const char * p : parts) {
            const StTensor & t = f.t.at(std::string("yue2.blk.5.") + tag + p + ".lora_magnitude_vector.weight");
            assert(t.dtype == "F32" && t.shape.size() == 1 && t.bytes.size() == (size_t) t.shape[0] * 4);
            cat.insert(cat.end(), t.bytes.begin(), t.bytes.end());
        }
        const StTensor & whole = fused.t.at(fused_name);
        (void) whole;
        return cat;
    };
    // Reference values come from the in-memory inventory, not the fused file.
    const auto ref = [&](const std::string & name) -> const Yue2AitkF32Matrix & {
        for (const auto & m : dora.m) if (m.name == name) return m;
        assert(false); return dora.m[0];
    };
    const auto same = [&](const std::vector<uint8_t> & cat, const Yue2AitkF32Matrix & m) {
        return cat.size() == (size_t) m.cols * 4 && std::memcmp(cat.data(), m.data, cat.size()) == 0;
    };
    const std::string ar_qkv = "text_encoders.model.layers.5.self_attn.qkv_proj.lora_magnitude_vector.weight";
    const std::string ar_gu = "text_encoders.model.layers.5.mlp.gate_up_proj.lora_magnitude_vector.weight";
    assert(ar.t.at("yue2.blk.5.attn_q.lora_magnitude_vector.weight").shape[0] == 2048);
    assert(ar.t.at("yue2.blk.5.attn_k.lora_magnitude_vector.weight").shape[0] == 1024);
    assert(ar.t.at("yue2.blk.5.attn_v.lora_magnitude_vector.weight").shape[0] == 1024);
    assert(same(join(ar, "", {"attn_q", "attn_k", "attn_v"}, qkvm), ref(ar_qkv)));
    assert(same(join(ar, "", {"ffn_gate", "ffn_up"}, qkvm), ref(ar_gu)));
    assert(same(join(ar, "", {"attn_output"}, qkvm), ref("text_encoders.model.layers.5.self_attn.o_proj.lora_magnitude_vector.weight")));
    assert(same(join(ar, "", {"ffn_down"}, qkvm), ref("text_encoders.model.layers.5.mlp.down_proj.lora_magnitude_vector.weight")));
    assert(same(join(nar, "nar_", {"attn_q", "attn_k", "attn_v"}, qkvm), ref("diffusion_model.model.layers.5.self_attn.qkv_proj.lora_magnitude_vector.weight")));
    assert(same(join(nar, "", {"nar_ffn_gate", "nar_ffn_up"}, qkvm), ref("diffusion_model.model.layers.5.mlp.gate_up_proj.lora_magnitude_vector.weight")));
    // The B rows split the same way: B row 2048 of AR qkv is k's first row (BF16 truncation tolerance).
    const Yue2AitkF32Matrix & B = ref("text_encoders.model.layers.5.self_attn.qkv_proj.lora_B.weight");
    const StTensor & kB = ar.t.at("yue2.blk.5.attn_k.lora_B.weight");
    assert(kB.shape == (std::vector<int64_t>{1024, kRank}));
    assert(std::fabs(bf16(kB.bytes.data()) - B.data[2048 * kRank]) < 1e-2f);

    // --- malformed inventories are refused and leave nothing behind ---
    const size_t before = file_count(dir);
    auto reject = [&](const std::vector<Yue2AitkF32Matrix> & inv, const char * tag) {
        const std::string a = P((std::string(tag) + ".f.safetensors").c_str());
        const std::string b = P((std::string(tag) + ".ar.safetensors").c_str());
        const std::string c = P((std::string(tag) + ".nar.safetensors").c_str());
        const bool ok = yue2_aitk_write_fused_dora(inv, kRank, 8.0f, 1, a.c_str()) ||
                        yue2_aitk_write_native_split_dora(inv, kRank, 8.0f, 1, b.c_str(), c.c_str());
        if (ok) std::fprintf(stderr, "accepted malformed inventory: %s\n", tag);
        assert(!ok);
    };
    reject(with(dora, [](std::vector<Yue2AitkF32Matrix> & v) { v.erase(v.begin() + 2); }), "missing-m");
    reject(with(dora, [](std::vector<Yue2AitkF32Matrix> & v) { v.pop_back(); }), "short");
    reject(lora.m, "lora-inventory");
    reject(with(dora, [](std::vector<Yue2AitkF32Matrix> & v) { v[2].rows = 2; v[2].cols = v[2].cols / 2; }), "m-rows");
    reject(with(dora, [](std::vector<Yue2AitkF32Matrix> & v) { v[2].cols -= 1; }), "m-cols");
    reject(with(dora, [](std::vector<Yue2AitkF32Matrix> & v) { v[2].name = v[2].name + "x"; }), "m-name");
    reject(with(dora, [](std::vector<Yue2AitkF32Matrix> & v) { v[5] = v[2]; }), "dup-name");
    reject(with(dora, [](std::vector<Yue2AitkF32Matrix> & v) { v[2].data = nullptr; }), "null-m");
    std::vector<float> bad(4096, 1.0f); bad[17] = NAN;
    reject(with(dora, [&](std::vector<Yue2AitkF32Matrix> & v) { v[2].data = bad.data(); }), "nan-m");
    // Orphan magnitude: a DoRA-sized list where a pair's A is replaced by a stray m.
    reject(with(dora, [&](std::vector<Yue2AitkF32Matrix> & v) { v[0].name = "diffusion_model.model.layers.0.self_attn.qkv_proj.lora_magnitude_vector.weight"; }), "orphan");
    assert(file_count(dir) == before);

    // --- legacy LoRA-v1 unchanged: 448 in, no magnitude out ---
    assert(yue2_aitk_write_fused_lora(lora.m, kRank, 8.0f, 7, P("lora-fused.safetensors").c_str()));
    assert(yue2_aitk_write_native_split(lora.m, kRank, 8.0f, 7, P("lar.safetensors").c_str(), P("lnar.safetensors").c_str()));
    StFile lf, la; assert(read_st(P("lora-fused.safetensors"), &lf) && read_st(P("lar.safetensors"), &la));
    assert(lf.t.size() == 448 && la.t.size() == 28 * 7 * 2);
    assert(lf.header.find("yue2-aitk-fused-lora-v1") != std::string::npos && lf.header.find("use_dora") == std::string::npos);
    assert(la.header.find("yue2-ar-lora-v1") != std::string::npos && la.header.find("use_dora") == std::string::npos);
    assert(la.header.find("magnitude") == std::string::npos);
    // A DoRA inventory must not sneak through the LoRA writers.
    assert(!yue2_aitk_write_fused_lora(dora.m, kRank, 8.0f, 7, P("x1.safetensors").c_str()));
    assert(!yue2_aitk_write_native_split(dora.m, kRank, 8.0f, 7, P("x2.safetensors").c_str(), P("x3.safetensors").c_str()));

    // --- resume: 672 slots, A/B/m, optimizer moments, both versions ---
    for (int version : {1, 2}) {
        yue2_aitk::HostStateSnapshot s;
        s.step = 11;
        const size_t N = dora.m.size();
        s.names.resize(N); s.elements.resize(N); s.parameters.resize(N);
        s.state1_fp32.resize(N); s.state2_fp32.resize(N); s.state1_u8.resize(N); s.state2_u8.resize(N);
        s.absmax1.resize(N); s.absmax2.resize(N); s.state3_fp32.resize(N); s.state4_fp32.resize(N);
        s.state3_u8.resize(N); s.state4_u8.resize(N); s.absmax3.resize(N); s.absmax4.resize(N);
        size_t mags = 0, quantized_mags = 0;
        for (size_t i = 0; i < N; ++i) {
            const auto & mat = dora.m[i];
            const size_t n = (size_t) (mat.rows * mat.cols);
            s.names[i] = mat.name; s.elements[i] = n;
            s.parameters[i].assign(mat.data, mat.data + n);
            if (mat.name.find("lora_magnitude_vector") != std::string::npos) { ++mags; if (n >= 4096) ++quantized_mags; }
            std::vector<float> m1(n), m2(n);
            for (size_t k = 0; k < n; ++k) { m1[k] = std::sin((float) (k + i)) * 0.01f; m2[k] = std::fabs(std::cos((float) (k * 3 + i))) * 0.001f; }
            if (n < 4096) { s.state1_fp32[i] = m1; s.state2_fp32[i] = m2; }
            else {
                assert(yue2_aitk::resume_detail::quantize_f32(m1, &s.state1_u8[i], &s.absmax1[i]));
                assert(yue2_aitk::resume_detail::quantize_f32(m2, &s.state2_u8[i], &s.absmax2[i]));
            }
        }
        assert(mags == 224 && quantized_mags > 0);
        const std::string path = P(version == 1 ? "r1.resume" : "r2.resume");
        const std::string meta = "{\"adapter_type\":\"dora\"}";
        assert(version == 1 ? yue2_aitk::yue2_aitk_write_resume(path.c_str(), s, meta)
                            : yue2_aitk::yue2_aitk_write_resume_v2(path.c_str(), s, meta));
        yue2_aitk::ResumeRecord r;
        assert(yue2_aitk::yue2_aitk_read_resume(path.c_str(), &r));
        assert(r.version == version && r.runner_metadata == meta && r.state.step == 11 && r.state.names == s.names);
        assert(r.state.elements == s.elements && r.state.parameters == s.parameters);
        assert(r.state.state1_fp32 == s.state1_fp32 && r.state.state2_fp32 == s.state2_fp32);
        assert(r.state.state1_u8 == s.state1_u8 && r.state.state2_u8 == s.state2_u8);
        assert(r.state.absmax1 == s.absmax1 && r.state.absmax2 == s.absmax2);
        // A flipped payload byte (inside a magnitude slot's parameters) fails the checksum.
        std::vector<uint8_t> raw; assert(yue2_aitk::resume_detail::read_file(path.c_str(), &raw));
        raw[raw.size() - 5] ^= 0x40;
        FILE * f = std::fopen(P("corrupt.resume").c_str(), "wb"); std::fwrite(raw.data(), 1, raw.size(), f); std::fclose(f);
        yue2_aitk::ResumeRecord bad_r; assert(!yue2_aitk::yue2_aitk_read_resume(P("corrupt.resume").c_str(), &bad_r));
        fs::remove(P("corrupt.resume"));
        // Truncation is refused too.
        raw.resize(raw.size() / 2);
        f = std::fopen(P("trunc.resume").c_str(), "wb"); std::fwrite(raw.data(), 1, raw.size(), f); std::fclose(f);
        assert(!yue2_aitk::yue2_aitk_read_resume(P("trunc.resume").c_str(), &bad_r));
        fs::remove(P("trunc.resume"));
    }

    fs::remove_all(dir);
    std::puts("PASS test-yue2-aitk-dora-io");
    return 0;
}
