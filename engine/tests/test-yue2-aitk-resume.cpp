// Self-check for the YuE2 LmOptim resume record: optimizer moments must come
// back bit-exact, including the tiny second moments the old 8-bit storage
// zeroed. Header-only, no CUDA, no model files:
//   cl /std:c++17 /EHsc /I..\src /I..\src\train /I..\ggml\include test-yue2-aitk-resume.cpp
//   c++ -std=c++17 -I../src -I../src/train -I../ggml/include test-yue2-aitk-resume.cpp
#include "yue2-aitk-resume.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>

namespace fs = std::filesystem;

static yue2_aitk::HostStateSnapshot make_snapshot(size_t tensors, size_t n, int slots, uint32_t seed) {
    yue2_aitk::HostStateSnapshot s;
    s.step = 2;
    auto sized = [&](auto & v) { v.resize(tensors); };
    sized(s.names); sized(s.elements); sized(s.parameters);
    sized(s.state1_fp32); sized(s.state2_fp32); sized(s.state3_fp32); sized(s.state4_fp32);
    sized(s.state1_u8); sized(s.state2_u8); sized(s.state3_u8); sized(s.state4_u8);
    sized(s.absmax1); sized(s.absmax2); sized(s.absmax3); sized(s.absmax4);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f), lg(-12.0f, -6.0f);
    std::vector<float> * fp32[4] = {};
    for (size_t i = 0; i < tensors; ++i) {
        s.names[i] = "lora." + std::to_string(i) + (i % 2 ? ".lora_B" : ".lora_A");
        s.elements[i] = n;
        s.parameters[i].resize(n);
        for (float & x : s.parameters[i]) x = u(rng) * 0.02f;
        fp32[0] = &s.state1_fp32[i]; fp32[1] = &s.state2_fp32[i]; fp32[2] = &s.state3_fp32[i]; fp32[3] = &s.state4_fp32[i];
        for (int k = 0; k < slots; ++k) {
            fp32[k]->resize(n);
            // Adam after a few steps: v spans six decades within one block, so
            // most entries sit far below absmax/254.
            for (float & x : *fp32[k]) x = k == 1 ? std::pow(10.0f, lg(rng)) : u(rng) * 1e-5f;
        }
    }
    return s;
}

static bool bits_equal(const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i].size() != b[i].size() || (!a[i].empty() && std::memcmp(a[i].data(), b[i].data(), a[i].size() * sizeof(float)) != 0)) return false;
    return true;
}

int main() {
    const fs::path dir = fs::temp_directory_path() / ("y2resume-" + std::to_string(std::random_device{}()));
    fs::create_directories(dir);
    auto P = [&](const char * f) { return (dir / f).string(); };
    const size_t kN = 8192;  // above the old 4096-element quantization cut

    // What 8-bit storage did to v: the small entries decode to exactly 0 and
    // the next AdamW step (lr * m / (sqrt(v) + eps)) blows up for them.
    {
        auto s = make_snapshot(1, kN, 2, 7);
        std::vector<uint8_t> q; std::vector<float> am, back;
        assert(yue2_aitk::resume_detail::quantize_f32(s.state2_fp32[0], &q, &am));
        assert(yue2_aitk::resume_detail::dequantize_f32(q, am, &back));
        size_t zeroed = 0; double worst = 0.0;
        for (size_t k = 0; k < kN; ++k) {
            const double m = s.state1_fp32[0][k], eps = 1e-6;
            if (back[k] == 0.0f) ++zeroed;
            const double exact = std::fabs(m) / (std::sqrt(double(s.state2_fp32[0][k])) + eps);
            const double lossy = std::fabs(m) / (std::sqrt(double(back[k])) + eps);
            worst = std::max(worst, lossy / exact);
        }
        std::printf("legacy 8-bit v: %zu/%zu entries decode to 0, worst step ratio %.1fx\n", zeroed, kN, worst);
        assert(zeroed > kN / 4 && worst > 10.0);
    }

    // AdamW (m, v) and Prodigy (m, v, s, x0): F32 slots of any size round trip
    // bit-exact as version 3.
    for (int slots : {2, 4}) {
        auto s = make_snapshot(6, kN, slots, 11 + slots);
        s.elements[5] = 300; for (auto * v : {&s.parameters[5], &s.state1_fp32[5], &s.state2_fp32[5], &s.state3_fp32[5], &s.state4_fp32[5]}) if (!v->empty()) v->resize(300);
        const std::string path = P(slots == 2 ? "adamw.resume" : "prodigy.resume");
        const std::string meta = "{\"optimizer\":\"adamw-lm\"}";
        assert(yue2_aitk::yue2_aitk_write_resume_v2(path.c_str(), s, meta));
        yue2_aitk::ResumeRecord r;
        assert(yue2_aitk::yue2_aitk_read_resume(path.c_str(), &r));
        assert(r.version == 3 && r.runner_metadata == meta && r.state.step == s.step && r.state.names == s.names && r.state.elements == s.elements);
        assert(bits_equal(r.state.parameters, s.parameters));
        assert(bits_equal(r.state.state1_fp32, s.state1_fp32) && bits_equal(r.state.state2_fp32, s.state2_fp32));
        assert(bits_equal(r.state.state3_fp32, s.state3_fp32) && bits_equal(r.state.state4_fp32, s.state4_fp32));
        for (size_t i = 0; i < s.names.size(); ++i) assert(r.state.state1_u8[i].empty() && r.state.state2_u8[i].empty() && r.state.absmax2[i].empty());
        // A flipped payload byte fails the checksum; truncation is refused.
        std::vector<uint8_t> raw; assert(yue2_aitk::resume_detail::read_file(path.c_str(), &raw));
        raw[raw.size() - 7] ^= 0x20;
        FILE * f = std::fopen(P("bad.resume").c_str(), "wb"); std::fwrite(raw.data(), 1, raw.size(), f); std::fclose(f);
        yue2_aitk::ResumeRecord bad; assert(!yue2_aitk::yue2_aitk_read_resume(P("bad.resume").c_str(), &bad));
        raw.resize(raw.size() - 1);
        f = std::fopen(P("bad2.resume").c_str(), "wb"); std::fwrite(raw.data(), 1, raw.size(), f); std::fclose(f);
        assert(!yue2_aitk::yue2_aitk_read_resume(P("bad2.resume").c_str(), &bad));
    }

    // Older version-2 records (8-bit slots above 4096 elements) still load as
    // they were written; the runner warns that they are lossy.
    {
        auto s = make_snapshot(2, kN, 2, 23);
        for (size_t i = 0; i < 2; ++i) {
            assert(yue2_aitk::resume_detail::quantize_f32(s.state1_fp32[i], &s.state1_u8[i], &s.absmax1[i]));
            assert(yue2_aitk::resume_detail::quantize_f32(s.state2_fp32[i], &s.state2_u8[i], &s.absmax2[i]));
            s.state1_fp32[i].clear(); s.state2_fp32[i].clear();
        }
        assert(yue2_aitk::yue2_aitk_write_resume_v2(P("legacy.resume").c_str(), s, "{}"));
        yue2_aitk::ResumeRecord r;
        assert(yue2_aitk::yue2_aitk_read_resume(P("legacy.resume").c_str(), &r));
        assert(r.version == 2 && r.state.state1_u8 == s.state1_u8 && r.state.state2_u8 == s.state2_u8 && r.state.absmax2 == s.absmax2);
        assert(bits_equal(r.state.parameters, s.parameters) && r.state.state2_fp32[0].empty());
        // The legacy layout cannot carry a large F32 slot: mixing is refused.
        s.state1_u8[1].clear(); s.absmax1[1].clear(); s.state1_fp32[1].assign(kN, 1e-6f);
        assert(!yue2_aitk::yue2_aitk_write_resume_v2(P("mixed.resume").c_str(), s, "{}"));
    }

    fs::remove_all(dir);
    std::puts("PASS test-yue2-aitk-resume");
    return 0;
}
