// Self-check for the HeartMuLa trainer's song-structure options and loss
// weights (train/heartmula-train-state.h): start-anchored crops (--crop-frames),
// the end-of-song frame (--eos) and the --cb0-weighted factors. Host only, no
// GPU, no model files. From engine\build\test-standalone in a VS x64 prompt:
//   cl /std:c++17 /EHsc /O2 /utf-8 /I..\..\src /I..\..\src\train /I..\..\ggml\include /I..\..\vendor\yyjson /I..
//      ..\..\tests\test-heartmula-train-schedule.cpp ..\ggml\src\Release\ggml.lib ..\ggml\src\Release\ggml-base.lib
//      ..\Release\yyjson.lib
// Optional argument: a heartmula-prepare directory, whose manifest is scheduled too.
#include "heartmula-train-state.h"

#include <cstdio>

static void jl(const char *, ...) {}  // lm-common.h declares it; ace-train.cpp defines it

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { g_fail++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

// Every pick of `steps` optimizer steps: crop windows start at 0, are N frames or the whole
// song, and carry the end frame exactly when they reach the song's real end.
static void check_schedule(const HmtData & d, int accum, int64_t crop, int64_t steps, const char * what) {
    int64_t picks = 0, whole = 0, eos = 0, frames = 0, cropped_long = 0;
    for (int64_t step = 0; step < steps; step++) {
        for (const HmtPick & p : hmt_schedule(d, accum, 0, 7, step, crop, true, 8192)) {
            const HmtSample & s = d.samples[(size_t) p.sample];
            const int64_t     F = s.frames;
            const bool        fits = (int64_t) s.prompt.size() + F <= 8192;
            CHECK(p.w0 == 0, "%s step %lld: window starts at %lld, not 0", what, (long long) step, (long long) p.w0);
            CHECK(p.T == F || (p.T == crop && crop < F), "%s step %lld: T %lld is neither the song (%lld) nor the crop",
                  what, (long long) step, (long long) p.T, (long long) F);
            CHECK(p.eos == (p.T == F && fits), "%s step %lld: eos %d on a window of %lld / %lld frames", what, (long long) step,
                  (int) p.eos, (long long) p.T, (long long) F);
            picks++;
            whole += p.T == F;
            eos += p.eos;
            frames += p.T + p.eos;
            cropped_long += F > crop && p.T == F;
        }
    }
    int64_t longer = 0;
    for (const auto & s : d.samples) longer += s.frames > crop;
    printf("%s: %lld picks over %lld steps, %lld whole (%lld of them songs longer than the crop), %lld with the end frame, "
           "%lld frames trained (%.1f per pick)\n",
           what, (long long) picks, (long long) steps, (long long) whole, (long long) cropped_long, (long long) eos,
           (long long) frames, (double) frames / (double) picks);
    if (longer == (int64_t) d.samples.size()) {
        const double share = (double) cropped_long / (double) picks;
        CHECK(share > 0.15 && share < 0.35, "%s: %.3f of long songs trained whole, HMT_FULL_EVERY says 1/4", what, share);
    }
}

int main(int argc, char ** argv) {
    namespace fs = std::filesystem;
    // Synthetic set: 10 songs of 1800..3600 frames, prompts of 400 rows; crop 750.
    HmtData d;
    for (int i = 0; i < 10; i++) {
        HmtSample s;
        s.id     = "s" + std::to_string(i);
        s.frames = 1800 + 200 * i;
        s.prompt.assign(400, 1);
        d.samples.push_back(s);
    }
    check_schedule(d, 4, 750, 300, "synthetic, crop 750");
    // A song too long for the context with its end frame trains whole without one.
    HmtData big = d;
    big.samples[0].frames = 8192 - 400;  // S + T == 8192: still fits (rows S + T - 1 + 1)
    big.samples[1].frames = 8192 - 399;  // one past: no end frame
    for (int64_t step = 0; step < 40; step++)
        for (const HmtPick & p : hmt_schedule(big, 4, 0, 7, step, 750, true, 8192)) {
            if (p.sample == 0 && p.T == big.samples[0].frames) CHECK(p.eos, "S + T == ctx must keep the end frame");
            if (p.sample == 1) CHECK(!p.eos, "S + T > ctx must drop the end frame");
        }
    // The shared rule hmt_eval scores with: S 400, F 7792 keeps the end frame (L = 8192),
    // F 7793 drops it rather than scoring L = 8193 past the context.
    CHECK(hmt_eos_fits(400, 7792, 8192) && !hmt_eos_fits(400, 7793, 8192), "hmt_eos_fits boundary");
    printf("eos boundary: S 400 + F 7792 keeps the end frame, S 400 + F 7793 drops it (context 8192)\n");
    // Without --eos nothing gets an end frame; without --crop-frames windows are whole songs.
    for (int64_t step = 0; step < 20; step++) {
        for (const HmtPick & p : hmt_schedule(d, 4, 0, 7, step, 750, false, 8192)) CHECK(!p.eos, "end frame without --eos");
        for (const HmtPick & p : hmt_schedule(d, 4, 0, 7, step, 0, false, 8192))
            CHECK(p.w0 == 0 && p.T == d.samples[(size_t) p.sample].frames && !p.eos, "whole-song default changed");
    }

    // The window the run loop trains (run.h): hmt_read_codes(w0, T), then 8 x HMT_EOS_CODE when
    // p.eos. The last real frame must be the song's last frame, and nothing past a crop.
    {
        const fs::path dir = fs::temp_directory_path() / "hm-sched-test";
        fs::create_directories(dir);
        HmtSample s;
        s.frames     = 900;
        s.codes_path = (dir / "c.i32").u8string();
        std::vector<int32_t> all((size_t) (s.frames * 8));
        for (int64_t t = 0; t < s.frames; t++)
            for (int k = 0; k < 8; k++) all[(size_t) (t * 8 + k)] = (int32_t) ((t * 8 + k) % 8192);
        FILE * f = fopen(s.codes_path.c_str(), "wb");
        fwrite(all.data(), 4, all.size(), f);
        fclose(f);
        std::vector<int32_t> codes;
        std::string          err;
        for (int64_t T : { (int64_t) 750, s.frames }) {
            const bool eos = T == s.frames;
            CHECK(hmt_read_codes(s, 0, T, &codes, &err), "%s", err.c_str());
            if (eos) codes.insert(codes.end(), 8, HMT_EOS_CODE);
            const int64_t Tt = T + (eos ? 1 : 0);
            CHECK((int64_t) codes.size() == Tt * 8, "window holds %zu codes", codes.size());
            CHECK(memcmp(codes.data(), all.data(), (size_t) T * 32) == 0, "window frames differ from the song's first %lld", (long long) T);
            if (eos) {
                for (int k = 0; k < 8; k++) CHECK(codes[(size_t) (T * 8 + k)] == HMT_EOS_CODE, "end frame codebook %d", k);
                CHECK(HMT_EOS_CODE < HEARTMULA_AUDIO_VOCAB, "the end code must be a label column");
            } else {
                CHECK(codes.back() == all[(size_t) (T * 8 - 1)], "a crop must end at frame %lld", (long long) T - 1);
            }
            // hmt_sample_upload: frame t feeds backbone row S + t for t < Tt - 1, and frame t is the
            // label of row S - 1 + t. So the end frame is a label only, and frame T - 1 (a real
            // frame) is the input that predicts it; a crop's last label is real frame T - 1.
            printf("window %s: %lld frames + %d end frame; backbone inputs frames 0..%lld, labels frames 0..%lld%s\n",
                   eos ? "whole" : "crop ", (long long) T, eos ? 1 : 0, (long long) Tt - 2, (long long) Tt - 1,
                   eos ? " (last label = end frame 8193 x 8)" : " (no label past the crop)");
        }
        fs::remove_all(dir);
    }

    // --cb0-weighted: loss = sum_k wce * cbw[k] * CE_k with wce = 1/8 (one micro-step).
    {
        float u[8], w[8];
        hmt_cb_weights(false, u);
        hmt_cb_weights(true, w);
        double su = 0, sw = 0;
        for (int k = 0; k < 8; k++) su += u[k] / 8.0, sw += w[k] / 8.0;
        CHECK(std::fabs(su - 1.0) < 1e-6 && std::fabs(sw - 1.0) < 1e-6, "factors / 8 must sum to 1 (%.7f, %.7f)", su, sw);
        for (int k = 0; k < 8; k++) CHECK(u[k] == 1.0f, "default factor %d is %g, not 1", k, u[k]);
        static const double paper[8] = { 2.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 };
        for (int k = 0; k < 8; k++)
            CHECK(std::fabs(w[k] / 8.0 - paper[k] / 4.8) < 1e-6, "codebook %d: dLoss/dCE %.6f, want %.6f", k, w[k] / 8.0, paper[k] / 4.8);
        // Equal CEs give the same loss either way; CEs as a fresh run sees them (step 1 of the
        // 12-song run) shift the objective toward CB0.
        const double ce[8] = { 3.819441, 5.481514, 6.030139, 6.434746, 6.928698, 7.273015, 7.460541, 7.727061 };
        double lu = 0, lw = 0, eu = 0, ew = 0;
        for (int k = 0; k < 8; k++) lu += ce[k] * u[k] / 8, lw += ce[k] * w[k] / 8, eu += 5.0 * u[k] / 8, ew += 5.0 * w[k] / 8;
        CHECK(std::fabs(eu - ew) < 1e-6, "equal CEs: %.6f vs %.6f", eu, ew);
        printf("cb weights (dLoss/dCE_k):");
        for (int k = 0; k < 8; k++) printf(" %.5f", w[k] / 8.0);
        printf("\nstep-1 CEs: plain loss %.4f, cb0-weighted %.4f; CB0 share of the gradient scale %.3f -> %.3f\n", lu, lw,
               1.0 / 8.0, w[0] / 8.0);
    }

    if (argc > 1) {
        HmtData real;
        std::string err;
        if (!hmt_data_load(argv[1], &real, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        check_schedule(real, 4, 750, 300, "manifest, crop 750");
    }
    printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
