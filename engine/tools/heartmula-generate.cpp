// heartmula-generate: HeartMuLa text to music from the command line, through the same
// pipeline as POST /heartmula/synth (heartmula/pipeline.h). The smoke test's HeartMuLa
// case and the S6 receipts (seed repeat, cancel per phase, unload and reload, admission)
// run here.
//
//   heartmula-generate --models DIR --request FILE.json --out FILE.wav [--check]
//                      [--runs N] [--cancel-at STAGE[:STEP]] [--codes-out FILE.json]
//
// --runs N renders N times in one process, freeing every buffer between runs, and
// prints a hash of each run's codes and audio. --cancel-at cancels once the pipeline
// reports STAGE (prefill, frames, codec_decode, loading_lm, loading_codec, admission)
// with a step >= STEP. --check measures the raw float output as ace-synth --check
// does; HOTSTEP_CHECK_INJECT (none|nan|subnormal|silence|dc|clip, comma list) checks
// one corrupted copy per entry, to prove each check can fail.
#include "heartmula/pipeline.h"

#include "audio-io.h"

#include <cstdio>
#include <fstream>
#include <sstream>

static uint64_t fnv1a(const void * data, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) h = (h ^ ((const uint8_t *) data)[i]) * 1099511628211ull;
    return h;
}

// ace-synth's --check thresholds: non-finite or subnormal samples, peak below -60 dBFS,
// |mean| above 0.02 on either channel, more than 0.1% of samples past full scale.
static bool check_audio(int b, std::vector<float> a, int64_t n, const char * inject) {
    if (inject && *inject && strcmp(inject, "none") != 0) {
        fprintf(stderr, "[Check Batch%d] injecting '%s'\n", b, inject);
        for (int64_t i = 0; i < 2 * n; i++) {
            float & x = a[(size_t) i];
            if (!strcmp(inject, "nan") && i == n / 2) x = NAN;
            if (!strcmp(inject, "subnormal") && i == n / 2) x = 1e-40f;
            if (!strcmp(inject, "silence")) x = 0.0f;
            if (!strcmp(inject, "dc")) x = x * 0.5f + 0.1f;
            if (!strcmp(inject, "clip")) x = x * 1000.0f;
        }
    }
    int64_t nonfin = 0, subn = 0, over = 0;
    double  sum[2] = { 0, 0 }, sq = 0;
    float   peak   = 0;
    for (int c = 0; c < 2; c++) {
        for (int64_t i = 0; i < n; i++) {
            const float x = a[(size_t) (c * n + i)];
            if (!std::isfinite(x)) { nonfin++; continue; }
            if (std::fpclassify(x) == FP_SUBNORMAL) subn++;
            peak = std::max(peak, std::fabs(x));
            over += std::fabs(x) > 1.0f;
            sum[c] += x;
            sq += (double) x * x;
        }
    }
    const double total = n > 0 ? 2.0 * n : 1.0, dc_l = n ? sum[0] / n : 0, dc_r = n ? sum[1] / n : 0;
    const double over_pct = 100.0 * over / total, peak_db = peak > 0 ? 20.0 * log10(peak) : -999.0;
    const bool f_nan = nonfin > 0, f_subn = subn > 0, f_sil = n == 0 || peak_db < -60.0;
    const bool f_dc = fabs(dc_l) > 0.02 || fabs(dc_r) > 0.02, f_clip = over_pct > 0.1;
    const bool ok = !(f_nan || f_subn || f_sil || f_dc || f_clip);
    fprintf(stderr,
            "[Check Batch%d] %s samples=%lld nonfinite=%lld%s subnormal=%lld%s peak=%.4f (%.1f dBFS)%s rms=%.5f "
            "dc_l=%+.5f dc_r=%+.5f%s over_fs=%lld (%.4f%%)%s\n",
            b, ok ? "PASS" : "FAIL", (long long) n, (long long) nonfin, f_nan ? " [FAIL]" : "", (long long) subn,
            f_subn ? " [FAIL]" : "", peak, peak_db, f_sil ? " [FAIL silence]" : "", sqrt(sq / total), dc_l, dc_r,
            f_dc ? " [FAIL dc]" : "", (long long) over, over_pct, f_clip ? " [FAIL clip]" : "");
    return ok;
}

int main(int argc, char ** argv) {
    std::string models, request_path, out, codes_out, cancel_at;
    bool        check = false;
    int         runs  = 1;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto value = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--models") models = value();
        else if (a == "--request") request_path = value();
        else if (a == "--out") out = value();
        else if (a == "--codes-out") codes_out = value();
        else if (a == "--cancel-at") cancel_at = value();
        else if (a == "--runs") runs = std::max(1, atoi(value().c_str()));
        else if (a == "--check") check = true;
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (models.empty() || request_path.empty()) {
        fprintf(stderr, "usage: heartmula-generate --models DIR --request FILE.json [--out FILE.wav] [--check] [--runs N]\n"
                        "                          [--cancel-at STAGE[:STEP]] [--codes-out FILE.json]\n");
        return 2;
    }
    std::ifstream     rf(request_path, std::ios::binary);
    std::stringstream body;
    body << rf.rdbuf();
    HeartMuLaRequest req;
    std::string      err;
    if (!rf || !heartmula_parse_request(body.str(), &req, &err)) {
        fprintf(stderr, "FATAL: request: %s\n", rf ? err.c_str() : "cannot read the file");
        return 1;
    }
    const HeartMuLaCatalogue cat = heartmula_discover(models);
    for (const auto & e : cat.errors) fprintf(stderr, "[HeartMuLa] catalogue: %s\n", e.c_str());
    HeartMuLaResolved pick;
    int               status = 0;
    if (!heartmula_resolve(cat, heartmula_request_pick({}, req), &pick, &status, &err)) {
        fprintf(stderr, "FATAL: pick (%d): %s\n", status, err.c_str());
        return 1;
    }
    for (int r = 0; r <= HM_TOKENIZER; r++) fprintf(stderr, "[HeartMuLa] %s: %s\n", k_heartmula_roles[r], pick.file[r].filename.c_str());
    HeartMuLaTokenizer tok;
    if (!heartmula_tokenizer_load(&tok, pick.file[HM_TOKENIZER].path.c_str(), &err)) {
        fprintf(stderr, "FATAL: tokenizer: %s\n", err.c_str());
        return 1;
    }
    const std::string cancel_stage = cancel_at.substr(0, cancel_at.find(':'));
    const int cancel_step = cancel_at.find(':') == std::string::npos ? 0 : atoi(cancel_at.substr(cancel_at.find(':') + 1).c_str());

    HeartMuLaRuntime rt;
    int              rc = 0;
    for (int run = 0; run < runs; run++) {
        bool           cancel = false;
        std::string    last;
        HeartMuLaHooks hooks;
        hooks.cancelled = [&]() { return cancel; };
        hooks.progress  = [&](const char * stage, int step, int total) {
            if (stage != last) fprintf(stderr, "[HeartMuLa] stage %s (%d/%d)\n", stage, step, total);
            last = stage;
            if (!cancel_stage.empty() && cancel_stage == stage && step >= cancel_step) cancel = true;
        };
        HeartMuLaResult res;
        const bool ok = heartmula_generate(&rt, pick, tok, req, false, hooks, &res);
        printf("run %d: %s end_reason=%s frames=%lld/%lld prompt=%lld samples=%lld codes_hash=%016llx audio_hash=%016llx\n",
               run, ok ? "ok" : res.error_code.c_str(), res.end_reason.c_str(), (long long) res.frames,
               (long long) res.max_frames, (long long) res.prompt_tokens, (long long) res.samples,
               (unsigned long long) fnv1a(res.codes.data(), res.codes.size() * 4),
               (unsigned long long) fnv1a(res.audio.data(), res.audio.size() * 4));
        printf("timings: admission %.0f ms, load %.0f ms, prefill %.0f ms, frames %.0f ms, codec %.0f ms\n", res.t_admission,
               res.t_load, res.t_prefill, res.t_frames, res.t_codec);
        if (res.admission) {
            const auto & a = *res.admission;
            printf("admission %s: need %llu free %llu reclaimable %llu reserve %llu weights %llu kv %llu positions %lld -> %s\n",
                   a.phase.c_str(), (unsigned long long) a.need, (unsigned long long) a.free, (unsigned long long) a.reclaimable,
                   (unsigned long long) a.reserve, (unsigned long long) a.weights, (unsigned long long) a.kv,
                   (long long) a.positions, a.ok ? "admitted" : "refused");
        }
        printf("resident after run: %llu bytes\n", (unsigned long long) heartmula_resident_bytes(rt));
        if (!ok) {
            fprintf(stderr, "FATAL: %s: %s\n", res.error_code.c_str(), res.error.c_str());
            rc = res.error_code == "cancelled" ? 3 : 1;
            continue;
        }
        if (!codes_out.empty()) {
            std::ofstream cf(codes_out, std::ios::binary);
            cf << "{\"frames\":" << res.frames << ",\"codebooks\":8,\"frame_rate\":12.5,\"codes\":[";
            for (int64_t t = 0; t < res.frames; t++) {
                cf << (t ? ",[" : "[");
                for (int q = 0; q < HEARTMULA_CODEBOOKS; q++) cf << (q ? "," : "") << res.codes[(size_t) (t * 8 + q)];
                cf << "]";
            }
            cf << "]}";
        }
        if (res.audio.empty()) continue;
        if (check) {
            std::string list = std::getenv("HOTSTEP_CHECK_INJECT") ? std::getenv("HOTSTEP_CHECK_INJECT") : "none";
            std::stringstream ls(list);
            std::string       entry;
            for (int b = 0; std::getline(ls, entry, ','); b++) {
                if (!check_audio(b, res.audio, res.samples, entry.c_str())) rc = rc ? rc : 2;
            }
        }
        if (!out.empty()) {
            const std::string wav = audio_encode_wav_s16(res.audio.data(), (int) res.samples, 48000);
            std::ofstream     wf(out, std::ios::binary);
            wf.write(wav.data(), (std::streamsize) wav.size());
            if (!wf) {
                fprintf(stderr, "FATAL: failed to write %s\n", out.c_str());
                return 1;
            }
        }
    }
    heartmula_free_all(&rt);
    return rc;
}
