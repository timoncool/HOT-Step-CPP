#pragma once
// train/heartmula-prepare.h — `ace-train heartmula-prepare`: a sample list
// (audio + tags + lyrics) -> HeartCodec codes and prompt ids, the cache the
// HeartMuLa LoRA trainer reads.
//
// HOT-Step file, TRAINING-SIDE (ace-train only). Implements the data half of
// engine/docs/HEARTMULA_TRAINING_CONTRACT.md: section 2 (prompt) and section 3
// (context limit). Nothing here is ACE: no latents, no FSQ, no VAE.
//
//   ace-train heartmula-prepare --dataset <samples.json> --out <cache dir>
//                               --models <dir> [--encoder <dir>] [--ffmpeg <path>] [--force]
//
// ── Input ──────────────────────────────────────────────────────────────────
//
// <samples.json> is a JSON array, or an object with a "samples" array, of
// { "id": "...", "audio": "<path>", "tags": "...", "lyrics": "..." }. The caller
// (the server) owns caption resolution and exclusions; `id` comes back in every
// report so skips map onto its own sample ids. `id` defaults to the audio basename.
// One entry is one training sample, as long as its audio: there is no cutting here.
//
// ── What one sample becomes ────────────────────────────────────────────────
//
//   prompt  heartmula_prompt_ids(), the inference builder itself: tags lower-cased
//           and wrapped in <tag>...</tag>, BOS/EOS once each, a 0 at the MuQ slot,
//           then the lyrics. muq_idx = len(tag ids). With no reference audio the
//           trainer fills that row with muq_linear(0), as inference does.
//   codes   heartcodec::encode(), the inference-side encoder, on the audio at its
//           NATIVE rate: the encoder does its own resample to 24 kHz, as heartlib's
//           tokenize does after torchaudio.load. WAV is read natively; every other
//           format goes through ffmpeg to f32 WAV with no -ar, so no second
//           resampler touches the signal. Written frame-major int32 little-endian,
//           index t * 8 + k, to <out>/codes/<key>.i32 (T * 32 bytes).
//
// Skipped samples are reported with the server's reasons:
//   invalid       empty or over-4096-byte tags (inference refuses both), text the
//                 tokenizer rejects, or audio shorter than 2 frames
//   over-context  prompt + T - 1 positions over the backbone's 8192 (contract §3);
//                 never truncated
//   no-audio      the file is missing or cannot be decoded
//   encode-failed the encoder threw on this sample
// Running out of VRAM is not a sample fault: it fails the run.
//
// ── Cache fingerprint ──────────────────────────────────────────────────────
//
// key = first 16 hex of sha256("heartmula-prepare|v<VERSION>|<audio path>|<bytes>|
//       <mtime s>|<encoder identity>"), encoder identity = each encoder file's
//       name and content sha256 plus the RVQ codebook's sha256 (hashing the
//       pack costs a few seconds per run). Codes depend on nothing
//       else, so a codes file named by its key is reused as-is (a cancelled run
//       resumes for free) unless --force. Prompt ids are cheap and are rebuilt
//       every run. HEARTMULA_PREPARE_VERSION bumps whenever the codes or the
//       manifest change meaning; a reader refuses any other version.
//
// The server works out staleness without running the engine: a sample is stale
// when its audio's size or whole-second mtime, or its tags or lyrics, differ
// from what its manifest row recorded.
//
// <out>/heartmula_prepare.json is written atomically at the end; codes files no
// manifest row names are removed then.
#include "heartmula/codec-encode.h"
#include <memory>

#include "audio-io.h"
#include "hot-step-fsutf8.h"
#include "train/preprocess-io.h"
#include "version.h"
#include "yyjson.h"

#include <filesystem>
#include <set>

#define HEARTMULA_PREPARE_VERSION 2
#define HEARTMULA_PREPARE_MANIFEST "heartmula_prepare.json"

struct HeartmulaPrepareArgs {
    std::string dataset, out_dir, models_dir, encoder_dir, ffmpeg = "ffmpeg";
    bool        force = false;
};

struct HeartmulaPrepareSample {
    std::string id, audio, tags, lyrics, key;
    long long   bytes = 0, mtime_s = 0;
    std::vector<int32_t> prompt;
    int64_t     muq_idx = 0, frames = 0;
    double      seconds = 0;
    bool        reused  = false;
};

struct HeartmulaPrepareSkip { std::string id, reason, detail; };

static bool hp_read_file(const std::string & path, std::vector<uint8_t> * out) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out->resize(n > 0 ? (size_t) n : 0);
    const bool ok = n > 0 && fread(out->data(), 1, out->size(), f) == out->size();
    fclose(f);
    return ok;
}

// Planar stereo at the file's own rate. WAV natively, everything else via ffmpeg.
static float * hp_decode(const HeartmulaPrepareArgs & a, const std::string & src, const std::string & tmp_wav,
                         int * T, int * sr, std::string * err) {
    std::vector<uint8_t> buf;
    if (pm_ext_of(src) != ".wav") {
        if (a.ffmpeg.empty()) { *err = "not WAV and no --ffmpeg"; return nullptr; }
        const std::string cmd = "\"" + a.ffmpeg + "\" -y -v error -i \"" + src + "\" -ac 2 -c:a pcm_f32le -f wav \"" + tmp_wav + "\"";
#ifdef _WIN32
        const int rc = hs_system("\"" + cmd + "\"");  // cmd.exe strips the outer pair
#else
        const int rc = hs_system(cmd);
#endif
        const bool ok = rc == 0 && hp_read_file(tmp_wav, &buf);
        hs_remove(tmp_wav);
        if (!ok) { *err = "ffmpeg exit " + std::to_string(rc); return nullptr; }
    } else if (!hp_read_file(src, &buf)) {
        *err = "cannot read the file";
        return nullptr;
    }
    float * planar = audio_io_read_wav_buf(buf.data(), buf.size(), T, sr);
    if (!planar || *T <= 0 || *sr <= 0) { free(planar); *err = "cannot decode the WAV"; return nullptr; }
    return planar;
}

static bool hp_load_samples(const std::string & path, std::vector<HeartmulaPrepareSample> * out, std::string * err) {
    std::vector<uint8_t> buf;
    if (!hp_read_file(path, &buf)) { *err = "cannot read " + path; return false; }
    yyjson_doc * doc = yyjson_read((const char *) buf.data(), buf.size(), 0);
    if (!doc) { *err = path + " is not valid JSON"; return false; }
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * arr  = yyjson_is_obj(root) ? yyjson_obj_get(root, "samples") : root;
    if (!yyjson_is_arr(arr)) { yyjson_doc_free(doc); *err = path + ": expected an array of samples"; return false; }
    std::set<std::string> ids;
    size_t i, n;
    yyjson_val * v;
    yyjson_arr_foreach(arr, i, n, v) {
        HeartmulaPrepareSample s;
        s.audio  = pm_js_str(v, "audio");
        s.tags   = pm_js_str(v, "tags");
        s.lyrics = pm_js_str(v, "lyrics");
        s.id     = pm_js_str(v, "id", pm_basename(s.audio).c_str());
        if (s.audio.empty() || !ids.insert(s.id).second) {
            yyjson_doc_free(doc);
            *err = "sample " + std::to_string(i) + ": " + (s.audio.empty() ? "no audio path" : "duplicate id " + s.id);
            return false;
        }
        out->push_back(std::move(s));
    }
    yyjson_doc_free(doc);
    return true;
}

static bool hp_write_codes(const std::string & path, const heartcodec::EncodeResult & r) {
    std::string bytes((size_t) r.frames * 8 * sizeof(int32_t), '\0');
    int32_t * dst = (int32_t *) bytes.data();
    for (int64_t t = 0; t < r.frames; ++t)
        for (int k = 0; k < 8; ++k) dst[t * 8 + k] = (int32_t) r.codes[(size_t) (k * r.frames + t)];
    return pm_write_atomic(path, bytes);
}

static int heartmula_prepare_run(const HeartmulaPrepareArgs & a) {
    namespace fs = std::filesystem;
    if (a.dataset.empty() || a.out_dir.empty() || a.models_dir.empty()) {
        fprintf(stderr, "heartmula-prepare: --dataset, --out and --models are required\n");
        return 2;
    }
    const char * tf32 = getenv("NVIDIA_TF32_OVERRIDE");
    if (!tf32 || strcmp(tf32, "0") != 0) {
        fprintf(stderr, "heartmula-prepare: NVIDIA_TF32_OVERRIDE=0 must be set before the first CUDA context\n");
        return 1;
    }
    std::vector<HeartmulaPrepareSample> samples;
    std::string err;
    if (!hp_load_samples(a.dataset, &samples, &err)) { fprintf(stderr, "heartmula-prepare: %s\n", err.c_str()); return 1; }

    // Tokenizer and RVQ as the inference catalogue resolves them; the encoder pack
    // must have been converted against that same RVQ (encode() checks).
    const HeartMuLaCatalogue cat = heartmula_discover(a.models_dir);
    const HeartMuLaFile * tok_file = nullptr, * rvq_file = nullptr;
    for (const auto & f : cat.files) {
        if (f.compatible && f.role == HM_TOKENIZER && !tok_file) tok_file = &f;
        if (f.compatible && f.role == HM_RVQ && !rvq_file) rvq_file = &f;
    }
    if (!tok_file || !rvq_file) {
        fprintf(stderr, "heartmula-prepare: no compatible heartmula-tokenizer.gguf and heartcodec-rvq under %s\n", a.models_dir.c_str());
        return 1;
    }
    HeartMuLaTokenizer tok;
    if (!heartmula_tokenizer_load(&tok, tok_file->path.c_str(), &err)) {
        fprintf(stderr, "heartmula-prepare: %s: %s\n", tok_file->path.c_str(), err.c_str());
        return 1;
    }
    std::string enc_dir = a.encoder_dir;
    if (enc_dir.empty()) enc_dir = fs::exists(fs::u8path(a.models_dir) / "heartmula") ? (fs::u8path(a.models_dir) / "heartmula").u8string() : a.models_dir;
    heartcodec::EncoderPaths paths = heartcodec::encoder_paths(enc_dir);
    paths.rvq = rvq_file->path;
    fprintf(stderr, "[HeartMuLa-Prepare] stage hashing the encoder pack\n");
    std::string identity;
    for (const std::string * p : { &paths.muencoder, &paths.wavlm, &paths.whisper, &paths.query }) {
        yue2_aitk::sha256::digest digest;
        if (!pm_file_exists(*p) || !yue2_aitk::sha256::file(fs::u8path(*p), digest)) {
            fprintf(stderr, "heartmula-prepare: encoder file missing or unreadable: %s (pass --encoder <dir>)\n", p->c_str());
            return 1;
        }
        identity += pm_basename(*p) + ":" + digest.hex() + "|";
    }
    identity += "rvq:" + rvq_file->rvq_sha256;
    const int64_t limit = heartmula_global_config().n_ctx;

    const std::string codes_dir = (fs::u8path(a.out_dir) / "codes").u8string();
    if (!pm_mkdir_p(codes_dir)) { fprintf(stderr, "heartmula-prepare: cannot create %s\n", codes_dir.c_str()); return 1; }

    std::vector<HeartmulaPrepareSample> ok;
    std::vector<HeartmulaPrepareSkip>   skipped;
    BackendPair bp = {};
    bool        bp_up = false;
    // Encoder weights for the whole run; freed before the backend they live on.
    auto        encoders = std::make_unique<heartcodec::Encoders>();
    double      enc_seconds = 0, enc_audio = 0;
    int         encoded = 0;
    fprintf(stderr, "[HeartMuLa-Prepare] stage encoding %zu sample(s)\n", samples.size());
    auto skip = [&](const HeartmulaPrepareSample & s, const char * reason, const std::string & detail) {
        fprintf(stderr, "[HeartMuLa-Prepare] skip %s: %s (%s)\n", s.id.c_str(), reason, detail.c_str());
        skipped.push_back({ s.id, reason, detail });
    };
    for (size_t i = 0; i < samples.size(); ++i) {
        HeartmulaPrepareSample s = samples[i];
        fprintf(stderr, "[HeartMuLa-Prepare] %zu/%zu %s: working\n", i + 1, samples.size(), s.id.c_str());
        if (!pm_stat_file(s.audio, &s.bytes, &s.mtime_s)) { skip(s, "no-audio", "file not found"); continue; }
        s.mtime_s /= 1000;
        if (pm_trim(s.tags).empty() || s.tags.size() > 4096) { skip(s, "invalid", "tags must be non-empty text of at most 4096 bytes"); continue; }
        try {
            s.prompt = heartmula_prompt_ids(tok, s.tags, s.lyrics, &s.muq_idx);
        } catch (const std::exception & e) { skip(s, "invalid", std::string("tokenizer: ") + e.what()); continue; }
        const int64_t S = (int64_t) s.prompt.size();
        const std::string id_text = "heartmula-prepare|v" + std::to_string(HEARTMULA_PREPARE_VERSION) + "|" + s.audio + "|" +
                                    std::to_string(s.bytes) + "|" + std::to_string(s.mtime_s) + "|" + identity;
        s.key = yue2_aitk::sha256::bytes(id_text.data(), id_text.size()).hex().substr(0, 16);
        const std::string codes_path = (fs::u8path(codes_dir) / (s.key + ".i32")).u8string();
        long long cached = 0;
        if (!a.force && pm_stat_file(codes_path, &cached, nullptr) && cached > 0 && cached % 32 == 0) {
            s.frames = cached / 32;
            s.reused = true;
        } else {
            int T = 0, sr = 0;
            const auto td = std::chrono::steady_clock::now();
            float * planar = hp_decode(a, s.audio, (fs::u8path(a.out_dir) / (s.key + ".tmp.wav")).u8string(), &T, &sr, &err);
            if (!planar) { skip(s, "no-audio", err); continue; }
            std::vector<float> audio(planar, planar + (size_t) 2 * T);
            free(planar);
            const double decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - td).count();
            // encode()'s own frame count, before spending an encode on a sample that cannot fit.
            const int64_t est = (int64_t) (double(T) / sr * 12.5) + 1;
            if (est < 2) { skip(s, "invalid", "audio shorter than 2 frames"); continue; }
            if (S + est - 1 > limit + 2) {
                skip(s, "over-context", std::to_string(S + est - 1) + " positions, limit " + std::to_string(limit));
                continue;
            }
            if (!bp_up) { bp = backend_init("HeartMuLa prepare"); bp_up = true; }
            const auto t0 = std::chrono::steady_clock::now();
            heartcodec::EncodeResult r;
            try {
                r = heartcodec::encode(paths, bp.backend, audio, 2, T, sr, {}, {}, encoders.get());
            } catch (const std::exception & e) {
                if (strstr(e.what(), "not enough free memory")) {
                    fprintf(stderr, "heartmula-prepare: %s\n", e.what());
                    { encoders.reset(); backend_release(bp.backend, bp.cpu_backend); }
                    return 1;
                }
                if (!encoders->loaded) encoders = std::make_unique<heartcodec::Encoders>();  // a failed load starts over
                skip(s, "encode-failed", e.what());
                continue;
            }
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (!hp_write_codes(codes_path, r)) {
                fprintf(stderr, "heartmula-prepare: cannot write %s\n", codes_path.c_str());
                { encoders.reset(); backend_release(bp.backend, bp.cpu_backend); }
                return 1;
            }
            s.frames = r.frames;
            enc_seconds += dt;
            enc_audio += double(T) / sr;
            encoded++;
            fprintf(stderr, "[HeartMuLa-Prepare] %zu/%zu %s: %.1f s of audio encoded in %.1f s (decode %.1f s;",
                    i + 1, samples.size(), s.id.c_str(), double(T) / sr, dt, decode_s);
            for (const auto & kv : r.seconds) fprintf(stderr, " %s %.1f", kv.first.c_str(), kv.second);
            fprintf(stderr, ")\n");
        }
        s.seconds = double(s.frames) / HEARTMULA_FRAME_RATE;
        if (s.frames < 2) { skip(s, "invalid", "audio shorter than 2 frames"); continue; }
        if (S + s.frames - 1 > limit) {
            skip(s, "over-context", std::to_string(S + s.frames - 1) + " positions, limit " + std::to_string(limit));
            continue;
        }
        fprintf(stderr, "[HeartMuLa-Prepare] %zu/%zu %s: %s, S=%lld T=%lld\n", i + 1, samples.size(), s.id.c_str(),
                s.reused ? "cached" : "encoded", (long long) S, (long long) s.frames);
        ok.push_back(std::move(s));
    }
    if (bp_up) { encoders.reset(); backend_release(bp.backend, bp.cpu_backend); }

    // Manifest.
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "format", "heartmula-prepare");
    yyjson_mut_obj_add_int(doc, root, "version", HEARTMULA_PREPARE_VERSION);
    yyjson_mut_obj_add_strcpy(doc, root, "producer", (std::string("ace-train ") + ACE_VERSION).c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "created_at", pm_iso8601_utc_now().c_str());
    yyjson_mut_obj_add_int(doc, root, "context_limit", limit);
    yyjson_mut_obj_add_real(doc, root, "frame_rate", HEARTMULA_FRAME_RATE);
    yyjson_mut_obj_add_int(doc, root, "codebooks", 8);
    yyjson_mut_obj_add_str(doc, root, "codes_layout", "frame_major int32le, index t * 8 + k");
    yyjson_mut_obj_add_strcpy(doc, root, "tokenizer_sha256", tok_file->tokenizer_json.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "encoder_identity", identity.c_str());
    yyjson_mut_val * arr = yyjson_mut_arr(doc);
    std::set<std::string> keep;
    for (const auto & s : ok) {
        yyjson_mut_val * o = yyjson_mut_arr_add_obj(doc, arr);
        yyjson_mut_obj_add_strcpy(doc, o, "id", s.id.c_str());
        yyjson_mut_obj_add_strcpy(doc, o, "audio", s.audio.c_str());
        yyjson_mut_obj_add_int(doc, o, "audio_bytes", s.bytes);
        yyjson_mut_obj_add_int(doc, o, "audio_mtime_s", s.mtime_s);
        yyjson_mut_obj_add_strcpy(doc, o, "tags", s.tags.c_str());
        yyjson_mut_obj_add_strcpy(doc, o, "lyrics", s.lyrics.c_str());
        yyjson_mut_obj_add_strcpy(doc, o, "key", s.key.c_str());
        yyjson_mut_obj_add_strcpy(doc, o, "codes", ("codes/" + s.key + ".i32").c_str());
        yyjson_mut_obj_add_int(doc, o, "frames", s.frames);
        yyjson_mut_obj_add_real(doc, o, "seconds", s.seconds);
        yyjson_mut_obj_add_int(doc, o, "muq_idx", s.muq_idx);
        yyjson_mut_obj_add_int(doc, o, "positions", (int64_t) s.prompt.size() + s.frames - 1);
        yyjson_mut_val * p = yyjson_mut_arr(doc);
        for (int32_t id : s.prompt) yyjson_mut_arr_add_int(doc, p, id);
        yyjson_mut_obj_add_val(doc, o, "prompt_ids", p);
        keep.insert(s.key + ".i32");
    }
    yyjson_mut_obj_add_val(doc, root, "samples", arr);
    yyjson_mut_val * sk = yyjson_mut_arr(doc);
    for (const auto & s : skipped) {
        yyjson_mut_val * o = yyjson_mut_arr_add_obj(doc, sk);
        yyjson_mut_obj_add_strcpy(doc, o, "id", s.id.c_str());
        yyjson_mut_obj_add_strcpy(doc, o, "reason", s.reason.c_str());
        yyjson_mut_obj_add_strcpy(doc, o, "detail", s.detail.c_str());
    }
    yyjson_mut_obj_add_val(doc, root, "skipped", sk);
    yyjson_mut_val * st = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_int(doc, st, "encoded", encoded);
    yyjson_mut_obj_add_int(doc, st, "reused", (int64_t) std::count_if(ok.begin(), ok.end(), [](const auto & s) { return s.reused; }));
    yyjson_mut_obj_add_real(doc, st, "encoded_audio_seconds", enc_audio);
    yyjson_mut_obj_add_real(doc, st, "encode_seconds", enc_seconds);
    yyjson_mut_obj_add_real(doc, st, "seconds_per_audio_minute", enc_audio > 0 ? enc_seconds / (enc_audio / 60.0) : 0.0);
    yyjson_mut_obj_add_val(doc, root, "stats", st);
    size_t len  = 0;
    char * json = yyjson_mut_write(doc, YYJSON_WRITE_PRETTY, &len);
    const bool wrote = json && pm_write_atomic((fs::u8path(a.out_dir) / HEARTMULA_PREPARE_MANIFEST).u8string(), std::string(json, len));
    free(json);
    yyjson_mut_doc_free(doc);
    if (!wrote) { fprintf(stderr, "heartmula-prepare: cannot write the manifest\n"); return 1; }

    std::error_code ec;
    for (const auto & e : fs::directory_iterator(fs::u8path(codes_dir), ec))
        if (e.is_regular_file() && !keep.count(e.path().filename().u8string())) fs::remove(e.path(), ec);

    fprintf(stderr, "[HeartMuLa-Prepare] %zu prepared (%d encoded), %zu skipped; %.1f s per minute of audio\n", ok.size(),
            encoded, skipped.size(), enc_audio > 0 ? enc_seconds / (enc_audio / 60.0) : 0.0);
    if (ok.empty()) { fprintf(stderr, "heartmula-prepare: no sample could be prepared\n"); return 1; }
    return 0;
}
