#pragma once
// HeartCodec waveform to RVQ tokens, as heartlib's HeartCodecEncoder.tokenize:
// mono doubles to stereo, functional resample to 24 kHz, 29.76 s chunks each with a
// 240-sample tail, four feature streams, query fusion and the shared RVQ.
//
// Phases. One encoder component is resident at a time: MuEncoder, WavLM, Whisper,
// then query, each over every chunk, each freed before the next loads. Admission
// before each load takes the S6 figure (NVML free VRAM on CUDA, ggml's otherwise)
// against the file's bytes plus HEARTMULA_VRAM_RESERVE_MB. Cancel is checked between
// components, chunks, channels and MuEncoder windows and throws heartcodec::Cancelled.
// The encoder pack is never loaded for text generation.
#include "heartmula/encoder-query.h"
#include "heartmula/pipeline.h"

#include <atomic>
#include <exception>
#include <filesystem>
#include <mutex>
#include <thread>

namespace heartcodec {

struct EncoderPaths { std::string rvq, muencoder, wavlm, whisper, query; };

// Paths in `dir` with the converter's default names at the given quant ("F16").
inline EncoderPaths encoder_paths(const std::string & dir, const std::string & quant = "F16") {
    auto f = [&](const std::string & role) { return (std::filesystem::path(dir) / ("heartcodec-" + role + "-" + quant + ".gguf")).string(); };
    return {(std::filesystem::path(dir) / "heartcodec-rvq-F32.gguf").string(), f("muencoder"), f("wavlm"), f("whisper"), f("query")};
}

// Stage outputs for parity tracing: stage, chunk, channel (-1 for stereo), values, dims.
using EncodeTrace = std::function<void(const std::string &, int, int, const std::vector<float> &, const std::vector<int64_t> &)>;

struct EncodeResult {
    std::vector<int64_t> codes;  // [8][frames]
    int64_t frames = 0, chunks = 0;
    std::map<std::string, double> seconds;
};

inline double encoder_config_number(const std::string & config, const char * a, const char * b) {
    using Json = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;
    Json doc(yyjson_read(config.data(), config.size(), 0), yyjson_doc_free);
    require(bool(doc), "invalid encoder config JSON");
    auto v = yyjson_obj_get(yyjson_obj_get(yyjson_obj_get(yyjson_doc_get_root(doc.get()), "feature_extractors"), a), b);
    require(yyjson_is_num(v), std::string("encoder config lacks ") + a + "." + b);
    return yyjson_get_num(v);
}

// f(i) for i in [0, n) on up to hardware_concurrency threads. The per-cell CPU features
// (mels, the 16 kHz resample) are independent, so the result is the serial one bit for bit.
template <typename F> inline void parallel_cells(size_t n, F && f) {
    const size_t k = std::min<size_t>(n, std::max(1u, std::thread::hardware_concurrency()));
    std::atomic<size_t> next{ 0 };
    std::exception_ptr  err;
    std::mutex          m;
    std::vector<std::thread> pool;
    for (size_t t = 0; t < k; ++t) pool.emplace_back([&] {
        for (size_t i; (i = next++) < n;) {
            try { f(i); } catch (...) { std::lock_guard<std::mutex> l(m); if (!err) err = std::current_exception(); }
        }
    });
    for (auto & th : pool) th.join();
    if (err) std::rethrow_exception(err);
}

// The four encoder parts and the RVQ, loaded once. heartmula-prepare keeps one across a whole
// dataset; reloading ~1.1 GB of weights per song was a fixed cost on every sample.
struct Encoders {
    Weights     mu, wavlm, whisper, query, rvq;
    std::string config;
    double      mean = 0, std_dev = 1;
    bool        loaded = false;
};

inline void encoders_load(Encoders & e, const EncoderPaths & paths, ggml_backend_t backend) {
    namespace fs = std::filesystem;
    const uint64_t need = fs::file_size(paths.muencoder) + fs::file_size(paths.wavlm) + fs::file_size(paths.whisper) +
                          fs::file_size(paths.query) + fs::file_size(paths.rvq) + heartmula_reserve_bytes();
    const uint64_t free = heartmula_device_free(ggml_backend_get_device(backend));
    fprintf(stderr, "[HeartCodec] encoder admission: need %.2f GiB, free %.2f GiB -> %s\n", need / 1073741824.0,
            free / 1073741824.0, need <= free ? "admitted" : "refused");
    require(need <= free, "not enough free memory to load the encoder");
    load_encoder_part(e.mu, paths.muencoder, "muencoder", backend);
    load_encoder_part(e.wavlm, paths.wavlm, "wavlm", backend);
    load_encoder_part(e.whisper, paths.whisper, "whisper", backend);
    load_encoder_part(e.query, paths.query, "query", backend);
    e.rvq.load(paths.rvq, "rvq", backend);
    e.config = e.mu.config;
    for (const Weights * w : { &e.wavlm, &e.whisper, &e.query })
        require(w->config == e.config, "encoder pack files come from different configs");
    for (const Weights * w : { &e.mu, &e.wavlm, &e.whisper, &e.query })
        require(w->rvq_hash == e.rvq.file_hash, "encoder pack was converted against a different RVQ");
    e.mean    = encoder_config_number(e.config, "muencoder_config", "mean");
    e.std_dev = encoder_config_number(e.config, "muencoder_config", "std");
    e.loaded  = true;
}

// `cache` keeps the weights for the next call; null loads and frees them here.
inline EncodeResult encode(const EncoderPaths & paths, ggml_backend_t backend, const std::vector<float> & planar,
                           int channels, int64_t samples, int rate, const Cancel & cancelled = {},
                           const EncodeTrace & trace = {}, Encoders * cache = nullptr) {
    require(channels == 1 || channels == 2, "audio must be mono or stereo");
    require(samples > 0 && rate > 0 && planar.size() == size_t(channels * samples), "invalid audio shape");
    for (float v : planar) require(std::isfinite(v), "audio must be finite");
    EncodeResult r;
    auto clock = [] { return std::chrono::steady_clock::now(); };
    auto since = [&](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(clock() - t).count(); };
    auto t0 = clock();
    std::vector<float> stereo(planar);
    if (channels == 1) stereo.insert(stereo.end(), planar.begin(), planar.end());
    int64_t length = samples;
    if (rate != 24000) stereo = resample(stereo, 2, samples, rate, 24000, false, &length);
    r.frames = int64_t(double(length) / 24000.0 * 12.5) + 1;
    r.chunks = (length + enc_chunk_samples - 1) / enc_chunk_samples;
    const int64_t span = enc_chunk_samples + enc_tail;
    auto chunk_audio = [&](int64_t n, int ch) {
        std::vector<float> a(size_t(span), 0.0f);
        const int64_t start = n * enc_chunk_samples, count = std::min(enc_chunk_samples, length - start);
        std::copy_n(stereo.data() + ch * length + start, count, a.data());
        return a;
    };
    if (trace) trace("audio_24k", 0, -1, stereo, {2, length});
    r.seconds["resample"] = since(t0);
    // Per chunk, per channel: acoustic, semantic, whisper, wavlm [dim, 744].
    const size_t cells = size_t(r.chunks) * 2;
    std::vector<std::vector<float>> acoustic(cells), semantic(cells), whisper(cells), wavlm(cells);
    auto cell = [](int64_t n, int ch) { return size_t(n * 2 + ch); };
    Encoders   local;
    Encoders & e = cache ? *cache : local;
    if (!e.loaded) {
        t0 = clock();
        check_cancel(cancelled);
        encoders_load(e, paths, backend);
        r.seconds["load"] = since(t0);
    }
    {
        t0 = clock();
        const Weights & w = e.mu;
        const double mean = e.mean, std_dev = e.std_dev;
        // Every cell's mel on the CPU threads first; the GPU passes then run back to back.
        const MuMelBuffers bufs = muencoder_mel_buffers(w);
        std::vector<std::vector<float>> mels(cells);
        parallel_cells(cells, [&](size_t c) {
            int64_t frames = 0;
            const auto a = chunk_audio(int64_t(c / 2), int(c % 2));
            mels[c] = muencoder_mel(bufs, a.data(), (enc_chunk_samples + enc_tail) / 960 * 960, mean, std_dev, &frames);
            require(frames == enc_frames * 4, "unexpected MuEncoder mel length");
        });
        r.seconds["mu_mel"] = since(t0);
        for (int64_t n = 0; n < r.chunks; ++n) for (int ch = 0; ch < 2; ++ch) {
            check_cancel(cancelled);
            auto audio = chunk_audio(n, ch);
            if (trace) trace("chunk_24k", int(n), ch, audio, {span});
            std::vector<float> mel;
            muencoder_run(w, backend, audio.data(), mean, std_dev, acoustic[cell(n, ch)], semantic[cell(n, ch)], cancelled,
                          trace ? &mel : nullptr, &mels[cell(n, ch)]);
            mels[cell(n, ch)].clear();
            mels[cell(n, ch)].shrink_to_fit();
            if (trace) {
                trace("mu_mel", int(n), ch, mel, {128, enc_frames * 4});
                trace("mu_acoustic", int(n), ch, acoustic[cell(n, ch)], {enc_frames, 1024});
                trace("mu_semantic", int(n), ch, semantic[cell(n, ch)], {enc_frames, 1024});
            }
        }
        r.seconds["muencoder"] = since(t0);
    }
    // Both 16 kHz paths share the resampler; keep its output for Whisper too.
    std::vector<std::vector<float>> audio16(cells);
    {
        t0 = clock();
        check_cancel(cancelled);
        const Weights & w = e.wavlm;
        parallel_cells(cells, [&](size_t c) {
            int64_t out = 0;
            audio16[c] = resample(chunk_audio(int64_t(c / 2), int(c % 2)), 1, span, 24000, 16000, true, &out);
            require(out == enc_samples_16k, "unexpected 16 kHz length");
        });
        for (int64_t n = 0; n < r.chunks; ++n) for (int ch = 0; ch < 2; ++ch) {
            check_cancel(cancelled);
            const int64_t out = int64_t(audio16[cell(n, ch)].size());
            if (trace) trace("wav_16k", int(n), ch, audio16[cell(n, ch)], {out});
            wavlm[cell(n, ch)] = wavlm_run(w, backend, audio16[cell(n, ch)]);
            if (trace) trace("wavlm", int(n), ch, wavlm[cell(n, ch)], {enc_frames, 768});
        }
        r.seconds["wavlm"] = since(t0);
    }
    {
        t0 = clock();
        check_cancel(cancelled);
        const Weights & w = e.whisper;
        std::vector<std::vector<float>> mels(cells);
        parallel_cells(cells, [&](size_t c) { mels[c] = whisper_mel(audio16[c].data(), int64_t(audio16[c].size())); });
        r.seconds["whisper_mel"] = since(t0);
        for (int64_t n = 0; n < r.chunks; ++n) for (int ch = 0; ch < 2; ++ch) {
            check_cancel(cancelled);
            auto mel = std::move(mels[cell(n, ch)]);
            if (trace) trace("whisper_mel", int(n), ch, mel, {80, 3000});
            whisper[cell(n, ch)] = whisper_run(w, backend, mel);
            if (trace) trace("whisper", int(n), ch, whisper[cell(n, ch)], {enc_frames, 768});
        }
        r.seconds["whisper"] = since(t0);
    }
    {
        t0 = clock();
        check_cancel(cancelled);
        const Weights & w = e.query;
        const Weights & rvq = e.rvq;
        // Interleave channels per frame: [left dims, right dims].
        auto stereo_features = [&](const std::vector<std::vector<float>> & f, int64_t n, int64_t dim) {
            std::vector<float> out(size_t(2 * dim * enc_frames));
            for (int ch = 0; ch < 2; ++ch) for (int64_t t = 0; t < enc_frames; ++t)
                std::copy_n(f[cell(n, ch)].data() + t * dim, dim, out.data() + t * 2 * dim + ch * dim);
            return out;
        };
        std::vector<int64_t> packed;
        for (int64_t n = 0; n < r.chunks; ++n) {
            check_cancel(cancelled);
            auto fa = stereo_features(acoustic, n, 1024), fs = stereo_features(semantic, n, 1024);
            auto fw = stereo_features(whisper, n, 768), fl = stereo_features(wavlm, n, 768);
            if (trace) {
                trace("feature_acoustic", int(n), -1, fa, {enc_frames, 2048});
                trace("feature_semantic", int(n), -1, fs, {enc_frames, 2048});
                trace("feature_whisper", int(n), -1, fw, {enc_frames, 1536});
                trace("feature_wavlm", int(n), -1, fl, {enc_frames, 1536});
            }
            auto queries = query_run(w, backend, fa, fs, fw, fl, enc_frames);
            if (trace) trace("query", int(n), -1, queries, {enc_frames / 2, 512});
            auto codes = rvq_encode(rvq, queries, enc_frames / 2);
            packed.insert(packed.end(), codes.begin(), codes.end());
        }
        r.seconds["query"] = since(t0);
        // The first `frames` [frame][8] rows, transposed to [8][frames]. heartlib's
        // [:output_frames] slice clamps: a length of exactly N chunks asks for one
        // frame more than N * 372 exist.
        r.frames = std::min(r.frames, int64_t(packed.size() / 8));
        r.codes.resize(size_t(8 * r.frames));
        for (int64_t t = 0; t < r.frames; ++t) for (int q = 0; q < 8; ++q)
            r.codes[size_t(q * r.frames + t)] = packed[size_t(t * 8 + q)];
    }
    return r;
}

} // namespace heartcodec
