#pragma once
#include "codec-scalar.h"

namespace heartcodec {

constexpr int sample_rate = 48000;
constexpr int code_samples = 3840;
constexpr int chunk_codes = 372;
constexpr int hop_codes = 320;
constexpr int overlap_codes = chunk_codes - hop_codes;
constexpr int chunk_frames = chunk_codes * 2;
constexpr int overlap_frames = overlap_codes * 2;

using Noise = std::function<std::vector<float>(size_t)>;
using Progress = std::function<void(int, int)>;
using Trace = std::function<void(const std::string &, int, const std::vector<float> &, const std::vector<int64_t> &)>;
// A replacement for solve_euler, same arguments, called once per window (codec-plugins.h).
// Empty = the native, parity-proven Euler loop.
using Solve = std::function<std::vector<float>(const Velocity &, const std::vector<float> &, const std::vector<float> &,
                                               const std::vector<float> &, int64_t, int64_t, int, float, const Cancel &)>;

inline int64_t padded_code_length(int64_t frames) {
    require(frames > 0 && frames <= 3000, "code frame count must be in [1,3000]");
    return std::max<int64_t>(chunk_codes, ((frames - overlap_codes + hop_codes - 1) / hop_codes) * hop_codes + overlap_codes);
}

// The repeat source changes after the first short-clip crop. This mirrors the
// reference's two separate repeat-and-crop loops at the chunk boundary.
inline std::vector<int64_t> pad_codes(const std::vector<int64_t> & codes, int64_t frames) {
    require(codes.size() == size_t(8 * frames), "expected [8,frames] codes");
    auto padded = padded_code_length(frames);
    int64_t repeat_length = std::max<int64_t>(chunk_codes, frames);
    std::vector<int64_t> result(size_t(padded * 8));
    for (int q = 0; q < 8; ++q) for (int64_t t = 0; t < padded; ++t) {
        auto value = codes[size_t(q * frames + (t % repeat_length) % frames)];
        require(value >= 0 && value < 8192, "code outside [0,8192)");
        result[size_t(q * padded + t)] = value;
    }
    return result;
}

inline void append_crossfade(std::vector<float> & output, const std::vector<float> & next, int64_t samples) {
    require(samples >= overlap_codes * code_samples && next.size() == size_t(samples * 2), "invalid crossfade chunk");
    if (output.empty()) { output = next; return; }
    const int64_t old = int64_t(output.size() / 2), overlap = overlap_codes * code_samples;
    require(old >= overlap, "crossfade output too short");
    const int64_t total = old + samples - overlap;
    std::vector<float> combined(size_t(total * 2));
    for (int ch = 0; ch < 2; ++ch) {
        const float * previous = output.data() + ch * old;
        const float * current = next.data() + ch * samples;
        float * dst = combined.data() + ch * total;
        std::copy_n(previous, old, dst);
        for (int64_t t = 0; t < overlap; ++t) {
            double ramp = double(t) / double(overlap - 1);
            dst[old - overlap + t] = float(previous[old - overlap + t] * (1.0 - ramp) + current[t] * ramp);
        }
        std::copy_n(current + overlap, samples - overlap, dst + old);
    }
    output.swap(combined);
}

inline std::vector<float> decode(const Weights & rvq, const Weights & flow, const Weights & scalar,
                                 ggml_backend_t backend, const std::vector<int64_t> & codes, int64_t frames,
                                 const Noise & noise, int steps = 10, float guidance = 1.25f,
                                 const Cancel & cancelled = {}, const Progress & progress = {}, const Trace & trace = {},
                                 const Solve & solve = {}) {
    require(flow.fingerprint == scalar.fingerprint, "flow/scalar base fingerprint mismatch");
    require(!rvq.file_hash.empty() && flow.rvq_hash == rvq.file_hash && scalar.rvq_hash == rvq.file_hash,
            "flow/scalar RVQ dependency hash mismatch");
    require(bool(noise), "noise provider is required");
    auto padded = pad_codes(codes, frames);
    const int64_t padded_frames = int64_t(padded.size() / 8);
    const int chunks = int((padded_frames - overlap_codes) / hop_codes);
    auto get_noise = [&](size_t size) {
        auto v = noise(size);
        require(v.size() == size, "noise provider returned wrong shape");
        for (float f : v) require(std::isfinite(f), "nonfinite noise");
        return v;
    };
    // Preserve the first_latent random draw, even though the initial overlap is zero.
    auto initial = get_noise(size_t(chunk_frames * 256));
    std::vector<float> previous, output;
    int velocity_index = 0;
    Velocity velocity = [&](const std::vector<float> & x, int64_t f, const std::vector<float> & t) {
        auto v = flow_velocity(flow, backend, x, f, t);
        if (trace) trace("flow_velocity", velocity_index, v, {int64_t(t.size()), f, 256});
        ++velocity_index;
        return v;
    };
    for (int n = 0; n < chunks; ++n) {
        check_cancel(cancelled);
        if (progress) progress(n, chunks);
        std::vector<int64_t> chunk(size_t(8 * chunk_codes));
        for (int q = 0; q < 8; ++q)
            std::copy_n(padded.data() + q * padded_frames + n * hop_codes, chunk_codes, chunk.data() + q * chunk_codes);
        auto features = rvq_decode(rvq, backend, chunk, chunk_codes);
        if (trace) trace("rvq_vectors", n, features, {1, chunk_codes, 512});
        auto mu = condition(flow, backend, features, chunk_codes);
        std::vector<float> context(size_t(chunk_frames * 256), 0);
        int overlap = n == 0 ? 0 : overlap_frames;
        if (n == 0) context = initial;
        else {
            std::copy_n(previous.end() - overlap_frames * 256, overlap_frames * 256, context.begin());
            auto tail = get_noise(size_t((chunk_frames - overlap_frames) * 256));
            std::copy(tail.begin(), tail.end(), context.begin() + overlap_frames * 256);
        }
        auto supplied = get_noise(size_t(chunk_frames * 256));
        previous = solve ? solve(velocity, supplied, context, mu, chunk_frames, overlap, steps, guidance, cancelled)
                         : solve_euler(velocity, supplied, context, mu, chunk_frames, overlap, steps, guidance, cancelled);
        if (trace) trace("latent", n, previous, {1, chunk_frames, 256});
        auto wav = scalar_decode(scalar, backend, previous, chunk_frames, cancelled);
        if (trace) trace("scalar_output", n, wav, {2, 1, chunk_frames * 1920});
        append_crossfade(output, wav, chunk_codes * code_samples);
    }
    const int64_t target = frames * code_samples, available = int64_t(output.size() / 2);
    require(available >= target, "decoded audio shorter than requested");
    std::vector<float> result(size_t(target * 2));
    for (int ch = 0; ch < 2; ++ch) std::copy_n(output.data() + ch * available, target, result.data() + ch * target);
    if (progress) progress(chunks, chunks);
    return result;
}

} // namespace heartcodec
