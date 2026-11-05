// moss-mel.h: Whisper-style log-mel frontend for the MOSS-Music audio tower.
//
// audio (16 kHz mono f32) -> log-mel [n_mels=128, frames], 100 frames/second.
// The encoder's conv stem then divides the time axis by 8, giving the 12.5
// audio-tokens/second that moss.audio.tokens_per_second records.
//
// Validated against a captured HF WhisperFeatureExtractor dump by
// engine/tools/moss-mel-ref.py (corr 0.9999942 / 0.9999982; the residual is the
// reference's own bf16 storage precision).
//
// WHY THIS FILE EXISTS AT ALL
// ---------------------------
// There is no mel filterbank anywhere else in engine/src/. denoiser.h and
// spectral-lifter.h both carry a minimal radix-2 FFT, but NEITHER can be reused
// here: Whisper's n_fft is 400, which is 2^4 * 25 and not a power of two. The
// transform below is therefore recursive radix-2 with a naive-DFT fallback for
// odd sizes -- it bottoms out at N=25 for our case. That is the same strategy
// whisper.cpp uses, for the same reason.
//
// FOUR DETAILS THAT FAIL QUIETLY IF YOU GET THEM WRONG
// ----------------------------------------------------
// 1. mel_scale AND norm are both "slaney" (librosa/HF defaults), not HTK. Using
//    HTK mel shifts every filter centre and the encoder still runs.
// 2. The final STFT frame is DROPPED. An off-by-one shifts the whole time axis
//    by one hop (10 ms) relative to the reference.
// 3. The dynamic-range clamp derives its floor from the GLOBAL maximum:
//        log = max(log, log.max() - 8)
//    so it couples the entire spectrogram. If long audio is ever chunked, the
//    max MUST be taken over the whole utterance -- per-chunk maxima give each
//    chunk a different floor and the frames stop being comparable. That would
//    present as encoder drift on long tracks only, which is a horrible bug to
//    find. mel_log_scale() is split out from mel_power() precisely so a chunked
//    caller can accumulate power first and scale once.
// 4. The final rescale is (log + 4) / 4. Omitting it leaves values ~4x too
//    large and the conv stem saturates.

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <map>
#include <vector>

namespace moss {

struct MelParams {
    int sample_rate = 16000;
    int n_fft       = 400;
    int hop_length  = 160;
    int n_mels      = 128;
};

namespace mel_detail {

using Cpx = std::complex<double>;

// MSVC does not define M_PI unless _USE_MATH_DEFINES is set before <cmath>, and
// this is a header others will include in any order. Carry our own constant.
constexpr double kPi = 3.14159265358979323846;

// exp(-2*pi*i*m/N) for m in [0, N), built once per size per thread. Computing the
// twiddles inline cost ~10k cos/sin calls per 400-point frame and made the mel
// ~90 % of an encode. thread_local: Flamingo runs one mel per thread.
// std::map, not a vector: the recursion holds a reference to its own size's table
// while a deeper level inserts the next one, so entries must never move.
inline const std::vector<Cpx> & twiddles(size_t N) {
    thread_local std::map<size_t, std::vector<Cpx>> cache;
    std::vector<Cpx> & t = cache[N];
    if (t.empty()) {
        t.resize(N);
        for (size_t m = 0; m < N; ++m) {
            const double a = -2.0 * kPi * (double) m / (double) N;
            t[m] = Cpx(std::cos(a), std::sin(a));
        }
    }
    return t;
}

// Recursive Cooley-Tukey, falling back to a naive DFT on odd N. n_fft=400
// factors as 2^4 * 25, so the recursion bottoms out in a 25-point DFT.
inline void fft(std::vector<Cpx> & x) {
    const size_t N = x.size();
    if (N <= 1) {
        return;
    }
    const std::vector<Cpx> & tw = twiddles(N);
    if (N % 2 != 0) {
        std::vector<Cpx> out(N);
        for (size_t k = 0; k < N; ++k) {
            Cpx s(0.0, 0.0);
            for (size_t n = 0; n < N; ++n) {
                s += x[n] * tw[(k * n) % N];
            }
            out[k] = s;
        }
        x.swap(out);
        return;
    }
    std::vector<Cpx> even(N / 2), odd(N / 2);
    for (size_t i = 0; i < N / 2; ++i) {
        even[i] = x[2 * i];
        odd[i]  = x[2 * i + 1];
    }
    fft(even);
    fft(odd);
    for (size_t k = 0; k < N / 2; ++k) {
        const Cpx t = tw[k] * odd[k];
        x[k]         = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}

inline double hz_to_mel_slaney(double f) {
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.0;
    if (f >= min_log_hz) {
        return min_log_mel + std::log(std::max(f, 1e-9) / min_log_hz) / logstep;
    }
    return f / f_sp;
}

inline double mel_to_hz_slaney(double m) {
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.0;
    if (m >= min_log_mel) {
        return min_log_hz * std::exp(logstep * (m - min_log_mel));
    }
    return f_sp * m;
}

// Slaney-scale, Slaney-area-normalised triangular filterbank. Row-major
// [n_mels, n_freq].
inline std::vector<double> filterbank(int n_mels, int n_freq, int sample_rate) {
    const double fmax = sample_rate / 2.0;
    const double mel_min = hz_to_mel_slaney(0.0);
    const double mel_max = hz_to_mel_slaney(fmax);

    std::vector<double> hz(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        const double m = mel_min + (mel_max - mel_min) * (double) i / (double) (n_mels + 1);
        hz[i] = mel_to_hz_slaney(m);
    }

    std::vector<double> fb((size_t) n_mels * n_freq, 0.0);
    for (int i = 0; i < n_mels; ++i) {
        const double lo = hz[i], ce = hz[i + 1], hi = hz[i + 2];
        // Slaney area normalisation: constant area per filter, not constant peak.
        const double enorm = 2.0 / (hi - lo);
        for (int b = 0; b < n_freq; ++b) {
            const double f = fmax * (double) b / (double) (n_freq - 1);
            double w = 0.0;
            if (f > lo && f < hi) {
                w = (f <= ce) ? (f - lo) / (ce - lo) : (hi - f) / (hi - ce);
                if (w < 0.0) {
                    w = 0.0;
                }
            }
            fb[(size_t) i * n_freq + b] = w * enorm;
        }
    }
    return fb;
}

}  // namespace mel_detail

// Linear mel power spectrogram, row-major [n_mels, frames]. Split out from the
// log/clamp/rescale so a chunked caller can accumulate the whole utterance
// before scaling (see note 3 in the header comment).
inline std::vector<float> mel_power(const float * pcm, size_t n_samples,
                                    const MelParams & p, int * out_frames) {
    using namespace mel_detail;

    const int n_fft = p.n_fft;
    const int hop   = p.hop_length;
    const int pad   = n_fft / 2;
    const int n_freq = n_fft / 2 + 1;

    // Reflect-pad, matching torch.stft(center=True, pad_mode="reflect").
    std::vector<double> a((size_t) n_samples + 2 * (size_t) pad);
    for (int i = 0; i < pad; ++i) {
        a[i] = (n_samples > (size_t) (pad - i)) ? (double) pcm[pad - i] : 0.0;
    }
    for (size_t i = 0; i < n_samples; ++i) {
        a[(size_t) pad + i] = (double) pcm[i];
    }
    for (int i = 0; i < pad; ++i) {
        const long src = (long) n_samples - 2 - i;
        a[(size_t) pad + n_samples + i] = (src >= 0) ? (double) pcm[src] : 0.0;
    }

    const long total = (long) a.size();
    const int n_all = (total >= n_fft) ? (int) (1 + (total - n_fft) / hop) : 0;
    // Whisper drops the final frame.
    const int frames = std::max(0, n_all - 1);
    if (out_frames) {
        *out_frames = frames;
    }
    if (frames == 0) {
        return {};
    }

    // Periodic Hann (np.hanning(N+1)[:-1]), NOT the symmetric variant.
    std::vector<double> win(n_fft);
    for (int i = 0; i < n_fft; ++i) {
        win[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * (double) i / (double) n_fft);
    }

    const std::vector<double> fb = filterbank(p.n_mels, n_freq, p.sample_rate);

    std::vector<float> mel((size_t) p.n_mels * frames, 0.0f);
    std::vector<Cpx> buf(n_fft);
    std::vector<double> power(n_freq);

    for (int t = 0; t < frames; ++t) {
        const size_t off = (size_t) t * hop;
        for (int i = 0; i < n_fft; ++i) {
            buf[i] = Cpx(a[off + i] * win[i], 0.0);
        }
        fft(buf);
        for (int b = 0; b < n_freq; ++b) {
            const double re = buf[b].real(), im = buf[b].imag();
            power[b] = re * re + im * im;
        }
        for (int m = 0; m < p.n_mels; ++m) {
            const double * row = &fb[(size_t) m * n_freq];
            double s = 0.0;
            for (int b = 0; b < n_freq; ++b) {
                s += row[b] * power[b];
            }
            mel[(size_t) m * frames + t] = (float) s;
        }
    }
    return mel;
}

// log10 -> global-max clamp -> rescale, in place. Call ONCE over the whole
// utterance; see note 3.
inline void mel_log_scale(std::vector<float> & mel) {
    float mx = -1e30f;
    for (float & v : mel) {
        v = std::log10(std::max(v, 1e-10f));
        mx = std::max(mx, v);
    }
    const float floor_v = mx - 8.0f;
    for (float & v : mel) {
        v = (std::max(v, floor_v) + 4.0f) / 4.0f;
    }
}

// Convenience: full frontend for one utterance held in memory.
inline std::vector<float> log_mel(const float * pcm, size_t n_samples,
                                  const MelParams & p, int * out_frames) {
    std::vector<float> mel = mel_power(pcm, n_samples, p, out_frames);
    mel_log_scale(mel);
    return mel;
}

}  // namespace moss
