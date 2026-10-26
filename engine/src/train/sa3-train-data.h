// sa3-train-data.h: data side of ace-train sa3-train.
//
// Pre-encode: every manifest sample is read once (44.1 kHz stereo WAV), run
// through the frozen SAME encoder and T5Gemma, and cached; the encoders are
// then released before the DiT loads. This is upstream's pre-encoded mode
// (scripts/pre_encode_dataset.py + train_lora.py --encoded_dir): the whole
// clip is encoded once and each step crops latents. The encode itself follows
// autoencoders.py encode_audio: pad to a multiple of 4096 samples, one
// unchunked pass below 128 latents, else 128-latent chunks with 32 overlap.
// The prompt embedding gets the TRAINING base's learned padding embedding
// (it differs from the inference model's), taken from the base DiT GGUF.
//
// Per-step randomness (sample order, crop, silence extension, inpaint mask,
// t, noise, CFG dropout) is a pure function of (seed, step), so an
// interrupted run resumes onto the identical draw sequence with no RNG state
// to save. The draws follow training/diffusion.py training_step at upstream
// 77e0398c, with train_lora.py's settings: trunc_logit_normal (flipped) +
// DistributionShift(256, 4096) on the effective length from seconds_total,
// silence extension Exp(4 s), inpaint mask types [0.1 segments, 0.8 full,
// 0.1 causal] with mask_padding, cfg_dropout_prob 0.1, mask_loss_weight 1.
// Upstream's PhaseFlipper augmentation acts on raw audio and has no latent
// equivalent; like upstream's own pre-encoded mode, it is not applied.

#pragma once

#include "audio-io.h"
#include "hot-step-fsutf8.h"
#include "sa3-same-ggml.h"
#include "sa3-t5gemma-enc.h"
#include "train/yue2-aitk-sha256.h"
#include "yyjson.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static const int     SA3T_SR        = 44100;
static const int     SA3T_DS        = 4096;   // audio samples per latent
static const int     SA3T_CHUNK     = 128;    // SAME encode chunk, latents
static const int     SA3T_OVERLAP   = 32;
static const int     SA3T_LAT       = 256;    // latent channels
static const int     SA3T_TOK       = 256;    // T5Gemma max_length
static const int     SA3T_TEXT_DIM  = 768;

// ── deterministic RNG (splitmix64, cross-platform) ──────────────────────────
struct Sa3Rng {
    uint64_t s;
    Sa3Rng(uint64_t seed, uint64_t step, uint64_t stream) : s(seed ^ (step * 0x9E3779B97F4A7C15ull) ^ (stream << 56)) {
        next();
        next();
    }
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uni() { return (double) (next() >> 11) * (1.0 / 9007199254740992.0); }  // [0, 1)
    int64_t randint(int64_t lo, int64_t hi) { return lo + (int64_t) (uni() * (double) (hi - lo + 1)); }  // inclusive
    double normal() {
        double u1 = uni();
        if (u1 < 1e-300) u1 = 1e-300;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * uni());
    }
};

// Inverse standard normal CDF (Wichura AS241, ~1e-16).
static double sa3t_ppnd(double p) {
    const double q = p - 0.5;
    if (std::fabs(q) <= 0.425) {
        const double r = 0.180625 - q * q;
        return q * (((((((2509.0809287301226727 * r + 33430.575583588128105) * r + 67265.770927008700853) * r +
                         45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r +
                      133.14166789178437745) * r + 3.387132872796366608) /
               (((((((5226.495278852545925 * r + 28729.085735721942674) * r + 39307.89580009271061) * r +
                    21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r +
                 42.313330701600911252) * r + 1.0);
    }
    double r = q < 0 ? p : 1.0 - p;
    r = std::sqrt(-std::log(r));
    double v;
    if (r <= 5.0) {
        r -= 1.6;
        v = (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r + 0.24178072517745061177) * r +
                 1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r +
              4.6303378461565452959) * r + 1.42343711074968357734) /
            (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r +
                 0.14810397642748007459) * r + 0.68976733498510000455) * r + 1.6763848301838038494) * r +
              2.05319162663775882187) * r + 1.0);
    } else {
        r -= 5.0;
        v = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r +
                 0.026532189526576123093) * r + 0.29656057182850489123) * r + 1.7848265399172913358) * r +
              5.4637849111641143699) * r + 6.6579046435011037772) /
            (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r +
                 7.868691311456132591e-4) * r + 0.0148753612908506148525) * r + 0.13692988092273580531) * r +
              0.59983220655588793769) * r + 1.0);
    }
    return q < 0 ? -v : v;
}

// trunc_logit_normal, flipped (sampling.py truncated_logistic_normal_rescaled,
// left_trunc 0.075, right_trunc 1), then DistributionShift(base 0.5, max 1.15,
// lengths 256..4096) on the effective length.
static double sa3t_sample_t(Sa3Rng & r, int64_t effective_len) {
    const double phi = [](double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }(std::log(0.075 / 0.925));
    const double u   = phi + (1.0 - phi) * r.uni();
    const double s   = 1.0 / (1.0 + std::exp(-sa3t_ppnd(u)));
    double       t   = 1.0 - (s - 0.075) / (1.0 - 0.075);
    const double L   = (double) std::min<int64_t>(std::max<int64_t>(effective_len, 256), 4096);
    const double mu  = -(0.5 + (1.15 - 0.5) * (L - 256.0) / (4096.0 - 256.0));
    if (t >= 1.0) return 1.0;
    t = 1.0 - std::exp(mu) / (std::exp(mu) + (1.0 / (1.0 - t) - 1.0));
    return t;
}

// ExpoFourierFeatures(dim, 0.5, 10000) (blocks.py), as sa3_dit_forward: [cos | sin].
static void sa3t_expo(float x, int dim, float * out) {
    const int    half = dim / 2;
    const double lmin = std::log(0.5), lmax = std::log(10000.0);
    for (int i = 0; i < half; i++) {
        const double ramp = half > 1 ? (double) i / (double) (half - 1) : 0.0;
        const float  freq = (float) std::exp(ramp * (lmax - lmin) + lmin);
        float        arg  = x * freq;
        arg               = arg * 2.0f;
        arg               = arg * 3.14159265358979323846f;
        out[i]            = std::cos(arg);
        out[half + i]     = std::sin(arg);
    }
}

// ── cached sample ──────────────────────────────────────────────────────────
struct Sa3Sample {
    std::string        audio_path, prompt;
    int64_t            n_samples = 0;      // source length, 44.1 kHz
    int64_t            valid = 0;          // ceil(n_samples / 4096) latents
    int                seconds_total = 0;  // ceil(n_samples / 44100), upstream PadCrop_Normalized_T
    std::vector<float> latents;            // [256][valid] channel-major (torch (C, L))
    std::vector<float> text;               // [256][768] token-major, padding rows = base padding embedding
};

// One step's DiT inputs, torch layouts.
struct Sa3StepInputs {
    int                sample = 0;
    int64_t            crop_start = 0;
    double             t = 0.0;
    bool               cfg_drop = false;
    int                mask_type = 1;   // 0 segments, 1 full, 2 causal
    std::vector<float> x0, noise, x_t, target;  // [256][C]
    std::vector<float> local;                    // [257][C]: inpaint mask, masked input
    std::vector<float> aug_pad, loss_mask, ctx_mask;  // [C] 0/1
};

// Upstream inpaint mask (models/inpainting.py random_inpaint_mask, mask_padding=True).
static int sa3t_inpaint_mask(Sa3Rng & r, int64_t real, int64_t C, std::vector<float> & mask) {
    mask.assign((size_t) C, 1.0f);
    const double u    = r.uni();
    const int    type = u < 0.1 ? 0 : (u < 0.9 ? 1 : 2);
    if (type == 1) {
        mask.assign((size_t) C, 0.0f);
        return type;
    }
    if (real == 0) return type;
    if (type == 0) {
        const int64_t n   = r.randint(1, 10);
        const int64_t mx  = std::max<int64_t>(1, real / n);
        for (int64_t k = 0; k < n; k++) {
            const int64_t len = r.randint(1, mx);
            if (real - len < 0) continue;
            const int64_t st = r.randint(0, real - len);
            for (int64_t i = st; i < st + len; i++) mask[(size_t) i] = 0.0f;
        }
    } else {
        const int64_t pre = r.randint(0, real);
        for (int64_t i = pre; i < real; i++) mask[(size_t) i] = 0.0f;
    }
    for (int64_t i = real; i < C; i++) mask[(size_t) i] = 0.0f;
    return type;
}

// Epoch-shuffled sample order: index for `step`.
static int sa3t_sample_for_step(uint64_t seed, int64_t step, int n) {
    const int64_t epoch = step / n;
    std::vector<int> perm((size_t) n);
    for (int i = 0; i < n; i++) perm[(size_t) i] = i;
    Sa3Rng r(seed, (uint64_t) epoch, 1);
    for (int i = n - 1; i > 0; i--) std::swap(perm[(size_t) i], perm[(size_t) r.randint(0, i)]);
    return perm[(size_t) (step % n)];
}

// Draw step `step`'s inputs. `silence` is the SAME encoding of zeros, >= C latents.
static void sa3t_make_step(const std::vector<Sa3Sample> & data, const std::vector<float> & silence, int64_t sil_len,
                           int64_t C, uint64_t seed, int64_t step, Sa3StepInputs & s) {
    Sa3Rng r(seed, (uint64_t) step, 2);
    s.sample = sa3t_sample_for_step(seed, step, (int) data.size());
    const Sa3Sample & d = data[(size_t) s.sample];
    // crop (data/dataset.py PreEncodedDataset): random start below the last valid latent
    const int64_t last = d.valid - 1;
    s.crop_start = (d.valid > C && last > C) ? r.randint(0, last - C) : 0;
    s.x0.assign((size_t) (SA3T_LAT * C), 0.0f);
    std::vector<float> pad((size_t) C, 0.0f);
    for (int64_t j = 0; j < C; j++) {
        const int64_t src = s.crop_start + j;
        const bool    in  = src < d.valid;
        pad[(size_t) j]   = in ? 1.0f : 0.0f;
        for (int c = 0; c < SA3T_LAT; c++)
            s.x0[(size_t) (c * C + j)] = in ? d.latents[(size_t) (c * d.valid + src)]
                                            : silence[(size_t) (c * sil_len + (src - d.valid))];
    }
    // silence extension (training/utils.py create_augmented_padding_mask)
    int64_t valid = 0;
    for (float p : pad) valid += p > 0.5f;
    const double  scale_tok = 4.0 * SA3T_SR / SA3T_DS;
    const int64_t ext       = (int64_t) (-std::log(1.0 - r.uni()) * scale_tok);
    const int64_t aug_len   = std::min<int64_t>(valid + ext, C);
    s.aug_pad.assign((size_t) C, 0.0f);
    for (int64_t j = 0; j < aug_len; j++) s.aug_pad[(size_t) j] = 1.0f;
    // timestep on the effective length from seconds_total
    const int64_t eff = (int64_t) std::ceil((double) ((int64_t) d.seconds_total * SA3T_SR) / SA3T_DS);
    s.t = sa3t_sample_t(r, eff);
    // inpaint mask over the augmented valid region
    std::vector<float> im;
    s.mask_type = sa3t_inpaint_mask(r, aug_len, C, im);
    s.local.assign((size_t) ((SA3T_LAT + 1) * C), 0.0f);
    for (int64_t j = 0; j < C; j++) {
        s.local[(size_t) j] = im[(size_t) j];
        for (int c = 0; c < SA3T_LAT; c++) s.local[(size_t) ((c + 1) * C + j)] = s.x0[(size_t) (c * C + j)] * im[(size_t) j];
    }
    s.loss_mask.assign((size_t) C, 0.0f);
    s.ctx_mask.assign((size_t) C, 0.0f);
    for (int64_t j = 0; j < C; j++) {
        s.loss_mask[(size_t) j] = (s.aug_pad[(size_t) j] > 0.5f && im[(size_t) j] < 0.5f) ? 1.0f : 0.0f;
        s.ctx_mask[(size_t) j]  = (s.aug_pad[(size_t) j] > 0.5f && im[(size_t) j] > 0.5f) ? 1.0f : 0.0f;
    }
    // noise, rectified-flow interpolation and target
    const float t = (float) s.t;
    s.noise.resize(s.x0.size());
    s.x_t.resize(s.x0.size());
    s.target.resize(s.x0.size());
    for (size_t i = 0; i < s.x0.size(); i++) {
        s.noise[i]  = (float) r.normal();
        s.x_t[i]    = s.x0[i] * (1.0f - t) + s.noise[i] * t;
        s.target[i] = s.noise[i] - s.x0[i];
    }
    s.cfg_drop = r.uni() < 0.1;
}

// ── encoders ───────────────────────────────────────────────────────────────

// autoencoders.py encode_audio. audio planar [2][n], n a multiple of 4096.
static void sa3t_same_encode(SA3Same * enc, const float * audio, int64_t n, std::vector<float> & lat) {
    const int64_t L = n / SA3T_DS;
    lat.assign((size_t) (SA3T_LAT * L), 0.0f);
    if (L < SA3T_CHUNK) {
        sa3_same_forward(enc, audio, lat.data(), (int) L);
        return;
    }
    const int64_t cs = (int64_t) SA3T_CHUNK * SA3T_DS, hop = (int64_t) (SA3T_CHUNK - SA3T_OVERLAP) * SA3T_DS;
    std::vector<int64_t> starts;
    for (int64_t s = 0; s + cs <= n; s += hop) starts.push_back(s);
    if (starts.back() != n - cs) starts.push_back(n - cs);
    std::vector<float> chunk((size_t) (2 * cs)), cl((size_t) (SA3T_LAT * SA3T_CHUNK));
    const int half = SA3T_OVERLAP / 2;
    for (size_t i = 0; i < starts.size(); i++) {
        std::copy(audio + starts[i], audio + starts[i] + cs, chunk.begin());
        std::copy(audio + n + starts[i], audio + n + starts[i] + cs, chunk.begin() + cs);
        sa3_same_forward(enc, chunk.data(), cl.data(), SA3T_CHUNK);
        const bool    first = i == 0, last = i + 1 == starts.size();
        const int64_t os    = last ? L - SA3T_CHUNK : starts[i] / SA3T_DS;
        const int     left = first ? 0 : half, right = last ? SA3T_CHUNK : SA3T_CHUNK - half;
        for (int c = 0; c < SA3T_LAT; c++)
            std::copy(cl.begin() + c * SA3T_CHUNK + left, cl.begin() + c * SA3T_CHUNK + right,
                      lat.begin() + c * L + os + left);
    }
}

// Base model's learned padding embedding (768) from the base DiT GGUF.
static bool sa3t_read_padding(const std::string & base_gguf, std::vector<float> & pad, std::string * err) {
    GGUFModel gf = {};
    if (!gf_load(&gf, base_gguf.c_str())) {
        *err = "cannot open " + base_gguf;
        return false;
    }
    const char *  name = "conditioner.conditioners.prompt.padding_embedding";
    ggml_tensor * t    = ggml_get_tensor(gf.meta, name);
    const void *  d    = gf_get_data(gf, name);
    const bool    ok   = t && d && t->type == GGML_TYPE_F32 && ggml_nelements(t) == SA3T_TEXT_DIM;
    if (ok) pad.assign((const float *) d, (const float *) d + SA3T_TEXT_DIM);
    gf_close(&gf);
    if (!ok) *err = "base GGUF has no f32 prompt padding_embedding";
    return ok;
}

// T5Gemma on 256 padded ids with n valid, then the base padding embedding on padded rows.
static void sa3t_text_encode(SA3T5GemmaEnc * t5, const int32_t * ids, int n, const std::vector<float> & pad,
                             std::vector<float> & out) {
    uint8_t mask[SA3T_TOK];
    for (int i = 0; i < SA3T_TOK; i++) mask[i] = i < n ? 1 : 0;
    out.resize((size_t) SA3T_TOK * SA3T_TEXT_DIM);
    sa3_t5gemma_forward(t5, ids, mask, SA3T_TOK, out.data());
    for (int s = n; s < SA3T_TOK; s++) std::copy(pad.begin(), pad.end(), out.begin() + (size_t) s * SA3T_TEXT_DIM);
}

// ── manifest + cache ───────────────────────────────────────────────────────
//
// Manifest (written by the server runner): {"samples": [{"audio_path": "...wav",
// "token_ids": [256 ints, padded], "n_tokens": N, "prompt": "..."}]}. Relative
// audio paths resolve against the manifest's folder. Token ids come from the
// server's T5Gemma tokenizer (sa3Tokenizer.ts tokenizeForSa3), the same
// contract as POST /sa3-refine.
struct Sa3ManifestItem {
    std::string          audio_path, prompt;
    std::vector<int32_t> ids;
    int                  n_tokens = 0;
};

static bool sa3t_read_manifest(const std::string & path, std::vector<Sa3ManifestItem> & items, std::string * err) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        *err = "cannot open manifest " + path;
        return false;
    }
    std::string txt;
    char        buf[65536];
    size_t      n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) txt.append(buf, n);
    fclose(f);
    yyjson_doc * doc = yyjson_read(txt.c_str(), txt.size(), 0);
    yyjson_val * arr = doc ? yyjson_obj_get(yyjson_doc_get_root(doc), "samples") : nullptr;
    if (!arr || !yyjson_is_arr(arr)) {
        if (doc) yyjson_doc_free(doc);
        *err = "manifest has no samples array: " + path;
        return false;
    }
    std::string dir = path;
    const size_t slash = dir.find_last_of("/\\");
    dir = slash == std::string::npos ? std::string(".") : dir.substr(0, slash);
    size_t i, mx;
    yyjson_val * v;
    bool ok = true;
    yyjson_arr_foreach(arr, i, mx, v) {
        Sa3ManifestItem it;
        yyjson_val * ap = yyjson_obj_get(v, "audio_path");
        yyjson_val * ti = yyjson_obj_get(v, "token_ids");
        yyjson_val * nt = yyjson_obj_get(v, "n_tokens");
        yyjson_val * pr = yyjson_obj_get(v, "prompt");
        if (!ap || !yyjson_is_str(ap) || !ti || !yyjson_is_arr(ti) || yyjson_arr_size(ti) != SA3T_TOK || !nt ||
            !yyjson_is_int(nt)) {
            *err = "manifest sample " + std::to_string(i) + " needs audio_path, token_ids[256] and n_tokens";
            ok = false;
            break;
        }
        it.audio_path = yyjson_get_str(ap);
        const bool absolute = it.audio_path.size() > 1 && (it.audio_path[0] == '/' || it.audio_path[0] == '\\' || it.audio_path[1] == ':');
        if (!absolute) it.audio_path = dir + "/" + it.audio_path;
        size_t j, jm;
        yyjson_val * x;
        yyjson_arr_foreach(ti, j, jm, x) it.ids.push_back((int32_t) yyjson_get_int(x));
        it.n_tokens = (int) yyjson_get_int(nt);
        if (it.n_tokens < 1 || it.n_tokens > SA3T_TOK) {
            *err = "manifest sample " + std::to_string(i) + ": n_tokens out of range";
            ok = false;
            break;
        }
        if (pr && yyjson_is_str(pr)) it.prompt = yyjson_get_str(pr);
        items.push_back(std::move(it));
    }
    yyjson_doc_free(doc);
    if (ok && items.empty()) {
        *err = "manifest has no samples";
        ok = false;
    }
    return ok;
}

static std::string sa3t_hex_key(const void * a, size_t na, const std::string & extra) {
    yue2_aitk::sha256::hasher h;
    h.update(a, na);
    h.update(extra.data(), extra.size());
    return h.final().hex().substr(0, 32);
}

static bool sa3t_write_file(const std::string & path, const std::vector<std::pair<const void *, size_t>> & parts) {
    const std::string tmp = path + ".tmp";
    FILE * f = hs_fopen(tmp, "wb");
    if (!f) return false;
    bool ok = true;
    for (const auto & p : parts) ok = ok && fwrite(p.first, 1, p.second, f) == p.second;
    ok = (fclose(f) == 0) && ok;
    if (!ok) {
        std::remove(tmp.c_str());
        return false;
    }
    std::remove(path.c_str());
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

static bool sa3t_read_file(const std::string & path, std::string & data) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) return false;
    char   buf[1 << 16];
    size_t n;
    data.clear();
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) data.append(buf, n);
    fclose(f);
    return true;
}
