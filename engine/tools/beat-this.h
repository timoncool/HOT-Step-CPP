#pragma once
// beat-this.h: Beat This! beat + downbeat tracker (GGML port)
//
// Foscarin, Schlüter & Widmer (CPJKU), "Beat This! Accurate Beat Tracking
// Without DBN Postprocessing", ISMIR 2024 (arXiv:2407.21658). Code and
// weights MIT. Ports beat_this/{preprocessing,inference}.py,
// model/{beat_tracker,roformer}.py and the "minimal" postprocessor (no DBN).
// The ggml layout notes follow CrispASR's port (MIT), which validated them
// stage by stage against PyTorch.
//
// Weights: the `final0` checkpoint as safetensors (torch state_dict keys,
// BatchNorm unfolded; folded here at load). MuScriptor's tempo detection
// (utils/beats.py detect_grid) runs this same model.
//
//   BeatThis bt;
//   bt_load(&bt, "<dir>/model.safetensors");
//   BeatTrack tr = bt_track(&bt, mono, n, sr);   // any rate, mono
//   bt_free(&bt);

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "audio-resample.h"
#include "backend.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "safetensors.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define BT_SAMPLE_RATE 22050
#define BT_N_FFT       1024
#define BT_HOP         441     // 50 frames per second
#define BT_FPS         50
#define BT_N_MELS      128
#define BT_CHUNK       1500    // frames per model call (30 s), inference.py
#define BT_BORDER      6       // frames discarded at each chunk edge
#define BT_DIM         512
#define BT_HEAD_DIM    32
#define BT_LAYERS      6

struct BtAttn {             // roformer.Attention: qkv/out bias-free, gates biased
    ggml_tensor * gamma, * qkv, * gates_w, * gates_b, * out;
    int           heads;
};

struct BtFF {               // roformer.FeedForward: RMSNorm -> Linear -> GELU -> Linear
    ggml_tensor * gamma, * w1, * b1, * w2, * b2;
};

struct BtBlock {            // PartialFTTransformer, then conv(2x3, stride f2) + BN + GELU
    BtAttn        attn_f, attn_t;
    BtFF          ff_f, ff_t;
    ggml_tensor * conv_w, * conv_b;  // BN folded in
};

struct BeatThis {
    ggml_context *        wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_backend_t        backend = nullptr, cpu_backend = nullptr;
    ggml_backend_sched_t  sched = nullptr;

    std::vector<float> bn1d_scale, bn1d_shift;  // stem BatchNorm1d over mel bins, applied on the host
    std::vector<float> mel_fb;                  // [513, 128] row-major
    std::vector<float> window;                  // periodic Hann, 1024

    ggml_tensor * stem_w, * stem_b;             // conv 4x3 stride f4, BN folded in
    BtBlock       blocks[3];
    ggml_tensor * linear_w, * linear_b;
    BtAttn        attn[BT_LAYERS];
    BtFF          ff[BT_LAYERS];
    ggml_tensor * out_norm, * head_w, * head_b;
};

// Beats and downbeats in seconds. Every downbeat is also a beat.
struct BeatTrack {
    std::vector<double> beats, downbeats;
};

// ---------------------------------------------------------------------------
// Log-mel front end (preprocessing.py LogMelSpect, torchaudio MelSpectrogram):
// 22050 Hz, n_fft 1024, hop 441, center + reflect pad, periodic Hann,
// MAGNITUDE (power 1) scaled by 1/sqrt(n_fft) ("frame_length" normalisation),
// slaney mel 30..11000 Hz with no filter normalisation, log1p(1000 x).
// ---------------------------------------------------------------------------

static void bt_fft(float * re, float * im, int n) {
    static std::vector<float> tw_re, tw_im;  // exact per-stage twiddles, as ace-midi's fft_radix2
    static int tw_n = 0;
    if (tw_n != n) {
        tw_re.assign((size_t) n, 0.0f);
        tw_im.assign((size_t) n, 0.0f);
        for (int len = 2, base = 0; len <= n; len <<= 1, base += len >> 2) {
            for (int j = 0; j < len / 2; j++) {
                double ang = -2.0 * M_PI * j / len;
                tw_re[(size_t) base + j] = (float) cos(ang);
                tw_im[(size_t) base + j] = (float) sin(ang);
            }
        }
        tw_n = n;
    }
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j |= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2, base = 0; len <= n; len <<= 1, base += len >> 2) {
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len / 2; j++) {
                int   a = i + j, b = i + j + len / 2;
                float cr = tw_re[(size_t) base + j], ci = tw_im[(size_t) base + j];
                float xr = re[b] * cr - im[b] * ci;
                float xi = re[b] * ci + im[b] * cr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr;        im[a] += xi;
            }
        }
    }
}

// torchaudio melscale_fbanks(513, 30, 11000, 128, 22050, norm=None, "slaney"),
// [n_freqs, n_mels] row-major. Matches the exported model's matrix to ~1e-5.
static std::vector<float> bt_mel_filterbank() {
    auto hz_to_mel = [](double f) {
        const double f_sp = 200.0 / 3.0, min_log_mel = 1000.0 / f_sp, logstep = log(6.4) / 27.0;
        return f >= 1000.0 ? min_log_mel + log(f / 1000.0) / logstep : f / f_sp;
    };
    auto mel_to_hz = [](double m) {
        const double f_sp = 200.0 / 3.0, min_log_mel = 1000.0 / f_sp, logstep = log(6.4) / 27.0;
        return m >= min_log_mel ? 1000.0 * exp(logstep * (m - min_log_mel)) : f_sp * m;
    };
    const int n_freqs = BT_N_FFT / 2 + 1;
    std::vector<double> f_pts(BT_N_MELS + 2);
    const double m_min = hz_to_mel(30.0), m_max = hz_to_mel(11000.0);
    for (int i = 0; i < BT_N_MELS + 2; i++) f_pts[i] = mel_to_hz(m_min + (m_max - m_min) * i / (BT_N_MELS + 1));
    std::vector<float> fb((size_t) n_freqs * BT_N_MELS);
    for (int k = 0; k < n_freqs; k++) {
        const double f = (double) (BT_SAMPLE_RATE / 2) * k / (n_freqs - 1);
        for (int m = 0; m < BT_N_MELS; m++) {
            double down = (f - f_pts[m]) / (f_pts[m + 1] - f_pts[m]);
            double up   = (f_pts[m + 2] - f) / (f_pts[m + 2] - f_pts[m + 1]);
            fb[(size_t) k * BT_N_MELS + m] = (float) std::max(0.0, std::min(down, up));
        }
    }
    return fb;
}

// Mono 22050 Hz in, (T, 128) row-major log-mel out, T = 1 + n / 441.
static std::vector<float> bt_logmel(const BeatThis * bt, const float * pcm, int n) {
    const int T = 1 + n / BT_HOP, half = BT_N_FFT / 2, n_freqs = BT_N_FFT / 2 + 1;
    auto reflect = [n](int i) {
        while (i < 0 || i >= n) {
            if (i < 0) i = -i;
            if (i >= n) i = 2 * (n - 1) - i;
        }
        return i;
    };
    const float norm = 1.0f / sqrtf((float) BT_N_FFT);
    std::vector<float>  out((size_t) T * BT_N_MELS);
    std::vector<float>  re(BT_N_FFT), im(BT_N_FFT);
    std::vector<double> acc(BT_N_MELS);
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < BT_N_FFT; i++) {
            re[i] = pcm[reflect(t * BT_HOP - half + i)] * bt->window[i];
            im[i] = 0.0f;
        }
        bt_fft(re.data(), im.data(), BT_N_FFT);
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int k = 0; k < n_freqs; k++) {
            double mag = sqrt((double) re[k] * re[k] + (double) im[k] * im[k]) * norm;
            if (mag == 0.0) continue;
            const float * row = bt->mel_fb.data() + (size_t) k * BT_N_MELS;
            for (int m = 0; m < BT_N_MELS; m++) acc[m] += mag * row[m];
        }
        for (int m = 0; m < BT_N_MELS; m++) out[(size_t) t * BT_N_MELS + m] = log1pf(1000.0f * (float) acc[m]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

static std::vector<float> bt_read(const STFile & st, const std::string & name) {
    const STEntry * e = st_find(st, name.c_str());
    if (!e || e->dtype != "F32") {
        fprintf(stderr, "[beat-this] FATAL: missing or non-F32 tensor %s\n", name.c_str());
        exit(1);
    }
    int64_t n = 1;
    for (int i = 0; i < e->n_dims; i++) n *= e->shape[i];
    std::vector<float> v((size_t) n);
    memcpy(v.data(), st_data(st, *e), (size_t) n * 4);
    return v;
}

// eval-mode BatchNorm as y = x * scale + shift (eps 1e-5, torch default)
static void bt_bn(const STFile & st, const std::string & p, std::vector<float> & scale, std::vector<float> & shift) {
    std::vector<float> w = bt_read(st, p + ".weight"), b = bt_read(st, p + ".bias");
    std::vector<float> mean = bt_read(st, p + ".running_mean"), var = bt_read(st, p + ".running_var");
    scale.resize(w.size());
    shift.resize(w.size());
    for (size_t i = 0; i < w.size(); i++) {
        scale[i] = w[i] / sqrtf(var[i] + 1e-5f);
        shift[i] = b[i] - mean[i] * scale[i];
    }
}

static bool bt_load(BeatThis * bt, const std::string & path) {
    STFile st;
    if (!st_open(&st, path.c_str())) return false;

    BackendPair bp  = backend_init("BeatThis");
    bt->backend     = bp.backend;
    bt->cpu_backend = bp.cpu_backend;
    bt->sched       = backend_sched_new(bp, 4096);

    ggml_init_params ip = { 256 * ggml_tensor_overhead(), NULL, true };
    bt->wctx = ggml_init(ip);

    // Tensors are created now and filled once the buffer exists.
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> pending;
    // torch [out, in] row-major == ggml ne (in, out), same memory
    auto mk = [&](std::vector<float> data, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1) {
        ggml_tensor * t = ggml_new_tensor_4d(bt->wctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
        if ((int64_t) data.size() != ggml_nelements(t)) {
            fprintf(stderr, "[beat-this] FATAL: tensor size %zu != %lld\n", data.size(), (long long) ggml_nelements(t));
            exit(1);
        }
        pending.push_back({ t, std::move(data) });
        return t;
    };
    auto linear = [&](const std::string & name, int64_t in, int64_t out) { return mk(bt_read(st, name), in, out); };
    auto vec    = [&](const std::string & name, int64_t n) { return mk(bt_read(st, name), n); };
    auto attn   = [&](const std::string & p, int dim) {
        BtAttn a;
        a.heads   = dim / BT_HEAD_DIM;
        a.gamma   = vec(p + ".norm.gamma", dim);
        a.qkv     = linear(p + ".to_qkv.weight", dim, 3 * dim);
        a.gates_w = linear(p + ".to_gates.weight", dim, a.heads);
        a.gates_b = vec(p + ".to_gates.bias", a.heads);
        a.out     = linear(p + ".to_out.0.weight", dim, dim);
        return a;
    };
    auto ff = [&](const std::string & p, int dim) {
        BtFF f;
        f.gamma = vec(p + ".net.0.gamma", dim);
        f.w1    = linear(p + ".net.1.weight", dim, 4 * dim);
        f.b1    = vec(p + ".net.1.bias", 4 * dim);
        f.w2    = linear(p + ".net.4.weight", 4 * dim, dim);
        f.b2    = vec(p + ".net.4.bias", dim);
        return f;
    };
    // conv weight torch (oc, ic, kf, kt) == ggml ne (kt, kf, ic, oc); the BN
    // after it (bias-free conv) folds into a per-channel scale and a bias
    auto conv_bn = [&](const std::string & conv, const std::string & bn, int ic, int oc, int kf,
                       ggml_tensor ** w_out, ggml_tensor ** b_out) {
        std::vector<float> w = bt_read(st, conv), scale, shift;
        bt_bn(st, bn, scale, shift);
        const size_t per = w.size() / oc;
        for (int c = 0; c < oc; c++) {
            for (size_t i = 0; i < per; i++) w[c * per + i] *= scale[c];
        }
        *w_out = mk(std::move(w), 3, kf, ic, oc);
        *b_out = mk(std::move(shift), oc);
    };

    bt_bn(st, "frontend.stem.bn1d", bt->bn1d_scale, bt->bn1d_shift);
    conv_bn("frontend.stem.conv2d.weight", "frontend.stem.bn2d", 1, 32, 4, &bt->stem_w, &bt->stem_b);
    for (int i = 0; i < 3; i++) {
        const int         dim = 32 << i;
        const std::string p   = "frontend.blocks." + std::to_string(i);
        BtBlock &         b   = bt->blocks[i];
        b.attn_f = attn(p + ".partial.attnF", dim);
        b.ff_f   = ff(p + ".partial.ffF", dim);
        b.attn_t = attn(p + ".partial.attnT", dim);
        b.ff_t   = ff(p + ".partial.ffT", dim);
        conv_bn(p + ".conv2d.weight", p + ".norm", dim, 2 * dim, 2, &b.conv_w, &b.conv_b);
    }
    bt->linear_w = linear("frontend.linear.weight", 1024, BT_DIM);
    bt->linear_b = vec("frontend.linear.bias", BT_DIM);
    for (int l = 0; l < BT_LAYERS; l++) {
        const std::string p = "transformer_blocks.layers." + std::to_string(l);
        bt->attn[l] = attn(p + ".0", BT_DIM);
        bt->ff[l]   = ff(p + ".1", BT_DIM);
    }
    bt->out_norm = vec("transformer_blocks.norm.gamma", BT_DIM);
    bt->head_w   = linear("task_heads.beat_downbeat_lin.weight", BT_DIM, 2);
    bt->head_b   = vec("task_heads.beat_downbeat_lin.bias", 2);
    st_close(&st);

    bt->wbuf = ggml_backend_alloc_ctx_tensors(bt->wctx, bt->backend);
    if (!bt->wbuf) {
        fprintf(stderr, "[beat-this] FATAL: weight buffer alloc failed\n");
        return false;
    }
    for (auto & p : pending) ggml_backend_tensor_set(p.first, p.second.data(), 0, p.second.size() * 4);

    bt->mel_fb = bt_mel_filterbank();
    bt->window.resize(BT_N_FFT);
    for (int i = 0; i < BT_N_FFT; i++) bt->window[i] = (float) (0.5 - 0.5 * cos(2.0 * M_PI * i / BT_N_FFT));
    return true;
}

static void bt_free(BeatThis * bt) {
    if (bt->sched) { ggml_backend_sched_free(bt->sched);  bt->sched = nullptr; }
    if (bt->wbuf)  { ggml_backend_buffer_free(bt->wbuf);  bt->wbuf  = nullptr; }
    if (bt->wctx)  { ggml_free(bt->wctx);                 bt->wctx  = nullptr; }
    if (bt->backend || bt->cpu_backend) backend_release(bt->backend, bt->cpu_backend);
    bt->backend = bt->cpu_backend = nullptr;
}

// ---------------------------------------------------------------------------
// Graph. Torch works in (b, c, f, t); ggml reverses that, so a block-level
// activation is ne (t, f, c). Inside a partial transformer the sequence axis
// moves to ne[1] and the other spatial axis becomes the batch in ne[2], as
// einops folds it upstream.
// ---------------------------------------------------------------------------

// F.normalize(x) * sqrt(dim) * gamma is RMSNorm * gamma; eps far below noise
static ggml_tensor * bt_norm(ggml_context * c, ggml_tensor * x, ggml_tensor * gamma) {
    return ggml_mul(c, ggml_rms_norm(c, x, 1e-12f), gamma);
}

// Linear on a (C, N, B) activation, flattened so the weight is never broadcast
static ggml_tensor * bt_mm(ggml_context * c, ggml_tensor * w, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(c, w, ggml_reshape_2d(c, x, x->ne[0], x->ne[1] * x->ne[2]));
    return ggml_reshape_3d(c, y, w->ne[1], x->ne[1], x->ne[2]);
}

// Attention branch on x ne (C, N, B), N the sequence; the caller adds the residual.
static ggml_tensor * bt_attention(ggml_context * c, const BtAttn & a, ggml_tensor * x, ggml_tensor * pos) {
    const int64_t N = x->ne[1], B = x->ne[2], H = a.heads, D = BT_HEAD_DIM;
    ggml_tensor * xn  = bt_norm(c, x, a.gamma);   // the gates read the NORMED x too
    ggml_tensor * qkv = bt_mm(c, a.qkv, xn);      // (3*H*D, N, B), packed "(qkv h d)"
    const size_t  es  = ggml_element_size(qkv);
    auto slab = [&](int64_t which) {
        return ggml_view_4d(c, qkv, D, H, N, B, D * es, qkv->nb[1], qkv->nb[2], which * H * D * es);
    };
    // rotary-embedding-torch rotates ADJACENT pairs with theta 10000^(-2i/32):
    // ggml's NORMAL rope, not NEOX
    ggml_tensor * q = ggml_rope(c, ggml_cont(c, slab(0)), pos, (int) D, GGML_ROPE_TYPE_NORMAL);
    ggml_tensor * k = ggml_rope(c, ggml_cont(c, slab(1)), pos, (int) D, GGML_ROPE_TYPE_NORMAL);
    q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));                  // (D, N, H, B)
    k = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
    ggml_tensor * vt = ggml_cont(c, ggml_permute(c, slab(2), 1, 2, 0, 3));  // (N, D, H, B)

    // SDPA, non-causal, default scale 1/sqrt(D)
    ggml_tensor * kq  = ggml_soft_max_ext(c, ggml_mul_mat(c, k, q), nullptr, 1.0f / sqrtf((float) D), 0.0f);
    ggml_tensor * out = ggml_mul_mat(c, vt, kq);                        // (D, N, H, B)
    out = ggml_cont(c, ggml_permute(c, out, 0, 2, 1, 3));               // (D, H, N, B)

    // one sigmoid gate per head, applied before to_out
    ggml_tensor * g = ggml_sigmoid(c, ggml_add(c, bt_mm(c, a.gates_w, xn), a.gates_b));  // (H, N, B)
    out = ggml_mul(c, out, ggml_reshape_4d(c, g, 1, H, N, B));
    return bt_mm(c, a.out, ggml_reshape_3d(c, out, D * H, N, B));
}

// exact (erf) GELU: torch nn.GELU's default
static ggml_tensor * bt_feedforward(ggml_context * c, const BtFF & f, ggml_tensor * x) {
    ggml_tensor * h = ggml_add(c, bt_mm(c, f.w1, bt_norm(c, x, f.gamma)), f.b1);
    return ggml_add(c, bt_mm(c, f.w2, ggml_gelu_erf(c, h)), f.b2);
}

// x ne (t, f, c) -> same; attention across frequency, then across time
static ggml_tensor * bt_partial(ggml_context * c, const BtBlock & b, ggml_tensor * x,
                                ggml_tensor * pos_f, ggml_tensor * pos_t) {
    ggml_tensor * h = ggml_cont(c, ggml_permute(c, x, 2, 1, 0, 3));  // (c, f, t)
    h = ggml_add(c, h, bt_attention(c, b.attn_f, h, pos_f));
    h = ggml_add(c, h, bt_feedforward(c, b.ff_f, h));
    h = ggml_cont(c, ggml_permute(c, h, 0, 2, 1, 3));                // (c, t, f)
    h = ggml_add(c, h, bt_attention(c, b.attn_t, h, pos_t));
    h = ggml_add(c, h, bt_feedforward(c, b.ff_t, h));
    return ggml_cont(c, ggml_permute(c, h, 2, 0, 1, 3));             // (t, f, c)
}

// One chunk of T frames (already BN1d-normalised, (T, 128) row-major) ->
// beat and downbeat logits, T each.
static void bt_forward(BeatThis * bt, const float * mel, int T, float * beat, float * downbeat) {
    ggml_init_params ip = { ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false), NULL, true };
    ggml_context * c  = ggml_init(ip);
    ggml_cgraph *  gf = ggml_new_graph_custom(c, 4096, false);

    ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, BT_N_MELS, T);
    ggml_set_input(x);
    ggml_tensor * pos_t = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_set_input(pos_t);
    ggml_tensor * pos_f[3];

    // stem: conv over (time, freq) with time fastest -> (T, 32, 32)
    ggml_tensor * h = ggml_reshape_4d(c, ggml_cont(c, ggml_transpose(c, x)), T, BT_N_MELS, 1, 1);
    h = ggml_conv_2d(c, bt->stem_w, h, 1, 4, 1, 0, 1, 1);
    h = ggml_gelu_erf(c, ggml_add(c, h, ggml_reshape_3d(c, bt->stem_b, 1, 1, bt->stem_b->ne[0])));

    for (int i = 0; i < 3; i++) {
        const BtBlock & b = bt->blocks[i];
        pos_f[i] = ggml_new_tensor_1d(c, GGML_TYPE_I32, h->ne[1]);
        ggml_set_input(pos_f[i]);
        h = bt_partial(c, b, ggml_reshape_3d(c, h, h->ne[0], h->ne[1], h->ne[2]), pos_f[i], pos_t);
        h = ggml_conv_2d(c, b.conv_w, ggml_reshape_4d(c, h, h->ne[0], h->ne[1], h->ne[2], 1), 1, 2, 1, 0, 1, 1);
        h = ggml_gelu_erf(c, ggml_add(c, h, ggml_reshape_3d(c, b.conv_b, 1, 1, b.conv_b->ne[0])));
    }

    // "b c f t -> b t (c f)": frequency fastest within each channel
    h = ggml_cont(c, ggml_permute(c, h, 2, 0, 1, 3));  // (t, f, c) -> (f, c, t)
    h = ggml_reshape_2d(c, h, h->ne[0] * h->ne[1], T);
    h = ggml_add(c, ggml_mul_mat(c, bt->linear_w, h), bt->linear_b);
    h = ggml_reshape_3d(c, h, BT_DIM, T, 1);
    for (int l = 0; l < BT_LAYERS; l++) {
        h = ggml_add(c, h, bt_attention(c, bt->attn[l], h, pos_t));
        h = ggml_add(c, h, bt_feedforward(c, bt->ff[l], h));
    }
    h = bt_norm(c, h, bt->out_norm);
    ggml_tensor * out = ggml_add(c, bt_mm(c, bt->head_w, h), bt->head_b);  // (2, T, 1)
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_backend_sched_reset(bt->sched);
    if (!ggml_backend_sched_alloc_graph(bt->sched, gf)) {
        fprintf(stderr, "[beat-this] FATAL: graph alloc failed\n");
        exit(1);
    }
    ggml_backend_tensor_set(x, mel, 0, (size_t) T * BT_N_MELS * 4);
    std::vector<int32_t> pos((size_t) std::max(T, 32));
    for (size_t i = 0; i < pos.size(); i++) pos[i] = (int32_t) i;
    ggml_backend_tensor_set(pos_t, pos.data(), 0, (size_t) T * 4);
    for (int i = 0; i < 3; i++) ggml_backend_tensor_set(pos_f[i], pos.data(), 0, (size_t) pos_f[i]->ne[0] * 4);
    if (ggml_backend_sched_graph_compute(bt->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[beat-this] FATAL: graph compute failed\n");
        exit(1);
    }
    std::vector<float> bd((size_t) 2 * T);
    ggml_backend_tensor_get(out, bd.data(), 0, bd.size() * 4);
    ggml_free(c);
    // SumHead: beat = beat + downbeat logits (suppresses downbeats that aren't beats)
    for (int t = 0; t < T; t++) {
        beat[t]     = bd[2 * t] + bd[2 * t + 1];
        downbeat[t] = bd[2 * t + 1];
    }
}

// Framewise logits for a whole piece: inference.py split_piece +
// aggregate_prediction, chunk 1500, border 6, keep_first, avoid_short_end.
static void bt_logits(BeatThis * bt, const std::vector<float> & mel, int T,
                      std::vector<float> & beat, std::vector<float> & downbeat) {
    const int stride = BT_CHUNK - 2 * BT_BORDER;
    std::vector<int> starts;
    for (int s = -BT_BORDER; s < T - BT_BORDER; s += stride) starts.push_back(s);
    if (T > stride) starts.back() = T - (BT_CHUNK - BT_BORDER);

    beat.assign((size_t) T, -1000.0f);      // upstream's "no prediction" logit
    downbeat.assign((size_t) T, -1000.0f);
    std::vector<float> chunk((size_t) BT_CHUNK * BT_N_MELS), cb(BT_CHUNK), cd(BT_CHUNK);
    for (int ci = (int) starts.size() - 1; ci >= 0; ci--) {  // reversed: earlier chunks win
        const int start = starts[ci];
        const int lo = std::max(start, 0), hi = std::min(start + BT_CHUNK, T);
        const int left = std::max(0, -start), right = std::max(0, std::min(BT_BORDER, start + BT_CHUNK - T));
        const int len = left + (hi - lo) + right;
        // zero padding happens BEFORE the stem BatchNorm, as in the model
        std::fill(chunk.begin(), chunk.begin() + (size_t) len * BT_N_MELS, 0.0f);
        memcpy(chunk.data() + (size_t) left * BT_N_MELS, mel.data() + (size_t) lo * BT_N_MELS,
               (size_t) (hi - lo) * BT_N_MELS * 4);
        for (int t = 0; t < len; t++) {
            for (int m = 0; m < BT_N_MELS; m++) {
                float & v = chunk[(size_t) t * BT_N_MELS + m];
                v = v * bt->bn1d_scale[m] + bt->bn1d_shift[m];
            }
        }
        bt_forward(bt, chunk.data(), len, cb.data(), cd.data());
        for (int j = BT_BORDER; j < len - BT_BORDER; j++) {
            const int dst = start + j;
            if (dst < 0 || dst >= T) continue;
            beat[dst]     = cb[j];
            downbeat[dst] = cd[j];
        }
    }
}

// postprocessor.py "minimal": a frame is a peak when it equals the max over
// +/-3 frames and its logit is > 0 (p > 0.5); runs of peaks at most one frame
// apart collapse to their running mean.
static std::vector<double> bt_peaks(const std::vector<float> & logit) {
    const int T = (int) logit.size();
    std::vector<int> peaks;
    for (int i = 0; i < T; i++) {
        if (!(logit[i] > 0.0f)) continue;
        float mx = logit[i];
        for (int j = std::max(0, i - 3); j <= std::min(T - 1, i + 3); j++) mx = std::max(mx, logit[j]);
        if (logit[i] == mx) peaks.push_back(i);
    }
    std::vector<double> out;
    if (peaks.empty()) return out;
    double p = peaks[0];
    int    n = 1;
    for (size_t i = 1; i < peaks.size(); i++) {
        if (peaks[i] - p <= 1.0) {
            n++;
            p += (peaks[i] - p) / n;
        } else {
            out.push_back(p / BT_FPS);
            p = peaks[i];
            n = 1;
        }
    }
    out.push_back(p / BT_FPS);
    return out;
}

static BeatTrack bt_from_logits(const std::vector<float> & beat, const std::vector<float> & downbeat) {
    BeatTrack tr;
    tr.beats = bt_peaks(beat);
    std::vector<double> db = bt_peaks(downbeat);
    if (!tr.beats.empty()) {
        for (double & d : db) {  // each downbeat moves to its nearest beat (first on a tie)
            size_t best = 0;
            for (size_t i = 1; i < tr.beats.size(); i++) {
                if (fabs(tr.beats[i] - d) < fabs(tr.beats[best] - d)) best = i;
            }
            d = tr.beats[best];
        }
    }
    std::sort(db.begin(), db.end());
    db.erase(std::unique(db.begin(), db.end()), db.end());
    tr.downbeats = db;
    return tr;
}

// Mono audio at any rate -> beats + downbeats. Upstream MuScriptor feeds its
// 16 kHz transcription audio; this takes the source rate, so the tracker sees
// the full band it was trained on.
static BeatTrack bt_track(BeatThis * bt, const float * mono, int n, int sr) {
    std::vector<float> pcm;
    if (sr != BT_SAMPLE_RATE) {
        int     n_out = 0;
        float * r     = audio_resample(mono, n, sr, BT_SAMPLE_RATE, 1, &n_out);
        if (!r || n_out <= 0) {
            fprintf(stderr, "[beat-this] resample %d -> %d failed\n", sr, BT_SAMPLE_RATE);
            return {};
        }
        pcm.assign(r, r + n_out);
        free(r);
    } else {
        pcm.assign(mono, mono + n);
    }
    if (pcm.size() < (size_t) BT_N_FFT) return {};
    const int          T   = 1 + (int) pcm.size() / BT_HOP;
    std::vector<float> mel = bt_logmel(bt, pcm.data(), (int) pcm.size());
    std::vector<float> beat, downbeat;
    bt_logits(bt, mel, T, beat, downbeat);
    return bt_from_logits(beat, downbeat);
}
