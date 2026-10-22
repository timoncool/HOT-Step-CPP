#pragma once
// train/yue2-nar-audio-loss.h — the audio-domain loss behind
// `ace-train yue2-nar-train --audio-loss <w>`.
//
// HOT-Step file, TRAINING-SIDE (ace-train only).
//
// ── What it computes ───────────────────────────────────────────────────────
//
// The flow loss compares velocities in latent space. This term decodes the
// network's own clean-latent estimate through the frozen YuE2 VAE decoder and
// compares the resulting audio with the real recording:
//
//     x0_hat = x_t - t * v_pred          (x_t = (1-t) z + t n, v* = n - z)
//     audio  = VAE_dec(x0_hat[s0 : s0 + win])[margin .. margin + frames]
//     aux    = mel + 0.5 * mrstft + 0.5 * |side(audio) - side(ref)|
//
// `aux` is the recipe of the joint_v6 trainer that produced the companion
// adapter (Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4,
// scripts/joint_v6.py, `spec_loss`), so its weights carry over:
//   mel     L1 of log(mel + 1e-5), 128 HTK mels over |STFT_2048|, mono
//   mrstft  mean over N in {512, 1024, 2048} of spectral convergence plus L1
//           of log(|STFT| + 1e-5), hop N/4, Hann, mono
//   side    log(E[side^2] + 1e-7) - log(E[mid^2] + 1e-7), the stereo width
// Two deliberate differences: frames are not centre-padded (both sides are
// framed identically, so only the edge frames differ), and the mel hop is 512
// rather than 480 (framing below needs the hop to divide N).
//
// ── Why the decoder is rebuilt here instead of reusing yue2-vae-graph.h ────
//
// The inference decoder is not differentiable in this ggml: IM2COL's backward
// is IM2COL_BACK, which has no CUDA kernel (every conv would round-trip to the
// CPU), and PAD has no backward at all. So this file builds the same network
// from ops whose backward runs on CUDA:
//   * a k-tap conv is k matmuls over shifted contiguous views of a zero-padded
//     input, summed (no im2col);
//   * zero padding is ACC of the input into a view of a persistent zero
//     buffer (no PAD);
//   * ConvTranspose1d is the inference path's GEMM + overlap-add, with the
//     overlap-add done by two ACCs onto a zero canvas.
// The layout is [C, T] (ne0 = channels) rather than inference's [T, C], which
// makes every shifted tap a contiguous column span and lets the NAR's own
// [LD, T] prediction feed the decoder with no transpose.
//
// `--fd-check` with `--audio-loss` checks this against the inference decoder
// (yue2_vae_decode) on the same latent before it finite-differences anything.
//
// STFT framing uses the same trick: with hop N/4 the frames split into four
// non-overlapping sets, each a contiguous [N, F] view of the signal, so a
// frame set is one matmul against a windowed DFT basis. Overlapping strided
// views would be wrong here: VIEW's backward is an ACC, and the CUDA ACC
// kernel gives each output element one source element.

#include "train/yue2-preprocess-run.h"  // yp_read_ffmpeg / yp_read_native_48k, yue2-loudness.h

#include "yue2/yue2-model.h"
#include "yue2/yue2-vae-graph.h"  // yue2_vae_readback, yue2_vae_stage2d

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// A conv's frozen weight, repacked once on the host to [IC, OC*K]: tap k is
// the contiguous [IC, OC] column block at k*OC, ready for mul_mat.
struct Yue2AlConv {
    ggml_tensor * w  = nullptr;
    ggml_tensor * b  = nullptr;  // [OC] raw model tensor, or nullptr
    int64_t       ic = 0, oc = 0;
    int           k  = 1;
};

struct Yue2AlSnake {
    ggml_tensor * exp_alpha = nullptr;  // [C, 1]
    ggml_tensor * inv_beta  = nullptr;  // [C, 1]
};

struct Yue2AlRes {
    Yue2AlSnake s1, s2;
    Yue2AlConv  c1, c2;
    int         dil = 1;
};

struct Yue2AlBlock {
    Yue2AlSnake            pre;
    ggml_tensor *          up_w = nullptr;  // [IC, OC*K], row index oc + k*OC
    ggml_tensor *          up_b = nullptr;  // [OC]
    int64_t                ic = 0, oc = 0;
    int                    stride = 1;
    std::vector<Yue2AlRes> res;
};

struct Yue2AlStft {
    int           n  = 0;
    ggml_tensor * re = nullptr;  // [N, N/2+1] Hann * cos
    ggml_tensor * im = nullptr;  // [N, N/2+1] Hann * -sin
};

struct Yue2AudioLoss {
    WeightCtx                w = {};
    Yue2AlConv               conv_in, conv_out;
    std::vector<Yue2AlBlock> blk;
    Yue2AlSnake              snake_out;
    ggml_tensor *            zero   = nullptr;  // the one zero canvas every pad/overlap-add views
    int64_t                  zero_n = 0;
    std::vector<Yue2AlStft>  stft;              // 512, 1024, 2048
    ggml_tensor *            mel_fb = nullptr;  // [1025, 128]
    int64_t                  frames = 0;        // decoded frames the loss scores
    int64_t                  margin = 0;        // decoded context each side, cropped off
    int64_t                  ratio  = 1920;     // samples per latent frame
    int64_t                  latent_dim = 64;

    int64_t win() const { return frames + 2 * margin; }
    int64_t samples() const { return frames * ratio; }
};

static void yue2_al_free(Yue2AudioLoss * A) {
    wctx_free(&A->w);
    *A = Yue2AudioLoss{};
}

// ── Prep ───────────────────────────────────────────────────────────────────

static bool yue2_al_conv_prep(WeightCtx * wc, const ggml_tensor * w, ggml_tensor * b, const char * name,
                              Yue2AlConv * out, std::string * err) {
    std::vector<float> src;
    if (!yue2_vae_readback(w, &src, err, name)) {
        return false;
    }
    const int64_t K = w->ne[0], IC = w->ne[1], OC = w->ne[2];  // [K, IC, OC]
    auto          dst = std::make_unique<float[]>((size_t) (IC * OC * K));
    for (int64_t oc = 0; oc < OC; oc++) {
        for (int64_t ic = 0; ic < IC; ic++) {
            for (int64_t k = 0; k < K; k++) {
                dst[(size_t) (ic + (oc + k * OC) * IC)] = src[(size_t) (k + ic * K + oc * K * IC)];
            }
        }
    }
    out->w  = yue2_vae_stage2d(wc, IC, OC * K, std::move(dst), name);
    out->b  = b;
    out->ic = IC;
    out->oc = OC;
    out->k  = (int) K;
    return true;
}

static bool yue2_al_snake_prep(WeightCtx * wc, const ggml_tensor * alpha, const ggml_tensor * beta, float eps,
                               const std::string & name, Yue2AlSnake * out, std::string * err) {
    std::vector<float> a, b;
    if (!yue2_vae_readback(alpha, &a, err, (name + ".alpha").c_str()) ||
        !yue2_vae_readback(beta, &b, err, (name + ".beta").c_str())) {
        return false;
    }
    const int64_t C  = (int64_t) a.size();
    auto          ea = std::make_unique<float[]>((size_t) C);
    auto          ib = std::make_unique<float[]>((size_t) C);
    for (int64_t i = 0; i < C; i++) {
        ea[(size_t) i] = std::exp(a[(size_t) i]);
        ib[(size_t) i] = 1.0f / (std::exp(b[(size_t) i]) + eps);
    }
    out->exp_alpha = yue2_vae_stage2d(wc, C, 1, std::move(ea), (name + ".ea").c_str());
    out->inv_beta  = yue2_vae_stage2d(wc, C, 1, std::move(ib), (name + ".ib").c_str());
    return true;
}

// torchaudio.functional.melscale_fbanks(n_freqs, 0, sr/2, n_mels, sr,
// norm=None, mel_scale="htk"), the filterbank MelSpectrogram's defaults use.
// Stored [n_freqs, n_mels] so mul_mat(fb, mag[n_freqs, F]) gives [n_mels, F].
static std::unique_ptr<float[]> yue2_al_mel_fb(int n_freqs, int n_mels, double sr) {
    auto hz2mel = [](double f) { return 2595.0 * std::log10(1.0 + f / 700.0); };
    auto mel2hz = [](double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); };
    std::vector<double> f_pts((size_t) n_mels + 2);
    const double        m0 = hz2mel(0.0), m1 = hz2mel(sr / 2.0);
    for (int i = 0; i < n_mels + 2; i++) {
        f_pts[(size_t) i] = mel2hz(m0 + (m1 - m0) * i / (n_mels + 1));
    }
    auto fb = std::make_unique<float[]>((size_t) n_freqs * n_mels);
    for (int k = 0; k < n_freqs; k++) {
        const double f = (sr / 2.0) * k / (n_freqs - 1);
        for (int m = 0; m < n_mels; m++) {
            const double down = (f - f_pts[(size_t) m]) / (f_pts[(size_t) m + 1] - f_pts[(size_t) m]);
            const double up   = (f_pts[(size_t) m + 2] - f) / (f_pts[(size_t) m + 2] - f_pts[(size_t) m + 1]);
            fb[(size_t) (k + (int64_t) m * n_freqs)] = (float) std::max(0.0, std::min(down, up));
        }
    }
    return fb;
}

// Largest canvas any pad or overlap-add needs for a `win`-frame decode.
static int64_t yue2_al_zero_need(const Yue2Model & m, const Yue2AudioLoss & A, int64_t win) {
    (void) m;
    int64_t               T    = win;
    int64_t               need = A.conv_in.ic * (T + 6);
    for (const Yue2AlBlock & b : A.blk) {
        const int S = b.stride, p = (S + 1) / 2;
        need        = std::max(need, b.oc * (int64_t) S * (T + 1));
        T           = (T + 1) * S - 2 * p;
        for (const Yue2AlRes & r : b.res) {
            need = std::max(need, b.oc * (T + 6 * (int64_t) r.dil));
        }
    }
    return std::max(need, A.conv_out.ic * (T + 6));
}

// Load the decoder twin for `frames` scored frames with `margin` frames of
// context each side. Needs the VAE resident (yue2_load_parts want_vae=true).
static bool yue2_al_init(const Yue2Model & m, int64_t frames, int64_t margin, Yue2AudioLoss * A,
                         std::string * err) {
    if (!m.vae_resident) {
        *err = "the YuE2 VAE is not resident";
        return false;
    }
    const Yue2VaeConfig &  vc  = m.vae_cfg;
    const Yue2VaeWeights & vw  = m.vae;
    const int              NB  = (int) vc.strides.size();
    const int              NR  = (int) vc.dec_res_dilations.size();
    const float            eps = vc.snake_eps > 0.0f ? vc.snake_eps : 1e-9f;
    A->frames     = frames;
    A->margin     = margin;
    A->ratio      = (int64_t) vc.downsampling_ratio;
    A->latent_dim = (int64_t) vc.latent_dim;

    // conv_in, per block (snake 2 + upsample 1 + NR * (2 snakes 4 + 2 convs 2)),
    // snake_out 2, conv_out 1, zero 1, 3 * 2 STFT bases, mel 1.
    wctx_init(&A->w, 2 + NB * (3 + NR * 6) + 2 + 1 + 1 + 6 + 1 + 8);
    WeightCtx * wc = &A->w;
    bool        ok = yue2_al_conv_prep(wc, vw.dec_conv_in_w, vw.dec_conv_in_b, "al.conv_in", &A->conv_in, err);
    A->blk.assign((size_t) NB, Yue2AlBlock{});
    for (int b = 0; b < NB && ok; b++) {
        const Yue2VaeDecBlock & wb = vw.dec_blk[(size_t) b];
        Yue2AlBlock &           ab = A->blk[(size_t) b];
        char                    nm[96];
        snprintf(nm, sizeof(nm), "al.blk.%d.pre", b);
        ok = yue2_al_snake_prep(wc, wb.snake_pre_alpha, wb.snake_pre_beta, eps, nm, &ab.pre, err);
        if (!ok) {
            break;
        }
        ab.stride = vc.strides[(size_t) (NB - 1 - b)];  // the decoder runs the strides reversed
        // upsample raw [K, OC, IC] -> [IC, OC*K] with row index oc + k*OC
        std::vector<float> src;
        snprintf(nm, sizeof(nm), "al.blk.%d.up", b);
        ok = yue2_vae_readback(wb.upsample_w, &src, err, nm);
        if (!ok) {
            break;
        }
        const int64_t K = wb.upsample_w->ne[0], OC = wb.upsample_w->ne[1], IC = wb.upsample_w->ne[2];
        if (K != 2 * ab.stride) {
            *err = std::string(nm) + ": kernel " + std::to_string(K) + " is not 2*stride";
            ok   = false;
            break;
        }
        auto dst = std::make_unique<float[]>((size_t) (IC * OC * K));
        for (int64_t ic = 0; ic < IC; ic++) {
            for (int64_t oc = 0; oc < OC; oc++) {
                for (int64_t k = 0; k < K; k++) {
                    dst[(size_t) (ic + (oc + k * OC) * IC)] = src[(size_t) (k + oc * K + ic * K * OC)];
                }
            }
        }
        ab.up_w = yue2_vae_stage2d(wc, IC, OC * K, std::move(dst), nm);
        ab.up_b = wb.upsample_b;
        ab.ic   = IC;
        ab.oc   = OC;
        ab.res.assign((size_t) NR, Yue2AlRes{});
        for (int r = 0; r < NR && ok; r++) {
            const Yue2VaeResUnit & wr = wb.res[(size_t) r];
            Yue2AlRes &            ar = ab.res[(size_t) r];
            ar.dil                    = (int) vc.dec_res_dilations[(size_t) r];
            const std::string p       = "al.blk." + std::to_string(b) + ".res." + std::to_string(r);
            ok = yue2_al_snake_prep(wc, wr.snake1_alpha, wr.snake1_beta, eps, p + ".s1", &ar.s1, err) &&
                 yue2_al_snake_prep(wc, wr.snake2_alpha, wr.snake2_beta, eps, p + ".s2", &ar.s2, err) &&
                 yue2_al_conv_prep(wc, wr.conv1_w, wr.conv1_b, (p + ".c1").c_str(), &ar.c1, err) &&
                 yue2_al_conv_prep(wc, wr.conv2_w, wr.conv2_b, (p + ".c2").c_str(), &ar.c2, err);
        }
    }
    ok = ok && yue2_al_snake_prep(wc, vw.dec_snake_out_alpha, vw.dec_snake_out_beta, eps, "al.snake_out",
                                  &A->snake_out, err);
    ok = ok && yue2_al_conv_prep(wc, vw.dec_conv_out_w, nullptr, "al.conv_out", &A->conv_out, err);
    if (!ok) {
        yue2_al_free(A);
        return false;
    }

    A->zero_n = yue2_al_zero_need(m, *A, A->win());
    A->zero   = yue2_vae_stage2d(wc, A->zero_n, 1, std::make_unique<float[]>((size_t) A->zero_n), "al.zero");

    for (int n : { 512, 1024, 2048 }) {
        const int  B = n / 2 + 1;
        auto       re = std::make_unique<float[]>((size_t) n * B), im = std::make_unique<float[]>((size_t) n * B);
        for (int k = 0; k < B; k++) {
            for (int i = 0; i < n; i++) {
                const double w  = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * i / n);  // periodic Hann, torch.hann_window
                const double ph = 2.0 * 3.14159265358979323846 * (double) ((int64_t) k * i % n) / n;
                re[(size_t) (i + (int64_t) k * n)] = (float) (w * std::cos(ph));
                im[(size_t) (i + (int64_t) k * n)] = (float) (-w * std::sin(ph));
            }
        }
        Yue2AlStft st;
        st.n  = n;
        st.re = yue2_vae_stage2d(wc, n, B, std::move(re), ("al.stft" + std::to_string(n) + ".re").c_str());
        st.im = yue2_vae_stage2d(wc, n, B, std::move(im), ("al.stft" + std::to_string(n) + ".im").c_str());
        A->stft.push_back(st);
    }
    A->mel_fb = yue2_vae_stage2d(wc, 1025, 128, yue2_al_mel_fb(1025, 128, 48000.0), "al.mel_fb");

    if (!wctx_alloc(wc, m.backend)) {
        *err = "audio-loss weight buffer allocation failed";
        yue2_al_free(A);
        return false;
    }
    fprintf(stderr, "[yue2-audio] decoder twin ready: %lld+2x%lld frames per decode, zero canvas %.1f MB, %.1f MB "
                    "total\n",
            (long long) frames, (long long) margin, A->zero_n * 4.0 / 1048576.0,
            ggml_backend_buffer_get_size(wc->buffer) / 1048576.0);
    return true;
}

// ── The decoder twin, [C, T] layout ────────────────────────────────────────

static ggml_tensor * yue2_al_canvas(ggml_context * ctx, const Yue2AudioLoss & A, int64_t ne0, int64_t ne1) {
    GGML_ASSERT(ne0 * ne1 <= A.zero_n && "audio-loss zero canvas too small for this window");
    return ggml_view_2d(ctx, A.zero, ne0, ne1, (size_t) ne0 * sizeof(float), 0);
}

// x [C, T] placed at column `at` of a zero [C, cols] canvas.
static ggml_tensor * yue2_al_place(ggml_context * ctx, const Yue2AudioLoss & A, ggml_tensor * x, int64_t cols,
                                   int64_t at) {
    ggml_tensor * z = yue2_al_canvas(ctx, A, x->ne[0], cols);
    return ggml_acc(ctx, z, x, z->nb[1], z->nb[2], z->nb[3], (size_t) at * z->nb[1]);
}

static ggml_tensor * yue2_al_snake(ggml_context * ctx, ggml_tensor * x, const Yue2AlSnake & s) {
    ggml_tensor * d = ggml_mul(ctx, ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, x, s.exp_alpha))), s.inv_beta);
    return ggml_add(ctx, x, d);
}

// 'same' conv, stride 1: pad (k-1)/2*dil each side, then one matmul per tap.
static ggml_tensor * yue2_al_conv(ggml_context * ctx, const Yue2AudioLoss & A, const Yue2AlConv & c,
                                  ggml_tensor * x, int dil) {
    const int64_t T  = x->ne[1];
    const int64_t p  = (int64_t) ((c.k - 1) / 2) * dil;
    ggml_tensor * xp = p > 0 ? yue2_al_place(ctx, A, x, T + 2 * p, p) : x;
    ggml_tensor * y  = nullptr;
    for (int k = 0; k < c.k; k++) {
        ggml_tensor * wk = ggml_view_2d(ctx, c.w, c.ic, c.oc, c.w->nb[1], (size_t) k * c.oc * c.w->nb[1]);
        ggml_tensor * xk = ggml_view_2d(ctx, xp, c.ic, T, xp->nb[1], (size_t) k * dil * xp->nb[1]);
        ggml_tensor * t  = ggml_mul_mat(ctx, wk, xk);  // [OC, T]
        y                = y ? ggml_add(ctx, y, t) : t;
    }
    if (c.b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, c.b, c.oc, 1));
    }
    return y;
}

// ConvTranspose1d, k = 2*S, pad = ceil(S/2): out[j*S + k] += W[k] x[j].
static ggml_tensor * yue2_al_convt(ggml_context * ctx, const Yue2AudioLoss & A, const Yue2AlBlock & b,
                                   ggml_tensor * x) {
    const int64_t S = b.stride, K = 2 * S, OC = b.oc, Tin = x->ne[1];
    ggml_tensor * c3 = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, b.up_w, x), OC, K, Tin);  // [OC, K, Tin]
    ggml_tensor * lo = ggml_cont(ctx, ggml_view_3d(ctx, c3, OC, S, Tin, c3->nb[1], c3->nb[2], 0));
    ggml_tensor * hi = ggml_cont(ctx, ggml_view_3d(ctx, c3, OC, S, Tin, c3->nb[1], c3->nb[2], (size_t) S * c3->nb[1]));
    lo               = ggml_reshape_2d(ctx, lo, OC, S * Tin);
    hi               = ggml_reshape_2d(ctx, hi, OC, S * Tin);
    const int64_t Tf = (Tin + 1) * S, p = (S + 1) / 2;
    ggml_tensor * y  = yue2_al_place(ctx, A, lo, Tf, 0);
    y                = ggml_acc(ctx, y, hi, y->nb[1], y->nb[2], y->nb[3], (size_t) S * y->nb[1]);
    y                = ggml_view_2d(ctx, y, OC, Tf - 2 * p, y->nb[1], (size_t) p * y->nb[1]);
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, b.up_b, OC, 1));
}

// latent [LD, W] -> audio [S_dec, 2] (ne0 = samples: channel 0, then channel 1,
// the inference decoder's planar layout). S_dec = ratio*W minus the decoder's
// fixed few-sample deficit.
static ggml_tensor * yue2_al_decode(ggml_context * ctx, const Yue2AudioLoss & A, ggml_tensor * latent) {
    ggml_tensor * x = yue2_al_conv(ctx, A, A.conv_in, latent, 1);
    for (const Yue2AlBlock & b : A.blk) {
        x = yue2_al_convt(ctx, A, b, yue2_al_snake(ctx, x, b.pre));
        for (const Yue2AlRes & r : b.res) {
            ggml_tensor * y = yue2_al_conv(ctx, A, r.c1, yue2_al_snake(ctx, x, r.s1), r.dil);
            y               = yue2_al_conv(ctx, A, r.c2, yue2_al_snake(ctx, y, r.s2), 1);
            x               = ggml_add(ctx, x, y);
        }
    }
    x = yue2_al_conv(ctx, A, A.conv_out, yue2_al_snake(ctx, x, A.snake_out), 1);  // [2, S_dec]
    return ggml_cont(ctx, ggml_transpose(ctx, x));                                 // [S_dec, 2]
}

// Decode `latent` (a [LD, W] tensor) and keep the scored span: [samples(), 2].
static ggml_tensor * yue2_al_decode_scored(ggml_context * ctx, const Yue2AudioLoss & A, ggml_tensor * latent) {
    ggml_tensor * a = yue2_al_decode(ctx, A, latent);
    GGML_ASSERT(a->ne[0] >= (A.margin + A.frames) * A.ratio);
    return ggml_cont(ctx, ggml_view_2d(ctx, a, A.samples(), 2, a->nb[1], (size_t) (A.margin * A.ratio) * sizeof(float)));
}

// ── The spectral loss ──────────────────────────────────────────────────────

static ggml_tensor * yue2_al_log_eps(ggml_context * ctx, ggml_tensor * x, float eps) {
    return ggml_log(ctx, ggml_scale_bias(ctx, x, 1.0f, eps));
}

// |STFT| of a contiguous 1-D signal, as four non-overlapping frame sets.
static std::vector<ggml_tensor *> yue2_al_mag(ggml_context * ctx, const Yue2AlStft & st, ggml_tensor * sig) {
    std::vector<ggml_tensor *> out;
    const int64_t              N = st.n, H = N / 4, S = sig->ne[0];
    for (int64_t j = 0; j < 4; j++) {
        const int64_t F = (S - j * H) / N;
        if (F <= 0) {
            continue;
        }
        ggml_tensor * fr = ggml_view_2d(ctx, sig, N, F, (size_t) N * sizeof(float), (size_t) (j * H) * sizeof(float));
        ggml_tensor * re = ggml_mul_mat(ctx, st.re, fr);
        ggml_tensor * im = ggml_mul_mat(ctx, st.im, fr);
        out.push_back(ggml_sqrt(ctx, ggml_scale_bias(ctx, ggml_add(ctx, ggml_sqr(ctx, re), ggml_sqr(ctx, im)), 1.0f,
                                                     1e-12f)));
    }
    return out;
}

static ggml_tensor * yue2_al_side(ggml_context * ctx, ggml_tensor * a) {
    const int64_t S = a->ne[0];
    ggml_tensor * l = ggml_view_1d(ctx, a, S, 0);
    ggml_tensor * r = ggml_view_1d(ctx, a, S, a->nb[1]);
    ggml_tensor * side = ggml_mean(ctx, ggml_sqr(ctx, ggml_scale(ctx, ggml_sub(ctx, l, r), 0.5f)));
    ggml_tensor * mid  = ggml_mean(ctx, ggml_sqr(ctx, ggml_scale(ctx, ggml_add(ctx, l, r), 0.5f)));
    return ggml_sub(ctx, yue2_al_log_eps(ctx, side, 1e-7f), yue2_al_log_eps(ctx, mid, 1e-7f));
}

struct Yue2AlTerms {
    ggml_tensor * total = nullptr, * mel = nullptr, * mr = nullptr, * side = nullptr;
};

// pred, ref: [S, 2]. joint_v6's spec_loss (see the file header).
static Yue2AlTerms yue2_al_spec_loss(ggml_context * ctx, const Yue2AudioLoss & A, ggml_tensor * pred,
                                     ggml_tensor * ref) {
    auto mono = [&](ggml_tensor * a) {
        const int64_t S = a->ne[0];
        return ggml_scale(ctx, ggml_add(ctx, ggml_view_1d(ctx, a, S, 0), ggml_view_1d(ctx, a, S, a->nb[1])), 0.5f);
    };
    ggml_tensor * pm = mono(pred);
    ggml_tensor * rm = mono(ref);

    Yue2AlTerms t;
    for (const Yue2AlStft & st : A.stft) {
        const std::vector<ggml_tensor *> P = yue2_al_mag(ctx, st, pm), R = yue2_al_mag(ctx, st, rm);
        ggml_tensor * num2 = nullptr, * den2 = nullptr, * l1 = nullptr;
        int64_t       cnt  = 0;
        for (size_t i = 0; i < P.size(); i++) {
            ggml_tensor * n2 = ggml_sum(ctx, ggml_sqr(ctx, ggml_sub(ctx, P[i], R[i])));
            ggml_tensor * d2 = ggml_sum(ctx, ggml_sqr(ctx, R[i]));
            ggml_tensor * ab = ggml_sum(ctx, ggml_abs(ctx, ggml_sub(ctx, yue2_al_log_eps(ctx, P[i], 1e-5f),
                                                                     yue2_al_log_eps(ctx, R[i], 1e-5f))));
            num2 = num2 ? ggml_add(ctx, num2, n2) : n2;
            den2 = den2 ? ggml_add(ctx, den2, d2) : d2;
            l1   = l1 ? ggml_add(ctx, l1, ab) : ab;
            cnt += ggml_nelements(P[i]);
            if (st.n == 2048) {
                ggml_tensor * mp = yue2_al_log_eps(ctx, ggml_mul_mat(ctx, A.mel_fb, P[i]), 1e-5f);
                ggml_tensor * mr = yue2_al_log_eps(ctx, ggml_mul_mat(ctx, A.mel_fb, R[i]), 1e-5f);
                ggml_tensor * md = ggml_sum(ctx, ggml_abs(ctx, ggml_sub(ctx, mp, mr)));
                t.mel            = t.mel ? ggml_add(ctx, t.mel, md) : md;
            }
        }
        ggml_tensor * sc = ggml_div(ctx, ggml_sqrt(ctx, num2), ggml_scale_bias(ctx, ggml_sqrt(ctx, den2), 1.0f, 1e-6f));
        ggml_tensor * r  = ggml_scale(ctx, ggml_add(ctx, sc, ggml_scale(ctx, l1, 1.0f / (float) cnt)), 1.0f / 3.0f);
        t.mr             = t.mr ? ggml_add(ctx, t.mr, r) : r;
        if (st.n == 2048) {
            t.mel = ggml_scale(ctx, t.mel, 1.0f / (float) (cnt / 1025 * 128));  // mean over every mel bin
        }
    }
    t.side  = ggml_abs(ctx, ggml_sub(ctx, yue2_al_side(ctx, pred), yue2_al_side(ctx, ref)));
    t.total = ggml_add(ctx, ggml_add(ctx, t.mel, ggml_scale(ctx, t.mr, 0.5f)), ggml_scale(ctx, t.side, 0.5f));
    return t;
}

// The same loss on the host in double, from downloaded audio: the FD gate's
// numeric arm. In-graph f32 sums over ~1e5 spectral bins are deterministic but
// too coarse to difference (the cancellation the flow gate also avoids), and
// this is an independent implementation of the forward besides. Same framing:
// frames at hop N/4 that fit inside the signal, no centre padding.
static double yue2_al_spec_loss_host(const std::vector<float> & pred, const std::vector<float> & ref) {
    const int64_t S = (int64_t) pred.size() / 2;
    auto          mono = [&](const std::vector<float> & a) {
        std::vector<double> m((size_t) S);
        for (int64_t i = 0; i < S; i++) {
            m[(size_t) i] = 0.5 * ((double) a[(size_t) i] + (double) a[(size_t) (S + i)]);
        }
        return m;
    };
    const std::vector<double> pm = mono(pred), rm = mono(ref);
    static std::vector<float> fb;
    if (fb.empty()) {
        auto f = yue2_al_mel_fb(1025, 128, 48000.0);
        fb.assign(f.get(), f.get() + 1025 * 128);
    }
    double mel = 0.0, mr = 0.0;
    for (int N : { 512, 1024, 2048 }) {
        const int           B = N / 2 + 1, H = N / 4;
        std::vector<double> cs((size_t) B * N), sn((size_t) B * N);
        for (int k = 0; k < B; k++) {
            for (int i = 0; i < N; i++) {
                const double w  = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * i / N);
                const double ph = 2.0 * 3.14159265358979323846 * (double) ((int64_t) k * i % N) / N;
                cs[(size_t) k * N + i] = w * std::cos(ph);
                sn[(size_t) k * N + i] = w * std::sin(ph);
            }
        }
        double              num2 = 0, den2 = 0, l1 = 0;
        int64_t             cnt = 0, mcnt = 0;
        std::vector<double> P((size_t) B), R((size_t) B);
        for (int64_t st = 0; st + N <= S; st += H) {
            for (int k = 0; k < B; k++) {
                double pr = 0, pi = 0, rr = 0, ri = 0;
                const double * c = &cs[(size_t) k * N];
                const double * s = &sn[(size_t) k * N];
                for (int i = 0; i < N; i++) {
                    pr += pm[(size_t) (st + i)] * c[i];
                    pi += pm[(size_t) (st + i)] * s[i];
                    rr += rm[(size_t) (st + i)] * c[i];
                    ri += rm[(size_t) (st + i)] * s[i];
                }
                P[(size_t) k] = std::sqrt(pr * pr + pi * pi + 1e-12);
                R[(size_t) k] = std::sqrt(rr * rr + ri * ri + 1e-12);
                num2 += (P[(size_t) k] - R[(size_t) k]) * (P[(size_t) k] - R[(size_t) k]);
                den2 += R[(size_t) k] * R[(size_t) k];
                l1 += std::fabs(std::log(P[(size_t) k] + 1e-5) - std::log(R[(size_t) k] + 1e-5));
                cnt++;
            }
            if (N == 2048) {
                for (int mi = 0; mi < 128; mi++) {
                    double mp = 0, mrr = 0;
                    for (int k = 0; k < B; k++) {
                        mp += fb[(size_t) (k + mi * B)] * P[(size_t) k];
                        mrr += fb[(size_t) (k + mi * B)] * R[(size_t) k];
                    }
                    mel += std::fabs(std::log(mp + 1e-5) - std::log(mrr + 1e-5));
                    mcnt++;
                }
            }
        }
        mr += (std::sqrt(num2) / (std::sqrt(den2) + 1e-6) + l1 / (double) cnt) / 3.0;
        if (N == 2048) {
            mel /= (double) mcnt;
        }
    }
    auto side = [&](const std::vector<float> & a) {
        double s2 = 0, m2 = 0;
        for (int64_t i = 0; i < S; i++) {
            const double l = a[(size_t) i], r = a[(size_t) (S + i)];
            s2 += 0.25 * (l - r) * (l - r);
            m2 += 0.25 * (l + r) * (l + r);
        }
        return std::log(s2 / S + 1e-7) - std::log(m2 / S + 1e-7);
    };
    return mel + 0.5 * mr + 0.5 * std::fabs(side(pred) - side(ref));
}

// ── Reference audio ────────────────────────────────────────────────────────
//
// The latents were encoded from the source decoded to 48 kHz, scaled by the
// source's `loudness_gain_db` (normalisation plus any peak scaling, folded into
// one gain by yue2-preprocess). The reference here is that same audio: same
// decode route (WAV/MP3 native, anything else through ffmpeg, as preprocess
// and tokenize choose it), same gain. Whole tracks are cached LRU, since the
// clip working set revisits a handful of sources for a block of steps.
struct Yue2AlRefCache {
    std::string ffmpeg = "ffmpeg";
    std::string tmp_wav;
    size_t      cap = 8;
    std::list<std::pair<std::string, std::vector<float>>> lru;  // planar [ch0 S][ch1 S], front = newest
    int64_t     loads = 0;
};

static const std::vector<float> * yue2_al_ref_track(Yue2AlRefCache * C, const std::string & path, double gain_db,
                                                    std::string * err) {
    for (auto it = C->lru.begin(); it != C->lru.end(); ++it) {
        if (it->first == path) {
            C->lru.splice(C->lru.begin(), C->lru, it);
            return &C->lru.front().second;
        }
    }
    std::string ext = path.size() > 4 ? path.substr(path.size() - 4) : std::string();
    for (char & ch : ext) {
        ch = (char) tolower((unsigned char) ch);
    }
    int     T      = 0;
    float * planar = nullptr;
    if (ext == ".wav" || ext == ".mp3") {
        planar = yp_read_native_48k(path, &T);
    } else {
        planar = yp_read_ffmpeg(C->ffmpeg, path, C->tmp_wav, 48000, &T, err);
    }
    if (!planar || T <= 0) {
        free(planar);
        if (err && err->empty()) {
            *err = "decode failed";
        }
        return nullptr;
    }
    yue2_apply_gain_db(planar, T, gain_db);
    std::vector<float> v(planar, planar + (size_t) T * 2);
    free(planar);
    C->loads++;
    C->lru.emplace_front(path, std::move(v));
    while (C->lru.size() > C->cap) {
        C->lru.pop_back();
    }
    return &C->lru.front().second;
}

// Samples [start, start+n) of both channels into [n, 2]. False when the span
// runs past the track (the step then trains on the flow loss alone).
static bool yue2_al_ref_slice(const std::vector<float> & planar, int64_t start, int64_t n, std::vector<float> * out) {
    const int64_t S = (int64_t) planar.size() / 2;
    if (start < 0 || start + n > S) {
        return false;
    }
    out->resize((size_t) n * 2);
    for (int ch = 0; ch < 2; ch++) {
        std::copy(planar.begin() + ch * S + start, planar.begin() + ch * S + start + n, out->begin() + ch * n);
    }
    return true;
}
