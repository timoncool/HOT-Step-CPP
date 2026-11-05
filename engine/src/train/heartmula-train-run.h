#pragma once
// train/heartmula-train-run.h — `ace-train heartmula-train`: HeartMuLa backbone
// LoRA training on a heartmula-prepare cache.
//
// HOT-Step file, TRAINING-SIDE (ace-train only). The maths is
// engine/docs/HEARTMULA_TRAINING_CONTRACT.md; the graphs are
// heartmula-train-graph.h, the state heartmula-train-state.h.
//
// ── CLI contract (what the server drives) ──────────────────────────────────
//
//   ace-train heartmula-train --data <prepare dir> --out <run dir>
//       (--models <dir> [--variant 3B] [--type F16] | --global <gguf> --local <gguf>)
//       [--steps 480] [--rank 64] [--alpha 64] [--lr 1e-4] [--weight-decay 0.01]
//       [--grad-accum 4] [--seed 0] [--max-frames 0] [--max-grad-norm 0]
//       [--save-every 100] [--stop-after N] [--resume] [--name <adapter name>]
//       [--weights f16|bf16] [--local-chunk 0] [--lora-dropout 0]
//       [--attn auto|flash-bf16|flash|exact] [--keep-layers auto|-1|0..28]
//       [--lora-mlp]   also train the backbone MLP (w1/w2/w3), not only attention
//       [--trigger <word>]  the word the tags lead with, stored as heartmula.lora.trigger
//       [--eos]        train the end-of-song frame after every window that reaches a song's end
//       [--crop-frames N]  start-anchored windows of N frames, every 4th sample whole (state.h)
//       [--cb0-weighted]  loss factors CB0 2.0, CB1..7 0.7..0.1, normalised (state.h hmt_cb_weights)
//
// Files under --out:
//   hm_train_state.bin          the resumable checkpoint (latest only)
//   adapters/step-NNNNNN.gguf   one immutable adapter per checkpoint, loadable
//                               by heartmula/adapter.h
//   control/checkpoint-now      created by the caller: finish the current
//                               optimizer step, checkpoint, exit 3
//   control/cancel              created by the caller: stop at the next
//                               micro-step, exit 4, last checkpoint untouched
// Stale control files are removed at start, so write them only after the
// `start` event.
//
// stdout carries one JSON object per line, prefixed "HMTRAIN ":
//   {"event":"start", ...config, samples, lmax, tmax, local_chunk, params,
//                     alloc_mib, device_free_mib, weights, attention, bwd}
//   {"event":"step", step, total, loss, ce:[8], lr, grad_norm, ms_step,
//                    ms_micro, micro, peak_alloc_mib, device_used_mib}
//                   loss is the objective; with --cb0-weighted it is sum_k (w_k / 4.8) ce_k
//                   and loss_mean adds the plain (1/8) sum_k ce_k
//   {"event":"skip", step, reason}       non-finite loss or gradient, no update
//   {"event":"checkpoint", step, state, adapter, reason: every|now|stop|final}
//   {"event":"done", step, adapter}
//   {"event":"error", message}
// stderr is the human log.
//
// Exit codes: 0 finished (or already finished on --resume), 3 paused at a
// checkpoint (checkpoint-now or --stop-after), 4 cancelled, 1 failure, 2 usage.
//
// A preview cycle is: --stop-after N (or checkpoint-now), render with
// adapters/step-N.gguf, then the same command line plus --resume.
//
// ── Held-out scoring ───────────────────────────────────────────────────────
//
//   ace-train heartmula-train --data <dir> --models <dir> --eval base|<adapter.gguf> [--eval ...]
//                             [--eval-samples id,id] [--eos] [--attn ...] [--local-chunk N]
//
// Forward only: no backward, no optimizer, no LoRA dropout, nothing written.
// Each --eval model (base = no adapter) scores every sample of --data, or the
// listed ids (manifest id or codes key), as whole songs; --eos appends the
// end frame where training would (hmt_eos_fits). Per sample and per model, on stdout:
//   {"event":"eval_sample", model, sample, frames, eos, loss, ce:[8]}
//   {"event":"eval", model, samples, frames, loss, loss_cb0w, ce:[8]}
// ce are mean CEs per codebook; the model line averages over every scored
// frame (sum_s T_s ce_s / sum_s T_s, summed in --eval-samples order), loss is
// their plain mean and loss_cb0w the --cb0-weighted objective.
//
// ── Selftest ───────────────────────────────────────────────────────────────
//
//   ace-train heartmula-train --selftest <fixture dir> --models <dir> [--weights ...]
//
// Checks contract §5 against the slice-1 fixture set: forward taps and losses
// (§5.1), gradients at the init and perturbed states (§5.2), the optimizer
// replay and the three-step end-to-end run (§5.3), resume bit-identity
// (§5.4); then times clip A whole (S 148, T 1112) for §7. Exit 0 = every
// bound holds.

#include "train/gpu-mem.h"
#include "train/heartmula-train-state.h"

#include <chrono>
#include <map>

struct HeartmulaTrainArgs {
    std::string data_dir, out_dir, models_dir, global_path, local_path, variant = "3B", type = "F16";
    std::string name = "heartmula-lora", selftest, weights = "f16";
    int      rank = 64, accum = 4, save_every = 100;
    float    alpha = 64.0f, lr = 1e-4f, wd = 0.01f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, clip = 0.0f, dropout = 0.0f;
    int64_t  steps = 480, stop_after = 0, max_frames = 0, local_chunk = 0;
    int      keep = -2;            // --keep-layers: -2 auto, -1 recompute every block, 0..28 kept blocks
    std::string attn = "auto";     // --attn auto|flash-bf16|flash|exact
    uint64_t seed = 0;
    bool     resume = false;
    bool     mlp    = false;       // --lora-mlp
    std::string trigger;           // --trigger: written to the adapter, not part of the run identity
    bool     eos = false;          // --eos
    int64_t  crop_frames = 0;      // --crop-frames
    bool     cb0_weighted = false; // --cb0-weighted
    std::vector<std::string> eval; // --eval base|<adapter>: score instead of training
    std::string eval_samples;      // --eval-samples id,id
};

static std::string hmt_json_str(const std::string & s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if ((unsigned char) c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
        else o += c;
    }
    return o + "\"";
}

static void hmt_emit(const std::string & body) {
    fprintf(stdout, "HMTRAIN {%s}\n", body.c_str());
    fflush(stdout);
}

static std::string hmt_num(double v) {
    if (!std::isfinite(v)) return "null";
    char b[40];
    snprintf(b, sizeof(b), "%.7g", v);
    return b;
}

// ── The run ────────────────────────────────────────────────────────────────

struct HmtRun {
    HeartMuLaModel m;
    HmtLora        lora;
    LmOptim        opt;
    HmtCkpt        ck;
    HmtSampleBuf   sb;
    HmtGraphCtx    g;
    int64_t        chunk     = 0;
    int            keep      = -1;  // --keep-layers: -1 recompute everything, else kept global blocks
    float          drop_p    = 0.0f;
    uint64_t       seed      = 0;
    ggml_context *        drop_ctx = nullptr;  // the dropout mask pool (lever 6)
    ggml_backend_buffer_t drop_buf = nullptr;
    size_t         peak_sched = 0;
    bool           taps      = false;
    // selftest taps, filled when `taps` is set
    std::vector<float> tap_last, tap_dec, tap_logits[8];
    // HMT_PROFILE=1: per-segment wall time (build = since the previous compute ended,
    // alloc, compute to a device sync). The sync costs overlap, so profile runs only.
    struct Prof { double build = 0, alloc = 0, comp = 0; int n = 0; };
    bool                        prof = getenv("HMT_PROFILE") != nullptr;
    const char *                prof_tag = "?";
    std::map<std::string, Prof> prof_acc;
    std::chrono::steady_clock::time_point prof_t = std::chrono::steady_clock::now();
};

// Host-side work between graphs, recorded under its own tag (inside the next
// segment's build time as well, so it is a breakdown, not an addition).
template <typename F> static auto hmt_prof_host(HmtRun & R, const char * tag, F && f) {
    if (!R.prof) return f();
    const auto t0 = std::chrono::steady_clock::now();
    auto       r  = f();
    HmtRun::Prof & p = R.prof_acc[tag];
    p.build += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    p.n++;
    return r;
}

static void hmt_prof_dump(HmtRun & R, const char * what) {
    if (!R.prof) return;
    double b = 0, a = 0, c = 0;
    for (auto & [k, p] : R.prof_acc) {
        fprintf(stderr, "[hm-prof] %-14s n %5d  build %8.1f  alloc %8.1f  compute %8.1f ms\n", k.c_str(), p.n, p.build, p.alloc, p.comp);
        b += p.build, a += p.alloc, c += p.comp;
    }
    fprintf(stderr, "[hm-prof] %s total: build %.1f alloc %.1f compute %.1f ms\n", what, b, a, c);
    R.prof_acc.clear();
}

static void hmt_run_free(HmtRun * R) {
    if (R->g.sched) ggml_backend_sched_free(R->g.sched);
    if (R->g.ctx) ggml_free(R->g.ctx);
    R->g.sched = nullptr;
    R->g.ctx   = nullptr;
    hmt_sample_free(&R->sb);
    hmt_ckpt_free(&R->ck);
    if (R->drop_buf) ggml_backend_buffer_free(R->drop_buf);
    if (R->drop_ctx) ggml_free(R->drop_ctx);
    R->drop_buf = nullptr;
    R->drop_ctx = nullptr;
    lm_optim_free(&R->opt);
    hmt_lora_free(&R->lora);
    heartmula_model_free(&R->m);
}

static size_t hmt_alloc_bytes(const HmtRun & R) {
    size_t n = 0;
    for (ggml_backend_buffer_t b : { R.m.wctx_global.buffer, R.m.wctx_local.buffer, R.lora.buf, R.opt.buf_grad, R.opt.buf_mom,
                                     R.opt.buf_alt, R.ck.buf, R.sb.buf, R.sb.lbuf, R.drop_buf }) {
        if (b) n += ggml_backend_buffer_get_size(b);
    }
    return n + R.peak_sched;
}

static bool hmt_compute(HmtRun & R, ggml_cgraph * gf, std::string * err) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    ggml_backend_sched_reset(R.g.sched);
    if (!ggml_backend_sched_alloc_graph(R.g.sched, gf)) { *err = "graph allocation failed (out of VRAM?)"; return false; }
    const auto t1 = clk::now();
    R.peak_sched = std::max(R.peak_sched, ggml_backend_sched_get_buffer_size(R.g.sched, R.m.bp.backend));
    if (ggml_backend_sched_graph_compute(R.g.sched, gf) != GGML_STATUS_SUCCESS) { *err = "graph compute failed"; return false; }
    if (R.prof) {
        ggml_backend_sched_synchronize(R.g.sched);
        const auto t2 = clk::now();
        auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        HmtRun::Prof & p = R.prof_acc[R.prof_tag];
        p.build += ms(R.prof_t, t0), p.alloc += ms(t0, t1), p.comp += ms(t1, t2), p.n++;
        R.prof_t = t2;
    }
    return true;
}

static float hmt_read1(ggml_tensor * t) {
    float v = 0.0f;
    ggml_backend_tensor_get(t, &v, 0, sizeof(v));
    return v;
}

// One sample, forward and backward; LoRA gradients accumulate in R.opt.acc.
// `ce` receives the eight per-codebook mean CEs. `wce` = 1 / (8 x micro-steps in the step).
static bool hmt_micro_step(HmtRun & R, const int32_t * prompt, int64_t S, int64_t muq_idx, const int32_t * codes,
                           int64_t T, float wce, double ce[8], std::string * err) {
    HmtGraphCtx & g = R.g;
    if (!hmt_sample_alloc(&R.sb, R.m.bp.backend, S, T, err)) return false;
    hmt_prof_host(R, "host upload", [&] { hmt_sample_upload(&R.sb, prompt, muq_idx, codes); return 0; });
    ggml_backend_buffer_clear(R.sb.lbuf, 0);
    const int64_t chunk = R.chunk > 0 ? std::min(R.chunk, T) : T;
    auto run = [&](ggml_cgraph * gf) { return hmt_compute(R, gf, err); };
    auto tag = [&](const char * t) { R.prof_tag = t; };

    // Forward, checkpointing every block boundary.
    tag("embed");
    if (!run(hmt_g_labels(g)) || !run(hmt_g_embed(g))) return false;
    if (R.keep >= 0) {
        // Kept tail (--keep-layers): blocks k0..27 and the local decoder in one graph, no recompute.
        const int k0 = 28 - R.keep;
        tag("global fwd");
        for (int l = 0; l < k0; l++) if (!run(hmt_g_global_fwd(g, l))) return false;
        tag("kept");
        if (k0 > 0) ggml_backend_tensor_memset(R.ck.Gg[0], 0, 0, ggml_nbytes(R.ck.Gg[0]));
        HmtHeadOut head;
        HmtTailOut tail;
        if (!run(hmt_g_kept_bwd(g, k0, wce, &head, &tail, R.taps))) return false;
        ce[0] = hmt_read1(head.ce0);
        for (int i = 1; i < 8; i++) ce[i] = hmt_read1(tail.ce[i]);
        if (R.taps) {
            auto get = [](ggml_tensor * t, std::vector<float> & v) {
                v.resize((size_t) ggml_nelements(t));
                ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
            };
            get(tail.dec, R.tap_dec);
            get(head.last, R.tap_last);
            get(head.logits, R.tap_logits[0]);
            for (int i = 1; i < 8; i++) get(tail.logits[i], R.tap_logits[i]);
        }
        tag("global bwd");
        int cur = 0;
        for (int l = k0 - 1; l >= 0; l--) {
            if (l > 0) ggml_backend_tensor_memset(R.ck.Gg[1 - cur], 0, 0, ggml_nbytes(R.ck.Gg[1 - cur]));
            if (!run(hmt_g_global_bwd(g, l, cur))) return false;
            cur = 1 - cur;
        }
        tag("optimizer");
        return true;
    }
    tag("global fwd");
    for (int l = 0; l < 28; l++) if (!run(hmt_g_global_fwd(g, l))) return false;
    tag("head fwd");
    if (!run(hmt_g_head_fwd(g))) return false;
    tag("local fwd");
    for (int l = 0; l < 3; l++)
        for (int64_t f0 = 0; f0 < T; f0 += chunk) if (!run(hmt_g_local_fwd(g, l, f0, std::min(chunk, T - f0)))) return false;
    tag("tail bwd");

    // Backward, last segment first.
    for (int k = 0; k < 8; k++) ce[k] = 0.0;
    ggml_backend_tensor_memset(R.ck.Gl[0], 0, 0, ggml_nbytes(R.ck.Gl[0]));
    for (int64_t f0 = 0; f0 < T; f0 += chunk) {
        const int64_t Tc = std::min(chunk, T - f0);
        HmtTailOut    out;
        if (!run(hmt_g_tail_bwd(g, f0, Tc, wce, 0, &out, R.taps))) return false;
        for (int i = 1; i < 8; i++) ce[i] += (double) hmt_read1(out.ce[i]) * (double) Tc / (double) T;
        if (R.taps) {
            R.tap_dec.resize((size_t) ggml_nelements(out.dec));
            ggml_backend_tensor_get(out.dec, R.tap_dec.data(), 0, ggml_nbytes(out.dec));
            for (int i = 1; i < 8; i++) {
                R.tap_logits[i].resize((size_t) ggml_nelements(out.logits[i]));
                ggml_backend_tensor_get(out.logits[i], R.tap_logits[i].data(), 0, ggml_nbytes(out.logits[i]));
            }
        }
    }
    tag("local bwd");
    int cur = 0;
    for (int l = 2; l >= 0; l--) {
        ggml_backend_tensor_memset(R.ck.Gl[1 - cur], 0, 0, ggml_nbytes(R.ck.Gl[1 - cur]));
        for (int64_t f0 = 0; f0 < T; f0 += chunk) if (!run(hmt_g_local_bwd(g, l, f0, std::min(chunk, T - f0), cur))) return false;
        cur = 1 - cur;
    }
    ggml_backend_tensor_memset(R.ck.Gg[0], 0, 0, ggml_nbytes(R.ck.Gg[0]));
    HmtHeadOut head;
    tag("head bwd");
    if (!run(hmt_g_head_bwd(g, cur, wce, &head, R.taps))) return false;
    ce[0] = hmt_read1(head.ce0);
    if (R.taps) {
        R.tap_last.resize((size_t) ggml_nelements(head.last));
        ggml_backend_tensor_get(head.last, R.tap_last.data(), 0, ggml_nbytes(head.last));
        R.tap_logits[0].resize((size_t) ggml_nelements(head.logits));
        ggml_backend_tensor_get(head.logits, R.tap_logits[0].data(), 0, ggml_nbytes(head.logits));
    }
    tag("global bwd");
    cur = 0;
    for (int l = 27; l >= 0; l--) {
        if (l > 0) ggml_backend_tensor_memset(R.ck.Gg[1 - cur], 0, 0, ggml_nbytes(R.ck.Gg[1 - cur]));
        if (!run(hmt_g_global_bwd(g, l, cur))) return false;
        cur = 1 - cur;
    }
    tag("optimizer");
    return true;
}

// Sum of squares of the accumulated gradients (finite check before a step).
static bool hmt_grad_norm2(HmtRun & R, double * out, std::string * err) {
    if (!hmt_compute(R, hmt_g_gradnorm(R.g), err)) return false;
    *out = hmt_read1(R.ck.gn2);
    return true;
}

static bool hmt_resolve_models(HeartmulaTrainArgs * a, std::string * err) {
    if (!a->global_path.empty() || !a->local_path.empty()) {
        if (a->global_path.empty() || a->local_path.empty()) { *err = "--global and --local go together"; return false; }
        return true;
    }
    if (a->models_dir.empty()) { *err = "one of --models or --global/--local is required"; return false; }
    auto lower = [](std::string s) { for (auto & c : s) c = (char) std::tolower((unsigned char) c); return s; };
    const HeartMuLaCatalogue cat = heartmula_discover(a->models_dir);
    for (const auto & f : cat.files) {
        if (!f.compatible || lower(f.variant) != lower(a->variant) || lower(f.type) != lower(a->type)) continue;
        if (f.role == HM_GLOBAL && a->global_path.empty()) a->global_path = f.path;
        if (f.role == HM_LOCAL && a->local_path.empty()) a->local_path = f.path;
    }
    if (a->global_path.empty() || a->local_path.empty()) {
        *err = "no compatible HeartMuLa " + a->variant + " " + a->type + " global+local pair under " + a->models_dir;
        return false;
    }
    return true;
}

// Model, factors, optimizer. The optimizer runs constant-LR AdamW (contract §1).
static bool hmt_setup(HmtRun & R, const HeartmulaTrainArgs & a, std::string * err) {
    if (!hmt_model_load(&R.m, a.global_path.c_str(), a.local_path.c_str(), a.weights == "bf16", err)) return false;
    if (!hmt_lora_init(&R.lora, R.m, a.rank, a.alpha, a.seed, a.mlp, err)) return false;
    LmOptim & o = R.opt;
    o.optimizer    = "adamw";
    o.t_adamw      = R.lora.t_adamw;
    o.t_lossgrad   = R.lora.t_lossgrad;
    o.t_clip       = R.lora.t_clip;
    o.t_eps        = R.lora.t_eps;
    o.t_gnorm2     = R.lora.t_gnorm2;
    o.base_lr      = a.lr;
    o.lr_floor     = 1.0f;  // with warmup 0 this makes lm_lr_lambda identically 1: constant LR
    o.warmup_steps = 0;
    o.total_steps  = 1;
    o.weight_decay = a.wd;
    o.adam_beta1   = a.beta1;
    o.adam_beta2   = a.beta2;
    o.adam_eps     = a.eps;
    o.grad_clip    = a.clip;
    ggml_backend_tensor_set(R.lora.t_clip, &a.clip, 0, sizeof(float));
    if (!lm_optim_init(&o, R.lora.params, R.m.bp.backend, err)) return false;
    R.g.m     = &R.m;
    R.g.lora  = &R.lora.layers;
    R.g.scale = a.alpha / (float) a.rank;
    R.g.opt   = &R.opt;
    R.g.ck    = &R.ck;
    R.g.sb    = &R.sb;
    R.g.sched = backend_sched_new(R.m.bp, HMT_MAX_NODES);
    R.drop_p  = a.dropout;
    R.seed    = a.seed;
    hmt_cb_weights(a.cb0_weighted, R.g.cbw);
    return true;
}

// The dropout mask pool: max(16M, 4 x the largest mask) F32 values, each 0 with
// probability p and 1 / (1 - p) otherwise, drawn once from the seed (so a resume
// rebuilds the same pool). Windows overlap between sites, but every window is
// an i.i.d. Bernoulli field and two sites never read the same value at the same
// position unless their offsets collide.
static bool hmt_drop_alloc(HmtRun & R, int64_t Lcap, std::string * err) {
    if (R.drop_p <= 0.0f) return true;
    const int64_t n = std::max<int64_t>((int64_t) 16 << 20, 4 * 3072 * Lcap);
    ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
    R.drop_ctx          = ggml_init(ip);
    R.g.drop_pool       = hmt_new(R.drop_ctx, GGML_TYPE_F32, n, 1, 1, "hmt.drop_pool");
    R.drop_buf          = ggml_backend_alloc_ctx_tensors(R.drop_ctx, R.m.bp.backend);
    if (!R.drop_buf) { *err = "the dropout mask pool does not fit"; return false; }
    std::mt19937_64    rng(hmt_mix(R.seed, 0, 0xD209));
    std::vector<float> v((size_t) n);
    const uint64_t     keep_below = (uint64_t) ((1.0 - (double) R.drop_p) * 18446744073709551616.0);
    const float        scale      = 1.0f / (1.0f - R.drop_p);
    for (auto & x : v) x = rng() < keep_below ? scale : 0.0f;
    ggml_backend_tensor_set(R.g.drop_pool, v.data(), 0, ggml_nbytes(R.g.drop_pool));
    return true;
}

static const char * hmt_attn_name(int attn) {
    return attn == 2 ? "flash-bf16" : attn == 1 ? "flash" : "exact";
}

// Memory admission and the op probe (contract §6.3). Builds every segment at
// the largest shapes the run will see, reserves it, and refuses a node the GPU
// would not run.
//
// Attention (--attn auto): the first of flash-bf16, flash, exact whose global
// block the device runs entirely (supports_op is capability-exact for
// FLASH_ATTN_TRAIN: BF16 needs CUDA sm_80+). Recompute (--keep-layers auto):
// the most kept global blocks out of 28, 24, ..., 0 whose graphs fit the free
// VRAM, then full recompute (-1). With full recompute and --local-chunk 0 the
// frame chunk of the local segments halves until the compute buffer fits.
static bool hmt_admit(HmtRun & R, int64_t Smax, int64_t Tmax, int64_t chunk_arg, const std::string & attn_arg, int keep_arg,
                      std::string * err) {
    ggml_backend_t gpu    = R.m.bp.backend;
    const bool     on_gpu = gpu != R.m.bp.cpu_backend;
    const int64_t  Lcap   = Smax + Tmax - 1;
    if (!hmt_sample_alloc(&R.sb, gpu, Smax, Tmax, err) || !hmt_drop_alloc(R, Lcap, err)) return false;

    std::vector<int> attns = attn_arg == "auto" ? std::vector<int>{ 2, 1, 0 }
                           : std::vector<int>{ attn_arg == "flash-bf16" ? 2 : attn_arg == "flash" ? 1 : 0 };
    if (!on_gpu && attn_arg == "auto") attns = { 1, 0 };  // the CPU has no BF16 kernels
    if (!hmt_ckpt_alloc(&R.ck, gpu, Lcap, Tmax, 2, false, err)) return false;
    std::string why;
    for (size_t i = 0; i < attns.size(); i++) {
        R.g.attn = attns[i];
        std::string e;
        if (!on_gpu || hmt_probe_graph(gpu, hmt_g_global_bwd(R.g, 1, 0), "global backward", &e)) break;
        if (i + 1 == attns.size()) { *err = e; return false; }
        why += (why.empty() ? "" : "; ") + std::string(hmt_attn_name(attns[i])) + ": " + e;
    }
    if (!why.empty()) fprintf(stderr, "[hm-train] attention fallback: %s\n", why.c_str());

    std::vector<int> keeps;
    if (keep_arg == -2) {
        for (int k = 28; k >= 0; k -= 4) keeps.push_back(k);
        keeps.push_back(-1);
    } else {
        keeps.push_back(keep_arg);
    }
    size_t need = 0, budget = 0;
    for (int keep : keeps) {
        R.keep = keep;
        if (R.g.sched) ggml_backend_sched_free(R.g.sched);
        R.g.sched = nullptr;
        // kept mode: C[0..k0] only, no local checkpoints
        if (!hmt_ckpt_alloc(&R.ck, gpu, Lcap, Tmax, keep < 0 ? 29 : 29 - keep, keep < 0, err)) return false;
        const DitGpuMem mem    = dit_gpu_mem_query(gpu);
        const size_t    margin = (size_t) 768 << 20;
        budget                 = mem.free > margin ? mem.free - margin : 0;
        const int64_t chunk0 = std::min(chunk_arg > 0 ? chunk_arg : Tmax, std::min(Tmax, hmt_local_frames_max(R.g)));
        for (int64_t chunk = chunk0;; chunk = (chunk + 1) / 2) {
            if (R.g.sched) ggml_backend_sched_free(R.g.sched);
            R.g.sched = backend_sched_new(R.m.bp, HMT_MAX_NODES);
            HmtTailOut tail;
            HmtHeadOut head;
            struct Seg { const char * what; std::function<ggml_cgraph *()> build; };
            std::vector<Seg> segs = {
                { "labels", [&] { return hmt_g_labels(R.g); } },
                { "embed", [&] { return hmt_g_embed(R.g); } },
                { "grad norm", [&] { return hmt_g_gradnorm(R.g); } },
            };
            if (keep < 28) {
                segs.push_back({ "global forward", [&] { return hmt_g_global_fwd(R.g, 0); } });
                segs.push_back({ "global backward", [&] { return hmt_g_global_bwd(R.g, 1, 0); } });
            }
            if (keep >= 0) {
                segs.push_back({ "kept", [&] { return hmt_g_kept_bwd(R.g, 28 - keep, 1.0f, &head, &tail, false); } });
            } else {
                segs.push_back({ "head forward", [&] { return hmt_g_head_fwd(R.g); } });
                segs.push_back({ "head backward", [&] { return hmt_g_head_bwd(R.g, 0, 1.0f, &head, false); } });
                segs.push_back({ "local forward", [&] { return hmt_g_local_fwd(R.g, 0, 0, chunk); } });
                segs.push_back({ "local backward", [&] { return hmt_g_local_bwd(R.g, 0, 0, chunk, 0); } });
                segs.push_back({ "tail backward", [&] { return hmt_g_tail_bwd(R.g, 0, chunk, 1.0f, 0, &tail, false); } });
            }
            bool        ok = true;
            std::string splits;  // graph splits per segment: 1 = the whole graph on one backend
            for (auto & s : segs) {
                ggml_cgraph * gf = s.build();
                if (on_gpu && !hmt_probe_graph(gpu, gf, s.what, err)) return false;
                if (!ggml_backend_sched_reserve(R.g.sched, gf)) { ok = false; break; }
                splits += std::string(splits.empty() ? "" : ", ") + s.what + " " + std::to_string(ggml_graph_n_nodes(gf)) +
                          " nodes/" + std::to_string(ggml_backend_sched_get_n_splits(R.g.sched)) + " split";
            }
            need = ok ? ggml_backend_sched_get_buffer_size(R.g.sched, gpu) : SIZE_MAX;
            if (ok && (need <= budget || !R.m.bp.has_gpu)) {
                R.chunk      = chunk < Tmax ? chunk : 0;
                R.peak_sched = need;
                fprintf(stderr, "[hm-train] admission: %s attention, keep-layers %d, compute %.0f MiB, persistent %.0f MiB, "
                                "local chunk %lld of %lld frames, device free %.0f MiB (%s)\n",
                        hmt_attn_name(R.g.attn), keep, need / 1048576.0, (hmt_alloc_bytes(R) - need) / 1048576.0,
                        (long long) chunk, (long long) Tmax, mem.free / 1048576.0, mem.source());
                fprintf(stderr, "[hm-train] placement: %s\n", splits.c_str());
                return true;
            }
            if (keep >= 0 || chunk_arg > 0 || chunk <= 16) break;
        }
    }
    *err = "does not fit: the training graphs need " + (need == SIZE_MAX ? std::string("more than the free") : std::to_string(need >> 20)) +
           " MiB of compute memory, " + std::to_string(budget >> 20) + " MiB is free beyond the weights (S " +
           std::to_string((long long) Smax) + ", T " + std::to_string((long long) Tmax) +
           "). Shorten the clips (--max-frames) or free VRAM";
    return false;
}

// ── Training loop ──────────────────────────────────────────────────────────

static bool hmt_exists(const std::string & p) {
    std::error_code ec;
    return std::filesystem::exists(std::filesystem::u8path(p), ec);
}

static int hmt_train(const HeartmulaTrainArgs & a_in) {
    namespace fs = std::filesystem;
    HeartmulaTrainArgs a = a_in;
    std::string        err;
    auto fail = [&](const std::string & msg) {
        fprintf(stderr, "heartmula-train: %s\n", msg.c_str());
        hmt_emit("\"event\":\"error\",\"message\":" + hmt_json_str(msg));
        return 1;
    };
    if (a.data_dir.empty() || a.out_dir.empty()) { fprintf(stderr, "heartmula-train: --data and --out are required\n"); return 2; }
    if (!hmt_resolve_models(&a, &err)) return fail(err);
    HmtData data;
    if (!hmt_data_load(a.data_dir, &data, &err)) return fail(err);

    const std::string ctl = (fs::u8path(a.out_dir) / "control").u8string();
    const std::string f_now = (fs::u8path(ctl) / "checkpoint-now").u8string(), f_cancel = (fs::u8path(ctl) / "cancel").u8string();
    const std::string f_state = (fs::u8path(a.out_dir) / HMT_STATE_FILE).u8string();
    std::error_code ec;
    fs::create_directories(fs::u8path(ctl), ec);
    fs::remove(fs::u8path(f_now), ec);
    fs::remove(fs::u8path(f_cancel), ec);
    fs::remove(fs::u8path(f_state + ".tmp"), ec);  // an interrupted write never replaced the real file
    if (!a.resume && hmt_exists(f_state)) return fail(f_state + " exists: pass --resume to continue that run, or use a new --out");

    HmtRun R;
    if (!hmt_setup(R, a, &err)) { hmt_run_free(&R); return fail(err); }
    int64_t Smax = 0, Tmax = 0;
    for (const auto & s : data.samples) {
        Smax = std::max<int64_t>(Smax, (int64_t) s.prompt.size());
        // Crop windows still train some songs whole, so they do not lower the ceiling; EOS adds a frame.
        Tmax = std::max<int64_t>(Tmax, (a.max_frames > 0 && a.crop_frames <= 0 ? std::min(s.frames, a.max_frames) : s.frames) + (a.eos ? 1 : 0));
    }
    if (!hmt_admit(R, Smax, Tmax, a.local_chunk, a.attn, a.keep, &err)) { hmt_run_free(&R); return fail(err); }

    HmtIdentity id;
    id.rank = a.rank, id.accum = a.accum, id.bf16 = a.weights == "bf16";
    id.alpha = a.alpha, id.lr = a.lr, id.wd = a.wd, id.beta1 = a.beta1, id.beta2 = a.beta2, id.eps = a.eps, id.clip = a.clip;
    id.dropout = a.dropout;
    id.mlp = a.mlp;
    id.eos = a.eos;
    id.crop_frames = a.crop_frames;
    id.cb0_weighted = a.cb0_weighted;
    id.seed = a.seed, id.data = data.fingerprint, id.max_frames = a.max_frames, id.base_fingerprint = R.m.base_fingerprint;
    id.weights = hmt_weights_id(a.global_path) + "|" + hmt_weights_id(a.local_path);
    const uint64_t ident = hmt_identity_hash(id);
    HmtCounters    cnt;
    if (a.resume) {
        if (!hmt_state_load(f_state, ident, &cnt, &R.lora, &R.opt, &err)) { hmt_run_free(&R); return fail(err); }
        fprintf(stderr, "[hm-train] resumed at step %lld\n", (long long) cnt.steps_done);
    }
    auto adapter_path = [&](int64_t step) {
        char b[32];
        snprintf(b, sizeof(b), "step-%06lld.gguf", (long long) step);
        return (fs::u8path(a.out_dir) / "adapters" / b).u8string();
    };

    {
        const DitGpuMem mem = dit_gpu_mem_query(R.m.bp.backend);
        hmt_emit("\"event\":\"start\",\"step\":" + std::to_string((long long) cnt.steps_done) + ",\"total\":" +
                 std::to_string((long long) a.steps) + ",\"samples\":" + std::to_string(data.samples.size()) +
                 ",\"rank\":" + std::to_string(a.rank) + ",\"alpha\":" + hmt_num(a.alpha) + ",\"lr\":" + hmt_num(a.lr) +
                 ",\"grad_accum\":" + std::to_string(a.accum) + ",\"smax\":" + std::to_string((long long) Smax) +
                 ",\"tmax\":" + std::to_string((long long) Tmax) + ",\"local_chunk\":" + std::to_string((long long) R.chunk) + ",\"keep_layers\":" + std::to_string(R.keep) + ",\"lora_dropout\":" + hmt_num(a.dropout) + ",\"lora_mlp\":" + (a.mlp ? "true" : "false") +
                 ",\"eos\":" + (a.eos ? "true" : "false") + ",\"crop_frames\":" + std::to_string((long long) a.crop_frames) +
                 ",\"cb0_weighted\":" + (a.cb0_weighted ? "true" : "false") +
                 ",\"params\":" + std::to_string((long long) hmt_lora_param_count(R.lora)) +
                 ",\"alloc_mib\":" + hmt_num(hmt_alloc_bytes(R) / 1048576.0) + ",\"device_free_mib\":" + hmt_num(mem.free / 1048576.0) +
                 ",\"weights\":" + hmt_json_str(a.weights) + ",\"attention\":\"" + hmt_attn_name(R.g.attn) + "\",\"bwd\":\"mm\",\"global\":" +
                 hmt_json_str(a.global_path) + ",\"local\":" + hmt_json_str(a.local_path));
    }
    fprintf(stderr, "[hm-train] %zu samples, %lld LoRA parameters (rank %d, alpha %g), %s weights, %s attention "
                    "(%s), keep-layers %d, backward arm mm, AdamW lr %g wd %g, grad-accum %d, %lld steps\n",
            data.samples.size(), (long long) hmt_lora_param_count(R.lora), a.rank, (double) a.alpha, a.weights.c_str(),
            hmt_attn_name(R.g.attn), R.g.attn == 2 ? "bf16" : R.g.attn == 1 ? "tf32" : "f32 softmax", R.keep, (double) a.lr, (double) a.wd, a.accum, (long long) a.steps);

    auto checkpoint = [&](const char * reason) -> bool {
        cnt.opt_step = R.opt.opt_step;
        cnt.opt_iter = R.opt.opt_iter;
        const std::string ad = adapter_path(cnt.steps_done);
        if (!hmt_state_save(f_state, ident, cnt, R.lora, R.opt, &err) ||
            !hmt_export(ad, a.name, cnt.steps_done, R.lora, R.m, &err, a.trigger)) return false;
        hmt_emit("\"event\":\"checkpoint\",\"step\":" + std::to_string((long long) cnt.steps_done) + ",\"state\":" +
                 hmt_json_str(f_state) + ",\"adapter\":" + hmt_json_str(ad) + ",\"reason\":\"" + reason + "\"");
        return true;
    };

    std::vector<int32_t> codes;
    int consecutive_bad = 0;
    DitGpuMem mem;
    auto      mem_at = std::chrono::steady_clock::now();
    while (cnt.steps_done < a.steps) {
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<HmtPick> picks = hmt_schedule(data, a.accum, a.max_frames, a.seed, cnt.steps_done, a.crop_frames, a.eos,
                                                        heartmula_global_config().n_ctx);
        const float wce = 1.0f / (8.0f * (float) picks.size());
        double ce_sum[8] = {}, loss = 0.0, loss_mean = 0.0;  // loss: the objective (cbw-weighted)
        bool   finite = true;
        for (const HmtPick & p : picks) {
            R.g.drop_key = hmt_mix(a.seed, (uint64_t) cnt.steps_done, (uint64_t) (&p - picks.data()));
            if (hmt_exists(f_cancel)) {
                fs::remove(fs::u8path(f_cancel), ec);
                fprintf(stderr, "[hm-train] cancelled at step %lld; the last checkpoint stands\n", (long long) cnt.steps_done);
                hmt_emit("\"event\":\"cancelled\",\"step\":" + std::to_string((long long) cnt.steps_done));
                hmt_run_free(&R);
                return 4;
            }
            const HmtSample & s = data.samples[(size_t) p.sample];
            double ce[8];
            if (!hmt_read_codes(s, p.w0, p.T, &codes, &err)) { hmt_run_free(&R); return fail(err); }
            if (p.eos) codes.insert(codes.end(), 8, HMT_EOS_CODE);
            if (!hmt_micro_step(R, s.prompt.data(), (int64_t) s.prompt.size(), s.muq_idx, codes.data(), p.T + (p.eos ? 1 : 0), wce, ce, &err)) {
                hmt_run_free(&R);
                return fail(err);
            }
            double l = 0.0, lm = 0.0;
            for (int k = 0; k < 8; k++) {
                l += ce[k] * R.g.cbw[k] / 8.0;
                lm += ce[k] / 8.0;
                ce_sum[k] += ce[k] / (double) picks.size();
            }
            finite = finite && std::isfinite(l);
            loss += l / (double) picks.size();
            loss_mean += lm / (double) picks.size();
        }
        double gn2 = 0.0;
        if (!hmt_grad_norm2(R, &gn2, &err)) { hmt_run_free(&R); return fail(err); }
        LmStepStats st;
        if (!finite || !std::isfinite(gn2)) {
            lm_optim_zero_grad(&R.opt);
            cnt.skipped++;
            hmt_emit("\"event\":\"skip\",\"step\":" + std::to_string((long long) cnt.steps_done + 1) + ",\"reason\":\"" +
                     (finite ? "non-finite gradient" : "non-finite loss") + "\"");
            if (++consecutive_bad >= 3) { hmt_run_free(&R); return fail("three optimizer steps in a row had a non-finite loss or gradient"); }
        } else {
            consecutive_bad = 0;
            if (!hmt_prof_host(R, "host adamw", [&] { return lm_optim_step(&R.opt, R.g.sched, &st); })) { hmt_run_free(&R); return fail("optimizer step failed"); }
        }
        cnt.steps_done++;
        const double    ms  = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        // An NVML query costs ~9 ms, so the step event's device_used_mib refreshes at most every 5 s.
        if (cnt.steps_done == 1 || std::chrono::steady_clock::now() - mem_at > std::chrono::seconds(5)) {
            mem    = hmt_prof_host(R, "host nvml", [&] { return dit_gpu_mem_query(R.m.bp.backend); });
            mem_at = std::chrono::steady_clock::now();
        }
        std::string     ces;
        for (int k = 0; k < 8; k++) ces += (k ? "," : "") + hmt_num(ce_sum[k]);
        hmt_emit("\"event\":\"step\",\"step\":" + std::to_string((long long) cnt.steps_done) + ",\"total\":" +
                 std::to_string((long long) a.steps) + ",\"loss\":" + hmt_num(loss) +
                 (a.cb0_weighted ? ",\"loss_mean\":" + hmt_num(loss_mean) : std::string()) + ",\"ce\":[" + ces + "],\"lr\":" +
                 hmt_num(a.lr) + ",\"grad_norm\":" + hmt_num(std::sqrt(gn2)) + ",\"ms_step\":" + hmt_num(ms) +
                 ",\"ms_micro\":" + hmt_num(ms / (double) picks.size()) + ",\"micro\":" + std::to_string(picks.size()) +
                 ",\"peak_alloc_mib\":" + hmt_num(hmt_alloc_bytes(R) / 1048576.0) + ",\"device_used_mib\":" + hmt_num(mem.used / 1048576.0));
        fprintf(stderr, "[hm-train] step %lld/%lld loss %.4f grad-norm %.4g %.0f ms (%zu micro)\n", (long long) cnt.steps_done,
                (long long) a.steps, loss, std::sqrt(gn2), ms, picks.size());
        hmt_prof_dump(R, "step");

        const bool final = cnt.steps_done >= a.steps;
        const bool now   = hmt_exists(f_now);
        const bool stop  = a.stop_after > 0 && cnt.steps_done >= a.stop_after && !final;
        const bool every = a.save_every > 0 && cnt.steps_done % a.save_every == 0;
        if (final || now || stop || every) {
            if (!checkpoint(final ? "final" : now ? "now" : stop ? "stop" : "every")) { hmt_run_free(&R); return fail(err); }
        }
        if (now) fs::remove(fs::u8path(f_now), ec);
        if (!final && (now || stop)) {
            fprintf(stderr, "[hm-train] paused at step %lld (resume with --resume)\n", (long long) cnt.steps_done);
            hmt_run_free(&R);
            return 3;
        }
    }
    // Reached on a finished run, and on --resume of one that already was.
    if (!hmt_exists(adapter_path(cnt.steps_done)) && !checkpoint("final")) { hmt_run_free(&R); return fail(err); }
    hmt_emit("\"event\":\"done\",\"step\":" + std::to_string((long long) cnt.steps_done) + ",\"adapter\":" +
             hmt_json_str(adapter_path(cnt.steps_done)));
    hmt_run_free(&R);
    return 0;
}

// ── Held-out scoring (--eval) ──────────────────────────────────────────────

// One sample's eight mean CEs, forward only, from frame-major `codes` [T][8].
static bool hmt_eval_sample(HmtRun & R, const HmtSample & s, const int32_t * codes, int64_t T, double ce[8], std::string * err) {
    HmtGraphCtx & g = R.g;
    if (!hmt_sample_alloc(&R.sb, R.m.bp.backend, (int64_t) s.prompt.size(), T, err)) return false;
    hmt_sample_upload(&R.sb, s.prompt.data(), s.muq_idx, codes);
    ggml_backend_buffer_clear(R.sb.lbuf, 0);
    auto run = [&](ggml_cgraph * gf) { return hmt_compute(R, gf, err); };
    if (!run(hmt_g_labels(g)) || !run(hmt_g_embed(g))) return false;
    for (int l = 0; l < 28; l++) if (!run(hmt_g_global_fwd(g, l))) return false;
    HmtHeadOut head;
    if (!run(hmt_g_head_eval(g, &head))) return false;
    ce[0] = hmt_read1(head.ce0);
    for (int l = 0; l < 3; l++)
        for (int64_t f0 = 0; f0 < T; f0 += R.chunk) if (!run(hmt_g_local_fwd(g, l, f0, std::min(R.chunk, T - f0)))) return false;
    for (int k = 1; k < 8; k++) ce[k] = 0.0;
    for (int64_t f0 = 0; f0 < T; f0 += R.chunk) {
        const int64_t Tc = std::min(R.chunk, T - f0);
        HmtTailOut    out;
        if (!run(hmt_g_tail_eval(g, f0, Tc, &out))) return false;
        for (int k = 1; k < 8; k++) ce[k] += (double) hmt_read1(out.ce[k]) * (double) Tc / (double) T;
    }
    return true;
}

static int hmt_eval(const HeartmulaTrainArgs & a_in) {
    HeartmulaTrainArgs a = a_in;
    std::string        err;
    auto fail = [&](const std::string & msg) {
        fprintf(stderr, "heartmula-train --eval: %s\n", msg.c_str());
        hmt_emit("\"event\":\"error\",\"message\":" + hmt_json_str(msg));
        return 1;
    };
    if (a.data_dir.empty()) { fprintf(stderr, "heartmula-train --eval: --data is required\n"); return 2; }
    if (!hmt_resolve_models(&a, &err)) return fail(err);
    HmtData data;
    if (!hmt_data_load(a.data_dir, &data, &err)) return fail(err);
    std::vector<const HmtSample *> picked;
    for (size_t p = 0; p < a.eval_samples.size();) {
        const size_t      c  = std::min(a.eval_samples.find(',', p), a.eval_samples.size());
        const std::string id = a.eval_samples.substr(p, c - p);
        p = c + 1;
        if (id.empty()) continue;
        const HmtSample * hit = nullptr;
        for (const auto & s : data.samples) if (s.id == id || s.key == id) hit = &s;
        if (!hit) return fail("no sample '" + id + "' in " + a.data_dir);
        picked.push_back(hit);
    }
    if (a.eval_samples.empty()) for (const auto & s : data.samples) picked.push_back(&s);
    if (picked.empty()) return fail("--eval-samples names no sample");

    HmtRun R;
    if (!hmt_model_load(&R.m, a.global_path.c_str(), a.local_path.c_str(), a.weights == "bf16", &err)) { hmt_run_free(&R); return fail(err); }
    // The end frame follows training's rule: only where it still fits the context.
    auto eos_of = [&](const HmtSample & s) {
        return a.eos && hmt_eos_fits((int64_t) s.prompt.size(), s.frames, heartmula_global_config().n_ctx);
    };
    int64_t Smax = 0, Tmax = 0;
    for (const HmtSample * s : picked) {
        Smax = std::max<int64_t>(Smax, (int64_t) s->prompt.size());
        Tmax = std::max<int64_t>(Tmax, s->frames + (eos_of(*s) ? 1 : 0));
    }
    ggml_backend_t gpu = R.m.bp.backend;
    R.g.m     = &R.m;
    R.g.ck    = &R.ck;
    R.g.sb    = &R.sb;
    R.g.sched = backend_sched_new(R.m.bp, HMT_MAX_NODES);
    R.chunk   = std::min(Tmax, a.local_chunk > 0 ? a.local_chunk : (int64_t) 1024);  // fixed, so the sums reproduce
    if (!hmt_sample_alloc(&R.sb, gpu, Smax, Tmax, &err) || !hmt_ckpt_alloc(&R.ck, gpu, Smax + Tmax - 1, Tmax, 29, true, &err)) {
        hmt_run_free(&R);
        return fail(err);
    }
    // Attention picked as training picks it, so a score compares with the run's loss.
    std::vector<int> attns = a.attn == "auto" ? std::vector<int>{ 2, 1, 0 }
                           : std::vector<int>{ a.attn == "flash-bf16" ? 2 : a.attn == "flash" ? 1 : 0 };
    if (gpu == R.m.bp.cpu_backend && a.attn == "auto") attns = { 1, 0 };
    for (size_t i = 0; i < attns.size(); i++) {
        R.g.attn = attns[i];
        std::string e;
        if (gpu == R.m.bp.cpu_backend || hmt_probe_graph(gpu, hmt_g_global_fwd(R.g, 0), "global forward", &e)) break;
        if (i + 1 == attns.size()) { hmt_run_free(&R); return fail(e); }
    }
    fprintf(stderr, "[hm-eval] %zu samples, %zu models, %s attention, local chunk %lld, %s\n", picked.size(), a.eval.size(),
            hmt_attn_name(R.g.attn), (long long) R.chunk, a.eos ? "with the end frame" : "no end frame");

    float cbw[8];
    hmt_cb_weights(true, cbw);
    std::vector<int32_t> codes;
    for (const std::string & model : a.eval) {
        HeartMuLaAdapter          ad;
        std::vector<HmtLayerLora> lora;
        R.g.lora  = nullptr;
        R.g.scale = 1.0f;
        if (model != "base") {
            if (!heartmula_adapter_load(&ad, R.m, model.c_str(), &err)) { hmt_run_free(&R); return fail(err); }
            lora.assign(ad.layers.size(), HmtLayerLora{});
            for (size_t l = 0; l < ad.layers.size(); l++)
                for (int s = 0; s < HMT_SITES; s++) {
                    const QwLoraPair & p = ad.layers[l].p[heartmula_adapter_detail::proj_slot(s)];
                    lora[l].s[s].A = p.A, lora[l].s[s].B = p.B;
                }
            R.g.lora  = &lora;
            R.g.scale = ad.alpha / (float) ad.rank;
        }
        double  sum[8] = {};
        int64_t frames = 0;
        for (const HmtSample * s : picked) {
            double ce[8];
            if (!hmt_read_codes(*s, 0, s->frames, &codes, &err)) { heartmula_adapter_free(&ad); hmt_run_free(&R); return fail(err); }
            const bool eos = eos_of(*s);
            if (eos) codes.insert(codes.end(), 8, HMT_EOS_CODE);
            const int64_t T = s->frames + (eos ? 1 : 0);
            if (!hmt_eval_sample(R, *s, codes.data(), T, ce, &err)) { heartmula_adapter_free(&ad); hmt_run_free(&R); return fail(err); }
            std::string ces;
            double      loss = 0.0;
            for (int k = 0; k < 8; k++) {
                ces += (k ? "," : "") + hmt_num(ce[k]);
                loss += ce[k] / 8.0;
                sum[k] += ce[k] * (double) T;
            }
            frames += T;
            hmt_emit("\"event\":\"eval_sample\",\"model\":" + hmt_json_str(model) + ",\"sample\":" + hmt_json_str(s->id) +
                     ",\"frames\":" + std::to_string((long long) T) + ",\"eos\":" + (eos ? "true" : "false") +
                     ",\"loss\":" + hmt_num(loss) + ",\"ce\":[" + ces + "]");
        }
        std::string ces;
        double      loss = 0.0, loss_w = 0.0;
        for (int k = 0; k < 8; k++) {
            const double c = sum[k] / (double) frames;
            ces += (k ? "," : "") + hmt_num(c);
            loss += c / 8.0;
            loss_w += c * cbw[k] / 8.0;
        }
        hmt_emit("\"event\":\"eval\",\"model\":" + hmt_json_str(model) + ",\"samples\":" + std::to_string(picked.size()) +
                 ",\"frames\":" + std::to_string((long long) frames) + ",\"loss\":" + hmt_num(loss) + ",\"loss_cb0w\":" +
                 hmt_num(loss_w) + ",\"ce\":[" + ces + "]");
        fprintf(stderr, "[hm-eval] %s: loss %.4f over %lld frames\n", model.c_str(), loss, (long long) frames);
        heartmula_adapter_free(&ad);
    }
    hmt_run_free(&R);
    return 0;
}

// ── Selftest (contract §5) ─────────────────────────────────────────────────

struct HmtCmp {
    double rel = 0, cos = 0, max_abs = 0;
};

static HmtCmp hmt_cmp(const float * got, const float * want, size_t n) {
    double num = 0, den = 0, dot = 0, gg = 0;
    HmtCmp c;
    for (size_t i = 0; i < n; i++) {
        const double e = (double) got[i] - want[i];
        num += e * e;
        den += (double) want[i] * want[i];
        dot += (double) got[i] * want[i];
        gg += (double) got[i] * got[i];
        c.max_abs = std::max(c.max_abs, std::fabs(e));
    }
    c.rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    c.cos = gg > 0 && den > 0 ? dot / std::sqrt(gg * den) : (gg == 0 && den == 0 ? 1.0 : 0.0);
    return c;
}

// Stacked fixture name of slot j of a layer: A_q, B_q, A_k, ...
static std::string hmt_fx_name(int j) {
    static const char * p[4] = { "q", "k", "v", "o" };
    return std::string(j % 2 ? "B_" : "A_") + p[j / 2];
}

// Set every factor (or, with `acc`, every gradient accumulator) from stacked [28, ...] fixtures.
static bool hmt_fx_set(HmtRun & R, const std::string & dir, const std::string & prefix_a, const std::string & prefix_b,
                       const std::vector<ggml_tensor *> & dst, std::string * err) {
    for (int j = 0; j < 8; j++) {
        HmtNpy a;
        const std::string name = hmt_fx_name(j);
        if (!hmt_npy(dir + "/" + (j % 2 ? prefix_b : prefix_a) + "/" + name + ".npy", &a, err)) return false;
        const size_t per = a.f.size() / 28;
        for (int l = 0; l < 28; l++) {
            ggml_tensor * t = dst[(size_t) (l * 8 + j)];
            if ((size_t) ggml_nelements(t) != per) { *err = name + ": fixture shape differs from the factor"; return false; }
            ggml_backend_tensor_set(t, a.f.data() + (size_t) l * per, 0, per * sizeof(float));
        }
    }
    (void) R;
    return true;
}

static bool hmt_fx_cmp(const std::string & dir, const std::string & prefix, const std::vector<ggml_tensor *> & src,
                       std::vector<std::pair<std::string, HmtCmp>> * out, std::string * err) {
    for (int j = 0; j < 8; j++) {
        HmtNpy w;
        const std::string name = hmt_fx_name(j);
        if (!hmt_npy(dir + "/" + prefix + "/" + name + ".npy", &w, err)) return false;
        std::vector<float> got(w.f.size());
        const size_t per = w.f.size() / 28;
        for (int l = 0; l < 28; l++) ggml_backend_tensor_get(src[(size_t) (l * 8 + j)], got.data() + (size_t) l * per, 0, per * sizeof(float));
        out->push_back({ name, hmt_cmp(got.data(), w.f.data(), got.size()) });
    }
    return true;
}

static int hmt_selftest(const HeartmulaTrainArgs & a_in) {
    HeartmulaTrainArgs a = a_in;
    std::string        err, dir = a.selftest;
    int                failures = 0;
    auto check = [&](bool ok, const char * fmt, auto... v) {
        fprintf(stderr, "[hm-selftest] %s ", ok ? "PASS" : "FAIL");
        fprintf(stderr, fmt, v...);
        fprintf(stderr, "\n");
        failures += ok ? 0 : 1;
    };
    if (!hmt_resolve_models(&a, &err)) { fprintf(stderr, "heartmula-train: %s\n", err.c_str()); return 2; }
    a.accum = 2;
    HmtNpy prompt_n, muq_n, codes_n, crops_n;
    if (!hmt_npy(dir + "/input/prompt_ids.npy", &prompt_n, &err) || !hmt_npy(dir + "/input/muq_idx.npy", &muq_n, &err) ||
        !hmt_npy(dir + "/input/codes.npy", &codes_n, &err) || !hmt_npy(dir + "/input/crops.npy", &crops_n, &err)) {
        fprintf(stderr, "heartmula-train: %s\n", err.c_str());
        return 1;
    }
    std::vector<int32_t> prompt(prompt_n.f.begin(), prompt_n.f.end()), codes(codes_n.f.begin(), codes_n.f.end());
    const int64_t S = (int64_t) prompt.size(), muq = (int64_t) muq_n.f[0], Tall = codes_n.shape[0];
    const int64_t crop[2] = { (int64_t) crops_n.f[0], (int64_t) crops_n.f[2] }, CT = (int64_t) (crops_n.f[1] - crops_n.f[0]);

    HmtRun R;
    if (!hmt_setup(R, a, &err) || !hmt_admit(R, S, Tall, a.local_chunk, a.attn, a.keep, &err)) {
        fprintf(stderr, "heartmula-train: %s\n", err.c_str());
        hmt_run_free(&R);
        return 1;
    }
    fprintf(stderr, "[hm-selftest] %s weights, S %lld, crop %lld frames at %lld and %lld, local chunk %lld\n", a.weights.c_str(),
            (long long) S, (long long) CT, (long long) crop[0], (long long) crop[1], (long long) R.chunk);
    const std::vector<ggml_tensor *> & P = R.lora.params;
    auto zero_b = [&]() {
        for (size_t j = 1; j < P.size(); j += 2) ggml_backend_tensor_memset(P[j], 0, 0, ggml_nbytes(P[j]));
    };
    auto set_state = [&](const char * b_prefix) {
        if (!hmt_fx_set(R, dir, "lora/init", b_prefix, P, &err)) return false;
        if (!strcmp(b_prefix, "lora/init")) zero_b();
        return true;
    };
    auto grads_report = [&](const char * prefix, bool init) {
        std::vector<std::pair<std::string, HmtCmp>> cmp;
        if (!hmt_fx_cmp(dir, prefix, R.opt.acc, &cmp, &err)) return false;
        for (auto & [name, c] : cmp) {
            const bool zero_a = init && name[0] == 'A';
            if (zero_a) {
                std::vector<float> g;
                double             mx = 0;
                for (int l = 0; l < 28; l++) {
                    ggml_tensor * t = R.opt.acc[(size_t) (l * 8 + (name[2] == 'q' ? 0 : name[2] == 'k' ? 2 : name[2] == 'v' ? 4 : 6))];
                    g.resize((size_t) ggml_nelements(t));
                    ggml_backend_tensor_get(t, g.data(), 0, ggml_nbytes(t));
                    for (float x : g) mx = std::max(mx, (double) std::fabs(x));
                }
                check(mx == 0.0, "%s grad %s: exactly zero at init (max |g| %.3g)", prefix, name.c_str(), mx);
            } else {
                check(c.rel <= 1e-1 && c.cos >= 0.995, "%s grad %s: rel L2 %.4g (<= 0.1), cos %.6f (>= 0.995)", prefix,
                      name.c_str(), c.rel, c.cos);
            }
        }
        return true;
    };
    const int32_t * crop1 = codes.data() + crop[0] * 8;
    double          ce[8];

    // §5.1 forward, base state (B = 0) then perturbed, crop 1, with taps.
    {
        if (!set_state("lora/init")) goto io_fail;
        if (!hmt_micro_step(R, prompt.data(), S, muq, crop1, CT, 1.0f / 8.0f, ce, &err)) goto run_fail;
        HmtNpy base;
        if (!hmt_npy(dir + "/fp32/forward_base/ce.npy", &base, &err)) goto io_fail;
        double mx = 0;
        for (int k = 0; k < 8; k++) mx = std::max(mx, std::fabs(ce[k] - base.f[(size_t) k]));
        check(mx <= 1.5e-2, "base forward: per-codebook CE max |diff| %.3g (<= 1.5e-2)", mx);
        if (!grads_report("fp32/init/grad", true)) goto io_fail;
        lm_optim_zero_grad(&R.opt);

        if (!set_state("lora/perturbed")) goto io_fail;
        R.taps = true;
        if (!hmt_micro_step(R, prompt.data(), S, muq, crop1, CT, 1.0f / 8.0f, ce, &err)) goto run_fail;
        R.taps = false;
        HmtNpy fce, floss, t_emb, t_last, t_dec, t_logits, t_muq;
        if (!hmt_npy(dir + "/fp32/forward/ce.npy", &fce, &err) || !hmt_npy(dir + "/fp32/forward/loss.npy", &floss, &err) ||
            !hmt_npy(dir + "/fp32/taps/input_embedding.npy", &t_emb, &err) || !hmt_npy(dir + "/fp32/taps/backbone_last.npy", &t_last, &err) ||
            !hmt_npy(dir + "/fp32/taps/decoder_out.npy", &t_dec, &err) || !hmt_npy(dir + "/fp32/taps/logits.npy", &t_logits, &err) ||
            !hmt_npy(dir + "/fp32/taps/muq_slot.npy", &t_muq, &err)) goto io_fail;
        double cmx = 0, loss = 0;
        for (int k = 0; k < 8; k++) { cmx = std::max(cmx, std::fabs(ce[k] - fce.f[(size_t) k])); loss += ce[k] / 8.0; }
        check(cmx <= 1.5e-2, "perturbed forward: per-codebook CE max |diff| %.3g (<= 1.5e-2); CE %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
              cmx, ce[0], ce[1], ce[2], ce[3], ce[4], ce[5], ce[6], ce[7]);
        check(std::fabs(loss - floss.f[0]) <= 5e-3, "perturbed forward: loss %.6f vs %.6f (|diff| <= 5e-3)", loss, (double) floss.f[0]);
        std::vector<float> emb((size_t) (3072 * (S + CT - 1)));
        ggml_backend_tensor_get(R.ck.C[0], emb.data(), 0, emb.size() * sizeof(float));
        HmtCmp c = hmt_cmp(emb.data(), t_emb.f.data(), emb.size());
        check(c.rel <= 5e-3, "input_embedding rel L2 %.3g (<= 5e-3)", c.rel);
        c = hmt_cmp(emb.data() + (size_t) muq * 3072, t_muq.f.data(), 3072);
        check(c.rel <= 1e-3, "muq_slot rel L2 %.3g (<= 1e-3)", c.rel);
        c = hmt_cmp(R.tap_last.data(), t_last.f.data(), R.tap_last.size());
        check(c.rel <= 3e-2, "backbone_last rel L2 %.3g (<= 3e-2)", c.rel);
        c = hmt_cmp(R.tap_dec.data(), t_dec.f.data(), R.tap_dec.size());
        check(c.rel <= 2e-2, "decoder_out rel L2 %.3g (<= 2e-2)", c.rel);
        std::vector<float> lg(t_logits.f.size());
        const int64_t V = HEARTMULA_AUDIO_VOCAB;
        for (int64_t t = 0; t < CT; t++)
            for (int k = 0; k < 8; k++)
                std::copy(R.tap_logits[k].begin() + t * V, R.tap_logits[k].begin() + (t + 1) * V, lg.begin() + (t * 8 + k) * V);
        c = hmt_cmp(lg.data(), t_logits.f.data(), lg.size());
        check(c.rel <= 2e-2, "logits rel L2 %.3g (<= 2e-2)", c.rel);
        if (!grads_report("fp32/perturbed/grad", false)) goto io_fail;
        lm_optim_zero_grad(&R.opt);
    }

    // §5.3 optimizer replay: fixture gradients through our AdamW, no model.
    {
        if (!set_state("lora/perturbed")) goto io_fail;
        ggml_backend_buffer_clear(R.opt.buf_mom, 0);
        R.opt.opt_step = R.opt.opt_iter = 0;
        LmStepStats st;
        for (int s = 1; s <= 3; s++) {
            if (!hmt_fx_set(R, dir, "fp32/adamw/step" + std::to_string(s) + "/grad", "fp32/adamw/step" + std::to_string(s) + "/grad",
                            R.opt.acc, &err)) goto io_fail;
            if (!lm_optim_step(&R.opt, R.g.sched, &st)) { err = "optimizer step failed"; goto run_fail; }
            if (s == 1 || s == 3) {
                std::vector<std::pair<std::string, HmtCmp>> cmp;
                if (!hmt_fx_cmp(dir, "fp32/adamw/step" + std::to_string(s), P, &cmp, &err)) goto io_fail;
                double mx = 0;
                for (auto & x : cmp) mx = std::max(mx, x.second.rel);
                check(mx <= 1e-5, "optimizer replay: factors after step %d, max rel L2 %.3g (<= 1e-5)", s, mx);
            }
        }
        for (const char * mom : { "exp_avg", "exp_avg_sq" }) {
            std::vector<std::pair<std::string, HmtCmp>> cmp;
            if (!hmt_fx_cmp(dir, std::string("fp32/adamw/step3/") + mom, mom[7] ? R.opt.mom_v : R.opt.mom_m, &cmp, &err)) goto io_fail;
            double mx = 0;
            for (auto & x : cmp) mx = std::max(mx, x.second.rel);
            // Contract §5.3: ggml's AdamW kernel forms 1 - beta2 in F32 (0.0010000467) where torch
            // passes 0.001, so v alone differs by 4.7e-5 relative per increment.
            const double bound = mom[7] ? 3e-5 : 1e-5;
            check(mx <= bound, "optimizer replay: %s after step 3, max rel L2 %.3g (<= %.0e)", mom, mx, bound);
        }
    }

    // §5.3 end to end: three steps of two micro-batches. §5.4 resume: 2 steps, save, reload, 1 step.
    {
        HmtNpy micro;
        if (!hmt_npy(dir + "/fp32/adamw/micro_losses.npy", &micro, &err)) goto io_fail;
        auto run_steps = [&](int from, int to, std::vector<double> * losses) {
            for (int s = from; s < to; s++) {
                for (int b = 0; b < 2; b++) {
                    if (!hmt_micro_step(R, prompt.data(), S, muq, codes.data() + crop[b] * 8, CT, 1.0f / 16.0f, ce, &err)) return false;
                    double l = 0;
                    for (int k = 0; k < 8; k++) l += ce[k] / 8.0;
                    if (losses) losses->push_back(l);
                }
                LmStepStats st;
                if (!lm_optim_step(&R.opt, R.g.sched, &st)) { err = "optimizer step failed"; return false; }
            }
            return true;
        };
        auto reset = [&]() {
            ggml_backend_buffer_clear(R.opt.buf_mom, 0);
            lm_optim_zero_grad(&R.opt);
            R.opt.opt_step = R.opt.opt_iter = 0;
            return set_state("lora/perturbed");
        };
        auto snapshot = [&](std::vector<float> * out) {
            out->clear();
            for (auto * t : hmt_state_tensors(R.lora, R.opt)) {
                const size_t n = out->size();
                out->resize(n + (size_t) ggml_nelements(t));
                ggml_backend_tensor_get(t, out->data() + n, 0, ggml_nbytes(t));
            }
        };
        std::vector<double> losses;
        std::vector<float>  start, full, resumed;
        if (!reset()) goto io_fail;
        for (auto * p : P) {
            const size_t n = start.size();
            start.resize(n + (size_t) ggml_nelements(p));
            ggml_backend_tensor_get(p, start.data() + n, 0, ggml_nbytes(p));
        }
        if (!run_steps(0, 3, &losses)) goto run_fail;
        double mx = 0;
        for (int i = 0; i < 6; i++) mx = std::max(mx, std::fabs(losses[(size_t) i] - micro.f[(size_t) i]));
        check(mx <= 1e-2, "end to end: micro-losses %.4f %.4f | %.4f %.4f | %.4f %.4f, max |diff| %.3g (<= 1e-2)", losses[0],
              losses[1], losses[2], losses[3], losses[4], losses[5], mx);
        {
            // Update = step-3 factors minus start, against the FP32 update.
            size_t off = 0;
            std::vector<std::vector<float>> got(8), want(8);
            for (int l = 0; l < 28; l++) {
                for (int j = 0; j < 8; j++) {
                    ggml_tensor *      t = P[(size_t) (l * 8 + j)];
                    std::vector<float> now((size_t) ggml_nelements(t));
                    ggml_backend_tensor_get(t, now.data(), 0, ggml_nbytes(t));
                    for (size_t i = 0; i < now.size(); i++) got[(size_t) j].push_back(now[i] - start[off + i]);
                    off += now.size();
                }
            }
            for (int j = 0; j < 8; j++) {
                HmtNpy w, s0;
                const std::string n = hmt_fx_name(j);
                if (!hmt_npy(dir + "/fp32/adamw/step3/" + n + ".npy", &w, &err) ||
                    !hmt_npy(dir + "/" + (j % 2 ? "lora/perturbed/" : "lora/init/") + n + ".npy", &s0, &err)) goto io_fail;
                want[(size_t) j].resize(w.f.size());
                for (size_t i = 0; i < w.f.size(); i++) want[(size_t) j][i] = w.f[i] - s0.f[i];
                const HmtCmp c = hmt_cmp(got[(size_t) j].data(), want[(size_t) j].data(), want[(size_t) j].size());
                check(c.cos >= 0.98 && c.rel <= 0.25, "end to end: update %s cos %.5f (>= 0.98), rel L2 %.4f (<= 0.25)", n.c_str(), c.cos, c.rel);
            }
        }
        snapshot(&full);
        const std::string tmp_state = a.out_dir.empty() ? std::string("hm_selftest_state.bin")
                                                         : (std::filesystem::u8path(a.out_dir) / "hm_selftest_state.bin").u8string();
        HmtCounters cnt;
        if (!reset() || !run_steps(0, 2, nullptr)) goto run_fail;
        cnt.steps_done = 2, cnt.opt_step = R.opt.opt_step, cnt.opt_iter = R.opt.opt_iter;
        if (!hmt_state_save(tmp_state, 42, cnt, R.lora, R.opt, &err)) goto io_fail;
        if (!reset()) goto io_fail;  // scramble: back to the start state with zero moments
        if (!hmt_state_load(tmp_state, 42, &cnt, &R.lora, &R.opt, &err)) goto io_fail;
        hs_remove(tmp_state);
        if (!run_steps(2, 3, nullptr)) goto run_fail;
        snapshot(&resumed);
        size_t diff = 0;
        for (size_t i = 0; i < full.size(); i++) diff += full[i] != resumed[i];
        check(diff == 0 && full.size() == resumed.size(), "resume: factors and moments after 2 + save/load + 1 steps vs 3 steps, %zu of %zu values differ (bit-identical)",
              diff, full.size());
    }

    // §7 numbers on clip A whole: median of 3 timed micro-steps after a warm-up.
    {
        std::vector<double> ms;
        for (int i = 0; i < 4; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            if (!hmt_micro_step(R, prompt.data(), S, muq, codes.data(), Tall, 1.0f / 32.0f, ce, &err)) goto run_fail;
            if (i) ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            hmt_prof_dump(R, "micro-batch");
        }
        lm_optim_zero_grad(&R.opt);
        std::sort(ms.begin(), ms.end());
        const DitGpuMem mem = dit_gpu_mem_query(R.m.bp.backend);
        fprintf(stderr, "[hm-selftest] BENCH clip A S %lld T %lld L %lld: %.0f ms per micro-batch (median of 3; Python 343), "
                        "peak allocated %.2f GiB (Python 17.81), device used %.2f GiB (%s), local chunk %lld\n",
                (long long) S, (long long) Tall, (long long) (S + Tall - 1), ms[1], hmt_alloc_bytes(R) / 1073741824.0,
                mem.used / 1073741824.0, mem.source(), (long long) R.chunk);
    }
    hmt_run_free(&R);
    fprintf(stderr, "[hm-selftest] %s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
io_fail:
run_fail:
    fprintf(stderr, "heartmula-train selftest: %s\n", err.c_str());
    hmt_run_free(&R);
    return 1;
}

static int heartmula_train_run(const HeartmulaTrainArgs & a) {
    if (a.rank < 1 || a.accum < 1 || a.steps < 1 || !(a.lr > 0) || !(a.alpha > 0) || a.wd < 0 || a.clip < 0 ||
        a.keep < -2 || a.keep > 28 || a.crop_frames < 0 || !(a.dropout >= 0.0f && a.dropout < 1.0f) || (a.attn != "auto" && a.attn != "exact" && a.attn != "flash" && a.attn != "flash-bf16") || (a.weights != "bf16" && a.weights != "f16")) {
        fprintf(stderr, "heartmula-train: invalid settings (rank, grad-accum, steps >= 1; lr, alpha > 0; --weights bf16|f16; "
                        "--attn auto|flash-bf16|flash|exact; --keep-layers auto|-1..28; 0 <= --lora-dropout < 1)\n");
        return 2;
    }
    if (!a.eval.empty()) return hmt_eval(a);
    return a.selftest.empty() ? hmt_train(a) : hmt_selftest(a);
}
