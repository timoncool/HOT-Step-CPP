#pragma once
// train/mm3-dit-train-run.h — MM3 flow-DiT LoRA trainer.
//
// Step 4b: the loop around mm3-dit-train-graph.h. Reads the two caches
// ace-train mm3-preprocess / mm3-condition produce and trains a LoRA.
//
// ── Scope of THIS increment ────────────────────────────────────────────────
//
// Deliberately the smallest thing that can be judged: forward -> loss ->
// backward -> AdamW step, with loss reported per step. It exists to clear
// validation-ladder rungs 1-4 from the design doc (forward parity, zero-adapter
// neutrality, finite differences, single-song overfit) — NOT to be a production
// trainer. No checkpoint/resume, no JSONL event stream, no Muon, no LoKR, no
// gradient accumulation beyond a simple counter, no export. Those all have
// working ACE implementations to lift once the numbers say this trains at all.
//
// ── The graph/backward pattern, copied from dit-train-run.h ────────────────
//
//   forward -> loss
//   ggml_set_loss(loss)
//   ggml_build_forward_expand(gf, loss)
//   lm_optim_fill_gacc(&opt, gf, &gacc)      <- AFTER the forward expansion:
//                                               gacc is indexed by forward-node
//                                               order and is wrong if built early
//   ggml_build_backward_expand(ctx, gf, gacc.data())
//   ggml_backend_sched_reset(sched)
//   ggml_backend_sched_graph_compute(sched, gf)
//
// ── The loss-sign gate, which runs BEFORE any training ─────────────────────
//
// The velocity target's direction is the one thing the design doc refused to
// settle on paper, because SimpleTuner states its convention against a different
// sampler and MM3 carries an output-negation trap. So `--sign-check` runs first:
// with the adapter at ZERO it measures the loss at high sigma against both
// candidate targets and reports which is lower. A model that already knows how
// to denoise must score better on the correct one. If they are within noise of
// each other, something else is wrong and training would be meaningless.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <cfloat>

#include "dora-common.h"
#include "train/mm3-dit-train-graph.h"
#include "train/mm3-dit-train-ckpt.h"
#include "train/lm-optim.h"

// One cached song: target latents + conditioning + where the seams are.
struct MM3TrainSong {
    std::string name;
    std::string latents_path;    // f32 [128, L]
    std::string cond_path;       // f16 [L', 2048]
    // TWO lengths, and confusing them silently corrupts every channel but the
    // first. The latents file is [128, L_audio] so L_audio is the ROW STRIDE;
    // L is how much of it is usable (conditioning is almost always shorter, both
    // from the rounding mismatch and from EOS-early rollouts). Seek with
    // L_audio, sample crops within L.
    int64_t     L_audio = 0;     // row stride of the latents file
    int64_t     L       = 0;     // usable = min(audio L, cond L')
    int64_t     cond_dim = 0;
    std::vector<int64_t> seams;  // latent indices where one rollout segment ends
};

struct MM3TrainArgs {
    std::string cache_dir, models_dir, out_dir;
    int64_t     rank       = 32;
    float       alpha      = 32.0f;
    float       lr         = 1e-4f;
    int64_t     steps      = 200;
    int64_t     crop       = 689;   // one 200-frame DiT window
    int64_t     grad_accum = 1;
    uint64_t    seed       = 42;
    float       logit_mean = 0.0f;
    float       logit_std  = 1.0f;
    bool        sign_check = false;
    int64_t     eval_every = 0;     // 0 = off
    int64_t     eval_n     = 24;    // fixed tuples in the eval set
    // Eval window length, INDEPENDENT of the training crop. Pinned so that a
    // run at crop 1378 is still measurable against one at crop 689 -- otherwise
    // changing the crop silently changes the eval and destroys the only
    // cross-run yardstick we have. 0 = follow the training crop (old behaviour).
    int64_t     eval_crop  = 689;
    // "random" samples the crop start uniformly, covering the whole song.
    // "beginning" always starts at 0, which is what SimpleTuner's
    // truncation_mode default does (it keeps the head of each clip).
    std::string crop_mode  = "random";
    // Gradient checkpointing. 0/1 = off (one monolithic graph). >1 splits the
    // block stack into that many segments, cutting peak attention memory by
    // roughly the same factor for about one extra forward pass of compute.
    // Required for crop 2584 (30 s), which needs ~30.8 GB monolithic.
    int64_t     ckpt_segments = 0;
    // Run BOTH paths on the same micro-batch and report the difference. This is
    // the acceptance gate: a checkpointed run that silently disagreed with the
    // monolithic one is worse than no checkpointing.
    bool        ckpt_verify   = false;
    // Snapshot export every N steps -> <out>/mm3_lora_step<N>.safetensors.
    // Exists for the DELTA LADDER: the audible-style band is bracketed between
    // 0.074% relative delta (SimpleTuner 200-step reference: provably harmless,
    // provably inaudible) and ~17% (ours: audibly destroyed). Nobody has ever
    // rendered the 1-5% band. Snapshots + offline relnorm.py let one run
    // produce the whole ladder; at ~0.2 s/step that is minutes, not their 15 h.
    int64_t     export_every  = 0;
    // "all" = every site (run-09 behaviour). "mlpv" = timbre-only: MLPs +
    // attention V + to_out; no q,k (structure poison), no proj heads (fuzz).
    std::string target        = "all";
    std::string only_song;          // restrict to one song (overfit test)
    // DoRA: learn a per-output magnitude on every site (see MM3TrainLora).
    // The CLI flag is wired by the Stage A integration slice.
    bool        dora          = false;
};

// ── Cache loading ──────────────────────────────────────────────────────────

// Read the two manifests and pair them by id. A song is usable only if BOTH
// halves exist; conditioning can be shorter than the audio (see the design
// doc's rounding note and the EOS-early behaviour), so L is always the min.
static bool mm3_train_load_cache(const std::string & cache_dir, const std::string & only,
                                 std::vector<MM3TrainSong> * out, std::string * err) {
    auto slurp = [&](const std::string & p, std::string * dst) -> bool {
        FILE * f = hs_fopen(p, "rb");
        if (!f) { if (err) *err = "cannot open " + p; return false; }
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        dst->assign((size_t) n, '\0');
        const bool ok = fread(&(*dst)[0], 1, (size_t) n, f) == (size_t) n;
        fclose(f);
        if (!ok && err) *err = "short read on " + p;
        return ok;
    };
    std::string pre_s, cond_s;
    if (!slurp(cache_dir + "/mm3_preprocess.json", &pre_s)) return false;
    if (!slurp(cache_dir + "/mm3_condition.json", &cond_s)) return false;

    yyjson_doc * pd = yyjson_read(pre_s.c_str(), pre_s.size(), 0);
    yyjson_doc * cd = yyjson_read(cond_s.c_str(), cond_s.size(), 0);
    if (!pd || !cd) { if (err) *err = "malformed manifest JSON"; return false; }

    // id -> conditioning record
    std::map<std::string, yyjson_val *> cond_by_id;
    yyjson_val * carr = yyjson_obj_get(yyjson_doc_get_root(cd), "samples");
    yyjson_val * cv; yyjson_arr_iter cit = yyjson_arr_iter_with(carr);
    while ((cv = yyjson_arr_iter_next(&cit))) {
        yyjson_val * idv = yyjson_obj_get(cv, "id");
        if (idv && yyjson_is_str(idv)) cond_by_id[yyjson_get_str(idv)] = cv;
    }

    yyjson_val * parr = yyjson_obj_get(yyjson_doc_get_root(pd), "samples");
    yyjson_val * pv; yyjson_arr_iter pit = yyjson_arr_iter_with(parr);
    while ((pv = yyjson_arr_iter_next(&pit))) {
        auto gs = [&](yyjson_val * o, const char * k) -> std::string {
            yyjson_val * v = yyjson_obj_get(o, k);
            return (v && yyjson_is_str(v)) ? std::string(yyjson_get_str(v)) : std::string();
        };
        auto gi = [&](yyjson_val * o, const char * k) -> int64_t {
            yyjson_val * v = yyjson_obj_get(o, k);
            return v ? (int64_t) yyjson_get_int(v) : 0;
        };
        const std::string id = gs(pv, "id"), fn = gs(pv, "filename");
        if (!only.empty() && fn.find(only) == std::string::npos) continue;
        auto it = cond_by_id.find(id);
        if (it == cond_by_id.end()) {
            fprintf(stderr, "[mm3-train] %s has no conditioning, skipped\n", fn.c_str());
            continue;
        }
        MM3TrainSong s;
        s.name         = fn;
        s.latents_path = cache_dir + "/" + gs(pv, "latents");
        s.cond_path    = cache_dir + "/" + gs(it->second, "condition");
        s.cond_dim     = gi(it->second, "cond_dim");
        s.L_audio      = gi(pv, "latent_frames");
        s.L            = std::min<int64_t>(s.L_audio, gi(it->second, "cond_latents"));
        yyjson_val * sb = yyjson_obj_get(it->second, "segment_latent_starts");
        if (sb && yyjson_is_arr(sb)) {
            yyjson_val * b; yyjson_arr_iter bi = yyjson_arr_iter_with(sb);
            while ((b = yyjson_arr_iter_next(&bi))) s.seams.push_back((int64_t) yyjson_get_int(b));
        }
        out->push_back(std::move(s));
    }
    yyjson_doc_free(pd); yyjson_doc_free(cd);
    if (out->empty()) { if (err) *err = "no usable songs in the cache"; return false; }
    return true;
}

// Read one crop. latents are f32 [128, L] channel-major; conditioning is f16
// [L', 2048] latent-major, i.e. exactly what mm3_cond_encode emitted.
static bool mm3_train_read_crop(const MM3TrainSong & s, int64_t start, int64_t crop, int64_t IC,
                                std::vector<float> * lat, std::vector<float> * cond, std::string * err) {
    lat->assign((size_t) (IC * crop), 0.0f);
    FILE * f = hs_fopen(s.latents_path, "rb");
    if (!f) { if (err) *err = "cannot open " + s.latents_path; return false; }
    for (int64_t c = 0; c < IC; c++) {
        // channel c of the full [128, L] block, offset to the crop
        // stride is L_audio, NOT s.L — see the struct comment.
        if (fseek(f, (long) (((size_t) (c * s.L_audio + start)) * sizeof(float)), SEEK_SET) != 0 ||
            fread(lat->data() + c * crop, sizeof(float), (size_t) crop, f) != (size_t) crop) {
            fclose(f); if (err) *err = "short read on " + s.latents_path; return false;
        }
    }
    fclose(f);

    const int64_t CD = s.cond_dim;
    cond->assign((size_t) (CD * crop), 0.0f);
    FILE * g = hs_fopen(s.cond_path, "rb");
    if (!g) { if (err) *err = "cannot open " + s.cond_path; return false; }
    std::vector<ggml_fp16_t> half((size_t) (CD * crop));
    if (fseek(g, (long) (((size_t) (start * CD)) * sizeof(ggml_fp16_t)), SEEK_SET) != 0 ||
        fread(half.data(), sizeof(ggml_fp16_t), half.size(), g) != half.size()) {
        fclose(g); if (err) *err = "short read on " + s.cond_path; return false;
    }
    fclose(g);
    // NO transpose. The cache stores [crop, CD] and the graph's cond tensor is
    // ne0=CD, ne1=crop -- which is the SAME memory order. An earlier version
    // transposed here and then also declared the tensor transposed, i.e. wrong
    // twice, and ggml_concat caught it.
    for (size_t i = 0; i < half.size(); i++) {
        (*cond)[i] = ggml_fp16_to_fp32(half[i]);
    }
    return true;
}

// ── Adapter parameters ─────────────────────────────────────────────────────

// Allocate LoRA A/B for every site. A is init'd small-random and B ZERO, so the
// adapter starts as an exact no-op — which is also validation rung 2: a fresh
// adapter must leave the forward bit-identical.
//
// `mlpv` restricts training to the timbre groups, from the run-09 group
// ablation (Rob's ears, 2026-08-16, all at 2.85% median delta / scale 1.0):
//   full set        -> "no apparent coherence, random clips"
//   minus proj_out  -> still a mess            (q,k routing is the poison)
//   minus q,k       -> coherent but "radio tuning" fuzz   (proj heads)
//   MLP+V+out only  -> coherent, CLEAR LYRICS, stable; style under-dosed
// So: no proj_in/proj_out sites at all, and the fused qkv trains V rows only
// via out_mask. Filtering at export would waste the optimizer's budget on
// deltas we discard; excluding here concentrates ALL fitting in what ships.
static bool mm3_train_make_adapters(ggml_context * ctx, const MM3Model & m, int64_t rank, float alpha,
                                    bool mlpv, bool dora, MM3TrainAdapters * ad,
                                    std::vector<ggml_tensor *> * params) {
    const MM3DitConfig & c = m.synth_cfg.dit;
    const int64_t E = (int64_t) c.embedding_length, FI = (int64_t) c.ff_inner;
    const int64_t CC = (int64_t) c.concat_channels, IC = (int64_t) c.in_channels;
    const float   sc = alpha / (float) rank;

    auto mk = [&](MM3TrainLora * lo, int64_t in, int64_t out, const char * tag) {
        lo->a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, rank);
        lo->b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, out);
        lo->scale = sc;
        ggml_set_name(lo->a, (std::string(tag) + ".A").c_str());
        ggml_set_name(lo->b, (std::string(tag) + ".B").c_str());
        ggml_set_param(lo->a);
        ggml_set_param(lo->b);
        params->push_back(lo->a);
        params->push_back(lo->b);
        if (dora) {
            // After A/B in params, and m is not named "...A", so the init loop's
            // rng stream (and with it every A) matches a plain LoRA run.
            lo->m   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out);
            lo->nrm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out);
            ggml_set_name(lo->m, (std::string(tag) + ".dora_m").c_str());
            ggml_set_name(lo->nrm, (std::string(tag) + ".dora_nrm").c_str());
            ggml_set_param(lo->m);
            params->push_back(lo->m);
        }
    };

    // ONE shared V-only mask for every block's qkv (identical shape [3E, 1]):
    // rows [0, 2E) = q,k -> 0;  [2E, 3E) = v -> 1. NOT a param — it is a
    // constant, and the chain rule through the mul is what freezes q,k.
    // Row order q|k|v matches the fused base tensor; validated empirically by
    // the MLPV merge-surgery ear test before being trusted here.
    ggml_tensor * vmask = nullptr;
    if (mlpv) {
        vmask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3 * E, 1);
        ggml_set_name(vmask, "dit.qkv_vmask");
        ad->qkv_vmask = vmask;
        if (dora) {
            ad->qkv_vkeep = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3 * E, 1);
            ggml_set_name(ad->qkv_vkeep, "dit.qkv_vkeep");
        }
    }

    ad->blk.assign(m.synth.dit.blk.size(), MM3TrainBlockAdapters{});
    for (size_t i = 0; i < ad->blk.size(); i++) {
        const std::string p = "dit.blk." + std::to_string(i);
        mk(&ad->blk[i].qkv,      E,  3 * E,      (p + ".attn_qkv").c_str());
        ad->blk[i].qkv.out_mask = vmask;   // nullptr unless mlpv
        ad->blk[i].qkv.keep     = ad->qkv_vkeep;
        mk(&ad->blk[i].attn_out, E,  E,          (p + ".attn_output").c_str());
        mk(&ad->blk[i].ff_in,    E,  2 * FI,     (p + ".ffn_in").c_str());
        mk(&ad->blk[i].ff_out,   FI, E,          (p + ".ffn_out").c_str());
    }
    if (!mlpv) {
        mk(&ad->proj_in,  CC, E,  "dit.proj_in");
        mk(&ad->proj_out, E,  IC, "dit.proj_out");
    }
    return true;
}

// ── DoRA norm refresh ──────────────────────────────────────────────────────
//
// nrm = max(||W + s*BA||_col, floor) for every DoRA site. A/B only move at
// optimizer steps, so refreshing after each step is exact for the next window
// and for any eval or export in between. One graph per SITE: the [in, out] F32
// transients are the whole cost, they land in the optimizer scheduler's buffer
// (which keeps its high-water mark), and one site at a time keeps that at the
// largest single projection instead of a whole block.
//
// Zero-norm policy: the floor keeps an all-zero row finite. Its output is 0
// whatever m is, where 0/0 would be NaN. mm3-adapter.h's merge uses the same
// floor, so trainer and merge agree on that row too.
static const float MM3_DORA_NRM_FLOOR = 1e-12f;

static bool mm3_train_dora_refresh(const MM3Model & m, const MM3TrainAdapters & ad, ggml_backend_sched_t sched,
                                   std::string * err) {
    const size_t         n_nodes = 32;
    std::vector<uint8_t> arena(ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false));
    auto run = [&](const MM3TrainLora & lo, ggml_tensor * w) -> bool {
        if (!lo.m) return true;
        ggml_init_params ip  = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        if (!ctx) { *err = "DoRA norm pass: cannot create context"; return false; }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_nodes, false);
        // B is an F32 param; the F16 base is cast here, as in mm3-lm-dora.h.
        ggml_tensor * nr = dora_col_norm(ctx, ggml_cast(ctx, w, GGML_TYPE_F32), lo.a, lo.b, lo.scale);
        nr               = ggml_clamp(ctx, nr, MM3_DORA_NRM_FLOOR, FLT_MAX);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, nr, ggml_reshape_2d(ctx, lo.nrm, 1, lo.nrm->ne[0])));
        ggml_backend_sched_reset(sched);
        const bool ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
        ggml_free(ctx);
        if (!ok) *err = std::string("DoRA norm pass: graph compute failed on ") + ggml_get_name(lo.m) +
                        " (out of VRAM?)";
        return ok;
    };
    const MM3DitWeights & w = m.synth.dit;
    for (size_t i = 0; i < ad.blk.size(); i++) {
        const MM3TrainBlockAdapters & b  = ad.blk[i];
        const MM3DitBlock &           wb = w.blk[i];
        if (!run(b.qkv, wb.attn_qkv) || !run(b.attn_out, wb.attn_output) || !run(b.ff_in, wb.ffn_in_w) ||
            !run(b.ff_out, wb.ffn_out_w)) {
            return false;
        }
    }
    return run(ad.proj_in, w.proj_in) && run(ad.proj_out, w.proj_out);
}

// ── Sigma / target ─────────────────────────────────────────────────────────
//
// Rectified flow, matching the scheduler this port already replicates:
//   sigma = 1 - sigmoid(logit_normal(mean, std)),  x_t = sigma*noise + (1-sigma)*x0
//
// The velocity target's SIGN is the one thing the design doc refused to settle
// on paper. mm3-dit-graph.h's Euler step is `x + (sigma_next - sigma) * v` with
// sigma RISING 0->1, i.e. integrating from data toward noise; so the field it
// wants is d(x)/d(sigma) = noise - x0. `--sign-check` verifies that empirically
// against the alternative before any training happens.
static inline float mm3_sigma_from(std::mt19937_64 & rng, float mean, float std_) {
    std::normal_distribution<double> nd(mean, std_);
    const double u = 1.0 / (1.0 + std::exp(-nd(rng)));
    return (float) std::min(1.0, std::max(0.0, 1.0 - u));
}

// target = (noise - x0) when `positive`, else (x0 - noise).
static void mm3_make_xt_target(const std::vector<float> & x0, const std::vector<float> & noise,
                               float sigma, bool positive, std::vector<float> * xt,
                               std::vector<float> * target) {
    xt->resize(x0.size());
    target->resize(x0.size());
    for (size_t i = 0; i < x0.size(); i++) {
        (*xt)[i]     = sigma * noise[i] + (1.0f - sigma) * x0[i];
        (*target)[i] = positive ? (noise[i] - x0[i]) : (x0[i] - noise[i]);
    }
}

// Deterministic Gaussian noise. NOT std::normal_distribution — its byte stream
// is stdlib-dependent, which is trap #8 in the mm3-backend skill and would make
// a "same seed" run unreproducible across builds. splitmix64 + Box-Muller.
static void mm3_fill_noise_train(std::vector<float> * v, uint64_t seed) {
    uint64_t s = seed;
    auto nxt = [&]() -> double {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= (z >> 31);
        return ((z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    };
    for (size_t i = 0; i < v->size(); i += 2) {
        const double u1 = std::max(1e-12, nxt()), u2 = nxt();
        const double r = std::sqrt(-2.0 * std::log(u1)), th = 6.283185307179586 * u2;
        (*v)[i] = (float) (r * std::cos(th));
        if (i + 1 < v->size()) (*v)[i + 1] = (float) (r * std::sin(th));
    }
}

// ── One micro-step ─────────────────────────────────────────────────────────
//
// Builds forward -> loss (and backward when training), computes, returns the
// loss. `backward == false` is the measurement path used by --sign-check and by
// zero-adapter neutrality.
static bool mm3_train_micro(MM3Model & m, ggml_backend_sched_t sched, const MM3TrainAdapters & ad,
                            LmOptim * opt,
                            const std::vector<float> & xt_h, const std::vector<float> & cond_h,
                            const std::vector<float> & tgt_h, const std::vector<float> & fourier_h,
                            int64_t crop, bool backward, float * loss_out, std::string * err) {
    const MM3DitConfig & c  = m.synth_cfg.dit;
    const int64_t        IC = (int64_t) c.in_channels;
    const int64_t        CD = (int64_t) c.condition_dim;
    const int64_t        E  = (int64_t) c.embedding_length;

    // Node budget. The backward graph is ~6.1k nodes at crop 344 and scales
    // with crop and depth, so 32k is generous; 262144 was not "safe", it made
    // the graph hash enormous while the TENSOR arena stayed too small for the
    // backward pass to allocate into. When ggml_new_tensor runs out of arena it
    // returns NULL, the expansion keeps going, and the crash lands later in
    // alloc/compute with no message -- which is exactly what happened.
    const size_t MAX_NODES = 32768;
    const size_t meta = ggml_tensor_overhead() * (MAX_NODES * 4)
                      + ggml_graph_overhead_custom(MAX_NODES, true)
                      + (size_t) 32 * 1024 * 1024;
    ggml_init_params ip   = { meta, nullptr, true };
    ggml_context *   ctx  = ggml_init(ip);
    if (!ctx) { if (err) *err = "ggml_init failed"; return false; }

    MM3TrainInputs in;
    in.xt        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, crop, IC);
    in.cond      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, CD, crop);
    in.fourier   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) c.fourier_dim, 1);
    in.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, crop + 1);
    in.vtarget   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, crop, IC);
    in.seq_zeros = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, crop + 1);
    ggml_set_input(in.xt); ggml_set_input(in.cond); ggml_set_input(in.fourier);
    ggml_set_input(in.positions); ggml_set_input(in.vtarget); ggml_set_input(in.seq_zeros);

    if (!in.xt || !in.cond || !in.fourier || !in.positions || !in.vtarget || !in.seq_zeros) {
        ggml_free(ctx); if (err) *err = "input tensor alloc failed (ctx arena too small)";
        return false;
    }
    ggml_tensor * pred = mm3_dt_forward(ctx, m, ad, in);
    ggml_tensor * loss = mm3_dt_loss(ctx, pred, in.vtarget);
    if (backward) ggml_set_loss(loss);
    ggml_set_output(loss);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, MAX_NODES, backward);
    if (!gf) { ggml_free(ctx); if (err) *err = "ggml_new_graph_custom failed"; return false; }
    ggml_build_forward_expand(gf, loss);
    if (getenv("MM3_TRAIN_TRACE")) fprintf(stderr, "[trace] fwd expanded, %d nodes\n", ggml_graph_n_nodes(gf));
    if (backward) {
        // AFTER the forward expansion — gacc is indexed by forward-node order.
        std::vector<ggml_tensor *> gacc;
        lm_optim_fill_gacc(opt, gf, &gacc);
        if (getenv("MM3_TRAIN_TRACE")) fprintf(stderr, "[trace] gacc %zu entries\n", gacc.size());
        ggml_build_backward_expand(ctx, gf, gacc.data());
        if (getenv("MM3_TRAIN_TRACE")) fprintf(stderr, "[trace] bwd expanded, %d nodes\n", ggml_graph_n_nodes(gf));
    }

    if (getenv("MM3_TRAIN_TRACE")) fprintf(stderr, "[trace] %s\n", "pre-reset");
    ggml_backend_sched_reset(sched);
    if (getenv("MM3_TRAIN_TRACE")) fprintf(stderr, "[trace] %s\n", "post-reset");
    if (getenv("MM3_TRAIN_TRACE")) fprintf(stderr, "[trace] %s\n", "pre-alloc");
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        ggml_free(ctx); if (err) *err = "sched_alloc_graph failed (out of VRAM?)"; return false;
    }
    if (getenv("MM3_TRAIN_TRACE")) fprintf(stderr, "[trace] %s\n", "post-alloc, uploading");
    // MM3_TRAIN_SPLITS=1: after allocation, walk the graph and report every
    // point where the assigned backend CHANGES. That is what a "split" is, and
    // naming the op on each side of the boundary says which node is dragging
    // work off the GPU. All eleven ops this graph uses have CUDA kernels, so
    // the cause is placement, not a missing implementation.
    if (getenv("MM3_TRAIN_SPLITS")) {
        ggml_backend_t prev = nullptr;
        int changes = 0;
        std::map<std::string, int> culprit;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * nd = ggml_graph_node(gf, i);
            ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, nd);
            if (b && prev && b != prev) {
                changes++;
                culprit[std::string(ggml_op_name(nd->op)) + " -> " +
                        std::string(ggml_backend_name(b))]++;
            }
            if (b) prev = b;
        }
        fprintf(stderr, "[splits] %d backend changes over %d nodes\n", changes,
                ggml_graph_n_nodes(gf));
        int shown = 0;
        for (auto it = culprit.begin(); it != culprit.end() && shown < 12; ++it, ++shown) {
            fprintf(stderr, "[splits]   %5d x  %s\n", it->second, it->first.c_str());
        }
    }

    ggml_backend_tensor_set(in.xt,      xt_h.data(),   0, xt_h.size()   * sizeof(float));
    ggml_backend_tensor_set(in.cond,    cond_h.data(), 0, cond_h.size() * sizeof(float));
    ggml_backend_tensor_set(in.vtarget, tgt_h.data(),  0, tgt_h.size()  * sizeof(float));
    ggml_backend_tensor_set(in.fourier, fourier_h.data(), 0, fourier_h.size() * sizeof(float));
    {
        // Time token holds position 0, so latent frames are 1..crop.
        std::vector<int32_t> pos((size_t) crop + 1);
        for (int64_t i = 0; i <= crop; i++) pos[(size_t) i] = (int32_t) i;
        ggml_backend_tensor_set(in.positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    }
    {
        std::vector<float> z((size_t) (E * (crop + 1)), 0.0f);
        ggml_backend_tensor_set(in.seq_zeros, z.data(), 0, z.size() * sizeof(float));
    }

    if (getenv("MM3_TRAIN_TRACE")) fprintf(stderr, "[trace] %s\n", "uploaded, computing");
    const int64_t t_c0 = ggml_time_us();
    const bool ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    if (getenv("MM3_TRAIN_TRACE")) {
        fprintf(stderr, "[trace] compute %.1f ms | %d sched splits | %d nodes\n",
                (double) (ggml_time_us() - t_c0) / 1000.0,
                ggml_backend_sched_get_n_splits(sched), ggml_graph_n_nodes(gf));
    }
    // Are gradients reaching the accumulators? Scan ALL of them, not acc[0]:
    // acc[0] is a LoRA **A** factor and dL/dA = B^T.grad is LEGITIMATELY zero
    // while B is still zero-initialised. Only the B factors carry gradient on
    // the first step, so a single-tensor probe reads as "no gradients" when
    // everything is in fact fine.
    if (backward && getenv("MM3_TRAIN_TRACE") && opt && !opt->acc.empty()) {
        int n_nonzero = 0, n_null = 0, n_nonfinite = 0;
        double best = 0.0; const char * best_name = "";
        std::vector<float> tmp;
        for (size_t j = 0; j < opt->acc.size(); j++) {
            ggml_tensor * g = opt->acc[j];
            if (!g || !g->data) { n_null++; continue; }
            tmp.assign((size_t) ggml_nelements(g), 0.0f);
            ggml_backend_tensor_get(g, tmp.data(), 0, tmp.size() * sizeof(float));
            double ss = 0.0;
            for (float v : tmp) { ss += (double) v * v; if (!std::isfinite(v)) n_nonfinite++; }
            if (ss > 0.0) n_nonzero++;
            if (ss > best) { best = ss; best_name = ggml_get_name(g); }
        }
        fprintf(stderr, "[trace] acc: %d/%zu nonzero, %d null, %d nonfinite; max |g|=%.4e (%s)\n",
                n_nonzero, opt->acc.size(), n_null, n_nonfinite, sqrt(best), best_name);
    }
    if (ok && loss_out) ggml_backend_tensor_get(loss, loss_out, 0, sizeof(float));
    if (getenv("MM3_TRAIN_TRACE")) fprintf(stderr, "[trace] %s\n", "loss read");
    ggml_free(ctx);
    if (getenv("MM3_TRAIN_TRACE")) fprintf(stderr, "[trace] %s\n", "ctx freed");
    if (!ok && err) *err = "graph compute failed";
    return ok;
}

// Host-side Fourier features for timestep t, cos-first, computed in double.
// Byte-for-byte the same formula as mm3-dit-graph.h's inference path — if that
// ever changes, this must follow or training and inference silently diverge.
static void mm3_train_fourier(const std::vector<float> & fourier_w, float t, std::vector<float> * out) {
    const size_t H = fourier_w.size();
    out->assign(2 * H, 0.0f);
    for (size_t i = 0; i < H; i++) {
        const double a = 2.0 * 3.14159265358979323846 * (double) t * (double) fourier_w[i];
        (*out)[i]     = (float) std::cos(a);
        (*out)[H + i] = (float) std::sin(a);
    }
}

static bool mm3_train_export(const MM3TrainAdapters & ad, const MM3Model & m, int64_t rank, float alpha,
                             const std::string & path, std::string * err);

// ── Driver ─────────────────────────────────────────────────────────────────

static int mm3_train_dit_run(const MM3TrainArgs & a) {
    std::string err;

    std::vector<MM3TrainSong> songs;
    if (!mm3_train_load_cache(a.cache_dir, a.only_song, &songs, &err)) {
        fprintf(stderr, "[mm3-train] %s\n", err.c_str());
        return 1;
    }
    int64_t usable = 0;
    for (const auto & s : songs) if (s.L > a.crop) usable++;
    fprintf(stderr, "[mm3-train] %zu song(s), %lld with >= %lld latents\n", songs.size(),
            (long long) usable, (long long) a.crop);
    if (!usable) { fprintf(stderr, "[mm3-train] no song is longer than the crop\n"); return 1; }

    // DiT only. The LM and depth decoder produced the conditioning cache and are
    // not needed again; everything this loop reads is on disk.
    static MM3Model m;
    mm3_discover(&m, a.models_dir.c_str());
    if (!mm3_load_parts(&m, /*lm*/ false, /*depth*/ false, /*rest*/ true, &err)) {
        fprintf(stderr, "[mm3-train] load: %s\n", err.c_str());
        return 1;
    }
    if (!mm3_dit_prepare(m, &g_mm3_dit, &err)) {
        fprintf(stderr, "[mm3-train] dit prepare: %s\n", err.c_str());
        return 1;
    }

    const MM3DitConfig & c  = m.synth_cfg.dit;
    const int64_t        IC = (int64_t) c.in_channels;

    // The trainer gets its OWN scheduler. g_mm3_dit.sched is sized
    // MM3_DIT_MAX_NODES*2 = 8192 for the inference graph; a backward graph is
    // 6084 nodes at crop 128 alone and grows with depth, so reusing it runs
    // right up against the limit and shares split state with inference for no
    // benefit. Sized generously — the cost is bookkeeping, not VRAM.
    BackendPair tbp{};
    tbp.backend     = m.backend;
    tbp.cpu_backend = m.cpu_backend;
    ggml_backend_sched_t tsched = backend_sched_new(tbp, 65536);
    ggml_backend_sched_t osched = backend_sched_new(tbp, 16384);   // optimizer only
    if (!tsched) { fprintf(stderr, "[mm3-train] scheduler alloc failed\n"); return 1; }

    // Adapter parameters in their own context + backend buffer.
    const size_t     n_sites = (m.synth.dit.blk.size() * 4 + 2) * 4;   // A, B, m, nrm
    ggml_init_params aip     = { (n_sites + 16) * ggml_tensor_overhead(), nullptr, true };
    ggml_context *   actx    = ggml_init(aip);
    MM3TrainAdapters ad;
    std::vector<ggml_tensor *> params;
    const bool target_mlpv = (a.target == "mlpv");
    mm3_train_make_adapters(actx, m, a.rank, a.alpha, target_mlpv, a.dora, &ad, &params);
    // THE SEED GRADIENT. LmOptim::t_lossgrad defaults to nullptr and is NOT
    // created by lm_optim_init — ACE builds it itself (dit-train-run.h) and
    // assigns it. Leaving it null makes lm_optim_fill_gacc hand a null
    // accumulator to the LOSS node, so no gradient is ever seeded and
    // lm_optim_step dereferences it. Holds 1/grad_accum, which is how the
    // micro-batches average.
    ggml_tensor * t_lossgrad = ggml_new_tensor_1d(actx, GGML_TYPE_F32, 1);
    ggml_set_name(t_lossgrad, "lossgrad");
    // THE SAME TRAP, and this one is the actual segfault: LmOptim::t_adamw is
    // also nullptr by default and also NOT created by lm_optim_init. ACE builds
    // both in its own static context. lm_optim_step writes the 7 AdamW
    // hyper-parameters into it UNCONDITIONALLY near the end
    // (ggml_backend_tensor_set(o->t_adamw, p7, ...)), so a null lands as a
    // write through a null tensor -- after the graph has built and computed,
    // which is why the crash looked like it was in the optimizer maths.
    // {alpha, beta1, beta2, eps, wd, beta1_hat, beta2_hat}
    ggml_tensor * t_adamw = ggml_new_tensor_1d(actx, GGML_TYPE_F32, 7);
    ggml_set_name(t_adamw, "adamw_params");
    // FOUR host-owned scalars in total, not two. LmOptim declares t_adamw,
    // t_lossgrad, t_clip and t_eps as nullptr and lm_optim_init creates NONE of
    // them -- the caller owns them (ACE builds all four in its ctx_static).
    // Every one is written or read inside lm_optim_step, so a missing one is a
    // null dereference AFTER the graph has computed, which is why this looked
    // like a fault in the optimizer maths rather than missing setup.
    ggml_tensor * t_clip = ggml_new_tensor_1d(actx, GGML_TYPE_F32, 1);
    ggml_tensor * t_eps  = ggml_new_tensor_1d(actx, GGML_TYPE_F32, 1);
    ggml_set_name(t_clip, "grad_clip");
    ggml_set_name(t_eps, "eps");
    // FIVE, not four. t_gnorm2 is the global-norm readback slot; lm_optim_step
    // does ggml_cpy(ctx, gn2, o->t_gnorm2) unconditionally, so a null here is a
    // copy into nothing -- the segfault. It is comment-labelled "LOGGING ONLY",
    // which is true of the VALUE and not of the pointer.
    ggml_tensor * t_gnorm2 = ggml_new_tensor_1d(actx, GGML_TYPE_F32, 1);
    ggml_set_name(t_gnorm2, "gnorm2");
    ggml_backend_buffer_t abuf = ggml_backend_alloc_ctx_tensors(actx, m.backend);
    if (!abuf) { fprintf(stderr, "[mm3-train] adapter alloc failed\n"); return 1; }
    // Tell the scheduler these live on the compute backend. Without it the graph
    // was cut into 293 splits — one per LoRA parameter, plus one — and spent
    // 47.6 s per step synchronising at each boundary while the GPU sat at 9 %.
    // weight-ctx.h flags exactly this for the model weights: the usage hint
    // "assigns ops to the correct backend based on weight location (avoids
    // fallback through expansion)".
    ggml_backend_buffer_set_usage(abuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    {
        const float lg = 1.0f / (float) std::max<int64_t>(1, a.grad_accum);
        ggml_backend_tensor_set(t_lossgrad, &lg, 0, sizeof(float));
        const float clip = 1.0f;    // global-norm clip, as the ACE trainer
        const float eps  = 1e-6f;   // Newton-Schulz / norm epsilon
        ggml_backend_tensor_set(t_clip, &clip, 0, sizeof(float));
        ggml_backend_tensor_set(t_eps,  &eps,  0, sizeof(float));
    }

    // A small-random, B ZERO -> the adapter starts as an exact no-op.
    {
        std::mt19937_64 rng(a.seed);
        std::normal_distribution<float> nd(0.0f, 0.02f);
        for (ggml_tensor * t : params) {
            const size_t n = (size_t) ggml_nelements(t);
            std::vector<float> v(n, 0.0f);
            const char * nm = ggml_get_name(t);
            if (nm && strlen(nm) > 2 && nm[strlen(nm) - 1] == 'A') {
                for (size_t i = 0; i < n; i++) v[i] = nd(rng);
            }
            ggml_backend_tensor_set(t, v.data(), 0, n * sizeof(float));
        }
    }
    // Fill the V-only mask AFTER buffer allocation. It is not a param, so the
    // init loop above skipped it; an unfilled mask would be allocator garbage
    // and silently random-mask the delta.
    if (ad.qkv_vmask) {
        const int64_t E3 = ad.qkv_vmask->ne[0];
        std::vector<float> mv((size_t) E3, 0.0f);
        for (int64_t i = 2 * (E3 / 3); i < E3; i++) mv[(size_t) i] = 1.0f;   // v rows only
        ggml_backend_tensor_set(ad.qkv_vmask, mv.data(), 0, mv.size() * sizeof(float));
        if (ad.qkv_vkeep) {
            for (float & x : mv) x = 1.0f - x;
            ggml_backend_tensor_set(ad.qkv_vkeep, mv.data(), 0, mv.size() * sizeof(float));
        }
    }
    // DoRA init: B is zero, so the refresh yields ||W||_col, and m starts there.
    // The ratio is then exactly 1 and the first forward is the plain one.
    if (a.dora) {
        if (!mm3_train_dora_refresh(m, ad, osched, &err)) {
            fprintf(stderr, "[mm3-train] %s\n", err.c_str()); return 1;
        }
        std::vector<float> col;
        for (ggml_tensor * t : params) {
            const char * nm = ggml_get_name(t);
            const size_t ln = strlen(nm);
            if (ln < 7 || strcmp(nm + ln - 7, ".dora_m") != 0) continue;
            ggml_tensor * nrm = ggml_get_tensor(actx, (std::string(nm, ln - 1) + "nrm").c_str());
            col.resize((size_t) ggml_nelements(t));
            ggml_backend_tensor_get(nrm, col.data(), 0, col.size() * sizeof(float));
            ggml_backend_tensor_set(t, col.data(), 0, col.size() * sizeof(float));
        }
    }
    fprintf(stderr, "[mm3-train] %zu %s tensors, rank %lld, alpha %.1f, target %s\n", params.size(),
            a.dora ? "DoRA" : "LoRA", (long long) a.rank, (double) a.alpha, a.target.c_str());

    // Report the sigma distribution this run will actually see, by sampling it
    // rather than by quoting logit_mean. sigma near 0 means the crop is mostly
    // REAL AUDIO and the step can learn "render this content in this style";
    // sigma near 1 is near-pure noise, where the only learnable signal is the
    // caption marginal. `frac<0.5` is therefore the fraction of steps that can
    // teach style at all -- it is 0.5 at the default mean of 0.
    {
        std::mt19937_64 probe(12345);
        double sum = 0.0; int lo = 0;
        const int N = 20000;
        for (int i = 0; i < N; i++) {
            const float s = mm3_sigma_from(probe, a.logit_mean, a.logit_std);
            sum += s;
            if (s < 0.5f) lo++;
        }
        fprintf(stderr, "[mm3-train] sigma: logit_mean %.2f std %.2f -> mean sigma %.3f, %.1f%% below 0.5\n",
                (double) a.logit_mean, (double) a.logit_std, sum / N, 100.0 * lo / N);
    }

    LmOptim opt;
    opt.t_lossgrad = t_lossgrad;   // must be set; see above
    opt.t_adamw    = t_adamw;      // ditto -- lm_optim_step writes into it
    opt.t_clip     = t_clip;
    opt.t_eps      = t_eps;
    opt.t_gnorm2   = t_gnorm2;
    opt.grad_clip  = 1.0f;
    if (!lm_optim_init(&opt, params, m.backend, &err)) {
        fprintf(stderr, "[mm3-train] optim: %s\n", err.c_str());
        return 1;
    }
    // LmOptim drives a cosine schedule off base_lr; lr(0) is 0 by design
    // (lm-optim.h uses `step`, not `step+1`), so a very short run spends a
    // real fraction of itself warming up. Warmup is 5% as in the ACE trainer.
    opt.base_lr      = a.lr;
    opt.total_steps  = (int) a.steps;
    opt.warmup_steps = (int) std::max<int64_t>(1, a.steps / 20);

    std::mt19937_64 rng(a.seed);
    std::vector<float> x0, cond, noise, xt, tgt, fourier;

    const bool crop_from_start = (a.crop_mode == "beginning");
    auto sample_crop = [&](const MM3TrainSong ** song, int64_t * start) {
        for (;;) {
            const MM3TrainSong & s = songs[rng() % songs.size()];
            if (s.L <= a.crop) continue;
            *song = &s;
            // Draw the position even when we discard it, so the rng stream --
            // and therefore the whole training trajectory -- stays aligned
            // between the two modes. Otherwise "beginning" would also silently
            // reshuffle which songs get picked, confounding the comparison.
            const int64_t r = (int64_t) (rng() % (uint64_t) (s.L - a.crop));
            // REJECT crops that straddle a rollout seam. The conditioning cache
            // is built from INDEPENDENT 60 s rollout segments; at a seam it
            // jumps to an unrelated rollout mid-crop while the audio target
            // flows on. Training on that teaches "conditioning sometimes lies,
            // smooth over it" -- direct pressure toward ignoring conditioning.
            // The design doc specified this rejection; the seams were parsed
            // and then never consulted. At crop 689 ~13% of uniform crops
            // straddle; at 1378 it is ~27%. Retry rather than clamp, so the
            // position distribution stays uniform over the valid set.
            if (!crop_from_start) {
                bool straddles = false;
                for (int64_t seam : s.seams) {
                    if (seam > r && seam < r + a.crop) { straddles = true; break; }
                }
                if (straddles) continue;
            }
            *start = crop_from_start ? 0 : r;
            return;
        }
    };

    // MEASURED, not derived. --sign-check on albumA, sigma 0.9, adapter at
    // zero, 6 crops:
    //
    //     target = (noise - x0)   mean loss 11.167
    //     target = (x0 - noise)   mean loss  3.593   <- 67.8% lower, all 6 agree
    //
    // I had argued from mm3-dit-graph.h's Euler step (`x + (sigma_next - sigma)*v`
    // with sigma rising 0->1) that it should be (noise - x0). That was WRONG.
    // SimpleTuner's flow_matching_target_direction = -1.0 agrees with the
    // measurement. Re-run --sign-check before ever flipping this.
    //
    // The margin is also the strongest evidence the rest of the path is right: a
    // stock DiT could not score 3.59 against one target and 11.17 against its
    // negation unless the conditioning, the crop alignment and the forward were
    // all actually working.
    //
    // Declared HERE rather than at the loop because the eval below must score
    // against the same target; two different signs would make eval and training
    // loss silently incomparable.
    const bool positive_target = false;  // (x0 - noise)

    // ── eval set ────────────────────────────────────────────────────────────
    //
    // Training loss cannot answer "is this run better than the last one", for
    // two reasons that both bit run 01/02:
    //
    //   1. It is measured at sigmas drawn from logit_normal(logit_mean), so
    //      CHANGING logit_mean changes the loss surface itself. Run 02's 1.53
    //      and run 01's 2.01 are not the same quantity and must never be
    //      subtracted.
    //   2. Most of it is an irreducible floor. The target is (x0 - noise) and
    //      the noise term is unguessable, so a large constant sits under every
    //      number and DIVIDES OUT any real improvement when read as a percent.
    //
    // The fix is a fixed eval set with sigma on a STRATIFIED GRID rather than
    // sampled: (i + 0.5)/N across (0,1), the same grid for every run whatever
    // logit_mean is. Same songs, same crops, same noise, same sigmas => the
    // number is comparable across runs and across configurations.
    //
    // It draws from its OWN rng. Reusing the training rng would consume draws
    // and change the training trajectory, which would silently break the
    // controlled comparison this exists to support.
    // Eval window is PINNED (a.eval_crop), not the training crop. A run at crop
    // 1378 must still be comparable to one at 689; if the eval followed the
    // training crop, changing it would move the yardstick with the thing being
    // measured and every crop experiment would be uninterpretable.
    const int64_t EC = a.eval_crop > 0 ? a.eval_crop : a.crop;
    struct MM3EvalItem { const MM3TrainSong * song; int64_t start; float sigma; uint64_t nseed; };
    std::vector<MM3EvalItem> evalset;
    if (a.eval_every > 0) {
        std::mt19937_64 erng(a.seed ^ 0xE7A1ULL);
        for (int64_t i = 0; i < a.eval_n; i++) {
            const MM3TrainSong * s = nullptr;
            for (;;) {
                const MM3TrainSong & cand = songs[erng() % songs.size()];
                if (cand.L > EC) { s = &cand; break; }
            }
            MM3EvalItem it;
            it.song  = s;
            it.start = (int64_t) (erng() % (uint64_t) (s->L - EC));
            it.sigma = (float) ((i + 0.5) / (double) a.eval_n);
            it.nseed = 0xE0A1ULL + (uint64_t) i;
            evalset.push_back(it);
        }
        fprintf(stderr, "[mm3-train] eval set: %lld fixed crops of %lld frames (%.1f s), sigma on a "
                        "stratified grid, every %lld steps\n",
                (long long) evalset.size(), (long long) EC, (double) EC / 86.1328125,
                (long long) a.eval_every);
    }

    // Forward-only pass over the eval set. Reports the mean plus three sigma
    // bands, because WHERE the loss moves is the actual diagnostic: the whole
    // point of logit_mean is to buy improvement at LOW sigma, and a run that
    // only improves the high band has learned the genre marginal again.
    std::vector<float> ex0, econd, enoise, ext, etgt, efour;
    auto run_eval = [&](int64_t at_step) -> bool {
        double sum = 0.0, band[3] = {0, 0, 0};
        int    cnt[3] = {0, 0, 0};
        for (const MM3EvalItem & it : evalset) {
            if (!mm3_train_read_crop(*it.song, it.start, EC, IC, &ex0, &econd, &err)) return false;
            enoise.resize(ex0.size());
            mm3_fill_noise_train(&enoise, it.nseed);
            mm3_train_fourier(g_mm3_dit.fourier_w, 1.0f - it.sigma, &efour);
            mm3_make_xt_target(ex0, enoise, it.sigma, positive_target, &ext, &etgt);
            float lv = 0.0f;
            if (!mm3_train_micro(m, tsched, ad, &opt, ext, econd, etgt, efour, EC, false, &lv, &err))
                return false;
            sum += lv;
            const int b = it.sigma < 0.33f ? 0 : (it.sigma < 0.67f ? 1 : 2);
            band[b] += lv; cnt[b]++;
        }
        fprintf(stderr, "[mm3-eval] step %4lld  loss %.5f   sigma<0.33 %.5f | 0.33-0.67 %.5f | >0.67 %.5f\n",
                (long long) at_step, sum / (double) evalset.size(),
                cnt[0] ? band[0] / cnt[0] : 0.0, cnt[1] ? band[1] / cnt[1] : 0.0,
                cnt[2] ? band[2] / cnt[2] : 0.0);
        return true;
    };

    // ── sign check ──
    //
    // With the adapter at zero the model is the stock DiT. At high sigma it must
    // score better against the CORRECT velocity target than the negated one; a
    // model that can denoise cannot be indifferent to the direction.
    if (a.sign_check) {
        double sum_pos = 0.0, sum_neg = 0.0;
        const int N = 6;
        for (int i = 0; i < N; i++) {
            const MM3TrainSong * s; int64_t st;
            sample_crop(&s, &st);
            if (!mm3_train_read_crop(*s, st, a.crop, IC, &x0, &cond, &err)) {
                fprintf(stderr, "[mm3-train] %s\n", err.c_str()); return 1;
            }
            noise.resize(x0.size());
            mm3_fill_noise_train(&noise, a.seed * 1000 + i);
            const float sigma = 0.9f;                       // high noise
            mm3_train_fourier(g_mm3_dit.fourier_w, 1.0f - sigma, &fourier);
            float lp = 0.0f, ln = 0.0f;
            mm3_make_xt_target(x0, noise, sigma, true,  &xt, &tgt);
            if (!mm3_train_micro(m, tsched, ad, &opt, xt, cond, tgt, fourier, a.crop, false, &lp, &err)) {
                fprintf(stderr, "[mm3-train] %s\n", err.c_str()); return 1;
            }
            mm3_make_xt_target(x0, noise, sigma, false, &xt, &tgt);
            if (!mm3_train_micro(m, tsched, ad, &opt, xt, cond, tgt, fourier, a.crop, false, &ln, &err)) {
                fprintf(stderr, "[mm3-train] %s\n", err.c_str()); return 1;
            }
            sum_pos += lp; sum_neg += ln;
            fprintf(stderr, "[sign-check] %d/%d  (noise-x0) %.5f   (x0-noise) %.5f\n", i + 1, N, lp, ln);
        }
        const double mp = sum_pos / N, mn = sum_neg / N;
        fprintf(stderr, "\n[sign-check] mean loss: (noise - x0) %.5f | (x0 - noise) %.5f\n", mp, mn);
        const double rel = std::fabs(mp - mn) / std::max(1e-9, std::max(mp, mn));
        if (rel < 0.05) {
            fprintf(stderr, "[sign-check] INCONCLUSIVE — the two targets differ by only %.1f%%. A model "
                            "that can denoise should not be indifferent; suspect the conditioning, the "
                            "crop alignment or the forward, NOT the sign.\n", 100.0 * rel);
            return 1;
        }
        fprintf(stderr, "[sign-check] VERDICT: target = %s  (%.1f%% lower loss)\n",
                mp < mn ? "(noise - x0)" : "(x0 - noise)", 100.0 * rel);
        return 0;
    }

    // ── gradient checkpointing ──
    MM3CkptPlan ckpt_plan;
    MM3CkptBufs ckpt_bufs;
    const bool  use_ckpt = a.ckpt_segments > 1;
    if (use_ckpt || a.ckpt_verify) {
        const int n_blk = (int) m.synth.dit.blk.size();
        ckpt_plan = mm3_ckpt_plan(n_blk, (int) std::max<int64_t>(2, a.ckpt_segments));
        if (!mm3_ckpt_alloc(&ckpt_bufs, m.backend, (int64_t) m.synth_cfg.dit.embedding_length,
                            a.crop + 1, ckpt_plan.segments, &err)) {
            fprintf(stderr, "[mm3-train] %s\n", err.c_str()); return 1;
        }
        fprintf(stderr, "[mm3-train] gradient checkpointing: %d segments over %d blocks, "
                        "boundaries [%lld, %lld] f32\n",
                ckpt_plan.segments, n_blk, (long long) m.synth_cfg.dit.embedding_length,
                (long long) (a.crop + 1));
    }

    // ACCEPTANCE GATE. Same crop, same noise, same sigma, both paths. If these
    // disagree beyond float noise the port is wrong and nothing trained with it
    // can be trusted -- so this refuses to continue rather than warn.
    if (a.ckpt_verify) {
        const MM3TrainSong * s; int64_t st;
        sample_crop(&s, &st);
        if (!mm3_train_read_crop(*s, st, a.crop, IC, &x0, &cond, &err)) {
            fprintf(stderr, "[mm3-train] %s\n", err.c_str()); return 1;
        }
        noise.resize(x0.size());
        mm3_fill_noise_train(&noise, a.seed * 7919 + 1);
        const float sigma = mm3_sigma_from(rng, a.logit_mean, a.logit_std);
        mm3_train_fourier(g_mm3_dit.fourier_w, 1.0f - sigma, &fourier);
        mm3_make_xt_target(x0, noise, sigma, positive_target, &xt, &tgt);

        float l_mono = 0.0f, l_ckpt = 0.0f;
        lm_optim_zero_grad(&opt);
        if (!mm3_train_micro(m, tsched, ad, &opt, xt, cond, tgt, fourier, a.crop, true, &l_mono, &err)) {
            fprintf(stderr, "[mm3-train] verify (monolithic): %s\n", err.c_str()); return 1;
        }
        lm_optim_zero_grad(&opt);
        if (!mm3_train_micro_ckpt(m, tsched, ad, &opt, xt, cond, tgt, fourier, a.crop, ckpt_plan,
                                  &ckpt_bufs, &l_ckpt, &err)) {
            fprintf(stderr, "[mm3-train] verify (checkpointed): %s\n", err.c_str()); return 1;
        }
        lm_optim_zero_grad(&opt);
        const double rel = std::fabs((double) l_mono - (double) l_ckpt)
                         / std::max(1e-9, std::fabs((double) l_mono));
        fprintf(stderr, "[ckpt-verify] monolithic %.7f | checkpointed %.7f | rel %.3e (%d segments)\n",
                (double) l_mono, (double) l_ckpt, rel, ckpt_plan.segments);
        if (rel > 1e-4) {
            fprintf(stderr, "[ckpt-verify] FAIL — the two paths disagree. The port is wrong; "
                            "do not train with --ckpt-segments.\n");
            mm3_ckpt_free(&ckpt_bufs);
            return 1;
        }
        fprintf(stderr, "[ckpt-verify] PASS\n");
        if (!use_ckpt) { mm3_ckpt_free(&ckpt_bufs); return 0; }
    }

    // ── training ──
    // The target sign is `positive_target`, declared above with its evidence.
    //
    // Eval at step 0 too: that is the UNTRAINED baseline every later number is
    // read against, and without it a run can only be compared to other runs.
    if (a.eval_every > 0 && !run_eval(0)) {
        fprintf(stderr, "[mm3-train] eval failed: %s\n", err.c_str()); return 1;
    }
    for (int64_t step = 1; step <= a.steps; step++) {
        double acc = 0.0;
        lm_optim_zero_grad(&opt);   // accumulators are per optimizer window
        for (int64_t g = 0; g < a.grad_accum; g++) {
            const MM3TrainSong * s; int64_t st;
            sample_crop(&s, &st);
            if (!mm3_train_read_crop(*s, st, a.crop, IC, &x0, &cond, &err)) {
                fprintf(stderr, "[mm3-train] %s\n", err.c_str()); return 1;
            }
            noise.resize(x0.size());
            mm3_fill_noise_train(&noise, a.seed * 7919 + (uint64_t) (step * a.grad_accum + g));
            const float sigma = mm3_sigma_from(rng, a.logit_mean, a.logit_std);
            mm3_train_fourier(g_mm3_dit.fourier_w, 1.0f - sigma, &fourier);
            mm3_make_xt_target(x0, noise, sigma, positive_target, &xt, &tgt);
            float lv = 0.0f;
            const bool mok = use_ckpt
                ? mm3_train_micro_ckpt(m, tsched, ad, &opt, xt, cond, tgt, fourier, a.crop, ckpt_plan,
                                       &ckpt_bufs, &lv, &err)
                : mm3_train_micro(m, tsched, ad, &opt, xt, cond, tgt, fourier, a.crop, true, &lv, &err);
            if (!mok) { fprintf(stderr, "[mm3-train] %s\n", err.c_str()); return 1; }
            acc += lv;
        }
        LmStepStats st{};
        if (getenv("MM3_TRAIN_TRACE")) fprintf(stderr, "[trace] %s\n", "pre optim step");
        if (!lm_optim_step(&opt, osched, &st)) {
            fprintf(stderr, "[mm3-train] optimizer step failed\n"); return 1;
        }
        if (a.dora && !mm3_train_dora_refresh(m, ad, osched, &err)) {
            fprintf(stderr, "[mm3-train] %s\n", err.c_str()); return 1;
        }
        if (getenv("MM3_TRAIN_TRACE")) fprintf(stderr, "[trace] %s\n", "post optim step");
        fprintf(stderr, "[mm3-train] step %4lld  loss %.5f  |g| %.4f  lr %.2e\n", (long long) step,
                acc / (double) a.grad_accum, (double) st.grad_norm, (double) st.lr);

        if (a.eval_every > 0 && step % a.eval_every == 0 && !run_eval(step)) {
            fprintf(stderr, "[mm3-train] eval failed: %s\n", err.c_str()); return 1;
        }
        if (a.export_every > 0 && step % a.export_every == 0 && step < a.steps && !a.out_dir.empty()) {
            pm_mkdir_p(a.out_dir);
            char snap[64];
            snprintf(snap, sizeof(snap), "/mm3_lora_step%lld.safetensors", (long long) step);
            const std::string sp = a.out_dir + snap;
            if (!mm3_train_export(ad, m, a.rank, a.alpha, sp, &err)) {
                fprintf(stderr, "[mm3-train] snapshot export failed: %s\n", err.c_str()); return 1;
            }
        }
    }

    if (!a.out_dir.empty()) {
        pm_mkdir_p(a.out_dir);
        const std::string out = a.out_dir + "/mm3_lora.safetensors";
        if (!mm3_train_export(ad, m, a.rank, a.alpha, out, &err)) {
            fprintf(stderr, "[mm3-train] export failed: %s\n", err.c_str());
            return 1;
        }
        fprintf(stderr, "[mm3-train] load it with: MM3_ADAPTER=%s\n", out.c_str());
        mm3_ckpt_free(&ckpt_bufs);
    }
    return 0;
}

// ── Export ─────────────────────────────────────────────────────────────────
//
// Writes the trained LoRA as a single safetensors file in the COMFYUI form,
// because that is the one mm3-adapter.h parses whose QKV is FUSED — exactly how
// this trainer parameterises it. The diffusers form splits q/k/v into three
// modules and would need the fused factor decomposed, which is not possible in
// general.
//
// Round-tripping through mm3-adapter.h is the permanent gate: an adapter this
// writes must load back and reproduce the trainer's own loss.
//
// ggml ne order is reversed vs torch, so a:[in, rank] IS torch [rank, in] and
// b:[rank, out] IS torch [out, rank] — the bytes need no transposition.
static bool mm3_train_export(const MM3TrainAdapters & ad, const MM3Model & m, int64_t rank, float alpha,
                             const std::string & path, std::string * err) {
    // shape is torch order; a 0-dim entry is a scalar.
    struct Ent { std::string name; std::vector<int64_t> shape; std::vector<float> data; };
    std::vector<Ent> ents;
    bool             dora = false;

    auto fetch = [](const ggml_tensor * t) {
        std::vector<float> v((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        return v;
    };
    auto add_site = [&](const MM3TrainLora & lo, const std::string & mod) {
        if (!lo.on()) return;
        // ggml ne reversed is the torch shape
        ents.push_back({ mod + ".lora_A.weight", { lo.a->ne[1], lo.a->ne[0] }, fetch(lo.a) });
        ents.push_back({ mod + ".lora_B.weight", { lo.b->ne[1], lo.b->ne[0] }, fetch(lo.b) });
        if (!lo.m) return;
        dora = true;
        // PEFT name and layout: F32 [out]. A row the forward holds at factor 1
        // (the mlpv q,k rows) exports its own norm, so the merge leaves it at
        // the base even though weight decay has moved the unused m.
        std::vector<float> mv = fetch(lo.m);
        if (lo.keep) {
            const std::vector<float> nv = fetch(lo.nrm), kv = fetch(lo.keep);
            for (size_t r = 0; r < mv.size(); r++) {
                if (kv[r] != 0.0f) mv[r] = nv[r];
            }
        }
        ents.push_back({ mod + ".lora_magnitude_vector.weight", { lo.m->ne[0] }, std::move(mv) });
        // The loader takes alpha from per-module scalars, else adapter_config,
        // else scaling 1. A DoRA norm is only right at the trained scaling.
        ents.push_back({ mod + ".alpha", {}, { alpha } });
    };
    const std::string P = "diffusion_model.diffusion_transformer.";
    for (size_t i = 0; i < ad.blk.size(); i++) {
        const std::string b = P + "transformer.layers." + std::to_string(i);
        add_site(ad.blk[i].qkv,      b + ".self_attn.to_qkv");
        add_site(ad.blk[i].attn_out, b + ".self_attn.to_out");
        add_site(ad.blk[i].ff_in,    b + ".ff.ff.0.proj");
        add_site(ad.blk[i].ff_out,   b + ".ff.ff.2");
    }
    add_site(ad.proj_in,  P + "transformer.project_in");
    add_site(ad.proj_out, P + "transformer.project_out");

    // header + payload
    std::string hdr = "{";
    size_t      off = 0;
    for (const Ent & e : ents) {
        const size_t nb = e.data.size() * sizeof(float);
        std::string  shp;
        for (int64_t d : e.shape) shp += (shp.empty() ? "" : ",") + std::to_string(d);
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "%s\"%s\":{\"dtype\":\"F32\",\"shape\":[%s],\"data_offsets\":[%zu,%zu]}",
                 off ? "," : "", e.name.c_str(), shp.c_str(), off, off + nb);
        hdr += buf;
        off += nb;
    }
    {
        char buf[384];
        // DoRA files carry the format tag mm3-adapter.h validates against:
        // W' = m * (W + s*BA) / max(||W + s*BA||_col, 1e-12), s = alpha/rank,
        // strength u gives W0 + u*(W' - W0).
        snprintf(buf, sizeof(buf), ",\"__metadata__\":{\"format\":\"pt\",\"hot_step_mm3_lora\":\"1\","
                                   "\"rank\":\"%lld\",\"alpha\":\"%.6f\"%s}",
                 (long long) rank, (double) alpha,
                 dora ? ",\"use_dora\":\"true\",\"hot_step_mm3_dora\":\"dit-dora-v1\"" : "");
        hdr += buf;
    }
    hdr += "}";
    while ((8 + hdr.size()) % 8) hdr += " ";

    FILE * f = hs_fopen(path, "wb");
    if (!f) { if (err) *err = "cannot write " + path; return false; }
    const uint64_t hlen = (uint64_t) hdr.size();
    fwrite(&hlen, sizeof(hlen), 1, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    for (const Ent & e : ents) fwrite(e.data.data(), sizeof(float), e.data.size(), f);
    fclose(f);
    fprintf(stderr, "[mm3-train] exported %zu tensors -> %s\n", ents.size(), path.c_str());
    (void) m;
    return true;
}
