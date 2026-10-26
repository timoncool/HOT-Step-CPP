// sa3-train-model.h: trainable SA3 medium DiT for ace-train sa3-train.
//
// The official LoRA recipe (stable_audio_3 scripts/train_lora.py, upstream
// 77e0398c) puts a DoRA-rows adapter on every Linear/Conv1d of the DiT (24
// blocks x 9 sites + 12 outer layers = 228) and on the seconds_total number
// embedder (conditioner.conditioners.seconds_total.embedder.embedding.1). The
// base DiT stays frozen as BF16; adapters are f32 masters on the host.
//
// Graph layout (batch 1, one fixed crop length, so every shape is fixed):
//   head  : seconds embed, cross/global/timestep projections, global_cond
//           embedder, preprocess conv + project_in + memory tokens -> x_0
//   block : sa3_train_build_layer (train/sa3-train-gate.h), x_l -> x_{l+1}
//   tail  : drop memory tokens, project_out + postprocess conv, masked MSE
// A step runs the head and all blocks forward once, keeping only the block
// boundaries x_l, then backpropagates tail -> block 23 ... block 0 -> head,
// recomputing each block's activations inside its own backward graph. Peak
// activation memory is one block. dL/dx is handed between graphs through two
// device carriers; the cross and global_cond gradients every block adds are
// summed on the device and fed to the head's backward.
//
// The seconds_total adapter is outside the DiT and the dora-rows-v1
// inference contract (train/sa3-adapter-export.h) cannot carry it, so it is
// built only when Sa3TrainModel::train_seconds is set (fixture parity, or a
// future inference path); otherwise the embedder is the frozen base Linear,
// which is what inference applies.

#pragma once

#include "train/sa3-train-gate.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// One adapted Linear. Host arrays are torch row-major, which is also the ggml
// layout of the device tensors: A (r, in) = [in, r], B (out, r) = [r, out].
struct Sa3Site {
    std::string   stem;   // export stem (dit.*), "seconds" for the embedder
    std::string   param;  // upstream parameter prefix, for fixtures
    ggml_tensor * w = nullptr;  // frozen base weight [in, out] (BF16; F32 on a CPU backend)
    ggml_tensor ** slot = nullptr;  // where the DiT struct keeps it
    int64_t       in = 0, out = 0;
    Sa3DoraRows   d;      // device A, B, m, base_nsq
    std::vector<float> a, b, m, nsq;     // host masters + ||W_row||^2
    std::vector<float> ga, gb, gm;       // last step's gradients
};

struct Sa3TrainModel {
    SA3DiT        dit{};
    SA3DiTConfig  c{};
    ggml_backend_t backend = nullptr;
    WeightCtx     sec_wctx{};
    ggml_context *        up_ctx = nullptr;  // F32 copies of the site weights (CPU backend)
    ggml_backend_buffer_t up_buf = nullptr;
    ggml_tensor * sec_w = nullptr, * sec_b = nullptr;  // seconds embedder Linear(256 -> 768)
    int64_t       rank = 16;
    float         alpha = 16.0f;
    bool          train_seconds = false;
    int64_t       T = 0, S = 0, S_c = 0;
    std::vector<Sa3Site> sites;     // 24*9 block sites, then 12 outer, then seconds (optional)
    Sa3LayerAdapters layer_ad[SA3_DIT_MAX_LAYERS];
    int           outer0 = 0;            // index of the first outer site

    // persistent device state
    ggml_context *        st_ctx = nullptr;
    ggml_backend_buffer_t st_buf = nullptr;
    ggml_tensor * x_t = nullptr, * target = nullptr, * w_loss = nullptr;
    ggml_tensor * cross_base = nullptr, * tfeat = nullptr, * sfeat = nullptr, * keep = nullptr;
    ggml_tensor * local = nullptr, * positions = nullptr, * vmask = nullptr, * memfull = nullptr;
    ggml_tensor * cross_p = nullptr, * gcond_p = nullptr, * gcross = nullptr, * ggcond = nullptr;
    std::vector<ggml_tensor *> xs;       // depth + 1 block boundaries [dim, S]
    ggml_tensor * carrier[2] = { nullptr, nullptr };
    ggml_gallocr_t alloc = nullptr;
    std::vector<uint8_t> arena;
    std::vector<float>   out_host;       // last step's DiT output, token-major [io, T]
    bool          audited = false;
};

static ggml_tensor * sa3t_site_lin(Sa3TrainModel & m, int s, ggml_context * ctx, ggml_tensor * x) {
    Sa3Site & st = m.sites[(size_t) s];
    return sa3_train_lin(ctx, st.w, x, &st.d);
}

// Upstream DoRA-rows init (lora/model.py): A kaiming_uniform(a=sqrt(5)) =
// U(-1/sqrt(in), 1/sqrt(in)), B = 0, magnitude = ||W0_row|| of the bf16 base.
// DiffusionCondTrainingWrapper then casts the whole model to bf16 and restores
// the adapters to f32 (cast_base_to_precision), so A and the magnitude start
// bf16-rounded. `uni` returns U[0, 1).
template <typename Uni>
static void sa3t_site_init(Sa3Site & st, int64_t rank, Uni && uni) {
    const float bound = 1.0f / std::sqrt((float) st.in);
    st.a.resize((size_t) (rank * st.in));
    for (float & v : st.a) v = ggml_bf16_to_fp32(ggml_fp32_to_bf16((float) ((2.0 * uni() - 1.0) * bound)));
    st.b.assign((size_t) (st.out * rank), 0.0f);
    st.m.resize((size_t) st.out);
    for (int64_t o = 0; o < st.out; o++)
        st.m[(size_t) o] = ggml_bf16_to_fp32(ggml_fp32_to_bf16((float) std::sqrt((double) st.nsq[(size_t) o])));
}

static std::vector<float> sa3t_get(ggml_tensor * t) {
    std::vector<float> v((size_t) ggml_nelements(t));
    if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> h(v.size());
        ggml_backend_tensor_get(t, h.data(), 0, ggml_nbytes(t));
        ggml_bf16_to_fp32_row(h.data(), v.data(), (int64_t) v.size());
    } else {
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    }
    return v;
}

// Load the frozen base, enumerate sites, allocate device state for crop length T.
static bool sa3t_model_load(Sa3TrainModel & m, const std::string & base_path, int64_t T, int64_t rank, float alpha,
                            bool train_seconds, std::string * err) {
    if (!sa3_dit_load(&m.dit, base_path.c_str())) {
        *err = "cannot load base DiT " + base_path;
        return false;
    }
    m.c = m.dit.cfg;
    m.backend = m.dit.backend;
    m.rank = rank;
    m.alpha = alpha;
    m.train_seconds = train_seconds;
    const SA3DiTConfig & c = m.c;
    m.T = T;
    m.S = c.n_memory_tokens + T;
    m.S_c = 256 + 1;  // prompt tokens + the seconds_total token

    {   // seconds embedder weights ride in the DiT GGUF (convert-sa3.py conditioner.* extras)
        GGUFModel gf = {};
        if (!gf_load(&gf, base_path.c_str())) {
            *err = "cannot reopen " + base_path;
            return false;
        }
        wctx_init(&m.sec_wctx, 2);
        m.sec_w = gf_load_tensor(&m.sec_wctx, gf, "conditioner.conditioners.seconds_total.embedder.embedding.1.weight");
        m.sec_b = gf_load_tensor_f32(&m.sec_wctx, gf, "conditioner.conditioners.seconds_total.embedder.embedding.1.bias");
        const bool ok = wctx_alloc(&m.sec_wctx, m.backend);
        gf_close(&gf);
        if (!ok) {
            *err = "cannot allocate the seconds embedder";
            return false;
        }
    }

    static const char * const export_tail[SA3_SITE_COUNT] = {
        "self.qkv", "self.out", "cross.q", "cross.kv", "cross.out", "ff.proj", "ff.out", "local.0", "local.2",
    };
    const std::string up = "diffusion.model.model.";
    for (int l = 0; l < c.depth; l++) {
        SA3DiTLayer & ly = m.dit.layers[l];
        ggml_tensor ** w[SA3_SITE_COUNT] = { &ly.attn_qkv, &ly.attn_out, &ly.cross_q, &ly.cross_kv, &ly.cross_out,
                                             &ly.ff0_w,    &ly.ff2_w,    &ly.loc0_w,  &ly.loc2_w };
        for (int s = 0; s < SA3_SITE_COUNT; s++) {
            Sa3Site st;
            st.stem  = "dit." + std::to_string(l) + "." + export_tail[s];
            st.param = up + "transformer.layers." + std::to_string(l) + "." + sa3_train_site_names[s];
            st.slot  = w[s];
            m.sites.push_back(st);
        }
    }
    m.outer0 = (int) m.sites.size();
    struct Outer { const char * stem, * param; ggml_tensor ** w; };
    const Outer outer[] = {
        { "dit.pre_conv", "preprocess_conv", &m.dit.pre_conv_w },
        { "dit.post_conv", "postprocess_conv", &m.dit.post_conv_w },
        { "dit.proj_in", "transformer.project_in", &m.dit.proj_in_w },
        { "dit.proj_out", "transformer.project_out", &m.dit.proj_out_w },
        { "dit.cond_embed.0", "to_cond_embed.0", &m.dit.cond0_w },
        { "dit.cond_embed.2", "to_cond_embed.2", &m.dit.cond2_w },
        { "dit.global_embed.0", "to_global_embed.0", &m.dit.glob0_w },
        { "dit.global_embed.2", "to_global_embed.2", &m.dit.glob2_w },
        { "dit.time_embed.0", "to_timestep_embed.0", &m.dit.ts0_w },
        { "dit.time_embed.2", "to_timestep_embed.2", &m.dit.ts2_w },
        { "dit.gce.0", "transformer.global_cond_embedder.0", &m.dit.gc0_w },
        { "dit.gce.2", "transformer.global_cond_embedder.2", &m.dit.gc2_w },
    };
    for (const Outer & o : outer) {
        Sa3Site st;
        st.stem  = o.stem;
        st.param = up + o.param;
        st.slot  = o.w;
        m.sites.push_back(st);
    }
    if (train_seconds) {
        Sa3Site st;
        st.stem  = "seconds";
        st.param = "diffusion.conditioner.conditioners.seconds_total.embedder.embedding.1";
        st.slot  = &m.sec_w;
        m.sites.push_back(st);
    }

    // The DoRA norm term W.A gets its gradient through a transpose, so GGML_BACKWARD_MM
    // keeps that arm an OUT_PROD over the frozen weight; ggml-cpu has no BF16 OUT_PROD
    // (ggml-cuda does). On a CPU backend the site weights are therefore held as F32
    // copies of the same BF16 values.
    if (ggml_backend_dev_type(ggml_backend_get_device(m.backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        ggml_init_params uip{ (m.sites.size() + 1) * ggml_tensor_overhead(), nullptr, true };
        m.up_ctx = ggml_init(uip);
        std::vector<ggml_tensor *> up;
        for (Sa3Site & st : m.sites) up.push_back(ggml_new_tensor_2d(m.up_ctx, GGML_TYPE_F32, (*st.slot)->ne[0], (*st.slot)->ne[1]));
        m.up_buf = ggml_backend_alloc_ctx_tensors(m.up_ctx, m.backend);
        if (!m.up_buf) {
            *err = "cannot allocate F32 base weights";
            return false;
        }
        for (size_t i = 0; i < m.sites.size(); i++) {
            const std::vector<float> w = sa3t_get(*m.sites[i].slot);
            ggml_backend_tensor_set(up[i], w.data(), 0, ggml_nbytes(up[i]));
            *m.sites[i].slot = up[i];
        }
    }
    for (Sa3Site & st : m.sites) st.w = *st.slot;

    // device state: adapters, step inputs, boundaries, carriers
    const size_t n_t = m.sites.size() * 4 + c.depth + 32;
    ggml_init_params ip{ n_t * ggml_tensor_overhead(), nullptr, true };
    m.st_ctx = ggml_init(ip);
    ggml_context * sc = m.st_ctx;
    auto f2 = [&](int64_t a, int64_t b) { return ggml_new_tensor_2d(sc, GGML_TYPE_F32, a, b); };
    auto f1 = [&](int64_t a) { return ggml_new_tensor_1d(sc, GGML_TYPE_F32, a); };
    const float scale = alpha / (float) rank;
    for (Sa3Site & st : m.sites) {
        st.in  = st.w->ne[0];
        st.out = st.w->ne[1];
        st.d.a = f2(st.in, rank);
        st.d.b = f2(rank, st.out);
        st.d.m = f1(st.out);
        st.d.base_nsq = f1(st.out);
        st.d.scale = scale;
        ggml_set_param(st.d.a);
        ggml_set_param(st.d.b);
        ggml_set_param(st.d.m);
    }
    const int64_t dim = c.dim, io = c.io_channels, S = m.S;
    m.x_t        = f2(io, T);
    m.target     = f2(io, T);
    m.w_loss     = f2(1, T);
    m.cross_base = f2(c.cond_token_dim, m.S_c);
    m.tfeat      = f2(c.ts_feat_dim, 1);
    m.sfeat      = f2(256, 1);
    m.keep       = f2(1, 1);
    m.local      = f2(c.local_cond_dim, T);
    m.positions  = ggml_new_tensor_1d(sc, GGML_TYPE_I32, S);
    m.vmask      = ggml_new_tensor_3d(sc, GGML_TYPE_F32, 1, 1, S);
    m.memfull    = f2(dim, S);
    m.cross_p    = f2(dim, m.S_c);
    m.gcond_p    = f2(6 * dim, 1);
    m.gcross     = f2(dim, m.S_c);
    m.ggcond     = f2(6 * dim, 1);
    for (int l = 0; l <= c.depth; l++) m.xs.push_back(f2(dim, S));
    m.carrier[0] = f2(dim, S);
    m.carrier[1] = f2(dim, S);
    m.st_buf = ggml_backend_alloc_ctx_tensors(sc, m.backend);
    if (!m.st_buf) {
        *err = "cannot allocate trainer state on " + std::string(ggml_backend_name(m.backend));
        return false;
    }

    // constants: base row norms, positions, memory-token canvas
    for (Sa3Site & st : m.sites) {
        const std::vector<float> w = sa3t_get(st.w);
        st.nsq.resize((size_t) st.out);
        for (int64_t o = 0; o < st.out; o++) {
            double acc = 0.0;
            for (int64_t i = 0; i < st.in; i++) acc += (double) w[(size_t) (o * st.in + i)] * w[(size_t) (o * st.in + i)];
            st.nsq[(size_t) o] = (float) acc;
        }
        ggml_backend_tensor_set(st.d.base_nsq, st.nsq.data(), 0, ggml_nbytes(st.d.base_nsq));
    }
    std::vector<int32_t> pos((size_t) S);
    for (int64_t i = 0; i < S; i++) pos[(size_t) i] = (int32_t) i;
    ggml_backend_tensor_set(m.positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    std::vector<float> mem((size_t) (dim * S), 0.0f);
    const std::vector<float> tok = sa3t_get(m.dit.memory_toks);  // [dim, n_mem]
    std::copy(tok.begin(), tok.end(), mem.begin());
    ggml_backend_tensor_set(m.memfull, mem.data(), 0, mem.size() * sizeof(float));

    for (int l = 0; l < c.depth; l++)
        for (int s = 0; s < SA3_SITE_COUNT; s++) m.layer_ad[l].site[s] = m.sites[(size_t) (l * SA3_SITE_COUNT + s)].d;
    m.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    const size_t nodes = 16384;
    m.arena.resize(ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, true));
    return true;
}

static void sa3t_model_free(Sa3TrainModel & m) {
    if (m.alloc) ggml_gallocr_free(m.alloc);
    if (m.up_buf) ggml_backend_buffer_free(m.up_buf);
    if (m.up_ctx) ggml_free(m.up_ctx);
    if (m.st_buf) ggml_backend_buffer_free(m.st_buf);
    if (m.st_ctx) ggml_free(m.st_ctx);
    if (m.sec_wctx.buffer) ggml_backend_buffer_free(m.sec_wctx.buffer);
    if (m.sec_wctx.ctx) ggml_free(m.sec_wctx.ctx);
    if (m.dit.sched) sa3_dit_free(&m.dit);
    m = Sa3TrainModel{};
}

static void sa3t_upload_adapters(Sa3TrainModel & m) {
    for (Sa3Site & st : m.sites) {
        ggml_backend_tensor_set(st.d.a, st.a.data(), 0, ggml_nbytes(st.d.a));
        ggml_backend_tensor_set(st.d.b, st.b.data(), 0, ggml_nbytes(st.d.b));
        ggml_backend_tensor_set(st.d.m, st.m.data(), 0, ggml_nbytes(st.d.m));
    }
}

// ── graphs ─────────────────────────────────────────────────────────────────

// Fresh graph context over the shared arena.
static ggml_context * sa3t_ctx(Sa3TrainModel & m) {
    ggml_init_params ip{ m.arena.size(), m.arena.data(), true };
    return ggml_init(ip);
}

struct Sa3HeadOut { ggml_tensor * cross, * gcond, * seq0; };

static Sa3HeadOut sa3t_build_head(Sa3TrainModel & m, ggml_context * ctx) {
    const SA3DiTConfig & c = m.c;
    SA3DiT &             d = m.dit;
    const int o = m.outer0;
    // seconds_total: NumberEmbedder Linear on host-computed Expo features
    ggml_tensor * sec = m.train_seconds ? sa3t_site_lin(m, (int) m.sites.size() - 1, ctx, m.sfeat)
                                        : ggml_mul_mat(ctx, m.sec_w, m.sfeat);
    sec = ggml_add(ctx, sec, m.sec_b);  // [768, 1]
    // cross = [prompt | seconds], all zero when CFG dropout fired (host zeroes the prompt rows)
    ggml_tensor * cin = ggml_acc(ctx, m.cross_base, ggml_mul(ctx, sec, m.keep), m.cross_base->nb[1], m.cross_base->nb[2],
                                 m.cross_base->nb[3], (size_t) (m.S_c - 1) * m.cross_base->nb[1]);
    ggml_tensor * cross = sa3t_site_lin(m, o + 5, ctx, ggml_silu(ctx, sa3t_site_lin(m, o + 4, ctx, cin)));
    ggml_tensor * glob  = sa3t_site_lin(m, o + 7, ctx, ggml_silu(ctx, sa3t_site_lin(m, o + 6, ctx, sec)));
    ggml_tensor * ts    = ggml_silu(ctx, ggml_add(ctx, sa3t_site_lin(m, o + 8, ctx, m.tfeat), d.ts0_b));
    ts                  = ggml_add(ctx, sa3t_site_lin(m, o + 9, ctx, ts), d.ts2_b);
    glob                = ggml_add(ctx, glob, ts);
    ggml_tensor * gc    = ggml_silu(ctx, ggml_add(ctx, sa3t_site_lin(m, o + 10, ctx, glob), d.gc0_b));
    gc                  = ggml_add(ctx, sa3t_site_lin(m, o + 11, ctx, gc), d.gc2_b);  // [6*dim, 1]
    ggml_tensor * h     = ggml_add(ctx, sa3t_site_lin(m, o + 0, ctx, m.x_t), m.x_t);
    h                   = sa3t_site_lin(m, o + 2, ctx, h);  // [dim, T]
    ggml_tensor * seq   = ggml_acc(ctx, m.memfull, h, m.memfull->nb[1], m.memfull->nb[2], m.memfull->nb[3],
                                   (size_t) c.n_memory_tokens * m.memfull->nb[1]);
    return { cross, gc, seq };
}

static ggml_tensor * sa3t_build_block(Sa3TrainModel & m, ggml_context * ctx, int l) {
    return sa3_train_build_layer(ctx, m.c, &m.dit.layers[l], &m.layer_ad[l], m.xs[(size_t) l], m.gcond_p, m.positions,
                                 m.vmask, m.cross_p, m.local, m.S, m.S_c);
}

// project_out + postprocess conv on the data rows of x_depth -> [io, T]
static ggml_tensor * sa3t_build_tail(Sa3TrainModel & m, ggml_context * ctx) {
    ggml_tensor * x    = m.xs.back();
    ggml_tensor * tail = ggml_cont(ctx, ggml_view_2d(ctx, x, m.c.dim, m.T, x->nb[1], (size_t) m.c.n_memory_tokens * x->nb[1]));
    ggml_tensor * o    = sa3t_site_lin(m, m.outer0 + 3, ctx, tail);
    return ggml_add(ctx, sa3t_site_lin(m, m.outer0 + 1, ctx, o), o);
}

static bool sa3t_run(Sa3TrainModel & m, ggml_cgraph * gf, const char * what, std::string * err, bool backward = true) {
    if (!m.audited) {
        std::string refused;
        const int bad = sa3_train_audit_graph(m.backend, gf, nullptr, &refused);
        if (bad) {
            *err = std::string(what) + ": " + ggml_backend_name(m.backend) + " refuses " + std::to_string(bad) +
                   " node(s): " + refused + " (no CPU fallback in the trainer)";
            return false;
        }
    }
    if (!ggml_gallocr_alloc_graph(m.alloc, gf)) {
        *err = std::string(what) + ": graph allocation failed (out of memory?)";
        return false;
    }
    if (backward) ggml_graph_reset(gf);
    if (ggml_backend_graph_compute(m.backend, gf) != GGML_STATUS_SUCCESS) {
        *err = std::string(what) + ": compute failed";
        return false;
    }
    return true;
}

static bool sa3t_read_grads(Sa3TrainModel & m, ggml_cgraph * gf, int s0, int s1, std::string * err) {
    for (int s = s0; s < s1; s++) {
        Sa3Site & st = m.sites[(size_t) s];
        ggml_tensor *  g[3] = { ggml_graph_get_grad(gf, st.d.a), ggml_graph_get_grad(gf, st.d.b),
                                ggml_graph_get_grad(gf, st.d.m) };
        if (!g[0] || !g[1] || !g[2]) {
            *err = "missing gradient for " + st.stem;
            return false;
        }
        st.ga = sa3t_get(g[0]);
        st.gb = sa3t_get(g[1]);
        st.gm = sa3t_get(g[2]);
    }
    return true;
}

static void sa3t_mark_grads(ggml_cgraph * gf, Sa3TrainModel & m, int s0, int s1) {
    for (int s = s0; s < s1; s++)
        for (ggml_tensor * p : { m.sites[(size_t) s].d.a, m.sites[(size_t) s].d.b, m.sites[(size_t) s].d.m })
            if (ggml_tensor * g = ggml_graph_get_grad(gf, p)) ggml_set_output(g);
}

// One forward + backward over the inputs already uploaded (sa3t_set_inputs).
// Fills every site's ga/gb/gm and out_host; returns the loss, NaN on error.
static double sa3t_step(Sa3TrainModel & m, std::string * err) {
    const int depth = m.c.depth;
    const int nsites = (int) m.sites.size();
    sa3t_upload_adapters(m);
    const size_t nodes = 16384;

    // F: head forward, then every block forward, keeping x_l.
    {
        ggml_context * ctx = sa3t_ctx(m);
        ggml_cgraph *  gf  = ggml_new_graph_custom(ctx, nodes, false);
        Sa3HeadOut     h   = sa3t_build_head(m, ctx);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, h.cross, m.cross_p));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, h.gcond, m.gcond_p));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, h.seq0, m.xs[0]));
        const bool ok = sa3t_run(m, gf, "head forward", err, false);
        ggml_free(ctx);
        if (!ok) return NAN;
    }
    for (int l = 0; l < depth; l++) {
        ggml_context * ctx = sa3t_ctx(m);
        ggml_cgraph *  gf  = ggml_new_graph_custom(ctx, nodes, false);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, sa3t_build_block(m, ctx, l), m.xs[(size_t) l + 1]));
        const bool ok = sa3t_run(m, gf, "block forward", err, false);
        ggml_free(ctx);
        if (!ok) return NAN;
    }

    // T: tail + loss; dL/dx_depth -> carrier[0]
    double loss = NAN;
    {
        ggml_context * ctx = sa3t_ctx(m);
        ggml_set_param(m.xs.back());
        ggml_tensor * out = sa3t_build_tail(m, ctx);
        ggml_set_output(out);
        ggml_tensor * d  = ggml_sub(ctx, out, m.target);
        ggml_tensor * ls = ggml_sum(ctx, ggml_mul(ctx, ggml_sqr(ctx, d), m.w_loss));
        ggml_set_output(ls);
        ggml_set_loss(ls);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, nodes, true);
        ggml_build_forward_expand(gf, ls);
        ggml_build_backward_expand(ctx, gf, nullptr);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_graph_get_grad(gf, m.xs.back()), m.carrier[0]));
        sa3t_mark_grads(gf, m, m.outer0 + 1, m.outer0 + 2);
        sa3t_mark_grads(gf, m, m.outer0 + 3, m.outer0 + 4);
        bool ok = sa3t_run(m, gf, "tail backward", err) && sa3t_read_grads(m, gf, m.outer0 + 1, m.outer0 + 2, err) &&
                  sa3t_read_grads(m, gf, m.outer0 + 3, m.outer0 + 4, err);
        if (ok) {
            float l = 0.0f;
            ggml_backend_tensor_get(ls, &l, 0, sizeof(float));
            loss = l;
            m.out_host = sa3t_get(out);
        }
        ggml_free(ctx);
        m.xs.back()->flags &= ~GGML_TENSOR_FLAG_PARAM;
        if (!ok) return NAN;
    }

    // B_l in reverse: recompute block l, backprop sum(y * dL/dy). The cross and
    // global_cond gradients of all blocks accumulate in gcross / ggcond.
    {
        std::vector<float> z((size_t) ggml_nelements(m.gcross), 0.0f);
        ggml_backend_tensor_set(m.gcross, z.data(), 0, ggml_nbytes(m.gcross));
        z.assign((size_t) ggml_nelements(m.ggcond), 0.0f);
        ggml_backend_tensor_set(m.ggcond, z.data(), 0, ggml_nbytes(m.ggcond));
    }
    int cur = 0;  // carrier holding dL/dx_{l+1}
    for (int l = depth - 1; l >= 0; l--) {
        ggml_context * ctx = sa3t_ctx(m);
        ggml_tensor *  x   = m.xs[(size_t) l];
        ggml_set_param(x);
        ggml_set_param(m.cross_p);
        ggml_set_param(m.gcond_p);
        ggml_tensor * ls = ggml_sum(ctx, ggml_mul(ctx, sa3t_build_block(m, ctx, l), m.carrier[cur]));
        ggml_set_loss(ls);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, nodes, true);
        ggml_build_forward_expand(gf, ls);
        ggml_build_backward_expand(ctx, gf, nullptr);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_graph_get_grad(gf, x), m.carrier[cur ^ 1]));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_add(ctx, m.gcross, ggml_graph_get_grad(gf, m.cross_p)), m.gcross));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_add(ctx, m.ggcond, ggml_graph_get_grad(gf, m.gcond_p)), m.ggcond));
        const int s0 = l * SA3_SITE_COUNT;
        sa3t_mark_grads(gf, m, s0, s0 + SA3_SITE_COUNT);
        const bool ok = sa3t_run(m, gf, "block backward", err) && sa3t_read_grads(m, gf, s0, s0 + SA3_SITE_COUNT, err);
        ggml_free(ctx);
        for (ggml_tensor * p : { x, m.cross_p, m.gcond_p }) p->flags &= ~GGML_TENSOR_FLAG_PARAM;
        if (!ok) return NAN;
        cur ^= 1;
    }

    // H: head backward against dL/dx_0, dL/dcross and dL/dglobal_cond
    {
        ggml_context * ctx = sa3t_ctx(m);
        Sa3HeadOut     h   = sa3t_build_head(m, ctx);
        ggml_tensor *  ls  = ggml_add(ctx, ggml_add(ctx, ggml_sum(ctx, ggml_mul(ctx, h.cross, m.gcross)),
                                                    ggml_sum(ctx, ggml_mul(ctx, h.gcond, m.ggcond))),
                                      ggml_sum(ctx, ggml_mul(ctx, h.seq0, m.carrier[cur])));
        ggml_set_loss(ls);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, nodes, true);
        ggml_build_forward_expand(gf, ls);
        ggml_build_backward_expand(ctx, gf, nullptr);
        // head sites: everything outer except proj_out / post_conv (tail), plus seconds
        std::vector<int> hs = { 0, 2, 4, 5, 6, 7, 8, 9, 10, 11 };
        for (int s : hs) sa3t_mark_grads(gf, m, m.outer0 + s, m.outer0 + s + 1);
        if (m.train_seconds) sa3t_mark_grads(gf, m, nsites - 1, nsites);
        bool ok = sa3t_run(m, gf, "head backward", err);
        for (int s : hs) ok = ok && sa3t_read_grads(m, gf, m.outer0 + s, m.outer0 + s + 1, err);
        if (m.train_seconds) ok = ok && sa3t_read_grads(m, gf, nsites - 1, nsites, err);
        ggml_free(ctx);
        if (!ok) return NAN;
    }
    m.audited = true;  // every graph shape has passed the placement audit once
    return loss;
}
