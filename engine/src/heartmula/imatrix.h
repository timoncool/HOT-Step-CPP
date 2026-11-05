#pragma once
// heartmula/imatrix.h: activation-importance ("imatrix") collection for the
// HeartMuLa global and local LMs, and the GGUF file engine/tools/quantize.cpp
// reads with --imatrix.
//
// HOT-Step file. The accumulator, eval-callback and file layout are those of
// yue2/yue2-imatrix.h (<name>.in_sum2 + <name>.counts per weight; see that file
// and minimax/mm3-imatrix.h for the rationale), renamed. Differences:
//
//   1. The hook is armed by the caller on the schedulers it owns. The graphs'
//      HeartMuLaSlot schedulers are created lazily inside lm-graph.h, so
//      heartmula_imatrix_arm_* creates a missing one first (the same
//      backend_sched_new the slot would have made) and leaves it for the slot.
//   2. One state holds both components; save() writes one file per component by
//      name set, because the quantizer is run per GGUF.
//   3. Collect on the unadapted F16 pair only. An adapter delta is applied by
//      separate matmuls over the base weights, so the base matmuls are still
//      what is measured, but the activations then describe the ADAPTED model.

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "heartmula/local-graph.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

struct HeartMuLaImatrixEntry {
    std::vector<double> sum2;      // [n_in] summed squares down each input column
    int64_t             rows = 0;  // activation rows accepted into sum2
    int64_t             bad  = 0;  // rows rejected for holding a non-finite value
};

struct HeartMuLaImatrixState {
    bool                                    armed = false;
    std::set<std::string>                   allow;    // collectable weight names
    std::map<std::string, HeartMuLaImatrixEntry> ent;
    std::vector<std::string>                sources;  // labels of the runs so far
    int64_t                                 runs = 0;

    // Diagnostics. A silent zero here is the failure mode that would otherwise
    // produce a plausible-looking but useless imatrix, so they are reported.
    int64_t nodes          = 0;  // matmuls collected
    int64_t skipped_type   = 0;  // src[1] was neither F32 nor F16
    int64_t skipped_stride = 0;  // src[1] rows were not contiguous in ne[0]
    int64_t skipped_shape  = 0;  // src[1] ne[0] disagreed with an earlier call
    int64_t bad_rows       = 0;  // rows dropped for holding a non-finite value

    // Scratch, kept across calls so a 28-layer forward does not churn the heap.
    std::vector<uint8_t> scratch;
    std::vector<float>   conv;
};

static HeartMuLaImatrixState g_heartmula_imatrix;

// Follow a view chain back to the tensor that actually owns the data.
static const ggml_tensor * heartmula_imatrix_base(const ggml_tensor * t) {
    while (t && t->view_src) {
        t = t->view_src;
    }
    return t;
}

static bool heartmula_imatrix_wanted(const ggml_tensor * t, std::string * name) {
    if (!t || t->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * w = heartmula_imatrix_base(t->src[0]);
    if (!w || w->name[0] == '\0') {
        return false;
    }
    if (g_heartmula_imatrix.allow.find(w->name) == g_heartmula_imatrix.allow.end()) {
        return false;
    }
    if (name) {
        name->assign(w->name);
    }
    return true;
}

// One activation row into the accumulator, or dropped whole (not
// element-wise — see minimax/mm3-imatrix.h's mm3_imatrix_accumulate comment
// for why a partially-finite row is not trustworthy either).
static bool heartmula_imatrix_accumulate(HeartMuLaImatrixEntry & e, const float * f, int64_t n_in) {
    for (int64_t j = 0; j < n_in; j++) {
        if (!std::isfinite(f[j])) {
            e.bad++;
            g_heartmula_imatrix.bad_rows++;
            return false;
        }
    }
    for (int64_t j = 0; j < n_in; j++) {
        const double v = (double) f[j];
        e.sum2[(size_t) j] += v * v;
    }
    e.rows++;
    return true;
}

static bool heartmula_imatrix_eval_cb(ggml_tensor * t, bool ask, void * ud) {
    (void) ud;

    std::string wname;
    if (!heartmula_imatrix_wanted(t, &wname)) {
        return false;
    }
    if (ask) {
        return true;
    }

    const ggml_tensor * x = t->src[1];
    if (!x) {
        return true;
    }
    if (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16) {
        g_heartmula_imatrix.skipped_type++;
        return true;
    }
    const size_t ts = ggml_type_size(x->type);
    if (x->nb[0] != ts) {
        // A transposed src[1] would make "one row" non-contiguous; the column
        // walk below would read the wrong numbers rather than fail, so refuse.
        g_heartmula_imatrix.skipped_stride++;
        return true;
    }

    const int64_t n_in = x->ne[0];

    HeartMuLaImatrixEntry & e = g_heartmula_imatrix.ent[wname];
    if (e.sum2.empty()) {
        e.sum2.assign((size_t) n_in, 0.0);
    } else if ((int64_t) e.sum2.size() != n_in) {
        g_heartmula_imatrix.skipped_shape++;
        return true;
    }

    g_heartmula_imatrix.conv.resize((size_t) n_in);

    if (ggml_is_contiguous(x)) {
        // One device->host copy for the whole activation block rather than
        // one memcpy per row — prefill can be a few hundred rows per matmul.
        const size_t nb = ggml_nbytes(x);
        g_heartmula_imatrix.scratch.resize(nb);
        ggml_backend_tensor_get(x, g_heartmula_imatrix.scratch.data(), 0, nb);

        const int64_t nrows = ggml_nelements(x) / n_in;
        for (int64_t r = 0; r < nrows; r++) {
            const uint8_t * p = g_heartmula_imatrix.scratch.data() + (size_t) r * (size_t) n_in * ts;
            if (x->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) p, g_heartmula_imatrix.conv.data(), n_in);
                heartmula_imatrix_accumulate(e, g_heartmula_imatrix.conv.data(), n_in);
            } else {
                heartmula_imatrix_accumulate(e, (const float *) p, n_in);
            }
        }
    } else {
        g_heartmula_imatrix.scratch.resize((size_t) n_in * ts);
        for (int64_t i3 = 0; i3 < x->ne[3]; i3++) {
            for (int64_t i2 = 0; i2 < x->ne[2]; i2++) {
                for (int64_t i1 = 0; i1 < x->ne[1]; i1++) {
                    const size_t off = (size_t) i1 * x->nb[1] + (size_t) i2 * x->nb[2] + (size_t) i3 * x->nb[3];
                    ggml_backend_tensor_get(x, g_heartmula_imatrix.scratch.data(), off, (size_t) n_in * ts);
                    if (x->type == GGML_TYPE_F16) {
                        ggml_fp16_to_fp32_row((const ggml_fp16_t *) g_heartmula_imatrix.scratch.data(),
                                              g_heartmula_imatrix.conv.data(), n_in);
                        heartmula_imatrix_accumulate(e, g_heartmula_imatrix.conv.data(), n_in);
                    } else {
                        heartmula_imatrix_accumulate(e, (const float *) g_heartmula_imatrix.scratch.data(), n_in);
                    }
                }
            }
        }
    }

    g_heartmula_imatrix.nodes++;
    return true;
}

// Attach or detach the hook on one scheduler.
static void heartmula_imatrix_hook(ggml_backend_sched_t sched) {
    if (!sched) {
        return;
    }
    ggml_backend_sched_set_eval_callback(sched, g_heartmula_imatrix.armed ? heartmula_imatrix_eval_cb : nullptr,
                                          &g_heartmula_imatrix);
}

static void heartmula_imatrix_reset() {
    g_heartmula_imatrix.ent.clear();
    g_heartmula_imatrix.sources.clear();
    g_heartmula_imatrix.runs           = 0;
    g_heartmula_imatrix.nodes          = 0;
    g_heartmula_imatrix.skipped_type   = 0;
    g_heartmula_imatrix.skipped_stride = 0;
    g_heartmula_imatrix.skipped_shape  = 0;
    g_heartmula_imatrix.bad_rows       = 0;
}

// Tensors that have seen at least one finite row. The gap between this and
// ent.size() is the number that would be written as all-zeros, so it is the
// number that matters when judging whether a collection run is usable.
static size_t heartmula_imatrix_usable() {
    size_t n = 0;
    for (const auto & kv : g_heartmula_imatrix.ent) {
        if (kv.second.rows > 0) {
            n++;
        }
    }
    return n;
}

// Total activation rows seen, summed over every collected tensor. Reporting
// only: a session that ends with 0 here collected nothing at all.
static int64_t heartmula_imatrix_total_rows() {
    int64_t n = 0;
    for (const auto & kv : g_heartmula_imatrix.ent) {
        n += kv.second.rows;
    }
    return n;
}

static bool heartmula_imatrix_save(const std::string & path, const std::set<std::string> & only, const char * name,
                                   std::string * err) {
    if (g_heartmula_imatrix.ent.empty()) {
        if (err) {
            *err = "nothing collected for this component";
        }
        return false;
    }

    // An entry with no accepted rows is all zeros. Writing it would hand the
    // quantizer an importance vector that says "every column is worthless",
    // which is strictly worse than handing it nothing -- so it is omitted, and
    // quantize.cpp falls back to round-to-nearest for that tensor.
    size_t                                                          n_nonfinite_sums = 0;
    std::vector<const std::pair<const std::string, HeartMuLaImatrixEntry> *> keep;
    for (const auto & kv : g_heartmula_imatrix.ent) {
        if (kv.second.rows <= 0 || only.find(kv.first) == only.end()) {
            continue;
        }
        bool ok = true;
        for (double v : kv.second.sum2) {
            if (!std::isfinite(v) || v < 0.0) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            n_nonfinite_sums++;
            fprintf(stderr, "[HM-IMAT] '%s' accumulated a non-finite sum over %lld rows - omitted\n",
                    kv.first.c_str(), (long long) kv.second.rows);
            continue;
        }
        keep.push_back(&kv);
    }
    if (n_nonfinite_sums) {
        fprintf(stderr, "[HM-IMAT] %zu tensors had non-finite accumulators\n", n_nonfinite_sums);
    }
    if (keep.empty()) {
        if (err) {
            *err = "every collected tensor had only non-finite rows — nothing worth writing";
        }
        return false;
    }
    if (keep.size() < g_heartmula_imatrix.ent.size()) {
        fprintf(stderr, "[HM-IMAT] %zu of %zu tensors had no finite rows and were omitted\n",
                g_heartmula_imatrix.ent.size() - keep.size(), g_heartmula_imatrix.ent.size());
    }

    size_t n_floats = 0;
    for (const auto * kv : keep) {
        n_floats += kv->second.sum2.size() + 1;  // in_sum2 + counts
    }

    const size_t n_tensors = keep.size() * 2;
    const size_t mem       = ggml_tensor_overhead() * (n_tensors + 16) + n_floats * sizeof(float) +
                       n_tensors * GGML_MEM_ALIGN * 2 + (1u << 20);

    ggml_init_params ip  = { mem, nullptr, /*no_alloc*/ false };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        if (err) {
            *err = "could not allocate the imatrix write context";
        }
        return false;
    }

    gguf_context * out = gguf_init_empty();
    gguf_set_val_str(out, "general.type", "imatrix");
    gguf_set_val_str(out, "general.architecture", "heartmula");
    gguf_set_val_str(out, "general.name", name);
    gguf_set_val_u32(out, "imatrix.chunk_count", (uint32_t) g_heartmula_imatrix.runs);
    gguf_set_val_u32(out, "imatrix.chunk_size", 0);
    if (!g_heartmula_imatrix.sources.empty()) {
        std::vector<const char *> ds;
        ds.reserve(g_heartmula_imatrix.sources.size());
        for (const auto & s : g_heartmula_imatrix.sources) {
            ds.push_back(s.c_str());
        }
        gguf_set_arr_str(out, "imatrix.datasets", ds.data(), (int) ds.size());
    }

    for (const auto * kvp : keep) {
        const std::string &      name = kvp->first;
        const HeartMuLaImatrixEntry & e    = kvp->second;

        ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) e.sum2.size());
        ggml_set_name(s, (name + ".in_sum2").c_str());
        float * sd = (float *) s->data;
        for (size_t j = 0; j < e.sum2.size(); j++) {
            sd[j] = (float) e.sum2[j];
        }
        gguf_add_tensor(out, s);

        ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_name(c, (name + ".counts").c_str());
        ((float *) c->data)[0] = (float) e.rows;
        gguf_add_tensor(out, c);
    }

    const bool ok = gguf_write_to_file(out, path.c_str(), /*only_meta*/ false);
    gguf_free(out);
    ggml_free(ctx);

    if (!ok && err) {
        *err = "could not write " + path;
    }
    return ok;
}

// ── Arming ──────────────────────────────────────────────────────────────────
// Needs lm-graph.h / local-graph.h in the including file (HeartMuLaSlot).

static void heartmula_imatrix_arm_slot(HeartMuLaSlot * s, const BackendPair & bp) {
    if (!s->sched) {
        s->sched = backend_sched_new(bp, HEARTMULA_MAX_NODES);
    }
    heartmula_imatrix_hook(s->sched);
}

// heartmula_lm_step frees and rebuilds a slot (scheduler included) whenever its T or n_kv
// changes, so the callback must go on AFTER the build and BEFORE the compute. Build the
// slot here with the same test lm_step applies, then arm it; lm_step then finds the graph
// ready and keeps the scheduler. Call with the arguments you pass to lm_step.
static bool heartmula_imatrix_arm_global(const HeartMuLaModel & m, HeartMuLaLmGraph * g, const int64_t * positions,
                                         int64_t T, std::string * err) {
    heartmula_lm_sync_adapter(g);
    int64_t hi = 0;
    for (int64_t i = 0; i < T; i++) {
        hi = std::max(hi, positions[i] + 1);
    }
    const int64_t   n_kv = std::min(heartmula_bucket(hi), g->n_ctx);
    HeartMuLaSlot * s    = T > 1 ? &g->prefill : &g->decode;
    if (!s->graph || s->T != T || s->n_kv != n_kv) {
        if (!heartmula_lm_build(m, g, s, T, n_kv, err)) {
            return false;
        }
    }
    heartmula_imatrix_arm_slot(s, m.bp);
    return true;
}

// The local slots are built once by heartmula_local_prepare and never rebuilt.
static void heartmula_imatrix_arm_local(const BackendPair & bp, HeartMuLaLocalGraph * g) {
    for (int i = 1; i < HEARTMULA_CODEBOOKS; i++) {
        heartmula_imatrix_arm_slot(&g->slot[i], bp);
    }
}
