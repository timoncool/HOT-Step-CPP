// heartmula-quant: calibration and quant-ladder sanity for the HeartMuLa LMs.
//
//   heartmula-quant --mode collect --models DIR --prompts FILE.json --set calib
//                   [--variant V] [--global TYPE] [--local TYPE] --out-prefix P
//   heartmula-quant --mode ref  ... --set heldout --ref-dir DIR      (F16 pair)
//   heartmula-quant --mode eval ... --set heldout --ref-dir DIR      (quantized pair)
//
// All three run the LM phase of heartmula/pipeline.h (same prompt ids, CFG rows,
// KV cache, prefill chunks, one global step plus one local frame per frame):
//
//   collect  samples as production does and accumulates activation importance for
//            every global and local matmul weight; writes <P>-global.imatrix.gguf
//            and <P>-local.imatrix.gguf (heartmula/imatrix.h).
//   ref      samples and stores each prompt's codes and CFG-blended logits for all
//            eight codebooks, per frame, in DIR/<id>.ref.
//   eval     teacher-forces the stored codes through the loaded pair and prints
//            KL(ref || this), argmax agreement and cross-entropy of the stored
//            code against the stored (ce_ref) and this model's (ce) logits, for
//            codebook 0 and codebooks 1-7, as one JSON line per run.
//
// The prompt file is {"calib":[P...],"heldout":[P...]} with P = {id, tags,
// lyrics, seconds, seed}. Calibration and held-out sets never share a prompt.
#include "heartmula/imatrix.h"
#include "heartmula/pipeline.h"

#include <cstdio>
#include <fstream>
#include <sstream>

struct QuantPrompt {
    std::string id, tags, lyrics;
    double      seconds = 20.0;
    uint64_t    seed    = 1;
};

static std::vector<QuantPrompt> load_prompts(const std::string & path, const std::string & set) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    yyjson_doc * doc = yyjson_read(text.data(), text.size(), 0);
    if (!f || !doc) {
        throw std::runtime_error("cannot read prompts " + path);
    }
    std::vector<QuantPrompt> out;
    yyjson_val * arr = yyjson_obj_get(yyjson_doc_get_root(doc), set.c_str());
    size_t i, n;
    yyjson_val * v;
    yyjson_arr_foreach(arr, i, n, v) {
        QuantPrompt p;
        auto str = [&](const char * k) {
            yyjson_val * s = yyjson_obj_get(v, k);
            return yyjson_is_str(s) ? std::string(yyjson_get_str(s)) : std::string();
        };
        p.id     = str("id");
        p.tags   = str("tags");
        p.lyrics = str("lyrics");
        if (yyjson_val * s = yyjson_obj_get(v, "seconds")) p.seconds = yyjson_get_num(s);
        if (yyjson_val * s = yyjson_obj_get(v, "seed")) p.seed = (uint64_t) yyjson_get_num(s);
        out.push_back(p);
    }
    yyjson_doc_free(doc);
    if (out.empty()) {
        throw std::runtime_error("prompt set '" + set + "' is empty or missing");
    }
    return out;
}

struct RefTrack {
    int64_t              frames = 0;
    std::vector<int32_t> codes;   // [frames][8]
    std::vector<float>   logits;  // [frames][8][V], CFG-blended
};

static void write_ref(const std::string & path, const RefTrack & r) {
    std::ofstream f(path, std::ios::binary);
    const int64_t n = r.frames;
    f.write((const char *) &n, sizeof(n));
    f.write((const char *) r.codes.data(), (std::streamsize) (r.codes.size() * 4));
    f.write((const char *) r.logits.data(), (std::streamsize) (r.logits.size() * 4));
    if (!f) throw std::runtime_error("cannot write " + path);
}

static RefTrack read_ref(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    RefTrack r;
    if (!f.read((char *) &r.frames, sizeof(r.frames))) throw std::runtime_error("cannot read " + path);
    r.codes.resize((size_t) r.frames * HEARTMULA_CODEBOOKS);
    r.logits.resize((size_t) r.frames * HEARTMULA_CODEBOOKS * HEARTMULA_AUDIO_VOCAB);
    f.read((char *) r.codes.data(), (std::streamsize) (r.codes.size() * 4));
    f.read((char *) r.logits.data(), (std::streamsize) (r.logits.size() * 4));
    if (!f) throw std::runtime_error(path + ": truncated");
    return r;
}

// log-softmax into `lp`, returns nothing; max-subtracted in double.
static void log_softmax(const float * x, int64_t n, std::vector<double> * lp) {
    float mx = x[0];
    for (int64_t i = 1; i < n; i++) mx = std::max(mx, x[i]);
    double sum = 0;
    for (int64_t i = 0; i < n; i++) sum += std::exp((double) x[i] - mx);
    const double lse = std::log(sum) + mx;
    lp->resize((size_t) n);
    for (int64_t i = 0; i < n; i++) (*lp)[(size_t) i] = (double) x[i] - lse;
}

struct Acc {
    double  kl = 0, ce = 0, ce_ref = 0;
    int64_t agree = 0, n = 0, nonfinite = 0;
};

static void score(Acc * a, const float * ref, const float * got, int32_t code) {
    const int64_t V = HEARTMULA_AUDIO_VOCAB;
    for (int64_t i = 0; i < V; i++) {
        if (!std::isfinite(got[i])) {
            a->nonfinite++;
            return;
        }
    }
    std::vector<double> lr, lg;
    log_softmax(ref, V, &lr);
    log_softmax(got, V, &lg);
    double kl = 0;
    int64_t ar = 0, ag = 0;
    for (int64_t i = 0; i < V; i++) {
        kl += std::exp(lr[(size_t) i]) * (lr[(size_t) i] - lg[(size_t) i]);
        if (ref[i] > ref[ar]) ar = i;
        if (got[i] > got[ag]) ag = i;
    }
    a->kl += kl;
    a->ce += -lg[(size_t) code];
    a->ce_ref += -lr[(size_t) code];
    a->agree += ar == ag;
    a->n++;
}

struct Options {
    std::string mode, models, prompts, set, variant, gtype, ltype, out_prefix, ref_dir;
    int         max_frames = 0;
};

static bool run_prompt(HeartMuLaRuntime * rt, const HeartMuLaTokenizer & tok, const QuantPrompt & qp, const Options & o,
                       RefTrack * ref, Acc acc[2]) {
    const bool collect = o.mode == "collect", forced = o.mode == "eval";
    const float cfg    = 1.5f;
    const int   B      = 2;
    int64_t     muq    = 0;
    const auto  ids    = heartmula_prompt_ids(tok, qp.tags, qp.lyrics, &muq);
    const int64_t T    = (int64_t) ids.size();
    int64_t     frames = heartmula_max_frames(qp.seconds);
    if (o.max_frames > 0) frames = std::min<int64_t>(frames, o.max_frames);
    if (forced) frames = std::min<int64_t>(frames, ref->frames);
    const int64_t V = HEARTMULA_AUDIO_VOCAB, K = HEARTMULA_CODEBOOKS;

    HeartMuLaLmGraph    g;
    HeartMuLaLocalGraph lg;
    std::string         err;
    auto done = [&](bool ok) {
        heartmula_local_free(&lg);
        heartmula_lm_free(&g);
        if (!ok) fprintf(stderr, "[quant] %s: %s\n", qp.id.c_str(), err.c_str());
        return ok;
    };
    if (T + frames + 1 > heartmula_global_config().n_ctx) {
        err = "prompt too long";
        return done(false);
    }
    if (!heartmula_lm_reserve(rt->lm, &g, B, T + frames + 1, &err) || !heartmula_local_prepare(rt->lm, &lg, B, &err)) {
        return done(false);
    }
    heartmula_lm_set_adapter(&g, nullptr, 1.0f);
    const BackendPair & bp = rt->lm.bp;

    std::vector<HeartMuLaPosition> prompt((size_t) T);
    for (int64_t i = 0; i < T; i++) {
        prompt[(size_t) i].id[HEARTMULA_COLUMNS - 1]   = ids[(size_t) i];
        prompt[(size_t) i].mask[HEARTMULA_COLUMNS - 1] = true;
    }
    const int64_t H = rt->lm.global.cfg.n_embd;
    std::vector<float>   hidden((size_t) (B * H)), c0((size_t) (B * V));
    std::vector<int64_t> pos;
    for (int64_t start = 0; start < T; start += HEARTMULA_PREFILL_CHUNK) {
        const int64_t n = std::min<int64_t>(HEARTMULA_PREFILL_CHUNK, T - start);
        pos.resize((size_t) n);
        for (int64_t i = 0; i < n; i++) pos[(size_t) i] = start + i;
        const bool last = start + n == T;
        if (collect && !heartmula_imatrix_arm_global(rt->lm, &g, pos.data(), n, &err)) return done(false);
        if (!heartmula_lm_step(rt->lm, &g, prompt.data() + start, pos.data(), n,
                               muq >= start && muq < start + n ? muq - start : -1, last ? hidden.data() : nullptr,
                               last ? c0.data() : nullptr, &err)) {
            return done(false);
        }
    }

    HeartMuLaDraws draws(qp.seed);
    int64_t        f = 0;
    std::vector<float> guided((size_t) V);
    const HeartMuLaChooser choose = [&](int k, const float * logits, int b) -> int32_t {
        if (b == 2) heartmula_cfg_blend(logits, logits + V, V, cfg, guided.data());
        else std::copy(logits, logits + V, guided.begin());
        if (forced) {
            score(&acc[k == 0 ? 0 : 1], &ref->logits[(size_t) ((f * K + k) * V)], guided.data(),
                  ref->codes[(size_t) (f * K + k)]);
            return ref->codes[(size_t) (f * K + k)];
        }
        if (ref) std::copy(guided.begin(), guided.end(), ref->logits.begin() + (ptrdiff_t) ((f * K + k) * V));
        return heartmula_sample_topk(guided.data(), V, 50, 1.0f, draws);
    };
    if (ref && !forced) ref->logits.assign((size_t) (frames * K * V), 0.0f);
    std::vector<int32_t> codes_all;
    for (f = 0; f < frames; f++) {
        if (f > 0) {
            HeartMuLaPosition in;
            for (int q = 0; q < K; q++) {
                in.id[q]   = codes_all[(size_t) ((f - 1) * K + q)];
                in.mask[q] = true;
            }
            const int64_t position = T + f - 1;
            if (collect && !heartmula_imatrix_arm_global(rt->lm, &g, &position, 1, &err)) return done(false);
            if (!heartmula_lm_step(rt->lm, &g, &in, &position, 1, -1, hidden.data(), c0.data(), &err)) return done(false);
        }
        int32_t codes[HEARTMULA_CODEBOOKS];
        if (collect) heartmula_imatrix_arm_local(bp, &lg);
        if (!heartmula_local_frame(rt->lm, &lg, hidden.data(), c0.data(), choose, codes, nullptr, &err)) return done(false);
        if (*std::max_element(codes, codes + K) >= HEARTMULA_AUDIO_EOS) {
            frames = f;
            break;
        }
        codes_all.insert(codes_all.end(), codes, codes + K);
    }
    if (ref && !forced) {
        ref->frames = frames;
        ref->codes  = codes_all;
        ref->logits.resize((size_t) (frames * K * V));
    }
    fprintf(stderr, "[quant] %s: %lld prompt tokens, %lld frames\n", qp.id.c_str(), (long long) T, (long long) frames);
    return done(true);
}

static std::set<std::string> weight_names(const std::string & path) {
    std::set<std::string> out;
    ggml_context * meta = nullptr;
    gguf_context * g    = gguf_init_from_file(path.c_str(), { true, &meta });
    if (!g) throw std::runtime_error("cannot read " + path);
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        const char * n = gguf_get_tensor_name(g, i);
        const size_t l = strlen(n);
        if (ggml_n_dims(ggml_get_tensor(meta, n)) == 2 && l > 7 && !strcmp(n + l - 7, ".weight")) out.insert(n);
    }
    gguf_free(g);
    ggml_free(meta);
    return out;
}

int main(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto value = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--mode") o.mode = value();
        else if (a == "--models") o.models = value();
        else if (a == "--prompts") o.prompts = value();
        else if (a == "--set") o.set = value();
        else if (a == "--variant") o.variant = value();
        else if (a == "--global") o.gtype = value();
        else if (a == "--local") o.ltype = value();
        else if (a == "--out-prefix") o.out_prefix = value();
        else if (a == "--ref-dir") o.ref_dir = value();
        else if (a == "--max-frames") o.max_frames = atoi(value().c_str());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    const bool collect = o.mode == "collect", is_ref = o.mode == "ref", eval = o.mode == "eval";
    if (!(collect || is_ref || eval) || o.models.empty() || o.prompts.empty() || o.set.empty() ||
        (collect && o.out_prefix.empty()) || ((is_ref || eval) && o.ref_dir.empty())) {
        fprintf(stderr, "usage: heartmula-quant --mode collect|ref|eval --models DIR --prompts FILE --set calib|heldout\n"
                        "         [--variant V] [--global TYPE] [--local TYPE] [--max-frames N]\n"
                        "         collect: --out-prefix P    ref/eval: --ref-dir DIR\n");
        return 2;
    }
    try {
        const auto prompts = load_prompts(o.prompts, o.set);
        const HeartMuLaCatalogue cat = heartmula_discover(o.models);
        for (const auto & e : cat.errors) fprintf(stderr, "[HeartMuLa] catalogue: %s\n", e.c_str());
        HeartMuLaPick pick;
        pick.variant = o.variant;
        pick.global  = o.gtype;
        pick.local   = o.ltype;
        HeartMuLaResolved res;
        int               status = 0;
        std::string       err;
        if (!heartmula_resolve(cat, pick, &res, &status, &err)) {
            fprintf(stderr, "FATAL: pick (%d): %s\n", status, err.c_str());
            return 1;
        }
        HeartMuLaTokenizer tok;
        if (!heartmula_tokenizer_load(&tok, res.file[HM_TOKENIZER].path.c_str(), &err)) {
            fprintf(stderr, "FATAL: tokenizer: %s\n", err.c_str());
            return 1;
        }
        fprintf(stderr, "[quant] global %s, local %s\n", res.file[HM_GLOBAL].filename.c_str(),
                res.file[HM_LOCAL].filename.c_str());
        HeartMuLaRuntime rt;
        heartmula_ensure_backend(&rt);
        if (!heartmula_load_lm(&rt, res, &err)) {
            fprintf(stderr, "FATAL: load: %s\n", err.c_str());
            return 1;
        }
        std::set<std::string> gnames, lnames;
        if (collect) {
            gnames = weight_names(res.file[HM_GLOBAL].path);
            lnames = weight_names(res.file[HM_LOCAL].path);
            g_heartmula_imatrix.allow = gnames;
            g_heartmula_imatrix.allow.insert(lnames.begin(), lnames.end());
            g_heartmula_imatrix.armed = true;
        }
        Acc acc[2];
        int ok = 0;
        for (const auto & qp : prompts) {
            RefTrack ref;
            const std::string path = o.ref_dir + "/" + qp.id + ".ref";
            if (eval) ref = read_ref(path);
            if (!run_prompt(&rt, tok, qp, o, (is_ref || eval) ? &ref : nullptr, acc)) return 1;
            if (is_ref) write_ref(path, ref);
            if (collect) {
                g_heartmula_imatrix.sources.push_back(qp.id);
                g_heartmula_imatrix.runs++;
            }
            ok++;
        }
        if (collect) {
            auto & st = g_heartmula_imatrix;
            size_t missing = 0;
            for (const auto & n : st.allow) {
                auto it = st.ent.find(n);
                if (it == st.ent.end() || it->second.rows == 0) {
                    fprintf(stderr, "[quant] no rows for %s\n", n.c_str());
                    missing++;
                }
            }
            fprintf(stderr, "[quant] %d prompts, %lld matmuls, %lld rows, %lld bad rows, skipped type/stride/shape %lld/%lld/%lld, "
                            "%zu of %zu weights without rows\n",
                    ok, (long long) st.nodes, (long long) heartmula_imatrix_total_rows(), (long long) st.bad_rows,
                    (long long) st.skipped_type, (long long) st.skipped_stride, (long long) st.skipped_shape, missing,
                    st.allow.size());
            if (!heartmula_imatrix_save(o.out_prefix + "-global.imatrix.gguf", gnames, "HeartMuLa global imatrix", &err) ||
                !heartmula_imatrix_save(o.out_prefix + "-local.imatrix.gguf", lnames, "HeartMuLa local imatrix", &err)) {
                fprintf(stderr, "FATAL: %s\n", err.c_str());
                return 1;
            }
        }
        if (eval) {
            printf("{\"global\":\"%s\",\"local\":\"%s\",\"prompts\":%d", res.file[HM_GLOBAL].filename.c_str(),
                   res.file[HM_LOCAL].filename.c_str(), ok);
            for (int c = 0; c < 2; c++) {
                const Acc & a = acc[c];
                const double n = std::max<int64_t>(1, a.n);
                printf(",\"%s\":{\"n\":%lld,\"kl\":%.5f,\"argmax\":%.4f,\"ce\":%.4f,\"ce_ref\":%.4f,\"nonfinite\":%lld}",
                       c == 0 ? "c0" : "c1_7", (long long) a.n, a.kl / n, (double) a.agree / n, a.ce / n, a.ce_ref / n,
                       (long long) a.nonfinite);
            }
            printf("}\n");
        }
        heartmula_free_all(&rt);
    } catch (const std::exception & e) {
        fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return 0;
}
