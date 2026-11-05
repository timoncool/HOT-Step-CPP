// sa3-train-run.h: ace-train sa3-train (the loop) and sa3-train-check (parity).
//
// sa3-train: pre-encode the manifest (train/sa3-train-data.h), load the frozen
// medium-base DiT, train DoRA-rows rank/alpha 16 with AdamW (lr 1e-4, betas
// 0.9/0.95, weight decay 0.01, eps 1e-8, no clipping: train_lora.py), batch 1.
//   <out>/state.gguf             latest resumable state (atomic): adapters,
//                                AdamW moments, step, run fingerprint
//   <out>/adapter.gguf           dora-rows-v1 inference export (sa3-adapter-export.h)
//                                against the installed inference DiT
//   <out>/adapter-raw.gguf       raw sa3-lora DiT adapter (training-base parameters)
//   <out>/snapshots/step-N.gguf  inference exports every --export-every steps
// Cancel: create <out>/STOP; the loop saves state at the next step and exits 0.
// A hard kill loses at most --checkpoint-every steps: --resume continues from
// state.gguf onto the identical draw sequence (draws are keyed on seed+step).
// Progress is JSONL on stdout with --jsonl (types start, data, step, checkpoint,
// export, stopped, done, error), human logs on stderr.
//
// sa3-train-check: replays the stage-4 official capture (fixtures from
// tools/sa3-training-step-fixture.py) through the same model code and gates
// output, loss, all adapter gradients, AdamW, SAME and T5Gemma against it.

#pragma once

#include "train/sa3-adapter-export.h"
#include "train/sa3-train-data.h"
#include "train/sa3-train-model.h"

#include "safetensors.h"

#include <chrono>
#include <filesystem>
#include <map>

static void jl(const char * fmt, ...);

struct Sa3TrainArgs {
    std::string manifest, models_dir, base_dit, out_dir, cache_dir, inference_dit;
    int64_t     steps = 1000, rank = 16, ckpt_every = 100, export_every = 0;
    float       alpha = 16.0f, lr = 1e-4f, wd = 0.01f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f;
    double      duration = 380.0;  // crop window, seconds (train_lora.py --duration default)
    uint64_t    seed = 42;
    bool        resume = false, train_seconds = false;
};

// ── AdamW (torch.optim.AdamW, single-tensor path, f32) ─────────────────────
struct Sa3Adam {
    std::vector<std::vector<float>> m, v;  // 3 per site: a, b, magnitude
};

static void sa3t_adamw(std::vector<float> & p, const std::vector<float> & g, std::vector<float> & m,
                       std::vector<float> & v, int64_t step, const Sa3TrainArgs & a) {
    if (m.empty()) m.assign(p.size(), 0.0f), v.assign(p.size(), 0.0f);
    const float bc1 = 1.0f - std::pow(a.beta1, (float) step), bc2 = 1.0f - std::pow(a.beta2, (float) step);
    const float step_size = a.lr / bc1, bc2s = std::sqrt(bc2), decay = 1.0f - a.lr * a.wd;
    for (size_t i = 0; i < p.size(); i++) {
        p[i] *= decay;
        m[i] += (g[i] - m[i]) * (1.0f - a.beta1);  // lerp_
        v[i] = v[i] * a.beta2 + (1.0f - a.beta2) * g[i] * g[i];
        p[i] -= step_size * (m[i] / (std::sqrt(v[i]) / bc2s + a.eps));
    }
}

static void sa3t_optimizer_step(Sa3TrainModel & mod, Sa3Adam & opt, int64_t step, const Sa3TrainArgs & a) {
    opt.m.resize(mod.sites.size() * 3);
    opt.v.resize(mod.sites.size() * 3);
    for (size_t s = 0; s < mod.sites.size(); s++) {
        Sa3Site & st = mod.sites[s];
        sa3t_adamw(st.a, st.ga, opt.m[3 * s], opt.v[3 * s], step, a);
        sa3t_adamw(st.b, st.gb, opt.m[3 * s + 1], opt.v[3 * s + 1], step, a);
        sa3t_adamw(st.m, st.gm, opt.m[3 * s + 2], opt.v[3 * s + 2], step, a);
    }
}

// Upload one step's inputs (torch layouts in s) into the model's device state.
static void sa3t_set_inputs(Sa3TrainModel & m, const Sa3StepInputs & s, const std::vector<float> & text, int seconds_total) {
    const int64_t C = m.T, io = m.c.io_channels;
    std::vector<float> tm((size_t) (io * C));
    auto token_major = [&](const std::vector<float> & cm, int64_t ch, ggml_tensor * dst) {
        tm.resize((size_t) (ch * C));
        for (int64_t j = 0; j < C; j++)
            for (int64_t c = 0; c < ch; c++) tm[(size_t) (j * ch + c)] = cm[(size_t) (c * C + j)];
        ggml_backend_tensor_set(dst, tm.data(), 0, ggml_nbytes(dst));
    };
    token_major(s.x_t, io, m.x_t);
    token_major(s.target, io, m.target);
    token_major(s.local, m.c.local_cond_dim, m.local);
    std::vector<float> cb((size_t) (SA3T_TEXT_DIM * m.S_c), 0.0f);
    if (!s.cfg_drop) std::copy(text.begin(), text.end(), cb.begin());
    ggml_backend_tensor_set(m.cross_base, cb.data(), 0, ggml_nbytes(m.cross_base));
    const float keep = s.cfg_drop ? 0.0f : 1.0f;
    ggml_backend_tensor_set(m.keep, &keep, 0, sizeof(float));
    std::vector<float> f(256);
    sa3t_expo((float) s.t, m.c.ts_feat_dim, f.data());
    ggml_backend_tensor_set(m.tfeat, f.data(), 0, ggml_nbytes(m.tfeat));
    const float secn = std::min(std::max((float) seconds_total, 0.0f), 384.0f) / 384.0f;  // NumberConditioner [0, 384]
    sa3t_expo(secn, 256, f.data());
    ggml_backend_tensor_set(m.sfeat, f.data(), 0, ggml_nbytes(m.sfeat));
    std::vector<float> vm((size_t) m.S, 1.0f);
    for (int64_t j = 0; j < C; j++) vm[(size_t) (m.c.n_memory_tokens + j)] = s.aug_pad[(size_t) j];
    ggml_backend_tensor_set(m.vmask, vm.data(), 0, ggml_nbytes(m.vmask));
    // loss = masked mean over loss_mask + masked mean over context (diffusion.py training_step)
    double nl = 0, nc = 0;
    for (int64_t j = 0; j < C; j++) nl += s.loss_mask[(size_t) j], nc += s.ctx_mask[(size_t) j];
    std::vector<float> w((size_t) C);
    for (int64_t j = 0; j < C; j++)
        w[(size_t) j] = (float) (s.loss_mask[(size_t) j] / (nl * io + 1e-8) + (nc > 0 ? s.ctx_mask[(size_t) j] / (nc * io + 1e-8) : 0.0));
    ggml_backend_tensor_set(m.w_loss, w.data(), 0, ggml_nbytes(m.w_loss));
}

// ── state file ─────────────────────────────────────────────────────────────
static bool sa3t_save_state(const Sa3TrainModel & m, const Sa3Adam & opt, int64_t step, const std::string & fp,
                            const std::string & path, std::string * err) {
    try {
        sa3_export::File f(m.sites.size() * 9);
        gguf_set_val_str(f.gguf, "general.architecture", "sa3-train-state");
        gguf_set_val_str(f.gguf, "sa3t.fingerprint", fp.c_str());
        gguf_set_val_u64(f.gguf, "sa3t.step", (uint64_t) step);
        const char * k[3] = { "a", "b", "m" };
        for (size_t s = 0; s < m.sites.size(); s++) {
            const Sa3Site & st = m.sites[s];
            const std::vector<float> * p[3] = { &st.a, &st.b, &st.m };
            const int64_t ne[3][2] = { { st.in, m.rank }, { m.rank, st.out }, { st.out, 0 } };
            for (int j = 0; j < 3; j++) {
                const std::string n = st.stem + "." + k[j];
                f.add(n, ne[j][0], ne[j][1], *p[j]);
                const std::vector<float> zero(p[j]->size(), 0.0f);
                const bool has = opt.m.size() > 3 * s + j && !opt.m[3 * s + j].empty();
                f.add(n + ".adam_m", ne[j][0], ne[j][1], has ? opt.m[3 * s + j] : zero);
                f.add(n + ".adam_v", ne[j][0], ne[j][1], has ? opt.v[3 * s + j] : zero);
            }
        }
        const std::string tmp = path + ".tmp";
        std::filesystem::remove(std::filesystem::u8path(tmp));
        f.write(tmp);
        std::filesystem::rename(std::filesystem::u8path(tmp), std::filesystem::u8path(path));
    } catch (const std::exception & e) {
        *err = std::string("state write failed: ") + e.what();
        return false;
    }
    return true;
}

static bool sa3t_load_state(Sa3TrainModel & m, Sa3Adam & opt, int64_t * step, const std::string & fp,
                            const std::string & path, std::string * err) {
    GGUFModel gf = {};
    if (!gf_load(&gf, path.c_str())) {
        *err = "cannot open " + path;
        return false;
    }
    const char * have = gf_get_str(gf, "sa3t.fingerprint");
    if (!have || fp != have) {
        gf_close(&gf);
        *err = "state.gguf belongs to a different run (rank/alpha/lr/seed/crop/dataset/base differ)";
        return false;
    }
    *step = (int64_t) gguf_get_val_u64(gf.gguf, gguf_find_key(gf.gguf, "sa3t.step"));
    opt.m.assign(m.sites.size() * 3, {});
    opt.v.assign(m.sites.size() * 3, {});
    const char * k[3] = { "a", "b", "m" };
    bool ok = true;
    auto fill = [&](const std::string & n, std::vector<float> & dst, size_t want) {
        ggml_tensor * t = ggml_get_tensor(gf.meta, n.c_str());
        const void *  d = gf_get_data(gf, n.c_str());
        if (!t || !d || t->type != GGML_TYPE_F32 || (size_t) ggml_nelements(t) != want) {
            *err = "state.gguf: bad tensor " + n;
            ok = false;
            return;
        }
        dst.assign((const float *) d, (const float *) d + want);
    };
    for (size_t s = 0; s < m.sites.size() && ok; s++) {
        Sa3Site &       st = m.sites[s];
        std::vector<float> * p[3] = { &st.a, &st.b, &st.m };
        const size_t         n[3] = { (size_t) (st.in * m.rank), (size_t) (m.rank * st.out), (size_t) st.out };
        for (int j = 0; j < 3 && ok; j++) {
            fill(st.stem + "." + k[j], *p[j], n[j]);
            fill(st.stem + "." + k[j] + ".adam_m", opt.m[3 * s + j], n[j]);
            fill(st.stem + "." + k[j] + ".adam_v", opt.v[3 * s + j], n[j]);
        }
    }
    gf_close(&gf);
    return ok;
}

static std::vector<sa3_export::Rows> sa3t_rows(const Sa3TrainModel & m) {
    std::vector<sa3_export::Rows> rows;
    for (const Sa3Site & st : m.sites) {
        if (st.stem == "seconds") continue;  // outside the dora-rows-v1 DiT contract
        rows.push_back({ st.stem, st.in, st.out, m.rank, m.alpha, st.a, st.b, st.m });
    }
    return rows;
}

static std::string sa3t_file_sha(const std::string & path) {
    yue2_aitk::sha256::digest d;
    return yue2_aitk::sha256::file(std::filesystem::u8path(path), d) ? d.hex() : std::string();
}

// ── pre-encode ─────────────────────────────────────────────────────────────
static bool sa3t_preencode(const Sa3TrainArgs & a, int64_t C, std::vector<Sa3Sample> & data, std::vector<float> & silence,
                           std::string * err) {
    std::vector<Sa3ManifestItem> items;
    if (!sa3t_read_manifest(a.manifest, items, err)) return false;
    std::vector<float> pad;
    if (!sa3t_read_padding(a.base_dit, pad, err)) return false;
    const std::string enc_path = a.models_dir + "/sa3-same-enc-F16.gguf", t5_path = a.models_dir + "/sa3-text-enc-BF16.gguf";
    std::error_code ec;
    // cache identity: encoder weights by content (a same-size replacement must miss) + base padding
    const std::string enc_sha = sa3t_file_sha(enc_path), t5_sha = sa3t_file_sha(t5_path);
    if (enc_sha.empty() || t5_sha.empty()) {
        *err = "cannot read the SAME encoder / T5Gemma in " + a.models_dir;
        return false;
    }
    const std::string ident = "v2|" + enc_sha + "|" + t5_sha + "|" + sa3t_hex_key(pad.data(), pad.size() * sizeof(float), "");
    std::filesystem::create_directories(std::filesystem::u8path(a.cache_dir), ec);
    SA3Same       enc{};
    SA3T5GemmaEnc t5{};
    bool          loaded = false;
    auto load = [&]() {
        if (loaded) return true;
        loaded = sa3_same_load(&enc, enc_path.c_str(), true) && sa3_t5gemma_load(&t5, t5_path.c_str());
        if (!loaded) *err = "cannot load the SAME encoder / T5Gemma from " + a.models_dir;
        return loaded;
    };
    int hits = 0, skipped = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < items.size(); i++) {
        const Sa3ManifestItem & it = items[i];
        std::string bytes;
        if (!sa3t_read_file(it.audio_path, bytes)) {
            *err = "cannot read " + it.audio_path;
            return false;
        }
        const std::string key  = sa3t_hex_key(bytes.data(), bytes.size(), ident + "|" + sa3t_hex_key(it.ids.data(), it.ids.size() * 4, std::to_string(it.n_tokens)));
        const std::string path = a.cache_dir + "/" + key + ".sa3c";
        Sa3Sample s;
        s.audio_path = it.audio_path;
        s.prompt     = it.prompt;
        std::string c;
        if (sa3t_read_file(path, c) && c.size() >= 24 && !memcmp(c.data(), "SA3C0001", 8)) {
            int64_t hdr[2];
            memcpy(hdr, c.data() + 8, sizeof(hdr));
            s.n_samples = hdr[0];
            s.valid     = (s.n_samples + SA3T_DS - 1) / SA3T_DS;
            s.seconds_total = (int) std::ceil((double) s.n_samples / SA3T_SR);
            const size_t nl = (size_t) (SA3T_LAT * s.valid), nt = (size_t) SA3T_TOK * SA3T_TEXT_DIM;
            if (hdr[1] == 1 && c.size() == 24 + (nl + nt) * sizeof(float)) {
                s.latents.assign((const float *) (c.data() + 24), (const float *) (c.data() + 24) + nl);
                s.text.assign((const float *) (c.data() + 24) + nl, (const float *) (c.data() + 24) + nl + nt);
                hits++;
                data.push_back(std::move(s));
                continue;
            }
        }
        int    T = 0, sr = 0;
        float * audio = audio_io_read_wav_buf((const uint8_t *) bytes.data(), bytes.size(), &T, &sr);
        if (!audio || sr != SA3T_SR || T <= 0) {
            free(audio);
            *err = it.audio_path + ": need a 44.1 kHz WAV (the SA3 dataset prep exports one)";
            return false;
        }
        float peak = 0.0f;
        for (int64_t j = 0; j < 2 * (int64_t) T; j++) peak = std::max(peak, std::fabs(audio[j]));
        if (peak < 1e-3f) {  // is_silence: below -60 dBFS, upstream redraws another file
            free(audio);
            skipped++;
            jl("{\"type\":\"log\",\"level\":\"warn\",\"message\":\"silent clip skipped: %s\"}", lm_json_escape(it.audio_path).c_str());
            continue;
        }
        if (!load()) {
            free(audio);
            return false;
        }
        const int64_t n = ((int64_t) T + SA3T_DS - 1) / SA3T_DS * SA3T_DS;
        std::vector<float> planar((size_t) (2 * n), 0.0f);
        for (int ch = 0; ch < 2; ch++)
            for (int64_t j = 0; j < T; j++) planar[(size_t) (ch * n + j)] = std::min(1.0f, std::max(-1.0f, audio[ch * (int64_t) T + j]));
        free(audio);
        s.n_samples     = T;
        s.valid         = n / SA3T_DS;
        s.seconds_total = (int) std::ceil((double) T / SA3T_SR);
        sa3t_same_encode(&enc, planar.data(), n, s.latents);
        sa3t_text_encode(&t5, it.ids.data(), it.n_tokens, pad, s.text);
        const int64_t hdr[2] = { s.n_samples, 1 };
        if (!sa3t_write_file(path, { { "SA3C0001", 8 }, { hdr, sizeof(hdr) },
                                     { s.latents.data(), s.latents.size() * sizeof(float) },
                                     { s.text.data(), s.text.size() * sizeof(float) } })) {
            *err = "cannot write cache " + path;
            return false;
        }
        data.push_back(std::move(s));
        jl("{\"type\":\"progress\",\"completed\":%zu,\"total\":%zu,\"phase\":\"encode\"}", i + 1, items.size());
    }
    if (data.empty()) {
        *err = "every clip is silent";
        return false;
    }
    // silence latent for crops longer than the clip (pre_encode_dataset.py silence.npy)
    const std::string spath = a.cache_dir + "/silence-" + sa3t_hex_key(ident.data(), ident.size(), std::to_string(C)) + ".f32";
    std::string sc;
    if (sa3t_read_file(spath, sc) && sc.size() == (size_t) (SA3T_LAT * C) * sizeof(float)) {
        silence.assign((const float *) sc.data(), (const float *) sc.data() + SA3T_LAT * C);
    } else {
        if (!load()) return false;
        std::vector<float> zeros((size_t) (2 * C * SA3T_DS), 0.0f);
        sa3t_same_encode(&enc, zeros.data(), C * SA3T_DS, silence);
        sa3t_write_file(spath, { { silence.data(), silence.size() * sizeof(float) } });
    }
    if (loaded) {
        sa3_same_free(&enc);
        sa3_t5gemma_free(&t5);
    }
    const long long ms = (long long) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    jl("{\"type\":\"data\",\"samples\":%zu,\"cached\":%d,\"skipped\":%d,\"ms\":%lld}", data.size(), hits, skipped, ms);
    fprintf(stderr, "[sa3-train] %zu clips (%d cached, %d silent skipped), %lld ms\n", data.size(), hits, skipped, ms);
    return true;
}

// ── sa3-train ──────────────────────────────────────────────────────────────
static int sa3_train_run(Sa3TrainArgs a) {
    std::string err;
    auto fail = [&](const std::string & e) {
        jl("{\"type\":\"error\",\"message\":\"%s\"}", lm_json_escape(e).c_str());
        fprintf(stderr, "[sa3-train] ERROR: %s\n", e.c_str());
        return 1;
    };
    if (a.cache_dir.empty()) a.cache_dir = a.out_dir + "/cache";
    if (a.inference_dit.empty()) a.inference_dit = a.models_dir + "/sa3-dit-BF16.gguf";
    const int64_t C = (int64_t) (a.duration * SA3T_SR) / SA3T_DS;  // (int(duration*sr) // ds) * ds, in latents
    if (C < 1 || a.steps < 1 || a.rank < 1) return fail("bad --duration/--steps/--rank");
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::u8path(a.out_dir), ec);
    const std::string state = a.out_dir + "/state.gguf", stop = a.out_dir + "/STOP";
    if (!a.resume && std::filesystem::exists(std::filesystem::u8path(state)))
        return fail("state.gguf exists in " + a.out_dir + ": pass --resume or use a new --out");
    std::filesystem::remove(std::filesystem::u8path(stop), ec);

    std::vector<Sa3Sample> data;
    std::vector<float>     silence;
    if (!sa3t_preencode(a, C, data, silence, &err)) return fail(err);

    const auto        tl = std::chrono::steady_clock::now();
    const std::string base_sha = sa3t_file_sha(a.base_dit), inf_sha = sa3t_file_sha(a.inference_dit);
    if (base_sha.empty() || inf_sha.empty()) return fail("cannot hash the base or inference DiT");
    std::string dhash;
    for (const Sa3Sample & s : data)  // latents, text conditioning and timing: any change makes a different run
        dhash += sa3t_hex_key(s.latents.data(), s.latents.size() * 4, s.audio_path) +
                 sa3t_hex_key(s.text.data(), s.text.size() * 4, std::to_string(s.n_samples) + "|" + std::to_string(s.seconds_total));
    char fpbuf[512];
    snprintf(fpbuf, sizeof(fpbuf), "r%lld a%g lr%g wd%g b%g/%g seed%llu C%lld secs%d base%s data%s", (long long) a.rank, a.alpha,
             a.lr, a.wd, a.beta1, a.beta2, (unsigned long long) a.seed, (long long) C, (int) a.train_seconds,
             base_sha.substr(0, 16).c_str(), sa3t_hex_key(dhash.data(), dhash.size(), "").c_str());
    const std::string fp = fpbuf;

    Sa3TrainModel m;
    if (!sa3t_model_load(m, a.base_dit, C, a.rank, a.alpha, a.train_seconds, &err)) return fail(err);
    Sa3Adam opt;
    int64_t done = 0;
    if (a.resume && std::filesystem::exists(std::filesystem::u8path(state))) {
        if (!sa3t_load_state(m, opt, &done, fp, state, &err)) return fail(err);
        jl("{\"type\":\"resume\",\"step\":%lld}", (long long) done);
        fprintf(stderr, "[sa3-train] resumed at step %lld\n", (long long) done);
    } else {
        Sa3Rng r(a.seed, 0, 3);
        for (Sa3Site & st : m.sites) sa3t_site_init(st, a.rank, [&]() { return r.uni(); });
    }
    const long long load_ms = (long long) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tl).count();
    jl("{\"type\":\"start\",\"backend\":\"%s\",\"sites\":%zu,\"rank\":%lld,\"alpha\":%g,\"lr\":%g,\"steps\":%lld,"
       "\"cropLatents\":%lld,\"cropSeconds\":%.3f,\"seed\":%llu,\"startStep\":%lld,\"trainSeconds\":%s,\"baseSha256\":\"%s\","
       "\"inferenceSha256\":\"%s\",\"loadMs\":%lld}",
       ggml_backend_name(m.backend), m.sites.size(), (long long) a.rank, a.alpha, a.lr, (long long) a.steps, (long long) C,
       (double) (C * SA3T_DS) / SA3T_SR, (unsigned long long) a.seed, (long long) done, a.train_seconds ? "true" : "false",
       base_sha.c_str(), inf_sha.c_str(), load_ms);

    GGUFModel inf = {};
    auto export_to = [&](const std::string & path, bool raw) {
        try {
            if (raw) {
                sa3_export::write_raw(path, "stabilityai/stable-audio-3-medium-base sha256:" + base_sha, sa3t_rows(m));
            } else {
                if (!inf.gguf && !gf_load(&inf, a.inference_dit.c_str())) throw std::runtime_error("cannot open " + a.inference_dit);
                sa3_export::write_inference(path, inf, "stabilityai/stable-audio-3-medium sha256:" + inf_sha, sa3t_rows(m));
            }
        } catch (const std::exception & e) {
            err = e.what();
            return false;
        }
        jl("{\"type\":\"export\",\"path\":\"%s\",\"raw\":%s,\"step\":%lld}", lm_json_escape(path).c_str(), raw ? "true" : "false",
           (long long) done);
        return true;
    };

    Sa3StepInputs in;
    bool          stopped = false;
    double        ema = NAN;
    while (done < a.steps) {
        if (std::filesystem::exists(std::filesystem::u8path(stop), ec)) {
            stopped = true;
            break;
        }
        const auto t0 = std::chrono::steady_clock::now();
        sa3t_make_step(data, silence, C, C, a.seed, done, in);
        sa3t_set_inputs(m, in, data[(size_t) in.sample].text, data[(size_t) in.sample].seconds_total);
        const double loss = sa3t_step(m, &err);
        if (!std::isfinite(loss)) return fail(err.empty() ? "non-finite loss at step " + std::to_string(done) : err);
        done++;
        sa3t_optimizer_step(m, opt, done, a);
        ema = std::isfinite(ema) ? 0.98 * ema + 0.02 * loss : loss;
        const long long ms = (long long) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        jl("{\"type\":\"step\",\"step\":%lld,\"totalSteps\":%lld,\"loss\":%.6f,\"lossEma\":%.6f,\"t\":%.4f,\"sample\":%d,"
           "\"cropStart\":%lld,\"maskType\":%d,\"cfgDrop\":%s,\"ms\":%lld}",
           (long long) done, (long long) a.steps, loss, ema, in.t, in.sample, (long long) in.crop_start, in.mask_type,
           in.cfg_drop ? "true" : "false", ms);
        fprintf(stderr, "[sa3-train] step %lld/%lld loss %.5f (ema %.5f) t %.3f %lld ms\n", (long long) done,
                (long long) a.steps, loss, ema, in.t, ms);
        if (a.ckpt_every > 0 && (done % a.ckpt_every == 0 || done == a.steps)) {
            if (!sa3t_save_state(m, opt, done, fp, state, &err)) return fail(err);
            jl("{\"type\":\"checkpoint\",\"step\":%lld,\"path\":\"%s\"}", (long long) done, lm_json_escape(state).c_str());
        }
        if (a.export_every > 0 && done % a.export_every == 0 && done < a.steps) {
            std::filesystem::create_directories(std::filesystem::u8path(a.out_dir + "/snapshots"), ec);
            const std::string p = a.out_dir + "/snapshots/step-" + std::to_string(done) + ".gguf";
            std::filesystem::remove(std::filesystem::u8path(p), ec);
            if (!export_to(p, false)) return fail(err);
        }
    }
    if (stopped) {
        if (!sa3t_save_state(m, opt, done, fp, state, &err)) return fail(err);
        jl("{\"type\":\"stopped\",\"step\":%lld,\"path\":\"%s\"}", (long long) done, lm_json_escape(state).c_str());
        fprintf(stderr, "[sa3-train] stopped at step %lld; --resume continues\n", (long long) done);
        if (inf.gguf) gf_close(&inf);
        sa3t_model_free(m);
        return 0;
    }
    for (const char * name : { "/adapter.gguf", "/adapter-raw.gguf" }) std::filesystem::remove(std::filesystem::u8path(a.out_dir + name), ec);
    const bool ok = export_to(a.out_dir + "/adapter.gguf", false) && export_to(a.out_dir + "/adapter-raw.gguf", true);
    if (inf.gguf) gf_close(&inf);
    sa3t_model_free(m);
    if (!ok) return fail(err);
    jl("{\"type\":\"done\",\"steps\":%lld,\"lossEma\":%.6f,\"adapter\":\"%s\"}", (long long) done, ema,
       lm_json_escape(a.out_dir + "/adapter.gguf").c_str());
    return 0;
}

// ── sa3-train-check: parity against the official capture ───────────────────
//
// Oracle: the official step recomputed in true fp32 on the trainer's exact
// weights (tools/sa3-training-step-fixture.py). The capture itself ran in bf16
// autocast and sits several percent from that oracle (tens of percent on a
// step that scores only a few latents), so gradients are gated relative to
// that gap: the trainer must land much closer to fp32 than upstream's own
// bf16 training does. Bounds and observed values are printed side by side.
static double sa3t_rel(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return NAN;
    double d = 0, n = 0;
    for (size_t i = 0; i < a.size(); i++) d += ((double) a[i] - b[i]) * ((double) a[i] - b[i]), n += (double) b[i] * b[i];
    return n > 0 ? std::sqrt(d / n) : std::sqrt(d);
}

static std::string sa3t_st_meta(const STFile & st, const char * key) {
    yyjson_doc * doc = yyjson_read((const char *) st.mapping + 8, st.data_offset - 8, 0);
    yyjson_val * md  = doc ? yyjson_obj_get(yyjson_doc_get_root(doc), "__metadata__") : nullptr;
    yyjson_val * v   = md ? yyjson_obj_get(md, key) : nullptr;
    std::string  out = v && yyjson_is_str(v) ? yyjson_get_str(v) : "";
    if (doc) yyjson_doc_free(doc);
    return out;
}

static bool sa3t_st_f32(const STFile & st, const std::string & name, std::vector<float> & out) {
    const STEntry * e = st_find(st, name.c_str());
    if (!e || e->dtype != "F32") {
        out.clear();
        return false;
    }
    const size_t n = (e->data_end - e->data_start) / sizeof(float);
    out.assign((const float *) st_data(st, *e), (const float *) st_data(st, *e) + n);
    return true;
}

// Returns 0 on PASS, 1 on FAIL, 77 when the fixtures are absent.
static int sa3_train_check(const std::string & dir, const std::string & base_dit, const std::string & models_dir) {
    int  fails = 0;
    auto check = [&](bool ok, const char * what, double v, double bound, const char * rel = "<=") {
        printf("  %-64s %.3e (%s %.2e) %s\n", what, v, rel, bound, ok ? "ok" : "FAIL");
        fails += !ok;
    };
    STFile f0{};
    if (!st_open(&f0, (dir + "/step0.safetensors").c_str())) {
        printf("sa3-train-check: no fixtures in %s (tools/sa3-training-step-fixture.py writes them)\n", dir.c_str());
        return 77;
    }
    if (!std::filesystem::exists(std::filesystem::u8path(base_dit))) {
        printf("sa3-train-check: no base DiT at %s (convert-sa3.py --dit-only --dit-name sa3-dit-base-BF16.gguf)\n", base_dit.c_str());
        st_close(&f0);
        return 77;
    }
    std::vector<float> v;
    sa3t_st_f32(f0, "noised_inputs", v);
    const int64_t C = (int64_t) v.size() / SA3T_LAT;
    Sa3TrainModel m;
    std::string   err;
    if (!sa3t_model_load(m, base_dit, C, 16, 16.0f, true, &err)) {
        fprintf(stderr, "sa3-train-check: %s\n", err.c_str());
        return 1;
    }
    const bool on_cpu = ggml_backend_dev_type(ggml_backend_get_device(m.backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    printf("sa3-train-check on %s, crop %lld latents, %zu sites\n", ggml_backend_name(m.backend), (long long) C, m.sites.size());
    const char * kinds[3] = { "lora_A", "lora_B", "magnitude" };
    auto pname = [&](const Sa3Site & st, int k) { return st.param + ".parametrizations.weight.0." + kinds[k]; };
    auto load_params = [&](const STFile & st, const char * prefix) {
        bool ok = true;
        for (Sa3Site & s : m.sites) {
            ok = sa3t_st_f32(st, prefix + pname(s, 0), s.a) && ok;
            ok = sa3t_st_f32(st, prefix + pname(s, 1), s.b) && ok;
            ok = sa3t_st_f32(st, prefix + pname(s, 2), s.m) && ok;
        }
        return ok;
    };
    // aggregate rel L2 of per-tensor gradient lists against a fixture's `prefix` tensors
    auto agg_vs = [&](const std::vector<std::vector<float>> & g, const STFile & st, const char * prefix) {
        double num = 0, den = 0;
        size_t k = 0;
        std::vector<float> r;
        for (Sa3Site & s : m.sites)
            for (int j = 0; j < 3; j++, k++) {
                sa3t_st_f32(st, prefix + pname(s, j), r);
                if (k >= g.size() || r.size() != g[k].size()) return (double) NAN;
                for (size_t i = 0; i < r.size(); i++) num += ((double) g[k][i] - r[i]) * ((double) g[k][i] - r[i]), den += (double) r[i] * r[i];
            }
        return std::sqrt(num / den);
    };

    // 1. init: magnitude = bf16(||W0_row||) on the bf16 base (one bf16 ulp allowed
    //    for the reduction order), B = 0, |A| <= 1/sqrt(in)
    {
        double      worst_m = 0;
        std::string bad;
        Sa3Rng      r(1, 0, 3);
        for (Sa3Site & st : m.sites) {
            Sa3Site mine = st;
            sa3t_site_init(mine, 16, [&]() { return r.uni(); });
            std::vector<float> ref, rb, ra;
            sa3t_st_f32(f0, "before." + pname(st, 2), ref);
            sa3t_st_f32(f0, "before." + pname(st, 1), rb);
            sa3t_st_f32(f0, "before." + pname(st, 0), ra);
            for (size_t i = 0; i < ref.size(); i++) worst_m = std::max(worst_m, (double) std::fabs(mine.m[i] - ref[i]) / std::fabs(ref[i]));
            const float bound = 1.0f / std::sqrt((float) st.in);
            bool        ok    = true;
            for (float x : rb) ok = ok && x == 0.0f;
            for (float x : ra) ok = ok && std::fabs(x) <= bound * (1.0f + 1.0f / 256) && ggml_bf16_to_fp32(ggml_fp32_to_bf16(x)) == x;
            for (float x : mine.a) ok = ok && std::fabs(x) <= bound * (1.0f + 1.0f / 256) && ggml_bf16_to_fp32(ggml_fp32_to_bf16(x)) == x;
            if (!ok && bad.empty()) bad = st.stem;
        }
        check(worst_m <= 1.0 / 128, "init magnitude vs upstream (worst element rel, 1 bf16 ulp)", worst_m, 1.0 / 128);
        if (!bad.empty()) printf("    first site outside the upstream A/B init: %s\n", bad.c_str());
        check(bad.empty(), "init B = 0, |A| <= 1/sqrt(in), A bf16-exact, ours and upstream (1 = no)", bad.empty() ? 0.0 : 1.0, 0.0);
    }

    // 2. steps: output, loss and all gradients vs the fp32 official recompute
    const double tol_out = on_cpu ? 1e-3 : 1e-2, ratio = 0.35;
    std::map<std::string, std::vector<std::vector<float>>> grads;
    std::map<std::string, double> own;
    const char * names[4] = { "step0", "step1", "step2", "step1-drop" };
    for (const char * name : names) {
        STFile st{};
        if (!st_open(&st, (dir + "/" + name + ".safetensors").c_str())) {
            printf("  %s missing\n", name);
            fails++;
            continue;
        }
        Sa3StepInputs      in;
        std::vector<float> tv, im, prompt;
        bool ok = load_params(st, "before.");
        ok = sa3t_st_f32(st, "noised_inputs", in.x_t) && sa3t_st_f32(st, "targets", in.target) &&
             sa3t_st_f32(st, "local_add_cond", in.local) && sa3t_st_f32(st, "augmented_padding_mask", in.aug_pad) &&
             sa3t_st_f32(st, "loss_mask", in.loss_mask) && sa3t_st_f32(st, "inpaint_mask", im) && sa3t_st_f32(st, "t", tv) &&
             sa3t_st_f32(st, "prompt", prompt) && ok;
        if (!ok) {
            printf("  %s: fixture tensors missing\n", name);
            fails++;
            st_close(&st);
            continue;
        }
        in.t        = tv[0];
        in.cfg_drop = sa3t_st_meta(st, "cfg_dropped") == "1";
        in.ctx_mask.resize(im.size());
        for (size_t j = 0; j < im.size(); j++) in.ctx_mask[j] = (im[j] > 0.5f && in.aug_pad[j] > 0.5f) ? 1.0f : 0.0f;
        const int secs = atoi(sa3t_st_meta(st, "seconds_total").c_str());
        sa3t_set_inputs(m, in, prompt, secs);
        const auto      t0   = std::chrono::steady_clock::now();
        const double    loss = sa3t_step(m, &err);
        const long long ms   = (long long) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        if (!std::isfinite(loss)) {
            printf("  %s: step failed: %s\n", name, err.c_str());
            fails++;
            st_close(&st);
            continue;
        }
        std::vector<float> lf, lc, of, pm;
        sa3t_st_f32(st, "loss_fp32", lf);
        sa3t_st_f32(st, "loss_capture", lc);
        sa3t_st_f32(st, "output_fp32", of);
        sa3t_st_f32(st, "padding_mask", pm);
        std::vector<float> out_cm(of.size());
        for (int64_t j = 0; j < C; j++)
            for (int c = 0; c < SA3T_LAT; c++) out_cm[(size_t) (c * C + j)] = m.out_host[(size_t) (j * SA3T_LAT + c)];
        int64_t npad = 0, naug = 0, nctx = 0;
        for (float p : pm) npad += p < 0.5f;
        for (float p : in.aug_pad) naug += p < 0.5f;
        for (float p : in.ctx_mask) nctx += p > 0.5f;
        printf("  %s: loss %.7f (fp32 oracle %.7f, bf16 capture %.7f); %lld padded latents (%lld after silence extension), "
               "%lld inpaint-context, cfg drop %d, %lld ms\n",
               name, loss, lf[0], lc[0], (long long) npad, (long long) naug, (long long) nctx, (int) in.cfg_drop, ms);
        char buf[160];
        snprintf(buf, sizeof(buf), "%s output vs fp32 oracle", name);
        check(sa3t_rel(out_cm, of) <= tol_out, buf, sa3t_rel(out_cm, of), tol_out);
        snprintf(buf, sizeof(buf), "%s loss vs fp32 oracle", name);
        check(std::fabs(loss - lf[0]) / lf[0] <= tol_out, buf, std::fabs(loss - lf[0]) / lf[0], tol_out);
        std::vector<std::vector<float>> mine;
        for (Sa3Site & s : m.sites) {
            mine.push_back(s.ga);
            mine.push_back(s.gb);
            mine.push_back(s.gm);
        }
        // the capture's own distance from the oracle sets the scale
        std::vector<std::vector<float>> cap;
        std::vector<float>              r;
        for (Sa3Site & s : m.sites)
            for (int j = 0; j < 3; j++) sa3t_st_f32(st, "grad_capture." + pname(s, j), r), cap.push_back(r);
        const double ours = agg_vs(mine, st, "grad_fp32."), gap = agg_vs(cap, st, "grad_fp32.");
        own[name] = ours;
        snprintf(buf, sizeof(buf), "%s %zu grads vs fp32 oracle (aggregate rel)", name, mine.size());
        check(ours <= ratio * gap, buf, ours, ratio * gap);
        printf("    bound = %.2f x the bf16 capture's own distance from the oracle (%.3e)\n", ratio, gap);
        grads[name] = std::move(mine);
        st_close(&st);
    }
    // negative controls: our gradients must sit much closer to their own oracle
    // than to another step's, or the kept branch's when CFG dropout fired
    auto control = [&](const char * ours, const char * other, const char * what) {
        STFile st{};
        if (!grads.count(ours) || !st_open(&st, (dir + "/" + other + ".safetensors").c_str())) {
            fails++;
            return;
        }
        const double d = agg_vs(grads[ours], st, "grad_fp32.");
        st_close(&st);
        check(d >= 2.0 * own[ours], what, d, 2.0 * own[ours], ">=");
    };
    control("step1", "step2", "control: step1 grads vs step2 oracle (>= 2x own error)");
    control("step1-drop", "step1", "control: CFG-drop grads vs kept-branch oracle (>= 2x own error)");

    // 3. export round trip: perturbed adapters -> dora-rows-v1 export against the
    //    base GGUF -> inference loader (sa3_dit_load, merge mode) -> sa3_dit_forward
    //    must equal the training forward on the same inputs. The seconds site is
    //    set to the identity (B = 0, m = ||W||) because the export cannot carry it.
    {
        STFile st{};
        std::vector<float> prompt, tv;
        Sa3StepInputs      in;
        bool               ok = st_open(&st, (dir + "/step2.safetensors").c_str()) && load_params(st, "before.");
        if (ok) {
            sa3t_st_f32(st, "noised_inputs", in.x_t), sa3t_st_f32(st, "targets", in.target), sa3t_st_f32(st, "local_add_cond", in.local);
            sa3t_st_f32(st, "augmented_padding_mask", in.aug_pad), sa3t_st_f32(st, "loss_mask", in.loss_mask), sa3t_st_f32(st, "t", tv);
            sa3t_st_f32(st, "prompt", prompt);
            in.t = tv[0];
            in.ctx_mask.assign(in.aug_pad.size(), 0.0f);
        }
        const int secs = ok ? atoi(sa3t_st_meta(st, "seconds_total").c_str()) : 0;
        if (ok) st_close(&st);
        Sa3Rng r(7, 0, 9);
        for (Sa3Site & s : m.sites) {
            if (s.stem == "seconds") {
                std::fill(s.b.begin(), s.b.end(), 0.0f);
                for (size_t o = 0; o < s.m.size(); o++) s.m[o] = std::sqrt(s.nsq[o]);
                continue;
            }
            for (float & x : s.b) x = (float) (0.01 * r.normal());
            for (float & x : s.m) x *= (float) (1.0 + 0.05 * r.normal());
        }
        std::vector<float> trained, plain;
        if (ok) {
            sa3t_set_inputs(m, in, prompt, secs);
            ok = std::isfinite(sa3t_step(m, &err));
            trained = m.out_host;
        }
        const std::string tmp = dir + "/.roundtrip-adapter.gguf";
        std::error_code   ec;
        std::filesystem::remove(std::filesystem::u8path(tmp), ec);
        if (ok) {
            try {
                GGUFModel gb = {};
                if (!gf_load(&gb, base_dit.c_str())) throw std::runtime_error("cannot open base");
                sa3_export::write_inference(tmp, gb, "round-trip test", sa3t_rows(m));
                gf_close(&gb);
            } catch (const std::exception & e) {
                printf("  export failed: %s\n", e.what());
                ok = false;
            }
        }
        if (ok) {
            // inference-side inputs: raw cross tokens [prompt | seconds embedding], global = seconds embedding
            SA3SecondsEmbedder se;
            ok = sa3_seconds_embedder_load(&se, base_dit.c_str());
            std::vector<float> sec(768), cross(prompt);
            if (ok) sa3_seconds_embed(se, (float) secs, sec.data());
            cross.insert(cross.end(), sec.begin(), sec.end());
            std::vector<uint8_t> pmask(in.aug_pad.size());
            for (size_t j = 0; j < pmask.size(); j++) pmask[j] = in.aug_pad[j] > 0.5f;
            SA3DiT inf{};
            std::vector<Sa3AdapterSpec> specs = { { tmp, 1.0f } };
            ok = ok && sa3_dit_load(&inf, base_dit.c_str(), &specs);
            std::vector<float> out_cm((size_t) (SA3T_LAT * C));
            if (ok) {
                sa3_dit_forward(&inf, in.x_t.data(), (float) in.t, cross.data(), m.S_c, sec.data(), in.local.data(), pmask.data(), C, out_cm.data());
                sa3_dit_free(&inf);
                std::vector<float> got((size_t) (SA3T_LAT * C));
                for (int64_t j = 0; j < C; j++)
                    for (int c = 0; c < SA3T_LAT; c++) got[(size_t) (j * SA3T_LAT + c)] = out_cm[(size_t) (c * C + j)];
                // reference scale: the same inputs with the adapters at their identity
                for (Sa3Site & s : m.sites) std::fill(s.b.begin(), s.b.end(), 0.0f), std::transform(s.nsq.begin(), s.nsq.end(), s.m.begin(), [](float v) { return std::sqrt(v); });
                ok = std::isfinite(sa3t_step(m, &err));
                plain = m.out_host;
                const double e = sa3t_rel(got, trained), effect = sa3t_rel(plain, trained);
                printf("    adapter effect on the output %.3e (training forward with vs without the perturbed adapters)\n", effect);
                // floor: the loader merges each adapted weight back into BF16 (~2^-9 per weight),
                // which alone puts an exact-f32 CPU forward ~5e-3 from the training forward
                check(e <= 0.1 * effect && e <= 1e-2, "export round trip: inference merge forward vs training forward", e,
                      std::min(0.1 * effect, 1e-2));
            }
        }
        std::filesystem::remove(std::filesystem::u8path(tmp), ec);
        if (!ok) {
            printf("  export round trip could not run: %s\n", err.c_str());
            fails++;
        }
    }

    // 4. AdamW chained over the capture's own gradients vs its updated adapters
    {
        Sa3TrainArgs a;
        Sa3Adam      opt;
        double       worst = 0;
        bool         ok    = load_params(f0, "before.");
        for (int k = 0; k < 3 && ok; k++) {
            STFile st{};
            if (!st_open(&st, (dir + "/step" + std::to_string(k) + ".safetensors").c_str())) {
                ok = false;
                break;
            }
            for (Sa3Site & s : m.sites) {
                ok = sa3t_st_f32(st, "grad_capture." + pname(s, 0), s.ga) && ok;
                ok = sa3t_st_f32(st, "grad_capture." + pname(s, 1), s.gb) && ok;
                ok = sa3t_st_f32(st, "grad_capture." + pname(s, 2), s.gm) && ok;
            }
            sa3t_optimizer_step(m, opt, k + 1, a);
            for (Sa3Site & s : m.sites) {
                const std::vector<float> * p[3] = { &s.a, &s.b, &s.m };
                for (int j = 0; j < 3; j++) {
                    std::vector<float> ref;
                    sa3t_st_f32(st, "after." + pname(s, j), ref);
                    for (size_t i = 0; i < ref.size() && i < p[j]->size(); i++)
                        worst = std::max(worst, (double) std::fabs((*p[j])[i] - ref[i]) / (1e-6 + 2e-5 * std::fabs(ref[i])));
                }
            }
            st_close(&st);
        }
        check(ok && worst <= 1.0, "AdamW steps 0-2 vs captured updates (worst/(1e-6+2e-5|p|))", worst, 1.0);
    }

    // 5. frozen encoders: SAME (short unchunked crop, 30 s chunked) and T5Gemma
    {
        SA3Same       enc{};
        SA3T5GemmaEnc t5{};
        if (!sa3_same_load(&enc, (models_dir + "/sa3-same-enc-F16.gguf").c_str(), true) ||
            !sa3_t5gemma_load(&t5, (models_dir + "/sa3-text-enc-BF16.gguf").c_str())) {
            printf("  encoders missing in %s\n", models_dir.c_str());
            fails++;
        } else {
            std::vector<float> audio, ref, lat;
            sa3t_st_f32(f0, "audio_crop", audio);
            sa3t_st_f32(f0, "latents_capture", ref);
            sa3t_same_encode(&enc, audio.data(), (int64_t) audio.size() / 2, lat);
            check(sa3t_rel(lat, ref) <= 3e-2, "SAME 3 s crop (unchunked) vs bf16 capture", sa3t_rel(lat, ref), 3e-2);
            STFile sl{};
            if (st_open(&sl, (dir + "/same-long.safetensors").c_str())) {
                sa3t_st_f32(sl, "audio", audio);
                sa3t_st_f32(sl, "latents", ref);
                sa3t_same_encode(&enc, audio.data(), (int64_t) audio.size() / 2, lat);
                // seams: the latents each chunk boundary hands over (+-4 around every trim point)
                const int64_t      L = (int64_t) ref.size() / SA3T_LAT;
                std::vector<bool>  seam((size_t) L, false);
                std::vector<int64_t> starts;  // chunk starts in latents, as sa3t_same_encode
                for (int64_t s = 0; s + SA3T_CHUNK <= L; s += SA3T_CHUNK - SA3T_OVERLAP) starts.push_back(s);
                if (starts.back() != L - SA3T_CHUNK) starts.push_back(L - SA3T_CHUNK);
                for (size_t i = 1; i < starts.size(); i++)
                    for (int64_t j = starts[i] + SA3T_OVERLAP / 2 - 4; j < starts[i] + SA3T_OVERLAP / 2 + 4; j++)
                        if (j >= 0 && j < L) seam[(size_t) j] = true;
                double ds = 0, ns = 0, di = 0, ni = 0;
                for (int c = 0; c < SA3T_LAT; c++)
                    for (int64_t j = 0; j < L; j++) {
                        const double a = lat[(size_t) (c * L + j)], b = ref[(size_t) (c * L + j)];
                        (seam[(size_t) j] ? ds : di) += (a - b) * (a - b);
                        (seam[(size_t) j] ? ns : ni) += b * b;
                    }
                const double es = std::sqrt(ds / ns), ei = std::sqrt(di / ni);
                check(sa3t_rel(lat, ref) <= 1e-2, "SAME 30 s (chunked) vs official fp32 chunked", sa3t_rel(lat, ref), 1e-2);
                check(es <= 1.5 * ei, "SAME 30 s seam latents vs interior error (seams no worse)", es, 1.5 * ei);
                st_close(&sl);
            } else {
                printf("  same-long.safetensors missing\n");
                fails++;
            }
            std::vector<float> pad, text, p32, pcap, pm;
            const STEntry *    ide = st_find(f0, "prompt_ids");
            sa3t_st_f32(f0, "prompt_fp32", p32);
            sa3t_st_f32(f0, "prompt", pcap);
            sa3t_st_f32(f0, "prompt_mask", pm);
            int n = 0;
            for (float x : pm) n += x > 0.5f;
            if (ide && ide->dtype == "I32" && !p32.empty() && sa3t_read_padding(base_dit, pad, &err)) {
                sa3t_text_encode(&t5, (const int32_t *) st_data(f0, *ide), n, pad, text);
                // the T5Gemma port lands ~7e-3 from fp32 on CUDA but ~3e-2 on ggml-cpu (its rows reach norm ~116)
                const double t5_tol = on_cpu ? 5e-2 : 2e-2;
                check(sa3t_rel(text, p32) <= t5_tol, "T5Gemma + base padding vs official fp32 T5Gemma", sa3t_rel(text, p32), t5_tol);
                printf("    (the bf16 capture's T5Gemma sits %.3e from fp32; upstream runs T5Gemma in bf16)\n", sa3t_rel(pcap, p32));
            } else {
                printf("  prompt_ids, prompt_fp32 or the base padding embedding missing\n");
                fails++;
            }
            sa3_same_free(&enc);
            sa3_t5gemma_free(&t5);
        }
    }
    st_close(&f0);
    sa3t_model_free(m);
    printf("sa3-train-check: %s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
