// sa3-preview-resolver.h: SA3 training snapshot -> adapter spec for /sa3-synth.
//
// Checkpoint previews render a trainer inference export
//   <adapters>/sa3-medium/<dataset>/<adapter>/<stamp>/snapshots/step-N.gguf
// (train/sa3-train-run.h, --export-every). This is an internal path beside the
// public adapter names: models/sa3-adapters/<name>.gguf and its name parser
// and catalogue are untouched, and snapshots never appear in /sa3/props.
//
// Bounds, in order:
//   - run = "<dataset>/<adapter>/<stamp>" with the server's slug, adapterName
//     and run-stamp charsets; step = positive decimal. No separators beyond
//     those two slashes, so no "..", no drive letters, no absolute paths.
//   - the canonical file (symlinks and junctions resolved) must be a regular
//     file under the canonical <adapters>/sa3-medium root.
//   - the step file is read ONCE. Those bytes are hashed and parked as an
//     immutable content-addressed copy, <root>/.preview-cache/<sha256>.gguf;
//     metadata is validated from that copy and the loader opens that copy.
//     The trainer may delete and rewrite a step file at any time
//     (sa3-train-run.h removes it before export), and none of that can reach
//     the validated, hashed or loaded bytes.
//   - the GGUF is a dora-rows-v1 sa3-adapter whose sa3.inference_base_id
//     names the sha256 of the DiT this request loads (wrong base = 400).
//   - the ModelKey carries the content sha256, so a rewritten step file loads
//     fresh instead of reusing the cached merge.
// ".preview-cache" can never be a dataset slug (slugs start alphanumeric).

#pragma once

#include "train/yue2-aitk-sha256.h"

#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <mutex>
#include <regex>
#include <string>
#include <tuple>
#include <vector>

static std::string sa3_preview_sha(const std::filesystem::path & p) {
    yue2_aitk::sha256::digest d;
    return yue2_aitk::sha256::file(p, d) ? d.hex() : std::string();
}

// The DiT is ~3 GB: hash once per (size, mtime), not per preview.
static std::string sa3_preview_dit_sha(const std::filesystem::path & p) {
    static std::mutex                                                          mtx;
    static std::map<std::string, std::tuple<uintmax_t, int64_t, std::string>> cache;
    std::error_code ec;
    const uintmax_t size = std::filesystem::file_size(p, ec);
    if (ec) return {};
    const int64_t mtime = (int64_t) std::filesystem::last_write_time(p, ec).time_since_epoch().count();
    if (ec) return {};
    std::lock_guard<std::mutex> lock(mtx);
    auto it = cache.find(p.u8string());
    if (it != cache.end() && std::get<0>(it->second) == size && std::get<1>(it->second) == mtime)
        return std::get<2>(it->second);
    std::string sha = sa3_preview_sha(p);
    if (!sha.empty()) cache[p.u8string()] = { size, mtime, sha };
    return sha;
}

static std::string sa3_preview_gguf_str(const gguf_context * g, const char * key) {
    const int64_t i = gguf_find_key(g, key);
    return (i >= 0 && gguf_get_kv_type(g, i) == GGUF_TYPE_STRING) ? gguf_get_val_str(g, i) : "";
}

// snapshot_scale: the whole string must parse to a finite float. strtof alone
// accepts "nan", "inf", overflow (HUGE_VALF) and trailing junk, and a
// non-finite strength poisons every merged weight.
static bool sa3_parse_snapshot_scale(const std::string & s, float & out) {
    if (s.empty()) return false;
    char *      end = nullptr;
    const float v   = std::strtof(s.c_str(), &end);
    if (end != s.c_str() + s.size() || !std::isfinite(v)) return false;
    out = v;
    return true;
}

// Park `bytes` (sha256 `sha`) as <cache>/<sha>.gguf, written once via a temp
// file + rename and never rewritten. An existing copy is reused only when its
// content still hashes to its name. Keeps the newest 16 copies.
static bool sa3_preview_park(const std::filesystem::path & cache, const std::string & sha,
                             const std::vector<char> & bytes, std::filesystem::path & out) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(cache, ec);
    out = cache / (sha + ".gguf");
    if (!(fs::is_regular_file(out, ec) && sa3_preview_sha(out) == sha)) {
        const fs::path tmp = cache / (sha + ".part");
        {
            std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
            o.write(bytes.data(), (std::streamsize) bytes.size());
            if (!o) return false;
        }
        fs::rename(tmp, out, ec);
        if (ec) {
            fs::remove(tmp, ec);
            return false;
        }
    }
    fs::last_write_time(out, fs::file_time_type::clock::now(), ec);
    std::vector<fs::directory_entry> all;
    for (auto & e : fs::directory_iterator(cache, ec))
        if (e.is_regular_file() && e.path().extension() == ".gguf") all.push_back(e);
    if (all.size() > 16) {
        std::sort(all.begin(), all.end(), [](const fs::directory_entry & a, const fs::directory_entry & b) {
            std::error_code e;
            return a.last_write_time(e) > b.last_write_time(e);
        });
        for (size_t i = 16; i < all.size(); i++) fs::remove(all[i].path(), ec);
    }
    return true;
}

// Resolve one snapshot. On success fills `path` (the parked copy, for the
// loader) and `sig` (the adapter_stack entry, content hash included) and
// returns "". Otherwise returns the client-facing error. `after_capture` is a
// test seam: it runs right after the step file was read.
static std::string sa3_resolve_snapshot(const char * adapters_dir, const std::string & run, const std::string & step,
                                        const std::string & dit_path, float scale, std::string & path,
                                        std::string & sig, const std::function<void()> & after_capture = {}) {
    namespace fs = std::filesystem;
    if (!adapters_dir || !*adapters_dir) return "SA3 snapshots need the engine's --adapters directory";
    static const std::regex run_re("[A-Za-z0-9][A-Za-z0-9_-]{0,127}/[A-Za-z0-9][A-Za-z0-9_-]{0,63}/[0-9][0-9_-]{0,63}");
    static const std::regex step_re("[1-9][0-9]{0,8}");
    if (!std::regex_match(run, run_re)) return "Invalid snapshot_run (expected <dataset>/<adapter>/<stamp>)";
    if (!std::regex_match(step, step_re)) return "Invalid snapshot_step (expected a positive integer)";

    std::error_code ec;
    const fs::path root = fs::canonical(fs::u8path(adapters_dir) / "sa3-medium", ec);
    if (ec) return "No SA3 training runs under the adapters directory";
    const fs::path file = fs::canonical(fs::u8path(adapters_dir) / "sa3-medium" / fs::u8path(run) / "snapshots" /
                                            ("step-" + step + ".gguf"), ec);
    if (ec || !fs::is_regular_file(file, ec)) return "SA3 snapshot not found: " + run + " step " + step;
    // Component-wise containment: a sibling such as "sa3-medium-x" must not pass.
    auto r = root.begin(), f = file.begin();
    for (; r != root.end(); ++r, ++f)
        if (f == file.end() || *r != *f) return "SA3 snapshot resolves outside the training runs";

    // One read of the step file; everything below works on these bytes.
    std::vector<char> bytes;
    {
        std::ifstream in(file, std::ios::binary);
        if (!in) return "Cannot read the SA3 snapshot";
        bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        if (in.bad()) return "Cannot read the SA3 snapshot";
    }
    if (after_capture) after_capture();
    const std::string snap_sha = yue2_aitk::sha256::bytes(bytes.data(), bytes.size()).hex();
    fs::path          parked;
    if (!sa3_preview_park(root / ".preview-cache", snap_sha, bytes, parked)) return "Cannot stage the SA3 snapshot";

    gguf_init_params gp = { /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context *   g  = gguf_init_from_file(parked.u8string().c_str(), gp);
    if (!g) return "SA3 snapshot is not a readable GGUF";
    const std::string arch = sa3_preview_gguf_str(g, "general.architecture");
    const std::string fmt  = sa3_preview_gguf_str(g, "sa3.adapter_format");
    const std::string base = sa3_preview_gguf_str(g, "sa3.inference_base_id");
    gguf_free(g);
    if (arch != "sa3-adapter" || fmt != "dora-rows-v1") return "SA3 snapshot is not a dora-rows-v1 inference export";

    const std::string dit_sha = sa3_preview_dit_sha(fs::u8path(dit_path));
    if (dit_sha.empty()) return "Cannot hash the SA3 generation DiT";
    if (base != "stabilityai/stable-audio-3-medium sha256:" + dit_sha)
        return "SA3 snapshot was exported for a different DiT (" + base + "), not " + fs::u8path(dit_path).filename().u8string();

    path = parked.u8string();
    // model-store parses "path=scale" entries; the "#sha256:" entry has no '='
    // so the parser skips it, but it is still part of the cache key.
    sig  = path + "=" + std::to_string(scale) + ";#sha256:" + snap_sha;
    return "";
}
