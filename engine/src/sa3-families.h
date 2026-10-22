#pragma once
// sa3-families.h — StableAudio3 boot-gate probe (directory/filename scan
// only, no weight reads, no GGUF header parse, no ONNX session creation).
//
// Mirrors minimax/mm3-model.h's mm3_weights_present in shape and intent (see
// that function's own comment for the pattern this follows), but lives
// outside engine/src/minimax/ on purpose: sa3-refine.h's own header comment
// says SA3 is deliberately self-contained (ONNX Runtime OR GGML, no shared
// subtree the way MM3 has one), so there is no "sa3/sa3-server.h" hook to
// piggyback on the way hot-step-families.h piggybacks on MM3's. A standalone
// header keeps hot-step-families.h's include list flat and lets any TU pull
// this in without pulling in sa3-refine.h's ONNX Runtime / GGML backends.
//
// Two ways an SA3 install can be complete enough to serve (same "onnx vs
// gguf" split hot-step-server.cpp's /sa3-synth and /sa3-refine routes probe
// with plain file_exists() checks — see hot-step-server.cpp's sa3-refine and
// sa3-synth handlers):
//   - GGUF: <models>/sa3-dit-<quant>.gguf (any quant token — "BF16" today,
//     matched by prefix "sa3-dit" + ".gguf" extension so a future quant
//     doesn't need this file touched).
//   - ONNX: <models>/onnx/sa3/sa3-dit.onnx (the DiT graph; the other four
//     ONNX graphs and the tokenizer files are assumed present alongside it,
//     same assumption the /sa3-synth and /sa3-props routes make).
// Either one present is enough to report "StableAudio3 could serve" — this
// is a boot gate, not a full readiness check.

#include <cstdio>
#include <string>
#include <vector>

#ifdef _WIN32
#    define SA3_SEP "\\"
#    include <windows.h>
#else
#    define SA3_SEP "/"
#    include <dirent.h>
#endif

// List plain files in a directory. Deliberately local rather than reusing
// mm3_list_dir or a registry helper — this header is meant to be includable
// on its own, same reasoning mm3_list_dir's own comment gives.
static void sa3_list_dir(const std::string & dir, std::vector<std::string> * names) {
#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    HANDLE           h = FindFirstFileA((dir + "\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            names->push_back(fd.cFileName);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR * d = opendir(dir.c_str());
    if (!d) {
        return;
    }
    while (struct dirent * e = readdir(d)) {
        if (e->d_type != DT_DIR) {
            names->push_back(e->d_name);
        }
    }
    closedir(d);
#endif
}

static bool sa3_file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fclose(f);
    return true;
}

static bool sa3_weights_present(const char * models_dir) {
    if (!models_dir || !models_dir[0]) {
        return false;
    }
    const std::string root = models_dir;

    // ONNX: <models>/onnx/sa3/sa3-dit.onnx.
    if (sa3_file_exists(root + SA3_SEP "onnx" SA3_SEP "sa3" SA3_SEP "sa3-dit.onnx")) {
        return true;
    }

    // GGUF: <models>/sa3-dit-<quant>.gguf (any quant token).
    std::vector<std::string> names;
    sa3_list_dir(root, &names);
    for (const auto & n : names) {
        if (n.size() > 12 && n.compare(0, 8, "sa3-dit-") == 0 && n.compare(n.size() - 5, 5, ".gguf") == 0) {
            return true;
        }
    }
    return false;
}
