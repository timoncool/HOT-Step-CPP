#pragma once
// hot-step-fsutf8.h — UTF-8-correct filesystem primitives (HOT-Step addition).
//
// Every path this engine handles arrives as UTF-8: manifests and dataset.json
// are JSON, written by the Node server, and JSON is UTF-8 by definition. The
// CRT's narrow file calls (fopen/_stat64/_mkdir/rename/remove/system) do NOT
// take UTF-8 on Windows — they decode their char* in the process ANSI codepage.
// Any path outside 7-bit ASCII is therefore mis-decoded and the file silently
// "does not exist".
//
// This is not theoretical. A 14-track album preprocessed to ZERO songs, every
// sample skipped with "audio file not found", because the filenames carried a
// literal U+0080 (mojibake from a rip: the UTF-8 bytes of '…' stored as three
// Latin-1 characters). cp1252 cannot represent U+0080 at all, so no narrow
// round-trip could ever have opened them. The training pipeline then failed one
// stage later with the far less helpful "no cached songs" (2026-08-02).
//
// On Windows these convert UTF-8 -> UTF-16 and call the _w variants. Elsewhere
// they are the plain calls unchanged: POSIX paths are bytes and UTF-8 is
// already the right thing to pass.
//
// Prefer these over the bare CRT calls for ANY path that can come from a
// dataset, a manifest, or a user-chosen directory.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <direct.h>
#    include <windows.h>
#    include <shellapi.h>  // CommandLineToArgvW
#    ifdef _MSC_VER
#        pragma comment(lib, "shell32.lib")
#    endif
#endif

// stat buffer type — 64-bit sizes on both platforms.
#ifdef _WIN32
#    define HS_STAT_T struct _stat64
#    define HS_ISREG(m) (((m) &_S_IFREG) != 0)
#else
#    define HS_STAT_T struct stat
#    define HS_ISREG(m) S_ISREG(m)
#endif

#ifdef _WIN32

// UTF-8 -> UTF-16. Invalid sequences are substituted rather than rejected: a
// mis-encoded name must still produce a lookup that fails honestly, never an
// empty string that would silently retarget the call at the CWD.
static inline std::wstring hs_widen(const char * utf8) {
    if (!utf8 || !*utf8) {
        return std::wstring();
    }
    const int len = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
    if (len <= 0) {
        return std::wstring();
    }
    std::vector<wchar_t> w((size_t) len);
    MultiByteToWideChar(CP_UTF8, 0, utf8, -1, w.data(), len);
    return std::wstring(w.data());  // -1 length includes the NUL; ctor stops at it
}

static inline std::wstring hs_widen(const std::string & utf8) {
    return hs_widen(utf8.c_str());
}

#endif  // _WIN32

// ─── the wrappers ───────────────────────────────────────────────────────────

static inline FILE * hs_fopen(const char * path, const char * mode) {
#ifdef _WIN32
    // Modes are always ASCII ("rb", "wb", …) so a widening cast is exact.
    const std::string  m(mode ? mode : "");
    const std::wstring wmode(m.begin(), m.end());
    return _wfopen(hs_widen(path).c_str(), wmode.c_str());
#else
    return fopen(path, mode);
#endif
}

static inline FILE * hs_fopen(const std::string & path, const char * mode) {
    return hs_fopen(path.c_str(), mode);
}

static inline int hs_stat(const std::string & path, HS_STAT_T * st) {
#ifdef _WIN32
    return _wstat64(hs_widen(path).c_str(), st);
#else
    return stat(path.c_str(), st);
#endif
}

static inline bool hs_path_exists(const std::string & path) {
    HS_STAT_T st;
    return hs_stat(path, &st) == 0;
}

static inline bool hs_file_exists(const std::string & path) {
    HS_STAT_T st;
    return hs_stat(path, &st) == 0 && HS_ISREG(st.st_mode);
}

static inline int hs_remove(const std::string & path) {
#ifdef _WIN32
    return _wremove(hs_widen(path).c_str());
#else
    return remove(path.c_str());
#endif
}

static inline int hs_rename(const std::string & from, const std::string & to) {
#ifdef _WIN32
    return _wrename(hs_widen(from).c_str(), hs_widen(to).c_str());
#else
    return rename(from.c_str(), to.c_str());
#endif
}

static inline int hs_mkdir(const std::string & path) {
#ifdef _WIN32
    return _wmkdir(hs_widen(path).c_str());
#else
    return mkdir(path.c_str(), 0755);
#endif
}

// The command line as UTF-8. On Windows main()'s argv is decoded in the ANSI
// codepage, so a path argument with any character outside it arrives mangled;
// this re-reads the UTF-16 command line instead.
static inline std::vector<std::string> hs_utf8_args(int argc, char ** argv) {
#ifdef _WIN32
    int     n = 0;
    LPWSTR * w = CommandLineToArgvW(GetCommandLineW(), &n);
    if (w) {
        std::vector<std::string> out;
        for (int i = 0; i < n; ++i) {
            const int len = WideCharToMultiByte(CP_UTF8, 0, w[i], -1, nullptr, 0, nullptr, nullptr);
            std::string s(len > 0 ? (size_t) len - 1 : 0, '\0');
            if (len > 1) {
                WideCharToMultiByte(CP_UTF8, 0, w[i], -1, &s[0], len, nullptr, nullptr);
            }
            out.push_back(s);
        }
        LocalFree(w);
        return out;
    }
#endif
    return std::vector<std::string>(argv, argv + argc);
}

// Run a shell command whose text may embed UTF-8 paths. cmd.exe is handed the
// UTF-16 form, so a non-ASCII filename survives into the child's argv — passing
// the UTF-8 bytes to the narrow system() would mangle them exactly as the file
// calls above do.
static inline int hs_system(const std::string & cmd) {
#ifdef _WIN32
    return _wsystem(hs_widen(cmd).c_str());
#else
    return system(cmd.c_str());
#endif
}
