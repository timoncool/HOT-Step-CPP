// HOTSTEP-353 probe: lifetime of the Metal half pre-pass scratch over varying sizes.
// Launches base-forward BF16 GEMMs in graphs built with grads (half pre-pass) asynchronously, with
// growing then shrinking N so the scratch is replaced while earlier command buffers still use it.
// Every output must match the same GEMM in a graph without grads (today's kernel) bit for bit, and
// the scratch counts ggml-metal logs must show every allocated buffer freed by ggml_metal_free.
//   usage: scratch-life        exit 0 pass, 1 fail, 2 setup error
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::vector<std::string> g_log;
static void log_cb(ggml_log_level, const char * text, void *) {
    if (strstr(text, "half pre-pass scratch")) {
        g_log.push_back(text);
        fputs(text, stdout);
    }
}

static uint32_t rng = 12345;
static float frand() { rng = rng*1664525u + 1013904223u; return (rng >> 8)*(1.0f/16777216.0f)*2.0f - 1.0f; }

// values across half's normal and subnormal range, and below it, so steps take both paths
static void fill(ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    std::vector<uint8_t> buf(ggml_nbytes(t));
    for (int64_t i = 0; i < n; ++i) {
        float v = frand();
        if (i % 4099 == 0) v = std::ldexp(v, -30);
        if (t->type == GGML_TYPE_F32) {
            memcpy(buf.data() + 4*i, &v, 4);
        } else {
            uint32_t u; memcpy(&u, &v, 4);
            const uint16_t h = uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
            memcpy(buf.data() + 2*i, &h, 2);
        }
    }
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
}

struct Case {
    ggml_context * ctx;
    ggml_backend_buffer_t buf;
    ggml_cgraph * g_hp, * g_ref;
    ggml_tensor * y_hp, * y_ref;
};

int main() {
    ggml_log_set(log_cb, nullptr);
    ggml_backend_t be = ggml_backend_metal_init();
    if (!be) { fprintf(stderr, "no Metal backend\n"); return 2; }

    const int K = 2048, M = 128;
    std::vector<int> ns;
    for (int n = 2048; n <= 24576; n += 2048) ns.push_back(n);
    for (int n = 24576 - 3072; n >= 1024; n -= 3072) ns.push_back(n);
    for (int n = 2048; n <= 24576; n += 4096) ns.push_back(n + 1); // partial tiles

    std::vector<Case> cases;
    for (int n : ns) {
        Case c = {};
        ggml_init_params ip = {16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(16, true), nullptr, true};
        c.ctx = ggml_init(ip);
        ggml_tensor * w = ggml_new_tensor_2d(c.ctx, GGML_TYPE_BF16, K, M);
        ggml_tensor * x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32,  K, n);
        c.y_hp  = ggml_mul_mat(c.ctx, w, x);
        c.y_ref = ggml_mul_mat(c.ctx, w, x);
        c.g_hp  = ggml_new_graph_custom(c.ctx, 16, true);
        c.g_ref = ggml_new_graph_custom(c.ctx, 16, false);
        ggml_build_forward_expand(c.g_hp,  c.y_hp);
        ggml_build_forward_expand(c.g_ref, c.y_ref);
        c.buf = ggml_backend_alloc_ctx_tensors(c.ctx, be);
        if (!c.buf) { fprintf(stderr, "alloc failed at N=%d\n", n); return 2; }
        fill(w);
        fill(x);
        // no synchronize between launches: replaced buffers are still in flight
        if (ggml_backend_graph_compute_async(be, c.g_hp) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 2; }
        cases.push_back(c);
    }
    ggml_backend_synchronize(be);

    bool ok = true;
    for (size_t i = 0; i < cases.size(); ++i) {
        Case & c = cases[i];
        if (ggml_backend_graph_compute(be, c.g_ref) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 2; }
        std::vector<uint8_t> a(ggml_nbytes(c.y_hp)), b(ggml_nbytes(c.y_ref));
        ggml_backend_tensor_get(c.y_hp,  a.data(), 0, a.size());
        ggml_backend_tensor_get(c.y_ref, b.data(), 0, b.size());
        const bool eq = a == b;
        ok &= eq;
        printf("N=%6d %s\n", ns[i], eq ? "equal" : "DIFFER");
        ggml_backend_buffer_free(c.buf);
        ggml_free(c.ctx);
    }

    ggml_backend_free(be);

    // the release line ggml_metal_free logs: "<n> allocated, <n> freed, <n> still in flight"
    size_t n_alloc = 0, n_free = 0, n_live = 1;
    bool released = false;
    for (const std::string & l : g_log) {
        const char * p = strstr(l.c_str(), "scratch: ");
        if (p && sscanf(p, "scratch: %zu allocated, %zu freed, %zu still in flight", &n_alloc, &n_free, &n_live) == 3) {
            released = true;
        }
    }
    const bool life = released && n_alloc >= 3 && n_free == n_alloc && n_live == 0;
    printf("scratch: %zu allocated, %zu freed, %zu in flight at teardown -> %s\n", n_alloc, n_free, n_live, life ? "ok" : "BAD");
    printf("RESULT %s\n", ok && life ? "PASS" : "FAIL");
    return ok && life ? 0 : 1;
}
