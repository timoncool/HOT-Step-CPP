// backend-health-test: logic test for the #242 poisoned-backend handling in
// backend.h (backend_healthy, backend_fault_point, backend_compute_ok, and the
// teardown-and-reinit recovery). It mirrors the YuE2 flow: an "align" compute,
// then an AR "prefill" compute on the same shared backend.
//
// Usage: backend-health-test [--no-probe] [--other-holder]
//   --no-probe      behave like the code before the fix (no health probes)
//   --other-holder  a non-YuE2 model keeps a backend ref, so recovery must fail
// Set ACE_FAULT_BACKEND_ERROR=align to inject the fault. Exit 0 when the flow
// ends the way the fixed code should (align fails loudly, then prefill runs on
// a fresh backend; or a clear error when another model holds the backend).

#include <string>

#include "backend.h"

#include <cstdio>
#include <cstring>
#include <vector>

// One small compute on the shared backend: y = x * k. Returns the status
// folded with the poison flag, the way the YuE2 compute sites check it.
static bool run_graph(ggml_backend_t be, float k, std::vector<float> * out) {
    ggml_init_params ip  = { ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(8, false), nullptr, true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_tensor *    x   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    ggml_tensor *    y   = ggml_scale(ctx, x, k);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::vector<float>    in(16);
    for (int i = 0; i < 16; i++) {
        in[i] = (float) i;
    }
    ggml_backend_tensor_set(x, in.data(), 0, in.size() * sizeof(float));
    bool ok = backend_compute_ok(ggml_backend_graph_compute(be, gf));
    out->assign(16, 0.0f);
    ggml_backend_tensor_get(y, out->data(), 0, 16 * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    bool probe = true, other = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--no-probe")) probe = false;
        if (!strcmp(argv[i], "--other-holder")) other = true;
    }
    BackendPair ext = {};
    if (other) {
        ext = backend_init("OtherModel");
    }
    BackendPair ar = backend_init("YuE2-AR");  // resident AR
    BackendPair mm = backend_init("MMSFA");    // align model, shared backend

    // Align: compute succeeds, then the fault point fires (async Metal error).
    std::vector<float> em;
    bool align_ok = run_graph(mm.backend, 3.0f, &em);
    backend_fault_point("align");
    if (align_ok && probe && !backend_healthy()) {
        align_ok = false;
        printf("ALIGN: 500 MMS_FA compute failed on the GPU backend (device error)\n");
    } else if (align_ok) {
        printf("ALIGN: success, em[5]=%.1f\n", em[5]);
    } else {
        printf("ALIGN: 500 MMS_FA graph compute failed\n");
    }
    backend_release(mm.backend, mm.cpu_backend);

    // Next job: probe before the prefill, recover through the same teardown
    // helper the YuE2 job path uses (backend_teardown_recover).
    if (probe && !backend_healthy()) {
        printf("JOB: backend unhealthy before prefill, unloading YuE2\n");
        BackendRecover rc = backend_teardown_recover([&] { backend_release(ar.backend, ar.cpu_backend); },
                                                     [&] {
                                                         ar = backend_init("YuE2-AR");
                                                         return ar.backend != nullptr;
                                                     });
        if (rc == BackendRecover::OTHER_HOLDER) {
            printf("JOB: FAILED: GPU backend in error state, unload models to recover (refs=%d)\n", g_backend_refs);
            backend_release(ext.backend, ext.cpu_backend);
            bool pass = other && g_backend_refs == 0;
            printf("RESULT: %s (expected other-holder failure, refs=%d)\n", pass ? "PASS" : "FAIL", g_backend_refs);
            return pass ? 0 : 1;
        }
        const char * rc_name = rc == BackendRecover::OK             ? "OK" :
                               rc == BackendRecover::OTHER_HOLDER   ? "OTHER_HOLDER" :
                                                                      "RELOAD_FAILED";
        printf("JOB: recover rc=%s, healthy=%d, poisoned=%d\n", rc_name, (int) backend_healthy(),
               (int) g_backend_poisoned);
        if (rc != BackendRecover::OK) {
            return 1;
        }
    }
    std::vector<float> pf;
    bool               pre_ok = run_graph(ar.backend, 0.5f, &pf);
    printf("PREFILL: %s, out[5]=%.2f\n", pre_ok ? "ok" : "YuE2 prefill graph compute failed", pf[5]);
    backend_release(ar.backend, ar.cpu_backend);
    if (other) backend_release(ext.backend, ext.cpu_backend);
    printf("RELEASE: refs=%d\n", g_backend_refs);

    // State asserts: align output must be right when it reports success, the
    // prefill must compute 5 * 0.5 = 2.5 on a healthy backend, nothing leaks.
    const bool faulted = getenv("ACE_FAULT_BACKEND_ERROR") != nullptr;
    bool       pass    = pre_ok && pf[5] == 2.5f && g_backend_refs == 0 && !g_backend_poisoned;
    if (align_ok && em[5] != 15.0f) pass = false;
    if (faulted && align_ok) pass = false;     // the fault must surface at align
    if (!faulted && !align_ok) pass = false;   // and not without it
    if (other && faulted) pass = false;        // should have failed with the clear message
    printf("RESULT: %s (align_ok=%d em[5]=%.2f pre_ok=%d out[5]=%.2f refs=%d poisoned=%d)\n", pass ? "PASS" : "FAIL",
           (int) align_ok, em[5], (int) pre_ok, pf[5], g_backend_refs, (int) g_backend_poisoned);
    return pass ? 0 : 1;
}
