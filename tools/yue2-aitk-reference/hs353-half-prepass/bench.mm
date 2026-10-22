// HOTSTEP-353: half pre-pass microbenchmark for kernel_mul_mm_bf16_f32 (hp.metal appended to the
// fork's embedded mul_mm source). Times today's BF16 kernel, the two pre-pass kernels, the pre-passed
// GEMM and the F16-weight kernel (ceiling) on the base-forward shapes with the HOTSTEP-347 fallback
// mix, and checks the pre-passed GEMM against today's kernel bit for bit.
//   usage: bench <mul_mm+hp.metal> [iters]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml-metal-impl.h"

static uint16_t f2bf(float f) { // RNE, finite inputs only
    uint32_t u; memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}
static float bf2f(uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2h(float f) { __fp16 h = (__fp16) f; uint16_t r; memcpy(&r, &h, 2); return r; }

struct Shape { const char * name; int K, M, N; double wfail, afail; };
enum Mix { MIX_AUDIT, MIX_PASS, MIX_FAIL, MIX_SUBNORMAL };
static const char * mix_name[] = {"audit", "all-half", "all-bfloat", "subnormal"};

static id<MTLDevice> dev;
static id<MTLLibrary> lib;
static id<MTLCommandQueue> queue;

static id<MTLComputePipelineState> pipeline(const char * name, bool bc_out, bool rnd) {
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    bool bc_inp = false; int16_t one = 1;
    [cv setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:FC_MUL_MM + 0];
    [cv setConstantValue:&bc_out type:MTLDataTypeBool atIndex:FC_MUL_MM + 1];
    for (int i = 2; i <= 5; i++) [cv setConstantValue:&one type:MTLDataTypeShort atIndex:FC_MUL_MM + i];
    [cv setConstantValue:&rnd type:MTLDataTypeBool atIndex:FC_MUL_MM + 7];
    NSError * err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name] constantValues:cv error:&err];
    if (!fn) { fprintf(stderr, "function %s: %s\n", name, err.localizedDescription.UTF8String); exit(2); }
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) { fprintf(stderr, "pipeline %s: %s\n", name, err.localizedDescription.UTF8String); exit(2); }
    return p;
}

// Weights [M][K] BF16 and activations [N][K] F32. Normal-range values are exact in half after the
// BF16 round; a planted 1e-9 makes its tile (64 weight rows or 32 activation rows by 32 K) fall back.
// (Same generator as HOTSTEP-348.)
static void fill(const Shape & s, Mix mix, std::vector<uint16_t> & w, std::vector<float> & a, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    w.resize((size_t) s.M * s.K);
    a.resize((size_t) s.N * s.K);
    auto mag = [&](float lo, float hi) { return std::exp2(lo + (hi - lo) * u(rng)) * (u(rng) < 0.5f ? -1.f : 1.f); };
    if (mix == MIX_SUBNORMAL) {
        for (auto & v : w) v = f2bf(u(rng) < 0.5f ? (float)((int)(u(rng) * 255) + 1) * std::exp2(-24.f) : mag(-10, -4));
        for (auto & v : a) v = u(rng) < 0.5f ? (float)((int)(u(rng) * 255) + 1) * std::exp2(-24.f) : mag(-4, 2);
        return;
    }
    for (auto & v : w) v = f2bf(mag(-10, -4));
    for (auto & v : a) v = mag(-4, 2);
    const double wf = mix == MIX_AUDIT ? s.wfail : mix == MIX_FAIL ? 1.0 : 0.0;
    const double af = mix == MIX_AUDIT ? s.afail : mix == MIX_FAIL ? 1.0 : 0.0;
    for (int r0 = 0; r0 < s.M; r0 += 64)
        for (int k0 = 0; k0 < s.K; k0 += 32)
            if (u(rng) < wf) {
                const int r = std::min(s.M - 1, r0 + (int)(u(rng) * 64)), k = k0 + (int)(u(rng) * 32);
                w[(size_t) r * s.K + k] = f2bf(1e-9f);
            }
    for (int r1 = 0; r1 < s.N; r1 += 32)
        for (int k0 = 0; k0 < s.K; k0 += 32)
            if (u(rng) < af) {
                const int r = std::min(s.N - 1, r1 + (int)(u(rng) * 32)), k = k0 + (int)(u(rng) * 32);
                a[(size_t) r * s.K + k] = 1e-9f;
            }
}

struct Bufs { id<MTLBuffer> w, wh16, a, wh, ah, ab, wm, am, d; };

enum Arm { ARM_BF16, ARM_HP, ARM_CVT_W, ARM_CVT_A, ARM_F16 };

// encode one arm `iters` times in one command buffer; returns per-dispatch GPU ms
static double run(Arm arm, id<MTLComputePipelineState> p, const Shape & s, const Bufs & b, int iters) {
    ggml_metal_kargs_mul_mm args = {};
    args.ne00 = s.K; args.ne02 = 1;
    args.nb01 = (uint64_t) s.K * 2; args.nb02 = args.nb01 * s.M; args.nb03 = args.nb02;
    args.ne12 = 1;
    args.nb10 = 4; args.nb11 = (uint64_t) s.K * 4; args.nb12 = args.nb11 * s.N; args.nb13 = args.nb12;
    args.ne0 = s.M; args.ne1 = s.N; args.r2 = 1; args.r3 = 1;
    const int KM[2] = {s.K, s.M}, KN[2] = {s.K, s.N};
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    switch (arm) {
        case ARM_BF16: case ARM_F16: case ARM_HP:
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:(arm == ARM_F16 ? b.wh16 : b.w) offset:0 atIndex:1];
            [enc setBuffer:b.a offset:0 atIndex:2];
            [enc setBuffer:b.d offset:0 atIndex:3];
            if (arm == ARM_HP) {
                [enc setBuffer:b.wh offset:0 atIndex:4];
                [enc setBuffer:b.ah offset:0 atIndex:5];
                [enc setBuffer:b.wm offset:0 atIndex:6];
                [enc setBuffer:b.am offset:0 atIndex:7];
                [enc setBuffer:b.ab offset:0 atIndex:8];
            }
            [enc setThreadgroupMemoryLength:8192 atIndex:0];
            for (int i = 0; i < iters; i++)
                [enc dispatchThreadgroups:MTLSizeMake((s.N + 31) / 32, (s.M + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            break;
        case ARM_CVT_W:
            [enc setBuffer:b.w offset:0 atIndex:0]; [enc setBuffer:b.wh offset:0 atIndex:1];
            [enc setBuffer:b.wm offset:0 atIndex:2]; [enc setBytes:KM length:8 atIndex:3];
            for (int i = 0; i < iters; i++)
                [enc dispatchThreadgroups:MTLSizeMake(s.K / 32, (s.M + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            break;
        case ARM_CVT_A:
            [enc setBuffer:b.a offset:0 atIndex:0]; [enc setBuffer:b.ah offset:0 atIndex:1];
            [enc setBuffer:b.am offset:0 atIndex:2]; [enc setBytes:KN length:8 atIndex:3];
            [enc setBuffer:b.ab offset:0 atIndex:4];
            for (int i = 0; i < iters; i++)
                [enc dispatchThreadgroups:MTLSizeMake(s.K / 32, (s.N + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            break;
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { fprintf(stderr, "command buffer: %s\n", cb.error.localizedDescription.UTF8String); exit(2); }
    return (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / iters;
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: bench <mul_mm+hp.metal> [iters]\n"); return 2; }
        const int iters = argc > 2 ? atoi(argv[2]) : 10;
        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newCommandQueue];
        NSError * err = nil;
        NSString * src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "read: %s\n", err.localizedDescription.UTF8String); return 2; }
        MTLCompileOptions * opt = [MTLCompileOptions new]; // ggml-metal's defaults
        NSMutableDictionary * macros = [@{@"GGML_METAL_HAS_BF16": @"1", @"GGML_METAL_EMBED_LIBRARY": @"1"} mutableCopy];
        for (const char * k : {"HP_FORCE", "HP_VARIANT"}) // bench-only variants
            if (const char * v = getenv(k)) macros[[NSString stringWithUTF8String:k]] = [NSString stringWithUTF8String:v];
        opt.preprocessorMacros = macros;
        lib = [dev newLibraryWithSource:src options:opt error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 2; }
        printf("device %s, iters %d\n", dev.name.UTF8String, iters);

        // base-forward families from the HOTSTEP-347 audit (AR length 7668): fallback rates per
        // weight tile and per activation tile, FLOP weights of the families in base forward
        const Shape shapes[] = {
            {"gate_up", 2048, 12288, 7668, 0.2516, 0.00056},
            {"qkv",     2048,  4096, 7668, 0.2656, 0.0079},
            {"down",    6144,  2048, 7668, 0.2519, 0.2976},
            {"o",       2048,  2048, 7668, 0.2560, 0.0832},
        };
        const double flop_w[] = {31.8, 10.6, 15.9, 5.3};

        auto p_cw = pipeline("kernel_hp_cvt_w", false, false);
        auto p_ca = pipeline("kernel_hp_cvt_a", false, false);

        bool all_ok = true;
        double w_base = 0, w_hp = 0, w_tot = 0, w_totc = 0, w_f16 = 0;
        printf("| shape | mix | bf16 | hp gemm | cvt w | cvt a | hp+cvt a | hp+both | f16 | (hp+cvt a)/bf16 | (hp+both)/bf16 | half steps | half 8-blocks | bits |\n"
               "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
        for (int si = 0; si < (int)(sizeof(shapes) / sizeof(shapes[0])); si++) {
            const Shape & s = shapes[si];
            const bool bc_out = s.M % 64 != 0 || s.N % 32 != 0;
            auto p_base   = pipeline("kernel_mul_mm_bf16_f32", bc_out, false);
            auto p_base_r = pipeline("kernel_mul_mm_bf16_f32", bc_out, true);
            auto p_hp     = pipeline("kernel_mul_mm_hp",       bc_out, false);
            auto p_hp_r   = pipeline("kernel_mul_mm_hp",       bc_out, true);
            auto p_f16    = pipeline("kernel_mul_mm_f16_f32",  bc_out, false);
            for (int mi = 0; mi < 4; mi++) {
                const Mix mix = (Mix) mi;
                std::vector<uint16_t> w; std::vector<float> a;
                fill(s, mix, w, a, 42 + si * 10 + mi);
                std::vector<uint16_t> wh(w.size());
                for (size_t i = 0; i < w.size(); i++) wh[i] = f2h(bf2f(w[i]));
                const int nks = s.K / 32;
                const size_t dsz = (size_t) s.M * s.N * 4;
                Bufs b;
                b.w    = [dev newBufferWithBytes:w.data()  length:w.size() * 2 options:MTLResourceStorageModeShared];
                b.wh16 = [dev newBufferWithBytes:wh.data() length:wh.size() * 2 options:MTLResourceStorageModeShared];
                b.a    = [dev newBufferWithBytes:a.data()  length:a.size() * 4 options:MTLResourceStorageModeShared];
                b.wh   = [dev newBufferWithLength:w.size() * 2 options:MTLResourceStorageModeShared];
                b.ah   = [dev newBufferWithLength:a.size() * 2 options:MTLResourceStorageModeShared];
                b.ab   = [dev newBufferWithLength:a.size() * 2 options:MTLResourceStorageModeShared];
                b.wm   = [dev newBufferWithLength:(size_t)((s.M + 63) / 64) * nks * 4 options:MTLResourceStorageModeShared];
                b.am   = [dev newBufferWithLength:(size_t)((s.N + 31) / 32) * nks * 4 options:MTLResourceStorageModeShared];
                b.d    = [dev newBufferWithLength:dsz options:MTLResourceStorageModeShared];

                // pre-pass, then parity against today's kernel (plain and BF16_ROUND)
                run(ARM_CVT_W, p_cw, s, b, 1);
                run(ARM_CVT_A, p_ca, s, b, 1);
                // half share of 32-wide steps (variants 1-4) and of 8-wide blocks (variant 5)
                long pass = 0, pass8 = 0, tot = 0;
                {
                    const uint8_t * wm = (const uint8_t *) b.wm.contents, * am = (const uint8_t *) b.am.contents;
                    for (int ty = 0; ty < (s.M + 63) / 64; ty++)
                        for (int tx = 0; tx < (s.N + 31) / 32; tx++)
                            for (int k = 0; k < nks; k++) {
                                int all = 1;
                                for (int j = 0; j < 4; j++) {
                                    const int f = wm[ty * nks * 4 + 4 * k + j] & am[tx * nks * 4 + 4 * k + j];
                                    pass8 += f; all &= f;
                                }
                                pass += all; tot++;
                            }
                }
                std::vector<uint8_t> ref(dsz);
                bool eq = true;
                for (int r = 0; r < 2; r++) {
                    memset(b.d.contents, 0xff, dsz);
                    run(ARM_BF16, r ? p_base_r : p_base, s, b, 1);
                    memcpy(ref.data(), b.d.contents, dsz);
                    memset(b.d.contents, 0x7f, dsz);
                    run(ARM_HP, r ? p_hp_r : p_hp, s, b, 1);
                    eq &= memcmp(ref.data(), b.d.contents, dsz) == 0;
                }
                all_ok &= eq;

                double t[5] = {};
                if (mix != MIX_SUBNORMAL) {
                    run(ARM_BF16, p_base, s, b, 3); // warm
                    // interleave so drift and thermals hit every arm alike
                    std::vector<double> v[5];
                    for (int r = 0; r < 5; r++) {
                        v[0].push_back(run(ARM_BF16,  p_base, s, b, iters));
                        v[1].push_back(run(ARM_HP,    p_hp,   s, b, iters));
                        v[2].push_back(run(ARM_CVT_W, p_cw,   s, b, iters));
                        v[3].push_back(run(ARM_CVT_A, p_ca,   s, b, iters));
                        v[4].push_back(run(ARM_F16,   p_f16,  s, b, iters));
                    }
                    for (int k = 0; k < 5; k++) { std::sort(v[k].begin(), v[k].end()); t[k] = v[k][2]; }
                    if (mix == MIX_AUDIT) {
                        w_base += flop_w[si]; w_hp += flop_w[si] * t[1] / t[0];
                        w_tot  += flop_w[si] * (t[1] + t[3]) / t[0];
                        w_totc += flop_w[si] * (t[1] + t[2] + t[3]) / t[0];
                        w_f16  += flop_w[si] * t[4] / t[0];
                    }
                }
                printf("| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.4f | %.4f | %.1f%% | %.1f%% | %s |\n",
                       s.name, mix_name[mi], t[0], t[1], t[2], t[3], t[1] + t[3], t[1] + t[2] + t[3], t[4],
                       t[0] > 0 ? (t[1] + t[3]) / t[0] : 0.0, t[0] > 0 ? (t[1] + t[2] + t[3]) / t[0] : 0.0,
                       100.0 * pass / tot, 100.0 * pass8 / (4.0 * tot), eq ? "equal" : "DIFFER");
                fflush(stdout);
            }
        }
        printf("\nFLOP-weighted base-forward ratio (audit mix): hp gemm %.4f, hp+cvt a (weights cached) %.4f, "
               "hp+both (no cache) %.4f, f16 ceiling %.4f\n", w_hp / w_base, w_tot / w_base, w_totc / w_base, w_f16 / w_base);
        printf("BITS %s\n", all_ok ? "PASS" : "FAIL");
        return all_ok ? 0 : 1;
    }
}
