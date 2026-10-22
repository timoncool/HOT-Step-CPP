// HOTSTEP-347: how this GPU converts to half and bfloat, and whether a half
// simdgroup product matches the bfloat one bit for bit on values both types
// hold exactly. Each test runs twice: with the default MTLCompileOptions (what
// ggml-metal uses for its embedded source) and with fast math off.
//   swiftc -O metal-half-probe.swift -o metal-half-probe && ./metal-half-probe
import Foundation
import Metal

let source = """
#include <metal_stdlib>
using namespace metal;

kernel void cvt_f32(device const float * in [[buffer(0)]], device ushort * hb [[buffer(1)]],
                    device ushort * bb [[buffer(2)]], constant uint & n [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const float x = in[i];
    hb[i] = as_type<ushort>((half) x);
    bb[i] = as_type<ushort>((bfloat) x);
}

// BF16 in device memory -> half, the way a half-staged tile would load it.
kernel void cvt_bf16(device const bfloat * in [[buffer(0)]], device ushort * hb [[buffer(1)]],
                     device float * back [[buffer(2)]], constant uint & n [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const half h = (half)(float) in[i];
    hb[i] = as_type<ushort>(h);
    back[i] = (float) h;
}

// One simdgroup per 8x8 output over K, staged through threadgroup memory as
// kernel_mul_mm does. mode 0: half tiles, 1: bfloat tiles, 2: alternate per
// 8-wide K block (half on even blocks, bfloat on odd).
kernel void mac(device const float * A [[buffer(0)]], device const float * B [[buffer(1)]],
                device float * C [[buffer(2)]], constant uint & K [[buffer(3)]], constant uint & mode [[buffer(4)]],
                uint tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half   sah[64], sbh[64];
    threadgroup bfloat sab[64], sbb[64];
    threadgroup float  sc[64];
    simdgroup_float8x8 mc = make_filled_simdgroup_matrix<float, 8>(0.f);
    device const float * a = A + (ulong) tg * 8 * K;
    device const float * b = B + (ulong) tg * 8 * K;
    for (uint k0 = 0; k0 < K; k0 += 8) {
        const bool use_half = mode == 0 || (mode == 2 && ((k0 / 8) % 2) == 0);
        for (ushort e = tid; e < 64; e += 32) {
            const ushort r = e / 8, c = e % 8;
            const float x = a[r * K + k0 + c], y = b[(k0 + r) * 8 + c];
            if (use_half) { sah[e] = (half) x; sbh[e] = (half) y; }
            else          { sab[e] = (bfloat) x; sbb[e] = (bfloat) y; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (use_half) {
            simdgroup_half8x8 ma, mb;
            simdgroup_load(ma, sah, 8); simdgroup_load(mb, sbh, 8);
            simdgroup_multiply_accumulate(mc, ma, mb, mc);
        } else {
            simdgroup_bfloat8x8 ma, mb;
            simdgroup_load(ma, sab, 8); simdgroup_load(mb, sbb, 8);
            simdgroup_multiply_accumulate(mc, ma, mb, mc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(mc, sc, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort e = tid; e < 64; e += 32) C[(ulong) tg * 64 + e] = sc[e];
}
"""

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
let queue = device.makeCommandQueue()!
print("device: \(device.name), apple8: \(device.supportsFamily(.apple8)), apple9: \(device.supportsFamily(.apple9))")

func buf<T>(_ a: [T]) -> MTLBuffer {
    a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 16), options: .storageModeShared)! }
}
func zeros<T>(_ n: Int, _: T.Type) -> MTLBuffer {
    device.makeBuffer(length: max(n * MemoryLayout<T>.stride, 16), options: .storageModeShared)!
}
func read<T>(_ b: MTLBuffer, _ n: Int, _: T.Type) -> [T] {
    Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: T.self, capacity: n), count: n))
}
func run(_ lib: MTLLibrary, _ name: String, _ bufs: [MTLBuffer], _ consts: [UInt32], groups: Int, threads: Int) {
    let pipeline = try! device.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    for (i, b) in bufs.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
    for (i, v) in consts.enumerated() { var x = v; enc.setBytes(&x, length: 4, index: bufs.count + i) }
    enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let e = cb.error { fatalError("\(name): \(e)") }
}

struct Rng { // splitmix64
    var s: UInt64
    mutating func next() -> UInt64 { s &+= 0x9E3779B97F4A7C15; var z = s; z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9; z = (z ^ (z >> 27)) &* 0x94D049BB133111EB; return z ^ (z >> 31) }
    mutating func int(_ lo: Int, _ hi: Int) -> Int { lo + Int(next() % UInt64(hi - lo + 1)) }
}
func rneBF16(_ x: Float) -> UInt16 {
    let u = x.bitPattern
    if (u & 0x7fffffff) > 0x7f800000 { return UInt16(u >> 16) | 0x40 }
    return UInt16((u &+ 0x7fff &+ ((u >> 16) & 1)) >> 16)
}
func isNaNHalf(_ h: UInt16) -> Bool { (h & 0x7c00) == 0x7c00 && (h & 0x03ff) != 0 }

func probe(_ label: String, _ options: MTLCompileOptions) {
    let lib = try! device.makeLibrary(source: source, options: options)
    print("== \(label)")

    // 1. Every BF16 bit pattern -> half, against host RNE (Float16, FZ16 off).
    let all = (0..<65536).map { UInt16($0) }
    let n = all.count
    let hb = zeros(n, UInt16.self), back = zeros(n, Float.self)
    run(lib, "cvt_bf16", [buf(all), hb, back], [UInt32(n)], groups: (n + 255) / 256, threads: 256)
    let gpu = read(hb, n, UInt16.self)
    var mismatch = 0, flushed = 0, subKept = 0, subTotal = 0
    for i in 0..<n {
        let x = Float(bitPattern: UInt32(all[i]) << 16)
        let host = Float16(x).bitPattern
        if isNaNHalf(host) && isNaNHalf(gpu[i]) { continue }
        if host != gpu[i] { mismatch += 1; if (gpu[i] & 0x7fff) == 0 && (host & 0x7fff) != 0 { flushed += 1 } }
        if (host & 0x7c00) == 0 && (host & 0x03ff) != 0 { subTotal += 1; if gpu[i] == host { subKept += 1 } }
    }
    print("bf16->half, all 65536 patterns: mismatches vs host RNE \(mismatch), of which flushed to zero \(flushed); half-subnormal results kept \(subKept)/\(subTotal)")

    // 2. F32 -> half and F32 -> bfloat on random floats, ties and the half subnormal range.
    var rng = Rng(s: 347)
    var xs: [Float] = []
    for _ in 0..<(1 << 20) { xs.append(Float(bitPattern: UInt32(truncatingIfNeeded: rng.next()))) }
    for _ in 0..<(1 << 18) { xs.append(Float(bitPattern: (UInt32(truncatingIfNeeded: rng.next()) & 0xffff0000) | 0x8000)) } // bf16 ties
    for _ in 0..<(1 << 18) { xs.append(Float(sign: rng.int(0, 1) == 0 ? .plus : .minus, exponent: rng.int(-27, -14), significand: 1 + Float(rng.int(0, (1 << 23) - 1)) / Float(1 << 23))) }
    let m = xs.count
    let h2 = zeros(m, UInt16.self), b2 = zeros(m, UInt16.self)
    run(lib, "cvt_f32", [buf(xs), h2, b2], [UInt32(m)], groups: (m + 255) / 256, threads: 256)
    let gh = read(h2, m, UInt16.self), gb = read(b2, m, UInt16.self)
    var hMis = 0, hFlush = 0, bMis = 0, bTieMis = 0
    for i in 0..<m {
        let x = xs[i]
        let host = Float16(x).bitPattern
        if !(isNaNHalf(host) && isNaNHalf(gh[i])) && host != gh[i] { hMis += 1; if (gh[i] & 0x7fff) == 0 { hFlush += 1 } }
        if x.isNaN { continue }
        if rneBF16(x) != gb[i] { bMis += 1; if (x.bitPattern & 0xffff) == 0x8000 { bTieMis += 1 } }
    }
    print("f32->half, \(m) values: mismatches vs host RNE \(hMis), flushed \(hFlush)")
    print("f32->bfloat, \(m) values: mismatches vs host RNE \(bMis), of which exact ties \(bTieMis)")

    // 3. half vs bfloat vs per-block mixed simdgroup products, values exact in both types.
    func value(_ kind: Int) -> Float {
        let sign: Float = rng.int(0, 1) == 0 ? 1 : -1
        if (kind == 2 && rng.int(0, 3) == 0) || kind == 3 { return sign * Float(rng.int(1, 255)) * 0x1p-24 } // exact half subnormal, exact bf16
        let e = kind == 0 ? rng.int(-4, 4) : rng.int(-14, 15)
        return sign * (1 + Float(rng.int(0, 127)) / 128) * Float(sign: .plus, exponent: e, significand: 1)
    }
    // kind 3: A is all half subnormals against narrow B, so a flush would zero the outputs.
    for (kind, name) in [(0, "narrow 2^-4..2^4"), (1, "wide 2^-14..2^15"), (2, "with half subnormals"), (3, "subnormal A x narrow B")] {
        for K in [256, 2048] {
            let groups = K == 256 ? 4096 : 512
            let A = (0..<(groups * 8 * K)).map { _ in value(kind) }, B = (0..<(groups * 8 * K)).map { _ in value(kind) }
            let ba = buf(A), bb = buf(B)
            var out: [[Float]] = []
            for mode in 0..<3 {
                let c = zeros(groups * 64, Float.self)
                run(lib, "mac", [ba, bb, c], [UInt32(K), UInt32(mode)], groups: groups, threads: 32)
                out.append(read(c, groups * 64, Float.self))
            }
            var hb = 0, mb = 0, maxRel: Double = 0, zeros = 0
            for i in 0..<(groups * 64) {
                if out[0][i].bitPattern != out[1][i].bitPattern {
                    hb += 1
                    let d = Double(abs(out[0][i] - out[1][i])) / max(Double(abs(out[1][i])), 1e-30)
                    maxRel = max(maxRel, d)
                }
                if out[2][i].bitPattern != out[1][i].bitPattern { mb += 1 }
                if out[0][i] == 0 { zeros += 1 }
            }
            print("mac \(name), K \(K), \(groups * 64) outputs: half!=bfloat \(hb) (max rel \(maxRel)), mixed!=bfloat \(mb), zero outputs (half) \(zeros)")
        }
    }
}

probe("default compile options (ggml-metal)", MTLCompileOptions())
let safe = MTLCompileOptions()
safe.fastMathEnabled = false
probe("fast math off", safe)
