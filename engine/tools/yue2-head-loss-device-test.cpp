// yue2-head-loss-device-test.cpp -- head CE/KL device mode
// (yue2_aitk_head_loss::compute with adapted_rows set, as the joint step runs
// it behind Yue2AitkArHeadRows) against the host mode it replaces
// (YUE2_AITK_HEAD_HOST=1). CPU backend, a random BF16 GGUF-style head with the
// full 184,704-row vocabulary and hidden 16, 300 positions: two full 128-row
// chunks and a 44-row tail, so the chunk offsets and the tail workspace are
// both exercised.
//
// Pass: CE and KL sums bit-identical, and the device F32 hidden gradient,
// rounded to BF16, equals the host mode's BF16 gradient bit for bit.

#include "train/yue2-aitk-head-loss.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int main() {
    using namespace yue2_aitk_head_loss;
    constexpr std::size_t H = 16, N = 300;
    ggml_backend_load_all();
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t backend = cpu_dev ? ggml_backend_dev_init(cpu_dev, nullptr) : nullptr;
    if (!backend) { std::printf("FAIL: no CPU backend\n"); return 1; }

    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 8;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    Yue2AitkConvRotLinear head;
    head.rows = int64_t(kVocab);
    head.cols = int64_t(H);
    head.parts.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, H, kVocab));
    head.dense_t = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, kVocab, H);
    ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
    ggml_tensor * base_rows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
    ggml_tensor * grad_rows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { std::printf("FAIL: buffer allocation\n"); return 1; }

    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<ggml_bf16_t> w(kVocab * H), wt(kVocab * H);
    for (std::size_t v = 0; v < kVocab; ++v)
        for (std::size_t j = 0; j < H; ++j) {
            w[v * H + j] = ggml_fp32_to_bf16(0.5f * normal(rng));
            wt[j * kVocab + v] = w[v * H + j];
        }
    ggml_backend_tensor_set(head.parts[0], w.data(), 0, ggml_nbytes(head.parts[0]));
    ggml_backend_tensor_set(head.dense_t, wt.data(), 0, ggml_nbytes(head.dense_t));
    auto bf16_valued = [&](std::vector<float> & x) { for (float & v : x) v = ggml_bf16_to_fp32(ggml_fp32_to_bf16(normal(rng))); };
    std::vector<float> adapted(N * H), base(N * H);
    bf16_valued(adapted);
    bf16_valued(base);
    for (std::size_t i = 0; i < N * H; ++i) base[i] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(adapted[i] + 0.1f * base[i]));
    std::vector<uint32_t> targets(N);
    for (auto & t : targets) t = uint32_t(rng() % kVocab);
    ggml_backend_tensor_set(rows, adapted.data(), 0, ggml_nbytes(rows));
    ggml_backend_tensor_set(base_rows, base.data(), 0, ggml_nbytes(base_rows));

    Request request;
    request.backend = backend;
    request.head = &head;
    request.targets = targets.data();
    request.positions = N;
    request.hidden = H;
    request.kl_weight = 0.4f;
    float host_ce = 0, host_kl = 0, device_ce = 0, device_kl = 0;
    std::vector<uint16_t> host_grad(N * H);
    Request host = request;
    host.adapted_hidden = adapted.data();
    host.base_hidden = base.data();
    host.adapted_hidden_grad_bf16 = host_grad.data();
    host.ce_sum = &host_ce;
    host.kl_sum = &host_kl;
    std::string error;
    if (compute(host, &error) != Status::success) { std::printf("FAIL: host mode: %s\n", error.c_str()); return 1; }
    Request device = request;
    device.adapted_rows = rows;
    device.base_rows = base_rows;
    device.adapted_grad_rows = grad_rows;
    device.ce_sum = &device_ce;
    device.kl_sum = &device_kl;
    if (compute(device, &error) != Status::success) { std::printf("FAIL: device mode: %s\n", error.c_str()); return 1; }

    std::vector<float> device_grad(N * H);
    ggml_backend_tensor_get(grad_rows, device_grad.data(), 0, ggml_nbytes(grad_rows));
    std::size_t mismatched = 0, nonzero = 0;
    for (std::size_t i = 0; i < N * H; ++i) {
        mismatched += f32_to_bf16(device_grad[i]) != host_grad[i];
        nonzero += host_grad[i] != 0;
    }
    const bool sums = std::memcmp(&host_ce, &device_ce, sizeof(float)) == 0 && std::memcmp(&host_kl, &device_kl, sizeof(float)) == 0;
    std::printf("ce host %.9g device %.9g | kl host %.9g device %.9g | gradient %zu of %zu differ (%zu nonzero)\n",
                host_ce, device_ce, host_kl, device_kl, mismatched, N * H, nonzero);
    const bool pass = sums && mismatched == 0 && nonzero > 0 && host_kl > 0;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return pass ? 0 : 1;
}
