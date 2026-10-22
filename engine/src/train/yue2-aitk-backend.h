#pragma once
#include "ggml-backend.h"
#include <cstring>

// Backends are loadable GGML modules in portable builds. Identify one through
// the backend registry, without directly linking symbols from that module.
inline bool yue2_aitk_backend_is(ggml_backend_t backend, const char * registry_name) {
    if (!backend) return false;
    const auto device = ggml_backend_get_device(backend);
    if (!device) return false;
    const auto registry = ggml_backend_dev_backend_reg(device);
    const char * name = registry ? ggml_backend_reg_name(registry) : nullptr;
    return name && std::strcmp(name, registry_name) == 0;
}
inline bool yue2_aitk_is_cuda(ggml_backend_t backend) { return yue2_aitk_backend_is(backend, "CUDA"); }
// Metal only; CPU and Vulkan are neither of these.
inline bool yue2_aitk_is_metal(ggml_backend_t backend) { return yue2_aitk_backend_is(backend, "MTL"); }
