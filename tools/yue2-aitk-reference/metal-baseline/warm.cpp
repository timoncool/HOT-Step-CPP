// Run the unchanged trainer twice in one process. Pipeline and frozen-teacher
// caches survive; each call constructs its own model, adapter, optimizer and RNG.
#include "train/yue2-aitk-runtime.h"
#include <unistd.h>

int main(int argc, char ** argv) {
    using namespace yue2_aitk_runtime;
    Config config;
    std::string error;
    if (argc < 2 || parse(argc - 1, argv + 1, &config, &error) != ParseResult::ok ||
        config.device_name != "MTL0" || !config.resume.empty() ||
        !config.fixture_replay.empty()) {
        std::fprintf(stderr, "Warm probe requires fresh MTL0 training: %s\n", error.c_str());
        return 2;
    }
    Config warmup = config;
    warmup.steps = 1;
    warmup.output += "-warmup";
    std::fprintf(stderr, "HS342_PHASE warmup\n");
    int rc = yue2_aitk_run(warmup, &error);
    if (rc == 0) {
        std::fprintf(stderr, "HS342_PHASE measured_fresh_state\n");
        rc = yue2_aitk_run(config, &error);
    }
    if (rc) std::fprintf(stderr, "%s\n", error.c_str());
    std::fflush(nullptr);
    _exit(rc); // Match ace-train's macOS static-destructor handling.
}
