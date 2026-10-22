// One fresh training run through the unchanged runtime, compiled with
// range-audit.h force-included; writes the audit JSON at the end.
#include "range-audit.h"
#include "train/yue2-aitk-runtime.h"
#include <unistd.h>

int main(int argc, char ** argv) {
    using namespace yue2_aitk_runtime;
    Config config;
    std::string error;
    if (!hs347::out_path() || argc < 2 || parse(argc - 1, argv + 1, &config, &error) != ParseResult::ok ||
        config.device_name != "MTL0" || !config.resume.empty() || !config.fixture_replay.empty()) {
        std::fprintf(stderr, "Range audit requires HS347_AUDIT_OUT and fresh MTL0 training: %s\n", error.c_str());
        return 2;
    }
    int rc = yue2_aitk_run(config, &error);
    if (rc) std::fprintf(stderr, "%s\n", error.c_str());
    hs347::dump();
    std::fflush(nullptr);
    _exit(rc);  // Match ace-train's macOS static-destructor handling.
}
