#!/bin/bash
# HOT-Step CPP — macOS build script (Apple Silicon + Metal)
#
# Prerequisites:
#   - Xcode (full install, not just command line tools — needed for Metal)
#   - CMake 3.20+   (brew install cmake)
#   - Ninja          (brew install ninja)  — optional but faster
#   - PowerShell     (brew install powershell)  — runs verify-hooks.ps1
#
# Usage:
#   ./build-mac.sh          # Release build with Metal
#   ./build-mac.sh Debug    # Debug build
#
set -e

cd "$(dirname "$0")"

BUILD_TYPE="${1:-Release}"
BUILD_DIR="build"

echo "╔══════════════════════════════════════════╗"
echo "║       HOT-Step CPP — macOS Build         ║"
echo "║        Apple Silicon + Metal             ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "Build type: ${BUILD_TYPE}"

# Same hook check build.cmd runs: stop before CMake if a fork hook or a
# HOT-ggml change is missing. Fails closed when pwsh is not installed.
if ! command -v pwsh &>/dev/null; then
    echo "ERROR: pwsh not found; it runs the engine hook check (verify-hooks.ps1)." >&2
    echo "       Install it with: brew install powershell" >&2
    exit 1
fi
if ! pwsh -NoProfile -File verify-hooks.ps1; then
    echo "ERROR: hook check failed; fix the hooks above before building." >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Detect Ninja
GENERATOR=""
if command -v ninja &>/dev/null; then
    GENERATOR="-G Ninja"
    echo "Generator: Ninja"
else
    echo "Generator: Unix Makefiles (install ninja for faster builds: brew install ninja)"
fi

cmake .. \
    ${GENERATOR} \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_BACKEND_DL=OFF

# Build using all available cores
NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake --build . --config "${BUILD_TYPE}" -j "${NCPU}"

echo ""
echo "═══════════════════════════════════════════"
echo "  Build complete! Binaries in: $(pwd)/"
echo ""
echo "  Targets built:"
[ -f ace-server ] && echo "    ✓ ace-server"
[ -f mastering ] && echo "    ✓ mastering"
[ -f mp3-codec ] && echo "    ✓ mp3-codec"
[ -f vst-host ] && echo "    ✓ vst-host"
echo ""
echo "  Next: cd .. && ./launch.sh"
echo "═══════════════════════════════════════════"
