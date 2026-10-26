#!/bin/bash
# HOT-Step CPP — macOS Release Packager
#
# Builds a portable .tar.gz release that requires ZERO prerequisites.
# End users just extract and run ./HOT-Step.sh — no Xcode, no brew, no npm.
#
# What gets bundled:
#   - Pre-built C++ engine binaries (Metal GPU)
#   - Bundled Node.js 22 runtime (~40 MB)
#   - Pre-installed server dependencies (production only)
#   - Pre-built React UI
#   - Launch scripts
#
# Usage:
#   ./package-release.sh                # Build release package
#   ./package-release.sh --skip-build   # Package only (engine already built)
#
# Output:
#   HOT-Step-CPP-v{VERSION}-macOS-arm64.tar.gz

set -e

cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"

# ── Config ───────────────────────────────────────────────────────────
VERSION="${VERSION:-1.0.0}"
ARCH="arm64"
NODE_VERSION="22.22.2"
NODE_URL="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-darwin-${ARCH}.tar.xz"
RELEASE_NAME="HOT-Step-CPP-v${VERSION}-macOS-${ARCH}"
SKIP_BUILD=0

for arg in "$@"; do
    case $arg in
        --skip-build) SKIP_BUILD=1 ;;
        --version=*) VERSION="${arg#*=}"; RELEASE_NAME="HOT-Step-CPP-v${VERSION}-macOS-${ARCH}" ;;
    esac
done

STAGING_DIR="${ROOT_DIR}/release-staging/${RELEASE_NAME}"
OUTPUT_FILE="${ROOT_DIR}/${RELEASE_NAME}.tar.gz"

# ── Colors ───────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

step() { echo -e "\n${CYAN}${BOLD}[$1/$TOTAL_STEPS]${NC} ${BOLD}$2${NC}"; }
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗ $1${NC}"; exit 1; }

TOTAL_STEPS=7

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   HOT-Step CPP — Release Packager        ║${NC}"
echo -e "${BOLD}║      macOS ${ARCH} Portable Build       ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "  Version:  ${VERSION}"
echo "  Output:   ${RELEASE_NAME}.tar.gz"
echo ""

# ── Find compatible Node for the build process ──────────────────────
find_node() {
    for bp in "/opt/homebrew/opt/node@24/bin" "/usr/local/opt/node@24/bin"; do
        if [ -x "${bp}/node" ]; then echo "$bp"; return 0; fi
    done
    if command -v node &>/dev/null; then
        local ver; ver="$(node --version 2>/dev/null | tr -d 'v')"
        local major="${ver%%.*}"
        if [ "$major" -ge 20 ] && [ "$major" -lt 25 ] 2>/dev/null; then
            echo "$(dirname "$(command -v node)")"; return 0
        fi
    fi
    return 1
}

NODE_DIR=""
if NODE_DIR=$(find_node); then
    export PATH="${NODE_DIR}:${PATH}"
else
    fail "Node.js 20-24 required for building. Run: brew install node@24"
fi

# ── Step 1: Build engine ─────────────────────────────────────────────
step 1 "Building C++ engine..."

if [ "$SKIP_BUILD" -eq 1 ] && [ -f "engine/build/ace-server" ]; then
    ok "Skipping build (--skip-build, binaries exist)"
else
    cd engine
    rm -rf build && mkdir build && cd build
    cmake .. \
        -DGGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DGGML_BACKEND_DL=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        2>&1 | tail -3
    NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cmake --build . --config Release -j "${NCPU}" 2>&1 | grep -E '^\[100|Built target|error:'
    cd "${ROOT_DIR}"
fi

# Verify binaries
for bin in ace-server mastering mp3-codec vst-host; do
    [ -f "engine/build/${bin}" ] || fail "engine/build/${bin} not found"
done
ok "All engine binaries built"

# Fix rpaths: replace any absolute build-dir rpaths with @executable_path
# so dylibs are found relative to the binary (portable release requirement).
echo "  Fixing dylib rpaths for portability..."
for bin in engine/build/ace-server engine/build/ace-train engine/build/ace-midi engine/build/ace-caption \
           engine/build/ace-lm engine/build/ace-synth engine/build/ace-understand \
           engine/build/mastering engine/build/mp3-codec engine/build/neural-codec \
           engine/build/vst-host engine/build/quantize; do
    if [ -f "$bin" ]; then
        # Remove all existing rpaths (may be absolute build paths)
        for old_rpath in $(otool -l "$bin" 2>/dev/null | grep -A1 LC_RPATH | grep "path " | awk '{print $2}'); do
            install_name_tool -delete_rpath "$old_rpath" "$bin" 2>/dev/null || true
        done
        # Add the portable @executable_path rpath
        install_name_tool -add_rpath "@executable_path" "$bin" 2>/dev/null || true
    fi
done
ok "Rpaths fixed (@executable_path)"

# ── Step 2: Install server deps (production) ─────────────────────────
step 2 "Installing server dependencies (production)..."

cd server
rm -rf node_modules
npm install --omit=dev --loglevel=warn 2>&1 | tail -3
# Rebuild native modules
npm rebuild better-sqlite3 2>&1 | tail -1
# Bundle the server with the same entry point as CI.
(cd ../release && npm ci && node esbuild.config.mjs)
ok "Server dependencies installed and server bundled"
cd "${ROOT_DIR}"

# ── Step 3: Build UI ─────────────────────────────────────────────────
step 3 "Building production UI..."

cd ui
[ -d "node_modules" ] || npm install --loglevel=warn
npx vite build 2>&1 | tail -3
ok "UI built"
cd "${ROOT_DIR}"

# ── Step 4: Download Node.js runtime ─────────────────────────────────
step 4 "Downloading Node.js ${NODE_VERSION} runtime..."

NODE_CACHE="${ROOT_DIR}/release-staging/node-cache"
NODE_ARCHIVE="${NODE_CACHE}/node-v${NODE_VERSION}-darwin-${ARCH}.tar.xz"
NODE_EXTRACTED="${NODE_CACHE}/node-v${NODE_VERSION}-darwin-${ARCH}"

mkdir -p "${NODE_CACHE}"

if [ -f "${NODE_EXTRACTED}/bin/node" ]; then
    ok "Using cached Node.js download"
else
    echo "  Downloading from ${NODE_URL}..."
    curl -L --progress-bar -o "${NODE_ARCHIVE}" "${NODE_URL}"
    cd "${NODE_CACHE}"
    tar xf "$(basename "${NODE_ARCHIVE}")"
    cd "${ROOT_DIR}"
    ok "Node.js ${NODE_VERSION} downloaded"
fi

# ── Step 5: Assemble release ─────────────────────────────────────────
step 5 "Assembling release package..."

rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

# Engine binaries + dylibs
mkdir -p "${STAGING_DIR}/engine"
for bin in ace-server ace-train ace-midi ace-caption ace-lm ace-synth ace-understand mastering mp3-codec neural-codec quantize vst-host; do
    [ -f "engine/build/${bin}" ] || fail "Missing engine binary: ${bin}"
    cp "engine/build/${bin}" "${STAGING_DIR}/engine/"
done
# Copy ggml dylibs
cp engine/build/libggml*.dylib "${STAGING_DIR}/engine/" 2>/dev/null || true

# Lua plugins — dual directory system:
#   engine/plugins/  = native/built-in plugins (solvers, schedulers, guidance)
#   plugins/         = community/user plugins (same structure, overrides native)
# ace-server scans both via engine_dir and project_dir resolution from binary path
if [ -d "engine/plugins" ]; then
    mkdir -p "${STAGING_DIR}/engine/plugins"
    cp -r engine/plugins/* "${STAGING_DIR}/engine/plugins/"
    PLUGIN_COUNT=$(find "${STAGING_DIR}/engine/plugins" -name "*.lua" | wc -l | tr -d ' ')
    ok "${PLUGIN_COUNT} native plugins copied (engine/plugins/)"
fi
if [ -d "plugins" ]; then
    cp -r plugins "${STAGING_DIR}/"
    ok "Community plugins dir copied (plugins/)"
fi

# App-owned optional resources.
[ ! -d noise_samples ] || cp -R noise_samples "${STAGING_DIR}/noise_samples"
[ ! -d Essentia ] || cp -R Essentia "${STAGING_DIR}/Essentia"

# Bundled server and app-owned resources, matching CI.
mkdir -p "${STAGING_DIR}/server/data"
cp release/staging/server/server.mjs "${STAGING_DIR}/server/"
[ ! -d release/staging/server/chunks ] || cp -R release/staging/server/chunks "${STAGING_DIR}/server/"
cp -r server/node_modules "${STAGING_DIR}/server/"
cp -R server/src/data/. "${STAGING_DIR}/server/data/"
cp server/node_modules/ffmpeg-static/ffmpeg "${STAGING_DIR}/server/ffmpeg"
ok "Server copied"

# Pre-built UI
mkdir -p "${STAGING_DIR}/ui"
cp -r ui/dist "${STAGING_DIR}/ui/"
ok "UI copied"

# Bundled runtime and shared release inputs.
mkdir -p "${STAGING_DIR}/runtime/bin"
cp "${NODE_EXTRACTED}/bin/node" "${STAGING_DIR}/runtime/bin/node"
cp release/HOT-Step.sh release/profile-path.mjs release/README.txt "${STAGING_DIR}/"
cp release/.env.example "${STAGING_DIR}/.env.example"
printf '%s\n' "v${VERSION}" > "${STAGING_DIR}/VERSION"
chmod +x "${STAGING_DIR}/HOT-Step.sh"
# Exact package identity for the updater (docs/dev/releasing.md)
"${STAGING_DIR}/runtime/bin/node" release/build-info.mjs "${STAGING_DIR}" --os macos --arch arm64 --variant metal --tag "v${VERSION}" || fail "build-info.mjs failed"
"${STAGING_DIR}/runtime/bin/node" release/check-layout.mjs --package "${STAGING_DIR}" --os macos --variant metal || fail "build-info.json check failed"

ok "Release assembled"

# Boot the assembled server with the shipped node, so a missing chunk fails here.
"${STAGING_DIR}/runtime/bin/node" release/smoke-boot.mjs "${STAGING_DIR}/server/server.mjs"     || fail "Assembled server failed its boot test"
ok "Server boot test passed"

# ── Step 6: Set permissions ──────────────────────────────────────────
step 6 "Setting permissions..."

find "${STAGING_DIR}/engine" -type f -perm +0111 -exec chmod +x {} \; 2>/dev/null || true
chmod +x "${STAGING_DIR}/engine/"* 2>/dev/null || true
chmod +x "${STAGING_DIR}/runtime/bin/node"
chmod +x "${STAGING_DIR}/HOT-Step.sh"
ok "Permissions set"

# ── Step 7: Create archive ──────────────────────────────────────────
step 7 "Creating archive..."

cd release-staging
tar czf "${OUTPUT_FILE}" "${RELEASE_NAME}"
cd "${ROOT_DIR}"

# Size info
SIZE_MB=$(du -sm "${OUTPUT_FILE}" | cut -f1)
ok "${RELEASE_NAME}.tar.gz (${SIZE_MB} MB)"

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║        Release package ready! 🎉         ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "  File:  ${OUTPUT_FILE}"
echo "  Size:  ${SIZE_MB} MB"
echo ""
echo "  Test locally:"
echo "    cd release-staging/${RELEASE_NAME}"
echo "    ./HOT-Step.sh"
echo ""

# Clean up staging (keep the archive)
# rm -rf release-staging
