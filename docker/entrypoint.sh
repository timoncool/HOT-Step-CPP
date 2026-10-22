#!/bin/bash
# ============================================================================
# HOT-Step 9000 CPP — Docker Entrypoint
# Verifies GPU access and starts the Node.js server (which spawns ace-server)
# ============================================================================

set -e

# Persistent storage is opt-in; Compose keeps its existing bind mounts.
if [ -n "${HOT_STEP_DATA_ROOT:-}" ]; then
    case "$HOT_STEP_DATA_ROOT" in
        /*) ;;
        *) echo "HOT_STEP_DATA_ROOT must be an absolute path" >&2; exit 1 ;;
    esac
    for name in models adapters data logs; do
        target="${HOT_STEP_DATA_ROOT%/}/$name"
        case "$name" in
            data) link=/app/server/data ;;
            *) link="/app/$name" ;;
        esac
        mkdir -p "$target"
        if [ -d "$link" ] && [ ! -L "$link" ]; then
            rmdir "$link" 2>/dev/null || {
                echo "Cannot link non-empty or mounted $link to $target" >&2
                exit 1
            }
        elif [ -e "$link" ] && [ ! -L "$link" ]; then
            echo "Cannot replace existing file $link" >&2
            exit 1
        fi
        ln -sfnT "$target" "$link"
    done
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║     HOT-Step 9000 ⚡ Docker             ║"
echo "║    High-Performance Music Generation     ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Verify GPU access ───────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU:    ${GPU_NAME} (${GPU_MEM})"
    echo "  Driver: ${GPU_DRIVER}"
else
    echo "  ⚠ nvidia-smi not found — GPU may not be available"
    echo "  Check that Docker has GPU access (--gpus=all or deploy.resources in compose)"
fi

# ── Verify engine binary exists ─────────────────────────────────────
if [ -f /app/engine/ace-server ]; then
    echo "  Engine: /app/engine/ace-server ✓"
else
    echo "  ⚠ ace-server binary not found at /app/engine/"
    echo "  The build may have failed — check Docker build logs"
fi

# ── Ensure bind-mount directories exist (in case host dirs are empty) ──
mkdir -p /app/models /app/adapters /app/server/data

# ── Check for models ────────────────────────────────────────────────
MODEL_COUNT=$(find /app/models -name '*.gguf' 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -gt 0 ]; then
    echo "  Models: ${MODEL_COUNT} GGUF file(s) found"
else
    echo "  ⚠ No .gguf models found in /app/models"
    echo "  Place model files in the ./models/ directory on the host"
fi

echo ""
echo "  Server:  http://localhost:${SERVER_PORT:-3001}"
echo "  Engine:  http://localhost:${ACESTEPCPP_PORT:-8085}"
echo ""
echo "  Starting server..."
echo ""

# ── Start the Node.js server ────────────────────────────────────────
# The server spawns ace-server as a child process automatically.
# Use exec to replace the shell — proper signal handling for graceful shutdown.
cd /app/server
exec node --import tsx/esm src/index.ts
