#!/bin/bash
# HOT-Step CPP — Development Mode (Linux / WSL / macOS)
#
# Starts Vite dev server (hot-reload UI) + Node.js server (tsx watch).
# Automatically finds a compatible Node.js.
#
# Restart loop, like dev.bat's server/restart-loop.cmd: when the server writes
# .restart-requested (in-app Restart, or a finished worker update) and exits,
# the whole stack comes back up from the current checkout. Quit in the app,
# Ctrl-C or SIGTERM stops it. Only this script's own children are stopped.
#
# Usage:
#   ./dev.sh
#   nohup ./dev.sh > logs/dev.log 2>&1 < /dev/null &   # over ssh; PID in logs/dev.pid

set -e

cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"
MARKER="${ROOT_DIR}/.restart-requested"
mkdir -p logs
QUIT_FLAG="${ROOT_DIR}/logs/dev.quit"
echo $$ > logs/dev.pid

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║     HOT-Step 9000 ⚡ DEV MODE           ║"
echo "║       Vite HMR + tsx watch               ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Find Node.js ────────────────────────────────────────────────────
find_node() {
    if [ -x "${ROOT_DIR}/runtime/node" ]; then
        echo "${ROOT_DIR}/runtime"; return 0
    fi
    # Homebrew's plain `node` can be newer than 24; its versioned kegs are not.
    for v in 24 22 20; do
        for bp in "/opt/homebrew/opt/node@${v}/bin" "/usr/local/opt/node@${v}/bin"; do
            if [ -x "${bp}/node" ]; then echo "$bp"; return 0; fi
        done
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
    echo "  Node.js: $(node --version)"
else
    echo "❌ No compatible Node.js found (need 20-24)."
    echo "   Fix: brew install node@24"
    exit 1
fi

# ── Install deps if needed ───────────────────────────────────────────
if [ ! -d "server/node_modules" ]; then
    echo "📦 Installing server dependencies..."
    cd server && npm install --loglevel=warn && cd ..
fi
if [ ! -d "ui/node_modules" ]; then
    echo "📦 Installing UI dependencies..."
    cd ui && npm install --loglevel=warn && cd ..
fi
# better-sqlite3 is native: node_modules installed under another Node major
# (say Homebrew's newer plain `node`) will not load. Rebuild it for this one.
if ! (cd server && node -e "new (require('better-sqlite3'))(':memory:').close()") 2>/dev/null; then
    echo "📦 Rebuilding better-sqlite3 for Node $(node --version)..."
    (cd server && npm rebuild better-sqlite3 --loglevel=warn)
fi

# ── Process helpers ──────────────────────────────────────────────────
# Every descendant of a PID, children before parents.
descendants() {
    local c
    for c in $(pgrep -P "$1"); do descendants "$c"; echo "$c"; done
}

# Stop a PID and everything under it: TERM, up to 15 s, then KILL.
# The tree is listed first, so children reparented mid-shutdown still go.
stop_tree() {
    [ -n "$1" ] || return 0
    local pids; pids="$(descendants "$1") $1"
    kill -TERM $pids 2>/dev/null || return 0
    for _ in $(seq 1 30); do
        kill -0 $pids 2>/dev/null || return 0
        sleep 0.5
    done
    kill -KILL $pids 2>/dev/null || true
}

cleanup() {
    trap - EXIT INT TERM
    echo ""
    echo "[dev.sh] Shutting down..."
    stop_tree "${SERVER_PID}"
    stop_tree "${VITE_PID}"
    rm -f "${QUIT_FLAG}" logs/dev.pid
    echo "[dev.sh] Stopped"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# tsx watch outlives the server process it runs: after Quit or Restart it sits
# waiting for a file change. So watch its child. Restart writes the marker
# before exiting; Quit logs this line when the request arrives.
watch_server_output() {
    local line
    while IFS= read -r line; do
        printf '%s\n' "$line"
        case "$line" in *"[Server] Shutdown requested via API"*) : > "${QUIT_FLAG}" ;; esac
    done
}

# ── Run the stack until Quit, restarting on the marker ──────────────
rm -f "${MARKER}" "${QUIT_FLAG}"
while :; do
    echo "🚀 Starting Vite dev server..."
    (cd ui && exec npx vite --port 3000 --host) &
    VITE_PID=$!
    sleep 2

    echo "🚀 Starting Node.js server..."
    echo "   UI:   http://localhost:3000 (Vite HMR)"
    echo "   API:  http://localhost:3001"
    echo ""
    # tsx directly, not through npx, so the server process is its direct child.
    (cd server && HOT_STEP_DEV=1 exec ./node_modules/.bin/tsx watch src/index.ts) \
        > >(watch_server_output) 2>&1 &
    SERVER_PID=$!

    # Leave when tsx dies, or when its server exited after Quit or Restart.
    # A crash without either (a typo mid-edit) waits for the fix, as before.
    # Match `node` only: bash forks the output filter above before the exec,
    # so a bash child of tsx is always there.
    while kill -0 "${SERVER_PID}" 2>/dev/null; do
        if ! pgrep -x -P "${SERVER_PID}" node >/dev/null \
            && { [ -f "${MARKER}" ] || [ -f "${QUIT_FLAG}" ]; }; then
            break
        fi
        sleep 1
    done

    stop_tree "${SERVER_PID}"; SERVER_PID=""
    stop_tree "${VITE_PID}"; VITE_PID=""
    if [ -f "${MARKER}" ] && [ ! -f "${QUIT_FLAG}" ]; then
        rm -f "${MARKER}"
        echo ""
        echo "[dev.sh] Restarting..."
        sleep 2
        continue
    fi
    break
done
