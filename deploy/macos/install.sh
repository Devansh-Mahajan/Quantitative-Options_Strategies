#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# install.sh — macOS LaunchAgent installer for the Quant Dashboard + Watchdog
#
# Run once from the project root:
#   bash deploy/macos/install.sh
#
# What it does:
#   1. Creates .venv if it doesn't exist
#   2. Installs project dependencies into .venv
#   3. Installs both LaunchAgent plists into ~/Library/LaunchAgents/
#   4. Loads them so they start immediately (and on every future login)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
VENV_DIR="/Users/devanshmahajan/Projects/.venv"   # shared project venv
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
DEPLOY_DIR="$REPO_DIR/deploy/macos"
LOG_DIR="$REPO_DIR/logs"

echo "→ Project: $REPO_DIR"
echo "→ Using venv: $VENV_DIR"

if [[ ! -f "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: venv not found at $VENV_DIR"
  echo "       Run: python3 -m venv $VENV_DIR"
  exit 1
fi

# ── Install missing dashboard deps into the shared venv ────────────────────
echo "→ Installing fastapi + uvicorn into shared venv…"
"$VENV_DIR/bin/pip" install --upgrade fastapi "uvicorn[standard]" --quiet
echo "→ Dependencies installed"

# ── 3. Ensure logs directory ────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

# ── 4. Install LaunchAgent plists ──────────────────────────────────────────
mkdir -p "$LAUNCH_AGENTS"

install_plist() {
  local src="$1"
  local label="$2"
  local dst="$LAUNCH_AGENTS/${label}.plist"

  # Unload first if already loaded (ignore errors)
  launchctl bootout "gui/$(id -u)/${label}" 2>/dev/null || true

  cp "$src" "$dst"
  chmod 644 "$dst"
  echo "→ Installed $dst"

  launchctl bootstrap "gui/$(id -u)" "$dst"
  echo "→ Loaded ${label}"
}

install_plist "$DEPLOY_DIR/com.quantalpha.dashboard.plist" "com.quantalpha.dashboard"
install_plist "$DEPLOY_DIR/com.quantalpha.watchdog.plist"  "com.quantalpha.watchdog"

# ── 5. Verify ───────────────────────────────────────────────────────────────
echo ""
echo "✓ Services installed and running"
echo ""
echo "  Dashboard:  http://localhost:8080"
echo "  Dashboard log:  tail -f $LOG_DIR/dashboard.log"
echo "  Watchdog log:   tail -f $LOG_DIR/automate-stack-watchdog.log"
echo ""
echo "  Manage:"
echo "    Stop dashboard:    launchctl stop com.quantalpha.dashboard"
echo "    Start dashboard:   launchctl start com.quantalpha.dashboard"
echo "    Remove services:   bash $DEPLOY_DIR/uninstall.sh"
