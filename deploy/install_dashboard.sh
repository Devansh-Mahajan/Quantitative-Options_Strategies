#!/usr/bin/env bash
# Install QuantAlpha systemd services on Ubuntu.
#
# Usage (from project root):
#   sudo bash deploy/install_dashboard.sh [--user USER] [--dir /path/to/project]
#
# Defaults:
#   --user   the user who owns the project (default: current sudo caller, or whoami)
#   --dir    project root (default: directory containing this script's parent)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

# ── Defaults ────────────────────────────────────────────────────────────────
DEPLOY_USER="${SUDO_USER:-$(whoami)}"
DEPLOY_DIR="$REPO_DIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) DEPLOY_USER="$2"; shift 2 ;;
    --dir)  DEPLOY_DIR="$2";  shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# ── Resolve Python ───────────────────────────────────────────────────────────
PYTHON_BIN=""
for candidate in \
    "$DEPLOY_DIR/.venv/bin/python" \
    "/home/${DEPLOY_USER}/Projects/.venv/bin/python" \
    "/opt/quant/.venv/bin/python" \
    "$(su -c 'command -v python3' "$DEPLOY_USER" 2>/dev/null || true)"; do
  if [[ -x "$candidate" ]]; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: no Python interpreter found. Create a venv first:"
  echo "  python3 -m venv $DEPLOY_DIR/.venv && $DEPLOY_DIR/.venv/bin/pip install -r requirements.txt"
  exit 1
fi

SYSTEMD_DIR="/etc/systemd/system"
DEPLOY_SYSTEMD="$SCRIPT_DIR/systemd"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  QuantAlpha Service Installer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  User:    $DEPLOY_USER"
echo "  Dir:     $DEPLOY_DIR"
echo "  Python:  $PYTHON_BIN"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p "$DEPLOY_DIR/logs" "$DEPLOY_DIR/.runtime"
chown -R "$DEPLOY_USER":"$DEPLOY_USER" "$DEPLOY_DIR/logs" "$DEPLOY_DIR/.runtime"

# ── Install fastapi + uvicorn if missing ─────────────────────────────────────
echo "→ Checking fastapi/uvicorn..."
if ! "$PYTHON_BIN" -c "import fastapi, uvicorn" 2>/dev/null; then
  echo "→ Installing fastapi + uvicorn into venv..."
  "$(dirname "$PYTHON_BIN")/pip" install --upgrade fastapi "uvicorn[standard]" --quiet
  echo "→ Done"
else
  echo "→ Already installed"
fi

# ── Install unit files ────────────────────────────────────────────────────────
install_unit() {
  local name="$1"
  local src="$DEPLOY_SYSTEMD/${name}"
  local dst="$SYSTEMD_DIR/${name}"

  sed \
    -e "s|__USER__|${DEPLOY_USER}|g" \
    -e "s|__DIR__|${DEPLOY_DIR}|g"  \
    -e "s|__PYTHON__|${PYTHON_BIN}|g" \
    "$src" > "$dst"

  chmod 644 "$dst"
  echo "→ Installed $dst"
}

install_unit "quant-dashboard.service"
install_unit "quant-bot.service"
install_unit "quant-watchdog.service"
install_unit "quant-watchdog.timer"

# ── Reload + enable + start ───────────────────────────────────────────────────
systemctl daemon-reload

for svc in quant-dashboard quant-bot; do
  systemctl enable  "$svc"
  systemctl restart "$svc"
  echo "→ $svc: $(systemctl is-active $svc)"
done

# Timer (not the oneshot service directly)
systemctl enable  quant-watchdog.timer
systemctl restart quant-watchdog.timer
echo "→ quant-watchdog.timer: $(systemctl is-active quant-watchdog.timer)"

# ── Status ────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ All services installed and running"
echo ""
echo "  Dashboard:   http://$(hostname -I | awk '{print $1}'):8080"
echo ""
echo "  Logs:"
echo "    journalctl -fu quant-dashboard"
echo "    journalctl -fu quant-bot"
echo "    journalctl -fu quant-watchdog"
echo ""
echo "  Control:"
echo "    sudo systemctl stop    quant-dashboard"
echo "    sudo systemctl start   quant-dashboard"
echo "    sudo systemctl restart quant-bot"
echo "    sudo systemctl status  quant-watchdog.timer"
echo ""
echo "  Remove:  sudo bash deploy/uninstall.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
