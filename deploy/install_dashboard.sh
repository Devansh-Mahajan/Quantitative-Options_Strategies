#!/usr/bin/env bash
# Deploy quant-dashboard systemd service.
# Run as root on the target server after git pull.
#
# Usage:
#   sudo bash deploy/install_dashboard.sh
#   sudo bash deploy/install_dashboard.sh --user myuser --dir /opt/quant-bot

set -euo pipefail

DEPLOY_USER="${DEPLOY_USER:-$(whoami)}"
DEPLOY_DIR="${DEPLOY_DIR:-$(pwd)}"

# Parse simple flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) DEPLOY_USER="$2"; shift 2;;
    --dir)  DEPLOY_DIR="$2";  shift 2;;
    *) echo "Unknown flag: $1"; exit 1;;
  esac
done

SERVICE_SRC="$(dirname "$0")/systemd/quant-dashboard.service"
SERVICE_DST="/etc/systemd/system/quant-dashboard.service"

echo "→ Installing quant-dashboard.service"
echo "  User: ${DEPLOY_USER}"
echo "  Dir:  ${DEPLOY_DIR}"

# Substitute placeholders
sed \
  -e "s|CHANGE_ME|${DEPLOY_USER}|g" \
  -e "s|/opt/quant-bot|${DEPLOY_DIR}|g" \
  "$SERVICE_SRC" > "$SERVICE_DST"

systemctl daemon-reload
systemctl enable quant-dashboard.service
systemctl restart quant-dashboard.service

echo ""
echo "✓ quant-dashboard.service enabled and started"
echo "  Dashboard URL: http://$(hostname -I | awk '{print $1}'):8080"
echo "  Logs: journalctl -fu quant-dashboard"
