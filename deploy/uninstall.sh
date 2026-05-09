#!/usr/bin/env bash
# Remove all QuantAlpha systemd services.
# Usage: sudo bash deploy/uninstall.sh
set -euo pipefail

for unit in quant-watchdog.timer quant-watchdog.service quant-dashboard.service quant-bot.service; do
  systemctl stop    "$unit" 2>/dev/null || true
  systemctl disable "$unit" 2>/dev/null || true
  rm -f "/etc/systemd/system/$unit"
  echo "→ Removed $unit"
done

systemctl daemon-reload
echo "✓ All QuantAlpha services removed"
