#!/usr/bin/env bash
# Stops and removes the LaunchAgent plists.
set -euo pipefail

LAUNCH_AGENTS="$HOME/Library/LaunchAgents"

for label in com.quantalpha.dashboard com.quantalpha.watchdog; do
  plist="$LAUNCH_AGENTS/${label}.plist"
  launchctl bootout "gui/$(id -u)/${label}" 2>/dev/null && echo "→ Unloaded ${label}" || true
  rm -f "$plist" && echo "→ Removed $plist" || true
done

echo "✓ Services removed"
