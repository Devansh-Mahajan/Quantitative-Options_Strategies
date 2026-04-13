#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/workspace/Quantitative-Options_Strategies"
LOG_DIR="$REPO_DIR/logs"
RUN_LOG="$LOG_DIR/automate-stack.log"
WATCHDOG_LOG="$LOG_DIR/automate-stack-watchdog.log"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if pgrep -f "automate-stack" >/dev/null 2>&1; then
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] automate-stack is running" >> "$WATCHDOG_LOG"
else
  nohup automate-stack --restart-on-failure >> "$RUN_LOG" 2>&1 &
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] automate-stack was down; restarted" >> "$WATCHDOG_LOG"
fi

if [[ -n "${HEALTHCHECK_URL:-}" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl -fsS -m 10 "$HEALTHCHECK_URL" >/dev/null && \
      echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] healthcheck ping ok" >> "$WATCHDOG_LOG" || \
      echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] healthcheck ping failed" >> "$WATCHDOG_LOG"
  else
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] curl missing; healthcheck ping skipped" >> "$WATCHDOG_LOG"
  fi
fi
