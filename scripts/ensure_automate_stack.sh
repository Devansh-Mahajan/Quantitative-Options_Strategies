#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
RUNTIME_DIR="$REPO_DIR/.runtime"
LOG_DIR="$REPO_DIR/logs"
RUN_LOG="$LOG_DIR/automate-stack.log"
WATCHDOG_LOG="$LOG_DIR/automate-stack-watchdog.log"
PID_FILE="$RUNTIME_DIR/automate-stack.pid"
LOCK_FILE="$RUNTIME_DIR/automate-stack-watchdog.lock"
PYTHON_BIN="$REPO_DIR/.venv/bin/python"

DASHBOARD_PID_FILE="$RUNTIME_DIR/dashboard.pid"
DASHBOARD_LOG="$LOG_DIR/dashboard.log"
DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_watchdog() {
  echo "[$(timestamp)] $*" >> "$WATCHDOG_LOG"
}

process_matches() {
  local pid="$1"
  local args

  if ! kill -0 "$pid" 2>/dev/null; then
    return 1
  fi

  args="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  [[ "$args" =~ scripts\.automation_controller|automate-stack ]]
}

discover_running_pid() {
  local pid

  if [[ -f "$PID_FILE" ]]; then
    pid="$(<"$PID_FILE")"
    if [[ -n "$pid" ]] && process_matches "$pid"; then
      printf "%s\n" "$pid"
      return 0
    fi
  fi

  pid="$(pgrep -fo -f "scripts\\.automation_controller|automate-stack" || true)"
  if [[ -n "$pid" ]] && process_matches "$pid"; then
    printf "%s\n" "$pid" > "$PID_FILE"
    printf "%s\n" "$pid"
    return 0
  fi

  return 1
}

start_stack() {
  export PYTHONUNBUFFERED=1
  nohup "$PYTHON_BIN" -m scripts.automation_controller --restart-on-failure >> "$RUN_LOG" 2>&1 &
  local pid="$!"
  printf "%s\n" "$pid" > "$PID_FILE"
  sleep 1

  if process_matches "$pid"; then
    log_watchdog "automate-stack was down; restarted with pid $pid"
    return 0
  fi

  log_watchdog "automate-stack failed to start; inspect $RUN_LOG"
  return 1
}

mkdir -p "$LOG_DIR" "$RUNTIME_DIR"
cd "$REPO_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  log_watchdog "missing interpreter at $PYTHON_BIN"
  exit 1
fi

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    log_watchdog "watchdog already running; skipping overlapping invocation"
    exit 0
  fi
fi

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# ── Dashboard watchdog ────────────────────────────────────────────────────────

dashboard_running() {
  local pid
  if [[ -f "$DASHBOARD_PID_FILE" ]]; then
    pid="$(<"$DASHBOARD_PID_FILE")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

start_dashboard() {
  export PYTHONUNBUFFERED=1
  nohup "$PYTHON_BIN" -m scripts.run_dashboard --port "$DASHBOARD_PORT" >> "$DASHBOARD_LOG" 2>&1 &
  local pid="$!"
  printf "%s\n" "$pid" > "$DASHBOARD_PID_FILE"
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    log_watchdog "dashboard started on port $DASHBOARD_PORT with pid $pid"
  else
    log_watchdog "dashboard failed to start — check $DASHBOARD_LOG"
  fi
}

if dashboard_running; then
  log_watchdog "dashboard is already running"
else
  start_dashboard
fi

# ── Trading stack ──────────────────────────────────────────────────────────

if running_pid="$(discover_running_pid)"; then
  log_watchdog "automate-stack is running with pid $running_pid"
else
  start_stack
fi

if [[ -n "${HEALTHCHECK_URL:-}" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl -fsS -m 10 "$HEALTHCHECK_URL" >/dev/null && \
      log_watchdog "healthcheck ping ok" || \
      log_watchdog "healthcheck ping failed"
  else
    log_watchdog "curl missing; healthcheck ping skipped"
  fi
fi
