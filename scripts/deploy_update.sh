#!/usr/bin/env bash
# =============================================================================
# deploy_update.sh — Zero-downtime deploy from GitHub
#
# Usage: ./scripts/deploy_update.sh [--no-restart] [--branch main]
#
# What it does:
#   1. git fetch + fast-forward pull (safe — aborts if local changes exist)
#   2. pip install -e . (picks up any new dependencies)
#   3. Gracefully restarts binance-bot via systemd (SIGTERM → waits → starts)
#   4. Waits 15s and confirms the service is running
#   5. Sends a Discord/Telegram notification (if webhooks configured in .env)
#
# Called by crontab or GitHub Actions webhook.
# Never force-pushes, never deletes branches.
# =============================================================================
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
BRANCH="${BRANCH:-main}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_BOT="binance-bot"
SERVICE_RISK="binance-risk-monitor"
LOG_FILE="${REPO_DIR}/.runtime/deploy.log"
VENV_PYTHON="${REPO_DIR}/.venv/bin/python"
VENV_PIP="${REPO_DIR}/.venv/bin/pip"
NO_RESTART=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-restart) NO_RESTART=true ;;
        --branch) BRANCH="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Deploy started at $(ts)"
echo "  Repo: ${REPO_DIR}  →  branch: ${BRANCH}"
echo "══════════════════════════════════════════════════════"

cd "$REPO_DIR"

# ── 1. Safety: refuse if working tree is dirty ───────────────────────────────
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: Working tree has uncommitted changes. Aborting deploy."
    echo "       Commit or stash your changes before deploying."
    exit 1
fi

# ── 2. Git pull ───────────────────────────────────────────────────────────────
echo "[$(ts)] git fetch origin ${BRANCH}"
git fetch --quiet origin "${BRANCH}"

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/${BRANCH}")

if [[ "$LOCAL" == "$REMOTE" ]]; then
    echo "[$(ts)] Already up-to-date (${LOCAL:0:8}). Nothing to deploy."
    # Still restart if --no-restart not passed and service exists
    if $NO_RESTART; then
        exit 0
    fi
    # Skip to restart step
else
    echo "[$(ts)] Pulling ${LOCAL:0:8} → ${REMOTE:0:8}"
    git merge --ff-only "origin/${BRANCH}"
    echo "[$(ts)] Pull complete."

    # Show what changed (for the log)
    echo "[$(ts)] Changed files:"
    git diff --name-only "${LOCAL}" "${REMOTE}" | sed 's/^/         /'
fi

# ── 3. Install / update Python dependencies ───────────────────────────────────
echo "[$(ts)] pip install -e . (silent)"
"$VENV_PIP" install -e . --quiet

# ── 4. Restart services ───────────────────────────────────────────────────────
if $NO_RESTART; then
    echo "[$(ts)] --no-restart passed, skipping service restart."
else
    for SVC in "$SERVICE_RISK" "$SERVICE_BOT"; do
        if systemctl is-active --quiet "$SVC" 2>/dev/null; then
            echo "[$(ts)] Restarting ${SVC} …"
            systemctl restart "$SVC"
        elif systemctl is-enabled --quiet "$SVC" 2>/dev/null; then
            echo "[$(ts)] Starting ${SVC} (was inactive) …"
            systemctl start "$SVC"
        else
            echo "[$(ts)] ${SVC} not found / not enabled — skipping"
        fi
    done

    # Wait and confirm
    sleep 15
    echo "[$(ts)] Service status after restart:"
    for SVC in "$SERVICE_BOT" "$SERVICE_RISK"; do
        STATUS=$(systemctl is-active "$SVC" 2>/dev/null || echo "not-found")
        echo "         ${SVC}: ${STATUS}"
        if [[ "$STATUS" != "active" ]]; then
            echo "WARNING: ${SVC} is not active after restart!"
            journalctl -u "$SVC" --no-pager -n 20 || true
        fi
    done
fi

# ── 5. Optional notification ─────────────────────────────────────────────────
# Source .env if it exists so we can read DISCORD_WEBHOOK / TELEGRAM_BOT_TOKEN
ENV_FILE="${REPO_DIR}/.env"
DISCORD_WEBHOOK=""
TELEGRAM_BOT_TOKEN=""
TELEGRAM_CHAT_ID=""

if [[ -f "$ENV_FILE" ]]; then
    # Safe extraction — only pull the variables we need
    DISCORD_WEBHOOK=$(grep -E '^DISCORD_WEBHOOK=' "$ENV_FILE" | cut -d'=' -f2- | tr -d '"' || true)
    TELEGRAM_BOT_TOKEN=$(grep -E '^TELEGRAM_BOT_TOKEN=' "$ENV_FILE" | cut -d'=' -f2- | tr -d '"' || true)
    TELEGRAM_CHAT_ID=$(grep -E '^TELEGRAM_CHAT_ID=' "$ENV_FILE" | cut -d'=' -f2- | tr -d '"' || true)
fi

MSG="Deploy complete @ $(ts) | branch=${BRANCH} | commit=${REMOTE:0:8}"

if [[ -n "$DISCORD_WEBHOOK" ]]; then
    curl -s -X POST "$DISCORD_WEBHOOK" \
         -H "Content-Type: application/json" \
         -d "{\"content\": \"${MSG}\"}" >/dev/null || true
fi

if [[ -n "$TELEGRAM_BOT_TOKEN" && -n "$TELEGRAM_CHAT_ID" ]]; then
    curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
         -d "chat_id=${TELEGRAM_CHAT_ID}&text=${MSG}" >/dev/null || true
fi

echo "[$(ts)] Deploy finished successfully."
echo "══════════════════════════════════════════════════════"
