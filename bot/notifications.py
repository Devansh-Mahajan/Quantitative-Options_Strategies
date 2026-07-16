"""
Crypto/Binance-stack Discord + Telegram alerting with non-blocking async sends.
Not the same module as core.telemetry.notifications, which is the sync
Discord-only alerter used by the options/Alpaca-stack strategy loop — both are
live, serving different trading stacks.
"""

from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

from bot.config import cfg

log = logging.getLogger("bot.notifications")
_warned_invalid_discord_url = False


def _normalize_webhook_url(url: str) -> str:
    value = str(url or "").strip().strip("\"'")
    if not value or value.startswith("#"):
        return ""
    if value.startswith("discord.com/") or value.startswith("www.discord.com/"):
        value = f"https://{value}"
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return value
    return ""


async def _post(url: str, payload: dict) -> None:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, json=payload)
            if r.status_code not in (200, 204):
                log.warning("Notification HTTP %d: %s", r.status_code, r.text[:200])
    except Exception as exc:
        log.warning("Notification send failed: %s", exc)


async def send_discord(message: str, title: str = "", color: int = 0x00B0F4) -> None:
    global _warned_invalid_discord_url

    url = _normalize_webhook_url(cfg.discord_webhook_url)
    if not url:
        if cfg.discord_webhook_url and not _warned_invalid_discord_url:
            log.warning("Discord notifications disabled: DISCORD_WEBHOOK_URL is blank or malformed.")
            _warned_invalid_discord_url = True
        return
    payload: dict = {
        "embeds": [
            {
                "title": title or "Bot Alert",
                "description": message[:2048],
                "color": color,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
    }
    await _post(url, payload)


async def send_telegram(message: str) -> None:
    if not cfg.telegram_bot_token or not cfg.telegram_chat_id:
        return
    url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage"
    await _post(url, {"chat_id": cfg.telegram_chat_id, "text": message[:4096], "parse_mode": "HTML"})


async def alert(message: str, title: str = "", level: str = "INFO") -> None:
    color_map = {"INFO": 0x00B0F4, "WARNING": 0xFFA500, "ERROR": 0xFF0000, "CRITICAL": 0x8B0000}
    color = color_map.get(level.upper(), 0x00B0F4)
    log.log(getattr(logging, level.upper(), logging.INFO), "[ALERT] %s", message)
    await asyncio.gather(
        send_discord(message, title=title, color=color),
        send_telegram(f"<b>[{level}]</b> {title}\n{message}"),
        return_exceptions=True,
    )


def alert_sync(message: str, title: str = "", level: str = "INFO") -> None:
    """Fire-and-forget wrapper for non-async contexts."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(alert(message, title=title, level=level))
        else:
            loop.run_until_complete(alert(message, title=title, level=level))
    except Exception:
        log.warning("alert_sync failed — Discord/Telegram config may be missing")
