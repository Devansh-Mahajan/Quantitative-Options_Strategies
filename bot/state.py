"""SQLite-backed persistent state: trades, equity curve, model metadata."""

from __future__ import annotations
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("bot.state")

_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = _ROOT / ".runtime" / "bot_state.db"


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(_DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            symbol      TEXT    NOT NULL,
            market      TEXT    NOT NULL,  -- spot | futures | options
            side        TEXT    NOT NULL,  -- BUY | SELL
            quantity    REAL    NOT NULL,
            price       REAL    NOT NULL,
            strategy    TEXT    NOT NULL,
            order_id    TEXT,
            pnl         REAL    DEFAULT 0,
            status      TEXT    DEFAULT 'open',
            meta        TEXT    DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS equity_curve (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            ts       TEXT    NOT NULL,
            equity   REAL    NOT NULL,
            drawdown REAL    NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS risk_events (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            ts      TEXT    NOT NULL,
            event   TEXT    NOT NULL,
            detail  TEXT    DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS model_metadata (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS kv (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """)
    log.debug("State DB initialised at %s", _DB_PATH)


def record_trade(
    symbol: str,
    market: str,
    side: str,
    quantity: float,
    price: float,
    strategy: str,
    order_id: str = "",
    meta: dict | None = None,
) -> int:
    ts = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO trades (ts,symbol,market,side,quantity,price,strategy,order_id,meta) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (ts, symbol, market, side, quantity, price, strategy, order_id, json.dumps(meta or {})),
        )
        return cur.lastrowid  # type: ignore[return-value]


def update_trade_pnl(trade_id: int, pnl: float, status: str = "closed") -> None:
    with _conn() as c:
        c.execute("UPDATE trades SET pnl=?, status=? WHERE id=?", (pnl, status, trade_id))


def record_equity(equity: float, drawdown: float) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute("INSERT INTO equity_curve (ts,equity,drawdown) VALUES (?,?,?)", (ts, equity, drawdown))


def record_risk_event(event: str, detail: str = "") -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute("INSERT INTO risk_events (ts,event,detail) VALUES (?,?,?)", (ts, event, detail))
    log.warning("RISK EVENT [%s]: %s", event, detail)


def set_model_meta(key: str, value: Any) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO model_metadata (key,value) VALUES (?,?)",
            (key, json.dumps(value)),
        )


def get_model_meta(key: str, default: Any = None) -> Any:
    with _conn() as c:
        row = c.execute("SELECT value FROM model_metadata WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def kv_set(key: str, value: Any) -> None:
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO kv (key,value) VALUES (?,?)", (key, json.dumps(value)))


def kv_get(key: str, default: Any = None) -> Any:
    with _conn() as c:
        row = c.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def get_open_trades() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM trades WHERE status='open' ORDER BY ts DESC").fetchall()
    return [dict(r) for r in rows]


def _parse_ts(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def get_daily_pnl_snapshot() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_floor = day_start.isoformat()

    with _conn() as c:
        latest = c.execute(
            "SELECT ts, equity FROM equity_curve ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        anchor = c.execute(
            "SELECT ts, equity FROM equity_curve WHERE ts < ? ORDER BY ts DESC LIMIT 1",
            (day_floor,),
        ).fetchone()
        first_today = c.execute(
            "SELECT ts, equity FROM equity_curve WHERE ts >= ? ORDER BY ts ASC LIMIT 1",
            (day_floor,),
        ).fetchone()
        intraday = c.execute(
            "SELECT COUNT(*) AS samples, MIN(equity) AS low_equity, MAX(equity) AS high_equity "
            "FROM equity_curve WHERE ts >= ?",
            (day_floor,),
        ).fetchone()
        closed = c.execute(
            "SELECT COALESCE(SUM(pnl),0) AS total, COUNT(*) AS count "
            "FROM trades WHERE ts >= ? AND status='closed'",
            (day_floor,),
        ).fetchone()
        peak = c.execute("SELECT MAX(equity) AS peak FROM equity_curve").fetchone()

    closed_trade_pnl = float((closed["total"] if closed else 0.0) or 0.0)
    closed_trade_count = int((closed["count"] if closed else 0) or 0)
    peak_equity = float((peak["peak"] if peak else 0.0) or 0.0)

    if not latest:
        return {
            "source": "closed_trades_only",
            "day_boundary_utc": day_floor,
            "daily_pnl": closed_trade_pnl,
            "daily_pnl_pct": None,
            "session_start_equity": None,
            "session_start_ts": None,
            "current_equity": None,
            "latest_equity_ts": None,
            "equity_freshness_seconds": None,
            "intraday_low_equity": None,
            "intraday_high_equity": None,
            "intraday_range_pct": None,
            "peak_equity": peak_equity if peak_equity > 0 else None,
            "distance_to_peak_pct": None,
            "equity_samples_today": 0,
            "closed_trade_pnl_today": closed_trade_pnl,
            "closed_trade_count_today": closed_trade_count,
        }

    current_equity = float(latest["equity"] or 0.0)
    anchor_row = anchor or first_today or latest
    start_equity = float(anchor_row["equity"] or 0.0) if anchor_row else current_equity
    daily_pnl = current_equity - start_equity
    daily_pnl_pct = (daily_pnl / start_equity * 100.0) if start_equity else None

    low_equity = float((intraday["low_equity"] if intraday else 0.0) or current_equity)
    high_equity = float((intraday["high_equity"] if intraday else 0.0) or current_equity)
    sample_count = int((intraday["samples"] if intraday else 0) or 0)
    intraday_range_pct = ((high_equity - low_equity) / start_equity * 100.0) if start_equity else None
    distance_to_peak_pct = ((current_equity / peak_equity) - 1.0) * 100.0 if peak_equity > 0 else None

    latest_ts = _parse_ts(latest["ts"])
    freshness_seconds = max(0.0, (now - latest_ts).total_seconds()) if latest_ts else None

    return {
        "source": "equity_curve_mark_to_market",
        "day_boundary_utc": day_floor,
        "daily_pnl": daily_pnl,
        "daily_pnl_pct": daily_pnl_pct,
        "session_start_equity": start_equity,
        "session_start_ts": anchor_row["ts"] if anchor_row else None,
        "current_equity": current_equity,
        "latest_equity_ts": latest["ts"],
        "equity_freshness_seconds": freshness_seconds,
        "intraday_low_equity": low_equity,
        "intraday_high_equity": high_equity,
        "intraday_range_pct": intraday_range_pct,
        "peak_equity": peak_equity if peak_equity > 0 else current_equity,
        "distance_to_peak_pct": distance_to_peak_pct,
        "equity_samples_today": sample_count,
        "closed_trade_pnl_today": closed_trade_pnl,
        "closed_trade_count_today": closed_trade_count,
    }


def get_daily_pnl() -> float:
    snapshot = get_daily_pnl_snapshot()
    return float(snapshot.get("daily_pnl") or 0.0)


def get_peak_equity() -> float:
    with _conn() as c:
        row = c.execute("SELECT MAX(equity) AS peak FROM equity_curve").fetchone()
    return float(row["peak"] or 0)
