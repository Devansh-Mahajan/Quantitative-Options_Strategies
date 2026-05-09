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


def get_daily_pnl() -> float:
    today = datetime.now(timezone.utc).date().isoformat()
    with _conn() as c:
        row = c.execute(
            "SELECT COALESCE(SUM(pnl),0) AS total FROM trades WHERE ts >= ? AND status='closed'",
            (today,),
        ).fetchone()
    return float(row["total"])


def get_peak_equity() -> float:
    with _conn() as c:
        row = c.execute("SELECT MAX(equity) AS peak FROM equity_curve").fetchone()
    return float(row["peak"] or 0)
