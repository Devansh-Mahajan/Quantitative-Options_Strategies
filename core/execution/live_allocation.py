from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.trading.enums import AssetClass

from bot.config import cfg
from config.params import RISK_ALLOCATION, SWEEP_TICKER
from core.utils import try_parse_option_symbol

ROOT = Path(__file__).resolve().parents[1]
LIVE_ALLOCATION_PATH = ROOT / ".runtime" / "live_allocation.json"
NY_TZ = ZoneInfo("America/New_York")

OPTIONS_EQUITY_SHARE_OPEN = 0.65
OPTIONS_EQUITY_SHARE_CLOSED = 0.15


@dataclass(frozen=True)
class BookBudget:
    share: float
    equity_budget: float
    risk_budget: float
    used_market_value: float
    active_positions: int
    remaining_risk_budget: float


@dataclass(frozen=True)
class LiveAllocationSnapshot:
    generated_at_utc: str
    market_session_open: bool
    total_equity: float
    gross_market_value: float
    options_equity: BookBudget
    crypto: BookBudget

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "market_session_open": self.market_session_open,
            "total_equity": round(float(self.total_equity), 2),
            "gross_market_value": round(float(self.gross_market_value), 2),
            "books": {
                "options_equity": _rounded_dict(self.options_equity),
                "crypto": _rounded_dict(self.crypto),
            },
        }


def _rounded_dict(book: BookBudget) -> dict[str, Any]:
    payload = asdict(book)
    for key in ("share", "equity_budget", "risk_budget", "used_market_value", "remaining_risk_budget"):
        payload[key] = round(float(payload[key]), 6 if key == "share" else 2)
    payload["active_positions"] = int(payload["active_positions"])
    return payload


def market_session_open_now(now: datetime | None = None) -> bool:
    ts = now or datetime.now(timezone.utc).astimezone(NY_TZ)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=NY_TZ)
    else:
        ts = ts.astimezone(NY_TZ)
    if ts.weekday() >= 5:
        return False
    open_mark = ts.replace(hour=9, minute=30, second=0, microsecond=0)
    close_mark = ts.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_mark <= ts < close_mark


def normalize_asset_class(asset_class: object) -> str:
    return str(getattr(asset_class, "value", asset_class) or "").strip().lower()


def classify_position_book(position: object) -> str:
    symbol = str(getattr(position, "symbol", "") or "").strip().upper()
    if not symbol:
        return "unknown"
    if symbol == SWEEP_TICKER:
        return "cash"

    asset_class = normalize_asset_class(getattr(position, "asset_class", None))
    if asset_class == normalize_asset_class(getattr(AssetClass, "US_OPTION", "us_option")):
        return "options_equity"
    if asset_class == normalize_asset_class(getattr(AssetClass, "CRYPTO", "crypto")):
        return "crypto"
    if try_parse_option_symbol(symbol) is not None:
        return "options_equity"
    if cfg.is_crypto_symbol(symbol):
        return "crypto"
    return "options_equity"


def estimate_position_market_value(position: object) -> float:
    book = classify_position_book(position)
    if book == "cash":
        return 0.0

    market_value = _safe_float(getattr(position, "market_value", None))
    if market_value is not None and abs(market_value) > 0.0:
        return abs(market_value)

    current_price = _safe_float(getattr(position, "current_price", None))
    qty = abs(_safe_float(getattr(position, "qty", None), fallback=0.0) or 0.0)
    if current_price is not None and qty > 0.0:
        return abs(current_price) * qty

    entry_price = _safe_float(getattr(position, "avg_entry_price", None))
    if entry_price is not None and qty > 0.0:
        multiplier = 100.0 if try_parse_option_symbol(str(getattr(position, "symbol", "") or "")) is not None else 1.0
        return abs(entry_price) * qty * multiplier
    return 0.0


def summarize_live_books(positions: list[object]) -> dict[str, dict[str, float]]:
    summary = {
        "options_equity": {"used_market_value": 0.0, "active_positions": 0},
        "crypto": {"used_market_value": 0.0, "active_positions": 0},
        "cash": {"used_market_value": 0.0, "active_positions": 0},
        "unknown": {"used_market_value": 0.0, "active_positions": 0},
    }
    for position in positions:
        book = classify_position_book(position)
        summary.setdefault(book, {"used_market_value": 0.0, "active_positions": 0})
        summary[book]["used_market_value"] += estimate_position_market_value(position)
        summary[book]["active_positions"] += 1
    return summary


def build_live_allocation_snapshot(
    *,
    total_equity: float,
    positions: list[object],
    market_open: bool,
) -> LiveAllocationSnapshot:
    total_equity = max(0.0, float(total_equity or 0.0))
    book_summary = summarize_live_books(list(positions))
    gross_market_value = float(
        book_summary["options_equity"]["used_market_value"] + book_summary["crypto"]["used_market_value"]
    )
    options_share = OPTIONS_EQUITY_SHARE_OPEN if market_open else OPTIONS_EQUITY_SHARE_CLOSED
    crypto_share = max(0.0, 1.0 - options_share)
    total_risk_budget = total_equity * float(RISK_ALLOCATION)

    def _budget(book: str, share: float) -> BookBudget:
        used_market_value = float(book_summary.get(book, {}).get("used_market_value", 0.0) or 0.0)
        risk_budget = total_risk_budget * share
        return BookBudget(
            share=float(share),
            equity_budget=total_equity * share,
            risk_budget=risk_budget,
            used_market_value=used_market_value,
            active_positions=int(book_summary.get(book, {}).get("active_positions", 0) or 0),
            remaining_risk_budget=max(0.0, risk_budget - used_market_value),
        )

    return LiveAllocationSnapshot(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        market_session_open=bool(market_open),
        total_equity=total_equity,
        gross_market_value=gross_market_value,
        options_equity=_budget("options_equity", options_share),
        crypto=_budget("crypto", crypto_share),
    )


def write_live_allocation(snapshot: LiveAllocationSnapshot, path: Path = LIVE_ALLOCATION_PATH) -> dict[str, Any]:
    payload = snapshot.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)
    return payload


def read_live_allocation(path: Path = LIVE_ALLOCATION_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(value: object, fallback: float | None = None) -> float | None:
    try:
        if value is None:
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback
