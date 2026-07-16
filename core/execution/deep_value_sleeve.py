"""
Deep-value (Graham net-net) execution sleeve — the share-buying counterpart to
the Cornwall options sleeve in core/execution/execution.py.

Consumes the nightly scan snapshot written by scripts/nightly_deep_value_scan.py
(via core/ml/deep_value.py) and manages buy-and-hold share positions tagged
mode="deep_value" in the equity-overlay registry. The overlay's own rebalance
loop skips that mode; entries AND exits happen exclusively here.

Self-budgeted like the Cornwall sleeve: uses the cash sweep, not the options
cycle's buying_power.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from alpaca.trading.enums import AssetClass

from config.params import (
    DEEP_VALUE_AUTO_EXECUTE,
    DEEP_VALUE_MAX_ALLOCATION,
    DEEP_VALUE_MAX_HOLD_DAYS,
    DEEP_VALUE_MAX_POSITIONS,
    DEEP_VALUE_MAX_SYMBOL_WEIGHT,
    DEEP_VALUE_MIN_ENTRY_SCORE,
    DEEP_VALUE_SCAN_MAX_AGE_HOURS,
    DEEP_VALUE_STOP_LOSS,
    DEEP_VALUE_TARGET_NCAV_FRACTION,
    ENABLE_DEEP_VALUE,
    SWEEP_TICKER,
)
from core.execution.manager import release_cash_from_sweep
from core.ml.deep_value import load_scan_snapshot
from core.telemetry.notifications import send_alert
from core.telemetry.state_manager import (
    get_equity_overlay_metadata,
    register_equity_overlay,
    remove_equity_overlay_metadata,
)

logger = logging.getLogger(f"strategy.{__name__}")

DEEP_VALUE_MODE = "deep_value"


def _f(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _held_deep_value_positions(positions: list, overlay_meta: dict) -> dict[str, object]:
    held = {}
    for pos in positions:
        symbol = str(getattr(pos, "symbol", "") or "").upper()
        if getattr(pos, "asset_class", None) != AssetClass.US_EQUITY or symbol == SWEEP_TICKER:
            continue
        meta = overlay_meta.get(symbol) or {}
        if str(meta.get("mode") or "") == DEEP_VALUE_MODE:
            held[symbol] = pos
    return held


def _held_days(metadata: dict) -> float:
    try:
        entered = datetime.fromisoformat(str(metadata.get("entered_at_utc")))
        if entered.tzinfo is None:
            entered = entered.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - entered.astimezone(timezone.utc)).total_seconds() / 86400.0
    except (TypeError, ValueError):
        return 0.0


def _exit_reason(pos, metadata: dict, fresh_by_symbol: dict[str, dict]) -> str | None:
    price = _f(getattr(pos, "current_price", None)) or _f(getattr(pos, "avg_entry_price", None))
    pnl_pct = _f(getattr(pos, "unrealized_plpc", None))
    entry_ncav = _f(metadata.get("ncav_per_share"))

    if entry_ncav > 0 and price >= DEEP_VALUE_TARGET_NCAV_FRACTION * entry_ncav:
        return "fair value target"
    if pnl_pct <= -float(DEEP_VALUE_STOP_LOSS):
        return "hard stop"
    if _held_days(metadata) > float(DEEP_VALUE_MAX_HOLD_DAYS):
        return "time stop"

    # Thesis break: last night's scan re-priced this symbol and the asset floor
    # is now below what we're paying to hold it.
    fresh = fresh_by_symbol.get(str(getattr(pos, "symbol", "") or "").upper())
    if fresh is not None and price > 0:
        fresh_ncav = _f(fresh.get("ncav_per_share"))
        if fresh_ncav <= 0 or fresh_ncav < price:
            return "thesis break (NCAV floor gone)"
    return None


def deploy_deep_value_bets(client, total_equity: float, positions: list) -> list[str]:
    """
    Manage the deep-value sleeve for one strategy cycle: run exits on held
    positions first, then deploy capped entries from the freshest nightly scan.
    Returns human-readable action strings (mirrors the equity overlay).
    """
    actions: list[str] = []
    if not ENABLE_DEEP_VALUE:
        return actions

    overlay_meta = get_equity_overlay_metadata()
    held = _held_deep_value_positions(positions, overlay_meta)

    candidates = load_scan_snapshot(DEEP_VALUE_SCAN_MAX_AGE_HOURS) or []
    fresh_by_symbol = {str(row.get("symbol") or "").upper(): row for row in candidates}

    # ---- Exits first --------------------------------------------------- #
    for symbol, pos in list(held.items()):
        metadata = overlay_meta.get(symbol) or {}
        reason = _exit_reason(pos, metadata, fresh_by_symbol)
        if reason is None:
            continue
        qty = max(0, int(_f(getattr(pos, "qty", 0))))
        if qty <= 0:
            remove_equity_overlay_metadata(symbol)
            held.pop(symbol, None)
            continue
        try:
            client.market_sell(symbol, qty=qty, order_label=f"Deep value exit {symbol}")
            remove_equity_overlay_metadata(symbol)
            held.pop(symbol, None)
            action = f"Closed deep value position {symbol} ({qty} shares): {reason}."
            actions.append(action)
            logger.info(action)
            send_alert(f"💎 Deep value exit: {symbol} — {reason}", level="INFO")
        except Exception as exc:
            logger.error("Deep value exit failed for %s: %s", symbol, exc)

    # ---- Entries -------------------------------------------------------- #
    if not DEEP_VALUE_AUTO_EXECUTE or not candidates:
        return actions

    sleeve_value = sum(
        _f(getattr(pos, "market_value", None)) or (_f(getattr(pos, "current_price", None)) * _f(getattr(pos, "qty", 0)))
        for pos in held.values()
    )
    budget = max(0.0, total_equity * float(DEEP_VALUE_MAX_ALLOCATION) - sleeve_value)
    per_name_cap = total_equity * float(DEEP_VALUE_MAX_SYMBOL_WEIGHT)

    entry_candidates = sorted(
        (
            row
            for row in candidates
            if not row.get("failed_gates")
            and _f(row.get("score")) >= float(DEEP_VALUE_MIN_ENTRY_SCORE)
            and str(row.get("symbol") or "").upper() not in held
        ),
        key=lambda row: _f(row.get("score")),
        reverse=True,
    )

    for row in entry_candidates:
        if len(held) >= int(DEEP_VALUE_MAX_POSITIONS) or budget <= 0:
            break
        symbol = str(row.get("symbol") or "").upper()
        price = _f(row.get("price"))
        if price <= 0:
            continue
        spend = min(per_name_cap, budget)
        qty = int(spend // price)
        if qty <= 0:
            continue
        spend = qty * price
        try:
            release_cash_from_sweep(client, required_cash=spend, reason=f"deep value {symbol}")
            client.market_buy(symbol, qty=qty, order_label=f"Deep value entry {symbol}")
            register_equity_overlay(
                symbol,
                {
                    "symbol": symbol,
                    "mode": DEEP_VALUE_MODE,
                    "entered_at_utc": datetime.now(timezone.utc).isoformat(),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "entry_price": round(price, 4),
                    "ncav_per_share": _f(row.get("ncav_per_share")),
                    "liquidation_per_share": _f(row.get("liquidation_per_share")),
                    "score": _f(row.get("score")),
                    "model_probability": row.get("model_probability"),
                },
            )
            held[symbol] = object()  # placeholder so max-position count holds within this cycle
            budget -= spend
            action = (
                f"Opened deep value position {symbol}: {qty} shares @ ~${price:.2f} "
                f"(NCAV/share ${_f(row.get('ncav_per_share')):.2f}, score {_f(row.get('score')):.2f})."
            )
            actions.append(action)
            logger.info(action)
            send_alert(f"💎 Deep value entry: {symbol} {qty} shares @ ~${price:.2f}", level="INFO")
        except Exception as exc:
            logger.error("Deep value entry failed for %s: %s", symbol, exc)

    return actions
