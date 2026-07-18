"""
Asymmetric Bets Engine — rebuilt 2026-07-18.

One job, stated plainly: find setups where the MOST we can lose is small and
KNOWN, the plausible gain is a large multiple of that loss, and then bet a
meaningful, portfolio-scaled amount. No setup enters the book without a
defined max loss and an asymmetry ratio clearing ASYM_MIN_RATIO.

Setup sources (each produces AsymmetricSetup objects):
  1. deep_value    — stocks below net current asset value: downside floored by
                     hard assets (liquidation value), upside = re-rating to NCAV.
  2. option_tail   — cheap long options: max loss = premium (structurally
                     capped), upside = modeled tail payoff. Delegates execution
                     to the proven Cornwall leg-selection pipeline with a
                     budget from THIS engine's risk budget.
  3. capitulation  — multi-day crash inside an intact uptrend: stop just under
                     the capitulation low (small defined loss), target = the
                     pre-crash level (typically 3-8x the stop distance).
  4. breakout_r    — range breakout with a structural stop under the base and
                     a measured-move target; only taken when the R multiple
                     clears the gate.

Sizing: risk budget = equity * ASYM_TOTAL_RISK_PCT across all engine bets;
each bet risks at most equity * ASYM_PER_BET_RISK_PCT. Quantity is derived
from the DEFINED per-unit loss, so the worst case is a known fraction of the
portfolio while the payoff target is ASYM_MIN_RATIO+ times that.

Positions opened here are tagged mode="asym_r" in the equity-overlay registry
(with their stop/target) and managed by manage_asym_positions() each cycle.
Every evaluated setup — taken or skipped, with the reason — is written to
.runtime/asymmetric_setups.json for the ops console.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.params import (
    ASYM_MIN_RATIO,
    ASYM_OPTION_BUDGET_FRACTION,
    ASYM_PER_BET_RISK_PCT,
    ASYM_TIME_STOP_DAYS,
    ASYM_TOTAL_RISK_PCT,
    DEEP_VALUE_SCAN_MAX_AGE_HOURS,
    ENABLE_ASYMMETRIC_ENGINE,
)
from core.execution.manager import release_cash_from_sweep
from core.ml.deep_value import load_scan_snapshot
from core.telemetry.notifications import send_alert
from core.telemetry.state_manager import (
    get_equity_overlay_metadata,
    register_equity_overlay,
    remove_equity_overlay_metadata,
)
from core.telemetry.trade_decision_tape import record_trade_decision

logger = logging.getLogger(f"strategy.{__name__}")

_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = _ROOT / ".runtime" / "asymmetric_setups.json"

ASYM_MODE = "asym_r"


@dataclass
class AsymmetricSetup:
    source: str              # deep_value | option_tail | capitulation | breakout_r
    symbol: str
    instrument: str          # equity | option
    entry_price: float
    max_loss_per_unit: float   # DEFINED downside per unit — the engine's contract
    target_gain_per_unit: float
    confidence: float          # 0..1
    stop_price: float | None = None
    target_price: float | None = None
    note: str = ""

    @property
    def asymmetry_ratio(self) -> float:
        if self.max_loss_per_unit <= 0:
            return 0.0
        return self.target_gain_per_unit / self.max_loss_per_unit

    def to_row(self, status: str, reason: str = "") -> dict:
        row = asdict(self)
        row["asymmetry_ratio"] = round(self.asymmetry_ratio, 2)
        row["status"] = status
        row["reason"] = reason
        return row


def _f(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Setup sources
# --------------------------------------------------------------------------- #

def deep_value_setups() -> list[AsymmetricSetup]:
    """Net-nets from the nightly scan: hard-asset floor under the price."""
    setups: list[AsymmetricSetup] = []
    for row in load_scan_snapshot(DEEP_VALUE_SCAN_MAX_AGE_HOURS) or []:
        if row.get("failed_gates"):
            continue
        price = _f(row.get("price"))
        ncav = _f(row.get("ncav_per_share"))
        liquidation = _f(row.get("liquidation_per_share"))
        if price <= 0 or ncav <= price:
            continue
        # Downside: price falling to 80% of fire-sale liquidation value.
        floor = max(0.0, liquidation * 0.8)
        max_loss = max(price * 0.10, price - floor)   # never assume < 10% loss possible
        target_gain = ncav * 0.9 - price
        setups.append(AsymmetricSetup(
            source="deep_value", symbol=str(row.get("symbol")), instrument="equity",
            entry_price=price, max_loss_per_unit=max_loss, target_gain_per_unit=target_gain,
            confidence=_f(row.get("score"), 0.5),
            stop_price=round(floor, 2) if floor > 0 else None,
            target_price=round(ncav * 0.9, 2),
            note=f"NCAV {ncav:.2f} vs price {price:.2f}, liq floor {liquidation:.2f}",
        ))
    return setups


def capitulation_setups(daily_by_symbol: dict[str, pd.DataFrame]) -> list[AsymmetricSetup]:
    """Multi-day crash in an intact uptrend: tight stop, retrace target."""
    setups: list[AsymmetricSetup] = []
    for symbol, daily in daily_by_symbol.items():
        try:
            if daily is None or len(daily) < 80:
                continue
            close = daily["close"].astype(float)
            low = daily["low"].astype(float)
            price = float(close.iloc[-1])

            ret3 = close.pct_change(3).dropna()
            window = ret3.iloc[-90:]
            z = (float(ret3.iloc[-1]) - float(window.mean())) / (float(window.std()) or 1e-9)
            # Trend gate on the PRE-crash price: a genuine capitulation usually
            # pierces the 60d mean itself — what matters is that the trend was
            # intact before the washout.
            pre_crash_price = float(close.iloc[-4])
            trend_ok = pre_crash_price > float(close.rolling(60).mean().iloc[-4])
            if z >= -2.0 or not trend_ok:
                continue

            diffs = close.diff().abs()
            atr = float(diffs.rolling(14).mean().iloc[-1] or 0.0)
            if atr <= 0:
                continue
            capit_low = float(low.iloc[-3:].min())
            stop = capit_low - 0.5 * atr
            pre_crash = float(close.iloc[-4])
            max_loss = price - stop
            target_gain = pre_crash - price
            if max_loss <= 0 or target_gain <= 0:
                continue
            setups.append(AsymmetricSetup(
                source="capitulation", symbol=symbol, instrument="equity",
                entry_price=price, max_loss_per_unit=max_loss, target_gain_per_unit=target_gain,
                confidence=float(np.clip(0.5 + 0.1 * (abs(z) - 2.0), 0.5, 0.85)),
                stop_price=round(stop, 2), target_price=round(pre_crash, 2),
                note=f"3d z={z:.1f}, stop under capitulation low",
            ))
        except Exception as exc:
            logger.debug("capitulation setup %s failed: %s", symbol, exc)
    return setups


def breakout_r_setups(daily_by_symbol: dict[str, pd.DataFrame]) -> list[AsymmetricSetup]:
    """Range breakout with structural stop + measured-move target."""
    setups: list[AsymmetricSetup] = []
    for symbol, daily in daily_by_symbol.items():
        try:
            if daily is None or len(daily) < 60:
                continue
            close = daily["close"].astype(float)
            high = daily["high"].astype(float)
            low = daily["low"].astype(float)
            price = float(close.iloc[-1])

            base_high = float(high.iloc[-31:-1].max())
            base_low = float(low.iloc[-31:-1].min())
            range_height = base_high - base_low
            if range_height <= 0 or price <= base_high:
                continue
            diffs = close.diff().abs()
            atr = float(diffs.rolling(14).mean().iloc[-1] or 0.0)
            # Structural stop just below the broken level.
            stop = base_high - max(0.5 * atr, 0.01 * price)
            max_loss = price - stop
            target = base_high + range_height        # measured move
            target_gain = target - price
            if max_loss <= 0 or target_gain <= 0:
                continue
            setups.append(AsymmetricSetup(
                source="breakout_r", symbol=symbol, instrument="equity",
                entry_price=price, max_loss_per_unit=max_loss, target_gain_per_unit=target_gain,
                confidence=0.55,
                stop_price=round(stop, 2), target_price=round(target, 2),
                note=f"30d range {base_low:.2f}-{base_high:.2f}, measured move",
            ))
        except Exception as exc:
            logger.debug("breakout setup %s failed: %s", symbol, exc)
    return setups


def fetch_daily_frames(symbols: list[str], period: str = "6mo") -> dict[str, pd.DataFrame]:
    """Thin yfinance daily-bars fetch for the equity setup sources."""
    import yfinance as yf

    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
            if hist is None or hist.empty:
                continue
            frames[symbol] = hist.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        except Exception:
            continue
    return frames


# --------------------------------------------------------------------------- #
# Risk budget + engine
# --------------------------------------------------------------------------- #

def _deployed_risk(positions: list, overlay_meta: dict) -> float:
    """Dollars currently at risk in engine-held positions (defined losses)."""
    at_risk = 0.0
    for pos in positions:
        symbol = str(getattr(pos, "symbol", "") or "").upper()
        meta = overlay_meta.get(symbol) or {}
        if str(meta.get("mode") or "") != ASYM_MODE:
            continue
        qty = abs(_f(getattr(pos, "qty", 0)))
        price = _f(getattr(pos, "current_price", None)) or _f(getattr(pos, "avg_entry_price", None))
        stop = _f(meta.get("stop_price"))
        if qty > 0 and price > 0 and 0 < stop < price:
            at_risk += (price - stop) * qty
        # Cheap long option tickets: whole premium is the defined risk.
    for pos in positions:
        try:
            cost = abs(_f(getattr(pos, "market_value", 0)))
            if 0 < cost < 150 and _f(getattr(pos, "qty", 0)) > 0 and len(str(getattr(pos, "symbol", ""))) > 10:
                at_risk += cost
        except Exception:
            continue
    return at_risk


def run_asymmetric_engine(
    client,
    total_equity: float,
    positions: list,
    equity_universe: list[str] | None = None,
    daily_frames: dict[str, pd.DataFrame] | None = None,
) -> list[str]:
    """
    One engine pass: collect setups from every source, gate on the asymmetry
    ratio, size from the portfolio risk budget, execute, and snapshot the
    reasoning. Returns human-readable action strings.
    """
    actions: list[str] = []
    if not ENABLE_ASYMMETRIC_ENGINE or total_equity <= 0:
        return actions

    overlay_meta = get_equity_overlay_metadata()
    held_symbols = {
        str(getattr(p, "symbol", "") or "").upper()
        for p in positions
        if _f(getattr(p, "qty", 0)) != 0
    }

    risk_budget_total = total_equity * float(ASYM_TOTAL_RISK_PCT)
    risk_deployed = _deployed_risk(positions, overlay_meta)
    risk_remaining = max(0.0, risk_budget_total - risk_deployed)
    per_bet_risk_cap = total_equity * float(ASYM_PER_BET_RISK_PCT)

    # ---- Collect equity setups --------------------------------------- #
    if daily_frames is None:
        universe = [s for s in (equity_universe or []) if s]
        daily_frames = fetch_daily_frames(universe[:20]) if universe else {}

    setups: list[AsymmetricSetup] = []
    setups += deep_value_setups()
    setups += capitulation_setups(daily_frames)
    setups += breakout_r_setups(daily_frames)
    setups.sort(key=lambda s: s.asymmetry_ratio * s.confidence, reverse=True)

    evaluated: list[dict] = []
    for setup in setups:
        ratio = setup.asymmetry_ratio
        if ratio < float(ASYM_MIN_RATIO):
            evaluated.append(setup.to_row("skipped", f"ratio {ratio:.1f} < {ASYM_MIN_RATIO}"))
            continue
        if setup.symbol.upper() in held_symbols:
            evaluated.append(setup.to_row("skipped", "already held"))
            continue
        if risk_remaining < per_bet_risk_cap * 0.25:
            evaluated.append(setup.to_row("skipped", "risk budget exhausted"))
            continue

        bet_risk = min(per_bet_risk_cap, risk_remaining)
        qty = int(bet_risk // setup.max_loss_per_unit)
        if qty <= 0 or setup.entry_price <= 0:
            evaluated.append(setup.to_row("skipped", "too small to size"))
            continue
        notional = qty * setup.entry_price

        try:
            release_cash_from_sweep(client, required_cash=notional, reason=f"asym {setup.source} {setup.symbol}")
            client.market_buy(setup.symbol, qty=qty, order_label=f"Asym {setup.source} {setup.symbol}")
            register_equity_overlay(setup.symbol.upper(), {
                "symbol": setup.symbol.upper(),
                "mode": ASYM_MODE,
                "source": setup.source,
                "entered_at_utc": datetime.now(timezone.utc).isoformat(),
                "entry_price": round(setup.entry_price, 4),
                "stop_price": setup.stop_price,
                "target_price": setup.target_price,
                "max_loss_per_unit": round(setup.max_loss_per_unit, 4),
                "asymmetry_ratio": round(ratio, 2),
            })
            held_symbols.add(setup.symbol.upper())
            risk_remaining -= qty * setup.max_loss_per_unit
            action = (
                f"ASYM BET [{setup.source}] {setup.symbol}: {qty} @ ~${setup.entry_price:.2f} "
                f"risking ${qty * setup.max_loss_per_unit:,.0f} for {ratio:.1f}x payoff "
                f"(target ${setup.target_price}, stop ${setup.stop_price})"
            )
            actions.append(action)
            logger.info(action)
            record_trade_decision(
                status="EXECUTED", strategy=f"asym_{setup.source}", symbol=setup.symbol,
                action="buy_equity", confidence=setup.confidence,
                reason=f"Asymmetric bet: risk ${qty * setup.max_loss_per_unit:,.0f} for {ratio:.1f}x upside. {setup.note}",
                details=setup.to_row("taken"),
            )
            send_alert(f"🎯 Asymmetric bet [{setup.source}]: {setup.symbol} — {ratio:.1f}x payoff-to-risk", "INFO")
            evaluated.append(setup.to_row("taken"))
        except Exception as exc:
            logger.error("Asym entry failed %s: %s", setup.symbol, exc)
            evaluated.append(setup.to_row("error", str(exc)))

    # ---- Option tails via the proven Cornwall pipeline ---------------- #
    option_budget = risk_remaining * float(ASYM_OPTION_BUDGET_FRACTION)
    if option_budget >= 25.0 and equity_universe:
        try:
            from core.execution.execution import deploy_asymmetric_bets

            deploy_asymmetric_bets(
                client, equity_universe, total_equity, positions,
                budget_override=option_budget,
            )
            evaluated.append({
                "source": "option_tail", "symbol": "(universe)", "status": "delegated",
                "reason": f"budget ${option_budget:,.0f} to Cornwall leg selector (max loss = premium)",
            })
        except Exception as exc:
            logger.debug("option tail source skipped: %s", exc)

    _write_snapshot(evaluated, risk_budget_total, risk_deployed, risk_remaining, total_equity)
    return actions


def manage_asym_positions(client, positions: list) -> list[str]:
    """Enforce each asym_r position's stop / target / time stop."""
    actions: list[str] = []
    overlay_meta = get_equity_overlay_metadata()
    for pos in positions:
        symbol = str(getattr(pos, "symbol", "") or "").upper()
        meta = overlay_meta.get(symbol) or {}
        if str(meta.get("mode") or "") != ASYM_MODE:
            continue
        qty = int(abs(_f(getattr(pos, "qty", 0))))
        price = _f(getattr(pos, "current_price", None)) or _f(getattr(pos, "avg_entry_price", None))
        if qty <= 0 or price <= 0:
            continue
        stop = _f(meta.get("stop_price"))
        target = _f(meta.get("target_price"))
        reason = None
        if 0 < stop and price <= stop:
            reason = "stop hit (defined max loss realized)"
        elif 0 < target <= price:
            reason = "target hit (asymmetric payoff realized)"
        else:
            try:
                entered = datetime.fromisoformat(str(meta.get("entered_at_utc")))
                if entered.tzinfo is None:
                    entered = entered.replace(tzinfo=timezone.utc)
                held_days = (datetime.now(timezone.utc) - entered).total_seconds() / 86400.0
                if held_days > float(ASYM_TIME_STOP_DAYS):
                    reason = "time stop"
            except (TypeError, ValueError):
                pass
        if reason is None:
            continue
        try:
            client.market_sell(symbol, qty=qty, order_label=f"Asym exit {symbol}")
            remove_equity_overlay_metadata(symbol)
            action = f"ASYM EXIT {symbol} ({qty}): {reason}"
            actions.append(action)
            logger.info(action)
            record_trade_decision(
                status="EXECUTED", strategy=f"asym_{meta.get('source', 'r')}", symbol=symbol,
                action="sell_equity", confidence=0.6, reason=reason, details=dict(meta),
            )
        except Exception as exc:
            logger.error("Asym exit failed %s: %s", symbol, exc)
    return actions


def _write_snapshot(evaluated: list[dict], budget: float, deployed: float, remaining: float, equity: float) -> None:
    try:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "equity": round(equity, 2),
            "risk_budget": round(budget, 2),
            "risk_deployed": round(deployed, 2),
            "risk_remaining": round(remaining, 2),
            "min_ratio": float(ASYM_MIN_RATIO),
            "setups": evaluated,
        }, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("asym snapshot write failed: %s", exc)
