from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from alpaca.trading.enums import AssetClass

from config.params import (
    DELTA_HEDGE_INVERSE_SYMBOL,
    DELTA_HEDGE_LONG_SYMBOL,
    DELTA_HEDGE_MAX_ALLOCATION,
    DELTA_HEDGE_MAX_VIX,
    DELTA_HEDGE_MIN_GAP,
    DELTA_HEDGE_SCALING,
    ENABLE_DELTA_HEDGE_OVERLAY,
    ENABLE_EQUITY_OVERLAY,
    EQUITY_OVERLAY_EVENT_BOOST,
    EQUITY_OVERLAY_EVENT_WINDOW_DAYS,
    EQUITY_OVERLAY_EXIT_SIGNAL,
    EQUITY_OVERLAY_IV_FLOOR_RANK,
    EQUITY_OVERLAY_MAX_ALLOCATION,
    EQUITY_OVERLAY_MAX_DISTRIBUTION_ZSCORE,
    EQUITY_OVERLAY_MAX_EVENT_CANDIDATES,
    EQUITY_OVERLAY_MAX_IV_RANK,
    EQUITY_OVERLAY_MAX_POSITIONS,
    EQUITY_OVERLAY_MAX_SYMBOL_WEIGHT,
    EQUITY_OVERLAY_MAX_VIX,
    EQUITY_OVERLAY_MAX_IV_REALIZED_RATIO,
    EQUITY_OVERLAY_MIN_EXPECTED_MOVE,
    EQUITY_OVERLAY_MIN_SIGNAL,
    EQUITY_OVERLAY_PROFIT_TARGET,
    EQUITY_OVERLAY_STOP_LOSS,
    SWEEP_TICKER,
)
from core.movement_predictor import MovementSignal
from core.ml_alpha import AlphaSignal
from core.notifications import send_alert
from core.state_manager import (
    get_equity_overlay_metadata,
    register_equity_overlay,
    remove_equity_overlay_metadata,
)

logger = logging.getLogger(f"strategy.{__name__}")
_EVENT_CONTEXT_CACHE: dict[str, "EquitySignalContext"] = {}


@dataclass
class EquitySignalContext:
    symbol: str
    earnings_days: int | None = None
    iv_rank: float | None = None
    iv_realized_ratio: float | None = None
    distribution_zscore: float | None = None
    price_percentile: float | None = None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_flow(flow_map: Mapping[str, float] | None) -> dict[str, float]:
    raw = {str(symbol).upper(): float(value) for symbol, value in (flow_map or {}).items()}
    if not raw:
        return {}
    low = min(raw.values())
    high = max(raw.values())
    if high - low <= 1e-9:
        return {symbol: 0.5 for symbol in raw}
    return {symbol: (value - low) / (high - low) for symbol, value in raw.items()}


def _is_defensive_mode(runtime_calibration, current_vix: float, movement_bias: str) -> bool:
    if movement_bias != "bullish":
        return True
    if float(current_vix) >= float(EQUITY_OVERLAY_MAX_VIX):
        return True
    if not getattr(runtime_calibration, "directional_enabled", True):
        return True
    if getattr(runtime_calibration, "current_market_state", None) in {"panic", "volatile_bear", "calm_bear"}:
        return True
    if getattr(runtime_calibration, "selected_profile", None) in {"panic_hedge", "bear_trend"}:
        return True
    return False


def _market_session_open_now() -> bool:
    current = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    if current.weekday() >= 5:
        return False
    session_open = current.replace(hour=9, minute=30, second=0, microsecond=0)
    session_close = current.replace(hour=16, minute=0, second=0, microsecond=0)
    return session_open <= current < session_close


def _extract_earnings_days(ticker: yf.Ticker) -> int | None:
    try:
        calendar = ticker.calendar
        found_date = None
        if isinstance(calendar, dict) and calendar.get("Earnings Date"):
            found_date = calendar["Earnings Date"][0]
        elif hasattr(calendar, "empty") and not calendar.empty and "Earnings Date" in calendar:
            found_date = calendar["Earnings Date"][0]
        if found_date is None:
            return None
        if hasattr(found_date, "date"):
            found_date = found_date.date()
        elif isinstance(found_date, str):
            found_date = datetime.strptime(found_date[:10], "%Y-%m-%d").date()
        return int((found_date - datetime.now(timezone.utc).date()).days)
    except Exception:
        return None


def _build_signal_context(symbol: str) -> EquitySignalContext | None:
    symbol = str(symbol).upper()
    if symbol in _EVENT_CONTEXT_CACHE:
        return _EVENT_CONTEXT_CACHE[symbol]

    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="1y")
        if history.empty or len(history) < 60:
            return None

        close = history["Close"].dropna().astype(float)
        if close.empty:
            return None
        current_price = float(close.iloc[-1])

        realized_vol_60 = _safe_float(close.pct_change().tail(60).std(), 0.0)
        if realized_vol_60:
            realized_vol_60 *= 252.0 ** 0.5
        rolling_vol = close.pct_change().rolling(window=20).std() * (252.0 ** 0.5)
        vol_min = _safe_float(rolling_vol.min())
        vol_max = _safe_float(rolling_vol.max())

        rolling_mean = close.rolling(20).mean().iloc[-1]
        rolling_std = close.rolling(20).std(ddof=0).iloc[-1]
        distribution_zscore = (
            (current_price - float(rolling_mean)) / float(rolling_std)
            if _safe_float(rolling_std, 0.0) and float(rolling_std) > 1e-9
            else 0.0
        )

        price_min = _safe_float(close.tail(252).min(), current_price)
        price_max = _safe_float(close.tail(252).max(), current_price)
        if price_max is not None and price_min is not None and (price_max - price_min) > 1e-9:
            price_percentile = (current_price - price_min) / (price_max - price_min)
        else:
            price_percentile = 0.5

        current_iv = None
        options = list(ticker.options or [])
        if options:
            target_exp = options[0]
            chain = ticker.option_chain(target_exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            if not calls.empty and not puts.empty:
                calls["strike_diff"] = (calls["strike"] - current_price).abs()
                atm_call = calls.sort_values("strike_diff").iloc[0]
                strike = float(atm_call["strike"])
                puts["strike_diff"] = (puts["strike"] - strike).abs()
                atm_put = puts.sort_values("strike_diff").iloc[0]
                current_iv = _safe_float(
                    (_safe_float(atm_call.get("impliedVolatility"), 0.0) + _safe_float(atm_put.get("impliedVolatility"), 0.0)) / 2.0
                )

        iv_rank = None
        iv_realized_ratio = None
        if current_iv is not None and vol_min is not None and vol_max is not None and vol_max > vol_min:
            iv_rank = ((current_iv - vol_min) / (vol_max - vol_min)) * 100.0
        if current_iv is not None and realized_vol_60 and realized_vol_60 > 1e-9:
            iv_realized_ratio = current_iv / realized_vol_60

        context = EquitySignalContext(
            symbol=symbol,
            earnings_days=_extract_earnings_days(ticker),
            iv_rank=_safe_float(iv_rank),
            iv_realized_ratio=_safe_float(iv_realized_ratio),
            distribution_zscore=_safe_float(distribution_zscore, 0.0),
            price_percentile=_clamp(_safe_float(price_percentile, 0.5)),
        )
        _EVENT_CONTEXT_CACHE[symbol] = context
        return context
    except Exception as exc:
        logger.debug("Could not build equity overlay event context for %s: %s", symbol, exc)
        return None


def _score_directional_candidate(
    signal: MovementSignal,
    flow_score: float,
    context: EquitySignalContext | None,
    alpha_signal: AlphaSignal | None = None,
) -> tuple[float, dict] | None:
    move_score = _clamp(abs(float(signal.expected_daily_move)) * 100.0)
    distribution_quality = 0.65
    if context and context.distribution_zscore is not None:
        if float(context.distribution_zscore) > float(EQUITY_OVERLAY_MAX_DISTRIBUTION_ZSCORE):
            return None
        distribution_quality = _clamp(
            1.0 - max(0.0, float(context.distribution_zscore)) / max(float(EQUITY_OVERLAY_MAX_DISTRIBUTION_ZSCORE), 0.5)
        )

    score = (
        0.62 * float(signal.probability_up)
        + 0.18 * float(flow_score)
        + 0.12 * move_score
        + 0.08 * distribution_quality
    )
    metadata = {
        "mode": "directional",
        "signal_probability_up": round(float(signal.probability_up), 4),
        "expected_daily_move": round(float(signal.expected_daily_move), 6),
        "flow_score": round(float(flow_score), 4),
        "distribution_quality": round(float(distribution_quality), 4),
    }

    if alpha_signal is not None:
        score += 0.16 * max(0.0, min(1.0, float(alpha_signal.percentile)))
        score += 0.08 * max(0.0, min(1.0, float(alpha_signal.alpha_score)))
        score -= 0.05 * max(0.0, min(1.0, float(alpha_signal.model_dispersion) * 25.0))
        metadata["alpha_percentile"] = round(float(alpha_signal.percentile), 4)
        metadata["alpha_score"] = round(float(alpha_signal.alpha_score), 4)
        metadata["alpha_direction"] = alpha_signal.direction

    if context:
        metadata["price_percentile"] = round(float(context.price_percentile or 0.5), 4)
        if context.iv_rank is not None:
            metadata["iv_rank"] = round(float(context.iv_rank), 2)
        if context.iv_realized_ratio is not None:
            metadata["iv_realized_ratio"] = round(float(context.iv_realized_ratio), 4)
        if context.earnings_days is not None:
            metadata["earnings_days"] = int(context.earnings_days)

        if context.earnings_days is not None and 0 <= int(context.earnings_days) <= int(EQUITY_OVERLAY_EVENT_WINDOW_DAYS):
            if context.iv_rank is not None and float(context.iv_rank) > float(EQUITY_OVERLAY_MAX_IV_RANK):
                return None
            if context.iv_realized_ratio is not None and float(context.iv_realized_ratio) > float(EQUITY_OVERLAY_MAX_IV_REALIZED_RATIO):
                return None
            score += float(EQUITY_OVERLAY_EVENT_BOOST)
            metadata["event_window"] = True

        if context.iv_rank is not None and float(context.iv_rank) <= float(EQUITY_OVERLAY_IV_FLOOR_RANK):
            score += 0.05
            metadata["iv_floor_bonus"] = True

    return score, metadata


def _build_directional_targets(
    client,
    movement_signals: Iterable[MovementSignal],
    alpha_signals: Mapping[str, AlphaSignal] | None,
    flow_map: Mapping[str, float] | None,
    total_equity: float,
    buying_power: float,
    deployment_scale: float,
    defensive_mode: bool,
    allow_new_entries: bool,
) -> dict[str, dict]:
    if defensive_mode or not allow_new_entries:
        return {}

    normalized_flow = _normalize_flow(flow_map)
    raw_candidates = []
    for signal in movement_signals:
        symbol = signal.symbol.upper()
        if signal.expected_direction != "up":
            continue
        if float(signal.probability_up) < float(EQUITY_OVERLAY_MIN_SIGNAL):
            continue
        if float(signal.expected_daily_move) < float(EQUITY_OVERLAY_MIN_EXPECTED_MOVE):
            continue
        alpha_signal = (alpha_signals or {}).get(symbol)
        base_score = (
            0.75 * float(signal.probability_up)
            + 0.15 * normalized_flow.get(symbol, 0.5)
            + 0.10 * _clamp(abs(float(signal.expected_daily_move)) * 100.0)
        )
        if alpha_signal is not None:
            base_score += 0.15 * _clamp(float(alpha_signal.percentile))
            base_score += 0.05 * _clamp(float(alpha_signal.alpha_score))
        raw_candidates.append((base_score, symbol, signal))

    raw_candidates.sort(reverse=True)
    shortlisted = raw_candidates[: max(1, int(EQUITY_OVERLAY_MAX_EVENT_CANDIDATES))]

    scored_candidates = []
    for _, symbol, signal in shortlisted:
        context = _build_signal_context(symbol)
        scored = _score_directional_candidate(signal, normalized_flow.get(symbol, 0.5), context, (alpha_signals or {}).get(symbol))
        if scored is None:
            continue
        score, metadata = scored
        scored_candidates.append((score, symbol, signal, metadata))

    scored_candidates.sort(reverse=True)
    target_symbols = scored_candidates[: max(1, int(EQUITY_OVERLAY_MAX_POSITIONS))]
    target_budget = min(
        max(0.0, float(buying_power)) * 0.50,
        float(total_equity) * float(EQUITY_OVERLAY_MAX_ALLOCATION) * max(0.20, min(1.0, float(deployment_scale))),
    )

    targets: dict[str, dict] = {}
    if not target_symbols or target_budget <= 0:
        return targets

    per_symbol_budget = min(
        target_budget / len(target_symbols),
        float(total_equity) * float(EQUITY_OVERLAY_MAX_SYMBOL_WEIGHT),
    )
    for score, symbol, signal, metadata in target_symbols:
        try:
            latest_trade = client.get_stock_latest_trade(symbol)
            price = float(latest_trade[symbol].price)
        except Exception as exc:
            logger.error("Failed reading latest stock price for %s: %s", symbol, exc)
            continue
        qty = int(per_symbol_budget // max(price, 0.01))
        if qty <= 0:
            continue
        targets[symbol] = {
            "qty": qty,
            "price": price,
            "mode": "directional",
            "score": round(float(score), 4),
            "metadata": metadata,
        }
    return targets


def _build_delta_hedge_targets(
    client,
    total_equity: float,
    buying_power: float,
    current_vix: float,
    current_port_delta: float | None,
    target_port_delta: float | None,
    allow_delta_hedge_entries: bool,
) -> dict[str, dict]:
    if not ENABLE_DELTA_HEDGE_OVERLAY or not allow_delta_hedge_entries:
        return {}
    if current_port_delta is None or target_port_delta is None:
        return {}
    if float(current_vix) > float(DELTA_HEDGE_MAX_VIX):
        return {}

    delta_gap = float(target_port_delta) - float(current_port_delta)
    if abs(delta_gap) < float(DELTA_HEDGE_MIN_GAP):
        return {}

    hedge_symbol = DELTA_HEDGE_LONG_SYMBOL if delta_gap > 0 else DELTA_HEDGE_INVERSE_SYMBOL
    try:
        latest_trade = client.get_stock_latest_trade(hedge_symbol)
        price = float(latest_trade[hedge_symbol].price)
    except Exception as exc:
        logger.error("Failed reading latest stock price for delta hedge %s: %s", hedge_symbol, exc)
        return {}

    target_budget = min(
        max(0.0, float(buying_power)) * 0.35,
        float(total_equity) * float(DELTA_HEDGE_MAX_ALLOCATION),
    )
    if target_budget <= 0:
        return {}

    target_qty = int(min(abs(delta_gap) * float(DELTA_HEDGE_SCALING), target_budget // max(price, 0.01)))
    if target_qty <= 0:
        return {}

    return {
        hedge_symbol: {
            "qty": target_qty,
            "price": price,
            "mode": "delta_hedge",
            "score": round(abs(delta_gap), 4),
            "metadata": {
                "mode": "delta_hedge",
                "delta_gap": round(float(delta_gap), 4),
                "hedge_symbol": hedge_symbol,
            },
        }
    }


def rebalance_equity_overlay(
    client,
    positions: list,
    movement_signals: Iterable[MovementSignal],
    flow_map: Mapping[str, float] | None,
    total_equity: float,
    buying_power: float,
    deployment_scale: float,
    current_vix: float,
    movement_bias: str,
    runtime_calibration,
    alpha_signals: Mapping[str, AlphaSignal] | None = None,
    current_port_delta: float | None = None,
    target_port_delta: float | None = None,
    allow_new_entries: bool = True,
    allow_delta_hedge_entries: bool = True,
) -> tuple[float, list[str]]:
    actions: list[str] = []
    if not ENABLE_EQUITY_OVERLAY:
        return buying_power, actions
    if not _market_session_open_now():
        return buying_power, actions

    overlay_meta = get_equity_overlay_metadata()
    held_equities = {
        pos.symbol.upper(): pos
        for pos in positions
        if pos.asset_class == AssetClass.US_EQUITY and pos.symbol != SWEEP_TICKER
    }
    overlay_positions = {symbol: pos for symbol, pos in held_equities.items() if symbol in overlay_meta}

    for symbol in list(overlay_meta):
        if symbol not in held_equities:
            remove_equity_overlay_metadata(symbol)

    defensive_mode = _is_defensive_mode(runtime_calibration, current_vix=current_vix, movement_bias=movement_bias)
    target_position_values = _build_directional_targets(
        client=client,
        movement_signals=movement_signals,
        alpha_signals=alpha_signals,
        flow_map=flow_map,
        total_equity=total_equity,
        buying_power=buying_power,
        deployment_scale=deployment_scale,
        defensive_mode=defensive_mode,
        allow_new_entries=allow_new_entries,
    )
    target_position_values.update(
        _build_delta_hedge_targets(
            client=client,
            total_equity=total_equity,
            buying_power=buying_power,
            current_vix=current_vix,
            current_port_delta=current_port_delta,
            target_port_delta=target_port_delta,
            allow_delta_hedge_entries=allow_delta_hedge_entries,
        )
    )

    signal_map = {signal.symbol.upper(): signal for signal in movement_signals}

    for symbol, pos in overlay_positions.items():
        current_qty = max(0, int(float(pos.qty)))
        if current_qty <= 0:
            remove_equity_overlay_metadata(symbol)
            continue

        metadata = overlay_meta.get(symbol) or {}
        mode = str(metadata.get("mode") or "directional")
        signal = signal_map.get(symbol)
        pnl_pct = float(getattr(pos, "unrealized_plpc", 0.0) or 0.0)
        target = target_position_values.get(symbol)

        should_exit = False
        if mode == "delta_hedge":
            should_exit = target is None
        else:
            should_exit = (
                target is None
                or signal is None
                or float(signal.probability_up) < float(EQUITY_OVERLAY_EXIT_SIGNAL)
                or pnl_pct <= -float(EQUITY_OVERLAY_STOP_LOSS)
                or pnl_pct >= float(EQUITY_OVERLAY_PROFIT_TARGET)
            )
            if defensive_mode:
                should_exit = True

        if should_exit:
            try:
                client.market_sell(symbol, qty=current_qty)
                remove_equity_overlay_metadata(symbol)
                action = f"Closed {mode.replace('_', ' ')} overlay in {symbol} ({current_qty} shares)."
                actions.append(action)
                logger.info(action)
                continue
            except Exception as exc:
                logger.error("Failed exiting %s overlay %s: %s", mode, symbol, exc)
                continue

        target_qty = int(target["qty"])
        if current_qty > target_qty:
            trim_qty = current_qty - target_qty
            try:
                client.market_sell(symbol, qty=trim_qty)
                action = f"Trimmed {mode.replace('_', ' ')} overlay in {symbol} by {trim_qty} shares."
                actions.append(action)
                logger.info(action)
            except Exception as exc:
                logger.error("Failed trimming %s overlay %s: %s", mode, symbol, exc)

    for symbol, target in target_position_values.items():
        current_qty = max(0, int(float(overlay_positions.get(symbol).qty))) if symbol in overlay_positions else 0
        if current_qty >= int(target["qty"]):
            register_equity_overlay(
                symbol,
                {
                    "symbol": symbol,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "mode": target["mode"],
                    **target["metadata"],
                },
            )
            continue

        buy_qty = int(target["qty"]) - current_qty
        spend = buy_qty * float(target["price"])
        if buy_qty <= 0 or spend > max(0.0, buying_power):
            continue

        try:
            client.market_buy(symbol, qty=buy_qty)
            buying_power -= spend
            register_equity_overlay(
                symbol,
                {
                    "symbol": symbol,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "mode": target["mode"],
                    "entry_price": round(float(target["price"]), 4),
                    "target_qty": int(target["qty"]),
                    "score": target["score"],
                    **target["metadata"],
                },
            )
            action = f"Opened {target['mode'].replace('_', ' ')} overlay in {symbol} ({buy_qty} shares at ~${target['price']:.2f})."
            actions.append(action)
            logger.info(action)
            send_alert(f"📈 **EQUITY OVERLAY UPDATE**\n{action}", "INFO")
        except Exception as exc:
            logger.error("Failed entering %s overlay %s: %s", target["mode"], symbol, exc)

    return buying_power, actions
