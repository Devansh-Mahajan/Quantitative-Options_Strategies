from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import yfinance as yf
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config.params import (
    OPTION_DELAY_MAX_UNDERLYING_MOVE_PCT,
    OPTION_DELAY_MIN_PRICING_CONFIDENCE,
    OPTION_DELAY_QUOTE_HAIRCUT,
    OPTION_PRICING_RISK_FREE_RATE,
    OPTION_QUOTE_DELAY_MINUTES,
)
from core.utils import parse_option_symbol
from models.contract import Contract

try:
    from py_vollib.black_scholes import black_scholes as vollib_black_scholes
    from py_vollib.black_scholes.greeks.analytical import (
        delta as vollib_delta,
        gamma as vollib_gamma,
        theta as vollib_theta,
        vega as vollib_vega,
    )
    from py_vollib.black_scholes.implied_volatility import implied_volatility as vollib_implied_volatility

    HAS_PY_VOLLIB = True
except Exception:  # pragma: no cover - exercised only when py_vollib is absent
    HAS_PY_VOLLIB = False

logger = logging.getLogger(f"strategy.{__name__}")


@dataclass(frozen=True)
class UnderlyingQuoteContext:
    symbol: str
    current_price: float | None
    delayed_price: float | None
    current_timestamp: datetime | None = None
    delayed_timestamp: datetime | None = None
    source: str = "alpaca"


def effective_bid_price(contract: Contract | None) -> float:
    if contract is None:
        return 0.0
    return float(contract.fair_bid_price or contract.bid_price or 0.0)


def effective_ask_price(contract: Contract | None) -> float:
    if contract is None:
        return 0.0
    return float(contract.fair_ask_price or contract.ask_price or 0.0)


def effective_mid_price(contract: Contract | None) -> float:
    if contract is None:
        return 0.0
    if contract.fair_value is not None and contract.fair_value > 0:
        return float(contract.fair_value)
    bid = effective_bid_price(contract)
    ask = effective_ask_price(contract)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return float(contract.last_price or 0.0)


def build_delay_adjusted_contracts(
    client,
    option_contracts: Iterable[object],
    snapshots: dict | None = None,
    delay_minutes: int = OPTION_QUOTE_DELAY_MINUTES,
    risk_free_rate: float = OPTION_PRICING_RISK_FREE_RATE,
) -> list[Contract]:
    contracts = list(option_contracts or [])
    if not contracts:
        return []

    symbols = [_contract_symbol(contract) for contract in contracts]
    snapshots = snapshots or client.get_option_snapshot(symbols)
    underlyings = sorted({_contract_underlying(contract) for contract in contracts if _contract_underlying(contract)})
    stock_context = get_underlying_quote_context(client, underlyings, delay_minutes=delay_minutes)

    repriced_contracts: list[Contract] = []
    for raw_contract in contracts:
        symbol = _contract_symbol(raw_contract)
        snapshot = (snapshots or {}).get(symbol)
        if snapshot is None:
            continue

        contract = _contract_to_model(raw_contract, snapshot)
        contract = reprice_contract(
            contract,
            snapshot=snapshot,
            stock_context=stock_context.get(contract.underlying),
            risk_free_rate=risk_free_rate,
        )
        repriced_contracts.append(contract)
    return repriced_contracts


def get_underlying_quote_context(client, symbols: Iterable[str], delay_minutes: int = OPTION_QUOTE_DELAY_MINUTES) -> dict[str, UnderlyingQuoteContext]:
    requested = [symbol for symbol in symbols if symbol]
    if not requested:
        return {}

    contexts: dict[str, UnderlyingQuoteContext] = {
        symbol: UnderlyingQuoteContext(symbol=symbol, current_price=None, delayed_price=None)
        for symbol in requested
    }

    try:
        latest_trades = client.get_stock_latest_trade(requested)
        for symbol in requested:
            trade = (latest_trades or {}).get(symbol)
            if trade is None:
                continue
            contexts[symbol] = UnderlyingQuoteContext(
                symbol=symbol,
                current_price=_safe_float(getattr(trade, "price", None)),
                delayed_price=contexts[symbol].delayed_price,
                current_timestamp=_normalize_timestamp(getattr(trade, "timestamp", None)),
                delayed_timestamp=contexts[symbol].delayed_timestamp,
                source="alpaca",
            )
    except Exception as exc:
        logger.debug("Could not fetch current stock trades for delay-aware pricing: %s", exc)

    delayed_end = datetime.now(timezone.utc) - timedelta(minutes=max(1, delay_minutes))
    delayed_start = delayed_end - timedelta(minutes=10)
    try:
        bars = client.stock_client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=requested,
                timeframe=TimeFrame.Minute,
                start=delayed_start,
                end=delayed_end,
            )
        )
        frame = getattr(bars, "df", None)
        delayed_map = _extract_delayed_prices(frame)
        for symbol, (price, ts) in delayed_map.items():
            ctx = contexts.get(symbol)
            if ctx is None:
                continue
            contexts[symbol] = UnderlyingQuoteContext(
                symbol=symbol,
                current_price=ctx.current_price,
                delayed_price=price,
                current_timestamp=ctx.current_timestamp,
                delayed_timestamp=ts,
                source=ctx.source,
            )
    except Exception as exc:
        logger.debug("Could not fetch delayed stock bars for delay-aware pricing: %s", exc)

    missing = [
        symbol
        for symbol, ctx in contexts.items()
        if ctx.current_price is None or ctx.delayed_price is None
    ]
    if missing:
        fallback = _fetch_yfinance_quote_context(missing, delay_minutes=delay_minutes)
        for symbol, fallback_ctx in fallback.items():
            existing = contexts.get(symbol)
            if existing is None:
                contexts[symbol] = fallback_ctx
                continue
            contexts[symbol] = UnderlyingQuoteContext(
                symbol=symbol,
                current_price=existing.current_price if existing.current_price is not None else fallback_ctx.current_price,
                delayed_price=existing.delayed_price if existing.delayed_price is not None else fallback_ctx.delayed_price,
                current_timestamp=existing.current_timestamp or fallback_ctx.current_timestamp,
                delayed_timestamp=existing.delayed_timestamp or fallback_ctx.delayed_timestamp,
                source="alpaca+yfinance" if existing.current_price is not None or existing.delayed_price is not None else "yfinance",
            )

    for symbol, ctx in list(contexts.items()):
        if ctx.current_price is None and ctx.delayed_price is not None:
            contexts[symbol] = UnderlyingQuoteContext(
                symbol=symbol,
                current_price=ctx.delayed_price,
                delayed_price=ctx.delayed_price,
                current_timestamp=ctx.delayed_timestamp,
                delayed_timestamp=ctx.delayed_timestamp,
                source=ctx.source,
            )
        elif ctx.delayed_price is None and ctx.current_price is not None:
            contexts[symbol] = UnderlyingQuoteContext(
                symbol=symbol,
                current_price=ctx.current_price,
                delayed_price=ctx.current_price,
                current_timestamp=ctx.current_timestamp,
                delayed_timestamp=ctx.current_timestamp,
                source=ctx.source,
            )

    return contexts


def reprice_contract(
    contract: Contract,
    snapshot,
    stock_context: UnderlyingQuoteContext | None,
    risk_free_rate: float = OPTION_PRICING_RISK_FREE_RATE,
) -> Contract:
    contract.underlying_price = stock_context.current_price if stock_context else contract.underlying_price
    contract.delayed_underlying_price = stock_context.delayed_price if stock_context else contract.delayed_underlying_price
    contract.quote_age_minutes = _quote_age_minutes(snapshot)

    delayed_spot = _safe_float(contract.delayed_underlying_price)
    current_spot = _safe_float(contract.underlying_price)
    if delayed_spot is None and current_spot is not None:
        delayed_spot = current_spot
        contract.delayed_underlying_price = delayed_spot
    if current_spot is None and delayed_spot is not None:
        current_spot = delayed_spot
        contract.underlying_price = current_spot

    contract.staleness_pct = _calculate_staleness_pct(current_spot, delayed_spot)

    delayed_mark = _delayed_mark(contract)
    if not delayed_mark or not delayed_spot or not current_spot or not contract.strike:
        contract.pricing_confidence = _pricing_confidence(contract, solved_iv=False)
        return contract

    years_to_expiry = _years_to_expiry(contract.symbol, contract.dte)
    if years_to_expiry <= 0:
        contract.pricing_confidence = _pricing_confidence(contract, solved_iv=False)
        return contract

    flag = "c" if str(contract.contract_type).lower().startswith("call") else "p"
    snapshot_iv = _safe_float(getattr(snapshot, "implied_volatility", None))
    solved_iv = snapshot_iv if snapshot_iv and 0.0001 < snapshot_iv < 5.0 else infer_implied_volatility(
        price=delayed_mark,
        spot=delayed_spot,
        strike=float(contract.strike),
        years_to_expiry=years_to_expiry,
        risk_free_rate=risk_free_rate,
        flag=flag,
    )

    if solved_iv is None:
        contract.pricing_confidence = _pricing_confidence(contract, solved_iv=False)
        return contract

    contract.implied_volatility = solved_iv
    fair_mid = option_price(flag, current_spot, float(contract.strike), years_to_expiry, risk_free_rate, solved_iv)
    fair_bid, fair_ask = _widen_fair_quote(
        fair_mid=fair_mid,
        delayed_bid=_safe_float(contract.bid_price),
        delayed_ask=_safe_float(contract.ask_price),
        staleness_pct=contract.staleness_pct or 0.0,
    )
    greeks = option_greeks(flag, current_spot, float(contract.strike), years_to_expiry, risk_free_rate, solved_iv)

    contract.fair_value = round(fair_mid, 4)
    contract.fair_bid_price = round(fair_bid, 4)
    contract.fair_ask_price = round(fair_ask, 4)
    contract.delta = greeks["delta"]
    contract.gamma = greeks["gamma"]
    contract.theta = greeks["theta"]
    contract.vega = greeks["vega"]
    contract.pricing_confidence = _pricing_confidence(contract, solved_iv=True)
    return contract


def infer_implied_volatility(
    price: float,
    spot: float,
    strike: float,
    years_to_expiry: float,
    risk_free_rate: float,
    flag: str,
) -> float | None:
    if price <= 0 or spot <= 0 or strike <= 0 or years_to_expiry <= 0:
        return None

    intrinsic = max(0.0, spot - strike) if flag == "c" else max(0.0, strike - spot)
    upper_bound = spot if flag == "c" else strike * math.exp(-risk_free_rate * years_to_expiry)
    lower_bound = intrinsic + 1e-6
    capped_price = min(max(price, lower_bound), max(lower_bound + 1e-6, upper_bound - 1e-6))

    if HAS_PY_VOLLIB:
        try:
            return float(vollib_implied_volatility(capped_price, spot, strike, years_to_expiry, risk_free_rate, flag))
        except Exception:
            pass

    low_sigma = 1e-4
    high_sigma = 5.0
    low_price = option_price(flag, spot, strike, years_to_expiry, risk_free_rate, low_sigma)
    high_price = option_price(flag, spot, strike, years_to_expiry, risk_free_rate, high_sigma)
    if capped_price < low_price or capped_price > high_price:
        return None

    for _ in range(60):
        mid_sigma = (low_sigma + high_sigma) / 2.0
        mid_price = option_price(flag, spot, strike, years_to_expiry, risk_free_rate, mid_sigma)
        if abs(mid_price - capped_price) <= 1e-5:
            return mid_sigma
        if mid_price < capped_price:
            low_sigma = mid_sigma
        else:
            high_sigma = mid_sigma
    return (low_sigma + high_sigma) / 2.0


def option_price(flag: str, spot: float, strike: float, years_to_expiry: float, risk_free_rate: float, volatility: float) -> float:
    if spot <= 0 or strike <= 0 or years_to_expiry <= 0 or volatility <= 0:
        intrinsic = max(0.0, spot - strike) if flag == "c" else max(0.0, strike - spot)
        return intrinsic

    if HAS_PY_VOLLIB:
        try:
            return float(vollib_black_scholes(flag, spot, strike, years_to_expiry, risk_free_rate, volatility))
        except Exception:
            pass

    d1, d2 = _d1_d2(spot, strike, years_to_expiry, risk_free_rate, volatility)
    if flag == "c":
        return (spot * _norm_cdf(d1)) - (strike * math.exp(-risk_free_rate * years_to_expiry) * _norm_cdf(d2))
    return (strike * math.exp(-risk_free_rate * years_to_expiry) * _norm_cdf(-d2)) - (spot * _norm_cdf(-d1))


def option_greeks(flag: str, spot: float, strike: float, years_to_expiry: float, risk_free_rate: float, volatility: float) -> dict[str, float]:
    if spot <= 0 or strike <= 0 or years_to_expiry <= 0 or volatility <= 0:
        intrinsic_delta = 1.0 if flag == "c" and spot > strike else -1.0 if flag == "p" and spot < strike else 0.0
        return {"delta": intrinsic_delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    if HAS_PY_VOLLIB:
        try:
            return {
                "delta": float(vollib_delta(flag, spot, strike, years_to_expiry, risk_free_rate, volatility)),
                "gamma": float(vollib_gamma(flag, spot, strike, years_to_expiry, risk_free_rate, volatility)),
                "theta": float(vollib_theta(flag, spot, strike, years_to_expiry, risk_free_rate, volatility)),
                "vega": float(vollib_vega(flag, spot, strike, years_to_expiry, risk_free_rate, volatility)),
            }
        except Exception:
            pass

    d1, d2 = _d1_d2(spot, strike, years_to_expiry, risk_free_rate, volatility)
    pdf = _norm_pdf(d1)
    gamma = pdf / (spot * volatility * math.sqrt(years_to_expiry))
    vega = spot * pdf * math.sqrt(years_to_expiry) * 0.01
    if flag == "c":
        delta = _norm_cdf(d1)
        theta = (
            -spot * pdf * volatility / (2.0 * math.sqrt(years_to_expiry))
            - risk_free_rate * strike * math.exp(-risk_free_rate * years_to_expiry) * _norm_cdf(d2)
        ) / 365.0
    else:
        delta = _norm_cdf(d1) - 1.0
        theta = (
            -spot * pdf * volatility / (2.0 * math.sqrt(years_to_expiry))
            + risk_free_rate * strike * math.exp(-risk_free_rate * years_to_expiry) * _norm_cdf(-d2)
        ) / 365.0
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def _contract_to_model(raw_contract, snapshot) -> Contract:
    symbol = _contract_symbol(raw_contract)
    contract_type = _contract_type(raw_contract)
    return Contract(
        underlying=_contract_underlying(raw_contract),
        symbol=symbol,
        contract_type=contract_type,
        dte=_contract_dte(raw_contract),
        strike=_contract_strike(raw_contract),
        delta=getattr(getattr(snapshot, "greeks", None), "delta", None),
        gamma=getattr(getattr(snapshot, "greeks", None), "gamma", None),
        theta=getattr(getattr(snapshot, "greeks", None), "theta", None),
        vega=getattr(getattr(snapshot, "greeks", None), "vega", None),
        bid_price=getattr(getattr(snapshot, "latest_quote", None), "bid_price", None),
        ask_price=getattr(getattr(snapshot, "latest_quote", None), "ask_price", None),
        last_price=getattr(getattr(snapshot, "latest_trade", None), "price", None),
        oi=_contract_open_interest(raw_contract),
        implied_volatility=getattr(snapshot, "implied_volatility", None),
    )


def _contract_symbol(contract) -> str:
    return str(getattr(contract, "symbol"))


def _contract_underlying(contract) -> str:
    underlying = getattr(contract, "underlying_symbol", None) or getattr(contract, "underlying", None)
    if underlying:
        return str(underlying)
    return parse_option_symbol(_contract_symbol(contract))[0]


def _contract_type(contract) -> str:
    raw_type = getattr(contract, "type", None) or getattr(contract, "contract_type", None)
    if raw_type is None:
        return "call" if parse_option_symbol(_contract_symbol(contract))[1] == "C" else "put"
    if hasattr(raw_type, "value"):
        raw_type = raw_type.value
    raw_text = str(raw_type).lower()
    if "call" in raw_text:
        return "call"
    if "put" in raw_text:
        return "put"
    return "call" if parse_option_symbol(_contract_symbol(contract))[1] == "C" else "put"


def _contract_strike(contract) -> float | None:
    strike = getattr(contract, "strike_price", None) or getattr(contract, "strike", None)
    if strike is not None:
        return float(strike)
    return float(parse_option_symbol(_contract_symbol(contract))[2])


def _contract_open_interest(contract) -> float | None:
    oi = getattr(contract, "open_interest", None) or getattr(contract, "oi", None)
    return float(oi) if oi is not None else None


def _contract_dte(contract) -> float | None:
    raw_dte = getattr(contract, "dte", None)
    if raw_dte is not None:
        return float(raw_dte)
    expiry = _expiry_date(_contract_symbol(contract))
    return float((expiry - date.today()).days)


def _expiry_date(symbol: str) -> date:
    digits = "".join(ch for ch in symbol if ch.isdigit())
    expiry_digits = digits[:6]
    return datetime.strptime(expiry_digits, "%y%m%d").date()


def _quote_age_minutes(snapshot) -> float | None:
    quote_ts = _normalize_timestamp(getattr(getattr(snapshot, "latest_quote", None), "timestamp", None))
    if quote_ts is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - quote_ts).total_seconds() / 60.0)


def _normalize_timestamp(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if hasattr(value, "to_pydatetime"):
        return _normalize_timestamp(value.to_pydatetime())
    return None


def _extract_delayed_prices(frame) -> dict[str, tuple[float, datetime | None]]:
    if frame is None or getattr(frame, "empty", True):
        return {}

    delayed: dict[str, tuple[float, datetime | None]] = {}
    if getattr(frame.index, "nlevels", 1) > 1:
        for symbol in frame.index.get_level_values(0).unique():
            symbol_frame = frame.xs(symbol)
            if symbol_frame.empty:
                continue
            last_row = symbol_frame.iloc[-1]
            delayed[str(symbol)] = (_safe_float(last_row.get("close")) or 0.0, _normalize_timestamp(symbol_frame.index[-1]))
    else:
        if "close" in frame.columns and not frame.empty:
            delayed_value = _safe_float(frame.iloc[-1]["close"])
            if delayed_value is not None:
                delayed[""] = (delayed_value, _normalize_timestamp(frame.index[-1]))
    return {symbol: payload for symbol, payload in delayed.items() if payload[0] > 0}


def _fetch_yfinance_quote_context(symbols: Iterable[str], delay_minutes: int) -> dict[str, UnderlyingQuoteContext]:
    out: dict[str, UnderlyingQuoteContext] = {}
    for symbol in symbols:
        try:
            history = yf.Ticker(symbol).history(period="2d", interval="1m")
            if history.empty:
                continue
            history = history.dropna(subset=["Close"])
            if history.empty:
                continue
            current_row = history.iloc[-1]
            delayed_cutoff = history.index[-1] - timedelta(minutes=max(1, delay_minutes))
            delayed_rows = history.loc[history.index <= delayed_cutoff]
            delayed_row = delayed_rows.iloc[-1] if not delayed_rows.empty else history.iloc[max(0, len(history) - delay_minutes - 1)]
            out[symbol] = UnderlyingQuoteContext(
                symbol=symbol,
                current_price=_safe_float(current_row["Close"]),
                delayed_price=_safe_float(delayed_row["Close"]),
                current_timestamp=_normalize_timestamp(history.index[-1]),
                delayed_timestamp=_normalize_timestamp(delayed_row.name),
                source="yfinance",
            )
        except Exception as exc:
            logger.debug("yfinance fallback failed for %s: %s", symbol, exc)
    return out


def _widen_fair_quote(
    fair_mid: float,
    delayed_bid: float | None,
    delayed_ask: float | None,
    staleness_pct: float,
) -> tuple[float, float]:
    delayed_bid = delayed_bid or 0.0
    delayed_ask = delayed_ask or 0.0
    delayed_mid = ((delayed_bid + delayed_ask) / 2.0) if delayed_bid > 0 and delayed_ask > 0 else 0.0
    delayed_width = max(0.01, delayed_ask - delayed_bid) if delayed_ask > 0 and delayed_bid >= 0 else max(0.02, fair_mid * 0.12)
    width_ratio = delayed_width / max(delayed_mid, 0.05) if delayed_mid > 0 else 0.18
    width_ratio = min(max(width_ratio, 0.03), 1.5)

    spread_multiplier = 1.0 + min(1.5, staleness_pct * 25.0) + float(OPTION_DELAY_QUOTE_HAIRCUT)
    adjusted_width = max(0.02, fair_mid * width_ratio * spread_multiplier)
    fair_bid = max(0.01, fair_mid - adjusted_width / 2.0)
    fair_ask = max(fair_bid + 0.01, fair_mid + adjusted_width / 2.0)
    return fair_bid, fair_ask


def _delayed_mark(contract: Contract) -> float:
    bid = _safe_float(contract.bid_price)
    ask = _safe_float(contract.ask_price)
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if ask is not None and ask > 0:
        return ask
    if bid is not None and bid > 0:
        return bid
    return float(contract.last_price or 0.0)


def _pricing_confidence(contract: Contract, solved_iv: bool) -> float:
    bid = _safe_float(contract.bid_price) or 0.0
    ask = _safe_float(contract.ask_price) or 0.0
    delayed_mid = max((_delayed_mark(contract) or 0.0), 0.05)
    spread_ratio = max(0.0, (ask - bid) / delayed_mid) if ask > 0 and bid >= 0 else 1.0
    staleness_penalty = min(0.60, float(contract.staleness_pct or 0.0) * 12.0)
    age_penalty = 0.0
    if contract.quote_age_minutes is not None:
        age_penalty = min(0.20, max(0.0, contract.quote_age_minutes - OPTION_QUOTE_DELAY_MINUTES) / 60.0)
    iv_penalty = 0.0 if solved_iv else 0.28
    confidence = 1.0 - min(0.95, (spread_ratio * 0.35) + staleness_penalty + age_penalty + iv_penalty)
    return round(max(0.05, min(1.0, confidence)), 4)


def _calculate_staleness_pct(current_price: float | None, delayed_price: float | None) -> float | None:
    if current_price is None or delayed_price is None or delayed_price <= 0:
        return None
    return abs(current_price - delayed_price) / delayed_price


def _years_to_expiry(symbol: str, dte: float | None) -> float:
    if dte is not None:
        return max(1.0 / (365.0 * 24.0), float(dte) / 365.0)
    expiry = _expiry_date(symbol)
    return max(1.0 / (365.0 * 24.0), (expiry - date.today()).days / 365.0)


def _d1_d2(spot: float, strike: float, years_to_expiry: float, risk_free_rate: float, volatility: float) -> tuple[float, float]:
    variance_term = volatility * math.sqrt(years_to_expiry)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility * volatility) * years_to_expiry) / variance_term
    d2 = d1 - variance_term
    return d1, d2


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _norm_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric
