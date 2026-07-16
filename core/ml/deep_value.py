"""
Deep-value (Graham net-net) scanner for the options/Alpaca equity stack.

Finds stocks trading below net current asset value (NCAV) — i.e. the market
prices the company below what its liquid assets minus ALL liabilities would
fetch — sourced nightly from Yahoo's screener (day gainers/losers, most
actives, small-cap screens). Companion execution sleeve lives in
core/execution/deep_value_sleeve.py; nightly CLI in
scripts/nightly_deep_value_scan.py; weekend ML trainer in
scripts/train_deep_value_model.py.

Design: all valuation math, filters, and scoring are pure functions over plain
dicts/dataclasses so tests never need the network. Yahoo access is confined to
the two thin fetch wrappers at the bottom.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from config.params import (
    DEEP_VALUE_EXCLUDED_SECTORS,
    DEEP_VALUE_MARGIN_OF_SAFETY,
    DEEP_VALUE_MIN_DOLLAR_VOLUME,
    DEEP_VALUE_MIN_MARKET_CAP,
    DEEP_VALUE_MIN_PRICE,
    DEEP_VALUE_MIN_RUNWAY_QUARTERS,
)
from core.risk.universe_maintenance import dedupe_symbols, resolve_download_symbol

logger = logging.getLogger(f"strategy.{__name__}")

_ROOT = Path(__file__).resolve().parents[2]
SCAN_SNAPSHOT_PATH = _ROOT / ".runtime" / "deep_value_scan.json"
TRAINING_HISTORY_PATH = _ROOT / ".runtime" / "deep_value_history.jsonl"
MODEL_PATH = _ROOT / "config" / "deep_value_model.pkl"

SCREENS = ["day_losers", "day_gainers", "most_actives", "small_cap_gainers", "aggressive_small_caps"]

# Liquidation haircuts: what each current-asset class realistically fetches in a
# wind-down (Graham's classic discounts).
HAIRCUT_CASH = 1.00
HAIRCUT_RECEIVABLES = 0.75
HAIRCUT_INVENTORY = 0.50
HAIRCUT_OTHER_CURRENT = 0.25

# Feature order shared by the live scorer and scripts/train_deep_value_model.py
# — keep in sync with feature_vector() below.
FEATURE_NAMES = [
    "price_to_ncav",
    "liquidation_coverage",
    "net_cash_coverage",
    "burn_runway_quarters",
    "log_market_cap",
    "log_dollar_volume",
]


@dataclass
class ValueMetrics:
    ncav_per_share: float
    liquidation_per_share: float
    net_cash_per_share: float
    burn_runway_quarters: float  # math.inf when FCF-positive
    price_to_ncav: float


@dataclass
class DeepValueCandidate:
    symbol: str
    price: float
    ncav_per_share: float
    liquidation_per_share: float
    net_cash_per_share: float
    burn_runway_quarters: float
    price_to_ncav: float
    market_cap: float
    avg_dollar_volume: float
    sector: str
    score: float
    model_probability: float | None
    failed_gates: list[str]
    as_of: str


# --------------------------------------------------------------------------- #
# Pure valuation math
# --------------------------------------------------------------------------- #

def _f(value, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def compute_value_metrics(fundamentals: dict) -> ValueMetrics | None:
    """
    Compute NCAV/liquidation value per share from a flat fundamentals dict.

    Required keys: price, shares_outstanding, current_assets, total_liabilities.
    Optional: cash, receivables, inventory, free_cash_flow.
    Returns None when the required inputs are missing/invalid — callers skip
    the ticker rather than guessing.
    """
    price = _f(fundamentals.get("price"))
    shares = _f(fundamentals.get("shares_outstanding"))
    current_assets = _f(fundamentals.get("current_assets"))
    total_liabilities = _f(fundamentals.get("total_liabilities"))
    if not price or price <= 0 or not shares or shares <= 0:
        return None
    if current_assets is None or total_liabilities is None:
        return None

    cash = _f(fundamentals.get("cash"), 0.0)
    receivables = _f(fundamentals.get("receivables"), 0.0)
    inventory = _f(fundamentals.get("inventory"), 0.0)
    other_current = max(0.0, current_assets - cash - receivables - inventory)

    ncav = current_assets - total_liabilities
    liquidation = (
        cash * HAIRCUT_CASH
        + receivables * HAIRCUT_RECEIVABLES
        + inventory * HAIRCUT_INVENTORY
        + other_current * HAIRCUT_OTHER_CURRENT
        - total_liabilities
    )
    net_cash = cash - total_liabilities

    fcf = _f(fundamentals.get("free_cash_flow"))
    if fcf is None or fcf >= 0:
        runway = math.inf
    else:
        quarterly_burn = abs(fcf) / 4.0
        runway = cash / quarterly_burn if quarterly_burn > 0 else math.inf

    ncav_per_share = ncav / shares
    return ValueMetrics(
        ncav_per_share=ncav_per_share,
        liquidation_per_share=liquidation / shares,
        net_cash_per_share=net_cash / shares,
        burn_runway_quarters=runway,
        price_to_ncav=price / ncav_per_share if ncav_per_share > 0 else math.inf,
    )


def passes_quality_filters(metrics: ValueMetrics, quote: dict) -> tuple[bool, list[str]]:
    """
    Gate a candidate on the classic value-trap filters. Returns
    (passed, failed_gate_names) so the scan report shows WHY a name was
    rejected — daily-loser lists are full of stocks that are cheap for a
    reason, and the sleeve's edge depends on these gates as much as on NCAV.
    """
    failed: list[str] = []
    price = _f(quote.get("price"), 0.0)
    market_cap = _f(quote.get("market_cap"), 0.0)
    dollar_volume = _f(quote.get("avg_dollar_volume"), 0.0)
    sector = str(quote.get("sector") or "").strip()

    if metrics.ncav_per_share <= 0:
        failed.append("negative_ncav")
    elif price >= DEEP_VALUE_MARGIN_OF_SAFETY * metrics.ncav_per_share:
        failed.append("insufficient_margin_of_safety")
    elif metrics.price_to_ncav < 0.05:
        # A >95% discount to NCAV is essentially never a real opportunity —
        # it's bad data (share-count unit mismatch) or a fraud in progress.
        failed.append("implausible_discount")

    # Reconcile the balance-sheet share count against market cap: if
    # price x shares disagrees with the quoted market cap by >2.5x, the
    # share count (and therefore NCAV/share) can't be trusted.
    shares = _f(quote.get("shares_outstanding"), 0.0)
    if market_cap > 0 and shares > 0 and price > 0:
        implied = price * shares
        ratio = implied / market_cap
        if ratio > 2.5 or ratio < 0.4:
            failed.append("share_count_mismatch")
    if price < DEEP_VALUE_MIN_PRICE:
        failed.append("penny_stock")
    if market_cap < DEEP_VALUE_MIN_MARKET_CAP:
        failed.append("market_cap_floor")
    if dollar_volume < DEEP_VALUE_MIN_DOLLAR_VOLUME:
        failed.append("illiquid")
    if sector in DEEP_VALUE_EXCLUDED_SECTORS:
        failed.append("excluded_sector")
    if metrics.burn_runway_quarters < DEEP_VALUE_MIN_RUNWAY_QUARTERS:
        failed.append("cash_burn_runway")

    return (not failed, failed)


def heuristic_score(metrics: ValueMetrics, quote: dict) -> float:
    """
    0-1 asymmetry score. Dominated by the discount to NCAV (the margin of
    safety IS the edge); liquidation coverage, burn runway, and liquidity are
    secondary robustness terms.
    """
    price = _f(quote.get("price"), 0.0)
    if not price or price <= 0 or metrics.ncav_per_share <= 0:
        return 0.0

    upside = metrics.ncav_per_share / price - 1.0                # 0 at par, 1 at 50% discount
    discount_term = max(0.0, min(1.0, upside))
    liquidation_term = max(0.0, min(1.0, metrics.liquidation_per_share / price))
    runway = metrics.burn_runway_quarters
    runway_term = 1.0 if math.isinf(runway) else max(0.0, min(1.0, runway / 8.0))
    dollar_volume = _f(quote.get("avg_dollar_volume"), 0.0) or 0.0
    liquidity_term = max(0.0, min(1.0, dollar_volume / 20e6))

    return round(
        0.45 * discount_term + 0.25 * liquidation_term + 0.20 * runway_term + 0.10 * liquidity_term,
        4,
    )


def feature_vector(metrics: ValueMetrics, quote: dict) -> list[float]:
    """Model features — order must match FEATURE_NAMES (trainer + scorer)."""
    market_cap = _f(quote.get("market_cap"), 0.0) or 0.0
    dollar_volume = _f(quote.get("avg_dollar_volume"), 0.0) or 0.0
    price = _f(quote.get("price"), 0.0) or 0.0
    runway = metrics.burn_runway_quarters
    return [
        min(metrics.price_to_ncav, 10.0),
        metrics.liquidation_per_share / price if price > 0 else 0.0,
        metrics.net_cash_per_share / price if price > 0 else 0.0,
        min(runway, 40.0) if not math.isinf(runway) else 40.0,
        math.log10(max(market_cap, 1.0)),
        math.log10(max(dollar_volume, 1.0)),
    ]


# --------------------------------------------------------------------------- #
# Optional ML scorer (weekend-trained; heuristic-only until a model exists)
# --------------------------------------------------------------------------- #

_MODEL_CACHE: dict = {"loaded": False, "payload": None}


def _load_model_payload() -> dict | None:
    if _MODEL_CACHE["loaded"]:
        return _MODEL_CACHE["payload"]
    _MODEL_CACHE["loaded"] = True
    try:
        import joblib

        payload = joblib.load(MODEL_PATH)
        if payload.get("feature_names") != FEATURE_NAMES:
            logger.warning("deep_value model feature mismatch — ignoring stale model at %s", MODEL_PATH)
            payload = None
        _MODEL_CACHE["payload"] = payload
    except Exception:
        _MODEL_CACHE["payload"] = None
    return _MODEL_CACHE["payload"]


def model_probability(metrics: ValueMetrics, quote: dict) -> float | None:
    payload = _load_model_payload()
    if not payload:
        return None
    try:
        proba = payload["model"].predict_proba([feature_vector(metrics, quote)])[0][1]
        return round(float(proba), 4)
    except Exception as exc:
        logger.warning("deep_value model inference failed: %s", exc)
        return None


def build_candidate(symbol: str, fundamentals: dict) -> DeepValueCandidate | None:
    """Metrics + filters + scores for one ticker's fundamentals dict."""
    metrics = compute_value_metrics(fundamentals)
    if metrics is None:
        return None
    passed, failed_gates = passes_quality_filters(metrics, fundamentals)
    score = heuristic_score(metrics, fundamentals) if passed else 0.0
    proba = model_probability(metrics, fundamentals) if passed else None
    return DeepValueCandidate(
        symbol=symbol.upper(),
        price=_f(fundamentals.get("price"), 0.0) or 0.0,
        ncav_per_share=round(metrics.ncav_per_share, 4),
        liquidation_per_share=round(metrics.liquidation_per_share, 4),
        net_cash_per_share=round(metrics.net_cash_per_share, 4),
        burn_runway_quarters=(-1.0 if math.isinf(metrics.burn_runway_quarters) else round(metrics.burn_runway_quarters, 2)),
        price_to_ncav=round(min(metrics.price_to_ncav, 999.0), 4),
        market_cap=_f(fundamentals.get("market_cap"), 0.0) or 0.0,
        avg_dollar_volume=_f(fundamentals.get("avg_dollar_volume"), 0.0) or 0.0,
        sector=str(fundamentals.get("sector") or ""),
        score=score,
        model_probability=proba,
        failed_gates=failed_gates,
        as_of=str(fundamentals.get("as_of") or ""),
    )


# --------------------------------------------------------------------------- #
# Snapshot persistence (nightly scan → next-day strategy cycle handoff)
# --------------------------------------------------------------------------- #

def save_scan_snapshot(candidates: list[DeepValueCandidate], path: Path = SCAN_SNAPSHOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_scan_snapshot(max_age_hours: float, path: Path = SCAN_SNAPSHOT_PATH) -> list[dict] | None:
    """Age-gated read; None when missing/stale/corrupt so callers skip entries."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        generated = datetime.fromisoformat(str(payload.get("generated_at_utc")))
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - generated.astimezone(timezone.utc)).total_seconds() / 3600.0
        if age_hours > max(0.25, float(max_age_hours)):
            return None
        return list(payload.get("candidates", []))
    except Exception:
        return None


def append_training_snapshot(rows: list[dict], path: Path = TRAINING_HISTORY_PATH) -> None:
    """One JSON line per scanned ticker per night — matures into training data."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"recorded_at_utc": stamp, **row}) + "\n")


# --------------------------------------------------------------------------- #
# Thin network wrappers (the ONLY functions here that touch Yahoo)
# --------------------------------------------------------------------------- #

def fetch_screener_candidates(limit_per_screen: int = 50) -> list[dict]:
    """
    Pull quote rows from Yahoo's predefined screens plus a custom
    price-to-book < 1 query. A failing screen is skipped, not fatal; returns
    deduped rows keyed by symbol with the cheap pre-filter fields populated.
    """
    import yfinance as yf

    quotes: dict[str, dict] = {}
    screens_ok = 0

    def _ingest(result: dict) -> None:
        for quote in result.get("quotes", []) or []:
            symbol = str(quote.get("symbol") or "").strip().upper()
            if not symbol or symbol in quotes:
                continue
            price = _f(quote.get("regularMarketPrice"))
            volume = _f(quote.get("averageDailyVolume3Month")) or _f(quote.get("regularMarketVolume")) or 0.0
            quotes[symbol] = {
                "symbol": symbol,
                "price": price,
                "market_cap": _f(quote.get("marketCap"), 0.0),
                "avg_dollar_volume": (price or 0.0) * volume,
                "change_percent": _f(quote.get("regularMarketChangePercent"), 0.0),
            }

    for screen_name in SCREENS:
        try:
            _ingest(yf.screen(screen_name, count=limit_per_screen))
            screens_ok += 1
        except Exception as exc:
            logger.warning("deep_value screen '%s' failed: %s", screen_name, exc)

    try:
        from yfinance import EquityQuery

        low_pb = EquityQuery(
            "and",
            [
                EquityQuery("lt", ["pricebookratio.quarterly", 1.0]),
                EquityQuery("gt", ["pricebookratio.quarterly", 0.0]),
                EquityQuery("eq", ["region", "us"]),
            ],
        )
        _ingest(yf.screen(low_pb, size=limit_per_screen, sortField="pricebookratio.quarterly", sortAsc=True))
        screens_ok += 1
    except Exception as exc:
        logger.warning("deep_value custom low-P/B screen failed: %s", exc)

    if screens_ok == 0:
        raise RuntimeError("all deep-value screens failed — Yahoo screener unavailable")

    ordered = dedupe_symbols(quotes.keys())
    return [quotes[symbol] for symbol in ordered]


def fetch_fundamentals(symbol: str, quote: dict | None = None) -> dict | None:
    """
    Balance-sheet + profile fields for one ticker. Returns the flat dict
    consumed by compute_value_metrics, or None when critical fields are
    missing (common for micro caps — skip, don't guess).
    """
    import yfinance as yf

    def _row(frame, labels: list[str]) -> float | None:
        for label in labels:
            if label in frame.index:
                series = frame.loc[label].dropna()
                if len(series):
                    return _f(series.iloc[0])
        return None

    try:
        ticker = yf.Ticker(resolve_download_symbol(symbol))
        sheet = ticker.quarterly_balance_sheet
        if sheet is None or sheet.empty:
            sheet = ticker.balance_sheet
        if sheet is None or sheet.empty:
            return None
        info = ticker.info or {}
    except Exception as exc:
        logger.debug("deep_value fundamentals fetch failed for %s: %s", symbol, exc)
        return None

    current_assets = _row(sheet, ["Current Assets", "Total Current Assets"])
    total_liabilities = _row(sheet, ["Total Liabilities Net Minority Interest", "Total Liab", "Total Liabilities"])
    shares = _row(sheet, ["Ordinary Shares Number", "Share Issued"]) or _f(info.get("sharesOutstanding"))
    if current_assets is None or total_liabilities is None or not shares:
        return None

    quote = quote or {}
    price = _f(quote.get("price")) or _f(info.get("regularMarketPrice")) or _f(info.get("currentPrice"))
    avg_volume = _f(info.get("averageVolume"), 0.0) or 0.0
    return {
        "symbol": symbol.upper(),
        "price": price,
        "shares_outstanding": shares,
        "current_assets": current_assets,
        "total_liabilities": total_liabilities,
        "cash": _row(sheet, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"]),
        "receivables": _row(sheet, ["Accounts Receivable", "Receivables"]),
        "inventory": _row(sheet, ["Inventory"]),
        "free_cash_flow": _f(info.get("freeCashflow")) if _f(info.get("freeCashflow")) is not None else _f(info.get("operatingCashflow")),
        "market_cap": _f(info.get("marketCap")) or _f(quote.get("market_cap"), 0.0),
        "avg_dollar_volume": _f(quote.get("avg_dollar_volume")) or (price or 0.0) * avg_volume,
        "sector": str(info.get("sector") or ""),
        "as_of": str(sheet.columns[0].date()) if len(sheet.columns) else "",
    }
