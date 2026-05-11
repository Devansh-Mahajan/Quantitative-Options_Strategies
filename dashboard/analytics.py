from __future__ import annotations

import importlib
import inspect
import itertools
import json
import math
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bot import state as bot_state
from bot.config import cfg
from core.universe_maintenance import download_close_matrix

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / ".runtime"
REPORTS_DIR = ROOT / "reports"
LATEST_BACKTEST_REPORT = REPORTS_DIR / "latest_backtest_report.json"
STOCK_UNIVERSE_RESEARCH = REPORTS_DIR / "stock_universe_research.json"
UNIVERSE_REPORT = REPORTS_DIR / "universe_validation_report.json"
RISK_SNAPSHOT_PATH = RUNTIME_DIR / "risk_snapshot.json"
PORTFOLIO_GUARD_PATH = RUNTIME_DIR / "portfolio_risk_guard.json"
EXECUTION_LEDGER_PATH = RUNTIME_DIR / "execution_ledger.json"
EXECUTION_SUMMARY_PATH = RUNTIME_DIR / "execution_quality_snapshot.json"
SYSTEM_RESOURCE_PATH = RUNTIME_DIR / "system_resource_snapshot.json"
BOT_STATE_DB_PATH = RUNTIME_DIR / "bot_state.db"
CORRELATION_CACHE_PATH = RUNTIME_DIR / "dashboard_correlation_cache.json"
RESEARCH_DESK_CACHE_PATH = RUNTIME_DIR / "dashboard_research_desk.json"

OPTION_MULTIPLIER = 100.0
DEFAULT_MC_PATHS = 2500
DEFAULT_FORECAST_DAYS = 30
DEFAULT_OPTION_CHAIN_UNDERLYINGS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]
STOCK_CORRELATION_PERIOD = "1y"
STOCK_CORRELATION_CACHE_TTL_SECONDS = 4 * 60 * 60
CRYPTO_CORRELATION_CACHE_TTL_SECONDS = 30 * 60
CRYPTO_CORRELATION_LOOKBACK_DAYS = 90
RESEARCH_DESK_CACHE_TTL_SECONDS = 30 * 60
BROKER_LIVE_CACHE_TTL_SECONDS = 5

_BROKER_POSITION_CACHE: dict[str, Any] = {"generated_at": None, "payload": ([], {"available": False, "reason": "cold_cache"})}
_BROKER_SNAPSHOT_CACHE: dict[str, Any] = {"generated_at": None, "payload": {"available": False, "reason": "cold_cache"}}
_LIVE_OPTION_CACHE: dict[str, Any] = {"generated_at": None, "payload": ([], {"available": False, "reason": "cold_cache"})}

STRATEGY_CLASS_MAP: dict[str, tuple[str, str]] = {
    "momentum": ("strategies.momentum", "MomentumStrategy"),
    "mean_reversion": ("strategies.mean_reversion", "MeanReversionStrategy"),
    "breakout": ("strategies.breakout", "BreakoutStrategy"),
    "statistical_arb": ("strategies.statistical_arb", "StatisticalArbStrategy"),
    "cross_sectional_momentum": ("strategies.cross_sectional_momentum", "CrossSectionalMomentumStrategy"),
    "tsmom": ("strategies.tsmom", "TSMomentumStrategy"),
    "quant_factors": ("strategies.quant_factors", "QuantFactorsStrategy"),
    "contrarian_oi": ("strategies.contrarian_oi", "ContrarianOIStrategy"),
    "rma_strategy": ("strategies.rma_strategy", "RMAStrategy"),
    "vpin_flow": ("strategies.vpin_flow", "VPINFlowStrategy"),
    "knn_predictor": ("strategies.knn_predictor", "KNNPredictorStrategy"),
    "pivot_sr": ("strategies.pivot_sr", "PivotSRStrategy"),
    "hp_trend": ("strategies.hp_trend", "HPTrendStrategy"),
    "carry_portfolio": ("strategies.carry_portfolio", "CarryPortfolioStrategy"),
    "momentum_carry_combo": ("strategies.momentum_carry_combo", "MomentumCarryComboStrategy"),
    "order_flow": ("strategies.order_flow", "OrderFlowStrategy"),
    "liquidation_cascade": ("strategies.liquidation_cascade", "LiquidationCascadeStrategy"),
    "microstructure_pressure": ("strategies.microstructure_pressure", "MicrostructurePressureStrategy"),
    "pullback_confluence": ("strategies.pullback_confluence", "PullbackConfluenceStrategy"),
}

STRATEGY_RESEARCH_LIBRARY: dict[str, dict[str, Any]] = {
    "momentum": {
        "label": "Momentum",
        "paper_family": "Kakushadze & Serur 2018",
        "category": "trend",
        "asset_class": "multi-asset",
        "thesis": "Classic directional trend capture.",
    },
    "mean_reversion": {
        "label": "Mean Reversion",
        "paper_family": "Kakushadze & Serur 2018",
        "category": "reversion",
        "asset_class": "multi-asset",
        "thesis": "Fade statistically stretched prices.",
    },
    "breakout": {
        "label": "Breakout",
        "paper_family": "Bloch 2023 Futuretesting",
        "category": "pattern",
        "asset_class": "multi-asset",
        "thesis": "Ride directional expansion through range breaks.",
    },
    "pullback_confluence": {
        "label": "Pullback Confluence",
        "paper_family": "Bloch 2023 Futuretesting",
        "category": "pattern",
        "asset_class": "multi-asset",
        "thesis": "Wait for trend pullbacks and confluence-based resumptions.",
    },
    "pairs_arb": {
        "label": "Pairs Arb",
        "paper_family": "Kakushadze & Serur 2018",
        "category": "relative value",
        "asset_class": "stocks+crypto",
        "thesis": "Exploit spread dislocations in correlated assets.",
    },
    "statistical_arb": {
        "label": "Statistical Arbitrage",
        "paper_family": "Kakushadze & Serur 2018",
        "category": "relative value",
        "asset_class": "multi-asset",
        "thesis": "Mean-revert z-scored residuals and spreads.",
    },
    "cross_sectional_momentum": {
        "label": "Cross-Sectional Momentum",
        "paper_family": "Kakushadze & Serur 2018",
        "category": "factors",
        "asset_class": "stocks+crypto",
        "thesis": "Rank the universe and rotate into relative winners.",
    },
    "tsmom": {
        "label": "Time-Series Momentum",
        "paper_family": "Kakushadze & Serur 2018",
        "category": "factors",
        "asset_class": "multi-asset",
        "thesis": "Trade absolute trend persistence.",
    },
    "quant_factors": {
        "label": "Quant Factors",
        "paper_family": "Kakushadze & Serur 2018",
        "category": "factors",
        "asset_class": "multi-asset",
        "thesis": "Blend factor sleeves such as intraday strength and volatility.",
    },
    "options_vol": {
        "label": "Options Volatility",
        "paper_family": "Kakushadze & Serur 2018",
        "category": "options",
        "asset_class": "crypto options",
        "thesis": "Trade implied vs realised volatility structures.",
    },
    "rma_strategy": {
        "label": "RMA",
        "paper_family": "Bloch 2025 RMA",
        "category": "adaptive",
        "asset_class": "multi-asset",
        "thesis": "Use relative moving-average disequilibrium as an adaptive signal.",
    },
    "pivot_sr": {
        "label": "Pivot S/R",
        "paper_family": "Bloch 2023 Futuretesting",
        "category": "pattern",
        "asset_class": "multi-asset",
        "thesis": "Trade support/resistance and intrabar confirmation.",
    },
    "hp_trend": {
        "label": "HP Trend",
        "paper_family": "Kakushadze & Serur 2018",
        "category": "trend",
        "asset_class": "multi-asset",
        "thesis": "Smooth price noise and follow structural trend.",
    },
    "order_flow": {
        "label": "Order Flow",
        "paper_family": "Cartea et al. 2024",
        "category": "microstructure",
        "asset_class": "stocks+crypto",
        "thesis": "Trade short-horizon imbalance and VWAP displacement.",
    },
    "vpin_flow": {
        "label": "VPIN Flow",
        "paper_family": "Cartea et al. 2024",
        "category": "microstructure",
        "asset_class": "stocks+crypto",
        "thesis": "Track toxicity and aggressive flow to separate trend from noise.",
    },
    "microstructure_pressure": {
        "label": "Microstructure Pressure",
        "paper_family": "Cartea et al. 2024",
        "category": "microstructure",
        "asset_class": "stocks+crypto",
        "thesis": "Fuse signed flow, price pressure, VWAP gap, and volume shock.",
    },
    "carry_portfolio": {
        "label": "Carry Portfolio",
        "paper_family": "Kakushadze & Serur 2018",
        "category": "carry",
        "asset_class": "crypto+macro",
        "thesis": "Harvest persistent carry premia across instruments.",
    },
    "momentum_carry_combo": {
        "label": "Momentum + Carry",
        "paper_family": "Kakushadze & Serur 2018",
        "category": "carry",
        "asset_class": "crypto+macro",
        "thesis": "Blend trend confirmation with carry alignment.",
    },
    "liquidation_cascade": {
        "label": "Liquidation Cascade",
        "paper_family": "Cartea et al. 2024",
        "category": "microstructure",
        "asset_class": "crypto",
        "thesis": "Exploit forced-flow cascades and transient imbalance.",
    },
    "knn_predictor": {
        "label": "kNN Predictor",
        "paper_family": "Kakushadze & Serur 2018",
        "category": "machine learning",
        "asset_class": "multi-asset",
        "thesis": "Nearest-neighbour pattern matching for short-horizon direction.",
    },
    "contrarian_oi": {
        "label": "Contrarian OI",
        "paper_family": "Kakushadze & Serur 2018",
        "category": "alternative data",
        "asset_class": "crypto",
        "thesis": "Fade crowded open-interest build-ups.",
    },
}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_mean(values: Any) -> float:
    nums = [float(value) for value in (values or []) if value is not None]
    return float(sum(nums) / len(nums)) if nums else 0.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_ts(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _is_cache_block_fresh(block: Any, ttl_seconds: int) -> bool:
    if not isinstance(block, dict):
        return False
    generated_at = _parse_iso_ts(block.get("generated_at_utc"))
    if generated_at is None:
        return False
    return (datetime.now(timezone.utc) - generated_at).total_seconds() <= ttl_seconds


def _cache_payload_fresh(cache: dict[str, Any], ttl_seconds: int) -> bool:
    generated_at = _parse_iso_ts(cache.get("generated_at"))
    if generated_at is None:
        return False
    return (datetime.now(timezone.utc) - generated_at).total_seconds() <= ttl_seconds


def sp500_universe_symbols() -> list[str]:
    universe_report = read_json(UNIVERSE_REPORT, {})
    raw = universe_report.get("valid_symbols") or []
    symbols: list[str] = []
    seen: set[str] = set()
    for symbol in raw:
        cleaned = str(symbol or "").strip().upper()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        symbols.append(cleaned)
    return symbols


def _cluster_correlation_symbols(corr_df: pd.DataFrame) -> list[str]:
    symbols = list(corr_df.columns)
    if len(symbols) <= 2:
        return symbols
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform

        dist = (1.0 - corr_df.fillna(0.0).clip(-1.0, 1.0)).to_numpy(dtype=float, copy=True)
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        leaves = leaves_list(linkage(condensed, method="average"))
        return [symbols[int(idx)] for idx in leaves]
    except Exception:
        return sorted(symbols)


def _peer_map_from_corr(corr_df: pd.DataFrame, limit: int = 10) -> dict[str, dict[str, list[dict[str, Any]]]]:
    peer_map: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for symbol in corr_df.columns:
        row = corr_df.loc[symbol].drop(labels=[symbol], errors="ignore").dropna()
        positive = [
            {"symbol": peer, "correlation": round(float(value), 3)}
            for peer, value in row.sort_values(ascending=False).head(limit).items()
        ]
        negative = [
            {"symbol": peer, "correlation": round(float(value), 3)}
            for peer, value in row.sort_values(ascending=True).head(limit).items()
        ]
        peer_map[symbol] = {"positive": positive, "negative": negative}
    return peer_map


def _pair_extremes_from_corr(corr_df: pd.DataFrame, limit: int = 15) -> dict[str, list[dict[str, Any]]]:
    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    symbols = list(corr_df.columns)
    for left_idx, left_symbol in enumerate(symbols):
        for right_idx in range(left_idx + 1, len(symbols)):
            right_symbol = symbols[right_idx]
            value = safe_float(corr_df.iat[left_idx, right_idx])
            if value is None or math.isnan(value):
                continue
            row = {"pair": f"{left_symbol}/{right_symbol}", "correlation": round(float(value), 3)}
            positive.append(row)
            negative.append(row)
    positive.sort(key=lambda item: item["correlation"], reverse=True)
    negative.sort(key=lambda item: item["correlation"])
    return {"positive": positive[:limit], "negative": negative[:limit]}


def _correlation_stats(corr_df: pd.DataFrame, returns_df: pd.DataFrame) -> dict[str, Any]:
    if corr_df.empty:
        return {}
    mask = np.triu(np.ones(corr_df.shape, dtype=bool), k=1)
    upper = corr_df.where(mask).stack().astype(float)
    if upper.empty:
        return {}
    strongest = upper.sort_values(ascending=False)
    weakest = upper.sort_values(ascending=True)
    return {
        "symbol_count": int(corr_df.shape[0]),
        "bar_count": int(len(returns_df)),
        "avg_corr": round(float(upper.mean()), 3),
        "avg_abs_corr": round(float(upper.abs().mean()), 3),
        "median_corr": round(float(upper.median()), 3),
        "dispersion": round(float(upper.std()), 3),
        "diversification_score": round(float(max(0.0, 1.0 - upper.abs().mean())), 3),
        "strongest_pair": {
            "pair": f"{strongest.index[0][0]}/{strongest.index[0][1]}",
            "correlation": round(float(strongest.iloc[0]), 3),
        },
        "weakest_pair": {
            "pair": f"{weakest.index[0][0]}/{weakest.index[0][1]}",
            "correlation": round(float(weakest.iloc[0]), 3),
        },
    }


def _leaders_from_closes(closes: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if closes.empty or len(closes) < 3:
        return {"leaders_1d": [], "laggards_1d": [], "leaders_1m": [], "laggards_1m": []}

    one_day = closes.pct_change().iloc[-1].replace([np.inf, -np.inf], np.nan).dropna()
    month_lookback = min(21, len(closes) - 1)
    one_month = closes.pct_change(month_lookback).iloc[-1].replace([np.inf, -np.inf], np.nan).dropna()

    def _rows(series: pd.Series, reverse: bool) -> list[dict[str, Any]]:
        if series.empty:
            return []
        ranked = series.sort_values(ascending=not reverse).head(8)
        return [{"symbol": symbol, "return_pct": round(float(value) * 100.0, 2)} for symbol, value in ranked.items()]

    return {
        "leaders_1d": _rows(one_day, True),
        "laggards_1d": _rows(one_day, False),
        "leaders_1m": _rows(one_month, True),
        "laggards_1m": _rows(one_month, False),
    }


def _breadth_from_closes(closes: pd.DataFrame, annualization_factor: float) -> dict[str, Any]:
    if closes.empty or len(closes) < 3:
        return {}

    last = closes.iloc[-1]
    one_day = closes.pct_change().iloc[-1].replace([np.inf, -np.inf], np.nan).dropna()
    week_lookback = min(5, len(closes) - 1)
    month_lookback = min(21, len(closes) - 1)
    one_week = closes.pct_change(week_lookback).iloc[-1].replace([np.inf, -np.inf], np.nan).dropna()
    one_month = closes.pct_change(month_lookback).iloc[-1].replace([np.inf, -np.inf], np.nan).dropna()

    ma20 = closes.tail(min(20, len(closes))).mean()
    ma50 = closes.tail(min(50, len(closes))).mean()
    pct_above_20 = float((last > ma20).mean() * 100.0) if len(ma20) else 0.0
    pct_above_50 = float((last > ma50).mean() * 100.0) if len(ma50) else 0.0
    advancers = int((one_day > 0).sum())
    decliners = int((one_day < 0).sum())
    unchanged = int(len(one_day) - advancers - decliners)

    xs_dispersion = closes.pct_change().tail(min(20, len(closes) - 1)).std(axis=1).dropna()
    realised_vol = closes.pct_change().tail(min(20, len(closes) - 1)).std().dropna()
    breadth_score = max(
        0.0,
        min(
            1.0,
            0.45 * (pct_above_20 / 100.0)
            + 0.35 * (pct_above_50 / 100.0)
            + 0.20 * ((advancers - decliners) / max(len(one_day), 1) + 1.0) / 2.0,
        ),
    )

    return {
        "advancers_1d": advancers,
        "decliners_1d": decliners,
        "unchanged_1d": unchanged,
        "pct_above_20d": round(pct_above_20, 2),
        "pct_above_50d": round(pct_above_50, 2),
        "avg_return_1d_pct": round(float(one_day.mean() * 100.0), 2) if not one_day.empty else None,
        "avg_return_1w_pct": round(float(one_week.mean() * 100.0), 2) if not one_week.empty else None,
        "avg_return_1m_pct": round(float(one_month.mean() * 100.0), 2) if not one_month.empty else None,
        "dispersion_20d_pct": round(float(xs_dispersion.mean() * 100.0), 2) if not xs_dispersion.empty else None,
        "avg_realized_vol_pct": round(float(realised_vol.mean() * math.sqrt(annualization_factor) * 100.0), 2) if not realised_vol.empty else None,
        "breadth_score": round(float(breadth_score), 3),
    }


def _serialize_correlation_block(
    close_df: pd.DataFrame,
    *,
    asset_class: str,
    annualization_factor: float,
    requested_symbols: list[str],
    focus_symbol: str | None = None,
) -> dict[str, Any]:
    if close_df is None or close_df.empty:
        return {
            "generated_at_utc": utc_now_iso(),
            "asset_class": asset_class,
            "requested_symbol_count": len(requested_symbols),
            "symbol_count": 0,
            "coverage_pct": 0.0,
            "symbols": [],
            "corr": [],
            "vols": {},
            "peer_map": {},
            "leaders": {"positive": [], "negative": []},
            "stats": {},
            "focus_symbol": focus_symbol,
        }

    closes = close_df.sort_index().ffill().dropna(axis=1, how="all")
    returns = closes.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    min_obs = max(2, min(20, len(returns)))
    returns = returns.dropna(axis=1, thresh=min_obs)
    if returns.empty:
        return {
            "generated_at_utc": utc_now_iso(),
            "asset_class": asset_class,
            "requested_symbol_count": len(requested_symbols),
            "symbol_count": 0,
            "coverage_pct": 0.0,
            "symbols": [],
            "corr": [],
            "vols": {},
            "peer_map": {},
            "leaders": {"positive": [], "negative": []},
            "stats": {},
            "focus_symbol": focus_symbol,
        }

    corr_df = returns.corr(min_periods=min_obs).fillna(0.0).copy()
    for idx in range(len(corr_df.index)):
        corr_df.iat[idx, idx] = 1.0
    ordered_symbols = _cluster_correlation_symbols(corr_df)
    corr_df = corr_df.reindex(index=ordered_symbols, columns=ordered_symbols)
    vols = (returns[ordered_symbols].std() * math.sqrt(annualization_factor) * 100.0).round(2)
    stats = _correlation_stats(corr_df, returns[ordered_symbols])
    peer_map = _peer_map_from_corr(corr_df, limit=10)
    leaders = _pair_extremes_from_corr(corr_df, limit=15)
    closes = closes[ordered_symbols]
    breadth = _breadth_from_closes(closes, annualization_factor)
    leaders_by_return = _leaders_from_closes(closes)

    resolved_focus = focus_symbol if focus_symbol in ordered_symbols else (ordered_symbols[0] if ordered_symbols else None)
    return {
        "generated_at_utc": utc_now_iso(),
        "asset_class": asset_class,
        "requested_symbol_count": len(requested_symbols),
        "symbol_count": len(ordered_symbols),
        "coverage_pct": round((len(ordered_symbols) / max(len(requested_symbols), 1)) * 100.0, 2),
        "symbols": ordered_symbols,
        "corr": corr_df.round(3).values.tolist(),
        "vols": {symbol: float(vols.get(symbol, 0.0)) for symbol in ordered_symbols},
        "peer_map": peer_map,
        "leaders": leaders,
        "leaders_by_return": leaders_by_return,
        "breadth": breadth,
        "stats": stats,
        "focus_symbol": resolved_focus,
    }


def _build_stock_correlation_block() -> dict[str, Any]:
    symbols = sp500_universe_symbols()
    close_df = download_close_matrix(symbols, period=STOCK_CORRELATION_PERIOD, auto_adjust=True, progress=False)
    close_df = close_df.tail(252)
    block = _serialize_correlation_block(
        close_df,
        asset_class="stocks",
        annualization_factor=252.0,
        requested_symbols=symbols,
        focus_symbol="AAPL",
    )
    block["period"] = STOCK_CORRELATION_PERIOD
    return block


def _build_crypto_correlation_block() -> dict[str, Any]:
    from backtester.data_loader import load_multi

    symbols = list(dict.fromkeys(cfg.crypto_symbols))
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=CRYPTO_CORRELATION_LOOKBACK_DAYS)
    raw = load_multi(symbols, "4h", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    frames = []
    for symbol, df in raw.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        series = df["close"].astype(float).rename(symbol)
        frames.append(series)
    close_df = pd.concat(frames, axis=1).sort_index().ffill() if frames else pd.DataFrame()
    block = _serialize_correlation_block(
        close_df,
        asset_class="crypto",
        annualization_factor=365.0 * 6.0,
        requested_symbols=symbols,
        focus_symbol="BTCUSDT",
    )
    block["lookback_days"] = CRYPTO_CORRELATION_LOOKBACK_DAYS
    block["interval"] = "4h"
    return block


def build_correlation_payload(force: bool = False) -> dict[str, Any]:
    cached = read_json(CORRELATION_CACHE_PATH, {})
    payload: dict[str, Any] = dict(cached) if isinstance(cached, dict) else {}

    stocks_block = payload.get("stocks") if isinstance(payload.get("stocks"), dict) else None
    crypto_block = payload.get("crypto") if isinstance(payload.get("crypto"), dict) else None

    if force or not _is_cache_block_fresh(stocks_block, STOCK_CORRELATION_CACHE_TTL_SECONDS):
        try:
            stocks_block = _build_stock_correlation_block()
        except Exception as exc:
            stocks_block = dict(stocks_block or {})
            stocks_block["note"] = f"stock_correlation_refresh_failed: {exc}"

    if force or not _is_cache_block_fresh(crypto_block, CRYPTO_CORRELATION_CACHE_TTL_SECONDS):
        try:
            crypto_block = _build_crypto_correlation_block()
        except Exception as exc:
            crypto_block = dict(crypto_block or {})
            crypto_block["note"] = f"crypto_correlation_refresh_failed: {exc}"

    risk_snapshot = read_json(RISK_SNAPSHOT_PATH, {})
    guard_snapshot = read_json(PORTFOLIO_GUARD_PATH, {})
    risk_engine = guard_snapshot.get("portfolio_risk_engine") or risk_snapshot.get("portfolio_risk_engine") or {}

    payload = {
        "generated_at_utc": utc_now_iso(),
        "stocks": stocks_block or {},
        "crypto": crypto_block or {},
        "options_greeks": {
            "portfolio_delta": safe_float(risk_snapshot.get("portfolio_delta")),
            "portfolio_theta": safe_float(risk_snapshot.get("portfolio_theta")),
            "portfolio_vega": safe_float(risk_snapshot.get("portfolio_vega")),
            "portfolio_gamma": safe_float(risk_snapshot.get("portfolio_gamma")),
            "target_delta": safe_float(risk_snapshot.get("target_delta")),
            "target_theta": safe_float(risk_snapshot.get("target_theta")),
            "target_vega": safe_float(risk_snapshot.get("target_vega")),
            "var_pct_equity": safe_float(risk_engine.get("var_pct_equity")),
            "cvar_pct_equity": safe_float(risk_engine.get("cvar_pct_equity")),
            "stress_pct_equity": safe_float(risk_engine.get("stress_pct_equity")),
            "gross_exposure": safe_float(risk_engine.get("gross_exposure_pct_equity")),
            "net_delta_exposure": safe_float(risk_engine.get("net_delta_exposure")),
            "risk_score": safe_float(risk_engine.get("risk_score")),
            "kill_switch": bool(risk_engine.get("kill_switch_active")),
            "breaches": list(risk_engine.get("breaches") or []),
            "top_underlyings": _normalize_top_underlyings(risk_engine.get("top_underlyings")),
            "macro_regime": risk_snapshot.get("macro_regime"),
            "vix": safe_float(risk_snapshot.get("vix")),
            "movement_bias": risk_snapshot.get("movement_bias"),
            "allowed_symbols": safe_int(risk_snapshot.get("allowed_symbols")),
            "correlation_concentration": safe_float(risk_engine.get("correlation_concentration")),
        },
        "stress_scenarios": risk_engine.get("stress_losses", {}) or {},
    }

    try:
        CORRELATION_CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass
    return payload


def _strategy_enabled(name: str) -> bool:
    attr_name = f"enable_{name}"
    if hasattr(cfg, attr_name):
        return bool(getattr(cfg, attr_name))
    if name == "options_vol":
        return bool(cfg.enable_options_vol and cfg.enable_options)
    return True


def _build_strategy_catalog() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy, meta in STRATEGY_RESEARCH_LIBRARY.items():
        rows.append(
            {
                "strategy": strategy,
                "label": meta.get("label") or strategy.replace("_", " ").title(),
                "paper_family": meta.get("paper_family"),
                "category": meta.get("category"),
                "asset_class": meta.get("asset_class"),
                "thesis": meta.get("thesis"),
                "enabled": _strategy_enabled(strategy),
                "implemented": strategy in STRATEGY_CLASS_MAP or strategy in {"options_vol", "pairs_arb", "momentum", "mean_reversion"},
            }
        )
    rows.sort(key=lambda row: (not row["enabled"], row["paper_family"] or "", row["label"]))
    return rows


def _build_paper_playbooks(strategy_catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    for row in strategy_catalog:
        family = str(row.get("paper_family") or "Unmapped")
        grouped.setdefault(family, []).append(str(row.get("label") or row.get("strategy")))
    summaries = {
        "Kakushadze & Serur 2018": "Broad strategy encyclopedia spanning factors, stat-arb, carry, options, and ML playbooks.",
        "Cartea et al. 2024": "Microstructure playbook built around direction, price, volume, and trading-behaviour prediction.",
        "Bloch 2023 Futuretesting": "Robust pattern, channel, pullback, and futuretesting workflow under scenario uncertainty.",
        "Bloch 2025 RMA": "Adaptive disequilibrium framework using relative moving-average structure and robust risk control.",
    }
    papers: list[dict[str, Any]] = []
    for family, strategies in grouped.items():
        papers.append(
            {
                "paper_family": family,
                "implemented_strategies": len(strategies),
                "strategies": strategies[:8],
                "summary": summaries.get(family, "Mapped into the dashboard strategy library."),
            }
        )
    papers.sort(key=lambda row: row["paper_family"])
    return papers


def _build_microstructure_board() -> list[dict[str, Any]]:
    from backtester.data_loader import load_multi

    symbols = list(cfg.model_symbols)
    if not symbols:
        return []

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=14)
    frames = load_multi(symbols, "1h", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    rows: list[dict[str, Any]] = []

    for symbol, df in frames.items():
        if df is None or df.empty or len(df) < 40:
            continue
        frame = df.copy()
        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        open_ = frame["open"].astype(float)
        volume = frame["volume"].astype(float).clip(lower=1e-9)

        typical = (high + low + close) / 3.0
        vwap = (typical * volume).rolling(16).sum() / volume.rolling(16).sum().clip(lower=1e-9)
        vwap_gap = (float(close.iloc[-1]) - float(vwap.iloc[-1])) / max(abs(float(vwap.iloc[-1])), 1e-9)
        volume_shock = float(volume.iloc[-1] / max(float(volume.tail(24).mean()), 1e-9))
        ret_1d = float(close.pct_change(min(24, len(close) - 1)).iloc[-1] or 0.0)
        rv_20 = float(close.pct_change().tail(20).std() * math.sqrt(24 * 365) * 100.0)

        if "taker_buy_base" in frame.columns and not frame["taker_buy_base"].isna().all():
            taker = frame["taker_buy_base"].astype(float).clip(lower=0.0)
            flow_series = ((2.0 * taker / volume) - 1.0).clip(-1.5, 1.5).fillna(0.0)
        else:
            candle_range = (high - low).replace(0.0, np.nan)
            close_location = (((close - low) - (high - close)) / candle_range).clip(-1.0, 1.0).fillna(0.0)
            flow_series = close_location
        flow_imbalance = float(flow_series.tail(12).mean())

        trend_term = np.tanh(float(close.pct_change(min(6, len(close) - 1)).iloc[-1] or 0.0) * 18.0)
        pressure = float(
            0.40 * flow_imbalance
            + 0.25 * np.tanh(vwap_gap * 80.0)
            + 0.20 * np.tanh((volume_shock - 1.0) * 1.4)
            + 0.15 * trend_term
        )
        archetype = "market_making"
        if abs(flow_imbalance) > 0.22 and abs(ret_1d) > 0.012:
            archetype = "directional"
        elif volume_shock > 1.4 and abs(vwap_gap) > 0.004:
            archetype = "opportunistic"

        rows.append(
            {
                "symbol": symbol,
                "asset_class": "crypto" if cfg.is_crypto_symbol(symbol) else "stock",
                "pressure": round(pressure, 3),
                "direction": "BUY_PRESSURE" if pressure > 0.15 else ("SELL_PRESSURE" if pressure < -0.15 else "BALANCED"),
                "flow_imbalance": round(flow_imbalance, 3),
                "vwap_gap_pct": round(vwap_gap * 100.0, 2),
                "volume_shock": round(volume_shock, 2),
                "return_1d_pct": round(ret_1d * 100.0, 2),
                "realized_vol_pct": round(rv_20, 2),
                "archetype": archetype,
            }
        )

    rows.sort(key=lambda row: abs(float(row.get("pressure") or 0.0)), reverse=True)
    return rows[:12]


def build_research_desk(force: bool = False) -> dict[str, Any]:
    cached = read_json(RESEARCH_DESK_CACHE_PATH, {})
    if not force and _is_cache_block_fresh(cached, RESEARCH_DESK_CACHE_TTL_SECONDS):
        return cached

    corr = build_correlation_payload(force=force)
    strategy_catalog = _build_strategy_catalog()
    stock_block = corr.get("stocks") or {}
    crypto_block = corr.get("crypto") or {}
    try:
        microstructure_board = _build_microstructure_board()
    except Exception as exc:
        microstructure_board = [{"symbol": "UNAVAILABLE", "direction": "BALANCED", "pressure": 0.0, "archetype": f"error: {exc}"}]

    payload = {
        "generated_at_utc": utc_now_iso(),
        "stock_breadth": stock_block.get("breadth") or {},
        "crypto_breadth": crypto_block.get("breadth") or {},
        "stock_leaders": stock_block.get("leaders_by_return") or {},
        "crypto_leaders": crypto_block.get("leaders_by_return") or {},
        "stock_corr_stats": stock_block.get("stats") or {},
        "crypto_corr_stats": crypto_block.get("stats") or {},
        "strategy_catalog": strategy_catalog,
        "papers": _build_paper_playbooks(strategy_catalog),
        "microstructure_board": microstructure_board,
    }
    try:
        RESEARCH_DESK_CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass
    return payload


def parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def normalize_side(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if any(token in raw for token in ("sell", "short")):
        return "SHORT"
    return "LONG"


def side_sign(side: str) -> float:
    return -1.0 if str(side).upper() == "SHORT" else 1.0


def _broker_asset_class_label(value: Any) -> str:
    raw = str(getattr(value, "value", value) or "").strip().lower()
    if "option" in raw:
        return "options"
    if "crypto" in raw:
        return "crypto"
    if "equity" in raw or "stock" in raw:
        return "stock"
    return raw or "unknown"


def _normalize_broker_symbol(symbol: Any, asset_class: Any) -> str:
    raw = str(symbol or "").strip().upper()
    if not raw:
        return ""
    if "/" in raw:
        base, quote = raw.split("/", 1)
        if quote in {"USD", "USDT", "BUSD"}:
            return f"{base}USDT"
        return raw.replace("/", "")

    asset_label = _broker_asset_class_label(asset_class)
    if asset_label == "crypto" and raw.endswith("USD") and not raw.endswith("USDT"):
        return f"{raw[:-3]}USDT"
    return raw


def option_meta(symbol: str) -> dict[str, Any]:
    try:
        from core.utils import get_option_days_to_expiry, get_option_expiry_date, try_parse_option_symbol
    except Exception:
        return {
            "underlying": symbol,
            "option_type": None,
            "strike": None,
            "expiry": None,
            "dte": None,
        }

    parsed = try_parse_option_symbol(symbol)
    if parsed is None:
        return {
            "underlying": symbol,
            "option_type": None,
            "strike": None,
            "expiry": None,
            "dte": None,
        }

    underlying, option_type, strike = parsed
    expiry = None
    dte = None
    try:
        expiry = get_option_expiry_date(symbol).isoformat()
        dte = get_option_days_to_expiry(symbol)
    except Exception:
        pass

    return {
        "underlying": underlying,
        "option_type": "CALL" if option_type == "C" else "PUT",
        "strike": float(strike),
        "expiry": expiry,
        "dte": dte,
    }


def is_option_symbol(symbol: str) -> bool:
    meta = option_meta(symbol)
    return bool(meta.get("option_type"))


def load_execution_records() -> list[dict[str, Any]]:
    payload = read_json(EXECUTION_LEDGER_PATH, [])
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [dict(item) for item in payload["records"] if isinstance(item, dict)]
    return []


def load_runtime_trades(limit: int | None = None) -> list[dict[str, Any]]:
    if not BOT_STATE_DB_PATH.exists():
        return []
    query = "SELECT * FROM trades ORDER BY ts DESC"
    params: tuple[Any, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (int(limit),)
    try:
        with sqlite3.connect(BOT_STATE_DB_PATH, check_same_thread=False) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
    except Exception:
        return []

    payload: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["meta_payload"] = parse_json_dict(item.get("meta"))
        payload.append(item)
    return payload


def ledger_option_activity(limit: int = 24) -> list[dict[str, Any]]:
    records = sorted(load_execution_records(), key=lambda item: str(item.get("filled_at_utc") or item.get("updated_at_utc") or ""), reverse=True)
    rows: list[dict[str, Any]] = []
    for record in records:
        status = str(record.get("status") or "").lower()
        is_filled = "filled" in status or bool(record.get("partial_fill"))
        if not is_filled:
            continue
        qty = safe_float(record.get("filled_qty") or record.get("qty") or 0.0) or 0.0
        if qty <= 0:
            continue
        fill_price = safe_float(record.get("filled_avg_price") or record.get("limit_price"))
        pricing = record.get("pricing_snapshot") or {}
        execution_quality = record.get("execution_quality") or {}
        for leg in record.get("legs") or []:
            symbol = str((leg or {}).get("symbol") or "").upper()
            if not symbol or not is_option_symbol(symbol):
                continue
            meta = option_meta(symbol)
            side = normalize_side((leg or {}).get("side"))
            rows.append(
                {
                    "symbol": symbol,
                    "underlying": meta.get("underlying"),
                    "option_type": meta.get("option_type"),
                    "strike": meta.get("strike"),
                    "expiry": meta.get("expiry"),
                    "dte": meta.get("dte"),
                    "side": side,
                    "quantity": qty,
                    "entry_price": abs(fill_price) if fill_price is not None else None,
                    "filled_at": record.get("filled_at_utc") or record.get("updated_at_utc") or record.get("recorded_at_utc"),
                    "pricing_confidence": safe_float(pricing.get("pricing_confidence")),
                    "staleness_pct": safe_float(pricing.get("staleness_pct")),
                    "fair_price": safe_float(pricing.get("fair_price")),
                    "natural_price": safe_float(pricing.get("natural_price")),
                    "mc_expected_price": safe_float(pricing.get("mc_expected_price")),
                    "mc_var_95": safe_float(pricing.get("mc_var_95")),
                    "mc_cvar_95": safe_float(pricing.get("mc_cvar_95")),
                    "execution_score": safe_float(execution_quality.get("score")),
                    "execution_tier": execution_quality.get("tier"),
                    "status": status,
                    "partial_fill": bool(record.get("partial_fill")),
                    "source": "execution_ledger",
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def _snapshot_value(snapshot: Any, attr: str, nested: str | None = None) -> float | None:
    if snapshot is None:
        return None
    target = getattr(snapshot, nested, None) if nested else snapshot
    return safe_float(getattr(target, attr, None)) if target is not None else None


def live_option_positions() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if _cache_payload_fresh(_LIVE_OPTION_CACHE, BROKER_LIVE_CACHE_TTL_SECONDS):
        rows, meta = _LIVE_OPTION_CACHE.get("payload") or ([], {})
        return [dict(row) for row in rows], dict(meta)

    if not cfg.is_alpaca or not cfg.alpaca_api_key or not cfg.alpaca_api_secret:
        return [], {"available": False, "reason": "alpaca_broker_unavailable"}

    try:
        from core.broker_client import BrokerClient
    except Exception as exc:
        return [], {"available": False, "reason": f"broker_client_import_failed: {exc}"}

    try:
        client = BrokerClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
        positions = client.get_positions() or []
        option_positions = [
            pos for pos in positions
            if "option" in str(getattr(pos, "asset_class", "")).lower()
        ]
        if not option_positions:
            payload = ([], {"available": True, "source": "alpaca_live", "count": 0})
            _LIVE_OPTION_CACHE["generated_at"] = utc_now_iso()
            _LIVE_OPTION_CACHE["payload"] = payload
            return [], dict(payload[1])

        symbols = [str(getattr(pos, "symbol", "")).upper() for pos in option_positions if getattr(pos, "symbol", None)]
        snapshots = client.get_option_snapshot(symbols) if symbols else {}
        rows: list[dict[str, Any]] = []
        for pos in option_positions:
            symbol = str(getattr(pos, "symbol", "")).upper()
            meta = option_meta(symbol)
            snapshot = snapshots.get(symbol) if isinstance(snapshots, dict) else None
            greeks = getattr(snapshot, "greeks", None)
            qty = abs(safe_float(getattr(pos, "qty", 0.0)) or 0.0)
            side = normalize_side(getattr(pos, "side", None))
            sign = side_sign(side)
            theta = _snapshot_value(greeks, "theta")
            delta = _snapshot_value(greeks, "delta")
            gamma = _snapshot_value(greeks, "gamma")
            vega = _snapshot_value(greeks, "vega")
            last_price = _snapshot_value(snapshot, "price", nested="latest_trade")
            rows.append(
                {
                    "symbol": symbol,
                    "underlying": meta.get("underlying"),
                    "option_type": meta.get("option_type"),
                    "strike": meta.get("strike"),
                    "expiry": meta.get("expiry"),
                    "dte": meta.get("dte"),
                    "side": side,
                    "quantity": qty,
                    "entry_price": safe_float(getattr(pos, "avg_entry_price", None)),
                    "current_price": safe_float(getattr(pos, "current_price", None)) or last_price,
                    "market_value": safe_float(getattr(pos, "market_value", None)),
                    "unrealized_pnl": safe_float(getattr(pos, "unrealized_pl", None)),
                    "delta": delta,
                    "gamma": gamma,
                    "theta": theta,
                    "vega": vega,
                    "iv": _snapshot_value(snapshot, "implied_volatility"),
                    "bid_price": _snapshot_value(snapshot, "bid_price", nested="latest_quote"),
                    "ask_price": _snapshot_value(snapshot, "ask_price", nested="latest_quote"),
                    "last_price": last_price,
                    "portfolio_delta": None if delta is None else delta * qty * OPTION_MULTIPLIER * sign,
                    "portfolio_gamma": None if gamma is None else gamma * qty * OPTION_MULTIPLIER * sign,
                    "portfolio_theta": None if theta is None else theta * qty * OPTION_MULTIPLIER * sign,
                    "portfolio_vega": None if vega is None else vega * qty * OPTION_MULTIPLIER * sign,
                    "source": "alpaca_live",
                }
            )
        payload = (rows, {"available": True, "source": "alpaca_live", "count": len(rows)})
        _LIVE_OPTION_CACHE["generated_at"] = utc_now_iso()
        _LIVE_OPTION_CACHE["payload"] = payload
        return [dict(row) for row in rows], dict(payload[1])
    except Exception as exc:
        payload = ([], {"available": False, "reason": f"alpaca_fetch_failed: {exc}"})
        _LIVE_OPTION_CACHE["generated_at"] = utc_now_iso()
        _LIVE_OPTION_CACHE["payload"] = payload
        return [], dict(payload[1])


def live_broker_positions() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if _cache_payload_fresh(_BROKER_POSITION_CACHE, BROKER_LIVE_CACHE_TTL_SECONDS):
        rows, meta = _BROKER_POSITION_CACHE.get("payload") or ([], {})
        return [dict(row) for row in rows], dict(meta)

    if not cfg.is_alpaca or not cfg.alpaca_api_key or not cfg.alpaca_api_secret:
        return [], {"available": False, "reason": "alpaca_broker_unavailable"}

    try:
        from core.broker_client import BrokerClient
    except Exception as exc:
        return [], {"available": False, "reason": f"broker_client_import_failed: {exc}"}

    try:
        client = BrokerClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
        account = client.trade_client.get_account()
        portfolio_value = safe_float(getattr(account, "portfolio_value", None)) or safe_float(getattr(account, "equity", None))
        positions = client.get_positions() or []
        rows: list[dict[str, Any]] = []
        asset_mix: dict[str, int] = {}
        total_unrealized = 0.0
        gross_market_value = 0.0

        for pos in positions:
            raw_symbol = str(getattr(pos, "symbol", "") or "").upper()
            asset_class = _broker_asset_class_label(getattr(pos, "asset_class", None))
            symbol = _normalize_broker_symbol(raw_symbol, asset_class)
            qty = abs(safe_float(getattr(pos, "qty", 0.0)) or 0.0)
            side = normalize_side(getattr(pos, "side", None))
            market_value = safe_float(getattr(pos, "market_value", None))
            unrealized_pnl = safe_float(getattr(pos, "unrealized_pl", None))
            unrealized_pnl_pct = safe_float(getattr(pos, "unrealized_plpc", None))
            weight_pct = (market_value / portfolio_value * 100.0) if market_value is not None and portfolio_value else None
            meta = option_meta(raw_symbol) if asset_class == "options" else {
                "underlying": symbol,
                "option_type": None,
                "strike": None,
                "expiry": None,
                "dte": None,
            }

            row = {
                "symbol": symbol or raw_symbol,
                "broker_symbol": raw_symbol,
                "asset_class": asset_class,
                "market": asset_class,
                "underlying": meta.get("underlying") or symbol or raw_symbol,
                "option_type": meta.get("option_type"),
                "strike": meta.get("strike"),
                "expiry": meta.get("expiry"),
                "dte": meta.get("dte"),
                "side": side,
                "quantity": qty,
                "entry_price": safe_float(getattr(pos, "avg_entry_price", None)),
                "current_price": safe_float(getattr(pos, "current_price", None)),
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": None if unrealized_pnl_pct is None else unrealized_pnl_pct * 100.0,
                "weight_pct": weight_pct,
                "exchange": str(getattr(getattr(pos, "exchange", None), "value", getattr(pos, "exchange", "")) or "").upper() or None,
                "strategy": "broker_live",
                "source": "alpaca_live",
                "opened_at": None,
            }
            rows.append(row)
            asset_mix[asset_class] = asset_mix.get(asset_class, 0) + 1
            total_unrealized += unrealized_pnl or 0.0
            gross_market_value += abs(market_value or 0.0)

        rows.sort(key=lambda row: abs(row.get("market_value") or 0.0), reverse=True)
        payload = (rows, {
            "available": True,
            "source": "alpaca_live",
            "count": len(rows),
            "asset_mix": asset_mix,
            "gross_market_value": round(float(gross_market_value), 2),
            "net_unrealized_pnl": round(float(total_unrealized), 2),
            "portfolio_value": portfolio_value,
        })
        _BROKER_POSITION_CACHE["generated_at"] = utc_now_iso()
        _BROKER_POSITION_CACHE["payload"] = payload
        return [dict(row) for row in rows], dict(payload[1])
    except Exception as exc:
        return [], {"available": False, "reason": f"alpaca_fetch_failed: {exc}"}


def live_broker_snapshot() -> dict[str, Any]:
    if _cache_payload_fresh(_BROKER_SNAPSHOT_CACHE, BROKER_LIVE_CACHE_TTL_SECONDS):
        return dict(_BROKER_SNAPSHOT_CACHE.get("payload") or {})

    if not cfg.is_alpaca or not cfg.alpaca_api_key or not cfg.alpaca_api_secret:
        return {"available": False, "reason": "alpaca_broker_unavailable"}

    try:
        from core.broker_client import BrokerClient
        from alpaca.trading.requests import GetPortfolioHistoryRequest
    except Exception as exc:
        return {"available": False, "reason": f"broker_client_import_failed: {exc}"}

    try:
        client = BrokerClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
        account = client.trade_client.get_account()
        positions = client.get_positions() or []

        total_equity = safe_float(getattr(account, "portfolio_value", None)) or safe_float(getattr(account, "equity", None))
        cash = safe_float(getattr(account, "cash", None))
        buying_power = safe_float(getattr(account, "buying_power", None))
        long_market_value = safe_float(getattr(account, "long_market_value", None))
        short_market_value = safe_float(getattr(account, "short_market_value", None))
        last_equity = safe_float(getattr(account, "last_equity", None))
        daytrade_count = safe_int(getattr(account, "daytrade_count", None))
        account_status = str(getattr(getattr(account, "status", None), "value", getattr(account, "status", "")) or "").upper() or None

        day_pnl_dollars = (total_equity - last_equity) if total_equity is not None and last_equity else None
        day_pnl_pct = ((day_pnl_dollars / last_equity) * 100.0) if day_pnl_dollars is not None and last_equity else None

        tape_timestamps: list[str] = []
        tape_equity: list[float] = []
        try:
            history = client.trade_client.get_portfolio_history(
                GetPortfolioHistoryRequest(period="1D", timeframe="1Min", intraday_reporting="continuous")
            )
            raw_ts = list(getattr(history, "timestamp", []) or [])
            raw_eq = list(getattr(history, "equity", []) or [])
            for ts, eq in zip(raw_ts, raw_eq):
                eq_value = safe_float(eq)
                if eq_value is None or eq_value <= 0:
                    continue
                tape_timestamps.append(datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat())
                tape_equity.append(eq_value)
        except Exception:
            tape_timestamps = []
            tape_equity = []

        inception_equity = None
        all_time_high_equity = None
        try:
            full_history = client.trade_client.get_portfolio_history(
                GetPortfolioHistoryRequest(period="all", timeframe="1D", intraday_reporting="continuous")
            )
            full_equity = [safe_float(value) for value in list(getattr(full_history, "equity", []) or [])]
            full_equity = [value for value in full_equity if value is not None and value > 0]
            if full_equity:
                inception_equity = full_equity[0]
                all_time_high_equity = max(full_equity)
        except Exception:
            inception_equity = None
            all_time_high_equity = None

        first_equity = tape_equity[0] if tape_equity else total_equity
        session_high = max(tape_equity) if tape_equity else total_equity
        session_low = min(tape_equity) if tape_equity else total_equity
        if total_equity is not None:
            if session_high is None:
                session_high = total_equity
            else:
                session_high = max(session_high, total_equity)
            if session_low is None:
                session_low = total_equity
            else:
                session_low = min(session_low, total_equity)
            if all_time_high_equity is None:
                all_time_high_equity = session_high
            else:
                all_time_high_equity = max(all_time_high_equity, session_high, total_equity)
        latest_equity_ts = tape_timestamps[-1] if tape_timestamps else utc_now_iso()
        position_market_value = sum(abs(safe_float(getattr(pos, "market_value", None)) or 0.0) for pos in positions)
        unrealized_pnl = sum(safe_float(getattr(pos, "unrealized_pl", None)) or 0.0 for pos in positions)
        total_return_pct = (
            ((total_equity / inception_equity) - 1.0) * 100.0
            if total_equity is not None and inception_equity not in (None, 0.0)
            else None
        )

        payload = {
            "available": True,
            "source": "alpaca_account",
            "total_equity": total_equity,
            "cash": cash,
            "buying_power": buying_power,
            "long_market_value": long_market_value,
            "short_market_value": short_market_value,
            "position_market_value": round(float(position_market_value), 2),
            "net_unrealized_pnl": round(float(unrealized_pnl), 2),
            "last_equity": last_equity,
            "day_pnl_dollars": day_pnl_dollars,
            "day_pnl_pct": day_pnl_pct,
            "positions_count": len(positions),
            "daytrade_count": daytrade_count,
            "account_status": account_status,
            "latest_equity_ts": latest_equity_ts,
            "session_start_equity": first_equity,
            "session_high_equity": session_high,
            "session_low_equity": session_low,
            "inception_equity": inception_equity,
            "all_time_high_equity": all_time_high_equity,
            "total_return_pct": total_return_pct,
            "tape_timestamps": tape_timestamps,
            "tape_equity": tape_equity,
        }
        _BROKER_SNAPSHOT_CACHE["generated_at"] = utc_now_iso()
        _BROKER_SNAPSHOT_CACHE["payload"] = payload
        return dict(payload)
    except Exception as exc:
        return {"available": False, "reason": f"alpaca_fetch_failed: {exc}"}


def _sum_field(rows: list[dict[str, Any]], field: str) -> float:
    return float(sum(safe_float(row.get(field)) or 0.0 for row in rows))


def _avg_field(rows: list[dict[str, Any]], field: str) -> float | None:
    vals = [safe_float(row.get(field)) for row in rows]
    vals = [val for val in vals if val is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _preferred_pairs_lookup(backtest_report: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pairs_payload = backtest_report.get("pairs_suite") or {}
    lookbacks = pairs_payload.get("results_by_lookback") or {}
    preferred_lookback = "5y" if "5y" in lookbacks else ("10y" if "10y" in lookbacks else next(iter(lookbacks.keys()), None))
    pair_results: list[dict[str, Any]] = []
    top_pairs: list[dict[str, Any]] = []
    if preferred_lookback:
        selected_payload = lookbacks.get(preferred_lookback) or {}
        pair_results = list(selected_payload.get("pair_results") or [])
        top_pairs = (((selected_payload.get("summary") or {}).get("top_pairs")) or pair_results[:8])
    return preferred_lookback, pair_results, top_pairs, pairs_payload.get("summary") or {}


def _normalize_top_underlyings(items: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items or []:
        if isinstance(item, dict):
            symbol = item.get("symbol") or item.get("underlying") or item.get("ticker") or item.get("name")
            rows.append(
                {
                    "symbol": str(symbol).upper() if symbol else None,
                    "weight": safe_float(item.get("weight") or item.get("gross_weight") or item.get("share") or item.get("contribution")),
                }
            )
        elif isinstance(item, (list, tuple)) and item:
            rows.append(
                {
                    "symbol": str(item[0]).upper(),
                    "weight": safe_float(item[1]) if len(item) > 1 else None,
                }
            )
        elif item:
            rows.append({"symbol": str(item).upper(), "weight": None})
    return [row for row in rows if row.get("symbol")]


def _best_option_model(option_models: dict[str, Any]) -> dict[str, Any] | None:
    ranked: list[tuple[str, dict[str, Any]]] = [
        (name, stats)
        for name, stats in option_models.items()
        if isinstance(stats, dict)
    ]
    if not ranked:
        return None
    ranked.sort(
        key=lambda item: (
            safe_float(item[1].get("avg_edge_pct")) or -1e9,
            safe_float(item[1].get("long_win_rate")) or -1e9,
            safe_float(item[1].get("avg_signals_per_symbol")) or -1e9,
        ),
        reverse=True,
    )
    name, stats = ranked[0]
    return {
        "model": name,
        "avg_edge_pct": safe_float(stats.get("avg_edge_pct")),
        "long_win_rate": safe_float(stats.get("long_win_rate")),
        "avg_signals_per_symbol": safe_float(stats.get("avg_signals_per_symbol")),
        "avg_long_pnl": safe_float(stats.get("avg_long_pnl")),
    }


def _snapshot_age_hours(payload: dict[str, Any] | None) -> float | None:
    if not isinstance(payload, dict):
        return None
    ts = _parse_iso_ts(payload.get("generated_at_utc"))
    if ts is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0)


def _snapshot_is_fresh(payload: dict[str, Any] | None, *, max_age_hours: float) -> bool:
    age = _snapshot_age_hours(payload)
    return age is not None and age <= max_age_hours


def _latest_equity_curve_point() -> dict[str, Any]:
    if not BOT_STATE_DB_PATH.exists():
        return {}
    try:
        with sqlite3.connect(BOT_STATE_DB_PATH, check_same_thread=False) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT ts, equity, drawdown FROM equity_curve ORDER BY ts DESC LIMIT 1"
            ).fetchone()
    except Exception:
        return {}
    if not row:
        return {}
    return {
        "ts": row["ts"],
        "equity": safe_float(row["equity"]),
        "drawdown": safe_float(row["drawdown"]),
    }


def _greek_alignment_component(actual: float | None, target: float | None) -> float | None:
    if actual is None or target is None:
        return None
    scale = max(abs(target), 1.0)
    score = 1.0 - min(1.0, abs(actual - target) / scale)
    return max(0.0, min(1.0, float(score)))


def _system_readiness_score(system_snapshot: dict[str, Any]) -> float:
    status = system_snapshot.get("status") or {}
    host = system_snapshot.get("host_metrics") or {}
    memory = host.get("memory") or {}
    disk = host.get("disk") or {}
    pressure_map = {"normal": 0.95, "elevated": 0.72, "high": 0.45}
    pressure_score = pressure_map.get(str(status.get("pressure") or "").lower(), 0.65)
    cpu_score = 1.0 - min(1.0, max(0.0, float(safe_float(host.get("normalized_cpu_load_pct")) or 0.0) / 100.0))
    mem_score = 1.0 - min(1.0, max(0.0, float(safe_float(memory.get("usage_pct")) or 0.0) / 100.0))
    disk_score = 1.0 - min(1.0, max(0.0, float(safe_float(disk.get("usage_pct")) or 0.0) / 100.0))
    return round(float(max(0.0, min(1.0, safe_mean([pressure_score, cpu_score, mem_score, disk_score])))), 6)


def _execution_readiness_score(exec_summary: dict[str, Any]) -> float:
    quality = safe_float(exec_summary.get("avg_execution_quality_score"))
    fill_rate = safe_float(exec_summary.get("fill_rate"))
    coverage = safe_float(exec_summary.get("broker_fill_price_coverage"))
    records = safe_int(exec_summary.get("records")) or 0
    degraded = safe_int(exec_summary.get("degraded_execution_count")) or 0
    degradation_score = 1.0 - min(1.0, degraded / max(records, 1))
    return round(
        float(
            max(
                0.0,
                min(
                    1.0,
                    0.45 * float(quality if quality is not None else 0.60)
                    + 0.25 * float(fill_rate if fill_rate is not None else 0.60)
                    + 0.20 * float(coverage if coverage is not None else 0.60)
                    + 0.10 * degradation_score,
                ),
            )
        ),
        6,
    )


def _risk_readiness_score(risk_engine: dict[str, Any]) -> float:
    pressure = max(0.0, min(1.0, float(safe_float(risk_engine.get("risk_score")) or 0.0)))
    kill_switch = bool(risk_engine.get("kill_switch_active"))
    breaches = list(risk_engine.get("breaches") or [])
    underlying_count = safe_int(risk_engine.get("underlying_count")) or 0
    gross = safe_float(risk_engine.get("gross_exposure_pct_equity")) or 0.0

    base = 0.90 if underlying_count == 0 and gross == 0.0 else 1.0 - pressure
    if breaches:
        base -= min(0.25, 0.05 * len(breaches))
    if kill_switch:
        base = min(base, 0.15)
    return round(float(max(0.0, min(1.0, base))), 6)


def _paper_edge_profile(backtest_report: dict[str, Any]) -> dict[str, Any]:
    strategy_catalog = _build_strategy_catalog()
    enabled_rows = [row for row in strategy_catalog if row.get("enabled") and row.get("implemented")]
    family_targets = {
        "Kakushadze & Serur 2018": 5,
        "Cartea et al. 2024": 3,
        "Bloch 2023 Futuretesting": 3,
        "Bloch 2025 RMA": 1,
    }
    family_breakdown: list[dict[str, Any]] = []
    family_scores: list[float] = []
    for family, target in family_targets.items():
        strategies = [row for row in enabled_rows if row.get("paper_family") == family]
        coverage = min(1.0, len(strategies) / max(target, 1))
        family_scores.append(coverage)
        family_breakdown.append(
            {
                "paper_family": family,
                "enabled_count": len(strategies),
                "target_count": target,
                "coverage_score": round(float(coverage), 3),
            }
        )

    categories = {str(row.get("category") or "").strip() for row in enabled_rows if row.get("category")}
    all_categories = {
        str(meta.get("category") or "").strip()
        for meta in STRATEGY_RESEARCH_LIBRARY.values()
        if meta.get("category")
    }
    category_diversity = len(categories) / max(len(all_categories), 1)

    asset_tokens = set()
    for row in enabled_rows:
        for token in str(row.get("asset_class") or "").replace("+", " ").replace("-", " ").split():
            cleaned = token.strip().lower()
            if cleaned:
                asset_tokens.add(cleaned)
    has_stock = any(token.startswith("stock") for token in asset_tokens)
    has_crypto = any(token.startswith("crypto") for token in asset_tokens)
    has_options = any("option" in token for token in asset_tokens)
    asset_diversity = safe_mean(
        [
            1.0 if has_stock else 0.0,
            1.0 if has_crypto else 0.0,
            1.0 if has_options else (0.75 if (backtest_report.get("option_model_suite") or {}).get("summary") else 0.0),
        ]
    )

    microstructure_names = {"order_flow", "vpin_flow", "microstructure_pressure", "liquidation_cascade"}
    microstructure_enabled = sum(1 for row in enabled_rows if row.get("strategy") in microstructure_names)
    microstructure_depth = min(1.0, microstructure_enabled / 3.0)

    score = round(
        float(
            max(
                0.0,
                min(
                    1.0,
                    0.55 * safe_mean(family_scores)
                    + 0.20 * category_diversity
                    + 0.15 * asset_diversity
                    + 0.10 * microstructure_depth,
                ),
            )
        ),
        6,
    )
    return {
        "score": score,
        "enabled_strategies": len(enabled_rows),
        "family_breakdown": family_breakdown,
        "category_diversity": round(float(category_diversity), 3),
        "asset_diversity": round(float(asset_diversity), 3),
        "microstructure_depth": round(float(microstructure_depth), 3),
    }


def _deployment_tier_from_score(score: float) -> str:
    if score >= 0.75:
        return "institutional_candidate"
    if score >= 0.62:
        return "paper_candidate"
    return "research_only"


def build_options_overview() -> dict[str, Any]:
    risk_snapshot = read_json(RISK_SNAPSHOT_PATH, {})
    guard_snapshot = read_json(PORTFOLIO_GUARD_PATH, {})
    exec_summary = read_json(EXECUTION_SUMMARY_PATH, {})
    backtest_report = read_json(LATEST_BACKTEST_REPORT, {})
    live_rows, live_meta = live_option_positions()
    activity_rows = ledger_option_activity()

    rows = live_rows
    source = "live_options" if live_meta.get("available") else "runtime_snapshot"

    portfolio_theta_live = _sum_field(live_rows, "portfolio_theta") if live_rows else None
    portfolio_delta_live = _sum_field(live_rows, "portfolio_delta") if live_rows else None
    portfolio_vega_live = _sum_field(live_rows, "portfolio_vega") if live_rows else None
    portfolio_gamma_live = _sum_field(live_rows, "portfolio_gamma") if live_rows else None

    portfolio_theta = portfolio_theta_live if portfolio_theta_live is not None else safe_float(risk_snapshot.get("portfolio_theta"))
    portfolio_delta = portfolio_delta_live if portfolio_delta_live is not None else safe_float(risk_snapshot.get("portfolio_delta"))
    portfolio_vega = portfolio_vega_live if portfolio_vega_live is not None else safe_float(risk_snapshot.get("portfolio_vega"))
    target_theta = safe_float(risk_snapshot.get("target_theta"))
    target_delta = safe_float(risk_snapshot.get("target_delta"))
    target_vega = safe_float(risk_snapshot.get("target_vega"))

    option_model_summary = ((backtest_report.get("option_model_suite") or {}).get("summary") or {}).get("models") or {}

    return {
        "generated_at_utc": utc_now_iso(),
        "source": source,
        "live_meta": live_meta,
        "portfolio": {
            "total_equity": safe_float(risk_snapshot.get("total_equity")),
            "portfolio_delta": portfolio_delta,
            "portfolio_gamma": portfolio_gamma_live,
            "portfolio_theta": portfolio_theta,
            "portfolio_vega": portfolio_vega,
            "target_delta": target_delta,
            "target_theta": target_theta,
            "target_vega": target_vega,
            "theta_gap": None if target_theta is None or portfolio_theta is None else target_theta - portfolio_theta,
            "delta_gap": None if target_delta is None or portfolio_delta is None else target_delta - portfolio_delta,
            "vega_gap": None if target_vega is None or portfolio_vega is None else target_vega - portfolio_vega,
            "market_value": _sum_field(live_rows, "market_value") if live_rows else None,
            "unrealized_pnl": _sum_field(live_rows, "unrealized_pnl") if live_rows else None,
            "avg_iv": _avg_field(live_rows, "iv"),
            "avg_pricing_confidence": safe_float(exec_summary.get("avg_pricing_confidence")),
            "avg_execution_quality": safe_float(exec_summary.get("avg_execution_quality_score")),
        },
        "positions": rows[:18],
        "recent_activity": activity_rows[:16],
        "recent_activity_source": "execution_ledger",
        "risk_snapshot": {
            "macro_regime": risk_snapshot.get("macro_regime"),
            "movement_bias": risk_snapshot.get("movement_bias"),
            "runtime_profile": risk_snapshot.get("runtime_profile"),
            "runtime_policy_mode": risk_snapshot.get("runtime_policy_mode"),
            "open_positions": safe_int(risk_snapshot.get("open_positions")),
            "risk_score": safe_float(((guard_snapshot.get("portfolio_risk_engine") or {}).get("risk_score"))),
        },
        "pricing_models": option_model_summary,
    }


def build_elite_overview() -> dict[str, Any]:
    risk_snapshot = read_json(RISK_SNAPSHOT_PATH, {})
    guard_snapshot = read_json(PORTFOLIO_GUARD_PATH, {})
    exec_summary = read_json(EXECUTION_SUMMARY_PATH, {})
    system_snapshot = read_json(SYSTEM_RESOURCE_PATH, {})
    backtest_report = read_json(LATEST_BACKTEST_REPORT, {})
    stocks_overview = build_stocks_overview()
    runtime_trades = load_runtime_trades()
    live_pnl_snapshot = bot_state.get_daily_pnl_snapshot()
    broker_snapshot = live_broker_snapshot()
    broker_positions, broker_position_meta = live_broker_positions()

    movement_summary = ((backtest_report.get("movement_suite") or {}).get("summary") or {})
    ml_summary = ((backtest_report.get("ml_alpha_suite") or {}).get("summary") or {})
    regime_summary = ((backtest_report.get("regime_suite") or {}).get("summary") or {})
    strategy_profile_summary = ((backtest_report.get("strategy_profile_suite") or {}).get("summary") or {})
    option_models = (((backtest_report.get("option_model_suite") or {}).get("summary") or {}).get("models") or {})
    massive_overview = backtest_report.get("massive_overview") or {}
    institutional = backtest_report.get("institutional_robustness") or {}
    paper_edge = _paper_edge_profile(backtest_report)

    preferred_pairs_lookback, _, _, pairs_summary = _preferred_pairs_lookup(backtest_report)
    risk_engine = (guard_snapshot.get("portfolio_risk_engine") or risk_snapshot.get("portfolio_risk_engine") or {})
    host_metrics = system_snapshot.get("host_metrics") or {}
    memory = host_metrics.get("memory") or {}
    disk = host_metrics.get("disk") or {}
    latest_equity = _latest_equity_curve_point()
    latest_trade_at_utc = runtime_trades[0].get("ts") if runtime_trades else None

    open_trade_count = sum(1 for row in runtime_trades if str(row.get("status") or "").lower() == "open")
    closed_trade_count = sum(1 for row in runtime_trades if str(row.get("status") or "").lower() == "closed")

    open_positions = safe_int(risk_snapshot.get("open_positions"))
    if open_positions is None:
        open_positions = open_trade_count
    if broker_snapshot.get("available") and broker_snapshot.get("positions_count") is not None:
        open_positions = safe_int(broker_snapshot.get("positions_count"))

    risk_snapshot_fresh = _snapshot_is_fresh(risk_snapshot, max_age_hours=6)
    latest_equity_ts = _parse_iso_ts(latest_equity.get("ts"))
    risk_snapshot_ts = _parse_iso_ts(risk_snapshot.get("generated_at_utc"))
    broker_equity = safe_float(broker_snapshot.get("total_equity")) if broker_snapshot.get("available") else None
    headline_equity = broker_equity if broker_equity is not None else safe_float(live_pnl_snapshot.get("current_equity"))
    if headline_equity is None:
        headline_equity = safe_float(risk_snapshot.get("total_equity"))
        if latest_equity.get("equity") is not None and (risk_snapshot_ts is None or (latest_equity_ts and latest_equity_ts > risk_snapshot_ts)):
            headline_equity = safe_float(latest_equity.get("equity"))

    predictive_score = float(safe_float(massive_overview.get("predictive_score")) or 0.0)
    backtest_institutional_score = float(
        safe_float(institutional.get("institutional_score"))
        or safe_float(massive_overview.get("institutional_score"))
        or predictive_score
    )
    execution_readiness = _execution_readiness_score(exec_summary)
    risk_readiness = _risk_readiness_score(risk_engine)
    system_readiness = _system_readiness_score(system_snapshot)
    live_operations_score = round(float(safe_mean([execution_readiness, risk_readiness, system_readiness])), 6)
    live_institutional_score = round(
        float(
            max(
                0.0,
                min(
                    1.0,
                    0.50 * backtest_institutional_score
                    + 0.18 * paper_edge["score"]
                    + 0.12 * system_readiness
                    + 0.10 * execution_readiness
                    + 0.10 * risk_readiness,
                ),
            )
        ),
        6,
    )
    deployment_tier = _deployment_tier_from_score(live_institutional_score)
    daily_pnl_dollars = safe_float(broker_snapshot.get("day_pnl_dollars")) if broker_snapshot.get("available") else safe_float(live_pnl_snapshot.get("daily_pnl"))
    daily_pnl_pct = safe_float(broker_snapshot.get("day_pnl_pct")) if broker_snapshot.get("available") else safe_float(live_pnl_snapshot.get("daily_pnl_pct"))
    if daily_pnl_dollars is None and risk_snapshot_fresh:
        daily_pnl_dollars = safe_float(risk_snapshot.get("daily_pnl_dollars"))
    if daily_pnl_pct is None and risk_snapshot_fresh:
        daily_pnl_pct = safe_float(risk_snapshot.get("daily_pnl_pct"))

    portfolio_delta = safe_float(risk_snapshot.get("portfolio_delta")) if risk_snapshot_fresh else None
    portfolio_theta = safe_float(risk_snapshot.get("portfolio_theta")) if risk_snapshot_fresh else None
    portfolio_vega = safe_float(risk_snapshot.get("portfolio_vega")) if risk_snapshot_fresh else None
    target_delta = safe_float(risk_snapshot.get("target_delta")) if risk_snapshot_fresh else None
    target_theta = safe_float(risk_snapshot.get("target_theta")) if risk_snapshot_fresh else None
    target_vega = safe_float(risk_snapshot.get("target_vega")) if risk_snapshot_fresh else None
    alignment_components = [
        _greek_alignment_component(portfolio_delta, target_delta),
        _greek_alignment_component(portfolio_theta, target_theta),
        _greek_alignment_component(portfolio_vega, target_vega),
    ]
    greek_alignment_score = (
        round(float(safe_mean(alignment_components)), 6)
        if any(value is not None for value in alignment_components)
        else None
    )
    current_live_equity = broker_equity if broker_equity is not None else safe_float(live_pnl_snapshot.get("current_equity"))

    risk_snapshot_age_hours = _snapshot_age_hours(risk_snapshot)
    execution_snapshot_age_hours = _snapshot_age_hours(exec_summary)
    system_snapshot_age_hours = _snapshot_age_hours(system_snapshot)
    latest_equity_mark_ts = _parse_iso_ts(broker_snapshot.get("latest_equity_ts") or live_pnl_snapshot.get("latest_equity_ts"))
    equity_snapshot_age_seconds = (
        max(0.0, (datetime.now(timezone.utc) - latest_equity_mark_ts).total_seconds())
        if latest_equity_mark_ts
        else safe_float(live_pnl_snapshot.get("equity_freshness_seconds"))
    )
    broker_tape_timestamps = list(broker_snapshot.get("tape_timestamps") or [])
    kill_switch_active = bool(risk_engine.get("kill_switch_active"))
    if kill_switch_active:
        trading_status = "HALTED"
        trading_reason = ", ".join(list(risk_engine.get("hard_kill_reasons") or []) or ["Kill switch active"])
    elif not risk_snapshot_fresh:
        trading_status = "LIVE / RISK STALE"
        trading_reason = "Broker account is live but the runtime risk snapshot is stale."
    elif broker_snapshot.get("available"):
        trading_status = "LIVE"
        trading_reason = f"Broker account status {broker_snapshot.get('account_status') or 'ACTIVE'}"
    else:
        trading_status = "MONITORING"
        trading_reason = "Runtime feed only"
    peak_equity = max(
        value
        for value in [
            safe_float(live_pnl_snapshot.get("peak_equity")),
            safe_float(broker_snapshot.get("session_high_equity")),
            safe_float(broker_snapshot.get("all_time_high_equity")),
            current_live_equity,
        ]
        if value is not None
    ) if any(value is not None for value in [safe_float(live_pnl_snapshot.get("peak_equity")), safe_float(broker_snapshot.get("session_high_equity")), safe_float(broker_snapshot.get("all_time_high_equity")), current_live_equity]) else None
    distance_to_peak_pct = (
        ((current_live_equity / peak_equity) - 1.0) * 100.0
        if current_live_equity is not None and peak_equity not in (None, 0.0)
        else safe_float(live_pnl_snapshot.get("distance_to_peak_pct"))
    )

    return {
        "generated_at_utc": utc_now_iso(),
        "status": {
            "trading_status": trading_status,
            "trading_reason": trading_reason,
            "kill_switch_active": kill_switch_active,
        },
        "headline": {
            "total_equity": headline_equity,
            "daily_pnl_pct": daily_pnl_pct,
            "daily_pnl_dollars": daily_pnl_dollars,
            "buying_power_budget": safe_float(broker_snapshot.get("buying_power")) if broker_snapshot.get("available") else safe_float(risk_snapshot.get("buying_power_budget")),
            "vix": safe_float(risk_snapshot.get("vix")) if risk_snapshot_fresh else None,
            "macro_regime": risk_snapshot.get("macro_regime") if risk_snapshot_fresh else None,
            "macro_confidence": safe_float(risk_snapshot.get("macro_confidence")) if risk_snapshot_fresh else None,
            "movement_bias": risk_snapshot.get("movement_bias") if risk_snapshot_fresh else None,
            "runtime_profile": risk_snapshot.get("runtime_profile") if risk_snapshot_fresh else None,
            "runtime_policy_mode": risk_snapshot.get("runtime_policy_mode") if risk_snapshot_fresh else None,
            "runtime_market_state": risk_snapshot.get("runtime_market_state") if risk_snapshot_fresh else None,
            "allowed_symbols": safe_int(risk_snapshot.get("allowed_symbols")) if risk_snapshot_fresh else None,
            "open_positions": open_positions,
            "portfolio_delta": portfolio_delta,
            "portfolio_theta": portfolio_theta,
            "portfolio_vega": portfolio_vega,
            "target_delta": target_delta,
            "target_theta": target_theta,
            "target_vega": target_vega,
            "risk_snapshot_fresh": risk_snapshot_fresh,
        },
        "performance": {
            "source": broker_snapshot.get("source") if broker_snapshot.get("available") else live_pnl_snapshot.get("source"),
            "current_equity": current_live_equity if current_live_equity is not None else headline_equity,
            "session_start_equity": safe_float(broker_snapshot.get("session_start_equity")) if broker_snapshot.get("available") else safe_float(live_pnl_snapshot.get("session_start_equity")),
            "session_start_ts": (broker_tape_timestamps[0] if broker_tape_timestamps else None) if broker_snapshot.get("available") else live_pnl_snapshot.get("session_start_ts"),
            "daily_pnl": daily_pnl_dollars,
            "daily_pnl_pct": daily_pnl_pct,
            "intraday_low_equity": safe_float(broker_snapshot.get("session_low_equity")) if broker_snapshot.get("available") else safe_float(live_pnl_snapshot.get("intraday_low_equity")),
            "intraday_high_equity": safe_float(broker_snapshot.get("session_high_equity")) if broker_snapshot.get("available") else safe_float(live_pnl_snapshot.get("intraday_high_equity")),
            "intraday_range_pct": (
                ((safe_float(broker_snapshot.get("session_high_equity")) - safe_float(broker_snapshot.get("session_low_equity"))) / safe_float(broker_snapshot.get("session_start_equity")) * 100.0)
                if broker_snapshot.get("available")
                and safe_float(broker_snapshot.get("session_high_equity")) is not None
                and safe_float(broker_snapshot.get("session_low_equity")) is not None
                and safe_float(broker_snapshot.get("session_start_equity"))
                else safe_float(live_pnl_snapshot.get("intraday_range_pct"))
            ),
            "peak_equity": peak_equity,
            "distance_to_peak_pct": distance_to_peak_pct,
            "latest_equity_ts": broker_snapshot.get("latest_equity_ts") if broker_snapshot.get("available") else (live_pnl_snapshot.get("latest_equity_ts") or latest_equity.get("ts")),
            "equity_samples_today": len(list(broker_snapshot.get("tape_equity") or [])) if broker_snapshot.get("available") else safe_int(live_pnl_snapshot.get("equity_samples_today")),
            "closed_trade_pnl_today": safe_float(live_pnl_snapshot.get("closed_trade_pnl_today")),
            "closed_trade_count_today": safe_int(live_pnl_snapshot.get("closed_trade_count_today")),
            "cash": safe_float(broker_snapshot.get("cash")) if broker_snapshot.get("available") else None,
            "position_market_value": safe_float(broker_snapshot.get("position_market_value")) if broker_snapshot.get("available") else None,
            "long_market_value": safe_float(broker_snapshot.get("long_market_value")) if broker_snapshot.get("available") else None,
            "short_market_value": safe_float(broker_snapshot.get("short_market_value")) if broker_snapshot.get("available") else None,
            "net_unrealized_pnl": safe_float(broker_snapshot.get("net_unrealized_pnl")) if broker_snapshot.get("available") else safe_float(broker_position_meta.get("net_unrealized_pnl")),
            "account_status": broker_snapshot.get("account_status"),
            "total_return_pct": safe_float(broker_snapshot.get("total_return_pct")) if broker_snapshot.get("available") else None,
        },
        "risk": {
            "risk_score": safe_float(risk_engine.get("risk_score")),
            "var_pct_equity": safe_float(risk_engine.get("var_pct_equity")),
            "cvar_pct_equity": safe_float(risk_engine.get("cvar_pct_equity")),
            "stress_pct_equity": safe_float(risk_engine.get("stress_pct_equity")),
            "gross_exposure_pct_equity": safe_float(risk_engine.get("gross_exposure_pct_equity")),
            "net_delta_exposure": safe_float(risk_engine.get("net_delta_exposure")),
            "correlation_concentration": safe_float(risk_engine.get("correlation_concentration")),
            "max_underlying_weight": safe_float(risk_engine.get("max_underlying_weight")),
            "value_volatility": safe_float(risk_engine.get("value_volatility")),
            "simulation_paths": safe_int(risk_engine.get("simulation_paths")),
            "confidence": safe_float(risk_engine.get("confidence")),
            "greek_alignment_score": greek_alignment_score,
            "kill_switch_active": bool(risk_engine.get("kill_switch_active")),
            "underlying_count": safe_int(risk_engine.get("underlying_count")),
            "breaches": list(risk_engine.get("breaches") or []),
            "hard_kill_reasons": list(risk_engine.get("hard_kill_reasons") or []),
            "top_underlyings": _normalize_top_underlyings(risk_engine.get("top_underlyings")),
        },
        "execution": {
            "records": safe_int(exec_summary.get("records")),
            "fill_events": safe_int(exec_summary.get("fill_events")),
            "fill_rate": safe_float(exec_summary.get("fill_rate")),
            "full_fill_rate": safe_float(exec_summary.get("full_fill_rate")),
            "avg_execution_quality_score": safe_float(exec_summary.get("avg_execution_quality_score")),
            "avg_pricing_confidence": safe_float(exec_summary.get("avg_pricing_confidence")),
            "avg_limit_edge_bps": safe_float(exec_summary.get("avg_limit_edge_bps")),
            "avg_reference_edge_bps": safe_float(exec_summary.get("avg_reference_edge_bps")),
            "broker_fill_price_coverage": safe_float(exec_summary.get("broker_fill_price_coverage")),
            "degraded_execution_count": safe_int(exec_summary.get("degraded_execution_count")),
            "latest_fill_at_utc": exec_summary.get("latest_fill_at_utc"),
            "tier_counts": exec_summary.get("tier_counts") or {},
        },
        "research": {
            "predictive_score": predictive_score,
            "institutional_score": live_institutional_score,
            "backtest_institutional_score": backtest_institutional_score,
            "deployment_tier": deployment_tier,
            "backtest_deployment_tier": institutional.get("deployment_tier") or massive_overview.get("deployment_tier"),
            "consensus_profile": strategy_profile_summary.get("consensus_profile"),
            "consensus_state": strategy_profile_summary.get("consensus_state"),
            "movement_avg_accuracy": safe_float(movement_summary.get("avg_accuracy")),
            "movement_avg_alpha_daily": safe_float(movement_summary.get("avg_alpha_daily")),
            "pairs_win_rate": safe_float(pairs_summary.get("win_rate")),
            "pairs_avg_trade_return": safe_float(pairs_summary.get("avg_trade_return")),
            "pairs_lookback": preferred_pairs_lookback,
            "regime_accuracy": safe_float(regime_summary.get("directional_accuracy_proxy")),
            "ml_information_coefficient": safe_float(ml_summary.get("avg_information_coefficient")),
            "ml_long_only_return": safe_float(((ml_summary.get("long_only") or {}).get("annualized_return"))),
            "ml_long_only_sharpe": safe_float(((ml_summary.get("long_only") or {}).get("sharpe_ratio"))),
            "best_option_model": _best_option_model(option_models),
            "top_pairs": (stocks_overview.get("pairs", {}) or {}).get("top_pairs", [])[:8],
            "research_leaders": stocks_overview.get("research_leaders", [])[:8],
            "paper_edge_score": paper_edge["score"],
            "paper_edge_profile": paper_edge,
            "live_operations_score": live_operations_score,
            "execution_readiness_score": execution_readiness,
            "risk_readiness_score": risk_readiness,
            "system_readiness_score": system_readiness,
        },
        "freshness": {
            "equity_snapshot_age_seconds": round(float(equity_snapshot_age_seconds), 2) if equity_snapshot_age_seconds is not None else None,
            "risk_snapshot_age_hours": round(float(risk_snapshot_age_hours), 2) if risk_snapshot_age_hours is not None else None,
            "execution_snapshot_age_hours": round(float(execution_snapshot_age_hours), 2) if execution_snapshot_age_hours is not None else None,
            "system_snapshot_age_hours": round(float(system_snapshot_age_hours), 2) if system_snapshot_age_hours is not None else None,
            "equity_last_update_utc": broker_snapshot.get("latest_equity_ts") if broker_snapshot.get("available") else (live_pnl_snapshot.get("latest_equity_ts") or latest_equity.get("ts")),
            "broker_account_live": bool(broker_snapshot.get("available")),
            "risk_generated_at_utc": risk_snapshot.get("generated_at_utc"),
            "execution_generated_at_utc": exec_summary.get("generated_at_utc"),
            "system_generated_at_utc": system_snapshot.get("generated_at_utc"),
            "equity_snapshot_fresh": equity_snapshot_age_seconds is not None and equity_snapshot_age_seconds <= 300.0,
            "risk_snapshot_fresh": risk_snapshot_fresh,
        },
        "infrastructure": {
            "pressure": (system_snapshot.get("status") or {}).get("pressure"),
            "note": (system_snapshot.get("status") or {}).get("note"),
            "cpu_load_pct": safe_float(host_metrics.get("normalized_cpu_load_pct")),
            "loadavg_1m": safe_float(host_metrics.get("loadavg_1m")),
            "loadavg_5m": safe_float(host_metrics.get("loadavg_5m")),
            "memory_usage_pct": safe_float(memory.get("usage_pct")),
            "memory_available_gb": safe_float(memory.get("available_gb")),
            "disk_usage_pct": safe_float(disk.get("usage_pct")),
            "disk_free_gb": safe_float(disk.get("free_gb")),
            "backtest_workers": safe_int((system_snapshot.get("resource_profile") or {}).get("backtest_workers")),
            "research_rf_jobs": safe_int((system_snapshot.get("resource_profile") or {}).get("research_rf_jobs")),
            "model_parallelism": safe_int((system_snapshot.get("resource_profile") or {}).get("model_parallelism")),
            "strategy_interval_seconds": safe_int((system_snapshot.get("resource_profile") or {}).get("strategy_interval_seconds")),
            "risk_interval_seconds": safe_int((system_snapshot.get("resource_profile") or {}).get("risk_interval_seconds")),
        },
        "activity": {
            "db_trades": len(runtime_trades),
            "open_trades": open_trade_count,
            "closed_trades": closed_trade_count,
            "latest_trade_at_utc": latest_trade_at_utc,
            "broker_positions": len(broker_positions),
            "broker_asset_mix": broker_position_meta.get("asset_mix") or {},
        },
    }


def _returns_from_equity(equity: list[float]) -> np.ndarray:
    arr = np.asarray([float(v) for v in equity if v is not None], dtype=float)
    if arr.size < 3:
        return np.array([], dtype=float)
    rets = np.diff(np.log(np.clip(arr, 1e-6, None)))
    if rets.size == 0:
        return np.array([], dtype=float)
    if np.allclose(rets, 0.0):
        return np.array([], dtype=float)
    return rets


def _proxy_returns_from_snapshot(risk_snapshot: dict[str, Any], n_obs: int = 120) -> np.ndarray:
    vix = safe_float(risk_snapshot.get("vix")) or 18.0
    theta = safe_float(risk_snapshot.get("portfolio_theta")) or safe_float(risk_snapshot.get("target_theta")) or 0.0
    equity = safe_float(risk_snapshot.get("total_equity")) or 10000.0
    daily_pnl_pct = safe_float(risk_snapshot.get("daily_pnl_pct")) or 0.0
    vol = max(0.08, min(0.65, (vix / 100.0) * 0.85 + 0.06))
    drift = daily_pnl_pct / 7.0 + (theta / max(equity, 1.0)) / 252.0
    np.random.seed(7)
    series = np.random.normal(loc=drift, scale=vol / math.sqrt(252.0), size=n_obs)
    np.random.seed(None)
    return series


def _simulate_forward_paths(
    base_value: float,
    returns: np.ndarray,
    horizon_days: int = DEFAULT_FORECAST_DAYS,
    n_paths: int = DEFAULT_MC_PATHS,
    mode: str = "bootstrap",
) -> tuple[np.ndarray, np.ndarray]:
    if returns.size == 0:
        return np.empty((0, horizon_days)), np.empty((0,))

    paths = np.empty((n_paths, horizon_days), dtype=float)
    if mode == "bootstrap":
        for i in range(n_paths):
            draws = np.random.choice(returns, size=horizon_days, replace=True)
            paths[i] = base_value * np.exp(np.cumsum(draws))
    else:
        pool = returns.copy()
        if pool.size < horizon_days:
            reps = int(math.ceil(horizon_days / max(pool.size, 1)))
            pool = np.tile(pool, reps)
        for i in range(n_paths):
            perm = np.random.permutation(pool)[:horizon_days]
            paths[i] = base_value * np.exp(np.cumsum(perm))
    return paths, paths[:, -1]


def _equity_bands(paths: np.ndarray) -> dict[str, list[float]]:
    if paths.size == 0:
        return {}
    return {
        "p5": np.percentile(paths, 5, axis=0).round(2).tolist(),
        "p25": np.percentile(paths, 25, axis=0).round(2).tolist(),
        "p50": np.percentile(paths, 50, axis=0).round(2).tolist(),
        "p75": np.percentile(paths, 75, axis=0).round(2).tolist(),
        "p95": np.percentile(paths, 95, axis=0).round(2).tolist(),
    }


def build_simulation_payload(equity: list[float], timestamps: list[str]) -> dict[str, Any]:
    risk_snapshot = read_json(RISK_SNAPSHOT_PATH, {})
    guard_snapshot = read_json(PORTFOLIO_GUARD_PATH, {})
    base_equity = float(equity[-1]) if equity else (safe_float(risk_snapshot.get("total_equity")) or 10000.0)

    returns = _returns_from_equity(equity)
    observation_count = int(returns.size)
    source = "equity_curve"
    notice: str | None = None
    available = bool(returns.size >= 20 and not np.allclose(np.std(returns), 0.0))

    if not available:
        notice = (
            "Simulation Lab needs at least 20 non-flat equity observations from a continuous portfolio history. "
            "Current runtime history is too sparse or reset-heavy, so synthetic paths were suppressed."
        )
        horizons: list[int] = []
        return {
            "generated_at_utc": utc_now_iso(),
            "available": False,
            "source": "insufficient_history",
            "base_equity": round(base_equity, 2),
            "horizon_days": 0,
            "monte_carlo": {"paths": 0, "prob_profit_pct": None, "terminal": {"p5": None, "p50": None, "p95": None}, "bands": {}},
            "las_vegas": {"paths": 0, "prob_profit_pct": None, "terminal": {"p5": None, "p50": None, "p95": None}, "bands": {}},
            "labels": horizons,
            "observation_count": observation_count,
            "notice": notice,
            "risk_context": {
                "daily_pnl_pct": safe_float(risk_snapshot.get("daily_pnl_pct")),
                "vix": safe_float(risk_snapshot.get("vix")),
                "macro_regime": risk_snapshot.get("macro_regime"),
                "movement_bias": risk_snapshot.get("movement_bias"),
                "risk_score": safe_float(((guard_snapshot.get("portfolio_risk_engine") or {}).get("risk_score"))),
                "simulation_paths": safe_int(((guard_snapshot.get("portfolio_risk_engine") or {}).get("simulation_paths"))),
            },
        }

    np.random.seed(11)
    mc_paths, mc_terminal = _simulate_forward_paths(base_equity, returns, mode="bootstrap")
    lv_paths, lv_terminal = _simulate_forward_paths(base_equity, returns, mode="permutation")
    np.random.seed(None)

    horizons = list(range(1, (mc_paths.shape[1] if mc_paths.size else DEFAULT_FORECAST_DAYS) + 1))
    mc_prob_profit = float((mc_terminal > base_equity).mean() * 100.0) if mc_terminal.size else None
    lv_prob_profit = float((lv_terminal > base_equity).mean() * 100.0) if lv_terminal.size else None

    return {
        "generated_at_utc": utc_now_iso(),
        "available": True,
        "source": source,
        "base_equity": round(base_equity, 2),
        "horizon_days": len(horizons),
        "observation_count": observation_count,
        "notice": notice,
        "monte_carlo": {
            "paths": int(mc_paths.shape[0]) if mc_paths.size else 0,
            "prob_profit_pct": None if mc_prob_profit is None else round(mc_prob_profit, 2),
            "terminal": {
                "p5": None if mc_terminal.size == 0 else round(float(np.percentile(mc_terminal, 5)), 2),
                "p50": None if mc_terminal.size == 0 else round(float(np.percentile(mc_terminal, 50)), 2),
                "p95": None if mc_terminal.size == 0 else round(float(np.percentile(mc_terminal, 95)), 2),
            },
            "bands": _equity_bands(mc_paths),
        },
        "las_vegas": {
            "paths": int(lv_paths.shape[0]) if lv_paths.size else 0,
            "prob_profit_pct": None if lv_prob_profit is None else round(lv_prob_profit, 2),
            "terminal": {
                "p5": None if lv_terminal.size == 0 else round(float(np.percentile(lv_terminal, 5)), 2),
                "p50": None if lv_terminal.size == 0 else round(float(np.percentile(lv_terminal, 50)), 2),
                "p95": None if lv_terminal.size == 0 else round(float(np.percentile(lv_terminal, 95)), 2),
            },
            "bands": _equity_bands(lv_paths),
        },
        "labels": horizons,
        "risk_context": {
            "daily_pnl_pct": safe_float(risk_snapshot.get("daily_pnl_pct")),
            "vix": safe_float(risk_snapshot.get("vix")),
            "macro_regime": risk_snapshot.get("macro_regime"),
            "movement_bias": risk_snapshot.get("movement_bias"),
            "risk_score": safe_float(((guard_snapshot.get("portfolio_risk_engine") or {}).get("risk_score"))),
            "simulation_paths": safe_int(((guard_snapshot.get("portfolio_risk_engine") or {}).get("simulation_paths"))),
        },
    }


def _aggregate_stock_research(movement_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_symbol: dict[str, dict[str, Any]] = {}
    lookback_rank = {"1y": 3, "3y": 2, "5y": 2, "10y": 1}
    for row in movement_results:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        score = lookback_rank.get(str(row.get("lookback")), 0)
        existing = per_symbol.get(symbol)
        current = {
            "symbol": symbol,
            "lookback": row.get("lookback"),
            "accuracy": safe_float(row.get("accuracy")),
            "hit_ratio": safe_float(row.get("hit_ratio")),
            "strategy_return": safe_float(row.get("strategy_return")),
            "buy_hold_return": safe_float(row.get("buy_hold_return")),
            "alpha_daily": safe_float(row.get("alpha_daily")),
            "meets_targets": bool(row.get("meets_targets")),
            "_score": score,
        }
        if existing is None or current["_score"] > existing["_score"]:
            per_symbol[symbol] = current
    rows = list(per_symbol.values())
    for row in rows:
        row.pop("_score", None)
    rows.sort(key=lambda item: (item.get("alpha_daily") or -1e9, item.get("accuracy") or -1e9), reverse=True)
    return rows


def build_stocks_overview() -> dict[str, Any]:
    universe_report = read_json(UNIVERSE_REPORT, {})
    backtest_report = read_json(LATEST_BACKTEST_REPORT, {})
    stock_research_report = read_json(STOCK_UNIVERSE_RESEARCH, {})
    movement_results = ((backtest_report.get("movement_suite") or {}).get("results") or [])
    ml_summary = ((backtest_report.get("ml_alpha_suite") or {}).get("summary") or {})

    research_rows = _aggregate_stock_research(movement_results)
    full_universe_rows = stock_research_report.get("rows") if isinstance(stock_research_report.get("rows"), list) else []
    preferred_lookback, pair_results, top_pairs, pairs_summary = _preferred_pairs_lookup(backtest_report)
    universe_symbols = [str(symbol).upper() for symbol in (universe_report.get("valid_symbols") or []) if symbol]
    if not universe_symbols and full_universe_rows:
        universe_symbols = [str(row.get("symbol")).upper() for row in full_universe_rows if row.get("symbol")]
    universe_size = safe_int(universe_report.get("symbols_valid")) or len(universe_symbols)
    research_by_symbol = {str(row.get("symbol")).upper(): row for row in research_rows if row.get("symbol")}
    full_research_by_symbol = {str(row.get("symbol")).upper(): row for row in full_universe_rows if row.get("symbol")}
    researched_symbols = 0
    price_ready_symbols = 0
    positive_alpha_symbols = 0
    universe_rows: list[dict[str, Any]] = []
    for rank, symbol in enumerate(universe_symbols, start=1):
        movement_base = research_by_symbol.get(symbol) or {}
        universe_base = full_research_by_symbol.get(symbol) or {}
        base = movement_base or universe_base
        has_research = bool(base and (movement_base or universe_base.get("has_research")))
        has_movement_research = bool(movement_base)
        data_status = str(universe_base.get("data_status") or ("ready" if has_movement_research else "pending"))
        if data_status == "ready":
            price_ready_symbols += 1
        if has_research:
            researched_symbols += 1
            if (safe_float(base.get("alpha_daily")) or 0.0) > 0:
                positive_alpha_symbols += 1
        universe_rows.append(
            {
                "rank": rank,
                "symbol": symbol,
                "has_research": has_research,
                "has_movement_research": has_movement_research,
                "research_source": base.get("research_source") or ("movement_suite" if has_movement_research else None),
                "data_status": data_status,
                "data_rows": safe_int(universe_base.get("data_rows")),
                "last_price": safe_float(universe_base.get("last_price")),
                "last_date": universe_base.get("last_date"),
                "lookback": movement_base.get("lookback") or stock_research_report.get("period"),
                "accuracy": safe_float(movement_base.get("accuracy")),
                "hit_ratio": safe_float(movement_base.get("hit_ratio")),
                "alpha_daily": safe_float(base.get("alpha_daily")),
                "strategy_return": safe_float(movement_base.get("strategy_return")),
                "buy_hold_return": safe_float(movement_base.get("buy_hold_return")) or safe_float(universe_base.get("momentum_12m")),
                "volatility_annualized": safe_float(universe_base.get("volatility_annualized")),
                "sharpe_proxy": safe_float(universe_base.get("sharpe_proxy")),
                "max_drawdown": safe_float(universe_base.get("max_drawdown")),
                "beta_spy": safe_float(universe_base.get("beta_spy")),
                "corr_spy": safe_float(universe_base.get("corr_spy")),
                "momentum_1m": safe_float(universe_base.get("momentum_1m")),
                "momentum_3m": safe_float(universe_base.get("momentum_3m")),
                "momentum_12m": safe_float(universe_base.get("momentum_12m")),
                "meets_targets": bool(movement_base.get("meets_targets")) if has_movement_research else False,
            }
        )

    pair_symbols: set[str] = set()
    for row in pair_results:
        pair = str(row.get("pair") or "")
        if "/" not in pair:
            continue
        left, right = pair.split("/", 1)
        pair_symbols.add(left.upper())
        pair_symbols.add(right.upper())

    return {
        "generated_at_utc": utc_now_iso(),
        "universe_size": universe_size,
        "universe_sample": universe_symbols[:18],
        "universe_symbols": universe_symbols,
        "research_leaders": research_rows,
        "universe_rows": universe_rows,
        "research_coverage_pct": round((researched_symbols / max(universe_size, 1)) * 100.0, 2),
        "price_data_coverage_pct": round((price_ready_symbols / max(universe_size, 1)) * 100.0, 2),
        "price_ready_symbols": price_ready_symbols,
        "researched_symbols": researched_symbols,
        "positive_alpha_symbols": positive_alpha_symbols,
        "stock_universe_research": {
            "generated_at_utc": stock_research_report.get("generated_at_utc"),
            "period": stock_research_report.get("period"),
            "symbols_with_price_data": stock_research_report.get("symbols_with_price_data"),
            "ready_symbols": stock_research_report.get("ready_symbols"),
            "thin_symbols": stock_research_report.get("thin_symbols"),
            "missing_symbols": stock_research_report.get("missing_symbols"),
            "cache_path": stock_research_report.get("cache_path"),
        },
        "pairs": {
            "lookback": preferred_lookback,
            "summary": pairs_summary,
            "top_pairs": top_pairs,
            "all_pairs": pair_results,
            "pair_symbol_count": len(pair_symbols),
        },
        "ml_alpha": ml_summary,
    }


def _normal_pdf(x: float, mean: float, std: float) -> float:
    std = max(std, 1e-6)
    z = (x - mean) / std
    return math.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))


def option_chain_underlyings() -> list[str]:
    backtest_report = read_json(LATEST_BACKTEST_REPORT, {})
    per_symbol = ((backtest_report.get("option_model_suite") or {}).get("per_symbol") or {})
    symbols = {str(symbol).upper() for symbol in per_symbol.keys() if symbol}

    for row in ledger_option_activity(limit=200):
        underlying = row.get("underlying")
        if underlying:
            symbols.add(str(underlying).upper())

    live_rows, _ = live_option_positions()
    for row in live_rows:
        underlying = row.get("underlying")
        if underlying:
            symbols.add(str(underlying).upper())

    return sorted(symbols) if symbols else list(DEFAULT_OPTION_CHAIN_UNDERLYINGS)


def option_research_rows(underlying: str) -> list[dict[str, Any]]:
    backtest_report = read_json(LATEST_BACKTEST_REPORT, {})
    per_symbol = ((backtest_report.get("option_model_suite") or {}).get("per_symbol") or {})
    payload = per_symbol.get(str(underlying).upper()) or {}
    rows: list[dict[str, Any]] = []
    for option_side, models in payload.items():
        if not isinstance(models, dict):
            continue
        for model_name, stats in models.items():
            if not isinstance(stats, dict):
                continue
            rows.append(
                {
                    "option_side": str(option_side).upper(),
                    "model": str(model_name),
                    "signals": safe_float(stats.get("signals")),
                    "avg_edge_pct": safe_float(stats.get("avg_edge_pct")),
                    "avg_long_pnl": safe_float(stats.get("avg_long_pnl")),
                    "long_win_rate": safe_float(stats.get("long_win_rate")),
                }
            )
    rows.sort(
        key=lambda item: (
            item.get("avg_edge_pct") or -1e9,
            item.get("long_win_rate") or -1e9,
            item.get("signals") or -1e9,
        ),
        reverse=True,
    )
    return rows


def build_options_chain(
    *,
    underlying: str | None = None,
    contract_type: str = "all",
    min_dte: int = 7,
    max_dte: int = 45,
    limit: int = 72,
) -> dict[str, Any]:
    underlyings = option_chain_underlyings()
    selected = str(underlying or (underlyings[0] if underlyings else DEFAULT_OPTION_CHAIN_UNDERLYINGS[0])).upper()
    side_filter = str(contract_type or "all").lower()
    if side_filter not in {"all", "call", "put"}:
        side_filter = "all"

    research_rows = option_research_rows(selected)
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "available": False,
        "reason": None,
        "source": "research_only",
        "underlyings": underlyings,
        "selected_underlying": selected,
        "filters": {
            "contract_type": side_filter,
            "min_dte": int(min_dte),
            "max_dte": int(max_dte),
            "limit": int(limit),
        },
        "summary": {
            "spot_price": None,
            "contract_count": 0,
            "expiry_count": 0,
            "avg_iv": None,
            "atm_iv": None,
            "put_call_iv_gap": None,
            "avg_pricing_confidence": None,
            "avg_spread_pct": None,
            "total_open_interest": None,
        },
        "expiries": [],
        "smile": {"expiry": None, "points": []},
        "chain": [],
        "research_models": research_rows[:10],
    }

    if not cfg.alpaca_api_key or not cfg.alpaca_api_secret:
        payload["reason"] = "alpaca_credentials_unavailable"
        return payload

    try:
        from core.broker_client import BrokerClient
        from alpaca.trading.enums import ContractType
    except Exception as exc:
        payload["reason"] = f"broker_client_import_failed: {exc}"
        return payload

    try:
        client = BrokerClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
        raw_contracts: list[Any] = []
        if side_filter in {"all", "call"}:
            raw_contracts.extend(client.get_options_contracts([selected], ContractType.CALL, min_days=min_dte, max_days=max_dte) or [])
        if side_filter in {"all", "put"}:
            raw_contracts.extend(client.get_options_contracts([selected], ContractType.PUT, min_days=min_dte, max_days=max_dte) or [])

        deduped: dict[str, Any] = {}
        for contract in raw_contracts:
            symbol = str(getattr(contract, "symbol", "") or "")
            if symbol:
                deduped[symbol] = contract
        raw_contracts = list(deduped.values())
        if not raw_contracts:
            payload["reason"] = "no_contracts_found"
            return payload

        snapshots = client.get_option_snapshot([str(getattr(contract, "symbol", "")) for contract in raw_contracts])
        source = "alpaca_delay_adjusted"
        try:
            from core.delay_aware_options import build_delay_adjusted_contracts

            contracts = build_delay_adjusted_contracts(client, raw_contracts, snapshots=snapshots)
        except Exception:
            from models.contract import Contract

            source = "alpaca_snapshot"
            contracts = []
            for contract in raw_contracts:
                snapshot = (snapshots or {}).get(getattr(contract, "symbol", None))
                if snapshot is None:
                    continue
                try:
                    contracts.append(Contract.from_contract_snapshot(contract, snapshot))
                except Exception:
                    continue

        if not contracts:
            payload["reason"] = "no_snapshot_contracts"
            return payload

        rows: list[dict[str, Any]] = []
        for contract in contracts:
            symbol = str(getattr(contract, "symbol", "") or "").upper()
            meta = option_meta(symbol)
            spot = safe_float(getattr(contract, "underlying_price", None)) or safe_float(getattr(contract, "delayed_underlying_price", None))
            bid = safe_float(getattr(contract, "fair_bid_price", None)) or safe_float(getattr(contract, "bid_price", None))
            ask = safe_float(getattr(contract, "fair_ask_price", None)) or safe_float(getattr(contract, "ask_price", None))
            last = safe_float(getattr(contract, "last_price", None))
            fair_value = safe_float(getattr(contract, "fair_value", None))
            mark = fair_value
            if mark is None:
                if bid is not None and ask is not None and bid > 0 and ask > 0:
                    mark = (bid + ask) / 2.0
                else:
                    mark = last
            spread_pct = None
            if bid is not None and ask is not None and mark is not None and mark > 0:
                spread_pct = (ask - bid) / mark
            distance_to_spot_pct = None
            strike = safe_float(getattr(contract, "strike", None))
            if spot is not None and strike is not None and spot > 0:
                distance_to_spot_pct = (strike - spot) / spot
            option_side = str(getattr(contract, "contract_type", "") or "").upper()
            rows.append(
                {
                    "underlying": selected,
                    "symbol": symbol,
                    "option_type": option_side,
                    "expiry": meta.get("expiry"),
                    "dte": safe_float(getattr(contract, "dte", None)),
                    "strike": strike,
                    "underlying_price": spot,
                    "distance_to_spot_pct": distance_to_spot_pct,
                    "delta": safe_float(getattr(contract, "delta", None)),
                    "gamma": safe_float(getattr(contract, "gamma", None)),
                    "theta": safe_float(getattr(contract, "theta", None)),
                    "vega": safe_float(getattr(contract, "vega", None)),
                    "bid_price": bid,
                    "ask_price": ask,
                    "last_price": last,
                    "mark_price": mark,
                    "fair_value": fair_value,
                    "iv": safe_float(getattr(contract, "implied_volatility", None)),
                    "open_interest": safe_float(getattr(contract, "oi", None)),
                    "pricing_confidence": safe_float(getattr(contract, "pricing_confidence", None)),
                    "quote_age_minutes": safe_float(getattr(contract, "quote_age_minutes", None)),
                    "staleness_pct": safe_float(getattr(contract, "staleness_pct", None)),
                    "spread_pct": spread_pct,
                }
            )

        rows.sort(
            key=lambda item: (
                item.get("dte") if item.get("dte") is not None else 1e9,
                abs(item.get("distance_to_spot_pct") or 1e9),
                item.get("strike") if item.get("strike") is not None else 1e9,
                item.get("symbol") or "",
            )
        )
        visible_rows = rows[: max(1, int(limit))]

        expiry_groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            expiry_key = str(row.get("expiry") or "UNSPECIFIED")
            expiry_groups.setdefault(expiry_key, []).append(row)

        expiries: list[dict[str, Any]] = []
        for expiry, group in sorted(
            expiry_groups.items(),
            key=lambda item: min((safe_float(row.get("dte")) or 1e9) for row in item[1]),
        ):
            iv_vals = [safe_float(row.get("iv")) for row in group]
            iv_vals = [val for val in iv_vals if val is not None]
            theta_vals = [safe_float(row.get("theta")) for row in group]
            theta_vals = [val for val in theta_vals if val is not None]
            expiries.append(
                {
                    "expiry": None if expiry == "UNSPECIFIED" else expiry,
                    "count": len(group),
                    "avg_dte": _avg_field(group, "dte"),
                    "avg_iv": None if not iv_vals else float(sum(iv_vals) / len(iv_vals)),
                    "avg_theta": None if not theta_vals else float(sum(theta_vals) / len(theta_vals)),
                    "total_open_interest": float(sum(safe_float(row.get("open_interest")) or 0.0 for row in group)),
                    "call_count": sum(1 for row in group if str(row.get("option_type") or "").upper() == "CALL"),
                    "put_count": sum(1 for row in group if str(row.get("option_type") or "").upper() == "PUT"),
                }
            )

        nearest_expiry = next((row.get("expiry") for row in rows if row.get("expiry")), None)
        smile_points = [
            {
                "strike": row.get("strike"),
                "iv": row.get("iv"),
                "option_type": row.get("option_type"),
                "delta": row.get("delta"),
                "open_interest": row.get("open_interest"),
            }
            for row in rows
            if row.get("expiry") == nearest_expiry and row.get("strike") is not None and row.get("iv") is not None
        ]
        smile_points.sort(key=lambda item: safe_float(item.get("strike")) or 1e9)

        atm_candidates = [row for row in rows if row.get("iv") is not None and row.get("distance_to_spot_pct") is not None]
        atm_candidates.sort(key=lambda row: abs(safe_float(row.get("distance_to_spot_pct")) or 1e9))
        atm_slice = atm_candidates[:4]
        call_ivs = [safe_float(row.get("iv")) for row in rows if str(row.get("option_type") or "").upper() == "CALL" and row.get("iv") is not None]
        put_ivs = [safe_float(row.get("iv")) for row in rows if str(row.get("option_type") or "").upper() == "PUT" and row.get("iv") is not None]

        payload.update(
            {
                "available": True,
                "reason": None,
                "source": source,
                "summary": {
                    "spot_price": safe_float(rows[0].get("underlying_price")) if rows else None,
                    "contract_count": len(rows),
                    "expiry_count": len(expiries),
                    "avg_iv": _avg_field(rows, "iv"),
                    "atm_iv": None if not atm_slice else float(sum(safe_float(row.get("iv")) or 0.0 for row in atm_slice) / len(atm_slice)),
                    "put_call_iv_gap": None if not put_ivs or not call_ivs else (sum(put_ivs) / len(put_ivs)) - (sum(call_ivs) / len(call_ivs)),
                    "avg_pricing_confidence": _avg_field(rows, "pricing_confidence"),
                    "avg_spread_pct": _avg_field(rows, "spread_pct"),
                    "total_open_interest": float(sum(safe_float(row.get("open_interest")) or 0.0 for row in rows)),
                },
                "expiries": expiries[:14],
                "smile": {"expiry": nearest_expiry, "points": smile_points[:60]},
                "chain": visible_rows,
            }
        )
        return payload
    except Exception as exc:
        payload["reason"] = f"options_chain_failed: {exc}"
        return payload


def simulate_trade_profile(
    *,
    symbol: str,
    market: str,
    side: str,
    entry_price: float,
    annual_vol: float,
    stop_loss_price: float,
    take_profit_price: float,
    n_paths: int = 4000,
) -> dict[str, Any]:
    side = "SHORT" if str(side).upper() == "SHORT" else "LONG"
    dt = 1.0 / 252.0
    drift = -0.5 * annual_vol * annual_vol * dt
    vol_step = annual_vol * math.sqrt(dt)

    z = np.random.normal(size=(n_paths, 21))
    log_returns = drift + (vol_step * z)
    price_paths = entry_price * np.exp(np.cumsum(log_returns, axis=1))

    if side == "LONG":
        stop_hit = (price_paths <= stop_loss_price).any(axis=1)
        tp_hit = (price_paths >= take_profit_price).any(axis=1) & ~stop_hit
        exit_prices = np.where(stop_hit, stop_loss_price, np.where(tp_hit, take_profit_price, price_paths[:, -1]))
        pnl = (exit_prices - entry_price) / max(entry_price, 1e-6)
    else:
        stop_hit = (price_paths >= stop_loss_price).any(axis=1)
        tp_hit = (price_paths <= take_profit_price).any(axis=1) & ~stop_hit
        exit_prices = np.where(stop_hit, stop_loss_price, np.where(tp_hit, take_profit_price, price_paths[:, -1]))
        pnl = (entry_price - exit_prices) / max(entry_price, 1e-6)

    prob_profit = float((pnl > 0).mean())
    expected = float(pnl.mean())
    var_95 = float(-np.percentile(pnl, 5))
    cvar_95 = float(-pnl[pnl <= np.percentile(pnl, 5)].mean())
    return {
        "symbol": symbol,
        "market": market,
        "side": side,
        "entry_price": round(entry_price, 4),
        "stop_loss_price": round(stop_loss_price, 4),
        "take_profit_price": round(take_profit_price, 4),
        "annual_vol": round(annual_vol, 4),
        "prob_profit_pct": round(prob_profit * 100.0, 2),
        "expected_pnl_pct": round(expected * 100.0, 2),
        "var_95_pct": round(var_95 * 100.0, 2),
        "cvar_95_pct": round(cvar_95 * 100.0, 2),
    }


def _trade_market(symbol: str, market: Any, meta: dict[str, Any]) -> str:
    raw_market = str(market or meta.get("market") or "").strip().lower()
    if is_option_symbol(symbol):
        return "options"
    if raw_market in {"options", "option"}:
        return "options"
    if raw_market in {"spot", "stocks", "stock", "equity", "equities"}:
        return "stocks"
    if raw_market in {"pairs", "pair"}:
        return "pairs"
    if raw_market in {"futures", "future", "perps", "perp", "crypto"}:
        return "crypto"
    if symbol.endswith("USDT") or symbol.endswith("USD"):
        return "crypto"
    return raw_market or "stocks"


def _trade_annual_vol(symbol: str, market: str, meta: dict[str, Any], fallback: float | None = None) -> float:
    for key in ("annual_vol", "annual_volatility", "volatility", "iv", "implied_volatility", "expected_volatility"):
        value = safe_float(meta.get(key))
        if value is None:
            continue
        normalized = value / 100.0 if value > 5.0 else value
        return max(0.08, min(normalized, 1.8))
    if fallback is not None:
        return max(0.08, min(float(fallback), 1.8))
    if market == "options":
        return 0.70
    if market == "crypto":
        return 0.55
    if market == "pairs":
        return 0.24
    return 0.22


def _trade_barriers(entry_price: float, market: str, side: str, meta: dict[str, Any]) -> tuple[float, float]:
    stop = safe_float(meta.get("stop_loss_price") or meta.get("stop") or meta.get("stop_price"))
    take = safe_float(meta.get("take_profit_price") or meta.get("take_profit") or meta.get("target_price"))
    if stop is not None and take is not None and stop > 0 and take > 0:
        return stop, take

    if market == "options":
        long_stop_mult, long_take_mult = 0.82, 1.28
        short_stop_mult, short_take_mult = 1.35, 0.70
    elif market == "crypto":
        long_stop_mult, long_take_mult = 0.91, 1.14
        short_stop_mult, short_take_mult = 1.09, 0.86
    else:
        long_stop_mult, long_take_mult = 0.94, 1.10
        short_stop_mult, short_take_mult = 1.06, 0.90

    if side == "LONG":
        return entry_price * long_stop_mult, entry_price * long_take_mult
    return entry_price * short_stop_mult, entry_price * short_take_mult


def build_trade_odds() -> dict[str, Any]:
    np.random.seed(21)
    trades: list[dict[str, Any]] = []
    actual_trade_count = 0
    research_overlay_count = 0
    market_counts: dict[str, int] = {}

    runtime_trades = load_runtime_trades()
    open_db_option_symbols = {
        str(row.get("symbol") or "").upper()
        for row in runtime_trades
        if str(row.get("status") or "").lower() == "open" and _trade_market(str(row.get("symbol") or "").upper(), row.get("market"), row.get("meta_payload") or {}) == "options"
    }
    db_symbols = {str(row.get("symbol") or "").upper() for row in runtime_trades if row.get("symbol")}

    for row in runtime_trades:
        symbol = str(row.get("symbol") or "").upper()
        price = safe_float(row.get("price"))
        status = str(row.get("status") or "").lower()
        if not symbol or price is None or price <= 0:
            continue
        if status in {"closed", "cancelled", "canceled", "rejected"}:
            continue
        if not (status == "open" or "filled" in status or "partial" in status):
            continue
        meta = row.get("meta_payload") or {}
        side = normalize_side(row.get("side") or meta.get("position_side"))
        market = _trade_market(symbol, row.get("market"), meta)
        stop, take = _trade_barriers(price, market, side, meta)
        profile = simulate_trade_profile(
            symbol=symbol,
            market=market,
            side=side,
            entry_price=price,
            annual_vol=_trade_annual_vol(symbol, market, meta),
            stop_loss_price=stop,
            take_profit_price=take,
        )
        profile["source"] = "bot_state"
        profile["timestamp"] = row.get("ts")
        profile["strategy"] = row.get("strategy")
        profile["status"] = row.get("status")
        trades.append(profile)
        actual_trade_count += 1
        market_counts[market] = market_counts.get(market, 0) + 1

    live_rows, _ = live_option_positions()
    for row in live_rows:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol or symbol in open_db_option_symbols:
            continue
        price = safe_float(row.get("current_price")) or safe_float(row.get("entry_price"))
        if price is None or price <= 0:
            continue
        side = str(row.get("side") or "LONG").upper()
        stop, take = _trade_barriers(price, "options", side, {"iv": row.get("iv")})
        profile = simulate_trade_profile(
            symbol=symbol,
            market="options",
            side=side,
            entry_price=price,
            annual_vol=_trade_annual_vol(symbol, "options", {"iv": row.get("iv")}, fallback=safe_float(row.get("iv"))),
            stop_loss_price=stop,
            take_profit_price=take,
        )
        profile["source"] = "live_options"
        profile["status"] = "open"
        trades.append(profile)
        actual_trade_count += 1
        market_counts["options"] = market_counts.get("options", 0) + 1

    np.random.seed(None)

    if not trades:
        return {
            "generated_at_utc": utc_now_iso(),
            "curve": [],
            "trades": [],
            "note": "No active confirmed positions were available for odds modelling. Trader Odds now suppresses historical fill replays and research-only placeholders.",
            "summary": {"actual_trade_count": 0, "research_overlay_count": 0, "market_counts": {}},
        }

    xs = np.array([trade["expected_pnl_pct"] for trade in trades], dtype=float)
    spread = np.array([max(1.0, abs(trade["var_95_pct"])) for trade in trades], dtype=float)
    mean = float(xs.mean())
    std = float(max(np.std(xs), spread.mean() / 1.7, 2.0))

    x_grid = np.linspace(mean - 3.5 * std, mean + 3.5 * std, 120)
    curve = [{"x": round(float(x), 3), "y": round(float(_normal_pdf(float(x), mean, std)), 6)} for x in x_grid]

    for trade in trades:
        trade["curve_y"] = round(float(_normal_pdf(float(trade["expected_pnl_pct"]), mean, std)), 6)
        trade["bubble_size"] = round(5.0 + (float(trade["prob_profit_pct"]) / 25.0), 2)

    return {
        "generated_at_utc": utc_now_iso(),
        "distribution": {"mean": round(mean, 3), "std": round(std, 3)},
        "curve": curve,
        "summary": {
            "actual_trade_count": actual_trade_count,
            "research_overlay_count": research_overlay_count,
            "market_counts": market_counts,
        },
        "trades": trades,
    }


def instantiate_strategy(strategy_name: str, params: dict[str, Any] | None = None):
    if strategy_name not in STRATEGY_CLASS_MAP:
        raise ValueError(f"Unsupported strategy '{strategy_name}'")
    module_name, class_name = STRATEGY_CLASS_MAP[strategy_name]
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    sig = inspect.signature(cls)
    accepted = {
        key: value
        for key, value in (params or {}).items()
        if key in sig.parameters
    }
    return cls(**accepted)


def supported_strategy_params(strategy_name: str, base_params: dict[str, Any]) -> list[str]:
    if strategy_name not in STRATEGY_CLASS_MAP:
        return []
    module_name, class_name = STRATEGY_CLASS_MAP[strategy_name]
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    sig = inspect.signature(cls)
    return [
        key for key in base_params.keys()
        if key in sig.parameters and isinstance(base_params.get(key), (int, float))
    ]


def permutation_grid(base_params: dict[str, Any], selected_keys: list[str], max_variants: int = 81) -> list[dict[str, Any]]:
    numeric_keys = [key for key in selected_keys if isinstance(base_params.get(key), (int, float))]
    if not numeric_keys:
        return [dict(base_params)]

    values_by_key: dict[str, list[Any]] = {}
    for key in numeric_keys[:3]:
        value = base_params.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int):
            candidates = sorted({max(1, int(round(value * 0.75))), int(value), max(1, int(round(value * 1.25)))})
        else:
            candidates = sorted({round(float(value) * 0.75, 4), round(float(value), 4), round(float(value) * 1.25, 4)})
        values_by_key[key] = candidates

    combos: list[dict[str, Any]] = []
    keys = list(values_by_key.keys())
    for variant in itertools.product(*(values_by_key[key] for key in keys)):
        payload = dict(base_params)
        payload.update({key: value for key, value in zip(keys, variant)})
        combos.append(payload)
        if len(combos) >= max_variants:
            break
    return combos or [dict(base_params)]
