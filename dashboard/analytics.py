from __future__ import annotations

import importlib
import inspect
import itertools
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from bot.config import cfg

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / ".runtime"
REPORTS_DIR = ROOT / "reports"
LATEST_BACKTEST_REPORT = REPORTS_DIR / "latest_backtest_report.json"
UNIVERSE_REPORT = REPORTS_DIR / "universe_validation_report.json"
RISK_SNAPSHOT_PATH = RUNTIME_DIR / "risk_snapshot.json"
PORTFOLIO_GUARD_PATH = RUNTIME_DIR / "portfolio_risk_guard.json"
EXECUTION_LEDGER_PATH = RUNTIME_DIR / "execution_ledger.json"
EXECUTION_SUMMARY_PATH = RUNTIME_DIR / "execution_quality_snapshot.json"
SYSTEM_RESOURCE_PATH = RUNTIME_DIR / "system_resource_snapshot.json"
BOT_STATE_DB_PATH = RUNTIME_DIR / "bot_state.db"

OPTION_MULTIPLIER = 100.0
DEFAULT_MC_PATHS = 2500
DEFAULT_FORECAST_DAYS = 30
DEFAULT_OPTION_CHAIN_UNDERLYINGS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]

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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        qty = safe_float(record.get("filled_qty") or record.get("qty") or 0.0) or 0.0
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
            return [], {"available": True, "source": "alpaca_live", "count": 0}

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
        return rows, {"available": True, "source": "alpaca_live", "count": len(rows)}
    except Exception as exc:
        return [], {"available": False, "reason": f"alpaca_fetch_failed: {exc}"}


def _sum_field(rows: list[dict[str, Any]], field: str) -> float:
    return float(sum(safe_float(row.get(field)) or 0.0 for row in rows))


def _avg_field(rows: list[dict[str, Any]], field: str) -> float | None:
    vals = [safe_float(row.get(field)) for row in rows]
    vals = [val for val in vals if val is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _preferred_pairs_lookup(backtest_report: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]], dict[str, Any]]:
    pairs_payload = backtest_report.get("pairs_suite") or {}
    lookbacks = pairs_payload.get("results_by_lookback") or {}
    preferred_lookback = "5y" if "5y" in lookbacks else ("10y" if "10y" in lookbacks else next(iter(lookbacks.keys()), None))
    top_pairs: list[dict[str, Any]] = []
    if preferred_lookback:
        top_pairs = (((lookbacks.get(preferred_lookback) or {}).get("summary") or {}).get("top_pairs") or [])
    return preferred_lookback, top_pairs, pairs_payload.get("summary") or {}


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


def build_options_overview() -> dict[str, Any]:
    risk_snapshot = read_json(RISK_SNAPSHOT_PATH, {})
    guard_snapshot = read_json(PORTFOLIO_GUARD_PATH, {})
    exec_summary = read_json(EXECUTION_SUMMARY_PATH, {})
    backtest_report = read_json(LATEST_BACKTEST_REPORT, {})
    live_rows, live_meta = live_option_positions()
    activity_rows = ledger_option_activity()

    rows = live_rows if live_rows else activity_rows
    source = "live_options" if live_rows else ("execution_ledger" if activity_rows else "runtime_snapshot")

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

    movement_summary = ((backtest_report.get("movement_suite") or {}).get("summary") or {})
    ml_summary = ((backtest_report.get("ml_alpha_suite") or {}).get("summary") or {})
    regime_summary = ((backtest_report.get("regime_suite") or {}).get("summary") or {})
    strategy_profile_summary = ((backtest_report.get("strategy_profile_suite") or {}).get("summary") or {})
    option_models = (((backtest_report.get("option_model_suite") or {}).get("summary") or {}).get("models") or {})
    massive_overview = backtest_report.get("massive_overview") or {}
    institutional = backtest_report.get("institutional_robustness") or {}

    preferred_pairs_lookback, _, pairs_summary = _preferred_pairs_lookup(backtest_report)
    risk_engine = (guard_snapshot.get("portfolio_risk_engine") or risk_snapshot.get("portfolio_risk_engine") or {})
    host_metrics = system_snapshot.get("host_metrics") or {}
    memory = host_metrics.get("memory") or {}
    disk = host_metrics.get("disk") or {}

    open_trade_count = sum(1 for row in runtime_trades if str(row.get("status") or "").lower() == "open")
    closed_trade_count = sum(1 for row in runtime_trades if str(row.get("status") or "").lower() == "closed")

    open_positions = safe_int(risk_snapshot.get("open_positions"))
    if open_positions is None:
        open_positions = open_trade_count

    return {
        "generated_at_utc": utc_now_iso(),
        "headline": {
            "total_equity": safe_float(risk_snapshot.get("total_equity")),
            "daily_pnl_pct": safe_float(risk_snapshot.get("daily_pnl_pct")),
            "daily_pnl_dollars": safe_float(risk_snapshot.get("daily_pnl_dollars")),
            "buying_power_budget": safe_float(risk_snapshot.get("buying_power_budget")),
            "vix": safe_float(risk_snapshot.get("vix")),
            "macro_regime": risk_snapshot.get("macro_regime"),
            "macro_confidence": safe_float(risk_snapshot.get("macro_confidence")),
            "movement_bias": risk_snapshot.get("movement_bias"),
            "runtime_profile": risk_snapshot.get("runtime_profile"),
            "runtime_policy_mode": risk_snapshot.get("runtime_policy_mode"),
            "runtime_market_state": risk_snapshot.get("runtime_market_state"),
            "allowed_symbols": safe_int(risk_snapshot.get("allowed_symbols")),
            "open_positions": open_positions,
            "portfolio_delta": safe_float(risk_snapshot.get("portfolio_delta")),
            "portfolio_theta": safe_float(risk_snapshot.get("portfolio_theta")),
            "portfolio_vega": safe_float(risk_snapshot.get("portfolio_vega")),
            "target_delta": safe_float(risk_snapshot.get("target_delta")),
            "target_theta": safe_float(risk_snapshot.get("target_theta")),
            "target_vega": safe_float(risk_snapshot.get("target_vega")),
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
            "predictive_score": safe_float(massive_overview.get("predictive_score")),
            "institutional_score": safe_float(institutional.get("institutional_score")),
            "deployment_tier": institutional.get("deployment_tier") or massive_overview.get("deployment_tier"),
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
    source = "equity_curve"
    if returns.size == 0:
        returns = _proxy_returns_from_snapshot(risk_snapshot)
        source = "runtime_proxy"

    np.random.seed(11)
    mc_paths, mc_terminal = _simulate_forward_paths(base_equity, returns, mode="bootstrap")
    lv_paths, lv_terminal = _simulate_forward_paths(base_equity, returns, mode="permutation")
    np.random.seed(None)

    horizons = list(range(1, (mc_paths.shape[1] if mc_paths.size else DEFAULT_FORECAST_DAYS) + 1))
    mc_prob_profit = float((mc_terminal > base_equity).mean() * 100.0) if mc_terminal.size else None
    lv_prob_profit = float((lv_terminal > base_equity).mean() * 100.0) if lv_terminal.size else None

    return {
        "generated_at_utc": utc_now_iso(),
        "source": source,
        "base_equity": round(base_equity, 2),
        "horizon_days": len(horizons),
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
    movement_results = ((backtest_report.get("movement_suite") or {}).get("results") or [])
    ml_summary = ((backtest_report.get("ml_alpha_suite") or {}).get("summary") or {})

    research_rows = _aggregate_stock_research(movement_results)
    preferred_lookback, top_pairs, pairs_summary = _preferred_pairs_lookup(backtest_report)

    return {
        "generated_at_utc": utc_now_iso(),
        "universe_size": safe_int(universe_report.get("symbols_valid")) or len(universe_report.get("valid_symbols") or []),
        "universe_sample": (universe_report.get("valid_symbols") or [])[:18],
        "research_leaders": research_rows[:60],
        "pairs": {
            "lookback": preferred_lookback,
            "summary": pairs_summary,
            "top_pairs": top_pairs[:20],
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
        if not symbol or price is None or price <= 0:
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

    if not runtime_trades:
        for row in ledger_option_activity(limit=250):
            symbol = str(row.get("symbol") or "").upper()
            price = safe_float(row.get("entry_price"))
            if not symbol or price is None or price <= 0:
                continue
            side = str(row.get("side") or "LONG").upper()
            meta = {"iv": row.get("iv"), "implied_volatility": row.get("iv")}
            stop, take = _trade_barriers(price, "options", side, meta)
            profile = simulate_trade_profile(
                symbol=symbol,
                market="options",
                side=side,
                entry_price=price,
                annual_vol=_trade_annual_vol(symbol, "options", meta, fallback=0.70),
                stop_loss_price=stop,
                take_profit_price=take,
            )
            profile["source"] = "execution_ledger"
            profile["timestamp"] = row.get("filled_at")
            trades.append(profile)
            actual_trade_count += 1
            market_counts["options"] = market_counts.get("options", 0) + 1

    if len(trades) < 12:
        stock_overview = build_stocks_overview()
        needed = max(0, 12 - len(trades))
        for pair in (stock_overview.get("pairs", {}) or {}).get("top_pairs", [])[:needed]:
            corr = safe_float(pair.get("correlation")) or 0.7
            avg_ret = safe_float(pair.get("avg_trade_return")) or 0.01
            win_rate = safe_float(pair.get("win_rate")) or 0.5
            annual_vol = max(0.10, min(0.45, (1.0 - corr) + 0.10))
            profile = simulate_trade_profile(
                symbol=str(pair.get("pair")),
                market="pairs",
                side="LONG",
                entry_price=1.0,
                annual_vol=annual_vol,
                stop_loss_price=0.97,
                take_profit_price=1.0 + max(0.02, min(abs(avg_ret) * 3.0, 0.08)),
            )
            profile["research_win_rate_pct"] = round(win_rate * 100.0, 2)
            profile["correlation"] = round(corr, 3)
            profile["source"] = "pairs_research"
            trades.append(profile)
            research_overlay_count += 1
            market_counts["pairs"] = market_counts.get("pairs", 0) + 1

    np.random.seed(None)

    if not trades:
        return {"generated_at_utc": utc_now_iso(), "curve": [], "trades": [], "summary": {"actual_trade_count": 0, "research_overlay_count": 0, "market_counts": {}}}

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
