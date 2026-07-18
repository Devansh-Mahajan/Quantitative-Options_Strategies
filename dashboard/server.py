"""
Quant Trading Dashboard — FastAPI backend.

Serves on 0.0.0.0:8080 (LAN-accessible via VPN).
Reads live trading state from .runtime/bot_state.db, streams logs,
manages backtest jobs, and exposes all data to the SPA frontend.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import math
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.config import cfg
from bot import state
from dashboard.analytics import (
    build_correlation_payload,
    build_elite_overview,
    build_options_chain,
    build_options_overview,
    build_research_desk,
    build_simulation_payload,
    build_stocks_overview,
    build_trade_odds,
    instantiate_strategy,
    is_option_symbol,
    live_broker_positions,
    live_broker_snapshot,
    permutation_grid,
    supported_strategy_params,
    RUNTIME_DIR,
    RISK_SNAPSHOT_PATH,
    PORTFOLIO_GUARD_PATH,
    read_json,
    safe_float,
    safe_int,
)
from core.telemetry.trade_decision_tape import DEFAULT_DECISION_TAPE_PATH, read_trade_decisions

log = logging.getLogger("dashboard.server")

PORT          = int(cfg.__dict__.get("dashboard_port", 8080) if hasattr(cfg, "__dict__") else 8080)
ROOT          = Path(__file__).resolve().parents[1]
STATIC        = Path(__file__).parent / "static"
LOG_DIR       = cfg.log_dir
BACKTEST_DIR  = ROOT / "backtest_reports"
WF_CONN: set[WebSocket] = set()   # live broadcast connections
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy default parameters (editable from UI)
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_DEFAULTS: dict[str, dict] = {
    "momentum":          {"ema_fast": 9,  "ema_slow": 21, "ema_trend": 55, "vol_ratio": 1.2},
    "mean_reversion":    {"bb_period": 20, "bb_std": 2.0, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70},
    "breakout":          {"donchian_period": 20, "vol_surge": 1.5},
    "statistical_arb":   {"entry_z": 2.2, "exit_z": 0.4, "warmup": 60},
    "cross_sectional_momentum": {"rebal_bars": 4, "long_n": 2, "short_n": 2},
    "tsmom":             {"lookback": 20, "vol_window": 20, "entry_threshold": 0.10},
    "quant_factors":     {"ibs_low": 0.20, "ibs_high": 0.80, "vol_lookback": 48},
    "contrarian_oi":     {"return_bars": 4, "entry_threshold": 0.15},
    "rma_strategy":      {"short_window": 12, "long_window": 48, "entry_k": 1.8},
    "vpin_flow":         {"vpin_high": 0.65, "vpin_low": 0.30, "imbalance_entry": 0.40},
    "knn_predictor":     {"train_window": 500, "pred_horizon": 4, "signal_thresh": 0.0015},
    "pivot_sr":          {"proximity_pct": 0.003, "ibs_confirm": 0.30, "lookback": 24},
    "hp_trend":          {"ma_fast": 8, "ma_slow": 21},
    "carry_portfolio":   {"lookback": 48, "rebal_bars": 8, "long_n": 3, "short_n": 2},
    "momentum_carry_combo": {"rebal_bars": 8, "cov_window": 60, "entry_score": 0.20},
    "order_flow":        {"imbalance_threshold": 0.65, "vwap_threshold": 0.003},
    "microstructure_pressure": {"flow_window": 12, "volume_window": 24, "vwap_window": 16, "pressure_threshold": 0.48, "spread_ceiling_bps": 30.0},
    "pullback_confluence": {"fast_window": 20, "slow_window": 55, "breakout_lookback": 20, "pullback_atr": 1.0, "trend_floor": 0.003},
    # Batch 3 — Avellaneda-Stoikov, Taleb gamma scalping, Gatheral vol surface
    "market_making":    {"gamma": 0.05, "kappa": 1.5, "vol_window": 20, "spread_floor_bps": 5.0, "spread_cap_bps": 30.0, "max_inventory": 3},
    "gamma_scalping":   {"garch_alpha": 0.10, "garch_beta": 0.85, "rv_window": 20, "long_gamma_threshold": 0.15, "short_gamma_threshold": 0.20, "min_iv": 0.08},
    "vol_surface_arb":  {"short_window": 5, "long_window": 60, "ts_lookback_norm": 120, "ts_contango_threshold": 1.25, "ts_backwdn_threshold": 0.80, "skew_zscore_entry": 2.0},
}

# Engine parameter defaults
ENGINE_DEFAULTS = {
    "stop_loss_pct":      0.05,
    "take_profit_pct":    0.10,
    "max_open_positions": 8,
    "lookback":           60,
    "max_risk_per_trade": 0.02,
    "initial_equity":     1027.0,
}

# In-memory job stores
_backtest_jobs: dict[str, dict] = {}
_ft_jobs:       dict[str, dict] = {}
_perm_jobs:     dict[str, dict] = {}
_train_jobs:    dict[str, dict] = {}
_JOB_UNSET = object()

TRAINING_SCRIPTS: dict[str, dict] = {
    "weekend_training":        {"label": "Weekend Pipeline",       "module": "scripts.weekend_training",             "est_min": 60,  "desc": "Full retrain — data download, all models, validation, deploy"},
    "train_hmm":               {"label": "HMM Macro Regime",       "module": "scripts.train_hmm",                    "est_min": 5,   "desc": "Retrain Hidden Markov Model for macro-regime detection"},
    "train_mega_brain":        {"label": "Mega Brain GPU",         "module": "scripts.train_mega_brain",             "est_min": 90,  "desc": "Full GPU-accelerated ensemble (wraps mega_gpu_training)"},
    "train_gpu_brain":         {"label": "GPU Brain",              "module": "scripts.train_gpu_brain",              "est_min": 45,  "desc": "GPU model training pipeline"},
    "train_correlation_alpha": {"label": "Correlation Alpha",      "module": "scripts.train_correlation_alpha",      "est_min": 15,  "desc": "Cross-asset correlation alpha models"},
    "train_regime_movement":   {"label": "Regime + Movement",      "module": "scripts.train_regime_movement_models", "est_min": 20,  "desc": "Regime detection and movement prediction retraining"},
    "weekend_recalibration":   {"label": "Weekend Recalibration",  "module": "scripts.weekend_recalibration",        "est_min": 30,  "desc": "Recalibrate all live strategy parameters from fresh data"},
    "train_xgb_alpha":         {"label": "XGBoost Alpha Engine",   "module": "scripts.train_xgb_alpha",              "est_min": 10,  "desc": "Train XGBoost alpha model on all symbols (Gu, Kelly & Xiu 2020)"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _db_conn() -> sqlite3.Connection:
    c = sqlite3.connect(state._DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _parse_dashboard_ts(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _metric_resample_rule(index: pd.DatetimeIndex) -> str | None:
    if len(index) < 3:
        return None
    span_seconds = max((index[-1] - index[0]).total_seconds(), 0.0)
    diffs = index.to_series().diff().dropna()
    median_seconds = float(diffs.dt.total_seconds().median()) if not diffs.empty else span_seconds

    if span_seconds <= 3 * 24 * 3600 or median_seconds < 10 * 60:
        return "15min"
    if span_seconds <= 31 * 24 * 3600 or median_seconds < 2 * 3600:
        return "1h"
    if span_seconds <= 180 * 24 * 3600:
        return "4h"
    return "1d"


def _prepare_metric_equity_curve(
    equity: list[float],
    timestamps: list[str],
) -> tuple[pd.Series, dict[str, Any]]:
    series = pd.Series(dtype=float)
    meta = {
        "sample_points": 0,
        "effective_points": 0,
        "span_hours": 0.0,
        "resample_rule": None,
        "sufficient_history": False,
    }
    if len(equity) < 2 or len(timestamps) < 2:
        return series, meta

    parsed_ts = [_parse_dashboard_ts(ts) for ts in timestamps]
    rows = [
        (ts, float(value))
        for ts, value in zip(parsed_ts, equity)
        if ts is not None and value is not None
    ]
    if len(rows) < 2:
        return series, meta

    idx = pd.DatetimeIndex([row[0] for row in rows], tz="UTC")
    series = pd.Series([row[1] for row in rows], index=idx).sort_index()
    series = series[series > 0].groupby(level=0).last()
    meta["sample_points"] = int(len(series))
    if len(series) < 2:
        return series, meta

    rule = _metric_resample_rule(series.index)
    if rule:
        series = series.resample(rule).last().dropna()
    reset_events = 0
    if len(series) >= 2:
        pct_moves = series.pct_change().abs().replace([np.inf, -np.inf], np.nan).dropna()
        breaks = pct_moves[pct_moves >= 0.25]
        reset_events = int(len(breaks))
        if len(breaks):
            trimmed = series.loc[breaks.index[-1]:].dropna()
            if len(trimmed) >= 2:
                series = trimmed
    span_hours = max((series.index[-1] - series.index[0]).total_seconds() / 3600.0, 0.0) if len(series) > 1 else 0.0
    meta["effective_points"] = int(len(series))
    meta["span_hours"] = round(float(span_hours), 2)
    meta["resample_rule"] = rule
    meta["sufficient_history"] = bool(len(series) >= 20 and span_hours >= 24 * 5)
    meta["reset_events"] = reset_events
    meta["segment_start_utc"] = series.index[0].isoformat() if len(series) else None
    meta["segment_end_utc"] = series.index[-1].isoformat() if len(series) else None
    return series, meta


def _latest_metric_equity_series(limit: int = 2000) -> tuple[pd.Series, dict[str, Any]]:
    timestamps, equity, _ = _equity_series(limit=limit)
    return _prepare_metric_equity_curve(equity, timestamps)


def _metric_history_notice(meta: dict[str, Any]) -> str:
    span_hours = safe_float(meta.get("span_hours")) or 0.0
    effective_points = safe_int(meta.get("effective_points")) or 0
    reset_events = safe_int(meta.get("reset_events")) or 0
    return (
        "Continuous equity history is not yet stable enough for professional risk analytics. "
        f"Latest clean segment spans {span_hours:.2f}h across {effective_points} points after trimming "
        f"{reset_events} balance-reset events."
    )


def _equity_series(limit: int = 2000) -> tuple[list[str], list[float], list[float]]:
    with _db_conn() as c:
        rows = c.execute(
            "SELECT ts, equity, drawdown FROM equity_curve ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    if not rows:
        return [], [], []
    rows = list(reversed(rows))
    return (
        [str(r["ts"]) for r in rows],
        [float(r["equity"]) for r in rows],
        [float(r["drawdown"]) for r in rows],
    )


def _equity_window(window: str) -> str:
    """Convert window string to ISO timestamp cutoff."""
    now = datetime.now(timezone.utc)
    delta_map = {
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
        "1m": timedelta(days=30),
        "3m": timedelta(days=90),
        "1y": timedelta(days=365),
        "max": timedelta(days=36500),
    }
    delta = delta_map.get(window, timedelta(days=36500))
    return (now - delta).isoformat()


def _coerce_utc_datetime(value: Any, *, default: datetime) -> datetime:
    if value is None:
        return default
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    raw = str(value).strip()
    if not raw:
        return default
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return default
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _sanitize_history_request(
    symbols: list[str],
    interval: str,
    start: str | datetime | None,
    end: str | datetime | None = None,
    *,
    now: datetime | None = None,
) -> tuple[str, str, str | None]:
    """
    Keep dashboard data requests inside the supported stock-history window when
    we have to fall back to Yahoo intraday bars.
    """
    from backtester.data_loader import YF_MAX_WINDOW_DAYS, _alpaca_stock_data_available

    current_time = now or datetime.now(timezone.utc)
    effective_end = _coerce_utc_datetime(end, default=current_time)
    if effective_end > current_time:
        effective_end = current_time
    effective_start = _coerce_utc_datetime(start, default=effective_end - timedelta(days=365))
    notice: str | None = None

    needs_yahoo_intraday_guard = (
        any(cfg.is_stock_symbol(symbol) for symbol in symbols)
        and interval in YF_MAX_WINDOW_DAYS
        and not _alpaca_stock_data_available()
    )
    if needs_yahoo_intraday_guard:
        lookback_days = int(YF_MAX_WINDOW_DAYS[interval])
        cutoff = current_time - timedelta(days=lookback_days)
        if effective_end < cutoff:
            raise ValueError(
                f"{interval} stock history via Yahoo Finance is only available for roughly the last "
                f"{lookback_days} days. Choose a more recent date range or use Alpaca credentials."
            )
        if effective_start < cutoff:
            effective_start = cutoff
            notice = (
                f"Adjusted stock history start to {effective_start.date().isoformat()} because Yahoo Finance "
                f"only retains about {lookback_days} days of {interval} stock bars."
            )

    if effective_start >= effective_end:
        raise ValueError("Start date must be earlier than end date.")

    return (
        effective_start.date().isoformat(),
        effective_end.date().isoformat(),
        notice,
    )


def _rounded(value: Any, digits: int = 4) -> float | None:
    num = safe_float(value)
    return round(num, digits) if num is not None else None


def _compute_metrics(equity: list[float], timestamps: list[str]) -> dict:
    """Compute portfolio performance metrics from equity curve."""
    series, meta = _prepare_metric_equity_curve(equity, timestamps)
    if len(series) < 2:
        return {"metrics_meta": meta} if meta["sample_points"] else {}

    arr = series.to_numpy(dtype=float)
    total_ret = (arr[-1] / arr[0]) - 1.0 if arr[0] != 0 else 0.0
    running_peak = np.maximum.accumulate(arr)
    dd = (arr - running_peak) / (running_peak + 1e-10)
    max_dd = float(dd.min()) if dd.size else 0.0

    result: dict[str, Any] = {
        "total_return_pct": round(total_ret * 100.0, 2),
        "annualised_return_pct": None,
        "sharpe": None,
        "sortino": None,
        "calmar": None,
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "volatility_pct": None,
        "current_equity": round(float(arr[-1]), 2),
        "peak_equity": round(float(arr.max()), 2),
        "metrics_meta": meta,
    }

    if len(series) < 3:
        return result

    rets = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if rets.empty:
        return result

    time_diffs = series.index.to_series().diff().dropna()
    if time_diffs.empty:
        return result
    median_seconds = float(time_diffs.dt.total_seconds().median())
    if median_seconds <= 0:
        return result

    periods_per_year = (365.25 * 24.0 * 3600.0) / median_seconds
    span_seconds = max((series.index[-1] - series.index[0]).total_seconds(), 0.0)
    span_days = span_seconds / 86400.0

    if span_days >= 30 and arr[0] > 0:
        years = max(span_seconds / (365.25 * 24.0 * 3600.0), 1e-9)
        annualised_return = (arr[-1] / arr[0]) ** (1.0 / years) - 1.0
        result["annualised_return_pct"] = round(float(annualised_return) * 100.0, 2)
    else:
        annualised_return = None

    if meta["sufficient_history"]:
        vol = float(rets.std() * math.sqrt(periods_per_year))
        if math.isfinite(vol):
            result["volatility_pct"] = round(vol * 100.0, 2)

        ret_std = float(rets.std())
        if ret_std > 1e-10:
            sharpe = float((rets.mean() * periods_per_year) / (ret_std * math.sqrt(periods_per_year)))
            result["sharpe"] = round(sharpe, 3)

        neg = rets[rets < 0]
        if len(neg) >= 3:
            neg_std = float(neg.std())
            if neg_std > 1e-10:
                sortino = float((rets.mean() * periods_per_year) / (neg_std * math.sqrt(periods_per_year)))
                result["sortino"] = round(sortino, 3)

        if annualised_return is not None and max_dd < -1e-9:
            calmar = float(annualised_return / abs(max_dd))
            result["calmar"] = round(calmar, 3)

    return result


def _live_tape_change(series: pd.Series, delta: timedelta) -> float | None:
    if len(series) < 2:
        return None
    end_ts = series.index[-1]
    cutoff = end_ts - delta
    eligible = series[series.index <= cutoff]
    if eligible.empty:
        return None
    base = float(eligible.iloc[-1])
    current = float(series.iloc[-1])
    if abs(base) < 1e-9:
        return None
    return ((current / base) - 1.0) * 100.0


def _build_live_tape_payload(timestamps: list[str], equity: list[float], *, source: str) -> dict[str, Any]:
    if len(timestamps) < 2 or len(equity) < 2:
        return {
            "source": source,
            "timestamps": timestamps,
            "equity": equity,
            "summary": {},
        }

    rows = [
        (ts, float(value))
        for ts, value in zip((_parse_dashboard_ts(ts) for ts in timestamps), equity)
        if ts is not None and value is not None
    ]
    if len(rows) < 2:
        return {"source": source, "timestamps": timestamps, "equity": equity, "summary": {}}

    idx = pd.DatetimeIndex([row[0] for row in rows], tz="UTC")
    series = pd.Series([row[1] for row in rows], index=idx).sort_index()
    series = series.groupby(level=0).last()
    current = float(series.iloc[-1])
    session_open = float(series.iloc[0])
    session_high = float(series.max())
    session_low = float(series.min())
    session_return_pct = ((current / session_open) - 1.0) * 100.0 if abs(session_open) > 1e-9 else None
    change_1m = _live_tape_change(series, timedelta(minutes=1))
    change_5m = _live_tape_change(series, timedelta(minutes=5))
    change_15m = _live_tape_change(series, timedelta(minutes=15))
    change_60m = _live_tape_change(series, timedelta(hours=1))

    return {
        "source": source,
        "timestamps": [ts.isoformat() for ts in series.index.to_pydatetime()],
        "equity": [round(float(val), 4) for val in series.to_numpy(dtype=float)],
        "summary": {
            "current_equity": round(current, 2),
            "session_open": round(session_open, 2),
            "session_high": round(session_high, 2),
            "session_low": round(session_low, 2),
            "session_return_pct": round(session_return_pct, 2) if session_return_pct is not None else None,
            "change_1m_pct": round(change_1m, 3) if change_1m is not None else None,
            "change_5m_pct": round(change_5m, 3) if change_5m is not None else None,
            "change_15m_pct": round(change_15m, 3) if change_15m is not None else None,
            "change_60m_pct": round(change_60m, 3) if change_60m is not None else None,
            "samples": int(len(series)),
        },
    }


def _histogram(values: list[float] | np.ndarray, bins: int = 20) -> dict[str, list[float] | list[int]]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(arr, bins=bins)
    bin_centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    return {"bins": bin_centers, "counts": counts.tolist()}


def _update_ft_job(
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    progress: float | int | None = None,
    detail: str | None = None,
    result: Any = _JOB_UNSET,
    error: Any = _JOB_UNSET,
) -> None:
    job = _ft_jobs.get(job_id)
    if not job:
        return
    if status is not None:
        job["status"] = status
    if stage is not None:
        job["stage"] = stage
    if progress is not None:
        job["progress"] = round(float(progress), 1)
    if detail is not None:
        job["detail"] = detail
    if result is not _JOB_UNSET:
        job["result"] = result
    if error is not _JOB_UNSET:
        job["error"] = error
    job["updated_at"] = datetime.now(timezone.utc).isoformat()


def _summarize_forwardtest_result(ft_result: Any) -> dict[str, Any]:
    returns_frac = np.asarray(getattr(ft_result, "return_dist", []), dtype=float)
    returns = returns_frac * 100
    sharpes = np.asarray(getattr(ft_result, "sharpe_dist", []), dtype=float)
    maxdds = np.asarray(getattr(ft_result, "max_dd_dist", []), dtype=float) * 100
    win_rates = np.asarray(getattr(ft_result, "win_rate_dist", []), dtype=float) * 100

    def _maybe_stat(values: np.ndarray, reducer, digits: int) -> float | None:
        if values.size == 0:
            return None
        return round(float(reducer(values)), digits)

    profitable_pct = round(float(np.mean(returns_frac > 0)) * 100, 1) if returns_frac.size else None

    return {
        "n_paths":           ft_result.n_paths,
        "failed_paths":      ft_result.failed_paths,
        "acceptance_rate":   round(ft_result.acceptance_rate * 100, 1),
        "median_return_pct": _maybe_stat(returns, np.median, 3),
        "mean_return_pct":   _maybe_stat(returns, np.mean, 3),
        "p5_return_pct":     _maybe_stat(returns, lambda arr: np.percentile(arr, 5), 3),
        "p10_return_pct":    _maybe_stat(returns, lambda arr: np.percentile(arr, 10), 3),
        "p25_return_pct":    _maybe_stat(returns, lambda arr: np.percentile(arr, 25), 3),
        "p75_return_pct":    _maybe_stat(returns, lambda arr: np.percentile(arr, 75), 3),
        "p90_return_pct":    _maybe_stat(returns, lambda arr: np.percentile(arr, 90), 3),
        "p95_return_pct":    _maybe_stat(returns, lambda arr: np.percentile(arr, 95), 3),
        "median_sharpe":     _maybe_stat(sharpes, np.median, 4),
        "p10_sharpe":        _maybe_stat(sharpes, lambda arr: np.percentile(arr, 10), 4),
        "p90_sharpe":        _maybe_stat(sharpes, lambda arr: np.percentile(arr, 90), 4),
        "median_maxdd_pct":  _maybe_stat(maxdds, np.median, 3),
        "p95_maxdd_pct":     _maybe_stat(maxdds, lambda arr: np.percentile(arr, 95), 3),
        "median_win_rate":   _maybe_stat(win_rates, np.median, 1),
        "profitable_pct":    profitable_pct,
        "return_hist":       _histogram(returns),
        "sharpe_hist":       _histogram(sharpes),
        "maxdd_hist":        _histogram(maxdds),
        "winrate_hist":      _histogram(win_rates),
    }


async def _broadcast(data: dict) -> None:
    dead = set()
    for ws in WF_CONN:
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    WF_CONN.difference_update(dead)


async def _poll_exchange_loop() -> None:
    """Background: poll broker for equity every 30s and broadcast to live WS clients."""
    from exchange import get_client
    client = None
    while True:
        try:
            if client is None:
                client = get_client()
                await client.start()

            broker_snapshot = live_broker_snapshot()
            daily_pnl = broker_snapshot.get("day_pnl_dollars") if broker_snapshot.get("available") else None
            daily_pnl_pct = broker_snapshot.get("day_pnl_pct") if broker_snapshot.get("available") else None
            equity = safe_float(broker_snapshot.get("total_equity")) if broker_snapshot.get("available") else None

            if equity is None:
                bal = await client.get_futures_balance()
                account_snapshot = await client.get_account_snapshot() if hasattr(client, "get_account_snapshot") else {}
                equity = safe_float(
                    account_snapshot.get("equity", 0)
                    or bal.get("USDT", 0)
                    or bal.get("USD", 0)
                    or 0
                )
                if daily_pnl is None:
                    daily_pnl = account_snapshot.get("day_pnl")
                if daily_pnl_pct is None:
                    daily_pnl_pct = account_snapshot.get("day_pnl_pct")

            if equity is not None and equity > 0:
                peak_equity = state.get_peak_equity()
                drawdown = ((equity / peak_equity) - 1.0) if peak_equity and peak_equity > 0 else 0.0
                drawdown = min(0.0, drawdown)
                state.record_equity(equity, drawdown)
                pnl_snapshot = state.get_daily_pnl_snapshot()
                await _broadcast(
                    {
                        "type": "equity_tick",
                        "equity": equity,
                        "daily_pnl": _rounded(daily_pnl if daily_pnl is not None else pnl_snapshot.get("daily_pnl"), 4),
                        "daily_pnl_pct": _rounded(daily_pnl_pct if daily_pnl_pct is not None else pnl_snapshot.get("daily_pnl_pct"), 2),
                        "daily_pnl_source": "alpaca_account" if daily_pnl is not None else pnl_snapshot.get("source"),
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                )
        except Exception as exc:
            log.debug("Exchange poll error: %s", exc)
            client = None

        await asyncio.sleep(30)


# ─────────────────────────────────────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    state.init_db()
    log.info("Dashboard starting on port %d", PORT)
    task = asyncio.create_task(_poll_exchange_loop())
    yield
    task.cancel()


app = FastAPI(title="Quant Dashboard", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# REST — General
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(
        str(STATIC / "index.html"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.head("/")
async def root_head():
    return Response(
        status_code=200,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/legacy")
async def legacy_dashboard():
    """Full research dashboard (pre-2026-07 ops-console rebuild)."""
    return FileResponse(
        str(STATIC / "legacy.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/favicon.ico")
@app.get("/apple-touch-icon.png")
@app.get("/apple-touch-icon-precomposed.png")
async def browser_icon_stub():
    return Response(status_code=204)


# ─────────────────────────────────────────────────────────────────────────────
# REST — Ops console (2026-07 rebuild)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/ops/summary")
async def ops_summary():
    """Live account values + freshness of the decision/execution telemetry."""
    snapshot = live_broker_snapshot()

    def _age_seconds(path: Path) -> float | None:
        try:
            if path.exists():
                return round(datetime.now(timezone.utc).timestamp() - path.stat().st_mtime, 1)
        except Exception:
            pass
        return None

    from core.telemetry.trade_decision_tape import DEFAULT_DECISION_TAPE_PATH as _tape_path

    # When broker credentials are absent (e.g. dev machine without .env),
    # fall back to the bot-book equity curve so the console shows something
    # truthful — clearly labeled as book data, not the live account.
    book = None
    if not snapshot or snapshot.get("available") is False:
        try:
            with _db_conn() as conn:
                row = conn.execute(
                    "SELECT ts, equity FROM equity_curve ORDER BY ts DESC LIMIT 1"
                ).fetchone()
            if row:
                book = {"equity": float(row["equity"]), "ts": row["ts"]}
        except Exception:
            book = None

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "account": snapshot,
        "book_fallback": book,
        "freshness": {
            "decision_tape_age_s": _age_seconds(Path(_tape_path)),
            "execution_ledger_age_s": _age_seconds(EXEC_LEDGER_PATH),
            "live_allocation_age_s": _age_seconds(RUNTIME_DIR / "live_allocation.json"),
            "risk_snapshot_age_s": _age_seconds(RUNTIME_DIR / "risk_snapshot.json"),
        },
        "broker": cfg.broker,
        "paper": bool(getattr(cfg, "alpaca_paper", True)),
    }


@app.get("/api/ops/league")
async def ops_league():
    """Measured strategy league joined with each strategy's live enable flag."""
    league = read_json(ROOT / "reports" / "strategy_league_gated.json", {})
    rows = league.get("rows") or []
    enabled_flags = {
        k.replace("enable_", ""): bool(v)
        for k, v in vars(cfg).items()
        if k.startswith("enable_")
    }
    try:
        from strategies.registry import build_registry

        active = {s.name for s in build_registry()}
    except Exception:
        active = set()
    for row in rows:
        name = row.get("strategy")
        row["enabled"] = enabled_flags.get(name, False)
        row["active_on_venue"] = name in active
    return {
        "window": league.get("window"),
        "fees": league.get("fees"),
        "generated_at_utc": league.get("generated_at_utc"),
        "rows": rows,
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "broker": cfg.broker,
        "ts": datetime.now(timezone.utc).isoformat(),
        "strategies_enabled": [
            k.replace("enable_", "") for k, v in vars(cfg).items()
            if k.startswith("enable_") and v
        ],
    }


@app.get("/api/config")
async def get_config():
    return {
        "broker":          cfg.broker,
        "initial_capital": cfg.initial_capital,
        "max_risk":        cfg.max_risk_per_trade,
        "max_leverage":    cfg.max_leverage,
        "max_drawdown":    cfg.max_drawdown,
        "daily_loss_limit":cfg.daily_loss_limit,
        "futures_symbols": cfg.futures_symbols,
        "spot_symbols":    cfg.spot_symbols,
        "stock_symbols":   cfg.stock_symbols,
        "mixed_symbols":   cfg.runtime_symbols,
        "model_symbols":   cfg.model_symbols,
        "risk_symbols":    cfg.risk_symbols,
        "benchmark_symbols": cfg.benchmark_symbols,
        "default_benchmark_symbol": cfg.default_benchmark_symbol,
        "engine_defaults": ENGINE_DEFAULTS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REST — Live Performance
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/equity")
async def get_equity(window: str = Query("1w")):
    cutoff = _equity_window(window)
    with _db_conn() as c:
        rows = c.execute(
            "SELECT ts, equity, drawdown FROM equity_curve WHERE ts >= ? ORDER BY ts",
            (cutoff,),
        ).fetchall()

    if not rows:
        return {"timestamps": [], "equity": [], "drawdown": [], "metrics": {}}

    ts     = [r["ts"] for r in rows]
    equity = [float(r["equity"]) for r in rows]
    clean_series, clean_meta = _prepare_metric_equity_curve(equity, ts)
    if len(clean_series) >= 2:
        clean_equity = [round(float(value), 2) for value in clean_series.to_list()]
        running_peak = np.maximum.accumulate(np.array(clean_equity, dtype=float))
        clean_dd = ((np.array(clean_equity, dtype=float) - running_peak) / (running_peak + 1e-10)).tolist()
        return {
            "timestamps": [idx.isoformat() for idx in clean_series.index],
            "equity":     clean_equity,
            "drawdown":   [float(value) for value in clean_dd],
            "metrics":    _compute_metrics(equity, ts),
            "source":     "clean_equity_segment",
            "raw_points": len(equity),
            "metrics_meta": clean_meta,
        }
    dd     = [float(r["drawdown"]) for r in rows]
    return {
        "timestamps": ts,
        "equity":     equity,
        "drawdown":   dd,
        "metrics":    _compute_metrics(equity, ts),
        "source":     "raw_equity_curve",
    }


@app.get("/api/benchmark")
async def get_benchmark(symbol: str | None = Query(None), window: str = Query("1w")):
    """Return benchmark close prices via backtester data loader."""
    try:
        from backtester.data_loader import load_multi
        benchmark_symbol = symbol or cfg.default_benchmark_symbol
        end   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        delta = {
            "1d": 2, "1w": 8, "1m": 32, "3m": 95, "1y": 366, "max": 1000
        }.get(window, 8)
        start = (datetime.now(timezone.utc) - timedelta(days=delta)).strftime("%Y-%m-%d")
        start, end, notice = _sanitize_history_request([benchmark_symbol], "1h", start, end)
        data  = load_multi([benchmark_symbol], "1h", start, end)
        df    = data.get(benchmark_symbol, pd.DataFrame())
        if df.empty:
            return {"timestamps": [], "prices": [], "notice": notice}
        df = df.reset_index()
        ts_col = "open_time" if "open_time" in df.columns else df.columns[0]
        ts_series = pd.to_datetime(df[ts_col], utc=True)
        return {
            "timestamps": ts_series.dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
            "prices":     df["close"].round(4).tolist(),
            "notice":     notice,
        }
    except Exception as exc:
        log.warning("Benchmark fetch failed: %s", exc)
        return {"timestamps": [], "prices": []}


@app.get("/api/positions")
async def get_positions():
    broker_positions, broker_meta = live_broker_positions()
    trades = state.get_open_trades()
    db_positions = []
    for t in trades:
        side = t.get("side", "")
        db_positions.append({
            **t,
            "asset_class":    t.get("market") or "runtime_db",
            "entry_price":    t.get("price"),
            "opened_at":      t.get("ts"),
            "side":           "LONG" if side in ("BUY", "LONG") else "SHORT" if side in ("SELL", "SHORT") else side,
            "unrealized_pnl": t.get("pnl", 0),
            "source":         "runtime_db",
        })

    if broker_positions:
        seen = {
            (
                str(pos.get("broker_symbol") or pos.get("symbol") or "").upper(),
                str(pos.get("asset_class") or "").lower(),
            )
            for pos in broker_positions
        }
        for pos in db_positions:
            key = (
                str(pos.get("broker_symbol") or pos.get("symbol") or "").upper(),
                str(pos.get("asset_class") or pos.get("market") or "").lower(),
            )
            if key not in seen:
                broker_positions.append(pos)
        broker_positions.sort(key=lambda row: abs(safe_float(row.get("market_value")) or safe_float(row.get("unrealized_pnl")) or 0.0), reverse=True)
        return {
            "positions": broker_positions,
            "count": len(broker_positions),
            "source": broker_meta.get("source"),
            "summary": broker_meta,
        }

    return {
        "positions": db_positions,
        "count": len(db_positions),
        "source": "runtime_db",
        "summary": {"available": False, "reason": broker_meta.get("reason", "runtime_db_only")},
    }


@app.get("/api/live/tape")
async def live_tape(limit: int = Query(360, ge=30, le=2000)):
    broker_snapshot = live_broker_snapshot()
    if broker_snapshot.get("available"):
        timestamps = list(broker_snapshot.get("tape_timestamps") or [])
        equity = [float(v) for v in list(broker_snapshot.get("tape_equity") or [])]
        latest_equity_ts = broker_snapshot.get("latest_equity_ts")
        latest_equity = safe_float(broker_snapshot.get("total_equity"))
        if latest_equity_ts and latest_equity is not None:
            if timestamps and timestamps[-1] == latest_equity_ts:
                equity[-1] = latest_equity
            else:
                timestamps.append(str(latest_equity_ts))
                equity.append(float(latest_equity))
        timestamps = timestamps[-limit:]
        equity = equity[-limit:]
        payload = _build_live_tape_payload(timestamps, equity, source=str(broker_snapshot.get("source") or "alpaca_account"))
        payload["account"] = {
            "total_equity": _rounded(broker_snapshot.get("total_equity"), 2),
            "cash": _rounded(broker_snapshot.get("cash"), 2),
            "buying_power": _rounded(broker_snapshot.get("buying_power"), 2),
            "day_pnl_dollars": _rounded(broker_snapshot.get("day_pnl_dollars"), 2),
            "day_pnl_pct": _rounded(broker_snapshot.get("day_pnl_pct"), 2),
            "total_return_pct": _rounded(broker_snapshot.get("total_return_pct"), 2),
            "all_time_high_equity": _rounded(broker_snapshot.get("all_time_high_equity"), 2),
            "positions_count": broker_snapshot.get("positions_count"),
            "account_status": broker_snapshot.get("account_status"),
        }
        return payload

    with _db_conn() as c:
        rows = c.execute(
            "SELECT ts, equity FROM equity_curve ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    rows = list(reversed(rows))
    ts = [str(row["ts"]) for row in rows]
    eq = [float(row["equity"]) for row in rows]
    payload = _build_live_tape_payload(ts, eq, source="runtime_equity_curve")
    payload["account"] = {}
    return payload


@app.get("/api/trades")
async def get_trades(
    start: str = Query(None),
    end:   str = Query(None),
    download: bool = Query(False),
    limit: int = Query(500),
):
    cutoff_start = start or (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    cutoff_end   = end   or datetime.now(timezone.utc).isoformat()

    with _db_conn() as c:
        rows = c.execute(
            "SELECT * FROM trades WHERE ts >= ? AND ts <= ? ORDER BY ts DESC LIMIT ?",
            (cutoff_start, cutoff_end, limit),
        ).fetchall()
    trades = [dict(r) for r in rows]

    if download:
        def _gen():
            buf = io.StringIO()
            if trades:
                w = csv.DictWriter(buf, fieldnames=trades[0].keys())
                w.writeheader()
                for t in trades:
                    w.writerow(t)
            yield buf.getvalue()
        return StreamingResponse(
            _gen(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=trades_{cutoff_start[:10]}_{cutoff_end[:10]}.csv"},
        )

    return {"trades": trades, "count": len(trades)}


@app.get("/api/daily_pnl")
async def get_daily_pnl():
    snapshot = state.get_daily_pnl_snapshot()
    broker_snapshot = live_broker_snapshot()
    if broker_snapshot.get("available"):
        snapshot = {
            **snapshot,
            "source": broker_snapshot.get("source"),
            "daily_pnl": broker_snapshot.get("day_pnl_dollars"),
            "daily_pnl_pct": broker_snapshot.get("day_pnl_pct"),
            "session_start_equity": broker_snapshot.get("last_equity") or broker_snapshot.get("session_start_equity"),
            "current_equity": broker_snapshot.get("total_equity"),
            "latest_equity_ts": broker_snapshot.get("latest_equity_ts"),
            "intraday_low_equity": broker_snapshot.get("session_low_equity"),
            "intraday_high_equity": broker_snapshot.get("session_high_equity"),
            "intraday_range_pct": (
                ((safe_float(broker_snapshot.get("session_high_equity")) - safe_float(broker_snapshot.get("session_low_equity"))) / safe_float(broker_snapshot.get("session_start_equity")) * 100.0)
                if safe_float(broker_snapshot.get("session_high_equity")) is not None
                and safe_float(broker_snapshot.get("session_low_equity")) is not None
                and safe_float(broker_snapshot.get("session_start_equity"))
                else snapshot.get("intraday_range_pct")
            ),
            "equity_freshness_seconds": (
                max(
                    0.0,
                    (datetime.now(timezone.utc) - _parse_dashboard_ts(broker_snapshot.get("latest_equity_ts"))).total_seconds(),
                )
                if _parse_dashboard_ts(broker_snapshot.get("latest_equity_ts"))
                else snapshot.get("equity_freshness_seconds")
            ),
        }
    peak = snapshot.get("peak_equity")
    if peak is None:
        peak = state.get_peak_equity()
    return {
        **snapshot,
        "daily_pnl": _rounded(snapshot.get("daily_pnl"), 4) or 0.0,
        "daily_pnl_pct": _rounded(snapshot.get("daily_pnl_pct"), 2),
        "session_start_equity": _rounded(snapshot.get("session_start_equity"), 2),
        "current_equity": _rounded(snapshot.get("current_equity"), 2),
        "intraday_low_equity": _rounded(snapshot.get("intraday_low_equity"), 2),
        "intraday_high_equity": _rounded(snapshot.get("intraday_high_equity"), 2),
        "intraday_range_pct": _rounded(snapshot.get("intraday_range_pct"), 2),
        "peak_equity": _rounded(peak, 2),
        "distance_to_peak_pct": _rounded(snapshot.get("distance_to_peak_pct"), 2),
        "equity_freshness_seconds": _rounded(snapshot.get("equity_freshness_seconds"), 2),
        "closed_trade_pnl_today": _rounded(snapshot.get("closed_trade_pnl_today"), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# REST — Risk
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/risk/snapshot")
async def risk_snapshot():
    """Compute risk metrics from recent equity history."""
    series, meta = _latest_metric_equity_series(limit=8760)
    if len(series) < 2:
        return {
            "available": False,
            "var_95_pct": None,
            "var_99_pct": None,
            "cvar_95_pct": None,
            "cvar_99_pct": None,
            "current_drawdown_pct": None,
            "max_drawdown_pct": None,
            "rolling_vol_pct": None,
            "drawdown_series": [],
            "metrics_meta": meta,
            "notice": "Risk analytics are waiting for more equity observations.",
        }

    if not meta.get("sufficient_history"):
        return {
            "available": False,
            "var_95_pct": None,
            "var_99_pct": None,
            "cvar_95_pct": None,
            "cvar_99_pct": None,
            "current_drawdown_pct": None,
            "max_drawdown_pct": None,
            "rolling_vol_pct": None,
            "drawdown_series": [],
            "metrics_meta": meta,
            "notice": _metric_history_notice(meta),
        }

    arr = series.to_numpy(dtype=float)
    rets = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if rets.size == 0:
        rets = np.array([0.0], dtype=float)

    # VaR / CVaR
    var_95  = float(np.percentile(rets, 5))
    var_99  = float(np.percentile(rets, 1))
    cvar_95 = float(rets[rets <= var_95].mean()) if (rets <= var_95).any() else var_95
    cvar_99 = float(rets[rets <= var_99].mean()) if (rets <= var_99).any() else var_99

    # Drawdown series
    peak = np.maximum.accumulate(arr)
    dd   = ((arr - peak) / (peak + 1e-10)).tolist()

    # Rolling 30-day vol (annualised)
    rolling_vol = []
    time_diffs = series.index.to_series().diff().dropna()
    median_seconds = float(time_diffs.dt.total_seconds().median()) if not time_diffs.empty else 3600.0
    periods_per_year = max((365.25 * 24.0 * 3600.0) / max(median_seconds, 1.0), 1.0)
    window = min(90, len(rets))
    for i in range(window, len(rets) + 1):
        rolling_vol.append(float(rets[i - window: i].std() * np.sqrt(periods_per_year) * 100))

    return {
        "available": True,
        "var_95_pct":  round(var_95 * 100, 3),
        "var_99_pct":  round(var_99 * 100, 3),
        "cvar_95_pct": round(cvar_95 * 100, 3),
        "cvar_99_pct": round(cvar_99 * 100, 3),
        "current_drawdown_pct": round(dd[-1] * 100, 2) if dd else 0,
        "max_drawdown_pct":     round(min(dd) * 100, 2) if dd else 0,
        "rolling_vol_pct":      rolling_vol[-1] if rolling_vol else 0,
        "drawdown_series":      [round(x * 100, 3) for x in dd[-720:]],
        "metrics_meta":         meta,
        "notice":               None,
    }


@app.get("/api/risk/heatmap")
async def risk_heatmap():
    payload = await run_in_threadpool(build_correlation_payload, False)
    crypto = payload.get("crypto") or {}
    return {
        "symbols": crypto.get("symbols") or [],
        "corr": crypto.get("corr") or [],
        "vols": crypto.get("vols") or {},
        "peer_map": crypto.get("peer_map") or {},
        "leaders": crypto.get("leaders") or {"positive": [], "negative": []},
        "stats": crypto.get("stats") or {},
        "universe_symbols": crypto.get("symbols") or [],
        "options_greeks": payload.get("options_greeks") or {},
        "stress_scenarios": payload.get("stress_scenarios") or {},
    }


@app.get("/api/risk/correlations")
async def risk_correlations(refresh: bool = Query(False)):
    return await run_in_threadpool(build_correlation_payload, bool(refresh))


@app.get("/api/risk/var_surface")
async def var_surface():
    """VaR surface at multiple confidence levels and horizons."""
    series, meta = _latest_metric_equity_series(limit=2000)
    if len(series) < 30 or not meta.get("sufficient_history"):
        return {
            "available": False,
            "levels": [],
            "horizons": [],
            "var": [],
            "cvar": [],
            "metrics_meta": meta,
            "notice": _metric_history_notice(meta) if len(series) >= 2 else "VaR surface is waiting for more equity observations.",
        }

    equity = series.to_numpy(dtype=float)
    rets   = np.diff(np.log(equity))

    levels  = [0.90, 0.95, 0.99, 0.999]
    horizons = [1, 5, 10, 21]
    var_grid  = []
    cvar_grid = []

    for h in horizons:
        var_row  = []
        cvar_row = []
        for lvl in levels:
            # Scale 1-day VaR to h-day via sqrt(h)
            v    = float(np.percentile(rets, (1 - lvl) * 100)) * np.sqrt(h)
            tail = rets[rets <= np.percentile(rets, (1 - lvl) * 100)]
            cv   = float(tail.mean() * np.sqrt(h)) if len(tail) else v
            var_row.append(round(v * 100, 3))
            cvar_row.append(round(cv * 100, 3))
        var_grid.append(var_row)
        cvar_grid.append(cvar_row)

    return {
        "available": True,
        "levels":   [f"{int(l*100)}%" for l in levels],
        "horizons": [f"{h}d" for h in horizons],
        "var":      var_grid,
        "cvar":     cvar_grid,
        "metrics_meta": meta,
        "notice": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REST — Risk Engine (live portfolio guard snapshot)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/risk/engine")
async def risk_engine_snapshot():
    """
    Return the live risk engine state directly from runtime files.
    No heavy computation — just JSON reads. Safe to poll every 60s.
    """
    risk_snap  = read_json(RISK_SNAPSHOT_PATH, {})
    guard_snap = read_json(PORTFOLIO_GUARD_PATH, {})
    eng = (guard_snap.get("portfolio_risk_engine") or risk_snap.get("portfolio_risk_engine") or {})
    risk_generated_at = _parse_dashboard_ts(risk_snap.get("generated_at_utc"))
    risk_age_hours = (
        max(0.0, (datetime.now(timezone.utc) - risk_generated_at).total_seconds() / 3600.0)
        if risk_generated_at
        else None
    )
    risk_snapshot_fresh = risk_age_hours is not None and risk_age_hours <= 6.0
    trading_status = "HALTED" if bool(eng.get("kill_switch_active")) else ("LIVE" if risk_snapshot_fresh else "RISK STALE")
    return {
        "generated_at_utc":            datetime.now(timezone.utc).isoformat(),
        "portfolio_delta":             safe_float(risk_snap.get("portfolio_delta")),
        "portfolio_theta":             safe_float(risk_snap.get("portfolio_theta")),
        "portfolio_vega":              safe_float(risk_snap.get("portfolio_vega")),
        "portfolio_gamma":             safe_float(risk_snap.get("portfolio_gamma")),
        "target_delta":                safe_float(risk_snap.get("target_delta")),
        "target_theta":                safe_float(risk_snap.get("target_theta")),
        "target_vega":                 safe_float(risk_snap.get("target_vega")),
        "risk_score":                  safe_float(eng.get("risk_score")),
        "var_pct_equity":              safe_float(eng.get("var_pct_equity")),
        "cvar_pct_equity":             safe_float(eng.get("cvar_pct_equity")),
        "stress_pct_equity":           safe_float(eng.get("stress_pct_equity")),
        "gross_exposure_pct_equity":   safe_float(eng.get("gross_exposure_pct_equity")),
        "net_delta_exposure":          safe_float(eng.get("net_delta_exposure")),
        "correlation_concentration":   safe_float(eng.get("correlation_concentration")),
        "max_underlying_weight":       safe_float(eng.get("max_underlying_weight")),
        "value_volatility":            safe_float(eng.get("value_volatility")),
        "simulation_paths":            safe_int(eng.get("simulation_paths")),
        "kill_switch_active":          bool(eng.get("kill_switch_active")),
        "underlying_count":            safe_int(eng.get("underlying_count")),
        "breaches":                    list(eng.get("breaches") or []),
        "hard_kill_reasons":           list(eng.get("hard_kill_reasons") or []),
        "top_underlyings":             eng.get("top_underlyings") or [],
        "macro_regime":                risk_snap.get("macro_regime"),
        "movement_bias":               risk_snap.get("movement_bias"),
        "runtime_policy_mode":         risk_snap.get("runtime_policy_mode"),
        "vix":                         safe_float(risk_snap.get("vix")),
        "open_positions":              safe_int(risk_snap.get("open_positions")),
        "buying_power":                safe_float(risk_snap.get("buying_power_budget")),
        "total_equity":                safe_float(risk_snap.get("total_equity")),
        "daily_pnl_pct":               safe_float(risk_snap.get("daily_pnl_pct")),
        "allowed_symbols":             safe_int(risk_snap.get("allowed_symbols")),
        "risk_snapshot_age_hours":     round(float(risk_age_hours), 2) if risk_age_hours is not None else None,
        "risk_snapshot_fresh":         risk_snapshot_fresh,
        "trading_status":              trading_status,
    }


@app.get("/api/strategies/pnl")
async def strategies_pnl():
    """Per-strategy realized P&L from the trades DB."""
    with _db_conn() as c:
        try:
            rows = c.execute(
                "SELECT strategy, pnl, ts FROM trades WHERE status='closed' ORDER BY ts DESC LIMIT 3000"
            ).fetchall()
        except Exception:
            return {"strategies": [], "total_pnl": 0}

    agg: dict[str, dict] = {}
    for row in rows:
        name = row["strategy"] or "unknown"
        pnl  = float(row["pnl"] or 0)
        if name not in agg:
            agg[name] = {"total_pnl": 0.0, "trades": 0, "wins": 0, "losses": 0}
        agg[name]["total_pnl"] += pnl
        agg[name]["trades"]   += 1
        if pnl > 0:
            agg[name]["wins"]   += 1
        elif pnl < 0:
            agg[name]["losses"] += 1

    result = []
    for name, d in sorted(agg.items(), key=lambda x: x[1]["total_pnl"], reverse=True):
        t = max(d["trades"], 1)
        result.append({
            "strategy":     name,
            "total_pnl":    round(d["total_pnl"], 2),
            "trades":       d["trades"],
            "wins":         d["wins"],
            "losses":       d["losses"],
            "win_rate_pct": round(d["wins"] / t * 100, 1),
        })

    return {
        "strategies": result,
        "total_pnl":  round(sum(s["total_pnl"] for s in result), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# REST — Elite dashboard analytics
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/elite/overview")
async def elite_overview():
    return await run_in_threadpool(build_elite_overview)


@app.get("/api/options/overview")
async def options_overview():
    return await run_in_threadpool(build_options_overview)


@app.get("/api/options/chain")
async def options_chain(
    underlying: str | None = Query(None),
    contract_type: str = Query("all"),
    min_dte: int = Query(7, ge=1, le=365),
    max_dte: int = Query(45, ge=1, le=365),
    limit: int = Query(72, ge=12, le=200),
):
    return await run_in_threadpool(
        lambda: build_options_chain(
            underlying=underlying,
            contract_type=contract_type,
            min_dte=min_dte,
            max_dte=max_dte,
            limit=limit,
        )
    )


@app.get("/api/options/simulations")
async def options_simulations():
    series, _meta = _latest_metric_equity_series(limit=1200)
    if len(series) >= 2:
        timestamps = [ts.isoformat() for ts in series.index]
        equity = [float(value) for value in series.to_list()]
    else:
        timestamps, equity, _ = _equity_series(limit=1200)
    return await run_in_threadpool(build_simulation_payload, equity, timestamps)


@app.get("/api/stocks/overview")
async def stocks_overview():
    return await run_in_threadpool(build_stocks_overview)


@app.get("/api/trades/odds")
async def trade_odds():
    return await run_in_threadpool(build_trade_odds)


@app.get("/api/research/desk")
async def research_desk(refresh: bool = Query(False)):
    return await run_in_threadpool(build_research_desk, bool(refresh))


@app.get("/api/permutations/strategies")
async def permutation_strategies():
    payload = []
    for name, params in STRATEGY_DEFAULTS.items():
        try:
            supported = supported_strategy_params(name, params)
        except Exception:
            supported = []
        payload.append({
            "strategy": name,
            "supported_params": supported,
        })
    return {"strategies": payload}


# ─────────────────────────────────────────────────────────────────────────────
# REST — Backtest
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/backtest/list")
async def backtest_list():
    report_dir = BACKTEST_DIR
    results = []
    if report_dir.exists():
        for f in sorted(report_dir.glob("backtest_*.json"), reverse=True)[:20]:
            try:
                data = json.loads(f.read_text())
                results.append({
                    "run_id":   f.stem.replace("backtest_", ""),
                    "file":     f.name,
                    "total_return_pct": data.get("metrics", {}).get("total_return_pct"),
                    "sharpe":           data.get("metrics", {}).get("sharpe"),
                    "max_drawdown_pct": data.get("metrics", {}).get("max_drawdown_pct"),
                    "symbols":          data.get("config", {}).get("symbols", []),
                    "strategies":       data.get("config", {}).get("strategies", []),
                    "trades":           data.get("metrics", {}).get("total_trades"),
                })
            except Exception:
                pass
    return {"reports": results}


@app.get("/api/backtest/{run_id}")
async def backtest_detail(run_id: str):
    path = BACKTEST_DIR / f"backtest_{run_id}.json"
    if not path.exists():
        raise HTTPException(404, "Report not found")
    return json.loads(path.read_text())


@app.post("/api/backtest/run")
async def run_backtest(body: dict):
    """
    Submit a new backtest job. Runs in background thread.
    Body: {symbols, interval, start, end, strategies, capital, stop_loss, take_profit, ...}
    """
    job_id = str(uuid.uuid4())[:8]
    _backtest_jobs[job_id] = {"status": "PENDING", "progress": 0, "result": None, "error": None}

    def _run():
        try:
            _backtest_jobs[job_id]["status"] = "RUNNING"
            from backtester.data_loader import load_multi, align_and_ffill
            from backtester.engine import BacktestEngine
            from backtester.report import save_report

            symbols    = body.get("symbols", cfg.model_symbols[:4])
            interval   = body.get("interval", "1h")
            start, end, range_notice = _sanitize_history_request(
                list(symbols),
                interval,
                body.get("start", "2024-01-01"),
                body.get("end"),
            )
            strategies = body.get("strategies", ["momentum", "breakout"])
            capital    = float(body.get("capital", cfg.initial_capital))
            sl         = float(body.get("stop_loss_pct", ENGINE_DEFAULTS["stop_loss_pct"]))
            tp         = float(body.get("take_profit_pct", ENGINE_DEFAULTS["take_profit_pct"]))
            max_pos    = int(body.get("max_open_positions", ENGINE_DEFAULTS["max_open_positions"]))
            lookback   = int(body.get("lookback", ENGINE_DEFAULTS["lookback"]))
            _backtest_jobs[job_id]["range_notice"] = range_notice

            data = load_multi(symbols, interval, start, end)
            data = {s: df for s, df in data.items() if not df.empty}
            if not data:
                raise ValueError("No backtest data available for the selected symbols/date range.")
            data = align_and_ffill(data)

            # Build strategies
            from scripts.run_backtest import _build_strategies
            strat_list = _build_strategies(strategies)

            engine = BacktestEngine(
                data=data, strategies=strat_list, interval=interval,
                initial_equity=capital, lookback=lookback,
                stop_loss_pct=sl, take_profit_pct=tp,
                max_open_positions=max_pos,
            )
            result = engine.run()
            path   = save_report(result, run_id=f"ui_{job_id}")
            _backtest_jobs[job_id]["status"]  = "DONE"
            _backtest_jobs[job_id]["result"]  = str(path)
            _backtest_jobs[job_id]["metrics"] = {
                "total_return_pct": result.metrics.total_return_pct,
                "sharpe": result.metrics.sharpe,
                "max_drawdown_pct": result.metrics.max_drawdown_pct,
                "win_rate_pct": result.metrics.win_rate_pct,
                "profit_factor": result.metrics.profit_factor,
                "trades": len(result.trades),
            }
            _backtest_jobs[job_id]["equity"] = result.equity_curve.values.tolist()
            _backtest_jobs[job_id]["timestamps"] = [str(t) for t in result.equity_curve.index]
            _backtest_jobs[job_id]["per_strategy"] = {
                s: {
                    "sharpe": m.sharpe,
                    "total_return_pct": m.total_return_pct,
                    "max_drawdown_pct": m.max_drawdown_pct,
                    "win_rate_pct": m.win_rate_pct,
                    "profit_factor": m.profit_factor,
                }
                for s, m in result.per_strategy_metrics.items()
            }
        except Exception as exc:
            import traceback
            _backtest_jobs[job_id]["status"] = "ERROR"
            _backtest_jobs[job_id]["error"]  = str(exc)
            log.error("Backtest job %s failed: %s\n%s", job_id, exc, traceback.format_exc())

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "PENDING"}


@app.get("/api/backtest/jobs/{job_id}")
async def backtest_job_status(job_id: str):
    job = _backtest_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


# ─────────────────────────────────────────────────────────────────────────────
# REST — Strategy Parameters
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/strategies/params")
async def get_strategy_params():
    """Return current params for all strategies (defaults + any overrides stored in kv)."""
    result = {}
    for name, defaults in STRATEGY_DEFAULTS.items():
        override = state.kv_get(f"strategy_params_{name}", {})
        result[name] = {**defaults, **override}
    return {"strategies": result, "engine": ENGINE_DEFAULTS}


@app.post("/api/strategies/{name}/params")
async def set_strategy_params(name: str, body: dict):
    if name not in STRATEGY_DEFAULTS:
        raise HTTPException(404, f"Strategy '{name}' not found")
    state.kv_set(f"strategy_params_{name}", body)
    return {"ok": True, "name": name, "params": body}


@app.post("/api/strategies/{name}/reset")
async def reset_strategy_params(name: str):
    if name not in STRATEGY_DEFAULTS:
        raise HTTPException(404, f"Strategy '{name}' not found")
    state.kv_set(f"strategy_params_{name}", {})
    return {"ok": True, "name": name, "params": STRATEGY_DEFAULTS[name]}


@app.get("/api/strategies/defaults")
async def get_strategy_defaults():
    return {"strategies": STRATEGY_DEFAULTS, "engine": ENGINE_DEFAULTS}


# ─────────────────────────────────────────────────────────────────────────────
# REST — Strategy Command Center (enable/disable, kill switch, universe)
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_enabled_key(name: str) -> str:
    return f"strategy_enabled_{name}"


@app.get("/api/strategies/status")
async def get_strategies_status():
    """Return enabled state and current params for every known strategy."""
    result = {}
    for name, defaults in STRATEGY_DEFAULTS.items():
        override   = state.kv_get(f"strategy_params_{name}", {})
        # Enabled state defaults to True (matches config defaults) unless overridden
        enabled    = state.kv_get(_strategy_enabled_key(name), True)
        result[name] = {
            "enabled": bool(enabled),
            "params":  {**defaults, **override},
        }
    return {
        "strategies": result,
        "kill_switch": bool(state.kv_get("global_kill_switch", False)),
    }


@app.post("/api/strategies/{name}/enable")
async def enable_strategy(name: str):
    if name not in STRATEGY_DEFAULTS:
        raise HTTPException(404, f"Strategy '{name}' not found")
    state.kv_set(_strategy_enabled_key(name), True)
    log.info("Dashboard: strategy '%s' ENABLED", name)
    return {"ok": True, "name": name, "enabled": True}


@app.post("/api/strategies/{name}/disable")
async def disable_strategy(name: str):
    if name not in STRATEGY_DEFAULTS:
        raise HTTPException(404, f"Strategy '{name}' not found")
    state.kv_set(_strategy_enabled_key(name), False)
    log.info("Dashboard: strategy '%s' DISABLED", name)
    return {"ok": True, "name": name, "enabled": False}


@app.post("/api/trading/kill_switch")
async def toggle_kill_switch(body: dict):
    """Activate or deactivate the global kill switch. Body: {active: bool}"""
    active = bool(body.get("active", True))
    state.kv_set("global_kill_switch", active)
    log.warning("Dashboard: global kill switch set to %s", active)
    return {"ok": True, "kill_switch_active": active}


@app.get("/api/config/universe")
async def get_universe():
    """Return current symbol universes (overridable from dashboard)."""
    futures  = state.kv_get("universe_futures",  cfg.futures_symbols)
    spot     = state.kv_get("universe_spot",     cfg.spot_symbols)
    stocks   = state.kv_get("universe_stocks",   cfg.stock_symbols)
    return {
        "futures": futures,
        "spot":    spot,
        "stocks":  stocks,
        "defaults": {
            "futures": cfg.futures_symbols,
            "spot":    cfg.spot_symbols,
            "stocks":  cfg.stock_symbols,
        },
    }


@app.post("/api/config/universe")
async def set_universe(body: dict):
    """Update symbol universe. Body: {futures: [...], spot: [...], stocks: [...]}"""
    if "futures" in body:
        syms = [s.strip().upper() for s in body["futures"] if str(s).strip()]
        state.kv_set("universe_futures", syms)
    if "spot" in body:
        syms = [s.strip().upper() for s in body["spot"] if str(s).strip()]
        state.kv_set("universe_spot", syms)
    if "stocks" in body:
        syms = [s.strip().upper() for s in body["stocks"] if str(s).strip()]
        state.kv_set("universe_stocks", syms)
    return {"ok": True, "universe": await get_universe()}


@app.post("/api/config/universe/reset")
async def reset_universe():
    """Reset symbol universe to config defaults."""
    state.kv_set("universe_futures", cfg.futures_symbols)
    state.kv_set("universe_spot",    cfg.spot_symbols)
    state.kv_set("universe_stocks",  cfg.stock_symbols)
    return {"ok": True, "universe": await get_universe()}


@app.post("/api/config/risk")
async def set_risk_config(body: dict):
    """Update runtime risk parameters. Body: {max_risk_per_trade, daily_loss_limit, max_drawdown, max_leverage, ...}"""
    allowed_keys = {
        "max_risk_per_trade", "daily_loss_limit", "max_drawdown",
        "max_leverage", "kelly_fraction", "max_portfolio_risk",
        "stop_loss_pct", "take_profit_pct", "max_open_positions",
    }
    saved = {}
    for k, v in body.items():
        if k in allowed_keys:
            state.kv_set(f"risk_override_{k}", v)
            saved[k] = v
    return {"ok": True, "saved": saved}


@app.get("/api/config/risk")
async def get_risk_config():
    """Return current risk configuration (config defaults + dashboard overrides)."""
    keys = [
        "max_risk_per_trade", "daily_loss_limit", "max_drawdown",
        "max_leverage", "kelly_fraction", "max_portfolio_risk",
    ]
    result = {}
    for k in keys:
        default = getattr(cfg, k, None)
        override = state.kv_get(f"risk_override_{k}", None)
        result[k] = {"default": default, "override": override, "effective": override if override is not None else default}
    return {"risk": result}


# ─────────────────────────────────────────────────────────────────────────────
# REST — Forward Test (Bloch Futuretesting Framework)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/forwardtest/run")
async def run_forwardtest(body: dict):
    """
    Submit a forward-test job.
    Body: { symbol, model, cal_start, cal_end, n_paths, horizon, strategy, interval }
    """
    job_id = str(uuid.uuid4())[:8]
    _ft_jobs[job_id] = {
        "status": "PENDING",
        "stage": "QUEUED",
        "progress": 0.0,
        "detail": "Waiting to start",
        "result": None,
        "error": None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    def _run():
        try:
            _update_ft_job(
                job_id,
                status="RUNNING",
                stage="PREPARING",
                progress=2.0,
                detail="Preparing forward-test inputs",
                error=None,
            )
            from backtester.data_loader import load_multi
            from backtester.futuretesting import FuturetestEngine
            from backtester.engine import BacktestEngine
            from scripts.run_backtest import _build_strategies

            symbol    = body.get("symbol", cfg.default_benchmark_symbol)
            model     = body.get("model", "gbm")
            n_paths   = int(body.get("n_paths", 50))
            horizon   = int(body.get("horizon", 500))
            strategy  = body.get("strategy", "momentum")
            interval  = body.get("interval", "1h")
            cal_start, cal_end, range_notice = _sanitize_history_request(
                [symbol],
                interval,
                body.get("cal_start", "2023-01-01"),
                body.get("cal_end") or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            )
            _ft_jobs[job_id]["range_notice"] = range_notice

            def _loader_progress(event: dict[str, Any]) -> None:
                overall_progress = float(event.get("overall_progress", event.get("progress", 0.0)))
                mapped_progress = 5.0 + (overall_progress * 0.35)
                detail = event.get("message") or f"Loading {symbol} {interval} history"
                _update_ft_job(
                    job_id,
                    status="RUNNING",
                    stage="DOWNLOADING_DATA",
                    progress=mapped_progress,
                    detail=detail,
                )

            # Load historical data for calibration
            _update_ft_job(
                job_id,
                status="RUNNING",
                stage="DOWNLOADING_DATA",
                progress=5.0,
                detail=range_notice or f"Preparing {symbol} {interval} history",
            )
            data = load_multi(
                [symbol],
                interval,
                cal_start,
                cal_end,
                progress_callback=_loader_progress,
            )
            hist_df = data.get(symbol, None)
            if hist_df is None or hist_df.empty:
                _update_ft_job(
                    job_id,
                    status="RUNNING",
                    stage="DOWNLOADING_DATA",
                    progress=12.0,
                    detail=f"Retrying fresh download for {symbol} {interval}",
                )
                data = load_multi(
                    [symbol],
                    interval,
                    cal_start,
                    cal_end,
                    force_download=True,
                    progress_callback=_loader_progress,
                )
                hist_df = data.get(symbol, None)
            if hist_df is None or hist_df.empty:
                raise ValueError(f"No data for {symbol} [{cal_start}→{cal_end}]")

            _update_ft_job(
                job_id,
                status="RUNNING",
                stage="CALIBRATING",
                progress=45.0,
                detail=f"Calibrating {model.upper()} model on {len(hist_df)} bars",
            )

            strat_list = _build_strategies([strategy])

            def engine_factory(sim_data):
                return BacktestEngine(
                    data={symbol: sim_data},
                    strategies=strat_list,
                    interval=interval,
                    initial_equity=float(cfg.initial_capital),
                    lookback=60,
                    stop_loss_pct=0.05,
                    take_profit_pct=0.10,
                    max_open_positions=8,
                )

            ft = FuturetestEngine(
                historical_df=hist_df,
                model=model,
                n_paths=n_paths,
                horizon_bars=horizon,
            )
            _update_ft_job(
                job_id,
                status="RUNNING",
                stage="RUNNING_PATHS",
                progress=55.0,
                detail=f"Running {n_paths} synthetic path(s) for {strategy}",
            )

            def _path_progress(event: dict[str, Any]) -> None:
                path_progress = float(event.get("progress", 0.0))
                mapped_progress = 55.0 + (path_progress * 0.4)
                detail = event.get("message") or f"Running synthetic paths for {symbol}"
                _update_ft_job(
                    job_id,
                    status="RUNNING",
                    stage="RUNNING_PATHS",
                    progress=mapped_progress,
                    detail=detail,
                )

            ft_result = ft.run(
                engine_factory,
                lambda: strat_list,
                symbol=symbol,
                verbose=False,
                progress_callback=_path_progress,
            )

            _update_ft_job(
                job_id,
                status="RUNNING",
                stage="SUMMARISING",
                progress=97.0,
                detail="Summarising synthetic-path results",
            )
            _update_ft_job(
                job_id,
                status="DONE",
                stage="DONE",
                progress=100.0,
                detail=f"Forward test complete for {symbol}",
                result={
                    **_summarize_forwardtest_result(ft_result),
                    "range_notice": range_notice,
                },
                error=None,
            )

        except Exception as exc:
            import traceback
            _update_ft_job(
                job_id,
                status="ERROR",
                stage="ERROR",
                detail=str(exc),
                error=str(exc),
            )
            log.error("Forward test job %s failed: %s\n%s", job_id, exc, traceback.format_exc())

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "PENDING", "stage": "QUEUED", "progress": 0.0}


@app.get("/api/forwardtest/jobs/{job_id}")
async def forwardtest_job_status(job_id: str):
    job = _ft_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


# ─────────────────────────────────────────────────────────────────────────────
# REST — Strategy permutations lab
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/permutations/run")
async def run_permutations(body: dict):
    """
    Run a compact parameter sweep for one selected strategy.
    Body:
      {
        strategy, params, symbols, interval, start, end,
        capital, stop_loss_pct, take_profit_pct, max_open_positions, lookback
      }
    """
    job_id = str(uuid.uuid4())[:8]
    _perm_jobs[job_id] = {
        "status": "PENDING",
        "progress": 0,
        "current_variant": 0,
        "total_variants": 0,
        "results": [],
        "best": None,
        "error": None,
    }

    def _run():
        try:
            _perm_jobs[job_id]["status"] = "RUNNING"
            from backtester.data_loader import load_multi, align_and_ffill
            from backtester.engine import BacktestEngine

            strategy_name = str(body.get("strategy") or "statistical_arb")
            current_params = {
                **STRATEGY_DEFAULTS.get(strategy_name, {}),
                **state.kv_get(f"strategy_params_{strategy_name}", {}),
            }
            sweepable_params = supported_strategy_params(strategy_name, current_params)
            selected_params = [
                str(item) for item in (body.get("params") or [])
                if str(item) in sweepable_params
            ]
            if not selected_params:
                selected_params = sweepable_params[:3]
            variants = permutation_grid(current_params, selected_params, max_variants=81)

            interval = str(body.get("interval") or "1h")
            symbols = body.get("symbols", cfg.model_symbols[:4])
            start, end, range_notice = _sanitize_history_request(
                symbols,
                interval,
                str(body.get("start") or "2024-01-01"),
                body.get("end"),
            )
            capital = float(body.get("capital", cfg.initial_capital))
            stop_loss_pct = float(body.get("stop_loss_pct", ENGINE_DEFAULTS["stop_loss_pct"]))
            take_profit_pct = float(body.get("take_profit_pct", ENGINE_DEFAULTS["take_profit_pct"]))
            max_open_positions = int(body.get("max_open_positions", ENGINE_DEFAULTS["max_open_positions"]))
            lookback = int(body.get("lookback", ENGINE_DEFAULTS["lookback"]))
            _perm_jobs[job_id]["range_notice"] = range_notice

            data = load_multi(symbols, interval, start, end)
            data = {sym: df for sym, df in data.items() if not df.empty}
            if not data:
                raise ValueError("No backtest data available for the selected symbols/date range.")
            data = align_and_ffill(data)

            _perm_jobs[job_id]["total_variants"] = len(variants)
            result_rows: list[dict] = []
            best_payload: dict[str, Any] | None = None

            for idx, variant in enumerate(variants, start=1):
                strategy = instantiate_strategy(strategy_name, variant)
                engine = BacktestEngine(
                    data=data,
                    strategies=[strategy],
                    interval=interval,
                    initial_equity=capital,
                    lookback=lookback,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    max_open_positions=max_open_positions,
                )
                result = engine.run()
                metrics = result.metrics
                row = {
                    "variant_id": idx,
                    "params": {key: variant.get(key) for key in variant if key in current_params},
                    "total_return_pct": round(float(metrics.total_return_pct), 3),
                    "annualised_return_pct": round(float(metrics.annualised_return_pct), 3),
                    "annualised_vol_pct": round(float(metrics.annualised_vol_pct), 3),
                    "sharpe": round(float(metrics.sharpe), 4),
                    "sortino": round(float(metrics.sortino), 4),
                    "calmar": round(float(metrics.calmar), 4),
                    "max_drawdown_pct": round(float(metrics.max_drawdown_pct), 3),
                    "win_rate_pct": round(float(metrics.win_rate_pct), 3),
                    "profit_factor": round(float(metrics.profit_factor), 4),
                    "trades": int(metrics.num_trades),
                    "score": round(float((metrics.sharpe * 0.55) + (metrics.calmar * 0.20) + (metrics.total_return_pct / 100.0 * 0.25)), 5),
                }
                result_rows.append(row)

                if best_payload is None or row["score"] > best_payload["score"]:
                    best_payload = {
                        **row,
                        "equity": [round(float(x), 4) for x in result.equity_curve.values.tolist()],
                        "timestamps": [str(ts) for ts in result.equity_curve.index],
                    }

                _perm_jobs[job_id]["current_variant"] = idx
                _perm_jobs[job_id]["progress"] = round((idx / max(len(variants), 1)) * 100, 1)

            result_rows.sort(key=lambda item: item["score"], reverse=True)
            _perm_jobs[job_id]["status"] = "DONE"
            _perm_jobs[job_id]["results"] = result_rows
            _perm_jobs[job_id]["best"] = best_payload
            _perm_jobs[job_id]["selected_params"] = selected_params
            _perm_jobs[job_id]["supported_params"] = sweepable_params
        except Exception as exc:
            import traceback
            _perm_jobs[job_id]["status"] = "ERROR"
            _perm_jobs[job_id]["error"] = str(exc)
            log.error("Permutation job %s failed: %s\n%s", job_id, exc, traceback.format_exc())

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "PENDING"}


@app.get("/api/permutations/jobs/{job_id}")
async def permutation_job_status(job_id: str):
    job = _perm_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


# ─────────────────────────────────────────────────────────────────────────────
# REST — ML Training
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/training/scripts")
async def list_training_scripts():
    return {"scripts": [{"id": k, **v} for k, v in TRAINING_SCRIPTS.items()]}


@app.post("/api/training/run")
async def run_training(body: dict):
    script_id = str(body.get("script", ""))
    if script_id not in TRAINING_SCRIPTS:
        raise HTTPException(400, f"Unknown script: {script_id!r}. Valid: {list(TRAINING_SCRIPTS)}")

    job_id = str(uuid.uuid4())[:8]
    info   = TRAINING_SCRIPTS[script_id]
    _train_jobs[job_id] = {
        "status":      "PENDING",
        "script":      script_id,
        "label":       info["label"],
        "started_at":  None,
        "finished_at": None,
        "exit_code":   None,
        "log_lines":   [],
        "error":       None,
    }

    def _run():
        import subprocess
        _train_jobs[job_id]["status"]     = "RUNNING"
        _train_jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
        root = Path(__file__).parent.parent
        python_bin = str(root / ".venv" / "bin" / "python")
        if not Path(python_bin).exists():
            python_bin = sys.executable
        try:
            proc = subprocess.Popen(
                [python_bin, "-m", info["module"]],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(root),
            )
            for line in iter(proc.stdout.readline, ""):
                line = line.rstrip()
                if line:
                    _train_jobs[job_id]["log_lines"].append(line)
                    # Cap memory
                    if len(_train_jobs[job_id]["log_lines"]) > 2000:
                        _train_jobs[job_id]["log_lines"] = _train_jobs[job_id]["log_lines"][-1500:]
            proc.wait()
            _train_jobs[job_id]["exit_code"] = proc.returncode
            _train_jobs[job_id]["status"]    = "DONE" if proc.returncode == 0 else "ERROR"
            if proc.returncode != 0:
                _train_jobs[job_id]["error"] = f"Process exited with code {proc.returncode}"
        except Exception as exc:
            import traceback
            _train_jobs[job_id]["status"] = "ERROR"
            _train_jobs[job_id]["error"]  = str(exc)
            log.error("Training job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
        finally:
            _train_jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "PENDING", "script": script_id}


@app.get("/api/training/jobs")
async def list_training_jobs():
    jobs = [
        {"job_id": jid, **{k: v for k, v in job.items() if k != "log_lines"}}
        for jid, job in reversed(list(_train_jobs.items()))
    ]
    return {"jobs": jobs}


@app.get("/api/training/jobs/{job_id}")
async def training_job_status(job_id: str):
    job = _train_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — Live equity + positions broadcast
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    WF_CONN.add(ws)
    try:
        # Send current state immediately on connect
        with _db_conn() as c:
            rows = c.execute(
                "SELECT ts, equity FROM equity_curve ORDER BY ts DESC LIMIT 1"
            ).fetchall()
        if rows:
            pnl_snapshot = state.get_daily_pnl_snapshot()
            broker_snapshot = live_broker_snapshot()
            current_equity = (
                safe_float(broker_snapshot.get("total_equity"))
                if broker_snapshot.get("available")
                else float(rows[0]["equity"])
            )
            await ws.send_json({
                "type": "equity_tick",
                "equity": current_equity,
                "daily_pnl": _rounded(broker_snapshot.get("day_pnl_dollars") if broker_snapshot.get("available") else pnl_snapshot.get("daily_pnl"), 4),
                "daily_pnl_pct": _rounded(broker_snapshot.get("day_pnl_pct") if broker_snapshot.get("available") else pnl_snapshot.get("daily_pnl_pct"), 2),
                "daily_pnl_source": broker_snapshot.get("source") if broker_snapshot.get("available") else pnl_snapshot.get("source"),
                "ts": broker_snapshot.get("latest_equity_ts") if broker_snapshot.get("available") else rows[0]["ts"],
            })
        while True:
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        WF_CONN.discard(ws)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — Log streaming
# ─────────────────────────────────────────────────────────────────────────────

async def _tail_log(ws: WebSocket, log_path: Path) -> None:
    """Stream a log file line by line, starting from last 100 lines."""
    if not log_path.exists():
        await ws.send_json({"line": f"[Log file not found: {log_path}]", "level": "WARN"})
        return

    await ws.send_json({"line": f"[Streaming {log_path.name}]", "level": "INFO"})

    # Send last 100 lines of history
    lines = log_path.read_text(errors="replace").splitlines()
    for line in lines[-100:]:
        clean = _ANSI_ESCAPE_RE.sub("", line)
        level = "ERROR" if "ERROR" in clean else ("WARN" if "WARNING" in clean else
                "INFO" if "INFO" in clean else "DEBUG")
        await ws.send_json({"line": clean, "level": level})

    # Tail new lines
    with log_path.open("r", errors="replace") as f:
        f.seek(0, 2)   # seek to end
        while True:
            line = f.readline()
            if line:
                clean = _ANSI_ESCAPE_RE.sub("", line.rstrip())
                level = ("ERROR" if "ERROR" in clean else
                         "WARN"  if "WARNING" in clean else
                         "INFO"  if "INFO" in clean else "DEBUG")
                await ws.send_json({"line": clean, "level": level})
            else:
                await asyncio.sleep(0.5)


@app.websocket("/ws/logs/trader")
async def ws_logs_trader(ws: WebSocket):
    await ws.accept()
    log_path = LOG_DIR / "bot.log"
    try:
        await _tail_log(ws, log_path)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.debug("Trader log WS error: %s", exc)


@app.websocket("/ws/logs/risk")
async def ws_logs_risk(ws: WebSocket):
    await ws.accept()
    risk_log = LOG_DIR / "risk.log"
    trader_log = LOG_DIR / "bot.log"
    candidates = [path for path in (risk_log, trader_log) if path.exists()]
    log_path = max(candidates, key=lambda path: path.stat().st_mtime) if candidates else risk_log
    try:
        await _tail_log(ws, log_path)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.debug("Risk log WS error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Elite intelligence endpoints
# ─────────────────────────────────────────────────────────────────────────────

ML_ALPHA_PATH         = RUNTIME_DIR / "ml_alpha_snapshot.json"
EXEC_QUALITY_PATH     = RUNTIME_DIR / "execution_quality_snapshot.json"
SYSTEM_RESOURCE_PATH  = RUNTIME_DIR / "system_resource_snapshot.json"
PREFLIGHT_PATH        = RUNTIME_DIR / "preflight_state.json"
AUTOMATION_STATE_PATH = RUNTIME_DIR / "automation_state.json"


@app.get("/api/ml/alpha")
async def ml_alpha_snapshot():
    """ML model alpha signals — predicted returns, direction, confidence per symbol."""
    snap = read_json(ML_ALPHA_PATH, {})
    signals = snap.get("signals", [])
    if isinstance(signals, dict):
        signals = list(signals.values())
    return {
        "generated_at_utc": snap.get("generated_at_utc"),
        "requested_symbols": snap.get("requested_symbols", 0),
        "signals": sorted(signals, key=lambda s: abs(s.get("alpha_score") or 0), reverse=True),
    }


@app.get("/api/execution/quality")
async def execution_quality():
    """Live execution quality metrics — fill rates, slippage, tier breakdown."""
    q = read_json(EXEC_QUALITY_PATH, {})
    return {
        "generated_at_utc":          q.get("generated_at_utc"),
        "records":                   safe_int(q.get("records")),
        "fill_events":               safe_int(q.get("fill_events")),
        "full_fills":                safe_int(q.get("full_fills")),
        "partial_fills":             safe_int(q.get("partial_fills")),
        "fill_rate":                 safe_float(q.get("fill_rate")),
        "full_fill_rate":            safe_float(q.get("full_fill_rate")),
        "avg_execution_quality_score": safe_float(q.get("avg_execution_quality_score")),
        "avg_limit_edge_bps":        safe_float(q.get("avg_limit_edge_bps")),
        "avg_reference_edge_bps":    safe_float(q.get("avg_reference_edge_bps")),
        "avg_pricing_confidence":    safe_float(q.get("avg_pricing_confidence")),
        "avg_staleness_pct":         safe_float(q.get("avg_staleness_pct")),
        "degraded_execution_count":  safe_int(q.get("degraded_execution_count")),
        "adaptive_reprice_factor":   safe_float(q.get("adaptive_reprice_factor")),
        "tier_counts":               q.get("tier_counts", {}),
        "latest_fill_at_utc":        q.get("latest_fill_at_utc"),
        "note":                      q.get("note", ""),
    }


EXEC_LEDGER_PATH = RUNTIME_DIR / "execution_ledger.json"


@app.get("/api/execution/ledger")
async def execution_ledger_api():
    """All trades from the execution ledger — actual broker fills."""
    records: list[dict] = []
    try:
        raw = read_json(EXEC_LEDGER_PATH, [])
        if not isinstance(raw, list):
            raw = []
    except Exception:
        raw = []
    for rec in raw:
        status = str(rec.get("status", "")).lower()
        is_filled = "filled" in status
        for leg in rec.get("legs") or []:
            sym = str((leg or {}).get("symbol", "")).upper()
            side_raw = str((leg or {}).get("side", "")).lower()
            side = "SELL" if "sell" in side_raw else "BUY"
            qty = safe_float(rec.get("filled_qty") or rec.get("qty"))
            fill_price = safe_float(rec.get("filled_avg_price"))
            multiplier = 100.0 if is_option_symbol(sym) else 1.0
            records.append({
                "order_id":    rec.get("order_id"),
                "symbol":      sym,
                "side":        side,
                "qty":         qty,
                "fill_price":  fill_price,
                "filled_at":   rec.get("filled_at_utc") or rec.get("updated_at_utc"),
                "status":      status,
                "is_filled":   is_filled,
                "partial_fill": bool(rec.get("partial_fill")),
                "exec_score":  safe_float((rec.get("execution_quality") or {}).get("score")),
                "exec_tier":   (rec.get("execution_quality") or {}).get("tier"),
                "source":      rec.get("source"),
                "market":      "options" if is_option_symbol(sym) else "spot",
                "notional":    None if qty is None or fill_price is None else round(float(qty) * float(fill_price) * multiplier, 4),
            })
    filled = [r for r in records if r["is_filled"] or r["partial_fill"]]
    return {
        "total":     len(records),
        "filled":    len(filled),
        "records":   records,
        "fills":     filled,
        "fill_rate": round(len(filled) / max(len(records), 1), 4),
        "latest_fill_at_utc": filled[0]["filled_at"] if filled else None,
    }


@app.get("/api/trade_decisions")
async def trade_decisions_api(
    limit: int = Query(500, ge=1, le=2000),
    status: str | None = Query(None),
    strategy: str | None = Query(None),
):
    """Structured strategy decision tape — suggestions, gates, rejects, executions."""
    records = read_trade_decisions(limit=limit, status=status, strategy=strategy)
    ordered = list(reversed(records))
    counts: dict[str, int] = {}
    strategies: set[str] = set()
    for rec in ordered:
        rec_status = str(rec.get("status") or "INFO").upper()
        counts[rec_status] = counts.get(rec_status, 0) + 1
        rec_strategy = str(rec.get("strategy") or "").strip()
        if rec_strategy:
            strategies.add(rec_strategy)
    latest = ordered[0].get("ts") if ordered else None
    return {
        "total": len(ordered),
        "records": ordered,
        "counts": counts,
        "strategies": sorted(strategies),
        "latest_ts": latest,
        "source": str(DEFAULT_DECISION_TAPE_PATH),
    }


# --------------------------------------------------------------------------- #
# Asymmetric sleeves + live allocation (2026-07 dashboard update)
# --------------------------------------------------------------------------- #

@app.get("/api/sleeves/deep_value")
async def sleeves_deep_value_api():
    """Deep-value (Graham net-net) sleeve: nightly scan + held positions."""
    from dashboard.analytics import build_deep_value_payload

    return build_deep_value_payload()


@app.get("/api/sleeves/cornwall")
async def sleeves_cornwall_api():
    """Cornwall convexity sleeve: standalone long option lottery tickets."""
    from dashboard.analytics import build_cornwall_payload

    return build_cornwall_payload()


@app.get("/api/allocation/live")
async def live_allocation_api():
    """Live allocation snapshot written by the crypto orchestrator."""
    from dashboard.analytics import build_live_allocation_payload

    return build_live_allocation_payload()


@app.get("/api/crypto/book")
async def crypto_book_api(limit: int = Query(200, ge=1, le=1000)):
    """Crypto book: live crypto positions + per-strategy trade activity."""
    from dashboard.analytics import build_crypto_book_payload

    return build_crypto_book_payload(limit=limit)


@app.get("/api/equity/analytics")
async def equity_analytics():
    """Rolling Sharpe, Calmar, volatility, regime-annotated equity curve."""
    series, meta = _latest_metric_equity_series(limit=5000)
    if len(series) < 10 or not meta.get("sufficient_history"):
        return {
            "available": False,
            "metrics_meta": meta,
            "notice": _metric_history_notice(meta) if len(series) >= 2 else "Rolling equity analytics are waiting for more observations.",
        }

    ts_list = [ts.isoformat() for ts in series.index]
    eq_list = [float(v) for v in series.to_list()]

    eq_arr = np.array(eq_list)
    rets   = np.diff(eq_arr) / (eq_arr[:-1] + 1e-10)

    # Rolling 50-bar Sharpe (annualised — bars are ~1-min apart)
    window = min(50, len(rets) - 1)
    rolling_sharpe: list = []
    for i in range(len(rets)):
        start = max(0, i - window)
        chunk = rets[start : i + 1]
        if len(chunk) < 5:
            rolling_sharpe.append(None)
        else:
            mean_r = float(np.mean(chunk))
            std_r  = float(np.std(chunk)) + 1e-10
            ann    = np.sqrt(252 * 24 * 60)
            rolling_sharpe.append(round(mean_r / std_r * ann, 3))

    total_ret    = (eq_arr[-1] / eq_arr[0]) - 1 if eq_arr[0] > 0 else 0
    running_peak = np.maximum.accumulate(eq_arr)
    dd_arr       = (eq_arr - running_peak) / (running_peak + 1e-10)
    max_dd       = float(dd_arr.min())

    overall_sharpe = None
    if len(rets) > 5:
        m = float(np.mean(rets))
        s = float(np.std(rets)) + 1e-10
        overall_sharpe = round(m / s * np.sqrt(252 * 24 * 60), 3)

    calmar = round(total_ret / abs(max_dd), 3) if max_dd < 0 else None

    step       = max(1, len(ts_list) // 500)
    sampled_ts = ts_list[::step]
    sampled_eq = [round(v, 2) for v in eq_list[::step]]
    sampled_dd = [round(v * 100, 3) for v in dd_arr[::step]]
    sampled_rs = (rolling_sharpe + [None])[::step][: len(sampled_ts)]

    return {
        "available":         True,
        "total_return_pct":  round(total_ret * 100, 3),
        "max_drawdown_pct":  round(max_dd * 100, 3),
        "overall_sharpe":    overall_sharpe,
        "calmar":            calmar,
        "start_equity":      round(float(eq_arr[0]), 2),
        "end_equity":        round(float(eq_arr[-1]), 2),
        "peak_equity":       round(float(eq_arr.max()), 2),
        "timestamps":        sampled_ts,
        "equity":            sampled_eq,
        "drawdown_pct":      sampled_dd,
        "rolling_sharpe":    sampled_rs,
        "n_bars":            len(series),
        "metrics_meta":      meta,
    }


@app.get("/api/market/breadth")
async def market_breadth():
    """Market breadth, leaders/laggards, and microstructure from research desk cache."""
    d = await run_in_threadpool(build_research_desk, False)
    return {
        "generated_at_utc": d.get("generated_at_utc"),
        "stock_breadth":    d.get("stock_breadth", {}),
        "crypto_breadth":   d.get("crypto_breadth", {}),
        "stock_leaders":    d.get("stock_leaders", {}),
        "crypto_leaders":   d.get("crypto_leaders", {}),
        "microstructure":   d.get("microstructure_board", []),
    }


@app.get("/api/system/health")
async def system_health():
    """Host resource utilization + automation scheduling state."""
    res  = read_json(SYSTEM_RESOURCE_PATH, {})
    auto = read_json(AUTOMATION_STATE_PATH, {})
    pre  = read_json(PREFLIGHT_PATH, {})

    host = res.get("host_metrics", {})
    mem  = host.get("memory", {})
    disk = host.get("disk", {})
    rp   = res.get("resource_profile", {})

    return {
        "generated_at_utc":    res.get("generated_at_utc"),
        "pressure":            res.get("status", {}).get("pressure", "unknown"),
        "loadavg_1m":          safe_float(host.get("loadavg_1m")),
        "loadavg_5m":          safe_float(host.get("loadavg_5m")),
        "normalized_cpu_pct":  safe_float(host.get("normalized_cpu_load_pct")),
        "memory_total_gb":     safe_float(mem.get("total_gb")),
        "memory_used_gb":      safe_float(mem.get("used_gb")),
        "memory_usage_pct":    safe_float(mem.get("usage_pct")),
        "disk_total_gb":       safe_float(disk.get("total_gb")),
        "disk_used_gb":        safe_float(disk.get("used_gb")),
        "disk_usage_pct":      safe_float(disk.get("usage_pct")),
        "cpu_count":           safe_int(rp.get("cpu_count")),
        "backtest_workers":    safe_int(rp.get("backtest_workers")),
        "model_parallelism":   safe_int(rp.get("model_parallelism")),
        "risk_interval_s":     safe_int(rp.get("risk_interval_seconds")),
        "regime_interval_s":   safe_int(rp.get("regime_interval_seconds")),
        "automation_state":    auto,
        "preflight_passed":    pre.get("ok"),
        "preflight_summary":   {
            "Passed":             pre.get("ok"),
            "Issues":             len(pre.get("issues", [])) if pre.get("issues") is not None else "—",
            "Source files":       pre.get("source_files"),
            "JSON files":         pre.get("json_files"),
            "Artifacts checked":  pre.get("artifacts_checked"),
        } if pre else {},
    }


@app.get("/api/intelligence")
async def intelligence_snapshot():
    """Compatibility snapshot for live intelligence tiles."""
    alpha = read_json(ML_ALPHA_PATH, {})
    signals = alpha.get("signals", [])
    if isinstance(signals, dict):
        signals = list(signals.values())
    return {
        "generated_at_utc": alpha.get("generated_at_utc"),
        "signals": sorted(signals, key=lambda s: abs(s.get("alpha_score") or 0), reverse=True) if isinstance(signals, list) else [],
    }


@app.get("/api/risk/events")
async def risk_events(limit: int = Query(100, ge=1, le=500)):
    """Recent risk events from the trades database."""
    with _db_conn() as c:
        try:
            rows = c.execute(
                "SELECT ts, event, detail FROM risk_events ORDER BY ts DESC LIMIT ?",
                (limit,)
            ).fetchall()
        except Exception:
            return {"events": []}
    return {
        "events": [{"ts": r["ts"], "event": r["event"], "detail": r["detail"]} for r in rows]
    }


@app.get("/api/trades/analysis")
async def trades_analysis():
    """Deep trade-level analytics — P&L distribution, streaks, expectancy, by market/symbol."""
    with _db_conn() as c:
        try:
            rows = c.execute(
                "SELECT ts, symbol, market, side, quantity, price, strategy, pnl, status "
                "FROM trades ORDER BY ts DESC LIMIT 5000"
            ).fetchall()
        except Exception:
            return {}

    if not rows:
        try:
            ledger_raw = read_json(EXEC_LEDGER_PATH, [])
            if not isinstance(ledger_raw, list):
                ledger_raw = []
        except Exception:
            ledger_raw = []

        fill_rows: list[dict[str, Any]] = []
        for rec in ledger_raw:
            status = str(rec.get("status", "")).lower()
            if "filled" not in status and not rec.get("partial_fill"):
                continue
            filled_at = rec.get("filled_at_utc") or rec.get("updated_at_utc")
            for leg in rec.get("legs") or []:
                symbol = str((leg or {}).get("symbol", "")).upper()
                if not symbol:
                    continue
                fill_rows.append({"symbol": symbol, "filled_at": filled_at, "status": status})

        by_symbol: dict[str, int] = {}
        latest_fill_at = None
        for row in fill_rows:
            symbol = row["symbol"]
            by_symbol[symbol] = by_symbol.get(symbol, 0) + 1
            filled_at = row.get("filled_at")
            if filled_at and (latest_fill_at is None or str(filled_at) > str(latest_fill_at)):
                latest_fill_at = filled_at

        return {
            "total_trades": 0,
            "realized_pnl_available": False,
            "ledger_fallback": True,
            "note": "No realized trade P&L is available in the trades database. Synthetic ledger-based P&L proxies have been removed; only confirmed fill activity is shown below.",
            "fill_activity": {
                "records": len(fill_rows),
                "latest_fill_at_utc": latest_fill_at,
                "symbols": [{"symbol": sym, "fills": count} for sym, count in sorted(by_symbol.items(), key=lambda item: (-item[1], item[0]))[:20]],
            },
            "by_market": {},
            "top_symbols": [],
            "pnl_hist": {"bins": [], "counts": []},
        }

    pnls      = [float(r["pnl"] or 0) for r in rows]
    wins      = [p for p in pnls if p > 0]
    losses    = [p for p in pnls if p < 0]
    flat      = [p for p in pnls if p == 0]

    # Streak calculation
    max_win_streak = max_loss_streak = cur = 0
    cur_win = cur_loss = 0
    for p in reversed(pnls):
        if p > 0:
            cur_win += 1; cur_loss = 0
        elif p < 0:
            cur_loss += 1; cur_win = 0
        else:
            cur_win = cur_loss = 0
        max_win_streak  = max(max_win_streak, cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)

    # By market
    by_market: dict[str, dict] = {}
    for r in rows:
        m = r["market"] or "spot"
        p = float(r["pnl"] or 0)
        if m not in by_market:
            by_market[m] = {"trades": 0, "total_pnl": 0.0, "wins": 0}
        by_market[m]["trades"]    += 1
        by_market[m]["total_pnl"] += p
        if p > 0:
            by_market[m]["wins"] += 1

    # By symbol (top 20)
    by_symbol: dict[str, dict] = {}
    for r in rows:
        sym = r["symbol"]
        p   = float(r["pnl"] or 0)
        if sym not in by_symbol:
            by_symbol[sym] = {"trades": 0, "total_pnl": 0.0, "wins": 0}
        by_symbol[sym]["trades"]    += 1
        by_symbol[sym]["total_pnl"] += p
        if p > 0:
            by_symbol[sym]["wins"] += 1

    top_symbols = sorted(by_symbol.items(), key=lambda x: abs(x[1]["total_pnl"]), reverse=True)[:20]

    # P&L histogram
    if pnls:
        counts, edges = np.histogram(pnls, bins=30)
        pnl_hist = {"bins": ((np.array(edges[:-1]) + np.array(edges[1:])) / 2).tolist(), "counts": counts.tolist()}
    else:
        pnl_hist = {"bins": [], "counts": []}

    total = len(pnls)
    win_rate = len(wins) / total * 100 if total else 0
    avg_win  = float(np.mean(wins))  if wins   else 0
    avg_loss = float(np.mean(losses)) if losses else 0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss) if total else 0

    return {
        "total_trades":    total,
        "win_count":       len(wins),
        "loss_count":      len(losses),
        "flat_count":      len(flat),
        "win_rate_pct":    round(win_rate, 2),
        "total_pnl":       round(sum(pnls), 2),
        "avg_win":         round(avg_win, 4),
        "avg_loss":        round(avg_loss, 4),
        "profit_factor":   round(sum(wins) / abs(sum(losses)), 3) if losses else None,
        "expectancy":      round(expectancy, 4),
        "max_win_streak":  max_win_streak,
        "max_loss_streak": max_loss_streak,
        "largest_win":     round(max(wins), 4)  if wins   else 0,
        "largest_loss":    round(min(losses), 4) if losses else 0,
        "by_market":       {k: {**v, "win_rate_pct": round(v["wins"]/max(v["trades"],1)*100, 1)} for k, v in by_market.items()},
        "top_symbols":     [{"symbol": s, **d, "win_rate_pct": round(d["wins"]/max(d["trades"],1)*100,1)} for s,d in top_symbols],
        "pnl_hist":        pnl_hist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REST — Master Recalibration Pipeline
# ─────────────────────────────────────────────────────────────────────────────

_recal_job: dict = {}

RECAL_SCRIPTS = [
    ("stock_universe_research", "Download + research full stock universe"),
    ("weekend_training",        "Full pipeline: data download + all models"),
    ("train_hmm",               "HMM macro regime"),
    ("train_correlation_alpha", "Correlation alpha models"),
    ("train_xgb_alpha",         "XGBoost alpha engine"),
    ("weekend_recalibration",   "Strategy parameter recalibration"),
]
RECAL_SCRIPT_DESCRIPTIONS = dict(RECAL_SCRIPTS)
RECAL_SCRIPT_EST_SECONDS = {
    script: max(60, int((TRAINING_SCRIPTS.get(script) or {}).get("est_min", 10)) * 60)
    for script, _ in RECAL_SCRIPTS
}


def _append_recal_log(job: dict, line: str, *, max_lines: int = 320) -> None:
    cleaned = _ANSI_ESCAPE_RE.sub("", str(line or "")).strip()
    if not cleaned:
        return
    job.setdefault("log", []).append(cleaned[-1200:])
    if len(job["log"]) > max_lines:
        job["log"] = job["log"][-max_lines:]


def _parse_pipeline_progress(line: str) -> float | None:
    match = re.search(r"\[pipeline\].*?\]\s*(\d{1,3})%", line or "")
    if not match:
        return None
    return max(0.0, min(100.0, float(match.group(1)))) / 100.0


def _recal_progress_payload(job: dict) -> dict:
    scripts = job.get("scripts") or []
    completed = job.get("completed") or []
    failed = job.get("failed") or []
    total = len(scripts)
    finished = len(completed) + len(failed)
    current = job.get("current")
    if job.get("status") in {"DONE", "PARTIAL"}:
        progress = 100.0
        current_index = total
    elif total:
        current_fraction = 0.0
        current_elapsed_seconds = None
        current_est_seconds = job.get("current_est_seconds") or RECAL_SCRIPT_EST_SECONDS.get(current, 600)
        current_started_at = job.get("current_started_at")
        if current and current_started_at:
            try:
                started = datetime.fromisoformat(str(current_started_at).replace("Z", "+00:00"))
                current_elapsed_seconds = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
                elapsed_fraction = current_elapsed_seconds / max(float(current_est_seconds or 600), 1.0)
                current_fraction = min(0.90, elapsed_fraction)
            except Exception:
                current_elapsed_seconds = None
        hint = job.get("current_progress_hint")
        if hint is not None:
            current_fraction = max(current_fraction, min(0.95, float(hint)))
        progress = min(99.0, ((finished + current_fraction) / total) * 100.0)
        current_index = min(total, finished + (1 if current else 0))
    else:
        progress = 0.0
        current_index = 0
    return {
        **job,
        "total_steps": total,
        "completed_steps": len(completed),
        "failed_steps": len(failed),
        "current_step_index": current_index,
        "current_description": RECAL_SCRIPT_DESCRIPTIONS.get(current, current),
        "current_elapsed_seconds": current_elapsed_seconds if total and job.get("status") == "RUNNING" else None,
        "current_est_seconds": job.get("current_est_seconds"),
        "current_pid": job.get("current_pid"),
        "updated_at": job.get("updated_at"),
        "progress_pct": round(progress, 1),
    }


@app.post("/api/system/master_recalibrate")
async def master_recalibrate(body: dict | None = None):
    """
    Trigger the full data download + model retrain + recalibration pipeline.
    Body: { scripts: ["weekend_training", ...] }  (default = all)
    """
    global _recal_job
    body = body or {}
    if _recal_job.get("status") == "RUNNING":
        return {"ok": False, "error": "A recalibration job is already running", "job": _recal_progress_payload(_recal_job)}

    requested = body.get("scripts") or [s for s, _ in RECAL_SCRIPTS]
    valid = {s for s, _ in RECAL_SCRIPTS}
    scripts = [s for s in requested if s in valid]
    if not scripts:
        raise HTTPException(400, "No valid scripts specified")

    job_id = str(uuid.uuid4())[:8]
    _recal_job = {
        "job_id": job_id,
        "status": "RUNNING",
        "scripts": scripts,
        "script_descriptions": {s: RECAL_SCRIPT_DESCRIPTIONS.get(s, s) for s in scripts},
        "current": None,
        "current_started_at": None,
        "current_est_seconds": None,
        "current_pid": None,
        "current_progress_hint": None,
        "completed": [],
        "failed": [],
        "log": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    def _run():
        for idx, script_name in enumerate(scripts, start=1):
            _recal_job["current"] = script_name
            _recal_job["current_started_at"] = datetime.now(timezone.utc).isoformat()
            _recal_job["current_est_seconds"] = RECAL_SCRIPT_EST_SECONDS.get(script_name, 600)
            _recal_job["current_pid"] = None
            _recal_job["current_progress_hint"] = None
            _recal_job["updated_at"] = datetime.now(timezone.utc).isoformat()
            description = RECAL_SCRIPT_DESCRIPTIONS.get(script_name, script_name)
            _append_recal_log(_recal_job, f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] [{idx}/{len(scripts)}] Starting {description} ({script_name})…")
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", f"scripts.{script_name}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(ROOT),
                )
            except Exception as exc:
                _recal_job["failed"].append(script_name)
                _append_recal_log(_recal_job, f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] ✗ {description} exception: {exc}")
                continue

            _recal_job["current_pid"] = process.pid
            try:
                assert process.stdout is not None
                deadline = time.monotonic() + 3600
                while process.poll() is None:
                    line = process.stdout.readline()
                    if line:
                        hint = _parse_pipeline_progress(line)
                        if hint is not None:
                            _recal_job["current_progress_hint"] = hint
                        _append_recal_log(_recal_job, line)
                        _recal_job["updated_at"] = datetime.now(timezone.utc).isoformat()
                    elif time.monotonic() > deadline:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise TimeoutError(f"{script_name} exceeded 3600s timeout")
                    else:
                        time.sleep(0.5)
                for line in process.stdout.readlines():
                    hint = _parse_pipeline_progress(line)
                    if hint is not None:
                        _recal_job["current_progress_hint"] = hint
                    _append_recal_log(_recal_job, line)
                if process.returncode == 0:
                    _recal_job["completed"].append(script_name)
                    _append_recal_log(_recal_job, f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] ✓ {description} done")
                else:
                    _recal_job["failed"].append(script_name)
                    _append_recal_log(_recal_job, f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] ✗ {description} failed with exit code {process.returncode}")
            except Exception as exc:
                if process.poll() is None:
                    process.terminate()
                _recal_job["failed"].append(script_name)
                _append_recal_log(_recal_job, f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] ✗ {description} exception: {exc}")
            finally:
                _recal_job["current_pid"] = None
                _recal_job["updated_at"] = datetime.now(timezone.utc).isoformat()

        _recal_job["current"] = None
        _recal_job["current_started_at"] = None
        _recal_job["current_est_seconds"] = None
        _recal_job["current_pid"] = None
        _recal_job["current_progress_hint"] = None
        _recal_job["status"] = "DONE" if not _recal_job["failed"] else "PARTIAL"
        _recal_job["finished_at"] = datetime.now(timezone.utc).isoformat()
        _recal_job["updated_at"] = _recal_job["finished_at"]

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "job_id": job_id, "scripts": scripts, "job": _recal_progress_payload(_recal_job)}


@app.get("/api/system/master_recalibrate")
async def master_recalibrate_status():
    """Current status of the master recalibration pipeline."""
    return _recal_progress_payload(_recal_job) if _recal_job else {"status": "IDLE", "progress_pct": 0, "total_steps": 0, "completed_steps": 0, "failed_steps": 0}


@app.get("/api/system/recal_scripts")
async def recal_scripts_list():
    return {"scripts": [{"id": s, "description": d} for s, d in RECAL_SCRIPTS]}


@app.get("/api/digital_twin")
async def digital_twin_snapshot():
    series, meta = _latest_metric_equity_series(limit=720)
    values = [float(v) for v in series.to_list()] if len(series) else []
    timestamps = [idx.isoformat() for idx in series.index] if len(series) else []
    returns = pd.Series(values).pct_change().dropna() if len(values) >= 2 else pd.Series(dtype=float)
    current_equity = values[-1] if values else cfg.initial_capital
    vol = float(returns.std() * math.sqrt(252)) if len(returns) >= 2 and returns.std() == returns.std() else 0.0
    drift = float((values[-1] - values[0]) / max(values[0], 1e-10)) if len(values) >= 2 else 0.0
    spread = returns - returns.rolling(24, min_periods=4).mean()
    spread_std = float(spread.std()) if len(spread.dropna()) else 0.0
    drift_z = float((spread.iloc[-1] / spread_std)) if spread_std > 0 and len(spread) else 0.0
    risk_snapshot = read_json(RISK_SNAPSHOT_PATH, {})
    stock_overview = build_stocks_overview()
    stock_rows = stock_overview.get("universe_rows") or []
    top_stock_alpha = [
        row for row in stock_rows
        if row.get("alpha_daily") is not None
    ][:10]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "integrity": {
            "status": "RED" if abs(drift_z) > 1.5 else "AMBER" if abs(drift_z) > 1.1 else "GREEN",
            "drift_z": round(drift_z, 3),
            "history_points": len(values),
            "history_meta": meta,
        },
        "equity": {
            "timestamps": timestamps,
            "realized": values,
            "shadow": [round(values[0] * (1.0 + drift * i / max(len(values) - 1, 1)), 2) for i in range(len(values))] if values else [],
            "current": round(current_equity, 2),
            "realized_vol_annualized": round(vol, 4),
        },
        "risk": {
            "daily_pnl_pct": safe_float(risk_snapshot.get("daily_pnl_pct")),
            "current_drawdown": safe_float(risk_snapshot.get("current_drawdown")),
            "risk_score": safe_float(risk_snapshot.get("risk_score")),
            "vix": safe_float(risk_snapshot.get("vix")),
        },
        "stocks": {
            "universe_size": stock_overview.get("universe_size"),
            "price_data_coverage_pct": stock_overview.get("price_data_coverage_pct"),
            "research_coverage_pct": stock_overview.get("research_coverage_pct"),
            "top_alpha": top_stock_alpha,
        },
        "models": [
            {"id": "HMM-MACRO", "family": "HMM", "asset": "multi-asset", "status": "LIVE", "drift_z": round(abs(drift_z) * 0.7, 3)},
            {"id": "XGB-ALPHA", "family": "XGBoost", "asset": "stocks+crypto", "status": "LIVE", "drift_z": round(abs(drift_z), 3)},
            {"id": "TFT-SEQUENCE", "family": "TFT", "asset": "multi-horizon", "status": "SHADOW", "drift_z": round(abs(drift_z) * 1.15, 3)},
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# GS QUANT ANALYTICS ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

def _gs_load_data(symbol: str, lookback_days: int = 365):
    """Load OHLCV for a symbol from the local cache without blocking on downloads."""
    try:
        cache_dir = ROOT / ".backtest_cache"
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days + 2)
        candidates = sorted(cache_dir.glob(f"{symbol}_1h_*.parquet"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in candidates:
            df = pd.read_parquet(path)
            if df is None or len(df) == 0:
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            else:
                df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            df = df.sort_index()
            df = df[df.index >= cutoff]
            if len(df) >= 30:
                return df
        return None
    except Exception as e:
        log.warning("gs_load_data failed for %s: %s", symbol, e)
        return None


def _gs_load_multi(symbols: list, lookback_days: int = 365):
    try:
        result = {}
        for symbol in symbols:
            df = _gs_load_data(str(symbol).upper(), lookback_days)
            if df is not None and len(df) > 50:
                result[str(symbol).upper()] = df
        return result
    except Exception as e:
        log.warning("gs_load_multi failed: %s", e)
        return {}


@app.get("/api/gsquant/timeseries/{symbol}")
async def gs_timeseries(symbol: str, lookback: int = Query(365, ge=30, le=1000)):
    """Full gs-quant timeseries analytics for a symbol."""
    from ml.gs_analytics import compute_timeseries_analytics
    df = await run_in_threadpool(_gs_load_data, symbol.upper(), lookback)
    if df is None or len(df) < 30:
        return {"available": False, "error": "insufficient data"}
    return await run_in_threadpool(compute_timeseries_analytics, df, symbol.upper())


@app.get("/api/gsquant/technical/{symbol}")
async def gs_technical(symbol: str, lookback: int = Query(365, ge=30, le=1000)):
    """Technical signals: RSI, MACD, Bollinger Bands, EMA crossovers."""
    from ml.gs_analytics import compute_technical_signals
    df = await run_in_threadpool(_gs_load_data, symbol.upper(), lookback)
    if df is None or len(df) < 30:
        return {"available": False, "error": "insufficient data"}
    return await run_in_threadpool(compute_technical_signals, df, symbol.upper())


@app.get("/api/gsquant/risk/{symbol}")
async def gs_risk(symbol: str, confidence: float = Query(0.95, ge=0.8, le=0.999),
                   lookback: int = Query(365, ge=30, le=1000)):
    """Risk analytics: VaR, CVaR, drawdown stats, stress scenarios."""
    from ml.gs_analytics import compute_risk_analytics
    df = await run_in_threadpool(_gs_load_data, symbol.upper(), lookback)
    if df is None or len(df) < 30:
        return {"available": False, "error": "insufficient data"}
    return await run_in_threadpool(compute_risk_analytics, df, symbol.upper(), confidence)


@app.get("/api/gsquant/vol_cone/{symbol}")
async def gs_vol_cone(symbol: str, lookback: int = Query(365, ge=63, le=1000)):
    """Realized volatility cone at multiple windows."""
    from ml.gs_analytics import compute_vol_cone
    df = await run_in_threadpool(_gs_load_data, symbol.upper(), lookback)
    if df is None or len(df) < 63:
        return {"available": False, "error": "insufficient data"}
    cone = await run_in_threadpool(compute_vol_cone, df)
    return {"available": True, "symbol": symbol.upper(), "cone": cone}


@app.get("/api/gsquant/zscore/{symbol}")
async def gs_zscore(symbol: str, lookback: int = Query(500, ge=63, le=1000)):
    """Multi-window z-score regime signals."""
    from ml.gs_analytics import compute_zscore_regime
    df = await run_in_threadpool(_gs_load_data, symbol.upper(), lookback)
    if df is None or len(df) < 63:
        return {"available": False, "error": "insufficient data"}
    return await run_in_threadpool(compute_zscore_regime, df, symbol.upper())


@app.get("/api/gsquant/correlation")
async def gs_correlation(
    symbols: str = Query("SPY,QQQ,AAPL,MSFT,NVDA,BTCUSDT,ETHUSDT"),
    lookback: int = Query(252, ge=30, le=1000)
):
    """Rolling 63-day correlation matrix across universe."""
    from ml.gs_analytics import compute_correlation_matrix
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:15]
    df_dict = await run_in_threadpool(_gs_load_multi, sym_list, lookback)
    if len(df_dict) < 2:
        return {"available": False, "error": "need at least 2 symbols with data"}
    return await run_in_threadpool(compute_correlation_matrix, df_dict)


@app.get("/api/gsquant/beta/{symbol}")
async def gs_beta(symbol: str, benchmark: str = Query("SPY"),
                   window: int = Query(63, ge=10, le=252),
                   lookback: int = Query(365, ge=30, le=1000)):
    """Rolling beta to benchmark."""
    from ml.gs_analytics import compute_rolling_beta
    df = await run_in_threadpool(_gs_load_data, symbol.upper(), lookback)
    bdf = await run_in_threadpool(_gs_load_data, benchmark.upper(), lookback)
    if df is None or bdf is None or len(df) < window or len(bdf) < window:
        return {"available": False, "error": "insufficient data"}
    return await run_in_threadpool(compute_rolling_beta, df, bdf, symbol.upper(), window)


@app.get("/api/gsquant/factors")
async def gs_factors(
    symbols: str = Query("SPY,QQQ,AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,BTCUSDT,ETHUSDT"),
    lookback: int = Query(365, ge=63, le=1000)
):
    """Cross-sectional factor analytics: momentum, mean-reversion, vol factors."""
    from ml.gs_analytics import compute_factor_analytics
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:20]
    df_dict = await run_in_threadpool(_gs_load_multi, sym_list, lookback)
    return await run_in_threadpool(compute_factor_analytics, df_dict)


@app.get("/api/gsquant/portfolio")
async def gs_portfolio(
    symbols: str = Query("SPY,QQQ,AAPL,MSFT"),
    weights: str = Query(""),
    lookback: int = Query(365, ge=30, le=1000)
):
    """Portfolio-level analytics with configurable weights."""
    from ml.gs_analytics import compute_portfolio_analytics
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:15]
    w = None
    if weights:
        try:
            parts = weights.split(",")
            wvals = [float(p) for p in parts]
            if len(wvals) == len(sym_list):
                w = dict(zip(sym_list, wvals))
        except Exception:
            pass
    df_dict = await run_in_threadpool(_gs_load_multi, sym_list, lookback)
    return await run_in_threadpool(compute_portfolio_analytics, df_dict, w)


@app.post("/api/gsquant/pricer")
async def gs_pricer(body: dict):
    """
    Full Black-Scholes options pricer with complete Greeks suite.
    Body: {S, K, dte, sigma, option_type, r, q}
    """
    from strategies.options_pricer import price_option
    try:
        S           = float(body.get("S", 100))
        K           = float(body.get("K", 100))
        dte         = float(body.get("dte", 30))
        sigma       = float(body.get("sigma", 0.25))
        option_type = str(body.get("option_type", "Call"))
        r           = float(body.get("r", 0.05))
        q           = float(body.get("q", 0.0))
        return price_option(S=S, K=K, dte=dte, sigma=sigma,
                            option_type=option_type, r=r, q=q)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/gsquant/iv_solver")
async def gs_iv_solver(body: dict):
    """Newton-Raphson implied vol solver."""
    from strategies.options_pricer import BlackScholesEngine, OptionType
    try:
        market_price = float(body["market_price"])
        S            = float(body["S"])
        K            = float(body["K"])
        dte          = float(body["dte"])
        T            = max(dte / 365.0, 1e-6)
        r            = float(body.get("r", 0.05))
        q            = float(body.get("q", 0.0))
        option_type  = OptionType(body.get("option_type", "Call"))
        iv = BlackScholesEngine.implied_vol(market_price, S, K, T, r, option_type, q)
        if iv is None:
            return {"error": "IV solver did not converge", "iv": None}
        return {"iv": round(iv, 6), "iv_pct": round(iv * 100, 4)}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/gsquant/scenario_grid")
async def gs_scenario_grid(body: dict):
    """2D scenario P&L grid (spot × vol shocks) for heatmap."""
    from strategies.options_pricer import BlackScholesEngine, OptionType, full_scenario_grid
    try:
        S           = float(body.get("S", 100))
        K           = float(body.get("K", 100))
        dte         = float(body.get("dte", 30))
        sigma       = float(body.get("sigma", 0.25))
        option_type = OptionType(body.get("option_type", "Call"))
        r           = float(body.get("r", 0.05))
        T           = max(dte / 365.0, 1e-6)
        result = BlackScholesEngine.calc(S, K, T, r, sigma, option_type)
        grid = full_scenario_grid(result)
        return {"available": True, "grid": grid, "base_price": round(result.price, 4)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/gsquant/vol_surface/{symbol}")
async def gs_vol_surface(symbol: str, lookback: int = Query(180, ge=30, le=500)):
    """
    Construct a SABR vol surface from ATM IV + skew estimation.
    Falls back to a demo surface built from realized vol.
    """
    from ml.gs_analytics import compute_timeseries_analytics
    from strategies.vol_surface import build_demo_surface, VolSurface
    df = await run_in_threadpool(_gs_load_data, symbol.upper(), lookback)
    if df is None or len(df) < 30:
        return {"available": False, "error": "insufficient data"}

    analytics = await run_in_threadpool(compute_timeseries_analytics, df, symbol.upper())
    close = float(df["close"].iloc[-1]) if "close" in df.columns else float(df.iloc[-1, -1])
    atm_iv = analytics.get("annualized_vol_21") or 0.25
    skew   = -0.04

    surface = build_demo_surface(spot=close, atm_vol=atm_iv, skew=skew)
    term_ts = surface.term_structure()
    grid    = surface.surface_grid()
    skew_r  = surface.skew_report()

    # Vol regime signal
    rv21 = analytics.get("realized_vol_21") or analytics.get("annualized_vol_21") or atm_iv
    from strategies.vol_surface import vol_regime_analytics
    regime = vol_regime_analytics(atm_iv, rv21, term_ts)

    return {
        "available":      True,
        "symbol":         symbol.upper(),
        "spot":           round(close, 4),
        "atm_iv":         round(atm_iv, 6),
        "term_structure": term_ts,
        "surface_grid":   grid,
        "skew_report":    skew_r,
        "vol_regime":     regime,
    }


@app.get("/api/gsquant/summary")
async def gs_summary():
    """Summary dashboard: key analytics across the standard universe."""
    from bot.config import cfg
    symbols = (cfg.model_symbols or ["SPY", "QQQ", "AAPL", "BTCUSDT"])[:4]

    async def _symbol_summary(sym: str) -> tuple[str, dict[str, Any] | None]:
        try:
            df = await run_in_threadpool(_gs_load_data, sym, 180)
            if df is None or len(df) < 30:
                return sym, None
            from ml.gs_analytics import compute_risk_analytics, compute_timeseries_analytics
            ts_metrics = await run_in_threadpool(compute_timeseries_analytics, df, sym)
            risk = await run_in_threadpool(compute_risk_analytics, df, sym)
            payload = {
                "sharpe": ts_metrics.get("sharpe_ratio"),
                "max_dd": ts_metrics.get("max_drawdown"),
                "vol_21": ts_metrics.get("annualized_vol_21"),
                "rsi_14": ts_metrics.get("rsi_14"),
                "var_1d": risk.get("var_1d_pct"),
                "zscore_1m": ts_metrics.get("zscore_1m"),
            }
            return sym, payload
        except Exception as e:
            log.debug("gs_summary failed for %s: %s", sym, e)
            return sym, None

    results = {}
    for sym, payload in await asyncio.gather(*[_symbol_summary(sym) for sym in symbols]):
        if payload:
            results[sym] = payload
    return {"available": bool(results), "symbols": results}


# ─────────────────────────────────────────────────────────────────────────────
# Model Zoo — TFT / VAE Vol / XGBoost Alpha / HMM / Kill Switch
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/model_zoo/tft/{symbol}")
async def model_zoo_tft(symbol: str, lookback: int = Query(300, ge=100, le=1000)):
    """Temporal Fusion Transformer multi-horizon forecast."""
    try:
        from ml.tft_forecaster import get_tft_engine, _build_features, LOOKBACK, TFTEngine
        df = await run_in_threadpool(_gs_load_data, symbol.upper(), max(lookback, LOOKBACK + 50))
        if df is None or len(df) < LOOKBACK:
            return {"available": False, "error": "insufficient data"}

        engine = get_tft_engine()
        if not engine.model or symbol.upper() not in engine.sym_map:
            # Auto-train on this symbol with current data
            metrics = await run_in_threadpool(engine.train, {symbol.upper(): df}, 10, False)
            if not metrics:
                return {"available": False, "error": "training failed"}

        results = await run_in_threadpool(engine.predict, {symbol.upper(): df})
        return {
            "available": True,
            "symbol": symbol.upper(),
            "forecasts": [
                {
                    "horizon_bars": r.horizon,
                    "p10": round(r.p10, 6),
                    "p50": round(r.p50, 6),
                    "p90": round(r.p90, 6),
                    "direction": r.direction,
                    "uncertainty": round(r.p90 - r.p10, 6),
                }
                for r in results
            ],
        }
    except Exception as e:
        log.exception("TFT endpoint error")
        return {"available": False, "error": str(e)}


@app.get("/api/model_zoo/xgb/{symbol}")
async def model_zoo_xgb(symbol: str, lookback: int = Query(500, ge=200, le=2000)):
    """XGBoost alpha signal — multi-horizon directional forecast."""
    try:
        from ml.xgb_alpha import get_engine
        df = await run_in_threadpool(_gs_load_data, symbol.upper(), lookback)
        if df is None or len(df) < 500:
            return {"available": False, "error": "insufficient data (need 500+ bars)"}

        engine = get_engine()
        if not engine.is_trained(symbol.upper()):
            metrics = await run_in_threadpool(engine.train, {symbol.upper(): df})
            if not metrics:
                return {"available": False, "error": "training failed"}

        results = await run_in_threadpool(engine.predict, {symbol.upper(): df})
        return {
            "available": True,
            "symbol": symbol.upper(),
            "signals": [
                {
                    "horizon_bars": r.horizon,
                    "direction": r.direction,
                    "probability": r.probability,
                    "predicted_return": r.predicted_return,
                    "top_features": r.feature_importance,
                }
                for r in results
            ],
        }
    except Exception as e:
        log.exception("XGB endpoint error")
        return {"available": False, "error": str(e)}


@app.get("/api/model_zoo/vae_vol/{symbol}")
async def model_zoo_vae_vol(symbol: str):
    """VAE vol surface anomaly score — flags when IV surface diverges from learned manifold."""
    try:
        from ml.vae_vol import get_vae_engine
        from strategies.vol_surface import build_demo_surface
        from ml.gs_analytics import compute_timeseries_analytics
        from ml.vae_vol import surface_to_array

        df = await run_in_threadpool(_gs_load_data, symbol.upper(), 252)
        if df is None or len(df) < 30:
            return {"available": False, "error": "insufficient data"}

        analytics = await run_in_threadpool(compute_timeseries_analytics, df, symbol.upper())
        close = float(df["close"].iloc[-1])
        atm_iv = analytics.get("annualized_vol_21") or 0.25

        surface = build_demo_surface(spot=close, atm_vol=atm_iv)
        grid    = surface.surface_grid()
        arr     = surface_to_array(grid)
        if arr is None:
            return {"available": False, "error": "surface grid conversion failed"}

        engine = get_vae_engine()
        if engine.model is None:
            # Auto-train on generated demo surfaces
            surfaces_list = []
            for _ in range(80):
                iv_bump = np.random.uniform(-0.05, 0.05)
                s = build_demo_surface(spot=close, atm_vol=max(0.05, atm_iv + iv_bump))
                a = surface_to_array(s.surface_grid())
                if a is not None:
                    surfaces_list.append(a)
            if surfaces_list:
                await run_in_threadpool(engine.train, surfaces_list, 20, False)

        score, is_anom, label = await run_in_threadpool(engine.score, arr)
        latent = await run_in_threadpool(engine.latent, arr)
        recon  = await run_in_threadpool(engine.reconstruct, arr)

        return {
            "available": True,
            "symbol":     symbol.upper(),
            "atm_iv":     round(atm_iv, 6),
            "anomaly_score": round(score, 6),
            "is_anomaly": is_anom,
            "label":      label,
            "threshold":  round(engine._threshold, 6),
            "latent_z":   [round(float(v), 4) for v in latent],
            "reconstructed_atm": round(float(recon.mean()), 6),
        }
    except Exception as e:
        log.exception("VAE vol endpoint error")
        return {"available": False, "error": str(e)}


@app.get("/api/model_zoo/hmm/{symbol}")
async def model_zoo_hmm(symbol: str, lookback: int = Query(252, ge=60, le=500)):
    """HMM regime detection — state probabilities for latest bar."""
    try:
        from ml.regime_hmm import RegimeHMM, REGIME_LABELS
        df = await run_in_threadpool(_gs_load_data, symbol.upper(), lookback)
        if df is None or len(df) < 60:
            return {"available": False, "error": "insufficient data"}

        if not {"close", "volume"}.issubset(df.columns):
            return {"available": False, "error": "missing close/volume columns"}

        hmm_m = RegimeHMM(n_states=4)
        await run_in_threadpool(hmm_m.fit, df)
        current_regime  = await run_in_threadpool(hmm_m.predict_regime, df)
        proba           = await run_in_threadpool(hmm_m.predict_proba, df)
        sequence        = await run_in_threadpool(hmm_m.predict_sequence, df)
        regime_series   = sequence[-min(252, len(sequence)):]
        counts          = {r: regime_series.count(r) for r in set(regime_series)}

        return {
            "available":      True,
            "symbol":         symbol.upper(),
            "current_regime": current_regime,
            "probabilities":  {k: round(v, 4) for k, v in proba.items()},
            "regime_counts":  counts,
            "dominant_regime": max(counts, key=counts.get) if counts else current_regime,
        }
    except Exception as e:
        log.exception("HMM endpoint error")
        return {"available": False, "error": str(e)}


@app.get("/api/model_zoo/kill_switch/status")
async def kill_switch_status():
    """Kill switch current status."""
    from risk.kill_switch import get_kill_switch
    ks = get_kill_switch()
    return ks.status()


@app.post("/api/model_zoo/kill_switch/evaluate")
async def kill_switch_evaluate(body: dict):
    """Evaluate a risk snapshot against kill-switch thresholds."""
    from risk.kill_switch import get_kill_switch, RiskSnapshot
    try:
        snap = RiskSnapshot(
            nav              = float(body.get("nav", 100_000)),
            peak_nav         = float(body.get("peak_nav", 100_000)),
            start_nav        = float(body.get("start_nav", 100_000)),
            var_1d_pct       = float(body.get("var_1d_pct", 0.0)),
            es_1d_pct        = float(body.get("es_1d_pct", 0.0)),
            gross_leverage   = float(body.get("gross_leverage", 1.0)),
            hhi              = float(body.get("hhi", 0.1)),
            avg_pairwise_corr= float(body.get("avg_pairwise_corr", 0.0)),
        )
        ks = get_kill_switch()
        decision = ks.evaluate(snap)
        return {
            "action": decision.action.value,
            "reason": decision.reason,
            "scale":  decision.scale,
            "checks": decision.checks,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/model_zoo/kill_switch/reset")
async def kill_switch_reset(body: dict = {}):
    """Reset the kill switch (requires reason)."""
    from risk.kill_switch import get_kill_switch
    reason = str(body.get("reason", "manual_reset"))
    get_kill_switch().reset(reason)
    return {"status": "reset", "reason": reason}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("Starting Quant Dashboard on http://0.0.0.0:%d", PORT)
    uvicorn.run(
        "dashboard.server:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
