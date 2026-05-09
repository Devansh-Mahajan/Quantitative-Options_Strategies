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
import sqlite3
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
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.config import cfg
from bot import state
from dashboard.analytics import (
    build_elite_overview,
    build_options_chain,
    build_options_overview,
    build_simulation_payload,
    build_stocks_overview,
    build_trade_odds,
    instantiate_strategy,
    permutation_grid,
    supported_strategy_params,
    RUNTIME_DIR,
    RISK_SNAPSHOT_PATH,
    PORTFOLIO_GUARD_PATH,
    read_json,
    safe_float,
    safe_int,
)

log = logging.getLogger("dashboard.server")

PORT     = int(cfg.__dict__.get("dashboard_port", 8080) if hasattr(cfg, "__dict__") else 8080)
STATIC   = Path(__file__).parent / "static"
LOG_DIR  = cfg.log_dir
WF_CONN: set[WebSocket] = set()   # live broadcast connections


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

TRAINING_SCRIPTS: dict[str, dict] = {
    "weekend_training":        {"label": "Weekend Pipeline",       "module": "scripts.weekend_training",             "est_min": 60,  "desc": "Full retrain — data download, all models, validation, deploy"},
    "train_hmm":               {"label": "HMM Macro Regime",       "module": "scripts.train_hmm",                    "est_min": 5,   "desc": "Retrain Hidden Markov Model for macro-regime detection"},
    "train_mega_brain":        {"label": "Mega Brain GPU",         "module": "scripts.train_mega_brain",             "est_min": 90,  "desc": "Full GPU-accelerated ensemble (wraps mega_gpu_training)"},
    "train_gpu_brain":         {"label": "GPU Brain",              "module": "scripts.train_gpu_brain",              "est_min": 45,  "desc": "GPU model training pipeline"},
    "train_correlation_alpha": {"label": "Correlation Alpha",      "module": "scripts.train_correlation_alpha",      "est_min": 15,  "desc": "Cross-asset correlation alpha models"},
    "train_regime_movement":   {"label": "Regime + Movement",      "module": "scripts.train_regime_movement_models", "est_min": 20,  "desc": "Regime detection and movement prediction retraining"},
    "weekend_recalibration":   {"label": "Weekend Recalibration",  "module": "scripts.weekend_recalibration",        "est_min": 30,  "desc": "Recalibrate all live strategy parameters from fresh data"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _db_conn() -> sqlite3.Connection:
    c = sqlite3.connect(state._DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


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


def _compute_metrics(equity: list[float], timestamps: list[str]) -> dict:
    """Compute portfolio performance metrics from equity curve."""
    if len(equity) < 2:
        return {}
    arr = np.array(equity, dtype=float)
    rets = np.diff(arr) / (arr[:-1] + 1e-10)

    # Annualise assuming 1h bars (8760 per year)
    ann = 8760
    total_ret = (arr[-1] / arr[0]) - 1
    vol       = float(rets.std() * np.sqrt(ann))
    sharpe    = float((rets.mean() * ann) / (rets.std() * np.sqrt(ann) + 1e-10))

    # Drawdown
    peak = np.maximum.accumulate(arr)
    dd   = (arr - peak) / (peak + 1e-10)
    max_dd = float(dd.min())

    # Sortino (downside deviation)
    neg = rets[rets < 0]
    down_vol = float(neg.std() * np.sqrt(ann)) if len(neg) > 1 else 1e-10
    sortino  = float((rets.mean() * ann) / (down_vol + 1e-10))

    # Calmar
    calmar = float((total_ret / abs(max_dd + 1e-10)) if max_dd < 0 else 0)

    return {
        "total_return_pct": round(total_ret * 100, 2),
        "annualised_return_pct": round((((arr[-1] / arr[0]) ** (ann / max(len(arr)-1, 1))) - 1) * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "volatility_pct": round(vol * 100, 2),
        "current_equity": round(float(arr[-1]), 2),
        "peak_equity": round(float(arr.max()), 2),
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

            bal = await client.get_futures_balance()
            equity = float(bal.get("USDT", 0) or bal.get("USD", 0) or 0)

            if equity > 0:
                state.record_equity(equity, 0)
                await _broadcast({"type": "equity_tick", "equity": equity,
                                   "ts": datetime.now(timezone.utc).isoformat()})
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
    return FileResponse(str(STATIC / "index.html"))


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
    dd     = [float(r["drawdown"]) for r in rows]
    return {
        "timestamps": ts,
        "equity":     equity,
        "drawdown":   dd,
        "metrics":    _compute_metrics(equity, ts),
    }


@app.get("/api/benchmark")
async def get_benchmark(symbol: str = Query("BTCUSDT"), window: str = Query("1w")):
    """Return benchmark close prices via backtester data loader."""
    try:
        from backtester.data_loader import load_multi
        end   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        delta = {
            "1d": 2, "1w": 8, "1m": 32, "3m": 95, "1y": 366, "max": 1000
        }.get(window, 8)
        start = (datetime.now(timezone.utc) - timedelta(days=delta)).strftime("%Y-%m-%d")
        data  = load_multi([symbol], "1h", start, end)
        df    = data.get(symbol, pd.DataFrame())
        if df.empty:
            return {"timestamps": [], "prices": []}
        df = df.reset_index()
        ts_col = "open_time" if "open_time" in df.columns else df.columns[0]
        return {
            "timestamps": df[ts_col].astype(str).tolist(),
            "prices":     df["close"].round(4).tolist(),
        }
    except Exception as exc:
        log.warning("Benchmark fetch failed: %s", exc)
        return {"timestamps": [], "prices": []}


@app.get("/api/positions")
async def get_positions():
    trades = state.get_open_trades()
    return {"positions": trades, "count": len(trades)}


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
    pnl = state.get_daily_pnl()
    peak = state.get_peak_equity()
    return {"daily_pnl": round(pnl, 4), "peak_equity": round(peak, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# REST — Risk
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/risk/snapshot")
async def risk_snapshot():
    """Compute risk metrics from recent equity history."""
    with _db_conn() as c:
        rows = c.execute(
            "SELECT ts, equity, drawdown FROM equity_curve ORDER BY ts DESC LIMIT 8760"
        ).fetchall()

    if not rows:
        return {}

    equity = [float(r["equity"]) for r in reversed(rows)]
    arr    = np.array(equity, dtype=float)
    rets   = np.diff(arr) / (arr[:-1] + 1e-10)

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
    window = min(720, len(rets))
    for i in range(window, len(rets) + 1):
        rolling_vol.append(float(rets[i - window: i].std() * np.sqrt(8760) * 100))

    return {
        "var_95_pct":  round(var_95 * 100, 3),
        "var_99_pct":  round(var_99 * 100, 3),
        "cvar_95_pct": round(cvar_95 * 100, 3),
        "cvar_99_pct": round(cvar_99 * 100, 3),
        "current_drawdown_pct": round(dd[-1] * 100, 2) if dd else 0,
        "max_drawdown_pct":     round(min(dd) * 100, 2) if dd else 0,
        "rolling_vol_pct":      rolling_vol[-1] if rolling_vol else 0,
        "drawdown_series":      [round(x * 100, 3) for x in dd[-720:]],
    }


@app.get("/api/risk/heatmap")
async def risk_heatmap():
    """
    Return correlation matrix and per-symbol volatility for heatmap display.
    Uses recent Binance/Alpaca kline data.
    """
    try:
        from backtester.data_loader import load_multi
        end   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=32)).strftime("%Y-%m-%d")
        symbols = cfg.futures_symbols[:8]
        data  = load_multi(symbols, "1h", start, end)

        rets_dict = {}
        vols      = {}
        for sym, df in data.items():
            if df.empty:
                continue
            c  = df["close"].astype(float).values
            lr = np.diff(np.log(c))
            rets_dict[sym] = lr
            vols[sym] = round(float(lr.std() * np.sqrt(8760) * 100), 2)

        # Correlation matrix on common length
        syms = list(rets_dict.keys())
        min_len = min(len(v) for v in rets_dict.values())
        if min_len < 10:
            return {"symbols": [], "corr": [], "vols": {}}

        mat  = np.column_stack([rets_dict[s][-min_len:] for s in syms])
        corr = np.corrcoef(mat.T).round(3).tolist()

        return {"symbols": syms, "corr": corr, "vols": vols}

    except Exception as exc:
        log.warning("Heatmap error: %s", exc)
        return {"symbols": [], "corr": [], "vols": {}}


@app.get("/api/risk/var_surface")
async def var_surface():
    """VaR surface at multiple confidence levels and horizons."""
    with _db_conn() as c:
        rows = c.execute(
            "SELECT equity FROM equity_curve ORDER BY ts DESC LIMIT 2000"
        ).fetchall()

    if len(rows) < 30:
        return {"levels": [], "horizons": [], "var": [], "cvar": []}

    equity = np.array([float(r["equity"]) for r in reversed(rows)])
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
        "levels":   [f"{int(l*100)}%" for l in levels],
        "horizons": [f"{h}d" for h in horizons],
        "var":      var_grid,
        "cvar":     cvar_grid,
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
    }


@app.get("/api/strategies/pnl")
async def strategies_pnl():
    """Per-strategy realized P&L from the trades DB."""
    with _db_conn() as c:
        try:
            rows = c.execute(
                "SELECT strategy, pnl, ts FROM trades ORDER BY ts DESC LIMIT 3000"
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
    return build_elite_overview()


@app.get("/api/options/overview")
async def options_overview():
    return build_options_overview()


@app.get("/api/options/chain")
async def options_chain(
    underlying: str | None = Query(None),
    contract_type: str = Query("all"),
    min_dte: int = Query(7, ge=1, le=365),
    max_dte: int = Query(45, ge=1, le=365),
    limit: int = Query(72, ge=12, le=200),
):
    return build_options_chain(
        underlying=underlying,
        contract_type=contract_type,
        min_dte=min_dte,
        max_dte=max_dte,
        limit=limit,
    )


@app.get("/api/options/simulations")
async def options_simulations():
    ts, equity, _ = _equity_series(limit=1200)
    return build_simulation_payload(equity, ts)


@app.get("/api/stocks/overview")
async def stocks_overview():
    return build_stocks_overview()


@app.get("/api/trades/odds")
async def trade_odds():
    return build_trade_odds()


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
    report_dir = Path("backtest_reports")
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
    path = Path("backtest_reports") / f"backtest_{run_id}.json"
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

            symbols    = body.get("symbols", cfg.futures_symbols[:4])
            interval   = body.get("interval", "1h")
            start      = body.get("start", "2024-01-01")
            end        = body.get("end")
            strategies = body.get("strategies", ["momentum", "breakout"])
            capital    = float(body.get("capital", cfg.initial_capital))
            sl         = float(body.get("stop_loss_pct", ENGINE_DEFAULTS["stop_loss_pct"]))
            tp         = float(body.get("take_profit_pct", ENGINE_DEFAULTS["take_profit_pct"]))
            max_pos    = int(body.get("max_open_positions", ENGINE_DEFAULTS["max_open_positions"]))
            lookback   = int(body.get("lookback", ENGINE_DEFAULTS["lookback"]))

            data = load_multi(symbols, interval, start, end)
            data = {s: df for s, df in data.items() if not df.empty}
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
# REST — Forward Test (Bloch Futuretesting Framework)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/forwardtest/run")
async def run_forwardtest(body: dict):
    """
    Submit a forward-test job.
    Body: { symbol, model, cal_start, cal_end, n_paths, horizon, strategy, interval }
    """
    job_id = str(uuid.uuid4())[:8]
    _ft_jobs[job_id] = {"status": "PENDING", "result": None, "error": None}

    def _run():
        try:
            _ft_jobs[job_id]["status"] = "RUNNING"
            from backtester.data_loader import load_multi
            from backtester.futuretesting import FuturetestEngine
            from backtester.engine import BacktestEngine
            from scripts.run_backtest import _build_strategies

            symbol    = body.get("symbol", "BTCUSDT")
            model     = body.get("model", "gbm")
            cal_start = body.get("cal_start", "2023-01-01")
            cal_end   = body.get("cal_end") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            n_paths   = int(body.get("n_paths", 50))
            horizon   = int(body.get("horizon", 500))
            strategy  = body.get("strategy", "momentum")
            interval  = body.get("interval", "1h")

            # Load historical data for calibration
            data = load_multi([symbol], interval, cal_start, cal_end)
            hist_df = data.get(symbol, None)
            if hist_df is None or hist_df.empty:
                raise ValueError(f"No data for {symbol} [{cal_start}→{cal_end}]")

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
            ft_result = ft.run(engine_factory, lambda: strat_list)

            # Build histogram helper
            def histogram(values, bins=20):
                arr = np.array(values)
                counts, edges = np.histogram(arr, bins=bins)
                bin_centers = ((edges[:-1] + edges[1:]) / 2).tolist()
                return {"bins": bin_centers, "counts": counts.tolist()}

            # FuturetestResult stores distributions as numpy arrays (fractions, not pct)
            returns = (ft_result.return_dist * 100).tolist()
            sharpes = ft_result.sharpe_dist.tolist()
            maxdds  = (ft_result.max_dd_dist * 100).tolist()
            win_rates = (ft_result.win_rate_dist * 100).tolist()

            profitable_pct = float(np.mean(ft_result.return_dist > 0)) * 100

            _ft_jobs[job_id]["status"] = "DONE"
            _ft_jobs[job_id]["result"] = {
                "n_paths":           ft_result.n_paths,
                "failed_paths":      ft_result.failed_paths,
                "acceptance_rate":   round(ft_result.acceptance_rate * 100, 1),
                "median_return_pct": round(float(np.median(returns)), 3) if returns else None,
                "mean_return_pct":   round(float(np.mean(returns)), 3)   if returns else None,
                "p5_return_pct":     round(float(np.percentile(returns, 5)), 3)  if returns else None,
                "p10_return_pct":    round(float(np.percentile(returns, 10)), 3) if returns else None,
                "p25_return_pct":    round(float(np.percentile(returns, 25)), 3) if returns else None,
                "p75_return_pct":    round(float(np.percentile(returns, 75)), 3) if returns else None,
                "p90_return_pct":    round(float(np.percentile(returns, 90)), 3) if returns else None,
                "p95_return_pct":    round(float(np.percentile(returns, 95)), 3) if returns else None,
                "median_sharpe":     round(float(np.median(sharpes)), 4) if sharpes else None,
                "p10_sharpe":        round(float(np.percentile(sharpes, 10)), 4) if sharpes else None,
                "p90_sharpe":        round(float(np.percentile(sharpes, 90)), 4) if sharpes else None,
                "median_maxdd_pct":  round(float(np.median(maxdds)), 3)  if maxdds else None,
                "p95_maxdd_pct":     round(float(np.percentile(maxdds, 95)), 3) if maxdds else None,
                "median_win_rate":   round(float(np.median(win_rates)), 1) if win_rates else None,
                "profitable_pct":    round(profitable_pct, 1),
                "return_hist":       histogram(returns),
                "sharpe_hist":       histogram(sharpes),
                "maxdd_hist":        histogram(maxdds),
                "winrate_hist":      histogram(win_rates),
            }

        except Exception as exc:
            import traceback
            _ft_jobs[job_id]["status"] = "ERROR"
            _ft_jobs[job_id]["error"]  = str(exc)
            log.error("Forward test job %s failed: %s\n%s", job_id, exc, traceback.format_exc())

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "PENDING"}


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

            symbols = body.get("symbols", cfg.futures_symbols[:4])
            interval = str(body.get("interval") or "1h")
            start = str(body.get("start") or "2024-01-01")
            end = body.get("end")
            capital = float(body.get("capital", cfg.initial_capital))
            stop_loss_pct = float(body.get("stop_loss_pct", ENGINE_DEFAULTS["stop_loss_pct"]))
            take_profit_pct = float(body.get("take_profit_pct", ENGINE_DEFAULTS["take_profit_pct"]))
            max_open_positions = int(body.get("max_open_positions", ENGINE_DEFAULTS["max_open_positions"]))
            lookback = int(body.get("lookback", ENGINE_DEFAULTS["lookback"]))

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
            await ws.send_json({
                "type": "equity_tick",
                "equity": float(rows[0]["equity"]),
                "ts": rows[0]["ts"],
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

    # Send last 100 lines of history
    lines = log_path.read_text(errors="replace").splitlines()
    for line in lines[-100:]:
        level = "ERROR" if "ERROR" in line else ("WARN" if "WARNING" in line else
                "INFO" if "INFO" in line else "DEBUG")
        await ws.send_json({"line": line, "level": level})

    # Tail new lines
    with log_path.open("r", errors="replace") as f:
        f.seek(0, 2)   # seek to end
        while True:
            line = f.readline()
            if line:
                level = ("ERROR" if "ERROR" in line else
                         "WARN"  if "WARNING" in line else
                         "INFO"  if "INFO" in line else "DEBUG")
                await ws.send_json({"line": line.rstrip(), "level": level})
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
    log_path = LOG_DIR / "risk.log"
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
        "signals": sorted(signals, key=lambda s: abs(s.get("alpha_score", 0)), reverse=True),
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
        "preflight_passed":    pre.get("passed"),
        "preflight_summary":   pre.get("summary", {}),
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
        return {"total": 0}

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
