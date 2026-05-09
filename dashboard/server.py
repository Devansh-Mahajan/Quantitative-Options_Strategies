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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _db_conn() -> sqlite3.Connection:
    c = sqlite3.connect(state._DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


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

            returns = [r.total_return_pct for r in ft_result.results]
            sharpes = [r.sharpe for r in ft_result.results]
            maxdds  = [r.max_drawdown_pct for r in ft_result.results]

            profitable_pct = (sum(1 for r in returns if r > 0) / len(returns) * 100) if returns else 0

            _ft_jobs[job_id]["status"] = "DONE"
            _ft_jobs[job_id]["result"] = {
                "n_paths":          n_paths,
                "median_return_pct":round(float(np.median(returns)), 3) if returns else None,
                "mean_return_pct":  round(float(np.mean(returns)), 3)   if returns else None,
                "p10_return_pct":   round(float(np.percentile(returns, 10)), 3) if returns else None,
                "p90_return_pct":   round(float(np.percentile(returns, 90)), 3) if returns else None,
                "median_sharpe":    round(float(np.median(sharpes)), 4) if sharpes else None,
                "p10_sharpe":       round(float(np.percentile(sharpes, 10)), 4) if sharpes else None,
                "p90_sharpe":       round(float(np.percentile(sharpes, 90)), 4) if sharpes else None,
                "median_maxdd_pct": round(float(np.median(maxdds)), 3) if maxdds else None,
                "profitable_pct":   round(profitable_pct, 1),
                "return_hist":      histogram(returns),
                "sharpe_hist":      histogram(sharpes),
                "maxdd_hist":       histogram(maxdds),
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
