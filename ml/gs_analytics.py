"""
GS-Quant Analytics Engine.

Wraps Goldman Sachs gs-quant timeseries module to produce institutional-grade
analytics from our OHLCV data. All functions work without GS Marquee credentials —
they are pure mathematical implementations on pandas Series.

Modules used:
    gs_quant.timeseries  — 300+ financial time series analytics
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger("gs_analytics")

# ─────────────────────────────────────────────────────────────────────────────
# Lazy gs-quant imports (avoids hard failure if not installed)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import gs_quant.timeseries as ts
    _GS = True
except Exception:
    ts = None
    _GS = False
    log.warning("gs-quant not available — falling back to manual implementations")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(fn, *args, fallback=None, **kwargs):
    try:
        result = fn(*args, **kwargs)
        if isinstance(result, pd.Series):
            return result.dropna()
        return result
    except Exception as e:
        log.debug("gs_analytics %s failed: %s", fn.__name__ if hasattr(fn, '__name__') else fn, e)
        return fallback


def _or(gs_result, fallback_fn, *args, **kwargs):
    """Use gs_result if it's a non-empty Series/scalar, else call fallback_fn."""
    if gs_result is None:
        return fallback_fn(*args, **kwargs)
    if isinstance(gs_result, pd.Series) and len(gs_result) == 0:
        return fallback_fn(*args, **kwargs)
    if isinstance(gs_result, pd.DataFrame) and gs_result.empty:
        return fallback_fn(*args, **kwargs)
    return gs_result


def _last(series, fallback=None):
    if series is None or (hasattr(series, '__len__') and len(series) == 0):
        return fallback
    try:
        v = series.iloc[-1] if isinstance(series, pd.Series) else float(series)
        return None if (np.isnan(v) or np.isinf(v)) else float(v)
    except Exception:
        return fallback


def _to_close(df: pd.DataFrame) -> pd.Series:
    """Extract close prices as a sorted pd.Series with DatetimeIndex."""
    col = next((c for c in ("close", "Close", "c", "price") if c in df.columns), None)
    if col is None:
        raise ValueError("No close column found")
    s = df[col].astype(float)
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index, unit="ms")
        except Exception:
            s.index = pd.to_datetime(s.index)
    return s.sort_index().dropna()


def _to_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


# Manual ratio implementations — used when gs-quant needs GS credentials
def _sharpe(rets: pd.Series, rf: float = 0.0, ann: int = 252) -> float | None:
    mu = rets.mean()
    sd = rets.std()
    if sd == 0 or np.isnan(sd):
        return None
    return float((mu - rf / ann) / sd * np.sqrt(ann))

def _sortino(rets: pd.Series, rf: float = 0.0, ann: int = 252) -> float | None:
    mu = rets.mean()
    down = rets[rets < 0].std()
    if down == 0 or np.isnan(down):
        return None
    return float((mu - rf / ann) / down * np.sqrt(ann))

def _max_drawdown(close: pd.Series) -> float | None:
    roll_max = close.cummax()
    dd = (close - roll_max) / roll_max
    v = float(dd.min())
    return None if np.isnan(v) else v

def _calmar(close: pd.Series, ann: int = 252) -> float | None:
    rets = close.pct_change().dropna()
    annual_ret = float((1 + rets.mean()) ** ann - 1)
    mdd = _max_drawdown(close)
    if mdd is None or mdd == 0:
        return None
    return float(annual_ret / abs(mdd))

def _skewness(rets: pd.Series) -> float | None:
    v = float(rets.skew())
    return None if np.isnan(v) else v

def _kurtosis(rets: pd.Series) -> float | None:
    v = float(rets.kurt())
    return None if np.isnan(v) else v

def _semi_var(rets: pd.Series) -> float | None:
    down = rets[rets < 0]
    if len(down) == 0:
        return 0.0
    return float((down ** 2).mean())

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()

def _macd_calc(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    return pd.DataFrame({"macd": macd_line, "signal": signal_line})

def _bb_calc(close: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return pd.DataFrame({"upper": mid + n_std * std, "lower": mid - n_std * std, "mid": mid})

def _vol_ann(close: pd.Series, window: int) -> pd.Series:
    return np.log(close / close.shift(1)).rolling(window).std() * np.sqrt(252)

def _zscore(close: pd.Series, window: int) -> pd.Series:
    mu = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return (close - mu) / sd.replace(0, np.nan)

# ─────────────────────────────────────────────────────────────────────────────
# Vol Cone
# ─────────────────────────────────────────────────────────────────────────────

def compute_vol_cone(df: pd.DataFrame) -> dict:
    """
    Realized vol cone at 5/10/21/42/63/126/252 bar windows.
    Returns for each window: current, 5th, 25th, 50th, 75th, 95th percentile.
    """
    close = _to_close(df)
    windows = [5, 10, 21, 42, 63, 126, 252]
    cone: dict[str, Any] = {}
    for w in windows:
        if _GS:
            rv = _safe(ts.volatility, close, w)
        else:
            rv = None
        if rv is None or len(rv) == 0:
            rv = _vol_ann(close, w).dropna()
        if rv is None or len(rv) < 5:
            continue
        cone[str(w)] = {
            "current": _last(rv),
            "p5":  float(rv.quantile(0.05)),
            "p25": float(rv.quantile(0.25)),
            "p50": float(rv.quantile(0.50)),
            "p75": float(rv.quantile(0.75)),
            "p95": float(rv.quantile(0.95)),
        }
    return cone


# ─────────────────────────────────────────────────────────────────────────────
# Full Time Series Analytics
# ─────────────────────────────────────────────────────────────────────────────

def compute_timeseries_analytics(df: pd.DataFrame, symbol: str = "") -> dict:
    """
    Run the full gs-quant analytics suite on OHLCV data.
    Returns a rich dict consumed by the dashboard /api/gsquant/timeseries endpoint.
    """
    close = _to_close(df)
    rets  = _to_returns(close)

    out: dict[str, Any] = {"symbol": symbol, "bars": len(close), "available": True}

    # ── Core statistics — always compute locally (gs-quant ratios need GS creds)
    out["sharpe_ratio"]  = _sharpe(rets)
    out["sortino_ratio"] = _sortino(rets)
    out["calmar_ratio"]  = _calmar(close)
    out["max_drawdown"]  = _max_drawdown(close)
    out["skewness"]      = _skewness(rets)
    out["kurtosis"]      = _kurtosis(rets)
    out["semi_variance"] = _semi_var(rets)

    if _GS:
        v21  = _safe(ts.volatility, close, 21)
        v63  = _safe(ts.volatility, close, 63)
        v252 = _safe(ts.volatility, close, 252)
        out["annualized_vol_21"]  = _last(v21)  or _last(_vol_ann(close, 21))
        out["annualized_vol_63"]  = _last(v63)  or _last(_vol_ann(close, 63))
        out["annualized_vol_252"] = _last(v252) or _last(_vol_ann(close, 252))
    else:
        out["annualized_vol_21"]  = _last(_vol_ann(close, 21))
        out["annualized_vol_63"]  = _last(_vol_ann(close, 63))
        out["annualized_vol_252"] = _last(_vol_ann(close, 252))

    # ── Technical indicators — use gs-quant only for RSI; rest use local impls
    rsi_s  = _or(_safe(ts.relative_strength_index, close, 14) if _GS else None, _rsi, close, 14)
    bb     = _bb_calc(close, 20, 2)
    ema_21 = _ema(close, 21)
    ema_50 = _ema(close, 50)
    ema_200= _ema(close, 200)
    macd_s = _macd_calc(close)

    out["rsi14"]    = _last(rsi_s)
    out["rsi_14"]   = out["rsi14"]
    out["ema_21"]   = _last(ema_21)
    out["ema_50"]   = _last(ema_50)
    out["ema_200"]  = _last(ema_200)
    out["bb_upper"] = _last(bb["upper"] if isinstance(bb, pd.DataFrame) and "upper" in bb.columns else None)
    out["bb_lower"] = _last(bb["lower"] if isinstance(bb, pd.DataFrame) and "lower" in bb.columns else None)
    out["bb_mid"]   = _last(bb["mid"]   if isinstance(bb, pd.DataFrame) and "mid"   in bb.columns else None)
    out["macd"]     = _last(macd_s["macd"]   if isinstance(macd_s, pd.DataFrame) and "macd"   in macd_s.columns else None)
    out["macd_signal"] = _last(macd_s["signal"] if isinstance(macd_s, pd.DataFrame) and "signal" in macd_s.columns else None)

    # ── Z-scores ─────────────────────────────────────────────────────────
    zs_1y = _or(_safe(ts.zscores, close, 252) if _GS else None, _zscore, close, 252)
    zs_1m = _or(_safe(ts.zscores, close, 21)  if _GS else None, _zscore, close, 21)
    out["zscore_1y"] = _last(zs_1y)
    out["zscore_1m"] = _last(zs_1m)

    # ── Realised variance ─────────────────────────────────────────────────
    rv21 = _or(_safe(ts.realized_volatility, close, 21) if _GS else None, _vol_ann, close, 21)
    rv63 = _or(_safe(ts.realized_volatility, close, 63) if _GS else None, _vol_ann, close, 63)
    out["realized_vol_21"] = _last(rv21)
    out["realized_vol_63"] = _last(rv63)

    # ── Returns distribution ──────────────────────────────────────────────
    out["avg_daily_return"] = float(rets.mean()) if len(rets) else None
    out["std_daily_return"] = float(rets.std())  if len(rets) else None
    out["best_day"]         = float(rets.max())  if len(rets) else None
    out["worst_day"]        = float(rets.min())  if len(rets) else None
    out["positive_days_pct"]= float((rets > 0).mean()) if len(rets) else None

    # ── Rolling series (last 252 bars for charts) ─────────────────────────
    rv_series  = _or(_safe(ts.volatility, close, 21)               if _GS else None, _vol_ann, close, 21)
    rsi_series = _or(_safe(ts.relative_strength_index, close, 14)  if _GS else None, _rsi, close, 14)
    # Rolling Sharpe (60-day window): manually computed as it requires no external data
    sr_series  = rets.rolling(60).apply(lambda r: _sharpe(pd.Series(r)) or 0, raw=False)
    # Running max drawdown series (local)
    roll_max   = close.cummax()
    dd_series  = (close - roll_max) / roll_max

    def _tail(s, n=252):
        if s is None or len(s) == 0:
            return []
        tail = s.tail(n)
        return list(zip(
            [int(t.timestamp() * 1000) for t in tail.index],
            [round(float(v), 6) if v is not None and not np.isnan(v) else None for v in tail.values]
        ))

    out["series"] = {
        "rolling_vol_21":   _tail(rv_series),
        "rolling_sharpe":   _tail(sr_series),
        "max_drawdown":     _tail(dd_series),
        "rsi_14":           _tail(rsi_series),
        "close":            _tail(close.tail(252) if isinstance(close, pd.Series) else None),
    }

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Asset Correlation Matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_correlation_matrix(df_dict: dict[str, pd.DataFrame]) -> dict:
    """
    Rolling 63-day return correlation matrix across all symbols.
    Returns full matrix + pairwise list sorted by abs correlation.
    """
    closes = {}
    for sym, df in df_dict.items():
        try:
            closes[sym] = _to_close(df)
        except Exception:
            continue

    if len(closes) < 2:
        return {"available": False}

    combined = pd.DataFrame(closes).pct_change().dropna()

    if _GS:
        try:
            corr = _safe(ts.correlation, combined.iloc[:, 0], combined.iloc[:, 1], 63)
        except Exception:
            corr = None

    corr_matrix = combined.tail(63).corr()

    syms = list(corr_matrix.columns)
    pairs = []
    for i, s1 in enumerate(syms):
        for s2 in syms[i+1:]:
            v = corr_matrix.loc[s1, s2]
            if not np.isnan(v):
                pairs.append({"s1": s1, "s2": s2, "corr": round(float(v), 4)})

    pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)

    matrix = {
        sym: {
            sym2: round(float(corr_matrix.loc[sym, sym2]), 4)
            for sym2 in syms if not np.isnan(corr_matrix.loc[sym, sym2])
        }
        for sym in syms
    }

    return {
        "available": True,
        "symbols": syms,
        "matrix": matrix,
        "top_pairs": pairs[:20],
        "window": 63,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rolling Beta to Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_beta(df: pd.DataFrame, benchmark_df: pd.DataFrame,
                          symbol: str = "", window: int = 63) -> dict:
    """Rolling beta to benchmark using gs-quant beta function."""
    try:
        asset_close = _to_close(df)
        bench_close = _to_close(benchmark_df)
    except Exception as e:
        return {"available": False, "error": str(e)}

    asset_ret = _to_returns(asset_close)
    bench_ret = _to_returns(bench_close)

    # Align on common dates
    combined = pd.DataFrame({"asset": asset_ret, "bench": bench_ret}).dropna()

    if _GS:
        beta_series = _safe(ts.beta, combined["asset"], combined["bench"], window)
        r2_series   = _safe(ts.r_squared, combined["asset"], combined["bench"], window)
    else:
        beta_series = combined["asset"].rolling(window).cov(combined["bench"]) / combined["bench"].rolling(window).var()
        r2_series   = None

    def _tail(s, n=252):
        if s is None or len(s) == 0:
            return []
        tail = s.tail(n)
        return list(zip(
            [int(t.timestamp() * 1000) for t in tail.index],
            [round(float(v), 6) if not np.isnan(v) else None for v in tail.values]
        ))

    return {
        "available": True,
        "symbol": symbol,
        "window": window,
        "current_beta": _last(beta_series),
        "current_r2":   _last(r2_series),
        "series": {
            "beta": _tail(beta_series),
            "r2":   _tail(r2_series),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Risk Analytics (VaR, CVaR, Stress)
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_analytics(df: pd.DataFrame, symbol: str = "",
                            confidence: float = 0.95) -> dict:
    """
    Full risk analytics: historical VaR, CVaR, drawdown statistics,
    and vol-of-vol using gs-quant timeseries.
    """
    close = _to_close(df)
    rets  = _to_returns(close)

    if len(rets) < 20:
        return {"available": False}

    # Historical VaR / CVaR
    q = 1 - confidence
    var_1d  = float(rets.quantile(q))
    cvar_1d = float(rets[rets <= var_1d].mean()) if (rets <= var_1d).any() else var_1d

    # Scale to 5d / 10d / 21d (sqrt-of-time)
    var_5d  = var_1d  * np.sqrt(5)
    var_10d = var_1d  * np.sqrt(10)
    var_21d = var_1d  * np.sqrt(21)

    # Drawdown stats
    rolling_max = close.cummax()
    dd_series   = (close - rolling_max) / rolling_max
    current_dd  = float(dd_series.iloc[-1])

    semi_var  = _semi_var(rets)
    track_err = float(rets.tail(63).std() * np.sqrt(252))

    # Drawdown length (consecutive bars below peak)
    below_peak = (close < rolling_max)
    streak = 0
    for v in below_peak.values[::-1]:
        if v:
            streak += 1
        else:
            break
    current_dd_len = streak

    # Vol of vol
    rv21 = _vol_ann(close, 21)
    vol_of_vol = float(rv21.tail(21).std()) if len(rv21) >= 21 else None

    # Stress scenarios (parametric)
    current_price = float(close.iloc[-1])
    daily_vol = float(rets.std())
    stress = {
        "down_1sigma":  round(current_price * (1 - daily_vol),      2),
        "down_2sigma":  round(current_price * (1 - 2 * daily_vol),  2),
        "down_3sigma":  round(current_price * (1 - 3 * daily_vol),  2),
        "up_1sigma":    round(current_price * (1 + daily_vol),      2),
        "up_2sigma":    round(current_price * (1 + 2 * daily_vol),  2),
        "up_3sigma":    round(current_price * (1 + 3 * daily_vol),  2),
        "pct_move_1sigma":  round(daily_vol * 100, 4),
        "pct_move_2sigma":  round(2 * daily_vol * 100, 4),
        "pct_move_3sigma":  round(3 * daily_vol * 100, 4),
    }

    return {
        "available":     True,
        "symbol":        symbol,
        "confidence":    confidence,
        "var_1d":        round(var_1d  * 100, 4),
        "var_5d":        round(var_5d  * 100, 4),
        "var_10d":       round(var_10d * 100, 4),
        "var_21d":       round(var_21d * 100, 4),
        "var_1d_pct":    round(var_1d  * 100, 4),
        "var_5d_pct":    round(var_5d  * 100, 4),
        "var_10d_pct":   round(var_10d * 100, 4),
        "var_21d_pct":   round(var_21d * 100, 4),
        "cvar_1d_pct":   round(cvar_1d * 100, 4),
        "current_drawdown_pct": round(current_dd * 100, 4),
        "drawdown_length_bars": current_dd_len,
        "semi_variance": round(semi_var, 8) if semi_var is not None else None,
        "vol_of_vol":    round(vol_of_vol, 6) if vol_of_vol is not None else None,
        "tracking_error":round(track_err, 6),
        "stress_scenarios": stress,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio-Level Analytics (multi-asset)
# ─────────────────────────────────────────────────────────────────────────────

def compute_portfolio_analytics(df_dict: dict[str, pd.DataFrame],
                                 weights: dict[str, float] | None = None) -> dict:
    """
    Portfolio-level analytics: blended return stream, Sharpe, max DD,
    skewness, kurtosis using gs-quant portfolio_* functions.
    """
    closes = {}
    for sym, df in df_dict.items():
        try:
            closes[sym] = _to_close(df)
        except Exception:
            continue

    if not closes:
        return {"available": False}

    combined = pd.DataFrame(closes).pct_change().dropna()
    syms = list(combined.columns)

    if weights is None:
        w = {s: 1.0 / len(syms) for s in syms}
    else:
        total = sum(weights.get(s, 0) for s in syms)
        w = {s: weights.get(s, 0) / max(total, 1e-9) for s in syms}

    port_ret = sum(combined[s] * w.get(s, 0) for s in syms)

    cum_ret = (1 + port_ret).cumprod()
    sh = _sharpe(port_ret)
    so = _sortino(port_ret)
    ca = _calmar(cum_ret)
    dd = _max_drawdown(cum_ret)
    sk = _skewness(port_ret)
    ku = _kurtosis(port_ret)
    te = float(port_ret.std() * np.sqrt(252))

    annual_ret = (1 + float(port_ret.mean())) ** 252 - 1
    annual_vol = float(port_ret.std()) * np.sqrt(252)

    return {
        "available":       True,
        "symbols":         syms,
        "weights":         {s: round(w[s], 4) for s in syms},
        "annual_return":   round(annual_ret, 6),
        "annual_vol":      round(annual_vol, 6),
        "sharpe_ratio":    sh,
        "sortino_ratio":   so,
        "calmar_ratio":    ca,
        "max_drawdown":    dd,
        "skewness":        sk,
        "kurtosis":        ku,
        "tracking_error":  te,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Regime Detection via Z-Scores
# ─────────────────────────────────────────────────────────────────────────────

def compute_zscore_regime(df: pd.DataFrame, symbol: str = "") -> dict:
    """
    Multi-window z-score regime signals (1m, 3m, 1y).
    Positive z → above mean (momentum), negative → below (reversion candidate).
    """
    close = _to_close(df)

    out = {"symbol": symbol, "available": True}
    for label, window in [("1m", 21), ("3m", 63), ("1y", 252)]:
        zs = _or(_safe(ts.zscores, close, window) if _GS else None, _zscore, close, window)
        out[f"zscore_{label}"] = _last(zs)
        out[f"zscore_{label}_series"] = (
            list(zip(
                [int(t.timestamp() * 1000) for t in zs.tail(252).index],
                [round(float(v), 4) for v in zs.tail(252).values]
            )) if zs is not None and len(zs) else []
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MACD + RSI Signals
# ─────────────────────────────────────────────────────────────────────────────

def compute_technical_signals(df: pd.DataFrame, symbol: str = "") -> dict:
    """
    Full technical signal suite using gs-quant timeseries.
    Returns current values + signal direction + recent series.
    """
    close = _to_close(df)
    out = {"symbol": symbol, "available": True}

    # RSI — gs-quant works for this
    rsi = _or(_safe(ts.relative_strength_index, close, 14) if _GS else None, _rsi, close, 14)
    rsi_v = _last(rsi)
    out["rsi_14"] = rsi_v
    out["rsi_signal"] = "overbought" if (rsi_v or 50) > 70 else "oversold" if (rsi_v or 50) < 30 else "neutral"

    # MACD — local only (gs-quant returns single Series, not MACD+signal DataFrame)
    macd_df = _macd_calc(close)
    if isinstance(macd_df, pd.DataFrame):
        macd_v   = _last(macd_df.get("macd"))
        signal_v = _last(macd_df.get("signal"))
        hist_v   = (macd_v or 0) - (signal_v or 0)
        out["macd"] = macd_v
        out["macd_signal"] = signal_v
        out["macd_hist"] = round(hist_v, 6)
        out["macd_cross"] = "bullish" if hist_v > 0 else "bearish"

    # Bollinger Bands — local only (gs-quant returns unnamed integer columns)
    bb = _bb_calc(close, 20, 2)
    if isinstance(bb, pd.DataFrame):
        upper = _last(bb["upper"])
        lower = _last(bb["lower"])
        mid   = _last(bb["mid"])
        price = float(close.iloc[-1])
        out["bb_upper"] = upper
        out["bb_lower"] = lower
        out["bb_mid"]   = mid
        out["bb_width"] = round(((upper or 0) - (lower or 0)) / (mid or 1), 6)
        if upper and lower:
            out["bb_pct"] = round((price - lower) / max(upper - lower, 1e-9), 4)
            out["bb_signal"] = "overbought" if price >= upper else "oversold" if price <= lower else "neutral"

    # EMAs — local only (gs-quant EMA uses alpha param, incompatible)
    e21  = _ema(close, 21)
    e50  = _ema(close, 50)
    e200 = _ema(close, 200)
    out["ema_21"]  = _last(e21)
    out["ema_50"]  = _last(e50)
    out["ema_200"] = _last(e200)
    price = float(close.iloc[-1])
    if out.get("ema_50") and out.get("ema_200"):
        out["golden_cross"] = out["ema_50"] > out["ema_200"]

    # Z-score 21d — gs-quant works for this
    zs = _or(_safe(ts.zscores, close, 21) if _GS else None, _zscore, close, 21)
    out["zscore_21"] = _last(zs)

    # Rolling series for charting
    def _tail(s, n=252):
        if s is None or len(s) == 0:
            return []
        t = s.tail(n) if isinstance(s, pd.Series) else s
        return [[int(idx.timestamp() * 1000), round(float(v), 6) if not np.isnan(v) else None]
                for idx, v in zip(t.index, t.values)]

    out["series"] = {
        "rsi":    _tail(rsi),
        "ema_21": _tail(e21),
        "ema_50": _tail(e50),
        "close":  _tail(close),
    }
    if isinstance(macd_df, pd.DataFrame):
        out["series"]["macd"]        = _tail(macd_df["macd"])
        out["series"]["macd_signal"] = _tail(macd_df["signal"])

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Asset Factor Analytics
# ─────────────────────────────────────────────────────────────────────────────

def compute_factor_analytics(df_dict: dict[str, pd.DataFrame]) -> dict:
    """
    Compute factor analytics:  momentum factor, mean-reversion factor,
    vol factor — ranked across universe.
    """
    results = {}
    for sym, df in df_dict.items():
        try:
            close = _to_close(df)
            rets  = _to_returns(close)
            if len(rets) < 63:
                continue

            mom_1m  = float((close.iloc[-1] / close.iloc[-22]) - 1) if len(close) >= 22  else None
            mom_3m  = float((close.iloc[-1] / close.iloc[-63]) - 1) if len(close) >= 63  else None
            mom_12m = float((close.iloc[-1] / close.iloc[-252])- 1) if len(close) >= 252 else None

            vol_21 = float(rets.tail(21).std() * np.sqrt(252))
            vol_63 = float(rets.tail(63).std() * np.sqrt(252))

            mean_rev_z = float((close.iloc[-1] - close.tail(21).mean()) / max(close.tail(21).std(), 1e-9))

            results[sym] = {
                "momentum_1m":  round(mom_1m or 0, 6),
                "momentum_3m":  round(mom_3m or 0, 6),
                "momentum_12m": round(mom_12m or 0, 6),
                "vol_21":       round(vol_21, 6),
                "vol_63":       round(vol_63, 6),
                "mean_rev_zscore": round(mean_rev_z, 4),
                "vol_ratio":    round(vol_21 / max(vol_63, 1e-9), 4),
            }
        except Exception as e:
            log.debug("factor analytics failed for %s: %s", sym, e)

    if not results:
        return {"available": False}

    # Rank all factors cross-sectionally
    syms = list(results.keys())
    for factor in ("momentum_1m", "momentum_3m", "momentum_12m",
                   "vol_21", "vol_63", "mean_rev_zscore"):
        vals = [(s, results[s][factor]) for s in syms if results[s].get(factor) is not None]
        vals_sorted = sorted(vals, key=lambda x: x[1])
        n = len(vals_sorted)
        for rank, (s, _) in enumerate(vals_sorted):
            results[s][f"{factor}_rank"] = round(rank / max(n - 1, 1), 4)

    # Top/bottom momentum
    mom_ranked = sorted(syms, key=lambda s: results[s].get("momentum_1m", 0), reverse=True)

    return {
        "available": True,
        "universe_size": len(results),
        "factors": results,
        "top_momentum": mom_ranked[:10],
        "bottom_momentum": mom_ranked[-10:],
    }
