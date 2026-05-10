"""Feature engineering for all ML models: technical, microstructure, and regime-conditioned features."""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd

log = logging.getLogger("ml.features")


FEATURE_NAMES = [
    # Returns
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
    "log_ret_1", "log_ret_5",
    # Realized volatility (multiple horizons)
    "vol_5", "vol_20", "vol_60",
    "rv_parkinson_5", "rv_parkinson_20",
    # Vol-of-vol and term structure
    "vol_of_vol", "ts_ratio_5_20", "ts_ratio_5_60", "ts_ratio_20_60",
    # EMA and trend
    "ema_9", "ema_21", "ema_55", "ema_200",
    "ema_cross_9_21", "ema_cross_21_55", "price_vs_ema200",
    # Momentum
    "mom_10", "mom_20",
    "roc_10", "roc_20",
    "high52w_pct", "low52w_pct",
    # RSI at multiple periods
    "rsi_7", "rsi_14", "rsi_21",
    # MACD
    "macd", "macd_hist",
    # Bollinger Bands
    "bb_upper", "bb_lower", "bb_pct", "bb_width",
    # ATR
    "atr_7", "atr_14",
    # Volume features
    "vol_ratio", "vol_delta", "buy_sell_ratio",
    "vol_ratio_5_20",
    "mfi_14",
    # Candle microstructure
    "body", "wick_up", "wick_dn", "bar_range",
    # Stochastic
    "stoch_k", "stoch_d",
    # Donchian
    "donchian_high_20", "donchian_low_20",
    # Autocorrelation (Cont 2001 — return serial structure)
    "autocorr_1", "autocorr_5",
    # Distributional moments (higher-order)
    "skewness_20", "kurtosis_20",
    # Hurst proxy (long-range dependence)
    "hurst_proxy",
    # Amihud illiquidity (Kyle lambda proxy)
    "amihud_5", "amihud_20",
    # Crypto-specific
    "funding_rate", "oi_value",
]


def compute_features(df: pd.DataFrame, funding_rate: float = 0.0, open_interest: float = 0.0) -> pd.DataFrame:
    """
    Input df must have columns: open, high, low, close, volume.
    Returns DataFrame with all features, NaN rows dropped.
    """
    f = pd.DataFrame(index=df.index)

    c = df["close"].astype(float)
    h = df["high"].astype(float)
    lo = df["low"].astype(float)
    v = df["volume"].astype(float)
    o = df["open"].astype(float)

    # ── Returns ──────────────────────────────────────────────────────────────
    f["ret_1"]  = c.pct_change(1)
    f["ret_3"]  = c.pct_change(3)
    f["ret_5"]  = c.pct_change(5)
    f["ret_10"] = c.pct_change(10)
    f["ret_20"] = c.pct_change(20)
    f["ret_60"] = c.pct_change(60)
    f["log_ret_1"] = np.log(c / c.shift(1))
    f["log_ret_5"] = np.log(c / c.shift(5))

    lr = f["log_ret_1"]

    # ── Realized volatility (log-return std, annualised hourly) ──────────────
    ann = np.sqrt(252 * 24)
    f["vol_5"]  = lr.rolling(5).std()  * ann
    f["vol_20"] = lr.rolling(20).std() * ann
    f["vol_60"] = lr.rolling(60).std() * ann

    # ── Parkinson (high-low) estimator — more efficient than close-to-close ──
    # Parkinson (1980): σ² = 1/(4·ln2) · E[(ln H/L)²]
    hl_log = np.log(h / (lo + 1e-10))
    park_factor = 1.0 / (4.0 * np.log(2))
    f["rv_parkinson_5"]  = np.sqrt(park_factor * (hl_log ** 2).rolling(5).mean())  * ann
    f["rv_parkinson_20"] = np.sqrt(park_factor * (hl_log ** 2).rolling(20).mean()) * ann

    # ── Vol-of-vol and term structure ─────────────────────────────────────────
    f["vol_of_vol"]    = f["vol_20"].rolling(10).std()
    f["ts_ratio_5_20"] = f["vol_5"]  / (f["vol_20"] + 1e-10)
    f["ts_ratio_5_60"] = f["vol_5"]  / (f["vol_60"] + 1e-10)
    f["ts_ratio_20_60"]= f["vol_20"] / (f["vol_60"] + 1e-10)

    # ── EMA and trend ──────────────────────────────────────────────────────────
    e9  = c.ewm(span=9,   adjust=False).mean()
    e21 = c.ewm(span=21,  adjust=False).mean()
    e55 = c.ewm(span=55,  adjust=False).mean()
    e200= c.ewm(span=200, adjust=False).mean()
    f["ema_9"]  = (e9   - c) / (c + 1e-10)
    f["ema_21"] = (e21  - c) / (c + 1e-10)
    f["ema_55"] = (e55  - c) / (c + 1e-10)
    f["ema_200"]= (e200 - c) / (c + 1e-10)
    f["ema_cross_9_21"]  = (e9  - e21) / (c + 1e-10)
    f["ema_cross_21_55"] = (e21 - e55) / (c + 1e-10)
    f["price_vs_ema200"] = (c  - e200) / (c + 1e-10)

    # ── Momentum ──────────────────────────────────────────────────────────────
    f["mom_10"] = (c - c.shift(10)) / (c + 1e-10)
    f["mom_20"] = (c - c.shift(20)) / (c + 1e-10)
    f["roc_10"] = c.pct_change(10)
    f["roc_20"] = c.pct_change(20)
    roll_high = c.rolling(min(504, len(c))).max()
    roll_low  = c.rolling(min(504, len(c))).min()
    f["high52w_pct"] = (c - roll_high) / (roll_high + 1e-10)
    f["low52w_pct"]  = (c - roll_low)  / (roll_low  + 1e-10)

    # ── RSI ───────────────────────────────────────────────────────────────────
    f["rsi_7"]  = _rsi(c, 7)
    f["rsi_14"] = _rsi(c, 14)
    f["rsi_21"] = _rsi(c, 21)

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    f["macd"]      = macd_line   / (c + 1e-10)
    f["macd_hist"] = (macd_line - macd_signal) / (c + 1e-10)

    # ── Bollinger Bands ────────────────────────────────────────────────────────
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    f["bb_upper"] = (bb_mid + 2 * bb_std - c) / (c + 1e-10)
    f["bb_lower"] = (c - (bb_mid - 2 * bb_std)) / (c + 1e-10)
    f["bb_pct"]   = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-10)
    f["bb_width"]  = (4 * bb_std) / (bb_mid + 1e-10)

    # ── ATR ───────────────────────────────────────────────────────────────────
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_7"]  = tr.rolling(7).mean()  / (c + 1e-10)
    f["atr_14"] = tr.rolling(14).mean() / (c + 1e-10)

    # ── Volume features ────────────────────────────────────────────────────────
    vol_ma20 = v.rolling(20).mean()
    vol_ma5  = v.rolling(5).mean()
    f["vol_ratio"]    = v / (vol_ma20 + 1e-10)
    f["vol_delta"]    = (v - v.shift(1)) / (vol_ma20 + 1e-10)
    f["vol_ratio_5_20"] = vol_ma5 / (vol_ma20 + 1e-10)
    taker_vol = df.get("taker_buy_base", pd.Series(np.nan, index=df.index)).astype(float)
    bsr = (taker_vol / (v + 1e-10)).replace([np.inf, -np.inf], np.nan)
    f["buy_sell_ratio"] = bsr.fillna(0.5)

    # ── Money Flow Index (Quong & Soudack 1989) ───────────────────────────────
    typical = (h + lo + c) / 3.0
    raw_mf  = typical * v
    pos_mf  = raw_mf.where(typical > typical.shift(1), 0.0)
    neg_mf  = raw_mf.where(typical < typical.shift(1), 0.0)
    pos_sum = pos_mf.rolling(14).sum()
    neg_sum = neg_mf.rolling(14).sum()
    f["mfi_14"] = 100.0 - 100.0 / (1.0 + pos_sum / (neg_sum + 1e-10))

    # ── Candle microstructure ─────────────────────────────────────────────────
    range_ = (h - lo + 1e-10)
    f["body"]      = (c - o).abs() / range_
    f["wick_up"]   = (h - pd.concat([c, o], axis=1).max(axis=1)) / range_
    f["wick_dn"]   = (pd.concat([c, o], axis=1).min(axis=1) - lo) / range_
    f["bar_range"] = range_ / (c + 1e-10)

    # ── Stochastic ────────────────────────────────────────────────────────────
    low14  = lo.rolling(14).min()
    high14 = h.rolling(14).max()
    f["stoch_k"] = (c - low14) / (high14 - low14 + 1e-10)
    f["stoch_d"] = f["stoch_k"].rolling(3).mean()

    # ── Donchian channels ─────────────────────────────────────────────────────
    f["donchian_high_20"] = (h.rolling(20).max() - c) / (c + 1e-10)
    f["donchian_low_20"]  = (c - lo.rolling(20).min()) / (c + 1e-10)

    # ── Autocorrelation — Cont (2001) return serial structure ─────────────────
    f["autocorr_1"] = lr.rolling(20).apply(lambda x: float(pd.Series(x).autocorr(lag=1)), raw=False).fillna(0)
    f["autocorr_5"] = lr.rolling(40).apply(lambda x: float(pd.Series(x).autocorr(lag=5)), raw=False).fillna(0)

    # ── Higher-order moments (20-bar window) ──────────────────────────────────
    f["skewness_20"] = lr.rolling(20).skew().fillna(0)
    f["kurtosis_20"] = lr.rolling(20).kurt().fillna(0)

    # ── Hurst exponent proxy (Peters 1994 R/S analysis approximation) ─────────
    # Uses variance-ratio shortcut: Var[k-period ret] / k*Var[1-period ret]
    # H > 0.5 = trending, H < 0.5 = mean-reverting
    var1  = lr.rolling(20).var()
    var5  = lr.rolling(5).apply(lambda x: float(pd.Series(x).sum()), raw=False)
    var5s = var5.rolling(16).var()
    f["hurst_proxy"] = (var5s / (5 * var1 + 1e-10)).clip(0, 2).fillna(1.0)

    # ── Amihud (2002) illiquidity: |ret| / volume (Kyle lambda proxy) ─────────
    abs_ret = lr.abs()
    amihud_daily = abs_ret / (v + 1e-10)
    f["amihud_5"]  = amihud_daily.rolling(5).mean()
    f["amihud_20"] = amihud_daily.rolling(20).mean()

    # ── Crypto-specific: funding rate and open interest ────────────────────────
    f["funding_rate"] = funding_rate
    f["oi_value"]     = open_interest

    f.replace([np.inf, -np.inf], np.nan, inplace=True)
    for name in FEATURE_NAMES:
        if name not in f.columns:
            f[name] = 0.0
    return f.loc[:, FEATURE_NAMES].dropna()


def compute_feature_matrix(df: pd.DataFrame, funding_rate: float = 0.0, open_interest: float = 0.0) -> np.ndarray:
    """Returns (T, F) float32 array with no NaN rows."""
    feat_df = compute_features(df, funding_rate=funding_rate, open_interest=open_interest)
    return feat_df.values.astype(np.float32)


def build_sequences(X: np.ndarray, y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert flat feature matrix into (N, lookback, F) sequences."""
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback: i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta   = series.diff()
    up      = delta.clip(lower=0)
    down    = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_dn = down.ewm(alpha=1 / period, adjust=False).mean()
    rs      = roll_up / (roll_dn + 1e-10)
    return 100 - 100 / (1 + rs)
