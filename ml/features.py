"""Feature engineering for all ML models: technical indicators + crypto-specific signals."""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd

log = logging.getLogger("ml.features")


def compute_features(df: pd.DataFrame, funding_rate: float = 0.0, open_interest: float = 0.0) -> pd.DataFrame:
    """
    Input df must have columns: open, high, low, close, volume.
    Returns DataFrame with all features, NaN rows dropped.
    """
    f = pd.DataFrame(index=df.index)

    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)
    o = df["open"].astype(float)

    # --- Returns ---
    f["ret_1"] = c.pct_change(1)
    f["ret_5"] = c.pct_change(5)
    f["ret_20"] = c.pct_change(20)
    f["log_ret_1"] = np.log(c / c.shift(1))
    f["log_ret_5"] = np.log(c / c.shift(5))

    # --- Realized volatility (rolling std of log returns) ---
    lr = f["log_ret_1"]
    f["vol_5"] = lr.rolling(5).std() * np.sqrt(252 * 24)   # annualised (hourly)
    f["vol_20"] = lr.rolling(20).std() * np.sqrt(252 * 24)
    f["vol_60"] = lr.rolling(60).std() * np.sqrt(252 * 24)

    # --- EMA ---
    f["ema_9"] = c.ewm(span=9, adjust=False).mean()
    f["ema_21"] = c.ewm(span=21, adjust=False).mean()
    f["ema_55"] = c.ewm(span=55, adjust=False).mean()
    f["ema_200"] = c.ewm(span=200, adjust=False).mean()
    f["ema_cross_9_21"] = (f["ema_9"] - f["ema_21"]) / c
    f["ema_cross_21_55"] = (f["ema_21"] - f["ema_55"]) / c
    f["price_vs_ema200"] = (c - f["ema_200"]) / c

    # --- RSI ---
    f["rsi_14"] = _rsi(c, 14)
    f["rsi_7"] = _rsi(c, 7)

    # --- MACD ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    f["macd"] = macd_line / c
    f["macd_hist"] = (macd_line - macd_signal) / c

    # --- Bollinger Bands ---
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    f["bb_upper"] = (bb_mid + 2 * bb_std - c) / c
    f["bb_lower"] = (c - (bb_mid - 2 * bb_std)) / c
    f["bb_pct"] = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-10)
    f["bb_width"] = (4 * bb_std) / (bb_mid + 1e-10)

    # --- ATR ---
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.rolling(14).mean() / c
    f["atr_7"] = tr.rolling(7).mean() / c

    # --- Volume ---
    vol_ma = v.rolling(20).mean()
    f["vol_ratio"] = v / (vol_ma + 1e-10)
    f["vol_delta"] = (v - v.shift(1)) / (vol_ma + 1e-10)
    taker_vol = df.get("taker_buy_base", pd.Series(np.nan, index=df.index)).astype(float)
    if not taker_vol.isna().all():
        f["buy_sell_ratio"] = taker_vol / (v + 1e-10)

    # --- Candle body features ---
    f["body"] = (c - o).abs() / (h - l + 1e-10)
    f["wick_up"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / (h - l + 1e-10)
    f["wick_dn"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / (h - l + 1e-10)

    # --- Momentum oscillator ---
    f["mom_10"] = (c - c.shift(10)) / c
    f["mom_20"] = (c - c.shift(20)) / c

    # --- Stochastic %K ---
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    f["stoch_k"] = (c - low14) / (high14 - low14 + 1e-10)
    f["stoch_d"] = f["stoch_k"].rolling(3).mean()

    # --- Donchian channels ---
    f["donchian_high_20"] = (h.rolling(20).max() - c) / c
    f["donchian_low_20"] = (c - l.rolling(20).min()) / c

    # --- Crypto-specific ---
    f["funding_rate"] = funding_rate
    f["oi_value"] = open_interest

    # --- Regime proxy: vol-of-vol ---
    f["vol_of_vol"] = f["vol_20"].rolling(10).std()

    f.replace([np.inf, -np.inf], np.nan, inplace=True)
    return f.dropna()


def compute_feature_matrix(df: pd.DataFrame, funding_rate: float = 0.0, open_interest: float = 0.0) -> np.ndarray:
    """Returns (T, F) float32 array with no NaN rows."""
    feat_df = compute_features(df, funding_rate=funding_rate, open_interest=open_interest)
    return feat_df.values.astype(np.float32)


def build_sequences(X: np.ndarray, y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert flat feature matrix into (N, lookback, F) sequences.
    y should be aligned with X (same length).
    """
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback: i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-10)
    return 100 - 100 / (1 + rs)


FEATURE_NAMES = [
    "ret_1", "ret_5", "ret_20", "log_ret_1", "log_ret_5",
    "vol_5", "vol_20", "vol_60",
    "ema_9", "ema_21", "ema_55", "ema_200",
    "ema_cross_9_21", "ema_cross_21_55", "price_vs_ema200",
    "rsi_14", "rsi_7",
    "macd", "macd_hist",
    "bb_upper", "bb_lower", "bb_pct", "bb_width",
    "atr_14", "atr_7",
    "vol_ratio", "vol_delta",
    "body", "wick_up", "wick_dn",
    "mom_10", "mom_20",
    "stoch_k", "stoch_d",
    "donchian_high_20", "donchian_low_20",
    "funding_rate", "oi_value",
    "vol_of_vol",
]
