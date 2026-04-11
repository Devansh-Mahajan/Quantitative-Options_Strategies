import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(f"strategy.{__name__}")


def _download_close_and_volume(symbols: List[str], period: str = "6mo") -> tuple[pd.DataFrame, pd.DataFrame]:
    data = yf.download(symbols, period=period, progress=False, auto_adjust=False)
    close = data.get('Close', pd.DataFrame())
    volume = data.get('Volume', pd.DataFrame())

    if isinstance(close, pd.Series):
        close = close.to_frame(name=symbols[0])
    if isinstance(volume, pd.Series):
        volume = volume.to_frame(name=symbols[0])

    return close.dropna(how='all'), volume.dropna(how='all')


def rank_symbols_by_volatility(symbols: List[str], lookback: int = 30, top_n: int = 50) -> List[str]:
    """Ranks symbols by annualized realized volatility over the lookback window."""
    if not symbols:
        return []

    close, _ = _download_close_and_volume(symbols)
    returns = np.log(close / close.shift(1))
    realized_vol = returns.tail(lookback).std() * np.sqrt(252)
    ranked = realized_vol.dropna().sort_values(ascending=False)

    return ranked.head(top_n).index.tolist()


def estimate_institutional_flow(symbols: List[str], lookback: int = 40) -> Dict[str, float]:
    """
    Proxy for institutional/hedge-fund style flow:
    momentum + volume surprise + trend persistence.
    """
    if not symbols:
        return {}

    close, volume = _download_close_and_volume(symbols)
    if close.empty:
        return {}

    log_ret = np.log(close / close.shift(1))
    mom_20 = close.pct_change(20)

    vol_ma = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0, np.nan)
    vol_z = (volume - vol_ma) / vol_std

    trend_consistency = log_ret.rolling(lookback).apply(lambda x: np.mean(np.sign(x) == np.sign(np.nanmean(x))), raw=True)

    latest = (
        mom_20.iloc[-1].fillna(0) * 0.5
        + vol_z.iloc[-1].fillna(0) * 0.3
        + trend_consistency.iloc[-1].fillna(0) * 0.2
    )

    ranked = latest.sort_values(ascending=False)
    return ranked.to_dict()


def prioritize_symbols(symbols: List[str], top_n: int = 50) -> List[str]:
    """Keeps the highest-volatility symbols and reorders by flow proxy."""
    if not symbols:
        return []

    vol_top = rank_symbols_by_volatility(symbols, top_n=min(top_n, len(symbols)))
    flow_map = estimate_institutional_flow(vol_top)

    ordered = sorted(vol_top, key=lambda s: flow_map.get(s, 0.0), reverse=True)
    logger.info("📡 Intelligence prioritization complete: %d symbols retained.", len(ordered))
    return ordered
