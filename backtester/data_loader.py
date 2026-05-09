"""
Historical OHLCV data loader.
- Downloads from Binance public REST API (no API key needed for klines)
- Caches as Parquet files in .backtest_cache/ to avoid redundant downloads
- Supports arbitrary date ranges, multiple symbols and intervals
"""

from __future__ import annotations
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("backtester.data_loader")

_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = _ROOT / ".backtest_cache"
BINANCE_BASE = "https://api.binance.com"
MAX_KLINES_PER_REQUEST = 1000   # Binance hard limit
RATE_LIMIT_PAUSE = 0.12         # seconds between requests (~8 req/s, well under limit)

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "12h": 43_200_000,
    "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000,
}


def _to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _cache_path(symbol: str, interval: str, start_ms: int, end_ms: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{symbol}_{interval}_{start_ms}_{end_ms}.parquet"


def fetch_klines_raw(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    """Download all klines for a date range, paginating automatically."""
    url = f"{BINANCE_BASE}/api/v3/klines"
    all_klines: list[list] = []
    current_start = start_ms
    bar_ms = INTERVAL_MS.get(interval, 3_600_000)

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": MAX_KLINES_PER_REQUEST,
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as exc:
            log.error("Binance kline fetch error: %s", exc)
            break

        if not batch:
            break

        all_klines.extend(batch)
        last_ts = int(batch[-1][0])
        current_start = last_ts + bar_ms
        log.debug("  fetched %d bars up to %s", len(batch), _from_ms(last_ts).date())
        time.sleep(RATE_LIMIT_PAUSE)

    return all_klines


def _klines_to_df(raw: list[list]) -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = df[col].astype(float)
    df["trades"] = df["trades"].astype(int)
    df = df.drop(columns=["close_time", "ignore"])
    df = df.set_index("open_time").sort_index()
    return df


def load(
    symbol: str,
    interval: str,
    start: str | datetime,
    end: str | datetime | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load OHLCV data for a symbol/interval/date range.
    Returns a DataFrame indexed by open_time (UTC).

    Args:
        symbol:  Binance symbol, e.g. 'BTCUSDT'
        interval: '1m', '5m', '15m', '1h', '4h', '1d', …
        start:   '2023-01-01' or datetime
        end:     '2024-01-01' or datetime (defaults to now)
        force_download: bypass cache
    """
    if isinstance(start, str):
        start = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    if end is None:
        end = datetime.now(timezone.utc)
    elif isinstance(end, str):
        end = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

    start_ms = _to_ms(start)
    end_ms = _to_ms(end)
    cache = _cache_path(symbol, interval, start_ms, end_ms)

    if cache.exists() and not force_download:
        log.debug("Loading %s %s from cache", symbol, interval)
        return pd.read_parquet(cache)

    log.info("Downloading %s %s  %s → %s …", symbol, interval, start.date(), end.date())
    raw = fetch_klines_raw(symbol, interval, start_ms, end_ms)
    if not raw:
        log.warning("No data returned for %s %s", symbol, interval)
        return pd.DataFrame()

    df = _klines_to_df(raw)
    df = df[(df.index >= start) & (df.index <= end)]
    df.to_parquet(cache)
    log.info("  → %d bars saved to %s", len(df), cache.name)
    return df


def load_multi(
    symbols: list[str],
    interval: str,
    start: str | datetime,
    end: str | datetime | None = None,
    force_download: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load multiple symbols. Returns {symbol: DataFrame}."""
    return {
        sym: load(sym, interval, start, end, force_download=force_download)
        for sym in symbols
    }


def align_and_ffill(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Align all DataFrames to a common index and forward-fill gaps.
    Essential for pairs-arb and correlation calculations.
    """
    if not data:
        return data
    idx = data[next(iter(data))].index
    for sym, df in data.items():
        idx = idx.intersection(df.index)
    return {sym: df.reindex(idx).ffill() for sym, df in data.items()}
