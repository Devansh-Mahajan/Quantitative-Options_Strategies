"""
Historical OHLCV data loader.
- Downloads from Binance public REST API (no API key needed for klines)
- Caches as Parquet files in .backtest_cache/ to avoid redundant downloads
- Supports arbitrary date ranges, multiple symbols and intervals
"""

from __future__ import annotations
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

from bot.config import cfg

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

YF_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "4h": "60m",
    "1d": "1d",
    "1w": "1wk",
}

YF_MAX_WINDOW_DAYS = {
    "1m": 7,
    "5m": 59,
    "15m": 59,
    "30m": 59,
    "1h": 729,
    "4h": 729,
}

ALPACA_STOCK_CHUNK_DAYS = {
    "1m": 5,
    "5m": 30,
    "15m": 60,
    "30m": 90,
    "1h": 180,
    "4h": 365,
    "1d": 3650,
    "1w": 3650,
}

ProgressCallback = Callable[[dict[str, Any]], None]


def _emit_progress(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(payload)
    except Exception as exc:
        log.debug("Progress callback failed: %s", exc)


def _to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _cache_path(symbol: str, interval: str, start_ms: int, end_ms: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{symbol}_{interval}_{start_ms}_{end_ms}.parquet"


def fetch_klines_raw(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    progress_callback: ProgressCallback | None = None,
) -> list[list]:
    """Download all klines for a date range, paginating automatically."""
    url = f"{BINANCE_BASE}/api/v3/klines"
    all_klines: list[list] = []
    current_start = start_ms
    bar_ms = INTERVAL_MS.get(interval, 3_600_000)
    total_span = max(end_ms - start_ms, bar_ms)

    _emit_progress(
        progress_callback,
        phase="download",
        source="binance",
        symbol=symbol,
        interval=interval,
        progress=0.0,
        message=f"Downloading {symbol} {interval} history from Binance",
    )

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
        covered_until = min(current_start, end_ms)
        progress = min(100.0, max(0.0, ((covered_until - start_ms) / total_span) * 100))
        _emit_progress(
            progress_callback,
            phase="download",
            source="binance",
            symbol=symbol,
            interval=interval,
            progress=round(progress, 1),
            message=f"Downloading {symbol} {interval} history ({len(all_klines)} bars)",
        )
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


def _resample_stock_df(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty or interval not in {"4h", "1d", "1w"}:
        return df
    rule_map = {"4h": "4h", "1d": "1d", "1w": "1W-MON"}
    if interval == "4h":
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "quote_volume": "sum",
            "taker_buy_base": "sum",
            "taker_buy_quote": "sum",
            "trades": "sum",
        }
        return df.resample(rule_map[interval]).agg(agg).dropna(subset=["open", "high", "low", "close"])
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "taker_buy_base": "sum",
        "taker_buy_quote": "sum",
        "trades": "sum",
    }
    return df.resample(rule_map[interval]).agg(agg).dropna(subset=["open", "high", "low", "close"])


def _download_stock_chunk(
    symbol: str,
    yf_interval: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        symbol,
        start=start,
        end=end,
        interval=yf_interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )


def _alpaca_stock_data_available() -> bool:
    return bool(
        cfg.alpaca_api_key
        and cfg.alpaca_api_secret
        and cfg.alpaca_api_key != "your_alpaca_key_here"
        and cfg.alpaca_api_secret != "your_alpaca_secret_here"
    )


def _alpaca_timeframe(interval: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    timeframe_map = {
        "1m": TimeFrame.Minute,
        "5m": TimeFrame(5, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "30m": TimeFrame(30, TimeFrameUnit.Minute),
        "1h": TimeFrame.Hour,
        "4h": TimeFrame(4, TimeFrameUnit.Hour),
        "1d": TimeFrame.Day,
    }
    if interval not in timeframe_map:
        raise ValueError(f"Unsupported Alpaca stock interval: {interval}")
    return timeframe_map[interval]


def _download_stock_history_alpaca(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest

    timeframe = _alpaca_timeframe(interval)
    chunk_days = ALPACA_STOCK_CHUNK_DAYS.get(interval, 180)
    client = StockHistoricalDataClient(
        api_key=cfg.alpaca_api_key,
        secret_key=cfg.alpaca_api_secret,
    )

    parts: list[pd.DataFrame] = []
    chunk_start = start
    total_seconds = max((end - start).total_seconds(), 1.0)
    total_chunks = max(1, int(np.ceil(total_seconds / timedelta(days=chunk_days).total_seconds())))
    chunk_idx = 0

    while chunk_start < end:
        chunk_idx += 1
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
        progress = ((chunk_start - start).total_seconds() / total_seconds) * 100
        _emit_progress(
            progress_callback,
            phase="download",
            source="alpaca",
            symbol=symbol,
            interval=interval,
            progress=round(progress, 1),
            message=(
                f"Downloading {symbol} {interval} chunk {chunk_idx}/{total_chunks} "
                f"from Alpaca ({chunk_start.date()} -> {chunk_end.date()})"
            ),
        )
        response = client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=chunk_start,
                end=chunk_end,
                limit=10_000,
            )
        )
        frame = getattr(response, "df", None)
        if frame is None or frame.empty:
            chunk_start = chunk_end
            continue
        if isinstance(frame.index, pd.MultiIndex):
            symbols = frame.index.get_level_values(0)
            if symbol in symbols:
                frame = frame.xs(symbol, level=0)
            else:
                frame = frame.droplevel(0)
        parts.append(frame)
        chunk_start = chunk_end

    if not parts:
        return pd.DataFrame()

    raw = pd.concat(parts).sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    raw = raw.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    _emit_progress(
        progress_callback,
        phase="download",
        source="alpaca",
        symbol=symbol,
        interval=interval,
        progress=100.0,
        message=f"Downloaded {symbol} {interval} history from Alpaca",
    )
    return _normalise_stock_history(raw, interval, start, end)


def _normalise_stock_history(
    raw: pd.DataFrame,
    interval: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    keep_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df.loc[:, keep_cols].copy()
    if len(keep_cols) < 5:
        return pd.DataFrame()
    for col in keep_cols:
        df[col] = df[col].astype(float)

    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df.index = idx
    df.index.name = "open_time"
    df = df.sort_index()

    df["quote_volume"] = df["volume"].astype(float) * df["close"].astype(float)
    df["trades"] = 0
    df["taker_buy_base"] = df["volume"].astype(float) * 0.5
    df["taker_buy_quote"] = df["quote_volume"].astype(float) * 0.5

    if interval in {"4h", "1d", "1w"}:
        df = _resample_stock_df(df, interval)

    return df[(df.index >= start) & (df.index <= end)]


def _load_stock_history(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    yf_interval = YF_INTERVAL_MAP.get(interval, "60m")
    log.info("Downloading stock %s %s  %s → %s …", symbol, interval, start.date(), end.date())
    if _alpaca_stock_data_available():
        try:
            df = _download_stock_history_alpaca(
                symbol,
                interval,
                start,
                end,
                progress_callback=progress_callback,
            )
            if not df.empty:
                return df
            log.warning("Alpaca returned no stock data for %s %s; falling back to Yahoo", symbol, interval)
        except Exception as exc:
            log.warning("Alpaca stock history failed for %s %s: %s", symbol, interval, exc)

    max_window_days = YF_MAX_WINDOW_DAYS.get(interval)
    total_chunks = 1

    if max_window_days is not None:
        span_seconds = max((end - start).total_seconds(), 1.0)
        total_chunks = max(1, int(np.ceil(span_seconds / timedelta(days=max_window_days).total_seconds())))

    if max_window_days is not None and end - start > timedelta(days=max_window_days):
        parts: list[pd.DataFrame] = []
        chunk_start = start
        chunk_idx = 0
        while chunk_start < end:
            chunk_idx += 1
            chunk_end = min(chunk_start + timedelta(days=max_window_days), end)
            progress = ((chunk_start - start).total_seconds() / max((end - start).total_seconds(), 1.0)) * 100
            _emit_progress(
                progress_callback,
                phase="download",
                source="yfinance",
                symbol=symbol,
                interval=interval,
                progress=round(progress, 1),
                message=(
                    f"Downloading {symbol} {interval} chunk {chunk_idx}/{total_chunks} "
                    f"({chunk_start.date()} -> {chunk_end.date()})"
                ),
            )
            raw_part = _download_stock_chunk(symbol, yf_interval, chunk_start, chunk_end)
            if raw_part is None or raw_part.empty:
                log.warning(
                    "No stock data returned for %s %s chunk %s/%s",
                    symbol,
                    interval,
                    chunk_idx,
                    total_chunks,
                )
            else:
                parts.append(raw_part)
            chunk_start = chunk_end

        if not parts:
            log.warning("No stock data returned for %s %s", symbol, interval)
            return pd.DataFrame()

        raw = pd.concat(parts).sort_index()
        raw = raw[~raw.index.duplicated(keep="last")]
    else:
        _emit_progress(
            progress_callback,
            phase="download",
            source="yfinance",
            symbol=symbol,
            interval=interval,
            progress=0.0,
            message=f"Downloading {symbol} {interval} history from Yahoo Finance",
        )
        raw = _download_stock_chunk(symbol, yf_interval, start, end)
        if raw is None or raw.empty:
            log.warning("No stock data returned for %s %s", symbol, interval)
            return pd.DataFrame()

    _emit_progress(
        progress_callback,
        phase="download",
        source="yfinance",
        symbol=symbol,
        interval=interval,
        progress=100.0,
        message=f"Downloaded {symbol} {interval} history",
    )
    return _normalise_stock_history(raw, interval, start, end)


def load(
    symbol: str,
    interval: str,
    start: str | datetime,
    end: str | datetime | None = None,
    force_download: bool = False,
    progress_callback: ProgressCallback | None = None,
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
        _emit_progress(
            progress_callback,
            phase="cache",
            source="cache",
            symbol=symbol,
            interval=interval,
            progress=100.0,
            message=f"Loaded {symbol} {interval} history from cache",
        )
        return pd.read_parquet(cache)

    if cfg.is_stock_symbol(symbol):
        df = _load_stock_history(symbol, interval, start, end, progress_callback=progress_callback)
    else:
        log.info("Downloading %s %s  %s → %s …", symbol, interval, start.date(), end.date())
        raw = fetch_klines_raw(
            symbol,
            interval,
            start_ms,
            end_ms,
            progress_callback=progress_callback,
        )
        if not raw:
            log.warning("No data returned for %s %s", symbol, interval)
            return pd.DataFrame()
        df = _klines_to_df(raw)
        df = df[(df.index >= start) & (df.index <= end)]

    if df.empty:
        return pd.DataFrame()

    df.to_parquet(cache)
    log.info("  → %d bars saved to %s", len(df), cache.name)
    _emit_progress(
        progress_callback,
        phase="complete",
        source="cache",
        symbol=symbol,
        interval=interval,
        progress=100.0,
        message=f"Prepared {symbol} {interval} history ({len(df)} bars)",
    )
    return df


def load_multi(
    symbols: list[str],
    interval: str,
    start: str | datetime,
    end: str | datetime | None = None,
    force_download: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, pd.DataFrame]:
    """Load multiple symbols. Returns {symbol: DataFrame}."""
    if not symbols:
        return {}

    total = len(symbols)
    loaded: dict[str, pd.DataFrame] = {}

    for idx, sym in enumerate(symbols):
        base_progress = (idx / total) * 100.0
        span = 100.0 / total

        def _symbol_progress(event: dict[str, Any], *, _idx: int = idx) -> None:
            local_progress = float(event.get("progress", 0.0))
            overall_progress = min(100.0, (_idx * span) + (local_progress / 100.0) * span)
            enriched = dict(event)
            enriched["symbol_index"] = _idx
            enriched["symbol_count"] = total
            enriched["overall_progress"] = round(overall_progress, 1)
            _emit_progress(progress_callback, **enriched)

        _emit_progress(
            progress_callback,
            phase="prepare",
            symbol=sym,
            interval=interval,
            progress=round(base_progress, 1),
            overall_progress=round(base_progress, 1),
            message=f"Preparing {sym} {interval} history",
        )
        loaded[sym] = load(
            sym,
            interval,
            start,
            end,
            force_download=force_download,
            progress_callback=_symbol_progress if progress_callback is not None else None,
        )

    _emit_progress(
        progress_callback,
        phase="complete",
        progress=100.0,
        overall_progress=100.0,
        interval=interval,
        message=f"Loaded {len(symbols)} symbol(s) for {interval}",
    )
    return loaded


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
