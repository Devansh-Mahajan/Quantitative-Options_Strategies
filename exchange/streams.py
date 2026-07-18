"""WebSocket stream manager: klines, orderbook, liquidations, user data."""

from __future__ import annotations
import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

from binance import AsyncClient, BinanceSocketManager

from bot.config import cfg

log = logging.getLogger("exchange.streams")


# --------------------------------------------------------------------------- #
# Market data store — shared memory between streams and strategies
# --------------------------------------------------------------------------- #

@dataclass
class Candle:
    symbol: str
    interval: str
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = False


@dataclass
class BookTicker:
    symbol: str
    bid: float
    ask: float
    ts: float = field(default_factory=time.monotonic)


@dataclass
class Liquidation:
    symbol: str
    side: str       # BUY = long liquidated, SELL = short liquidated
    quantity: float
    price: float
    ts: float = field(default_factory=time.monotonic)


HISTORY_MAX_BARS = 2400   # ≈100 days of 1h bars


class MarketDataStore:
    """Thread-safe (GIL-protected) in-process market data bus."""

    def __init__(self) -> None:
        # latest closed candle per (symbol, interval)
        self.candles: dict[tuple[str, str], Candle] = {}
        # rolling candle history per (symbol, interval)
        # 2400 x 1h ≈ 100 days — the daily/weekly strategies (trend_follow_daily,
        # weekly_momentum_rotation, dip_buyer) need 30-90 days of history; the
        # old 500-bar cap (~21 days) silently starved them into permanent
        # no-signal mode in both live trading and backtests.
        self.history: dict[tuple[str, str], deque[Candle]] = defaultdict(lambda: deque(maxlen=HISTORY_MAX_BARS))
        # best bid/ask
        self.book: dict[str, BookTicker] = {}
        # recent liquidations per symbol
        self.liquidations: dict[str, deque[Liquidation]] = defaultdict(lambda: deque(maxlen=50))
        # funding rates
        self.funding: dict[str, float] = {}
        # open interest
        self.open_interest: dict[str, float] = {}
        # callbacks registered by strategies
        self._on_candle_close: list[Callable[[Candle], None]] = []
        # DataFrame cache: rebuilt only when new candle pushed for that key
        self._df_cache: dict[tuple[str, str], "pd.DataFrame"] = {}
        self._df_dirty: dict[tuple[str, str], bool] = defaultdict(lambda: True)

    def on_candle_close(self, callback: Callable[[Candle], None]) -> None:
        self._on_candle_close.append(callback)

    def push_candle(self, candle: Candle) -> None:
        key = (candle.symbol, candle.interval)
        self.candles[key] = candle
        self.history[key].append(candle)
        self._df_dirty[key] = True
        if candle.closed:
            for cb in self._on_candle_close:
                try:
                    cb(candle)
                except Exception as exc:
                    log.warning("on_candle_close callback error: %s", exc)

    def get_closes(self, symbol: str, interval: str, n: int) -> list[float]:
        key = (symbol, interval)
        hist = list(self.history[key])
        return [c.close for c in hist[-n:]]

    def get_volumes(self, symbol: str, interval: str, n: int) -> list[float]:
        key = (symbol, interval)
        hist = list(self.history[key])
        return [c.volume for c in hist[-n:]]

    def get_history_df(self, symbol: str, interval: str):
        import pandas as pd
        key = (symbol, interval)
        if not self._df_dirty[key] and key in self._df_cache:
            return self._df_cache[key]
        records = [
            {
                "open_time": c.open_time, "open": c.open, "high": c.high,
                "low": c.low, "close": c.close, "volume": c.volume,
            }
            for c in self.history[key]
        ]
        df = pd.DataFrame(records) if records else pd.DataFrame()
        self._df_cache[key] = df
        self._df_dirty[key] = False
        return df


# --------------------------------------------------------------------------- #
# StreamManager
# --------------------------------------------------------------------------- #

class StreamManager:
    """Manages all Binance WebSocket subscriptions."""

    def __init__(self, client: AsyncClient, store: MarketDataStore) -> None:
        self._client = client
        self.store = store
        self._bsm: BinanceSocketManager | None = None
        self._tasks: list[asyncio.Task] = []

    async def start(self, symbols: list[str], intervals: list[str] = None) -> None:
        intervals = intervals or ["1m", "5m", "15m", "1h", "4h"]
        self._bsm = BinanceSocketManager(self._client)

        for symbol in symbols:
            for interval in intervals:
                t = asyncio.create_task(
                    self._kline_stream(symbol.lower(), interval),
                    name=f"kline_{symbol}_{interval}",
                )
                self._tasks.append(t)

            t = asyncio.create_task(
                self._book_ticker_stream(symbol.lower()), name=f"book_{symbol}"
            )
            self._tasks.append(t)

        # All-symbol liquidation stream
        t = asyncio.create_task(self._liquidation_stream(), name="liquidations")
        self._tasks.append(t)

        log.info(
            "StreamManager started: %d symbols × %d intervals + book + liquidations",
            len(symbols), len(intervals),
        )

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        log.info("StreamManager stopped")

    # ------------------------------------------------------------------ #

    async def _kline_stream(self, symbol: str, interval: str) -> None:
        while True:
            try:
                async with self._bsm.futures_kline_socket(symbol=symbol, interval=interval) as stream:
                    async for msg in stream:
                        k = msg.get("k", {})
                        candle = Candle(
                            symbol=k.get("s", symbol.upper()),
                            interval=interval,
                            open_time=int(k["t"]),
                            open=float(k["o"]),
                            high=float(k["h"]),
                            low=float(k["l"]),
                            close=float(k["c"]),
                            volume=float(k["v"]),
                            closed=bool(k["x"]),
                        )
                        self.store.push_candle(candle)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("Kline stream %s/%s error: %s — reconnecting in 5s", symbol, interval, exc)
                await asyncio.sleep(5)

    async def _book_ticker_stream(self, symbol: str) -> None:
        while True:
            try:
                async with self._bsm.futures_symbol_book_ticker_socket(symbol=symbol) as stream:
                    async for msg in stream:
                        self.store.book[msg["s"]] = BookTicker(
                            symbol=msg["s"],
                            bid=float(msg["b"]),
                            ask=float(msg["a"]),
                        )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("Book ticker %s error: %s — reconnecting in 5s", symbol, exc)
                await asyncio.sleep(5)

    async def _liquidation_stream(self) -> None:
        while True:
            try:
                async with self._bsm.futures_socket() as stream:
                    async for msg in stream:
                        if msg.get("e") == "forceOrder":
                            o = msg["o"]
                            liq = Liquidation(
                                symbol=o["s"],
                                side=o["S"],          # BUY/SELL
                                quantity=float(o["q"]),
                                price=float(o["p"]),
                            )
                            self.store.liquidations[liq.symbol].append(liq)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("Liquidation stream error: %s — reconnecting in 5s", exc)
                await asyncio.sleep(5)
