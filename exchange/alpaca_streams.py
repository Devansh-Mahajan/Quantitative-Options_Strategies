"""
Alpaca WebSocket stream adapter.
Subscribes to crypto bars + quotes via CryptoDataStream and feeds the same
MarketDataStore that Binance streams use — strategies see identical data.
"""

from __future__ import annotations
import asyncio
import logging
import time

from exchange.streams import MarketDataStore, Candle, BookTicker
from bot.config import cfg

log = logging.getLogger("exchange.alpaca_streams")

# Interval string → approximate seconds (used to derive open_time)
_INTERVAL_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400,
}


def _to_alpaca(symbol: str) -> str:
    sym = symbol.upper()
    for suffix in ("USDT", "BUSD", "USD"):
        if sym.endswith(suffix):
            return f"{sym[:-len(suffix)]}/USD"
    return f"{sym}/USD"


def _from_alpaca(alpaca_sym: str) -> str:
    """'BTC/USD' → 'BTCUSDT'."""
    base = alpaca_sym.split("/")[0]
    return base + "USDT"


class AlpacaStreamManager:
    """Wraps Alpaca CryptoDataStream; exposes same start/stop interface as StreamManager."""

    def __init__(self, client, store: MarketDataStore) -> None:
        self._client = client
        self.store = store
        self._stream = None
        self._tasks: list[asyncio.Task] = []
        self._alpaca_symbols: list[str] = []

    async def start(self, symbols: list[str], intervals: list[str] = None) -> None:
        intervals = intervals or ["1m", "5m", "15m", "1h", "4h"]
        self._alpaca_symbols = [_to_alpaca(s) for s in symbols]
        self._intervals = intervals

        try:
            from alpaca.data.live import CryptoDataStream
        except ImportError:
            log.warning("alpaca-py not installed — falling back to polling mode")
            t = asyncio.create_task(self._polling_fallback(symbols, intervals))
            self._tasks.append(t)
            return

        self._stream = CryptoDataStream(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_api_secret,
        )

        # Subscribe bars for the primary interval (1m) — we derive others via accumulation
        self._stream.subscribe_bars(self._on_bar, *self._alpaca_symbols)
        self._stream.subscribe_quotes(self._on_quote, *self._alpaca_symbols)

        # Seed historical data via REST before live stream starts
        t_seed = asyncio.create_task(self._seed_history(symbols, intervals))
        self._tasks.append(t_seed)

        # Run WebSocket in a background task
        t_ws = asyncio.create_task(self._run_stream(), name="alpaca_ws")
        self._tasks.append(t_ws)

        log.info("AlpacaStreamManager started — %d symbols, seeding history...", len(symbols))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._stream:
            try:
                await asyncio.to_thread(self._stream.stop)
            except Exception:
                pass
        log.info("AlpacaStreamManager stopped")

    # ------------------------------------------------------------------ #
    # WebSocket callbacks
    # ------------------------------------------------------------------ #

    async def _on_bar(self, bar) -> None:
        """Called by Alpaca for every completed bar (1m by default)."""
        binance_sym = _from_alpaca(bar.symbol)
        ts = int(bar.timestamp.timestamp() * 1000)
        candle = Candle(
            symbol=binance_sym,
            interval="1m",
            open_time=ts,
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
            closed=True,
        )
        self.store.push_candle(candle)

    async def _on_quote(self, quote) -> None:
        """Best bid/ask — mirrors Binance book ticker stream."""
        binance_sym = _from_alpaca(quote.symbol)
        self.store.book[binance_sym] = BookTicker(
            symbol=binance_sym,
            bid=float(quote.bid_price),
            ask=float(quote.ask_price),
        )

    async def _run_stream(self) -> None:
        while True:
            try:
                await self._stream._run_forever()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("Alpaca WS error: %s — reconnecting in 10s", exc)
                await asyncio.sleep(10)

    # ------------------------------------------------------------------ #
    # Historical seeding
    # ------------------------------------------------------------------ #

    async def _seed_history(self, symbols: list[str], intervals: list[str]) -> None:
        """Load historical bars so strategies have data before the first live bar."""
        for symbol in symbols:
            for interval in intervals:
                try:
                    klines = await self._client.get_klines(symbol, interval=interval, limit=300)
                    for k in klines:
                        candle = Candle(
                            symbol=symbol,
                            interval=interval,
                            open_time=int(k["open_time"]),
                            open=float(k["open"]),
                            high=float(k["high"]),
                            low=float(k["low"]),
                            close=float(k["close"]),
                            volume=float(k["volume"]),
                            closed=True,
                        )
                        self.store.push_candle(candle)
                    log.debug("Seeded %d bars for %s/%s", len(klines), symbol, interval)
                except Exception as exc:
                    log.warning("Failed to seed %s/%s: %s", symbol, interval, exc)
        log.info("History seed complete for %d symbols", len(symbols))

    # ------------------------------------------------------------------ #
    # Polling fallback (if alpaca-py WebSocket fails or isn't available)
    # ------------------------------------------------------------------ #

    async def _polling_fallback(self, symbols: list[str], intervals: list[str]) -> None:
        """Poll REST API every 60s — slower but always works."""
        log.info("Alpaca polling fallback mode (REST every 60s)")
        while True:
            for symbol in symbols:
                for interval in intervals:
                    try:
                        klines = await self._client.get_klines(symbol, interval=interval, limit=10)
                        for k in klines:
                            candle = Candle(
                                symbol=symbol, interval=interval,
                                open_time=int(k["open_time"]),
                                open=float(k["open"]), high=float(k["high"]),
                                low=float(k["low"]), close=float(k["close"]),
                                volume=float(k["volume"]), closed=True,
                            )
                            self.store.push_candle(candle)
                    except Exception:
                        pass
            await asyncio.sleep(60)
