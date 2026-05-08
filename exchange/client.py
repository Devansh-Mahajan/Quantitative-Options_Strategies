"""Async Binance client: spot, USDM futures, COINM futures, and European options (EAPI)."""

from __future__ import annotations
import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable

from binance import AsyncClient
from binance.exceptions import BinanceAPIException

from bot.config import cfg

log = logging.getLogger("exchange.client")

# --------------------------------------------------------------------------- #
# Retry decorator
# --------------------------------------------------------------------------- #

def _retry(max_attempts: int = 3, base_delay: float = 1.0):
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except BinanceAPIException as exc:
                    if exc.code in (-1003, -1015):  # rate limit
                        wait = base_delay * (2 ** attempt)
                        log.warning("Rate limit hit, sleeping %.1fs (attempt %d)", wait, attempt)
                        await asyncio.sleep(wait)
                    elif attempt == max_attempts:
                        raise
                    else:
                        log.warning("BinanceAPIException %s — retrying (%d/%d)", exc, attempt, max_attempts)
                        await asyncio.sleep(base_delay * attempt)
                except Exception as exc:
                    if attempt == max_attempts:
                        raise
                    log.warning("Exchange error %s — retrying (%d/%d)", exc, attempt, max_attempts)
                    await asyncio.sleep(base_delay * attempt)
        return wrapper
    return decorator


# --------------------------------------------------------------------------- #
# BinanceClient
# --------------------------------------------------------------------------- #

class BinanceClient:
    """Thin async wrapper. Call `await client.start()` before use."""

    def __init__(self) -> None:
        self._client: AsyncClient | None = None
        self._price_cache: dict[str, tuple[float, float]] = {}  # symbol -> (price, ts)
        self._cache_ttl = 2.0  # seconds

    async def start(self) -> None:
        self._client = await AsyncClient.create(
            api_key=cfg.api_key,
            api_secret=cfg.api_secret,
            testnet=cfg.testnet,
        )
        log.info("Binance AsyncClient connected (testnet=%s)", cfg.testnet)

    async def close(self) -> None:
        if self._client:
            await self._client.close_connection()
            log.info("Binance client closed")

    @property
    def _c(self) -> AsyncClient:
        if self._client is None:
            raise RuntimeError("BinanceClient not started — call await client.start() first")
        return self._client

    # ------------------------------------------------------------------ #
    # Account
    # ------------------------------------------------------------------ #

    @_retry()
    async def get_spot_balance(self) -> dict[str, float]:
        """Return {asset: free_balance} for spot account."""
        account = await self._c.get_account()
        return {b["asset"]: float(b["free"]) for b in account["balances"] if float(b["free"]) > 0}

    @_retry()
    async def get_futures_balance(self) -> dict[str, float]:
        """USDM futures wallet balance."""
        balances = await self._c.futures_account_balance()
        return {b["asset"]: float(b["availableBalance"]) for b in balances}

    @_retry()
    async def get_futures_positions(self) -> list[dict]:
        """All open USDM futures positions (non-zero)."""
        info = await self._c.futures_position_information()
        return [p for p in info if float(p["positionAmt"]) != 0]

    @_retry()
    async def get_spot_positions(self) -> dict[str, float]:
        """Non-zero spot holdings."""
        account = await self._c.get_account()
        return {b["asset"]: float(b["free"]) + float(b["locked"])
                for b in account["balances"] if float(b["free"]) + float(b["locked"]) > 0}

    # ------------------------------------------------------------------ #
    # Market data
    # ------------------------------------------------------------------ #

    @_retry()
    async def get_price(self, symbol: str) -> float:
        now = time.monotonic()
        if symbol in self._price_cache:
            price, ts = self._price_cache[symbol]
            if now - ts < self._cache_ttl:
                return price
        ticker = await self._c.get_symbol_ticker(symbol=symbol)
        price = float(ticker["price"])
        self._price_cache[symbol] = (price, now)
        return price

    @_retry()
    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 200,
        futures: bool = True,
    ) -> list[dict]:
        """Return OHLCV list as dicts."""
        if futures:
            raw = await self._c.futures_klines(symbol=symbol, interval=interval, limit=limit)
        else:
            raw = await self._c.get_klines(symbol=symbol, interval=interval, limit=limit)
        keys = ["open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore"]
        return [dict(zip(keys, [float(v) if i > 0 else int(v) for i, v in enumerate(row)]))
                for row in raw]

    @_retry()
    async def get_funding_rate(self, symbol: str) -> float:
        """Current funding rate for a USDM perpetual."""
        data = await self._c.futures_funding_rate(symbol=symbol, limit=1)
        return float(data[-1]["fundingRate"]) if data else 0.0

    @_retry()
    async def get_open_interest(self, symbol: str) -> float:
        data = await self._c.futures_open_interest(symbol=symbol)
        return float(data["openInterest"])

    @_retry()
    async def get_orderbook(self, symbol: str, limit: int = 20, futures: bool = True) -> dict:
        if futures:
            return await self._c.futures_order_book(symbol=symbol, limit=limit)
        return await self._c.get_order_book(symbol=symbol, limit=limit)

    @_retry()
    async def get_futures_exchange_info(self) -> dict:
        return await self._c.futures_exchange_info()

    @_retry()
    async def get_mark_price(self, symbol: str) -> float:
        data = await self._c.futures_mark_price(symbol=symbol)
        return float(data["markPrice"])

    # ------------------------------------------------------------------ #
    # Spot orders
    # ------------------------------------------------------------------ #

    @_retry()
    async def place_spot_market(self, symbol: str, side: str, quantity: float) -> dict:
        log.info("SPOT MARKET %s %s qty=%.6f", side, symbol, quantity)
        return await self._c.order_market(symbol=symbol, side=side, quantity=f"{quantity:.6f}")

    @_retry()
    async def place_spot_limit(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        log.info("SPOT LIMIT %s %s qty=%.6f @ %.4f", side, symbol, quantity, price)
        return await self._c.order_limit(
            symbol=symbol, side=side,
            quantity=f"{quantity:.6f}", price=f"{price:.4f}",
            timeInForce="GTC",
        )

    # ------------------------------------------------------------------ #
    # USDM Futures orders
    # ------------------------------------------------------------------ #

    @_retry()
    async def set_leverage(self, symbol: str, leverage: int) -> None:
        await self._c.futures_change_leverage(symbol=symbol, leverage=leverage)

    @_retry()
    async def place_futures_market(self, symbol: str, side: str, quantity: float) -> dict:
        log.info("FUTURES MARKET %s %s qty=%.6f", side, symbol, quantity)
        return await self._c.futures_create_order(
            symbol=symbol, side=side, type="MARKET", quantity=f"{quantity:.6f}"
        )

    @_retry()
    async def place_futures_limit(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> dict:
        log.info("FUTURES LIMIT %s %s qty=%.6f @ %.4f", side, symbol, quantity, price)
        return await self._c.futures_create_order(
            symbol=symbol, side=side, type="LIMIT",
            quantity=f"{quantity:.6f}", price=f"{price:.4f}",
            timeInForce="GTC",
        )

    @_retry()
    async def cancel_futures_order(self, symbol: str, order_id: int) -> dict:
        return await self._c.futures_cancel_order(symbol=symbol, orderId=order_id)

    @_retry()
    async def get_futures_order(self, symbol: str, order_id: int) -> dict:
        return await self._c.futures_get_order(symbol=symbol, orderId=order_id)

    # ------------------------------------------------------------------ #
    # Binance Options (EAPI — European vanilla, BTC/ETH)
    # ------------------------------------------------------------------ #

    @_retry()
    async def get_options_chain(self, underlying: str) -> list[dict]:
        """All listed option contracts for 'BTC' or 'ETH'."""
        return await self._c.options_info(underlyingAsset=underlying)

    @_retry()
    async def get_options_mark_price(self, symbol: str) -> dict:
        return await self._c.options_mark_price(symbol=symbol)

    @_retry()
    async def place_options_limit(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> dict:
        log.info("OPTIONS LIMIT %s %s qty=%.2f @ %.4f", side, symbol, quantity, price)
        return await self._c.options_place_order(
            symbol=symbol, side=side, type="LIMIT",
            quantity=f"{quantity:.2f}", price=f"{price:.4f}",
            timeInForce="GTC",
        )

    @_retry()
    async def get_options_positions(self) -> list[dict]:
        return await self._c.options_user_trades()

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #

    @_retry()
    async def get_server_time(self) -> int:
        t = await self._c.get_server_time()
        return int(t["serverTime"])

    @_retry()
    async def get_recent_trades(self, symbol: str, limit: int = 500, futures: bool = True) -> list[dict]:
        if futures:
            return await self._c.futures_recent_trades(symbol=symbol, limit=limit)
        return await self._c.get_recent_trades(symbol=symbol, limit=limit)

    # ------------------------------------------------------------------ #
    # Stream manager factory (mirrors AlpacaClient interface)
    # ------------------------------------------------------------------ #

    def create_stream_manager(self, store):
        from exchange.streams import StreamManager
        return StreamManager(self._c, store)
