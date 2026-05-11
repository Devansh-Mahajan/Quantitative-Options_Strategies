"""
Alpaca broker adapter — implements the same interface as BinanceClient so every
strategy, risk engine, and execution layer works unchanged when BROKER=alpaca.

Supported on Alpaca:
  - Crypto spot (BTC/USD, ETH/USD, SOL/USD …)   → mapped from BTCUSDTstyle symbols
  - Paper trading ($100k simulated USD)
  - Historical bars and live WebSocket quotes/bars

Not supported on Alpaca (returns safe defaults):
  - Futures leverage, funding rates, open interest → returns 0
  - Binance options (EAPI)                        → not available
  - Coin-margined contracts                       → not available

Strategies that produce signals requiring unsupported features (funding_arb,
basis_trade, options_vol) will simply generate no signals because the data
they query comes back as 0.0 — no special handling needed.
"""

from __future__ import annotations
import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta

from alpaca.trading.enums import AssetClass

from bot.config import cfg

log = logging.getLogger("exchange.alpaca_client")

# --------------------------------------------------------------------------- #
# Symbol helpers
# --------------------------------------------------------------------------- #

def _to_alpaca(symbol: str) -> str:
    """Map internal symbols to Alpaca symbols while preserving stock tickers."""
    sym = symbol.upper()
    if "/" in sym:
        return sym
    if cfg.is_crypto_symbol(sym):
        for suffix in ("USDT", "BUSD", "USD"):
            if sym.endswith(suffix):
                base = sym[: -len(suffix)]
                return f"{base}/USD"
        return f"{sym}/USD"
    return sym


def _from_alpaca(alpaca_sym: str) -> str:
    """'BTC/USD' → 'BTCUSDT'; 'AAPL' → 'AAPL'."""
    sym = alpaca_sym.upper()
    if "/" not in sym:
        return sym
    base, quote = sym.split("/", 1)
    if quote in {"USD", "USDT", "BUSD"}:
        return f"{base}USDT"
    return sym.replace("/", "")


def _is_stock_symbol(symbol: str) -> bool:
    return cfg.is_stock_symbol(symbol)


def _is_crypto_symbol(symbol: str) -> bool:
    return cfg.is_crypto_symbol(symbol)


def _response_get(mapping, key: str):
    if mapping is None:
        return None
    if hasattr(mapping, "data"):
        data = getattr(mapping, "data", None) or {}
        return data.get(key)
    if hasattr(mapping, "get"):
        return mapping.get(key)
    try:
        return mapping[key]
    except (KeyError, TypeError):
        return None


def _alpaca_interval(interval: str):
    """Map Binance-style interval string to alpaca-py TimeFrame."""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    _map = {
        "1m":  TimeFrame.Minute,
        "5m":  TimeFrame(5, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "1h":  TimeFrame.Hour,
        "4h":  TimeFrame(4, TimeFrameUnit.Hour),
        "1d":  TimeFrame.Day,
    }
    return _map.get(interval, TimeFrame.Hour)


# --------------------------------------------------------------------------- #
# AlpacaClient
# --------------------------------------------------------------------------- #

class AlpacaClient:
    """Alpaca broker adapter with the same async interface as BinanceClient."""

    def __init__(self) -> None:
        self._trading = None   # alpaca.trading.client.TradingClient
        self._crypto_data = None
        self._stock_data = None
        self._price_cache: dict[str, tuple[float, float]] = {}
        self._cache_ttl = 3.0

    async def start(self) -> None:
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient
        from alpaca.data.historical.stock import StockHistoricalDataClient

        self._trading = TradingClient(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_api_secret,
            paper=cfg.alpaca_paper,
        )
        self._crypto_data = CryptoHistoricalDataClient(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_api_secret,
        )
        self._stock_data = StockHistoricalDataClient(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_api_secret,
        )
        # Verify connectivity
        account = await asyncio.to_thread(self._trading.get_account)
        log.info(
            "Alpaca client connected (paper=%s) — account=%s equity=$%.2f",
            cfg.alpaca_paper, account.id, float(account.equity),
        )

    async def close(self) -> None:
        log.info("Alpaca client closed")

    # ------------------------------------------------------------------ #
    # Account
    # ------------------------------------------------------------------ #

    async def get_futures_balance(self) -> dict[str, float]:
        """Returns equity-first balances for portfolio sizing and monitoring."""
        account = await asyncio.to_thread(self._trading.get_account)
        equity = float(getattr(account, "portfolio_value", None) or getattr(account, "equity", 0.0) or 0.0)
        cash = float(getattr(account, "cash", 0.0) or 0.0)
        buying_power = float(getattr(account, "buying_power", 0.0) or 0.0)
        return {
            "USDT": equity,
            "USD": cash,
            "CASH": cash,
            "BUYING_POWER": buying_power,
        }

    async def get_spot_balance(self) -> dict[str, float]:
        positions = await asyncio.to_thread(self._trading.get_all_positions)
        return {
            p.symbol.replace("/", ""): float(p.qty)
            for p in positions
            if getattr(p, "asset_class", None) == AssetClass.CRYPTO
        }

    async def get_futures_positions(self) -> list[dict]:
        """Returns positions in the same dict format as Binance futures_position_information."""
        positions = await asyncio.to_thread(self._trading.get_all_positions)
        result = []
        for p in positions:
            if getattr(p, "asset_class", None) != AssetClass.CRYPTO:
                continue
            binance_sym = _from_alpaca(p.symbol)
            result.append({
                "symbol": binance_sym,
                "positionAmt": str(float(p.qty) if p.side.value == "long" else -float(p.qty)),
                "entryPrice": str(p.avg_entry_price),
                "markPrice": str(p.current_price or p.avg_entry_price),
                "unRealizedProfit": str(p.unrealized_pl or 0),
                "leverage": "1",
                "market": "spot",
            })
        return result

    async def get_spot_positions(self) -> dict[str, float]:
        return await self.get_spot_balance()

    async def get_account_snapshot(self) -> dict[str, float | str | None]:
        account = await asyncio.to_thread(self._trading.get_account)
        total_equity = float(getattr(account, "portfolio_value", None) or getattr(account, "equity", 0.0) or 0.0)
        cash = float(getattr(account, "cash", 0.0) or 0.0)
        buying_power = float(getattr(account, "buying_power", 0.0) or 0.0)
        last_equity = float(getattr(account, "last_equity", 0.0) or 0.0)
        day_pnl = (total_equity - last_equity) if last_equity else None
        day_pnl_pct = ((day_pnl / last_equity) * 100.0) if day_pnl is not None and last_equity else None
        status = getattr(getattr(account, "status", None), "value", getattr(account, "status", None))
        return {
            "equity": total_equity,
            "cash": cash,
            "buying_power": buying_power,
            "last_equity": last_equity or None,
            "day_pnl": day_pnl,
            "day_pnl_pct": day_pnl_pct,
            "status": str(status).upper() if status else None,
        }

    async def get_all_positions_raw(self) -> list[object]:
        return await asyncio.to_thread(self._trading.get_all_positions)

    # ------------------------------------------------------------------ #
    # Market data
    # ------------------------------------------------------------------ #

    async def get_price(self, symbol: str) -> float:
        now = time.monotonic()
        if symbol in self._price_cache:
            price, ts = self._price_cache[symbol]
            if now - ts < self._cache_ttl:
                return price

        alpaca_sym = _to_alpaca(symbol)
        if _is_stock_symbol(symbol):
            from alpaca.data.requests import StockLatestTradeRequest

            req = StockLatestTradeRequest(symbol_or_symbols=alpaca_sym)
            trades_resp = await asyncio.to_thread(self._stock_data.get_stock_latest_trade, req)
            trade = _response_get(trades_resp, alpaca_sym)
            price = float(trade.price) if trade else 0.0
        else:
            try:
                from alpaca.data.requests import CryptoLatestBarRequest as _LatestBarReq
            except ImportError:
                from alpaca.data.requests import LatestCryptoBarRequest as _LatestBarReq
            req = _LatestBarReq(symbol_or_symbols=alpaca_sym)
            bars_resp = await asyncio.to_thread(self._crypto_data.get_crypto_latest_bar, req)
            bar = _response_get(bars_resp, alpaca_sym)
            price = float(bar.close) if bar else 0.0
        self._price_cache[symbol] = (price, now)
        return price

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 200,
        futures: bool = True,   # ignored — Alpaca has no futures; uses crypto spot bars
    ) -> list[dict]:
        alpaca_sym = _to_alpaca(symbol)
        tf = _alpaca_interval(interval)

        # Calculate start time from limit + interval
        end = datetime.now(timezone.utc)
        delta_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        minutes = delta_map.get(interval, 60) * limit
        start = end - timedelta(minutes=minutes + 60)  # +buffer

        if _is_stock_symbol(symbol):
            from alpaca.data.requests import StockBarsRequest

            req = StockBarsRequest(
                symbol_or_symbols=alpaca_sym,
                timeframe=tf,
                start=start,
                end=end,
                limit=limit,
            )
            bars_resp = await asyncio.to_thread(self._stock_data.get_stock_bars, req)
        else:
            from alpaca.data.requests import CryptoBarsRequest

            req = CryptoBarsRequest(
                symbol_or_symbols=alpaca_sym,
                timeframe=tf,
                start=start,
                end=end,
                limit=limit,
            )
            bars_resp = await asyncio.to_thread(self._crypto_data.get_crypto_bars, req)

        if hasattr(bars_resp, "data"):
            bars = bars_resp.data.get(alpaca_sym, [])
        elif hasattr(bars_resp, "get"):
            bars = bars_resp.get(alpaca_sym, [])
        else:
            try:
                bars = list(bars_resp[alpaca_sym])
            except (KeyError, TypeError):
                bars = []

        result = []
        for bar in bars[-limit:]:
            ts = int(bar.timestamp.timestamp() * 1000)
            quote_volume = float(bar.volume) * float(bar.close)
            result.append({
                "open_time": ts,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "close_time": ts + delta_map.get(interval, 60) * 60_000,
                "quote_volume": quote_volume,
                "trades": int(getattr(bar, "trade_count", 0)),
                "taker_buy_base": float(bar.volume) * 0.5,
                "taker_buy_quote": quote_volume * 0.5,
            })
        return result

    async def get_funding_rate(self, symbol: str) -> float:
        """Not available on Alpaca — returns 0.0 so funding_arb produces no signals."""
        return 0.0

    async def get_open_interest(self, symbol: str) -> float:
        """Not available on Alpaca — returns 0.0."""
        return 0.0

    async def get_orderbook(self, symbol: str, limit: int = 20, futures: bool = True) -> dict:
        alpaca_sym = _to_alpaca(symbol)
        try:
            if _is_stock_symbol(symbol):
                from alpaca.data.requests import StockLatestQuoteRequest

                req = StockLatestQuoteRequest(symbol_or_symbols=alpaca_sym)
                quotes = await asyncio.to_thread(self._stock_data.get_stock_latest_quote, req)
                quote = _response_get(quotes, alpaca_sym)
                if quote:
                    return {
                        "bids": [[float(quote.bid_price), float(getattr(quote, "bid_size", 0.0) or 0.0)]],
                        "asks": [[float(quote.ask_price), float(getattr(quote, "ask_size", 0.0) or 0.0)]],
                    }
            else:
                from alpaca.data.requests import CryptoLatestOrderbookRequest

                req = CryptoLatestOrderbookRequest(symbol_or_symbols=alpaca_sym)
                ob = await asyncio.to_thread(self._crypto_data.get_crypto_latest_orderbook, req)
                book = _response_get(ob, alpaca_sym)
                if book:
                    return {
                        "bids": [[b.p, b.s] for b in book.bids[:limit]],
                        "asks": [[a.p, a.s] for a in book.asks[:limit]],
                    }
        except Exception:
            pass
        return {"bids": [], "asks": []}

    async def get_mark_price(self, symbol: str) -> float:
        return await self.get_price(symbol)

    async def get_futures_exchange_info(self) -> dict:
        return {}

    async def get_server_time(self) -> int:
        return int(time.time() * 1000)

    async def get_recent_trades(self, symbol: str, limit: int = 500, futures: bool = True) -> list[dict]:
        alpaca_sym = _to_alpaca(symbol)
        try:
            if _is_stock_symbol(symbol):
                from alpaca.data.requests import StockLatestTradeRequest

                req = StockLatestTradeRequest(symbol_or_symbols=alpaca_sym)
                trades = await asyncio.to_thread(self._stock_data.get_stock_latest_trade, req)
            else:
                from alpaca.data.requests import CryptoLatestTradeRequest

                req = CryptoLatestTradeRequest(symbol_or_symbols=alpaca_sym)
                trades = await asyncio.to_thread(self._crypto_data.get_crypto_latest_trade, req)
            t = _response_get(trades, alpaca_sym)
            if t:
                return [{"price": str(t.price), "qty": str(t.size)}]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------ #
    # Spot orders
    # ------------------------------------------------------------------ #

    async def place_spot_market(self, symbol: str, side: str, quantity: float) -> dict:
        return await self._place_order(symbol, side, quantity, price=None, order_type="market")

    async def place_spot_limit(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        return await self._place_order(symbol, side, quantity, price=price, order_type="limit")

    # ------------------------------------------------------------------ #
    # "Futures" orders — mapped to Alpaca crypto spot (no real leverage)
    # ------------------------------------------------------------------ #

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        pass  # No leverage on Alpaca crypto

    async def place_futures_market(self, symbol: str, side: str, quantity: float) -> dict:
        log.info("ALPACA MARKET %s %s qty=%.6f (no leverage)", side, symbol, quantity)
        return await self._place_order(symbol, side, quantity, price=None, order_type="market")

    async def place_futures_limit(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        log.info("ALPACA LIMIT %s %s qty=%.6f @ %.4f", side, symbol, quantity, price)
        return await self._place_order(symbol, side, quantity, price=price, order_type="limit")

    async def cancel_futures_order(self, symbol: str, order_id: int) -> dict:
        try:
            await asyncio.to_thread(self._trading.cancel_order_by_id, str(order_id))
        except Exception as exc:
            log.warning("Cancel order %s failed: %s", order_id, exc)
        return {}

    async def get_futures_order(self, symbol: str, order_id: int) -> dict:
        try:
            order = await asyncio.to_thread(self._trading.get_order_by_id, str(order_id))
            status_map = {
                "filled": "FILLED",
                "canceled": "CANCELED",
                "new": "NEW",
                "partially_filled": "PARTIALLY_FILLED",
                "rejected": "REJECTED",
                "expired": "EXPIRED",
            }
            return {
                "orderId": order_id,
                "status": status_map.get(order.status.value, "NEW"),
                "avgPrice": str(order.filled_avg_price or 0),
                "executedQty": str(order.filled_qty or 0),
            }
        except Exception:
            return {"orderId": order_id, "status": "NEW"}

    # ------------------------------------------------------------------ #
    # Options — not available on Alpaca crypto
    # ------------------------------------------------------------------ #

    async def get_options_chain(self, underlying: str) -> list[dict]:
        return []

    async def get_options_mark_price(self, symbol: str) -> dict:
        return {}

    async def place_options_limit(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        log.warning("Options not supported on Alpaca — order skipped")
        return {}

    async def get_options_positions(self) -> list[dict]:
        return []

    # ------------------------------------------------------------------ #
    # Internal order placement
    # ------------------------------------------------------------------ #

    async def _place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None,
        order_type: str,
    ) -> dict:
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        alpaca_sym = _to_alpaca(symbol)
        alpaca_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        tif = TimeInForce.DAY if _is_stock_symbol(symbol) else TimeInForce.GTC
        price_decimals = 2 if _is_stock_symbol(symbol) else 4

        try:
            if order_type == "market" or price is None:
                req = MarketOrderRequest(
                    symbol=alpaca_sym,
                    qty=round(quantity, 6),
                    side=alpaca_side,
                    time_in_force=tif,
                )
            else:
                req = LimitOrderRequest(
                    symbol=alpaca_sym,
                    qty=round(quantity, 6),
                    side=alpaca_side,
                    limit_price=round(price, price_decimals),
                    time_in_force=tif,
                )
            order = await asyncio.to_thread(self._trading.submit_order, req)
            return {"orderId": int(order.id.int) if hasattr(order.id, "int") else hash(str(order.id)),
                    "status": order.status.value, "symbol": symbol}
        except Exception as exc:
            log.error("Alpaca order failed %s %s: %s", side, symbol, exc)
            return {}

    # ------------------------------------------------------------------ #
    # Stream manager factory (called by orchestrator)
    # ------------------------------------------------------------------ #

    def create_stream_manager(self, store):
        from exchange.alpaca_streams import AlpacaStreamManager
        return AlpacaStreamManager(self, store)
