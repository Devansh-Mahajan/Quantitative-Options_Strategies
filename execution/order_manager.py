"""
Order lifecycle management:
- Place limit orders with fallback to market
- Monitor fill status and reprice unfilled orders
- TWAP execution for large orders
- Track slippage vs expected price
"""

from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field

from bot.config import cfg
from bot import state
from exchange.client import BinanceClient
from strategies.base import Signal

log = logging.getLogger("execution.order_manager")


@dataclass
class ActiveOrder:
    order_id: int
    symbol: str
    market: str          # spot | futures | options
    side: str
    quantity: float
    expected_price: float
    placed_at: float = field(default_factory=time.monotonic)
    reprices: int = 0
    trade_db_id: int = 0
    strategy: str = ""


class OrderManager:
    def __init__(self, client: BinanceClient) -> None:
        self._client = client
        self._active: dict[int, ActiveOrder] = {}

    # ------------------------------------------------------------------ #
    # Main entry — execute a signal
    # ------------------------------------------------------------------ #

    async def execute(self, signal: Signal, leverage: int = 1) -> int | None:
        """
        Place order for signal. Returns Binance order_id (or None on failure).
        """
        if signal.quantity <= 0:
            return None

        try:
            if signal.market == "futures":
                return await self._execute_futures(signal, leverage)
            elif signal.market == "spot":
                return await self._execute_spot(signal)
            elif signal.market == "options":
                return await self._execute_options(signal)
        except Exception as exc:
            log.error("Order execution failed for %s %s: %s", signal.side, signal.symbol, exc)
        return None

    # ------------------------------------------------------------------ #
    # Futures
    # ------------------------------------------------------------------ #

    async def _execute_futures(self, signal: Signal, leverage: int) -> int | None:
        symbol = signal.symbol
        side = signal.side
        qty = signal.quantity
        price = signal.price

        # Set leverage
        safe_lev = min(leverage, cfg.max_leverage)
        try:
            await self._client.set_leverage(symbol, safe_lev)
        except Exception:
            pass  # already set or not changeable

        if price > 0:
            # Limit order — slightly inside the spread for maker fee
            limit_price = self._maker_price(side, price)
            resp = await self._client.place_futures_limit(symbol, side, qty, limit_price)
            exec_price = limit_price
        elif cfg.maker_only:
            # maker_only with no reference price — fall back to market to avoid $0 limit
            log.warning("maker_only=True but signal has no price for %s %s — using market order", side, symbol)
            resp = await self._client.place_futures_market(symbol, side, qty)
            exec_price = 0.0
        else:
            resp = await self._client.place_futures_market(symbol, side, qty)
            exec_price = 0.0

        order_id = int(resp.get("orderId", 0))
        trade_id = state.record_trade(
            symbol=symbol, market="futures", side=side,
            quantity=qty, price=price,
            strategy=signal.strategy, order_id=str(order_id),
        )
        if order_id:
            self._active[order_id] = ActiveOrder(
                order_id=order_id, symbol=symbol, market="futures",
                side=side, quantity=qty, expected_price=price,
                trade_db_id=trade_id, strategy=signal.strategy,
            )
            log.info("Futures %s %s %.6f @ %.2f [id=%d]", side, symbol, qty, exec_price, order_id)
        return order_id

    # ------------------------------------------------------------------ #
    # Spot
    # ------------------------------------------------------------------ #

    async def _execute_spot(self, signal: Signal) -> int | None:
        symbol = signal.symbol
        side = signal.side
        qty = signal.quantity
        price = signal.price

        if price > 0:
            limit_price = self._maker_price(side, price)
            resp = await self._client.place_spot_limit(symbol, side, qty, limit_price)
        else:
            resp = await self._client.place_spot_market(symbol, side, qty)

        order_id = int(resp.get("orderId", 0))
        trade_id = state.record_trade(
            symbol=symbol, market="spot", side=side,
            quantity=qty, price=price, strategy=signal.strategy,
            order_id=str(order_id),
        )
        if order_id:
            self._active[order_id] = ActiveOrder(
                order_id=order_id, symbol=symbol, market="spot",
                side=side, quantity=qty, expected_price=price,
                trade_db_id=trade_id, strategy=signal.strategy,
            )
        return order_id

    # ------------------------------------------------------------------ #
    # Options
    # ------------------------------------------------------------------ #

    async def _execute_options(self, signal: Signal) -> int | None:
        resp = await self._client.place_options_limit(
            signal.symbol, signal.side, signal.quantity, signal.price
        )
        order_id = int(resp.get("orderId", 0))
        state.record_trade(
            symbol=signal.symbol, market="options", side=signal.side,
            quantity=signal.quantity, price=signal.price,
            strategy=signal.strategy, order_id=str(order_id),
        )
        return order_id

    # ------------------------------------------------------------------ #
    # Monitoring + repricing
    # ------------------------------------------------------------------ #

    async def monitor_and_reprice(self) -> None:
        """
        Called periodically by orchestrator.
        Cancel and reprice orders that haven't filled within ORDER_TIMEOUT_SECONDS.
        """
        now = time.monotonic()
        to_cancel: list[int] = []

        for oid, order in list(self._active.items()):
            age = now - order.placed_at
            if age < cfg.order_timeout_seconds:
                continue

            if order.market == "futures":
                try:
                    status = await self._client.get_futures_order(order.symbol, oid)
                    if status.get("status") == "FILLED":
                        self._handle_fill(oid, status)
                        continue
                    elif status.get("status") in ("CANCELED", "EXPIRED", "REJECTED"):
                        del self._active[oid]
                        continue
                except Exception:
                    pass

            if order.reprices < cfg.max_reprices:
                await self._reprice(order)
            else:
                to_cancel.append(oid)

        for oid in to_cancel:
            await self._force_cancel_market(oid)

    async def _reprice(self, order: ActiveOrder) -> None:
        """Cancel existing order and re-submit at a more aggressive price."""
        try:
            if order.market == "futures":
                await self._client.cancel_futures_order(order.symbol, order.order_id)
        except Exception:
            pass

        # More aggressive: tighten by 0.1%
        adj = 0.999 if order.side == "BUY" else 1.001
        new_price = order.expected_price * adj

        try:
            if order.market == "futures":
                resp = await self._client.place_futures_limit(order.symbol, order.side, order.quantity, new_price)
                new_oid = int(resp.get("orderId", 0))
                del self._active[order.order_id]
                order = ActiveOrder(
                    order_id=new_oid, symbol=order.symbol, market=order.market,
                    side=order.side, quantity=order.quantity, expected_price=new_price,
                    reprices=order.reprices + 1, trade_db_id=order.trade_db_id,
                    strategy=order.strategy,
                )
                self._active[new_oid] = order
                log.debug("Repriced order %s -> %d @ %.4f (reprice #%d)", order.symbol, new_oid, new_price, order.reprices)
        except Exception as exc:
            log.warning("Reprice failed for %s: %s", order.symbol, exc)

    async def _force_cancel_market(self, oid: int) -> None:
        order = self._active.pop(oid, None)
        if order is None:
            return
        log.warning("Max reprices reached for %s — cancelling", order.symbol)
        try:
            if order.market == "futures":
                await self._client.cancel_futures_order(order.symbol, oid)
        except Exception:
            pass

    def _handle_fill(self, oid: int, status: dict) -> None:
        order = self._active.pop(oid, None)
        if order:
            fill_price = float(status.get("avgPrice", order.expected_price))
            slippage = abs(fill_price - order.expected_price) / (order.expected_price + 1e-10)
            log.info("FILLED %s %s @ %.4f (slippage=%.3f%%)", order.symbol, order.side, fill_price, slippage * 100)

    @staticmethod
    def _maker_price(side: str, mid_price: float) -> float:
        """Return maker-friendly limit price just inside spread."""
        return round(mid_price * 0.9995, 4) if side == "BUY" else round(mid_price * 1.0005, 4)

    # ------------------------------------------------------------------ #
    # TWAP execution
    # ------------------------------------------------------------------ #

    async def twap_execute(self, signal: Signal, slices: int | None = None) -> list[int | None]:
        """Split a large order into equal slices over time."""
        slices = slices or cfg.twap_slices
        slice_qty = signal.quantity / slices
        interval = cfg.order_timeout_seconds / slices
        ids = []
        for i in range(slices):
            sub = signal._replace(quantity=slice_qty)
            oid = await self.execute(sub)
            ids.append(oid)
            if i < slices - 1:
                await asyncio.sleep(interval)
        return ids
