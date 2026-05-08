"""
Position tracking: collects live positions from Binance, computes P&L,
tracks per-strategy attribution, and triggers exit signals.
"""

from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from exchange.client import BinanceClient
from bot.config import cfg

log = logging.getLogger("execution.position_manager")


@dataclass
class Position:
    symbol: str
    market: str          # spot | futures | options
    side: str            # LONG | SHORT
    quantity: float
    entry_price: float
    mark_price: float = 0.0
    unrealised_pnl: float = 0.0
    realised_pnl: float = 0.0
    strategy: str = ""
    opened_at: float = field(default_factory=time.monotonic)
    leverage: int = 1
    notional: float = 0.0  # mark_price * quantity

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return self.unrealised_pnl / (self.entry_price * self.quantity + 1e-10)

    @property
    def holding_hours(self) -> float:
        return (time.monotonic() - self.opened_at) / 3600


class PositionManager:
    def __init__(self, client: BinanceClient) -> None:
        self._client = client
        self.futures: dict[str, Position] = {}
        self.spot: dict[str, Position] = {}
        self.options: dict[str, Position] = {}
        self._last_refresh = 0.0
        self._refresh_interval = 30.0  # seconds

    # ------------------------------------------------------------------ #
    # Refresh from exchange
    # ------------------------------------------------------------------ #

    async def refresh(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_refresh < self._refresh_interval:
            return
        self._last_refresh = now

        await asyncio.gather(
            self._refresh_futures(),
            return_exceptions=True,
        )
        if cfg.enable_options:
            await self._refresh_options()

    async def _refresh_futures(self) -> None:
        try:
            raw = await self._client.get_futures_positions()
            self.futures = {}
            for p in raw:
                qty = float(p.get("positionAmt", 0))
                if qty == 0:
                    continue
                sym = p["symbol"]
                entry = float(p.get("entryPrice", 0))
                mark = float(p.get("markPrice", 0))
                upnl = float(p.get("unRealizedProfit", 0))
                lev = int(p.get("leverage", 1))
                notional = abs(qty) * mark
                self.futures[sym] = Position(
                    symbol=sym,
                    market="futures",
                    side="LONG" if qty > 0 else "SHORT",
                    quantity=abs(qty),
                    entry_price=entry,
                    mark_price=mark,
                    unrealised_pnl=upnl,
                    leverage=lev,
                    notional=notional,
                )
        except Exception as exc:
            log.warning("Failed to refresh futures positions: %s", exc)

    async def _refresh_options(self) -> None:
        try:
            raw = await self._client.get_options_positions()
            self.options = {}
            for p in raw:
                sym = p.get("symbol", "")
                qty = float(p.get("quantity", 0))
                if qty == 0:
                    continue
                self.options[sym] = Position(
                    symbol=sym,
                    market="options",
                    side="LONG" if float(p.get("side", 1)) > 0 else "SHORT",
                    quantity=abs(qty),
                    entry_price=float(p.get("averagePrice", 0)),
                    mark_price=float(p.get("markPrice", 0)),
                    unrealised_pnl=float(p.get("unrealizedPnl", 0)),
                )
        except Exception as exc:
            log.warning("Failed to refresh options positions: %s", exc)

    # ------------------------------------------------------------------ #
    # Aggregated metrics
    # ------------------------------------------------------------------ #

    def all_positions(self) -> list[Position]:
        return list(self.futures.values()) + list(self.spot.values()) + list(self.options.values())

    def total_unrealised_pnl(self) -> float:
        return sum(p.unrealised_pnl for p in self.all_positions())

    def gross_notional(self) -> float:
        return sum(p.notional for p in self.all_positions())

    def net_notional(self) -> float:
        return sum(
            p.notional if p.side == "LONG" else -p.notional
            for p in self.all_positions()
        )

    def position_list_for_risk(self) -> list[dict]:
        result = []
        for p in self.all_positions():
            result.append({
                "symbol": p.symbol,
                "notional": p.notional,
                "side": 1 if p.side == "LONG" else -1,
            })
        return result

    def hhi(self, equity: float) -> float:
        positions = self.all_positions()
        if not positions or equity <= 0:
            return 0.0
        weights = [p.notional / (equity + 1e-10) for p in positions]
        total = sum(weights)
        if total <= 0:
            return 0.0
        w = [x / total for x in weights]
        return sum(x ** 2 for x in w)

    # ------------------------------------------------------------------ #
    # Exit detection
    # ------------------------------------------------------------------ #

    def positions_needing_exit(self, stop_loss_pct: float = 0.05, take_profit_pct: float = 0.10) -> list[Position]:
        exits = []
        for pos in self.all_positions():
            if pos.pnl_pct <= -stop_loss_pct:
                log.warning("STOP LOSS triggered for %s: pnl=%.2f%%", pos.symbol, pos.pnl_pct * 100)
                exits.append(pos)
            elif pos.pnl_pct >= take_profit_pct:
                log.info("TAKE PROFIT triggered for %s: pnl=%.2f%%", pos.symbol, pos.pnl_pct * 100)
                exits.append(pos)
        return exits

    def position(self, symbol: str, market: str = "futures") -> Position | None:
        store = {"futures": self.futures, "spot": self.spot, "options": self.options}.get(market, {})
        return store.get(symbol)

    def is_flat(self) -> bool:
        return len(self.all_positions()) == 0
