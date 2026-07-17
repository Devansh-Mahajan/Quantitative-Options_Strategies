"""
Realistic order fill simulation.
Models venue fee tiers, bid-ask spread slippage, and volume impact.
No look-ahead bias: limit fills use the NEXT bar's OHLC; market fills at next open.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


# Default fees = Binance futures VIP-0. Callers should use venue_fees() so a
# backtest charges what the LIVE venue actually charges — Alpaca crypto is
# ~6x more expensive than these defaults, which materially changes the
# viability of high-turnover strategies.
MAKER_FEE = 0.0002    # 0.02%
TAKER_FEE = 0.0004    # 0.04%

# (maker, taker) per venue
VENUE_FEES = {
    "alpaca": (0.0015, 0.0025),          # Alpaca crypto retail tier
    "binance_spot": (0.0010, 0.0010),    # Binance spot VIP-0
    "binance_futures": (0.0002, 0.0004), # Binance USDM futures VIP-0
}


def venue_fees(broker: str, market: str = "spot") -> tuple[float, float]:
    """(maker, taker) for a broker/market combination."""
    broker = str(broker or "").lower()
    if broker == "alpaca":
        return VENUE_FEES["alpaca"]
    if broker == "binance":
        return VENUE_FEES["binance_futures" if market == "futures" else "binance_spot"]
    return (MAKER_FEE, TAKER_FEE)


# Slippage as fraction of ATR (market impact proxy)
MARKET_SLIPPAGE_ATR_FRAC = 0.05   # 5% of ATR per market order
LIMIT_PARTIAL_FILL_PROB = 0.85    # fill probability when a limit is only touched (not penetrated)
LIMIT_ORDER_TTL_BARS = 5          # unfilled limit orders expire after this many bars

# Maximum order size as fraction of bar volume before significant impact
MAX_VOL_FRACTION = 0.01


@dataclass
class Order:
    symbol: str
    side: str              # BUY | SELL
    order_type: OrderType
    quantity: float
    limit_price: float = 0.0
    strategy: str = ""
    bar_index: int = 0     # bar when order was submitted
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float = 0.0
    fill_bar: int = 0
    commission: float = 0.0
    slippage: float = 0.0
    # Internal: remaining quantity (for partial fills)
    remaining: float = field(init=False)

    def __post_init__(self):
        self.remaining = self.quantity

    @property
    def is_done(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED)


@dataclass
class Fill:
    order: Order
    fill_price: float
    quantity: float
    commission: float
    slippage_cost: float
    bar_index: int

    @property
    def net_cost(self) -> float:
        """Net cash moved: positive = cash out (buy), negative = cash in (sell)."""
        sign = 1.0 if self.order.side == "BUY" else -1.0
        return sign * (self.fill_price * self.quantity + self.commission + self.slippage_cost)

    @property
    def gross_notional(self) -> float:
        return self.fill_price * self.quantity


class FillModel:
    """
    Processes pending orders against each completed bar.
    Call `process_bar(bar, pending_orders, bar_index)` on every bar.
    """

    def __init__(
        self,
        maker_fee: float = MAKER_FEE,
        taker_fee: float = TAKER_FEE,
        slippage_atr_frac: float = MARKET_SLIPPAGE_ATR_FRAC,
        max_vol_fraction: float = MAX_VOL_FRACTION,
        seed: int = 42,
    ) -> None:
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_atr_frac = slippage_atr_frac
        self.max_vol_fraction = max_vol_fraction
        # Seeded RNG for touch-only limit fills — keeps backtests reproducible.
        self._rng = random.Random(seed)

    def process_bar(
        self,
        bar: dict,         # {"open":…,"high":…,"low":…,"close":…,"volume":…,"atr":…}
        pending: list[Order],
        bar_index: int,
    ) -> list[Fill]:
        """
        Attempt to fill pending orders against this bar.
        Returns a list of Fill objects for orders that were filled.
        """
        fills: list[Fill] = []
        atr = bar.get("atr", bar["close"] * 0.01)   # fallback: 1% of price
        volume = bar["volume"]

        for order in pending:
            if order.is_done:
                continue
            if bar_index <= order.bar_index:
                continue  # never fill on the same bar the order was submitted

            fill = None
            if order.order_type == OrderType.MARKET:
                fill = self._fill_market(order, bar, atr, volume, bar_index)
            elif order.order_type == OrderType.LIMIT:
                if bar_index - order.bar_index > LIMIT_ORDER_TTL_BARS:
                    order.status = OrderStatus.EXPIRED
                    continue
                fill = self._fill_limit(order, bar, atr, volume, bar_index)

            if fill:
                fills.append(fill)
                order.status = OrderStatus.FILLED
                order.fill_price = fill.fill_price
                order.fill_bar = bar_index
                order.commission = fill.commission

        return fills

    # ------------------------------------------------------------------ #

    def _fill_market(self, order: Order, bar: dict, atr: float, volume: float, idx: int) -> Fill:
        """Market orders fill at bar open with slippage based on ATR + volume."""
        base_price = bar["open"]
        vol_impact = self._volume_impact(order.quantity, base_price, volume)
        slip_frac = self.slippage_atr_frac * atr / (base_price + 1e-10)
        total_slip_frac = slip_frac + vol_impact

        if order.side == "BUY":
            fill_price = base_price * (1 + total_slip_frac)
        else:
            fill_price = base_price * (1 - total_slip_frac)

        slippage_cost = abs(fill_price - base_price) * order.quantity
        commission = fill_price * order.quantity * self.taker_fee

        return Fill(
            order=order,
            fill_price=fill_price,
            quantity=order.quantity,
            commission=commission,
            slippage_cost=slippage_cost,
            bar_index=idx,
        )

    def _fill_limit(self, order: Order, bar: dict, atr: float, volume: float, idx: int) -> Optional[Fill]:
        """
        Limit orders fill when bar high/low crosses the limit price.
        Strict penetration (price trades THROUGH the limit) = full fill; a bare
        touch means our order sat in the queue at that level, so it only fills
        with LIMIT_PARTIAL_FILL_PROB. Conservative: fills between limit and open.
        """
        lp = order.limit_price
        touched = (order.side == "BUY" and bar["low"] <= lp) or \
                  (order.side == "SELL" and bar["high"] >= lp)

        if not touched:
            return None

        penetrated = (order.side == "BUY" and bar["low"] < lp) or \
                     (order.side == "SELL" and bar["high"] > lp)
        if not penetrated and self._rng.random() > LIMIT_PARTIAL_FILL_PROB:
            return None  # touched our level but queue ahead of us didn't clear

        # Conservative fill: halfway between limit and open (not always at limit)
        if order.side == "BUY":
            fill_price = min(lp, max(bar["open"] * 0.9999, lp * 0.9995))
        else:
            fill_price = max(lp, min(bar["open"] * 1.0001, lp * 1.0005))

        slippage_cost = abs(fill_price - lp) * order.quantity
        commission = fill_price * order.quantity * self.maker_fee  # limit = maker

        return Fill(
            order=order,
            fill_price=fill_price,
            quantity=order.quantity,
            commission=commission,
            slippage_cost=slippage_cost,
            bar_index=idx,
        )

    def _volume_impact(self, quantity: float, price: float, bar_volume: float) -> float:
        """Market impact: larger orders relative to bar volume cost more."""
        order_vol_frac = (quantity * price) / (bar_volume * price + 1e-10)
        # Linear impact model: each % of bar volume = 0.01% extra slippage
        return min(order_vol_frac / self.max_vol_fraction * 0.0001, 0.005)


def make_market_order(symbol: str, side: str, quantity: float, strategy: str, bar_index: int) -> Order:
    return Order(symbol=symbol, side=side, order_type=OrderType.MARKET,
                 quantity=quantity, strategy=strategy, bar_index=bar_index)


def make_limit_order(
    symbol: str, side: str, quantity: float, limit_price: float,
    strategy: str, bar_index: int,
) -> Order:
    return Order(symbol=symbol, side=side, order_type=OrderType.LIMIT,
                 quantity=quantity, limit_price=limit_price,
                 strategy=strategy, bar_index=bar_index)
