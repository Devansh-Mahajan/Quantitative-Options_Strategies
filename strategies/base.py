"""Abstract base strategy and Signal dataclass."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import NamedTuple, Any


class Signal(NamedTuple):
    symbol: str
    market: str          # spot | futures | options
    side: str            # BUY | SELL
    quantity: float      # units (BTC, ETH, etc.)
    price: float         # 0 = market order
    confidence: float    # 0..1
    strategy: str
    required_regime: str | None = None   # bull|bear|ranging|volatile|None=any
    leverage: int = 1
    meta: dict | None = None


class SignalCooldown:
    """
    Turnover throttle shared by the live orchestrator and the backtester so
    both measure the same behavior.

    Drops a signal when the same (strategy, symbol, side) was emitted within
    the last `gap` counter ticks (cycles live, bars in backtests). Eleven
    strategies emit every cycle with no rebalance gating; at ~50bps roundtrip
    on Alpaca that re-entry churn is a systematic bleed. Rebalance-gated
    strategies are naturally slower, so this is effectively a no-op for them.
    """

    def __init__(self, gap: int = 4) -> None:
        self.gap = max(0, int(gap))
        self._last_emit: dict[tuple[str, str, str], int] = {}

    def filter(self, signals: list["Signal"], counter: int) -> list["Signal"]:
        if self.gap <= 0:
            return signals
        kept: list[Signal] = []
        for sig in signals:
            key = (sig.strategy, sig.symbol, sig.side)
            last = self._last_emit.get(key)
            if last is not None and (counter - last) < self.gap:
                continue
            self._last_emit[key] = counter
            kept.append(sig)
        return kept


class BaseStrategy(ABC):
    name: str = "base"
    market: str = "futures"     # spot | futures | options
    required_regime: str | None = None

    @abstractmethod
    def generate_signals(
        self,
        store,           # MarketDataStore
        regime: str,
        predictions: dict[str, Any],
    ) -> list[Signal]:
        ...

    @property
    def symbols(self) -> list[str]:
        return []
