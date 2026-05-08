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
    meta: dict = {}


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
