"""
Order flow / market microstructure strategy.
Signals: buy/sell order imbalance, large trade detection, VWAP deviation.
Works in all regimes, especially volatile.
"""

from __future__ import annotations
import logging

import numpy as np

from bot.config import cfg
from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategy.order_flow")

IMBALANCE_THRESHOLD = 0.65   # buy_volume / total_volume threshold
VWAP_THRESHOLD = 0.003       # 0.3% deviation from VWAP
LOOKBACK_BARS = 20


class OrderFlowStrategy(BaseStrategy):
    name = "order_flow"
    market = "futures"


    def __init__(
        self,
        imbalance_threshold: float = IMBALANCE_THRESHOLD,
        vwap_threshold: float = VWAP_THRESHOLD,
        lookback_bars: int = LOOKBACK_BARS,
    ) -> None:
        # Tunable thresholds (defaults = historical module constants).
        self._imbalance_threshold = imbalance_threshold
        self._vwap_threshold = vwap_threshold
        self._lookback_bars = int(lookback_bars)

    @property
    def symbols(self) -> list[str]:
        # Focus on the mixed top-of-book universe where quote-based microstructure is meaningful.
        return cfg.model_symbols[:4]

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        signals = []
        regime_scale = {"volatile": 0.9, "bull": 1.0, "bear": 0.8, "ranging": 0.7}.get(regime, 0.8)

        for symbol in self.symbols:
            try:
                df = store.get_history_df(symbol, "5m")
                if len(df) < self._lookback_bars + 5:
                    continue

                c = df["close"].astype(float).values
                v = df["volume"].astype(float).values
                taker = df.get("taker_buy_base", None)

                price = float(c[-1])

                # --- VWAP calculation ---
                typical = (df["high"].astype(float).values + df["low"].astype(float).values + c) / 3.0
                vwap = (typical[-self._lookback_bars:] * v[-self._lookback_bars:]).sum() / (v[-self._lookback_bars:].sum() + 1e-10)
                vwap_dev = (price - vwap) / (vwap + 1e-10)

                # --- Buy/sell imbalance ---
                imbalance = 0.5  # default neutral
                if taker is not None and not taker.isna().all():
                    tb = taker.astype(float).values[-self._lookback_bars:]
                    total = v[-self._lookback_bars:]
                    buy_vol = tb.sum()
                    total_vol = total.sum()
                    imbalance = buy_vol / (total_vol + 1e-10)

                # --- Order book spread from stream ---
                book = store.book.get(symbol)
                spread_pct = 0.0
                if book:
                    spread_pct = (book.ask - book.bid) / (book.bid + 1e-10)

                # Tight spread + strong buy imbalance + below VWAP = bullish microstructure
                if imbalance > self._imbalance_threshold and vwap_dev < -self._vwap_threshold and spread_pct < 0.001:
                    confidence = min(0.75, 0.45 + (imbalance - self._imbalance_threshold) * 2.0 + abs(vwap_dev) * 10)
                    confidence *= regime_scale
                    signals.append(Signal(
                        symbol=symbol, market="futures", side="BUY",
                        quantity=0.0, price=price,
                        confidence=confidence,
                        strategy=self.name,
                        meta={"imbalance": imbalance, "vwap_dev": vwap_dev},
                    ))

                elif imbalance < (1 - self._imbalance_threshold) and vwap_dev > self._vwap_threshold and spread_pct < 0.001:
                    confidence = min(0.75, 0.45 + ((1 - self._imbalance_threshold) - imbalance) * 2.0 + abs(vwap_dev) * 10)
                    confidence *= regime_scale
                    signals.append(Signal(
                        symbol=symbol, market="futures", side="SELL",
                        quantity=0.0, price=price,
                        confidence=confidence,
                        strategy=self.name,
                        meta={"imbalance": imbalance, "vwap_dev": vwap_dev},
                    ))

            except Exception as exc:
                log.debug("OrderFlow %s error: %s", symbol, exc)

        return signals
