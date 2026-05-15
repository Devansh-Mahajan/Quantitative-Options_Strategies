"""
Liquidation Cascade Detector.

This is a crypto-native signal that is largely unavailable in traditional
markets — hence a genuine structural edge for algorithmic crypto traders.

How it works:
  Binance broadcasts every forced liquidation in real-time. When a large
  long position is force-liquidated, the exchange sells into the market,
  pushing price down further — potentially triggering more stop-losses
  (cascade). Conversely, a cascade of short liquidations creates a
  short-squeeze where forced buy orders push price up.

Two regimes:

  CASCADE MOMENTUM (follow the cascade)
  ─────────────────────────────────────
  When a very large liquidation cluster fires, the market is in genuine
  panic/euphoria. Joining the cascade is profitable in the first few bars
  because more liquidations are queued (leverage is systemic).

  POST-CASCADE MEAN REVERSION (fade the exhaustion)
  ──────────────────────────────────────────────────
  After a cascade peaks (liquidations slowing after a spike), the move
  is usually overdone. Smart money buys the capitulation (long liq cascade)
  or sells the short squeeze (short liq cascade).

Strategy selects mode based on cascade intensity ratio:
  - Intensity very high (>MOMENTUM_THRESHOLD) → follow momentum for first N bars
  - Intensity moderate (>ENTRY_THRESHOLD)     → fade for mean reversion

Data source: store.liquidations[symbol] — deque of Liquidation namedtuples
  Each Liquidation: symbol, side (BUY=short-liq, SELL=long-liq), quantity, price, ts
"""

from __future__ import annotations
import logging
import time
from collections import defaultdict

import numpy as np

from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.liquidation_cascade")

WINDOW_BARS       = 3      # scan last N bars of liquidation history
ENTRY_THRESHOLD   = 0.5e6  # $500k in liquidations triggers entry (USD notional)
MOMENTUM_THRESH   = 2.0e6  # $2M+ → momentum mode (follow cascade)
COOLDOWN_BARS     = 8      # bars before same symbol can trigger again
SYMBOLS_FOCUS     = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"]


class LiquidationCascadeStrategy(BaseStrategy):
    """
    Trade liquidation cascade events: follow large cascades, fade moderate ones.

    Requires a live Binance liquidation feed (store.liquidations populated by
    StreamManager._liquidation_stream). Falls back to no signal if no liq data.
    """

    name = "liquidation_cascade"
    market = "futures"

    def __init__(
        self,
        entry_threshold: float = ENTRY_THRESHOLD,
        momentum_threshold: float = MOMENTUM_THRESH,
        cooldown_bars: int = COOLDOWN_BARS,
        symbols: list[str] = SYMBOLS_FOCUS,
    ) -> None:
        self._entry_threshold   = entry_threshold
        self._momentum_threshold = momentum_threshold
        self._cooldown = cooldown_bars
        self._symbols = symbols
        self._last_signal_bar: dict[str, int] = defaultdict(lambda: -999)
        self._bar_count = 0

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        self._bar_count += 1
        signals: list[Signal] = []

        # Liquidations are less useful in ranging (no momentum to follow)
        regime_scale = {
            "bull":     0.9,
            "bear":     0.9,
            "volatile": 1.0,   # cascades hit hardest in volatile
            "ranging":  0.5,
        }.get(regime, 0.6)

        for sym in self._symbols:
            # Cooldown guard
            if self._bar_count - self._last_signal_bar[sym] < self._cooldown:
                continue

            liqs = list(store.liquidations.get(sym, []))
            if not liqs:
                continue

            # Separate by direction
            long_liqs  = [lq for lq in liqs if lq.side == "SELL"]   # SELL = long forced out
            short_liqs = [lq for lq in liqs if lq.side == "BUY"]    # BUY  = short forced out

            long_notional  = sum(lq.quantity * lq.price for lq in long_liqs)
            short_notional = sum(lq.quantity * lq.price for lq in short_liqs)

            dominant_notional = max(long_notional, short_notional)
            if dominant_notional < self._entry_threshold:
                continue

            is_long_cascade  = long_notional >= short_notional
            intensity_ratio  = dominant_notional / (self._entry_threshold + 1e-10)
            confidence_raw   = min(intensity_ratio / 4.0, 1.0) * regime_scale

            if confidence_raw < 0.3:
                continue

            # Current price for context
            candle = store.candles.get((sym, "1h"))
            if candle is None:
                continue

            if dominant_notional >= self._momentum_threshold:
                # MOMENTUM mode: follow the cascade direction
                # Long cascade (longs getting rekt) → market falling → SHORT
                # Short cascade (shorts getting squeezed) → market rising → LONG
                side = "SELL" if is_long_cascade else "BUY"
                mode = "momentum"
                confidence = min(confidence_raw * 0.9, 0.85)
            else:
                # REVERSION mode: fade the exhaustion
                # Long cascade overdone → price too low → BUY
                # Short cascade overdone → price too high → SELL
                side = "BUY" if is_long_cascade else "SELL"
                mode = "reversion"
                confidence = min(confidence_raw * 0.75, 0.75)

            signals.append(Signal(
                symbol=sym, market="futures", side=side,
                quantity=0.0, price=0,
                confidence=confidence,
                strategy=self.name,
                meta={"mode": mode, "liq_usd": dominant_notional},
            ))
            self._last_signal_bar[sym] = self._bar_count
            log.info(
                "LiqCascade %s %s [%s]  $%.0fk  conf=%.2f",
                side, sym, mode, dominant_notional / 1000, confidence,
            )

        return signals
