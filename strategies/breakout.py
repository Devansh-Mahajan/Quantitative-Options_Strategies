"""
Volume-confirmed breakout strategy.
Donchian channel breakout on 4h, confirmed by volume surge and ATR expansion.
Best in bull and volatile regimes.
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from bot.config import cfg
from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategy.breakout")

DONCHIAN_PERIOD = 20
ATR_PERIOD = 14
VOL_SURGE = 1.5       # volume must be 1.5x 20-bar average
ATR_EXPAND = 1.2      # ATR must be expanding


class BreakoutStrategy(BaseStrategy):
    name = "breakout"
    market = "futures"
    required_regime = None


    def __init__(
        self,
        donchian_period: int = DONCHIAN_PERIOD,
        atr_period: int = ATR_PERIOD,
        vol_surge: float = VOL_SURGE,
        atr_expand: float = ATR_EXPAND,
    ) -> None:
        # Tunable thresholds (defaults = historical module constants).
        self._donchian_period = int(donchian_period)
        self._atr_period = int(atr_period)
        self._vol_surge = vol_surge
        self._atr_expand = atr_expand

    @property
    def symbols(self) -> list[str]:
        return cfg.runtime_symbols

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        signals = []
        regime_scale = {"bull": 1.0, "volatile": 0.8, "ranging": 0.3, "bear": 0.6}.get(regime, 0.7)

        for symbol in self.symbols:
            try:
                df = store.get_history_df(symbol, "4h")
                if len(df) < self._donchian_period + self._atr_period + 5:
                    continue

                c = df["close"].astype(float)
                h = df["high"].astype(float)
                l_ = df["low"].astype(float)
                v = df["volume"].astype(float)

                price = float(c.iloc[-1])
                prev_high = float(h.iloc[-(self._donchian_period + 1): -1].max())
                prev_low = float(l_.iloc[-(self._donchian_period + 1): -1].min())

                # ATR
                tr = pd.concat([h - l_, (h - c.shift()).abs(), (l_ - c.shift()).abs()], axis=1).max(axis=1)
                atr = tr.rolling(self._atr_period).mean()
                atr_curr = float(atr.iloc[-1])
                atr_prev = float(atr.iloc[-2])

                vol_ratio = float(v.iloc[-1]) / (float(v.rolling(20).mean().iloc[-1]) + 1e-10)
                vol_ok = vol_ratio >= self._vol_surge
                atr_ok = atr_curr >= atr_prev * self._atr_expand

                # ML prediction alignment
                pred = predictions.get(symbol, {})
                h4_ret = pred.get("h4_ret", 0.0)

                # Upside breakout
                if price > prev_high and vol_ok and atr_ok:
                    confidence = min(0.85, 0.55 + (price / prev_high - 1) * 10 + 0.15 * max(h4_ret, 0) * 5)
                    confidence *= regime_scale
                    signals.append(Signal(
                        symbol=symbol, market="futures", side="BUY",
                        quantity=0.0, price=price,
                        confidence=confidence,
                        strategy=self.name,
                        meta={"breakout_high": prev_high, "vol_ratio": vol_ratio, "atr": atr_curr},
                    ))

                # Downside breakout
                elif price < prev_low and vol_ok and atr_ok:
                    confidence = min(0.85, 0.55 + (prev_low / price - 1) * 10 + 0.15 * max(-h4_ret, 0) * 5)
                    confidence *= regime_scale
                    signals.append(Signal(
                        symbol=symbol, market="futures", side="SELL",
                        quantity=0.0, price=price,
                        confidence=confidence,
                        strategy=self.name,
                        meta={"breakout_low": prev_low, "vol_ratio": vol_ratio, "atr": atr_curr},
                    ))

            except Exception as exc:
                log.debug("Breakout %s error: %s", symbol, exc)

        log.debug("Breakout generated %d signals", len(signals))
        return signals
