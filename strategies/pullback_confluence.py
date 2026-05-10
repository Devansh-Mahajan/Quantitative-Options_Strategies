"""
Pullback + confluence trend strategy inspired by Bloch's futuretesting work.

The core idea is to avoid chasing raw breakouts. Instead we require:
  - a persistent trend
  - a controlled pullback toward the fast mean / breakout zone
  - a resumption candle confirming the trend
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bot.config import cfg
from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.pullback_confluence")


class PullbackConfluenceStrategy(BaseStrategy):
    name = "pullback_confluence"
    market = "futures"

    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 55,
        breakout_lookback: int = 20,
        pullback_atr: float = 1.0,
        trend_floor: float = 0.003,
    ) -> None:
        self._fast_window = max(5, int(fast_window))
        self._slow_window = max(self._fast_window + 5, int(slow_window))
        self._breakout_lookback = max(10, int(breakout_lookback))
        self._pullback_atr = float(pullback_atr)
        self._trend_floor = float(trend_floor)

    @property
    def symbols(self) -> list[str]:
        return list(cfg.model_symbols[:8])

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        signals: list[Signal] = []
        regime_scale = {
            "bull": 1.0,
            "bear": 1.0,
            "ranging": 0.65,
            "volatile": 0.8,
        }.get(regime, 0.8)
        min_bars = max(self._slow_window, self._breakout_lookback) + 8

        for symbol in self.symbols:
            try:
                df = store.get_history_df(symbol, "1h")
                if df is None or len(df) < min_bars:
                    continue

                frame = df.copy()
                close = frame["close"].astype(float)
                high = frame["high"].astype(float)
                low = frame["low"].astype(float)

                ema_fast = close.ewm(span=self._fast_window, adjust=False).mean()
                ema_slow = close.ewm(span=self._slow_window, adjust=False).mean()
                prev_close = close.shift(1)
                tr = pd.concat(
                    [
                        high - low,
                        (high - prev_close).abs(),
                        (low - prev_close).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                atr = tr.rolling(14).mean().bfill()
                atr_now = float(atr.iloc[-1] or 0.0)
                if atr_now <= 0:
                    continue

                breakout_high = float(high.iloc[-self._breakout_lookback - 1:-1].max())
                breakout_low = float(low.iloc[-self._breakout_lookback - 1:-1].min())
                slope = (float(ema_slow.iloc[-1]) / max(float(ema_slow.iloc[-5]), 1e-9)) - 1.0

                trend_up = float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1]) and slope > self._trend_floor
                trend_down = float(ema_fast.iloc[-1]) < float(ema_slow.iloc[-1]) and slope < -self._trend_floor

                prev_low = float(low.iloc[-2])
                prev_high = float(high.iloc[-2])
                prev_close_val = float(close.iloc[-2])
                last_close = float(close.iloc[-1])

                pullback_depth_long = (breakout_high - prev_close_val) / atr_now
                pullback_depth_short = (prev_close_val - breakout_low) / atr_now

                touched_fast_long = prev_low <= float(ema_fast.iloc[-2]) + self._pullback_atr * atr_now
                touched_fast_short = prev_high >= float(ema_fast.iloc[-2]) - self._pullback_atr * atr_now
                resume_long = last_close > max(float(ema_fast.iloc[-1]), prev_high)
                resume_short = last_close < min(float(ema_fast.iloc[-1]), prev_low)

                if trend_up and touched_fast_long and 0.35 <= pullback_depth_long <= 3.25 and resume_long:
                    trend_strength = min(abs(slope) / max(self._trend_floor, 1e-6), 3.0)
                    confidence = min(0.86, (0.28 + trend_strength * 0.12 + min(pullback_depth_long, 2.0) * 0.10) * regime_scale)
                    if confidence >= 0.25:
                        signals.append(
                            Signal(
                                symbol=symbol,
                                market="futures",
                                side="BUY",
                                quantity=0.0,
                                price=last_close,
                                confidence=confidence,
                                strategy=self.name,
                                meta={
                                    "setup": "pullback_resume_long",
                                    "atr": round(atr_now, 4),
                                    "slope": round(float(slope), 4),
                                    "pullback_atr": round(float(pullback_depth_long), 3),
                                    "breakout_level": round(float(breakout_high), 4),
                                },
                            )
                        )

                elif trend_down and touched_fast_short and 0.35 <= pullback_depth_short <= 3.25 and resume_short:
                    trend_strength = min(abs(slope) / max(self._trend_floor, 1e-6), 3.0)
                    confidence = min(0.86, (0.28 + trend_strength * 0.12 + min(pullback_depth_short, 2.0) * 0.10) * regime_scale)
                    if confidence >= 0.25:
                        signals.append(
                            Signal(
                                symbol=symbol,
                                market="futures",
                                side="SELL",
                                quantity=0.0,
                                price=last_close,
                                confidence=confidence,
                                strategy=self.name,
                                meta={
                                    "setup": "pullback_resume_short",
                                    "atr": round(atr_now, 4),
                                    "slope": round(float(slope), 4),
                                    "pullback_atr": round(float(pullback_depth_short), 3),
                                    "breakout_level": round(float(breakout_low), 4),
                                },
                            )
                        )
            except Exception as exc:
                log.debug("Pullback confluence %s error: %s", symbol, exc)

        return signals
