"""
Momentum / trend-following strategy.
Signals: EMA crossover (fast/slow) + MACD histogram + volume confirmation.
Best in bull and bear regimes.
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from bot.config import cfg
from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategy.momentum")

EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 55
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOL_CONFIRM_RATIO = 1.2   # volume must be ≥ this multiple of 20-bar average


class MomentumStrategy(BaseStrategy):
    name = "momentum"
    market = "futures"
    required_regime = None  # active in all regimes (scaled down in ranging/volatile)

    @property
    def symbols(self) -> list[str]:
        return cfg.futures_symbols

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        signals = []
        regime_scale = {"bull": 1.0, "bear": 0.8, "ranging": 0.4, "volatile": 0.3}.get(regime, 0.5)
        if regime_scale < 0.3:
            return signals

        for symbol in self.symbols:
            try:
                df = store.get_history_df(symbol, "1h")
                if len(df) < EMA_TREND + 10:
                    continue

                c = df["close"].astype(float)
                v = df["volume"].astype(float)

                ema_fast = c.ewm(span=EMA_FAST, adjust=False).mean()
                ema_slow = c.ewm(span=EMA_SLOW, adjust=False).mean()
                ema_trend = c.ewm(span=EMA_TREND, adjust=False).mean()

                # MACD
                macd_line = c.ewm(span=MACD_FAST).mean() - c.ewm(span=MACD_SLOW).mean()
                macd_sig = macd_line.ewm(span=MACD_SIGNAL).mean()
                hist = macd_line - macd_sig

                # Volume
                vol_ratio = v.iloc[-1] / (v.rolling(20).mean().iloc[-1] + 1e-10)

                price = float(c.iloc[-1])
                cross_up = float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1]) and \
                           float(ema_fast.iloc[-2]) <= float(ema_slow.iloc[-2])
                cross_dn = float(ema_fast.iloc[-1]) < float(ema_slow.iloc[-1]) and \
                           float(ema_fast.iloc[-2]) >= float(ema_slow.iloc[-2])

                above_trend = price > float(ema_trend.iloc[-1])
                macd_bullish = float(hist.iloc[-1]) > 0 and float(hist.iloc[-2]) <= 0
                macd_bearish = float(hist.iloc[-1]) < 0 and float(hist.iloc[-2]) >= 0
                vol_ok = vol_ratio >= VOL_CONFIRM_RATIO

                # ML prediction boost
                pred = predictions.get(symbol, {})
                h1_ret = pred.get("h1_ret", 0.0)

                if (cross_up or macd_bullish) and above_trend and vol_ok:
                    confidence = min(0.9, 0.55 + 0.2 * vol_ratio + 0.15 * max(h1_ret, 0) * 10)
                    confidence *= regime_scale
                    signals.append(Signal(
                        symbol=symbol, market="futures", side="BUY",
                        quantity=0.0,  # sized by orchestrator
                        price=price,
                        confidence=confidence,
                        strategy=self.name,
                        meta={"ema_cross": "bullish", "macd": float(hist.iloc[-1])},
                    ))

                elif (cross_dn or macd_bearish) and not above_trend and vol_ok:
                    confidence = min(0.9, 0.55 + 0.2 * vol_ratio + 0.15 * max(-h1_ret, 0) * 10)
                    confidence *= regime_scale
                    signals.append(Signal(
                        symbol=symbol, market="futures", side="SELL",
                        quantity=0.0, price=price,
                        confidence=confidence,
                        strategy=self.name,
                        meta={"ema_cross": "bearish", "macd": float(hist.iloc[-1])},
                    ))

            except Exception as exc:
                log.debug("MomentumStrategy %s error: %s", symbol, exc)

        log.debug("Momentum generated %d signals (regime=%s)", len(signals), regime)
        return signals
