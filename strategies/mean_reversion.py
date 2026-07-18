"""
Mean reversion strategy.
Signals: Bollinger Band extremes + RSI divergence + Keltner Channel.
Best in ranging regime.
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from bot.config import cfg
from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategy.mean_reversion")

BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
ATR_PERIOD = 14
KC_MULT = 1.5    # Keltner Channel multiplier


def _rsi(s: pd.Series, period: int) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-delta).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-10))


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"
    market = "futures"
    required_regime = "ranging"


    def __init__(
        self,
        bb_period: int = BB_PERIOD,
        bb_std: float = BB_STD,
        rsi_period: int = RSI_PERIOD,
        rsi_oversold: float = RSI_OVERSOLD,
        rsi_overbought: float = RSI_OVERBOUGHT,
        atr_period: int = ATR_PERIOD,
        kc_mult: float = KC_MULT,
    ) -> None:
        # Tunable thresholds (defaults = historical module constants).
        self._bb_period = int(bb_period)
        self._bb_std = bb_std
        self._rsi_period = int(rsi_period)
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought
        self._atr_period = int(atr_period)
        self._kc_mult = kc_mult

    @property
    def symbols(self) -> list[str]:
        return cfg.runtime_symbols[:6]

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        signals = []
        # Reduce size outside ranging; don't fully disable so RL can still use it
        regime_scale = {"ranging": 1.0, "bull": 0.5, "bear": 0.5, "volatile": 0.2}.get(regime, 0.5)

        for symbol in self.symbols:
            try:
                df = store.get_history_df(symbol, "1h")
                if len(df) < self._bb_period + self._atr_period + 5:
                    continue

                c = df["close"].astype(float)
                h = df["high"].astype(float)
                l_ = df["low"].astype(float)
                price = float(c.iloc[-1])

                # Bollinger Bands
                bb_mid = c.rolling(self._bb_period).mean()
                bb_std = c.rolling(self._bb_period).std()
                bb_upper = bb_mid + self._bb_std * bb_std
                bb_lower = bb_mid - self._bb_std * bb_std
                bb_pct = (c - bb_lower) / (bb_upper - bb_lower + 1e-10)

                # RSI
                rsi = _rsi(c, self._rsi_period)

                # ATR
                tr = pd.concat([h - l_, (h - c.shift()).abs(), (l_ - c.shift()).abs()], axis=1).max(axis=1)
                atr = tr.rolling(self._atr_period).mean()

                # Keltner Channel
                ema20 = c.ewm(span=20, adjust=False).mean()
                kc_upper = ema20 + self._kc_mult * atr
                kc_lower = ema20 - self._kc_mult * atr

                curr_rsi = float(rsi.iloc[-1])
                curr_bb = float(bb_pct.iloc[-1])

                # BB squeeze (both bands inside KC = low vol, prepare for breakout)
                in_squeeze = float(bb_upper.iloc[-1]) < float(kc_upper.iloc[-1]) and \
                             float(bb_lower.iloc[-1]) > float(kc_lower.iloc[-1])

                # Long: price at lower BB + RSI oversold
                if curr_bb < 0.10 and curr_rsi < self._rsi_oversold and not in_squeeze:
                    confidence = min(0.85, 0.50 + (self._rsi_oversold - curr_rsi) / 100 + (0.10 - curr_bb))
                    confidence *= regime_scale
                    signals.append(Signal(
                        symbol=symbol, market="futures", side="BUY",
                        quantity=0.0, price=price, confidence=confidence,
                        strategy=self.name,
                        required_regime="ranging",
                        meta={"bb_pct": curr_bb, "rsi": curr_rsi},
                    ))

                # Short: price at upper BB + RSI overbought
                elif curr_bb > 0.90 and curr_rsi > self._rsi_overbought and not in_squeeze:
                    confidence = min(0.85, 0.50 + (curr_rsi - self._rsi_overbought) / 100 + (curr_bb - 0.90))
                    confidence *= regime_scale
                    signals.append(Signal(
                        symbol=symbol, market="futures", side="SELL",
                        quantity=0.0, price=price, confidence=confidence,
                        strategy=self.name,
                        required_regime="ranging",
                        meta={"bb_pct": curr_bb, "rsi": curr_rsi},
                    ))

            except Exception as exc:
                log.debug("MeanReversion %s error: %s", symbol, exc)

        log.debug("MeanReversion generated %d signals", len(signals))
        return signals
