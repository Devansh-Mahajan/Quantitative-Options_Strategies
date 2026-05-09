"""
Pivot Point Support & Resistance — Kakushadze & Serur §3.14.

Classic institutional support/resistance framework used by market makers
and institutional desks to identify price level clusters.

Formulas (§3.14):
  C = (H + L + C_prev) / 3        [pivot center = typical price]
  R1 = 2·C - L                    [first resistance level]
  S1 = 2·C - H                    [first support level]
  R2 = C + (H - L)                [second resistance]
  S2 = C - (H - L)                [second support]
  R3 = H + 2·(C - L)              [third resistance]
  S3 = L - 2·(H - C)              [third support]

Signal logic:
  Long  if price bounces off S1/S2 (support holding = momentum entry)
  Short if price bounces off R1/R2 (resistance holding = reversal entry)

  Key insight: USE the pivot as an entry point, not the level itself.
  Enter when price tests the level and shows a reversal bar pattern
  (IBS confirmation: low IBS at support, high IBS at resistance).

Extension: daily/4h pivots used on 1h chart for multi-timeframe confluence.
"""

from __future__ import annotations
import logging

import numpy as np

from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.pivot_sr")

PROXIMITY_PCT  = 0.003   # price within 0.3% of pivot level triggers check
IBS_CONFIRM    = 0.30    # IBS must be < this to confirm support bounce (buy)
                          # or > (1-this) to confirm resistance rejection (sell)
LOOKBACK_BARS  = 24      # bars to compute high/low for daily-level pivots


class PivotSRStrategy(BaseStrategy):
    """
    Classic pivot point support & resistance system (Kakushadze §3.14).

    Computes pivot levels from the prior LOOKBACK_BARS high/low/close.
    Signals when price is near a pivot level AND the current bar confirms
    the expected direction via Internal Bar Strength (IBS).
    """

    name = "pivot_sr"
    market = "futures"

    def __init__(
        self,
        proximity_pct: float = PROXIMITY_PCT,
        ibs_confirm: float = IBS_CONFIRM,
        lookback: int = LOOKBACK_BARS,
    ) -> None:
        self._prox = proximity_pct
        self._ibs = ibs_confirm
        self._lookback = lookback

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        signals: list[Signal] = []

        # Pivot levels are most useful in ranging and volatile regimes
        regime_scale = {
            "ranging":  1.0,
            "volatile": 0.9,
            "bull":     0.6,    # support is tested more → trend continuation
            "bear":     0.6,
        }.get(regime, 0.6)

        symbols = [key[0] for key in store.candles if key[1] == "1h"]

        for sym in symbols:
            try:
                df = store.get_history_df(sym, "1h")
                if df is None or len(df) < self._lookback + 2:
                    continue

                closes = df["close"].astype(float).values
                highs  = df["high"].astype(float).values
                lows   = df["low"].astype(float).values

                # Pivot computed from prior lookback (no look-ahead)
                H_prev = float(highs[-(self._lookback + 1):-1].max())
                L_prev = float(lows [-(self._lookback + 1):-1].min())
                C_prev = float(closes[-(self._lookback + 2)])    # close 2 bars ago

                # Kakushadze §3.14 pivot levels
                C_piv = (H_prev + L_prev + C_prev) / 3
                R1 = 2 * C_piv - L_prev
                S1 = 2 * C_piv - H_prev
                R2 = C_piv + (H_prev - L_prev)
                S2 = C_piv - (H_prev - L_prev)

                # Current bar OHLC
                candle = store.candles.get((sym, "1h"))
                if candle is None:
                    continue
                price = candle.close
                bar_h = candle.high
                bar_l = candle.low

                # IBS for current bar
                ibs = (candle.close - bar_l) / (bar_h - bar_l + 1e-10)

                # Check proximity to each level
                def near(level: float) -> bool:
                    return abs(price - level) / (level + 1e-10) <= self._prox

                signal_generated = False
                for level, level_name in [(S1, "S1"), (S2, "S2")]:
                    if near(level) and ibs < self._ibs:
                        # Price at support with low IBS (closed near low) → bullish bounce
                        conf = (self._ibs - ibs) / self._ibs * 0.85 * regime_scale
                        if conf >= 0.25 and not signal_generated:
                            signals.append(Signal(
                                symbol=sym, market="futures", side="BUY",
                                quantity=0.0, price=price,
                                confidence=conf,
                                strategy=self.name,
                                meta={"level": level_name, "pivot": round(level, 2), "ibs": round(ibs, 3)},
                            ))
                            signal_generated = True
                            log.debug("Pivot BUY %s @ %s=%.2f  ibs=%.3f", sym, level_name, level, ibs)

                for level, level_name in [(R1, "R1"), (R2, "R2")]:
                    if near(level) and ibs > (1.0 - self._ibs):
                        # Price at resistance with high IBS (closed near high) → bearish rejection
                        conf = (ibs - (1.0 - self._ibs)) / self._ibs * 0.85 * regime_scale
                        if conf >= 0.25 and not signal_generated:
                            signals.append(Signal(
                                symbol=sym, market="futures", side="SELL",
                                quantity=0.0, price=price,
                                confidence=conf,
                                strategy=self.name,
                                meta={"level": level_name, "pivot": round(level, 2), "ibs": round(ibs, 3)},
                            ))
                            signal_generated = True
                            log.debug("Pivot SELL %s @ %s=%.2f  ibs=%.3f", sym, level_name, level, ibs)

            except Exception as exc:
                log.debug("PivotSR %s error: %s", sym, exc)

        return signals
