"""
Quantitative Factor Bundle — Kakushadze & Serur §4.4, §3.4, §9.5.

Three academically-validated signals bundled into one strategy:

1. Internal Bar Strength (IBS) Mean Reversion — §4.4
   IBS = (Close - Low) / (High - Low)
   Range: 0 (closed at low) → 1 (closed at high)
   Signal: Mean-revert. Buy low IBS (capitulation close), Sell high IBS (exhaustion close).
   Evidence: ETF daily IBS predicts next-day reversal with statistical significance.

2. Low-Volatility Anomaly — §3.4
   Realized volatility over 30-90 bars → long low-vol, short high-vol.
   Paradox: low-vol assets outperform high-vol on risk-adjusted basis due to
   investor preference for lottery-like payoffs (overpaying for volatile assets).
   Evidence: robust across equity, FX, and crypto markets.

3. Skewness Premium — §9.5
   S_i = (1/(σ³·T)) · Σ(R_i(t) - R̄_i)³
   Signal: Short high-skew (expensive lottery tickets), Long low-skew.
   Evidence: retail investors overpay for positively skewed assets (hope of big upside).

All three are cross-sectional: rank symbols, long bottom decile, short top decile.
Signals are averaged for each symbol and filtered by minimum threshold.
"""

from __future__ import annotations
import logging

import numpy as np

from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.quant_factors")

IBS_LOW        = 0.20    # buy if IBS ≤ this (closed near session low)
IBS_HIGH       = 0.80    # sell if IBS ≥ this (closed near session high)
VOL_LOOKBACK   = 48      # bars for realized vol ranking (§3.4)
SKEW_LOOKBACK  = 30      # bars for skewness calculation (§9.5)
LONG_N         = 2       # top N for each cross-sectional ranking
SHORT_N        = 2
REBAL_BARS     = 6       # cross-sectional factors rebalance every 6 bars


class QuantFactorsStrategy(BaseStrategy):
    """
    Three-factor quantitative strategy: IBS reversal + low-vol anomaly + skewness premium.

    IBS fires single-asset signals on every bar when at extremes.
    Low-vol and skewness are cross-sectional, rebalancing every REBAL_BARS.
    """

    name = "quant_factors"
    market = "futures"

    def __init__(
        self,
        ibs_low: float = IBS_LOW,
        ibs_high: float = IBS_HIGH,
        vol_lookback: int = VOL_LOOKBACK,
        skew_lookback: int = SKEW_LOOKBACK,
    ) -> None:
        self._ibs_low = ibs_low
        self._ibs_high = ibs_high
        self._vol_lb = vol_lookback
        self._skew_lb = skew_lookback
        self._bar_count = 0

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        self._bar_count += 1
        signals: list[Signal] = []

        regime_scale = {
            "ranging":  1.0,
            "bull":     0.7,
            "bear":     0.7,
            "volatile": 0.5,
        }.get(regime, 0.6)

        symbols = [key[0] for key in store.candles if key[1] == "1h"]

        # ── FACTOR 1: IBS Mean Reversion (single-asset, fires every bar) ──
        for sym in symbols:
            candle = store.candles.get((sym, "1h"))
            if candle is None:
                continue
            h, l, c = candle.high, candle.low, candle.close
            if h <= l:
                continue

            # Kakushadze §4.4: IBS = (C - L) / (H - L)
            ibs = (c - l) / (h - l + 1e-10)

            if ibs <= self._ibs_low:
                # Closed near low → likely oversold → BUY
                confidence = (self._ibs_low - ibs) / self._ibs_low * 0.80 * regime_scale
                if confidence >= 0.25:
                    signals.append(Signal(sym, "futures", "BUY", 0.0, 0,
                                          confidence, self.name,
                                          meta={"factor": "ibs", "ibs": round(ibs, 3)}))

            elif ibs >= self._ibs_high:
                # Closed near high → likely overbought → SELL
                confidence = (ibs - self._ibs_high) / (1 - self._ibs_high) * 0.80 * regime_scale
                if confidence >= 0.25:
                    signals.append(Signal(sym, "futures", "SELL", 0.0, 0,
                                          confidence, self.name,
                                          meta={"factor": "ibs", "ibs": round(ibs, 3)}))

        # ── FACTOR 2 & 3: Cross-Sectional (rebalance every REBAL_BARS) ───
        if self._bar_count % REBAL_BARS != 0:
            return signals

        vols:  dict[str, float] = {}
        skews: dict[str, float] = {}
        prices: dict[str, float] = {}

        need = max(self._vol_lb, self._skew_lb) + 2
        for sym in symbols:
            closes = store.get_closes(sym, "1h", need)
            if len(closes) < need:
                continue
            arr = np.array(closes, dtype=float)
            rets = np.diff(np.log(arr))

            # Vol (annualised)
            vol = float(rets[-self._vol_lb:].std() * np.sqrt(365 * 24))
            vols[sym] = vol

            # Skewness §9.5: S_i = E[(R - R̄)³] / σ³
            r_sub = rets[-self._skew_lb:]
            mu = r_sub.mean()
            sigma = r_sub.std() + 1e-10
            skew = float(((r_sub - mu) ** 3).mean() / sigma ** 3)
            skews[sym] = skew

            prices[sym] = float(arr[-1])

        # ── Low-Vol Anomaly §3.4: Long low-vol, Short high-vol ────────────
        if len(vols) >= LONG_N + SHORT_N + 1:
            ranked_vol = sorted(vols.items(), key=lambda kv: kv[1])
            for i, (sym, vol) in enumerate(ranked_vol):
                if i < LONG_N:
                    conf = (1.0 - vol / (max(vols.values()) + 1e-10)) * 0.70 * regime_scale
                    if conf >= 0.25:
                        signals.append(Signal(sym, "futures", "BUY", 0.0, prices.get(sym, 0),
                                              conf, self.name,
                                              meta={"factor": "low_vol", "vol": round(vol, 4)}))
                elif i >= len(ranked_vol) - SHORT_N:
                    conf = (vol / (max(vols.values()) + 1e-10)) * 0.70 * regime_scale
                    if conf >= 0.25:
                        signals.append(Signal(sym, "futures", "SELL", 0.0, prices.get(sym, 0),
                                              conf, self.name,
                                              meta={"factor": "high_vol", "vol": round(vol, 4)}))

        # ── Skewness Premium §9.5: Long low-skew, Short high-skew ─────────
        if len(skews) >= LONG_N + SHORT_N + 1:
            ranked_skew = sorted(skews.items(), key=lambda kv: kv[1])
            skew_range = max(skews.values()) - min(skews.values()) + 1e-10
            for i, (sym, skew) in enumerate(ranked_skew):
                if i < LONG_N:
                    # Low skew = cheap → BUY
                    conf = (1.0 - (skew - min(skews.values())) / skew_range) * 0.65 * regime_scale
                    if conf >= 0.25:
                        signals.append(Signal(sym, "futures", "BUY", 0.0, prices.get(sym, 0),
                                              conf, self.name,
                                              meta={"factor": "skewness", "skew": round(skew, 3)}))
                elif i >= len(ranked_skew) - SHORT_N:
                    # High skew = expensive lottery → SELL
                    conf = ((skew - min(skews.values())) / skew_range) * 0.65 * regime_scale
                    if conf >= 0.25:
                        signals.append(Signal(sym, "futures", "SELL", 0.0, prices.get(sym, 0),
                                              conf, self.name,
                                              meta={"factor": "skewness", "skew": round(skew, 3)}))

        return signals
