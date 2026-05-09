"""
Relative Moving Average (RMA) Strategy — Bloch (SSRN 4817751 + 5278107).

Bloch's Relative Moving Moments (RMM) framework measures the normalised
distance of price from a rolling statistical distribution.

Core quantity:
  MA_n(t)   = (1/n) · Σ P(t-s) for s in [0, n-1]    [n-bar simple moving average]
  σ_n(t)    = rolling standard deviation over n bars
  RMM_n(t)  = [P(t) - MA_n(t)] / σ_n(t)              [rolling z-score]

This is a theoretically grounded Bollinger-band z-score, but the RMA
framework extends it with:
  1. Adaptive threshold: threshold = k · σ_n (adapts to local vol)
  2. Multi-window blending: combine short and long RMM for regime detection
  3. Pairs extension: Spread = RMM_n(A) - RMM_n(B) → pairs on normalised deviations
  4. Risk-managed sizing: position ∝ signal / σ_n (inverse-vol, auto-deleverages)

Signal modes (selected by regime):
  Mean-reversion:  Long if RMM < -k  (price well below rolling mean)
                   Short if RMM > +k (price well above rolling mean)
  Trend-following: Long if RMM > 0 AND increasing (price in upper half, trending)
                   Short if RMM < 0 AND decreasing

References:
  Bloch (2024) "Dynamically Characterising Time Series With Relative Moving Moments"
  Bloch (2025) "A Course On Systematic Trading With RMA"
"""

from __future__ import annotations
import logging

import numpy as np

from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.rma_strategy")

SHORT_WINDOW   = 12     # fast RMM window (short-term distribution)
LONG_WINDOW    = 48     # slow RMM window (long-term distribution)
ENTRY_K        = 1.8    # RMM threshold for mean-reversion entry
EXIT_K         = 0.3    # exit when RMM returns inside this band
TREND_ENTRY_K  = 0.5    # RMM level to trigger trend-following entry
PAIRS = [
    ("BTCUSDT",  "ETHUSDT"),
    ("SOLUSDT",  "AVAXUSDT"),
]


def _rmm(closes: np.ndarray, window: int) -> float:
    """Compute Relative Moving Moment (rolling z-score) for latest bar."""
    if len(closes) < window + 1:
        return 0.0
    arr = closes[-window:]
    ma  = float(arr.mean())
    std = float(arr.std() + 1e-10)
    return (float(closes[-1]) - ma) / std


class RMAStrategy(BaseStrategy):
    """
    Bloch's Relative Moving Average strategy with regime-adaptive mode switching.

    In ranging regimes: trades mean-reversion (fade RMM extremes)
    In trending regimes: trades momentum (follow RMM direction)
    Pairs extension trades normalised deviation differential across correlated assets.
    Position sized inversely proportional to local volatility (risk-managed alpha).
    """

    name = "rma_strategy"
    market = "futures"

    def __init__(
        self,
        short_window: int = SHORT_WINDOW,
        long_window: int = LONG_WINDOW,
        entry_k: float = ENTRY_K,
        trend_entry_k: float = TREND_ENTRY_K,
    ) -> None:
        self._short_w  = short_window
        self._long_w   = long_window
        self._entry_k  = entry_k
        self._trend_k  = trend_entry_k
        self._positions: dict[str, str | None] = {}   # sym → "LONG"|"SHORT"|None

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        signals: list[Signal] = []
        symbols = [key[0] for key in store.candles if key[1] == "1h"]

        # ── Single-asset RMM signals ──────────────────────────────────────
        for sym in symbols:
            closes = store.get_closes(sym, "1h", self._long_w + 5)
            if len(closes) < self._long_w + 2:
                continue

            arr = np.array(closes, dtype=float)
            rmm_short = _rmm(arr, self._short_w)
            rmm_long  = _rmm(arr, self._long_w)

            # Local vol for risk-managed sizing
            local_vol  = float(np.diff(np.log(arr[-self._short_w:])).std() + 1e-10)
            vol_weight = min(0.02 / local_vol, 2.0)   # normalise: target 2% vol per unit

            pos = self._positions.get(sym)

            # ── Mean-Reversion mode (ranging regime) ──────────────────────
            if regime in ("ranging", "volatile"):
                # Exit existing position
                if pos == "LONG" and rmm_short > EXIT_K:
                    signals.append(Signal(sym, "futures", "SELL", 0.0, 0, 0.6, self.name,
                                          meta={"mode": "rmm_exit"}))
                    self._positions[sym] = None
                    continue
                if pos == "SHORT" and rmm_short < -EXIT_K:
                    signals.append(Signal(sym, "futures", "BUY", 0.0, 0, 0.6, self.name,
                                          meta={"mode": "rmm_exit"}))
                    self._positions[sym] = None
                    continue

                if pos is not None:
                    continue

                # Entry: extreme RMM on both short and long window (confirmation)
                if rmm_short <= -self._entry_k and rmm_long < -0.5:
                    conf = min(abs(rmm_short) / (self._entry_k * 2), 0.88) * vol_weight
                    conf = min(conf, 0.88)
                    if conf >= 0.25:
                        signals.append(Signal(sym, "futures", "BUY", 0.0, 0, conf, self.name,
                                              meta={"rmm_short": round(rmm_short, 3),
                                                    "mode": "mean_rev"}))
                        self._positions[sym] = "LONG"

                elif rmm_short >= self._entry_k and rmm_long > 0.5:
                    conf = min(abs(rmm_short) / (self._entry_k * 2), 0.88) * vol_weight
                    conf = min(conf, 0.88)
                    if conf >= 0.25:
                        signals.append(Signal(sym, "futures", "SELL", 0.0, 0, conf, self.name,
                                              meta={"rmm_short": round(rmm_short, 3),
                                                    "mode": "mean_rev"}))
                        self._positions[sym] = "SHORT"

            # ── Trend-Following mode (bull/bear regime) ───────────────────
            else:
                if pos is not None:
                    continue
                # Enter long when price is in upper half of distribution AND trending
                rmm_trending_up   = rmm_short > self._trend_k  and rmm_long > 0.1
                rmm_trending_down = rmm_short < -self._trend_k and rmm_long < -0.1

                if rmm_trending_up:
                    conf = min(abs(rmm_short) / (self._trend_k * 4), 0.80)
                    if conf >= 0.25:
                        signals.append(Signal(sym, "futures", "BUY", 0.0, 0, conf, self.name,
                                              meta={"rmm_short": round(rmm_short, 3),
                                                    "mode": "trend"}))
                        self._positions[sym] = "LONG"
                elif rmm_trending_down:
                    conf = min(abs(rmm_short) / (self._trend_k * 4), 0.80)
                    if conf >= 0.25:
                        signals.append(Signal(sym, "futures", "SELL", 0.0, 0, conf, self.name,
                                              meta={"rmm_short": round(rmm_short, 3),
                                                    "mode": "trend"}))
                        self._positions[sym] = "SHORT"

        # ── Pairs extension: trade RMM spread ────────────────────────────
        for sym_a, sym_b in PAIRS:
            closes_a = store.get_closes(sym_a, "1h", self._short_w + 5)
            closes_b = store.get_closes(sym_b, "1h", self._short_w + 5)
            if len(closes_a) < self._short_w + 2 or len(closes_b) < self._short_w + 2:
                continue

            rmm_a = _rmm(np.array(closes_a, dtype=float), self._short_w)
            rmm_b = _rmm(np.array(closes_b, dtype=float), self._short_w)
            spread = rmm_a - rmm_b   # positive → A rich vs B

            if spread > self._entry_k:
                conf = min(abs(spread) / (self._entry_k * 2), 0.80)
                if conf >= 0.30:
                    signals.append(Signal(sym_a, "futures", "SELL", 0.0, 0, conf, self.name,
                                          meta={"mode": "rma_pair", "spread": round(spread, 3)}))
                    signals.append(Signal(sym_b, "futures", "BUY",  0.0, 0, conf, self.name,
                                          meta={"mode": "rma_pair", "spread": round(spread, 3)}))
            elif spread < -self._entry_k:
                conf = min(abs(spread) / (self._entry_k * 2), 0.80)
                if conf >= 0.30:
                    signals.append(Signal(sym_a, "futures", "BUY",  0.0, 0, conf, self.name,
                                          meta={"mode": "rma_pair", "spread": round(spread, 3)}))
                    signals.append(Signal(sym_b, "futures", "SELL", 0.0, 0, conf, self.name,
                                          meta={"mode": "rma_pair", "spread": round(spread, 3)}))

        return signals
