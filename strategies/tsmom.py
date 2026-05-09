"""
Time-Series Momentum (TSMOM) — Kakushadze & Serur §10.4.

The academically validated momentum strategy (Moskowitz, Ooi, Pedersen 2012).
Unlike EMA crossovers that trade on relative position of two averages, TSMOM
trades the sign and magnitude of the actual return over a lookback, scaled by
realised volatility to ensure equal risk contribution per signal.

Mathematical formulation:
  R_i(t)     = cumulative return of asset i over T bars
  kappa(t)   = cross-sectional std of {R_i(t)} across all N assets
  eta_i(t)   = tanh(R_i / kappa)              [smoothed direction ∈ (-1,+1)]
  sigma_i(t) = annualised realised vol of asset i over T bars
  w_i(t)     = eta_i / sigma_i                [inverse-vol scaled weight]
  w_i_tilde  = w_i - (1/N)*sum_j w_j          [dollar-neutral demean]

The tanh(·) nonlinearity:
  - Linear in the middle (small returns → proportional signal)
  - Saturates at ±1 (extreme returns capped, regime protection)
  - Smoother than sign(·) function used in simpler TSMOM variants

References:
  Moskowitz, Ooi, Pedersen (2012) — "Time Series Momentum"
  Kakushadze & Serur (2018) §10.4 — "Futures Trend Following / Momentum"
"""

from __future__ import annotations
import logging

import numpy as np

from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.tsmom")

LOOKBACK_BARS  = 20      # T: momentum formation period (20 × 1h = ~20h)
LONG_LOOKBACK  = 60      # secondary long-term lookback for signal confirmation
VOL_WINDOW     = 20      # window for realised volatility estimate
ENTRY_THRESHOLD = 0.10   # |w_tilde| must exceed this to generate signal
REBAL_BARS     = 4       # rebalance every 4 bars (don't churn every bar)


class TSMomentumStrategy(BaseStrategy):
    """
    Volatility-scaled time-series momentum.

    Uses tanh of cumulative return divided by cross-sectional return dispersion
    as signal direction, then scales by inverse realised vol for risk parity.
    Dollar-neutralised across all symbols so net exposure is minimal.
    """

    name = "tsmom"
    market = "futures"

    def __init__(
        self,
        lookback: int = LOOKBACK_BARS,
        vol_window: int = VOL_WINDOW,
        entry_threshold: float = ENTRY_THRESHOLD,
        rebal_bars: int = REBAL_BARS,
    ) -> None:
        self._lookback = lookback
        self._vol_window = vol_window
        self._threshold = entry_threshold
        self._rebal_bars = rebal_bars
        self._bar_count = 0

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        self._bar_count += 1
        if self._bar_count % self._rebal_bars != 0:
            return []

        regime_scale = {
            "bull":     1.0,
            "bear":     0.9,
            "ranging":  0.4,
            "volatile": 0.5,
        }.get(regime, 0.5)

        # ── Collect returns and vols for all symbols ──────────────────────
        symbols = [key[0] for key in store.candles if key[1] == "1h"]
        returns:  dict[str, float] = {}
        vols:     dict[str, float] = {}
        prices:   dict[str, float] = {}

        for sym in symbols:
            closes = store.get_closes(sym, "1h", max(self._lookback, self._vol_window) + 5)
            if len(closes) < self._lookback + 2:
                continue
            arr = np.array(closes, dtype=float)

            # Cumulative log return over lookback
            R_i = np.log(arr[-1] / arr[-(self._lookback + 1)])
            returns[sym] = float(R_i)

            # Realised volatility (annualised, bar-freq)
            log_rets = np.diff(np.log(arr[-self._vol_window:]))
            sigma_i = float(log_rets.std() * np.sqrt(365 * 24) + 1e-10)
            vols[sym] = sigma_i
            prices[sym] = float(arr[-1])

        if len(returns) < 3:
            return []

        # ── Kakushadze §10.4 weight computation ──────────────────────────
        R_vals = np.array(list(returns.values()))
        kappa  = float(R_vals.std() + 1e-10)    # cross-sectional dispersion

        raw_weights: dict[str, float] = {}
        for sym in returns:
            eta_i = np.tanh(returns[sym] / kappa)
            w_i   = eta_i / vols[sym]
            raw_weights[sym] = float(w_i)

        # Dollar-neutral demean
        w_mean = np.mean(list(raw_weights.values()))
        weights_neutral = {sym: w - w_mean for sym, w in raw_weights.items()}

        # ── Generate signals ──────────────────────────────────────────────
        signals: list[Signal] = []
        for sym, w_tilde in weights_neutral.items():
            if abs(w_tilde) < self._threshold:
                continue

            # Confidence proportional to signal strength (capped at 0.90)
            confidence = min(abs(w_tilde) / (self._threshold * 3), 0.90) * regime_scale
            if confidence < 0.25:
                continue

            side = "BUY" if w_tilde > 0 else "SELL"
            signals.append(Signal(
                symbol=sym, market="futures", side=side,
                quantity=0.0, price=prices.get(sym, 0),
                confidence=confidence,
                strategy=self.name,
                meta={
                    "w_tilde": round(w_tilde, 4),
                    "R_i": round(returns[sym], 4),
                    "vol": round(vols[sym], 4),
                },
            ))
            log.debug("TSMOM %s %s  w=%.3f  R=%.3f  σ=%.3f", side, sym, w_tilde, returns[sym], vols[sym])

        return signals
