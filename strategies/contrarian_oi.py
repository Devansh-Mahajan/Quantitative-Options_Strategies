"""
Futures Contrarian with Open Interest Filter — Kakushadze & Serur §10.3.

Academic insight: short-term momentum in futures REVERSES when:
  - Volume is high (real conviction in the move — not just noise)
  - Open Interest is declining (losing side is closing out, move exhausted)

Conversely, a move that is increasing volume AND OI is more likely to continue.

The two-filter contrarian (§10.3.1):
  u_i = ln(OI_i(t) / OI_i(t-1))     [OI change; negative = OI declining]
  v_i = ln(Vol_i(t) / Vol_i(t-1))   [volume change; positive = volume rising]

  Trade contrarian ONLY when: v_i in top half (high volume) AND u_i in bottom half (OI falling)
  This combination means: strong participation in move, but longs/shorts capitulating = reversal likely

Signal:
  R_m = (1/N) * Σ R_i           [market return]
  epsilon_i = R_i - R_m         [demeaned return (alpha vs market)]
  w_i = -gamma * epsilon_i      [long losers vs market; short winners vs market]
  Entry only when OI filter and volume filter pass.
"""

from __future__ import annotations
import logging
from collections import defaultdict, deque

import numpy as np

from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.contrarian_oi")

RETURN_BARS     = 4       # bars to measure recent return for contrarian signal
REBAL_BARS      = 4       # rebalance frequency
ENTRY_THRESHOLD = 0.15    # minimum |epsilon| z-score to generate signal
VOL_RATIO_MIN   = 1.1     # volume must be > this × prev bar to confirm
OI_RATIO_MAX    = 0.99    # OI must be FALLING (ratio < 1) to confirm exhaustion


class ContrarianOIStrategy(BaseStrategy):
    """
    Cross-sectional contrarian mean-reversion gated by OI + volume filter.

    Only enters when a move has high volume (conviction) but declining open interest
    (the move's participants are closing rather than adding — exhaustion signal).
    """

    name = "contrarian_oi"
    market = "futures"

    def __init__(
        self,
        return_bars: int = RETURN_BARS,
        rebal_bars: int = REBAL_BARS,
        entry_threshold: float = ENTRY_THRESHOLD,
    ) -> None:
        self._ret_bars = return_bars
        self._rebal_bars = rebal_bars
        self._threshold = entry_threshold
        self._bar_count = 0
        # Rolling OI levels per symbol so we can compute a change RATIO —
        # store.open_interest only exposes the current level.
        self._oi_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=8))

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        self._bar_count += 1
        if self._bar_count % self._rebal_bars != 0:
            return []

        # Contrarian works best when there's a trend to fade
        regime_scale = {
            "ranging":  1.0,
            "volatile": 0.9,
            "bull":     0.7,
            "bear":     0.7,
        }.get(regime, 0.6)

        symbols = [key[0] for key in store.candles if key[1] == "1h"]
        returns:  dict[str, float] = {}
        vol_chgs: dict[str, float] = {}
        prices:   dict[str, float] = {}
        oi_chgs:  dict[str, float] = {}

        need = self._ret_bars + 3
        for sym in symbols:
            closes  = store.get_closes(sym, "1h", need)
            volumes = store.get_volumes(sym, "1h", need)
            if len(closes) < need or len(volumes) < need:
                continue

            arr = np.array(closes, dtype=float)
            vol = np.array(volumes, dtype=float)

            # Recent return
            R_i = float(np.log(arr[-1] / arr[-(self._ret_bars + 1)]))
            returns[sym] = R_i

            # Volume change (ln ratio)
            v_i = float(np.log(vol[-1] / (vol[-2] + 1e-10)))
            vol_chgs[sym] = v_i

            # OI change ratio: current level vs rolling mean of prior levels.
            # (store.open_interest is a raw LEVEL — comparing it directly to a
            # ratio threshold was a long-standing bug that disabled this filter.)
            oi_raw = float(store.open_interest.get(sym, 0.0))
            history = self._oi_history[sym]
            if oi_raw > 0 and len(history) > 0:
                oi_chgs[sym] = oi_raw / (float(np.mean(history)) + 1e-10)
            else:
                oi_chgs[sym] = 0.0  # no usable OI data -> filter passes open
            if oi_raw > 0:
                history.append(oi_raw)
            prices[sym] = float(arr[-1])

        if len(returns) < 3:
            return []

        # ── Market return (equal-weight benchmark) ────────────────────────
        R_vals = np.array(list(returns.values()))
        R_m = float(R_vals.mean())

        # Median volume change (for top-half filter)
        vol_median = float(np.median(list(vol_chgs.values()))) if vol_chgs else 0.0

        # ── Kakushadze §10.3 weight and filter ───────────────────────────
        signals: list[Signal] = []
        for sym in returns:
            epsilon_i = returns[sym] - R_m

            # Filter 1: volume must be in top half (v_i > median)
            vol_ok = vol_chgs.get(sym, 0.0) >= vol_median

            # Filter 2: OI ratio must show decline (current below recent mean),
            # confirming exhaustion. Passes open when no OI data (ratio 0.0).
            oi_ratio = oi_chgs.get(sym, 0.0)
            oi_ok = (oi_ratio == 0.0) or (oi_ratio < OI_RATIO_MAX)

            if not (vol_ok and oi_ok):
                continue

            if abs(epsilon_i) < self._threshold:
                continue

            # Contrarian: fade losers relative to market
            side = "BUY" if epsilon_i < 0 else "SELL"
            # Confidence: strength of deviation from market
            confidence = min(abs(epsilon_i) / (self._threshold * 3), 0.85) * regime_scale

            if confidence < 0.25:
                continue

            signals.append(Signal(
                symbol=sym, market="futures", side=side,
                quantity=0.0, price=prices.get(sym, 0),
                confidence=confidence,
                strategy=self.name,
                meta={
                    "epsilon": round(epsilon_i, 4),
                    "R_m": round(R_m, 4),
                    "vol_filter": vol_ok,
                },
            ))
            log.debug("Contrarian %s %s  ε=%.4f  R_m=%.4f", side, sym, epsilon_i, R_m)

        return signals
