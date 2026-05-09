"""
Cross-Sectional Momentum (Factor Momentum).

How top quant funds generate alpha in crypto:
  Rather than asking "will BTC go up?", ask "which assets will outperform
  each other in the next period?" This is cross-sectional (relative) alpha,
  not time-series (absolute) alpha.

Algorithm:
  1. Every REBAL_BARS bars, score all symbols using a composite momentum signal
  2. Rank them (highest score = strongest outperformer)
  3. Long the top N, short the bottom N with position sizes proportional to rank
  4. Hold until next rebalance, applying stop-loss / take-profit as usual

Momentum score is a weighted blend of:
  - 1-bar (short) return    — very recent price pressure
  - 5-bar return            — intraday momentum
  - 20-bar return           — trend component (most weight)
  - Vol-adjusted return (return / volatility) — quality-adjusted signal

Why this works in crypto:
  - Crypto has persistent short-horizon momentum (unlike equities)
  - Funding rates reinforce trending: high funding → strong buyers
  - Cross-sectional removes market beta, making the signal market-neutral

References:
  Moskowitz, Ooi, Pedersen (2012) — Time Series Momentum
  Asness, Moskowitz, Pedersen (2013) — Value and Momentum Everywhere
"""

from __future__ import annotations
import logging

import numpy as np

from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.cross_sectional_momentum")

REBAL_BARS  = 4      # rebalance every 4 bars (4h at 1h interval)
LOOKBACKS   = [1, 5, 20]
WEIGHTS     = [0.2, 0.3, 0.5]   # recent < intraday < trend
LONG_N      = 2      # number of symbols to go long
SHORT_N     = 2      # number of symbols to go short
MIN_VOL_PCT = 0.001  # ignore symbols with <0.1% vol (stale data)


class CrossSectionalMomentumStrategy(BaseStrategy):
    """
    Long/short factor momentum: buy the strongest symbols, sell the weakest.

    Works best in trending regimes (bull/bear) where relative momentum
    persists, and avoids ranging/volatile where reversals dominate.
    """

    name = "cross_sectional_momentum"
    market = "futures"

    def __init__(
        self,
        rebal_bars: int = REBAL_BARS,
        long_n: int = LONG_N,
        short_n: int = SHORT_N,
    ) -> None:
        self._rebal_bars = rebal_bars
        self._long_n = long_n
        self._short_n = short_n
        self._bar_count = 0
        self._last_signals: list[Signal] = []

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        self._bar_count += 1

        # Only rebalance every REBAL_BARS bars
        if self._bar_count % self._rebal_bars != 0:
            return []

        regime_scale = {
            "bull":     1.0,
            "bear":     0.9,    # short leg works well in bear
            "ranging":  0.4,    # cross-sec still works but weaker
            "volatile": 0.3,    # high reversals, reduce size
        }.get(regime, 0.5)

        if regime_scale < 0.3:
            return []

        # ── Score each symbol ─────────────────────────────────────────────
        scores: dict[str, float] = {}
        symbols = [key[0] for key in store.candles if key[1] == "1h"]

        if len(symbols) < self._long_n + self._short_n + 1:
            return []

        for sym in symbols:
            closes = store.get_closes(sym, "1h", max(LOOKBACKS) + 5)
            if len(closes) < max(LOOKBACKS) + 2:
                continue
            arr = np.array(closes, dtype=float)

            # Composite momentum score
            mom_score = 0.0
            for lb, w in zip(LOOKBACKS, WEIGHTS):
                if lb >= len(arr):
                    continue
                ret = (arr[-1] - arr[-lb - 1]) / (arr[-lb - 1] + 1e-10)
                mom_score += w * ret

            # Vol-adjust (quality filter): penalise noisy signals
            recent_vols = np.diff(np.log(arr[-21:])) if len(arr) >= 22 else np.array([0.01])
            vol = float(recent_vols.std() + 1e-10)
            if vol < MIN_VOL_PCT:
                continue

            vol_adj_score = mom_score / vol
            scores[sym] = vol_adj_score

        if len(scores) < self._long_n + self._short_n + 1:
            return []

        # ── Rank and select ───────────────────────────────────────────────
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        longs  = ranked[:self._long_n]
        shorts = ranked[-self._short_n:]

        n = len(ranked)
        signals: list[Signal] = []

        for rank, (sym, score) in enumerate(ranked):
            is_long  = sym in dict(longs)
            is_short = sym in dict(shorts)
            if not is_long and not is_short:
                continue

            # Position size proportional to rank extremity
            rank_score = abs(rank - (n - 1) / 2) / (n / 2)   # 0..1, highest at extremes
            confidence = min(rank_score * regime_scale * min(abs(score) / 0.05, 1.0), 0.95)

            if confidence < 0.25:
                continue

            side = "BUY" if is_long else "SELL"
            signals.append(Signal(
                symbol=sym, market="futures", side=side,
                quantity=0.001, price=0,
                confidence=confidence,
                strategy=self.name,
            ))
            log.debug("XSMom %s %s  score=%.4f  conf=%.2f", side, sym, score, confidence)

        return signals
