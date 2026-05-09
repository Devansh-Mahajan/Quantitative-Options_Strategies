"""
Hodrick-Prescott Filter Trend Following — Kakushadze & Serur §8.1.

The Hodrick-Prescott (HP) filter decomposes a price series into:
  - Trend component g(t): smooth long-run path through price
  - Cycle component c(t) = P(t) - g(t): short-term deviations

HP Filter minimization (§8.1):
  min_{g} Σ_{t=1}^{T} [P(t) - g(t)]² + λ · Σ_{t=2}^{T-1} [g(t+1) - 2g(t) + g(t-1)]²

  The λ parameter controls smoothness:
  λ = 100·n²  where n = frequency multiplier
    - Monthly data (n=1): λ = 100
    - Weekly (n=4):        λ = 1,600
    - Daily (n=22):        λ = 48,400  (≈ 1,600 × 52/5.2)
    - Hourly (n=130):      λ = 1,690,000

  The filter is the Tikhonov regularisation of the trend-cycle decomposition.
  In matrix form: g = (I + λ·D'D)^{-1} · P  where D is second-difference matrix.

Strategy:
  1. Apply HP filter to close prices → extract trend g(t)
  2. Apply fast MA(T') and slow MA(T) crossover ON g(t) (not raw price)
     This eliminates high-freq noise before crossover, reducing false signals
  3. Signal: Long if MA_fast(g) > MA_slow(g); Short if MA_fast(g) < MA_slow(g)

Key advantage over raw MA crossover:
  HP filter acts as low-pass filter, removing the cyclical noise that causes
  whipsaw (false crossovers) in volatile markets like crypto.
"""

from __future__ import annotations
import logging

import numpy as np

from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.hp_trend")

HP_LAMBDA    = 1_600_000   # λ for hourly data (n ≈ 130 hours/month; 100 × 130²)
MA_FAST      = 8            # fast MA on HP-filtered series
MA_SLOW      = 21           # slow MA on HP-filtered series
MIN_BARS     = 80           # minimum bars for HP filter to converge
REBAL_BARS   = 2            # recompute every 2 bars


def _hp_filter(series: np.ndarray, lam: float) -> np.ndarray:
    """
    Hodrick-Prescott filter implemented via sparse linear system.
    Returns the trend component g such that P = g + c.
    Uses the direct matrix formula: g = (I + λ·D'D)^{-1} · P
    """
    n = len(series)
    if n < 4:
        return series.copy()

    # Build second-difference matrix D (size n-2 × n)
    # Using scipy if available, else manual tridiagonal solve
    try:
        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve

        d2 = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n)).toarray()
        A  = np.eye(n) + lam * d2.T @ d2
        g  = np.linalg.solve(A, series)
    except ImportError:
        # Fallback: EMA-based approximation of HP trend
        alpha = 1.0 / (1.0 + lam ** 0.25)
        g = np.zeros(n)
        g[0] = series[0]
        for t in range(1, n):
            g[t] = alpha * g[t - 1] + (1 - alpha) * series[t]

    return g


class HPTrendStrategy(BaseStrategy):
    """
    HP-filtered moving average crossover (Kakushadze §8.1).

    Applies Hodrick-Prescott filter to remove cyclical noise, then runs
    a fast/slow MA crossover on the smooth trend component.
    Result: fewer false crossovers vs raw price, higher win rate in trending regimes.
    """

    name = "hp_trend"
    market = "futures"

    def __init__(
        self,
        lam: float = HP_LAMBDA,
        ma_fast: int = MA_FAST,
        ma_slow: int = MA_SLOW,
        min_bars: int = MIN_BARS,
        rebal_bars: int = REBAL_BARS,
    ) -> None:
        self._lam       = lam
        self._ma_fast   = ma_fast
        self._ma_slow   = ma_slow
        self._min_bars  = min_bars
        self._rebal     = rebal_bars
        self._bar_count = 0

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        self._bar_count += 1
        if self._bar_count % self._rebal != 0:
            return []

        # HP trend works best in sustained trending regimes
        regime_scale = {
            "bull":     1.0,
            "bear":     0.9,
            "volatile": 0.6,
            "ranging":  0.4,
        }.get(regime, 0.6)

        signals: list[Signal] = []
        symbols = [key[0] for key in store.candles if key[1] == "1h"]

        for sym in symbols:
            try:
                closes = store.get_closes(sym, "1h", self._min_bars + 10)
                if len(closes) < self._min_bars:
                    continue

                arr = np.array(closes, dtype=float)

                # Apply HP filter to extract trend
                trend = _hp_filter(arr, self._lam)

                if len(trend) < self._ma_slow + 2:
                    continue

                # MA crossover on HP trend (not raw price)
                g_fast_cur  = float(trend[-self._ma_fast:].mean())
                g_fast_prev = float(trend[-self._ma_fast - 1:-1].mean())
                g_slow_cur  = float(trend[-self._ma_slow:].mean())
                g_slow_prev = float(trend[-self._ma_slow - 1:-1].mean())

                cross_up = g_fast_cur > g_slow_cur and g_fast_prev <= g_slow_prev
                cross_dn = g_fast_cur < g_slow_cur and g_fast_prev >= g_slow_prev

                # Continuous signal: distance between fast and slow MAs
                ma_spread = (g_fast_cur - g_slow_cur) / (g_slow_cur + 1e-10)

                if not (cross_up or cross_dn) and abs(ma_spread) < 0.001:
                    continue

                price = float(arr[-1])
                conf  = min(abs(ma_spread) * 50, 0.88) * regime_scale

                if conf < 0.25:
                    continue

                if cross_up or ma_spread > 0.001:
                    signals.append(Signal(
                        symbol=sym, market="futures", side="BUY",
                        quantity=0.0, price=price, confidence=conf,
                        strategy=self.name,
                        meta={"hp_spread_pct": round(ma_spread * 100, 3), "cross": cross_up},
                    ))
                elif cross_dn or ma_spread < -0.001:
                    signals.append(Signal(
                        symbol=sym, market="futures", side="SELL",
                        quantity=0.0, price=price, confidence=conf,
                        strategy=self.name,
                        meta={"hp_spread_pct": round(ma_spread * 100, 3), "cross": cross_dn},
                    ))

            except Exception as exc:
                log.debug("HPTrend %s error: %s", sym, exc)

        return signals
