"""
Momentum + Carry Minimum-Variance Combination — Kakushadze & Serur §8.4.

Elite funds combine uncorrelated alpha signals using portfolio optimisation
rather than simple averaging. The minimum-variance weighting (§8.4) finds
weights that minimise total signal variance while preserving expected return.

Mathematical formulation for two signals (§8.4):
  σ₁², σ₂² : variance of signal 1 and signal 2 returns
  ρ         : correlation between the two signals
  σ₁σ₂ = σ₁ · σ₂ · ρ    [covariance]

  Minimum-variance weights (closed form):
    w₁ = (σ₂² - σ₁σ₂) / (σ₁² + σ₂² - 2·σ₁σ₂)
    w₂ = (σ₁² - σ₁σ₂) / (σ₁² + σ₂² - 2·σ₁σ₂)

Extension to N signals (matrix form):
  Σ   = covariance matrix of signal returns
  μ   = expected signal returns
  w*  = Σ⁻¹ · μ / (1ᵀ · Σ⁻¹ · μ)          [max Sharpe weights]
  OR  = Σ⁻¹ · 1 / (1ᵀ · Σ⁻¹ · 1)          [min variance weights]

This module implements the N-signal version combining:
  1. Cross-sectional momentum score
  2. Funding carry score
  3. Time-series momentum (TSMOM)
  4. RMA mean-reversion score (inverse — used as timing filter)

Each sub-signal is pre-computed as a per-symbol score in [-1, +1].
The optimal weight vector is re-estimated every REBAL_BARS using
rolling covariance of recent signal returns.
"""

from __future__ import annotations
import logging
from collections import defaultdict, deque

import numpy as np

from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.momentum_carry_combo")

REBAL_BARS     = 8        # recombine signals every 8 bars
COV_WINDOW     = 60       # bars of signal history for covariance estimation
MIN_COV_OBS    = 20       # minimum observations before using optimised weights
ENTRY_SCORE    = 0.20     # minimum composite score magnitude to enter
MAX_SYMBOLS    = 8        # process this many symbols maximum (sorted by liquidity)

_SYMBOLS_ORDERED = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT",
    "ADAUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT",
]


def _rmm_score(closes: np.ndarray, window: int = 20) -> float:
    """Relative Moving Moment: (P - MA) / σ, range ≈ (-3, +3)."""
    if len(closes) < window + 1:
        return 0.0
    arr = closes[-window:]
    ma  = arr.mean()
    std = arr.std() + 1e-10
    return float((closes[-1] - ma) / std)


def _tsmom_score(closes: np.ndarray, window: int = 20) -> float:
    """Vol-scaled TSMOM signal for single asset, range ≈ (-1, +1)."""
    if len(closes) < window + 2:
        return 0.0
    arr = np.array(closes, dtype=float)
    R   = float(np.log(arr[-1] / arr[-(window + 1)]))
    sig = float(np.diff(np.log(arr[-window:])).std() + 1e-10)
    return float(np.tanh(R / (sig * np.sqrt(window))))


def _carry_score(funding_rate: float) -> float:
    """
    Carry signal: positive funding → short perp is attractive (negative score → SELL),
    negative funding → long perp is attractive (positive score → BUY).
    Score in (-1, +1).
    """
    return float(np.tanh(-funding_rate * 1000))


class MomentumCarryComboStrategy(BaseStrategy):
    """
    Minimum-variance combination of momentum + carry + mean-reversion signals.

    Combines sub-signals using rolling covariance-optimal weights:
    - When signals agree (low covariance): all get high weight
    - When signals conflict (high covariance): down-weight them

    This is how multi-strat funds allocate between individual alpha engines.
    """

    name = "momentum_carry_combo"
    market = "futures"

    def __init__(
        self,
        rebal_bars: int = REBAL_BARS,
        cov_window: int = COV_WINDOW,
        entry_score: float = ENTRY_SCORE,
    ) -> None:
        self._rebal    = rebal_bars
        self._cov_win  = cov_window
        self._thresh   = entry_score
        self._bar_count = 0
        # History of sub-signals for each symbol: {sym: deque of [s1,s2,s3]}
        self._signal_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=cov_window))

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        self._bar_count += 1

        # Always update signal history (every bar)
        symbols = _SYMBOLS_ORDERED[:MAX_SYMBOLS]
        for sym in symbols:
            closes = store.get_closes(sym, "1h", 30)
            if len(closes) < 22:
                continue

            arr = np.array(closes, dtype=float)
            s_tsmom  = _tsmom_score(arr, 20)
            s_rmm    = _rmm_score(arr, 20)
            s_carry  = _carry_score(float(store.funding.get(sym, 0.0)))

            self._signal_history[sym].append([s_tsmom, s_rmm, s_carry])

        # Rebalance (compute optimal weights and generate signals) every REBAL_BARS
        if self._bar_count % self._rebal != 0:
            return []

        regime_scale = {
            "bull":     1.0,
            "bear":     0.9,
            "ranging":  0.8,
            "volatile": 0.6,
        }.get(regime, 0.7)

        signals: list[Signal] = []

        for sym in symbols:
            hist = list(self._signal_history[sym])
            if len(hist) < MIN_COV_OBS:
                continue

            H = np.array(hist, dtype=float)   # (T, 3) — [tsmom, rmm, carry]

            # Latest sub-signals
            s_latest = H[-1]   # [s_tsmom, s_rmm, s_carry]

            if len(H) >= MIN_COV_OBS:
                # Compute rolling covariance matrix
                Sigma = np.cov(H.T, ddof=1)   # (3, 3)

                # Minimum variance weights: w = Σ⁻¹·1 / (1ᵀ·Σ⁻¹·1)
                try:
                    Sigma_inv = np.linalg.inv(Sigma + np.eye(3) * 1e-6)
                    ones      = np.ones(3)
                    w_raw     = Sigma_inv @ ones
                    weights   = w_raw / (ones @ w_raw + 1e-10)
                except np.linalg.LinAlgError:
                    weights = np.array([1/3, 1/3, 1/3])

                # Note: RMM is a mean-reversion signal: negate it for combo
                # (high RMM = price expensive = bearish = negative momentum)
                combo_weights = np.array([weights[0], -weights[1], weights[2]])
            else:
                # Equal weights before covariance estimation has enough data
                combo_weights = np.array([1/3, -1/3, 1/3])

            # Composite score for this symbol
            score = float(combo_weights @ s_latest)

            # Normalise by the std of combo scores in history to get z-score
            if len(H) >= 10:
                hist_scores = np.array([
                    float(combo_weights @ row) for row in H[-30:]
                ])
                score_z = score / (hist_scores.std() + 1e-10)
            else:
                score_z = score / 0.5

            if abs(score_z) < self._thresh * 5:
                continue

            side = "BUY" if score_z > 0 else "SELL"
            conf = min(abs(score_z) / 3.0, 0.88) * regime_scale

            if conf < 0.25:
                continue

            closes = store.get_closes(sym, "1h", 2)
            price  = float(closes[-1]) if closes else 0.0

            signals.append(Signal(
                symbol=sym, market="futures", side=side,
                quantity=0.0, price=price,
                confidence=conf,
                strategy=self.name,
                meta={
                    "score_z":  round(score_z, 3),
                    "s_tsmom":  round(float(s_latest[0]), 3),
                    "s_rmm":    round(float(s_latest[1]), 3),
                    "s_carry":  round(float(s_latest[2]), 3),
                    "w": [round(float(w), 3) for w in combo_weights],
                },
            ))
            log.debug("MomCarryCombo %s %s  z=%.3f  w=[%.2f,%.2f,%.2f]",
                      side, sym, score_z, *combo_weights)

        return signals
