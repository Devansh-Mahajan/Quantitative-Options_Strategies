"""
Kalman Filter Statistical Arbitrage.

Replaces the naive fixed-window z-score pairs approach with a proper
Kalman filter that estimates a time-varying hedge ratio online.

Why this is better than z-score pairs_arb:
  - Hedge ratio updates every bar (market structure changes over time)
  - Innovation sequence (residuals) is the theoretically correct spread to trade
  - No lookback window to tune; Q and R are the only hyper-parameters
  - Handles regime changes automatically via the recursive Bayesian update

For each pair (A, B) the model is:
  price_A(t) = β(t) * price_B(t) + α(t) + ε(t)
  State θ = [β, α] follows a random walk: θ(t) = θ(t-1) + η(t)

Entry: |innovation / sqrt(S)| > entry_z  →  fade the spread
Exit : |innovation / sqrt(S)| < exit_z   →  close spread
"""

from __future__ import annotations
import logging
from collections import defaultdict

import numpy as np

from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.statistical_arb")

# ── Pairs to trade ────────────────────────────────────────────────────────────
# Add more pairs from your symbol universe; each must be in the data feed
PAIRS: list[tuple[str, str]] = [
    ("BTCUSDT",  "ETHUSDT"),
    ("SOLUSDT",  "AVAXUSDT"),
    ("ADAUSDT",  "XRPUSDT"),
    ("DOGEUSDT", "LINKUSDT"),
]

ENTRY_Z   = 2.2    # z-score to enter
EXIT_Z    = 0.4    # z-score to exit
WARMUP    = 60     # bars before trading (Kalman needs to converge)
Q_VAR     = 5e-5   # process noise — higher = faster adaptation
R_VAR     = 1e-2   # measurement noise


# ── Kalman filter (one per pair) ─────────────────────────────────────────────

class _KalmanHedge:
    """Recursive Kalman filter estimating dynamic hedge ratio for a pair."""

    def __init__(self, q_var: float = Q_VAR, r_var: float = R_VAR) -> None:
        self._theta = np.array([1.0, 0.0])       # [beta, alpha]
        self._P = np.eye(2) * 10.0               # large init uncertainty
        self._Q = np.diag([q_var, q_var])        # process noise
        self._R = float(r_var)                   # measurement noise
        self._n = 0

    def update(self, price_a: float, price_b: float) -> tuple[float, float]:
        """
        Returns (normalised_innovation, sqrt_innovation_variance).
        Innovation = price_a - (beta * price_b + alpha)
        """
        H = np.array([price_b, 1.0])

        # Predict
        P_pred = self._P + self._Q

        # Innovation
        innov = price_a - H @ self._theta
        S = float(H @ P_pred @ H) + self._R

        # Kalman gain
        K = P_pred @ H / S

        # Update
        self._theta = self._theta + K * innov
        self._P = (np.eye(2) - np.outer(K, H)) @ P_pred

        self._n += 1
        return float(innov), float(np.sqrt(S))

    @property
    def beta(self) -> float:
        return float(self._theta[0])

    @property
    def alpha(self) -> float:
        return float(self._theta[1])

    @property
    def n_updates(self) -> int:
        return self._n


# ── Strategy ─────────────────────────────────────────────────────────────────

class StatisticalArbStrategy(BaseStrategy):
    """
    Kalman-filter pairs arbitrage on cointegrated crypto pairs.

    For each pair (A, B) we maintain a running Kalman estimate of the
    dynamic hedge ratio β and intercept α. We trade the innovation
    (normalised residual) when it exceeds ±entry_z standard deviations.

    Long the spread  = BUY A + SELL B (when spread is too low)
    Short the spread = SELL A + BUY B (when spread is too high)
    """

    name = "statistical_arb"
    market = "futures"

    def __init__(
        self,
        pairs: list[tuple[str, str]] = PAIRS,
        entry_z: float = ENTRY_Z,
        exit_z: float = EXIT_Z,
        warmup: int = WARMUP,
    ) -> None:
        self._pairs = pairs
        self._entry_z = entry_z
        self._exit_z = exit_z
        self._warmup = warmup
        # One Kalman filter per pair
        self._filters: dict[tuple[str, str], _KalmanHedge] = {
            pair: _KalmanHedge() for pair in pairs
        }
        # Track open spreads: True = long spread (A-B), False = short spread
        self._positions: dict[tuple[str, str], bool | None] = defaultdict(lambda: None)

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        signals: list[Signal] = []

        regime_scale = {
            "ranging":  1.0,
            "bull":     0.7,
            "bear":     0.7,
            "volatile": 0.4,
        }.get(regime, 0.6)

        for sym_a, sym_b in self._pairs:
            kf = self._filters[(sym_a, sym_b)]
            pos = self._positions[(sym_a, sym_b)]

            # Pull latest prices
            ca = store.candles.get((sym_a, "1h"))
            cb = store.candles.get((sym_b, "1h"))
            if ca is None or cb is None:
                continue

            price_a = ca.close
            price_b = cb.close

            innov, innov_std = kf.update(price_a, price_b)

            if kf.n_updates < self._warmup:
                continue

            # Normalised z-score
            z = innov / (innov_std + 1e-10)
            beta = kf.beta

            # Confidence decays when β drifts far from 1 (unreliable pair)
            beta_penalty = max(1.0 - abs(beta - 1.0) * 0.5, 0.3)
            confidence = min(abs(z) / (self._entry_z + 2.0), 1.0) * regime_scale * beta_penalty

            # Exit open positions first
            if pos is not None:
                if abs(z) < self._exit_z:
                    # Close A leg
                    exit_side_a = "SELL" if pos else "BUY"
                    exit_side_b = "BUY"  if pos else "SELL"
                    qty = 0.001   # absolute minimum; engine will resize
                    signals.append(Signal(sym_a, "futures", exit_side_a, qty, 0, 0.5,
                                          self.name))
                    signals.append(Signal(sym_b, "futures", exit_side_b, qty, 0, 0.5,
                                          self.name))
                    self._positions[(sym_a, sym_b)] = None
                continue   # don't double-enter while managing a position

            if confidence < 0.3:
                continue

            # Entry: spread too high → short A, long B
            if z > self._entry_z:
                signals.append(Signal(sym_a, "futures", "SELL", 0.001, 0, confidence,
                                      self.name))
                signals.append(Signal(sym_b, "futures", "BUY",  0.001, 0, confidence,
                                      self.name))
                self._positions[(sym_a, sym_b)] = False   # short spread
                log.debug("StatArb SELL %s / BUY %s  z=%.2f  β=%.3f", sym_a, sym_b, z, beta)

            # Entry: spread too low → long A, short B
            elif z < -self._entry_z:
                signals.append(Signal(sym_a, "futures", "BUY",  0.001, 0, confidence,
                                      self.name))
                signals.append(Signal(sym_b, "futures", "SELL", 0.001, 0, confidence,
                                      self.name))
                self._positions[(sym_a, sym_b)] = True    # long spread
                log.debug("StatArb BUY %s / SELL %s  z=%.2f  β=%.3f", sym_a, sym_b, z, beta)

        return signals
