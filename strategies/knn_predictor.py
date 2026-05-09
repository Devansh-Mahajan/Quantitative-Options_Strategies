"""
K-Nearest Neighbor Return Predictor — Kakushadze & Serur §3.17.

Non-parametric ML strategy: given current market features, find the k most
similar historical market states and use their subsequent returns as the
prediction for the current period.

Mathematical formulation (§3.17):
  Features X_a(t):  normalized market indicators (moving averages, RSI, vol ratio)
  X̃_a(t) = (X_a(t) - X_min) / (X_max - X_min)         [min-max normalize to [0,1]]
  D(t, t')² = Σ_a (X̃_a(t) - X̃_a(t'))²                 [Euclidean distance]
  Ŷ(t) = (1/k) · Σ_{α=1}^{k} Y(t'_α(t))               [mean of k nearest returns]
  Y(t) = forward_return(t, T)                            [target: T-bar ahead return]

  Signal: Long if Ŷ(t) > threshold;  Short if Ŷ(t) < -threshold
  k ≈ √(training_window)

Feature set (computed from OHLCV candles):
  1. Price vs MA(5)   — short-term deviation
  2. Price vs MA(20)  — medium-term deviation
  3. Price vs MA(50)  — long-term deviation
  4. RSI(14)          — momentum oscillator
  5. Volume ratio     — volume vs 20-bar average
  6. ATR ratio        — current ATR vs 20-bar ATR (vol regime)
  7. Return(1 bar)    — very recent momentum
  8. Return(5 bars)   — intraday momentum
  9. BB position      — (close - lower_BB) / BB_width

This is non-parametric: no model training needed. New data points are compared
against all historical states in the in-sample window. Computationally cheap
at prediction time (O(N·F) distance computation).

Note: To prevent look-ahead, the feature history and target returns are built
exclusively from data available in the current store window.
"""

from __future__ import annotations
import logging

import numpy as np

from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.knn_predictor")

N_FEATURES      = 9
K_NEIGHBORS_PCT = 0.04    # k ≈ 4% of training window (≈ sqrt(N)/N at N=625)
TRAIN_WINDOW    = 500     # bars of history to build feature matrix from
PRED_HORIZON    = 4       # T: predict return T bars ahead
SIGNAL_THRESH   = 0.0015  # predicted return must exceed 0.15% to act
MIN_HISTORY     = 120     # minimum bars before any signal
REBAL_BARS      = 2       # only recompute every 2 bars (CPU efficiency)

_EPS = 1e-10


def _features(arr: np.ndarray) -> np.ndarray | None:
    """
    Compute feature vector for the latest bar in `arr` (OHLCV: shape (n, 5)).
    Returns 1D feature array of length N_FEATURES or None if insufficient data.
    """
    if len(arr) < 55:
        return None

    c = arr[:, 3].astype(float)     # close prices
    v = arr[:, 4].astype(float)     # volumes
    h = arr[:, 1].astype(float)
    l = arr[:, 2].astype(float)

    price = c[-1]

    # Moving averages
    ma5  = float(c[-5:].mean())
    ma20 = float(c[-20:].mean())
    ma50 = float(c[-50:].mean())

    # RSI(14) — simplified
    rets = np.diff(c[-16:])
    gains = rets[rets > 0].mean() if (rets > 0).any() else _EPS
    losses = -rets[rets < 0].mean() if (rets < 0).any() else _EPS
    rsi = 100 - 100 / (1 + gains / losses)

    # Volume ratio
    vol_ratio = float(v[-1] / (v[-20:].mean() + _EPS))

    # ATR ratio
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1))[-14:], abs(l - np.roll(c, 1))[-14:]))
    atr = float(np.mean(tr[:14]))
    atr_long = float(np.mean(tr[:14]))   # simplified: just ATR(14) vs ATR(20)
    atr_ratio = atr / (float(np.mean(tr)) + _EPS)

    # Short-term returns
    ret1 = float((c[-1] / (c[-2] + _EPS)) - 1)
    ret5 = float((c[-1] / (c[-6] + _EPS)) - 1)

    # Bollinger Band position
    bb_mean = float(c[-20:].mean())
    bb_std  = float(c[-20:].std() + _EPS)
    bb_pos  = (price - (bb_mean - 2 * bb_std)) / (4 * bb_std + _EPS)
    bb_pos  = float(np.clip(bb_pos, 0.0, 1.0))

    return np.array([
        (price - ma5)  / (ma5  + _EPS),
        (price - ma20) / (ma20 + _EPS),
        (price - ma50) / (ma50 + _EPS),
        rsi / 100.0,
        vol_ratio,
        atr_ratio,
        ret1,
        ret5,
        bb_pos,
    ], dtype=float)


class KNNPredictorStrategy(BaseStrategy):
    """
    K-nearest neighbor return prediction (Kakushadze §3.17).

    Builds a rolling feature matrix from historical bars, normalizes features,
    computes Euclidean distances from current state to all historical states,
    and predicts forward return as the weighted average of k-nearest outcomes.
    """

    name = "knn_predictor"
    market = "futures"

    def __init__(
        self,
        train_window: int = TRAIN_WINDOW,
        pred_horizon: int = PRED_HORIZON,
        signal_thresh: float = SIGNAL_THRESH,
        rebal_bars: int = REBAL_BARS,
    ) -> None:
        self._window     = train_window
        self._horizon    = pred_horizon
        self._thresh     = signal_thresh
        self._rebal_bars = rebal_bars
        self._bar_count  = 0

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        self._bar_count += 1
        if self._bar_count % self._rebal_bars != 0:
            return []

        regime_scale = {
            "bull":     1.0,
            "bear":     0.9,
            "ranging":  0.8,
            "volatile": 0.6,
        }.get(regime, 0.7)

        signals: list[Signal] = []
        symbols = [key[0] for key in store.candles if key[1] == "1h"]

        for sym in symbols:
            try:
                df = store.get_history_df(sym, "1h")
                if df is None or len(df) < MIN_HISTORY:
                    continue

                # Build OHLCV matrix
                cols = ["open", "high", "low", "close", "volume"]
                if not all(c in df.columns for c in cols):
                    continue
                mat = df[cols].values.astype(float)

                n = len(mat)
                if n < self._horizon + 60:
                    continue

                # Build feature matrix for all bars with enough history
                X_list: list[np.ndarray] = []
                Y_list: list[float] = []

                step = 3   # compute every 3 bars for speed
                for t in range(55, n - self._horizon, step):
                    feat = _features(mat[:t + 1])
                    if feat is None:
                        continue
                    fwd_ret = float(mat[t + self._horizon, 3] / (mat[t, 3] + _EPS) - 1)
                    X_list.append(feat)
                    Y_list.append(fwd_ret)

                if len(X_list) < 30:
                    continue

                X_train = np.array(X_list, dtype=float)
                Y_train = np.array(Y_list, dtype=float)

                # Current feature vector
                x_cur = _features(mat)
                if x_cur is None:
                    continue

                # Min-max normalize (Kakushadze §3.17 normalization)
                X_min = X_train.min(axis=0)
                X_max = X_train.max(axis=0)
                denom = X_max - X_min + _EPS
                X_norm = (X_train - X_min) / denom
                x_norm = np.clip((x_cur - X_min) / denom, 0, 1)

                # Euclidean distance to all training points
                dists = np.linalg.norm(X_norm - x_norm[None, :], axis=1)

                # k ≈ sqrt(N) (Kakushadze recommendation)
                k = max(5, int(np.sqrt(len(dists)) * K_NEIGHBORS_PCT * 25))
                k = min(k, len(dists) // 4)

                # k nearest neighbors
                knn_idx = np.argpartition(dists, k)[:k]
                y_hat   = float(Y_train[knn_idx].mean())

                if abs(y_hat) < self._thresh:
                    continue

                price = float(mat[-1, 3])
                side  = "BUY" if y_hat > 0 else "SELL"
                conf  = min(abs(y_hat) / (self._thresh * 5), 0.85) * regime_scale

                if conf < 0.25:
                    continue

                signals.append(Signal(
                    symbol=sym, market="futures", side=side,
                    quantity=0.0, price=price,
                    confidence=conf,
                    strategy=self.name,
                    meta={"y_hat": round(y_hat, 5), "k": k, "n_train": len(X_list)},
                ))
                log.debug("KNN %s %s  ŷ=%.4f  k=%d  conf=%.2f", side, sym, y_hat, k, conf)

            except Exception as exc:
                log.debug("KNN %s error: %s", sym, exc)

        return signals
