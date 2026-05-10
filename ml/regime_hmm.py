"""4-state Gaussian HMM for market regime detection: bull, bear, ranging, volatile."""

from __future__ import annotations
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from bot.config import cfg

log = logging.getLogger("ml.regime_hmm")

REGIME_LABELS = {0: "bull", 1: "bear", 2: "ranging", 3: "volatile"}


class RegimeHMM:
    """
    Trains a GaussianHMM on [log_return, realized_vol, volume_ratio] features.
    Assigns regime labels by sorting states on mean return.
    """

    def __init__(self, n_states: int = 4) -> None:
        self.n_states = n_states
        self.model: hmm.GaussianHMM | None = None
        self.scaler = StandardScaler()
        self._state_map: dict[int, str] = {}

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame) -> None:
        """
        df must have: close, volume (OHLCV style).
        Uses 3 features: log_ret, realised_vol (20-bar), volume_ratio.
        """
        X = self._build_obs(df)
        X_scaled = self.scaler.fit_transform(X)

        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=200,
            tol=1e-4,
            random_state=42,
        )
        self.model.fit(X_scaled)
        self._assign_labels()
        log.info("RegimeHMM trained — states: %s", self._state_map)

    def _assign_labels(self) -> None:
        """Sort states by mean log_return. Highest = bull, lowest = bear, etc."""
        assert self.model is not None
        mean_rets = self.model.means_[:, 0]  # first feature is log_ret
        vol_feature = self.model.means_[:, 1]  # second is realised vol
        order = np.argsort(mean_rets)
        # lowest mean_ret => bear, highest => bull
        # middle two: ranging (lower vol) vs volatile (higher vol)
        mid = order[1:3]
        mid_sorted = mid[np.argsort(vol_feature[mid])]
        self._state_map = {
            int(order[0]): "bear",
            int(mid_sorted[0]): "ranging",
            int(mid_sorted[1]): "volatile",
            int(order[-1]): "bull",
        }

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict_regime(self, df: pd.DataFrame) -> str:
        """Return regime string for the latest bar in df."""
        if self.model is None:
            return "unknown"
        X = self._build_obs(df)
        X_scaled = self.scaler.transform(X)
        state = self.model.predict(X_scaled)[-1]
        return self._state_map.get(int(state), "unknown")

    def predict_proba(self, df: pd.DataFrame) -> dict[str, float]:
        """Return probability over each regime for the latest bar."""
        if self.model is None:
            return {r: 0.25 for r in REGIME_LABELS.values()}
        X = self._build_obs(df)
        X_scaled = self.scaler.transform(X)
        log_proba = self.model.predict_proba(X_scaled)[-1]
        proba = np.exp(log_proba - log_proba.max())
        proba /= proba.sum()
        return {self._state_map.get(i, f"s{i}"): float(proba[i]) for i in range(self.n_states)}

    def predict_sequence(self, df: pd.DataFrame) -> list[str]:
        """Regime label for every bar in df."""
        if self.model is None:
            return ["unknown"] * len(df)
        X = self._build_obs(df)
        X_scaled = self.scaler.transform(X)
        states = self.model.predict(X_scaled)
        return [self._state_map.get(int(s), "unknown") for s in states]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler, "state_map": self._state_map}, f)
        log.info("RegimeHMM saved to %s", path)

    def load(self, path: Path | str) -> "RegimeHMM":
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self._state_map = data["state_map"]
        log.info("RegimeHMM loaded from %s — states: %s", path, self._state_map)
        return self

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_obs(df: pd.DataFrame) -> np.ndarray:
        c = df["close"].astype(float)
        v = df["volume"].astype(float)
        log_ret = np.log(c / c.shift(1)).fillna(0)
        realised_vol = log_ret.rolling(20).std().bfill().fillna(0.0)
        vol_ma = v.rolling(20).mean().bfill().fillna(v)
        vol_ratio = (v / (vol_ma + 1e-10)).fillna(1.0)
        X = np.column_stack([log_ret.values, realised_vol.values, vol_ratio.values])
        return X.astype(np.float32)


def load_or_init_hmm(symbol: str = "BTCUSDT") -> RegimeHMM:
    path = cfg.model_dir / f"hmm_{symbol}.pkl"
    hmm_model = RegimeHMM(n_states=cfg.hmm_states)
    if path.exists():
        try:
            hmm_model.load(path)
        except Exception as exc:
            log.warning("Could not load HMM from %s: %s — using untrained model", path, exc)
    return hmm_model
