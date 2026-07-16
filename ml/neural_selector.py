"""
NeuralSelector: wraps MegaStrategyNet for live macro strategy classification.

Outputs THETA/VEGA/BULL/BEAR probabilities once per trading cycle using the
feature matrix already computed for LSTM prediction (no extra data fetching).
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np

from bot.config import cfg

log = logging.getLogger("ml.neural_selector")

MACRO_NAMES = ["THETA", "VEGA", "BULL", "BEAR"]
_FALLBACK = {name: 0.25 for name in MACRO_NAMES}

_MACRO_STRATEGY_MAP = {
    "THETA": "THETA_ENGINE",
    "VEGA":  "VEGA_SNIPER",
    "BULL":  "BULL_TREND",
    "BEAR":  "BEAR_TREND",
}


class NeuralSelector:
    def __init__(self) -> None:
        self._model = None
        self._device = "cpu"
        self._last_probs: dict[str, float] = _FALLBACK.copy()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load(self, path: Path | str | None = None) -> "NeuralSelector":
        try:
            import torch
            from core.ml.mega_neural_brain import MegaStrategyNet
        except ImportError as exc:
            log.warning("NeuralSelector: torch/MegaStrategyNet unavailable: %s", exc)
            return self

        candidates = [
            Path(path) if path else None,
            cfg.model_dir / "mega_net.pt",
            Path("config/trading_model.pth"),
        ]
        model_path = next((p for p in candidates if p is not None and p.exists()), None)
        if model_path is None:
            log.info("NeuralSelector: no saved model found — using neutral priors")
            return self

        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                input_size = int(checkpoint.get("input_size", 66))
            else:
                state_dict = checkpoint
                # Infer input_size from first LSTM weight shape
                first_key = next(iter(state_dict))
                lstm_w = state_dict.get("lstm.weight_ih_l0")
                input_size = int(lstm_w.shape[1]) if lstm_w is not None else 66

            net = MegaStrategyNet(input_size=input_size)
            net.load_state_dict(state_dict)
            net.eval()
            self._device = cfg.device if _gpu_available() else "cpu"
            self._model = net.to(self._device)
            log.info("NeuralSelector loaded from %s (input_size=%d)", model_path, input_size)
        except Exception as exc:
            log.warning("NeuralSelector load failed: %s — using neutral priors", exc)
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict(self, feature_matrix: np.ndarray) -> dict[str, float]:
        """
        feature_matrix: (T, n_features) float32 — e.g. last 64 rows of feature matrix.
        Returns {THETA, VEGA, BULL, BEAR} probabilities summing to 1.
        """
        if self._model is None:
            return _FALLBACK.copy()

        try:
            import torch
            seq = np.asarray(feature_matrix, dtype=np.float32)
            # LSTM expects (batch=1, seq_len=T, input_size)
            tensor = torch.tensor(seq[np.newaxis], device=self._device)
            self._model.eval()
            with torch.no_grad():
                logits = self._model(tensor)  # (1, 4)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        except Exception as exc:
            log.debug("NeuralSelector inference error: %s", exc)
            return _FALLBACK.copy()

        self._last_probs = {name: float(p) for name, p in zip(MACRO_NAMES, probs)}
        return self._last_probs

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def macro_strategy(self) -> str:
        """THETA_ENGINE | VEGA_SNIPER | BULL_TREND | BEAR_TREND"""
        best = max(self._last_probs, key=lambda k: self._last_probs[k])
        return _MACRO_STRATEGY_MAP[best]

    @property
    def macro_confidence(self) -> float:
        return max(self._last_probs.values())

    @property
    def probs_array(self) -> np.ndarray:
        """[THETA, VEGA, BULL, BEAR] probabilities as float32 array."""
        return np.array([self._last_probs[k] for k in MACRO_NAMES], dtype=np.float32)


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def load_or_init_selector() -> NeuralSelector:
    return NeuralSelector().load()
