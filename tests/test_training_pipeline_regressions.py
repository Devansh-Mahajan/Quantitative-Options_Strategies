import unittest

import numpy as np
import pandas as pd
import torch

from bot.notifications import _normalize_webhook_url
from ml.features import FEATURE_NAMES, build_sequences, compute_feature_matrix
from ml.price_lstm import PricePredictorTrainer
from ml.regime_hmm import RegimeHMM
from scripts.weekend_training import build_training_data


def _sample_ohlcv(rows: int = 320, include_taker: bool = True) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=rows, freq="h")
    base = np.linspace(100.0, 130.0, rows) + np.sin(np.linspace(0.0, 20.0, rows))
    close = pd.Series(base, index=idx)
    frame = pd.DataFrame(
        {
            "open": close * 0.998,
            "high": close * 1.003,
            "low": close * 0.997,
            "close": close,
            "volume": np.linspace(1_000.0, 2_000.0, rows),
        },
        index=idx,
    )
    if include_taker:
        frame["taker_buy_base"] = frame["volume"] * 0.53
    return frame


class TrainingPipelineRegressionTests(unittest.TestCase):
    def test_feature_matrix_has_stable_schema_with_or_without_taker_volume(self):
        with_taker = compute_feature_matrix(_sample_ohlcv(include_taker=True))
        without_taker = compute_feature_matrix(_sample_ohlcv(include_taker=False))

        self.assertEqual(with_taker.shape[1], len(FEATURE_NAMES))
        self.assertEqual(without_taker.shape[1], len(FEATURE_NAMES))
        self.assertEqual(with_taker.shape[1], without_taker.shape[1])

    def test_weekend_training_data_returns_aligned_features_and_targets(self):
        X, y = build_training_data(_sample_ohlcv())

        self.assertEqual(len(X), len(y))
        self.assertEqual(X.shape[1], len(FEATURE_NAMES))
        self.assertEqual(y.shape[1], 3)

    def test_hmm_observation_builder_is_pandas_compatible(self):
        obs = RegimeHMM._build_obs(_sample_ohlcv())

        self.assertEqual(obs.shape[1], 3)
        self.assertFalse(np.isnan(obs).any())

    def test_lstm_forward_pass_accepts_current_feature_count(self):
        X, y = build_training_data(_sample_ohlcv())
        Xs, _ = build_sequences(X[:140], y[:140], lookback=50)
        trainer = PricePredictorTrainer(len(FEATURE_NAMES))

        batch = torch.tensor(Xs[:4], dtype=torch.float32, device=trainer.device)
        preds = trainer.model(batch)

        self.assertEqual(tuple(preds.shape), (4, 3))

    def test_discord_webhook_placeholder_is_treated_as_empty(self):
        self.assertEqual(_normalize_webhook_url("              # https://discord.com/api/webhooks/..."), "")
        self.assertEqual(_normalize_webhook_url("discord.com/api/webhooks/123/abc"), "https://discord.com/api/webhooks/123/abc")


if __name__ == "__main__":
    unittest.main()
