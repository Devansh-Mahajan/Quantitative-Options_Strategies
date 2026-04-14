import unittest

import numpy as np
import pandas as pd

from scripts.quant_research_foundry import (
    DEFAULT_MODEL_PARAMS,
    _sanitize_model_params,
    build_feature_matrix,
    fit_models,
)


class QuantFoundryTests(unittest.TestCase):
    def test_build_feature_matrix_has_unique_index_and_binary_target(self):
        idx = pd.date_range("2024-01-01", periods=80, freq="B")
        close = pd.DataFrame(
            {
                "AAA": 100 + np.linspace(0.0, 4.0, 80) + np.sin(np.linspace(0.0, 12.0, 80)) * 1.8,
                "BBB": 90 + np.linspace(0.0, 3.0, 80) - np.cos(np.linspace(0.0, 10.0, 80)) * 1.5,
            },
            index=idx,
        )

        X, y = build_feature_matrix(close, horizon_days=5)

        self.assertFalse(X.empty)
        self.assertTrue(X.index.is_unique)
        self.assertEqual(X.index.nlevels, 2)
        self.assertSetEqual(set(y.unique()), {0, 1})

    def test_sanitize_model_params_merges_defaults(self):
        merged = _sanitize_model_params({"rf": {"n_estimators": 150}})

        self.assertEqual(merged["rf"]["n_estimators"], 150)
        self.assertEqual(merged["logreg"]["solver"], DEFAULT_MODEL_PARAMS["logreg"]["solver"])
        self.assertEqual(merged["mlp"]["max_iter"], DEFAULT_MODEL_PARAMS["mlp"]["max_iter"])

    def test_fit_models_accepts_symbol_in_multiindex(self):
        idx = pd.date_range("2024-01-01", periods=260, freq="B")
        close = pd.DataFrame(
            {
                "AAA": 100 + (pd.Series(range(260), index=idx) * 0.15),
                "BBB": 90 + (pd.Series(range(260), index=idx) * 0.12),
                "CCC": 80 + (pd.Series(range(260), index=idx) * 0.09),
            },
            index=idx,
        )

        X, y = build_feature_matrix(close, horizon_days=5)
        metrics = fit_models(X, y, model_params=DEFAULT_MODEL_PARAMS, model_parallelism=1)

        self.assertIn("ensemble_accuracy", metrics)
        self.assertIn("logreg", metrics["models"])
        self.assertGreater(metrics["models"]["logreg"]["n_test"], 0)
        self.assertIn("candidate_leaderboard", metrics)
        self.assertGreaterEqual(metrics["candidate_count"], 1)
        self.assertIn("champion_score", metrics)
        self.assertEqual(metrics["selection_method"], "walk_forward_champion_challenger")


if __name__ == "__main__":
    unittest.main()
