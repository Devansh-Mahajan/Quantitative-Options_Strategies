import unittest

import pandas as pd

from scripts.quant_research_foundry import (
    DEFAULT_MODEL_PARAMS,
    _sanitize_model_params,
    build_feature_matrix,
)


class QuantFoundryTests(unittest.TestCase):
    def test_build_feature_matrix_has_unique_index_and_binary_target(self):
        idx = pd.date_range("2024-01-01", periods=80, freq="B")
        close = pd.DataFrame(
            {
                "AAA": 100 + (pd.Series(range(80), index=idx) * 0.2),
                "BBB": 90 + (pd.Series(range(80), index=idx) * 0.15),
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


if __name__ == "__main__":
    unittest.main()
