import unittest

import pandas as pd

from core.movement_predictor import _slice_features_to_lookback, lookback_days


class MovementPredictorLookbackTests(unittest.TestCase):
    def test_lookback_days_supports_new_windows(self):
        self.assertEqual(lookback_days("3y"), 1095)
        self.assertTrue(lookback_days("ytd") > 0)

    def test_slice_features_to_ytd_keeps_only_current_year(self):
        idx = pd.date_range("2025-10-01", "2026-04-10", freq="B")
        features = pd.DataFrame({"value": range(len(idx))}, index=idx)

        sliced = _slice_features_to_lookback(features, "ytd")

        self.assertFalse(sliced.empty)
        self.assertGreaterEqual(sliced.index.min(), pd.Timestamp("2026-01-01"))
        self.assertLessEqual(sliced.index.max(), pd.Timestamp("2026-12-31"))


if __name__ == "__main__":
    unittest.main()
