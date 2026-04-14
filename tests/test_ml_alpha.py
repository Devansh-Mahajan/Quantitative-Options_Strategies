import unittest

import numpy as np
import pandas as pd

from core.ml_alpha import _fit_live_ensemble, backtest_alpha_strategy, build_feature_frame, clean_feature_frame


class MLAlphaTests(unittest.TestCase):
    def _synthetic_daily_close(self) -> pd.DataFrame:
        periods = 2600
        index = pd.bdate_range("2016-01-01", periods=periods)
        trend = np.linspace(0.0, 0.45, periods)
        cyclical = 0.05 * np.sin(np.linspace(0.0, 40.0, periods))
        anti = 0.04 * np.cos(np.linspace(0.0, 32.0, periods))
        prices = pd.DataFrame(
            {
                "AAA": 100.0 * np.exp(trend + cyclical),
                "BBB": 95.0 * np.exp((trend * 0.85) - cyclical),
                "CCC": 90.0 * np.exp((trend * 0.65) + anti),
                "DDD": 105.0 * np.exp((trend * 0.40) - anti),
                "EEE": 98.0 * np.exp((trend * 0.55) + (0.03 * np.sin(np.linspace(0.0, 28.0, periods)))),
                "FFF": 102.0 * np.exp((trend * 0.30) - (0.03 * np.sin(np.linspace(0.0, 22.0, periods)))),
            },
            index=index,
        )
        return prices

    def test_live_ensemble_emits_ranked_alpha_signals(self):
        frame = clean_feature_frame(build_feature_frame(self._synthetic_daily_close()))

        signals = _fit_live_ensemble(frame, min_train_months=48)

        self.assertGreaterEqual(len(signals), 4)
        self.assertTrue(all(0.0 <= signal.percentile <= 1.0 for signal in signals))
        self.assertTrue(any(signal.direction == "up" for signal in signals))

    def test_backtest_alpha_strategy_returns_summary(self):
        prices = self._synthetic_daily_close()

        with unittest.mock.patch("core.ml_alpha.download_close_matrix", return_value=prices):
            result = backtest_alpha_strategy(prices.columns, min_train_months=48)

        self.assertNotIn("error", result)
        self.assertGreater(result["summary"]["months_tested"], 0)
        self.assertIn("long_short", result["summary"])


if __name__ == "__main__":
    unittest.main()
