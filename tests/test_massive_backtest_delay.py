import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.massive_backtest_engine import (
    _build_institutional_robustness,
    _download_intraday_close,
    _run_delay_quote_suite_from_close,
    _run_option_model_suite_from_close,
)


class MassiveBacktestDelaySuiteTests(unittest.TestCase):
    def test_institutional_robustness_builds_deployment_summary(self):
        payload = {
            "massive_overview": {
                "predictive_score": 0.61,
                "avg_recent_quote_error_pct": 0.03,
                "option_model_ensemble_win_rate": 0.56,
                "ml_alpha_information_coefficient": 0.02,
            },
            "movement_suite": {"summary": {"by_lookback": {"10y": {"avg_accuracy": 0.55}, "5y": {"avg_accuracy": 0.54}}}},
            "pairs_suite": {"results_by_lookback": {"10y": {"summary": {"win_rate": 0.53}}, "5y": {"summary": {"win_rate": 0.52}}}},
            "regime_suite": {"summary": {"by_lookback": {"10y": {"directional_accuracy_proxy": 0.56}, "5y": {"directional_accuracy_proxy": 0.55}}}},
            "strategy_proxy_suite": {"results_by_lookback": {"10y": {"summary": {"overall_hit_rate": 0.54}}, "5y": {"summary": {"overall_hit_rate": 0.52}}}},
        }

        robustness = _build_institutional_robustness(payload)

        self.assertIn("institutional_score", robustness)
        self.assertIn("deployment_tier", robustness)
        self.assertIn("movement", robustness["windows"])

    def test_intraday_download_resolves_yahoo_aliases(self):
        index = pd.date_range("2026-01-02 14:30:00+00:00", periods=4, freq="15min")
        raw = pd.concat(
            {
                "Close": pd.DataFrame(
                    {
                        "BRK-B": [500.0, 501.0, 502.0, 503.0],
                        "SPY": [600.0, 601.0, 602.0, 603.0],
                    },
                    index=index,
                )
            },
            axis=1,
        )

        with patch("scripts.massive_backtest_engine.yf.download", return_value=raw):
            close = _download_intraday_close(["BRK.B", "SPY"], period="60d", interval="15m")

        self.assertListEqual(list(close.columns), ["BRK.B", "SPY"])
        self.assertEqual(float(close["BRK.B"].iloc[-1]), 503.0)

    def test_delay_quote_suite_produces_summary_on_synthetic_intraday_data(self):
        periods = 420
        index = pd.date_range("2026-01-02 14:30:00+00:00", periods=periods, freq="15min")
        base = np.linspace(100.0, 108.0, periods)
        oscillation = 1.2 * np.sin(np.linspace(0.0, 16.0, periods))
        close = pd.DataFrame(
            {
                "SPY": base + oscillation,
                "QQQ": (base * 1.1) - (oscillation * 0.8),
            },
            index=index,
        )

        result = _run_delay_quote_suite_from_close(close, horizon_days=2)

        self.assertNotIn("error", result)
        self.assertGreaterEqual(result["summary"]["symbols"], 1)
        self.assertIn("puts", result["summary"])
        self.assertIn("calls", result["summary"])
        self.assertGreater(result["summary"]["puts"]["samples"], 0)
        self.assertGreater(result["summary"]["calls"]["samples"], 0)

    def test_option_model_suite_produces_summary_on_synthetic_intraday_data(self):
        periods = 420
        index = pd.date_range("2026-01-02 14:30:00+00:00", periods=periods, freq="15min")
        base = np.linspace(100.0, 109.0, periods)
        oscillation = 1.6 * np.sin(np.linspace(0.0, 20.0, periods))
        close = pd.DataFrame(
            {
                "SPY": base + oscillation,
                "QQQ": (base * 1.08) - (oscillation * 0.7),
            },
            index=index,
        )

        result = _run_option_model_suite_from_close(close, horizon_days=2)

        self.assertNotIn("error", result)
        self.assertGreaterEqual(result["summary"]["symbols"], 1)
        self.assertIn("models", result["summary"])
        self.assertIn("ensemble", result["summary"]["models"])
        self.assertGreaterEqual(result["summary"]["models"]["ensemble"]["avg_signals_per_symbol"], 0.0)


if __name__ == "__main__":
    unittest.main()
