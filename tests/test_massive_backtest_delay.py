import unittest

import numpy as np
import pandas as pd

from scripts.massive_backtest_engine import _run_delay_quote_suite_from_close, _run_option_model_suite_from_close


class MassiveBacktestDelaySuiteTests(unittest.TestCase):
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
