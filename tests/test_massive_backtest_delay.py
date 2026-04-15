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
        self.assertIn("methodology_score", robustness)

    def test_institutional_robustness_ignores_invalid_windows_and_tracks_evidence(self):
        payload = {
            "config": {"lookbacks": ["10y", "5y", "3y", "1y"]},
            "massive_overview": {
                "predictive_score": 0.55,
                "avg_recent_quote_error_pct": 0.03,
                "delay_filtered_put_win_rate": 0.56,
                "delay_filtered_call_win_rate": 0.57,
                "option_model_ensemble_win_rate": 0.49,
                "option_model_ensemble_edge_pct": 0.09,
                "ml_alpha_information_coefficient": 0.01,
                "ml_alpha_long_only_sharpe": 0.9,
            },
            "movement_suite": {
                "summary": {
                    "by_lookback": {
                        "10y": {"valid_runs": 5, "avg_accuracy": 0.51},
                        "5y": {"valid_runs": 5, "avg_accuracy": 0.508},
                        "3y": {"valid_runs": 5, "avg_accuracy": 0.507},
                        "1y": {"valid_runs": 5, "avg_accuracy": 0.499},
                        "6mo": {"valid_runs": 5, "avg_accuracy": 0.506},
                    }
                }
            },
            "pairs_suite": {
                "results_by_lookback": {
                    "10y": {"summary": {"pairs_evaluated": 12, "win_rate": 0.54}},
                    "5y": {"summary": {"pairs_evaluated": 10, "win_rate": 0.53}},
                    "3y": {"error": "insufficient_pairs", "summary": {"pairs_evaluated": 0}},
                    "1y": {"summary": {"pairs_evaluated": 0, "win_rate": 0.0}},
                }
            },
            "regime_suite": {
                "results_by_lookback": {
                    "10y": {"summary": {"n_samples": 400, "directional_accuracy_proxy": 0.51}},
                    "5y": {"summary": {"n_samples": 260, "directional_accuracy_proxy": 0.509}},
                    "3y": {"error": "insufficient_macro_features", "summary": {"n_samples": 80}},
                }
            },
            "strategy_profile_suite": {
                "results_by_lookback": {
                    "10y": {"summary": {"samples": 500, "current_state_best_profile": {"score": 0.63}}},
                    "5y": {"summary": {"samples": 450, "current_state_best_profile": {"score": 0.62}}},
                    "3y": {"error": "insufficient_profile_samples", "summary": {"samples": 200}},
                }
            },
            "delay_quote_suite": {
                "summary": {
                    "puts": {"delay_filtered_samples": 6000, "delay_filtered_win_rate": 0.56},
                    "calls": {"delay_filtered_samples": 5800, "delay_filtered_win_rate": 0.57},
                }
            },
            "option_model_suite": {
                "summary": {
                    "models": {
                        "ensemble": {
                            "avg_signals_per_symbol": 140.0,
                            "long_win_rate": 0.49,
                            "avg_edge_pct": 0.09,
                        }
                    }
                }
            },
            "ml_alpha_suite": {
                "summary": {
                    "months_tested": 40,
                    "avg_information_coefficient": 0.01,
                    "long_only": {"sharpe_ratio": 0.9},
                }
            },
        }

        robustness = _build_institutional_robustness(payload)

        self.assertEqual(robustness["windows"]["pairs"]["valid_windows"], 2)
        self.assertIn("strategy_profile", robustness["windows"])
        self.assertGreaterEqual(robustness["methodology_score"], 0.75)
        self.assertEqual(robustness["deployment_tier"], "institutional_candidate")

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
