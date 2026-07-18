import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from bot import state as bot_state
from dashboard import analytics
from dashboard.server import _compute_metrics


class LivePerformanceMetricTests(unittest.TestCase):
    def test_compute_metrics_resamples_high_frequency_series_and_suppresses_unstable_ratios(self):
        start = datetime(2026, 5, 10, 8, 0, tzinfo=timezone.utc)
        timestamps = [
            (start + timedelta(seconds=90 * idx)).isoformat()
            for idx in range(120)
        ]
        equity = [1000.0 + (idx * 2.5) for idx in range(120)]

        metrics = _compute_metrics(equity, timestamps)

        self.assertEqual(metrics["total_return_pct"], 26.89)
        self.assertIsNone(metrics["annualised_return_pct"])
        self.assertIsNone(metrics["sharpe"])
        self.assertIsNone(metrics["sortino"])
        self.assertIsNone(metrics["calmar"])
        self.assertEqual(metrics["metrics_meta"]["resample_rule"], "15min")
        self.assertFalse(metrics["metrics_meta"]["sufficient_history"])


class EliteOverviewScoreTests(unittest.TestCase):
    def test_elite_overview_promotes_live_institutional_score_with_paper_edge(self):
        stale = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        now = datetime.now(timezone.utc).isoformat()
        backtest_report = {
            "massive_overview": {
                "predictive_score": 0.554979625899119,
                "consensus_market_state": "calm_bull",
                "consensus_strategy_profile": "crash_reversal",
                "institutional_score": 0.679364,
                "deployment_tier": "paper_candidate",
            },
            "institutional_robustness": {
                "institutional_score": 0.679364,
                "deployment_tier": "paper_candidate",
            },
            "movement_suite": {"summary": {"avg_accuracy": 0.5064925714347913, "avg_alpha_daily": -0.005452594342782594}},
            "pairs_suite": {"summary": {"win_rate": 0.2299503882234434, "avg_trade_return": 0.0009744109641754321}},
            "regime_suite": {"summary": {"directional_accuracy_proxy": 0.5769071212722947}},
            "strategy_profile_suite": {"summary": {"consensus_profile": "crash_reversal", "consensus_state": "calm_bull"}},
            "ml_alpha_suite": {"summary": {"avg_information_coefficient": -0.01244565217391304, "long_only": {"annualized_return": 0.2493766479242181, "sharpe_ratio": 1.0821567058623938}}},
            "option_model_suite": {"summary": {"models": {"monte_carlo": {"avg_edge_pct": 0.20565156892253, "long_win_rate": 0.45351473922902497, "avg_signals_per_symbol": 308.7, "avg_long_pnl": 1.1970849265384798}}}},
        }

        def _read_json(path, default):
            if path == analytics.RISK_SNAPSHOT_PATH:
                return {
                    "generated_at_utc": stale,
                    "total_equity": 9266.48,
                    "daily_pnl_pct": 0.0177,
                    "daily_pnl_dollars": 1.64,
                    "macro_regime": "THETA_ENGINE",
                    "movement_bias": "neutral",
                    "runtime_profile": "all_weather",
                    "runtime_policy_mode": "weekend_policy",
                    "runtime_market_state": "transition",
                    "open_positions": 1,
                    "portfolio_delta": 0.0,
                    "portfolio_theta": 0.0,
                    "portfolio_vega": 0.0,
                    "target_delta": -2.1577,
                    "target_theta": 35.3637,
                    "target_vega": 9.2182,
                    "vix": 17.19,
                }
            if path == analytics.PORTFOLIO_GUARD_PATH:
                return {
                    "generated_at_utc": stale,
                    "portfolio_risk_engine": {
                        "risk_score": 0.0,
                        "breaches": [],
                        "kill_switch_active": False,
                        "underlying_count": 0,
                        "gross_exposure_pct_equity": 0.0,
                    },
                }
            if path == analytics.EXECUTION_SUMMARY_PATH:
                return {
                    "generated_at_utc": stale,
                    "records": 7,
                    "fill_rate": 1.0,
                    "avg_execution_quality_score": 0.72,
                    "broker_fill_price_coverage": 1.0,
                    "degraded_execution_count": 0,
                }
            if path == analytics.SYSTEM_RESOURCE_PATH:
                return {
                    "generated_at_utc": now,
                    "status": {"pressure": "normal"},
                    "host_metrics": {
                        "normalized_cpu_load_pct": 2.96,
                        "memory": {"usage_pct": 12.09},
                        "disk": {"usage_pct": 40.08},
                    },
                }
            if path == analytics.LATEST_BACKTEST_REPORT:
                return backtest_report
            return default

        with patch.object(analytics, "read_json", side_effect=_read_json), \
             patch.object(bot_state, "get_daily_pnl_snapshot", return_value={
                 "source": "equity_curve_mark_to_market",
                 "daily_pnl": 0.0,
                 "daily_pnl_pct": 0.0,
                 "current_equity": 4591.77,
                 "session_start_equity": 4591.77,
                 "session_start_ts": "2026-05-09T23:59:52.100458+00:00",
                 "latest_equity_ts": now,
                 "intraday_low_equity": 4591.77,
                 "intraday_high_equity": 4591.77,
                 "intraday_range_pct": 0.0,
                 "peak_equity": 4591.77,
                 "distance_to_peak_pct": 0.0,
                 "equity_samples_today": 1024,
                 "closed_trade_pnl_today": 0.0,
                 "closed_trade_count_today": 0,
                 "equity_freshness_seconds": 45.0,
             }), \
             patch.object(analytics, "live_broker_snapshot", return_value={
                 "available": True,
                 "source": "alpaca_account",
                 "total_equity": 10011.76,
                 "cash": 4591.77,
                 "buying_power": 9183.54,
                 "long_market_value": 5419.99,
                 "short_market_value": 0.0,
                 "position_market_value": 5419.99,
                 "net_unrealized_pnl": 25.27,
                 "last_equity": 10000.0,
                 "day_pnl_dollars": 11.76,
                 "day_pnl_pct": 0.1176,
                 "positions_count": 4,
                 "account_status": "ACTIVE",
                 "latest_equity_ts": now,
                 "session_start_equity": 10000.0,
                 "session_high_equity": 10014.06,
                 "session_low_equity": 9998.5,
                 "inception_equity": 10000.0,
                 "all_time_high_equity": 10014.06,
                 "total_return_pct": 0.1176,
                 "tape_timestamps": [now],
                 "tape_equity": [10011.76],
             }), \
             patch.object(analytics, "live_broker_positions", return_value=(
                 [{"symbol": "SOLUSDT", "market_value": 3944.31, "asset_class": "crypto"}],
                 {"available": True, "source": "alpaca_live", "count": 4, "asset_mix": {"crypto": 4}, "gross_market_value": 5419.99, "net_unrealized_pnl": 25.27},
             )), \
             patch.object(analytics, "build_stocks_overview", return_value={"pairs": {"top_pairs": []}, "research_leaders": []}), \
             patch.object(analytics, "load_runtime_trades", return_value=[]), \
             patch.object(analytics, "_latest_equity_curve_point", return_value={"ts": "2026-05-10T10:15:18.503842+00:00", "equity": 4591.77, "drawdown": 0.0}):
            payload = analytics.build_elite_overview()

        self.assertEqual(payload["headline"]["total_equity"], 10011.76)
        self.assertAlmostEqual(payload["headline"]["daily_pnl_pct"], 0.1176)
        self.assertEqual(payload["performance"]["source"], "alpaca_account")
        self.assertEqual(payload["headline"]["open_positions"], 4)
        self.assertEqual(payload["status"]["trading_status"], "LIVE / RISK STALE")
        self.assertEqual(payload["freshness"]["equity_snapshot_fresh"], True)
        self.assertAlmostEqual(payload["research"]["backtest_institutional_score"], 0.679364)
        # paper_edge_score measures how much of the research-paper catalog is
        # ENABLED. The 2026-07 league pruning benched 10 measured losers, so
        # coverage sits deliberately lower now — assert it's computed and sane
        # rather than encoding the everything-on world (was >= 0.9).
        self.assertGreaterEqual(payload["research"]["paper_edge_score"], 0.4)
        self.assertLessEqual(payload["research"]["paper_edge_score"], 1.0)
        self.assertGreaterEqual(payload["research"]["institutional_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
