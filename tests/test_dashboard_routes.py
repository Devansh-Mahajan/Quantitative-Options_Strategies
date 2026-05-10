import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from dashboard import server
from dashboard.server import app


class DashboardRouteTests(unittest.TestCase):
    def test_training_and_intelligence_routes_are_available(self):
        with TestClient(app) as client:
            route_expectations = {
                "/api/backtest/list": "reports",
                "/api/strategies/params": "strategies",
                "/api/strategies/defaults": "strategies",
                "/api/training/scripts": "scripts",
                "/api/training/jobs": "jobs",
                "/api/ml/alpha": "signals",
                "/api/execution/quality": "records",
                "/api/system/health": "pressure",
                "/api/risk/events": "events",
            }

            for path, expected_key in route_expectations.items():
                response = client.get(path)
                self.assertEqual(response.status_code, 200, path)
                payload = response.json()
                self.assertIsInstance(payload, dict, path)
                self.assertIn(expected_key, payload, path)

    def test_config_exposes_mixed_asset_universe(self):
        with TestClient(app) as client:
            response = client.get("/api/config")
            self.assertEqual(response.status_code, 200)
            payload = response.json()

        self.assertIn("stock_symbols", payload)
        self.assertIn("mixed_symbols", payload)
        self.assertIn("benchmark_symbols", payload)
        self.assertIn("SPY", payload["benchmark_symbols"])
        self.assertIn("BTCUSDT", payload["mixed_symbols"])

    def test_risk_correlations_route_returns_stock_and_crypto_blocks(self):
        mock_payload = {
            "stocks": {"symbol_count": 502, "symbols": ["SPY", "AAPL"], "corr": [[1.0, 0.9], [0.9, 1.0]]},
            "crypto": {"symbol_count": 8, "symbols": ["BTCUSDT", "ETHUSDT"], "corr": [[1.0, 0.8], [0.8, 1.0]]},
            "options_greeks": {"portfolio_delta": 0.0},
            "stress_scenarios": {"equity_gap_down": -0.04},
        }
        with patch("dashboard.server.build_correlation_payload", return_value=mock_payload):
            with TestClient(app) as client:
                response = client.get("/api/risk/correlations")
                self.assertEqual(response.status_code, 200)
                payload = response.json()

        self.assertIn("stocks", payload)
        self.assertIn("crypto", payload)
        self.assertEqual(payload["stocks"]["symbol_count"], 502)
        self.assertEqual(payload["crypto"]["symbol_count"], 8)

    def test_research_desk_route_returns_catalog_payload(self):
        mock_payload = {
            "strategy_catalog": [{"strategy": "rma_strategy", "enabled": True}],
            "papers": [{"paper_family": "Bloch 2025 RMA"}],
            "stock_breadth": {"breadth_score": 0.61},
            "crypto_breadth": {"breadth_score": 0.57},
            "microstructure_board": [{"symbol": "BTCUSDT"}],
        }
        with patch("dashboard.server.build_research_desk", return_value=mock_payload):
            with TestClient(app) as client:
                response = client.get("/api/research/desk")
                self.assertEqual(response.status_code, 200)
                payload = response.json()

        self.assertIn("strategy_catalog", payload)
        self.assertIn("papers", payload)
        self.assertEqual(payload["microstructure_board"][0]["symbol"], "BTCUSDT")

    def test_root_disables_caching_for_dashboard_shell(self):
        with TestClient(app) as client:
            response = client.get("/")
            self.assertEqual(response.status_code, 200)

        cache_control = response.headers.get("cache-control", "")
        self.assertIn("no-store", cache_control)

    def test_forwardtest_status_route_preserves_progress_fields(self):
        job_id = "ft_progress_test"
        server._ft_jobs[job_id] = {
            "status": "RUNNING",
            "stage": "DOWNLOADING_DATA",
            "progress": 27.5,
            "detail": "Downloading SPY 1h chunk 1/2",
            "result": None,
            "error": None,
        }
        try:
            with TestClient(app) as client:
                response = client.get(f"/api/forwardtest/jobs/{job_id}")
                self.assertEqual(response.status_code, 200)
                payload = response.json()
        finally:
            server._ft_jobs.pop(job_id, None)

        self.assertEqual(payload["stage"], "DOWNLOADING_DATA")
        self.assertEqual(payload["progress"], 27.5)
        self.assertIn("Downloading SPY", payload["detail"])

    def test_daily_pnl_route_returns_mark_to_market_snapshot(self):
        snapshot = {
            "source": "equity_curve_mark_to_market",
            "daily_pnl": 125.4321,
            "daily_pnl_pct": 2.79,
            "session_start_equity": 4500.0,
            "current_equity": 4625.4321,
            "peak_equity": 4701.55,
            "intraday_low_equity": 4488.1,
            "intraday_high_equity": 4632.0,
            "intraday_range_pct": 3.2,
            "distance_to_peak_pct": -1.62,
            "equity_freshness_seconds": 18.0,
            "closed_trade_pnl_today": 42.0,
            "closed_trade_count_today": 3,
        }
        broker_snapshot = {
            "available": True,
            "source": "alpaca_account",
            "day_pnl_dollars": 125.4321,
            "day_pnl_pct": 2.79,
            "last_equity": 4500.0,
            "total_equity": 4625.4321,
            "latest_equity_ts": "2026-05-10T10:45:00+00:00",
            "session_low_equity": 4488.1,
            "session_high_equity": 4632.0,
        }
        with patch("dashboard.server.state.get_daily_pnl_snapshot", return_value=snapshot), \
             patch("dashboard.server.live_broker_snapshot", return_value=broker_snapshot):
            with TestClient(app) as client:
                response = client.get("/api/daily_pnl")
                self.assertEqual(response.status_code, 200)
                payload = response.json()

        self.assertEqual(payload["source"], "alpaca_account")
        self.assertEqual(payload["daily_pnl"], 125.4321)
        self.assertEqual(payload["daily_pnl_pct"], 2.79)
        self.assertEqual(payload["current_equity"], 4625.43)

    def test_positions_route_prefers_live_broker_positions(self):
        broker_rows = [
            {
                "symbol": "SOLUSDT",
                "asset_class": "crypto",
                "side": "LONG",
                "quantity": 2.0,
                "entry_price": 90.0,
                "current_price": 95.0,
                "market_value": 190.0,
                "unrealized_pnl": 10.0,
                "weight_pct": 1.9,
                "source": "alpaca_live",
            }
        ]
        broker_meta = {
            "available": True,
            "source": "alpaca_live",
            "count": 1,
            "asset_mix": {"crypto": 1},
            "gross_market_value": 190.0,
            "net_unrealized_pnl": 10.0,
        }
        with patch("dashboard.server.live_broker_positions", return_value=(broker_rows, broker_meta)), \
             patch("dashboard.server.state.get_open_trades", return_value=[]):
            with TestClient(app) as client:
                response = client.get("/api/positions")
                self.assertEqual(response.status_code, 200)
                payload = response.json()

        self.assertEqual(payload["source"], "alpaca_live")
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["positions"][0]["symbol"], "SOLUSDT")

    def test_live_tape_route_exposes_broker_tape(self):
        broker_snapshot = {
            "available": True,
            "source": "alpaca_account",
            "total_equity": 10011.76,
            "cash": 4591.77,
            "buying_power": 9183.54,
            "day_pnl_dollars": 11.76,
            "day_pnl_pct": 0.12,
            "total_return_pct": 0.12,
            "all_time_high_equity": 10014.06,
            "positions_count": 4,
            "account_status": "ACTIVE",
            "tape_timestamps": [
                "2026-05-10T10:40:00+00:00",
                "2026-05-10T10:41:00+00:00",
                "2026-05-10T10:42:00+00:00",
            ],
            "tape_equity": [10000.0, 10008.0, 10011.76],
        }
        with patch("dashboard.server.live_broker_snapshot", return_value=broker_snapshot):
            with TestClient(app) as client:
                response = client.get("/api/live/tape")
                self.assertEqual(response.status_code, 200)
                payload = response.json()

        self.assertEqual(payload["source"], "alpaca_account")
        self.assertEqual(payload["account"]["total_equity"], 10011.76)
        self.assertEqual(payload["summary"]["current_equity"], 10011.76)

    def test_dashboard_shell_contains_forwardtest_progress_bar(self):
        with TestClient(app) as client:
            response = client.get("/")
            self.assertEqual(response.status_code, 200)

        self.assertIn('id="ftProgBar"', response.text)
        self.assertIn('id="ftProgPct"', response.text)
        self.assertIn('id="eliteSessionMeta"', response.text)
        self.assertIn('id="eliteScoreBars"', response.text)
        self.assertIn('id="eliteFreshnessMeta"', response.text)
        self.assertIn('id="tapeChart"', response.text)
        self.assertIn('id="posSummary"', response.text)


if __name__ == "__main__":
    unittest.main()
