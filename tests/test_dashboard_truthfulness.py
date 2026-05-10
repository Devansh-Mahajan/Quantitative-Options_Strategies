import sqlite3
import unittest
import asyncio
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pandas as pd
from fastapi.testclient import TestClient

from dashboard.analytics import build_options_overview, build_simulation_payload, build_trade_odds
from dashboard import server
from dashboard.server import app


class DashboardTruthfulnessTests(unittest.TestCase):
    def test_options_overview_keeps_current_book_empty_without_live_positions(self):
        activity_row = {
            "symbol": "SPY260515C00500000",
            "side": "LONG",
            "entry_price": 1.25,
            "filled_at": "2026-05-10T10:00:00+00:00",
        }
        with patch("dashboard.analytics.live_option_positions", return_value=([], {"available": False, "reason": "broker_unavailable"})), \
             patch("dashboard.analytics.ledger_option_activity", return_value=[activity_row]):
            payload = build_options_overview()

        self.assertEqual(payload["positions"], [])
        self.assertEqual(payload["recent_activity"][0]["symbol"], activity_row["symbol"])
        self.assertEqual(payload["source"], "runtime_snapshot")

    def test_trade_odds_suppresses_historical_fill_replays_when_no_active_positions(self):
        with patch("dashboard.analytics.load_runtime_trades", return_value=[]), \
             patch("dashboard.analytics.live_option_positions", return_value=([], {"available": False, "reason": "broker_unavailable"})), \
             patch("dashboard.analytics.ledger_option_activity", return_value=[{"symbol": "SPY260515C00500000"}]):
            payload = build_trade_odds()

        self.assertEqual(payload["trades"], [])
        self.assertIn("No active confirmed positions", payload["note"])

    def test_trade_odds_ignores_pending_runtime_rows(self):
        pending_trade = {
            "symbol": "SPY260515C00500000",
            "status": "pending_new",
            "price": 1.25,
            "side": "buy",
            "market": "options",
            "meta_payload": {},
        }
        with patch("dashboard.analytics.load_runtime_trades", return_value=[pending_trade]), \
             patch("dashboard.analytics.live_option_positions", return_value=([], {"available": False, "reason": "broker_unavailable"})):
            payload = build_trade_odds()

        self.assertEqual(payload["trades"], [])
        self.assertIn("No active confirmed positions", payload["note"])

    def test_simulation_payload_marks_insufficient_history_instead_of_proxy_paths(self):
        timestamps = [f"2026-05-10T10:{minute:02d}:00+00:00" for minute in range(15)]
        equity = [1000.0] * len(timestamps)

        payload = build_simulation_payload(equity, timestamps)

        self.assertFalse(payload["available"])
        self.assertEqual(payload["source"], "insufficient_history")
        self.assertEqual(payload["monte_carlo"]["paths"], 0)
        self.assertIn("suppressed", payload["notice"])

    def test_execution_ledger_route_returns_fills_with_correct_notional(self):
        ledger = [
            {
                "order_id": "ord-opt",
                "status": "orderstatus.filled",
                "filled_qty": 2,
                "filled_avg_price": 1.5,
                "filled_at_utc": "2026-05-10T10:00:00+00:00",
                "partial_fill": False,
                "legs": [{"symbol": "SPY260515C00500000", "side": "buy"}],
                "execution_quality": {"score": 0.9, "tier": "excellent"},
                "source": "broker_reconciliation",
            },
            {
                "order_id": "ord-spot",
                "status": "orderstatus.filled",
                "filled_qty": 3,
                "filled_avg_price": 100.0,
                "filled_at_utc": "2026-05-10T11:00:00+00:00",
                "partial_fill": False,
                "legs": [{"symbol": "SGOV", "side": "sell"}],
                "execution_quality": {"score": 0.7, "tier": "acceptable"},
                "source": "broker_reconciliation",
            },
        ]

        def fake_read_json(path, default):
            if str(path).endswith("execution_ledger.json"):
                return ledger
            return default

        with patch("dashboard.server.read_json", side_effect=fake_read_json):
            with TestClient(app) as client:
                response = client.get("/api/execution/ledger")
                self.assertEqual(response.status_code, 200)
                payload = response.json()

        self.assertEqual(payload["filled"], 2)
        self.assertEqual(len(payload["fills"]), 2)
        self.assertEqual(payload["fills"][0]["market"], "options")
        self.assertEqual(payload["fills"][0]["notional"], 300.0)
        self.assertEqual(payload["fills"][1]["market"], "spot")
        self.assertEqual(payload["fills"][1]["notional"], 300.0)

    def test_trades_analysis_refuses_synthetic_pnl_when_trades_table_is_empty(self):
        ledger = [
            {
                "status": "orderstatus.filled",
                "filled_at_utc": "2026-05-10T10:00:00+00:00",
                "legs": [{"symbol": "SPY260515C00500000", "side": "buy"}],
            }
        ]

        @contextmanager
        def fake_conn():
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            conn.execute(
                "CREATE TABLE trades (ts TEXT, symbol TEXT, market TEXT, side TEXT, quantity REAL, price REAL, strategy TEXT, pnl REAL, status TEXT)"
            )
            try:
                yield conn
            finally:
                conn.close()

        def fake_read_json(path, default):
            if str(path).endswith("execution_ledger.json"):
                return ledger
            return default

        with patch("dashboard.server._db_conn", side_effect=fake_conn), \
             patch("dashboard.server.read_json", side_effect=fake_read_json):
            with TestClient(app) as client:
                response = client.get("/api/trades/analysis")
                self.assertEqual(response.status_code, 200)
                payload = response.json()

        self.assertEqual(payload["total_trades"], 0)
        self.assertFalse(payload["realized_pnl_available"])
        self.assertIn("Synthetic", payload["note"])
        self.assertEqual(payload["fill_activity"]["records"], 1)

    def test_master_recalibrate_accepts_empty_body(self):
        thread = Mock()
        thread.start = Mock()
        original_job = dict(server._recal_job)
        server._recal_job = {}
        try:
            with patch("dashboard.server.threading.Thread", return_value=thread):
                payload = asyncio.run(server.master_recalibrate(None))
        finally:
            server._recal_job = original_job

        self.assertTrue(payload["ok"])
        self.assertTrue(payload["scripts"])
        thread.start.assert_called_once()

    def test_intelligence_route_exposes_signal_list(self):
        alpha_snapshot = {
            "generated_at_utc": "2026-05-10T10:00:00+00:00",
            "signals": [{"symbol": "SPY", "alpha_score": 0.8}, {"symbol": "QQQ", "alpha_score": 0.2}],
        }

        def fake_read_json(path, default):
            if str(path).endswith("ml_alpha_snapshot.json"):
                return alpha_snapshot
            return default

        with patch("dashboard.server.read_json", side_effect=fake_read_json):
            with TestClient(app) as client:
                response = client.get("/api/intelligence")
                self.assertEqual(response.status_code, 200)
                payload = response.json()

        self.assertEqual(payload["signals"][0]["symbol"], "SPY")

    def test_risk_snapshot_withholds_metrics_for_sparse_continuous_history(self):
        index = pd.date_range("2026-05-10T10:00:00Z", periods=8, freq="15min")
        series = pd.Series([10000.0, 10020.0, 10010.0, 10025.0, 10030.0, 10028.0, 10035.0, 10040.0], index=index)
        meta = {
            "sample_points": 8,
            "effective_points": 8,
            "span_hours": 1.75,
            "resample_rule": "15min",
            "sufficient_history": False,
            "reset_events": 0,
            "segment_start_utc": index[0].isoformat(),
            "segment_end_utc": index[-1].isoformat(),
        }

        with patch("dashboard.server._latest_metric_equity_series", return_value=(series, meta)):
            payload = asyncio.run(server.risk_snapshot())

        self.assertFalse(payload["available"])
        self.assertIsNone(payload["var_95_pct"])
        self.assertEqual(payload["drawdown_series"], [])
        self.assertIn("not yet stable", payload["notice"])


if __name__ == "__main__":
    unittest.main()
