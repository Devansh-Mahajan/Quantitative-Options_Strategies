import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from bot import state


class DailyPnlSnapshotTests(unittest.TestCase):
    def test_snapshot_prefers_mark_to_market_equity_curve(self):
        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        before_open = day_start - timedelta(minutes=1)
        first_today = now - timedelta(seconds=2)
        latest = now - timedelta(seconds=1)

        with tempfile.TemporaryDirectory() as td, patch.object(state, "_DB_PATH", Path(td) / "bot_state.db"):
            state.init_db()
            with state._conn() as conn:
                conn.execute(
                    "INSERT INTO equity_curve (ts,equity,drawdown) VALUES (?,?,?)",
                    (before_open.isoformat(), 1000.0, 0.0),
                )
                conn.execute(
                    "INSERT INTO equity_curve (ts,equity,drawdown) VALUES (?,?,?)",
                    (first_today.isoformat(), 1025.0, 0.0),
                )
                conn.execute(
                    "INSERT INTO equity_curve (ts,equity,drawdown) VALUES (?,?,?)",
                    (latest.isoformat(), 1050.0, 0.0),
                )
                conn.execute(
                    "INSERT INTO trades (ts,symbol,market,side,quantity,price,strategy,pnl,status,meta) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (latest.isoformat(), "SPY", "options", "SELL", 1.0, 2.5, "theta", 15.0, "closed", "{}"),
                )

            snapshot = state.get_daily_pnl_snapshot()

        self.assertEqual(snapshot["source"], "equity_curve_mark_to_market")
        self.assertAlmostEqual(snapshot["daily_pnl"], 50.0)
        self.assertAlmostEqual(snapshot["daily_pnl_pct"], 5.0)
        self.assertEqual(snapshot["session_start_equity"], 1000.0)
        self.assertEqual(snapshot["current_equity"], 1050.0)
        self.assertEqual(snapshot["closed_trade_pnl_today"], 15.0)
        self.assertEqual(snapshot["closed_trade_count_today"], 1)


if __name__ == "__main__":
    unittest.main()
