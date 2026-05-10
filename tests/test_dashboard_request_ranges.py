import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from dashboard import server


class DashboardRequestRangeTests(unittest.TestCase):
    def test_intraday_stock_requests_are_clamped_without_alpaca(self):
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)

        with patch("backtester.data_loader._alpaca_stock_data_available", return_value=False):
            start, end, notice = server._sanitize_history_request(
                ["SPY"],
                "1h",
                "2023-01-01",
                "2026-05-10",
                now=now,
            )

        self.assertEqual(start, "2024-05-11")
        self.assertEqual(end, "2026-05-10")
        self.assertIn("Yahoo Finance", notice)
        self.assertIn("729", notice)

    def test_intraday_stock_requests_fail_when_end_is_outside_retention_window(self):
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)

        with patch("backtester.data_loader._alpaca_stock_data_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "Yahoo Finance"):
                server._sanitize_history_request(
                    ["SPY"],
                    "1h",
                    "2023-01-01",
                    "2024-01-01",
                    now=now,
                )

    def test_crypto_requests_are_left_unchanged(self):
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)

        with patch("backtester.data_loader._alpaca_stock_data_available", return_value=False):
            start, end, notice = server._sanitize_history_request(
                ["BTCUSDT"],
                "1h",
                "2023-01-01",
                "2026-05-10",
                now=now,
            )

        self.assertEqual(start, "2023-01-01")
        self.assertEqual(end, "2026-05-10")
        self.assertIsNone(notice)


if __name__ == "__main__":
    unittest.main()
