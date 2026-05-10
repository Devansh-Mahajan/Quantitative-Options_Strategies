import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from dashboard import analytics


def _close_frame(symbol_to_values: dict[str, list[float]], freq: str = "D") -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(next(iter(symbol_to_values.values()))), freq=freq, tz="UTC")
    return pd.DataFrame(symbol_to_values, index=idx)


class DashboardCorrelationPayloadTests(unittest.TestCase):
    def test_build_correlation_payload_returns_stock_and_crypto_blocks(self):
        stocks_close = _close_frame(
            {
                "SPY": [100, 101, 102, 104, 103, 105, 106, 107],
                "AAPL": [200, 202, 205, 207, 206, 209, 212, 214],
                "MSFT": [300, 299, 301, 303, 304, 306, 308, 309],
            }
        )
        crypto_frames = {
            "BTCUSDT": _close_frame({"close": [50000, 50500, 51000, 51500, 51200, 51800, 52200, 52500]}, freq="4h"),
            "ETHUSDT": _close_frame({"close": [3000, 3025, 3050, 3075, 3060, 3090, 3110, 3135]}, freq="4h"),
            "SOLUSDT": _close_frame({"close": [100, 102, 103, 104, 103, 105, 107, 108]}, freq="4h"),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "corr_cache.json"
            with patch.object(analytics, "CORRELATION_CACHE_PATH", cache_path), \
                 patch.object(analytics, "sp500_universe_symbols", return_value=["SPY", "AAPL", "MSFT"]), \
                 patch.object(analytics, "download_close_matrix", return_value=stocks_close), \
                 patch("backtester.data_loader.load_multi", return_value=crypto_frames), \
                 patch.object(analytics.cfg, "futures_symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"]), \
                 patch.object(analytics.cfg, "spot_symbols", []):
                payload = analytics.build_correlation_payload(force=True)
                self.assertTrue(cache_path.exists())
                cached = json.loads(cache_path.read_text())

        self.assertIn("stocks", payload)
        self.assertIn("crypto", payload)
        self.assertEqual(payload["stocks"]["symbol_count"], 3)
        self.assertEqual(payload["crypto"]["symbol_count"], 3)
        self.assertIn("SPY", payload["stocks"]["peer_map"])
        self.assertIn("BTCUSDT", payload["crypto"]["peer_map"])
        self.assertIn("stocks", cached)
        self.assertIn("crypto", cached)


if __name__ == "__main__":
    unittest.main()
