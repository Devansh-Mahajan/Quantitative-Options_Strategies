import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from bot.config import cfg
from backtester import data_loader
from exchange.alpaca_client import _from_alpaca, _to_alpaca


def _sample_history() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500],
            "quote_volume": [100500, 111650, 123000, 134550, 146300, 158250],
            "trades": [0, 0, 0, 0, 0, 0],
            "taker_buy_base": [500, 550, 600, 650, 700, 750],
            "taker_buy_quote": [50250, 55825, 61500, 67275, 73150, 79125],
        },
        index=idx,
    )


class MixedAssetSupportTests(unittest.TestCase):
    def test_config_balances_stock_and_crypto_defaults(self):
        symbols = cfg.model_symbols

        self.assertIn("SPY", symbols)
        self.assertIn("BTCUSDT", symbols)
        self.assertEqual(symbols[0], "SPY")
        self.assertEqual(symbols[1], "BTCUSDT")

    def test_stock_symbols_use_stock_history_loader(self):
        sample = _sample_history()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "AAPL_1h.parquet"
            with patch.object(data_loader, "_cache_path", return_value=cache_path), \
                 patch.object(data_loader, "_load_stock_history", return_value=sample) as stock_loader, \
                 patch.object(data_loader, "fetch_klines_raw") as crypto_loader:
                loaded = data_loader.load("AAPL", "1h", "2026-01-01", "2026-01-02", force_download=True)

        stock_loader.assert_called_once()
        crypto_loader.assert_not_called()
        self.assertFalse(loaded.empty)
        self.assertIn("close", loaded.columns)

    def test_crypto_symbols_use_binance_kline_loader(self):
        raw = [
            [1735689600000, "100", "101", "99", "100.5", "1000", 1735693200000, "100500", 10, "500", "50250", "0"],
            [1735693200000, "100.5", "102", "100", "101.5", "1100", 1735696800000, "111650", 11, "550", "55825", "0"],
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "BTCUSDT_1h.parquet"
            with patch.object(data_loader, "_cache_path", return_value=cache_path), \
                 patch.object(data_loader, "fetch_klines_raw", return_value=raw) as crypto_loader, \
                 patch.object(data_loader, "_load_stock_history") as stock_loader:
                loaded = data_loader.load("BTCUSDT", "1h", "2025-01-01", "2025-01-02", force_download=True)

        crypto_loader.assert_called_once()
        stock_loader.assert_not_called()
        self.assertFalse(loaded.empty)
        self.assertIn("taker_buy_base", loaded.columns)

    def test_long_range_hourly_stock_download_is_chunked(self):
        chunk_calls = []
        progress_events = []

        def _fake_chunk(symbol, yf_interval, start, end):
            chunk_calls.append((symbol, yf_interval, start, end))
            idx = pd.date_range(start, periods=3, freq="h", tz="UTC")
            return pd.DataFrame(
                {
                    "Open": [100.0, 101.0, 102.0],
                    "High": [101.0, 102.0, 103.0],
                    "Low": [99.0, 100.0, 101.0],
                    "Close": [100.5, 101.5, 102.5],
                    "Volume": [1000.0, 1100.0, 1200.0],
                },
                index=idx,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "SPY_1h.parquet"
            with patch.object(data_loader, "_cache_path", return_value=cache_path), \
                 patch.object(data_loader, "_alpaca_stock_data_available", return_value=False), \
                 patch.object(data_loader, "_download_stock_chunk", side_effect=_fake_chunk):
                loaded = data_loader.load(
                    "SPY",
                    "1h",
                    "2023-01-01",
                    "2026-05-10",
                    force_download=True,
                    progress_callback=progress_events.append,
                )

        self.assertGreater(len(chunk_calls), 1)
        self.assertFalse(loaded.empty)
        self.assertEqual(chunk_calls[0][1], "60m")
        self.assertEqual(progress_events[-1]["progress"], 100.0)
        self.assertIn("Downloaded SPY 1h history", progress_events[-2]["message"])

    def test_stock_loader_prefers_alpaca_when_available(self):
        sample = _sample_history()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "SPY_1h.parquet"
            with patch.object(data_loader, "_cache_path", return_value=cache_path), \
                 patch.object(data_loader, "_alpaca_stock_data_available", return_value=True), \
                 patch.object(data_loader, "_download_stock_history_alpaca", return_value=sample) as alpaca_loader, \
                 patch.object(data_loader, "_download_stock_chunk") as yahoo_loader:
                loaded = data_loader.load("SPY", "1h", "2023-01-01", "2026-05-10", force_download=True)

        alpaca_loader.assert_called_once()
        yahoo_loader.assert_not_called()
        self.assertFalse(loaded.empty)
        self.assertIn("close", loaded.columns)

    def test_alpaca_symbol_mapping_preserves_stocks_and_crypto(self):
        self.assertEqual(_to_alpaca("AAPL"), "AAPL")
        self.assertEqual(_to_alpaca("BTCUSDT"), "BTC/USD")
        self.assertEqual(_from_alpaca("AAPL"), "AAPL")
        self.assertEqual(_from_alpaca("ETH/USD"), "ETHUSDT")


if __name__ == "__main__":
    unittest.main()
