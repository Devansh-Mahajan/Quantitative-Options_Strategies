import unittest
from types import SimpleNamespace

from alpaca.trading.enums import AssetClass

from core.live_allocation import build_live_allocation_snapshot, classify_position_book
from core.state_manager import calculate_risk


class LiveAllocationTests(unittest.TestCase):
    def test_classifies_crypto_and_options_equity_books(self):
        self.assertEqual(
            classify_position_book(
                SimpleNamespace(symbol="BTCUSD", asset_class=AssetClass.CRYPTO)
            ),
            "crypto",
        )
        self.assertEqual(
            classify_position_book(
                SimpleNamespace(symbol="AAPL", asset_class=AssetClass.US_EQUITY)
            ),
            "options_equity",
        )

    def test_build_snapshot_tracks_separate_books(self):
        positions = [
            SimpleNamespace(
                symbol="AAPL",
                asset_class=AssetClass.US_EQUITY,
                qty="10",
                current_price="110",
                market_value="1100",
                avg_entry_price="100",
            ),
            SimpleNamespace(
                symbol="BTCUSD",
                asset_class=AssetClass.CRYPTO,
                qty="0.5",
                current_price="60000",
                market_value="30000",
                avg_entry_price="58000",
            ),
        ]

        snapshot = build_live_allocation_snapshot(
            total_equity=100000.0,
            positions=positions,
            market_open=True,
        ).to_dict()

        self.assertEqual(snapshot["books"]["options_equity"]["used_market_value"], 1100.0)
        self.assertEqual(snapshot["books"]["crypto"]["used_market_value"], 30000.0)
        self.assertEqual(snapshot["books"]["options_equity"]["share"], 0.65)
        self.assertEqual(snapshot["books"]["crypto"]["share"], 0.35)

    def test_calculate_risk_can_exclude_crypto_from_options_book(self):
        positions = [
            SimpleNamespace(
                symbol="AAPL",
                asset_class=AssetClass.US_EQUITY,
                qty="5",
                avg_entry_price="100",
            ),
            SimpleNamespace(
                symbol="BTCUSD",
                asset_class=AssetClass.CRYPTO,
                qty="0.2",
                current_price="10000",
                avg_entry_price="9000",
            ),
        ]

        self.assertEqual(calculate_risk(positions, include_crypto=False), 500.0)
        self.assertEqual(calculate_risk(positions, include_crypto=True), 2500.0)


if __name__ == "__main__":
    unittest.main()
