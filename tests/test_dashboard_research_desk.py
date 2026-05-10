import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from dashboard import analytics


class DashboardResearchDeskTests(unittest.TestCase):
    def test_build_research_desk_uses_correlation_and_microstructure_blocks(self):
        corr_payload = {
            "stocks": {
                "breadth": {"advancers_1d": 320, "breadth_score": 0.62},
                "leaders_by_return": {"leaders_1d": [{"symbol": "AAPL", "return_pct": 2.1}]},
                "stats": {"avg_abs_corr": 0.41},
            },
            "crypto": {
                "breadth": {"advancers_1d": 5, "breadth_score": 0.58},
                "leaders_by_return": {"leaders_1d": [{"symbol": "BTCUSDT", "return_pct": 3.4}]},
                "stats": {"avg_abs_corr": 0.55},
            },
        }
        micro_board = [
            {
                "symbol": "BTCUSDT",
                "asset_class": "crypto",
                "pressure": 0.71,
                "direction": "BUY_PRESSURE",
                "archetype": "directional",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "research_cache.json"
            with patch.object(analytics, "RESEARCH_DESK_CACHE_PATH", cache_path), \
                 patch.object(analytics, "build_correlation_payload", return_value=corr_payload), \
                 patch.object(analytics, "_build_microstructure_board", return_value=micro_board):
                payload = analytics.build_research_desk(force=True)

        self.assertIn("strategy_catalog", payload)
        self.assertIn("papers", payload)
        self.assertEqual(payload["stock_breadth"]["advancers_1d"], 320)
        self.assertEqual(payload["crypto_breadth"]["advancers_1d"], 5)
        self.assertEqual(payload["microstructure_board"][0]["symbol"], "BTCUSDT")


if __name__ == "__main__":
    unittest.main()
