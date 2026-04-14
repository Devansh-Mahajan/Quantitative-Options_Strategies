import unittest

from core.equity_overlay import (
    EquitySignalContext,
    _build_delta_hedge_targets,
    _score_directional_candidate,
)
from core.movement_predictor import MovementSignal


class _Quote:
    def __init__(self, price: float):
        self.price = price


class _FakeClient:
    def __init__(self, prices):
        self.prices = prices

    def get_stock_latest_trade(self, symbol):
        return {symbol: _Quote(self.prices[symbol])}


class EquityOverlayTests(unittest.TestCase):
    def test_delta_hedge_uses_inverse_symbol_for_negative_gap(self):
        client = _FakeClient({"SH": 42.0, "SPY": 520.0})
        targets = _build_delta_hedge_targets(
            client=client,
            total_equity=100_000.0,
            buying_power=15_000.0,
            current_vix=18.0,
            current_port_delta=65.0,
            target_port_delta=10.0,
            allow_delta_hedge_entries=True,
        )

        self.assertIn("SH", targets)
        self.assertEqual(targets["SH"]["mode"], "delta_hedge")
        self.assertGreater(targets["SH"]["qty"], 0)

    def test_directional_candidate_blocks_hot_iv_event_setup(self):
        signal = MovementSignal(symbol="AAPL", probability_up=0.71, expected_daily_move=0.006, expected_direction="up")
        context = EquitySignalContext(
            symbol="AAPL",
            earnings_days=3,
            iv_rank=92.0,
            iv_realized_ratio=1.6,
            distribution_zscore=0.9,
            price_percentile=0.55,
        )

        scored = _score_directional_candidate(signal, flow_score=0.8, context=context)
        self.assertIsNone(scored)


if __name__ == "__main__":
    unittest.main()
