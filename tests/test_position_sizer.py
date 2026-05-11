import unittest

from risk.position_sizer import PositionSizer, SizingContext


class PositionSizerTests(unittest.TestCase):
    def test_higher_volatility_reduces_position_size(self):
        sizer = PositionSizer()
        low_vol = sizer.size_signal(
            SizingContext(
                signal_confidence=0.72,
                equity=10_000,
                price=100,
                realised_vol=0.20,
                regime="ranging",
            )
        )
        high_vol = sizer.size_signal(
            SizingContext(
                signal_confidence=0.72,
                equity=10_000,
                price=100,
                realised_vol=1.20,
                regime="ranging",
            )
        )

        self.assertGreater(low_vol.notional, 0)
        self.assertGreater(high_vol.notional, 0)
        self.assertLess(high_vol.notional, low_vol.notional)

    def test_existing_symbol_exposure_blocks_concentration(self):
        sizer = PositionSizer()
        decision = sizer.size_signal(
            SizingContext(
                signal_confidence=0.85,
                equity=10_000,
                price=100,
                realised_vol=0.35,
                regime="bull",
                current_gross_exposure=500,
                symbol_exposure=10_000,
            )
        )

        self.assertEqual(decision.quantity, 0)
        self.assertEqual(decision.reason, "no_remaining_risk_budget")

    def test_drawdown_throttles_risk_before_guard_halts(self):
        sizer = PositionSizer()
        normal = sizer.size_signal(
            SizingContext(
                signal_confidence=0.75,
                equity=10_000,
                peak_equity=10_000,
                daily_start_equity=10_000,
                price=100,
                realised_vol=0.40,
                regime="ranging",
            )
        )
        stressed = sizer.size_signal(
            SizingContext(
                signal_confidence=0.75,
                equity=9_000,
                peak_equity=10_000,
                daily_start_equity=9_000,
                price=100,
                realised_vol=0.40,
                regime="ranging",
            )
        )

        self.assertGreater(normal.notional, stressed.notional)
        self.assertGreater(stressed.notional, 0)
        self.assertLess(stressed.drawdown_scalar, normal.drawdown_scalar)

    def test_exit_plan_recalibrates_with_volatility_and_regime(self):
        sizer = PositionSizer()
        calm = sizer.exit_plan(realised_vol=0.20, regime="ranging", confidence=0.60)
        stressed = sizer.exit_plan(realised_vol=1.00, regime="volatile", confidence=0.60)

        self.assertGreater(stressed.stop_loss_pct, calm.stop_loss_pct)
        self.assertGreater(stressed.take_profit_pct, stressed.stop_loss_pct)
        self.assertLess(calm.trailing_stop_pct, calm.stop_loss_pct)

    def test_legacy_size_from_signal_still_returns_tuple(self):
        sizer = PositionSizer()
        notional, quantity = sizer.size_from_signal(
            signal_confidence=0.75,
            equity=10_000,
            price=100,
            realised_vol=0.35,
            regime="ranging",
        )

        self.assertGreater(notional, 0)
        self.assertEqual(quantity, notional / 100)


if __name__ == "__main__":
    unittest.main()
