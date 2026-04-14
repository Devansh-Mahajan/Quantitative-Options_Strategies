import unittest

from core.strategy_regime import build_live_controls, classify_market_state


class StrategyRegimeTests(unittest.TestCase):
    def test_calm_bull_state_prefers_bull_bucket(self):
        state = classify_market_state(0.03, 18.0, -0.01)
        controls = build_live_controls(
            profile_name="bull_trend",
            market_state=state,
            predictive_score=0.68,
            state_confidence=0.74,
        )

        self.assertEqual(state, "calm_bull")
        self.assertGreater(controls["strategy_weights"]["BULL"], controls["strategy_weights"]["BEAR"])
        self.assertTrue(controls["directional_enabled"])

    def test_panic_profile_disables_short_premium(self):
        controls = build_live_controls(
            profile_name="panic_hedge",
            market_state="panic",
            predictive_score=0.54,
            state_confidence=0.91,
            adaptive_profile={"rolling_avg_return_pct": -0.80},
        )

        self.assertFalse(controls["theta_enabled"])
        self.assertFalse(controls["directional_enabled"])
        self.assertGreater(controls["strategy_weights"]["VEGA"], controls["strategy_weights"]["THETA"])
        self.assertLessEqual(controls["max_vix_for_short_premium"], 16.0)


if __name__ == "__main__":
    unittest.main()
