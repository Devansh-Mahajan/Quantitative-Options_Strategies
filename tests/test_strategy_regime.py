import unittest

from core.strategy_regime import build_live_controls, classify_market_state, synthesize_live_controls


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

    def test_synthetic_live_controls_promote_trend_acceleration_in_confident_bull_tape(self):
        controls = synthesize_live_controls(
            macro_strategy="THETA_ENGINE",
            movement_bias="bullish",
            signal_confidence=0.74,
            macro_confidence=0.62,
            vix_level=17.5,
            adaptive_profile={"rolling_avg_return_pct": 0.35},
        )

        self.assertEqual(controls["market_state"], "calm_bull")
        self.assertEqual(controls["selected_profile"], "trend_acceleration")
        self.assertEqual(controls["control_source"], "synthetic_live_policy")
        self.assertTrue(controls["directional_enabled"])

    def test_synthetic_live_controls_can_switch_to_crash_reversal_when_tail_risk_and_rebound_align(self):
        controls = synthesize_live_controls(
            macro_strategy="TAIL_HEDGE",
            movement_bias="bullish",
            signal_confidence=0.72,
            macro_confidence=0.90,
            vix_level=32.0,
            adaptive_profile={"rolling_avg_return_pct": -0.9},
        )

        self.assertEqual(controls["market_state"], "panic")
        self.assertEqual(controls["selected_profile"], "crash_reversal")
        self.assertFalse(controls["theta_enabled"])
        self.assertTrue(controls["vega_enabled"])


if __name__ == "__main__":
    unittest.main()
