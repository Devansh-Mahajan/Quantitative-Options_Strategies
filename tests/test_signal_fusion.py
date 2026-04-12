import unittest

from core.greeks_targeting import PortfolioGreekTargets
from core.movement_predictor import MovementSignal
from core.signal_fusion import empty_ai_targets, route_strategy_candidates


def _greek_targets(
    *,
    target_theta: float = 0.0,
    target_vega: float = 0.0,
    movement_bias: str = "neutral",
) -> PortfolioGreekTargets:
    return PortfolioGreekTargets(
        target_delta=0.0,
        target_theta=target_theta,
        target_vega=target_vega,
        target_gamma=0.0,
        target_confidence=0.5,
        movement_bias=movement_bias,
    )


class SignalFusionTests(unittest.TestCase):
    def test_theta_macro_and_range_bound_symbol_rank_theta_first(self):
        plan = route_strategy_candidates(
            allowed_symbols=["AAPL", "MSFT"],
            ai_targets={
                "THETA": ["AAPL"],
                "VEGA": ["MSFT"],
                "BULL": [],
                "BEAR": [],
            },
            movement_signals=[
                MovementSignal("AAPL", 0.51, 0.001, "flat"),
                MovementSignal("MSFT", 0.72, 0.010, "up"),
            ],
            flow_map={"AAPL": 0.2, "MSFT": 0.9},
            pair_overlay={"signals": []},
            greek_targets=_greek_targets(target_theta=15.0),
            macro_strategy="THETA_ENGINE",
            macro_confidence=0.72,
            top_k=4,
        )

        self.assertEqual(plan.theta_candidates[0], "AAPL")
        self.assertGreater(plan.consensus_score, 0.0)
        self.assertGreaterEqual(plan.deployment_multiplier, 0.6)

    def test_pair_and_directional_alignment_boost_bull_bucket(self):
        plan = route_strategy_candidates(
            allowed_symbols=["NVDA", "AMD"],
            ai_targets=empty_ai_targets(),
            movement_signals=[
                MovementSignal("NVDA", 0.69, 0.007, "up"),
                MovementSignal("AMD", 0.43, 0.004, "down"),
            ],
            flow_map={"NVDA": 1.0, "AMD": 0.3},
            pair_overlay={
                "signals": [
                    {
                        "pair": "NVDA/AMD",
                        "confidence": 0.82,
                        "long": "NVDA",
                        "short": "AMD",
                    }
                ]
            },
            greek_targets=_greek_targets(movement_bias="bullish"),
            macro_strategy="THETA_ENGINE",
            macro_confidence=0.55,
            top_k=4,
        )

        self.assertIn("NVDA", plan.bull_candidates[:1])
        self.assertTrue(plan.diagnostics["top_scores"]["BULL"] > plan.diagnostics["top_scores"]["BEAR"])

    def test_tail_hedge_caps_deployment_multiplier(self):
        plan = route_strategy_candidates(
            allowed_symbols=["SPY", "QQQ"],
            ai_targets={"THETA": [], "VEGA": ["SPY"], "BULL": [], "BEAR": []},
            movement_signals=[
                MovementSignal("SPY", 0.40, 0.012, "down"),
                MovementSignal("QQQ", 0.39, 0.013, "down"),
            ],
            flow_map={"SPY": 0.4, "QQQ": 0.3},
            pair_overlay={"signals": []},
            greek_targets=_greek_targets(target_vega=20.0, movement_bias="bearish"),
            macro_strategy="TAIL_HEDGE",
            macro_confidence=0.91,
            top_k=4,
        )

        self.assertLessEqual(plan.deployment_multiplier, 0.75)
        self.assertIn("SPY", plan.vega_candidates)


if __name__ == "__main__":
    unittest.main()
