import unittest

from core.execution.portfolio_optimizer import effective_trade_risk_budget


class EffectiveTradeRiskBudgetTests(unittest.TestCase):
    def test_floor_prevents_unexecutable_trade_caps(self):
        budget = effective_trade_risk_budget(
            base_trade_risk=344.0,
            deployment_fraction=0.05,
            equity=10_000.0,
            minimum_dollars=100.0,
            minimum_pct_equity=0.025,
        )
        self.assertAlmostEqual(budget, 250.0)

    def test_floor_never_exceeds_pre_scaled_trade_risk(self):
        budget = effective_trade_risk_budget(
            base_trade_risk=80.0,
            deployment_fraction=0.05,
            equity=10_000.0,
            minimum_dollars=100.0,
            minimum_pct_equity=0.025,
        )
        self.assertAlmostEqual(budget, 80.0)

    def test_scaled_budget_is_used_when_it_stays_above_floor(self):
        budget = effective_trade_risk_budget(
            base_trade_risk=700.0,
            deployment_fraction=0.60,
            equity=10_000.0,
            minimum_dollars=100.0,
            minimum_pct_equity=0.025,
        )
        self.assertAlmostEqual(budget, 420.0)

    def test_zero_deployment_disables_trade_risk(self):
        budget = effective_trade_risk_budget(
            base_trade_risk=700.0,
            deployment_fraction=0.0,
            equity=10_000.0,
        )
        self.assertEqual(budget, 0.0)


if __name__ == "__main__":
    unittest.main()
