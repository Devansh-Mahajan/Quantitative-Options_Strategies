import unittest

from core.manager import build_credit_spread_exit_plan, build_long_option_exit_plan


class ManagerExitPlanTests(unittest.TestCase):
    def test_credit_spread_exit_plan_tightens_for_low_pop(self):
        plan = build_credit_spread_exit_plan(days_to_expiry=18, pop=48.0, implied_risk=0.52)

        self.assertLessEqual(plan.stop_loss, 1.20)
        self.assertGreaterEqual(plan.take_profit, 0.50)
        self.assertGreaterEqual(plan.time_stop_dte, 9)

    def test_long_option_exit_plan_adds_capital_protection(self):
        tail_plan = build_long_option_exit_plan(days_to_expiry=25, is_cornwall=False)
        cornwall_plan = build_long_option_exit_plan(days_to_expiry=25, is_cornwall=True)

        self.assertLess(tail_plan.stop_loss, 0.5)
        self.assertGreater(cornwall_plan.take_profit, tail_plan.take_profit)


if __name__ == "__main__":
    unittest.main()
