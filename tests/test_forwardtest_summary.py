import unittest

import numpy as np

from dashboard.server import _summarize_forwardtest_result


class _StubFuturetestResult:
    def __init__(self):
        self.n_paths = 4
        self.failed_paths = 1
        self.acceptance_rate = 0.5
        self.return_dist = np.array([0.10, -0.05, 0.00, 0.20])
        self.sharpe_dist = np.array([1.2, -0.8, 0.2, 1.8])
        self.max_dd_dist = np.array([0.03, 0.12, 0.07, 0.02])
        self.win_rate_dist = np.array([0.60, 0.30, 0.50, 0.75])


class ForwardtestSummaryTests(unittest.TestCase):
    def test_summary_uses_distribution_arrays_without_results_attribute(self):
        result = _summarize_forwardtest_result(_StubFuturetestResult())

        self.assertEqual(result["n_paths"], 4)
        self.assertEqual(result["failed_paths"], 1)
        self.assertEqual(result["acceptance_rate"], 50.0)
        self.assertEqual(result["median_return_pct"], 5.0)
        self.assertEqual(result["mean_return_pct"], 6.25)
        self.assertEqual(result["median_sharpe"], 0.7)
        self.assertEqual(result["median_maxdd_pct"], 5.0)
        self.assertEqual(result["median_win_rate"], 55.0)
        self.assertEqual(result["profitable_pct"], 50.0)
        self.assertEqual(len(result["return_hist"]["counts"]), 20)
        self.assertEqual(len(result["sharpe_hist"]["bins"]), 20)


if __name__ == "__main__":
    unittest.main()
