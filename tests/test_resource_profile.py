import tempfile
import unittest
from pathlib import Path

from core.resource_profile import build_resource_profile


class ResourceProfileTests(unittest.TestCase):
    def test_large_machine_profile_scales_workers_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = build_resource_profile(
                repo_root=Path(tmp),
                cpu_count=28,
                memory_gb=32.0,
                disk_gb=200.0,
            )

        self.assertEqual(profile.cpu_count, 28)
        self.assertEqual(profile.backtest_workers, 16)
        self.assertGreaterEqual(profile.research_rf_jobs, 6)
        self.assertLessEqual(profile.controller_blas_threads, 6)
        self.assertGreaterEqual(profile.daily_training_max_symbols, 20)


if __name__ == "__main__":
    unittest.main()
