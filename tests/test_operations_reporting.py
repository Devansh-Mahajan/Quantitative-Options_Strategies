import json
import tempfile
import unittest
from pathlib import Path

from core.operations_reporting import archive_backtest_artifacts, write_daily_ops_report


class OperationsReportingTests(unittest.TestCase):
    def test_archive_backtest_artifacts_writes_latest_and_archive_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports_root = root / "reports"
            reports_root.mkdir(parents=True, exist_ok=True)
            output_path = reports_root / "massive_backtest_report.json"
            payload = {
                "massive_overview": {
                    "predictive_score": 0.72,
                    "consensus_market_state": "calm_bull",
                    "consensus_strategy_profile": "bull_trend",
                },
                "movement_suite": {"summary": {"avg_accuracy": 0.58}},
            }

            paths = archive_backtest_artifacts(payload, output_path=output_path, reports_root=reports_root)

            self.assertTrue(Path(paths["latest_json"]).exists())
            self.assertTrue(Path(paths["latest_md"]).exists())
            self.assertTrue(Path(paths["archive_json"]).exists())
            self.assertTrue(Path(paths["archive_md"]).exists())

    def test_write_daily_ops_report_writes_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_root = root / ".runtime"
            runtime_root.mkdir(parents=True, exist_ok=True)
            (runtime_root / "execution_quality_snapshot.json").write_text(
                json.dumps(
                    {
                        "records": 12,
                        "fill_rate": 0.75,
                        "full_fill_rate": 0.50,
                        "broker_fill_price_coverage": 0.83,
                        "avg_execution_quality_score": 0.68,
                        "adaptive_reprice_factor": 1.04,
                        "degraded_execution_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            report_path = root / "reports" / "daily" / "daily_automation_report_2026-04-13.md"
            latest_path = root / "reports" / "daily" / "latest_daily_automation_report.md"

            payload = write_daily_ops_report(
                repo_root=root,
                report_path=report_path,
                latest_report_path=latest_path,
                context={"post_close_eval_rc": 0},
            )

            self.assertEqual(payload["context"]["post_close_eval_rc"], 0)
            self.assertEqual(payload["execution"]["records"], 12)
            self.assertEqual(payload["execution"]["adaptive_reprice_factor"], 1.04)
            self.assertTrue(report_path.exists())
            self.assertTrue(latest_path.exists())
            self.assertTrue(latest_path.with_suffix(".json").exists())


if __name__ == "__main__":
    unittest.main()
