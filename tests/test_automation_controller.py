import asyncio
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from scripts.automation_controller import AutomationController, parse_args


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        timezone="America/New_York",
        market_open_hour=9,
        market_open_minute=30,
        market_close_hour=16,
        market_close_minute=0,
        poll_seconds=60,
        strategy_interval_seconds=1800,
        risk_interval_seconds=300,
        regime_interval_seconds=900,
        overnight_training_interval_seconds=1800,
        weekend_training_interval_seconds=1800,
        burst_interval_seconds=45,
        regime_watch_interval_seconds=120,
        open_burst_minutes=12,
        close_burst_minutes=12,
        strategy_command="run-strategy",
        risk_command="run-strategy --manage-only",
        regime_command="run-strategy",
        open_bell_command="run-strategy --manage-only",
        market_open_deploy_command="run-strategy",
        close_bell_command="run-strategy --manage-only",
        market_close_rebalance_command="run-strategy --manage-only",
        critical_window_command="run-strategy --manage-only",
        regime_probe_command="python -m scripts.regime_probe",
        regime_shift_command="run-strategy --manage-only",
        pre_open_hour=9,
        pre_open_minute=0,
        post_close_hour=16,
        post_close_minute=20,
        pre_open_command="run-strategy --manage-only",
        post_close_eval_command="quant-foundry --mode zero-calibration",
        post_close_train_command="python -m scripts.model_maintenance --mode daily",
        post_close_tune_command="quant-foundry --mode weekend-calibrate",
        post_close_backtest_command="massive-backtest",
        weekend_hour=8,
        weekend_minute=0,
        weekend_recalibration_command="weekend-recalibrate",
        weekend_backtest_command="massive-backtest",
        daily_report_dir=str(tmp_path / "daily"),
        latest_daily_report_path=str(tmp_path / "daily" / "latest_daily_automation_report.md"),
        weekend_report_path=str(tmp_path / "weekend_professional_report.md"),
        massive_backtest_report_path=str(tmp_path / "massive_backtest_report.json"),
        state_path=str(tmp_path / "automation_state.json"),
        preflight_state_path=str(tmp_path / "preflight_state.json"),
        system_snapshot_path=str(tmp_path / "system_resource_snapshot.json"),
        preflight_max_age_seconds=300,
        skip_preflight=False,
        deep_model_checks=True,
        progress_ui=True,
        telemetry_interval_seconds=300,
        restart_on_failure=True,
        restart_delay_seconds=15,
    )


class AutomationControllerTests(unittest.TestCase):
    def test_market_session_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = AutomationController(_args(Path(tmp)))
            tz = ZoneInfo("America/New_York")

            during = datetime(2026, 4, 13, 10, 0, tzinfo=tz)
            pre = datetime(2026, 4, 13, 8, 59, tzinfo=tz)
            weekend = datetime(2026, 4, 12, 11, 0, tzinfo=tz)

            self.assertTrue(controller.is_market_session(during))
            self.assertFalse(controller.is_market_session(pre))
            self.assertFalse(controller.is_market_session(weekend))

    def test_critical_window_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = AutomationController(_args(Path(tmp)))
            tz = ZoneInfo("America/New_York")

            just_after_open = datetime(2026, 4, 13, 9, 35, tzinfo=tz)
            midday = datetime(2026, 4, 13, 12, 0, tzinfo=tz)
            just_before_close = datetime(2026, 4, 13, 15, 55, tzinfo=tz)

            self.assertTrue(controller._is_critical_window(just_after_open))
            self.assertFalse(controller._is_critical_window(midday))
            self.assertTrue(controller._is_critical_window(just_before_close))

    def test_offhours_cycle_kind_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = AutomationController(_args(Path(tmp)))
            tz = ZoneInfo("America/New_York")

            overnight = datetime(2026, 4, 13, 20, 0, tzinfo=tz)
            pre_open = datetime(2026, 4, 14, 8, 0, tzinfo=tz)
            weekend = datetime(2026, 4, 12, 11, 0, tzinfo=tz)
            session = datetime(2026, 4, 13, 11, 0, tzinfo=tz)

            self.assertEqual(controller._offhours_cycle_kind(overnight), "overnight")
            self.assertEqual(controller._offhours_cycle_kind(pre_open), "overnight")
            self.assertEqual(controller._offhours_cycle_kind(weekend), "weekend")
            self.assertIsNone(controller._offhours_cycle_kind(session))

    def test_weekend_report_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(Path(tmp))
            controller = AutomationController(args)

            payload = {
                "massive_overview": {
                    "predictive_score": 0.73,
                    "consensus_state": "calm_bull",
                    "consensus_profile": "bull_trend",
                },
                "movement_suite": {"summary": {"avg_accuracy": 0.58}},
                "regime_suite": {"summary": {"directional_accuracy_proxy": 0.61}},
                "pairs_suite": {"summary": {"win_rate": 0.54}},
                "strategy_profile_suite": {"summary": {"consensus_state": "calm_bull", "consensus_profile": "bull_trend"}},
            }
            Path(args.massive_backtest_report_path).write_text(json.dumps(payload), encoding="utf-8")

            controller._build_weekend_report()
            report_text = Path(args.weekend_report_path).read_text(encoding="utf-8")

            self.assertIn("Weekend Professional Report", report_text)
            self.assertIn("Predictive score: 0.73", report_text)
            self.assertIn("Consensus market state: calm_bull", report_text)
            self.assertIn("Consensus live strategy profile: bull_trend", report_text)

    def test_daily_report_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(Path(tmp))
            controller = AutomationController(args)
            now = datetime(2026, 4, 13, 16, 45, tzinfo=ZoneInfo("America/New_York"))

            controller._build_daily_report(now, context={"post_close_eval_rc": 0})
            latest_text = Path(args.latest_daily_report_path).read_text(encoding="utf-8")

            self.assertIn("Daily Automation Report", latest_text)
            self.assertIn("Portfolio And Risk", latest_text)

    def test_daily_maintenance_runs_after_post_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(Path(tmp))
            controller = AutomationController(args)
            executed = []

            async def _fake_run_command(label, command, lock):
                executed.append((label, command))
                return 0

            controller._run_command = _fake_run_command
            state = {}
            now = datetime(2026, 4, 13, 16, 30, tzinfo=ZoneInfo("America/New_York"))

            asyncio.run(controller._run_daily_maintenance_if_due(state, now))
            self.assertEqual(state["daily_maintenance"], "2026-04-13")
            self.assertEqual(
                [name for name, _ in executed],
                [
                    "pre-open-self-check",
                    "post-close-self-evaluate",
                    "post-close-model-maintenance",
                ],
            )
            self.assertTrue(Path(args.latest_daily_report_path).exists())

    def test_daily_maintenance_runs_friday_backtest_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(Path(tmp))
            controller = AutomationController(args)
            executed = []

            async def _fake_run_command(label, command, lock):
                executed.append((label, command))
                return 0

            controller._run_command = _fake_run_command
            state = {}
            now = datetime(2026, 4, 17, 16, 30, tzinfo=ZoneInfo("America/New_York"))  # Friday

            asyncio.run(controller._run_daily_maintenance_if_due(state, now))
            self.assertEqual(
                [name for name, _ in executed],
                [
                    "pre-open-self-check",
                    "post-close-self-evaluate",
                    "post-close-model-maintenance",
                    "post-close-fine-tune",
                    "post-close-backtest",
                ],
            )
            self.assertTrue(Path(args.latest_daily_report_path).exists())

    def test_daily_maintenance_retries_when_post_close_training_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(Path(tmp))
            controller = AutomationController(args)
            executed = []

            async def _fake_run_command(label, command, lock):
                executed.append((label, command))
                if label == "post-close-model-maintenance":
                    return 1
                return 0

            controller._run_command = _fake_run_command
            state = {}
            now = datetime(2026, 4, 13, 16, 30, tzinfo=ZoneInfo("America/New_York"))

            asyncio.run(controller._run_daily_maintenance_if_due(state, now))

            self.assertNotIn("daily_maintenance", state)
            self.assertEqual(
                [name for name, _ in executed],
                [
                    "pre-open-self-check",
                    "post-close-self-evaluate",
                    "post-close-model-maintenance",
                ],
            )
            self.assertTrue(Path(args.latest_daily_report_path).exists())

    def test_offhours_training_cycle_runs_continuous_weekday_training_after_daily_maintenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(Path(tmp))
            controller = AutomationController(args)
            executed = []

            async def _fake_run_command(label, command, lock):
                executed.append((label, command))
                return 0

            controller._run_command = _fake_run_command
            state = {
                "pre_open_self_check": "2026-04-13",
                "daily_maintenance": "2026-04-13",
            }
            now = datetime(2026, 4, 13, 20, 30, tzinfo=ZoneInfo("America/New_York"))

            cycle_kind = asyncio.run(controller._run_offhours_training_cycle(state, now))

            self.assertEqual(cycle_kind, "overnight")
            self.assertEqual(
                [name for name, _ in executed],
                ["overnight-model-maintenance"],
            )

    def test_offhours_training_cycle_runs_weekend_research_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(Path(tmp))
            controller = AutomationController(args)
            executed = []

            async def _fake_run_command(label, command, lock):
                executed.append((label, command))
                return 0

            controller._run_command = _fake_run_command
            controller._build_weekend_report = lambda: executed.append(("weekend-report", "built"))
            state = {}
            now = datetime(2026, 4, 12, 11, 0, tzinfo=ZoneInfo("America/New_York"))

            cycle_kind = asyncio.run(controller._run_offhours_training_cycle(state, now))

            self.assertEqual(cycle_kind, "weekend")
            self.assertEqual(
                [name for name, _ in executed],
                [
                    "weekend-recalibration",
                    "massive-backtest",
                    "weekend-report",
                ],
            )

    def test_pre_open_self_check_runs_once_per_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(Path(tmp))
            controller = AutomationController(args)
            executed = []

            async def _fake_run_command(label, command, lock):
                executed.append((label, command))
                return 0

            controller._run_command = _fake_run_command
            state = {}
            now = datetime(2026, 4, 13, 10, 0, tzinfo=ZoneInfo("America/New_York"))

            asyncio.run(controller._run_daily_maintenance_if_due(state, now))
            asyncio.run(controller._run_daily_maintenance_if_due(state, now))

            self.assertEqual(state["pre_open_self_check"], "2026-04-13")
            self.assertEqual([name for name, _ in executed], ["pre-open-self-check"])

    def test_parse_args_restart_defaults(self):
        with patch.object(sys, "argv", ["automate-stack"]):
            args = parse_args()
        self.assertTrue(args.restart_on_failure)
        self.assertEqual(args.restart_delay_seconds, 15)
        self.assertEqual(args.preflight_max_age_seconds, 300)
        self.assertEqual(args.overnight_training_interval_seconds, 1800)
        self.assertEqual(args.weekend_training_interval_seconds, 1800)

    def test_parse_args_default_commands_use_python_modules(self):
        with patch.object(sys, "argv", ["automate-stack"]):
            args = parse_args()

        self.assertIn(sys.executable, args.strategy_command)
        self.assertIn("scripts.run_strategy", args.strategy_command)
        self.assertIn("scripts.quant_research_foundry", args.post_close_eval_command)
        self.assertIn("scripts.model_maintenance", args.post_close_train_command)
        self.assertIn("scripts.quant_research_foundry", args.post_close_tune_command)
        self.assertIn("scripts.massive_backtest_engine", args.weekend_backtest_command)


if __name__ == "__main__":
    unittest.main()
