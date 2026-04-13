from argparse import Namespace
from datetime import datetime
from pathlib import Path
import sys
import asyncio
from zoneinfo import ZoneInfo

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.automation_controller import AutomationController
from scripts.automation_controller import parse_args


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
        strategy_command="run-strategy",
        risk_command="run-strategy --manage-only",
        regime_command="run-strategy",
        open_bell_command="run-strategy --manage-only",
        close_bell_command="run-strategy --manage-only",
        pre_open_hour=9,
        pre_open_minute=0,
        post_close_hour=16,
        post_close_minute=20,
        pre_open_command="run-strategy --manage-only",
        post_close_eval_command="quant-foundry --mode zero-calibration",
        post_close_tune_command="weekend-recalibrate",
        post_close_backtest_command="massive-backtest",
        weekend_hour=8,
        weekend_minute=0,
        weekend_recalibration_command="weekend-recalibrate",
        weekend_backtest_command="massive-backtest",
        weekend_report_path=str(tmp_path / "weekend_professional_report.md"),
        massive_backtest_report_path=str(tmp_path / "massive_backtest_report.json"),
        state_path=str(tmp_path / "automation_state.json"),
    )


def test_market_session_detection(tmp_path):
    controller = AutomationController(_args(tmp_path))
    tz = ZoneInfo("America/New_York")

    during = datetime(2026, 4, 13, 10, 0, tzinfo=tz)  # Monday
    pre = datetime(2026, 4, 13, 8, 59, tzinfo=tz)
    weekend = datetime(2026, 4, 12, 11, 0, tzinfo=tz)  # Sunday

    assert controller.is_market_session(during)
    assert not controller.is_market_session(pre)
    assert not controller.is_market_session(weekend)


def test_weekend_report_generation(tmp_path):
    args = _args(tmp_path)
    controller = AutomationController(args)

    payload = {
        "massive_overview": {
            "predictive_score": 0.73,
            "lookbacks": ["5y", "1y"],
        },
        "movement_suite": {"mean_directional_accuracy": 0.58},
        "regime_suite": {"regime_score": 0.61},
        "pairs_suite": {"win_rate": 0.54},
    }
    Path(args.massive_backtest_report_path).write_text(__import__("json").dumps(payload), encoding="utf-8")

    controller._build_weekend_report()
    report_text = Path(args.weekend_report_path).read_text(encoding="utf-8")

    assert "Weekend Professional Report" in report_text
    assert "Predictive score: 0.73" in report_text
    assert "Movement suite accuracy: 0.58" in report_text


def test_daily_maintenance_runs_after_post_close(tmp_path):
    args = _args(tmp_path)
    controller = AutomationController(args)
    executed = []

    async def _fake_run_command(label, command, lock):
        executed.append((label, command))
        return 0

    controller._run_command = _fake_run_command
    state = {}
    now = datetime(2026, 4, 13, 16, 30, tzinfo=ZoneInfo("America/New_York"))

    asyncio.run(controller._run_daily_maintenance_if_due(state, now))
    assert state["daily_maintenance"] == "2026-04-13"
    assert [name for name, _ in executed] == [
        "pre-open-self-check",
        "post-close-self-evaluate",
        "post-close-fine-tune",
        "post-close-backtest",
    ]


def test_parse_args_restart_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["automate-stack"])
    args = parse_args()
    assert args.restart_on_failure is True
    assert args.restart_delay_seconds == 15
