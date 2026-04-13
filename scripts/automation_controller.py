import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class MarketWindow:
    open_hour: int
    open_minute: int
    close_hour: int
    close_minute: int


class AutomationController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.timezone = ZoneInfo(args.timezone)
        self.market_window = MarketWindow(
            open_hour=args.market_open_hour,
            open_minute=args.market_open_minute,
            close_hour=args.market_close_hour,
            close_minute=args.market_close_minute,
        )
        self.execution_lock = asyncio.Lock()
        self.monitor_lock = asyncio.Lock()
        self.state_path = Path(args.state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def now(self) -> datetime:
        return datetime.now(self.timezone)

    def is_market_session(self, ts: Optional[datetime] = None) -> bool:
        current = ts or self.now()
        if current.weekday() >= 5:
            return False

        open_mark = current.replace(
            hour=self.market_window.open_hour,
            minute=self.market_window.open_minute,
            second=0,
            microsecond=0,
        )
        close_mark = current.replace(
            hour=self.market_window.close_hour,
            minute=self.market_window.close_minute,
            second=0,
            microsecond=0,
        )
        return open_mark <= current < close_mark

    async def _run_command(self, label: str, command: str, lock: asyncio.Lock) -> int:
        async with lock:
            print(f"[{self.now().isoformat()}] ▶ {label}: {command}")
            process = await asyncio.create_subprocess_shell(command)
            code = await process.wait()
            if code == 0:
                print(f"[{self.now().isoformat()}] ✅ {label} completed")
            else:
                print(f"[{self.now().isoformat()}] ❌ {label} failed with exit code {code}")
            return code

    async def strategy_loop(self):
        while True:
            if self.is_market_session():
                await self._run_command(
                    "portfolio-manager",
                    self.args.strategy_command,
                    self.execution_lock,
                )
                await asyncio.sleep(max(30, self.args.strategy_interval_seconds))
                continue
            await asyncio.sleep(self.args.poll_seconds)

    async def risk_monitor_loop(self):
        while True:
            await self._run_command(
                "risk-monitor",
                self.args.risk_command,
                self.monitor_lock,
            )
            await asyncio.sleep(max(30, self.args.risk_interval_seconds))

    async def regime_rebalance_loop(self):
        while True:
            if self.is_market_session():
                await self._run_command(
                    "regime-rebalance",
                    self.args.regime_command,
                    self.execution_lock,
                )
                await asyncio.sleep(max(30, self.args.regime_interval_seconds))
                continue
            await asyncio.sleep(self.args.poll_seconds)

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save_state(self, state: dict):
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    async def _run_daily_maintenance_if_due(self, state: dict, now: datetime):
        today_key = now.date().isoformat()
        if state.get("daily_maintenance") == today_key:
            return

        pre_open_mark = now.replace(
            hour=self.args.pre_open_hour,
            minute=self.args.pre_open_minute,
            second=0,
            microsecond=0,
        )
        post_close_mark = now.replace(
            hour=self.args.post_close_hour,
            minute=self.args.post_close_minute,
            second=0,
            microsecond=0,
        )

        if now < pre_open_mark:
            return

        await self._run_command(
            "pre-open-self-check",
            self.args.pre_open_command,
            self.monitor_lock,
        )

        if now < post_close_mark:
            return

        post_close_rc = await self._run_command(
            "post-close-self-evaluate",
            self.args.post_close_eval_command,
            self.execution_lock,
        )
        if post_close_rc == 0:
            tune_rc = await self._run_command(
                "post-close-fine-tune",
                self.args.post_close_tune_command,
                self.execution_lock,
            )
            if tune_rc == 0:
                await self._run_command(
                    "post-close-backtest",
                    self.args.post_close_backtest_command,
                    self.execution_lock,
                )
        state["daily_maintenance"] = today_key
        self._save_state(state)

    async def weekend_pipeline_loop(self):
        while True:
            now = self.now()
            state = self._load_state()

            if now.weekday() < 5:
                await self._run_daily_maintenance_if_due(state, now)

            if now.weekday() not in {5, 6}:
                await asyncio.sleep(self.args.poll_seconds)
                continue

            run_mark = now.replace(
                hour=self.args.weekend_hour,
                minute=self.args.weekend_minute,
                second=0,
                microsecond=0,
            )
            week_key = f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"
            already_ran = state.get("weekend_pipeline") == week_key

            if now >= run_mark and not already_ran:
                recalibrate_rc = await self._run_command(
                    "weekend-recalibration",
                    self.args.weekend_recalibration_command,
                    self.execution_lock,
                )
                if recalibrate_rc == 0:
                    backtest_rc = await self._run_command(
                        "massive-backtest",
                        self.args.weekend_backtest_command,
                        self.execution_lock,
                    )
                    if backtest_rc == 0:
                        self._build_weekend_report()
                state["weekend_pipeline"] = week_key
                self._save_state(state)

            await asyncio.sleep(self.args.poll_seconds)

    async def market_boundary_watch(self):
        previous = self.is_market_session()
        while True:
            await asyncio.sleep(self.args.poll_seconds)
            current = self.is_market_session()
            if current == previous:
                continue

            if current:
                await self._run_command(
                    "market-open-risk-sweep",
                    self.args.open_bell_command,
                    self.monitor_lock,
                )
            else:
                await self._run_command(
                    "market-close-risk-sweep",
                    self.args.close_bell_command,
                    self.monitor_lock,
                )
            previous = current

    def _build_weekend_report(self):
        report_path = Path(self.args.weekend_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        backtest_file = Path(self.args.massive_backtest_report_path)
        if not backtest_file.exists():
            report_path.write_text(
                "# Weekend Professional Report\n\nMassive backtest output not found.\n",
                encoding="utf-8",
            )
            return

        payload = json.loads(backtest_file.read_text(encoding="utf-8"))
        overview = payload.get("massive_overview", {})
        movement = payload.get("movement_suite", {})
        pairs = payload.get("pairs_suite", {})
        regime = payload.get("regime_suite", {})

        lines = [
            "# Weekend Professional Report",
            "",
            f"Generated: {self.now().isoformat()}",
            "",
            "## Executive Summary",
            f"- Predictive score: {overview.get('predictive_score', 'n/a')}",
            f"- Coverage windows: {', '.join(overview.get('lookbacks', [])) or 'n/a'}",
            "",
            "## Model Diagnostics",
            f"- Movement suite accuracy: {movement.get('mean_directional_accuracy', 'n/a')}",
            f"- Regime suite score: {regime.get('regime_score', 'n/a')}",
            f"- Pairs suite win-rate: {pairs.get('win_rate', 'n/a')}",
            "",
            "## Operational Actions",
            "- Confirm runtime calibration artifacts are loaded before Monday open.",
            "- Validate delta/theta targets against PM and risk desk tolerances.",
            "- Keep risk monitor and regime rebalance daemons active through the week.",
        ]
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    async def run(self):
        await asyncio.gather(
            self.strategy_loop(),
            self.risk_monitor_loop(),
            self.regime_rebalance_loop(),
            self.weekend_pipeline_loop(),
            self.market_boundary_watch(),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Always-on automation controller for strategy + risk operations.")
    parser.add_argument("--timezone", default="America/New_York", help="Timezone used for market and weekend schedules.")
    parser.add_argument("--market-open-hour", type=int, default=9)
    parser.add_argument("--market-open-minute", type=int, default=30)
    parser.add_argument("--market-close-hour", type=int, default=16)
    parser.add_argument("--market-close-minute", type=int, default=0)

    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--strategy-interval-seconds", type=int, default=1800)
    parser.add_argument("--risk-interval-seconds", type=int, default=300)
    parser.add_argument("--regime-interval-seconds", type=int, default=900)

    parser.add_argument("--strategy-command", default="run-strategy --strat-log --log-level INFO --log-to-file")
    parser.add_argument("--risk-command", default="run-strategy --manage-only --strat-log --log-level INFO")
    parser.add_argument("--regime-command", default="run-strategy --strat-log --predictor-universe-cap 25 --router-top-k 15")

    parser.add_argument("--open-bell-command", default="run-strategy --manage-only --strat-log --log-level INFO")
    parser.add_argument("--close-bell-command", default="run-strategy --manage-only --strat-log --log-level INFO")

    parser.add_argument("--pre-open-hour", type=int, default=9)
    parser.add_argument("--pre-open-minute", type=int, default=0)
    parser.add_argument("--post-close-hour", type=int, default=16)
    parser.add_argument("--post-close-minute", type=int, default=20)
    parser.add_argument("--pre-open-command", default="run-strategy --manage-only --strat-log --log-level INFO")
    parser.add_argument("--post-close-eval-command", default="quant-foundry --mode zero-calibration")
    parser.add_argument("--post-close-tune-command", default="weekend-recalibrate --target-daily-return 0.002 --target-accuracy 0.56")
    parser.add_argument("--post-close-backtest-command", default="massive-backtest")

    parser.add_argument("--weekend-hour", type=int, default=8)
    parser.add_argument("--weekend-minute", type=int, default=0)
    parser.add_argument("--weekend-recalibration-command", default="weekend-recalibrate --target-daily-return 0.002 --target-accuracy 0.56")
    parser.add_argument("--weekend-backtest-command", default="massive-backtest")
    parser.add_argument("--weekend-report-path", default="reports/weekend_professional_report.md")
    parser.add_argument("--massive-backtest-report-path", default="reports/massive_backtest_report.json")
    parser.add_argument("--state-path", default=".runtime/automation_state.json")
    parser.add_argument(
        "--restart-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restart the 24/7 controller automatically if it crashes.",
    )
    parser.add_argument(
        "--restart-delay-seconds",
        type=int,
        default=15,
        help="Wait time before restarting after an unexpected crash.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    while True:
        controller = AutomationController(args)
        try:
            asyncio.run(controller.run())
            return
        except KeyboardInterrupt:
            print(f"[{datetime.now(ZoneInfo(args.timezone)).isoformat()}] Received Ctrl+C. Exiting automate-stack.")
            return
        except Exception as exc:
            print(f"[{datetime.now(ZoneInfo(args.timezone)).isoformat()}] automate-stack crashed: {exc}")
            if not args.restart_on_failure:
                raise
            print(
                f"[{datetime.now(ZoneInfo(args.timezone)).isoformat()}] "
                f"Restarting in {args.restart_delay_seconds} seconds..."
            )
            asyncio.run(asyncio.sleep(max(1, args.restart_delay_seconds)))


if __name__ == "__main__":
    main()
