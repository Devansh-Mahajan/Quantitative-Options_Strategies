import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shlex
import sys
from typing import Optional
from zoneinfo import ZoneInfo

from core.operations_reporting import write_daily_ops_report
from core.resource_profile import load_resource_profile
from core.system_preflight import run_preflight
from core.system_telemetry import DEFAULT_SYSTEM_SNAPSHOT_PATH, write_system_resource_snapshot
from core.terminal_ui import format_status_line

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MarketWindow:
    open_hour: int
    open_minute: int
    close_hour: int
    close_minute: int


class AutomationController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = REPO_ROOT
        self.timezone = ZoneInfo(args.timezone)
        self.market_window = MarketWindow(
            open_hour=args.market_open_hour,
            open_minute=args.market_open_minute,
            close_hour=args.market_close_hour,
            close_minute=args.market_close_minute,
        )
        self.execution_lock = asyncio.Lock()
        self.monitor_lock = asyncio.Lock()
        self.preflight_lock = asyncio.Lock()
        self.resource_profile = load_resource_profile(self.repo_root)
        self.process_env = self._build_process_env()
        self.state_path = self._resolve_repo_path(args.state_path)
        self.preflight_state_path = self._resolve_repo_path(args.preflight_state_path)
        self.system_snapshot_path = self._resolve_repo_path(args.system_snapshot_path)
        self.daily_report_dir = self._resolve_repo_path(args.daily_report_dir)
        self.latest_daily_report_path = self._resolve_repo_path(args.latest_daily_report_path)
        self.weekend_report_path = self._resolve_repo_path(args.weekend_report_path)
        self.massive_backtest_report_path = self._resolve_repo_path(args.massive_backtest_report_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_report_dir.mkdir(parents=True, exist_ok=True)
        self.weekend_report_path.parent.mkdir(parents=True, exist_ok=True)

    def _resolve_repo_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self.repo_root / path

    def _build_process_env(self) -> dict[str, str]:
        environment = os.environ.copy()

        venv_bin = self.repo_root / ".venv" / "bin"
        if venv_bin.exists():
            current_path = environment.get("PATH", "")
            environment["PATH"] = (
                f"{venv_bin}{os.pathsep}{current_path}" if current_path else str(venv_bin)
            )

        current_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            f"{self.repo_root}{os.pathsep}{current_pythonpath}"
            if current_pythonpath
            else str(self.repo_root)
        )
        environment["PYTHONUNBUFFERED"] = "1"
        environment.update(self.resource_profile.to_env())
        return environment

    def now(self) -> datetime:
        return datetime.now(self.timezone)

    def _minutes_from_open(self, ts: Optional[datetime] = None) -> float:
        current = ts or self.now()
        open_mark = current.replace(
            hour=self.market_window.open_hour,
            minute=self.market_window.open_minute,
            second=0,
            microsecond=0,
        )
        return (current - open_mark).total_seconds() / 60.0

    def _minutes_to_close(self, ts: Optional[datetime] = None) -> float:
        current = ts or self.now()
        close_mark = current.replace(
            hour=self.market_window.close_hour,
            minute=self.market_window.close_minute,
            second=0,
            microsecond=0,
        )
        return (close_mark - current).total_seconds() / 60.0

    def _is_critical_window(self, ts: Optional[datetime] = None) -> bool:
        current = ts or self.now()
        if not self.is_market_session(current):
            return False
        return (
            self._minutes_from_open(current) <= max(1, self.args.open_burst_minutes)
            or self._minutes_to_close(current) <= max(1, self.args.close_burst_minutes)
        )

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

    def _emit_console(self, message: str) -> None:
        print(f"[{self.now().isoformat()}] {message}")

    def _emit_resource_banner(self) -> None:
        self._emit_console(
            format_status_line(
                "resource-profile",
                "active",
                cpu=self.resource_profile.cpu_count,
                memory_gb=f"{self.resource_profile.memory_gb:.1f}",
                disk_gb=f"{self.resource_profile.disk_gb:.1f}",
                backtest_workers=self.resource_profile.backtest_workers,
                rf_jobs=self.resource_profile.research_rf_jobs,
                blas_threads=self.resource_profile.controller_blas_threads,
            )
        )

    def _preflight_progress(self, percent: int, message: str, detail: str | None = None) -> None:
        suffix = f" | {detail}" if detail else ""
        self._emit_console(f"PRECHECK {percent:3d}% | {message}{suffix}")

    async def _ensure_preflight_ok(self, label: str) -> bool:
        if self.args.skip_preflight:
            return True

        async with self.preflight_lock:
            result = await asyncio.to_thread(
                run_preflight,
                root=self.repo_root,
                state_path=self.preflight_state_path,
                max_age_seconds=self.args.preflight_max_age_seconds,
                progress_callback=None if not self.args.progress_ui else self._preflight_progress,
                deep_model_checks=self.args.deep_model_checks,
            )
            if not result.ok:
                self._emit_console(f"⛔ Preflight blocked {label}: {result.summary}")
                for issue in result.issues:
                    if issue.severity == "error":
                        self._emit_console(f"   {issue.check}: {issue.detail}")
                return False
            if not result.skipped:
                self._emit_console(f"✅ {result.summary}")
            return True

    async def _stream_pipe(self, label: str, stream: asyncio.StreamReader | None, pipe_name: str) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                self._emit_console(f"{label} {pipe_name} | {text}")

    async def _run_command(self, label: str, command: str, lock: asyncio.Lock) -> int:
        async with lock:
            if not await self._ensure_preflight_ok(label):
                return 2

            started = time.monotonic()
            self._emit_console(format_status_line(label, "start", command=command))
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.repo_root),
                env=self.process_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_task = asyncio.create_task(self._stream_pipe(label, process.stdout, "stdout"))
            stderr_task = asyncio.create_task(self._stream_pipe(label, process.stderr, "stderr"))
            code = await process.wait()
            await asyncio.gather(stdout_task, stderr_task)
            duration = time.monotonic() - started
            if code == 0:
                self._emit_console(format_status_line(label, "done", exit_code=code, seconds=f"{duration:.1f}"))
            else:
                self._emit_console(format_status_line(label, "failed", exit_code=code, seconds=f"{duration:.1f}"))
            return code

    async def _capture_json_command(self, label: str, command: str) -> dict:
        if not await self._ensure_preflight_ok(label):
            return {}

        self._emit_console(format_status_line(label, "capture", command=command))
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self.repo_root),
            env=self.process_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            self._emit_console(format_status_line(label, "failed", exit_code=process.returncode))
            if stderr:
                self._emit_console(f"{label} stderr | {stderr.decode('utf-8', errors='replace').strip()}")
            return {}
        raw_text = stdout.decode("utf-8", errors="replace").strip().splitlines()
        if not raw_text:
            return {}
        try:
            payload = json.loads(raw_text[-1])
        except json.JSONDecodeError:
            self._emit_console(f"⚠ {label}: failed to parse JSON payload")
            return {}
        self._emit_console(format_status_line(label, "done", exit_code=0))
        return payload if isinstance(payload, dict) else {}

    async def strategy_loop(self):
        while True:
            if self.is_market_session():
                state = self._load_state()
                today_key = self.now().date().isoformat()
                if (
                    state.get("market_open_dispatch") == today_key
                    and self._minutes_from_open() <= max(1, self.args.open_burst_minutes)
                ):
                    await asyncio.sleep(max(15, self.args.burst_interval_seconds))
                    continue
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

    async def critical_window_loop(self):
        while True:
            if self._is_critical_window():
                await self._run_command(
                    "critical-window-monitor",
                    self.args.critical_window_command,
                    self.monitor_lock,
                )
                await asyncio.sleep(max(15, self.args.burst_interval_seconds))
                continue
            await asyncio.sleep(self.args.poll_seconds)

    async def regime_shift_watch_loop(self):
        while True:
            if not self.is_market_session():
                await asyncio.sleep(self.args.poll_seconds)
                continue

            payload = await self._capture_json_command("regime-probe", self.args.regime_probe_command)
            if not payload:
                await asyncio.sleep(max(30, self.args.regime_watch_interval_seconds))
                continue

            state = self._load_state()
            current_signature = {
                "macro_strategy": payload.get("macro_strategy"),
                "runtime_regime": payload.get("runtime_regime"),
                "runtime_market_state": payload.get("runtime_market_state"),
                "runtime_profile": payload.get("runtime_profile"),
            }
            previous_signature = state.get("regime_signature")
            state["regime_signature"] = current_signature
            self._save_state(state)

            if previous_signature and previous_signature != current_signature:
                await self._run_command(
                    "regime-shift-rebalance",
                    self.args.regime_shift_command,
                    self.execution_lock,
                )

            await asyncio.sleep(max(30, self.args.regime_watch_interval_seconds))

    async def resource_telemetry_loop(self):
        while True:
            snapshot = await asyncio.to_thread(
                write_system_resource_snapshot,
                self.system_snapshot_path,
                repo_root=self.repo_root,
                profile=self.resource_profile,
            )
            host_metrics = snapshot.get("host_metrics", {})
            pressure = (snapshot.get("status") or {}).get("pressure", "unknown")
            self._emit_console(
                format_status_line(
                    "system-telemetry",
                    pressure,
                    load_1m=host_metrics.get("loadavg_1m"),
                    cpu_load_pct=host_metrics.get("normalized_cpu_load_pct"),
                    mem_pct=(host_metrics.get("memory") or {}).get("usage_pct"),
                    disk_pct=(host_metrics.get("disk") or {}).get("usage_pct"),
                )
            )
            await asyncio.sleep(max(60, self.args.telemetry_interval_seconds))

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

        if state.get("pre_open_self_check") != today_key:
            pre_open_rc = await self._run_command(
                "pre-open-self-check",
                self.args.pre_open_command,
                self.monitor_lock,
            )
            if pre_open_rc == 0:
                state["pre_open_self_check"] = today_key
                self._save_state(state)

        if now < post_close_mark:
            return

        if state.get("daily_maintenance") == today_key:
            return
        maintenance_context = {
            "date": today_key,
            "post_close_eval_rc": None,
            "post_close_train_rc": None,
            "post_close_tune_rc": None,
            "post_close_backtest_rc": None,
        }

        post_close_rc = await self._run_command(
            "post-close-self-evaluate",
            self.args.post_close_eval_command,
            self.execution_lock,
        )
        maintenance_context["post_close_eval_rc"] = post_close_rc
        if post_close_rc == 0:
            train_rc = await self._run_command(
                "post-close-model-maintenance",
                self.args.post_close_train_command,
                self.execution_lock,
            )
            maintenance_context["post_close_train_rc"] = train_rc
            if train_rc == 0 and now.weekday() == 4:
                tune_rc = await self._run_command(
                    "post-close-fine-tune",
                    self.args.post_close_tune_command,
                    self.execution_lock,
                )
                maintenance_context["post_close_tune_rc"] = tune_rc
                if tune_rc == 0:
                    backtest_rc = await self._run_command(
                        "post-close-backtest",
                        self.args.post_close_backtest_command,
                        self.execution_lock,
                    )
                    maintenance_context["post_close_backtest_rc"] = backtest_rc
        self._build_daily_report(now, context=maintenance_context)
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
                await asyncio.gather(
                    self._run_command(
                        "market-open-risk-sweep",
                        self.args.open_bell_command,
                        self.monitor_lock,
                    ),
                    self._run_command(
                        "market-open-deploy",
                        self.args.market_open_deploy_command,
                        self.execution_lock,
                    ),
                )
                state = self._load_state()
                state["market_open_dispatch"] = self.now().date().isoformat()
                self._save_state(state)
            else:
                await asyncio.gather(
                    self._run_command(
                        "market-close-risk-sweep",
                        self.args.close_bell_command,
                        self.monitor_lock,
                    ),
                    self._run_command(
                        "market-close-rebalance",
                        self.args.market_close_rebalance_command,
                        self.execution_lock,
                    ),
                )
            previous = current

    def _build_weekend_report(self):
        report_path = self.weekend_report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)

        backtest_file = self.massive_backtest_report_path
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
        strategy_profiles = payload.get("strategy_profile_suite", {})
        current_state = (strategy_profiles.get("summary") or {}).get("consensus_state", "n/a")
        current_profile = (strategy_profiles.get("summary") or {}).get("consensus_profile", "n/a")

        lines = [
            "# Weekend Professional Report",
            "",
            f"Generated: {self.now().isoformat()}",
            "",
            "## Executive Summary",
            f"- Predictive score: {overview.get('predictive_score', 'n/a')}",
            f"- Consensus market state: {current_state}",
            f"- Consensus live strategy profile: {current_profile}",
            "",
            "## Model Diagnostics",
            f"- Movement suite accuracy: {(movement.get('summary') or {}).get('avg_accuracy', 'n/a')}",
            f"- Regime suite score: {(regime.get('summary') or {}).get('directional_accuracy_proxy', 'n/a')}",
            f"- Pairs suite win-rate: {(pairs.get('summary') or {}).get('win_rate', 'n/a')}",
            "",
            "## Operational Actions",
            "- Weekend recalibration and the backtest engine are chained automatically inside the 24/7 automation controller.",
            "- Confirm runtime calibration artifacts are loaded before Monday open.",
            "- Validate delta/theta targets against PM and risk desk tolerances.",
            "- Keep portfolio manager, risk monitor, critical-window monitor, and regime-shift daemons active through the week.",
        ]
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _build_daily_report(self, now: datetime, context: dict | None = None):
        report_path = self.daily_report_dir / f"daily_automation_report_{now.date().isoformat()}.md"
        write_daily_ops_report(
            repo_root=self.repo_root,
            report_path=report_path,
            latest_report_path=self.latest_daily_report_path,
            context=context,
        )

    async def run(self):
        self._emit_resource_banner()
        await asyncio.gather(
            self.strategy_loop(),
            self.risk_monitor_loop(),
            self.regime_rebalance_loop(),
            self.critical_window_loop(),
            self.regime_shift_watch_loop(),
            self.resource_telemetry_loop(),
            self.weekend_pipeline_loop(),
            self.market_boundary_watch(),
        )


def _python_module_command(module_name: str, *args: str) -> str:
    command_parts = [sys.executable, "-m", module_name, *args]
    return " ".join(shlex.quote(str(part)) for part in command_parts)


def parse_args() -> argparse.Namespace:
    resource_profile = load_resource_profile(REPO_ROOT)
    parser = argparse.ArgumentParser(description="Always-on automation controller for strategy + risk operations.")
    parser.add_argument("--timezone", default="America/New_York", help="Timezone used for market and weekend schedules.")
    parser.add_argument("--market-open-hour", type=int, default=9)
    parser.add_argument("--market-open-minute", type=int, default=30)
    parser.add_argument("--market-close-hour", type=int, default=16)
    parser.add_argument("--market-close-minute", type=int, default=0)

    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--strategy-interval-seconds", type=int, default=resource_profile.strategy_interval_seconds)
    parser.add_argument("--risk-interval-seconds", type=int, default=resource_profile.risk_interval_seconds)
    parser.add_argument("--regime-interval-seconds", type=int, default=resource_profile.regime_interval_seconds)
    parser.add_argument("--burst-interval-seconds", type=int, default=45)
    parser.add_argument("--regime-watch-interval-seconds", type=int, default=120)
    parser.add_argument("--telemetry-interval-seconds", type=int, default=resource_profile.telemetry_interval_seconds)
    parser.add_argument("--open-burst-minutes", type=int, default=12)
    parser.add_argument("--close-burst-minutes", type=int, default=12)

    parser.add_argument(
        "--strategy-command",
        default=_python_module_command(
            "scripts.run_strategy",
            "--strat-log",
            "--log-level",
            "INFO",
            "--log-to-file",
        ),
    )
    parser.add_argument(
        "--risk-command",
        default=_python_module_command(
            "scripts.run_strategy",
            "--manage-only",
            "--strat-log",
            "--log-level",
            "INFO",
        ),
    )
    parser.add_argument(
        "--regime-command",
        default=_python_module_command(
            "scripts.run_strategy",
            "--strat-log",
            "--predictor-universe-cap",
            "25",
            "--router-top-k",
            "15",
        ),
    )

    parser.add_argument(
        "--open-bell-command",
        default=_python_module_command(
            "scripts.run_strategy",
            "--manage-only",
            "--strat-log",
            "--log-level",
            "INFO",
        ),
    )
    parser.add_argument(
        "--market-open-deploy-command",
        default=_python_module_command(
            "scripts.run_strategy",
            "--strat-log",
            "--log-level",
            "INFO",
            "--log-to-file",
        ),
    )
    parser.add_argument(
        "--close-bell-command",
        default=_python_module_command(
            "scripts.run_strategy",
            "--manage-only",
            "--strat-log",
            "--log-level",
            "INFO",
        ),
    )
    parser.add_argument(
        "--market-close-rebalance-command",
        default=_python_module_command(
            "scripts.run_strategy",
            "--manage-only",
            "--strat-log",
            "--log-level",
            "INFO",
        ),
    )
    parser.add_argument(
        "--critical-window-command",
        default=_python_module_command(
            "scripts.run_strategy",
            "--manage-only",
            "--strat-log",
            "--log-level",
            "INFO",
        ),
    )
    parser.add_argument(
        "--regime-probe-command",
        default=_python_module_command("scripts.regime_probe"),
    )
    parser.add_argument(
        "--regime-shift-command",
        default=_python_module_command(
            "scripts.run_strategy",
            "--manage-only",
            "--strat-log",
            "--log-level",
            "INFO",
        ),
    )

    parser.add_argument("--pre-open-hour", type=int, default=9)
    parser.add_argument("--pre-open-minute", type=int, default=0)
    parser.add_argument("--post-close-hour", type=int, default=16)
    parser.add_argument("--post-close-minute", type=int, default=20)
    parser.add_argument(
        "--pre-open-command",
        default=_python_module_command(
            "scripts.run_strategy",
            "--manage-only",
            "--strat-log",
            "--log-level",
            "INFO",
        ),
    )
    parser.add_argument(
        "--post-close-eval-command",
        default=_python_module_command(
            "scripts.quant_research_foundry",
            "--mode",
            "zero-calibration",
        ),
    )
    parser.add_argument(
        "--post-close-train-command",
        default=_python_module_command(
            "scripts.model_maintenance",
            "--mode",
            "daily",
            "--max-symbols",
            str(resource_profile.daily_training_max_symbols),
            "--rf-jobs",
            str(resource_profile.research_rf_jobs),
            "--model-parallelism",
            str(resource_profile.model_parallelism),
            "--target-daily-return",
            "0.002",
            "--target-accuracy",
            "0.56",
        ),
    )
    parser.add_argument(
        "--post-close-tune-command",
        default=_python_module_command(
            "scripts.quant_research_foundry",
            "--mode",
            "weekend-calibrate",
            "--max-symbols",
            str(min(35, resource_profile.daily_training_max_symbols)),
            "--rf-jobs",
            str(resource_profile.research_rf_jobs),
            "--model-parallelism",
            str(resource_profile.model_parallelism),
            "--target-daily-return",
            "0.002",
            "--target-accuracy",
            "0.56",
        ),
    )
    parser.add_argument(
        "--post-close-backtest-command",
        default=_python_module_command(
            "scripts.massive_backtest_engine",
            "--workers",
            str(resource_profile.backtest_workers),
        ),
    )

    parser.add_argument("--weekend-hour", type=int, default=8)
    parser.add_argument("--weekend-minute", type=int, default=0)
    parser.add_argument(
        "--weekend-recalibration-command",
        default=_python_module_command(
            "scripts.weekend_recalibration",
            "--workers",
            str(resource_profile.backtest_workers),
            "--target-daily-return",
            "0.002",
            "--target-accuracy",
            "0.56",
        ),
    )
    parser.add_argument(
        "--weekend-backtest-command",
        default=_python_module_command(
            "scripts.massive_backtest_engine",
            "--workers",
            str(resource_profile.backtest_workers),
        ),
    )
    parser.add_argument("--weekend-report-path", default="reports/weekend_professional_report.md")
    parser.add_argument("--daily-report-dir", default="reports/daily")
    parser.add_argument("--latest-daily-report-path", default="reports/daily/latest_daily_automation_report.md")
    parser.add_argument("--massive-backtest-report-path", default="reports/massive_backtest_report.json")
    parser.add_argument("--state-path", default=".runtime/automation_state.json")
    parser.add_argument("--preflight-state-path", default=".runtime/preflight_state.json")
    parser.add_argument("--system-snapshot-path", default=str(DEFAULT_SYSTEM_SNAPSHOT_PATH.relative_to(REPO_ROOT)))
    parser.add_argument("--preflight-max-age-seconds", type=int, default=300)
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Bypass the compile/import/config gate before controller actions run.",
    )
    parser.add_argument(
        "--deep-model-checks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load critical model files during preflight instead of only checking that they exist.",
    )
    parser.add_argument(
        "--progress-ui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show percentage-based progress lines in the controller output.",
    )
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
