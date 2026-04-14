from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from core.resource_profile import load_resource_profile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = ROOT / "reports" / "daily_model_maintenance_report.json"


def parse_args() -> argparse.Namespace:
    profile = load_resource_profile(ROOT)
    parser = argparse.ArgumentParser(description="Lightweight post-close model maintenance pipeline.")
    parser.add_argument("--mode", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--target-daily-return", type=float, default=0.002)
    parser.add_argument("--target-accuracy", type=float, default=0.56)
    parser.add_argument("--max-symbols", type=int, default=profile.daily_training_max_symbols)
    parser.add_argument("--rf-jobs", type=int, default=profile.research_rf_jobs)
    parser.add_argument("--model-parallelism", type=int, default=profile.model_parallelism)
    parser.add_argument("--skip-foundry", action="store_true")
    parser.add_argument("--skip-regime-models", action="store_true")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    return parser.parse_args()


def _run_step(cmd: list[str], *, env: dict[str, str]) -> dict:
    started = datetime.now(timezone.utc)
    process = subprocess.Popen(cmd, cwd=ROOT, env=env)
    return_code = process.wait()
    finished = datetime.now(timezone.utc)
    result = {
        "command": cmd,
        "return_code": return_code,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 2),
    }
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    return result


def main() -> None:
    args = parse_args()
    profile = load_resource_profile(ROOT)
    env = {**os.environ, **profile.to_env()}
    python_exec = sys.executable

    steps: list[list[str]] = []
    if not args.skip_foundry:
        steps.append(
            [
                python_exec,
                "scripts/quant_research_foundry.py",
                "--mode",
                "weekend-calibrate",
                "--target-daily-return",
                str(args.target_daily_return),
                "--target-accuracy",
                str(args.target_accuracy),
                "--max-symbols",
                str(args.max_symbols),
                "--rf-jobs",
                str(args.rf_jobs),
                "--model-parallelism",
                str(args.model_parallelism),
            ]
        )
    if not args.skip_regime_models:
        steps.append(
            [
                python_exec,
                "scripts/train_regime_movement_models.py",
                "--target-accuracy",
                str(args.target_accuracy),
            ]
        )

    results = []
    for command in steps:
        print(f"[model-maintenance] running: {' '.join(command)}")
        results.append(_run_step(command, env=env))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "resource_profile": profile.to_dict(),
        "steps": results,
        "note": "Daily maintenance refreshes model packs and regime movement models between heavy weekend recalibration cycles.",
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
