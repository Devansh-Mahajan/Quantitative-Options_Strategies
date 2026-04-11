import sys

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.market_intelligence import prioritize_symbols
from core.state_manager import register_model_snapshot

ROOT = Path(__file__).resolve().parent.parent
SYMBOLS_FILE = ROOT / "config" / "symbol_list.txt"
VOL_SYMBOLS_FILE = ROOT / "config" / "volatile_symbols.txt"
ADAPTIVE_PROFILE_FILE = ROOT / "config" / "adaptive_profile.json"


def load_symbols():
    with open(SYMBOLS_FILE, 'r') as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def save_symbols(symbols):
    with open(VOL_SYMBOLS_FILE, 'w') as f:
        f.write("\n".join(symbols) + "\n")


def load_adaptive_profile():
    if not ADAPTIVE_PROFILE_FILE.exists():
        return {}
    try:
        with open(ADAPTIVE_PROFILE_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def run_step(cmd):
    print(f"[recalibration] running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, cwd=ROOT)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        print("\n[recalibration] interrupt received, stopping current step...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)





def print_pipeline_progress(current_step, total_steps, label):
    width = 32
    ratio = min(max(current_step / total_steps, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    percent = int(ratio * 100)
    print(f"[pipeline] [{bar}] {percent:>3}% ({current_step}/{total_steps}) {label}")


def main():
    parser = argparse.ArgumentParser(description="Weekend recalibration pipeline")
    parser.add_argument("--top-n", type=int, default=0, help="Deprecated: universe pruning is disabled and all symbols are always kept.")
    parser.add_argument("--train", action="store_true", help="Run full retraining pipeline")
    parser.add_argument("--target-daily-return", type=float, default=0.002, help="Optimization target for expected daily return.")
    parser.add_argument("--target-accuracy", type=float, default=0.56, help="Optimization target for validation accuracy.")
    args = parser.parse_args()

    base_step_count = 4  # load -> prioritize -> persist -> snapshot

    train_steps = []
    if args.train:
        train_steps = [
            ["python", "scripts/train_hmm.py"],
            ["python", "scripts/train_correlation_alpha.py"],
            ["python", "scripts/mega_matrix.py", "--target-annual-return", str(args.target_daily_return * 252)],
            ["python", "scripts/mega_gpu_training.py", "--target-annual-return", str(args.target_daily_return * 252), "--target-accuracy", str(args.target_accuracy)],
            ["python", "scripts/train_regime_movement_models.py", "--target-accuracy", str(args.target_accuracy)],
        ]

    total_steps = base_step_count + len(train_steps)
    step_idx = 1

    print_pipeline_progress(step_idx, total_steps, "Loading symbols")
    symbols = load_symbols()

    effective_top_n = len(symbols) if args.top_n <= 0 else min(args.top_n, len(symbols))

    step_idx += 1
    print_pipeline_progress(step_idx, total_steps, f"Prioritizing top {effective_top_n} symbols")
    prioritized = prioritize_symbols(symbols, top_n=effective_top_n)

    step_idx += 1
    print_pipeline_progress(step_idx, total_steps, "Writing volatile symbol list")
    save_symbols(prioritized)

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbols_input": len(symbols),
        "symbols_retained": len(prioritized),
        "top_n_requested": args.top_n,
        "top_n_applied": effective_top_n,
        "volatile_symbols_file": str(VOL_SYMBOLS_FILE.relative_to(ROOT)),
        "target_daily_return": args.target_daily_return,
        "target_accuracy": args.target_accuracy,
        "adaptive_profile": load_adaptive_profile(),
        "note": "Target is an optimization objective, not a guarantee.",
    }
    step_idx += 1
    print_pipeline_progress(step_idx, total_steps, "Saving recalibration snapshot")
    register_model_snapshot("weekend_recalibration", snapshot)

    for cmd in train_steps:
        step_idx += 1
        print_pipeline_progress(step_idx, total_steps, f"Running {' '.join(cmd[1:])}")
        run_step(cmd)

    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[recalibration] interrupted by user; exiting cleanly.")
        raise SystemExit(130)
