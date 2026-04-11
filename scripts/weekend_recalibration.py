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


def load_symbols():
    with open(SYMBOLS_FILE, 'r') as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def save_symbols(symbols):
    with open(VOL_SYMBOLS_FILE, 'w') as f:
        f.write("\n".join(symbols) + "\n")


def run_step(cmd):
    print(f"[recalibration] running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Weekend recalibration pipeline")
    parser.add_argument("--top-n", type=int, default=80, help="How many high-volatility symbols to keep")
    parser.add_argument("--train", action="store_true", help="Run full retraining pipeline")
    parser.add_argument("--target-daily-return", type=float, default=0.002, help="Optimization target for expected daily return.")
    parser.add_argument("--target-accuracy", type=float, default=0.56, help="Optimization target for validation accuracy.")
    args = parser.parse_args()

    symbols = load_symbols()
    prioritized = prioritize_symbols(symbols, top_n=args.top_n)
    save_symbols(prioritized)

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbols_input": len(symbols),
        "symbols_retained": len(prioritized),
        "volatile_symbols_file": str(VOL_SYMBOLS_FILE.relative_to(ROOT)),
        "target_daily_return": args.target_daily_return,
        "target_accuracy": args.target_accuracy,
        "note": "Target is an optimization objective, not a guarantee.",
    }
    register_model_snapshot("weekend_recalibration", snapshot)

    if args.train:
        run_step(["python", "scripts/train_hmm.py"])
        run_step(["python", "scripts/mega_matrix.py", "--target-annual-return", str(args.target_daily_return * 252)])
        run_step(["python", "scripts/mega_gpu_training.py", "--target-annual-return", str(args.target_daily_return * 252), "--target-accuracy", str(args.target_accuracy)])
        run_step(["python", "scripts/train_regime_movement_models.py", "--target-accuracy", str(args.target_accuracy)])

    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
