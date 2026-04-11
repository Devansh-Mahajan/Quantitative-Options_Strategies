import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.movement_predictor import LOOKBACK_MAP, backtest_symbol_movement


def parse_args():
    p = argparse.ArgumentParser(description="Massive backtest runner for movement prediction.")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--lookbacks", nargs="+", default=["10y", "5y", "1y", "6mo", "3mo"])
    p.add_argument("--target-daily-return", type=float, default=0.002)
    p.add_argument("--target-accuracy", type=float, default=0.56)
    p.add_argument("--output", default="reports/massive_backtest_report.json")
    return p.parse_args()


def annual_to_daily(ann):
    if ann <= -0.999:
        return -1.0
    return (1 + ann) ** (1 / 252) - 1


def main():
    args = parse_args()
    lookbacks = [lb for lb in args.lookbacks if lb in LOOKBACK_MAP]
    report = {"targets": {"daily_return": args.target_daily_return, "accuracy": args.target_accuracy}, "results": []}

    for symbol in args.symbols:
        for lb in lookbacks:
            res = backtest_symbol_movement(symbol, lb)
            if "error" in res:
                res["meets_targets"] = False
            else:
                daily_ret = annual_to_daily(res["strategy_return"])
                res["strategy_daily_return"] = daily_ret
                res["meets_targets"] = bool(res["accuracy"] >= args.target_accuracy and daily_ret >= args.target_daily_return)
            report["results"].append(res)

    valid = [x for x in report["results"] if "error" not in x]
    report["summary"] = {
        "total_runs": len(report["results"]),
        "valid_runs": len(valid),
        "hit_rate": sum(1 for x in valid if x.get("meets_targets")) / len(valid) if valid else 0.0,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
