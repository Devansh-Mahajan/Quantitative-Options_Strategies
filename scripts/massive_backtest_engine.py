import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    p.add_argument("--workers", type=int, default=8, help="Concurrent workers for symbol/lookback jobs.")
    p.add_argument("--output", default="reports/massive_backtest_report.json")
    return p.parse_args()


def annual_to_daily(ann):
    if ann <= -0.999:
        return -1.0
    return (1 + ann) ** (1 / 252) - 1


def run_job(symbol, lookback, target_accuracy, target_daily_return):
    res = backtest_symbol_movement(symbol, lookback)
    if "error" in res:
        res["meets_targets"] = False
        return res

    daily_ret = annual_to_daily(res["strategy_return"])
    buy_hold_daily = annual_to_daily(res["buy_hold_return"])
    alpha_daily = daily_ret - buy_hold_daily

    res["strategy_daily_return"] = daily_ret
    res["buy_hold_daily_return"] = buy_hold_daily
    res["alpha_daily"] = alpha_daily
    res["meets_targets"] = bool(res["accuracy"] >= target_accuracy and daily_ret >= target_daily_return)
    return res


def safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    args = parse_args()
    lookbacks = [lb for lb in args.lookbacks if lb in LOOKBACK_MAP]
    jobs = [(symbol, lb) for symbol in args.symbols for lb in lookbacks]

    report = {
        "targets": {"daily_return": args.target_daily_return, "accuracy": args.target_accuracy},
        "results": [],
    }

    workers = max(1, min(args.workers, len(jobs) or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(run_job, sym, lb, args.target_accuracy, args.target_daily_return): (sym, lb)
            for sym, lb in jobs
        }
        for future in as_completed(future_map):
            report["results"].append(future.result())

    report["results"] = sorted(report["results"], key=lambda x: (x.get("symbol", ""), x.get("lookback", "")))

    valid = [x for x in report["results"] if "error" not in x]
    hit_rate = sum(1 for x in valid if x.get("meets_targets")) / len(valid) if valid else 0.0

    daily_returns = [x.get("strategy_daily_return", 0.0) for x in valid]
    alpha_daily = [x.get("alpha_daily", 0.0) for x in valid]
    accuracies = [x.get("accuracy", 0.0) for x in valid]

    report["summary"] = {
        "total_runs": len(report["results"]),
        "valid_runs": len(valid),
        "hit_rate": hit_rate,
        "avg_strategy_daily_return": safe_mean(daily_returns),
        "avg_alpha_daily": safe_mean(alpha_daily),
        "avg_accuracy": safe_mean(accuracies),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
