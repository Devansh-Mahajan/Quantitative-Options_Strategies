import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.movement_predictor import LOOKBACK_MAP, backtest_symbol_movement


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtest stock movement predictor over multiple lookback windows."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Ticker symbols, e.g. --symbols SPY QQQ NVDA",
    )
    parser.add_argument(
        "--lookbacks",
        nargs="+",
        default=["10y", "5y", "1y", "6mo", "3mo"],
        help="Lookback windows. Supported: 10y 5y 1y 6mo 3mo",
    )
    parser.add_argument(
        "--output",
        default="reports/movement_predictor_backtest.json",
        help="JSON output report path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    lookbacks = [lb for lb in args.lookbacks if lb in LOOKBACK_MAP]
    if not lookbacks:
        raise SystemExit("No valid lookbacks provided.")

    results = []
    for symbol in args.symbols:
        for lb in lookbacks:
            res = backtest_symbol_movement(symbol, lb)
            results.append(res)
            if "error" in res:
                print(f"{symbol:<8} {lb:<4} ERROR: {res['error']}")
            else:
                print(
                    f"{symbol:<8} {lb:<4} acc={res['accuracy']:.3f} "
                    f"strat={res['strategy_return']:+.2%} "
                    f"buy&hold={res['buy_hold_return']:+.2%}"
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
