"""
Holdout-validated per-strategy parameter tuning.

Wraps BacktestOptimizer with a per-strategy search space (STRATEGY_PARAM_SPACES)
over the league window. Adoption rule: tuned params are recommended ONLY when
the untouched 20% holdout confirms them — holdout Sharpe must beat both zero
and the default-params holdout Sharpe. Otherwise defaults stand.

Usage:
    python -m scripts.tools.tune_strategy --strategy tsmom --trials 40
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.tools.strategy_league import LEAGUE_SYMBOLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune one strategy's params with holdout validation.")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--symbols", nargs="+", default=LEAGUE_SYMBOLS)
    parser.add_argument("--output-dir", default=str(ROOT / "reports"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from backtester.data_loader import load_multi
    from backtester.fill_model import FillModel, venue_fees
    from backtester.optimizer import STRATEGY_PARAM_SPACES, BacktestOptimizer
    from strategies.registry import _build_all_enabled

    if args.strategy not in STRATEGY_PARAM_SPACES:
        print(f"no param space for '{args.strategy}' — add it to STRATEGY_PARAM_SPACES", file=sys.stderr)
        return 1

    strategy_cls = next((type(s) for s in _build_all_enabled() if s.name == args.strategy), None)
    if strategy_cls is None:
        print(f"strategy '{args.strategy}' not constructible", file=sys.stderr)
        return 1

    end = pd.Timestamp.now("UTC").normalize().tz_localize(None)
    start = end - pd.Timedelta(days=int(args.months * 30.4))
    data = load_multi(args.symbols, "1h", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    data = {sym: df for sym, df in (data or {}).items() if df is not None and len(df) > 1000}
    if not data:
        print("no data", file=sys.stderr)
        return 1

    def factory(**params):
        return [strategy_cls(**params)]

    optimizer = BacktestOptimizer(
        data=data,
        interval="1h",
        strategy_factory=factory,
        n_trials=args.trials,
        n_wf_folds=args.folds,
        tune_strategy=args.strategy,
        output_dir=Path(args.output_dir) / "tuning",
    )
    result = optimizer.run(verbose=False)

    # Default-params holdout benchmark: same holdout window, default kwargs,
    # default engine params.
    default_holdout, default_metrics = optimizer._evaluate_holdout({})
    tuned_holdout = result.holdout_score

    strategy_kwargs = {
        key[2:]: value for key, value in result.best_params.items() if key.startswith("s_")
    }
    adopted = (
        tuned_holdout == tuned_holdout  # not NaN
        and tuned_holdout > 0
        and (default_holdout != default_holdout or tuned_holdout > default_holdout)
    )

    report = {
        "strategy": args.strategy,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trials": args.trials,
        "best_params": result.best_params,
        "strategy_kwargs": strategy_kwargs,
        "search_score": result.best_score,
        "tuned_holdout_sharpe": None if tuned_holdout != tuned_holdout else round(tuned_holdout, 3),
        "tuned_holdout_metrics": result.holdout_metrics,
        "default_holdout_sharpe": None if default_holdout != default_holdout else round(default_holdout, 3),
        "default_holdout_metrics": default_metrics,
        "adopted": bool(adopted),
        "adoption_rule": "tuned holdout Sharpe > 0 AND > default holdout Sharpe",
    }
    out = Path(args.output_dir) / "tuning" / f"tune_{args.strategy}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("strategy", "strategy_kwargs", "tuned_holdout_sharpe", "default_holdout_sharpe", "adopted")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
