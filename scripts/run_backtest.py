"""
CLI entry point for the backtester.

Usage examples:
  run-backtest --symbols BTCUSDT ETHUSDT --start 2023-01-01 --end 2024-01-01
  run-backtest --symbols BTCUSDT --start 2023-01-01 --interval 4h --walk-forward --folds 5
  run-backtest --symbols BTCUSDT ETHUSDT --start 2022-01-01 --strategies momentum breakout --save-report
  run-backtest --compare backtest_reports/backtest_20240101.json backtest_reports/backtest_20240201.json
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path when called as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from backtester.data_loader import load_multi, align_and_ffill
from backtester.engine import BacktestEngine
from backtester.report import print_report, save_report, save_equity_csv, compare_runs
from backtester.walk_forward import WalkForwardValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_backtest")

AVAILABLE_STRATEGIES = [
    "momentum", "mean_reversion", "funding_arb", "basis_trade",
    "pairs_arb", "options_vol", "order_flow", "breakout",
]

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def _build_strategies(names: list[str]):
    """Import and instantiate only the requested strategy classes."""
    from strategies.momentum import MomentumStrategy
    from strategies.mean_reversion import MeanReversionStrategy
    from strategies.funding_arb import FundingArbStrategy
    from strategies.basis_trade import BasisTradeStrategy
    from strategies.pairs_arb import PairsArbStrategy
    from strategies.options_vol import OptionsVolStrategy
    from strategies.order_flow import OrderFlowStrategy
    from strategies.breakout import BreakoutStrategy

    factory_map = {
        "momentum": MomentumStrategy,
        "mean_reversion": MeanReversionStrategy,
        "funding_arb": FundingArbStrategy,
        "basis_trade": BasisTradeStrategy,
        "pairs_arb": PairsArbStrategy,
        "options_vol": OptionsVolStrategy,
        "order_flow": OrderFlowStrategy,
        "breakout": BreakoutStrategy,
    }
    result = []
    for name in names:
        cls = factory_map.get(name)
        if cls is None:
            log.warning("Unknown strategy '%s', skipping", name)
            continue
        try:
            result.append(cls())
        except Exception as exc:
            log.warning("Could not instantiate %s: %s", name, exc)
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run-backtest",
        description="Vectorised bar-by-bar backtester with walk-forward validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                   metavar="SYM", help="Binance symbols (default: BTCUSDT ETHUSDT)")
    p.add_argument("--interval", default="1h",
                   choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                   help="OHLCV bar interval (default: 1h)")
    p.add_argument("--start", required=False, default="2023-01-01",
                   metavar="YYYY-MM-DD", help="Start date (default: 2023-01-01)")
    p.add_argument("--end", required=False, default=None,
                   metavar="YYYY-MM-DD", help="End date (default: today)")
    p.add_argument("--force-download", action="store_true",
                   help="Bypass Parquet cache and re-download data")

    # Strategies
    p.add_argument("--strategies", nargs="+", default=AVAILABLE_STRATEGIES,
                   metavar="NAME",
                   help=f"Strategies to run (default: all). Choices: {', '.join(AVAILABLE_STRATEGIES)}")

    # Engine parameters
    p.add_argument("--capital", type=float, default=10_000,
                   help="Initial equity in USD (default: 10000)")
    p.add_argument("--stop-loss", type=float, default=0.05,
                   help="Hard stop-loss per trade, e.g. 0.05 = 5%% (default: 0.05)")
    p.add_argument("--take-profit", type=float, default=0.10,
                   help="Take-profit per trade, e.g. 0.10 = 10%% (default: 0.10)")
    p.add_argument("--max-positions", type=int, default=8,
                   help="Max simultaneous open positions (default: 8)")
    p.add_argument("--max-risk", type=float, default=None,
                   help="Max risk fraction per trade (default: from .env)")
    p.add_argument("--lookback", type=int, default=60,
                   help="Warm-up bars before first signal (default: 60)")

    # Walk-forward
    p.add_argument("--walk-forward", action="store_true",
                   help="Run K-fold walk-forward validation in addition to full backtest")
    p.add_argument("--folds", type=int, default=5,
                   help="Number of walk-forward folds (default: 5)")
    p.add_argument("--rolling", action="store_true",
                   help="Use rolling (non-anchored) walk-forward windows")
    p.add_argument("--test-fraction", type=float, default=0.20,
                   help="Fraction of data used as each test window (default: 0.20)")

    # Monte Carlo + Las Vegas
    p.add_argument("--monte-carlo", action="store_true",
                   help="Bootstrap CI + Las Vegas permutation test on backtest trades")
    p.add_argument("--mc-paths", type=int, default=10_000,
                   help="Monte Carlo paths for bootstrap / permutation (default: 10000)")
    p.add_argument("--mc-distribution", default="student-t",
                   choices=["normal", "student-t"],
                   help="Return distribution for portfolio MC (default: student-t)")
    p.add_argument("--portfolio-mc", action="store_true",
                   help="Run portfolio-level Monte Carlo VaR / stress scenarios")

    # Output
    p.add_argument("--save-report", action="store_true",
                   help="Save JSON report to backtest_reports/")
    p.add_argument("--save-equity-csv", action="store_true",
                   help="Save equity curve CSV for charting")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override report output directory")
    p.add_argument("--run-id", type=str, default=None,
                   help="Custom run ID for report filenames")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Extra debug logging")

    # Compare mode (no backtest, just diff existing JSON reports)
    p.add_argument("--compare", nargs="+", type=Path, default=None,
                   metavar="PATH",
                   help="Compare existing JSON report files instead of running a new backtest")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Compare-only mode
    if args.compare:
        compare_runs(args.compare)
        return

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    log.info("Loading data for %s  %s  %s → %s",
             args.symbols, args.interval, args.start, args.end or "today")
    try:
        data = load_multi(
            args.symbols, args.interval, args.start, args.end,
            force_download=args.force_download,
        )
    except Exception as exc:
        log.error("Data load failed: %s", exc)
        sys.exit(1)

    data = {sym: df for sym, df in data.items() if not df.empty}
    if not data:
        log.error("No data downloaded — check symbols and date range")
        sys.exit(1)

    data = align_and_ffill(data)

    # ------------------------------------------------------------------ #
    # Full backtest
    # ------------------------------------------------------------------ #
    strategies = _build_strategies(args.strategies)
    if not strategies:
        log.error("No valid strategies — aborting")
        sys.exit(1)

    engine_kwargs = dict(
        interval=args.interval,
        initial_equity=args.capital,
        lookback=args.lookback,
        max_open_positions=args.max_positions,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
    )
    if args.max_risk is not None:
        engine_kwargs["max_risk_per_trade"] = args.max_risk

    engine = BacktestEngine(data=data, strategies=strategies, **engine_kwargs)
    result = engine.run()

    title = f"Full Backtest  [{args.start} → {args.end or 'today'}]  {args.interval}"
    print_report(result, title=title)

    if args.save_report:
        path = save_report(result, run_id=args.run_id, output_dir=args.output_dir)
        print(f"  JSON report → {path}")

    if args.save_equity_csv:
        csv_path = save_equity_csv(result, output_dir=args.output_dir)
        print(f"  Equity CSV  → {csv_path}")

    # ------------------------------------------------------------------ #
    # Walk-forward validation
    # ------------------------------------------------------------------ #
    if args.walk_forward:
        def strategy_factory():
            return _build_strategies(args.strategies)

        wf_kwargs = {k: v for k, v in engine_kwargs.items() if k != "strategies"}

        validator = WalkForwardValidator(
            data=data,
            interval=args.interval,
            n_folds=args.folds,
            test_fraction=args.test_fraction,
            anchored=not args.rolling,
            engine_kwargs=wf_kwargs,
            strategy_factory=strategy_factory,
        )
        wf_result = validator.run(verbose=True)

        if args.save_report:
            # Save WF summary alongside main report
            import json, datetime
            from pathlib import Path as P
            out_dir = args.output_dir or P("backtest_reports")
            out_dir.mkdir(parents=True, exist_ok=True)
            run_id = args.run_id or datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            wf_path = out_dir / f"walk_forward_{run_id}.json"
            payload = {
                "run_id": run_id,
                "n_folds": len(wf_result.folds),
                "efficiency_ratio": wf_result.efficiency_ratio,
                "degradation_pct": wf_result.degradation_pct,
                "oos_metrics": wf_result.oos_combined_metrics.to_dict(),
                "is_metrics": wf_result.is_combined_metrics.to_dict(),
                "folds": [
                    {
                        "fold": f.fold_index,
                        "train_start": str(f.train_start),
                        "train_end": str(f.train_end),
                        "test_start": str(f.test_start),
                        "test_end": str(f.test_end),
                        "is_sharpe": f.is_result.metrics.sharpe,
                        "oos_sharpe": f.oos_result.metrics.sharpe,
                        "is_return": f.is_result.metrics.total_return_pct,
                        "oos_return": f.oos_result.metrics.total_return_pct,
                        "is_max_dd": f.is_result.metrics.max_drawdown_pct,
                        "oos_max_dd": f.oos_result.metrics.max_drawdown_pct,
                    }
                    for f in wf_result.folds
                ],
            }
            wf_path.write_text(json.dumps(payload, indent=2, default=str))
            print(f"  WF report   → {wf_path}")

    # ------------------------------------------------------------------ #
    # Monte Carlo bootstrap + Las Vegas permutation test
    # ------------------------------------------------------------------ #
    if args.monte_carlo:
        from backtester.monte_carlo_bt import BacktestMonteCarloEngine, print_mc_bt_result

        trade_rets = result.trades["pnl_pct"].tolist() if "pnl_pct" in result.trades.columns else []
        hold_bars  = result.trades["holding_bars"].tolist() if "holding_bars" in result.trades.columns else []

        mc_engine = BacktestMonteCarloEngine(
            n_bootstrap=args.mc_paths,
            n_permutations=args.mc_paths,
        )
        mc_result = mc_engine.run(
            trade_returns=trade_rets,
            equity_curve=result.equity_curve,
            holding_periods=hold_bars,
            verbose=args.verbose,
        )
        print_mc_bt_result(mc_result)

        if args.save_report:
            import json, datetime
            from pathlib import Path as P
            out_dir = args.output_dir or P("backtest_reports")
            out_dir.mkdir(parents=True, exist_ok=True)
            run_id_mc = args.run_id or datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            mc_path = out_dir / f"monte_carlo_{run_id_mc}.json"
            mc_payload = {
                "verdict": mc_result.verdict,
                "bootstrap": {
                    "n_iterations": mc_result.bootstrap.n_iterations,
                    "sharpe":  vars(mc_result.bootstrap.sharpe_ci),
                    "return":  vars(mc_result.bootstrap.return_ci),
                    "max_dd":  vars(mc_result.bootstrap.max_dd_ci),
                    "win_rate": vars(mc_result.bootstrap.win_rate_ci),
                    "profit_factor": vars(mc_result.bootstrap.profit_factor_ci),
                    "ruin_probability": mc_result.bootstrap.ruin_probability,
                    "target_return_probability": mc_result.bootstrap.target_return_probability,
                },
                "las_vegas": {
                    "n_permutations": mc_result.las_vegas.n_permutations,
                    "observed_sharpe": mc_result.las_vegas.observed_sharpe,
                    "observed_return": mc_result.las_vegas.observed_return,
                    "p_value_sharpe": mc_result.las_vegas.p_value_sharpe,
                    "p_value_return": mc_result.las_vegas.p_value_return,
                    "is_significant_sharpe": mc_result.las_vegas.is_significant_sharpe,
                    "is_significant_return": mc_result.las_vegas.is_significant_return,
                    "edge_sharpe": mc_result.las_vegas.edge_sharpe,
                    "edge_return": mc_result.las_vegas.edge_return,
                    "random_baseline_sharpe": mc_result.las_vegas.random_baseline_sharpe,
                    "random_baseline_return": mc_result.las_vegas.random_baseline_return,
                },
            }
            mc_path.write_text(json.dumps(mc_payload, indent=2, default=str))
            print(f"  MC report   → {mc_path}")

    # ------------------------------------------------------------------ #
    # Portfolio-level Monte Carlo (VaR surface + stress scenarios)
    # ------------------------------------------------------------------ #
    if args.portfolio_mc and not result.equity_curve.empty:
        from risk.monte_carlo import PortfolioMonteCarloEngine, mc_var_from_equity, print_mc_summary

        log.info("Running portfolio-level Monte Carlo …")
        var_95, cvar_95 = mc_var_from_equity(
            result.equity_curve,
            n_paths=args.mc_paths,
            confidence=0.95,
            distribution=args.mc_distribution,
        )
        var_99, cvar_99 = mc_var_from_equity(
            result.equity_curve,
            n_paths=args.mc_paths,
            confidence=0.99,
            distribution=args.mc_distribution,
        )

        print(f"\n  Portfolio MC VaR  ({args.mc_distribution}  ·  {args.mc_paths:,} paths)")
        print(f"  {'─'*44}")
        print(f"  1-day VaR  95%  : {var_95:.3%}  of equity")
        print(f"  1-day CVaR 95%  : {cvar_95:.3%}  of equity")
        print(f"  1-day VaR  99%  : {var_99:.3%}  of equity")
        print(f"  1-day CVaR 99%  : {cvar_99:.3%}  of equity")

        # Full simulation with stress scenarios (single-asset approximation using equity curve)
        rets = result.equity_curve.pct_change().dropna()
        mu   = float(rets.mean()) * 365.25
        vol  = float(rets.std()) * (365.25 ** 0.5)
        cov  = np.array([[vol ** 2]])
        eq_engine = PortfolioMonteCarloEngine(
            n_paths=args.mc_paths,
            distribution=args.mc_distribution,
        )
        mc_port = eq_engine.simulate_portfolio(
            weights=np.array([1.0]),
            mean_returns=np.array([mu]),
            cov_matrix=cov,
            equity=float(result.equity_curve.iloc[-1]),
            horizons=[1, 5, 10],
            include_stress=True,
        )
        print_mc_summary(mc_port, float(result.equity_curve.iloc[-1]))

    log.info("Done.")


if __name__ == "__main__":
    main()
