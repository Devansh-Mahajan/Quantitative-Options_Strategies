"""
Backtest report generation.
Console summary + JSON export + optional HTML equity chart.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backtester.engine import BacktestResult

log = logging.getLogger("backtester.report")

REPORTS_DIR = Path("backtest_reports")


def print_report(result: "BacktestResult", title: str = "Backtest Report") -> None:
    """Print a formatted summary to stdout."""
    m = result.metrics
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

    # Config snapshot
    cfg = result.config_snapshot
    if cfg:
        syms = ", ".join(cfg.get("symbols", []))
        print(f"  Symbols   : {syms}")
        print(f"  Interval  : {cfg.get('interval', '?')}")
        print(f"  Capital   : ${cfg.get('initial_equity', 0):,.0f}")
        strats = ", ".join(cfg.get("strategies", []))
        print(f"  Strategies: {strats}")
        print()

    # Core metrics
    for line in m.summary_lines():
        print(line)

    print()
    print(f"  Signals Fired   : {result.signals_fired}")
    print(f"  Fills Executed  : {result.fills_executed}")

    # Per-strategy breakdown
    if result.per_strategy_metrics:
        print(f"\n  {'─'*56}")
        print("  Per-Strategy Breakdown")
        print(f"  {'─'*56}")
        header = f"  {'Strategy':<22} {'Trades':>6} {'WinRate':>8} {'PF':>6} {'Sharpe':>7} {'MaxDD':>7}"
        print(header)
        print(f"  {'─'*56}")
        for name, sm in result.per_strategy_metrics.items():
            print(
                f"  {name:<22} {sm.num_trades:>6} "
                f"{sm.win_rate_pct:>7.1f}% "
                f"{sm.profit_factor:>6.2f} "
                f"{sm.sharpe:>7.3f} "
                f"{sm.max_drawdown_pct:>6.1f}%"
            )

    print(sep)
    print()


def save_report(
    result: "BacktestResult",
    run_id: str | None = None,
    output_dir: Path | str | None = None,
    mc_result=None,       # optional BacktestMCResult
    portfolio_mc=None,    # optional MonteCarloResult
) -> Path:
    """
    Persist the full report as a JSON file.
    Returns the path written.
    """
    out_dir = Path(output_dir) if output_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    payload: dict = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": result.config_snapshot,
        "metrics": result.metrics.to_dict(),
        "per_strategy_metrics": {
            k: v.to_dict() for k, v in result.per_strategy_metrics.items()
        },
        "signals_fired": result.signals_fired,
        "fills_executed": result.fills_executed,
        "equity_curve": {
            str(ts): round(val, 4)
            for ts, val in zip(result.equity_curve.index.astype(str), result.equity_curve.values)
        },
        "trades": (
            result.trades.to_dict(orient="records")
            if not result.trades.empty
            else []
        ),
    }

    if mc_result is not None:
        lv = mc_result.las_vegas
        bs = mc_result.bootstrap
        payload["monte_carlo"] = {
            "verdict": mc_result.verdict,
            "las_vegas_p_value_sharpe": lv.p_value_sharpe,
            "las_vegas_p_value_return": lv.p_value_return,
            "edge_sharpe": lv.edge_sharpe,
            "edge_return": lv.edge_return,
            "bootstrap_sharpe_ci_90": [bs.sharpe_ci.ci_low_90, bs.sharpe_ci.ci_high_90],
            "bootstrap_return_ci_90": [bs.return_ci.ci_low_90, bs.return_ci.ci_high_90],
            "ruin_probability": bs.ruin_probability,
            "target_return_probability": bs.target_return_probability,
        }

    if portfolio_mc is not None:
        vs = portfolio_mc.var_surface
        payload["portfolio_mc"] = {
            "distribution": portfolio_mc.distribution,
            "n_paths": portfolio_mc.n_paths,
            "annualised_vol": vs.annualised_vol,
            "var_surface": {
                f"{h}d": {
                    f"var_{int(c*100)}": vs.var[i][j]
                    for j, c in enumerate(vs.confidence_levels)
                }
                for i, h in enumerate(vs.horizon_days)
            },
            "stress_scenarios": portfolio_mc.stress_results,
            "percentile_bands": portfolio_mc.percentile_bands,
        }

    json_path = out_dir / f"backtest_{run_id}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("Report saved → %s", json_path)
    return json_path


def save_equity_csv(result: "BacktestResult", output_dir: Path | str | None = None) -> Path:
    """Save equity curve to CSV for external charting (e.g. TradingView Pine Script import)."""
    out_dir = Path(output_dir) if output_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"equity_{run_id}.csv"
    result.equity_curve.to_csv(csv_path, header=["equity"])
    log.info("Equity CSV saved → %s", csv_path)
    return csv_path


def compare_runs(json_paths: list[Path]) -> None:
    """Print a side-by-side comparison table for multiple backtest JSON reports."""
    if not json_paths:
        return
    rows = []
    for p in json_paths:
        data = json.loads(p.read_text())
        m = data.get("metrics", {})
        cfg = data.get("config", {})
        rows.append({
            "run": data.get("run_id", p.stem),
            "symbols": ",".join(cfg.get("symbols", [])),
            "total_ret": m.get("total_return_pct", 0),
            "sharpe": m.get("sharpe", 0),
            "max_dd": m.get("max_drawdown_pct", 0),
            "win_rate": m.get("win_rate_pct", 0),
            "trades": m.get("num_trades", 0),
        })

    header = f"  {'Run':<22} {'Symbols':<16} {'TotalRet':>9} {'Sharpe':>7} {'MaxDD':>7} {'WinRate':>8} {'Trades':>7}"
    sep = "  " + "─" * 78
    print(f"\n{sep}")
    print("  Run Comparison")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"  {r['run']:<22} {r['symbols']:<16} "
            f"{r['total_ret']:>+8.2f}% "
            f"{r['sharpe']:>7.3f} "
            f"{r['max_dd']:>6.1f}% "
            f"{r['win_rate']:>7.1f}% "
            f"{r['trades']:>7}"
        )
    print(sep)
    print()
