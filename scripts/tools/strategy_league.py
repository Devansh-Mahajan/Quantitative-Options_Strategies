"""
Strategy League — single-strategy backtest measurement over a common window.

Runs each Alpaca-viable crypto strategy ALONE through the BacktestEngine
(single-strategy runs give clean attribution; combined runs interfere via the
shared max_open_positions gate and cash), with real Alpaca fee levels, and
emits a ranked league table used to prune losers and pick tuning candidates.

Usage:
    python -m scripts.tools.strategy_league --months 12 --tag baseline
    python -m scripts.tools.strategy_league --months 12 --tag gated
    python -m scripts.tools.strategy_league --strategies tsmom hp_trend --months 6
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

LEAGUE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT"]

# Strategies that cannot produce meaningful results on Alpaca crypto spot:
# venue-gated ones plus futures-data-dependent ones (documented inert).
EXCLUDED = {
    "gamma_scalping", "vol_surface_arb", "market_making",
    "carry_portfolio", "liquidation_cascade",
    "funding_arb", "basis_trade", "options_vol",
}

MIN_TRADES_FOR_VERDICT = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-strategy backtest league.")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--symbols", nargs="+", default=LEAGUE_SYMBOLS)
    parser.add_argument("--strategies", nargs="+", default=None, help="subset by name (default: all viable)")
    parser.add_argument("--tag", default="baseline", help="report tag (baseline | gated | tuned)")
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--output-dir", default=str(ROOT / "reports"))
    return parser.parse_args()


def viable_strategy_names() -> list[str]:
    from strategies.registry import canonical_strategy_names

    return [n for n in canonical_strategy_names() if n not in EXCLUDED]


def fresh_instance(name: str):
    from strategies.registry import _build_all_enabled

    for strategy in _build_all_enabled():
        if strategy.name == name:
            return strategy
    return None


def run_league(args: argparse.Namespace) -> dict:
    from backtester.data_loader import load_multi
    from backtester.engine import BacktestEngine
    from backtester.fill_model import FillModel, venue_fees

    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=int(args.months * 30.4))
    print(f"[league] window {start.date()} -> {end.date()}, symbols={args.symbols}")

    data = load_multi(args.symbols, "1h", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    data = {sym: df for sym, df in (data or {}).items() if df is not None and len(df) > 1000}
    if not data:
        print("[league] FATAL: no data loaded", file=sys.stderr)
        return {}
    print(f"[league] data ready: { {sym: len(df) for sym, df in data.items()} }")

    maker, taker = venue_fees("alpaca")
    names = args.strategies or viable_strategy_names()

    rows = []
    for name in names:
        instance = fresh_instance(name)
        if instance is None:
            print(f"[league] skip {name}: not constructible (flag off?)")
            continue
        try:
            engine = BacktestEngine(
                data=data,
                interval="1h",
                initial_equity=args.initial_equity,
                strategies=[instance],
                fill_model=FillModel(maker_fee=maker, taker_fee=taker),
                max_open_positions=6,
            )
            result = engine.run()
        except Exception as exc:
            print(f"[league] {name} FAILED: {exc}")
            rows.append({"strategy": name, "error": str(exc)})
            continue

        m = result.metrics
        trades = int(m.num_trades or 0)
        total_ret = float(m.total_return_pct or 0.0)
        pf = float(m.profit_factor or 0.0)
        if trades < MIN_TRADES_FOR_VERDICT:
            verdict = "insufficient_sample"
        elif total_ret > 0 and pf > 1.0:
            verdict = "positive"
        else:
            verdict = "negative"

        rows.append({
            "strategy": name,
            "trades": trades,
            "total_return_pct": round(total_ret, 3),
            "sharpe": round(float(m.sharpe or 0.0), 3),
            "profit_factor": round(pf, 3),
            "win_rate_pct": round(float(m.win_rate_pct or 0.0), 1),
            "max_drawdown_pct": round(float(m.max_drawdown_pct or 0.0), 2),
            "avg_holding_bars": round(float(m.avg_holding_bars or 0.0), 1),
            "signals_fired": int(result.signals_fired or 0),
            "verdict": verdict,
        })
        print(f"[league] {name:<28} trades={trades:<4} ret={total_ret:+7.2f}% PF={pf:5.2f} -> {verdict}")

    rows.sort(key=lambda r: (r.get("total_return_pct") or -999), reverse=True)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "window": {"start": str(start.date()), "end": str(end.date()), "months": args.months},
        "symbols": list(data.keys()),
        "fees": {"venue": "alpaca", "maker": maker, "taker": taker},
        "min_trades_for_verdict": MIN_TRADES_FOR_VERDICT,
        "rows": rows,
    }
    return report


def write_report(report: dict, output_dir: Path, tag: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"strategy_league_{tag}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        f"# Strategy League — {tag}",
        "",
        f"Window: {report['window']['start']} → {report['window']['end']} · "
        f"Symbols: {', '.join(report['symbols'])} · Fees: Alpaca "
        f"({report['fees']['taker']*1e4:.0f}bps taker)",
        "",
        "| Strategy | Trades | Return | Sharpe | PF | Win% | MaxDD | AvgHold | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        if "error" in r:
            lines.append(f"| {r['strategy']} | — | — | — | — | — | — | — | ERROR |")
            continue
        lines.append(
            f"| {r['strategy']} | {r['trades']} | {r['total_return_pct']:+.2f}% | "
            f"{r['sharpe']:.2f} | {r['profit_factor']:.2f} | {r['win_rate_pct']:.0f}% | "
            f"{r['max_drawdown_pct']:.1f}% | {r['avg_holding_bars']:.0f} | {r['verdict']} |"
        )
    (output_dir / f"strategy_league_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[league] report written: {json_path}")


def main() -> int:
    args = parse_args()
    report = run_league(args)
    if not report:
        return 1
    write_report(report, Path(args.output_dir), args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
