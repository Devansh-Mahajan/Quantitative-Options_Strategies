from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.universe_maintenance import download_close_matrix, load_symbol_file

ROOT = Path(__file__).resolve().parent.parent
SYMBOLS_FILE = ROOT / "config" / "symbol_list.txt"
REPORT_PATH = ROOT / "reports" / "stock_universe_research.json"
CACHE_PATH = ROOT / ".backtest_cache" / "stock_universe_close_1d.parquet"


def _safe_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _max_drawdown(returns: pd.Series) -> float | None:
    returns = returns.dropna()
    if returns.empty:
        return None
    equity = (1.0 + returns).cumprod()
    dd = equity / equity.cummax() - 1.0
    return _safe_float(dd.min())


def _row_for_symbol(symbol: str, prices: pd.DataFrame, benchmark_returns: pd.Series) -> dict:
    series = prices[symbol].dropna() if symbol in prices.columns else pd.Series(dtype=float)
    if series.empty:
        return {
            "symbol": symbol,
            "data_status": "missing",
            "data_rows": 0,
            "has_research": False,
        }

    returns = series.pct_change().dropna()
    aligned = pd.concat([returns.rename("asset"), benchmark_returns.rename("benchmark")], axis=1, sort=False).dropna()
    beta = None
    corr_spy = None
    alpha_daily = None
    if len(aligned) >= 60 and aligned["benchmark"].var() > 0:
        beta = _safe_float(aligned["asset"].cov(aligned["benchmark"]) / aligned["benchmark"].var())
        corr_spy = _safe_float(aligned["asset"].corr(aligned["benchmark"]))
        alpha_daily = _safe_float(aligned["asset"].mean() - (beta or 1.0) * aligned["benchmark"].mean())

    vol = _safe_float(returns.std() * math.sqrt(252)) if len(returns) else None
    momentum_1m = _safe_float(series.iloc[-1] / series.iloc[-22] - 1.0) if len(series) > 22 else None
    momentum_3m = _safe_float(series.iloc[-1] / series.iloc[-63] - 1.0) if len(series) > 63 else None
    momentum_12m = _safe_float(series.iloc[-1] / series.iloc[-252] - 1.0) if len(series) > 252 else None
    sharpe = None
    if returns.std() and returns.std() > 0:
        sharpe = _safe_float((returns.mean() / returns.std()) * math.sqrt(252))

    return {
        "symbol": symbol,
        "data_status": "ready" if len(series) >= 120 else "thin",
        "data_rows": int(len(series)),
        "first_date": series.index.min().isoformat() if len(series) else None,
        "last_date": series.index.max().isoformat() if len(series) else None,
        "last_price": _safe_float(series.iloc[-1]),
        "daily_return_mean": _safe_float(returns.mean()) if len(returns) else None,
        "volatility_annualized": vol,
        "sharpe_proxy": sharpe,
        "max_drawdown": _max_drawdown(returns),
        "momentum_1m": momentum_1m,
        "momentum_3m": momentum_3m,
        "momentum_12m": momentum_12m,
        "beta_spy": beta,
        "corr_spy": corr_spy,
        "alpha_daily": alpha_daily,
        "has_research": len(series) >= 120,
        "research_source": "stock_universe_research",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and research the full stock universe.")
    parser.add_argument("--period", default="2y", help="Yahoo Finance history period, e.g. 1y, 2y, 5y")
    parser.add_argument("--symbols-file", type=Path, default=SYMBOLS_FILE)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    symbols = load_symbol_file(args.symbols_file)
    if not symbols:
        raise SystemExit(f"No symbols found in {args.symbols_file}")

    print(f"[stock-research] downloading {len(symbols)} symbols period={args.period}", flush=True)
    prices = download_close_matrix(symbols, period=args.period, auto_adjust=True, progress=False)
    if prices.empty:
        raise SystemExit("No price data downloaded for stock universe.")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(CACHE_PATH)

    benchmark = prices["SPY"].pct_change().dropna() if "SPY" in prices.columns else prices.pct_change().mean(axis=1).dropna()
    rows = [_row_for_symbol(symbol, prices, benchmark) for symbol in symbols]
    covered = sum(1 for row in rows if row.get("has_research"))
    ready = sum(1 for row in rows if row.get("data_status") == "ready")
    thin = sum(1 for row in rows if row.get("data_status") == "thin")
    missing = sum(1 for row in rows if row.get("data_status") == "missing")
    leaders = sorted(
        [row for row in rows if row.get("alpha_daily") is not None],
        key=lambda row: (row.get("alpha_daily") or -1e9, row.get("sharpe_proxy") or -1e9),
        reverse=True,
    )[:50]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "period": args.period,
        "symbols_requested": len(symbols),
        "symbols_with_price_data": int(len(prices.columns)),
        "coverage_pct": round(ready / max(len(symbols), 1) * 100.0, 2),
        "research_coverage_pct": round(covered / max(len(symbols), 1) * 100.0, 2),
        "ready_symbols": ready,
        "thin_symbols": thin,
        "missing_symbols": missing,
        "cache_path": str(CACHE_PATH.relative_to(ROOT)),
        "rows": rows,
        "leaders": leaders,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[stock-research] complete ready={ready} thin={thin} missing={missing} "
        f"coverage={payload['coverage_pct']}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
