"""
XGBoost Alpha Engine Training Script.

Trains the XGBAlphaEngine on the full symbol universe, evaluates OOS Sharpe
per horizon, prints a summary table, and saves the model to disk.

Based on: Gu, Kelly & Xiu (2020) "Empirical Asset Pricing via Machine Learning"
          Review of Financial Studies — tree ensembles outperform neural nets
          on financial tabular data with purged time-series cross-validation.

Usage:
    python -m scripts.train_xgb_alpha [--symbols BTC ETH SPY] [--lookback 120] [--verbose]
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_xgb_alpha")


def _load_data(symbols: list[str], lookback_days: int) -> dict:
    """Load OHLCV data for all symbols using the backtester data loader."""
    from backtester.data_loader import load_multi

    end = None
    import datetime
    start = (datetime.datetime.utcnow() - datetime.timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    log.info("Loading data: %d symbols, lookback=%d days", len(symbols), lookback_days)
    df_dict = load_multi(symbols, interval="1h", start=start, end=end)
    loaded = {s: df for s, df in df_dict.items() if df is not None and len(df) > 200}
    log.info("Loaded %d / %d symbols with sufficient history", len(loaded), len(symbols))
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost Alpha Engine")
    parser.add_argument("--symbols",  nargs="*",   default=None,    help="Override symbol list")
    parser.add_argument("--lookback", type=int,    default=180,     help="Days of history to load")
    parser.add_argument("--verbose",  action="store_true",          help="Enable XGBoost verbose output")
    parser.add_argument("--no-save",  action="store_true",          help="Skip saving model to disk")
    args = parser.parse_args()

    # ── Symbol universe ───────────────────────────────────────────────────────
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        from bot.config import cfg
        symbols = cfg.model_symbols
    log.info("Training symbols: %s", symbols)

    # ── Load data ─────────────────────────────────────────────────────────────
    df_dict = _load_data(symbols, args.lookback)
    if not df_dict:
        log.error("No data loaded — aborting.")
        sys.exit(1)

    # ── Train ─────────────────────────────────────────────────────────────────
    from ml.xgb_alpha import XGBAlphaEngine

    engine = XGBAlphaEngine()
    log.info("Starting training across %d symbols × 3 horizons (1h, 4h, 24h)...", len(df_dict))
    metrics = engine.train(df_dict, verbose=args.verbose)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("XGBoost Alpha Engine — Training Summary")
    print("=" * 60)
    total_models = 0
    total_oos_sharpe = 0.0
    for symbol, horizons in metrics.items():
        print(f"\n  {symbol}")
        for horizon, m in horizons.items():
            oos = m.get("oos_sharpe", 0.0)
            n   = m.get("n_train", 0)
            acc = m.get("accuracy", 0.0)
            print(f"    {horizon:>4}h  OOS Sharpe: {oos:+.3f}  |  Train N: {n:>6}  |  Accuracy: {acc:.2%}")
            total_models += 1
            total_oos_sharpe += oos

    print("\n" + "-" * 60)
    avg_sharpe = total_oos_sharpe / max(total_models, 1)
    print(f"  Models trained: {total_models}")
    print(f"  Avg OOS Sharpe: {avg_sharpe:+.3f}")
    print("=" * 60 + "\n")

    if avg_sharpe < -0.5:
        log.warning("Average OOS Sharpe is very negative (%.3f) — check data quality", avg_sharpe)
    elif avg_sharpe > 0.0:
        log.info("Positive OOS Sharpe — model has predictive value")

    # ── Save ──────────────────────────────────────────────────────────────────
    if not args.no_save:
        engine.save()
        log.info("Model saved.")
    else:
        log.info("--no-save: model not persisted.")

    log.info("Done.")


if __name__ == "__main__":
    main()
