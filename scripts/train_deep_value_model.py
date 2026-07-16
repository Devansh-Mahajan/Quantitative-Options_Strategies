"""
Weekend trainer for the deep-value (net-net) ML scorer.

Builds a training set from two sources and fits a small gradient-boosted
classifier predicting P(forward 60-trading-day return >= +20%):

  (a) bootstrap: historical quarterly balance sheets (~4-8 periods per ticker
      via yfinance) joined with the price on each period-end date and the
      forward return from that date;
  (b) matured rows from .runtime/deep_value_history.jsonl (nightly scan
      snapshots old enough that their forward window has closed).

If fewer than --min-samples rows are available the script exits 0 WITHOUT
writing a model — the live scorer then keeps using the heuristic only.
Invoked from scripts/weekend_recalibration.py; safe to run manually.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from core.ml.deep_value import (
    FEATURE_NAMES,
    MODEL_PATH,
    TRAINING_HISTORY_PATH,
    ValueMetrics,
    compute_value_metrics,
    feature_vector,
)
from core.risk.universe_maintenance import load_symbol_file, resolve_download_symbol
from core.telemetry.state_manager import register_model_snapshot

SYMBOL_FILE = ROOT / "config" / "symbol_list.txt"
FORWARD_TRADING_DAYS = 60
LABEL_THRESHOLD = 0.20  # forward return counted as a win


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the deep-value ML scorer.")
    parser.add_argument("--max-symbols", type=int, default=150)
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--fetch-sleep", type=float, default=0.4)
    return parser.parse_args()


def _forward_return(closes, as_of) -> float | None:
    """Return over the FORWARD_TRADING_DAYS window starting at/after as_of."""
    try:
        window = closes.loc[closes.index >= as_of]
        if len(window) <= FORWARD_TRADING_DAYS:
            return None
        start = float(window.iloc[0])
        end = float(window.iloc[FORWARD_TRADING_DAYS])
        if start <= 0:
            return None
        return end / start - 1.0
    except Exception:
        return None


def _bootstrap_rows(symbols: list[str], fetch_sleep: float) -> tuple[list[list[float]], list[int]]:
    """Historical balance-sheet periods -> (features, labels)."""
    import pandas as pd
    import yfinance as yf

    X: list[list[float]] = []
    y: list[int] = []

    for index, symbol in enumerate(symbols):
        try:
            ticker = yf.Ticker(resolve_download_symbol(symbol))
            sheet = ticker.quarterly_balance_sheet
            if sheet is None or sheet.empty:
                continue
            history = ticker.history(period="5y", auto_adjust=True)
            if history is None or history.empty:
                continue
            history.index = history.index.tz_localize(None)
            closes = history["Close"]
            volumes = history["Volume"]
        except Exception:
            continue
        finally:
            if index < len(symbols) - 1 and fetch_sleep > 0:
                time.sleep(fetch_sleep)

        def _row(labels: list[str], column) -> float | None:
            for label in labels:
                if label in sheet.index:
                    value = sheet.loc[label, column]
                    if value is not None and not (isinstance(value, float) and math.isnan(value)):
                        return float(value)
            return None

        for column in sheet.columns:
            as_of = pd.Timestamp(column).tz_localize(None)
            price_window = closes.loc[closes.index >= as_of]
            if price_window.empty:
                continue
            price = float(price_window.iloc[0])

            shares = _row(["Ordinary Shares Number", "Share Issued"], column)
            fundamentals = {
                "price": price,
                "shares_outstanding": shares,
                "current_assets": _row(["Current Assets", "Total Current Assets"], column),
                "total_liabilities": _row(
                    ["Total Liabilities Net Minority Interest", "Total Liab", "Total Liabilities"], column
                ),
                "cash": _row(
                    ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"], column
                ),
                "receivables": _row(["Accounts Receivable", "Receivables"], column),
                "inventory": _row(["Inventory"], column),
            }
            metrics = compute_value_metrics(fundamentals)
            if metrics is None:
                continue
            forward = _forward_return(closes, as_of)
            if forward is None:
                continue

            vol_window = volumes.loc[volumes.index >= as_of]
            avg_dollar_volume = float(vol_window.iloc[:20].mean()) * price if len(vol_window) else 0.0
            quote = {
                "price": price,
                "market_cap": price * (shares or 0.0),
                "avg_dollar_volume": avg_dollar_volume,
            }
            X.append(feature_vector(metrics, quote))
            y.append(1 if forward >= LABEL_THRESHOLD else 0)

    return X, y


def _matured_history_rows() -> tuple[list[list[float]], list[int]]:
    """Nightly scan rows whose forward window has closed, labeled from prices."""
    import yfinance as yf

    if not TRAINING_HISTORY_PATH.exists():
        return [], []

    cutoff = datetime.now(timezone.utc) - timedelta(days=int(FORWARD_TRADING_DAYS * 1.5))
    matured: dict[str, list[dict]] = {}
    for line in TRAINING_HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            recorded = datetime.fromisoformat(str(row.get("recorded_at_utc")))
            if recorded.tzinfo is None:
                recorded = recorded.replace(tzinfo=timezone.utc)
            if recorded <= cutoff and row.get("price"):
                matured.setdefault(str(row.get("symbol") or "").upper(), []).append(row)
        except Exception:
            continue
    if not matured:
        return [], []

    X: list[list[float]] = []
    y: list[int] = []
    for symbol, rows in matured.items():
        try:
            history = yf.Ticker(resolve_download_symbol(symbol)).history(period="1y", auto_adjust=True)
            history.index = history.index.tz_localize(None)
            closes = history["Close"]
        except Exception:
            continue
        for row in rows:
            recorded = datetime.fromisoformat(str(row["recorded_at_utc"])).replace(tzinfo=None)
            forward = _forward_return(closes, recorded)
            if forward is None:
                continue
            runway = float(row.get("burn_runway_quarters", -1.0))
            metrics = ValueMetrics(
                ncav_per_share=float(row.get("ncav_per_share", 0.0)),
                liquidation_per_share=float(row.get("liquidation_per_share", 0.0)),
                net_cash_per_share=float(row.get("net_cash_per_share", 0.0)),
                burn_runway_quarters=math.inf if runway < 0 else runway,
                price_to_ncav=float(row.get("price_to_ncav", math.inf)),
            )
            X.append(feature_vector(metrics, row))
            y.append(1 if forward >= LABEL_THRESHOLD else 0)
    return X, y


def main() -> int:
    args = parse_args()

    history_symbols: list[str] = []
    if TRAINING_HISTORY_PATH.exists():
        for line in TRAINING_HISTORY_PATH.read_text(encoding="utf-8").splitlines():
            try:
                history_symbols.append(str(json.loads(line).get("symbol") or ""))
            except Exception:
                continue
    symbols = [s for s in dict.fromkeys(history_symbols + load_symbol_file(SYMBOL_FILE)) if s]
    symbols = symbols[: max(1, args.max_symbols)]
    print(f"[deep-value-train] symbols={len(symbols)}")

    X_boot, y_boot = _bootstrap_rows(symbols, args.fetch_sleep)
    X_hist, y_hist = _matured_history_rows()
    X = X_boot + X_hist
    y = y_boot + y_hist
    print(f"[deep-value-train] samples: bootstrap={len(X_boot)} matured_scans={len(X_hist)} total={len(X)}")

    if len(X) < args.min_samples or len(set(y)) < 2:
        print(
            f"[deep-value-train] insufficient training data ({len(X)} rows, need {args.min_samples} with both classes)"
            " — keeping heuristic-only scoring."
        )
        return 0

    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=4, random_state=42)
    try:
        cv_auc = float(np.mean(cross_val_score(model, np.array(X), np.array(y), cv=3, scoring="roc_auc")))
    except Exception:
        cv_auc = float("nan")
    model.fit(np.array(X), np.array(y))

    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(X),
        "positive_rate": round(float(np.mean(y)), 4),
        "cv_auc": None if math.isnan(cv_auc) else round(cv_auc, 4),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL_PATH)
    register_model_snapshot(
        "deep_value_model",
        {key: value for key, value in payload.items() if key != "model"},
    )
    print(f"[deep-value-train] saved {MODEL_PATH} (n={len(X)}, cv_auc={payload['cv_auc']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
