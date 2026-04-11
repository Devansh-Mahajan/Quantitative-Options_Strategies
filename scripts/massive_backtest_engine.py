import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.movement_predictor import LOOKBACK_MAP, backtest_symbol_movement
from scripts.train_hmm import build_macro_features

DEFAULT_LOOKBACKS = ["10y", "5y", "1y", "6mo", "3mo"]
DEFAULT_OUTPUT = "reports/massive_backtest_report.json"
SYMBOL_LIST_PATH = ROOT / "config" / "symbol_list.txt"
HMM_MODEL_PATH = ROOT / "config" / "hmm_macro_model.pkl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="All-in-one predictive backtest suite for movement, regime, pairs, and strategy-routing proxies."
    )
    parser.add_argument("--symbols", nargs="+", help="Universe symbols. Defaults to config/symbol_list.txt if omitted.")
    parser.add_argument("--max-symbols", type=int, default=40, help="Cap symbol universe size for faster runs.")
    parser.add_argument("--lookbacks", nargs="+", default=DEFAULT_LOOKBACKS)
    parser.add_argument("--target-daily-return", type=float, default=0.002)
    parser.add_argument("--target-accuracy", type=float, default=0.56)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pairs-entry-z", type=float, default=1.5)
    parser.add_argument("--pairs-exit-z", type=float, default=0.5)
    parser.add_argument("--horizon-days", type=int, default=5, help="Evaluation horizon for regime/pairs/proxy tests.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_default_symbols(max_symbols: int) -> list[str]:
    if not SYMBOL_LIST_PATH.exists():
        return []
    with SYMBOL_LIST_PATH.open("r", encoding="utf-8") as handle:
        symbols = [line.strip().upper() for line in handle if line.strip()]
    symbols = list(dict.fromkeys(symbols))
    return symbols[: max(1, max_symbols)]


def annual_to_daily(annual_return: float) -> float:
    if annual_return <= -0.999:
        return -1.0
    return float((1 + annual_return) ** (1 / 252) - 1)


def safe_mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def run_movement_job(symbol: str, lookback: str, target_accuracy: float, target_daily_return: float) -> dict:
    result = backtest_symbol_movement(symbol, lookback)
    if "error" in result:
        result["meets_targets"] = False
        return result

    daily_return = annual_to_daily(result["strategy_return"])
    daily_buy_hold = annual_to_daily(result["buy_hold_return"])
    result["strategy_daily_return"] = daily_return
    result["buy_hold_daily_return"] = daily_buy_hold
    result["alpha_daily"] = daily_return - daily_buy_hold
    result["meets_targets"] = bool(result["accuracy"] >= target_accuracy and daily_return >= target_daily_return)
    return result


def run_movement_suite(symbols: list[str], lookbacks: list[str], workers: int, target_accuracy: float, target_daily_return: float):
    jobs = [(symbol, lb) for symbol in symbols for lb in lookbacks]
    if not jobs:
        return {"results": [], "summary": {"total_runs": 0, "valid_runs": 0}}

    pool_workers = max(1, min(workers, len(jobs)))
    results = []
    with ThreadPoolExecutor(max_workers=pool_workers) as executor:
        futures = {
            executor.submit(run_movement_job, symbol, lookback, target_accuracy, target_daily_return): (symbol, lookback)
            for symbol, lookback in jobs
        }
        for future in as_completed(futures):
            results.append(future.result())

    results = sorted(results, key=lambda x: (x.get("symbol", ""), x.get("lookback", "")))
    valid = [row for row in results if "error" not in row]

    summary = {
        "total_runs": len(results),
        "valid_runs": len(valid),
        "hit_rate": safe_mean([1.0 if row.get("meets_targets") else 0.0 for row in valid]),
        "avg_accuracy": safe_mean([row.get("accuracy", 0.0) for row in valid]),
        "avg_strategy_daily_return": safe_mean([row.get("strategy_daily_return", 0.0) for row in valid]),
        "avg_alpha_daily": safe_mean([row.get("alpha_daily", 0.0) for row in valid]),
    }
    return {"results": results, "summary": summary}


def download_close(symbols: list[str], period: str = "10y") -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    data = yf.download(symbols, period=period, auto_adjust=True, progress=False)
    close = data.get("Close") if isinstance(data, pd.DataFrame) else None
    if close is None or close.empty:
        return pd.DataFrame()
    if isinstance(close, pd.Series):
        close = close.to_frame(name=symbols[0])
    close = close.dropna(how="all").ffill().dropna(axis=1, how="all")
    close.columns = [str(col).upper() for col in close.columns]
    return close


def run_pairs_suite(symbols: list[str], entry_z: float, exit_z: float, horizon_days: int):
    close = download_close(symbols, period="10y")
    if close.empty or close.shape[1] < 2:
        return {
            "pair_results": [],
            "summary": {"pairs_evaluated": 0, "signals": 0, "win_rate": 0.0, "avg_trade_return": 0.0},
            "error": "insufficient_price_matrix",
        }

    returns = np.log(close / close.shift(1)).dropna(how="all")
    corr = returns.corr().fillna(0.0)

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
    candidates = [(a, b, float(c)) for (a, b), c in upper.items() if float(c) >= 0.65][:30]

    pair_results = []
    all_trades = []
    for left, right, c in candidates:
        pair_df = close[[left, right]].dropna()
        if len(pair_df) < 500:
            continue

        spread = np.log(pair_df[left].clip(lower=1e-6)) - np.log(pair_df[right].clip(lower=1e-6))
        rolling_mean = spread.rolling(120).mean()
        rolling_std = spread.rolling(120).std(ddof=0)
        zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)

        trade_returns = []
        trade_wins = 0
        signals = 0
        for idx in range(130, len(pair_df) - horizon_days):
            z = zscore.iloc[idx]
            if np.isnan(z) or abs(z) < entry_z:
                continue

            entry_left = pair_df[left].iloc[idx]
            entry_right = pair_df[right].iloc[idx]
            exit_idx = min(idx + horizon_days, len(pair_df) - 1)

            # Optional early convergence exit
            for probe in range(idx + 1, min(idx + horizon_days + 1, len(pair_df))):
                probe_z = zscore.iloc[probe]
                if not np.isnan(probe_z) and abs(probe_z) <= exit_z:
                    exit_idx = probe
                    break

            exit_left = pair_df[left].iloc[exit_idx]
            exit_right = pair_df[right].iloc[exit_idx]

            if z > 0:
                # short left, long right
                pnl = ((entry_left - exit_left) / entry_left) + ((exit_right - entry_right) / entry_right)
            else:
                # long left, short right
                pnl = ((exit_left - entry_left) / entry_left) + ((entry_right - exit_right) / entry_right)

            signals += 1
            trade_returns.append(float(pnl))
            if pnl > 0:
                trade_wins += 1

        pair_summary = {
            "pair": f"{left}/{right}",
            "correlation": c,
            "signals": signals,
            "win_rate": (trade_wins / signals) if signals else 0.0,
            "avg_trade_return": safe_mean(trade_returns),
        }
        pair_results.append(pair_summary)
        all_trades.extend(trade_returns)

    summary = {
        "pairs_evaluated": len(pair_results),
        "signals": sum(p["signals"] for p in pair_results),
        "win_rate": (
            sum(p["win_rate"] * p["signals"] for p in pair_results) / max(1, sum(p["signals"] for p in pair_results))
        ),
        "avg_trade_return": safe_mean(all_trades),
        "top_pairs": sorted(pair_results, key=lambda x: (x["win_rate"], x["avg_trade_return"]), reverse=True)[:8],
    }
    return {"pair_results": pair_results, "summary": summary}


def _load_hmm_payload():
    if not HMM_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(HMM_MODEL_PATH)
    except Exception:
        return None


def run_regime_suite(horizon_days: int):
    payload = _load_hmm_payload()
    if not payload:
        return {"error": "hmm_model_missing_or_unreadable", "summary": {"n_samples": 0}}

    model = payload.get("model")
    scaler = payload.get("scaler")
    features_list = payload.get("features_list") or []
    state_map = payload.get("state_map") or {}
    if model is None or scaler is None or not features_list:
        return {"error": "hmm_payload_incomplete", "summary": {"n_samples": 0}}

    tickers = {
        "SPY": "SPY",
        "VIX": "^VIX",
        "TNX": "^TNX",
        "DXY": "DX-Y.NYB",
        "HYG": "HYG",
        "LQD": "LQD",
    }
    raw = yf.download(list(tickers.values()), period="15y", auto_adjust=True, progress=False)
    close = raw.get("Close") if isinstance(raw, pd.DataFrame) else None
    if close is None or close.empty:
        return {"error": "macro_download_failed", "summary": {"n_samples": 0}}

    close = close.rename(columns={v: k for k, v in tickers.items()}).dropna()
    features = build_macro_features(close)
    features = features[[c for c in features_list if c in features.columns]].dropna()
    if len(features) < 500:
        return {"error": "insufficient_macro_features", "summary": {"n_samples": int(len(features))}}

    X = scaler.transform(features.values)
    states = model.predict(X)

    spy_forward = np.log(close["SPY"].shift(-horizon_days) / close["SPY"]).reindex(features.index)
    vix_forward = np.log(close["VIX"].shift(-horizon_days) / close["VIX"]).reindex(features.index)

    rows = []
    for state_id in sorted(set(states.tolist())):
        mask = states == state_id
        idx = features.index[mask]
        spy_slice = spy_forward.reindex(idx).dropna()
        vix_slice = vix_forward.reindex(idx).dropna()
        rows.append(
            {
                "state_id": int(state_id),
                "state_label": state_map.get(int(state_id), f"STATE_{state_id}"),
                "samples": int(mask.sum()),
                "avg_spy_forward_return": float(spy_slice.mean()) if not spy_slice.empty else 0.0,
                "downside_prob": float((spy_slice < 0).mean()) if not spy_slice.empty else 0.0,
                "avg_vix_forward_change": float(vix_slice.mean()) if not vix_slice.empty else 0.0,
            }
        )

    inferred_risk_off = sorted(rows, key=lambda r: (r["avg_spy_forward_return"], -r["avg_vix_forward_change"]))[:1]
    risk_off_state = inferred_risk_off[0]["state_id"] if inferred_risk_off else None

    directional_hits = []
    if risk_off_state is not None:
        for i in range(len(states) - horizon_days):
            pred = -1 if states[i] == risk_off_state else 1
            realized = -1 if spy_forward.iloc[i] < 0 else 1
            directional_hits.append(1.0 if pred == realized else 0.0)

    summary = {
        "n_samples": int(len(features)),
        "horizon_days": int(horizon_days),
        "state_breakdown": rows,
        "risk_off_state_inferred": risk_off_state,
        "directional_accuracy_proxy": safe_mean(directional_hits),
    }
    return {"summary": summary}


def run_strategy_proxy_suite(symbols: list[str], horizon_days: int):
    close = download_close(symbols, period="10y")
    if close.empty:
        return {"error": "no_price_data", "summary": {"samples": 0}}

    market = close.mean(axis=1).dropna()
    ret1 = np.log(market / market.shift(1))
    realized_vol20 = ret1.rolling(20).std() * np.sqrt(252)
    vol_trend = realized_vol20.diff(5)
    mom20 = market / market.rolling(20).mean() - 1.0

    vix = yf.download("^VIX", period="10y", auto_adjust=True, progress=False)
    vix_close = vix.get("Close") if isinstance(vix, pd.DataFrame) else None
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.squeeze()
    if vix_close is None or vix_close.empty:
        vix_close = pd.Series(index=market.index, data=np.nan).ffill().fillna(20.0)

    frame = pd.DataFrame(
        {
            "ret1": ret1,
            "mom20": mom20,
            "vol20": realized_vol20,
            "vol_trend": vol_trend,
            "vix": vix_close.reindex(market.index).ffill(),
            "fwd_ret": np.log(market.shift(-horizon_days) / market),
            "fwd_vol_change": realized_vol20.shift(-horizon_days) - realized_vol20,
        }
    ).dropna()

    if len(frame) < 400:
        return {"error": "insufficient_proxy_samples", "summary": {"samples": int(len(frame))}}

    strategy_hits = {"BULL": [], "BEAR": [], "THETA": [], "VEGA": []}
    strategy_returns = {"BULL": [], "BEAR": [], "THETA": [], "VEGA": []}

    for _, row in frame.iterrows():
        predicted = "THETA"
        if row["mom20"] > 0.015 and row["vix"] < 24:
            predicted = "BULL"
        elif row["mom20"] < -0.015 and row["vix"] > 18:
            predicted = "BEAR"
        elif row["vol_trend"] > 0 and row["vix"] > 20:
            predicted = "VEGA"

        if predicted == "BULL":
            success = row["fwd_ret"] > 0
            proxy_return = row["fwd_ret"]
        elif predicted == "BEAR":
            success = row["fwd_ret"] < 0
            proxy_return = -row["fwd_ret"]
        elif predicted == "VEGA":
            success = row["fwd_vol_change"] > 0
            proxy_return = row["fwd_vol_change"]
        else:  # THETA
            success = abs(row["fwd_ret"]) < 0.02 and row["fwd_vol_change"] <= 0
            proxy_return = -abs(row["fwd_ret"]) + max(0.0, -row["fwd_vol_change"])

        strategy_hits[predicted].append(1.0 if success else 0.0)
        strategy_returns[predicted].append(float(proxy_return))

    by_strategy = {}
    for name in ["BULL", "BEAR", "THETA", "VEGA"]:
        by_strategy[name] = {
            "samples": len(strategy_hits[name]),
            "hit_rate": safe_mean(strategy_hits[name]),
            "avg_proxy_return": safe_mean(strategy_returns[name]),
        }

    summary = {
        "samples": int(len(frame)),
        "horizon_days": int(horizon_days),
        "by_strategy": by_strategy,
        "overall_hit_rate": safe_mean([x for values in strategy_hits.values() for x in values]),
    }
    return {"summary": summary}


def main():
    args = parse_args()

    lookbacks = [lb for lb in args.lookbacks if lb in LOOKBACK_MAP]
    if not lookbacks:
        raise SystemExit("No valid lookbacks provided.")

    symbols = [s.upper() for s in (args.symbols or [])]
    if not symbols:
        symbols = load_default_symbols(args.max_symbols)
    symbols = list(dict.fromkeys(symbols))[: max(1, args.max_symbols)]
    if not symbols:
        raise SystemExit("No symbols available. Pass --symbols or populate config/symbol_list.txt.")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "symbols": symbols,
            "lookbacks": lookbacks,
            "target_daily_return": args.target_daily_return,
            "target_accuracy": args.target_accuracy,
            "pairs_entry_z": args.pairs_entry_z,
            "pairs_exit_z": args.pairs_exit_z,
            "horizon_days": args.horizon_days,
        },
    }

    movement = run_movement_suite(
        symbols=symbols,
        lookbacks=lookbacks,
        workers=args.workers,
        target_accuracy=args.target_accuracy,
        target_daily_return=args.target_daily_return,
    )
    pairs = run_pairs_suite(symbols=symbols, entry_z=args.pairs_entry_z, exit_z=args.pairs_exit_z, horizon_days=args.horizon_days)
    regime = run_regime_suite(horizon_days=args.horizon_days)
    strategy_proxy = run_strategy_proxy_suite(symbols=symbols, horizon_days=args.horizon_days)

    report["movement_suite"] = movement
    report["pairs_suite"] = pairs
    report["regime_suite"] = regime
    report["strategy_proxy_suite"] = strategy_proxy

    aggregate_score_components = []
    aggregate_score_components.append(movement["summary"].get("avg_accuracy", 0.0))
    aggregate_score_components.append(max(0.0, min(1.0, pairs.get("summary", {}).get("win_rate", 0.0))))
    aggregate_score_components.append(max(0.0, min(1.0, regime.get("summary", {}).get("directional_accuracy_proxy", 0.0))))
    aggregate_score_components.append(max(0.0, min(1.0, strategy_proxy.get("summary", {}).get("overall_hit_rate", 0.0))))

    report["massive_overview"] = {
        "predictive_score": safe_mean(aggregate_score_components),
        "movement_hit_rate": movement["summary"].get("hit_rate", 0.0),
        "pairs_win_rate": pairs.get("summary", {}).get("win_rate", 0.0),
        "regime_directional_accuracy_proxy": regime.get("summary", {}).get("directional_accuracy_proxy", 0.0),
        "strategy_router_hit_rate": strategy_proxy.get("summary", {}).get("overall_hit_rate", 0.0),
        "note": "Proxy framework focuses on predictive quality. It is not a direct options PnL backtest.",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n=== MASSIVE PREDICTIVE OVERVIEW ===")
    print(json.dumps(report["massive_overview"], indent=2))
    print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
