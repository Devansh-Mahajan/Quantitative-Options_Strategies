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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.movement_predictor import LOOKBACK_MAP
from scripts.train_hmm import build_macro_features

DEFAULT_LOOKBACKS = ["10y", "5y", "1y", "6mo", "3mo"]
DEFAULT_OUTPUT = "reports/massive_backtest_report.json"
SYMBOL_LIST_PATH = ROOT / "config" / "symbol_list.txt"
HMM_MODEL_PATH = ROOT / "config" / "hmm_macro_model.pkl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Industrial-style all-in-one predictive audit: movement, regime, pairs, and strategy-routing proxies."
    )
    parser.add_argument("--symbols", nargs="+", help="Universe symbols. Defaults to config/symbol_list.txt if omitted.")
    parser.add_argument("--max-symbols", type=int, default=40, help="Cap symbol universe size for runtime control.")
    parser.add_argument("--lookbacks", nargs="+", default=DEFAULT_LOOKBACKS)
    parser.add_argument("--target-daily-return", type=float, default=0.002)
    parser.add_argument("--target-accuracy", type=float, default=0.56)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pairs-entry-z", type=float, default=1.5)
    parser.add_argument("--pairs-exit-z", type=float, default=0.5)
    parser.add_argument("--horizon-days", type=int, default=5, help="Forward evaluation horizon.")
    parser.add_argument("--period", default="10y", help="Yahoo period fallback when start/end dates are not provided.")
    parser.add_argument("--start-date", type=str, help="Backtest start date YYYY-MM-DD.")
    parser.add_argument("--end-date", type=str, help="Backtest end date YYYY-MM-DD.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split for movement model in (0.5, 0.95).")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_date(date_text: str | None):
    if not date_text:
        return None
    return datetime.strptime(date_text, "%Y-%m-%d").date().isoformat()


def load_default_symbols(max_symbols: int) -> list[str]:
    if not SYMBOL_LIST_PATH.exists():
        return []
    with SYMBOL_LIST_PATH.open("r", encoding="utf-8") as handle:
        symbols = [line.strip().upper() for line in handle if line.strip()]
    return list(dict.fromkeys(symbols))[: max(1, max_symbols)]


def safe_mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def safe_std(values):
    return float(np.std(values, ddof=0)) if values else 0.0


def annualize_log_returns(log_returns: np.ndarray) -> float:
    if len(log_returns) == 0:
        return 0.0
    return float(np.exp(np.nanmean(log_returns) * 252.0) - 1.0)


def compute_risk_metrics(log_returns: list[float]) -> dict:
    if not log_returns:
        return {
            "n": 0,
            "mean_daily": 0.0,
            "vol_daily": 0.0,
            "annual_return": 0.0,
            "annual_vol": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    arr = np.asarray(log_returns, dtype=float)
    mean_d = float(np.nanmean(arr))
    vol_d = float(np.nanstd(arr, ddof=0))
    annual_ret = float(np.exp(mean_d * 252.0) - 1.0)
    annual_vol = float(vol_d * np.sqrt(252.0))
    sharpe = (mean_d / vol_d * np.sqrt(252.0)) if vol_d > 1e-10 else 0.0

    downside = arr[arr < 0.0]
    downside_std = float(np.nanstd(downside, ddof=0)) if len(downside) else 0.0
    sortino = (mean_d / downside_std * np.sqrt(252.0)) if downside_std > 1e-10 else 0.0

    equity_curve = np.exp(np.nancumsum(arr))
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve / np.maximum(running_max, 1e-9)) - 1.0
    max_dd = float(np.nanmin(drawdowns)) if len(drawdowns) else 0.0
    calmar = (annual_ret / abs(max_dd)) if max_dd < -1e-10 else 0.0

    pos = arr[arr > 0]
    neg = arr[arr < 0]
    gross_profit = float(np.sum(pos)) if len(pos) else 0.0
    gross_loss = float(abs(np.sum(neg))) if len(neg) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-10 else 0.0

    return {
        "n": int(len(arr)),
        "mean_daily": mean_d,
        "vol_daily": vol_d,
        "annual_return": annual_ret,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": float(np.mean(arr > 0)),
        "profit_factor": profit_factor,
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "brier": 0.0}
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
    }


def download_close(
    symbols: list[str],
    period: str = "10y",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    kwargs = {"auto_adjust": True, "progress": False}
    if start_date:
        kwargs["start"] = start_date
        if end_date:
            kwargs["end"] = end_date
    else:
        kwargs["period"] = period

    data = yf.download(symbols, **kwargs)
    close = data.get("Close") if isinstance(data, pd.DataFrame) else None
    if close is None or close.empty:
        return pd.DataFrame()

    if isinstance(close, pd.Series):
        close = close.to_frame(name=symbols[0])

    close = close.dropna(how="all").ffill().dropna(axis=1, how="all")
    close.columns = [str(col).upper() for col in close.columns]
    return close


def movement_feature_frame(close: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=close.index)
    ret = np.log(close / close.shift(1))
    df["ret_1d"] = ret
    df["ret_5d"] = np.log(close / close.shift(5))
    df["ret_10d"] = np.log(close / close.shift(10))
    df["vol_20"] = ret.rolling(20).std() * np.sqrt(252)
    df["vol_60"] = ret.rolling(60).std() * np.sqrt(252)
    df["momentum_20"] = close / close.rolling(20).mean() - 1.0
    df["momentum_50"] = close / close.rolling(50).mean() - 1.0
    df["target"] = (close.shift(-1) > close).astype(int)
    return df.dropna()


def slice_lookback(df: pd.DataFrame, lookback: str) -> pd.DataFrame:
    if lookback not in LOOKBACK_MAP:
        return df
    rows = LOOKBACK_MAP[lookback]
    return df.tail(rows)


def backtest_symbol_movement_audit(
    symbol: str,
    lookback: str,
    close: pd.Series,
    train_ratio: float,
    target_accuracy: float,
    target_daily_return: float,
) -> dict:
    features = movement_feature_frame(close)
    features = slice_lookback(features, lookback)

    if len(features) < 180:
        return {
            "symbol": symbol,
            "lookback": lookback,
            "error": "insufficient_data",
            "available_rows": int(len(features)),
        }

    split = int(len(features) * train_ratio)
    train = features.iloc[:split]
    test = features.iloc[split:]
    if len(test) < 60:
        return {
            "symbol": symbol,
            "lookback": lookback,
            "error": "insufficient_test_set",
            "available_rows": int(len(features)),
        }

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    X_train = train.drop(columns=["target"])
    y_train = train["target"]
    X_test = test.drop(columns=["target"])
    y_test = test["target"].values.astype(int)

    model.fit(X_train, y_train)
    prob_up = model.predict_proba(X_test)[:, 1]
    pred = (prob_up >= 0.5).astype(int)

    cls = compute_classification_metrics(y_test, pred, prob_up)

    strat_log_rets = np.where(pred == 1, test["ret_1d"].values, -test["ret_1d"].values)
    buy_hold_log_rets = test["ret_1d"].values

    strategy_metrics = compute_risk_metrics(strat_log_rets.tolist())
    benchmark_metrics = compute_risk_metrics(buy_hold_log_rets.tolist())

    strategy_daily = float(np.nanmean(strat_log_rets))
    buyhold_daily = float(np.nanmean(buy_hold_log_rets))

    return {
        "symbol": symbol,
        "lookback": lookback,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "classification": cls,
        "strategy": strategy_metrics,
        "benchmark": benchmark_metrics,
        "strategy_daily_return": strategy_daily,
        "buy_hold_daily_return": buyhold_daily,
        "alpha_daily": strategy_daily - buyhold_daily,
        "meets_targets": bool(cls["accuracy"] >= target_accuracy and strategy_daily >= target_daily_return),
    }


def run_movement_suite(close_prices: pd.DataFrame, lookbacks: list[str], workers: int, train_ratio: float, target_accuracy: float, target_daily_return: float):
    jobs = []
    for symbol in close_prices.columns:
        close = close_prices[symbol].dropna()
        for lb in lookbacks:
            jobs.append((symbol, lb, close))

    if not jobs:
        return {"results": [], "summary": {"total_runs": 0, "valid_runs": 0}}

    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
        futures = {
            pool.submit(
                backtest_symbol_movement_audit,
                symbol,
                lb,
                close,
                train_ratio,
                target_accuracy,
                target_daily_return,
            ): (symbol, lb)
            for symbol, lb, close in jobs
        }
        for future in as_completed(futures):
            results.append(future.result())

    results = sorted(results, key=lambda x: (x.get("symbol", ""), x.get("lookback", "")))
    valid = [row for row in results if "error" not in row]

    summary = {
        "total_runs": len(results),
        "valid_runs": len(valid),
        "hit_rate": safe_mean([1.0 if row.get("meets_targets") else 0.0 for row in valid]),
        "avg_accuracy": safe_mean([row["classification"]["accuracy"] for row in valid]),
        "avg_precision": safe_mean([row["classification"]["precision"] for row in valid]),
        "avg_recall": safe_mean([row["classification"]["recall"] for row in valid]),
        "avg_f1": safe_mean([row["classification"]["f1"] for row in valid]),
        "avg_brier": safe_mean([row["classification"]["brier"] for row in valid]),
        "avg_strategy_daily_return": safe_mean([row["strategy_daily_return"] for row in valid]),
        "avg_alpha_daily": safe_mean([row["alpha_daily"] for row in valid]),
        "avg_strategy_sharpe": safe_mean([row["strategy"]["sharpe"] for row in valid]),
        "avg_strategy_sortino": safe_mean([row["strategy"]["sortino"] for row in valid]),
        "avg_strategy_max_drawdown": safe_mean([row["strategy"]["max_drawdown"] for row in valid]),
    }
    return {"results": results, "summary": summary}


def run_pairs_suite(close: pd.DataFrame, entry_z: float, exit_z: float, horizon_days: int):
    if close.empty or close.shape[1] < 2:
        return {
            "pair_results": [],
            "summary": {"pairs_evaluated": 0, "signals": 0, "win_rate": 0.0, "avg_trade_return": 0.0},
            "error": "insufficient_price_matrix",
        }

    returns = np.log(close / close.shift(1)).dropna(how="all")
    corr = returns.corr().fillna(0.0)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
    candidates = [(a, b, float(c)) for (a, b), c in upper.items() if float(c) >= 0.65][:40]

    pair_results = []
    all_trade_returns = []
    for left, right, corr_val in candidates:
        pair_df = close[[left, right]].dropna()
        if len(pair_df) < 500:
            continue

        spread = np.log(pair_df[left].clip(lower=1e-6)) - np.log(pair_df[right].clip(lower=1e-6))
        mean = spread.rolling(120).mean()
        std = spread.rolling(120).std(ddof=0).replace(0, np.nan)
        zscores = (spread - mean) / std

        trade_returns = []
        for idx in range(130, len(pair_df) - horizon_days):
            z = zscores.iloc[idx]
            if np.isnan(z) or abs(z) < entry_z:
                continue

            e_l = pair_df[left].iloc[idx]
            e_r = pair_df[right].iloc[idx]
            exit_idx = min(idx + horizon_days, len(pair_df) - 1)

            for probe in range(idx + 1, min(idx + horizon_days + 1, len(pair_df))):
                probe_z = zscores.iloc[probe]
                if not np.isnan(probe_z) and abs(probe_z) <= exit_z:
                    exit_idx = probe
                    break

            x_l = pair_df[left].iloc[exit_idx]
            x_r = pair_df[right].iloc[exit_idx]

            if z > 0:
                pnl = ((e_l - x_l) / e_l) + ((x_r - e_r) / e_r)
            else:
                pnl = ((x_l - e_l) / e_l) + ((e_r - x_r) / e_r)

            trade_returns.append(float(pnl))

        risk = compute_risk_metrics(trade_returns)
        pair_results.append(
            {
                "pair": f"{left}/{right}",
                "correlation": corr_val,
                "signals": len(trade_returns),
                "win_rate": risk["win_rate"],
                "avg_trade_return": safe_mean(trade_returns),
                "median_trade_return": float(np.median(trade_returns)) if trade_returns else 0.0,
                "sharpe": risk["sharpe"],
                "sortino": risk["sortino"],
                "max_drawdown": risk["max_drawdown"],
                "profit_factor": risk["profit_factor"],
            }
        )
        all_trade_returns.extend(trade_returns)

    total_signals = sum(row["signals"] for row in pair_results)
    weighted_win = sum(row["win_rate"] * row["signals"] for row in pair_results) / max(1, total_signals)
    portfolio_risk = compute_risk_metrics(all_trade_returns)

    summary = {
        "pairs_evaluated": len(pair_results),
        "signals": total_signals,
        "win_rate": weighted_win,
        "avg_trade_return": safe_mean(all_trade_returns),
        "portfolio_sharpe": portfolio_risk["sharpe"],
        "portfolio_sortino": portfolio_risk["sortino"],
        "portfolio_max_drawdown": portfolio_risk["max_drawdown"],
        "portfolio_profit_factor": portfolio_risk["profit_factor"],
        "top_pairs": sorted(pair_results, key=lambda x: (x["sharpe"], x["avg_trade_return"]), reverse=True)[:10],
    }
    return {"pair_results": pair_results, "summary": summary}


def _load_hmm_payload():
    if not HMM_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(HMM_MODEL_PATH)
    except Exception:
        return None


def run_regime_suite(start_date: str | None, end_date: str | None, period: str, horizon_days: int):
    payload = _load_hmm_payload()
    if not payload:
        return {"error": "hmm_model_missing_or_unreadable", "summary": {"n_samples": 0}}

    model = payload.get("model")
    scaler = payload.get("scaler")
    features_list = payload.get("features_list") or []
    state_map = payload.get("state_map") or {}
    if model is None or scaler is None or not features_list:
        return {"error": "hmm_payload_incomplete", "summary": {"n_samples": 0}}

    tickers = {"SPY": "SPY", "VIX": "^VIX", "TNX": "^TNX", "DXY": "DX-Y.NYB", "HYG": "HYG", "LQD": "LQD"}
    kwargs = {"auto_adjust": True, "progress": False}
    if start_date:
        kwargs["start"] = start_date
        if end_date:
            kwargs["end"] = end_date
    else:
        kwargs["period"] = max(period, "10y")

    raw = yf.download(list(tickers.values()), **kwargs)
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

    spy_fwd = np.log(close["SPY"].shift(-horizon_days) / close["SPY"]).reindex(features.index)
    vix_fwd = np.log(close["VIX"].shift(-horizon_days) / close["VIX"]).reindex(features.index)

    rows = []
    for state_id in sorted(set(states.tolist())):
        mask = states == state_id
        idx = features.index[mask]
        spy_slice = spy_fwd.reindex(idx).dropna()
        vix_slice = vix_fwd.reindex(idx).dropna()
        rows.append(
            {
                "state_id": int(state_id),
                "state_label": state_map.get(int(state_id), f"STATE_{state_id}"),
                "samples": int(mask.sum()),
                "avg_spy_forward_return": float(spy_slice.mean()) if len(spy_slice) else 0.0,
                "downside_prob": float((spy_slice < 0).mean()) if len(spy_slice) else 0.0,
                "avg_vix_forward_change": float(vix_slice.mean()) if len(vix_slice) else 0.0,
            }
        )

    inferred_risk_off = sorted(rows, key=lambda r: (r["avg_spy_forward_return"], -r["avg_vix_forward_change"]))[:1]
    risk_off_state = inferred_risk_off[0]["state_id"] if inferred_risk_off else None

    y_true = []
    y_pred = []
    proxy_returns = []
    if risk_off_state is not None:
        for i in range(len(states) - horizon_days):
            realized_down = 1 if spy_fwd.iloc[i] < 0 else 0
            pred_down = 1 if states[i] == risk_off_state else 0
            y_true.append(realized_down)
            y_pred.append(pred_down)
            proxy_returns.append(-spy_fwd.iloc[i] if pred_down else spy_fwd.iloc[i])

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    cls = compute_classification_metrics(y_true_arr, y_pred_arr, y_pred_arr.astype(float) if len(y_pred_arr) else np.array([]))
    risk = compute_risk_metrics(proxy_returns)

    summary = {
        "n_samples": int(len(features)),
        "horizon_days": int(horizon_days),
        "risk_off_state_inferred": risk_off_state,
        "state_breakdown": rows,
        "directional_accuracy_proxy": cls["accuracy"],
        "directional_precision_proxy": cls["precision"],
        "directional_recall_proxy": cls["recall"],
        "proxy_sharpe": risk["sharpe"],
        "proxy_sortino": risk["sortino"],
        "proxy_max_drawdown": risk["max_drawdown"],
    }
    return {"summary": summary}


def run_strategy_proxy_suite(close: pd.DataFrame, horizon_days: int):
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
    vix_series = vix_close.reindex(market.index).ffill() if isinstance(vix_close, pd.Series) else pd.Series(index=market.index, data=20.0)

    frame = pd.DataFrame(
        {
            "ret1": ret1,
            "mom20": mom20,
            "vol20": realized_vol20,
            "vol_trend": vol_trend,
            "vix": vix_series.fillna(20.0),
            "fwd_ret": np.log(market.shift(-horizon_days) / market),
            "fwd_vol_change": realized_vol20.shift(-horizon_days) - realized_vol20,
        }
    ).dropna()

    if len(frame) < 400:
        return {"error": "insufficient_proxy_samples", "summary": {"samples": int(len(frame))}}

    hits = {"BULL": [], "BEAR": [], "THETA": [], "VEGA": []}
    rets = {"BULL": [], "BEAR": [], "THETA": [], "VEGA": []}

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
            proxy_ret = row["fwd_ret"]
        elif predicted == "BEAR":
            success = row["fwd_ret"] < 0
            proxy_ret = -row["fwd_ret"]
        elif predicted == "VEGA":
            success = row["fwd_vol_change"] > 0
            proxy_ret = row["fwd_vol_change"]
        else:
            success = abs(row["fwd_ret"]) < 0.02 and row["fwd_vol_change"] <= 0
            proxy_ret = -abs(row["fwd_ret"]) + max(0.0, -row["fwd_vol_change"])

        hits[predicted].append(1.0 if success else 0.0)
        rets[predicted].append(float(proxy_ret))

    by_strategy = {}
    for name in ["BULL", "BEAR", "THETA", "VEGA"]:
        risk = compute_risk_metrics(rets[name])
        by_strategy[name] = {
            "samples": len(hits[name]),
            "hit_rate": safe_mean(hits[name]),
            "avg_proxy_return": safe_mean(rets[name]),
            "sharpe": risk["sharpe"],
            "sortino": risk["sortino"],
            "max_drawdown": risk["max_drawdown"],
            "profit_factor": risk["profit_factor"],
        }

    all_returns = [x for values in rets.values() for x in values]
    all_hits = [x for values in hits.values() for x in values]
    overall_risk = compute_risk_metrics(all_returns)

    return {
        "summary": {
            "samples": int(len(frame)),
            "horizon_days": int(horizon_days),
            "overall_hit_rate": safe_mean(all_hits),
            "overall_sharpe": overall_risk["sharpe"],
            "overall_sortino": overall_risk["sortino"],
            "overall_max_drawdown": overall_risk["max_drawdown"],
            "overall_profit_factor": overall_risk["profit_factor"],
            "by_strategy": by_strategy,
        }
    }


def main():
    args = parse_args()

    start_date = validate_date(args.start_date)
    end_date = validate_date(args.end_date)
    if start_date and end_date and start_date > end_date:
        raise SystemExit("--start-date must be <= --end-date")
    if not (0.5 < args.train_ratio < 0.95):
        raise SystemExit("--train-ratio must be between 0.5 and 0.95")

    lookbacks = [lb for lb in args.lookbacks if lb in LOOKBACK_MAP]
    if not lookbacks:
        raise SystemExit("No valid lookbacks provided.")

    symbols = [s.upper() for s in (args.symbols or [])]
    if not symbols:
        symbols = load_default_symbols(args.max_symbols)
    symbols = list(dict.fromkeys(symbols))[: max(1, args.max_symbols)]
    if not symbols:
        raise SystemExit("No symbols available. Pass --symbols or populate config/symbol_list.txt.")

    close = download_close(symbols, period=args.period, start_date=start_date, end_date=end_date)

    movement = run_movement_suite(
        close_prices=close,
        lookbacks=lookbacks,
        workers=args.workers,
        train_ratio=args.train_ratio,
        target_accuracy=args.target_accuracy,
        target_daily_return=args.target_daily_return,
    )
    pairs = run_pairs_suite(close=close, entry_z=args.pairs_entry_z, exit_z=args.pairs_exit_z, horizon_days=args.horizon_days)
    regime = run_regime_suite(start_date=start_date, end_date=end_date, period=args.period, horizon_days=args.horizon_days)
    strategy_proxy = run_strategy_proxy_suite(close=close, horizon_days=args.horizon_days)

    components = [
        movement["summary"].get("avg_accuracy", 0.0),
        max(0.0, min(1.0, pairs.get("summary", {}).get("win_rate", 0.0))),
        max(0.0, min(1.0, regime.get("summary", {}).get("directional_accuracy_proxy", 0.0))),
        max(0.0, min(1.0, strategy_proxy.get("summary", {}).get("overall_hit_rate", 0.0))),
    ]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "symbols": symbols,
            "max_symbols": args.max_symbols,
            "lookbacks": lookbacks,
            "target_daily_return": args.target_daily_return,
            "target_accuracy": args.target_accuracy,
            "pairs_entry_z": args.pairs_entry_z,
            "pairs_exit_z": args.pairs_exit_z,
            "horizon_days": args.horizon_days,
            "period": args.period,
            "start_date": start_date,
            "end_date": end_date,
            "train_ratio": args.train_ratio,
        },
        "movement_suite": movement,
        "pairs_suite": pairs,
        "regime_suite": regime,
        "strategy_proxy_suite": strategy_proxy,
        "massive_overview": {
            "predictive_score": safe_mean(components),
            "movement_hit_rate": movement["summary"].get("hit_rate", 0.0),
            "movement_avg_accuracy": movement["summary"].get("avg_accuracy", 0.0),
            "pairs_win_rate": pairs.get("summary", {}).get("win_rate", 0.0),
            "pairs_sharpe": pairs.get("summary", {}).get("portfolio_sharpe", 0.0),
            "regime_directional_accuracy_proxy": regime.get("summary", {}).get("directional_accuracy_proxy", 0.0),
            "strategy_router_hit_rate": strategy_proxy.get("summary", {}).get("overall_hit_rate", 0.0),
            "strategy_router_sharpe": strategy_proxy.get("summary", {}).get("overall_sharpe", 0.0),
            "note": "Predictive audit only. This is not a full options-PnL simulator.",
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n=== INDUSTRIAL PREDICTIVE AUDIT OVERVIEW ===")
    print(json.dumps(report["massive_overview"], indent=2))
    print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
