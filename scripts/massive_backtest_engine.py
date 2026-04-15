import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import math
import sys

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import (
    DELTA_MAX,
    DELTA_MIN,
    OPTION_DELAY_MAX_UNDERLYING_MOVE_PCT,
    OPTION_PRICING_RISK_FREE_RATE,
)
from core.delay_aware_options import option_greeks, option_price
from core.ml_alpha import backtest_alpha_strategy
from core.movement_predictor import LOOKBACK_MAP, backtest_symbol_movement, lookback_days
from core.operations_reporting import archive_backtest_artifacts
from core.quant_models import black_scholes_price, binomial_option_price, monte_carlo_option_price
from core.resource_profile import load_resource_profile
from core.strategy_regime import BUCKETS, STRATEGY_PROFILES, classify_market_state, combine_profile_with_state, clamp
from core.universe_maintenance import download_close_matrix, load_symbol_file, resolve_download_symbol
from scripts.train_hmm import build_macro_features

DEFAULT_LOOKBACKS = ["10y", "5y", "3y", "1y", "6mo", "3mo", "ytd"]
DEFAULT_OUTPUT = "reports/massive_backtest_report.json"
SYMBOL_LIST_PATH = ROOT / "config" / "symbol_list.txt"
HMM_MODEL_PATH = ROOT / "config" / "hmm_macro_model.pkl"
INTRADAY_BARS_PER_DAY = 26
TARGET_OPTION_ABS_DELTA = 0.20
VECTORBT_LOCAL_PATH = ROOT / "third_party" / "vectorbt"


def parse_args():
    resource_profile = load_resource_profile(ROOT)
    parser = argparse.ArgumentParser(
        description="All-in-one predictive backtest suite for movement, regime, pairs, and strategy-routing proxies."
    )
    parser.add_argument("--symbols", nargs="+", help="Universe symbols. Defaults to config/symbol_list.txt if omitted.")
    parser.add_argument("--max-symbols", type=int, default=40, help="Cap symbol universe size for faster runs.")
    parser.add_argument("--lookbacks", nargs="+", default=DEFAULT_LOOKBACKS)
    parser.add_argument("--target-daily-return", type=float, default=0.002)
    parser.add_argument("--target-accuracy", type=float, default=0.56)
    parser.add_argument("--workers", type=int, default=resource_profile.backtest_workers)
    parser.add_argument("--pairs-entry-z", type=float, default=1.5)
    parser.add_argument("--pairs-exit-z", type=float, default=0.5)
    parser.add_argument("--horizon-days", type=int, default=5, help="Evaluation horizon for regime/pairs/proxy tests.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_default_symbols(max_symbols: int) -> list[str]:
    symbols = load_symbol_file(SYMBOL_LIST_PATH)
    return symbols[: max(1, max_symbols)]


def annual_to_daily(annual_return: float) -> float:
    if annual_return <= -0.999:
        return -1.0
    return float((1 + annual_return) ** (1 / 252) - 1)


def safe_mean(values):
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(sum(clean) / len(clean)) if clean else 0.0


def _bootstrap_mean_ci(values, *, n_bootstrap: int = 300, seed: int = 42) -> dict:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return {"samples": 0, "mean": 0.0, "p10": 0.0, "p90": 0.0, "dispersion": 0.0}

    arr = np.asarray(clean, dtype=float)
    rng = np.random.default_rng(seed)
    if arr.size == 1:
        return {
            "samples": int(arr.size),
            "mean": float(arr[0]),
            "p10": float(arr[0]),
            "p90": float(arr[0]),
            "dispersion": 0.0,
        }

    draws = np.array([rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(max(50, n_bootstrap))], dtype=float)
    return {
        "samples": int(arr.size),
        "mean": float(arr.mean()),
        "p10": float(np.percentile(draws, 10)),
        "p90": float(np.percentile(draws, 90)),
        "dispersion": float(np.std(arr, ddof=0)),
    }


def _window_metric_summary(metric_map: dict[str, float], *, threshold: float) -> dict:
    clean = {str(k): float(v) for k, v in (metric_map or {}).items() if v is not None}
    values = list(clean.values())
    ci = _bootstrap_mean_ci(values)
    pass_rate = safe_mean([1.0 if value >= threshold else 0.0 for value in values])
    stability_score = clamp(1.0 - (ci.get("dispersion", 0.0) * 4.0), 0.0, 1.0)
    return {
        "threshold": float(threshold),
        "valid_windows": int(len(values)),
        "pass_rate": float(pass_rate),
        "stability_score": float(stability_score),
        "confidence_band": ci,
        "window_values": clean,
    }


def _safe_float_or_none(value) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _extract_valid_metric_windows(
    rows_by_lookback: dict | None,
    *,
    metric_key: str,
    min_count_key: str | None = None,
    min_count: float = 1.0,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for lookback, row in (rows_by_lookback or {}).items():
        if not isinstance(row, dict):
            continue
        if "error" in row:
            continue
        summary = row.get("summary", row)
        if not isinstance(summary, dict):
            continue
        if min_count_key is not None:
            count = _safe_float_or_none(summary.get(min_count_key, 0.0)) or 0.0
            if count < float(min_count):
                continue
        numeric = _safe_float_or_none(summary.get(metric_key))
        if numeric is None:
            continue
        out[str(lookback)] = numeric
    return out


def _extract_movement_window_scores(report: dict) -> dict[str, float]:
    movement_summary = (report.get("movement_suite") or {}).get("summary") or {}
    return _extract_valid_metric_windows(
        movement_summary.get("by_lookback"),
        metric_key="avg_accuracy",
        min_count_key="valid_runs",
        min_count=1,
    )


def _extract_pairs_window_scores(report: dict) -> dict[str, float]:
    return _extract_valid_metric_windows(
        (report.get("pairs_suite") or {}).get("results_by_lookback"),
        metric_key="win_rate",
        min_count_key="pairs_evaluated",
        min_count=1,
    )


def _extract_regime_window_scores(report: dict) -> dict[str, float]:
    return _extract_valid_metric_windows(
        (report.get("regime_suite") or {}).get("results_by_lookback"),
        metric_key="directional_accuracy_proxy",
        min_count_key="n_samples",
        min_count=120,
    )


def _extract_strategy_window_scores(report: dict) -> tuple[str, dict[str, float]]:
    strategy_profiles = (report.get("strategy_profile_suite") or {}).get("results_by_lookback") or {}
    profile_scores: dict[str, float] = {}
    for lookback, result in strategy_profiles.items():
        if not isinstance(result, dict) or "error" in result:
            continue
        summary = result.get("summary") or {}
        if (_safe_float_or_none(summary.get("samples")) or 0.0) < 1:
            continue
        current_state = summary.get("current_state_best_profile") or {}
        best_overall = summary.get("best_profile_overall") or {}
        numeric = _safe_float_or_none(current_state.get("score"))
        if numeric is None:
            numeric = _safe_float_or_none(best_overall.get("score"))
        if numeric is None:
            continue
        profile_scores[str(lookback)] = numeric
    if profile_scores:
        return "strategy_profile", profile_scores

    proxy_scores = _extract_valid_metric_windows(
        (report.get("strategy_proxy_suite") or {}).get("results_by_lookback"),
        metric_key="overall_hit_rate",
        min_count_key="samples",
        min_count=400,
    )
    return "strategy_proxy", proxy_scores


def _infer_total_windows(report: dict, *window_maps: dict[str, float]) -> int:
    configured = (report.get("config") or {}).get("lookbacks") or []
    if configured:
        return max(1, len(configured))
    discovered = [len(metric_map) for metric_map in window_maps if metric_map]
    return max([1, *discovered])


def _window_coverage(valid_windows: int, total_windows: int) -> float:
    return clamp(valid_windows / max(total_windows, 1), 0.0, 1.0)


def _option_signal_score(report: dict, overview: dict) -> float:
    delay_summary = (report.get("delay_quote_suite") or {}).get("summary") or {}
    put_summary = delay_summary.get("puts") or {}
    call_summary = delay_summary.get("calls") or {}
    short_win_rate = safe_mean(
        [
            _safe_float_or_none(put_summary.get("delay_filtered_win_rate")),
            _safe_float_or_none(call_summary.get("delay_filtered_win_rate")),
        ]
    )
    if short_win_rate <= 0:
        short_win_rate = safe_mean(
            [
                _safe_float_or_none(overview.get("delay_filtered_put_win_rate")),
                _safe_float_or_none(overview.get("delay_filtered_call_win_rate")),
            ]
        )

    ensemble_summary = (((report.get("option_model_suite") or {}).get("summary") or {}).get("models") or {}).get("ensemble") or {}
    long_win_rate = _safe_float_or_none(ensemble_summary.get("long_win_rate"))
    if long_win_rate is None:
        long_win_rate = _safe_float_or_none(overview.get("option_model_ensemble_win_rate")) or 0.0

    edge_pct = _safe_float_or_none(ensemble_summary.get("avg_edge_pct"))
    if edge_pct is None:
        edge_pct = _safe_float_or_none(overview.get("option_model_ensemble_edge_pct")) or 0.0
    edge_score = clamp(0.50 + (edge_pct * 3.0), 0.0, 1.0)

    return clamp((0.55 * short_win_rate) + (0.25 * long_win_rate) + (0.20 * edge_score), 0.0, 1.0)


def _alpha_signal_score(report: dict, overview: dict) -> float:
    ml_summary = (report.get("ml_alpha_suite") or {}).get("summary") or {}
    info_coeff = _safe_float_or_none(ml_summary.get("avg_information_coefficient"))
    if info_coeff is None:
        info_coeff = _safe_float_or_none(overview.get("ml_alpha_information_coefficient")) or 0.0
    long_only_summary = ml_summary.get("long_only") or {}
    long_only_sharpe = _safe_float_or_none(long_only_summary.get("sharpe_ratio"))
    if long_only_sharpe is None:
        long_only_sharpe = _safe_float_or_none(overview.get("ml_alpha_long_only_sharpe")) or 0.0
    return clamp(0.45 + (info_coeff * 6.0) + (long_only_sharpe * 0.08), 0.0, 1.0)


def _methodology_score(report: dict, *, total_windows: int, windows: dict[str, dict]) -> tuple[float, dict[str, float]]:
    delay_summary = (report.get("delay_quote_suite") or {}).get("summary") or {}
    delay_puts = delay_summary.get("puts") or {}
    delay_calls = delay_summary.get("calls") or {}
    avg_delay_samples = safe_mean(
        [
            _safe_float_or_none(delay_puts.get("delay_filtered_samples")),
            _safe_float_or_none(delay_calls.get("delay_filtered_samples")),
        ]
    )
    delay_sample_score = clamp(avg_delay_samples / 5000.0, 0.0, 1.0)

    ensemble_summary = (((report.get("option_model_suite") or {}).get("summary") or {}).get("models") or {}).get("ensemble") or {}
    option_signal_density = clamp(
        (_safe_float_or_none(ensemble_summary.get("avg_signals_per_symbol")) or 0.0) / 100.0,
        0.0,
        1.0,
    )

    ml_summary = (report.get("ml_alpha_suite") or {}).get("summary") or {}
    alpha_horizon_score = clamp((_safe_float_or_none(ml_summary.get("months_tested")) or 0.0) / 36.0, 0.0, 1.0)

    coverage = {
        "movement_coverage": _window_coverage(windows["movement"]["valid_windows"], total_windows),
        "pairs_coverage": _window_coverage(windows["pairs"]["valid_windows"], total_windows),
        "regime_coverage": _window_coverage(windows["regime"]["valid_windows"], total_windows),
        "strategy_coverage": _window_coverage(windows["strategy"]["valid_windows"], total_windows),
        "delay_sample_depth": delay_sample_score,
        "option_signal_depth": option_signal_density,
        "alpha_horizon_depth": alpha_horizon_score,
    }
    methodology_score = safe_mean(list(coverage.values()))
    return methodology_score, coverage


def _build_institutional_robustness(report: dict) -> dict:
    overview = report.get("massive_overview") or {}
    movement_by_lookback = _extract_movement_window_scores(report)
    pairs_by_lookback = _extract_pairs_window_scores(report)
    regime_by_lookback = _extract_regime_window_scores(report)
    strategy_label, strategy_by_lookback = _extract_strategy_window_scores(report)
    total_windows = _infer_total_windows(
        report,
        movement_by_lookback,
        pairs_by_lookback,
        regime_by_lookback,
        strategy_by_lookback,
    )

    movement_summary = _window_metric_summary(movement_by_lookback, threshold=0.505)
    pairs_summary = _window_metric_summary(pairs_by_lookback, threshold=0.52)
    regime_summary = _window_metric_summary(regime_by_lookback, threshold=0.505)
    strategy_summary = _window_metric_summary(strategy_by_lookback, threshold=0.58 if strategy_label == "strategy_profile" else 0.51)

    avg_quote_error = _safe_float_or_none(overview.get("avg_recent_quote_error_pct")) or 0.0
    option_score = _option_signal_score(report, overview)
    alpha_score = _alpha_signal_score(report, overview)

    breadth_score = safe_mean(
        [
            movement_summary["pass_rate"],
            pairs_summary["pass_rate"],
            regime_summary["pass_rate"],
            strategy_summary["pass_rate"],
        ]
    )
    stability_score = safe_mean(
        [
            movement_summary["stability_score"],
            pairs_summary["stability_score"],
            regime_summary["stability_score"],
            strategy_summary["stability_score"],
        ]
    )
    execution_score = clamp(1.0 - (avg_quote_error * 8.0), 0.0, 1.0)
    methodology_score, methodology_breakdown = _methodology_score(
        report,
        total_windows=total_windows,
        windows={
            "movement": movement_summary,
            "pairs": pairs_summary,
            "regime": regime_summary,
            "strategy": strategy_summary,
        },
    )

    institutional_score = round(
        (
            0.24 * float(overview.get("predictive_score", 0.0))
            + 0.18 * breadth_score
            + 0.17 * stability_score
            + 0.14 * execution_score
            + 0.12 * methodology_score
            + 0.10 * option_score
            + 0.05 * alpha_score
        ),
        6,
    )

    gates = {
        "predictive_score": bool(float(overview.get("predictive_score", 0.0)) >= 0.54),
        "movement_breadth": bool(movement_summary["pass_rate"] >= 0.50),
        "pairs_breadth": bool(pairs_summary["pass_rate"] >= 0.50),
        "regime_breadth": bool(regime_summary["pass_rate"] >= 0.60),
        "strategy_breadth": bool(strategy_summary["pass_rate"] >= 0.67),
        "execution_quality": bool(avg_quote_error <= 0.08),
        "option_quality": bool(option_score >= 0.55),
        "evidence_quality": bool(methodology_score >= 0.75),
    }

    passed_gates = sum(1 for value in gates.values() if value)
    critical_gates = (
        gates["predictive_score"]
        and gates["strategy_breadth"]
        and gates["execution_quality"]
        and gates["evidence_quality"]
    )

    if institutional_score >= 0.69 and critical_gates and passed_gates >= 6:
        deployment_tier = "institutional_candidate"
    elif institutional_score >= 0.58 and passed_gates >= 4:
        deployment_tier = "paper_candidate"
    else:
        deployment_tier = "research_only"

    return {
        "institutional_score": institutional_score,
        "deployment_tier": deployment_tier,
        "breadth_score": round(breadth_score, 6),
        "stability_score": round(stability_score, 6),
        "execution_quality_score": round(execution_score, 6),
        "methodology_score": round(methodology_score, 6),
        "option_quality_score": round(option_score, 6),
        "alpha_quality_score": round(alpha_score, 6),
        "gates": gates,
        "evidence": methodology_breakdown,
        "windows": {
            "movement": movement_summary,
            "pairs": pairs_summary,
            "regime": regime_summary,
            strategy_label: strategy_summary,
        },
        "note": "Institutional robustness emphasizes valid-window breadth, stability, execution realism, and evidence depth. It is a deployment filter, not a profit guarantee.",
    }


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
    by_lookback = {}
    for lookback in lookbacks:
        lb_rows = [row for row in valid if row.get("lookback") == lookback]
        by_lookback[lookback] = {
            "valid_runs": len(lb_rows),
            "avg_accuracy": safe_mean([row.get("accuracy", 0.0) for row in lb_rows]),
            "avg_strategy_daily_return": safe_mean([row.get("strategy_daily_return", 0.0) for row in lb_rows]),
            "avg_alpha_daily": safe_mean([row.get("alpha_daily", 0.0) for row in lb_rows]),
            "hit_rate": safe_mean([1.0 if row.get("meets_targets") else 0.0 for row in lb_rows]),
        }

    summary = {
        "total_runs": len(results),
        "valid_runs": len(valid),
        "hit_rate": safe_mean([1.0 if row.get("meets_targets") else 0.0 for row in valid]),
        "avg_accuracy": safe_mean([row.get("accuracy", 0.0) for row in valid]),
        "avg_strategy_daily_return": safe_mean([row.get("strategy_daily_return", 0.0) for row in valid]),
        "avg_alpha_daily": safe_mean([row.get("alpha_daily", 0.0) for row in valid]),
        "by_lookback": by_lookback,
    }
    return {"results": results, "summary": summary}


def download_close(symbols: list[str], period: str = "10y") -> pd.DataFrame:
    return download_close_matrix(symbols, period=period, auto_adjust=True, progress=False).ffill().dropna(axis=1, how="all")


def _slice_frame_to_lookback(frame: pd.DataFrame | pd.Series, lookback: str):
    if frame is None or getattr(frame, "empty", True):
        return frame
    if lookback == "ytd":
        cutoff = pd.Timestamp(datetime.now(timezone.utc).year, 1, 1)
    else:
        cutoff = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=lookback_days(lookback))
    return frame.loc[frame.index >= cutoff]


def _window_summary(results_by_lookback: dict[str, dict], metric_key: str) -> dict[str, float]:
    out = {}
    for lookback, result in results_by_lookback.items():
        summary = result.get("summary", {})
        if metric_key in summary:
            out[lookback] = float(summary.get(metric_key, 0.0))
    return out


def _run_pairs_suite_from_close(close: pd.DataFrame, entry_z: float, exit_z: float, horizon_days: int):
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


def run_pairs_suite(symbols: list[str], lookbacks: list[str], entry_z: float, exit_z: float, horizon_days: int):
    close = download_close(symbols, period="10y")
    results_by_lookback = {}
    for lookback in lookbacks:
        window_close = _slice_frame_to_lookback(close, lookback)
        results_by_lookback[lookback] = _run_pairs_suite_from_close(window_close, entry_z, exit_z, horizon_days)

    valid_win_rates = [
        result.get("summary", {}).get("win_rate", 0.0)
        for result in results_by_lookback.values()
        if "error" not in result
    ]
    valid_trade_returns = [
        result.get("summary", {}).get("avg_trade_return", 0.0)
        for result in results_by_lookback.values()
        if "error" not in result
    ]
    summary = {
        "by_lookback": {
            lookback: result.get("summary", {})
            for lookback, result in results_by_lookback.items()
        },
        "win_rate": safe_mean(valid_win_rates),
        "avg_trade_return": safe_mean(valid_trade_returns),
        "valid_windows": sum(1 for result in results_by_lookback.values() if "error" not in result),
    }
    return {"results_by_lookback": results_by_lookback, "summary": summary}


def _load_hmm_payload():
    if not HMM_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(HMM_MODEL_PATH)
    except Exception:
        return None


def run_regime_suite(horizon_days: int, lookbacks: list[str]):
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

    results_by_lookback = {}
    for lookback in lookbacks:
        window_features = _slice_frame_to_lookback(features, lookback)
        if window_features is None or window_features.empty or len(window_features) < 120:
            results_by_lookback[lookback] = {
                "error": "insufficient_macro_features",
                "summary": {"n_samples": int(len(window_features) if window_features is not None else 0)},
            }
            continue

        window_states = model.predict(scaler.transform(window_features.values))
        window_spy_forward = spy_forward.reindex(window_features.index)
        window_vix_forward = vix_forward.reindex(window_features.index)

        rows = []
        for state_id in sorted(set(window_states.tolist())):
            mask = window_states == state_id
            idx = window_features.index[mask]
            spy_slice = window_spy_forward.reindex(idx).dropna()
            vix_slice = window_vix_forward.reindex(idx).dropna()
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
            for i in range(len(window_states) - horizon_days):
                pred = -1 if window_states[i] == risk_off_state else 1
                realized = -1 if window_spy_forward.iloc[i] < 0 else 1
                directional_hits.append(1.0 if pred == realized else 0.0)

        results_by_lookback[lookback] = {
            "summary": {
                "n_samples": int(len(window_features)),
                "horizon_days": int(horizon_days),
                "state_breakdown": rows,
                "risk_off_state_inferred": risk_off_state,
                "directional_accuracy_proxy": safe_mean(directional_hits),
            }
        }

    summary = {
        "by_lookback": {lookback: result.get("summary", {}) for lookback, result in results_by_lookback.items()},
        "directional_accuracy_proxy": safe_mean(
            [
                result.get("summary", {}).get("directional_accuracy_proxy", 0.0)
                for result in results_by_lookback.values()
                if "error" not in result
            ]
        ),
        "valid_windows": sum(1 for result in results_by_lookback.values() if "error" not in result),
    }
    return {"results_by_lookback": results_by_lookback, "summary": summary}


def _run_strategy_proxy_suite_from_close(close: pd.DataFrame, vix_close: pd.Series, horizon_days: int):
    if close.empty:
        return {"error": "no_price_data", "summary": {"samples": 0}}

    market = close.mean(axis=1).dropna()
    ret1 = np.log(market / market.shift(1))
    realized_vol20 = ret1.rolling(20).std() * np.sqrt(252)
    vol_trend = realized_vol20.diff(5)
    mom20 = market / market.rolling(20).mean() - 1.0

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


def run_strategy_proxy_suite(symbols: list[str], lookbacks: list[str], horizon_days: int):
    close = download_close(symbols, period="10y")
    vix = download_close(["^VIX"], period="10y")
    vix_close = None
    if not vix.empty:
        if "^VIX" in vix.columns:
            vix_close = vix["^VIX"]
        elif "VIX" in vix.columns:
            vix_close = vix["VIX"]

    results_by_lookback = {}
    for lookback in lookbacks:
        window_close = _slice_frame_to_lookback(close, lookback)
        window_vix = _slice_frame_to_lookback(vix_close, lookback) if vix_close is not None else None
        results_by_lookback[lookback] = _run_strategy_proxy_suite_from_close(window_close, window_vix, horizon_days)

    summary = {
        "by_lookback": {lookback: result.get("summary", {}) for lookback, result in results_by_lookback.items()},
        "overall_hit_rate": safe_mean(
            [
                result.get("summary", {}).get("overall_hit_rate", 0.0)
                for result in results_by_lookback.values()
                if "error" not in result
            ]
        ),
        "valid_windows": sum(1 for result in results_by_lookback.values() if "error" not in result),
    }
    return {"results_by_lookback": results_by_lookback, "summary": summary}


def _bucket_proxy_outcome(bucket: str, row: pd.Series) -> tuple[bool, float]:
    fwd_ret = float(row["fwd_ret"])
    fwd_vol_change = float(row["fwd_vol_change"])

    if bucket == "BULL":
        return fwd_ret > 0, fwd_ret
    if bucket == "BEAR":
        return fwd_ret < 0, -fwd_ret
    if bucket == "VEGA":
        proxy_return = (0.60 * abs(fwd_ret)) + (0.40 * fwd_vol_change)
        return fwd_vol_change > 0 or abs(fwd_ret) > 0.015, proxy_return

    proxy_return = (-abs(fwd_ret)) + max(0.0, -fwd_vol_change)
    return abs(fwd_ret) < 0.02 and fwd_vol_change <= 0, proxy_return


def _profile_score(hit_rate: float, avg_proxy_return: float) -> float:
    return_component = clamp(0.50 + (avg_proxy_return * 18.0), 0.0, 1.0)
    return round((0.60 * float(hit_rate)) + (0.40 * return_component), 4)


def _run_strategy_profile_suite_from_close(close: pd.DataFrame, vix_close: pd.Series | None, horizon_days: int):
    if close.empty:
        return {"error": "no_price_data", "summary": {"samples": 0}}

    market = close.mean(axis=1).dropna()
    ret1 = np.log(market / market.shift(1))
    realized_vol20 = ret1.rolling(20).std() * np.sqrt(252)
    vol_trend = realized_vol20.diff(5)
    mom20 = market / market.rolling(20).mean() - 1.0

    if vix_close is None or vix_close.empty:
        vix_close = pd.Series(index=market.index, data=np.nan).ffill().fillna(20.0)

    frame = pd.DataFrame(
        {
            "mom20": mom20,
            "vol20": realized_vol20,
            "vol_trend": vol_trend,
            "vix": vix_close.reindex(market.index).ffill(),
            "fwd_ret": np.log(market.shift(-horizon_days) / market),
            "fwd_vol_change": realized_vol20.shift(-horizon_days) - realized_vol20,
        }
    ).dropna()

    if len(frame) < 400:
        return {"error": "insufficient_profile_samples", "summary": {"samples": int(len(frame))}}

    frame["market_state"] = [
        classify_market_state(row.mom20, row.vix, row.vol_trend)
        for row in frame.itertuples(index=False)
    ]

    state_distribution = {
        state: round(float((frame["market_state"] == state).mean()), 4)
        for state in sorted(frame["market_state"].unique())
    }
    current_market_state = str(frame["market_state"].iloc[-1])

    results_by_profile = {}
    best_profile_by_state: dict[str, dict[str, object]] = {}
    overall_candidates = []

    for profile_name in STRATEGY_PROFILES:
        profile_hits = []
        profile_returns = []
        bucket_selection = {bucket: 0 for bucket in BUCKETS}
        state_hits: dict[str, list[float]] = {}
        state_returns: dict[str, list[float]] = {}

        for row in frame.itertuples(index=False):
            market_state = str(row.market_state)
            combined = combine_profile_with_state(profile_name, market_state)
            selected_bucket = max(BUCKETS, key=lambda bucket: (combined.get(bucket, 0.0), bucket))
            success, proxy_return = _bucket_proxy_outcome(selected_bucket, pd.Series(row._asdict()))

            profile_hits.append(1.0 if success else 0.0)
            profile_returns.append(float(proxy_return))
            bucket_selection[selected_bucket] += 1
            state_hits.setdefault(market_state, []).append(1.0 if success else 0.0)
            state_returns.setdefault(market_state, []).append(float(proxy_return))

        by_state = {}
        for state, hits in state_hits.items():
            avg_proxy_return = safe_mean(state_returns.get(state, []))
            hit_rate = safe_mean(hits)
            score = _profile_score(hit_rate, avg_proxy_return)
            by_state[state] = {
                "samples": len(hits),
                "hit_rate": hit_rate,
                "avg_proxy_return": avg_proxy_return,
                "score": score,
            }

            incumbent = best_profile_by_state.get(state)
            if incumbent is None or score > float(incumbent.get("score", -1.0)):
                best_profile_by_state[state] = {
                    "profile": profile_name,
                    "score": score,
                    "hit_rate": hit_rate,
                    "avg_proxy_return": avg_proxy_return,
                    "samples": len(hits),
                }

        avg_proxy_return = safe_mean(profile_returns)
        hit_rate = safe_mean(profile_hits)
        score = _profile_score(hit_rate, avg_proxy_return)
        overall_candidates.append((score, profile_name, hit_rate, avg_proxy_return))

        results_by_profile[profile_name] = {
            "label": STRATEGY_PROFILES[profile_name]["label"],
            "thesis": STRATEGY_PROFILES[profile_name]["thesis"],
            "summary": {
                "samples": int(len(profile_hits)),
                "hit_rate": hit_rate,
                "avg_proxy_return": avg_proxy_return,
                "score": score,
                "bucket_selection_share": {
                    bucket: round(bucket_selection[bucket] / max(len(profile_hits), 1), 4)
                    for bucket in BUCKETS
                },
                "by_state": by_state,
            },
        }

    overall_candidates.sort(reverse=True)
    best_overall = overall_candidates[0] if overall_candidates else (0.0, "all_weather", 0.0, 0.0)

    summary = {
        "samples": int(len(frame)),
        "state_distribution": state_distribution,
        "current_market_state": current_market_state,
        "best_profile_overall": {
            "profile": best_overall[1],
            "score": best_overall[0],
            "hit_rate": best_overall[2],
            "avg_proxy_return": best_overall[3],
        },
        "best_profile_by_state": best_profile_by_state,
        "current_state_best_profile": best_profile_by_state.get(
            current_market_state,
            {
                "profile": best_overall[1],
                "score": best_overall[0],
                "hit_rate": best_overall[2],
                "avg_proxy_return": best_overall[3],
                "samples": int(len(frame)),
            },
        ),
    }
    return {"results_by_profile": results_by_profile, "summary": summary}


def run_strategy_profile_suite(symbols: list[str], lookbacks: list[str], horizon_days: int):
    close = download_close(symbols, period="10y")
    vix = download_close(["^VIX"], period="10y")
    vix_close = None
    if not vix.empty:
        if "^VIX" in vix.columns:
            vix_close = vix["^VIX"]
        elif "VIX" in vix.columns:
            vix_close = vix["VIX"]

    results_by_lookback = {}
    for lookback in lookbacks:
        window_close = _slice_frame_to_lookback(close, lookback)
        window_vix = _slice_frame_to_lookback(vix_close, lookback) if vix_close is not None else None
        results_by_lookback[lookback] = _run_strategy_profile_suite_from_close(window_close, window_vix, horizon_days)

    valid_results = [result for result in results_by_lookback.values() if "error" not in result]
    best_profile_votes: dict[str, int] = {}
    current_state_votes: dict[str, int] = {}
    profile_scores = []

    for result in valid_results:
        summary = result.get("summary", {})
        best_profile = (summary.get("best_profile_overall") or {}).get("profile")
        if best_profile:
            best_profile_votes[str(best_profile)] = best_profile_votes.get(str(best_profile), 0) + 1
            profile_scores.append(float((summary.get("best_profile_overall") or {}).get("score", 0.0)))
        current_state = summary.get("current_market_state")
        if current_state:
            current_state_votes[str(current_state)] = current_state_votes.get(str(current_state), 0) + 1

    consensus_profile = max(best_profile_votes.items(), key=lambda item: (item[1], item[0]))[0] if best_profile_votes else "all_weather"
    consensus_state = max(current_state_votes.items(), key=lambda item: (item[1], item[0]))[0] if current_state_votes else "transition"
    summary = {
        "by_lookback": {lookback: result.get("summary", {}) for lookback, result in results_by_lookback.items()},
        "consensus_profile": consensus_profile,
        "consensus_state": consensus_state,
        "profile_vote_share": best_profile_votes,
        "current_state_vote_share": current_state_votes,
        "avg_best_profile_score": safe_mean(profile_scores),
        "valid_windows": len(valid_results),
    }
    return {"results_by_lookback": results_by_lookback, "summary": summary}


def _load_vectorbt_module():
    try:
        import vectorbt as vbt

        return vbt
    except Exception:
        pass

    if VECTORBT_LOCAL_PATH.exists():
        local_path = str(VECTORBT_LOCAL_PATH)
        if local_path not in sys.path:
            sys.path.insert(0, local_path)
        try:
            import vectorbt as vbt

            return vbt
        except Exception:
            return None
    return None


def _download_intraday_close(symbols: list[str], period: str = "60d", interval: str = "15m") -> pd.DataFrame:
    canonical_symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if not canonical_symbols:
        return pd.DataFrame()

    resolved_to_canonical = {resolve_download_symbol(symbol): symbol for symbol in canonical_symbols}
    requested = list(resolved_to_canonical.keys())

    raw = yf.download(requested, period=period, interval=interval, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        else:
            close = raw.xs("Close", axis=1, level=0, drop_level=True)
    else:
        close = raw[["Close"]].rename(columns={"Close": requested[0]})

    close = close.rename(columns=resolved_to_canonical)
    return close.reindex(columns=[symbol for symbol in canonical_symbols if symbol in close.columns]).ffill().dropna(how="all")


def _target_delta_strike(flag: str, spot: float, volatility: float, years_to_expiry: float, risk_free_rate: float) -> float:
    if spot <= 0 or volatility <= 0 or years_to_expiry <= 0:
        return spot
    # Inverse normal constants for N^-1(0.80) and N^-1(0.20).
    d1 = 0.8416212336 if flag == "p" else -0.8416212336
    exponent = (d1 * volatility * math.sqrt(years_to_expiry)) - ((risk_free_rate + 0.5 * volatility * volatility) * years_to_expiry)
    return spot / math.exp(exponent)


def _build_vectorbt_summary(naive_returns: list[float], filtered_returns: list[float]) -> dict | None:
    vbt = _load_vectorbt_module()
    if vbt is None:
        return None

    def _portfolio_stats(returns: list[float]) -> dict:
        if not returns:
            return {"trades": 0}
        capped = pd.Series(np.clip(np.array(returns, dtype=float), -0.95, 3.0))
        equity_curve = 100.0 * (1.0 + capped).cumprod()
        portfolio = vbt.Portfolio.from_holding(equity_curve, init_cash=100.0, freq="15m")
        return {
            "trades": int(len(capped)),
            "total_return": float(portfolio.total_return()),
            "max_drawdown": float(portfolio.max_drawdown()),
            "sharpe_ratio": float(portfolio.sharpe_ratio()),
        }

    return {
        "naive": _portfolio_stats(naive_returns),
        "delay_filtered": _portfolio_stats(filtered_returns),
    }


def _run_delay_quote_suite_from_close(close: pd.DataFrame, horizon_days: int):
    if close.empty:
        return {"error": "no_intraday_data", "summary": {"symbols": 0}}

    horizon_bars = max(1, int(horizon_days * INTRADAY_BARS_PER_DAY))
    years_to_expiry = 45.0 / 365.0
    years_decay = horizon_bars / float(INTRADAY_BARS_PER_DAY * 252)
    risk_free_rate = float(OPTION_PRICING_RISK_FREE_RATE)
    annualization = math.sqrt(INTRADAY_BARS_PER_DAY * 252)

    per_symbol: dict[str, dict[str, object]] = {}
    aggregate = {
        "naive_pnl": {"p": [], "c": []},
        "naive_win": {"p": [], "c": []},
        "filtered_pnl": {"p": [], "c": []},
        "filtered_win": {"p": [], "c": []},
        "quote_error": {"p": [], "c": []},
        "delay_move": {"p": [], "c": []},
        "delta_breach": {"p": [], "c": []},
    }

    for symbol in close.columns:
        series = close[symbol].dropna()
        if len(series) < max(200, horizon_bars + 50):
            continue

        frame = pd.DataFrame({"close": series})
        frame["ret"] = np.log(frame["close"] / frame["close"].shift(1))
        frame["sigma"] = frame["ret"].rolling(INTRADAY_BARS_PER_DAY).std() * annualization
        frame = frame.dropna()
        if len(frame) < horizon_bars + 20:
            continue

        symbol_result = {}
        for flag in ("p", "c"):
            naive_pnl = []
            filtered_pnl = []
            naive_win = []
            filtered_win = []
            quote_error = []
            delay_move = []
            delta_breach = []

            for i in range(1, len(frame) - horizon_bars):
                delayed_spot = float(frame["close"].iloc[i - 1])
                entry_spot = float(frame["close"].iloc[i])
                exit_spot = float(frame["close"].iloc[i + horizon_bars])
                sigma_entry = float(frame["sigma"].iloc[i])
                sigma_exit = float(frame["sigma"].iloc[i + horizon_bars])
                if sigma_entry <= 0 or sigma_exit <= 0:
                    continue

                strike = _target_delta_strike(flag, delayed_spot, sigma_entry, years_to_expiry, risk_free_rate)
                delayed_price = option_price(flag, delayed_spot, strike, years_to_expiry, risk_free_rate, sigma_entry)
                entry_price = option_price(flag, entry_spot, strike, years_to_expiry, risk_free_rate, sigma_entry)
                exit_price = option_price(
                    flag,
                    exit_spot,
                    strike,
                    max(1.0 / 365.0, years_to_expiry - years_decay),
                    risk_free_rate,
                    sigma_exit,
                )

                if min(delayed_price, entry_price) <= 0:
                    continue

                live_delta = abs(option_greeks(flag, entry_spot, strike, years_to_expiry, risk_free_rate, sigma_entry)["delta"])
                live_in_band = DELTA_MIN <= live_delta <= DELTA_MAX
                move_pct = abs(entry_spot - delayed_spot) / max(delayed_spot, 1e-6)
                pnl = float(entry_price - exit_price)

                quote_error.append(abs(entry_price - delayed_price) / max(delayed_price, 0.05))
                delay_move.append(move_pct)
                delta_breach.append(0.0 if live_in_band else 1.0)
                naive_pnl.append(pnl)
                naive_win.append(1.0 if pnl > 0 else 0.0)

                if live_in_band and move_pct <= OPTION_DELAY_MAX_UNDERLYING_MOVE_PCT:
                    filtered_pnl.append(pnl)
                    filtered_win.append(1.0 if pnl > 0 else 0.0)

            if not naive_pnl:
                continue

            symbol_result["put" if flag == "p" else "call"] = {
                "samples": len(naive_pnl),
                "avg_quote_error_pct": safe_mean(quote_error),
                "avg_delay_move_pct": safe_mean(delay_move),
                "delta_band_breach_rate": safe_mean(delta_breach),
                "naive_avg_short_pnl": safe_mean(naive_pnl),
                "naive_win_rate": safe_mean(naive_win),
                "delay_filtered_samples": len(filtered_pnl),
                "delay_filtered_keep_rate": (len(filtered_pnl) / len(naive_pnl)) if naive_pnl else 0.0,
                "delay_filtered_avg_short_pnl": safe_mean(filtered_pnl),
                "delay_filtered_win_rate": safe_mean(filtered_win),
            }

            aggregate["naive_pnl"][flag].extend(naive_pnl)
            aggregate["naive_win"][flag].extend(naive_win)
            aggregate["filtered_pnl"][flag].extend(filtered_pnl)
            aggregate["filtered_win"][flag].extend(filtered_win)
            aggregate["quote_error"][flag].extend(quote_error)
            aggregate["delay_move"][flag].extend(delay_move)
            aggregate["delta_breach"][flag].extend(delta_breach)

        if symbol_result:
            per_symbol[str(symbol)] = symbol_result

    if not per_symbol:
        return {"error": "insufficient_intraday_samples", "summary": {"symbols": 0}}

    naive_returns = []
    filtered_returns = []
    for flag in ("p", "c"):
        naive_returns.extend(
            pnl / max(0.25, TARGET_OPTION_ABS_DELTA)
            for pnl in aggregate["naive_pnl"][flag]
        )
        filtered_returns.extend(
            pnl / max(0.25, TARGET_OPTION_ABS_DELTA)
            for pnl in aggregate["filtered_pnl"][flag]
        )

    summary = {
        "symbols": len(per_symbol),
        "horizon_days": int(horizon_days),
        "vectorbt_available": _load_vectorbt_module() is not None,
        "puts": {
            "samples": len(aggregate["naive_pnl"]["p"]),
            "avg_quote_error_pct": safe_mean(aggregate["quote_error"]["p"]),
            "avg_delay_move_pct": safe_mean(aggregate["delay_move"]["p"]),
            "delta_band_breach_rate": safe_mean(aggregate["delta_breach"]["p"]),
            "naive_avg_short_pnl": safe_mean(aggregate["naive_pnl"]["p"]),
            "naive_win_rate": safe_mean(aggregate["naive_win"]["p"]),
            "delay_filtered_samples": len(aggregate["filtered_pnl"]["p"]),
            "delay_filtered_avg_short_pnl": safe_mean(aggregate["filtered_pnl"]["p"]),
            "delay_filtered_win_rate": safe_mean(aggregate["filtered_win"]["p"]),
        },
        "calls": {
            "samples": len(aggregate["naive_pnl"]["c"]),
            "avg_quote_error_pct": safe_mean(aggregate["quote_error"]["c"]),
            "avg_delay_move_pct": safe_mean(aggregate["delay_move"]["c"]),
            "delta_band_breach_rate": safe_mean(aggregate["delta_breach"]["c"]),
            "naive_avg_short_pnl": safe_mean(aggregate["naive_pnl"]["c"]),
            "naive_win_rate": safe_mean(aggregate["naive_win"]["c"]),
            "delay_filtered_samples": len(aggregate["filtered_pnl"]["c"]),
            "delay_filtered_avg_short_pnl": safe_mean(aggregate["filtered_pnl"]["c"]),
            "delay_filtered_win_rate": safe_mean(aggregate["filtered_win"]["c"]),
        },
        "vectorbt_proxy": _build_vectorbt_summary(naive_returns, filtered_returns),
    }
    return {"per_symbol": per_symbol, "summary": summary}


def run_delay_quote_suite(symbols: list[str], horizon_days: int):
    intraday_symbols = list(dict.fromkeys(symbols))[:12]
    close = _download_intraday_close(intraday_symbols, period="60d", interval="15m")
    return _run_delay_quote_suite_from_close(close, horizon_days=horizon_days)


def _run_option_model_suite_from_close(close: pd.DataFrame, horizon_days: int):
    if close.empty:
        return {"error": "no_intraday_data", "summary": {"symbols": 0}}

    horizon_bars = max(1, int(horizon_days * INTRADAY_BARS_PER_DAY))
    years_to_expiry = 45.0 / 365.0
    years_decay = horizon_bars / float(INTRADAY_BARS_PER_DAY * 252)
    risk_free_rate = float(OPTION_PRICING_RISK_FREE_RATE)
    annualization = math.sqrt(INTRADAY_BARS_PER_DAY * 252)
    edge_threshold_pct = 0.03
    sample_stride = 2

    model_names = ("black_scholes", "binomial", "monte_carlo", "ensemble")
    per_symbol: dict[str, dict[str, object]] = {}
    aggregate = {
        model_name: {
            "signals": [],
            "pnl": [],
            "wins": [],
            "edge_pct": [],
        }
        for model_name in model_names
    }

    for symbol in close.columns:
        series = close[symbol].dropna()
        if len(series) < max(220, horizon_bars + 60):
            continue

        frame = pd.DataFrame({"close": series})
        frame["ret"] = np.log(frame["close"] / frame["close"].shift(1))
        frame["sigma"] = frame["ret"].rolling(INTRADAY_BARS_PER_DAY).std() * annualization
        frame = frame.dropna()
        if len(frame) < horizon_bars + 30:
            continue

        symbol_result = {}
        for flag in ("p", "c"):
            model_stats = {
                model_name: {
                    "signals": 0,
                    "pnl": [],
                    "wins": [],
                    "edge_pct": [],
                }
                for model_name in model_names
            }

            for i in range(1, len(frame) - horizon_bars, sample_stride):
                delayed_spot = float(frame["close"].iloc[i - 1])
                entry_spot = float(frame["close"].iloc[i])
                exit_spot = float(frame["close"].iloc[i + horizon_bars])
                sigma_entry = float(frame["sigma"].iloc[i])
                sigma_exit = float(frame["sigma"].iloc[i + horizon_bars])
                if sigma_entry <= 0 or sigma_exit <= 0:
                    continue

                strike = _target_delta_strike(flag, delayed_spot, sigma_entry, years_to_expiry, risk_free_rate)
                market_price = option_price(flag, delayed_spot, strike, years_to_expiry, risk_free_rate, sigma_entry)
                if market_price <= 0:
                    continue

                remaining_years = max(1.0 / 365.0, years_to_expiry - years_decay)
                exit_price = option_price(flag, exit_spot, strike, remaining_years, risk_free_rate, sigma_exit)
                fair_values = {
                    "black_scholes": black_scholes_price(flag, entry_spot, strike, years_to_expiry, risk_free_rate, sigma_entry),
                    "binomial": binomial_option_price(
                        flag,
                        entry_spot,
                        strike,
                        years_to_expiry,
                        risk_free_rate,
                        sigma_entry,
                        steps=60,
                    ),
                    "monte_carlo": monte_carlo_option_price(
                        flag,
                        entry_spot,
                        strike,
                        years_to_expiry,
                        risk_free_rate,
                        sigma_entry,
                        n_simulations=160,
                        seed=1000 + i,
                    ),
                }
                fair_values["ensemble"] = safe_mean(list(fair_values.values()))

                for model_name, fair_value in fair_values.items():
                    edge_pct = (fair_value - market_price) / max(market_price, 0.05)
                    if edge_pct < edge_threshold_pct:
                        continue
                    pnl = float(exit_price - market_price)
                    model_stats[model_name]["signals"] += 1
                    model_stats[model_name]["pnl"].append(pnl)
                    model_stats[model_name]["wins"].append(1.0 if pnl > 0 else 0.0)
                    model_stats[model_name]["edge_pct"].append(edge_pct)

            option_results = {
                model_name: {
                    "signals": int(stats["signals"]),
                    "avg_edge_pct": safe_mean(stats["edge_pct"]),
                    "avg_long_pnl": safe_mean(stats["pnl"]),
                    "long_win_rate": safe_mean(stats["wins"]),
                }
                for model_name, stats in model_stats.items()
                if stats["signals"] > 0
            }
            if option_results:
                symbol_result["put" if flag == "p" else "call"] = option_results
                for model_name in option_results:
                    aggregate[model_name]["signals"].append(float(option_results[model_name]["signals"]))
                    aggregate[model_name]["pnl"].extend(model_stats[model_name]["pnl"])
                    aggregate[model_name]["wins"].extend(model_stats[model_name]["wins"])
                    aggregate[model_name]["edge_pct"].extend(model_stats[model_name]["edge_pct"])

        if symbol_result:
            per_symbol[str(symbol)] = symbol_result

    if not per_symbol:
        return {"error": "insufficient_intraday_samples", "summary": {"symbols": 0}}

    summary = {
        "symbols": len(per_symbol),
        "edge_threshold_pct": edge_threshold_pct,
        "sample_stride_bars": sample_stride,
        "models": {
            model_name: {
                "avg_signals_per_symbol": safe_mean(aggregate[model_name]["signals"]),
                "avg_edge_pct": safe_mean(aggregate[model_name]["edge_pct"]),
                "avg_long_pnl": safe_mean(aggregate[model_name]["pnl"]),
                "long_win_rate": safe_mean(aggregate[model_name]["wins"]),
            }
            for model_name in model_names
        },
    }
    return {"per_symbol": per_symbol, "summary": summary}


def run_option_model_suite(symbols: list[str], horizon_days: int):
    intraday_symbols = list(dict.fromkeys(symbols))[:10]
    close = _download_intraday_close(intraday_symbols, period="60d", interval="15m")
    return _run_option_model_suite_from_close(close, horizon_days=horizon_days)


def run_ml_alpha_suite(symbols: list[str]):
    universe = list(dict.fromkeys(symbols))[:24]
    return backtest_alpha_strategy(universe)


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
    pairs = run_pairs_suite(symbols=symbols, lookbacks=lookbacks, entry_z=args.pairs_entry_z, exit_z=args.pairs_exit_z, horizon_days=args.horizon_days)
    regime = run_regime_suite(horizon_days=args.horizon_days, lookbacks=lookbacks)
    strategy_proxy = run_strategy_proxy_suite(symbols=symbols, lookbacks=lookbacks, horizon_days=args.horizon_days)
    strategy_profiles = run_strategy_profile_suite(symbols=symbols, lookbacks=lookbacks, horizon_days=args.horizon_days)
    delay_quote = run_delay_quote_suite(symbols=symbols, horizon_days=args.horizon_days)
    option_model = run_option_model_suite(symbols=symbols, horizon_days=args.horizon_days)
    ml_alpha = run_ml_alpha_suite(symbols=symbols)

    report["movement_suite"] = movement
    report["pairs_suite"] = pairs
    report["regime_suite"] = regime
    report["strategy_proxy_suite"] = strategy_proxy
    report["strategy_profile_suite"] = strategy_profiles
    report["delay_quote_suite"] = delay_quote
    report["option_model_suite"] = option_model
    report["ml_alpha_suite"] = ml_alpha

    aggregate_score_components = []
    aggregate_score_components.append(movement["summary"].get("avg_accuracy", 0.0))
    aggregate_score_components.append(max(0.0, min(1.0, pairs.get("summary", {}).get("win_rate", 0.0))))
    aggregate_score_components.append(max(0.0, min(1.0, regime.get("summary", {}).get("directional_accuracy_proxy", 0.0))))
    aggregate_score_components.append(max(0.0, min(1.0, strategy_proxy.get("summary", {}).get("overall_hit_rate", 0.0))))
    aggregate_score_components.append(max(0.0, min(1.0, strategy_profiles.get("summary", {}).get("avg_best_profile_score", 0.0))))
    aggregate_score_components.append(
        max(
            0.0,
            min(
                1.0,
                (
                    0.50
                    + (
                        float(delay_quote.get("summary", {}).get("puts", {}).get("delay_filtered_win_rate", 0.0))
                        + float(delay_quote.get("summary", {}).get("calls", {}).get("delay_filtered_win_rate", 0.0))
                    )
                    / 4.0
                ),
            ),
        )
    )
    aggregate_score_components.append(
        max(
            0.0,
            min(
                1.0,
                0.50 + float(option_model.get("summary", {}).get("models", {}).get("ensemble", {}).get("long_win_rate", 0.0)) / 2.0,
            ),
        )
    )
    aggregate_score_components.append(
        max(
            0.0,
            min(
                1.0,
                0.50 + float(ml_alpha.get("summary", {}).get("avg_information_coefficient", 0.0)),
            ),
        )
    )

    report["massive_overview"] = {
        "predictive_score": safe_mean(aggregate_score_components),
        "movement_hit_rate": movement["summary"].get("hit_rate", 0.0),
        "pairs_win_rate": pairs.get("summary", {}).get("win_rate", 0.0),
        "regime_directional_accuracy_proxy": regime.get("summary", {}).get("directional_accuracy_proxy", 0.0),
        "strategy_router_hit_rate": strategy_proxy.get("summary", {}).get("overall_hit_rate", 0.0),
        "strategy_profile_score": strategy_profiles.get("summary", {}).get("avg_best_profile_score", 0.0),
        "delay_filtered_put_win_rate": delay_quote.get("summary", {}).get("puts", {}).get("delay_filtered_win_rate", 0.0),
        "delay_filtered_call_win_rate": delay_quote.get("summary", {}).get("calls", {}).get("delay_filtered_win_rate", 0.0),
        "option_model_ensemble_win_rate": option_model.get("summary", {}).get("models", {}).get("ensemble", {}).get("long_win_rate", 0.0),
        "option_model_ensemble_edge_pct": option_model.get("summary", {}).get("models", {}).get("ensemble", {}).get("avg_edge_pct", 0.0),
        "avg_recent_quote_error_pct": safe_mean(
            [
                delay_quote.get("summary", {}).get("puts", {}).get("avg_quote_error_pct", 0.0),
                delay_quote.get("summary", {}).get("calls", {}).get("avg_quote_error_pct", 0.0),
            ]
        ),
        "ml_alpha_information_coefficient": ml_alpha.get("summary", {}).get("avg_information_coefficient", 0.0),
        "ml_alpha_long_only_sharpe": ml_alpha.get("summary", {}).get("long_only", {}).get("sharpe_ratio", 0.0),
        "ml_alpha_long_short_sharpe": ml_alpha.get("summary", {}).get("long_short", {}).get("sharpe_ratio", 0.0),
        "consensus_market_state": strategy_profiles.get("summary", {}).get("consensus_state"),
        "consensus_strategy_profile": strategy_profiles.get("summary", {}).get("consensus_profile"),
        "window_movement_accuracy": movement["summary"].get("by_lookback", {}),
        "window_pairs_win_rate": _window_summary(pairs.get("results_by_lookback", {}), "win_rate"),
        "window_regime_accuracy": _window_summary(regime.get("results_by_lookback", {}), "directional_accuracy_proxy"),
        "window_strategy_hit_rate": _window_summary(strategy_proxy.get("results_by_lookback", {}), "overall_hit_rate"),
        "window_strategy_profile_state": {
            lookback: result.get("summary", {}).get("current_market_state")
            for lookback, result in strategy_profiles.get("results_by_lookback", {}).items()
        },
        "note": "Framework emphasizes walk-forward predictive quality, state-conditioned strategy routing, and delay-aware execution realism. It is still a deployment filter rather than an exact broker-fill simulator.",
    }
    report["institutional_robustness"] = _build_institutional_robustness(report)
    report["massive_overview"]["institutional_score"] = report["institutional_robustness"]["institutional_score"]
    report["massive_overview"]["deployment_tier"] = report["institutional_robustness"]["deployment_tier"]
    report["massive_overview"]["methodology_score"] = report["institutional_robustness"]["methodology_score"]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    archive_paths = archive_backtest_artifacts(report, output_path=out, reports_root=ROOT / "reports")

    print("\n=== MASSIVE PREDICTIVE OVERVIEW ===")
    print(json.dumps(report["massive_overview"], indent=2))
    print(f"Saved report: {out}")
    print(f"Saved latest backtest summary: {archive_paths['latest_md']}")


if __name__ == "__main__":
    main()
