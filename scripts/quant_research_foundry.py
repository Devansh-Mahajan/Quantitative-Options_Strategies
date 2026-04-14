import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.state_manager import register_model_snapshot
from core.resource_profile import load_resource_profile
from core.universe_maintenance import download_close_matrix, load_symbol_file

DEFAULT_REPORT_PATH = ROOT / "reports" / "quant_foundry_report.json"
DEFAULT_PACK_PATH = ROOT / "config" / "quant_strategy_pack.json"
DEFAULT_SYMBOLS_PATH = ROOT / "config" / "symbol_list.txt"
ADAPTIVE_PROFILE_PATH = ROOT / "config" / "adaptive_profile.json"
DEFAULT_MODEL_CONFIG_PATH = ROOT / "config" / "foundry_last_model_config.json"


DEFAULT_MODEL_PARAMS = {
    "logreg": {"max_iter": 400, "solver": "lbfgs", "random_state": 42, "C": 1.0},
    "rf": {"n_estimators": 220, "max_depth": 8, "min_samples_leaf": 10, "random_state": 42, "n_jobs": -1},
    "mlp": {
        "hidden_layer_sizes": [64, 32],
        "learning_rate_init": 8e-4,
        "max_iter": 260,
        "random_state": 42,
        "alpha": 8e-4,
    },
}

FOUNDRY_SELECTION_MAX_ROWS = 12000
FOUNDRY_SELECTION_FOLDS = 3
FOUNDRY_ROUNDTRIP_COST_BPS = 8.0

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args():
    resource_profile = load_resource_profile(ROOT)
    parser = argparse.ArgumentParser(
        description=(
            "Build a multi-model strategy pack with walk-forward validation. "
            "Mode 'weekend-calibrate' fits ensemble weights; mode 'zero-calibration' emits priors only."
        )
    )
    parser.add_argument("--symbols", nargs="+", help="Universe symbols (defaults to config/symbol_list.txt).")
    parser.add_argument("--period", default="10y", help="Download period for training data.")
    parser.add_argument("--horizon-days", type=int, default=5, help="Prediction horizon in trading days.")
    parser.add_argument("--target-daily-return", type=float, default=0.002)
    parser.add_argument("--target-accuracy", type=float, default=0.56)
    parser.add_argument(
        "--mode",
        choices=["weekend-calibrate", "zero-calibration"],
        default="weekend-calibrate",
    )
    parser.add_argument("--max-symbols", type=int, default=35)
    parser.add_argument("--rf-jobs", type=int, default=resource_profile.research_rf_jobs)
    parser.add_argument("--model-parallelism", type=int, default=resource_profile.model_parallelism)
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--pack-path", default=str(DEFAULT_PACK_PATH))
    parser.add_argument("--model-config-path", default=str(DEFAULT_MODEL_CONFIG_PATH))
    parser.add_argument(
        "--disable-model-warm-start",
        action="store_true",
        help="Ignore previously saved model params and train from defaults.",
    )
    return parser.parse_args()


def load_symbols(cli_symbols: list[str] | None, max_symbols: int) -> list[str]:
    if cli_symbols:
        return list(dict.fromkeys([s.strip().upper() for s in cli_symbols if s.strip()]))[: max(1, max_symbols)]

    if not DEFAULT_SYMBOLS_PATH.exists():
        return ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT"]

    return load_symbol_file(DEFAULT_SYMBOLS_PATH)[: max(1, max_symbols)]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def safe_mean(values) -> float:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not clean:
        return 0.0
    return float(np.mean(clean))


def _annualized_proxy_from_horizon(returns: np.ndarray, horizon_days: int) -> float:
    clean = np.asarray(returns, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return 0.0
    mean_horizon = float(np.mean(clean))
    if mean_horizon <= -0.999:
        return -1.0
    return float((1.0 + mean_horizon) ** (252 / max(1, int(horizon_days))) - 1.0)


def _sharpe_proxy(returns: np.ndarray, horizon_days: int) -> float:
    clean = np.asarray(returns, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return 0.0
    std = float(np.std(clean, ddof=0))
    if std <= 1e-12:
        return 0.0
    mean = float(np.mean(clean))
    return float((mean / std) * math.sqrt(252 / max(1, int(horizon_days))))


def _reshape_hidden_layers(layers, scale: float) -> list[int]:
    seq = list(layers if isinstance(layers, (list, tuple)) else [layers])
    return [max(16, int(round(float(value) * scale))) for value in seq]


def _build_feature_panel(close: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    returns_1 = close.pct_change(1)
    returns_5 = close.pct_change(5)
    returns_20 = close.pct_change(20)
    vol_20 = returns_1.rolling(20).std()
    z_20 = (close - close.rolling(20).mean()) / close.rolling(20).std(ddof=0)

    market = close.mean(axis=1)
    market_ret = market.pct_change(1)
    market_vol = market_ret.rolling(20).std()

    frames = []
    for sym in close.columns:
        feature_df = pd.DataFrame(
            {
                "ret_1": returns_1[sym],
                "ret_5": returns_5[sym],
                "ret_20": returns_20[sym],
                "vol_20": vol_20[sym],
                "z_20": z_20[sym],
                "market_ret_1": market_ret,
                "market_vol_20": market_vol,
                "cross_rank_ret_5": returns_5.rank(axis=1, pct=True)[sym],
                "cross_rank_z_20": z_20.rank(axis=1, pct=True)[sym],
            }
        )
        forward_return = close[sym].shift(-horizon_days) / close[sym] - 1.0
        target = (forward_return > 0).astype(int)
        sym_frame = feature_df.assign(symbol=sym, target=target, forward_return=forward_return)
        sym_frame.index.name = "date"
        sym_frame = sym_frame.reset_index().set_index(["date", "symbol"])
        frames.append(sym_frame)

    joined = pd.concat(frames, axis=0)
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    return joined


def build_feature_matrix(close: pd.DataFrame, horizon_days: int) -> tuple[pd.DataFrame, pd.Series]:
    joined = _build_feature_panel(close, horizon_days)
    return joined.drop(columns=["target", "forward_return"]), joined["target"]


def annual_to_daily(annual_return: float) -> float:
    if annual_return <= -0.999:
        return -1.0
    return float((1 + annual_return) ** (1 / 252) - 1)


def _sanitize_model_params(raw_params: dict | None) -> dict:
    merged = json.loads(json.dumps(DEFAULT_MODEL_PARAMS))
    if not isinstance(raw_params, dict):
        return merged
    for model_name in ("logreg", "rf", "mlp"):
        raw_model_params = raw_params.get(model_name)
        if not isinstance(raw_model_params, dict):
            continue
        merged[model_name].update(raw_model_params)
    return merged


def load_last_model_params(path: Path) -> dict:
    if not path.exists():
        return _sanitize_model_params(None)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _sanitize_model_params(None)
    return _sanitize_model_params(payload.get("model_params"))


def _prepare_training_frame(X: pd.DataFrame) -> pd.DataFrame:
    if "symbol" in X.columns:
        return X.copy()
    if isinstance(X.index, pd.MultiIndex) and "symbol" in X.index.names:
        return X.reset_index(level="symbol")
    return X.assign(symbol="UNIVERSE")


def _serialize_candidate_id(model_params: dict) -> str:
    return json.dumps(_sanitize_model_params(model_params), sort_keys=True)


def _walk_forward_slices(n_samples: int, requested_folds: int = FOUNDRY_SELECTION_FOLDS) -> list[tuple[int, int]]:
    if n_samples <= 0:
        return []

    test_size = max(40, int(n_samples * 0.12))
    min_train = max(180, int(n_samples * 0.52))
    if n_samples - min_train < max(20, test_size // 2):
        train_end = max(60, int(n_samples * 0.8))
        if train_end >= n_samples:
            train_end = max(1, n_samples - 1)
        return [(train_end, n_samples)]

    remaining = n_samples - min_train
    possible_folds = max(1, min(int(requested_folds or 1), max(1, remaining // test_size)))
    step = max(test_size, remaining // possible_folds)

    slices: list[tuple[int, int]] = []
    for fold_idx in range(possible_folds):
        train_end = min_train + (fold_idx * step)
        test_end = min(n_samples, train_end + test_size)
        if test_end - train_end < 20 or train_end < 60:
            continue
        slices.append((train_end, test_end))

    if not slices:
        train_end = max(60, n_samples - max(20, test_size))
        slices.append((train_end, n_samples))
    return slices


def _model_constructor_map() -> dict[str, callable]:
    return {
        "logreg": lambda cfg: LogisticRegression(**cfg),
        "rf": lambda cfg: RandomForestClassifier(**cfg),
        "mlp": lambda cfg: MLPClassifier(**cfg),
    }


def _evaluate_holdout_model(
    name: str,
    constructor,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    forward_returns: pd.Series,
    horizon_days: int,
) -> tuple[str, dict]:
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    forward_test = np.asarray(forward_returns.iloc[split_idx:], dtype=float)
    roundtrip_cost = FOUNDRY_ROUNDTRIP_COST_BPS / 10000.0

    try:
        model = constructor(params)
        model.fit(X_train, y_train)
        train_prob = model.predict_proba(X_train)[:, 1]
        train_pred = (train_prob >= 0.5).astype(int)
        test_prob = model.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)

        train_accuracy = float(accuracy_score(y_train, train_pred))
        test_accuracy = float(accuracy_score(y_test, test_pred))
        train_loss = float(log_loss(y_train, train_prob, labels=[0, 1]))
        test_loss = float(log_loss(y_test, test_prob, labels=[0, 1]))

        gross_returns = np.where(test_pred == 1, forward_test, -forward_test)
        net_returns = gross_returns - roundtrip_cost
        daily_return_proxy = float(np.nanmean(net_returns) / max(1, int(horizon_days)))
        cumulative_return_proxy = float(np.exp(np.nansum(net_returns)) - 1.0)
        sharpe_proxy = _sharpe_proxy(net_returns, horizon_days)
        annualized_return_proxy = _annualized_proxy_from_horizon(net_returns, horizon_days)
        win_rate = float(np.mean(net_returns > 0)) if len(net_returns) else 0.0

        return name, {
            "accuracy": test_accuracy,
            "log_loss": test_loss,
            "n_test": int(len(y_test)),
            "train_accuracy": train_accuracy,
            "train_log_loss": train_loss,
            "generalization_gap": max(0.0, train_accuracy - test_accuracy),
            "daily_return_proxy": daily_return_proxy,
            "annualized_return_proxy": annualized_return_proxy,
            "cumulative_return_proxy": cumulative_return_proxy,
            "sharpe_proxy": sharpe_proxy,
            "win_rate": win_rate,
            "evaluation_mode": "holdout",
        }
    except Exception as exc:
        return name, {
            "error": str(exc),
            "accuracy": 0.0,
            "log_loss": 10.0,
            "n_test": int(len(y_test)),
            "train_accuracy": 0.0,
            "train_log_loss": 10.0,
            "generalization_gap": 1.0,
            "daily_return_proxy": -0.01,
            "annualized_return_proxy": -1.0,
            "cumulative_return_proxy": -1.0,
            "sharpe_proxy": -3.0,
            "win_rate": 0.0,
            "evaluation_mode": "holdout",
        }


def _evaluate_walk_forward_model(
    name: str,
    constructor,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    forward_returns: pd.Series,
    horizon_days: int,
) -> tuple[str, dict]:
    fold_rows = []
    roundtrip_cost = FOUNDRY_ROUNDTRIP_COST_BPS / 10000.0

    try:
        for fold_number, (train_end, test_end) in enumerate(_walk_forward_slices(len(X)), start=1):
            X_train, X_test = X.iloc[:train_end], X.iloc[train_end:test_end]
            y_train, y_test = y.iloc[:train_end], y.iloc[train_end:test_end]
            forward_test = np.asarray(forward_returns.iloc[train_end:test_end], dtype=float)

            model = constructor(params)
            model.fit(X_train, y_train)
            train_prob = model.predict_proba(X_train)[:, 1]
            train_pred = (train_prob >= 0.5).astype(int)
            test_prob = model.predict_proba(X_test)[:, 1]
            test_pred = (test_prob >= 0.5).astype(int)

            train_accuracy = float(accuracy_score(y_train, train_pred))
            test_accuracy = float(accuracy_score(y_test, test_pred))
            test_loss = float(log_loss(y_test, test_prob, labels=[0, 1]))
            gross_returns = np.where(test_pred == 1, forward_test, -forward_test)
            net_returns = gross_returns - roundtrip_cost

            fold_rows.append(
                {
                    "fold": fold_number,
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "train_accuracy": train_accuracy,
                    "accuracy": test_accuracy,
                    "log_loss": test_loss,
                    "generalization_gap": max(0.0, train_accuracy - test_accuracy),
                    "daily_return_proxy": float(np.nanmean(net_returns) / max(1, int(horizon_days))),
                    "annualized_return_proxy": _annualized_proxy_from_horizon(net_returns, horizon_days),
                    "sharpe_proxy": _sharpe_proxy(net_returns, horizon_days),
                    "win_rate": float(np.mean(net_returns > 0)) if len(net_returns) else 0.0,
                }
            )

        if not fold_rows:
            raise ValueError("No walk-forward folds available.")

        accuracy_values = [row["accuracy"] for row in fold_rows]
        log_losses = [row["log_loss"] for row in fold_rows]
        generalization_gaps = [row["generalization_gap"] for row in fold_rows]
        daily_returns = [row["daily_return_proxy"] for row in fold_rows]
        annualized_returns = [row["annualized_return_proxy"] for row in fold_rows]
        sharpe_values = [row["sharpe_proxy"] for row in fold_rows]
        win_rates = [row["win_rate"] for row in fold_rows]

        return name, {
            "accuracy": safe_mean(accuracy_values),
            "log_loss": safe_mean(log_losses),
            "n_test": int(sum(row["n_test"] for row in fold_rows)),
            "generalization_gap": safe_mean(generalization_gaps),
            "daily_return_proxy": safe_mean(daily_returns),
            "annualized_return_proxy": safe_mean(annualized_returns),
            "sharpe_proxy": safe_mean(sharpe_values),
            "win_rate": safe_mean(win_rates),
            "accuracy_std": float(np.std(accuracy_values, ddof=0)) if len(accuracy_values) > 1 else 0.0,
            "log_loss_std": float(np.std(log_losses, ddof=0)) if len(log_losses) > 1 else 0.0,
            "folds": fold_rows,
            "evaluation_mode": "walk_forward",
        }
    except Exception as exc:
        return name, {
            "error": str(exc),
            "accuracy": 0.0,
            "log_loss": 10.0,
            "n_test": 0,
            "generalization_gap": 1.0,
            "daily_return_proxy": -0.01,
            "annualized_return_proxy": -1.0,
            "sharpe_proxy": -3.0,
            "win_rate": 0.0,
            "accuracy_std": 1.0,
            "log_loss_std": 10.0,
            "folds": [],
            "evaluation_mode": "walk_forward",
        }


def _summarize_ensemble(model_rows: dict[str, dict]) -> dict:
    losses = {k: v.get("log_loss", 10.0) for k, v in model_rows.items() if "error" not in v}
    if losses:
        inv = {k: 1.0 / max(1e-5, v) for k, v in losses.items()}
        total = sum(inv.values())
        weights = {k: float(v / total) for k, v in inv.items()}
    else:
        weights = {"logreg": 0.4, "rf": 0.4, "mlp": 0.2}

    accuracy = float(sum(model_rows[k].get("accuracy", 0.0) * weights.get(k, 0.0) for k in model_rows.keys()))
    log_loss_value = float(sum(model_rows[k].get("log_loss", 10.0) * weights.get(k, 0.0) for k in model_rows.keys()))
    daily_return_proxy = float(
        sum(model_rows[k].get("daily_return_proxy", 0.0) * weights.get(k, 0.0) for k in model_rows.keys())
    )
    annualized_return_proxy = float(
        sum(model_rows[k].get("annualized_return_proxy", 0.0) * weights.get(k, 0.0) for k in model_rows.keys())
    )
    sharpe_proxy = float(sum(model_rows[k].get("sharpe_proxy", 0.0) * weights.get(k, 0.0) for k in model_rows.keys()))
    generalization_gap = float(
        sum(model_rows[k].get("generalization_gap", 0.0) * weights.get(k, 0.0) for k in model_rows.keys())
    )
    win_rate = float(sum(model_rows[k].get("win_rate", 0.0) * weights.get(k, 0.0) for k in model_rows.keys()))
    stability_penalty = float(
        sum(model_rows[k].get("accuracy_std", 0.0) * weights.get(k, 0.0) for k in model_rows.keys())
    )
    robustness_score = clamp(1.0 - (stability_penalty * 4.0) - generalization_gap)

    return {
        "ensemble_weights": weights,
        "ensemble_accuracy": accuracy,
        "ensemble_log_loss": log_loss_value,
        "ensemble_daily_return_proxy": daily_return_proxy,
        "ensemble_annualized_return_proxy": annualized_return_proxy,
        "ensemble_sharpe_proxy": sharpe_proxy,
        "ensemble_win_rate": win_rate,
        "generalization_gap": generalization_gap,
        "robustness_score": robustness_score,
        "stability_penalty": stability_penalty,
    }


def _score_candidate(summary: dict) -> float:
    loss_component = clamp(1.0 - (summary["ensemble_log_loss"] / 1.2))
    return_component = 0.5 + (math.tanh(summary["ensemble_daily_return_proxy"] * 400.0) / 2.0)
    sharpe_component = 0.5 + (math.tanh(summary["ensemble_sharpe_proxy"] / 3.0) / 2.0)
    score = (
        (0.34 * summary["ensemble_accuracy"])
        + (0.16 * loss_component)
        + (0.18 * return_component)
        + (0.10 * sharpe_component)
        + (0.12 * summary["ensemble_win_rate"])
        + (0.18 * summary["robustness_score"])
        - (0.08 * clamp(summary["generalization_gap"] * 2.0))
    )
    return round(clamp(score), 6)


def _build_candidate_configs(base_params: dict) -> list[dict]:
    base = _sanitize_model_params(base_params)
    candidates: list[dict] = []
    seen: set[str] = set()

    def _register(label: str, params: dict) -> None:
        sanitized = _sanitize_model_params(params)
        key = _serialize_candidate_id(sanitized)
        if key in seen:
            return
        seen.add(key)
        candidates.append({"label": label, "params": sanitized})

    _register("warm_start", base)
    defaults = _sanitize_model_params(DEFAULT_MODEL_PARAMS)
    if _serialize_candidate_id(defaults) != _serialize_candidate_id(base):
        _register("defaults", defaults)

    stability = deepcopy(base)
    stability["logreg"]["C"] = max(0.2, float(stability["logreg"].get("C", 1.0)) * 0.7)
    stability["rf"]["n_estimators"] = max(120, int(round(stability["rf"].get("n_estimators", 220) * 0.85)))
    stability["rf"]["max_depth"] = max(4, int(round(stability["rf"].get("max_depth", 8) - 2)))
    stability["rf"]["min_samples_leaf"] = max(4, int(round(stability["rf"].get("min_samples_leaf", 10) * 1.6)))
    stability["mlp"]["hidden_layer_sizes"] = _reshape_hidden_layers(stability["mlp"].get("hidden_layer_sizes", [64, 32]), 0.85)
    stability["mlp"]["learning_rate_init"] = float(stability["mlp"].get("learning_rate_init", 8e-4)) * 0.85
    stability["mlp"]["alpha"] = float(stability["mlp"].get("alpha", 8e-4)) * 1.35
    _register("stability_bias", stability)

    balanced = deepcopy(base)
    balanced["logreg"]["C"] = max(0.2, float(balanced["logreg"].get("C", 1.0)) * 1.15)
    balanced["rf"]["n_estimators"] = max(140, int(round(balanced["rf"].get("n_estimators", 220) * 1.12)))
    balanced["rf"]["max_depth"] = max(5, int(round(balanced["rf"].get("max_depth", 8) + 1)))
    balanced["rf"]["min_samples_leaf"] = max(2, int(round(balanced["rf"].get("min_samples_leaf", 10) * 0.9)))
    balanced["mlp"]["hidden_layer_sizes"] = _reshape_hidden_layers(balanced["mlp"].get("hidden_layer_sizes", [64, 32]), 1.15)
    balanced["mlp"]["learning_rate_init"] = float(balanced["mlp"].get("learning_rate_init", 8e-4)) * 1.05
    balanced["mlp"]["alpha"] = float(balanced["mlp"].get("alpha", 8e-4))
    _register("balanced_bias", balanced)

    aggressive = deepcopy(base)
    aggressive["logreg"]["C"] = max(0.25, float(aggressive["logreg"].get("C", 1.0)) * 1.55)
    aggressive["rf"]["n_estimators"] = max(160, int(round(aggressive["rf"].get("n_estimators", 220) * 1.3)))
    aggressive["rf"]["max_depth"] = max(6, int(round(aggressive["rf"].get("max_depth", 8) + 3)))
    aggressive["rf"]["min_samples_leaf"] = max(2, int(round(aggressive["rf"].get("min_samples_leaf", 10) * 0.65)))
    aggressive["mlp"]["hidden_layer_sizes"] = _reshape_hidden_layers(aggressive["mlp"].get("hidden_layer_sizes", [64, 32]), 1.35)
    aggressive["mlp"]["learning_rate_init"] = float(aggressive["mlp"].get("learning_rate_init", 8e-4)) * 1.2
    aggressive["mlp"]["alpha"] = float(aggressive["mlp"].get("alpha", 8e-4)) * 0.7
    _register("aggressive_bias", aggressive)
    return candidates[:4]


def _selection_stage_params(model_params: dict) -> dict:
    staged = _sanitize_model_params(model_params)
    staged["logreg"]["max_iter"] = min(int(staged["logreg"].get("max_iter", 400)), 220)
    staged["rf"]["n_estimators"] = max(80, int(round(staged["rf"].get("n_estimators", 220) * 0.6)))
    staged["mlp"]["max_iter"] = min(int(staged["mlp"].get("max_iter", 260)), 140)
    return staged


def _evaluate_bundle(
    X: pd.DataFrame,
    y: pd.Series,
    forward_returns: pd.Series,
    model_params: dict,
    *,
    model_parallelism: int,
    evaluation_mode: str,
    horizon_days: int,
) -> dict:
    constructors = _model_constructor_map()
    max_workers = max(1, min(3, int(model_parallelism or 1)))
    evaluate_one = _evaluate_walk_forward_model if evaluation_mode == "walk_forward" else _evaluate_holdout_model
    model_rows: dict[str, dict] = {}
    effective_params = _selection_stage_params(model_params) if evaluation_mode == "walk_forward" else _sanitize_model_params(model_params)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                evaluate_one,
                name,
                constructor,
                effective_params[name],
                X,
                y,
                forward_returns,
                horizon_days,
            ): name
            for name, constructor in constructors.items()
        }
        for future in as_completed(futures):
            name, result = future.result()
            model_rows[name] = result

    summary = _summarize_ensemble(model_rows)
    return {
        "models": model_rows,
        "model_params": _sanitize_model_params(model_params),
        "evaluation_params": effective_params,
        **summary,
    }


def fit_models(
    X: pd.DataFrame,
    y: pd.Series,
    model_params: dict,
    model_parallelism: int = 1,
    forward_returns: pd.Series | None = None,
    horizon_days: int = 5,
) -> dict:
    prepared = pd.get_dummies(_prepare_training_frame(X), columns=["symbol"], dtype=float)
    forward_series = (
        forward_returns.reset_index(drop=True)
        if isinstance(forward_returns, pd.Series)
        else pd.Series(np.zeros(len(prepared), dtype=float))
    )
    y_series = y.reset_index(drop=True)
    prepared = prepared.reset_index(drop=True)

    selection_start = max(0, len(prepared) - min(len(prepared), FOUNDRY_SELECTION_MAX_ROWS))
    X_selection = prepared.iloc[selection_start:].reset_index(drop=True)
    y_selection = y_series.iloc[selection_start:].reset_index(drop=True)
    forward_selection = forward_series.iloc[selection_start:].reset_index(drop=True)

    leaderboard = []
    candidates = _build_candidate_configs(model_params)
    for candidate in candidates:
        candidate_metrics = _evaluate_bundle(
            X_selection,
            y_selection,
            forward_selection,
            candidate["params"],
            model_parallelism=model_parallelism,
            evaluation_mode="walk_forward",
            horizon_days=horizon_days,
        )
        candidate_summary = {
            "label": candidate["label"],
            "score": _score_candidate(candidate_metrics),
            "ensemble_accuracy": candidate_metrics["ensemble_accuracy"],
            "ensemble_log_loss": candidate_metrics["ensemble_log_loss"],
            "ensemble_daily_return_proxy": candidate_metrics["ensemble_daily_return_proxy"],
            "ensemble_annualized_return_proxy": candidate_metrics["ensemble_annualized_return_proxy"],
            "ensemble_sharpe_proxy": candidate_metrics["ensemble_sharpe_proxy"],
            "ensemble_win_rate": candidate_metrics["ensemble_win_rate"],
            "robustness_score": candidate_metrics["robustness_score"],
            "generalization_gap": candidate_metrics["generalization_gap"],
            "selection_rows": int(len(X_selection)),
            "walk_forward_folds": max(
                [len((row.get("folds") or [])) for row in candidate_metrics["models"].values()] or [0]
            ),
            "model_errors": {
                model_name: row.get("error")
                for model_name, row in candidate_metrics["models"].items()
                if row.get("error")
            },
            "model_params": candidate_metrics["model_params"],
        }
        leaderboard.append(candidate_summary)

    leaderboard = sorted(
        leaderboard,
        key=lambda row: (
            row["score"],
            row["ensemble_accuracy"],
            row["ensemble_daily_return_proxy"],
            row["robustness_score"],
        ),
        reverse=True,
    )
    champion = leaderboard[0] if leaderboard else {"label": "defaults", "model_params": _sanitize_model_params(model_params), "score": 0.0}
    holdout_metrics = _evaluate_bundle(
        prepared,
        y_series,
        forward_series,
        champion["model_params"],
        model_parallelism=model_parallelism,
        evaluation_mode="holdout",
        horizon_days=horizon_days,
    )
    holdout_metrics["candidate_leaderboard"] = leaderboard
    holdout_metrics["champion_score"] = float(champion.get("score", 0.0))
    holdout_metrics["selected_candidate_label"] = champion.get("label", "warm_start")
    holdout_metrics["candidate_count"] = len(leaderboard)
    holdout_metrics["selection_method"] = "walk_forward_champion_challenger"
    holdout_metrics["selection_rows"] = int(len(X_selection))
    holdout_metrics["selection_robustness_score"] = float(champion.get("robustness_score", holdout_metrics["robustness_score"]))
    holdout_metrics["walk_forward_accuracy"] = float(champion.get("ensemble_accuracy", holdout_metrics["ensemble_accuracy"]))
    holdout_metrics["walk_forward_daily_return_proxy"] = float(
        champion.get("ensemble_daily_return_proxy", holdout_metrics["ensemble_daily_return_proxy"])
    )
    return holdout_metrics


def zero_calibration_pack(symbols: list[str], target_daily_return: float, target_accuracy: float):
    return {
        "mode": "zero-calibration",
        "universe": symbols,
        "ensemble_weights": {"logreg": 0.45, "rf": 0.35, "mlp": 0.20},
        "entry_policy": {
            "min_ensemble_confidence": max(0.52, target_accuracy - 0.02),
            "min_expected_daily_return": target_daily_return,
            "max_position_per_symbol": 0.03,
            "max_sector_exposure": 0.20,
        },
        "risk_policy": {
            "stop_volatility_regime": 0.035,
            "capital_throttle_if_drawdown_pct": -0.06,
            "notes": "Static priors for immediate deployment without data refit.",
        },
    }


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def to_repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def save_last_model_config(path: Path, model_params: dict, mode: str, symbols: list[str], horizon_days: int, metrics: dict):
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "horizon_days": horizon_days,
        "symbols": symbols,
        "model_params": _sanitize_model_params(model_params),
        "training_summary": {
            "ensemble_accuracy": metrics.get("ensemble_accuracy"),
            "ensemble_weights": metrics.get("ensemble_weights"),
            "ensemble_daily_return_proxy": metrics.get("ensemble_daily_return_proxy"),
            "robustness_score": metrics.get("robustness_score"),
            "champion_score": metrics.get("champion_score"),
            "selected_candidate_label": metrics.get("selected_candidate_label"),
        },
        "selection_method": metrics.get("selection_method"),
        "candidate_count": metrics.get("candidate_count"),
        "candidate_leaderboard": (metrics.get("candidate_leaderboard") or [])[:3],
        "note": "Last successful foundry model parameters for warm-start and overfitting control.",
    }
    write_json(path, payload)


def update_adaptive_profile_from_pack(pack: dict, mode: str):
    profile = {
        "version": 2,
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "lookback": 30,
        "daily_returns": [],
        "confidence_samples": [],
        "risk_multiplier": 1.0,
        "deployment_multiplier": 1.0,
        "trade_intensity_multiplier": 1.0,
        "regime": "neutral",
        "notes": "Foundry profile controls are optimization levers, not profit guarantees.",
        "foundry_mode": mode,
        "foundry_min_confidence": pack.get("entry_policy", {}).get("min_ensemble_confidence", 0.55),
    }
    write_json(ADAPTIVE_PROFILE_PATH, profile)


def main():
    args = parse_args()
    symbols = load_symbols(args.symbols, args.max_symbols)

    report_path = Path(args.report_path)
    pack_path = Path(args.pack_path)
    model_config_path = Path(args.model_config_path)

    if args.mode == "zero-calibration":
        pack = zero_calibration_pack(symbols, args.target_daily_return, args.target_accuracy)
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "symbols": symbols,
            "summary": {
                "n_symbols": len(symbols),
                "target_daily_return": args.target_daily_return,
                "target_accuracy": args.target_accuracy,
                "note": "No fitting performed; pack emitted from robust priors.",
            },
        }
    else:
        close = download_close_matrix(symbols, period=args.period, auto_adjust=True, progress=False)
        if close is None or close.empty:
            raise SystemExit("Failed to download prices for foundry calibration.")
        close = close.dropna(how="all").ffill().dropna(axis=1, how="all")

        panel = _build_feature_panel(close, args.horizon_days)
        X = panel.drop(columns=["target", "forward_return"])
        y = panel["target"]
        forward_returns = panel["forward_return"]
        if len(X) < 600:
            raise SystemExit("Not enough samples for calibration. Increase period or universe.")

        warm_start_params = None if args.disable_model_warm_start else load_last_model_params(model_config_path)
        working_model_params = _sanitize_model_params(warm_start_params)
        working_model_params["rf"]["n_jobs"] = max(1, int(args.rf_jobs))
        metrics = fit_models(
            X,
            y,
            model_params=working_model_params,
            model_parallelism=args.model_parallelism,
            forward_returns=forward_returns,
            horizon_days=args.horizon_days,
        )
        est_daily = float(metrics.get("ensemble_daily_return_proxy", 0.0))
        est_annual = float(metrics.get("ensemble_annualized_return_proxy", 0.0))

        pack = {
            "mode": "weekend-calibrate",
            "universe": list(close.columns),
            "horizon_days": args.horizon_days,
            "ensemble_weights": metrics["ensemble_weights"],
            "entry_policy": {
                "min_ensemble_confidence": max(0.51, args.target_accuracy - 0.01),
                "min_expected_daily_return": args.target_daily_return,
                "dynamic_top_k": 12,
            },
            "risk_policy": {
                "capital_throttle_if_drawdown_pct": -0.05,
                "max_symbol_weight": 0.035,
                "max_gross_exposure": 1.4,
            },
            "selection": {
                "method": metrics.get("selection_method"),
                "selected_candidate_label": metrics.get("selected_candidate_label"),
                "candidate_count": metrics.get("candidate_count"),
                "champion_score": metrics.get("champion_score"),
                "walk_forward_accuracy": metrics.get("walk_forward_accuracy"),
                "walk_forward_daily_return_proxy": metrics.get("walk_forward_daily_return_proxy"),
                "selection_robustness_score": metrics.get("selection_robustness_score"),
            },
            "validation": metrics,
        }
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "symbols": list(close.columns),
            "samples": int(len(X)),
            "summary": {
                "ensemble_accuracy": metrics["ensemble_accuracy"],
                "estimated_strategy_daily_return": est_daily,
                "estimated_strategy_annual_return": est_annual,
                "champion_score": metrics.get("champion_score"),
                "robustness_score": metrics.get("robustness_score"),
                "walk_forward_accuracy": metrics.get("walk_forward_accuracy"),
                "walk_forward_daily_return_proxy": metrics.get("walk_forward_daily_return_proxy"),
                "generalization_gap": metrics.get("generalization_gap"),
                "meets_accuracy_target": bool(metrics["ensemble_accuracy"] >= args.target_accuracy),
                "meets_daily_return_target": bool(est_daily >= args.target_daily_return),
                "targets": {
                    "target_accuracy": args.target_accuracy,
                    "target_daily_return": args.target_daily_return,
                },
                "note": "Champion selected with walk-forward stability, return-proxy, and anti-overfit penalties. Targets remain optimization objectives, not guarantees.",
            },
            "model_details": metrics["models"],
            "model_params_used": metrics["model_params"],
            "candidate_leaderboard": metrics.get("candidate_leaderboard"),
            "selection_method": metrics.get("selection_method"),
            "selected_candidate_label": metrics.get("selected_candidate_label"),
        }
        save_last_model_config(
            path=model_config_path,
            model_params=metrics["model_params"],
            mode=args.mode,
            symbols=list(close.columns),
            horizon_days=args.horizon_days,
            metrics=metrics,
        )

    write_json(pack_path, pack)
    write_json(report_path, report)
    update_adaptive_profile_from_pack(pack, args.mode)

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "pack_path": to_repo_relative(pack_path),
        "report_path": to_repo_relative(report_path),
        "symbols": symbols,
        "note": "Research pack for fine-tuning and staged paper-trading rollout.",
    }
    register_model_snapshot("quant_research_foundry", snapshot)

    print(json.dumps({"pack_path": str(pack_path), "report_path": str(report_path), "mode": args.mode}, indent=2))


if __name__ == "__main__":
    main()
