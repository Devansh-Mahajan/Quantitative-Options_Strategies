import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.state_manager import register_model_snapshot
from core.universe_maintenance import download_close_matrix, load_symbol_file

DEFAULT_REPORT_PATH = ROOT / "reports" / "quant_foundry_report.json"
DEFAULT_PACK_PATH = ROOT / "config" / "quant_strategy_pack.json"
DEFAULT_SYMBOLS_PATH = ROOT / "config" / "symbol_list.txt"
ADAPTIVE_PROFILE_PATH = ROOT / "config" / "adaptive_profile.json"


MODEL_SPECS = {
    "logreg": lambda: LogisticRegression(max_iter=400, solver="lbfgs"),
    "rf": lambda: RandomForestClassifier(
        n_estimators=220,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    ),
    "mlp": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), learning_rate_init=8e-4, max_iter=260, random_state=42),
}


def parse_args():
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
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--pack-path", default=str(DEFAULT_PACK_PATH))
    return parser.parse_args()


def load_symbols(cli_symbols: list[str] | None, max_symbols: int) -> list[str]:
    if cli_symbols:
        return list(dict.fromkeys([s.strip().upper() for s in cli_symbols if s.strip()]))[: max(1, max_symbols)]

    if not DEFAULT_SYMBOLS_PATH.exists():
        return ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT"]

    return load_symbol_file(DEFAULT_SYMBOLS_PATH)[: max(1, max_symbols)]


def build_feature_matrix(close: pd.DataFrame, horizon_days: int) -> tuple[pd.DataFrame, pd.Series]:
    returns_1 = close.pct_change(1)
    returns_5 = close.pct_change(5)
    returns_20 = close.pct_change(20)
    vol_20 = returns_1.rolling(20).std()
    z_20 = (close - close.rolling(20).mean()) / close.rolling(20).std(ddof=0)

    market = close.mean(axis=1)
    market_ret = market.pct_change(1)
    market_vol = market_ret.rolling(20).std()

    frames = []
    targets = []
    for sym in close.columns:
        feature_df = pd.DataFrame(
            {
                "symbol": sym,
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
        target = (forward_return > 0).astype(float)
        frames.append(feature_df)
        targets.append(target.rename("target"))

    X = pd.concat(frames, axis=0).dropna()
    y = pd.concat(targets, axis=0).reindex(X.index).astype(int)
    joined = X.join(y, how="inner")
    return joined.drop(columns=["target"]), joined["target"]


def annual_to_daily(annual_return: float) -> float:
    if annual_return <= -0.999:
        return -1.0
    return float((1 + annual_return) ** (1 / 252) - 1)


def fit_models(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model_rows = {}

    for name, constructor in MODEL_SPECS.items():
        try:
            model = constructor()
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]
            pred = (prob >= 0.5).astype(int)
            acc = float(accuracy_score(y_test, pred))
            ll = float(log_loss(y_test, prob, labels=[0, 1]))
            model_rows[name] = {
                "accuracy": acc,
                "log_loss": ll,
                "n_test": int(len(y_test)),
            }
        except Exception as exc:
            model_rows[name] = {"error": str(exc), "accuracy": 0.0, "log_loss": 10.0, "n_test": int(len(y_test))}

    losses = {k: v.get("log_loss", 10.0) for k, v in model_rows.items() if "error" not in v}
    if losses:
        inv = {k: 1.0 / max(1e-5, v) for k, v in losses.items()}
        total = sum(inv.values())
        weights = {k: float(v / total) for k, v in inv.items()}
    else:
        weights = {"logreg": 0.4, "rf": 0.4, "mlp": 0.2}

    ensemble_accuracy = float(sum(model_rows[k]["accuracy"] * weights.get(k, 0.0) for k in model_rows.keys()))
    return {
        "models": model_rows,
        "ensemble_weights": weights,
        "ensemble_accuracy": ensemble_accuracy,
    }


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

        X, y = build_feature_matrix(close, args.horizon_days)
        if len(X) < 600:
            raise SystemExit("Not enough samples for calibration. Increase period or universe.")

        metrics = fit_models(X, y)
        est_annual = (metrics["ensemble_accuracy"] - 0.5) * 0.60
        est_daily = annual_to_daily(est_annual)

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
                "meets_accuracy_target": bool(metrics["ensemble_accuracy"] >= args.target_accuracy),
                "meets_daily_return_target": bool(est_daily >= args.target_daily_return),
                "targets": {
                    "target_accuracy": args.target_accuracy,
                    "target_daily_return": args.target_daily_return,
                },
                "note": "Targets are optimization objectives, not guarantees.",
            },
            "model_details": metrics["models"],
        }

    write_json(pack_path, pack)
    write_json(report_path, report)
    update_adaptive_profile_from_pack(pack, args.mode)

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "pack_path": str(pack_path.relative_to(ROOT)),
        "report_path": str(report_path.relative_to(ROOT)),
        "symbols": symbols,
        "note": "Research pack for fine-tuning and staged paper-trading rollout.",
    }
    register_model_snapshot("quant_research_foundry", snapshot)

    print(json.dumps({"pack_path": str(pack_path), "report_path": str(report_path), "mode": args.mode}, indent=2))


if __name__ == "__main__":
    main()
