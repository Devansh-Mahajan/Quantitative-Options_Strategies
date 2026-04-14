import sys

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib

from core.market_intelligence import prioritize_symbols
from core.resource_profile import load_resource_profile
from core.runtime_calibration import MARKET_POLICY_PATH, save_market_regime_policy
from core.state_manager import register_model_snapshot
from core.strategy_regime import build_live_controls, clamp, macro_regime_to_market_state
from core.universe_maintenance import (
    download_close_matrix,
    load_symbol_file,
    save_symbol_file,
    save_validation_report,
    validate_symbol_universe,
)
from scripts.train_hmm import build_macro_features

ROOT = Path(__file__).resolve().parent.parent
SYMBOLS_FILE = ROOT / "config" / "symbol_list.txt"
VOL_SYMBOLS_FILE = ROOT / "config" / "volatile_symbols.txt"
ADAPTIVE_PROFILE_FILE = ROOT / "config" / "adaptive_profile.json"
HMM_MODEL_FILE = ROOT / "config" / "hmm_macro_model.pkl"
BACKTEST_REPORT_FILE = ROOT / "reports" / "massive_backtest_report.json"
FOUNDRY_REPORT_FILE = ROOT / "reports" / "quant_foundry_report.json"
UNIVERSE_REPORT_FILE = ROOT / "reports" / "universe_validation_report.json"
WEEKEND_REPORT_FILE = ROOT / "reports" / "weekend_recalibration_report.json"
DEFAULT_LOOKBACKS = ["10y", "5y", "3y", "1y", "6mo", "3mo", "ytd"]


def load_adaptive_profile():
    if not ADAPTIVE_PROFILE_FILE.exists():
        return {}
    try:
        return json.loads(ADAPTIVE_PROFILE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def run_step(cmd, env=None):
    print(f"[recalibration] running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, cwd=ROOT, env=env)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        print("\n[recalibration] interrupt received, stopping current step...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def print_pipeline_progress(current_step, total_steps, label):
    width = 32
    ratio = min(max(current_step / total_steps, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    percent = int(ratio * 100)
    print(f"[pipeline] [{bar}] {percent:>3}% ({current_step}/{total_steps}) {label}")


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _select_strategy_profile(backtest_report: dict | None, macro_regime_label: str | None) -> tuple[str, str, float]:
    profile_suite = (backtest_report or {}).get("strategy_profile_suite") or {}
    profile_summary = profile_suite.get("summary") or {}

    inferred_state = str(
        profile_summary.get("consensus_state")
        or macro_regime_to_market_state(macro_regime_label)
    )

    votes: dict[str, float] = {}
    windows = profile_suite.get("results_by_lookback") or {}
    for result in windows.values():
        summary = result.get("summary") or {}
        candidate = (summary.get("best_profile_by_state") or {}).get(inferred_state) or summary.get("current_state_best_profile") or {}
        profile_name = candidate.get("profile")
        if profile_name:
            votes[str(profile_name)] = votes.get(str(profile_name), 0.0) + float(candidate.get("score", 0.0))

    selected_profile = profile_summary.get("consensus_profile") or "all_weather"
    if votes:
        selected_profile = max(votes.items(), key=lambda item: (item[1], item[0]))[0]

    current_state_votes = profile_summary.get("current_state_vote_share") or {}
    if current_state_votes:
        inferred_state = max(current_state_votes.items(), key=lambda item: (item[1], item[0]))[0]

    proxy_score = 0.0
    if votes and selected_profile in votes:
        proxy_score = float(votes[selected_profile]) / max(1, len(windows))
    else:
        proxy_score = float(profile_summary.get("avg_best_profile_score", 0.0))

    return str(selected_profile), str(inferred_state), float(proxy_score)


def _build_market_regime_policy(backtest_report: dict | None, adaptive_profile: dict | None, args) -> dict:
    if not HMM_MODEL_FILE.exists():
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": "missing_hmm_model",
            "risk_multiplier": 1.0,
            "deployment_multiplier": 1.0,
            "trade_intensity_multiplier": 1.0,
        }

    payload = joblib.load(HMM_MODEL_FILE)
    model = payload.get("model")
    scaler = payload.get("scaler")
    features_list = payload.get("features_list") or []
    state_map = payload.get("state_map") or {}

    raw = download_close_matrix(
        ["SPY", "^VIX", "^TNX", "DX-Y.NYB", "HYG", "LQD"],
        period="5y",
        auto_adjust=False,
        progress=False,
    )
    raw = raw.rename(columns={"^VIX": "VIX", "^TNX": "TNX", "DX-Y.NYB": "DXY"}).dropna()
    if raw.empty or model is None or scaler is None or not features_list:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": "incomplete_hmm_payload",
            "risk_multiplier": 1.0,
            "deployment_multiplier": 1.0,
            "trade_intensity_multiplier": 1.0,
        }

    features = build_macro_features(raw)
    features = features.replace([float("inf"), float("-inf")], float("nan")).dropna()
    missing = [column for column in features_list if column not in features.columns]
    if missing:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"missing_features:{','.join(missing)}",
            "risk_multiplier": 1.0,
            "deployment_multiplier": 1.0,
            "trade_intensity_multiplier": 1.0,
        }

    X = scaler.transform(features[features_list].values)
    probs = model.predict_proba(X)
    current_state = int(probs[-1].argmax())
    current_confidence = float(probs[-1].max())
    current_label = state_map.get(current_state, f"STATE_{current_state}")

    base = {
        "GOLDILOCKS": (1.05, 1.00, 1.00),
        "TRANSITION": (0.92, 0.90, 0.92),
        "RISK_OFF": (0.74, 0.68, 0.78),
        "LIQUIDITY_CRUNCH": (0.60, 0.54, 0.64),
        "PANIC": (0.48, 0.42, 0.55),
    }.get(current_label, (0.90, 0.88, 0.90))

    predictive_score = float((backtest_report or {}).get("massive_overview", {}).get("predictive_score", 0.5))
    regime_by_lookback = (backtest_report or {}).get("regime_suite", {}).get("summary", {}).get("by_lookback", {})
    ytd_regime_accuracy = float((regime_by_lookback.get("ytd") or {}).get("directional_accuracy_proxy", predictive_score))
    selected_profile, inferred_market_state, profile_proxy_score = _select_strategy_profile(backtest_report, current_label)
    macro_state = macro_regime_to_market_state(current_label, inferred_market_state)
    if current_label in {"PANIC", "LIQUIDITY_CRUNCH"}:
        inferred_market_state = macro_state
        if selected_profile not in {"panic_hedge", "long_vol_breakout", "bear_trend"}:
            selected_profile = "panic_hedge"
    elif current_label == "RISK_OFF" and selected_profile == "theta_harvest":
        selected_profile = "bear_trend"

    live_controls = build_live_controls(
        profile_name=selected_profile,
        market_state=inferred_market_state,
        predictive_score=predictive_score,
        state_confidence=current_confidence,
        adaptive_profile=adaptive_profile or {},
    )

    quality_scalar = 0.85 + (0.30 * predictive_score)
    risk_multiplier = base[0] * quality_scalar * float(live_controls.get("risk_bias", 1.0))
    deployment_multiplier = base[1] * quality_scalar * float(live_controls.get("deployment_bias", 1.0))
    trade_intensity_multiplier = (
        base[2]
        * (0.90 + 0.20 * ytd_regime_accuracy)
        * float(live_controls.get("trade_intensity_bias", 1.0))
    )

    if ytd_regime_accuracy < 0.50:
        risk_multiplier *= 0.92
        deployment_multiplier *= 0.88
        trade_intensity_multiplier *= 0.90

    if profile_proxy_score < args.profile_score_floor:
        risk_multiplier *= 0.90
        deployment_multiplier *= 0.85
        trade_intensity_multiplier *= 0.88
        live_controls["min_signal_confidence"] = round(
            clamp(float(live_controls.get("min_signal_confidence", 0.18)) + 0.03, 0.12, 0.40),
            4,
        )

    risk_multiplier = max(args.policy_risk_floor, min(args.policy_risk_cap, risk_multiplier))
    deployment_multiplier = max(args.policy_deployment_floor, min(args.policy_deployment_cap, deployment_multiplier))
    trade_intensity_multiplier = max(args.policy_intensity_floor, min(args.policy_intensity_cap, trade_intensity_multiplier))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_state_id": current_state,
        "current_regime_label": current_label,
        "current_market_state": inferred_market_state,
        "selected_profile": selected_profile,
        "current_state_confidence": round(current_confidence, 4),
        "predictive_score": round(predictive_score, 4),
        "ytd_regime_accuracy": round(ytd_regime_accuracy, 4),
        "profile_proxy_score": round(profile_proxy_score, 4),
        "risk_multiplier": round(risk_multiplier, 4),
        "deployment_multiplier": round(deployment_multiplier, 4),
        "trade_intensity_multiplier": round(trade_intensity_multiplier, 4),
        "live_controls": live_controls,
        "note": "Weekend HMM policy for current market conditions. Multipliers are adaptive controls, not guarantees.",
    }


def _build_readiness_snapshot(
    symbols_before: list[str],
    symbols_after: list[str],
    prioritized: list[str],
    validation_report: dict,
    backtest_report: dict,
    foundry_report: dict,
    market_policy: dict,
    foundry_enabled: bool,
    args,
) -> dict:
    movement_summary = (backtest_report.get("movement_suite") or {}).get("summary", {})
    foundry_summary = foundry_report.get("summary") or {}
    predictive_score = float((backtest_report.get("massive_overview") or {}).get("predictive_score", 0.0))
    institutional = backtest_report.get("institutional_robustness") or {}
    institutional_score = float(institutional.get("institutional_score", predictive_score))
    deployment_tier = str(institutional.get("deployment_tier", "research_only"))
    consensus_strategy_profile = (backtest_report.get("massive_overview") or {}).get("consensus_strategy_profile")
    consensus_market_state = (backtest_report.get("massive_overview") or {}).get("consensus_market_state")
    avg_accuracy = float(movement_summary.get("avg_accuracy", 0.0))
    foundry_accuracy_ok = bool(foundry_summary.get("meets_accuracy_target", False))
    foundry_return_ok = bool(foundry_summary.get("meets_daily_return_target", False))

    live_ready = all(
        [
            len(symbols_after) >= 25,
            predictive_score >= max(0.50, args.target_accuracy - 0.06),
            institutional_score >= 0.58,
            avg_accuracy >= max(0.50, args.target_accuracy - 0.04),
            foundry_accuracy_ok or not foundry_enabled,
            foundry_return_ok or not foundry_enabled,
            "error" not in market_policy,
        ]
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "symbols_before_cleanup": len(symbols_before),
            "symbols_after_cleanup": len(symbols_after),
            "prioritized_symbols": len(prioritized),
            "lookbacks": args.lookbacks,
            "target_accuracy": args.target_accuracy,
            "target_daily_return": args.target_daily_return,
        },
        "artifacts": {
            "universe_validation_report": str(UNIVERSE_REPORT_FILE.relative_to(ROOT)),
            "massive_backtest_report": str(BACKTEST_REPORT_FILE.relative_to(ROOT)),
            "quant_foundry_report": str(FOUNDRY_REPORT_FILE.relative_to(ROOT)),
            "market_regime_policy": str(MARKET_POLICY_PATH.relative_to(ROOT)),
        },
        "validation_summary": {
            "symbols_invalid": validation_report.get("symbols_invalid", 0),
            "removed_symbols": [row.get("symbol") for row in validation_report.get("invalid_symbols", [])],
        },
        "predictive_summary": {
            "predictive_score": predictive_score,
            "institutional_score": institutional_score,
            "deployment_tier": deployment_tier,
            "movement_avg_accuracy": avg_accuracy,
            "consensus_strategy_profile": consensus_strategy_profile,
            "consensus_market_state": consensus_market_state,
            "foundry_accuracy_target_met": foundry_accuracy_ok,
            "foundry_daily_return_target_met": foundry_return_ok,
        },
        "current_market_policy": market_policy,
        "readiness": {
            "live_ready": live_ready,
            "note": "Weekend recalibration can prepare the stack, but no model or strategy can guarantee profits in every market regime.",
        },
    }


def main():
    resource_profile = load_resource_profile(ROOT)
    parser = argparse.ArgumentParser(description="Self-maintaining weekend recalibration pipeline")
    parser.add_argument("--top-n", type=int, default=0, help="Deprecated: universe pruning is disabled and all valid symbols are retained.")
    parser.add_argument("--train", action="store_true", help="Deprecated: training runs by default; use --no-train to skip.")
    parser.add_argument("--no-train", action="store_true", help="Skip model retraining.")
    parser.add_argument("--quant-foundry", action="store_true", help="Deprecated: foundry runs by default; use --no-quant-foundry to skip.")
    parser.add_argument("--no-quant-foundry", action="store_true", help="Skip quant research foundry pack generation.")
    parser.add_argument("--no-backtest", action="store_true", help="Skip the multi-window massive backtest suite.")
    parser.add_argument("--no-symbol-maintenance", action="store_true", help="Skip automatic symbol validation and cleanup.")
    parser.add_argument("--target-daily-return", type=float, default=0.002, help="Optimization target for expected daily return.")
    parser.add_argument("--target-accuracy", type=float, default=0.56, help="Optimization target for validation accuracy.")
    parser.add_argument("--quant-mode", choices=["weekend-calibrate", "zero-calibration"], default="weekend-calibrate",
                        help="Foundry mode: fit on weekend or emit a no-fit prior pack.")
    parser.add_argument("--lookbacks", nargs="+", default=DEFAULT_LOOKBACKS, help="Backtest windows to evaluate.")
    parser.add_argument("--workers", type=int, default=resource_profile.backtest_workers, help="Worker count for the massive backtest engine.")
    parser.add_argument("--rf-jobs", type=int, default=resource_profile.research_rf_jobs, help="RandomForest thread budget for the foundry.")
    parser.add_argument("--model-parallelism", type=int, default=resource_profile.model_parallelism, help="How many foundry model families fit concurrently.")
    parser.add_argument("--research-max-symbols", type=int, default=60, help="Cap the top prioritized symbols passed into research/backtests.")
    parser.add_argument("--validation-recent-period", default="6mo")
    parser.add_argument("--validation-history-period", default="2y")
    parser.add_argument("--min-recent-rows", type=int, default=40)
    parser.add_argument("--min-history-rows", type=int, default=120)
    parser.add_argument("--profile-score-floor", type=float, default=0.48, help="Minimum acceptable profile proxy score before policy gets more defensive.")
    parser.add_argument("--policy-risk-floor", type=float, default=0.40)
    parser.add_argument("--policy-risk-cap", type=float, default=1.15)
    parser.add_argument("--policy-deployment-floor", type=float, default=0.35)
    parser.add_argument("--policy-deployment-cap", type=float, default=1.10)
    parser.add_argument("--policy-intensity-floor", type=float, default=0.45)
    parser.add_argument("--policy-intensity-cap", type=float, default=1.10)
    args = parser.parse_args()
    print(
        f"[recalibration] resource profile -> cpu={resource_profile.cpu_count} mem={resource_profile.memory_gb:.1f}GB "
        f"backtest_workers={args.workers} rf_jobs={args.rf_jobs} model_parallelism={args.model_parallelism}"
    )
    subprocess_env = {**os.environ, **resource_profile.to_env()}

    train_enabled = not args.no_train
    foundry_enabled = not args.no_quant_foundry
    backtest_enabled = not args.no_backtest
    symbol_maintenance_enabled = not args.no_symbol_maintenance

    base_step_count = 5  # load -> validate -> prioritize -> persist -> snapshot
    total_steps = base_step_count + int(train_enabled) * 5 + int(foundry_enabled) + int(backtest_enabled) + 1
    step_idx = 1

    print_pipeline_progress(step_idx, total_steps, "Loading symbols")
    symbols_before = load_symbol_file(SYMBOLS_FILE)
    if not symbols_before:
        raise SystemExit("No symbols available in config/symbol_list.txt.")

    validation_report = {
        "symbols_input": len(symbols_before),
        "symbols_valid": len(symbols_before),
        "symbols_invalid": 0,
        "valid_symbols": symbols_before,
        "invalid_symbols": [],
    }
    symbols_after = list(symbols_before)
    if symbol_maintenance_enabled:
        step_idx += 1
        print_pipeline_progress(step_idx, total_steps, "Validating and cleaning symbol universe")
        validation_report = validate_symbol_universe(
            symbols_before,
            recent_period=args.validation_recent_period,
            history_period=args.validation_history_period,
            min_recent_rows=args.min_recent_rows,
            min_history_rows=args.min_history_rows,
        )
        symbols_after = validation_report["valid_symbols"]
        if not symbols_after:
            raise SystemExit("Universe validation removed every symbol. Check the download feed or lower the row thresholds.")
        save_symbol_file(SYMBOLS_FILE, symbols_after)
        save_validation_report(UNIVERSE_REPORT_FILE, validation_report)
    else:
        step_idx += 1
        print_pipeline_progress(step_idx, total_steps, "Skipping symbol maintenance")

    step_idx += 1
    print_pipeline_progress(step_idx, total_steps, f"Prioritizing {len(symbols_after)} valid symbols")
    prioritized = prioritize_symbols(symbols_after, top_n=len(symbols_after))
    prioritized = prioritized or symbols_after

    step_idx += 1
    print_pipeline_progress(step_idx, total_steps, "Writing volatile symbol list")
    save_symbol_file(VOL_SYMBOLS_FILE, prioritized)

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbols_input": len(symbols_before),
        "symbols_retained": len(symbols_after),
        "symbols_removed": len(validation_report.get("invalid_symbols", [])),
        "top_n_requested": args.top_n,
        "top_n_applied": len(prioritized),
        "volatile_symbols_file": str(VOL_SYMBOLS_FILE.relative_to(ROOT)),
        "target_daily_return": args.target_daily_return,
        "target_accuracy": args.target_accuracy,
        "adaptive_profile": load_adaptive_profile(),
        "profile_score_floor": args.profile_score_floor,
        "note": "Weekend recalibration prepares the stack automatically, but targets remain optimization objectives, not guarantees.",
    }
    step_idx += 1
    print_pipeline_progress(step_idx, total_steps, "Saving recalibration snapshot")
    register_model_snapshot("weekend_recalibration", snapshot)

    research_symbols = prioritized[: max(1, min(args.research_max_symbols, len(prioritized)))]
    python_exec = sys.executable

    steps = []
    if train_enabled:
        steps.extend(
            [
                [python_exec, "scripts/train_hmm.py"],
                [python_exec, "scripts/train_correlation_alpha.py"],
                [python_exec, "scripts/mega_matrix.py", "--target-annual-return", str(args.target_daily_return * 252)],
                [python_exec, "scripts/mega_gpu_training.py", "--target-annual-return", str(args.target_daily_return * 252), "--target-accuracy", str(args.target_accuracy)],
                [python_exec, "scripts/train_regime_movement_models.py", "--target-accuracy", str(args.target_accuracy)],
            ]
        )
    if foundry_enabled:
        steps.append(
            [
                python_exec,
                "scripts/quant_research_foundry.py",
                "--mode",
                args.quant_mode,
                "--target-daily-return",
                str(args.target_daily_return),
                "--target-accuracy",
                str(args.target_accuracy),
                "--max-symbols",
                str(min(35, len(research_symbols))),
                "--rf-jobs",
                str(args.rf_jobs),
                "--model-parallelism",
                str(args.model_parallelism),
                "--symbols",
                *research_symbols[:35],
            ]
        )
    if backtest_enabled:
        steps.append(
            [
                python_exec,
                "scripts/massive_backtest_engine.py",
                "--target-daily-return",
                str(args.target_daily_return),
                "--target-accuracy",
                str(args.target_accuracy),
                "--workers",
                str(args.workers),
                "--output",
                str(BACKTEST_REPORT_FILE),
                "--lookbacks",
                *args.lookbacks,
                "--symbols",
                *research_symbols,
            ]
        )

    for cmd in steps:
        step_idx += 1
        print_pipeline_progress(step_idx, total_steps, f"Running {' '.join(cmd[1:3])}")
        run_step(cmd, env=subprocess_env)

    step_idx += 1
    print_pipeline_progress(step_idx, total_steps, "Building market regime policy + readiness report")
    backtest_report = _safe_read_json(BACKTEST_REPORT_FILE)
    foundry_report = _safe_read_json(FOUNDRY_REPORT_FILE)
    adaptive_profile = load_adaptive_profile()
    market_policy = _build_market_regime_policy(backtest_report, adaptive_profile, args)
    save_market_regime_policy(MARKET_POLICY_PATH, market_policy)

    weekend_report = _build_readiness_snapshot(
        symbols_before=symbols_before,
        symbols_after=symbols_after,
        prioritized=prioritized,
        validation_report=validation_report,
        backtest_report=backtest_report,
        foundry_report=foundry_report,
        market_policy=market_policy,
        foundry_enabled=foundry_enabled,
        args=args,
    )
    WEEKEND_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    WEEKEND_REPORT_FILE.write_text(json.dumps(weekend_report, indent=2), encoding="utf-8")
    register_model_snapshot("weekend_recalibration", weekend_report)

    print(json.dumps(weekend_report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[recalibration] interrupted by user; exiting cleanly.")
        raise SystemExit(130)
