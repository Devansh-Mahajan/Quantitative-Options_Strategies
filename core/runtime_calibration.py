from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(f"strategy.{__name__}")

STRATEGY_PACK_PATH = Path("config") / "quant_strategy_pack.json"
MARKET_POLICY_PATH = Path("config") / "market_regime_policy.json"


@dataclass
class RuntimeCalibration:
    min_signal_confidence: float | None = None
    dynamic_top_k: int | None = None
    max_symbol_weight: float | None = None
    predictor_universe_cap: int | None = None
    mega_confidence_threshold: float | None = None
    risk_multiplier: float = 1.0
    deployment_multiplier: float = 1.0
    trade_intensity_multiplier: float = 1.0
    current_regime: str | None = None
    current_market_state: str | None = None
    selected_profile: str | None = None
    current_state_id: int | None = None
    regime_confidence: float | None = None
    strategy_weights: dict[str, float] = field(default_factory=dict)
    bucket_thresholds: dict[str, float] = field(default_factory=dict)
    bucket_cap_multipliers: dict[str, float] = field(default_factory=dict)
    theta_enabled: bool = True
    vega_enabled: bool = True
    directional_enabled: bool = True
    min_vix_for_directional_credit: float = 15.0
    max_vix_for_short_premium: float = 30.0
    profile_proxy_score: float | None = None
    notes: list[str] = field(default_factory=list)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Failed reading calibration artifact %s: %s", path, exc)
        return {}


def load_runtime_calibration(include_live_policy: bool = True) -> RuntimeCalibration:
    pack = _read_json(STRATEGY_PACK_PATH)
    policy = _read_json(MARKET_POLICY_PATH)
    live_controls = (policy.get("live_controls") or {}) if include_live_policy else {}

    entry_policy = pack.get("entry_policy") or {}
    risk_policy = pack.get("risk_policy") or {}

    calibration = RuntimeCalibration(
        min_signal_confidence=entry_policy.get("min_ensemble_confidence"),
        dynamic_top_k=entry_policy.get("dynamic_top_k"),
        max_symbol_weight=risk_policy.get("max_symbol_weight"),
    )

    if policy and include_live_policy:
        calibration.risk_multiplier = float(policy.get("risk_multiplier", 1.0))
        calibration.deployment_multiplier = float(policy.get("deployment_multiplier", 1.0))
        calibration.trade_intensity_multiplier = float(policy.get("trade_intensity_multiplier", 1.0))
        calibration.current_regime = policy.get("current_regime_label")
        calibration.current_market_state = policy.get("current_market_state")
        calibration.selected_profile = policy.get("selected_profile")
        calibration.current_state_id = policy.get("current_state_id")
        calibration.regime_confidence = policy.get("current_state_confidence")
        calibration.profile_proxy_score = policy.get("profile_proxy_score")

    if live_controls:
        calibration.min_signal_confidence = float(
            live_controls.get("min_signal_confidence", calibration.min_signal_confidence or 0.0)
        )
        calibration.dynamic_top_k = int(live_controls.get("dynamic_top_k", calibration.dynamic_top_k or 0) or 0) or None
        calibration.max_symbol_weight = float(
            live_controls.get("max_symbol_weight", calibration.max_symbol_weight or 0.0)
        ) or None
        calibration.predictor_universe_cap = int(live_controls.get("predictor_universe_cap", 0) or 0) or None
        calibration.mega_confidence_threshold = float(live_controls.get("mega_confidence_threshold", 0.0) or 0.0) or None
        calibration.strategy_weights = {str(k): float(v) for k, v in dict(live_controls.get("strategy_weights") or {}).items()}
        calibration.bucket_thresholds = {str(k): float(v) for k, v in dict(live_controls.get("bucket_thresholds") or {}).items()}
        calibration.bucket_cap_multipliers = {
            str(k): float(v) for k, v in dict(live_controls.get("bucket_cap_multipliers") or {}).items()
        }
        calibration.theta_enabled = bool(live_controls.get("theta_enabled", True))
        calibration.vega_enabled = bool(live_controls.get("vega_enabled", True))
        calibration.directional_enabled = bool(live_controls.get("directional_enabled", True))
        calibration.min_vix_for_directional_credit = float(
            live_controls.get("min_vix_for_directional_credit", calibration.min_vix_for_directional_credit)
        )
        calibration.max_vix_for_short_premium = float(
            live_controls.get("max_vix_for_short_premium", calibration.max_vix_for_short_premium)
        )
        calibration.current_market_state = live_controls.get("market_state", calibration.current_market_state)
        calibration.selected_profile = live_controls.get("selected_profile", calibration.selected_profile)

    if pack:
        calibration.notes.append("quant_strategy_pack")
    if policy and include_live_policy:
        calibration.notes.append("market_regime_policy")
    if live_controls:
        calibration.notes.append("live_controls")
    if calibration.selected_profile:
        calibration.notes.append(f"profile:{calibration.selected_profile}")
    return calibration


def save_market_regime_policy(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
