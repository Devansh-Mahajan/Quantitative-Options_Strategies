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
    risk_multiplier: float = 1.0
    deployment_multiplier: float = 1.0
    trade_intensity_multiplier: float = 1.0
    current_regime: str | None = None
    current_state_id: int | None = None
    regime_confidence: float | None = None
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


def load_runtime_calibration() -> RuntimeCalibration:
    pack = _read_json(STRATEGY_PACK_PATH)
    policy = _read_json(MARKET_POLICY_PATH)

    entry_policy = pack.get("entry_policy") or {}
    risk_policy = pack.get("risk_policy") or {}

    calibration = RuntimeCalibration(
        min_signal_confidence=entry_policy.get("min_ensemble_confidence"),
        dynamic_top_k=entry_policy.get("dynamic_top_k"),
        max_symbol_weight=risk_policy.get("max_symbol_weight"),
    )

    if policy:
        calibration.risk_multiplier = float(policy.get("risk_multiplier", 1.0))
        calibration.deployment_multiplier = float(policy.get("deployment_multiplier", 1.0))
        calibration.trade_intensity_multiplier = float(policy.get("trade_intensity_multiplier", 1.0))
        calibration.current_regime = policy.get("current_regime_label")
        calibration.current_state_id = policy.get("current_state_id")
        calibration.regime_confidence = policy.get("current_state_confidence")

    if pack:
        calibration.notes.append("quant_strategy_pack")
    if policy:
        calibration.notes.append("market_regime_policy")
    return calibration


def save_market_regime_policy(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
