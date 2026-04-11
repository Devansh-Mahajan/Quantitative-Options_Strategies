import json
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(f"strategy.{__name__}")

ADAPTIVE_PROFILE_FILE = Path("config") / "adaptive_profile.json"


class AdaptiveRecalibrationEngine:
    """Simple online policy that nudges risk/deployment knobs from recent outcomes."""

    def __init__(self, profile_path: Path = ADAPTIVE_PROFILE_FILE, lookback: int = 30):
        self.profile_path = Path(profile_path)
        self.lookback = max(5, int(lookback))

    def _default_profile(self):
        return {
            "version": 1,
            "last_updated_utc": None,
            "lookback": self.lookback,
            "daily_returns": [],
            "confidence_samples": [],
            "risk_multiplier": 1.0,
            "deployment_multiplier": 1.0,
            "trade_intensity_multiplier": 1.0,
            "regime": "neutral",
            "notes": "Adaptive controls are optimization levers, not profit guarantees.",
        }

    def load_profile(self):
        if not self.profile_path.exists():
            return self._default_profile()
        try:
            with self.profile_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return self._default_profile()
            merged = self._default_profile()
            merged.update(data)
            return merged
        except Exception as exc:
            logger.warning("Failed loading adaptive profile: %s", exc)
            return self._default_profile()

    def save_profile(self, profile):
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        with self.profile_path.open("w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)

    def update(self, daily_return_pct, signal_confidence, macro_confidence, vix_level):
        profile = self.load_profile()

        returns = deque(profile.get("daily_returns", []), maxlen=self.lookback)
        confidences = deque(profile.get("confidence_samples", []), maxlen=self.lookback)

        returns.append(float(daily_return_pct))
        blended_conf = (float(signal_confidence) + float(macro_confidence)) / 2.0
        confidences.append(max(0.0, min(1.0, blended_conf)))

        avg_return = sum(returns) / len(returns)
        avg_conf = sum(confidences) / len(confidences)

        if avg_return <= -1.5:
            regime = "capital_preservation"
            risk_multiplier = 0.55
            deployment_multiplier = 0.60
            trade_intensity_multiplier = 0.65
        elif avg_return < 0.25:
            regime = "defensive"
            risk_multiplier = 0.80
            deployment_multiplier = 0.85
            trade_intensity_multiplier = 0.85
        elif avg_return < 0.90:
            regime = "balanced"
            risk_multiplier = 1.00
            deployment_multiplier = 1.00
            trade_intensity_multiplier = 1.00
        else:
            regime = "offensive"
            risk_multiplier = 1.15
            deployment_multiplier = 1.10
            trade_intensity_multiplier = 1.05

        confidence_boost = 0.85 + (0.30 * avg_conf)
        volatility_drag = 0.90 if float(vix_level) >= 30.0 else 0.95 if float(vix_level) >= 24.0 else 1.0

        risk_multiplier = max(0.35, min(1.25, risk_multiplier * confidence_boost * volatility_drag))
        deployment_multiplier = max(0.40, min(1.20, deployment_multiplier * confidence_boost * volatility_drag))
        trade_intensity_multiplier = max(0.50, min(1.15, trade_intensity_multiplier * confidence_boost))

        profile.update(
            {
                "last_updated_utc": datetime.now(timezone.utc).isoformat(),
                "lookback": self.lookback,
                "daily_returns": list(returns),
                "confidence_samples": list(confidences),
                "risk_multiplier": round(risk_multiplier, 4),
                "deployment_multiplier": round(deployment_multiplier, 4),
                "trade_intensity_multiplier": round(trade_intensity_multiplier, 4),
                "regime": regime,
                "rolling_avg_return_pct": round(avg_return, 4),
                "rolling_avg_confidence": round(avg_conf, 4),
            }
        )

        self.save_profile(profile)
        return profile
