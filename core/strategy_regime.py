from __future__ import annotations

from typing import Mapping

BUCKETS = ("THETA", "VEGA", "BULL", "BEAR")

BASE_BUCKET_THRESHOLDS = {
    "THETA": 0.32,
    "VEGA": 0.32,
    "BULL": 0.30,
    "BEAR": 0.30,
}

MARKET_STATE_BUCKET_BASE = {
    "calm_bull": {"THETA": 0.80, "VEGA": 0.30, "BULL": 1.00, "BEAR": 0.20},
    "calm_bear": {"THETA": 0.55, "VEGA": 0.40, "BULL": 0.20, "BEAR": 1.00},
    "calm_range": {"THETA": 1.00, "VEGA": 0.20, "BULL": 0.45, "BEAR": 0.45},
    "volatile_bull": {"THETA": 0.30, "VEGA": 0.95, "BULL": 0.85, "BEAR": 0.25},
    "volatile_bear": {"THETA": 0.15, "VEGA": 1.00, "BULL": 0.15, "BEAR": 0.95},
    "panic": {"THETA": 0.05, "VEGA": 1.00, "BULL": 0.05, "BEAR": 0.85},
    "transition": {"THETA": 0.55, "VEGA": 0.65, "BULL": 0.65, "BEAR": 0.65},
}

STRATEGY_PROFILES = {
    "all_weather": {
        "label": "All Weather",
        "thesis": "Balanced deployment across premium capture, long volatility, and directional books.",
        "bucket_weights": {"THETA": 1.00, "VEGA": 1.00, "BULL": 1.00, "BEAR": 1.00},
        "risk_bias": 0.98,
        "deployment_bias": 1.00,
        "trade_intensity_bias": 1.00,
        "dynamic_top_k": 12,
        "predictor_universe_cap": 20,
        "mega_confidence_threshold": 75.0,
        "min_signal_confidence": 0.18,
        "max_symbol_weight": 0.08,
        "min_vix_for_directional_credit": 15.0,
        "max_vix_for_short_premium": 30.0,
        "theta_enabled": True,
        "vega_enabled": True,
        "directional_enabled": True,
    },
    "theta_harvest": {
        "label": "Theta Harvest",
        "thesis": "Lean into short premium when tape is orderly and realized volatility stays contained.",
        "bucket_weights": {"THETA": 1.45, "VEGA": 0.35, "BULL": 0.90, "BEAR": 0.85},
        "risk_bias": 1.05,
        "deployment_bias": 1.06,
        "trade_intensity_bias": 1.08,
        "dynamic_top_k": 14,
        "predictor_universe_cap": 18,
        "mega_confidence_threshold": 72.0,
        "min_signal_confidence": 0.16,
        "max_symbol_weight": 0.09,
        "min_vix_for_directional_credit": 14.0,
        "max_vix_for_short_premium": 24.0,
        "theta_enabled": True,
        "vega_enabled": True,
        "directional_enabled": True,
    },
    "bull_trend": {
        "label": "Bull Trend",
        "thesis": "Favor bullish spreads when momentum is positive and panic risk is low.",
        "bucket_weights": {"THETA": 0.75, "VEGA": 0.55, "BULL": 1.50, "BEAR": 0.35},
        "risk_bias": 1.02,
        "deployment_bias": 1.03,
        "trade_intensity_bias": 1.05,
        "dynamic_top_k": 13,
        "predictor_universe_cap": 24,
        "mega_confidence_threshold": 74.0,
        "min_signal_confidence": 0.17,
        "max_symbol_weight": 0.08,
        "min_vix_for_directional_credit": 15.0,
        "max_vix_for_short_premium": 26.0,
        "theta_enabled": True,
        "vega_enabled": True,
        "directional_enabled": True,
    },
    "bear_trend": {
        "label": "Bear Trend",
        "thesis": "Favor bearish spreads and protection when downside momentum builds.",
        "bucket_weights": {"THETA": 0.40, "VEGA": 0.85, "BULL": 0.20, "BEAR": 1.55},
        "risk_bias": 0.84,
        "deployment_bias": 0.82,
        "trade_intensity_bias": 0.88,
        "dynamic_top_k": 11,
        "predictor_universe_cap": 22,
        "mega_confidence_threshold": 78.0,
        "min_signal_confidence": 0.20,
        "max_symbol_weight": 0.07,
        "min_vix_for_directional_credit": 18.0,
        "max_vix_for_short_premium": 22.0,
        "theta_enabled": True,
        "vega_enabled": True,
        "directional_enabled": True,
    },
    "long_vol_breakout": {
        "label": "Long Vol Breakout",
        "thesis": "Prefer convexity and breakout trades when volatility is rising fast.",
        "bucket_weights": {"THETA": 0.15, "VEGA": 1.60, "BULL": 0.75, "BEAR": 0.75},
        "risk_bias": 0.78,
        "deployment_bias": 0.86,
        "trade_intensity_bias": 0.90,
        "dynamic_top_k": 10,
        "predictor_universe_cap": 26,
        "mega_confidence_threshold": 79.0,
        "min_signal_confidence": 0.21,
        "max_symbol_weight": 0.07,
        "min_vix_for_directional_credit": 18.0,
        "max_vix_for_short_premium": 18.0,
        "theta_enabled": False,
        "vega_enabled": True,
        "directional_enabled": True,
    },
    "panic_hedge": {
        "label": "Panic Hedge",
        "thesis": "Shrink short premium, keep convexity on, and let defense dominate in panic regimes.",
        "bucket_weights": {"THETA": 0.05, "VEGA": 1.85, "BULL": 0.10, "BEAR": 0.95},
        "risk_bias": 0.58,
        "deployment_bias": 0.52,
        "trade_intensity_bias": 0.60,
        "dynamic_top_k": 8,
        "predictor_universe_cap": 16,
        "mega_confidence_threshold": 82.0,
        "min_signal_confidence": 0.24,
        "max_symbol_weight": 0.05,
        "min_vix_for_directional_credit": 22.0,
        "max_vix_for_short_premium": 16.0,
        "theta_enabled": False,
        "vega_enabled": True,
        "directional_enabled": False,
    },
    "mean_reversion": {
        "label": "Mean Reversion",
        "thesis": "Favor range capture and measured directional fades in choppy, non-trending tapes.",
        "bucket_weights": {"THETA": 1.15, "VEGA": 0.55, "BULL": 0.85, "BEAR": 0.85},
        "risk_bias": 0.96,
        "deployment_bias": 0.98,
        "trade_intensity_bias": 0.94,
        "dynamic_top_k": 12,
        "predictor_universe_cap": 18,
        "mega_confidence_threshold": 76.0,
        "min_signal_confidence": 0.19,
        "max_symbol_weight": 0.08,
        "min_vix_for_directional_credit": 15.0,
        "max_vix_for_short_premium": 26.0,
        "theta_enabled": True,
        "vega_enabled": True,
        "directional_enabled": True,
    },
}

MACRO_REGIME_TO_STATE = {
    "GOLDILOCKS": "calm_bull",
    "TRANSITION": "transition",
    "RISK_OFF": "volatile_bear",
    "LIQUIDITY_CRUNCH": "volatile_bear",
    "PANIC": "panic",
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def normalize_bucket_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    raw = {bucket: max(0.0, float((weights or {}).get(bucket, 0.0))) for bucket in BUCKETS}
    total = sum(raw.values())
    if total <= 1e-9:
        even = 1.0 / len(BUCKETS)
        return {bucket: even for bucket in BUCKETS}
    return {bucket: raw[bucket] / total for bucket in BUCKETS}


def classify_market_state(momentum_20: float, vix_level: float, vol_trend: float) -> str:
    mom = float(momentum_20)
    vix = float(vix_level)
    vol = float(vol_trend)

    if vix >= 36.0:
        return "panic"
    if vix >= 27.0 and mom <= -0.015:
        return "volatile_bear"
    if vix >= 24.0 and mom >= 0.012:
        return "volatile_bull"
    if abs(mom) <= 0.008 and vix <= 18.0 and vol <= 0.0:
        return "calm_range"
    if mom >= 0.015 and vix <= 23.0:
        return "calm_bull"
    if mom <= -0.015 and vix <= 24.0:
        return "calm_bear"
    if vol > 0.0 and vix >= 22.0:
        return "volatile_bull" if mom >= 0.0 else "volatile_bear"
    return "transition"


def macro_regime_to_market_state(regime_label: str | None, fallback: str = "transition") -> str:
    if not regime_label:
        return fallback
    return MACRO_REGIME_TO_STATE.get(str(regime_label).upper(), fallback)


def profile_bucket_weights(profile_name: str) -> dict[str, float]:
    profile = STRATEGY_PROFILES.get(profile_name) or STRATEGY_PROFILES["all_weather"]
    return {bucket: float(profile["bucket_weights"].get(bucket, 1.0)) for bucket in BUCKETS}


def combine_profile_with_state(profile_name: str, market_state: str) -> dict[str, float]:
    profile_weights = profile_bucket_weights(profile_name)
    state_weights = MARKET_STATE_BUCKET_BASE.get(market_state, MARKET_STATE_BUCKET_BASE["transition"])
    return {
        bucket: float(profile_weights.get(bucket, 1.0)) * float(state_weights.get(bucket, 1.0))
        for bucket in BUCKETS
    }


def build_bucket_thresholds(
    combined_weights: Mapping[str, float] | None,
    base_thresholds: Mapping[str, float] | None = None,
) -> dict[str, float]:
    normalized = normalize_bucket_weights(combined_weights)
    thresholds = {}
    base = dict(base_thresholds or BASE_BUCKET_THRESHOLDS)
    for bucket in BUCKETS:
        relative_bias = normalized[bucket] - 0.25
        thresholds[bucket] = round(clamp(base.get(bucket, 0.30) - (0.18 * relative_bias), 0.18, 0.45), 4)
    return thresholds


def build_bucket_cap_multipliers(combined_weights: Mapping[str, float] | None) -> dict[str, float]:
    normalized = normalize_bucket_weights(combined_weights)
    return {
        bucket: round(clamp(0.55 + (normalized[bucket] * 3.0), 0.45, 1.65), 4)
        for bucket in BUCKETS
    }


def build_live_controls(
    profile_name: str,
    market_state: str,
    predictive_score: float,
    state_confidence: float,
    adaptive_profile: Mapping[str, object] | None = None,
) -> dict[str, object]:
    profile = STRATEGY_PROFILES.get(profile_name) or STRATEGY_PROFILES["all_weather"]
    combined_weights = combine_profile_with_state(profile_name, market_state)
    normalized_weights = normalize_bucket_weights(combined_weights)
    thresholds = build_bucket_thresholds(combined_weights)
    cap_multipliers = build_bucket_cap_multipliers(combined_weights)

    predictive = clamp(predictive_score)
    confidence = clamp(state_confidence)
    avg_return = float((adaptive_profile or {}).get("rolling_avg_return_pct", 0.0) or 0.0)
    correction_pressure = clamp(max(0.0, -avg_return) / 2.5, 0.0, 0.35)
    quality_scalar = 0.80 + (0.35 * predictive) + (0.15 * confidence)

    min_signal_confidence = clamp(
        float(profile["min_signal_confidence"])
        + max(0.0, 0.56 - predictive) * 0.20
        + correction_pressure * 0.12,
        0.12,
        0.40,
    )
    dynamic_top_k = int(round(max(4, profile["dynamic_top_k"] * quality_scalar)))
    predictor_universe_cap = int(round(max(8, profile["predictor_universe_cap"] * (0.85 + 0.35 * predictive))))
    mega_confidence_threshold = round(
        clamp(
            float(profile["mega_confidence_threshold"]) + max(0.0, 0.58 - predictive) * 30.0,
            60.0,
            92.0,
        ),
        2,
    )
    max_symbol_weight = round(
        clamp(float(profile["max_symbol_weight"]) * (0.90 + 0.25 * predictive) * (1.0 - 0.40 * correction_pressure), 0.03, 0.12),
        4,
    )

    return {
        "selected_profile": profile_name,
        "profile_label": profile["label"],
        "profile_thesis": profile["thesis"],
        "market_state": market_state,
        "strategy_weights": {bucket: round(weight, 4) for bucket, weight in normalized_weights.items()},
        "bucket_thresholds": thresholds,
        "bucket_cap_multipliers": cap_multipliers,
        "dynamic_top_k": dynamic_top_k,
        "predictor_universe_cap": predictor_universe_cap,
        "mega_confidence_threshold": mega_confidence_threshold,
        "min_signal_confidence": round(min_signal_confidence, 4),
        "max_symbol_weight": max_symbol_weight,
        "theta_enabled": bool(profile["theta_enabled"] and normalized_weights["THETA"] >= 0.10),
        "vega_enabled": bool(profile["vega_enabled"] and normalized_weights["VEGA"] >= 0.12),
        "directional_enabled": bool(profile["directional_enabled"] and (normalized_weights["BULL"] + normalized_weights["BEAR"]) >= 0.20),
        "min_vix_for_directional_credit": round(float(profile["min_vix_for_directional_credit"]), 2),
        "max_vix_for_short_premium": round(float(profile["max_vix_for_short_premium"]), 2),
        "risk_bias": round(float(profile["risk_bias"]) * (1.0 - 0.35 * correction_pressure), 4),
        "deployment_bias": round(float(profile["deployment_bias"]) * (1.0 - 0.25 * correction_pressure), 4),
        "trade_intensity_bias": round(float(profile["trade_intensity_bias"]) * (1.0 - 0.30 * correction_pressure), 4),
    }
