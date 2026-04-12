from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from core.greeks_targeting import PortfolioGreekTargets
from core.movement_predictor import MovementSignal

logger = logging.getLogger(f"strategy.{__name__}")

BUCKETS = ("THETA", "VEGA", "BULL", "BEAR")


@dataclass(frozen=True)
class RoutedCandidate:
    symbol: str
    score: float
    components: dict[str, float]


@dataclass(frozen=True)
class RoutingPlan:
    theta_candidates: list[str]
    vega_candidates: list[str]
    bull_candidates: list[str]
    bear_candidates: list[str]
    consensus_score: float
    deployment_multiplier: float
    diagnostics: dict[str, object]


def empty_ai_targets() -> dict[str, list[str]]:
    return {bucket: [] for bucket in BUCKETS}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _normalize_flow_scores(symbols: Sequence[str], flow_map: Mapping[str, float] | None) -> dict[str, float]:
    if not symbols:
        return {}

    raw = {symbol: float((flow_map or {}).get(symbol, 0.0)) for symbol in symbols}
    lo = min(raw.values())
    hi = max(raw.values())
    if hi - lo <= 1e-9:
        return {symbol: 0.5 for symbol in symbols}
    return {symbol: (value - lo) / (hi - lo) for symbol, value in raw.items()}


def _pair_confidence_maps(pair_overlay: Mapping[str, object] | None) -> tuple[dict[str, float], dict[str, float]]:
    long_conf = defaultdict(float)
    short_conf = defaultdict(float)

    for signal in (pair_overlay or {}).get("signals", []) or []:
        confidence = _clamp(float(signal.get("confidence", 0.0)))
        long_symbol = str(signal.get("long", "")).upper()
        short_symbol = str(signal.get("short", "")).upper()
        if long_symbol:
            long_conf[long_symbol] = max(long_conf[long_symbol], confidence)
        if short_symbol:
            short_conf[short_symbol] = max(short_conf[short_symbol], confidence)

    return dict(long_conf), dict(short_conf)


def _macro_adjustments(macro_strategy: str, macro_confidence: float) -> dict[str, float]:
    macro_conf = _clamp(macro_confidence)
    if macro_strategy == "THETA_ENGINE":
        weights = {"THETA": 0.24, "VEGA": -0.22, "BULL": 0.06, "BEAR": 0.06}
    elif macro_strategy == "VEGA_SNIPER":
        weights = {"THETA": -0.28, "VEGA": 0.22, "BULL": -0.04, "BEAR": -0.04}
    elif macro_strategy == "TAIL_HEDGE":
        weights = {"THETA": -0.42, "VEGA": 0.18, "BULL": -0.18, "BEAR": -0.18}
    else:
        weights = {bucket: 0.0 for bucket in BUCKETS}
    return {bucket: macro_conf * weight for bucket, weight in weights.items()}


def _candidate_dict(candidate: RoutedCandidate) -> dict[str, object]:
    return {
        "symbol": candidate.symbol,
        "score": round(candidate.score, 4),
        "components": {key: round(value, 4) for key, value in candidate.components.items()},
    }


def route_strategy_candidates(
    allowed_symbols: Sequence[str],
    ai_targets: Mapping[str, Sequence[str]] | None,
    movement_signals: Iterable[MovementSignal],
    flow_map: Mapping[str, float] | None,
    pair_overlay: Mapping[str, object] | None,
    greek_targets: PortfolioGreekTargets,
    macro_strategy: str,
    macro_confidence: float,
    top_k: int = 12,
) -> RoutingPlan:
    symbols = list(dict.fromkeys(str(symbol).upper() for symbol in allowed_symbols if symbol))
    if not symbols:
        return RoutingPlan([], [], [], [], 0.0, 0.45, {"reason": "no_allowed_symbols"})

    ai_sets = {
        bucket: {str(symbol).upper() for symbol in (ai_targets or {}).get(bucket, []) if symbol}
        for bucket in BUCKETS
    }
    movement_map = {
        str(signal.symbol).upper(): signal
        for signal in movement_signals
        if getattr(signal, "symbol", None)
    }
    flow_scores = _normalize_flow_scores(symbols, flow_map)
    pair_bulls, pair_bears = _pair_confidence_maps(pair_overlay)
    macro_conf = _clamp(macro_confidence)
    macro_adjust = _macro_adjustments(macro_strategy, macro_conf)

    theta_demand = 0.10 if greek_targets.target_theta > 0 else 0.0
    vega_demand = 0.10 if greek_targets.target_vega > 0 else 0.0
    bullish_bias_boost = 0.10 if greek_targets.movement_bias == "bullish" else 0.0
    bearish_bias_boost = 0.10 if greek_targets.movement_bias == "bearish" else 0.0

    ranked: dict[str, list[RoutedCandidate]] = {bucket: [] for bucket in BUCKETS}

    for symbol in symbols:
        flow_score = flow_scores.get(symbol, 0.5)
        balanced_flow = 1.0 - abs((flow_score * 2.0) - 1.0)
        trend_flow = 1.0 - balanced_flow

        movement = movement_map.get(symbol)
        if movement:
            direction_confidence = _clamp(abs(movement.probability_up - 0.5) * 2.0)
            bull_edge = _clamp((movement.probability_up - 0.5) * 2.0)
            bear_edge = _clamp((0.5 - movement.probability_up) * 2.0)
            move_amplitude = _clamp(abs(movement.expected_daily_move) * 80.0)
            range_bound = _clamp(1.0 - max(direction_confidence, move_amplitude))
            explosive = _clamp((0.65 * move_amplitude) + (0.35 * direction_confidence))
        else:
            direction_confidence = 0.0
            bull_edge = 0.0
            bear_edge = 0.0
            move_amplitude = 0.0
            range_bound = 0.55
            explosive = 0.25

        pair_bull = pair_bulls.get(symbol, 0.0)
        pair_bear = pair_bears.get(symbol, 0.0)
        pair_pressure = max(pair_bull, pair_bear)

        theta_score = _clamp(
            (0.52 if symbol in ai_sets["THETA"] else 0.0)
            + (0.20 * range_bound)
            + (0.12 * balanced_flow)
            + (0.08 * (1.0 - pair_pressure))
            + theta_demand
            + macro_adjust["THETA"],
            0.0,
            1.25,
        )
        vega_score = _clamp(
            (0.52 if symbol in ai_sets["VEGA"] else 0.0)
            + (0.22 * explosive)
            + (0.12 * trend_flow)
            + (0.10 * pair_pressure)
            + vega_demand
            + macro_adjust["VEGA"],
            0.0,
            1.25,
        )
        bull_score = _clamp(
            (0.52 if symbol in ai_sets["BULL"] else 0.0)
            + (0.22 * bull_edge)
            + (0.14 * flow_score)
            + (0.15 * pair_bull)
            + bullish_bias_boost
            - (0.08 * pair_bear)
            + macro_adjust["BULL"],
            0.0,
            1.25,
        )
        bear_score = _clamp(
            (0.52 if symbol in ai_sets["BEAR"] else 0.0)
            + (0.22 * bear_edge)
            + (0.14 * (1.0 - flow_score))
            + (0.15 * pair_bear)
            + bearish_bias_boost
            - (0.08 * pair_bull)
            + macro_adjust["BEAR"],
            0.0,
            1.25,
        )

        ranked["THETA"].append(
            RoutedCandidate(
                symbol=symbol,
                score=theta_score,
                components={
                    "mega": 1.0 if symbol in ai_sets["THETA"] else 0.0,
                    "range_bound": range_bound,
                    "balanced_flow": balanced_flow,
                    "pair_stability": 1.0 - pair_pressure,
                    "macro_adjust": macro_adjust["THETA"],
                },
            )
        )
        ranked["VEGA"].append(
            RoutedCandidate(
                symbol=symbol,
                score=vega_score,
                components={
                    "mega": 1.0 if symbol in ai_sets["VEGA"] else 0.0,
                    "explosive": explosive,
                    "trend_flow": trend_flow,
                    "pair_pressure": pair_pressure,
                    "macro_adjust": macro_adjust["VEGA"],
                },
            )
        )
        ranked["BULL"].append(
            RoutedCandidate(
                symbol=symbol,
                score=bull_score,
                components={
                    "mega": 1.0 if symbol in ai_sets["BULL"] else 0.0,
                    "bull_edge": bull_edge,
                    "flow_score": flow_score,
                    "pair_bull": pair_bull,
                    "macro_adjust": macro_adjust["BULL"],
                },
            )
        )
        ranked["BEAR"].append(
            RoutedCandidate(
                symbol=symbol,
                score=bear_score,
                components={
                    "mega": 1.0 if symbol in ai_sets["BEAR"] else 0.0,
                    "bear_edge": bear_edge,
                    "flow_score": 1.0 - flow_score,
                    "pair_bear": pair_bear,
                    "macro_adjust": macro_adjust["BEAR"],
                },
            )
        )

    thresholds = {"THETA": 0.32, "VEGA": 0.32, "BULL": 0.30, "BEAR": 0.30}
    top_k = max(1, int(top_k))
    selected: dict[str, list[str]] = {}
    diagnostics_candidates: dict[str, list[dict[str, object]]] = {}
    top_scores: dict[str, float] = {}

    for bucket in BUCKETS:
        ordered = sorted(ranked[bucket], key=lambda item: (-item.score, item.symbol))
        selected[bucket] = [item.symbol for item in ordered if item.score >= thresholds[bucket]][:top_k]
        diagnostics_candidates[bucket] = [_candidate_dict(item) for item in ordered[: min(top_k, 5)]]
        top_scores[bucket] = ordered[0].score if ordered else 0.0

    coverage = _clamp(len(movement_map) / max(len(symbols), 1))
    model_coverage = _clamp(len(set().union(*ai_sets.values())) / max(len(symbols), 1))
    pair_coverage = _clamp(len((pair_overlay or {}).get("signals", []) or []) / max(top_k, 1))

    if greek_targets.movement_bias == "bullish":
        directional_alignment = _clamp(0.5 + top_scores["BULL"] - top_scores["BEAR"])
    elif greek_targets.movement_bias == "bearish":
        directional_alignment = _clamp(0.5 + top_scores["BEAR"] - top_scores["BULL"])
    else:
        directional_alignment = _clamp(1.0 - abs(top_scores["BULL"] - top_scores["BEAR"]))

    macro_bucket = {"THETA_ENGINE": "THETA", "VEGA_SNIPER": "VEGA", "TAIL_HEDGE": "VEGA"}.get(
        macro_strategy,
        "THETA",
    )
    macro_alignment = top_scores.get(macro_bucket, 0.0)

    consensus_score = _clamp(
        (0.35 * macro_conf)
        + (0.25 * macro_alignment)
        + (0.20 * directional_alignment)
        + (0.10 * coverage)
        + (0.10 * max(model_coverage, pair_coverage))
    )

    deployment_multiplier = 0.60 + (0.45 * consensus_score)
    if macro_strategy == "TAIL_HEDGE":
        deployment_multiplier = min(deployment_multiplier, 0.75)
    if not any(ai_sets.values()) and not (pair_overlay or {}).get("signals"):
        deployment_multiplier *= 0.85
    deployment_multiplier = round(_clamp(deployment_multiplier, 0.45, 1.05), 4)

    diagnostics = {
        "macro_strategy": macro_strategy,
        "macro_confidence": round(macro_conf, 4),
        "movement_bias": greek_targets.movement_bias,
        "movement_signal_coverage": round(coverage, 4),
        "model_coverage": round(model_coverage, 4),
        "pair_signal_coverage": round(pair_coverage, 4),
        "top_scores": {bucket: round(score, 4) for bucket, score in top_scores.items()},
        "selected": selected,
        "ranked": diagnostics_candidates,
    }

    return RoutingPlan(
        theta_candidates=selected["THETA"],
        vega_candidates=selected["VEGA"],
        bull_candidates=selected["BULL"],
        bear_candidates=selected["BEAR"],
        consensus_score=round(consensus_score, 4),
        deployment_multiplier=deployment_multiplier,
        diagnostics=diagnostics,
    )
