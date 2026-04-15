import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionQualityAssessment:
    fill_price: float | None
    fill_ratio: float
    fill_source: str
    broker_fill_observed: bool
    limit_edge_bps: float | None
    reference_edge_bps: float | None
    score: float
    tier: str


def estimate_option_transaction_cost(
    bid_price: float | None,
    ask_price: float | None,
    open_interest: float | None,
    notional: float,
    spread_weight: float = 0.65,
    slippage_weight: float = 0.25,
    liquidity_weight: float = 0.10,
) -> float:
    """Return expected round-trip cost ratio in [0, 1.5+] where lower is better.

    This is a lightweight TCA proxy for option selection.
    """
    bid = float(bid_price or 0.0)
    ask = float(ask_price or 0.0)
    mid = max((bid + ask) / 2.0, 0.01)

    spread_ratio = max(0.0, (ask - bid) / mid)

    # Slippage increases when options are cheap and spreads are wide.
    # Normalize by contract notional so far OTM penny options are penalized.
    contract_notional = max(notional, 1.0)
    slippage_ratio = min(1.0, ((ask - bid) * 100.0) / contract_notional)

    oi = max(float(open_interest or 0.0), 0.0)
    liquidity_penalty = 1.0 / (1.0 + math.log1p(oi))

    return (
        spread_weight * spread_ratio
        + slippage_weight * slippage_ratio
        + liquidity_weight * liquidity_penalty
    )


def execution_quality_multiplier(cost_ratio: float) -> float:
    """Map transaction-cost ratio into [0.05, 1.0] multiplier."""
    return max(0.05, min(1.0, 1.0 - cost_ratio))


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _signed_edge_bps(observed_price: float | None, reference_price: float | None, *, is_credit: bool) -> float | None:
    observed = _safe_float(observed_price)
    reference = _safe_float(reference_price)
    if observed is None or reference is None:
        return None

    observed_abs = abs(observed)
    reference_abs = abs(reference)
    if reference_abs <= 0.0:
        return None

    edge = observed_abs - reference_abs if is_credit else reference_abs - observed_abs
    return round((edge / reference_abs) * 10000.0, 4)


def assess_execution_quality(
    *,
    fill_price: float | None,
    limit_price: float | None,
    reference_price: float | None = None,
    pricing_confidence: float | None = None,
    staleness_pct: float | None = None,
    is_credit: bool,
    fill_ratio: float = 1.0,
    broker_fill_observed: bool = True,
) -> ExecutionQualityAssessment:
    fill_ratio = max(0.0, min(1.0, float(fill_ratio or 0.0)))
    confidence = max(0.0, min(1.0, float(pricing_confidence or 0.0)))
    staleness = max(0.0, float(staleness_pct or 0.0))
    fill_source = "broker_reported" if broker_fill_observed and fill_price is not None else "limit_fallback"

    limit_edge_bps = _signed_edge_bps(fill_price, limit_price, is_credit=is_credit)
    reference_edge_bps = _signed_edge_bps(fill_price, reference_price, is_credit=is_credit)

    score = 0.60
    score += 0.12 * fill_ratio
    score += min(0.10, confidence * 0.10)
    score -= min(0.15, staleness * 2.0)

    if limit_edge_bps is not None:
        score += max(-0.08, min(0.08, limit_edge_bps / 1500.0))
    if reference_edge_bps is not None:
        score += max(-0.08, min(0.08, reference_edge_bps / 3000.0))
    if not broker_fill_observed:
        score -= 0.10
    if fill_ratio <= 0.0:
        score = min(score, 0.20)

    score = max(0.0, min(1.0, score))
    if score >= 0.82:
        tier = "excellent"
    elif score >= 0.60:
        tier = "acceptable"
    elif score >= 0.44:
        tier = "degraded"
    else:
        tier = "poor"

    return ExecutionQualityAssessment(
        fill_price=_safe_float(fill_price),
        fill_ratio=round(fill_ratio, 6),
        fill_source=fill_source,
        broker_fill_observed=bool(broker_fill_observed and fill_price is not None),
        limit_edge_bps=limit_edge_bps,
        reference_edge_bps=reference_edge_bps,
        score=round(score, 6),
        tier=tier,
    )
