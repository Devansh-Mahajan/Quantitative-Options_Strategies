import math


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
