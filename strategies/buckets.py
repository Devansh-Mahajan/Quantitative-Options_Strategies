"""
Strategy → macro bucket mapping.

Each bucket describes the market condition the strategy thrives in:
  THETA — range-bound, low vol, premium/income collection
  VEGA  — vol expansion, explosive directional moves
  BULL  — trending bull, positive momentum
  BEAR  — trending bear / short-biased opportunities

Used by SignalRouter to boost or dampen signal confidence based on whether
the current macro regime (from MegaStrategyNet + RoutingPlan) aligns with
the strategy's target environment.
"""

from __future__ import annotations

STRATEGY_BUCKETS: dict[str, str] = {
    # ── THETA: income, mean-reversion, neutral ──────────────────────────
    "mean_reversion":          "THETA",
    "options_vol":             "THETA",
    "funding_arb":             "THETA",
    "basis_trade":             "THETA",
    "pairs_arb":               "THETA",
    "pivot_sr":                "THETA",
    "rma_strategy":            "THETA",
    "statistical_arb":         "THETA",
    "carry_portfolio":         "THETA",
    "market_making":           "THETA",

    # ── VEGA: explosive moves, vol spikes ───────────────────────────────
    "breakout":                "VEGA",
    "liquidation_cascade":     "VEGA",
    "microstructure_pressure": "VEGA",
    "order_flow":              "VEGA",
    "vpin_flow":               "VEGA",
    "gamma_scalping":          "VEGA",
    "vol_surface_arb":         "VEGA",

    # ── BULL: directional long, trending ────────────────────────────────
    "momentum":                "BULL",
    "tsmom":                   "BULL",
    "hp_trend":                "BULL",
    "pullback_confluence":     "BULL",
    "knn_predictor":           "BULL",
    "contrarian_oi":           "BULL",
    "cross_sectional_momentum": "BULL",
    "momentum_carry_combo":    "BULL",
    "quant_factors":           "BULL",

    # ── BEAR: short-biased / bear market capture ─────────────────────────
    # Currently no explicit bear-only strategies in the registry; the routing
    # plan can still route symbols here via movement signals.
}

_OPPOSING: dict[str, str] = {
    "THETA": "VEGA",
    "VEGA":  "THETA",
    "BULL":  "BEAR",
    "BEAR":  "BULL",
}


def get_bucket(strategy_name: str) -> str:
    """Return macro bucket for a strategy name, defaulting to THETA."""
    return STRATEGY_BUCKETS.get(strategy_name, "THETA")


def opposing_bucket(bucket: str) -> str:
    return _OPPOSING.get(bucket, "THETA")


def build_ai_targets_from_bucket(
    dominant_bucket: str, symbols: list[str]
) -> dict[str, list[str]]:
    """
    Build ai_targets for signal_fusion: put all symbols in the dominant
    macro bucket so the router knows the MegaNet favours that regime.
    """
    targets: dict[str, list[str]] = {"THETA": [], "VEGA": [], "BULL": [], "BEAR": []}
    targets[dominant_bucket] = list(symbols)
    return targets
