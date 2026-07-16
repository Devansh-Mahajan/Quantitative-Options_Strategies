"""
SignalRouter: bridges the live bot's ML stack with signal_fusion.

Two entry points:
  build_routing_plan(...)   — called once per 60-second cycle; returns RoutingPlan
  confidence_modifier(...)  — called per-signal to boost/dampen based on routing
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import numpy as np

from core.ml.signal_fusion import route_strategy_candidates, RoutingPlan
from core.ml.greeks_targeting import derive_portfolio_greek_targets
from strategies.buckets import build_ai_targets_from_bucket, get_bucket, opposing_bucket

if TYPE_CHECKING:
    from ml.neural_selector import NeuralSelector
    from ml.movement_scorer import MovementScorer
    from ml.alpha_cache import AlphaCache
    from strategies.base import Signal

log = logging.getLogger("core.ml.signal_router")

_MACRO_TO_BUCKET = {
    "THETA_ENGINE": "THETA",
    "VEGA_SNIPER":  "VEGA",
    "BULL_TREND":   "BULL",
    "BEAR_TREND":   "BEAR",
}


def build_routing_plan(
    store,
    neural_selector: NeuralSelector,
    movement_scorer: MovementScorer,
    alpha_cache: AlphaCache,
    symbols: list[str],
    equity: float,
    primary_symbol: str = "",
) -> RoutingPlan:
    """
    Build a RoutingPlan from the current ML state.  Called once per cycle.

    Pipeline:
      1. Run NeuralSelector on primary symbol's feature matrix → macro regime
      2. Score all symbols via MovementScorer (store-backed, no yfinance)
      3. Pull alpha signals from AlphaCache (weekly snapshot)
      4. Derive greek targets from movement signals + equity
      5. Route candidates via signal_fusion
    """
    # 1. Neural macro classification
    anchor = primary_symbol or (symbols[0] if symbols else "")
    neural_probs = _run_neural(store, neural_selector, anchor)
    macro_strategy = neural_selector.macro_strategy
    macro_confidence = neural_selector.macro_confidence
    dominant_bucket = _MACRO_TO_BUCKET.get(macro_strategy, "THETA")

    # 2. Movement signals (no I/O — uses already-cached store data)
    movement_signals = movement_scorer.score_all(store, symbols)

    # 3. Alpha signals (JSON cache, re-reads at most once per hour)
    alpha_map = alpha_cache.load(symbols)

    # 4. Greek targets (required by signal_fusion — never None)
    greek_targets = derive_portfolio_greek_targets(movement_signals, equity, vix_level=20.0)

    # 5. ai_targets: symbols the MegaNet thinks belong in dominant bucket
    ai_targets = build_ai_targets_from_bucket(dominant_bucket, symbols)

    plan = route_strategy_candidates(
        allowed_symbols=symbols,
        ai_targets=ai_targets,
        movement_signals=movement_signals,
        flow_map=None,
        pair_overlay=None,
        greek_targets=greek_targets,
        macro_strategy=macro_strategy,
        macro_confidence=macro_confidence,
        alpha_signals=alpha_map if alpha_map else None,
        top_k=min(12, max(1, len(symbols))),
    )

    log.info(
        "RoutingPlan macro=%s conf=%.2f consensus=%.2f deploy=%.2f "
        "THETA=%d VEGA=%d BULL=%d BEAR=%d",
        macro_strategy,
        macro_confidence,
        plan.consensus_score,
        plan.deployment_multiplier,
        len(plan.theta_candidates),
        len(plan.vega_candidates),
        len(plan.bull_candidates),
        len(plan.bear_candidates),
    )
    return plan


def confidence_modifier(
    signal: Signal,
    routing_plan: RoutingPlan,
) -> float:
    """
    Return a float multiplier applied to signal.confidence before RL scaling.

    Logic:
      - symbol in matching bucket  → 1.25 × deployment_multiplier
      - symbol in opposing bucket  → 0.50 × deployment_multiplier
      - symbol in neither bucket   → 0.85 × deployment_multiplier
    """
    symbol = signal.symbol.upper()
    strat_bucket = get_bucket(signal.strategy)

    bucket_sets = {
        "THETA": set(routing_plan.theta_candidates),
        "VEGA":  set(routing_plan.vega_candidates),
        "BULL":  set(routing_plan.bull_candidates),
        "BEAR":  set(routing_plan.bear_candidates),
    }

    if symbol in bucket_sets.get(strat_bucket, set()):
        base = 1.25
    elif symbol in bucket_sets.get(opposing_bucket(strat_bucket), set()):
        base = 0.50
    else:
        base = 0.85

    return float(base * routing_plan.deployment_multiplier)


# ─────────────────────────────────────────────────────────────────────────── #
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────── #

def _run_neural(store, neural_selector: NeuralSelector, symbol: str) -> dict[str, float]:
    if not symbol:
        return {k: 0.25 for k in ["THETA", "VEGA", "BULL", "BEAR"]}
    try:
        df = store.get_history_df(symbol, "1h")
        if len(df) < 30:
            return {k: 0.25 for k in ["THETA", "VEGA", "BULL", "BEAR"]}
        from ml.features import compute_feature_matrix
        X = compute_feature_matrix(
            df,
            funding_rate=getattr(store, "funding", {}).get(symbol, 0.0),
            open_interest=getattr(store, "open_interest", {}).get(symbol, 0.0),
        )
        seq_len = min(64, len(X))
        return neural_selector.predict(X[-seq_len:])
    except Exception as exc:
        log.debug("_run_neural failed for %s: %s", symbol, exc)
        return {k: 0.25 for k in ["THETA", "VEGA", "BULL", "BEAR"]}
