from dataclasses import dataclass
from typing import Iterable

import numpy as np

from core.movement_predictor import MovementSignal


@dataclass
class PortfolioGreekTargets:
    target_delta: float
    target_theta: float
    target_vega: float
    target_gamma: float
    target_confidence: float
    movement_bias: str


def derive_portfolio_greek_targets(
    movement_signals: Iterable[MovementSignal],
    equity: float,
    vix_level: float,
) -> PortfolioGreekTargets:
    signals = list(movement_signals)
    if not signals:
        return PortfolioGreekTargets(0.0, 0.0, 0.0, 0.0, 0.0, "neutral")

    avg_prob_up = float(np.mean([s.probability_up for s in signals]))
    avg_expected_move = float(np.mean([s.expected_daily_move for s in signals]))

    confidence = abs(avg_prob_up - 0.5) * 2.0
    direction = 1.0 if avg_prob_up >= 0.5 else -1.0

    volatility_scale = max(0.35, min(1.5, 20.0 / max(vix_level, 10.0)))
    book_scale = max(1.0, equity / 100_000.0)

    target_delta = direction * confidence * 20.0 * book_scale * volatility_scale
    target_theta = (0.8 - confidence) * 50.0 * book_scale
    target_vega = (0.4 - confidence) * 30.0 * book_scale * (-1.0 if vix_level > 24 else 1.0)
    target_gamma = (abs(avg_expected_move) * 3000.0) * book_scale * (1.0 if confidence > 0.4 else -0.3)

    if avg_prob_up > 0.57:
        movement_bias = "bullish"
    elif avg_prob_up < 0.43:
        movement_bias = "bearish"
    else:
        movement_bias = "neutral"

    return PortfolioGreekTargets(
        target_delta=target_delta,
        target_theta=target_theta,
        target_vega=target_vega,
        target_gamma=target_gamma,
        target_confidence=confidence,
        movement_bias=movement_bias,
    )
