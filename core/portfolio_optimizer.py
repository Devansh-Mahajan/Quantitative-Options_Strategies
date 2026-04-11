import math


def clipped_kelly_fraction(edge: float, win_prob: float, loss_multiple: float = 1.0) -> float:
    """Conservative Kelly fraction with hard clipping.

    edge: expected fractional return per trade (e.g. 0.02)
    win_prob: probability of win [0,1]
    loss_multiple: avg loss / avg win
    """
    p = max(0.0, min(1.0, win_prob))
    q = 1.0 - p
    b = max(loss_multiple, 1e-6)
    raw = p - (q / b)
    # Blend with edge to avoid over-sizing on noisy p estimates.
    return max(0.0, min(1.0, 0.5 * raw + 0.5 * max(edge, 0.0)))


def recommend_deployment_fraction(
    signal_confidence: float,
    macro_confidence: float,
    pair_confidence: float,
    vix_level: float,
    target_daily_return: float,
    max_kelly_fraction: float,
) -> float:
    """Return conservative deployment fraction in [0.05, 1.0]."""
    signal_conf = max(0.0, min(1.0, signal_confidence))
    macro_conf = max(0.0, min(1.0, macro_confidence))
    pair_conf = max(0.0, min(1.0, pair_confidence))

    blended_conf = (0.45 * signal_conf) + (0.35 * macro_conf) + (0.20 * pair_conf)
    base_edge = min(0.10, max(0.0, target_daily_return) * (0.7 + 0.6 * blended_conf))

    # Higher VIX lowers usable edge quality for short-premium systems.
    vix_penalty = 1.0 if vix_level < 20 else 0.85 if vix_level < 28 else 0.65

    kelly = clipped_kelly_fraction(edge=base_edge, win_prob=blended_conf, loss_multiple=1.1)
    sized = min(max_kelly_fraction, kelly) * vix_penalty

    return max(0.05, min(1.0, sized))


def estimate_pair_overlay_confidence(signals: list[dict]) -> float:
    if not signals:
        return 0.0
    vals = [max(0.0, min(1.0, float(s.get("confidence", 0.0)))) for s in signals]
    return float(sum(vals) / max(len(vals), 1))
