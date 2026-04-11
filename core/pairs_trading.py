import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(f"strategy.{__name__}")

MODEL_PATH = Path(__file__).resolve().parent.parent / "config" / "hmm_macro_model.pkl"
ALPHA_MODEL_PATH = Path(__file__).resolve().parent.parent / "config" / "correlation_alpha_model.pkl"


def _load_hmm_pairs(model_path: Path = MODEL_PATH) -> list[dict]:
    try:
        payload = joblib.load(model_path)
    except Exception as exc:
        logger.debug("Could not load HMM macro model for pairs: %s", exc)
        return []

    ticker_patterns = payload.get("ticker_patterns") or {}
    top_pairs = ticker_patterns.get("top_positive_pairs") or []
    if not isinstance(top_pairs, list):
        return []
    return top_pairs


def _load_pair_alpha_priors(model_path: Path = ALPHA_MODEL_PATH) -> dict[str, dict]:
    try:
        payload = joblib.load(model_path)
    except Exception:
        return {}
    pairs = payload.get("pairs") or []
    out = {}
    for p in pairs:
        key = str(p.get("pair", "")).upper()
        if key:
            out[key] = p
    return out


def _compute_spread_zscore(price_a: pd.Series, price_b: pd.Series, lookback: int) -> float | None:
    merged = pd.concat([price_a, price_b], axis=1).dropna()
    if len(merged) < max(lookback, 60):
        return None

    log_a = np.log(merged.iloc[:, 0].astype(float).clip(lower=1e-6))
    log_b = np.log(merged.iloc[:, 1].astype(float).clip(lower=1e-6))
    spread = log_a - log_b

    rolling = spread.iloc[-lookback:]
    mean = float(rolling.mean())
    std = float(rolling.std(ddof=0))
    if std <= 1e-8:
        return None
    return float((spread.iloc[-1] - mean) / std)


def _mean_reversion_strength(price_a: pd.Series, price_b: pd.Series) -> tuple[float, float]:
    """Returns (reversion_speed, hedge_beta). Higher reversion_speed is better."""
    merged = pd.concat([price_a, price_b], axis=1).dropna()
    if len(merged) < 120:
        return 0.0, 1.0

    y = np.log(merged.iloc[:, 0].astype(float).clip(lower=1e-6))
    x = np.log(merged.iloc[:, 1].astype(float).clip(lower=1e-6))
    var_x = float(np.var(x))
    if var_x <= 1e-8:
        beta = 1.0
    else:
        beta = float(np.cov(y, x)[0, 1] / var_x)

    spread = y - beta * x
    lagged = spread.shift(1).dropna()
    aligned = spread.loc[lagged.index]
    if len(lagged) < 90:
        return 0.0, beta

    var_lag = float(np.var(lagged))
    if var_lag <= 1e-8:
        return 0.0, beta

    phi = float(np.cov(aligned, lagged)[0, 1] / var_lag)
    # AR(1) mean reversion speed proxy
    reversion_speed = max(0.0, min(1.0, 1.0 - phi))
    return reversion_speed, beta


def generate_pairs_trading_signals(
    allowed_symbols: list[str],
    max_signals: int = 6,
    entry_zscore: float = 1.5,
    lookback_days: int = 120,
    min_pair_corr: float = 0.65,
    min_confidence: float = 0.55,
) -> dict:
    if not allowed_symbols:
        return {"bull_symbols": [], "bear_symbols": [], "signals": []}

    allowed = {s.upper() for s in allowed_symbols}
    pairs = _load_hmm_pairs()
    pair_priors = _load_pair_alpha_priors()
    if not pairs:
        return {"bull_symbols": [], "bear_symbols": [], "signals": []}

    eligible_pairs = []
    for item in pairs:
        pair_str = str(item.get("pair", ""))
        corr = float(item.get("correlation", 0.0))
        if corr < min_pair_corr or "/" not in pair_str:
            continue
        left, right = pair_str.split("/", 1)
        left, right = left.strip().upper(), right.strip().upper()
        if left in allowed and right in allowed:
            eligible_pairs.append((left, right, corr))

    if not eligible_pairs:
        return {"bull_symbols": [], "bear_symbols": [], "signals": []}

    uniq_symbols = sorted({s for a, b, _ in eligible_pairs for s in (a, b)})
    history = yf.download(uniq_symbols, period="3y", progress=False)["Close"]
    if history is None or history.empty:
        return {"bull_symbols": [], "bear_symbols": [], "signals": []}

    if isinstance(history, pd.Series):
        history = history.to_frame()

    signals = []
    bull, bear = [], []
    for left, right, corr in eligible_pairs:
        if left not in history.columns or right not in history.columns:
            continue

        zscore = _compute_spread_zscore(history[left], history[right], lookback=lookback_days)
        if zscore is None or abs(zscore) < entry_zscore:
            continue
        reversion_speed, hedge_beta = _mean_reversion_strength(history[left], history[right])
        divergence_strength = min(1.0, abs(zscore) / max(entry_zscore * 2.0, 1e-6))
        corr_strength = min(1.0, max(0.0, corr))
        prior = pair_priors.get(f"{left}/{right}") or pair_priors.get(f"{right}/{left}") or {}
        prior_win = float(prior.get("win_rate", 0.5))
        prior_sharpe = float(prior.get("sharpe_like", 0.0))
        prior_strength = max(0.0, min(1.0, ((prior_win - 0.5) * 2.0) * 0.7 + (max(prior_sharpe, 0.0) / 2.0) * 0.3))
        confidence = (0.40 * divergence_strength) + (0.25 * corr_strength) + (0.20 * reversion_speed) + (0.15 * prior_strength)
        if confidence < min_confidence:
            continue

        if zscore >= entry_zscore:
            # left rich vs right: mean-reversion expects left down/right up
            short_sym, long_sym = left, right
        else:
            short_sym, long_sym = right, left

        bear.append(short_sym)
        bull.append(long_sym)
        signals.append(
            {
                "pair": f"{left}/{right}",
                "correlation": corr,
                "zscore": zscore,
                "long": long_sym,
                "short": short_sym,
                "hedge_beta": hedge_beta,
                "reversion_speed": reversion_speed,
                "confidence": confidence,
                "prior_win_rate": prior_win,
                "prior_sharpe_like": prior_sharpe,
            }
        )

    if signals:
        signals = sorted(signals, key=lambda x: x["confidence"], reverse=True)[:max_signals]
        selected_longs = [s["long"] for s in signals]
        selected_shorts = [s["short"] for s in signals]
        logger.info(
            "🔁 HMM Pairs Signals: %d active | %s",
            len(signals),
            ", ".join(f"{s['pair']} z={s['zscore']:+.2f} c={s['confidence']:.2f}" for s in signals),
        )
        return {
            "bull_symbols": selected_longs,
            "bear_symbols": selected_shorts,
            "signals": signals,
        }

    return {"bull_symbols": [], "bear_symbols": [], "signals": []}
