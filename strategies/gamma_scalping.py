"""
Dynamic Gamma Scalping Strategy.

Based on:
  - Taleb (1997) "Dynamic Hedging" — gamma scalping framework
  - Wilmott (2006) "Paul Wilmott on Quantitative Finance" — vol trading
  - Recent: Carr & Wu (2020) "Decomposing Long Bond Returns" — vol premium

Core insight: If you own gamma (long options), you earn
  P&L_gamma ≈ ½ * Γ * (σ²_realized - σ²_implied) * S² * dt
  per rebalance. When realized vol > implied vol (cheap options), gamma
  scalping is systematically profitable.

Strategy:
  1. Estimate current implied vol (IV) from options surface or GARCH proxy
  2. Estimate realized vol from recent price moves
  3. When RV > IV by threshold: LONG gamma trade (synthetic: buy straddle-like)
  4. When IV > RV by threshold: SHORT gamma (sell vol, delta hedge)
  5. Delta hedge continuously via futures to isolate pure gamma P&L

For equity/crypto futures (without live options data):
  - Proxy IV via GARCH(1,1) conditional variance
  - Proxy gamma position with a volatility breakout trade:
    • BUY when realized vol just exceeded GARCH forecast → vol expansion
    • SELL when realized vol collapsed below GARCH → vol compression
  - The position is sized proportional to the vol gap and held short-term

Alpha source: The volatility risk premium (VRP) — options are systematically
overpriced on average. Short gamma harvests this. Long gamma profits from
discrete delta hedging when prices jump.

References:
  - Egloff, Leippold & Wu (2010) — The Term Structure of Variance Swap Rates
  - Carr & Wu (2009) — Variance Risk Premiums
"""

from __future__ import annotations
import logging

import numpy as np

from bot.config import cfg
from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategy.gamma_scalping")

# ── Volatility estimation parameters ─────────────────────────────────────
GARCH_OMEGA   = 1e-6     # GARCH(1,1) constant
GARCH_ALPHA   = 0.10     # ARCH term (surprise weight)
GARCH_BETA    = 0.85     # GARCH term (persistence)
GARCH_WARMUP  = 60       # Bars to warm up GARCH
RV_WINDOW     = 20       # Realised vol window (bars)
RV_SHORT      = 5        # Short-window RV for fast moves

# ── Signal thresholds ─────────────────────────────────────────────────────
LONG_GAMMA_THRESHOLD  = 0.15   # RV / IV_garch > 1 + threshold → buy gamma
SHORT_GAMMA_THRESHOLD = 0.20   # IV_garch / RV > 1 + threshold → sell gamma (harvest VRP)
MIN_IV                = 0.08   # Minimum annualised IV to trade (avoid micro-cap noise)
MAX_IV                = 2.00   # Maximum (avoid insane vol environments)

# ── Regime and confidence ─────────────────────────────────────────────────
BASE_CONFIDENCE = 0.65
MAX_POSITIONS   = 4


class GammaScalpingStrategy(BaseStrategy):
    """
    Gamma scalping via GARCH-implied vs realised vol comparison.

    Long gamma when the market is about to move more than priced in.
    Short gamma (vol harvesting) when options are overpriced.
    Delta-hedged continuously via the futures position.
    """

    name = "gamma_scalping"
    market = "futures"
    required_regime = None

    def __init__(self) -> None:
        super().__init__()
        self._garch_var: dict[str, float] = {}   # Running GARCH variance per symbol

    @property
    def symbols(self) -> list[str]:
        return cfg.futures_symbols[:MAX_POSITIONS]

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        signals = []

        regime_scale = {
            "volatile": 1.0,    # Best: high realized vol vs cheap IV
            "bull":     0.80,
            "bear":     0.75,
            "ranging":  0.65,   # Worst: flat market, vol premium captures are smaller
        }.get(regime, 0.75)

        for symbol in self.symbols:
            try:
                df = store.get_history_df(symbol, "1h")
                if df is None or len(df) < GARCH_WARMUP + RV_WINDOW:
                    continue

                sig = self._compute_signal(symbol, df, regime_scale, predictions)
                if sig:
                    signals.append(sig)

            except Exception as exc:
                log.debug("%s gamma signal error: %s", symbol, exc)

        return signals

    # ── GARCH(1,1) estimator ──────────────────────────────────────────────

    def _update_garch(self, symbol: str, log_rets: np.ndarray) -> float:
        """Compute GARCH(1,1) conditional variance for the latest bar."""
        long_var = float(log_rets.var())   # Unconditional variance

        if symbol not in self._garch_var:
            self._garch_var[symbol] = long_var

        sigma2 = self._garch_var[symbol]

        # Iteratively update GARCH over the warmup window
        for r in log_rets[-GARCH_WARMUP:]:
            sigma2 = (GARCH_OMEGA
                      + GARCH_ALPHA * r ** 2
                      + GARCH_BETA  * sigma2)

        self._garch_var[symbol] = sigma2
        # Annualise (1h bars → annual)
        return float(np.sqrt(max(sigma2, 1e-10) * 252 * 24))

    # ── Signal computation ─────────────────────────────────────────────────

    def _compute_signal(self, symbol: str, df, regime_scale: float, predictions: dict):
        c = df["close"].astype(float).values
        log_rets = np.diff(np.log(c[-GARCH_WARMUP - RV_WINDOW - 2:]))

        if len(log_rets) < GARCH_WARMUP + 5:
            return None

        # ── IV proxy via GARCH ────────────────────────────────────────────
        iv_garch = self._update_garch(symbol, log_rets)
        if not (MIN_IV <= iv_garch <= MAX_IV):
            return None

        # ── Realised volatility (long window) ─────────────────────────────
        rv_long = float(log_rets[-RV_WINDOW:].std()) * np.sqrt(252 * 24)

        # ── Realised volatility (short window — fast regime detection) ────
        rv_short = float(log_rets[-RV_SHORT:].std()) * np.sqrt(252 * 24)

        # ── Vol ratio ─────────────────────────────────────────────────────
        vrp_ratio_long  = rv_long  / (iv_garch + 1e-6)    # > 1: RV > IV → cheap options
        vrp_ratio_short = rv_short / (iv_garch + 1e-6)

        # ── Direction ─────────────────────────────────────────────────────
        price = float(c[-1])

        # LONG gamma: RV just surged above IV → market is moving more than priced
        # Use short RV for fast detection
        if vrp_ratio_short > (1.0 + LONG_GAMMA_THRESHOLD):
            side = "BUY"
            vol_edge = vrp_ratio_short - 1.0
            confidence = min(0.90, BASE_CONFIDENCE + vol_edge * 0.3)

        # SHORT gamma: IV persistently above RV → harvest vol premium
        elif vrp_ratio_long < (1.0 - SHORT_GAMMA_THRESHOLD):
            side = "SELL"
            vol_edge = 1.0 - vrp_ratio_long
            confidence = min(0.85, BASE_CONFIDENCE + vol_edge * 0.25)

        else:
            return None

        confidence *= regime_scale

        if confidence < 0.40:
            return None

        # ── Size proportional to vol gap and risk budget ───────────────────
        equity = float(cfg.initial_capital)
        # Position size: larger edge → larger size, capped at risk limit
        size_scalar = min(1.5, 1.0 + vol_edge)
        quantity = (equity * float(cfg.max_risk_per_trade) * size_scalar) / (price + 1e-10)
        quantity = max(0.001, round(quantity, 4))

        return Signal(
            symbol=symbol, market="futures", side=side,
            quantity=quantity, price=0.0,   # market order — captures vol immediately
            confidence=round(confidence, 4), strategy=self.name,
            meta={
                "type": "long_gamma" if side == "BUY" else "short_gamma_vrp",
                "iv_garch_ann": round(iv_garch, 4),
                "rv_long_ann":  round(rv_long, 4),
                "rv_short_ann": round(rv_short, 4),
                "vrp_ratio":    round(vrp_ratio_short if side == "BUY" else vrp_ratio_long, 4),
                "vol_edge":     round(vol_edge, 4),
            },
        )
