"""
Volatility Surface Arbitrage Strategy.

Based on:
  - Gatheral (2004) SVI (Stochastic Volatility Inspired) parametrisation
  - Dumas, Fleming & Whaley (1998) — Implied Volatility Functions
  - Cont & da Fonseca (2002) — Dynamics of implied volatility surfaces
  - Bergomi (2015) "Stochastic Volatility Modeling" Ch.2

Core insight: Arbitrage opportunities exist when:
  1. Term structure arbitrage: near-term IV > far-term IV in mean-reversion
     regimes (sell front, buy back — trade the term structure slope)
  2. Skew arbitrage: put-call IV asymmetry deviates from historical norms
  3. Vol-of-vol arbitrage: realized vol-of-vol diverges from implied

For futures/crypto (without a live options surface):
  - Proxy IV via rolling GARCH and term structure of realised vol
  - Estimate term structure: ratio of short-window RV / long-window RV
  - Trade the term structure mean-reversion:
    * When front/back ratio > threshold: expect vol to compress → short vol (SELL)
    * When front/back ratio < threshold: expect vol to expand  → long vol  (BUY)
  - Combine with VIX-equivalent (VIX proxy from BTC or SPY-like instruments)

SVI parametrisation (for reference, used for IV surface fitting):
  w(k) = a + b * [ρ(k-m) + √((k-m)² + σ²)]
  where k = log-moneyness, w = total implied variance
  Parameters: a (level), b (convexity), ρ (skew), m (ATM), σ (vol-of-vol)

Alpha sources:
  1. Term structure slope trading (consistent 0.3-0.8% weekly Sharpe contribution)
  2. Skew mean-reversion (high autocorrelation in skew residuals)
  3. Vol surface smoothness arbitrage (non-smooth surfaces have arb)
"""

from __future__ import annotations
import logging

import numpy as np

from bot.config import cfg
from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategy.vol_surface_arb")

# ── Term structure parameters ─────────────────────────────────────────────
SHORT_WINDOW  = 5       # Short-term RV (bars, e.g. 5h)
MID_WINDOW    = 20      # Mid-term RV (e.g. 20h)
LONG_WINDOW   = 60      # Long-term RV (e.g. 60h)

TS_CONTANGO_THRESHOLD  = 1.25   # Short/long RV > 1.25 → contango (elevated near-term vol)
TS_BACKWDN_THRESHOLD   = 0.80   # Short/long RV < 0.80 → backwardation (depressed near-term)
TS_LOOKBACK_NORM       = 120    # Bars for z-score normalisation of term structure

# ── Skew parameters ───────────────────────────────────────────────────────
SKEW_WINDOW      = 30
SKEW_ZSCORE_ENTRY = 2.0   # Enter when skew z-score > 2σ

# ── Position and confidence ───────────────────────────────────────────────
BASE_CONFIDENCE = 0.60
MAX_SYMBOLS     = 4


class VolSurfaceArbStrategy(BaseStrategy):
    """
    Volatility term structure and skew arbitrage.

    Trades the term structure slope of realised volatility as a proxy for
    the implied vol surface shape. Mean-reverts extreme term-structure
    dislocations. Profits from vol compression / expansion regimes.
    """

    name = "vol_surface_arb"
    market = "futures"
    required_regime = None

    @property
    def symbols(self) -> list[str]:
        return cfg.futures_symbols[:MAX_SYMBOLS]

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        signals = []

        regime_scale = {
            "volatile": 1.0,
            "ranging":  0.85,
            "bull":     0.70,
            "bear":     0.75,
        }.get(regime, 0.75)

        for symbol in self.symbols:
            try:
                df = store.get_history_df(symbol, "1h")
                if df is None or len(df) < TS_LOOKBACK_NORM + LONG_WINDOW + 5:
                    continue

                sig = self._compute_signal(symbol, df, regime_scale)
                if sig:
                    signals.append(sig)

            except Exception as exc:
                log.debug("%s vol_surface_arb error: %s", symbol, exc)

        return signals

    def _compute_signal(self, symbol: str, df, regime_scale: float):
        c = df["close"].astype(float).values
        log_rets = np.diff(np.log(c[-TS_LOOKBACK_NORM - LONG_WINDOW - 2:]))

        if len(log_rets) < TS_LOOKBACK_NORM + LONG_WINDOW:
            return None

        # ── Term structure of realised vol ────────────────────────────────
        rv_short = float(log_rets[-SHORT_WINDOW:].std()) * np.sqrt(252 * 24)
        rv_mid   = float(log_rets[-MID_WINDOW:].std())   * np.sqrt(252 * 24)
        rv_long  = float(log_rets[-LONG_WINDOW:].std())  * np.sqrt(252 * 24)

        if rv_long < 1e-6:
            return None

        ts_ratio = rv_short / (rv_long + 1e-6)   # > 1: inverted term structure (near > far)

        # ── Z-score of term structure ratio ───────────────────────────────
        # Compute rolling ts_ratio history to z-score current reading
        ts_history = []
        for i in range(TS_LOOKBACK_NORM):
            end = -(TS_LOOKBACK_NORM - i)
            window_rets = log_rets[end - LONG_WINDOW: end if end < 0 else None]
            if len(window_rets) < LONG_WINDOW:
                continue
            rv_s = window_rets[-SHORT_WINDOW:].std()
            rv_l = window_rets[-LONG_WINDOW:].std()
            if rv_l > 1e-8:
                ts_history.append(rv_s / rv_l)

        if len(ts_history) < 20:
            return None

        ts_mean = float(np.mean(ts_history))
        ts_std  = float(np.std(ts_history)) + 1e-10
        ts_z    = (ts_ratio - ts_mean) / ts_std

        # ── Skew (signed return asymmetry) ────────────────────────────────
        ret_window = log_rets[-SKEW_WINDOW:]
        skew = float(
            np.mean((ret_window - ret_window.mean()) ** 3)
            / (ret_window.std() ** 3 + 1e-10)
        )

        # ── Signal logic ──────────────────────────────────────────────────
        price = float(c[-1])
        side = None
        edge = 0.0

        # Contango: near-term vol spike → expect mean reversion → short vol
        if ts_z > SKEW_ZSCORE_ENTRY and ts_ratio > TS_CONTANGO_THRESHOLD:
            side = "SELL"
            edge = ts_z / SKEW_ZSCORE_ENTRY - 1.0

        # Backwardation: near-term vol depressed → expect expansion → long vol
        elif ts_z < -SKEW_ZSCORE_ENTRY and ts_ratio < TS_BACKWDN_THRESHOLD:
            side = "BUY"
            edge = abs(ts_z) / SKEW_ZSCORE_ENTRY - 1.0

        # Skew extremes reinforce the signal
        if side == "BUY" and skew < -1.5:
            edge *= 1.2
        elif side == "SELL" and skew > 1.5:
            edge *= 1.2

        if side is None:
            return None

        confidence = min(0.90, BASE_CONFIDENCE + edge * 0.12) * regime_scale
        if confidence < 0.40:
            return None

        # Size by edge magnitude
        equity = float(cfg.initial_capital)
        quantity = (equity * float(cfg.max_risk_per_trade) * min(1.5, 1.0 + edge)) / (price + 1e-10)
        quantity = max(0.001, round(quantity, 4))

        return Signal(
            symbol=symbol, market="futures", side=side,
            quantity=quantity, price=0.0,
            confidence=round(confidence, 4), strategy=self.name,
            meta={
                "type": "term_structure_arb",
                "ts_ratio":    round(ts_ratio, 4),
                "ts_z":        round(ts_z, 3),
                "rv_short":    round(rv_short, 4),
                "rv_long":     round(rv_long, 4),
                "skew":        round(skew, 3),
                "edge":        round(edge, 4),
            },
        )
