"""
Avellaneda-Stoikov Optimal Market Making Strategy.

Based on: Avellaneda & Stoikov (2008) "High-frequency trading in a limit order book"
          SSRN: http://ssrn.com/abstract=1086061
Extended with: Guéant, Lehalle & Fernandez-Tapia (2013) inventory control extensions.

Core insight: Market makers earn the bid-ask spread by quoting simultaneously on both
sides. The key is optimally managing inventory risk. The AS model gives closed-form
optimal quotes based on current inventory, volatility, and risk aversion.

Key formulas:
  Reservation price:  r = s - q * γ * σ² * T
  Optimal spread:     δ = γ * σ² * T + (2/γ) * ln(1 + γ/κ)
  Bid quote:          r - δ/2
  Ask quote:          r + δ/2

Where:
  s = mid price
  q = signed inventory (+ = long, - = short)
  γ = risk aversion (tune: 0.01 low, 0.1 high)
  σ = instantaneous volatility
  T = time horizon (normalised)
  κ = order arrival rate (estimated from recent volume)

Alpha source: Earned spread minus inventory risk costs. Profitable when:
  - Spread earned > adverse selection cost
  - Inventory turnover is high
  - Volatility is moderate (not extreme)

Regime filter: Avoid during very high volatility (liquidation cascades) and
trending markets (adverse selection dominates).
"""

from __future__ import annotations
import logging
import math

import numpy as np

from bot.config import cfg
from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategy.market_making")

# ── Avellaneda-Stoikov parameters ─────────────────────────────────────────
GAMMA          = 0.05     # Risk aversion (higher = tighter quotes, more inventory control)
KAPPA          = 1.5      # Order arrival rate (higher = tighter quotes)
T_HORIZON      = 1.0      # Time horizon normalisation (bars)
VOL_WINDOW     = 20       # Bars for σ estimation
SPREAD_FLOOR   = 0.0005   # 5 bps minimum spread each side
SPREAD_CAP     = 0.003    # 30 bps maximum spread each side (avoid quoting in thin markets)
MAX_INVENTORY  = 3        # Max open positions before bias kicks in fully
INVENTORY_BIAS = 0.6      # How much to skew quotes when inventory is at limit

# ── Signal confidence tuning ──────────────────────────────────────────────
BASE_CONFIDENCE   = 0.62
TREND_PENALTY     = 0.15  # Reduce confidence in strong trending markets (adverse selection)
VOL_SPIKE_PENALTY = 0.20  # Reduce confidence during vol spikes


class MarketMakingStrategy(BaseStrategy):
    """
    Avellaneda-Stoikov optimal market making.

    Generates limit order BUY signals at the optimal bid (below mid) and
    SELL signals at the optimal ask (above mid). Inventory imbalance shifts
    the reservation price to mean-revert towards zero inventory.

    Best in: ranging, low-trend markets with moderate volume.
    Avoid:   strong trends (adverse selection eats the spread).
    """

    name = "market_making"
    market = "futures"
    required_regime = None   # active in all regimes, dampened in trending

    @property
    def symbols(self) -> list[str]:
        # Focus on the most liquid symbols where spread capture is viable
        return cfg.futures_symbols[:6]

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        signals = []

        # Dampening by regime: in trending markets adverse selection dominates
        regime_scale = {
            "ranging":  1.00,
            "volatile": 0.70,
            "bull":     0.55,
            "bear":     0.55,
        }.get(regime, 0.75)

        if regime_scale < 0.40:
            return signals

        for symbol in self.symbols:
            try:
                df = store.get_history_df(symbol, "5m")
                if df is None or len(df) < VOL_WINDOW + 5:
                    continue

                sig = self._compute_signal(store, symbol, df, regime_scale, predictions)
                if sig:
                    signals.extend(sig)

            except Exception as exc:
                log.debug("%s signal error: %s", symbol, exc)

        return signals

    # ── Core AS model ─────────────────────────────────────────────────────

    def _compute_signal(self, store, symbol: str, df, regime_scale: float, predictions: dict) -> list[Signal]:
        c = df["close"].astype(float).values
        v = df["volume"].astype(float).values
        h = df["high"].astype(float).values
        lo = df["low"].astype(float).values

        mid = float(c[-1])
        if mid <= 0:
            return []

        # ── σ: realised volatility (annualised log-returns std) ───────────
        log_rets = np.diff(np.log(c[-VOL_WINDOW - 1:]))
        sigma = float(log_rets.std()) * math.sqrt(252 * 24 * 12)   # 5-min bars → annual
        if sigma < 1e-6:
            return []

        # ── κ: estimate order arrival rate from volume rhythm ──────────────
        # Higher recent volume / average = more arrivals = tighter optimal spread
        recent_vol = float(v[-5:].mean())
        avg_vol = float(v[-VOL_WINDOW:].mean()) + 1e-10
        kappa = KAPPA * max(0.5, min(2.0, recent_vol / avg_vol))

        # ── Trend strength filter (ADX proxy) ─────────────────────────────
        # Strong trend = high adverse selection cost = reduce quotes
        tr = np.maximum(h[-14:] - lo[-14:],
             np.maximum(np.abs(h[-14:] - c[-15:-1]), np.abs(lo[-14:] - c[-15:-1])))
        adx_proxy = float(tr.mean()) / (mid + 1e-10)
        trend_confidence_mult = max(0.0, 1.0 - adx_proxy * 50)  # scale to ~0-1

        # ── Vol spike filter ───────────────────────────────────────────────
        recent_sigma = float(log_rets[-5:].std()) * math.sqrt(252 * 24 * 12)
        vol_ratio = recent_sigma / (sigma + 1e-10)
        if vol_ratio > 2.5:
            return []  # Vol spike — adverse selection explodes, do not quote

        # ── Inventory proxy (count open positions for this symbol) ─────────
        # In live trading this comes from the position manager.
        # Here we use predictions["inventory_{symbol}"] if available, else 0.
        inventory = float(predictions.get(f"inventory_{symbol}", 0))
        inv_frac = max(-1.0, min(1.0, inventory / (MAX_INVENTORY + 1e-10)))

        # ── Avellaneda-Stoikov reservation price ──────────────────────────
        # r = s - q * γ * σ² * T
        reservation = mid - inv_frac * GAMMA * (sigma ** 2) * T_HORIZON

        # ── Optimal half-spread ───────────────────────────────────────────
        # δ/2 = (γ * σ² * T) / 2 + (1/γ) * ln(1 + γ/κ)
        half_spread = (GAMMA * sigma ** 2 * T_HORIZON) / 2.0 + (1.0 / GAMMA) * math.log(1.0 + GAMMA / kappa)
        half_spread = max(SPREAD_FLOOR, min(SPREAD_CAP, half_spread))

        # ── Optimal bid / ask ─────────────────────────────────────────────
        bid_price = reservation - half_spread
        ask_price = reservation + half_spread

        # Sanity checks
        if bid_price <= 0 or ask_price <= bid_price:
            return []

        # ── Confidence ────────────────────────────────────────────────────
        spread_quality = min(1.0, half_spread / (SPREAD_FLOOR + 1e-10) * 0.1)
        confidence = (
            BASE_CONFIDENCE
            * regime_scale
            * trend_confidence_mult
            * (1.0 - TREND_PENALTY * adx_proxy * 20)
            * (1.0 - VOL_SPIKE_PENALTY * max(0, vol_ratio - 1.0))
        )
        confidence = max(0.0, min(1.0, confidence))

        if confidence < 0.35:
            return []

        # ── Position sizing: inverse volatility, scaled by half-spread ─────
        equity = float(cfg.initial_capital)
        risk_per_trade = equity * float(cfg.max_risk_per_trade) * 0.5  # halved for MM
        quantity = risk_per_trade / (mid * half_spread * 100 + 1e-10)
        quantity = max(0.001, round(quantity, 4))

        signals_out = []

        # Generate BID signal (buy at optimal bid — below mid) when not over-long
        if inv_frac < INVENTORY_BIAS:
            signals_out.append(Signal(
                symbol=symbol, market="futures", side="BUY",
                quantity=quantity, price=round(bid_price, 6),
                confidence=confidence, strategy=self.name,
                meta={
                    "type": "market_make_bid",
                    "mid": mid, "reservation": round(reservation, 6),
                    "half_spread_bps": round(half_spread * 10000, 2),
                    "sigma_ann": round(sigma, 4),
                    "inventory": inventory,
                },
            ))

        # Generate ASK signal (sell at optimal ask — above mid) when not over-short
        if inv_frac > -INVENTORY_BIAS:
            signals_out.append(Signal(
                symbol=symbol, market="futures", side="SELL",
                quantity=quantity, price=round(ask_price, 6),
                confidence=confidence, strategy=self.name,
                meta={
                    "type": "market_make_ask",
                    "mid": mid, "reservation": round(reservation, 6),
                    "half_spread_bps": round(half_spread * 10000, 2),
                    "sigma_ann": round(sigma, 4),
                    "inventory": inventory,
                },
            ))

        return signals_out
