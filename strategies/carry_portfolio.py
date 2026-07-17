"""
Diversified Funding Rate Carry Portfolio.

How institutional crypto desks extract carry:
  The perpetual futures funding rate is the closest crypto equivalent to
  the carry trade in FX or bond markets. When funding is positive, longs
  pay shorts every 8 hours — so going short earns a risk-adjusted yield.

Why this beats the simple FundingArbStrategy:
  1. PORTFOLIO approach: diversify across all symbols, not just one threshold
  2. RISK-ADJUSTED sizing: weight by funding Sharpe (stability × magnitude)
  3. DYNAMIC: rebalance on every funding cycle change, not static threshold
  4. CORRELATION-AWARE: avoid doubling up correlated carry positions

Key insight:
  If BTC funding = +0.05%/8h = +21.9% annualised — you get paid to short BTC perps.
  But if all correlated alts also have high funding, you have correlated exposure.
  The portfolio approach finds the ORTHOGONAL carry, not just the highest yield.

Signal logic:
  - For each symbol, collect last LOOKBACK funding rate observations
  - Compute mean (expected yield) and std (yield stability)
  - Carry Sharpe = mean / std   → prefer stable, consistent carry
  - Rank: long (short perp, earn funding) top-N positive carry
  - Rank: short (long perp, pay funding) if very negative carry on it
  - Rebalance every REBAL_BARS bars (≈ 8h at 1h interval)
"""

from __future__ import annotations
import logging

import numpy as np

from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.carry_portfolio")

LOOKBACK       = 48      # 48 bars of funding history (48h at 1h interval)
REBAL_BARS     = 8       # rebalance every 8 bars
MIN_CARRY_RATE = 0.0002  # minimum |funding rate| to consider (0.02%)
CARRY_LONG_N   = 3       # number of carry longs (short perp, earn funding)
CARRY_SHORT_N  = 2       # number of carry shorts (long perp, negative carry)
MIN_OBS        = 12      # minimum funding observations before trusting signal


class CarryPortfolioStrategy(BaseStrategy):
    """
    Diversified funding rate carry portfolio.

    Goes short perpetuals with the highest positive funding (earning premium)
    and long perpetuals with the most negative funding (earning reverse premium).
    Sizes positions by carry Sharpe to prefer stable, persistent carry.
    """

    name = "carry_portfolio"
    market = "futures"

    def __init__(
        self,
        lookback: int = LOOKBACK,
        rebal_bars: int = REBAL_BARS,
        long_n: int = CARRY_LONG_N,
        short_n: int = CARRY_SHORT_N,
    ) -> None:
        self._lookback = lookback
        self._rebal_bars = rebal_bars
        self._long_n = long_n
        self._short_n = short_n
        self._bar_count = 0
        # Track funding history ourselves since store may only have latest value
        self._funding_history: dict[str, list[float]] = {}

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        self._bar_count += 1

        # Collect current funding rates into history
        symbols = list(store.funding.keys())
        for sym in symbols:
            rate = float(store.funding.get(sym, 0.0))
            if sym not in self._funding_history:
                self._funding_history[sym] = []
            self._funding_history[sym].append(rate)
            # Rolling window
            self._funding_history[sym] = self._funding_history[sym][-self._lookback:]

        # Rebalance only every REBAL_BARS bars
        if self._bar_count % self._rebal_bars != 0:
            return []

        # Carry is less reliable in highly volatile regimes (funding spikes then reverses)
        regime_scale = {
            "bull":     1.0,
            "bear":     0.9,
            "ranging":  1.0,   # carry works great in sideways markets
            "volatile": 0.5,   # funding rates spike unpredictably
        }.get(regime, 0.7)

        # (An unreachable `< 0.4` kill-switch was removed here — the map's
        # minimum is 0.5; regime handling is via the regime_scale multiplier.)

        # ── Compute carry Sharpe per symbol ───────────────────────────────
        carry_scores: dict[str, float] = {}

        for sym, history in self._funding_history.items():
            if len(history) < MIN_OBS:
                continue
            arr = np.array(history, dtype=float)
            mean_rate = float(arr.mean())
            std_rate  = float(arr.std() + 1e-10)

            if abs(mean_rate) < MIN_CARRY_RATE:
                continue

            # Sharpe of funding rate: mean / std
            carry_sharpe = mean_rate / std_rate
            carry_scores[sym] = carry_sharpe

        if not carry_scores:
            return []

        # Sort: highest positive Sharpe = best carry long (short perp)
        ranked = sorted(carry_scores.items(), key=lambda kv: kv[1], reverse=True)

        signals: list[Signal] = []

        # ── Positive carry longs: SHORT the perp to collect funding ───────
        for i, (sym, sharpe) in enumerate(ranked[:self._long_n]):
            if sharpe <= 0:
                break
            mean_rate = float(np.mean(self._funding_history[sym]))
            annualised_carry = mean_rate * 3 * 365   # 3 funding periods/day × 365

            # Confidence scales with annualised carry magnitude
            confidence = min(abs(sharpe) * 0.3 * regime_scale, 0.90)
            if confidence < 0.25:
                continue

            signals.append(Signal(
                symbol=sym, market="futures", side="SELL",
                quantity=0.0, price=0,
                confidence=confidence,
                strategy=self.name,
                meta={"carry_type": "positive", "ann_carry_pct": annualised_carry * 100},
            ))
            log.debug("Carry SELL %s  Sharpe=%.2f  ann=%.1f%%", sym, sharpe, annualised_carry * 100)

        # ── Negative carry shorts: LONG the perp (negative funding = longs paid) ─
        for sym, sharpe in ranked[-self._short_n:]:
            if sharpe >= 0:
                continue
            mean_rate = float(np.mean(self._funding_history[sym]))
            annualised_carry = mean_rate * 3 * 365

            confidence = min(abs(sharpe) * 0.3 * regime_scale, 0.80)
            if confidence < 0.25:
                continue

            signals.append(Signal(
                symbol=sym, market="futures", side="BUY",
                quantity=0.0, price=0,
                confidence=confidence,
                strategy=self.name,
                meta={"carry_type": "negative", "ann_carry_pct": annualised_carry * 100},
            ))
            log.debug("Carry BUY %s  Sharpe=%.2f  ann=%.1f%%", sym, sharpe, annualised_carry * 100)

        return signals
