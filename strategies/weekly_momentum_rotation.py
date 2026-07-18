"""
Weekly cross-sectional momentum rotation.

Every ~7 days, rank the universe by blended 7d/30d return (skipping the most
recent day to sidestep short-term reversal), and hold the top N names ONLY
when their momentum is positive in absolute terms — otherwise stay in cash.
One-week holding periods make the expected move per position ~5-15%, dwarfing
the ~60bps Alpaca roundtrip that killed the hourly cross-sectional variant
(measured -1.35% over 12 months). Long-only by construction: rotation sells
simply exit; there is no short leg to orphan.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bot.config import cfg
from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.weekly_momentum_rotation")

REBAL_CYCLES = 7 * 24     # weekly at 1h cadence
TOP_N = 2
FAST_DAYS = 7
SLOW_DAYS = 30
SKIP_DAYS = 1             # skip the latest day (reversal effect)
MIN_ABS_MOMENTUM = 0.0    # only hold names with positive blended momentum


class WeeklyMomentumRotationStrategy(BaseStrategy):
    name = "weekly_momentum_rotation"
    market = "futures"
    required_regime = None

    def __init__(
        self,
        top_n: int = TOP_N,
        fast_days: int = FAST_DAYS,
        slow_days: int = SLOW_DAYS,
        rebal_cycles: int = REBAL_CYCLES,
        min_abs_momentum: float = MIN_ABS_MOMENTUM,
    ) -> None:
        self._top_n = int(top_n)
        self._fast = int(fast_days)
        self._slow = int(slow_days)
        self._rebal = max(1, int(rebal_cycles))
        self._min_mom = float(min_abs_momentum)
        self._cycle = 0
        self._held: set[str] = set()

    @property
    def symbols(self) -> list[str]:
        return cfg.runtime_symbols

    def _blended_momentum(self, closes: list[float]) -> float | None:
        need_hours = (self._slow + SKIP_DAYS) * 24 + 2
        if len(closes) < need_hours:
            return None
        arr = np.asarray(closes, dtype=float)
        # hourly series -> use day offsets in hours; skip the latest SKIP_DAYS
        end = -SKIP_DAYS * 24 if SKIP_DAYS else len(arr)
        anchor = arr[end - 1] if end < 0 else arr[-1]
        fast_ret = anchor / arr[end - self._fast * 24 - 1] - 1.0
        slow_ret = anchor / arr[end - self._slow * 24 - 1] - 1.0
        return 0.5 * fast_ret + 0.5 * slow_ret

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        self._cycle += 1
        if self._cycle % self._rebal != 0:
            return []

        scores: dict[str, float] = {}
        prices: dict[str, float] = {}
        need = (self._slow + SKIP_DAYS) * 24 + 4
        for symbol in self.symbols:
            try:
                closes = store.get_closes(symbol, "1h", need)
                momentum = self._blended_momentum(closes)
                if momentum is None:
                    continue
                scores[symbol] = momentum
                prices[symbol] = float(closes[-1])
            except Exception as exc:
                log.debug("weekly_rotation %s error: %s", symbol, exc)

        if len(scores) < max(3, self._top_n + 1):
            return []

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        target = {sym for sym, mom in ranked[: self._top_n] if mom > self._min_mom}

        signals: list[Signal] = []
        # Exit names that dropped out of the target basket.
        for symbol in sorted(self._held - target):
            signals.append(Signal(
                symbol, "futures", "SELL", 0.0, prices.get(symbol, 0.0), 0.6, self.name,
                meta={"mode": "rotation_exit"},
            ))
        # Enter new leaders; confidence scales with cross-sectional dominance.
        spread = (ranked[0][1] - ranked[-1][1]) or 1e-9
        for symbol in sorted(target - self._held):
            momentum = scores[symbol]
            rank_edge = (momentum - ranked[-1][1]) / spread   # 1.0 = best of universe
            confidence = float(np.clip(0.50 + 0.35 * rank_edge, 0.50, 0.85))
            signals.append(Signal(
                symbol, "futures", "BUY", 0.0, prices[symbol], confidence, self.name,
                meta={"mode": "rotation_entry", "momentum": round(momentum, 4)},
            ))

        self._held = target
        return signals
