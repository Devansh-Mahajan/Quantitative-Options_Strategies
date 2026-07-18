"""
Volatility-managed momentum (daily cadence).

The most robust documented return factor in crypto: hold assets whose
medium-term trend is up, with exposure scaled INVERSELY to recent realized
volatility (vol-managed momentum roughly doubles momentum's Sharpe in the
academic record because momentum crashes cluster in high-vol regimes).

Design for Alpaca's ~60bps roundtrip: evaluates once per day, holds for
days-to-weeks, exits only when the trend flips or volatility spikes into the
top decile — turnover measured in trades per month, not per hour.
"""

from __future__ import annotations

import logging

import numpy as np

from bot.config import cfg
from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.vol_managed_momentum")

TREND_HOURS = 30 * 24        # ~30d trend window
FAST_HOURS = 10 * 24         # ~10d confirmation window
VOL_HOURS = 14 * 24          # realized-vol window
VOL_SPIKE_PCTL = 90.0        # exit when current vol above this percentile
EVAL_EVERY = 24              # daily at 1h cadence
TARGET_DAILY_VOL = 0.03      # vol scaling anchor (3% daily)


class VolManagedMomentumStrategy(BaseStrategy):
    name = "vol_managed_momentum"
    market = "futures"
    required_regime = None

    def __init__(
        self,
        trend_hours: int = TREND_HOURS,
        fast_hours: int = FAST_HOURS,
        vol_hours: int = VOL_HOURS,
        vol_spike_pctl: float = VOL_SPIKE_PCTL,
        eval_every: int = EVAL_EVERY,
    ) -> None:
        self._trend_h = int(trend_hours)
        self._fast_h = int(fast_hours)
        self._vol_h = int(vol_hours)
        self._spike_pctl = float(vol_spike_pctl)
        self._eval_every = max(1, int(eval_every))
        self._cycle = 0
        self._held: set[str] = set()

    @property
    def symbols(self) -> list[str]:
        return cfg.runtime_symbols

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        self._cycle += 1
        if self._cycle % self._eval_every != 0:
            return []

        signals: list[Signal] = []
        need = self._trend_h + 24
        for symbol in self.symbols:
            try:
                closes = store.get_closes(symbol, "1h", need)
                if len(closes) < self._trend_h + 2:
                    continue
                arr = np.asarray(closes, dtype=float)
                price = float(arr[-1])

                trend_ret = price / arr[-self._trend_h - 1] - 1.0
                fast_ret = price / arr[-self._fast_h - 1] - 1.0

                hourly_rets = np.diff(np.log(arr))
                daily_vol = float(np.std(hourly_rets[-self._vol_h:]) * np.sqrt(24))
                # Vol percentile over the available window (rolling daily-vol proxy)
                chunks = [
                    float(np.std(hourly_rets[i:i + 24]) * np.sqrt(24))
                    for i in range(0, max(1, len(hourly_rets) - 24), 24)
                ]
                vol_pctl = float((np.asarray(chunks) <= daily_vol).mean() * 100) if chunks else 50.0

                uptrend = trend_ret > 0 and fast_ret > 0
                vol_calm = vol_pctl < self._spike_pctl

                if symbol in self._held:
                    if not uptrend or not vol_calm:
                        signals.append(Signal(
                            symbol, "futures", "SELL", 0.0, price, 0.6, self.name,
                            meta={"mode": "vmm_exit",
                                  "reason": "trend_flip" if not uptrend else "vol_spike"},
                        ))
                        self._held.discard(symbol)
                    continue

                if uptrend and vol_calm and daily_vol > 0:
                    # Inverse-vol scaling: calm trends get full conviction,
                    # hot ones get less — this scaling IS the documented edge.
                    vol_scalar = float(np.clip(TARGET_DAILY_VOL / daily_vol, 0.3, 1.5))
                    trend_strength = float(np.clip(trend_ret / (daily_vol * np.sqrt(30) + 1e-9), 0.0, 2.0))
                    confidence = float(np.clip(0.45 + 0.20 * trend_strength * vol_scalar, 0.45, 0.88))
                    if confidence < 0.5:
                        continue
                    signals.append(Signal(
                        symbol, "futures", "BUY", 0.0, price, confidence, self.name,
                        meta={"mode": "vmm_entry", "trend_30d": round(trend_ret, 4),
                              "daily_vol": round(daily_vol, 4), "vol_pctl": round(vol_pctl, 1)},
                    ))
                    self._held.add(symbol)
            except Exception as exc:
                log.debug("vol_managed_momentum %s error: %s", symbol, exc)

        return signals
