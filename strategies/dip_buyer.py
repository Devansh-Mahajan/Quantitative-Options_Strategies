"""
Daily-scale capitulation dip buyer.

Buys multi-day washouts INSIDE an intact longer-term uptrend: the 3-day return
must be an extreme negative outlier (z-score < -2 vs its own 90-day history)
while price still sits above the 60-day mean. Exits at a 50% retrace of the
dip, a 1.5x ATR hard stop, or a 10-day time stop.

This is the opposite corner of the design space from the pruned HOURLY
mean_reversion (-5.62% over 12 months): a handful of large-edge entries per
year per symbol instead of hundreds of fee-sized scalps.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bot.config import cfg
from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.dip_buyer")

DIP_DAYS = 3
Z_WINDOW_DAYS = 90
Z_ENTRY = -2.0
TREND_DAYS = 60
ATR_DAYS = 14
ATR_STOP_MULT = 1.5
TIME_STOP_DAYS = 10
EVAL_EVERY = 24            # daily at 1h cadence
STALE_CYCLES = 24 * 30


class DipBuyerStrategy(BaseStrategy):
    name = "dip_buyer"
    market = "futures"
    required_regime = None

    def __init__(
        self,
        z_entry: float = Z_ENTRY,
        dip_days: int = DIP_DAYS,
        trend_days: int = TREND_DAYS,
        atr_stop_mult: float = ATR_STOP_MULT,
        time_stop_days: int = TIME_STOP_DAYS,
        eval_every: int = EVAL_EVERY,
    ) -> None:
        self._z_entry = float(z_entry)
        self._dip_days = int(dip_days)
        self._trend_days = int(trend_days)
        self._atr_stop_mult = float(atr_stop_mult)
        self._time_stop_days = int(time_stop_days)
        self._eval_every = max(1, int(eval_every))
        self._cycle = 0
        # sym -> {"entry": price, "pre_dip": price, "stop": price, "opened": cycle}
        self._positions: dict[str, dict] = {}

    @property
    def symbols(self) -> list[str]:
        return cfg.runtime_symbols

    @staticmethod
    def _daily_closes(df: pd.DataFrame) -> pd.Series | None:
        if df is None or len(df) < 48:
            return None
        frame = df.copy()
        if "open_time" in frame.columns:
            ts = pd.to_datetime(frame["open_time"], unit="ms", errors="coerce")
            if ts.isna().all():
                ts = pd.to_datetime(frame["open_time"], errors="coerce")
            frame.index = ts
        if not isinstance(frame.index, pd.DatetimeIndex):
            return None
        daily = frame["close"].astype(float).resample("1D").last().dropna()
        return daily if len(daily) >= 30 else None

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        self._cycle += 1
        if self._cycle % self._eval_every != 0:
            return []

        signals: list[Signal] = []
        for symbol in self.symbols:
            try:
                daily = self._daily_closes(store.get_history_df(symbol, "1h"))
                if daily is None or len(daily) < self._dip_days + 20:
                    continue
                price = float(daily.iloc[-1])

                state = self._positions.get(symbol)
                if state and self._cycle - state.get("opened", 0) > STALE_CYCLES:
                    self._positions.pop(symbol, None)
                    state = None

                if state:  # manage: retrace target / hard stop / time stop
                    target = state["entry"] + 0.5 * (state["pre_dip"] - state["entry"])
                    held_days = (self._cycle - state["opened"]) / self._eval_every
                    reason = None
                    if price >= target:
                        reason = "retrace_target"
                    elif price <= state["stop"]:
                        reason = "hard_stop"
                    elif held_days >= self._time_stop_days:
                        reason = "time_stop"
                    if reason:
                        signals.append(Signal(
                            symbol, "futures", "SELL", 0.0, price, 0.6, self.name,
                            meta={"mode": "dip_exit", "reason": reason},
                        ))
                        self._positions.pop(symbol, None)
                    continue

                # Entry: 3-day return is an extreme negative outlier...
                rets = daily.pct_change(self._dip_days).dropna()
                if len(rets) < 30:
                    continue
                window = rets.iloc[-Z_WINDOW_DAYS:]
                mu, sd = float(window.mean()), float(window.std() or 1e-9)
                z = (float(rets.iloc[-1]) - mu) / sd
                # ...while the longer trend is still intact.
                if len(daily) < self._trend_days:
                    continue
                trend_ok = price > float(daily.rolling(self._trend_days).mean().iloc[-1])

                if z < self._z_entry and trend_ok:
                    diffs = daily.diff().abs()
                    atr = float(diffs.rolling(ATR_DAYS).mean().iloc[-1] or 0.0)
                    if atr <= 0:
                        continue
                    pre_dip = float(daily.iloc[-self._dip_days - 1])
                    confidence = float(np.clip(0.55 + 0.12 * (abs(z) - abs(self._z_entry)), 0.55, 0.88))
                    signals.append(Signal(
                        symbol, "futures", "BUY", 0.0, price, confidence, self.name,
                        meta={"mode": "dip_entry", "z": round(z, 2), "pre_dip": round(pre_dip, 4)},
                    ))
                    self._positions[symbol] = {
                        "entry": price,
                        "pre_dip": pre_dip,
                        "stop": price - self._atr_stop_mult * atr,
                        "opened": self._cycle,
                    }
            except Exception as exc:
                log.debug("dip_buyer %s error: %s", symbol, exc)

        return signals
