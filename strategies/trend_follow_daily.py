"""
Daily-timeframe trend following (Donchian style).

Designed explicitly to survive Alpaca's ~60bps roundtrip cost: entries fire on
20-DAY breakout highs confirmed by the 50-day mean, exits on a 10-day low or a
2.5x ATR(14) trailing stop — positions are held for weeks, so the expected move
per trade is an order of magnitude larger than fees. Evaluates once per 24
cycles (daily at the 1h cadence); everything else is a no-op cycle.

The classic turtle-style design: lose small often, win big rarely — profitable
iff the market trends, which crypto does in fat bursts.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bot.config import cfg
from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.trend_follow_daily")

ENTRY_DAYS = 20      # breakout lookback (days)
EXIT_DAYS = 10       # exit-channel lookback (days)
TREND_DAYS = 50      # long-term mean gate
ATR_DAYS = 14
ATR_TRAIL_MULT = 2.5
EVAL_EVERY = 24      # cycles between evaluations (~daily)
STALE_CYCLES = 24 * 90  # forget internal position state after ~90 days


class TrendFollowDailyStrategy(BaseStrategy):
    name = "trend_follow_daily"
    market = "futures"
    required_regime = None

    def __init__(
        self,
        entry_days: int = ENTRY_DAYS,
        exit_days: int = EXIT_DAYS,
        trend_days: int = TREND_DAYS,
        atr_trail_mult: float = ATR_TRAIL_MULT,
        eval_every: int = EVAL_EVERY,
    ) -> None:
        self._entry_days = int(entry_days)
        self._exit_days = int(exit_days)
        self._trend_days = int(trend_days)
        self._atr_trail_mult = float(atr_trail_mult)
        self._eval_every = max(1, int(eval_every))
        self._cycle = 0
        # sym -> {"side": "LONG", "peak": float, "opened": cycle}
        self._positions: dict[str, dict] = {}

    @property
    def symbols(self) -> list[str]:
        return cfg.runtime_symbols

    @staticmethod
    def _daily(df: pd.DataFrame) -> pd.DataFrame | None:
        """Resample 1h bars to daily OHLC (needs open_time or a datetime index)."""
        if df is None or len(df) < 30:
            return None
        frame = df.copy()
        if "open_time" in frame.columns:
            ts = pd.to_datetime(frame["open_time"], unit="ms", errors="coerce")
            if ts.isna().all():
                ts = pd.to_datetime(frame["open_time"], errors="coerce")
            frame.index = ts
        if not isinstance(frame.index, pd.DatetimeIndex):
            return None
        daily = frame.resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        return daily if len(daily) >= 25 else None

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        self._cycle += 1
        if self._cycle % self._eval_every != 0:
            return []

        signals: list[Signal] = []
        for symbol in self.symbols:
            try:
                daily = self._daily(store.get_history_df(symbol, "1h"))
                if daily is None or len(daily) < self._entry_days + 5:
                    continue

                close = daily["close"].astype(float)
                high = daily["high"].astype(float)
                low = daily["low"].astype(float)
                price = float(close.iloc[-1])

                prev_close = close.shift(1)
                tr = pd.concat(
                    [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
                ).max(axis=1)
                atr = float(tr.rolling(ATR_DAYS).mean().iloc[-1] or 0.0)

                entry_high = float(high.iloc[-self._entry_days - 1:-1].max())
                exit_low = float(low.iloc[-self._exit_days - 1:-1].min())
                trend_ok = len(close) >= self._trend_days and price > float(
                    close.rolling(self._trend_days).mean().iloc[-1]
                )

                state = self._positions.get(symbol)
                if state and self._cycle - state.get("opened", 0) > STALE_CYCLES:
                    state = None
                    self._positions.pop(symbol, None)

                if state:  # manage the open trend position
                    state["peak"] = max(state["peak"], price)
                    trail = state["peak"] - self._atr_trail_mult * atr
                    if price < exit_low or (atr > 0 and price < trail):
                        signals.append(Signal(
                            symbol, "futures", "SELL", 0.0, price, 0.6, self.name,
                            meta={"mode": "trend_exit",
                                  "reason": "exit_channel" if price < exit_low else "atr_trail"},
                        ))
                        self._positions.pop(symbol, None)
                    continue

                # Fresh 20-day breakout in an established uptrend
                if price > entry_high and trend_ok and atr > 0:
                    # Confidence scales with breakout strength vs daily noise.
                    strength = (price - entry_high) / atr
                    confidence = float(np.clip(0.55 + 0.15 * strength, 0.55, 0.90))
                    signals.append(Signal(
                        symbol, "futures", "BUY", 0.0, price, confidence, self.name,
                        meta={"mode": "trend_entry", "entry_high": round(entry_high, 4),
                              "atr": round(atr, 4)},
                    ))
                    self._positions[symbol] = {"side": "LONG", "peak": price, "opened": self._cycle}
            except Exception as exc:
                log.debug("trend_follow_daily %s error: %s", symbol, exc)

        return signals
