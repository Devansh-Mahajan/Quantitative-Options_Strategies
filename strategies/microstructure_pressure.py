"""
Microstructure pressure strategy inspired by Cartea et al. (2024).

The paper focuses on predicting direction, price, and volume from market-
observable state. We adapt that idea to the current data model by combining:
  - signed flow imbalance (real taker flow when available, otherwise a proxy)
  - short-horizon return pressure
  - VWAP displacement
  - volume shock
  - spread / liquidity sanity checks
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bot.config import cfg
from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategies.microstructure_pressure")


class MicrostructurePressureStrategy(BaseStrategy):
    name = "microstructure_pressure"
    market = "futures"

    def __init__(
        self,
        flow_window: int = 12,
        volume_window: int = 24,
        vwap_window: int = 16,
        pressure_threshold: float = 0.48,
        spread_ceiling_bps: float = 30.0,
    ) -> None:
        self._flow_window = max(4, int(flow_window))
        self._volume_window = max(self._flow_window + 4, int(volume_window))
        self._vwap_window = max(6, int(vwap_window))
        self._pressure_threshold = float(pressure_threshold)
        self._spread_ceiling_bps = float(spread_ceiling_bps)

    @property
    def symbols(self) -> list[str]:
        return list(cfg.model_symbols[:8])

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        signals: list[Signal] = []
        regime_scale = {
            "bull": 1.0,
            "bear": 1.0,
            "volatile": 0.9,
            "ranging": 0.75,
        }.get(regime, 0.8)
        min_bars = max(self._volume_window, self._vwap_window, self._flow_window) + 6

        for symbol in self.symbols:
            try:
                df = store.get_history_df(symbol, "1h")
                if df is None or len(df) < min_bars:
                    continue

                frame = df.copy()
                close = frame["close"].astype(float)
                high = frame["high"].astype(float)
                low = frame["low"].astype(float)
                open_ = frame["open"].astype(float)
                volume = frame["volume"].astype(float).clip(lower=1e-9)

                typical = (high + low + close) / 3.0
                rolling_vwap = (
                    (typical * volume).rolling(self._vwap_window).sum()
                    / volume.rolling(self._vwap_window).sum().clip(lower=1e-9)
                )
                vwap_now = float(rolling_vwap.iloc[-1]) if not pd.isna(rolling_vwap.iloc[-1]) else float(close.iloc[-1])
                vwap_gap = (float(close.iloc[-1]) - vwap_now) / max(abs(vwap_now), 1e-9)

                ret1 = float(close.pct_change().iloc[-1] or 0.0)
                ret3 = float(close.pct_change(3).iloc[-1] or 0.0)
                ret6 = float(close.pct_change(6).iloc[-1] or 0.0)
                ret_hist = close.pct_change().tail(self._volume_window).dropna()
                ret_std = float(ret_hist.std() + 1e-9)
                ret_pressure = np.tanh(ret3 / (ret_std * 2.5))

                vol_mean = float(volume.tail(self._volume_window).mean() or 0.0)
                if vol_mean <= 0:
                    continue
                volume_shock = float(volume.iloc[-1] / vol_mean)
                volume_term = np.tanh((volume_shock - 1.0) * 1.3)

                if "taker_buy_base" in frame.columns and not frame["taker_buy_base"].isna().all():
                    taker = frame["taker_buy_base"].astype(float).clip(lower=0.0)
                    flow_series = ((2.0 * taker / volume) - 1.0).clip(-1.5, 1.5).fillna(0.0)
                else:
                    candle_range = (high - low).replace(0.0, np.nan)
                    close_location = (((close - low) - (high - close)) / candle_range).clip(-1.0, 1.0).fillna(0.0)
                    open_close_bias = np.sign(close - open_) * np.minimum(np.abs(close.pct_change().fillna(0.0)) * 40.0, 1.0)
                    flow_series = (0.7 * close_location + 0.3 * open_close_bias).clip(-1.0, 1.0)
                flow_imbalance = float(flow_series.tail(self._flow_window).mean())

                book = store.book.get(symbol)
                if book and book.ask > book.bid > 0:
                    spread_bps = (book.ask - book.bid) / ((book.ask + book.bid) / 2.0) * 10000.0
                    spread_factor = max(0.25, 1.0 - min(spread_bps, self._spread_ceiling_bps * 2.0) / (self._spread_ceiling_bps * 2.0))
                    if spread_bps > self._spread_ceiling_bps * 2.0:
                        continue
                else:
                    spread_bps = float(((high - low) / close).tail(self._flow_window).mean() * 10000.0)
                    spread_factor = 0.7

                pressure = (
                    0.36 * flow_imbalance
                    + 0.24 * ret_pressure
                    + 0.20 * np.tanh(vwap_gap * 90.0)
                    + 0.20 * volume_term
                )

                if abs(pressure) < self._pressure_threshold:
                    continue

                directional_archetype = "market_making"
                if abs(flow_imbalance) > 0.22 and abs(ret6) > max(0.006, ret_std * 1.5):
                    directional_archetype = "directional_trading"
                elif volume_shock > 1.4 and abs(vwap_gap) > 0.004:
                    directional_archetype = "opportunistic_trading"

                side = "BUY" if pressure > 0 else "SELL"
                if side == "BUY" and ret1 < -max(0.012, ret_std * 1.5):
                    continue
                if side == "SELL" and ret1 > max(0.012, ret_std * 1.5):
                    continue

                confidence = min(
                    0.88,
                    (0.26 + abs(pressure) * 0.58 + max(volume_shock - 1.0, 0.0) * 0.08)
                    * regime_scale
                    * spread_factor,
                )
                if confidence < 0.25:
                    continue

                signals.append(
                    Signal(
                        symbol=symbol,
                        market="futures",
                        side=side,
                        quantity=0.0,
                        price=float(close.iloc[-1]),
                        confidence=confidence,
                        strategy=self.name,
                        meta={
                            "pressure": round(float(pressure), 3),
                            "flow_imbalance": round(flow_imbalance, 3),
                            "vwap_gap": round(float(vwap_gap), 4),
                            "volume_shock": round(volume_shock, 3),
                            "spread_bps": round(float(spread_bps), 2),
                            "archetype": directional_archetype,
                        },
                    )
                )
            except Exception as exc:
                log.debug("Microstructure pressure %s error: %s", symbol, exc)

        return signals
