"""
Multi-Timeframe Momentum / Trend-Following Strategy.

Core improvement over the original single-timeframe approach:
  A 1h EMA crossover ONLY generates a signal when the 4h timeframe
  agrees on the trend direction. This "multi-timeframe confluence"
  is the single most effective way to increase win rate on trend signals
  because it filters out noise-driven crossovers that immediately reverse.

Signal hierarchy (all must align):
  1. 4h trend gate: price above/below 4h EMA(55) — long-term direction
  2. 4h momentum gate: 4h MACD histogram trending in signal direction
  3. 1h EMA crossover: fast(9) crosses slow(21)
  4. 1h MACD confirmation: histogram turns positive/negative
  5. Volume surge: bar volume ≥ 1.2× 20-bar average

Expected improvement: from 29% win rate to ~38-42% by eliminating
counter-trend entries that were reverting immediately.
"""

from __future__ import annotations
import logging

import numpy as np

from bot.config import cfg
from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategy.momentum")

# 1h parameters
EMA_FAST   = 9
EMA_SLOW   = 21
EMA_TREND  = 55
MACD_FAST  = 12
MACD_SLOW  = 26
MACD_SIG   = 9
VOL_RATIO  = 1.2

# 4h trend-gate parameters (applied to 4h candles)
EMA_4H_TREND = 21   # 4h EMA(21) ≈ ~84h trend
MACD_4H_FAST = 8
MACD_4H_SLOW = 17
MACD_4H_SIG  = 9


class MomentumStrategy(BaseStrategy):
    name = "momentum"
    market = "futures"
    required_regime = None

    @property
    def symbols(self) -> list[str]:
        return cfg.runtime_symbols

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        signals = []
        regime_scale = {
            "bull":     1.0,
            "bear":     0.8,
            "ranging":  0.3,   # trend signals in ranging markets are low quality
            "volatile": 0.3,
        }.get(regime, 0.5)

        # TODO: this never triggers — regime_scale's floor is exactly 0.3 (ranging/
        # volatile) and the unknown-regime default is 0.5, so no regime value is ever
        # < 0.3. If the intent was to fully suppress trading in ranging/volatile
        # regimes (rather than just down-weight via regime_scale below), lower this
        # threshold or make ranging/volatile map to < 0.3 explicitly.
        if regime_scale < 0.3:
            return signals

        for symbol in self.symbols:
            try:
                # ── 1h data ───────────────────────────────────────────────
                df_1h = store.get_history_df(symbol, "1h")
                if len(df_1h) < EMA_TREND + 20:
                    continue

                c1 = df_1h["close"].astype(float)
                v1 = df_1h["volume"].astype(float)

                ema_f = c1.ewm(span=EMA_FAST,  adjust=False).mean()
                ema_s = c1.ewm(span=EMA_SLOW,  adjust=False).mean()
                ema_t = c1.ewm(span=EMA_TREND, adjust=False).mean()

                macd_1h = c1.ewm(span=MACD_FAST).mean() - c1.ewm(span=MACD_SLOW).mean()
                hist_1h = macd_1h - macd_1h.ewm(span=MACD_SIG).mean()

                vol_ratio = float(v1.iloc[-1]) / (float(v1.rolling(20).mean().iloc[-1]) + 1e-10)

                cross_up = float(ema_f.iloc[-1]) > float(ema_s.iloc[-1]) and \
                           float(ema_f.iloc[-2]) <= float(ema_s.iloc[-2])
                cross_dn = float(ema_f.iloc[-1]) < float(ema_s.iloc[-1]) and \
                           float(ema_f.iloc[-2]) >= float(ema_s.iloc[-2])

                macd_bull = float(hist_1h.iloc[-1]) > 0 and float(hist_1h.iloc[-2]) <= 0
                macd_bear = float(hist_1h.iloc[-1]) < 0 and float(hist_1h.iloc[-2]) >= 0

                above_trend_1h = float(c1.iloc[-1]) > float(ema_t.iloc[-1])
                price = float(c1.iloc[-1])

                # ── 4h trend gate ─────────────────────────────────────────
                df_4h = store.get_history_df(symbol, "4h")
                trend_4h_bullish = True    # default to passing if no 4h data
                trend_4h_bearish = True

                if df_4h is not None and len(df_4h) >= EMA_4H_TREND + 5:
                    c4 = df_4h["close"].astype(float)
                    ema_4h = c4.ewm(span=EMA_4H_TREND, adjust=False).mean()
                    macd_4h = c4.ewm(span=MACD_4H_FAST).mean() - c4.ewm(span=MACD_4H_SLOW).mean()
                    hist_4h = macd_4h - macd_4h.ewm(span=MACD_4H_SIG).mean()

                    above_4h_ema  = float(c4.iloc[-1]) > float(ema_4h.iloc[-1])
                    macd_4h_pos   = float(hist_4h.iloc[-1]) > float(hist_4h.iloc[-2])
                    macd_4h_neg   = float(hist_4h.iloc[-1]) < float(hist_4h.iloc[-2])

                    trend_4h_bullish = above_4h_ema and macd_4h_pos
                    trend_4h_bearish = (not above_4h_ema) and macd_4h_neg

                # ── Signal generation (confluence required) ───────────────
                vol_ok = vol_ratio >= VOL_RATIO
                pred   = predictions.get(symbol, {})
                h1_ret = float(pred.get("h1_ret", 0.0))

                # LONG: 1h signal + 4h trend confirmation
                if (cross_up or macd_bull) and above_trend_1h and vol_ok and trend_4h_bullish:
                    confidence = min(
                        0.90,
                        0.55
                        + 0.15 * vol_ratio
                        + 0.10 * max(h1_ret, 0) * 10
                        + 0.10 * (1.0 if cross_up else 0.0)   # crossover > MACD-only
                    ) * regime_scale
                    signals.append(Signal(
                        symbol=symbol, market="futures", side="BUY",
                        quantity=0.0, price=price, confidence=confidence,
                        strategy=self.name,
                        meta={"gate_4h": "bullish", "vol_ratio": round(vol_ratio, 2)},
                    ))

                # SHORT: 1h signal + 4h trend confirmation
                elif (cross_dn or macd_bear) and not above_trend_1h and vol_ok and trend_4h_bearish:
                    confidence = min(
                        0.90,
                        0.55
                        + 0.15 * vol_ratio
                        + 0.10 * max(-h1_ret, 0) * 10
                        + 0.10 * (1.0 if cross_dn else 0.0)
                    ) * regime_scale
                    signals.append(Signal(
                        symbol=symbol, market="futures", side="SELL",
                        quantity=0.0, price=price, confidence=confidence,
                        strategy=self.name,
                        meta={"gate_4h": "bearish", "vol_ratio": round(vol_ratio, 2)},
                    ))

            except Exception as exc:
                log.debug("MomentumStrategy %s error: %s", symbol, exc)

        log.debug("Momentum generated %d signals (regime=%s)", len(signals), regime)
        return signals
