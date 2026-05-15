"""
MovementScorer: per-symbol next-bar direction probability.

Uses the candle history already buffered in MarketDataStore — no yfinance
downloads, safe to call every 60-second trading cycle.
"""

from __future__ import annotations
import logging

import numpy as np

from core.movement_predictor import MovementSignal

log = logging.getLogger("ml.movement_scorer")


class MovementScorer:
    """
    Scores each symbol using a momentum/vol composite logistic proxy.
    Output matches the MovementSignal duck-type expected by signal_fusion and
    greeks_targeting: symbol, probability_up, expected_daily_move, expected_direction.
    """

    def score(self, store, symbol: str, interval: str = "1h") -> MovementSignal | None:
        df = store.get_history_df(symbol, interval)
        if len(df) < 30:
            return None
        try:
            close = df["close"].astype(float)
            ret_1 = close.pct_change().fillna(0.0)
            ret_5 = (close / close.shift(5) - 1.0).fillna(0.0)
            ret_20 = (close / close.shift(20) - 1.0).fillna(0.0)
            vol_20 = ret_1.rolling(20).std().fillna(0.02).clip(lower=1e-5)

            # Fraction of up-bars in last 14 (RSI proxy)
            up_frac = (ret_1 > 0).rolling(14).mean().fillna(0.5)

            r1  = float(ret_1.iloc[-1])
            r5  = float(ret_5.iloc[-1])
            r20 = float(ret_20.iloc[-1])
            vol = float(vol_20.iloc[-1])
            uf  = float(up_frac.iloc[-1])

            # Composite bull score: recent momentum normalised by vol + RSI tilt
            norm_r1  = float(np.clip(r1  / vol,  -2.0, 2.0))
            norm_r5  = float(np.clip(r5  / (vol * np.sqrt(5)),  -2.0, 2.0))
            norm_r20 = float(np.clip(r20 / (vol * np.sqrt(20)), -2.0, 2.0))

            # Weighted sum → logistic
            z = 0.40 * norm_r1 + 0.35 * norm_r5 + 0.15 * norm_r20 + 0.10 * (uf - 0.5) * 4.0
            prob_up = float(1.0 / (1.0 + np.exp(-z)))
            prob_up = float(np.clip(prob_up, 0.05, 0.95))

            expected_move = float(vol * (prob_up - 0.5) * 2.0)

            if prob_up > 0.55:
                direction = "up"
            elif prob_up < 0.45:
                direction = "down"
            else:
                direction = "flat"

            return MovementSignal(
                symbol=symbol,
                probability_up=prob_up,
                expected_daily_move=expected_move,
                expected_direction=direction,
            )
        except Exception as exc:
            log.debug("MovementScorer.score failed for %s: %s", symbol, exc)
            return None

    def score_all(self, store, symbols: list[str]) -> list[MovementSignal]:
        results: list[MovementSignal] = []
        for symbol in symbols:
            sig = self.score(store, symbol)
            if sig is not None:
                results.append(sig)
        return results
