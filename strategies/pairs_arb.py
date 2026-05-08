"""
Statistical pairs arbitrage: BTC/ETH spread mean reversion.
Uses Z-score of log price ratio; HMM regime filter.
"""

from __future__ import annotations
import logging

import numpy as np

from bot.config import cfg
from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategy.pairs_arb")

ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.5
LOOKBACK = 60       # bars for spread stats
MIN_CORRELATION = 0.65


class PairsArbStrategy(BaseStrategy):
    name = "pairs_arb"
    market = "futures"
    required_regime = None  # works in ranging and trending

    # BTC/ETH is the anchor pair; add more as needed
    PAIRS = [("BTCUSDT", "ETHUSDT"), ("BNBUSDT", "SOLUSDT")]

    @property
    def symbols(self) -> list[str]:
        seen: set[str] = set()
        for a, b in self.PAIRS:
            seen.add(a)
            seen.add(b)
        return [s for s in seen if s in cfg.futures_symbols]

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        signals = []
        # Pairs arb works best in ranging; reduce in strongly trending regimes
        regime_scale = {"ranging": 1.0, "volatile": 0.5, "bull": 0.6, "bear": 0.6}.get(regime, 0.7)

        for sym_a, sym_b in self.PAIRS:
            try:
                closes_a = store.get_closes(sym_a, "1h", LOOKBACK + 5)
                closes_b = store.get_closes(sym_b, "1h", LOOKBACK + 5)

                if len(closes_a) < LOOKBACK or len(closes_b) < LOOKBACK:
                    continue

                a = np.array(closes_a[-LOOKBACK:], dtype=float)
                b = np.array(closes_b[-LOOKBACK:], dtype=float)

                # Correlation check
                corr = float(np.corrcoef(a, b)[0, 1])
                if corr < MIN_CORRELATION:
                    continue

                spread = np.log(a) - np.log(b)
                z = (spread[-1] - spread.mean()) / (spread.std() + 1e-10)

                price_a = float(a[-1])
                price_b = float(b[-1])

                if z > ZSCORE_ENTRY:
                    # spread too high: sell A, buy B
                    confidence = min(0.85, 0.55 + (z - ZSCORE_ENTRY) * 0.10) * regime_scale
                    signals.append(Signal(
                        symbol=sym_a, market="futures", side="SELL",
                        quantity=0.0, price=price_a, confidence=confidence,
                        strategy=self.name,
                        meta={"zscore": z, "pair": f"{sym_a}/{sym_b}", "corr": corr},
                    ))
                    signals.append(Signal(
                        symbol=sym_b, market="futures", side="BUY",
                        quantity=0.0, price=price_b, confidence=confidence,
                        strategy=self.name,
                        meta={"zscore": z, "pair": f"{sym_a}/{sym_b}", "corr": corr},
                    ))

                elif z < -ZSCORE_ENTRY:
                    # spread too low: buy A, sell B
                    confidence = min(0.85, 0.55 + (abs(z) - ZSCORE_ENTRY) * 0.10) * regime_scale
                    signals.append(Signal(
                        symbol=sym_a, market="futures", side="BUY",
                        quantity=0.0, price=price_a, confidence=confidence,
                        strategy=self.name,
                        meta={"zscore": z, "pair": f"{sym_a}/{sym_b}", "corr": corr},
                    ))
                    signals.append(Signal(
                        symbol=sym_b, market="futures", side="SELL",
                        quantity=0.0, price=price_b, confidence=confidence,
                        strategy=self.name,
                        meta={"zscore": z, "pair": f"{sym_a}/{sym_b}", "corr": corr},
                    ))

            except Exception as exc:
                log.debug("PairsArb %s/%s error: %s", sym_a, sym_b, exc)

        log.debug("PairsArb generated %d signals", len(signals))
        return signals
