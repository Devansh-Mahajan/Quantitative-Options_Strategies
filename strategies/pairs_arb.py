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


    def __init__(
        self,
        zscore_entry: float = ZSCORE_ENTRY,
        zscore_exit: float = ZSCORE_EXIT,
        lookback: int = LOOKBACK,
        min_correlation: float = MIN_CORRELATION,
    ) -> None:
        # Tunable thresholds (defaults = historical module constants).
        self._zscore_entry = zscore_entry
        self._zscore_exit = zscore_exit
        self._lookback = int(lookback)
        self._min_correlation = min_correlation

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
                closes_a = store.get_closes(sym_a, "1h", self._lookback + 5)
                closes_b = store.get_closes(sym_b, "1h", self._lookback + 5)

                if len(closes_a) < self._lookback or len(closes_b) < self._lookback:
                    continue

                a = np.array(closes_a[-self._lookback:], dtype=float)
                b = np.array(closes_b[-self._lookback:], dtype=float)

                # Correlation check
                corr = float(np.corrcoef(a, b)[0, 1])
                if corr < self._min_correlation:
                    continue

                spread = np.log(a) - np.log(b)
                z = (spread[-1] - spread.mean()) / (spread.std() + 1e-10)

                price_a = float(a[-1])
                price_b = float(b[-1])

                if z > self._zscore_entry:
                    # spread too high: sell A, buy B
                    confidence = min(0.85, 0.55 + (z - self._zscore_entry) * 0.10) * regime_scale
                    signals.append(Signal(
                        symbol=sym_a, market="futures", side="SELL",
                        quantity=0.0, price=price_a, confidence=confidence,
                        strategy=self.name,
                        meta={"zscore": z, "pair": f"{sym_a}/{sym_b}", "corr": corr, "pair_id": f"pairs_arb:{sym_a}/{sym_b}"},
                    ))
                    signals.append(Signal(
                        symbol=sym_b, market="futures", side="BUY",
                        quantity=0.0, price=price_b, confidence=confidence,
                        strategy=self.name,
                        meta={"zscore": z, "pair": f"{sym_a}/{sym_b}", "corr": corr, "pair_id": f"pairs_arb:{sym_a}/{sym_b}"},
                    ))

                elif z < -self._zscore_entry:
                    # spread too low: buy A, sell B
                    confidence = min(0.85, 0.55 + (abs(z) - self._zscore_entry) * 0.10) * regime_scale
                    signals.append(Signal(
                        symbol=sym_a, market="futures", side="BUY",
                        quantity=0.0, price=price_a, confidence=confidence,
                        strategy=self.name,
                        meta={"zscore": z, "pair": f"{sym_a}/{sym_b}", "corr": corr, "pair_id": f"pairs_arb:{sym_a}/{sym_b}"},
                    ))
                    signals.append(Signal(
                        symbol=sym_b, market="futures", side="SELL",
                        quantity=0.0, price=price_b, confidence=confidence,
                        strategy=self.name,
                        meta={"zscore": z, "pair": f"{sym_a}/{sym_b}", "corr": corr, "pair_id": f"pairs_arb:{sym_a}/{sym_b}"},
                    ))

            except Exception as exc:
                log.debug("PairsArb %s/%s error: %s", sym_a, sym_b, exc)

        log.debug("PairsArb generated %d signals", len(signals))
        return signals
