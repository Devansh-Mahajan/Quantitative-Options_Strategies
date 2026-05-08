"""
Funding rate arbitrage.
When funding rate is high (longs pay shorts): short perp, buy spot — collect funding.
When funding rate is very negative (shorts pay longs): long perp, sell spot.
Pure cash flow play; delta-neutral.
"""

from __future__ import annotations
import logging

from bot.config import cfg
from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategy.funding_arb")

FUNDING_LONG_THRESHOLD = 0.0005    # > 0.05% per 8h = enter short-perp/long-spot
FUNDING_SHORT_THRESHOLD = -0.0003  # < -0.03% per 8h = enter long-perp/short-spot
FUNDING_EXIT_THRESHOLD = 0.0001   # close when funding normalises


class FundingArbStrategy(BaseStrategy):
    name = "funding_arb"
    market = "futures"

    @property
    def symbols(self) -> list[str]:
        return cfg.futures_symbols

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        signals = []

        for symbol in self.symbols:
            try:
                funding = store.funding.get(symbol, 0.0)
                price = store.candles.get((symbol, "1h"))
                if price is None:
                    continue
                price = price.close

                # Annualised funding: 3 periods/day × 365 days
                ann_funding = funding * 3 * 365

                if funding > FUNDING_LONG_THRESHOLD:
                    # Short perpetual + corresponding spot long (collect high funding)
                    confidence = min(0.95, 0.50 + funding / FUNDING_LONG_THRESHOLD * 0.25)
                    # Signal futures short (the spot hedge is handled by orchestrator)
                    signals.append(Signal(
                        symbol=symbol, market="futures", side="SELL",
                        quantity=0.0, price=price,
                        confidence=confidence,
                        strategy=self.name,
                        meta={"funding_rate": funding, "ann_funding": ann_funding, "leg": "perp_short"},
                    ))
                    # Spot long hedge
                    spot_sym = symbol  # same symbol for spot
                    if symbol in cfg.spot_symbols:
                        signals.append(Signal(
                            symbol=spot_sym, market="spot", side="BUY",
                            quantity=0.0, price=price,
                            confidence=confidence,
                            strategy=self.name,
                            meta={"funding_rate": funding, "leg": "spot_long"},
                        ))

                elif funding < FUNDING_SHORT_THRESHOLD:
                    # Long perpetual (collect negative funding from shorts)
                    confidence = min(0.90, 0.50 + abs(funding) / abs(FUNDING_SHORT_THRESHOLD) * 0.20)
                    signals.append(Signal(
                        symbol=symbol, market="futures", side="BUY",
                        quantity=0.0, price=price,
                        confidence=confidence,
                        strategy=self.name,
                        meta={"funding_rate": funding, "ann_funding": ann_funding},
                    ))

            except Exception as exc:
                log.debug("FundingArb %s error: %s", symbol, exc)

        log.debug("FundingArb generated %d signals", len(signals))
        return signals
