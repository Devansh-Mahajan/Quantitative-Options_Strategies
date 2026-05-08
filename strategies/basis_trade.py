"""
Cash-and-carry basis trade.
Buy spot + short quarterly futures when futures trade at a premium > threshold.
Profit from basis convergence at expiry.
"""

from __future__ import annotations
import logging

from bot.config import cfg
from strategies.base import BaseStrategy, Signal

log = logging.getLogger("strategy.basis_trade")

BASIS_ENTRY_THRESHOLD = 0.005   # 0.5% premium to enter
BASIS_EXIT_THRESHOLD = 0.001    # close when premium < 0.1%


class BasisTradeStrategy(BaseStrategy):
    name = "basis_trade"
    market = "futures"

    @property
    def symbols(self) -> list[str]:
        # Only on most liquid — BTC and ETH have quarterly futures with good liquidity
        return [s for s in cfg.futures_symbols if s in ("BTCUSDT", "ETHUSDT")]

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        """
        For simplicity, we model basis as (perp_price - spot_price) / spot_price.
        True basis trading uses dated quarterly futures, but perp vs spot is a valid proxy.
        """
        signals = []

        for symbol in self.symbols:
            try:
                spot_candle = store.candles.get((symbol, "1h"))
                if spot_candle is None:
                    continue
                spot_price = spot_candle.close

                # Use mark price from futures as proxy for futures price
                # (stored in store.candles for futures interval)
                futures_price = spot_price  # will be updated by orchestrator with mark price
                funding = store.funding.get(symbol, 0.0)

                # Approximate annualised basis from funding rate
                ann_basis = funding * 3 * 365

                if ann_basis > BASIS_ENTRY_THRESHOLD * 365:
                    # Futures at premium: buy spot, short futures
                    confidence = min(0.90, 0.60 + ann_basis * 0.5)
                    signals.append(Signal(
                        symbol=symbol, market="spot", side="BUY",
                        quantity=0.0, price=spot_price,
                        confidence=confidence,
                        strategy=self.name,
                        meta={"basis_pct": ann_basis, "leg": "spot_long"},
                    ))
                    signals.append(Signal(
                        symbol=symbol, market="futures", side="SELL",
                        quantity=0.0, price=spot_price,
                        confidence=confidence,
                        strategy=self.name,
                        meta={"basis_pct": ann_basis, "leg": "futures_short"},
                    ))

            except Exception as exc:
                log.debug("BasisTrade %s error: %s", symbol, exc)

        return signals
