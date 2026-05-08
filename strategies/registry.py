"""Strategy registry: instantiates and returns enabled strategies."""

from __future__ import annotations
import logging

from bot.config import cfg
from strategies.base import BaseStrategy

log = logging.getLogger("strategy.registry")


def build_registry() -> list[BaseStrategy]:
    """Return all enabled strategy instances based on config."""
    strategies: list[BaseStrategy] = []

    if cfg.enable_momentum:
        from strategies.momentum import MomentumStrategy
        strategies.append(MomentumStrategy())

    if cfg.enable_mean_reversion:
        from strategies.mean_reversion import MeanReversionStrategy
        strategies.append(MeanReversionStrategy())

    if cfg.enable_funding_arb:
        from strategies.funding_arb import FundingArbStrategy
        strategies.append(FundingArbStrategy())

    if cfg.enable_basis_trade:
        from strategies.basis_trade import BasisTradeStrategy
        strategies.append(BasisTradeStrategy())

    if cfg.enable_pairs_arb:
        from strategies.pairs_arb import PairsArbStrategy
        strategies.append(PairsArbStrategy())

    if cfg.enable_options_vol and cfg.enable_options:
        from strategies.options_vol import OptionsVolatilityStrategy
        strategies.append(OptionsVolatilityStrategy())

    if cfg.enable_order_flow:
        from strategies.order_flow import OrderFlowStrategy
        strategies.append(OrderFlowStrategy())

    if cfg.enable_breakout:
        from strategies.breakout import BreakoutStrategy
        strategies.append(BreakoutStrategy())

    log.info("Strategy registry: %d strategies active — %s",
             len(strategies), [s.name for s in strategies])
    return strategies
