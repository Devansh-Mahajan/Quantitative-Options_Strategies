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

    if cfg.enable_statistical_arb:
        from strategies.statistical_arb import StatisticalArbStrategy
        strategies.append(StatisticalArbStrategy())

    if cfg.enable_cross_sectional_momentum:
        from strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
        strategies.append(CrossSectionalMomentumStrategy())

    if cfg.enable_liquidation_cascade:
        from strategies.liquidation_cascade import LiquidationCascadeStrategy
        strategies.append(LiquidationCascadeStrategy())

    if cfg.enable_carry_portfolio:
        from strategies.carry_portfolio import CarryPortfolioStrategy
        strategies.append(CarryPortfolioStrategy())

    # ── Research-paper strategies (Kakushadze, Bloch, Cartea) ──────────
    if cfg.enable_tsmom:
        from strategies.tsmom import TSMomentumStrategy
        strategies.append(TSMomentumStrategy())

    if cfg.enable_quant_factors:
        from strategies.quant_factors import QuantFactorsStrategy
        strategies.append(QuantFactorsStrategy())

    if cfg.enable_contrarian_oi:
        from strategies.contrarian_oi import ContrarianOIStrategy
        strategies.append(ContrarianOIStrategy())

    if cfg.enable_rma_strategy:
        from strategies.rma_strategy import RMAStrategy
        strategies.append(RMAStrategy())

    if cfg.enable_vpin_flow:
        from strategies.vpin_flow import VPINFlowStrategy
        strategies.append(VPINFlowStrategy())

    if cfg.enable_knn_predictor:
        from strategies.knn_predictor import KNNPredictorStrategy
        strategies.append(KNNPredictorStrategy())

    if cfg.enable_pivot_sr:
        from strategies.pivot_sr import PivotSRStrategy
        strategies.append(PivotSRStrategy())

    if cfg.enable_hp_trend:
        from strategies.hp_trend import HPTrendStrategy
        strategies.append(HPTrendStrategy())

    if cfg.enable_momentum_carry_combo:
        from strategies.momentum_carry_combo import MomentumCarryComboStrategy
        strategies.append(MomentumCarryComboStrategy())

    log.info("Strategy registry: %d strategies active — %s",
             len(strategies), [s.name for s in strategies])
    return strategies
