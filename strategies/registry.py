"""Strategy registry: instantiates and returns enabled strategies."""

from __future__ import annotations
import logging

from bot.config import cfg
from strategies.base import BaseStrategy

log = logging.getLogger("strategy.registry")

# Strategies that cannot function on Alpaca's crypto-spot venue and are
# auto-skipped there regardless of their ENABLE_ flag:
#   - gamma_scalping / vol_surface_arb: express volatility views, which need an
#     options/futures venue — on spot they degrade to naked directional bets
#     with no directional thesis.
#   - market_making: needs two-sided quoting; on spot the SELL side is gated,
#     leaving one-sided inventory accumulation.
#   - carry_portfolio / liquidation_cascade: depend on funding-rate /
#     liquidation feeds that Alpaca never populates (permanently silent).
# Skipping them also stops inert strategies diluting equal-weight/RL allocation.
ALPACA_INCOMPATIBLE = {
    "gamma_scalping",
    "vol_surface_arb",
    "market_making",
    "carry_portfolio",
    "liquidation_cascade",
}


def canonical_strategy_names() -> list[str]:
    """
    Full, venue-independent, order-stable strategy name list (every strategy
    whose ENABLE_ flag is on, before venue gating). This is the canonical
    ordering the RL allocator's weights are defined against.
    """
    return [s.name for s in _build_all_enabled()]


def build_registry() -> list[BaseStrategy]:
    """Return enabled strategy instances, minus venue-incompatible ones."""
    strategies = _build_all_enabled()

    if cfg.is_alpaca:
        skipped = [s.name for s in strategies if s.name in ALPACA_INCOMPATIBLE]
        if skipped:
            log.info("Venue gating: skipping %s on Alpaca (need futures/options/feeds)", skipped)
        strategies = [s for s in strategies if s.name not in ALPACA_INCOMPATIBLE]

    log.info("Strategy registry: %d strategies active — %s",
             len(strategies), [s.name for s in strategies])
    return strategies


def _build_all_enabled() -> list[BaseStrategy]:
    """All enabled strategy instances based on config flags (no venue gating)."""
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

    if cfg.enable_microstructure_pressure:
        from strategies.microstructure_pressure import MicrostructurePressureStrategy
        strategies.append(MicrostructurePressureStrategy())

    if cfg.enable_pullback_confluence:
        from strategies.pullback_confluence import PullbackConfluenceStrategy
        strategies.append(PullbackConfluenceStrategy())

    # ── Cost-aware daily/weekly strategies (2026-07) ───────────────────
    if cfg.enable_trend_follow_daily:
        from strategies.trend_follow_daily import TrendFollowDailyStrategy
        strategies.append(TrendFollowDailyStrategy())

    if cfg.enable_weekly_momentum_rotation:
        from strategies.weekly_momentum_rotation import WeeklyMomentumRotationStrategy
        strategies.append(WeeklyMomentumRotationStrategy())

    if cfg.enable_dip_buyer:
        from strategies.dip_buyer import DipBuyerStrategy
        strategies.append(DipBuyerStrategy())

    # ── High-alpha quantitative strategies (Avellaneda-Stoikov, Taleb, Gatheral) ──
    if cfg.enable_market_making:
        from strategies.market_making import MarketMakingStrategy
        strategies.append(MarketMakingStrategy())

    if cfg.enable_gamma_scalping:
        from strategies.gamma_scalping import GammaScalpingStrategy
        strategies.append(GammaScalpingStrategy())

    if cfg.enable_vol_surface_arb:
        from strategies.vol_surface_arb import VolSurfaceArbStrategy
        strategies.append(VolSurfaceArbStrategy())

    return strategies
