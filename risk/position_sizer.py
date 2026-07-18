"""
Institutional position sizing.

The live bot should never size from confidence alone.  Each proposed trade is
passed through several independent constraints:

- fractional Kelly edge
- stop-loss dollar risk
- volatility contribution
- remaining portfolio exposure
- per-symbol concentration
- drawdown / daily-loss throttles
- market-regime and execution-quality scalars

The ``size_from_signal`` compatibility wrapper is retained for callers that
expect the old ``(notional, quantity)`` tuple.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
import logging
from math import isfinite, sqrt

import numpy as np

from bot.config import cfg

log = logging.getLogger("risk.position_sizer")


@dataclass(frozen=True)
class SizingContext:
    signal_confidence: float
    equity: float
    price: float
    realised_vol: float = 0.5
    regime: str = "ranging"
    side: str = "BUY"
    market: str = "futures"
    strategy: str = ""
    current_gross_exposure: float = 0.0
    symbol_exposure: float = 0.0
    open_positions: int = 0
    daily_start_equity: float = 0.0
    peak_equity: float = 0.0
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    liquidity_score: float = 1.0
    execution_quality: float = 1.0


@dataclass(frozen=True)
class ExitPlan:
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    reason: str = "vol_regime"

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class SizingDecision:
    notional: float
    quantity: float
    risk_dollars: float
    stop_loss_pct: float
    take_profit_pct: float
    kelly_fraction: float
    vol_target_notional: float
    max_loss_notional: float
    exposure_cap_notional: float
    concentration_cap_notional: float
    drawdown_scalar: float
    daily_loss_scalar: float
    regime_scalar: float
    confidence_scalar: float
    volatility_scalar: float
    liquidity_scalar: float
    execution_scalar: float
    binding_constraint: str
    reason: str

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


class PositionSizer:
    def __init__(self) -> None:
        self.kelly_fraction = cfg.kelly_fraction
        self.max_risk_per_trade = cfg.max_risk_per_trade
        self.max_portfolio_risk = cfg.max_portfolio_risk
        self.min_notional = cfg.min_notional
        self.max_symbol_risk = min(0.08, self.max_portfolio_risk)
        self.max_new_position_risk = min(0.10, self.max_portfolio_risk)
        self.target_trade_vol = 0.08

    # ------------------------------------------------------------------ #
    # Small numeric helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _finite(value: float, default: float = 0.0) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if isfinite(value) else default

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _roundtrip_cost_pct() -> float:
        """
        Expected roundtrip execution cost as a fraction of notional for the
        LIVE venue: 2 taker fees (entry + exit at market) + a 10bps slippage
        buffer. Alpaca crypto: 2x25bps + 10bps = 60bps. Binance futures:
        2x4bps + 10bps = 18bps.
        """
        try:
            from backtester.fill_model import venue_fees

            _, taker = venue_fees(cfg.broker, "spot" if cfg.is_alpaca else "futures")
        except Exception:
            taker = 0.0025  # conservative fallback (Alpaca level)
        return 2.0 * taker + 0.0010

    @staticmethod
    def _zero_decision(reason: str, context: SizingContext | None = None) -> SizingDecision:
        stop = PositionSizer._finite(context.stop_loss_pct, 0.0) if context else 0.0
        take = PositionSizer._finite(context.take_profit_pct, 0.0) if context else 0.0
        return SizingDecision(
            notional=0.0,
            quantity=0.0,
            risk_dollars=0.0,
            stop_loss_pct=stop,
            take_profit_pct=take,
            kelly_fraction=0.0,
            vol_target_notional=0.0,
            max_loss_notional=0.0,
            exposure_cap_notional=0.0,
            concentration_cap_notional=0.0,
            drawdown_scalar=0.0,
            daily_loss_scalar=0.0,
            regime_scalar=0.0,
            confidence_scalar=0.0,
            volatility_scalar=0.0,
            liquidity_scalar=0.0,
            execution_scalar=0.0,
            binding_constraint="blocked",
            reason=reason,
        )

    # ------------------------------------------------------------------ #
    # Dynamic stop/take-profit planning
    # ------------------------------------------------------------------ #

    def exit_plan(
        self,
        realised_vol: float = 0.5,
        regime: str = "ranging",
        confidence: float = 0.55,
    ) -> ExitPlan:
        """
        Recalculate stop and take-profit bands from current volatility/regime.

        ``realised_vol`` is treated as annualised if it is in normal model
        space. The plan is intentionally conservative: high-volatility regimes
        get wider stops but smaller sizing elsewhere in the decision engine.
        """
        vol = self._clamp(abs(self._finite(realised_vol, 0.5)), 0.03, 4.0)
        daily_vol = vol / sqrt(365.0)
        base_stop = self._clamp(1.80 * daily_vol, 0.0125, 0.0650)
        regime_stop_scalar = {
            "bull": 1.05,
            "ranging": 1.00,
            "volatile": 1.25,
            "bear": 0.90,
        }.get(regime, 1.00)
        stop = self._clamp(base_stop * regime_stop_scalar, 0.0100, 0.0800)
        conf = self._clamp(self._finite(confidence, 0.55), 0.0, 1.0)
        payoff = self._clamp(1.45 + conf * 0.90, 1.55, 2.35)
        take = self._clamp(stop * payoff, stop * 1.35, 0.1800)
        trailing = self._clamp(stop * 0.70, 0.0080, 0.0600)
        return ExitPlan(
            stop_loss_pct=round(stop, 5),
            take_profit_pct=round(take, 5),
            trailing_stop_pct=round(trailing, 5),
        )

    # ------------------------------------------------------------------ #
    # Full institutional signal sizing
    # ------------------------------------------------------------------ #

    def size_signal(self, context: SizingContext) -> SizingDecision:
        equity = self._finite(context.equity)
        price = self._finite(context.price)
        confidence = self._clamp(self._finite(context.signal_confidence), 0.0, 1.0)
        realised_vol = self._clamp(abs(self._finite(context.realised_vol, 0.5)), 0.03, 4.0)

        if equity <= 0:
            return self._zero_decision("invalid_equity", context)
        if price <= 0:
            return self._zero_decision("invalid_price", context)
        if confidence <= 0:
            return self._zero_decision("zero_confidence", context)

        plan = self.exit_plan(realised_vol, context.regime, confidence)
        stop_override = self._finite(context.stop_loss_pct, 0.0)
        take_override = self._finite(context.take_profit_pct, 0.0)
        stop = self._clamp(stop_override or plan.stop_loss_pct, 0.0025, 0.20)
        take = self._clamp(take_override or plan.take_profit_pct, stop * 1.01, 0.40)

        # ---- Cost-aware edge floor -------------------------------------- #
        # A trade whose profit target barely covers the roundtrip cost has
        # negative expectancy after the win rate is applied. Require the
        # target to clear COST_EDGE_MULTIPLE x (2 taker fees + slippage);
        # otherwise reject before any capital is committed.
        roundtrip_cost = self._roundtrip_cost_pct()
        min_take = float(getattr(cfg, "cost_edge_multiple", 3.0)) * roundtrip_cost
        if take < min_take:
            return self._zero_decision("edge_below_cost_floor", context)
        # Headroom scaling: thin edges over the floor get proportionally less.
        cost_edge_scalar = self._clamp(take / (2.0 * min_take), 0.5, 1.0)

        regime_scalar = {
            "bull": 1.00,
            "ranging": 0.85,
            "volatile": 0.60,   # raised from 0.45: crypto is routinely volatile
            "bear": 0.65,       # raised from 0.60: bear markets still produce trades
        }.get(context.regime, 0.75)

        # Convex confidence map: weak signals survive as watchlist context but
        # rarely clear min-notional; strong signals receive materially more risk.
        confidence_scalar = self._clamp((confidence - 0.45) / 0.55, 0.0, 1.0) ** 1.35
        if confidence_scalar <= 0:
            return self._zero_decision("confidence_below_edge_floor", context)

        peak_equity = self._finite(context.peak_equity, equity) or equity
        drawdown = max(0.0, (peak_equity - equity) / max(peak_equity, 1e-10))
        if drawdown >= cfg.max_drawdown:
            return self._zero_decision("max_drawdown_reached", context)
        drawdown_scalar = self._clamp(1.0 - drawdown / max(cfg.max_drawdown, 1e-6), 0.15, 1.0)

        daily_start = self._finite(context.daily_start_equity, equity) or equity
        daily_loss = max(0.0, (daily_start - equity) / max(daily_start, 1e-10))
        if daily_loss >= cfg.daily_loss_limit:
            return self._zero_decision("daily_loss_limit_reached", context)
        daily_loss_scalar = self._clamp(1.0 - daily_loss / max(cfg.daily_loss_limit, 1e-6), 0.20, 1.0)

        liquidity_scalar = self._clamp(self._finite(context.liquidity_score, 1.0), 0.10, 1.0)
        execution_scalar = self._clamp(self._finite(context.execution_quality, 1.0), 0.10, 1.0)
        volatility_scalar = self._clamp(0.35 / realised_vol, 0.15, 1.0)
        scalar = (
            regime_scalar
            * confidence_scalar
            * volatility_scalar
            * drawdown_scalar
            * daily_loss_scalar
            * liquidity_scalar
            * execution_scalar
            * cost_edge_scalar
        )

        # Treat confidence as calibrated win probability only after shrinkage
        # toward 50%; this avoids raw model scores creating oversized Kelly bets.
        win_prob = self._clamp(0.50 + (confidence - 0.50) * 0.65, 0.35, 0.68)
        payoff_ratio = max(take / max(stop, 1e-10), 0.10)
        raw_kelly = max(0.0, (win_prob * payoff_ratio - (1.0 - win_prob)) / payoff_ratio)
        kelly_fraction = min(raw_kelly * self.kelly_fraction, self.max_risk_per_trade)

        risk_budget = equity * self.max_risk_per_trade * scalar
        max_loss_notional = risk_budget / max(stop, 1e-10)
        kelly_notional = equity * kelly_fraction * scalar
        vol_target_notional = equity * (self.target_trade_vol / realised_vol) * confidence_scalar * regime_scalar

        gross_exposure = max(0.0, self._finite(context.current_gross_exposure))
        symbol_exposure = max(0.0, self._finite(context.symbol_exposure))
        exposure_cap_notional = max(0.0, equity * self.max_portfolio_risk - gross_exposure)
        concentration_cap_notional = max(0.0, equity * self.max_symbol_risk - symbol_exposure)
        single_trade_cap = equity * self.max_new_position_risk

        constraints = {
            "kelly_edge": kelly_notional,
            "stop_loss_budget": max_loss_notional,
            "vol_target": vol_target_notional,
            "portfolio_exposure": exposure_cap_notional,
            "symbol_concentration": concentration_cap_notional,
            "single_trade_cap": single_trade_cap,
        }
        positive_constraints = {k: v for k, v in constraints.items() if v > 0}
        if len(positive_constraints) != len(constraints):
            return self._zero_decision("no_remaining_risk_budget", context)

        binding_constraint, notional = min(positive_constraints.items(), key=lambda item: item[1])
        notional = round(max(0.0, notional), 2)
        if notional < self.min_notional:
            return self._zero_decision(f"below_min_notional:{binding_constraint}", context)

        quantity = notional / price
        risk_dollars = notional * stop
        return SizingDecision(
            notional=notional,
            quantity=quantity,
            risk_dollars=round(risk_dollars, 2),
            stop_loss_pct=round(stop, 5),
            take_profit_pct=round(take, 5),
            kelly_fraction=round(kelly_fraction, 6),
            vol_target_notional=round(vol_target_notional, 2),
            max_loss_notional=round(max_loss_notional, 2),
            exposure_cap_notional=round(exposure_cap_notional, 2),
            concentration_cap_notional=round(concentration_cap_notional, 2),
            drawdown_scalar=round(drawdown_scalar, 4),
            daily_loss_scalar=round(daily_loss_scalar, 4),
            regime_scalar=round(regime_scalar, 4),
            confidence_scalar=round(confidence_scalar, 4),
            volatility_scalar=round(volatility_scalar, 4),
            liquidity_scalar=round(liquidity_scalar, 4),
            execution_scalar=round(execution_scalar, 4),
            binding_constraint=binding_constraint,
            reason="approved",
        )

    # ------------------------------------------------------------------ #
    # Backwards-compatible signal-confidence-weighted sizing
    # ------------------------------------------------------------------ #

    def size_from_signal(
        self,
        signal_confidence: float,     # 0..1
        equity: float,
        price: float,
        realised_vol: float = 0.5,
        regime: str = "ranging",
    ) -> tuple[float, float]:
        """
        Returns (notional_usdt, quantity).
        Compatibility wrapper around the full institutional sizing decision.
        """
        decision = self.size_signal(
            SizingContext(
                signal_confidence=signal_confidence,
                equity=equity,
                price=price,
                realised_vol=realised_vol,
                regime=regime,
            )
        )
        return decision.notional, decision.quantity

    # ------------------------------------------------------------------ #
    # Risk-parity allocation across strategies
    # ------------------------------------------------------------------ #

    @staticmethod
    def risk_parity_weights(vols: np.ndarray) -> np.ndarray:
        """
        Inverse-vol weights. vols: per-strategy realised volatilities.
        Returns weights that sum to 1.
        """
        inv = 1.0 / (vols + 1e-10)
        return inv / inv.sum()

    # ------------------------------------------------------------------ #
    # Leverage calculation
    # ------------------------------------------------------------------ #

    def safe_leverage(self, realised_vol: float, regime: str = "ranging") -> int:
        """Return safe leverage cap given current volatility and regime."""
        max_lev = cfg.max_leverage
        regime_cap = {"bull": max_lev, "ranging": max_lev // 2, "volatile": 3, "bear": 5}.get(regime, 5)
        # realised_vol is annualised; convert to daily before sizing leverage.
        # vol_cap = target_daily_vol / daily_vol, where target_daily_vol = 20%.
        daily_vol = max(realised_vol / sqrt(365), 1e-4)
        vol_cap = max(1, int(0.20 / daily_vol))  # target 20% daily vol at 1x
        return min(regime_cap, vol_cap, max_lev)

    # ------------------------------------------------------------------ #
    # Quantity precision
    # ------------------------------------------------------------------ #

    @staticmethod
    def round_step(quantity: float, step_size: float) -> float:
        if step_size <= 0:
            return quantity
        return round(quantity // step_size * step_size, 10)
