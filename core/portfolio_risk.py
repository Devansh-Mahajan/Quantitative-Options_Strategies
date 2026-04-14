from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np
from alpaca.trading.enums import AssetClass

from config.params import (
    OPTION_PRICING_RISK_FREE_RATE,
    PORTFOLIO_RISK_CONFIDENCE,
    PORTFOLIO_RISK_ENGINE_ENABLED,
    PORTFOLIO_RISK_HARD_KILL_CVAR_PCT_EQUITY,
    PORTFOLIO_RISK_HARD_KILL_DAILY_PNL_PCT,
    PORTFOLIO_RISK_HARD_KILL_STRESS_PCT_EQUITY,
    PORTFOLIO_RISK_HISTORY_CACHE_SECONDS,
    PORTFOLIO_RISK_HORIZON_DAYS,
    PORTFOLIO_RISK_MAX_CORRELATION_CONCENTRATION,
    PORTFOLIO_RISK_MAX_CVAR_PCT_EQUITY,
    PORTFOLIO_RISK_MAX_GROSS_EXPOSURE_PCT_EQUITY,
    PORTFOLIO_RISK_MAX_SINGLE_UNDERLYING_WEIGHT,
    PORTFOLIO_RISK_MAX_STRESS_PCT_EQUITY,
    PORTFOLIO_RISK_MAX_VAR_PCT_EQUITY,
    PORTFOLIO_RISK_MC_PATHS,
    PORTFOLIO_RISK_STUDENT_T_DF,
    SWEEP_TICKER,
)
from core.quant_models import black_scholes_price
from core.system_telemetry import DEFAULT_PORTFOLIO_RISK_GUARD_SNAPSHOT_PATH, write_risk_snapshot
from core.utils import get_option_days_to_expiry, try_parse_option_symbol

logger = logging.getLogger(f"risk.{__name__}")


@dataclass(frozen=True)
class PortfolioTradeLeg:
    symbol: str
    side: str
    quantity: float = 1.0


@dataclass(frozen=True)
class PortfolioInstrument:
    symbol: str
    underlying: str
    instrument_type: str
    quantity: float
    spot: float
    volatility: float
    option_type: str | None = None
    strike: float | None = None
    years_to_expiry: float | None = None
    cash_equivalent: bool = False


@dataclass
class PortfolioRiskSnapshot:
    portfolio_equity: float
    modeled_portfolio_value: float
    expected_value: float
    var_95: float
    cvar_95: float
    value_volatility: float
    worst_case: float
    best_case: float
    stress_losses: dict[str, float]
    worst_stress_loss: float
    max_underlying_weight: float
    correlation_concentration: float
    gross_exposure: float
    gross_exposure_pct_equity: float
    net_delta_exposure: float
    top_underlyings: list[dict[str, float]]
    breaches: list[str] = field(default_factory=list)
    risk_score: float = 0.0
    daily_pnl_pct: float = 0.0
    kill_switch_active: bool = False
    hard_kill_reasons: list[str] = field(default_factory=list)
    underlying_count: int = 0
    simulation_paths: int = 0
    horizon_days: float = 0.0
    confidence: float = 0.95

    @property
    def var_pct_equity(self) -> float:
        return float(self.var_95 / self.portfolio_equity) if self.portfolio_equity > 0 else 0.0

    @property
    def cvar_pct_equity(self) -> float:
        return float(self.cvar_95 / self.portfolio_equity) if self.portfolio_equity > 0 else 0.0

    @property
    def stress_pct_equity(self) -> float:
        return float(self.worst_stress_loss / self.portfolio_equity) if self.portfolio_equity > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "portfolio_equity": round(float(self.portfolio_equity), 2),
            "modeled_portfolio_value": round(float(self.modeled_portfolio_value), 2),
            "expected_value": round(float(self.expected_value), 2),
            "var_95": round(float(self.var_95), 2),
            "cvar_95": round(float(self.cvar_95), 2),
            "var_pct_equity": round(float(self.var_pct_equity), 6),
            "cvar_pct_equity": round(float(self.cvar_pct_equity), 6),
            "value_volatility": round(float(self.value_volatility), 2),
            "worst_case": round(float(self.worst_case), 2),
            "best_case": round(float(self.best_case), 2),
            "stress_losses": {name: round(float(value), 2) for name, value in self.stress_losses.items()},
            "worst_stress_loss": round(float(self.worst_stress_loss), 2),
            "stress_pct_equity": round(float(self.stress_pct_equity), 6),
            "max_underlying_weight": round(float(self.max_underlying_weight), 6),
            "correlation_concentration": round(float(self.correlation_concentration), 6),
            "gross_exposure": round(float(self.gross_exposure), 2),
            "gross_exposure_pct_equity": round(float(self.gross_exposure_pct_equity), 6),
            "net_delta_exposure": round(float(self.net_delta_exposure), 2),
            "top_underlyings": [
                {
                    "symbol": str(item.get("symbol") or ""),
                    "weight": round(float(item.get("weight", 0.0)), 6),
                    "delta_notional": round(float(item.get("delta_notional", 0.0)), 2),
                }
                for item in self.top_underlyings
            ],
            "breaches": list(self.breaches),
            "risk_score": round(float(self.risk_score), 6),
            "daily_pnl_pct": round(float(self.daily_pnl_pct), 6),
            "kill_switch_active": bool(self.kill_switch_active),
            "hard_kill_reasons": list(self.hard_kill_reasons),
            "underlying_count": int(self.underlying_count),
            "simulation_paths": int(self.simulation_paths),
            "horizon_days": round(float(self.horizon_days), 4),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class PortfolioRiskDecision:
    allowed: bool
    reason: str
    order_label: str
    reduce_only: bool
    risk_reducing: bool
    current: PortfolioRiskSnapshot
    projected: PortfolioRiskSnapshot

    def to_dict(self) -> dict:
        return {
            "allowed": bool(self.allowed),
            "reason": str(self.reason),
            "order_label": str(self.order_label),
            "reduce_only": bool(self.reduce_only),
            "risk_reducing": bool(self.risk_reducing),
            "current": self.current.to_dict(),
            "projected": self.projected.to_dict(),
        }


class PortfolioRiskBlockedError(RuntimeError):
    def __init__(self, decision: PortfolioRiskDecision):
        self.decision = decision
        super().__init__(f"Portfolio risk gate blocked {decision.order_label}: {decision.reason}")


class PortfolioRiskEngine:
    def __init__(
        self,
        client,
        *,
        price_history_provider=None,
        guard_snapshot_path=DEFAULT_PORTFOLIO_RISK_GUARD_SNAPSHOT_PATH,
    ):
        self.client = client
        self.price_history_provider = price_history_provider
        self.guard_snapshot_path = guard_snapshot_path
        self._history_cache: dict[str, tuple[float, np.ndarray]] = {}

    def build_snapshot(self, *, positions=None, account=None) -> PortfolioRiskSnapshot:
        positions = list(positions if positions is not None else self.client.trade_client.get_all_positions())
        account = account or self.client.trade_client.get_account()
        equity = _safe_float(
            getattr(account, "equity", None),
            fallback=_safe_float(getattr(account, "portfolio_value", None), fallback=0.0),
        )
        daily_pnl_pct = _daily_pnl_pct(account)
        instruments = self._build_instruments(positions)
        if not instruments:
            snapshot = PortfolioRiskSnapshot(
                portfolio_equity=equity,
                modeled_portfolio_value=0.0,
                expected_value=0.0,
                var_95=0.0,
                cvar_95=0.0,
                value_volatility=0.0,
                worst_case=0.0,
                best_case=0.0,
                stress_losses={},
                worst_stress_loss=0.0,
                max_underlying_weight=0.0,
                correlation_concentration=0.0,
                gross_exposure=0.0,
                gross_exposure_pct_equity=0.0,
                net_delta_exposure=0.0,
                top_underlyings=[],
                daily_pnl_pct=daily_pnl_pct,
                underlying_count=0,
                simulation_paths=int(PORTFOLIO_RISK_MC_PATHS),
                horizon_days=float(PORTFOLIO_RISK_HORIZON_DAYS),
                confidence=float(PORTFOLIO_RISK_CONFIDENCE),
            )
            self._annotate_limits(snapshot)
            return snapshot

        modeled_value = sum(self._instrument_value(instrument) for instrument in instruments)
        exposure_map = self._build_delta_exposure_map(instruments)
        gross_exposure = float(sum(abs(value) for value in exposure_map.values()))
        max_weight, correlation_concentration, top_underlyings = self._concentration_metrics(exposure_map)

        risky_underlyings = sorted({instrument.underlying for instrument in instruments if not instrument.cash_equivalent})
        stress_losses: dict[str, float] = {}
        expected_value = modeled_value
        var_95 = 0.0
        cvar_95 = 0.0
        value_volatility = 0.0
        worst_case = modeled_value
        best_case = modeled_value
        if risky_underlyings:
            simulated_values = self._simulate_portfolio_values(
                instruments,
                risky_underlyings,
                horizon_days=float(PORTFOLIO_RISK_HORIZON_DAYS),
                n_simulations=int(PORTFOLIO_RISK_MC_PATHS),
            )
            if len(simulated_values):
                pnl = simulated_values - modeled_value
                downside_cut = float(np.quantile(pnl, 1.0 - float(PORTFOLIO_RISK_CONFIDENCE)))
                downside_tail = pnl[pnl <= downside_cut]
                expected_value = float(np.mean(simulated_values))
                var_95 = max(0.0, -downside_cut)
                cvar_95 = max(0.0, -float(np.mean(downside_tail))) if len(downside_tail) else var_95
                value_volatility = float(np.std(simulated_values))
                worst_case = float(np.min(simulated_values))
                best_case = float(np.max(simulated_values))
            stress_losses = self._run_stress_scenarios(instruments, risky_underlyings, modeled_value)

        snapshot = PortfolioRiskSnapshot(
            portfolio_equity=equity,
            modeled_portfolio_value=float(modeled_value),
            expected_value=float(expected_value),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            value_volatility=float(value_volatility),
            worst_case=float(worst_case),
            best_case=float(best_case),
            stress_losses=stress_losses,
            worst_stress_loss=float(max(stress_losses.values()) if stress_losses else 0.0),
            max_underlying_weight=float(max_weight),
            correlation_concentration=float(correlation_concentration),
            gross_exposure=float(gross_exposure),
            gross_exposure_pct_equity=(float(gross_exposure) / equity) if equity > 0 else 0.0,
            net_delta_exposure=float(sum(exposure_map.values())),
            top_underlyings=top_underlyings,
            daily_pnl_pct=float(daily_pnl_pct),
            underlying_count=len(risky_underlyings),
            simulation_paths=int(PORTFOLIO_RISK_MC_PATHS),
            horizon_days=float(PORTFOLIO_RISK_HORIZON_DAYS),
            confidence=float(PORTFOLIO_RISK_CONFIDENCE),
        )
        self._annotate_limits(snapshot)
        return snapshot

    def assess_trade(self, order_label: str, trade_legs: list[PortfolioTradeLeg]) -> PortfolioRiskDecision:
        positions = list(self.client.trade_client.get_all_positions())
        account = self.client.trade_client.get_account()
        current_snapshot = self.build_snapshot(positions=positions, account=account)
        if not PORTFOLIO_RISK_ENGINE_ENABLED:
            decision = PortfolioRiskDecision(
                allowed=True,
                reason="Portfolio risk engine disabled.",
                order_label=order_label,
                reduce_only=False,
                risk_reducing=False,
                current=current_snapshot,
                projected=current_snapshot,
            )
            self._write_guard_snapshot("pre-trade", decision=decision)
            return decision

        current_quantities = self._position_quantities(positions)
        projected_positions = self._apply_trade_legs(positions, trade_legs)
        projected_snapshot = self.build_snapshot(positions=projected_positions, account=account)
        projected_quantities = self._position_quantities(projected_positions)
        reduce_only = _is_reduce_only(current_quantities, projected_quantities, trade_legs)
        current_score = self._risk_pressure_score(current_snapshot)
        projected_score = self._risk_pressure_score(projected_snapshot)
        risk_reducing = projected_score < (current_score - 0.01)
        if projected_snapshot.cvar_pct_equity < current_snapshot.cvar_pct_equity - 0.0025:
            risk_reducing = True
        if projected_snapshot.stress_pct_equity < current_snapshot.stress_pct_equity - 0.0025:
            risk_reducing = True

        allowed = True
        reason = "Projected portfolio remains within configured limits."
        if reduce_only:
            reason = "Trade reduces existing exposure."
        elif current_snapshot.kill_switch_active and not risk_reducing:
            allowed = False
            reason = f"Hard kill switch active ({', '.join(current_snapshot.hard_kill_reasons)})."
        else:
            current_breaches = set(current_snapshot.breaches)
            projected_breaches = set(projected_snapshot.breaches)
            if not current_breaches and projected_breaches:
                allowed = False
                reason = f"Projected breach: {', '.join(sorted(projected_breaches))}."
            elif current_breaches and not risk_reducing:
                allowed = False
                reason = (
                    "Portfolio is already above risk limits; only risk-reducing trades are allowed until pressure comes down."
                )
            elif current_breaches and risk_reducing:
                reason = "Trade reduces portfolio risk while existing breaches are being worked down."
            elif projected_score > current_score + 0.10 and projected_breaches:
                allowed = False
                reason = "Projected trade materially worsens Monte Carlo or stress loss pressure."

        decision = PortfolioRiskDecision(
            allowed=allowed,
            reason=reason,
            order_label=order_label,
            reduce_only=reduce_only,
            risk_reducing=risk_reducing,
            current=current_snapshot,
            projected=projected_snapshot,
        )
        self._write_guard_snapshot("pre-trade", decision=decision)
        return decision

    def record_post_trade_snapshot(self, *, order_label: str, phase: str = "post-trade") -> PortfolioRiskSnapshot:
        snapshot = self.build_snapshot()
        self._write_guard_snapshot(
            phase,
            payload={
                "order_label": order_label,
                "portfolio_risk_engine": snapshot.to_dict(),
            },
        )
        return snapshot

    def _build_instruments(self, positions) -> list[PortfolioInstrument]:
        quantity_map: dict[str, float] = {}
        asset_map: dict[str, object] = {}
        for position in positions:
            symbol = str(getattr(position, "symbol", "") or "").strip().upper()
            qty = _safe_float(getattr(position, "qty", None))
            if not symbol or abs(qty) <= 1e-9:
                continue
            quantity_map[symbol] = quantity_map.get(symbol, 0.0) + qty
            asset_map[symbol] = getattr(position, "asset_class", None)

        underlyings = self._required_underlyings(quantity_map.keys())
        spot_map = {symbol: self._spot_for_underlying(symbol) for symbol in underlyings}
        vol_map = {symbol: self._volatility_for_underlying(symbol) for symbol in underlyings}

        instruments: list[PortfolioInstrument] = []
        for symbol, qty in quantity_map.items():
            if abs(qty) <= 1e-9:
                continue
            parsed = try_parse_option_symbol(symbol)
            cash_equivalent = symbol == SWEEP_TICKER
            if parsed is not None:
                underlying, option_type, strike = parsed
                underlying_spot = max(0.01, float(spot_map.get(underlying, strike or 1.0)))
                years_to_expiry = max(1.0 / 365.0, float(get_option_days_to_expiry(symbol)) / 365.0)
                instruments.append(
                    PortfolioInstrument(
                        symbol=symbol,
                        underlying=underlying,
                        instrument_type="option",
                        quantity=float(qty),
                        spot=underlying_spot,
                        volatility=max(0.08, float(vol_map.get(underlying, 0.30))),
                        option_type=option_type,
                        strike=float(strike),
                        years_to_expiry=float(years_to_expiry),
                    )
                )
                continue

            asset_class = asset_map.get(symbol)
            if asset_class not in (None, AssetClass.US_EQUITY, "us_equity"):
                continue
            spot = max(0.01, float(spot_map.get(symbol, self._spot_for_underlying(symbol))))
            volatility = 0.03 if cash_equivalent else max(0.08, float(vol_map.get(symbol, 0.25)))
            instruments.append(
                PortfolioInstrument(
                    symbol=symbol,
                    underlying=symbol,
                    instrument_type="equity",
                    quantity=float(qty),
                    spot=spot,
                    volatility=volatility,
                    cash_equivalent=cash_equivalent,
                )
            )
        return instruments

    def _required_underlyings(self, symbols) -> set[str]:
        underlyings: set[str] = set()
        for symbol in symbols:
            parsed = try_parse_option_symbol(str(symbol))
            if parsed is None:
                underlyings.add(str(symbol).upper())
            else:
                underlyings.add(parsed[0])
        return underlyings

    def _position_quantities(self, positions) -> dict[str, float]:
        quantities: dict[str, float] = {}
        for position in positions:
            symbol = str(getattr(position, "symbol", "") or "").strip().upper()
            qty = _safe_float(getattr(position, "qty", None))
            if symbol:
                quantities[symbol] = quantities.get(symbol, 0.0) + qty
        return quantities

    def _apply_trade_legs(self, positions, trade_legs: list[PortfolioTradeLeg]):
        position_map = {str(getattr(position, "symbol", "")).upper(): position for position in positions}
        quantity_map = self._position_quantities(positions)
        for leg in trade_legs:
            symbol = str(leg.symbol).upper()
            signed_qty = abs(float(leg.quantity or 0.0))
            if str(leg.side).lower() == "sell":
                signed_qty *= -1.0
            quantity_map[symbol] = quantity_map.get(symbol, 0.0) + signed_qty

        projected = []
        for symbol, qty in quantity_map.items():
            if abs(qty) <= 1e-9:
                continue
            if symbol in position_map:
                original = position_map[symbol]
                payload = {
                    key: value
                    for key, value in getattr(original, "__dict__", {}).items()
                    if key != "qty"
                }
                payload.setdefault("symbol", symbol)
                payload.setdefault("asset_class", getattr(original, "asset_class", None))
                payload["qty"] = str(qty)
                projected.append(type("ProjectedPosition", (), payload)())
            else:
                asset_class = AssetClass.US_OPTION if try_parse_option_symbol(symbol) is not None else AssetClass.US_EQUITY
                projected.append(
                    type(
                        "ProjectedPosition",
                        (),
                        {
                            "symbol": symbol,
                            "qty": str(qty),
                            "asset_class": asset_class,
                        },
                    )()
                )
        return projected

    def _spot_for_underlying(self, symbol: str) -> float:
        symbol = str(symbol).upper()
        try:
            latest_trade = self.client.get_stock_latest_trade(symbol)
            if isinstance(latest_trade, dict):
                trade = latest_trade.get(symbol)
                if trade is not None and getattr(trade, "price", None) is not None:
                    return float(trade.price)
        except Exception:
            pass

        history = self._history_for_symbol(symbol)
        if len(history):
            return float(history[-1])
        return 100.0 if symbol != SWEEP_TICKER else 100.0

    def _volatility_for_underlying(self, symbol: str) -> float:
        symbol = str(symbol).upper()
        if symbol == SWEEP_TICKER:
            return 0.03
        history = self._history_for_symbol(symbol)
        returns = _close_to_returns(history)
        if len(returns) >= 20:
            return float(np.clip(np.std(returns) * math.sqrt(252.0), 0.08, 2.50))
        return 0.30

    def _history_for_symbol(self, symbol: str) -> np.ndarray:
        symbol = str(symbol).upper()
        now = time.monotonic()
        cached = self._history_cache.get(symbol)
        if cached and (now - cached[0]) <= float(PORTFOLIO_RISK_HISTORY_CACHE_SECONDS):
            return cached[1]

        history = np.array([], dtype=float)
        if self.price_history_provider is not None:
            try:
                raw = self.price_history_provider(symbol)
                history = np.asarray(raw if raw is not None else [], dtype=float)
            except Exception:
                history = np.array([], dtype=float)
        else:
            try:
                import yfinance as yf

                frame = yf.Ticker(symbol).history(period="1y")
                if "Close" in frame:
                    history = np.asarray(frame["Close"].dropna().tolist(), dtype=float)
            except Exception:
                history = np.array([], dtype=float)

        self._history_cache[symbol] = (now, history)
        return history

    def _simulate_portfolio_values(
        self,
        instruments: list[PortfolioInstrument],
        risky_underlyings: list[str],
        *,
        horizon_days: float,
        n_simulations: int,
    ) -> np.ndarray:
        if not risky_underlyings:
            return np.array([], dtype=float)

        spot_vector = np.array([max(0.01, self._spot_for_underlying(symbol)) for symbol in risky_underlyings], dtype=float)
        vol_vector = np.array([max(0.05, self._volatility_for_underlying(symbol)) for symbol in risky_underlyings], dtype=float)
        corr = self._correlation_matrix(risky_underlyings)
        terminal_spots = _simulate_correlated_spots(
            spot_vector,
            vol_vector,
            corr,
            horizon_years=max(1.0 / 252.0, float(horizon_days) / 252.0),
            n_simulations=max(200, int(n_simulations)),
        )

        underlying_index = {symbol: idx for idx, symbol in enumerate(risky_underlyings)}
        simulated_values = np.zeros(len(terminal_spots), dtype=float)
        for instrument in instruments:
            if instrument.cash_equivalent:
                simulated_values += instrument.quantity * instrument.spot
                continue
            if instrument.instrument_type == "equity":
                prices = terminal_spots[:, underlying_index[instrument.underlying]]
                simulated_values += instrument.quantity * prices
                continue

            prices = terminal_spots[:, underlying_index[instrument.underlying]]
            remaining_years = max(1.0 / 365.0, float(instrument.years_to_expiry or 0.0) - (float(horizon_days) / 252.0))
            option_values = np.array(
                [
                    black_scholes_price(
                        "c" if str(instrument.option_type).upper().startswith("C") else "p",
                        max(0.01, float(spot)),
                        float(instrument.strike or 0.0),
                        remaining_years,
                        OPTION_PRICING_RISK_FREE_RATE,
                        max(0.05, float(instrument.volatility)),
                    )
                    for spot in prices
                ],
                dtype=float,
            )
            simulated_values += instrument.quantity * 100.0 * option_values
        return simulated_values

    def _run_stress_scenarios(
        self,
        instruments: list[PortfolioInstrument],
        risky_underlyings: list[str],
        baseline_value: float,
    ) -> dict[str, float]:
        if not risky_underlyings:
            return {}

        spot_map = {symbol: self._spot_for_underlying(symbol) for symbol in risky_underlyings}
        scenarios = {
            "market_down_5": {"spot_mult": 0.95, "vol_mult": 1.10},
            "market_down_10": {"spot_mult": 0.90, "vol_mult": 1.20},
            "gap_crash_15": {"spot_mult": 0.85, "vol_mult": 1.35},
            "market_up_5": {"spot_mult": 1.05, "vol_mult": 0.95},
        }

        losses = {}
        for name, spec in scenarios.items():
            shocked_value = self._scenario_value(
                instruments,
                {
                    symbol: max(0.01, spot * float(spec["spot_mult"]))
                    for symbol, spot in spot_map.items()
                },
                vol_multiplier=float(spec["vol_mult"]),
            )
            losses[name] = max(0.0, baseline_value - shocked_value)

        single_name_losses = []
        for stressed in risky_underlyings:
            shocked_spots = dict(spot_map)
            shocked_spots[stressed] = max(0.01, shocked_spots[stressed] * 0.88)
            single_name_losses.append(
                max(0.0, baseline_value - self._scenario_value(instruments, shocked_spots, vol_multiplier=1.20, stressed_symbol=stressed))
            )
        if single_name_losses:
            losses["single_name_gap_12"] = max(single_name_losses)
        return losses

    def _scenario_value(
        self,
        instruments: list[PortfolioInstrument],
        shocked_spots: dict[str, float],
        *,
        vol_multiplier: float,
        stressed_symbol: str | None = None,
    ) -> float:
        total = 0.0
        for instrument in instruments:
            if instrument.cash_equivalent:
                total += instrument.quantity * instrument.spot
                continue
            if instrument.instrument_type == "equity":
                total += instrument.quantity * max(0.01, float(shocked_spots.get(instrument.underlying, instrument.spot)))
                continue

            stress_bump = vol_multiplier if stressed_symbol in (None, instrument.underlying) else 1.05
            total += instrument.quantity * 100.0 * black_scholes_price(
                "c" if str(instrument.option_type).upper().startswith("C") else "p",
                max(0.01, float(shocked_spots.get(instrument.underlying, instrument.spot))),
                float(instrument.strike or 0.0),
                max(1.0 / 365.0, float(instrument.years_to_expiry or 0.0)),
                OPTION_PRICING_RISK_FREE_RATE,
                max(0.05, float(instrument.volatility) * float(stress_bump)),
            )
        return float(total)

    def _build_delta_exposure_map(self, instruments: list[PortfolioInstrument]) -> dict[str, float]:
        exposure_map: dict[str, float] = {}
        for instrument in instruments:
            if instrument.cash_equivalent:
                continue
            if instrument.instrument_type == "equity":
                exposure = float(instrument.quantity) * float(instrument.spot)
            else:
                exposure = float(instrument.quantity) * 100.0 * float(instrument.spot) * _black_scholes_delta(
                    "c" if str(instrument.option_type).upper().startswith("C") else "p",
                    float(instrument.spot),
                    float(instrument.strike or 0.0),
                    float(instrument.years_to_expiry or 0.0),
                    OPTION_PRICING_RISK_FREE_RATE,
                    float(instrument.volatility),
                )
            exposure_map[instrument.underlying] = exposure_map.get(instrument.underlying, 0.0) + exposure
        return exposure_map

    def _correlation_matrix(self, risky_underlyings: list[str]) -> np.ndarray:
        count = len(risky_underlyings)
        if count <= 1:
            return np.eye(max(1, count), dtype=float)

        matrix = np.eye(count, dtype=float)
        returns_map = {symbol: _close_to_returns(self._history_for_symbol(symbol)) for symbol in risky_underlyings}
        for i, left in enumerate(risky_underlyings):
            for j in range(i + 1, count):
                right = risky_underlyings[j]
                corr = _estimate_pair_correlation(left, right, returns_map.get(left), returns_map.get(right))
                matrix[i, j] = corr
                matrix[j, i] = corr
        return _nearest_psd_correlation(matrix)

    def _concentration_metrics(self, exposure_map: dict[str, float]) -> tuple[float, float, list[dict[str, float]]]:
        if not exposure_map:
            return 0.0, 0.0, []
        total_abs = float(sum(abs(value) for value in exposure_map.values()))
        if total_abs <= 0:
            return 0.0, 0.0, []

        symbols = list(exposure_map.keys())
        weights = np.array([abs(float(exposure_map[symbol])) / total_abs for symbol in symbols], dtype=float)
        correlation_concentration = float(weights @ self._correlation_matrix(symbols) @ weights)
        top_underlyings = [
            {
                "symbol": symbol,
                "weight": abs(float(exposure_map[symbol])) / total_abs,
                "delta_notional": float(exposure_map[symbol]),
            }
            for symbol in sorted(symbols, key=lambda item: abs(exposure_map[item]), reverse=True)[:5]
        ]
        return float(np.max(weights)), correlation_concentration, top_underlyings

    def _risk_pressure_score(self, snapshot: PortfolioRiskSnapshot) -> float:
        var_component = snapshot.var_pct_equity / max(1e-6, float(PORTFOLIO_RISK_MAX_VAR_PCT_EQUITY))
        cvar_component = snapshot.cvar_pct_equity / max(1e-6, float(PORTFOLIO_RISK_MAX_CVAR_PCT_EQUITY))
        stress_component = snapshot.stress_pct_equity / max(1e-6, float(PORTFOLIO_RISK_MAX_STRESS_PCT_EQUITY))
        weight_component = snapshot.max_underlying_weight / max(1e-6, float(PORTFOLIO_RISK_MAX_SINGLE_UNDERLYING_WEIGHT))
        corr_component = snapshot.correlation_concentration / max(1e-6, float(PORTFOLIO_RISK_MAX_CORRELATION_CONCENTRATION))
        gross_component = snapshot.gross_exposure_pct_equity / max(1e-6, float(PORTFOLIO_RISK_MAX_GROSS_EXPOSURE_PCT_EQUITY))
        return float(
            (0.24 * var_component)
            + (0.28 * cvar_component)
            + (0.20 * stress_component)
            + (0.12 * weight_component)
            + (0.08 * corr_component)
            + (0.08 * gross_component)
        )

    def _annotate_limits(self, snapshot: PortfolioRiskSnapshot) -> None:
        breaches = []
        if snapshot.var_pct_equity > float(PORTFOLIO_RISK_MAX_VAR_PCT_EQUITY):
            breaches.append("portfolio_var")
        if snapshot.cvar_pct_equity > float(PORTFOLIO_RISK_MAX_CVAR_PCT_EQUITY):
            breaches.append("portfolio_cvar")
        if snapshot.stress_pct_equity > float(PORTFOLIO_RISK_MAX_STRESS_PCT_EQUITY):
            breaches.append("stress_loss")
        if snapshot.max_underlying_weight > float(PORTFOLIO_RISK_MAX_SINGLE_UNDERLYING_WEIGHT):
            breaches.append("single_underlying_concentration")
        if snapshot.correlation_concentration > float(PORTFOLIO_RISK_MAX_CORRELATION_CONCENTRATION):
            breaches.append("correlation_concentration")
        if snapshot.gross_exposure_pct_equity > float(PORTFOLIO_RISK_MAX_GROSS_EXPOSURE_PCT_EQUITY):
            breaches.append("gross_exposure")

        hard_kill_reasons = []
        if snapshot.daily_pnl_pct <= float(PORTFOLIO_RISK_HARD_KILL_DAILY_PNL_PCT):
            hard_kill_reasons.append("daily_drawdown")
        if snapshot.cvar_pct_equity > float(PORTFOLIO_RISK_HARD_KILL_CVAR_PCT_EQUITY):
            hard_kill_reasons.append("cvar")
        if snapshot.stress_pct_equity > float(PORTFOLIO_RISK_HARD_KILL_STRESS_PCT_EQUITY):
            hard_kill_reasons.append("stress")

        snapshot.breaches = breaches
        snapshot.hard_kill_reasons = hard_kill_reasons
        snapshot.kill_switch_active = bool(hard_kill_reasons)
        snapshot.risk_score = self._risk_pressure_score(snapshot)

    def _write_guard_snapshot(self, phase: str, *, decision: PortfolioRiskDecision | None = None, payload: dict | None = None) -> None:
        guard_payload = {
            "phase": str(phase),
        }
        if payload:
            guard_payload.update(payload)
        if decision is not None:
            guard_payload["decision"] = decision.to_dict()
        try:
            write_risk_snapshot(self.guard_snapshot_path, guard_payload)
        except Exception as exc:
            logger.debug("Could not write portfolio risk guard snapshot: %s", exc)

    def _instrument_value(self, instrument: PortfolioInstrument) -> float:
        if instrument.cash_equivalent:
            return float(instrument.quantity) * float(instrument.spot)
        if instrument.instrument_type == "equity":
            return float(instrument.quantity) * float(instrument.spot)
        return float(instrument.quantity) * 100.0 * black_scholes_price(
            "c" if str(instrument.option_type).upper().startswith("C") else "p",
            max(0.01, float(instrument.spot)),
            float(instrument.strike or 0.0),
            max(1.0 / 365.0, float(instrument.years_to_expiry or 0.0)),
            OPTION_PRICING_RISK_FREE_RATE,
            max(0.05, float(instrument.volatility)),
        )


def _safe_float(value, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _daily_pnl_pct(account) -> float:
    equity = _safe_float(getattr(account, "equity", None), fallback=_safe_float(getattr(account, "portfolio_value", None), fallback=0.0))
    last_equity = _safe_float(getattr(account, "last_equity", None), fallback=equity)
    if last_equity <= 0:
        return 0.0
    return ((equity - last_equity) / last_equity) * 100.0


def _close_to_returns(history: np.ndarray) -> np.ndarray:
    if history is None or len(history) < 2:
        return np.array([], dtype=float)
    clipped = np.asarray(history, dtype=float)
    clipped = clipped[np.isfinite(clipped)]
    if len(clipped) < 2:
        return np.array([], dtype=float)
    return np.diff(clipped) / np.clip(clipped[:-1], 1e-6, None)


def _simulate_correlated_spots(
    spot_vector: np.ndarray,
    vol_vector: np.ndarray,
    correlation: np.ndarray,
    *,
    horizon_years: float,
    n_simulations: int,
) -> np.ndarray:
    rng = np.random.default_rng(42)
    count = len(spot_vector)
    df = max(3, int(PORTFOLIO_RISK_STUDENT_T_DF))
    gaussian = rng.standard_normal((max(1, int(n_simulations)), count))
    chi2 = rng.chisquare(df=df, size=max(1, int(n_simulations)))
    student_scale = np.sqrt(df / np.clip(chi2, 1e-6, None))[:, None] * math.sqrt((df - 2) / df)
    correlated = (gaussian @ np.linalg.cholesky(correlation).T) * student_scale
    drift = (OPTION_PRICING_RISK_FREE_RATE - 0.5 * (vol_vector ** 2)) * float(horizon_years)
    diffusion = vol_vector * math.sqrt(max(1e-6, float(horizon_years))) * correlated
    return spot_vector * np.exp(drift + diffusion)


def _nearest_psd_correlation(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 1e-6, None)
    repaired = eigvecs @ np.diag(eigvals) @ eigvecs.T
    diagonal = np.sqrt(np.clip(np.diag(repaired), 1e-6, None))
    repaired = repaired / np.outer(diagonal, diagonal)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def _estimate_pair_correlation(left: str, right: str, left_returns: np.ndarray | None, right_returns: np.ndarray | None) -> float:
    if left == SWEEP_TICKER or right == SWEEP_TICKER:
        return 0.05

    if left_returns is not None and right_returns is not None:
        count = min(len(left_returns), len(right_returns), 252)
        if count >= 20:
            corr = np.corrcoef(left_returns[-count:], right_returns[-count:])[0, 1]
            if np.isfinite(corr):
                return float(np.clip(corr, -0.80, 0.98))

    broad = {"SPY", "QQQ", "IWM", "DIA"}
    high_beta = {"SMH", "SOXL", "TSLA", "NVDA", "AMD", "PLTR", "MSTR", "COIN", "TSLL"}
    if left in broad and right in broad:
        return 0.82
    if left in high_beta and right in high_beta:
        return 0.66
    if (left in broad and right in high_beta) or (left in high_beta and right in broad):
        return 0.58
    return 0.32


def _black_scholes_delta(flag: str, spot: float, strike: float, years_to_expiry: float, risk_free_rate: float, volatility: float) -> float:
    if spot <= 0 or strike <= 0 or years_to_expiry <= 0 or volatility <= 0:
        if flag == "c":
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    variance_term = max(volatility * math.sqrt(years_to_expiry), 1e-9)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility * volatility) * years_to_expiry) / variance_term
    if flag == "c":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _is_reduce_only(current_quantities: dict[str, float], projected_quantities: dict[str, float], trade_legs: list[PortfolioTradeLeg]) -> bool:
    touched = {str(leg.symbol).upper() for leg in trade_legs}
    if not touched:
        return False
    any_reduction = False
    for symbol in touched:
        before = abs(float(current_quantities.get(symbol, 0.0)))
        after = abs(float(projected_quantities.get(symbol, 0.0)))
        if after > before + 1e-9:
            return False
        if after < before - 1e-9:
            any_reduction = True
    return any_reduction
