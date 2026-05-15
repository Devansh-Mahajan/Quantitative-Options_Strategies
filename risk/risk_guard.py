"""
Hard circuit breakers: drawdown halt, daily loss limit, leverage cap,
concentration limit. Returns filtered/scaled signals.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import date, timezone, datetime

from bot.config import cfg
from bot import state

log = logging.getLogger("risk.risk_guard")


@dataclass
class GuardDecision:
    allowed: bool
    reason: str = ""
    scale: float = 1.0   # scale applied to position size if allowed but reduced


class RiskGuard:
    def __init__(self) -> None:
        self._halted: bool = False
        self._halt_reason: str = ""
        self._daily_date: date | None = None
        self._daily_pnl_start_equity: float | None = None
        self._current_drawdown: float = 0.0

    # ------------------------------------------------------------------ #
    # Global halt
    # ------------------------------------------------------------------ #

    def halt(self, reason: str) -> None:
        if not self._halted:
            self._halted = True
            self._halt_reason = reason
            log.critical("TRADING HALTED: %s", reason)
            state.record_risk_event("HALT", reason)

    def resume(self) -> None:
        if self._halted:
            self._halted = False
            self._halt_reason = ""
            log.warning("Trading resumed")

    @property
    def is_halted(self) -> bool:
        return self._halted

    # ------------------------------------------------------------------ #
    # Per-cycle checks
    # ------------------------------------------------------------------ #

    @property
    def current_drawdown(self) -> float:
        return self._current_drawdown

    def check_drawdown(self, current_equity: float, peak_equity: float) -> GuardDecision:
        if peak_equity <= 0:
            return GuardDecision(True)
        dd = (peak_equity - current_equity) / peak_equity
        self._current_drawdown = max(0.0, dd)
        if dd >= cfg.max_drawdown:
            self.halt(f"Max drawdown {dd:.1%} >= limit {cfg.max_drawdown:.1%}")
            return GuardDecision(False, self._halt_reason)
        # Reduce size gradually as drawdown approaches limit
        scale = max(0.2, 1.0 - dd / cfg.max_drawdown)
        return GuardDecision(True, scale=scale)

    def check_daily_loss(self, current_equity: float, start_equity: float) -> GuardDecision:
        if start_equity <= 0:
            return GuardDecision(True)
        daily_ret = (current_equity - start_equity) / start_equity
        if daily_ret <= -cfg.daily_loss_limit:
            self.halt(f"Daily loss {daily_ret:.1%} <= -{cfg.daily_loss_limit:.1%}")
            return GuardDecision(False, self._halt_reason)
        return GuardDecision(True)

    def check_leverage(self, gross_exposure: float) -> GuardDecision:
        if gross_exposure > cfg.max_leverage:
            reason = f"Gross exposure {gross_exposure:.1f}x > max {cfg.max_leverage}x"
            log.warning(reason)
            return GuardDecision(False, reason)
        return GuardDecision(True)

    def check_concentration(self, hhi: float, threshold: float = 0.5) -> GuardDecision:
        """HHI > 0.5 means very concentrated portfolio — reduce new positions."""
        if hhi > threshold:
            scale = 1.0 - (hhi - threshold) / (1.0 - threshold + 1e-10)
            return GuardDecision(True, f"High concentration HHI={hhi:.2f}", scale=max(0.1, scale))
        return GuardDecision(True)

    def check_regime(self, regime: str, signal_regime: str | None) -> GuardDecision:
        """Block strategies that only work in specific regimes."""
        if signal_regime and regime not in (signal_regime, "any"):
            return GuardDecision(
                False, f"Strategy requires regime={signal_regime}, current={regime}"
            )
        return GuardDecision(True)

    # ------------------------------------------------------------------ #
    # Aggregate signal filter
    # ------------------------------------------------------------------ #

    def filter_signals(
        self,
        signals: list,
        current_equity: float,
        peak_equity: float,
        start_equity: float,
        gross_exposure: float,
        hhi: float,
        regime: str,
    ) -> list:
        """
        Apply all guards to a list of Signal objects.
        Returns filtered (and potentially size-scaled) signals.
        """
        if self._halted:
            log.warning("All signals blocked — trading halted: %s", self._halt_reason)
            return []

        decisions = [
            self.check_drawdown(current_equity, peak_equity),
            self.check_daily_loss(current_equity, start_equity),
            self.check_leverage(gross_exposure),
            self.check_concentration(hhi),
        ]

        for d in decisions:
            if not d.allowed:
                return []

        min_scale = min(d.scale for d in decisions)

        filtered = []
        for sig in signals:
            regime_ok = self.check_regime(regime, getattr(sig, "required_regime", None))
            if not regime_ok.allowed:
                continue
            if min_scale < 1.0:
                sig = sig._replace(quantity=sig.quantity * min_scale)
            if sig.quantity > 0:
                filtered.append(sig)

        return filtered

    # ------------------------------------------------------------------ #
    # New-day reset
    # ------------------------------------------------------------------ #

    def maybe_reset_daily(self, current_equity: float) -> None:
        today = datetime.now(timezone.utc).date()
        if self._daily_date != today:
            self._daily_date = today
            self._daily_pnl_start_equity = current_equity
            # Auto-resume halt if it was due to daily loss (new day = fresh start)
            if self._halted and "Daily loss" in self._halt_reason:
                self.resume()
            log.info("New trading day — start equity=%.2f", current_equity)

    @property
    def daily_start_equity(self) -> float:
        return self._daily_pnl_start_equity or 0.0
