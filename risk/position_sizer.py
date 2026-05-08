"""
Dynamic position sizing:
- Kelly criterion (quarter-Kelly default)
- Volatility targeting
- Risk-parity allocation
- Regime-adjusted scaling
"""

from __future__ import annotations
import logging

import numpy as np

from bot.config import cfg

log = logging.getLogger("risk.position_sizer")


class PositionSizer:
    def __init__(self) -> None:
        self.kelly_fraction = cfg.kelly_fraction
        self.max_risk_per_trade = cfg.max_risk_per_trade
        self.max_portfolio_risk = cfg.max_portfolio_risk
        self.min_notional = cfg.min_notional

    # ------------------------------------------------------------------ #
    # Kelly criterion
    # ------------------------------------------------------------------ #

    def kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        equity: float,
    ) -> float:
        """
        Full Kelly fraction of equity.
        k = (win_rate / avg_loss) - (loss_rate / avg_win)
        Applied at self.kelly_fraction (quarter-Kelly by default).
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 0.0
        loss_rate = 1.0 - win_rate
        k = (win_rate / avg_loss) - (loss_rate / avg_win)
        k = max(0.0, k)  # never go short from Kelly
        size_fraction = min(k * self.kelly_fraction, self.max_risk_per_trade)
        return equity * size_fraction

    # ------------------------------------------------------------------ #
    # Volatility targeting
    # ------------------------------------------------------------------ #

    def vol_target_size(
        self,
        target_vol: float,
        realised_vol: float,
        equity: float,
        price: float,
    ) -> float:
        """
        Size position so it contributes `target_vol` to portfolio volatility.
        target_vol and realised_vol are annualised fractions.
        Returns USDT notional.
        """
        if realised_vol <= 0 or price <= 0:
            return 0.0
        notional = (target_vol / realised_vol) * equity
        return min(notional, equity * self.max_risk_per_trade / max(realised_vol, 0.01))

    # ------------------------------------------------------------------ #
    # Signal-confidence-weighted sizing
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
        Scales down in volatile/bear regimes, scales up in bull.
        """
        regime_scalar = {"bull": 1.2, "ranging": 1.0, "volatile": 0.6, "bear": 0.8}.get(regime, 1.0)

        base_notional = equity * self.max_risk_per_trade * signal_confidence * regime_scalar

        # Vol-adjust: if vol is very high, reduce size
        vol_adj = min(1.0, 0.3 / max(realised_vol, 0.10))
        notional = base_notional * vol_adj

        # Hard limits
        notional = max(self.min_notional, notional)
        notional = min(notional, equity * cfg.max_portfolio_risk)

        qty = notional / max(price, 1e-8)
        return round(notional, 2), qty

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
        vol_cap = max(1, int(0.20 / max(realised_vol, 0.01)))  # target 20% daily vol at 1x
        return min(regime_cap, vol_cap, max_lev)

    # ------------------------------------------------------------------ #
    # Quantity precision
    # ------------------------------------------------------------------ #

    @staticmethod
    def round_step(quantity: float, step_size: float) -> float:
        if step_size <= 0:
            return quantity
        return round(quantity // step_size * step_size, 10)
