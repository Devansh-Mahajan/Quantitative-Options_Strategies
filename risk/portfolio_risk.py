"""
Portfolio risk metrics:
- Historical VaR & CVaR (95% and 99%)
- Monte Carlo VaR (GPU-accelerated)
- Correlation matrix and concentration Herfindahl index
- Drawdown tracking
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.config import cfg

log = logging.getLogger("risk.portfolio_risk")


@dataclass
class RiskSnapshot:
    equity: float
    var_95: float           # as fraction of equity (positive = loss)
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    current_drawdown: float
    gross_exposure: float   # total notional / equity
    net_exposure: float     # (longs - shorts) / equity
    hhi: float              # Herfindahl–Hirschman Index (concentration)
    regime: str = "unknown"


class PortfolioRiskEngine:
    def __init__(self) -> None:
        self._equity_history: list[float] = []
        self._return_history: list[float] = []  # per-cycle returns
        self._peak_equity: float = 0.0

    # ------------------------------------------------------------------ #
    # Equity tracking
    # ------------------------------------------------------------------ #

    def record_equity(self, equity: float) -> None:
        self._equity_history.append(equity)
        if equity > self._peak_equity:
            self._peak_equity = equity
        if len(self._equity_history) > 1:
            prev = self._equity_history[-2]
            self._return_history.append((equity - prev) / (prev + 1e-10))

    @property
    def current_drawdown(self) -> float:
        if not self._equity_history or self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self._equity_history[-1]) / self._peak_equity

    @property
    def max_drawdown(self) -> float:
        if len(self._equity_history) < 2:
            return 0.0
        arr = np.array(self._equity_history)
        peak = np.maximum.accumulate(arr)
        dd = (peak - arr) / (peak + 1e-10)
        return float(dd.max())

    # ------------------------------------------------------------------ #
    # VaR / CVaR
    # ------------------------------------------------------------------ #

    def historical_var(self, confidence: float = 0.95) -> float:
        if len(self._return_history) < 30:
            return 0.0
        rets = np.array(self._return_history[-252:])
        return float(-np.percentile(rets, (1 - confidence) * 100))

    def historical_cvar(self, confidence: float = 0.95) -> float:
        if len(self._return_history) < 30:
            return 0.0
        rets = np.array(self._return_history[-252:])
        threshold = np.percentile(rets, (1 - confidence) * 100)
        tail = rets[rets <= threshold]
        return float(-tail.mean()) if len(tail) > 0 else 0.0

    def monte_carlo_var(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        equity: float,
        n_paths: int = 10_000,
        horizon: int = 1,
        confidence: float = 0.99,
    ) -> float:
        """Parametric Monte Carlo VaR. Returns fraction of equity."""
        try:
            L = np.linalg.cholesky(cov_matrix + np.eye(len(cov_matrix)) * 1e-8)
            z = np.random.randn(n_paths, len(weights))
            correlated = z @ L.T
            port_rets = correlated @ weights * np.sqrt(horizon)
            return float(-np.percentile(port_rets, (1 - confidence) * 100))
        except np.linalg.LinAlgError:
            return self.historical_var(confidence)

    # ------------------------------------------------------------------ #
    # Position-level metrics
    # ------------------------------------------------------------------ #

    def compute_snapshot(
        self,
        positions: list[dict],
        equity: float,
        regime: str = "unknown",
    ) -> RiskSnapshot:
        """
        positions: list of dicts with keys: symbol, notional, side (+1 long, -1 short)
        """
        if not positions or equity <= 0:
            return RiskSnapshot(
                equity=equity, var_95=0, var_99=0, cvar_95=0, cvar_99=0,
                max_drawdown=self.max_drawdown, current_drawdown=self.current_drawdown,
                gross_exposure=0, net_exposure=0, hhi=0, regime=regime,
            )

        notionals = np.array([abs(p.get("notional", 0)) for p in positions], dtype=float)
        sides = np.array([p.get("side", 1) for p in positions], dtype=float)

        gross = float(notionals.sum()) / equity
        net = float((notionals * sides).sum()) / equity

        # Concentration (HHI)
        weights = notionals / (notionals.sum() + 1e-10)
        hhi = float((weights ** 2).sum())

        self.record_equity(equity)

        return RiskSnapshot(
            equity=equity,
            var_95=self.historical_var(0.95),
            var_99=self.historical_var(0.99),
            cvar_95=self.historical_cvar(0.95),
            cvar_99=self.historical_cvar(0.99),
            max_drawdown=self.max_drawdown,
            current_drawdown=self.current_drawdown,
            gross_exposure=gross,
            net_exposure=net,
            hhi=hhi,
            regime=regime,
        )

    # ------------------------------------------------------------------ #
    # Correlation
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_correlation(returns_dict: dict[str, list[float]]) -> pd.DataFrame:
        df = pd.DataFrame(returns_dict).dropna()
        return df.corr()

    @staticmethod
    def covariance_matrix(returns_dict: dict[str, list[float]], lookback: int = 60) -> np.ndarray:
        df = pd.DataFrame(returns_dict).dropna().tail(lookback)
        return df.cov().values
