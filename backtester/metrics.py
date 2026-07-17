"""
Performance metrics for backtest results.
All ratios are annualised. Frequency is inferred from the equity curve index.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    # Returns
    total_return_pct: float
    annualised_return_pct: float
    annualised_vol_pct: float

    # Risk-adjusted ratios
    sharpe: float
    sortino: float
    calmar: float

    # Drawdown
    max_drawdown_pct: float
    max_drawdown_duration_bars: int
    avg_drawdown_pct: float

    # Trade stats
    num_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_holding_bars: float
    best_trade_pct: float
    worst_trade_pct: float

    # Misc
    annualisation_factor: float    # bars per year used
    start_equity: float
    end_equity: float

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}

    def summary_lines(self) -> list[str]:
        lines = [
            f"  Total Return      : {self.total_return_pct:+.2f}%",
            f"  Annualised Return : {self.annualised_return_pct:+.2f}%",
            f"  Annualised Vol    : {self.annualised_vol_pct:.2f}%",
            f"  Sharpe Ratio      : {self.sharpe:.3f}",
            f"  Sortino Ratio     : {self.sortino:.3f}",
            f"  Calmar Ratio      : {self.calmar:.3f}",
            f"  Max Drawdown      : {self.max_drawdown_pct:.2f}%  ({self.max_drawdown_duration_bars} bars)",
            f"  Avg Drawdown      : {self.avg_drawdown_pct:.2f}%",
            f"  Trades            : {self.num_trades}",
            f"  Win Rate          : {self.win_rate_pct:.1f}%",
            f"  Profit Factor     : {self.profit_factor:.3f}",
            f"  Avg Win / Loss    : {self.avg_win_pct:+.2f}% / {self.avg_loss_pct:+.2f}%",
            f"  Best / Worst Trade: {self.best_trade_pct:+.2f}% / {self.worst_trade_pct:+.2f}%",
            f"  Avg Hold          : {self.avg_holding_bars:.1f} bars",
        ]
        return lines


def _infer_annualisation(index: pd.DatetimeIndex) -> float:
    """Estimate bars per year from the DatetimeIndex frequency."""
    if len(index) < 2:
        return 252 * 24
    median_delta = pd.Series(index).diff().dropna().median()
    seconds = median_delta.total_seconds()
    seconds_per_year = 365.25 * 24 * 3600
    return max(seconds_per_year / seconds, 1.0)


def compute_metrics(
    equity_curve: pd.Series,           # DatetimeIndex, values = portfolio equity
    trade_returns: list[float],        # per-trade percentage returns (e.g. 0.02 = +2%)
    holding_periods: list[int],        # per-trade holding duration in bars
    risk_free_rate: float = 0.05,      # annual risk-free rate
) -> PerformanceMetrics:
    """
    Compute full performance metrics from equity curve and trade list.
    """
    eq = equity_curve.dropna()
    if len(eq) < 2:
        return _empty_metrics(eq.iloc[0] if len(eq) else 1.0)

    ann = _infer_annualisation(eq.index)
    rfr_per_bar = risk_free_rate / ann

    # --- Returns ---
    bar_returns = eq.pct_change().dropna()
    total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1
    n_years = len(eq) / ann
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 1e-6)) - 1
    ann_vol = float(bar_returns.std() * np.sqrt(ann))

    # --- Sharpe ---
    excess = bar_returns - rfr_per_bar
    sharpe = float(excess.mean() / (bar_returns.std() + 1e-10) * np.sqrt(ann))

    # --- Sortino (downside deviation only) ---
    # Arithmetic annualized excess return over arithmetic downside deviation —
    # mixing the geometric ann_ret with an arithmetic denominator (as before)
    # produces a scale-inconsistent ratio.
    downside = bar_returns[bar_returns < rfr_per_bar]
    downside_std = float(downside.std() * np.sqrt(ann)) if len(downside) > 0 else 1e-10
    arith_ann_excess = float(excess.mean() * ann)
    sortino = float(arith_ann_excess / (downside_std + 1e-10))

    # --- Drawdown ---
    rolling_peak = eq.cummax()
    dd_series = (eq - rolling_peak) / (rolling_peak + 1e-10)
    max_dd = float(dd_series.min())
    avg_dd = float(dd_series[dd_series < 0].mean()) if (dd_series < 0).any() else 0.0

    # Max drawdown duration (consecutive bars below peak)
    in_dd = (dd_series < 0).astype(int)
    max_dd_dur = int(_max_consecutive(in_dd.values))

    # --- Calmar ---
    calmar = float(ann_ret / (abs(max_dd) + 1e-10))

    # --- Trade stats ---
    trets = np.array(trade_returns) if trade_returns else np.array([0.0])
    wins = trets[trets > 0]
    losses = trets[trets <= 0]
    win_rate = float(len(wins) / (len(trets) + 1e-10)) * 100
    pf = float(wins.sum() / (abs(losses.sum()) + 1e-10)) if len(wins) > 0 else 0.0
    avg_win = float(wins.mean() * 100) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean() * 100) if len(losses) > 0 else 0.0
    avg_hold = float(np.mean(holding_periods)) if holding_periods else 0.0
    best = float(trets.max() * 100) if len(trets) > 0 else 0.0
    worst = float(trets.min() * 100) if len(trets) > 0 else 0.0

    return PerformanceMetrics(
        total_return_pct=total_ret * 100,
        annualised_return_pct=ann_ret * 100,
        annualised_vol_pct=ann_vol * 100,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown_pct=max_dd * 100,
        max_drawdown_duration_bars=max_dd_dur,
        avg_drawdown_pct=avg_dd * 100,
        num_trades=len(trade_returns),
        win_rate_pct=win_rate,
        profit_factor=pf,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        avg_holding_bars=avg_hold,
        best_trade_pct=best,
        worst_trade_pct=worst,
        annualisation_factor=ann,
        start_equity=float(eq.iloc[0]),
        end_equity=float(eq.iloc[-1]),
    )


def _max_consecutive(arr: np.ndarray) -> int:
    max_run = curr = 0
    for v in arr:
        curr = curr + 1 if v else 0
        max_run = max(max_run, curr)
    return max_run


def _empty_metrics(equity: float) -> PerformanceMetrics:
    return PerformanceMetrics(
        total_return_pct=0, annualised_return_pct=0, annualised_vol_pct=0,
        sharpe=0, sortino=0, calmar=0,
        max_drawdown_pct=0, max_drawdown_duration_bars=0, avg_drawdown_pct=0,
        num_trades=0, win_rate_pct=0, profit_factor=0,
        avg_win_pct=0, avg_loss_pct=0, avg_holding_bars=0,
        best_trade_pct=0, worst_trade_pct=0,
        annualisation_factor=252 * 24, start_equity=equity, end_equity=equity,
    )
