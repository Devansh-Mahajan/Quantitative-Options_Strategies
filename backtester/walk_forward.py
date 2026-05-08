"""
Walk-forward validation for backtesting.

Classic expanding-window (anchored) and rolling-window walk-forward.
Each fold: train on [start, fold_end), test on [fold_end, fold_end + test_size).
Aggregated metrics expose whether in-sample performance generalises.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from backtester.engine import BacktestEngine, BacktestResult
from backtester.metrics import PerformanceMetrics, compute_metrics
from backtester.report import print_report

log = logging.getLogger("backtester.walk_forward")


@dataclass
class WalkForwardFold:
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    is_result: BacktestResult       # in-sample
    oos_result: BacktestResult      # out-of-sample


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold]
    oos_combined_metrics: PerformanceMetrics    # metrics on spliced OOS equity curve
    is_combined_metrics: PerformanceMetrics     # metrics on IS equity curve (sanity check)
    efficiency_ratio: float                     # OOS Sharpe / IS Sharpe  (target > 0.5)
    degradation_pct: float                      # (IS return - OOS return) / IS return * 100

    def summary(self) -> str:
        lines = [
            "══ Walk-Forward Validation ══════════════════════════",
            f"  Folds          : {len(self.folds)}",
            f"  IS  Sharpe     : {self.is_combined_metrics.sharpe:.3f}",
            f"  OOS Sharpe     : {self.oos_combined_metrics.sharpe:.3f}",
            f"  Efficiency     : {self.efficiency_ratio:.2%}  (target >50%)",
            f"  IS  Total Ret  : {self.is_combined_metrics.total_return_pct:+.2f}%",
            f"  OOS Total Ret  : {self.oos_combined_metrics.total_return_pct:+.2f}%",
            f"  Degradation    : {self.degradation_pct:.1f}%  (lower = less overfit)",
            f"  OOS Max DD     : {self.oos_combined_metrics.max_drawdown_pct:.2f}%",
            f"  OOS Win Rate   : {self.oos_combined_metrics.win_rate_pct:.1f}%",
            "═" * 52,
        ]
        return "\n".join(lines)


class WalkForwardValidator:
    """
    Runs expanding-window or rolling-window walk-forward analysis.

    Args:
        data:           {symbol: DataFrame} OHLCV, same format as BacktestEngine
        interval:       bar interval string ("1h", etc.)
        n_folds:        number of test folds
        test_fraction:  fraction of total bars used as each test window (default 0.2 = 20%)
        anchored:       True = expanding train window; False = rolling fixed-size window
        engine_kwargs:  passed through to BacktestEngine (strategies, stop_loss_pct, etc.)
        strategy_factory: callable() → list[strategy]; called fresh for each fold to reset
                          internal state (required if strategies are stateful)
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        interval: str,
        n_folds: int = 5,
        test_fraction: float = 0.20,
        anchored: bool = True,
        engine_kwargs: dict | None = None,
        strategy_factory: Callable | None = None,
    ) -> None:
        self.data = data
        self.interval = interval
        self.n_folds = n_folds
        self.test_fraction = test_fraction
        self.anchored = anchored
        self.engine_kwargs = engine_kwargs or {}
        self.strategy_factory = strategy_factory

        anchor_sym = next(iter(data))
        self._anchor_df = data[anchor_sym].reset_index()
        self._n_bars = len(self._anchor_df)
        self._ts_col = "open_time" if "open_time" in self._anchor_df.columns else self._anchor_df.columns[0]

    def run(self, verbose: bool = True) -> WalkForwardResult:
        test_size = int(self._n_bars * self.test_fraction)
        folds = self._build_fold_indices(test_size)

        if verbose:
            log.info(
                "Walk-forward: %d folds, test=%d bars, mode=%s",
                len(folds), test_size, "anchored" if self.anchored else "rolling",
            )

        fold_results: list[WalkForwardFold] = []
        all_oos_eq: list[pd.Series] = []
        all_is_eq: list[pd.Series] = []
        all_oos_rets: list[float] = []
        all_is_rets: list[float] = []
        all_oos_holds: list[int] = []
        all_is_holds: list[int] = []

        for k, (train_start_i, train_end_i, test_start_i, test_end_i) in enumerate(folds):
            log.info(
                "  Fold %d/%d  IS=[%d:%d]  OOS=[%d:%d]",
                k + 1, len(folds), train_start_i, train_end_i, test_start_i, test_end_i,
            )
            train_data = self._slice(train_start_i, train_end_i)
            test_data = self._slice(test_start_i, test_end_i)

            strategies = self.strategy_factory() if self.strategy_factory else self.engine_kwargs.get("strategies", [])

            is_result = self._run_engine(train_data, strategies)
            oos_result = self._run_engine(test_data, strategies)

            ts = self._anchor_df[self._ts_col]
            fold_obj = WalkForwardFold(
                fold_index=k,
                train_start=pd.Timestamp(ts.iloc[train_start_i]),
                train_end=pd.Timestamp(ts.iloc[train_end_i - 1]),
                test_start=pd.Timestamp(ts.iloc[test_start_i]),
                test_end=pd.Timestamp(ts.iloc[test_end_i - 1]),
                is_result=is_result,
                oos_result=oos_result,
            )
            fold_results.append(fold_obj)

            if not oos_result.equity_curve.empty:
                all_oos_eq.append(oos_result.equity_curve)
            if not is_result.equity_curve.empty:
                all_is_eq.append(is_result.equity_curve)

            if "pnl_pct" in oos_result.trades.columns:
                all_oos_rets.extend(oos_result.trades["pnl_pct"].tolist())
            if "holding_bars" in oos_result.trades.columns:
                all_oos_holds.extend(oos_result.trades["holding_bars"].tolist())
            if "pnl_pct" in is_result.trades.columns:
                all_is_rets.extend(is_result.trades["pnl_pct"].tolist())
            if "holding_bars" in is_result.trades.columns:
                all_is_holds.extend(is_result.trades["holding_bars"].tolist())

            if verbose:
                print(f"\n  [Fold {k+1}] In-Sample:")
                is_result.metrics.summary_lines()
                print(f"\n  [Fold {k+1}] Out-of-Sample:")
                for line in oos_result.metrics.summary_lines():
                    print(line)

        # Splice OOS equity curves end-to-end (renormalized per fold)
        oos_eq = _splice_equity_curves(all_oos_eq)
        is_eq = _splice_equity_curves(all_is_eq)

        oos_metrics = compute_metrics(oos_eq, all_oos_rets, all_oos_holds)
        is_metrics = compute_metrics(is_eq, all_is_rets, all_is_holds)

        efficiency = oos_metrics.sharpe / (abs(is_metrics.sharpe) + 1e-10)
        is_ret = is_metrics.total_return_pct
        oos_ret = oos_metrics.total_return_pct
        degradation = ((is_ret - oos_ret) / (abs(is_ret) + 1e-10)) * 100

        wf_result = WalkForwardResult(
            folds=fold_results,
            oos_combined_metrics=oos_metrics,
            is_combined_metrics=is_metrics,
            efficiency_ratio=efficiency,
            degradation_pct=degradation,
        )

        if verbose:
            print("\n" + wf_result.summary())

        return wf_result

    # ------------------------------------------------------------------ #

    def _build_fold_indices(self, test_size: int) -> list[tuple[int, int, int, int]]:
        """Return list of (train_start, train_end, test_start, test_end) bar indices."""
        lookback = self.engine_kwargs.get("lookback", 60)
        folds = []
        for k in range(self.n_folds):
            test_end = self._n_bars - k * test_size
            test_start = test_end - test_size
            train_end = test_start
            if self.anchored:
                train_start = 0
            else:
                train_start = max(0, train_end - (self._n_bars - self.n_folds * test_size))
            if train_end - train_start < lookback + test_size:
                log.warning("Fold %d: insufficient training bars (%d), skipping", k, train_end - train_start)
                continue
            folds.append((train_start, train_end, test_start, test_end))
        return list(reversed(folds))  # chronological order

    def _slice(self, start_i: int, end_i: int) -> dict[str, pd.DataFrame]:
        """Slice all symbol DataFrames to bar range [start_i, end_i)."""
        sliced: dict[str, pd.DataFrame] = {}
        for sym, df in self.data.items():
            df_reset = df.reset_index()
            sliced[sym] = df_reset.iloc[start_i:end_i].set_index(df_reset.columns[0])
        return sliced

    def _run_engine(self, data: dict[str, pd.DataFrame], strategies) -> BacktestResult:
        kwargs = {k: v for k, v in self.engine_kwargs.items() if k != "strategies"}
        engine = BacktestEngine(
            data=data,
            interval=self.interval,
            strategies=strategies,
            **kwargs,
        )
        return engine.run()


def _splice_equity_curves(curves: list[pd.Series]) -> pd.Series:
    """
    Concatenate equity curves, renormalising each fold to start where previous ended.
    Result behaves like a single continuous equity series.
    """
    if not curves:
        return pd.Series(dtype=float)

    pieces = []
    running_equity = curves[0].iloc[0]
    for curve in curves:
        if curve.empty:
            continue
        scale = running_equity / (curve.iloc[0] + 1e-10)
        renorm = curve * scale
        pieces.append(renorm)
        running_equity = renorm.iloc[-1]

    return pd.concat(pieces).sort_index()
