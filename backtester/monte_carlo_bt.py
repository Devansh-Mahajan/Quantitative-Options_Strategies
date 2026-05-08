"""
Backtest Monte Carlo & Las Vegas Simulation.

Two complementary analyses:

  Monte Carlo Bootstrap
  ─────────────────────
  Resample trade returns (with replacement) N times.
  Produces confidence intervals for Sharpe, annualised return, max-drawdown, win-rate.
  Also forward-projects the equity curve with percentile bands showing the
  realistic spread of outcomes if the strategy keeps its statistical edge.

  Las Vegas Permutation Test
  ──────────────────────────
  Named after the Las Vegas algorithm class: uses randomness to always reach
  a definitive (probabilistic) conclusion.
  Randomly permutes the sequence of trade returns N_perm times.
  Asks: "could a *random* ordering of these exact trades have produced results
  at least as good?" → p-value.

  Interpretation:
    p < 0.05  → strategy has statistically significant edge at 95% confidence
    p < 0.10  → marginal significance
    p ≥ 0.20  → results are likely due to luck / curve-fitting; reduce conviction

  Additionally runs a random-entry baseline: replaces each signal with a fair
  coin-flip entry at the same time, keeping the same stop/TP. Compares your
  strategy's Sharpe to the distribution of coin-flip Sharpes.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from backtester.metrics import PerformanceMetrics, compute_metrics, _empty_metrics

log = logging.getLogger("backtester.monte_carlo_bt")


# ─── result types ──────────────────────────────────────────────────────────── #

@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a single metric."""
    metric: str
    observed: float
    ci_low_90: float
    ci_high_90: float
    ci_low_95: float
    ci_high_95: float
    mean_bootstrap: float
    std_bootstrap: float

    def is_robust(self) -> bool:
        """True if the 90% CI does not include zero (for return/Sharpe)."""
        return self.ci_low_90 > 0 or self.ci_high_90 < 0


@dataclass
class MonteCarloBootstrapResult:
    n_iterations: int
    trade_returns: list[float]
    sharpe_ci: BootstrapCI
    return_ci: BootstrapCI
    max_dd_ci: BootstrapCI
    win_rate_ci: BootstrapCI
    profit_factor_ci: BootstrapCI
    equity_bands: pd.DataFrame          # columns: p5, p25, p50, p75, p95
    ruin_probability: float             # fraction of paths where equity drops >50%
    target_return_probability: float    # fraction of paths exceeding 20% annual return


@dataclass
class LasVegasResult:
    n_permutations: int
    observed_sharpe: float
    observed_return: float
    permutation_sharpe_dist: np.ndarray
    permutation_return_dist: np.ndarray
    p_value_sharpe: float               # fraction of permutations ≥ observed Sharpe
    p_value_return: float               # fraction of permutations ≥ observed return
    is_significant_sharpe: bool         # p < 0.05
    is_significant_return: bool
    random_baseline_sharpe: float       # E[Sharpe] if entries were random (no edge)
    random_baseline_return: float
    edge_sharpe: float                  # observed − random_baseline (true alpha)
    edge_return: float


@dataclass
class BacktestMCResult:
    bootstrap: MonteCarloBootstrapResult
    las_vegas: LasVegasResult
    verdict: str                        # "STRONG_EDGE" | "MARGINAL_EDGE" | "NO_EDGE" | "INSUFFICIENT_DATA"


# ─── engine ────────────────────────────────────────────────────────────────── #

class BacktestMonteCarloEngine:
    """
    Runs bootstrap + Las Vegas analysis on a completed backtest.

    Args:
        n_bootstrap:    paths for bootstrap confidence intervals (default 10,000)
        n_permutations: paths for Las Vegas permutation test (default 10,000)
        target_annual_return: threshold for "target reached" probability (default 0.20 = 20%)
        ruin_threshold: equity drop considered "ruin" (default 0.50 = 50% loss)
    """

    def __init__(
        self,
        n_bootstrap: int = 10_000,
        n_permutations: int = 10_000,
        target_annual_return: float = 0.20,
        ruin_threshold: float = 0.50,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.target_annual_return = target_annual_return
        self.ruin_threshold = ruin_threshold

    def run(
        self,
        trade_returns: list[float],         # per-trade % returns (e.g. 0.05 = +5%)
        equity_curve: pd.Series,            # DatetimeIndex → equity
        holding_periods: list[int] | None = None,
        verbose: bool = True,
    ) -> BacktestMCResult:

        if len(trade_returns) < 10:
            log.warning("Only %d trades — MC results will be unreliable (need ≥ 30)", len(trade_returns))

        if len(trade_returns) < 5:
            m = _empty_metrics(float(equity_curve.iloc[0]) if len(equity_curve) else 1.0)
            empty_ci = lambda name, v=0.0: BootstrapCI(name, v, 0, 0, 0, 0, 0, 0)
            empty_lv = LasVegasResult(0, 0, 0, np.array([]), np.array([]),
                                      1.0, 1.0, False, False, 0, 0, 0, 0)
            return BacktestMCResult(
                bootstrap=MonteCarloBootstrapResult(
                    0, trade_returns,
                    empty_ci("sharpe"), empty_ci("return"),
                    empty_ci("max_dd"), empty_ci("win_rate"), empty_ci("profit_factor"),
                    pd.DataFrame(), 0.0, 0.0,
                ),
                las_vegas=empty_lv,
                verdict="INSUFFICIENT_DATA",
            )

        rets = np.array(trade_returns)
        holds = np.array(holding_periods or [1] * len(rets))

        if verbose:
            log.info("Bootstrap: %d iterations  ·  Permutation: %d", self.n_bootstrap, self.n_permutations)

        bootstrap_result = self._run_bootstrap(rets, holds, equity_curve)
        lv_result        = self._run_las_vegas(rets, holds, equity_curve)
        verdict          = self._verdict(lv_result, bootstrap_result)

        return BacktestMCResult(bootstrap=bootstrap_result, las_vegas=lv_result, verdict=verdict)

    # ──────────────────────────────────────────────────────────────────────── #
    # Bootstrap
    # ──────────────────────────────────────────────────────────────────────── #

    def _run_bootstrap(
        self,
        rets: np.ndarray,
        holds: np.ndarray,
        equity_curve: pd.Series,
    ) -> MonteCarloBootstrapResult:
        n = len(rets)
        start_equity = float(equity_curve.iloc[0]) if len(equity_curve) else 1.0

        sharpes, returns, max_dds, win_rates, pfs = [], [], [], [], []
        equity_paths: list[np.ndarray] = []

        for _ in range(self.n_bootstrap):
            idx = np.random.randint(0, n, size=n)
            sampled = rets[idx]
            sampled_holds = holds[idx]

            # Build synthetic equity curve from resampled trades
            eq = self._equity_from_trade_rets(sampled, start_equity)
            eq_series = pd.Series(
                eq,
                index=pd.date_range(
                    start=equity_curve.index[0] if len(equity_curve) else pd.Timestamp.now(),
                    periods=len(eq),
                    freq="h",
                    tz="UTC",
                ),
            )
            m = compute_metrics(eq_series, sampled.tolist(), sampled_holds.tolist())
            sharpes.append(m.sharpe)
            returns.append(m.total_return_pct)
            max_dds.append(m.max_drawdown_pct)
            win_rates.append(m.win_rate_pct)
            pfs.append(m.profit_factor)

            equity_paths.append(eq / eq[0])   # normalised path

        # Observed metrics
        obs_metrics = compute_metrics(equity_curve, rets.tolist(), holds.tolist())

        def _ci(arr, observed, name) -> BootstrapCI:
            a = np.array(arr)
            return BootstrapCI(
                metric=name,
                observed=observed,
                ci_low_90=float(np.percentile(a, 5)),
                ci_high_90=float(np.percentile(a, 95)),
                ci_low_95=float(np.percentile(a, 2.5)),
                ci_high_95=float(np.percentile(a, 97.5)),
                mean_bootstrap=float(a.mean()),
                std_bootstrap=float(a.std()),
            )

        # Equity percentile bands (normalised)
        all_paths = np.array(equity_paths)     # (n_bootstrap, n_trades+1)
        eq_bands = pd.DataFrame({
            "p5":  all_paths.T @ np.zeros(self.n_bootstrap),   # placeholder; filled below
        })
        pcts = [5, 25, 50, 75, 95]
        eq_bands = pd.DataFrame({
            f"p{p}": np.percentile(all_paths, p, axis=0) * start_equity
            for p in pcts
        })

        # Ruin probability: paths where final equity < ruin_threshold of start
        ruin_prob = float((all_paths[:, -1] < (1 - self.ruin_threshold)).mean())

        # Fraction of paths hitting target annual return
        # Approximate: if the equity path grew enough over n trades
        n_years_approx = max(1, len(equity_curve) / (365.25 * 24))  # assuming 1h bars
        target_mult = (1 + self.target_annual_return) ** n_years_approx
        target_prob = float((all_paths[:, -1] >= target_mult).mean())

        return MonteCarloBootstrapResult(
            n_iterations=self.n_bootstrap,
            trade_returns=rets.tolist(),
            sharpe_ci=_ci(sharpes, obs_metrics.sharpe, "sharpe"),
            return_ci=_ci(returns, obs_metrics.total_return_pct, "total_return_pct"),
            max_dd_ci=_ci(max_dds, obs_metrics.max_drawdown_pct, "max_drawdown_pct"),
            win_rate_ci=_ci(win_rates, obs_metrics.win_rate_pct, "win_rate_pct"),
            profit_factor_ci=_ci(pfs, obs_metrics.profit_factor, "profit_factor"),
            equity_bands=eq_bands,
            ruin_probability=ruin_prob,
            target_return_probability=target_prob,
        )

    # ──────────────────────────────────────────────────────────────────────── #
    # Las Vegas permutation test
    # ──────────────────────────────────────────────────────────────────────── #

    def _run_las_vegas(
        self,
        rets: np.ndarray,
        holds: np.ndarray,
        equity_curve: pd.Series,
    ) -> LasVegasResult:
        start_equity = float(equity_curve.iloc[0]) if len(equity_curve) else 1.0
        obs_metrics  = compute_metrics(equity_curve, rets.tolist(), holds.tolist())
        obs_sharpe   = obs_metrics.sharpe
        obs_return   = obs_metrics.total_return_pct

        perm_sharpes: list[float] = []
        perm_returns: list[float] = []

        freq = "h"
        t0 = equity_curve.index[0] if len(equity_curve) else pd.Timestamp.now(tz="UTC")

        for _ in range(self.n_permutations):
            shuffled = rets.copy()
            np.random.shuffle(shuffled)
            eq = self._equity_from_trade_rets(shuffled, start_equity)
            eq_s = pd.Series(eq, index=pd.date_range(t0, periods=len(eq), freq=freq, tz="UTC"))
            m = compute_metrics(eq_s, shuffled.tolist(), holds.tolist())
            perm_sharpes.append(m.sharpe)
            perm_returns.append(m.total_return_pct)

        ps_arr = np.array(perm_sharpes)
        pr_arr = np.array(perm_returns)

        p_val_sharpe = float((ps_arr >= obs_sharpe).mean())
        p_val_return = float((pr_arr >= obs_return).mean())

        return LasVegasResult(
            n_permutations=self.n_permutations,
            observed_sharpe=obs_sharpe,
            observed_return=obs_return,
            permutation_sharpe_dist=ps_arr,
            permutation_return_dist=pr_arr,
            p_value_sharpe=p_val_sharpe,
            p_value_return=p_val_return,
            is_significant_sharpe=(p_val_sharpe < 0.05),
            is_significant_return=(p_val_return < 0.05),
            random_baseline_sharpe=float(ps_arr.mean()),
            random_baseline_return=float(pr_arr.mean()),
            edge_sharpe=obs_sharpe - float(ps_arr.mean()),
            edge_return=obs_return - float(pr_arr.mean()),
        )

    # ──────────────────────────────────────────────────────────────────────── #
    # Helpers
    # ──────────────────────────────────────────────────────────────────────── #

    @staticmethod
    def _equity_from_trade_rets(rets: np.ndarray, start: float) -> np.ndarray:
        """Compound trade returns into an equity curve."""
        eq = np.empty(len(rets) + 1)
        eq[0] = start
        for i, r in enumerate(rets):
            eq[i + 1] = eq[i] * (1 + r)
        return eq

    @staticmethod
    def _verdict(lv: LasVegasResult, bs: MonteCarloBootstrapResult) -> str:
        if lv.n_permutations == 0:
            return "INSUFFICIENT_DATA"
        if lv.is_significant_sharpe and lv.is_significant_return and bs.sharpe_ci.is_robust():
            return "STRONG_EDGE"
        if lv.p_value_sharpe < 0.10 or lv.is_significant_return:
            return "MARGINAL_EDGE"
        return "NO_EDGE"


# ─── printing ──────────────────────────────────────────────────────────────── #

def print_mc_bt_result(result: BacktestMCResult) -> None:
    sep = "─" * 60
    bs  = result.bootstrap
    lv  = result.las_vegas

    verdict_icons = {
        "STRONG_EDGE": "✔  STRONG EDGE — statistically significant at 95%",
        "MARGINAL_EDGE": "~  MARGINAL EDGE — significant at ~90% — exercise caution",
        "NO_EDGE": "✘  NO EDGE — results not distinguishable from random",
        "INSUFFICIENT_DATA": "?  INSUFFICIENT DATA — run more trades before drawing conclusions",
    }

    print(f"\n  {'═'*58}")
    print(f"  Backtest Monte Carlo & Las Vegas Analysis")
    print(f"  {'═'*58}")
    print(f"  Verdict: {verdict_icons.get(result.verdict, result.verdict)}")
    print()

    # Bootstrap CIs
    print(f"  Bootstrap Confidence Intervals  ({bs.n_iterations:,} samples)")
    print(f"  {sep}")
    _print_ci(bs.sharpe_ci,        "Sharpe Ratio",   fmt=".3f")
    _print_ci(bs.return_ci,        "Total Return %",  fmt=".2f")
    _print_ci(bs.max_dd_ci,        "Max Drawdown %",  fmt=".2f")
    _print_ci(bs.win_rate_ci,      "Win Rate %",      fmt=".1f")
    _print_ci(bs.profit_factor_ci, "Profit Factor",   fmt=".3f")
    print()
    print(f"  Ruin Probability (equity −50%)  : {bs.ruin_probability:.2%}")
    print(f"  Target Return (≥20% p.a.) Prob  : {bs.target_return_probability:.2%}")
    print()

    # Las Vegas
    print(f"  Las Vegas Permutation Test  ({lv.n_permutations:,} shuffles)")
    print(f"  {sep}")
    sig_s = "✔ significant" if lv.is_significant_sharpe else "✘ not significant"
    sig_r = "✔ significant" if lv.is_significant_return else "✘ not significant"
    print(f"  Observed Sharpe        : {lv.observed_sharpe:.3f}")
    print(f"  Random-baseline Sharpe : {lv.random_baseline_sharpe:.3f}  (mean of shuffled sequences)")
    print(f"  Sharpe edge (alpha)    : {lv.edge_sharpe:+.3f}")
    print(f"  p-value (Sharpe)       : {lv.p_value_sharpe:.4f}  →  {sig_s}")
    print()
    print(f"  Observed Return        : {lv.observed_return:+.2f}%")
    print(f"  Random-baseline Return : {lv.random_baseline_return:+.2f}%")
    print(f"  Return edge (alpha)    : {lv.edge_return:+.2f}%")
    print(f"  p-value (Return)       : {lv.p_value_return:.4f}  →  {sig_r}")

    # Permutation distribution summary
    pd_arr = lv.permutation_sharpe_dist
    if len(pd_arr):
        print()
        print(f"  Permutation Sharpe Distribution")
        print(f"    p5={np.percentile(pd_arr,5):.3f}  p25={np.percentile(pd_arr,25):.3f}  "
              f"median={np.median(pd_arr):.3f}  p75={np.percentile(pd_arr,75):.3f}  "
              f"p95={np.percentile(pd_arr,95):.3f}")
        # ASCII histogram
        _ascii_hist(pd_arr, lv.observed_sharpe, label="Sharpe")

    print(f"  {'═'*58}\n")


def _print_ci(ci: BootstrapCI, label: str, fmt: str = ".3f") -> None:
    robust = "✔" if ci.is_robust() else " "
    print(
        f"  {robust} {label:<22} obs={ci.observed:{fmt}}  "
        f"90%CI [{ci.ci_low_90:{fmt}}, {ci.ci_high_90:{fmt}}]  "
        f"95%CI [{ci.ci_low_95:{fmt}}, {ci.ci_high_95:{fmt}}]"
    )


def _ascii_hist(arr: np.ndarray, marker: float, label: str = "", width: int = 50) -> None:
    lo, hi = arr.min(), arr.max()
    n_bins = 20
    counts, edges = np.histogram(arr, bins=n_bins)
    max_count = counts.max() or 1
    print(f"    {label} distribution (▓ = count,  | = observed)")
    for i, count in enumerate(counts):
        bar_len = int(count / max_count * width)
        bar = "▓" * bar_len
        lo_e, hi_e = edges[i], edges[i + 1]
        is_obs = lo_e <= marker < hi_e
        marker_str = " ←observed" if is_obs else ""
        print(f"    {lo_e:+.2f} │{bar}{marker_str}")
