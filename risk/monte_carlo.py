"""
Portfolio Monte Carlo Risk Engine.

Two simulation modes:
  • Parametric GBM  — correlated Geometric Brownian Motion via Cholesky;
                      fast, GPU-accelerated, assumes log-normality.
  • Fat-tail GBM    — returns sampled from multivariate Student-t
                      (ν default=4); better for crypto tail risk.

Outputs:
  • VaR / CVaR / ES at 95%, 99%, 99.9% over 1d / 5d / 10d horizons
  • Full P&L distribution percentile bands
  • Path-based max-drawdown distribution
  • Stress scenario overlays (exchange hack, regulatory crash, BTC halving shock)
  • Forward equity projection with confidence envelope

GPU acceleration: uses PyTorch when CUDA is available, falls back to NumPy.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger("risk.monte_carlo")

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


# ─── result types ──────────────────────────────────────────────────────────── #

@dataclass
class VaRSurface:
    """VaR / CVaR at multiple confidence levels and time horizons."""
    horizon_days: list[int]
    confidence_levels: list[float]
    # var[i][j]  = VaR  at horizon_days[i], confidence_levels[j]  (fraction of portfolio)
    var: list[list[float]]
    cvar: list[list[float]]
    es_99: list[float]                  # Expected Shortfall 99% per horizon
    annualised_vol: float

    def summary(self) -> str:
        lines = ["  VaR Surface (fraction of portfolio)"]
        header = f"  {'Horizon':<10}" + "".join(f"  VaR{int(c*100)}%  CVaR{int(c*100)}%" for c in self.confidence_levels)
        lines.append(header)
        for i, h in enumerate(self.horizon_days):
            row = f"  {h}d{'':<7}"
            for j in range(len(self.confidence_levels)):
                row += f"  {self.var[i][j]:.3%}   {self.cvar[i][j]:.3%}  "
            lines.append(row)
        return "\n".join(lines)


@dataclass
class MonteCarloResult:
    n_paths: int
    horizon_days: int
    distribution: str                    # "normal" | "student-t"
    var_surface: VaRSurface
    path_returns: np.ndarray             # shape (n_paths,) — final portfolio P&L fraction
    path_max_dd: np.ndarray             # shape (n_paths,) — per-path max drawdown
    percentile_bands: dict[str, float]  # "p1", "p5", "p25", "p50", "p75", "p95", "p99"
    stress_results: dict[str, float]    # scenario → expected P&L fraction
    forward_equity_bands: pd.DataFrame  # DatetimeIndex × ["p5","p25","p50","p75","p95"]


@dataclass
class TradeMonteCarloResult:
    """Per-trade MC risk assessment."""
    symbol: str
    side: str
    entry_price: float
    quantity: float
    stop_loss_price: float
    take_profit_price: float
    n_paths: int
    prob_stop_hit: float
    prob_tp_hit: float
    expected_pnl_pct: float
    var_95: float
    cvar_95: float
    max_adverse_excursion_p95: float    # worst 5% of MFE paths


# ─── stress scenarios ──────────────────────────────────────────────────────── #

# Instantaneous shock applied to each asset as log-return
STRESS_SCENARIOS: dict[str, dict] = {
    "exchange_hack":        {"shock": -0.25, "vol_mult": 3.0,  "corr_boost": 0.9},
    "regulatory_ban":       {"shock": -0.35, "vol_mult": 4.0,  "corr_boost": 0.95},
    "btc_halving_squeeze":  {"shock": +0.15, "vol_mult": 2.0,  "corr_boost": 0.7},
    "flash_crash_recovery": {"shock": -0.20, "vol_mult": 5.0,  "corr_boost": 0.85},
    "defi_contagion":       {"shock": -0.30, "vol_mult": 3.5,  "corr_boost": 0.88},
    "tether_depeg":         {"shock": -0.15, "vol_mult": 2.5,  "corr_boost": 0.92},
    "mild_bear":            {"shock": -0.05, "vol_mult": 1.5,  "corr_boost": 0.6},
    "strong_bull":          {"shock": +0.10, "vol_mult": 1.2,  "corr_boost": 0.5},
}


# ─── main engine ───────────────────────────────────────────────────────────── #

class PortfolioMonteCarloEngine:
    """
    Full portfolio simulation engine.

    Args:
        n_paths:      number of Monte Carlo paths (default 50,000)
        distribution: "normal" | "student-t"
        nu:           degrees-of-freedom for Student-t (lower = fatter tails; 4 typical for crypto)
        use_gpu:      auto-detect CUDA; override with False to force CPU
    """

    def __init__(
        self,
        n_paths: int = 50_000,
        distribution: Literal["normal", "student-t"] = "student-t",
        nu: float = 4.0,
        use_gpu: bool = True,
    ) -> None:
        self.n_paths = n_paths
        self.distribution = distribution
        self.nu = nu
        self._device = self._pick_device(use_gpu)
        log.info(
            "MonteCarloEngine: n_paths=%d  dist=%s  nu=%.1f  device=%s",
            n_paths, distribution, nu, self._device,
        )

    # ──────────────────────────────────────────────────────────────────────── #
    # Portfolio-level simulation
    # ──────────────────────────────────────────────────────────────────────── #

    def simulate_portfolio(
        self,
        weights: np.ndarray,            # per-asset portfolio weight (sum=1)
        mean_returns: np.ndarray,       # annualised mean log-return per asset
        cov_matrix: np.ndarray,         # annualised covariance (n_assets × n_assets)
        equity: float,
        horizon_days: list[int] | None = None,
        confidence_levels: list[float] | None = None,
        include_stress: bool = True,
        bars_per_day: int = 24,         # 1h bars → 24 bars per day
    ) -> MonteCarloResult:
        """
        Simulate portfolio P&L distribution over multiple horizons.
        Returns MonteCarloResult with VaR surface, stress scenarios, equity bands.
        """
        horizons = horizon_days or [1, 5, 10]
        confs = confidence_levels or [0.95, 0.99, 0.999]

        # Use longest horizon for full path simulation
        max_horizon = max(horizons)
        n_steps = max_horizon * bars_per_day
        dt = 1.0 / (365.25 * bars_per_day)   # fraction of year per step

        # Annualised vol → per-step vol
        daily_cov = cov_matrix * (1.0 / 365.25)

        # Cholesky decomposition (add jitter for numerical stability)
        try:
            L = np.linalg.cholesky(daily_cov + np.eye(len(weights)) * 1e-8)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(np.diag(np.diag(daily_cov)) + np.eye(len(weights)) * 1e-8)

        # Simulate correlated paths
        paths = self._simulate_paths(L, mean_returns * dt, n_steps, len(weights))
        # paths shape: (n_paths, n_steps, n_assets)

        # Portfolio return at each step
        port_paths = (paths * weights[np.newaxis, np.newaxis, :]).sum(axis=2)
        # port_paths shape: (n_paths, n_steps)

        # Cumulative log-return paths
        cum_paths = np.cumsum(port_paths, axis=1)   # (n_paths, n_steps)

        # VaR surface
        var_surface = self._compute_var_surface(cum_paths, horizons, confs, bars_per_day)

        # Final returns at max horizon
        final_rets = cum_paths[:, -1]

        # Per-path max drawdown
        path_max_dd = self._path_max_drawdown(cum_paths)

        # Percentile bands
        bands = self._percentile_bands(final_rets)

        # Stress scenarios
        stress = {}
        if include_stress:
            stress = self._run_stress_scenarios(weights, mean_returns, cov_matrix, bars_per_day)

        # Forward equity bands as DataFrame
        fwd_bands = self._forward_equity_bands(cum_paths, equity, bars_per_day)

        return MonteCarloResult(
            n_paths=self.n_paths,
            horizon_days=max_horizon,
            distribution=self.distribution,
            var_surface=var_surface,
            path_returns=np.expm1(final_rets),   # log-return → simple return
            path_max_dd=path_max_dd,
            percentile_bands=bands,
            stress_results=stress,
            forward_equity_bands=fwd_bands,
        )

    # ──────────────────────────────────────────────────────────────────────── #
    # Trade-level simulation
    # ──────────────────────────────────────────────────────────────────────── #

    def simulate_trade(
        self,
        symbol: str,
        side: str,                   # "LONG" | "SHORT"
        entry_price: float,
        quantity: float,
        stop_loss_price: float,
        take_profit_price: float,
        annualised_vol: float,       # asset vol (e.g. 0.80 = 80% p.a.)
        holding_bars: int = 24,      # expected holding period in bars
        bars_per_day: int = 24,
    ) -> TradeMonteCarloResult:
        """
        Simulate a single trade's P&L distribution.
        Models GBM price paths and checks stop-loss / take-profit touch.
        """
        dt = 1.0 / (365.25 * bars_per_day)
        vol_per_step = annualised_vol * np.sqrt(dt)
        drift_per_step = -0.5 * annualised_vol ** 2 * dt   # risk-neutral drift

        # Simulate price paths: shape (n_paths, holding_bars)
        z = self._sample_z(self.n_paths, holding_bars)
        log_ret_steps = drift_per_step + vol_per_step * z
        log_prices = np.cumsum(log_ret_steps, axis=1)
        price_paths = entry_price * np.exp(log_prices)

        # Check stop and TP touch along path
        if side == "LONG":
            stop_hit = (price_paths <= stop_loss_price).any(axis=1)
            tp_hit   = (price_paths >= take_profit_price).any(axis=1) & ~stop_hit
            exit_prices = np.where(
                stop_hit, stop_loss_price,
                np.where(tp_hit, take_profit_price, price_paths[:, -1])
            )
            pnl_pct = (exit_prices - entry_price) / entry_price
        else:  # SHORT
            stop_hit = (price_paths >= stop_loss_price).any(axis=1)
            tp_hit   = (price_paths <= take_profit_price).any(axis=1) & ~stop_hit
            exit_prices = np.where(
                stop_hit, stop_loss_price,
                np.where(tp_hit, take_profit_price, price_paths[:, -1])
            )
            pnl_pct = (entry_price - exit_prices) / entry_price

        prob_stop = float(stop_hit.mean())
        prob_tp   = float(tp_hit.mean())
        ev        = float(pnl_pct.mean())
        var_95    = float(-np.percentile(pnl_pct, 5))
        cvar_95   = float(-pnl_pct[pnl_pct <= np.percentile(pnl_pct, 5)].mean())

        # Maximum adverse excursion at 95th percentile of bad paths
        if side == "LONG":
            adverse = (price_paths.min(axis=1) - entry_price) / entry_price
        else:
            adverse = (entry_price - price_paths.max(axis=1)) / entry_price
        mae_p95 = float(-np.percentile(adverse, 5))

        return TradeMonteCarloResult(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            n_paths=self.n_paths,
            prob_stop_hit=prob_stop,
            prob_tp_hit=prob_tp,
            expected_pnl_pct=ev,
            var_95=var_95,
            cvar_95=cvar_95,
            max_adverse_excursion_p95=mae_p95,
        )

    # ──────────────────────────────────────────────────────────────────────── #
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────── #

    def _simulate_paths(
        self, L: np.ndarray, drift: np.ndarray, n_steps: int, n_assets: int
    ) -> np.ndarray:
        """
        Returns shape (n_paths, n_steps, n_assets) of per-step log-returns.
        Uses GPU via PyTorch if available, otherwise NumPy.
        """
        if _TORCH and self._device != "cpu":
            return self._simulate_paths_gpu(L, drift, n_steps, n_assets)
        return self._simulate_paths_cpu(L, drift, n_steps, n_assets)

    def _simulate_paths_cpu(
        self, L: np.ndarray, drift: np.ndarray, n_steps: int, n_assets: int
    ) -> np.ndarray:
        z = self._sample_z(self.n_paths * n_steps, n_assets)
        z = z.reshape(self.n_paths, n_steps, n_assets)
        correlated = z @ L.T
        return correlated + drift[np.newaxis, np.newaxis, :]

    def _simulate_paths_gpu(
        self, L: np.ndarray, drift: np.ndarray, n_steps: int, n_assets: int
    ) -> np.ndarray:
        import torch
        device = torch.device(self._device)
        L_t = torch.tensor(L, dtype=torch.float32, device=device)
        d_t = torch.tensor(drift, dtype=torch.float32, device=device)

        z = self._sample_z_torch(self.n_paths * n_steps, n_assets, device)
        z = z.reshape(self.n_paths, n_steps, n_assets)
        correlated = z @ L_t.T + d_t
        return correlated.cpu().numpy()

    def _sample_z(self, n_rows: int, n_cols: int) -> np.ndarray:
        if self.distribution == "student-t":
            # Multivariate Student-t via ratio: z / sqrt(chi2/nu)
            z = np.random.randn(n_rows, n_cols)
            chi2 = np.random.chisquare(self.nu, (n_rows, 1))
            return z / np.sqrt(chi2 / self.nu)
        return np.random.randn(n_rows, n_cols)

    def _sample_z_torch(self, n_rows: int, n_cols: int, device) -> "torch.Tensor":
        import torch
        z = torch.randn(n_rows, n_cols, device=device)
        if self.distribution == "student-t":
            chi2 = torch.distributions.Chi2(self.nu).sample((n_rows, 1)).to(device)
            return z / torch.sqrt(chi2 / self.nu)
        return z

    def _compute_var_surface(
        self,
        cum_paths: np.ndarray,          # (n_paths, n_steps)
        horizons: list[int],
        confs: list[float],
        bars_per_day: int,
    ) -> VaRSurface:
        ann_factor = 365.25 * bars_per_day
        var_table: list[list[float]] = []
        cvar_table: list[list[float]] = []
        es_99: list[float] = []

        for h in horizons:
            step = min(h * bars_per_day, cum_paths.shape[1]) - 1
            rets = np.expm1(cum_paths[:, step])   # simple returns at horizon h
            var_row, cvar_row = [], []
            for c in confs:
                q = np.percentile(rets, (1 - c) * 100)
                tail = rets[rets <= q]
                var_row.append(float(-q))
                cvar_row.append(float(-tail.mean()) if len(tail) else float(-q))
            var_table.append(var_row)
            cvar_table.append(cvar_row)

            q99 = np.percentile(rets, 1.0)
            tail99 = rets[rets <= q99]
            es_99.append(float(-tail99.mean()) if len(tail99) else float(-q99))

        # Annualised vol from daily returns
        daily_rets = np.expm1(np.diff(cum_paths[:, ::bars_per_day], axis=1))
        ann_vol = float(daily_rets.std() * np.sqrt(365.25))

        return VaRSurface(
            horizon_days=horizons,
            confidence_levels=confs,
            var=var_table,
            cvar=cvar_table,
            es_99=es_99,
            annualised_vol=ann_vol,
        )

    def _path_max_drawdown(self, cum_paths: np.ndarray) -> np.ndarray:
        """Return per-path maximum drawdown fraction."""
        prices = np.exp(cum_paths)          # (n_paths, n_steps)
        peak = np.maximum.accumulate(prices, axis=1)
        dd = (peak - prices) / (peak + 1e-10)
        return dd.max(axis=1)               # (n_paths,)

    def _percentile_bands(self, final_rets: np.ndarray) -> dict[str, float]:
        pcts = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
        labels = ["p0.1", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99.9"]
        simple_rets = np.expm1(final_rets)
        return {lbl: float(np.percentile(simple_rets, p)) for lbl, p in zip(labels, pcts)}

    def _run_stress_scenarios(
        self,
        weights: np.ndarray,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        bars_per_day: int,
    ) -> dict[str, float]:
        """Apply instantaneous shock + elevated vol for each scenario."""
        results: dict[str, float] = {}
        n_assets = len(weights)
        n_steps = bars_per_day    # simulate 1 day under stress

        for name, params in STRESS_SCENARIOS.items():
            shock = params["shock"]
            vol_mult = params["vol_mult"]
            corr_boost = params["corr_boost"]

            # Stressed covariance: inflate vol + push correlations toward corr_boost
            vols = np.sqrt(np.diag(cov_matrix))
            stressed_vols = vols * vol_mult
            # Build correlation matrix and boost off-diagonal toward corr_boost
            corr = cov_matrix / (np.outer(vols, vols) + 1e-10)
            np.fill_diagonal(corr, 1.0)
            corr_stressed = corr + (corr_boost - corr) * (np.abs(corr) < 0.99)
            corr_stressed = np.clip(corr_stressed, -1, 1)
            np.fill_diagonal(corr_stressed, 1.0)
            stressed_cov = np.outer(stressed_vols, stressed_vols) * corr_stressed * (1.0 / 365.25)

            try:
                L_s = np.linalg.cholesky(stressed_cov + np.eye(n_assets) * 1e-8)
            except np.linalg.LinAlgError:
                results[name] = shock
                continue

            drift_s = (mean_returns * (1.0 / (365.25 * bars_per_day))
                       - 0.5 * np.diag(stressed_cov) / n_steps)
            paths = self._simulate_paths_cpu(L_s, drift_s, n_steps, n_assets)
            cum = paths.sum(axis=1)   # (n_paths, n_assets)
            port_ret = np.expm1((cum * weights).sum(axis=1) + shock)
            results[name] = float(port_ret.mean())

        return results

    def _forward_equity_bands(
        self, cum_paths: np.ndarray, equity: float, bars_per_day: int
    ) -> pd.DataFrame:
        """
        Return DataFrame with columns p5/p25/p50/p75/p95 showing equity bands
        at each daily step.
        """
        # Downsample to daily
        n_steps = cum_paths.shape[1]
        n_days  = max(1, n_steps // bars_per_day)
        daily_idx = [min((d + 1) * bars_per_day - 1, n_steps - 1) for d in range(n_days)]
        daily_cum = cum_paths[:, daily_idx]   # (n_paths, n_days)

        pcts = [5, 25, 50, 75, 95]
        data = {
            f"p{p}": equity * np.exp(np.percentile(daily_cum, p, axis=0))
            for p in pcts
        }
        return pd.DataFrame(data)

    @staticmethod
    def _pick_device(use_gpu: bool) -> str:
        if not use_gpu or not _TORCH:
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"


# ─── convenience functions ─────────────────────────────────────────────────── #

def mc_var_from_equity(
    equity_series: pd.Series,
    n_paths: int = 50_000,
    horizon_days: int = 1,
    confidence: float = 0.99,
    distribution: str = "student-t",
) -> tuple[float, float]:
    """
    Single-asset MC VaR/CVaR directly from an equity curve.
    Returns (VaR, CVaR) as fractions of current equity.
    Useful for quick integration into PortfolioRiskEngine.
    """
    rets = equity_series.pct_change().dropna()
    if len(rets) < 10:
        return 0.0, 0.0

    mu = float(rets.mean())
    sigma = float(rets.std())
    engine = PortfolioMonteCarloEngine(n_paths=n_paths, distribution=distribution)

    z = engine._sample_z(n_paths, 1).flatten()
    sim_rets = mu * horizon_days + sigma * np.sqrt(horizon_days) * z
    q = np.percentile(sim_rets, (1 - confidence) * 100)
    tail = sim_rets[sim_rets <= q]
    return float(-q), float(-tail.mean()) if len(tail) else float(-q)


def print_mc_summary(result: MonteCarloResult, equity: float) -> None:
    """Print a formatted summary of a MonteCarloResult."""
    sep = "─" * 60
    print(f"\n  Monte Carlo Risk  ({result.n_paths:,} paths  ·  {result.distribution})")
    print(f"  {sep}")
    print(result.var_surface.summary())
    print(f"  {sep}")

    bands = result.percentile_bands
    print("  P&L Distribution (fraction of portfolio)")
    print(f"    Worst 0.1%  : {bands.get('p0.1', 0):+.2%}")
    print(f"    Worst 1%    : {bands.get('p1', 0):+.2%}")
    print(f"    Worst 5%    : {bands.get('p5', 0):+.2%}")
    print(f"    Median      : {bands.get('p50', 0):+.2%}")
    print(f"    Best 5%     : {bands.get('p95', 0):+.2%}")
    print(f"    Best 1%     : {bands.get('p99', 0):+.2%}")
    print(f"  {sep}")

    dd = result.path_max_dd
    print(f"  Max Drawdown Distribution  (across all paths)")
    print(f"    Median path DD  : {np.median(dd):.2%}")
    print(f"    95th pct DD     : {np.percentile(dd, 95):.2%}")
    print(f"    99th pct DD     : {np.percentile(dd, 99):.2%}")
    print(f"  {sep}")

    if result.stress_results:
        print("  Stress Scenarios (1-day expected P&L fraction)")
        for name, ret in result.stress_results.items():
            bar = "▓" * min(int(abs(ret) * 200), 30)
            sign = "+" if ret >= 0 else "-"
            print(f"    {name:<28} {sign}{abs(ret):.2%}  {bar}")
    print()
