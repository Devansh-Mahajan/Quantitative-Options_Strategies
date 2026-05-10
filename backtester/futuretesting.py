"""
Futuretesting Framework — Bloch (SSRN 4647103).

The core problem with backtesting:
  A single historical price path is ONE realization of the underlying
  stochastic process. Strategies tuned to that path may not generalize.
  Backtesting on a single path gives overconfident Sharpe estimates.

Bloch's futuretesting approach:
  1. Calibrate a stochastic model (GBM, OU, jump-diffusion) to historical data
  2. Generate N synthetic price paths from the calibrated model
  3. Run your strategy on EVERY path and collect the full distribution of outcomes
  4. Accept the strategy only if it is robust ACROSS the distribution
  5. Optimize parameters on the synthetic ensemble, not the single history

This is fundamentally different from Monte Carlo on trade returns (which we
already have in monte_carlo_bt.py). Here we simulate ENTIRE price paths and
run the full strategy engine on each one — a much stronger test.

Models supported:
  - GBM (Geometric Brownian Motion): dS = μS dt + σS dW
  - GBM with jumps: adds Poisson jump process (crypto flash crashes)
  - OU (Ornstein-Uhlenbeck): dX = κ(α-X) dt + σ dW  [mean-reverting log-price]
  - Heston stochastic vol: dS = μS dt + √V S dW₁; dV = κ(θ-V) dt + ξ√V dW₂

Usage:
    from backtester.futuretesting import FuturetestEngine

    ft = FuturetestEngine(
        historical_df=data["BTCUSDT"],
        model="gbm_jumps",
        n_paths=200,
        horizon_bars=1000,
    )
    ft_result = ft.run(engine_factory, strategy_factory)
    ft_result.print_summary()
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger("backtester.futuretesting")


# ─────────────────────────────────────────────────────────────────────────────
# Parameter calibration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _GBMParams:
    mu:    float    # annualised drift
    sigma: float    # annualised volatility
    dt:    float    # time step in years

@dataclass
class _JumpParams:
    lam:    float   # jump intensity (jumps per year)
    mu_j:   float   # mean log-jump size
    sigma_j: float  # std of log-jump size

@dataclass
class _OUParams:
    kappa: float    # mean-reversion speed (per year)
    alpha: float    # long-run mean (log-price)
    sigma: float    # diffusion coefficient

@dataclass
class _HestonParams:
    mu:    float    # drift
    kappa: float    # vol mean-reversion speed
    theta: float    # long-run variance
    xi:    float    # vol of vol
    rho:   float    # correlation dW₁·dW₂


def _calibrate_gbm(log_rets: np.ndarray, bars_per_year: int) -> _GBMParams:
    mu_bar = float(log_rets.mean())
    sigma_bar = float(log_rets.std())
    mu    = mu_bar * bars_per_year
    sigma = sigma_bar * np.sqrt(bars_per_year)
    return _GBMParams(mu=mu, sigma=sigma, dt=1.0 / bars_per_year)


def _calibrate_ou(log_prices: np.ndarray, bars_per_year: int) -> _OUParams:
    """Least-squares calibration of OU: X_{t+1} = a + b·X_t + ε."""
    X = log_prices[:-1]
    Y = log_prices[1:]
    b = float(np.cov(X, Y)[0, 1] / (np.var(X) + 1e-10))
    a = float(Y.mean() - b * X.mean())
    kappa = -np.log(b) * bars_per_year
    alpha = a / (1 - b)
    sigma = float((Y - a - b * X).std() * np.sqrt(bars_per_year))
    return _OUParams(kappa=max(kappa, 0.01), alpha=alpha, sigma=sigma)


def _calibrate_jumps(log_rets: np.ndarray, bars_per_year: int) -> _JumpParams:
    """Simple jump detection via return quantiles."""
    threshold = log_rets.std() * 3
    jumps = log_rets[np.abs(log_rets) > threshold]
    lam   = len(jumps) / len(log_rets) * bars_per_year
    mu_j  = float(jumps.mean()) if len(jumps) > 0 else 0.0
    sigma_j = float(jumps.std()) if len(jumps) > 1 else 0.02
    return _JumpParams(lam=lam, mu_j=mu_j, sigma_j=sigma_j)


# ─────────────────────────────────────────────────────────────────────────────
# Path simulators
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_gbm(p: _GBMParams, n_bars: int, S0: float, rng: np.random.Generator) -> np.ndarray:
    """Geometric Brownian Motion."""
    dW = rng.normal(0, 1, n_bars) * np.sqrt(p.dt)
    log_S = np.log(S0) + np.cumsum((p.mu - 0.5 * p.sigma ** 2) * p.dt + p.sigma * dW)
    return np.exp(log_S)


def _simulate_gbm_jumps(
    p: _GBMParams, jp: _JumpParams, n_bars: int, S0: float, rng: np.random.Generator
) -> np.ndarray:
    """GBM + compound Poisson jumps (crypto crash model)."""
    dW   = rng.normal(0, 1, n_bars) * np.sqrt(p.dt)
    n_j  = rng.poisson(jp.lam * p.dt, n_bars)
    J    = np.array([
        rng.normal(jp.mu_j, jp.sigma_j, int(n)).sum() if n > 0 else 0.0
        for n in n_j
    ])
    log_S = np.log(S0) + np.cumsum((p.mu - 0.5 * p.sigma ** 2) * p.dt + p.sigma * dW + J)
    return np.exp(log_S)


def _simulate_ou(p: _OUParams, n_bars: int, X0: float, dt_yr: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Ornstein-Uhlenbeck for log-price (mean-reverting)."""
    X = np.zeros(n_bars + 1)
    X[0] = X0
    sigma_bar = p.sigma * np.sqrt(dt_yr)
    for t in range(n_bars):
        X[t + 1] = X[t] + p.kappa * (p.alpha - X[t]) * dt_yr + rng.normal(0, sigma_bar)
    return np.exp(X[1:])


def _simulate_heston(
    p: _HestonParams, n_bars: int, S0: float, V0: float, dt_yr: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Full Euler-Maruyama Heston model."""
    dW1 = rng.normal(0, 1, (n_bars, 2))
    # Correlate W1 and W2
    dW2 = p.rho * dW1[:, 0] + np.sqrt(1 - p.rho ** 2) * dW1[:, 1]
    dW1 = dW1[:, 0]

    S = np.zeros(n_bars + 1); S[0] = S0
    V = np.zeros(n_bars + 1); V[0] = V0
    sqrt_dt = np.sqrt(dt_yr)

    for t in range(n_bars):
        V_pos = max(V[t], 0.0)
        V[t+1] = V[t] + p.kappa * (p.theta - V[t]) * dt_yr + p.xi * np.sqrt(V_pos) * sqrt_dt * dW2[t]
        V[t+1] = max(V[t+1], 0.0)
        S[t+1] = S[t] * np.exp((p.mu - 0.5 * V_pos) * dt_yr + np.sqrt(V_pos) * sqrt_dt * dW1[t])

    return S[1:]


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FuturetestResult:
    model:            str
    n_paths:          int
    sharpe_dist:      np.ndarray      # Sharpe across all paths
    return_dist:      np.ndarray      # total return across all paths
    max_dd_dist:      np.ndarray      # max drawdown across all paths
    win_rate_dist:    np.ndarray
    failed_paths:     int             # paths where strategy crashed
    acceptance_rate:  float           # fraction of paths with Sharpe > 0.5
    results:          list = field(default_factory=list)  # per-path BacktestResult objects

    def print_summary(self) -> None:
        def pct(arr: np.ndarray, p: int) -> str:
            return f"{float(np.percentile(arr, p)):+.2f}"

        print("\n" + "═" * 60)
        print(f"  FUTURETEST SUMMARY  ({self.model.upper()}  ·  {self.n_paths} paths)")
        print("═" * 60)
        print(f"  {'Metric':<22} {'p5':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p95':>8}")
        print(f"  {'─'*54}")
        for label, arr in [
            ("Sharpe",        self.sharpe_dist),
            ("Total Return %", self.return_dist * 100),
            ("Max Drawdown %", self.max_dd_dist * 100),
            ("Win Rate %",    self.win_rate_dist * 100),
        ]:
            print(f"  {label:<22} "
                  f"{pct(arr,5):>8} {pct(arr,25):>8} {pct(arr,50):>8} "
                  f"{pct(arr,75):>8} {pct(arr,95):>8}")
        print(f"  {'─'*54}")
        print(f"  Acceptance rate (Sharpe>0.5):  {self.acceptance_rate:.1%}")
        print(f"  Failed paths:                  {self.failed_paths}/{self.n_paths}")

        # ASCII histogram of Sharpe distribution
        bins  = np.linspace(-2, 3, 12)
        hist, _ = np.histogram(self.sharpe_dist, bins=bins)
        max_h  = hist.max() or 1
        print("\n  Sharpe distribution across synthetic paths:")
        for i, (b, count) in enumerate(zip(bins[:-1], hist)):
            bar = "█" * int(count / max_h * 20)
            print(f"  [{b:+.1f}] {bar:<20} {count}")
        print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────

class FuturetestEngine:
    """
    Bloch's futuretesting: test a strategy across N synthetic price paths.

    Args:
        historical_df:    historical OHLCV DataFrame (one symbol)
        model:            "gbm" | "gbm_jumps" | "ou" | "heston"
        n_paths:          number of synthetic price paths
        horizon_bars:     how many bars to simulate per path
        bars_per_year:    calendar bars in one year (8760 for 1h crypto)
        seed:             random seed for reproducibility
    """

    def __init__(
        self,
        historical_df: pd.DataFrame,
        model: str = "gbm_jumps",
        n_paths: int = 200,
        horizon_bars: int = 1000,
        bars_per_year: int = 8760,
        seed: int = 42,
    ) -> None:
        self._df           = historical_df.reset_index()
        self._model        = model
        self._n_paths      = n_paths
        self._horizon      = horizon_bars
        self._bars_per_yr  = bars_per_year
        self._rng          = np.random.default_rng(seed)

        # Calibrate from historical data
        if "close" not in self._df.columns:
            raise ValueError("DataFrame must have 'close' column")
        closes  = self._df["close"].astype(float).values
        log_rets = np.diff(np.log(closes))

        self._gbm_p  = _calibrate_gbm(log_rets, bars_per_year)
        self._ou_p   = _calibrate_ou(np.log(closes), bars_per_year)
        self._jump_p = _calibrate_jumps(log_rets, bars_per_year)
        self._S0     = float(closes[-1])
        self._V0     = float(log_rets[-252:].std() ** 2 * bars_per_year)

        log.info(
            "FuturetestEngine calibrated: model=%s  μ=%.2f%%  σ=%.1f%%  κ=%.2f  "
            "λ_jump=%.2f/yr  S0=%.2f",
            model, self._gbm_p.mu * 100, self._gbm_p.sigma * 100,
            self._ou_p.kappa, self._jump_p.lam, self._S0,
        )

    def _generate_path(self) -> np.ndarray:
        """Generate one synthetic price path."""
        m = self._model
        n = self._horizon
        dt = self._gbm_p.dt

        if m == "gbm":
            return _simulate_gbm(self._gbm_p, n, self._S0, self._rng)
        elif m == "gbm_jumps":
            return _simulate_gbm_jumps(self._gbm_p, self._jump_p, n, self._S0, self._rng)
        elif m == "ou":
            return _simulate_ou(self._ou_p, n, np.log(self._S0), dt, self._rng)
        elif m == "heston":
            hp = _HestonParams(
                mu=self._gbm_p.mu, kappa=self._ou_p.kappa,
                theta=self._V0, xi=0.5, rho=-0.7,
            )
            return _simulate_heston(hp, n, self._S0, self._V0, dt, self._rng)
        else:
            raise ValueError(f"Unknown model: {m}")

    def _path_to_df(self, prices: np.ndarray) -> pd.DataFrame:
        """Convert price array to OHLCV DataFrame for the engine."""
        n = len(prices)
        noise = self._rng.uniform(0.998, 1.002, n)
        dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame({
            "open_time": (dates.astype(np.int64) // 10 ** 6).values,
            "open":      prices * noise,
            "high":      prices * self._rng.uniform(1.000, 1.008, n),
            "low":       prices * self._rng.uniform(0.992, 1.000, n),
            "close":     prices,
            "volume":    self._rng.uniform(100, 1000, n),
        })

    def run(
        self,
        engine_factory: Callable,
        strategy_factory: Callable,
        symbol: str = "BTCUSDT",
        verbose: bool = True,
        progress_callback: Callable | None = None,
    ) -> FuturetestResult:
        """
        Run the strategy on N synthetic paths.

        Args:
            engine_factory:   callable(sim_df: DataFrame) → BacktestEngine
            strategy_factory: callable() → list[strategy]  (unused in call; kept for API compat)
            symbol:           symbol name for the synthetic data
            verbose:          print progress every 10%
        """
        sharpes, returns, max_dds, win_rates = [], [], [], []
        path_results = []
        failed = 0

        def _emit_progress(path_index: int) -> None:
            if progress_callback is None:
                return
            try:
                progress = 100.0 * (path_index / max(self._n_paths, 1))
                progress_callback(
                    {
                        "path_index": path_index,
                        "n_paths": self._n_paths,
                        "progress": round(progress, 1),
                        "message": f"Running synthetic path {path_index}/{self._n_paths}",
                    }
                )
            except Exception as exc:
                log.debug("Futuretest progress callback failed: %s", exc)

        log.info("Futuretesting: %d paths × %d bars  model=%s", self._n_paths, self._horizon, self._model)
        _emit_progress(0)

        for i in range(self._n_paths):
            if verbose and i % max(1, self._n_paths // 10) == 0:
                log.info("  path %d/%d", i, self._n_paths)
            try:
                path = self._generate_path()
                df   = self._path_to_df(path)
                engine = engine_factory(df)
                path_result = engine.run()

                m = path_result.metrics
                sharpes.append(m.sharpe)
                returns.append(m.total_return_pct / 100)
                max_dds.append(abs(m.max_drawdown_pct) / 100)
                wr = m.win_rate_pct / 100 if m.win_rate_pct else 0.0
                win_rates.append(wr)
                path_results.append(path_result)

            except Exception as exc:
                log.debug("Path %d failed: %s", i, exc)
                failed += 1
                sharpes.append(-5.0)
                returns.append(-1.0)
                max_dds.append(1.0)
                win_rates.append(0.0)
            finally:
                _emit_progress(i + 1)

        sharpe_arr   = np.array(sharpes)
        acceptance   = float((sharpe_arr > 0.5).mean())

        result = FuturetestResult(
            model=self._model,
            n_paths=self._n_paths,
            sharpe_dist=sharpe_arr,
            return_dist=np.array(returns),
            max_dd_dist=np.array(max_dds),
            win_rate_dist=np.array(win_rates),
            failed_paths=failed,
            acceptance_rate=acceptance,
            results=path_results,
        )
        result.print_summary()
        return result
