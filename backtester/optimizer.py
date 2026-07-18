"""
Bayesian walk-forward parameter optimizer.

How elite hedge funds approach this:
  - Never grid-search (exponential complexity, guaranteed overfit)
  - Use Bayesian optimization (Optuna TPE) to efficiently navigate parameter space
  - Evaluate on OOS data only via mini walk-forward — prevents in-sample overfit
  - Objective is composite: Sharpe + Calmar + PF edge, minus an overfit penalty
  - Top-N parameter sets are returned; the BEST is not blindly deployed — you
    look for robustness (params that score well across multiple random seeds)

Usage:
    from backtester.optimizer import BacktestOptimizer
    opt = BacktestOptimizer(data, interval="1h", strategy_factory=factory, n_trials=100)
    result = opt.run()
    print(result.best_params)
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger("backtester.optimizer")


@dataclass
class OptimizationResult:
    best_params: dict
    best_score: float
    n_trials: int
    n_completed: int
    top_trials: list[dict]           # top-5 by score
    param_importances: dict          # which params mattered most
    robustness_score: float          # std of top-5 scores (lower = more robust)
    holdout_score: float = float("nan")   # best params on the untouched final window
    holdout_metrics: dict | None = None   # sharpe/return/max_dd on the holdout


# --------------------------------------------------------------------------- #
# Parameter search space definition
# --------------------------------------------------------------------------- #

def _suggest_params(trial) -> dict:
    """
    Define the hyper-parameter search space.
    Constraints are encoded inline (TP must beat SL by at least 1.5x).
    """
    stop_loss = trial.suggest_float("stop_loss_pct", 0.02, 0.12, step=0.005)
    # TP floor = 1.5× SL ensures minimum R:R of 1.5 even at ~40% win rate
    tp_min = round(stop_loss * 1.5, 3)
    take_profit = trial.suggest_float("take_profit_pct", tp_min, 0.30, step=0.005)
    max_positions = trial.suggest_int("max_open_positions", 3, 12)
    lookback = trial.suggest_int("lookback", 20, 100, step=10)
    max_risk = trial.suggest_float("max_risk_per_trade", 0.005, 0.04, step=0.005)

    return dict(
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
        max_open_positions=max_positions,
        lookback=lookback,
        max_risk_per_trade=max_risk,
    )


# Per-strategy Optuna search spaces. Keys are strategy names; each function
# returns __init__ kwargs for that strategy class (all promoted/tunable).
# Used when BacktestOptimizer is constructed with tune_strategy=<name>.
STRATEGY_PARAM_SPACES: dict = {
    "tsmom": lambda t: dict(
        lookback=t.suggest_int("s_lookback", 48, 336, step=24),
        entry_threshold=t.suggest_float("s_entry_threshold", 0.05, 0.30, step=0.05),
        rebal_bars=t.suggest_int("s_rebal_bars", 2, 12, step=2),
    ),
    "statistical_arb": lambda t: dict(
        entry_z=t.suggest_float("s_entry_z", 1.6, 3.2, step=0.2),
        exit_z=t.suggest_float("s_exit_z", 0.2, 1.0, step=0.2),
    ),
    "mean_reversion": lambda t: dict(
        bb_std=t.suggest_float("s_bb_std", 1.8, 3.0, step=0.2),
        rsi_oversold=t.suggest_int("s_rsi_oversold", 18, 34, step=4),
        rsi_overbought=t.suggest_int("s_rsi_overbought", 66, 82, step=4),
    ),
    "breakout": lambda t: dict(
        donchian_period=t.suggest_int("s_donchian_period", 16, 48, step=8),
        vol_surge=t.suggest_float("s_vol_surge", 1.2, 2.4, step=0.2),
    ),
    "hp_trend": lambda t: dict(
        ma_fast=t.suggest_int("s_ma_fast", 6, 18, step=3),
        ma_slow=t.suggest_int("s_ma_slow", 24, 72, step=8),
        rebal_bars=t.suggest_int("s_rebal_bars", 2, 8, step=2),
    ),
    "vpin_flow": lambda t: dict(),  # engine-level params only (thresholds internal)
    "quant_factors": lambda t: dict(
        ibs_low=t.suggest_float("s_ibs_low", 0.10, 0.30, step=0.05),
        ibs_high=t.suggest_float("s_ibs_high", 0.70, 0.90, step=0.05),
    ),
    "momentum_carry_combo": lambda t: dict(
        entry_score=t.suggest_float("s_entry_score", 0.10, 0.40, step=0.05),
        rebal_bars=t.suggest_int("s_rebal_bars", 4, 16, step=4),
    ),
    "momentum": lambda t: dict(
        ema_fast=t.suggest_int("s_ema_fast", 6, 14, step=2),
        ema_slow=t.suggest_int("s_ema_slow", 16, 34, step=3),
        vol_ratio=t.suggest_float("s_vol_ratio", 1.0, 1.8, step=0.2),
    ),
}


def _composite_score(oos_metrics, is_metrics, degradation_pct: float) -> float:
    """
    Composite objective maximised by the optimizer.

    Components:
      - OOS Sharpe  (40%) — risk-adjusted return quality
      - OOS Calmar  (25%) — return / max-drawdown
      - OOS PF edge (20%) — profit factor above 1.0, capped at 2.0
      - Overfit penalty (15%) — how much IS→OOS performance degrades
    """
    sharpe = float(oos_metrics.sharpe or 0)
    calmar = float(oos_metrics.calmar or 0)
    pf_edge = min(max(float(oos_metrics.profit_factor or 0) - 1.0, 0.0), 2.0)
    # Penalise degradation above 30%
    overfit_penalty = max((degradation_pct - 30.0) / 100.0, 0.0)

    score = (0.40 * sharpe + 0.25 * calmar + 0.20 * pf_edge - 0.15 * overfit_penalty)

    # Hard reject: if OOS is deeply negative, return very low score
    if sharpe < -2.0 or float(oos_metrics.max_drawdown_pct or 0) < -70.0:
        return -10.0
    return score


# --------------------------------------------------------------------------- #
# Main optimizer
# --------------------------------------------------------------------------- #

class BacktestOptimizer:
    """
    Bayesian parameter optimizer using Optuna + walk-forward OOS scoring.

    Args:
        data:             {symbol: DataFrame} — same as BacktestEngine
        interval:         bar interval string
        strategy_factory: callable() → list[strategy]; called fresh each trial
        n_trials:         Bayesian trials (50 is fast, 200 is thorough)
        n_wf_folds:       mini walk-forward folds per trial (3 is typical)
        initial_equity:   starting equity for all trial engines
        output_dir:       where to save optimal_params.json
    """

    def __init__(
        self,
        data: dict,
        interval: str,
        strategy_factory: Callable,
        n_trials: int = 60,
        n_wf_folds: int = 3,
        initial_equity: float = 10_000,
        output_dir: Path | None = None,
        holdout_fraction: float = 0.20,
        tune_strategy: str | None = None,
    ) -> None:
        # When tune_strategy is set, each trial ALSO suggests that strategy's
        # __init__ kwargs (STRATEGY_PARAM_SPACES) and strategy_factory must
        # accept them as keyword arguments.
        self.tune_strategy = tune_strategy
        if tune_strategy is not None and tune_strategy not in STRATEGY_PARAM_SPACES:
            raise ValueError(f"no param space defined for strategy '{tune_strategy}'")
        # Reserve the FINAL holdout_fraction of bars as an untouched test
        # window. Optuna only ever sees the earlier portion; without this, the
        # reported best_value is the max over n_trials evaluations of the same
        # OOS folds — selection bias masquerading as edge.
        self.holdout_fraction = max(0.0, min(0.4, float(holdout_fraction)))
        self.data, self.holdout_data = self._split_holdout(data, self.holdout_fraction)
        self.interval = interval
        self.strategy_factory = strategy_factory
        self.n_trials = n_trials
        self.n_wf_folds = n_wf_folds
        self.initial_equity = initial_equity
        self.output_dir = output_dir or Path(__file__).resolve().parents[1] / "backtest_reports"

    @staticmethod
    def _split_holdout(data: dict, fraction: float, warmup_bars: int = 100) -> tuple[dict, dict]:
        if fraction <= 0.0:
            return data, {}
        opt_data, holdout = {}, {}
        for sym, df in data.items():
            cut = int(len(df) * (1.0 - fraction))
            if cut < warmup_bars + 10 or len(df) - cut < 50:
                return data, {}  # too little data to split — skip holdout
            opt_data[sym] = df.iloc[:cut]
            holdout[sym] = df.iloc[max(0, cut - warmup_bars):]
        return opt_data, holdout

    def _evaluate_holdout(self, best_params: dict) -> tuple[float, dict | None]:
        if not self.holdout_data:
            return float("nan"), None
        from backtester.engine import BacktestEngine

        # Split trial params: "s_" prefix = strategy __init__ kwargs (threaded
        # through the factory), everything else = engine kwargs.
        engine_params = {k: v for k, v in best_params.items() if not k.startswith("s_")}
        strategy_params = {k[2:]: v for k, v in best_params.items() if k.startswith("s_")}

        try:
            strategies = (
                self.strategy_factory(**strategy_params) if strategy_params else self.strategy_factory()
            )
            engine = BacktestEngine(
                data=self.holdout_data,
                interval=self.interval,
                initial_equity=self.initial_equity,
                strategies=strategies,
                **engine_params,
            )
            result = engine.run()
            m = result.metrics
            return float(m.sharpe or 0.0), {
                "sharpe": float(m.sharpe or 0.0),
                "total_return_pct": float(m.total_return_pct or 0.0),
                "max_drawdown_pct": float(m.max_drawdown_pct or 0.0),
                "num_trades": int(m.num_trades or 0),
            }
        except Exception as exc:
            log.warning("Holdout evaluation failed: %s", exc)
            return float("nan"), None

    # ------------------------------------------------------------------ #
    # Optuna objective
    # ------------------------------------------------------------------ #

    def _objective(self, trial) -> float:
        import optuna
        params = _suggest_params(trial)

        strategy_params: dict = {}
        if self.tune_strategy is not None:
            strategy_params = STRATEGY_PARAM_SPACES[self.tune_strategy](trial)

        from backtester.walk_forward import WalkForwardValidator

        factory = self.strategy_factory
        if strategy_params:
            base_factory = self.strategy_factory
            factory = lambda: base_factory(**strategy_params)  # noqa: E731

        validator = WalkForwardValidator(
            data=self.data,
            interval=self.interval,
            n_folds=self.n_wf_folds,
            test_fraction=0.25,
            anchored=True,
            engine_kwargs=dict(
                initial_equity=self.initial_equity,
                **params,
            ),
            strategy_factory=factory,
        )
        try:
            wf = validator.run(verbose=False)
            if not wf.folds:
                return -5.0
            score = _composite_score(
                wf.oos_combined_metrics,
                wf.is_combined_metrics,
                wf.degradation_pct,
            )
            # Report intermediate value so MedianPruner can kill bad trials
            trial.report(score, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return score
        except Exception as exc:
            log.debug("Trial %d failed: %s", trial.number, exc)
            return -5.0

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def run(self, verbose: bool = True) -> OptimizationResult:
        try:
            import optuna
        except ImportError:
            raise ImportError("Run: pip install optuna")

        optuna.logging.set_verbosity(
            optuna.logging.INFO if verbose else optuna.logging.WARNING
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        )

        log.info(
            "Optimizer starting — %d trials × %d WF folds × %d strategies",
            self.n_trials, self.n_wf_folds, len(self.strategy_factory()),
        )

        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        completed = [t for t in study.trials if t.value is not None]
        top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
        top5_scores = [t.value for t in top5]

        # Parameter importances (requires ≥ 4 completed trials)
        importances = {}
        try:
            importances = optuna.importance.get_param_importances(study)
        except Exception:
            pass

        robustness = float(np.std(top5_scores)) if len(top5_scores) > 1 else 0.0

        holdout_score, holdout_metrics = self._evaluate_holdout(study.best_params)

        result = OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=self.n_trials,
            n_completed=len(completed),
            top_trials=[{"params": t.params, "score": t.value} for t in top5],
            param_importances=importances,
            robustness_score=robustness,
            holdout_score=holdout_score,
            holdout_metrics=holdout_metrics,
        )

        self._save(result)
        self._print(result)
        return result

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    def _save(self, result: OptimizationResult) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "optimal_params.json"
        payload = {
            "best_params": result.best_params,
            "best_score": result.best_score,
            "n_trials": result.n_trials,
            "n_completed": result.n_completed,
            "robustness_score": result.robustness_score,
            "param_importances": result.param_importances,
            "top_trials": result.top_trials,
            "holdout_score": None if result.holdout_score != result.holdout_score else result.holdout_score,
            "holdout_metrics": result.holdout_metrics,
        }
        path.write_text(json.dumps(payload, indent=2))
        log.info("Optimal params saved → %s", path)

    def _print(self, result: OptimizationResult) -> None:
        print("\n" + "═" * 58)
        print("  OPTIMIZATION RESULTS")
        print("═" * 58)
        print(f"  Trials completed : {result.n_completed} / {result.n_trials}")
        print(f"  Best score       : {result.best_score:.4f}  (selection-biased — trust holdout)")
        if result.holdout_metrics:
            print(f"  HOLDOUT Sharpe   : {result.holdout_score:.4f}  "
                  f"(return {result.holdout_metrics['total_return_pct']:+.2f}%, "
                  f"maxDD {result.holdout_metrics['max_drawdown_pct']:.2f}%, "
                  f"{result.holdout_metrics['num_trades']} trades on untouched data)")
        print(f"  Robustness (σ)   : {result.robustness_score:.4f}  (lower = more robust)")
        print()
        print("  Best parameters found:")
        for k, v in result.best_params.items():
            print(f"    {k:<28} {v}")
        if result.param_importances:
            print()
            print("  Parameter importance (what drove performance):")
            for k, v in sorted(result.param_importances.items(), key=lambda x: -x[1]):
                bar = "█" * int(v * 20)
                print(f"    {k:<28} {bar:<20} {v:.1%}")
        print()
        print("  Top-5 trials:")
        for i, t in enumerate(result.top_trials, 1):
            print(f"    #{i}  score={t['score']:.4f}  "
                  f"SL={t['params'].get('stop_loss_pct', '?'):.3f}  "
                  f"TP={t['params'].get('take_profit_pct', '?'):.3f}  "
                  f"pos={t['params'].get('max_open_positions', '?')}")
        print("═" * 58)
