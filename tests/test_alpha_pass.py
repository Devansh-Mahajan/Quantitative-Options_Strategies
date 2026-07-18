"""Regression tests for the 2026-07 alpha/profitability pass."""

import numpy as np
import pandas as pd
import pytest

from strategies.base import Signal, SignalCooldown


def _sig(strategy="momentum", symbol="BTCUSDT", side="BUY", conf=0.7):
    return Signal(symbol, "spot", side, 0.0, 100.0, conf, strategy)


# --------------------------------------------------------------------------- #
# SignalCooldown (turnover throttle)
# --------------------------------------------------------------------------- #

def test_cooldown_blocks_reemission_within_gap():
    cd = SignalCooldown(gap=4)
    assert len(cd.filter([_sig()], counter=10)) == 1
    assert len(cd.filter([_sig()], counter=12)) == 0   # within gap
    assert len(cd.filter([_sig()], counter=14)) == 1   # gap elapsed


def test_cooldown_keys_on_strategy_symbol_side():
    cd = SignalCooldown(gap=4)
    assert len(cd.filter([_sig()], counter=1)) == 1
    # Different side, symbol, or strategy is NOT throttled.
    assert len(cd.filter([_sig(side="SELL")], counter=2)) == 1
    assert len(cd.filter([_sig(symbol="ETHUSDT")], counter=2)) == 1
    assert len(cd.filter([_sig(strategy="tsmom")], counter=2)) == 1


def test_cooldown_gap_zero_is_noop():
    cd = SignalCooldown(gap=0)
    sigs = [_sig(), _sig()]
    assert cd.filter(sigs, counter=1) == sigs


# --------------------------------------------------------------------------- #
# Cost-aware edge floor (position sizer)
# --------------------------------------------------------------------------- #

def test_sizer_rejects_edge_below_cost_floor(monkeypatch):
    from risk.position_sizer import PositionSizer, SizingContext

    sizer = PositionSizer()
    # Force a tiny profit target far below 3x roundtrip cost (~1.8% at Alpaca).
    context = SizingContext(
        side="BUY", price=100.0, equity=10_000.0,
        signal_confidence=0.8, realised_vol=0.5, regime="bull",
        take_profit_pct=0.004, stop_loss_pct=0.003,
    )
    decision = sizer.size_signal(context)
    assert decision.notional == 0.0
    assert decision.reason == "edge_below_cost_floor"


def test_sizer_accepts_edge_above_cost_floor():
    from risk.position_sizer import PositionSizer, SizingContext

    sizer = PositionSizer()
    context = SizingContext(
        side="BUY", price=100.0, equity=10_000.0,
        signal_confidence=0.8, realised_vol=0.5, regime="bull",
        take_profit_pct=0.06, stop_loss_pct=0.02,
    )
    decision = sizer.size_signal(context)
    assert decision.notional > 0.0


def test_roundtrip_cost_reflects_alpaca_fees(monkeypatch):
    from risk.position_sizer import PositionSizer
    from bot.config import cfg

    monkeypatch.setattr(type(cfg), "is_alpaca", property(lambda self: True))
    monkeypatch.setattr(type(cfg), "broker", property(lambda self: "alpaca"), raising=False)
    cost = PositionSizer._roundtrip_cost_pct()
    assert cost == pytest.approx(2 * 0.0025 + 0.0010)


# --------------------------------------------------------------------------- #
# Promoted strategy params (defaults must equal the historical constants)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cls_path,attr,expected", [
    ("strategies.momentum.MomentumStrategy", "_ema_fast", 9),
    ("strategies.momentum.MomentumStrategy", "_vol_ratio", 1.2),
    ("strategies.mean_reversion.MeanReversionStrategy", "_bb_std", 2.0),
    ("strategies.mean_reversion.MeanReversionStrategy", "_rsi_oversold", 30),
    ("strategies.order_flow.OrderFlowStrategy", "_imbalance_threshold", 0.65),
    ("strategies.breakout.BreakoutStrategy", "_donchian_period", 20),
    ("strategies.pairs_arb.PairsArbStrategy", "_zscore_entry", 2.0),
])
def test_promoted_param_defaults(cls_path, attr, expected):
    module_path, cls_name = cls_path.rsplit(".", 1)
    import importlib

    cls = getattr(importlib.import_module(module_path), cls_name)
    assert getattr(cls(), attr) == expected


def test_promoted_params_accept_overrides():
    from strategies.breakout import BreakoutStrategy

    s = BreakoutStrategy(donchian_period=30, vol_surge=2.0)
    assert s._donchian_period == 30 and s._vol_surge == 2.0


# --------------------------------------------------------------------------- #
# Per-strategy attribution uses the strategy's OWN equity
# --------------------------------------------------------------------------- #

def test_per_strategy_metrics_use_own_equity():
    from backtester.engine import BacktestEngine

    engine = BacktestEngine.__new__(BacktestEngine)
    engine.initial_equity = 10_000.0
    engine.lookback = 0

    idx = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
    portfolio_eq = pd.Series(np.linspace(10_000, 15_000, 100), index=idx)  # big portfolio gain

    trades = pd.DataFrame([
        # A strategy that only LOST money — its Sharpe must not inherit the
        # portfolio's +50% run.
        {"strategy": "loser", "pnl_pct": -0.02, "pnl_abs": -100.0, "holding_bars": 5, "bar_index": 10},
        {"strategy": "loser", "pnl_pct": -0.03, "pnl_abs": -150.0, "holding_bars": 5, "bar_index": 50},
    ])
    metrics = engine._per_strategy_metrics(trades, portfolio_eq)
    assert metrics["loser"].total_return_pct < 0


def test_flat_curve_sharpe_is_zero_not_sentinel():
    from backtester.metrics import compute_metrics

    idx = pd.date_range("2026-01-01", periods=50, freq="1h", tz="UTC")
    flat = pd.Series(10_000.0, index=idx)
    m = compute_metrics(flat, [], [])
    assert m.sharpe == 0.0


# --------------------------------------------------------------------------- #
# Optimizer per-strategy param threading
# --------------------------------------------------------------------------- #

def test_optimizer_rejects_unknown_tune_strategy():
    from backtester.optimizer import BacktestOptimizer

    with pytest.raises(ValueError):
        BacktestOptimizer(
            data={"X": pd.DataFrame({"close": range(2000)})},
            interval="1h",
            strategy_factory=lambda **kw: [],
            tune_strategy="nonexistent_strategy",
        )


def test_strategy_param_spaces_produce_valid_kwargs():
    from backtester.optimizer import STRATEGY_PARAM_SPACES

    class FakeTrial:
        def suggest_int(self, name, lo, hi, step=1):
            return lo

        def suggest_float(self, name, lo, hi, step=None):
            return lo

    from strategies.registry import _build_all_enabled

    instances = {s.name: type(s) for s in _build_all_enabled()}
    for name, space in STRATEGY_PARAM_SPACES.items():
        kwargs = space(FakeTrial())
        if name in instances:
            instances[name](**kwargs)  # must construct without error
