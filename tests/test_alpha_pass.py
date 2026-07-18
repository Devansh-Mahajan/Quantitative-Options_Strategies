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


# --------------------------------------------------------------------------- #
# New cost-aware strategies (2026-07)
# --------------------------------------------------------------------------- #

class _FakeStore:
    """Minimal store: one symbol of hourly bars."""

    def __init__(self, closes, symbol="BTCUSDT"):
        n = len(closes)
        idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
        c = np.asarray(closes, dtype=float)
        self._df = pd.DataFrame({
            "open_time": [int(t.timestamp() * 1000) for t in idx],  # epoch ms
            "open": c, "high": c * 1.002, "low": c * 0.998, "close": c,
            "volume": np.full(n, 1e5),
        })
        self._symbol = symbol
        self.candles = {(symbol, "1h"): object()}

    def get_history_df(self, symbol, interval):
        return self._df if symbol == self._symbol else pd.DataFrame()

    def get_closes(self, symbol, interval, n):
        if symbol != self._symbol:
            return []
        return self._df["close"].tolist()[-n:]


def _hourly_from_daily(daily_closes):
    """Expand daily closes to flat hourly bars (24 per day)."""
    out = []
    for value in daily_closes:
        out.extend([value] * 24)
    return out


def test_trend_follow_daily_fires_on_breakout(monkeypatch):
    from strategies.trend_follow_daily import TrendFollowDailyStrategy

    # 80 flat days then a strong 8-day ramp to new highs.
    daily = [100.0] * 80 + [100 + 3 * i for i in range(1, 9)]
    store = _FakeStore(_hourly_from_daily(daily))
    strat = TrendFollowDailyStrategy(eval_every=1)
    monkeypatch.setattr(type(strat), "symbols", property(lambda self: ["BTCUSDT"]))

    sigs = strat.generate_signals(store, "bull", {})
    assert len(sigs) == 1 and sigs[0].side == "BUY"
    assert 0.5 <= sigs[0].confidence <= 0.9
    # No re-entry while the internal position is open.
    assert strat.generate_signals(store, "bull", {}) == []


def test_trend_follow_daily_silent_on_flat(monkeypatch):
    from strategies.trend_follow_daily import TrendFollowDailyStrategy

    store = _FakeStore(_hourly_from_daily([100.0] * 90))
    strat = TrendFollowDailyStrategy(eval_every=1)
    monkeypatch.setattr(type(strat), "symbols", property(lambda self: ["BTCUSDT"]))
    assert strat.generate_signals(store, "bull", {}) == []


def test_dip_buyer_fires_on_capitulation_in_uptrend(monkeypatch):
    from strategies.dip_buyer import DipBuyerStrategy

    # Steady uptrend for 120 days, then a sharp 3-day dump that keeps price
    # above the 60-day mean.
    daily = [100 + 0.8 * i for i in range(120)]
    peak = daily[-1]
    daily += [peak * 0.97, peak * 0.94, peak * 0.90]
    store = _FakeStore(_hourly_from_daily(daily))
    strat = DipBuyerStrategy(eval_every=1)
    monkeypatch.setattr(type(strat), "symbols", property(lambda self: ["BTCUSDT"]))

    sigs = strat.generate_signals(store, "bull", {})
    assert len(sigs) == 1 and sigs[0].side == "BUY"
    assert sigs[0].meta["mode"] == "dip_entry"


def test_dip_buyer_ignores_dip_in_downtrend(monkeypatch):
    from strategies.dip_buyer import DipBuyerStrategy

    # Long downtrend then a further dump: dip triggers but the trend gate fails.
    daily = [200 - 0.8 * i for i in range(120)]
    last = daily[-1]
    daily += [last * 0.97, last * 0.94, last * 0.90]
    store = _FakeStore(_hourly_from_daily(daily))
    strat = DipBuyerStrategy(eval_every=1)
    monkeypatch.setattr(type(strat), "symbols", property(lambda self: ["BTCUSDT"]))
    assert strat.generate_signals(store, "bear", {}) == []


class _MultiStore:
    def __init__(self, series_map):
        self._stores = {s: _FakeStore(c, s) for s, c in series_map.items()}
        self.candles = {(s, "1h"): object() for s in series_map}

    def get_history_df(self, symbol, interval):
        return self._stores[symbol].get_history_df(symbol, interval)

    def get_closes(self, symbol, interval, n):
        return self._stores[symbol].get_closes(symbol, interval, n)


def test_weekly_rotation_picks_leaders(monkeypatch):
    from strategies.weekly_momentum_rotation import WeeklyMomentumRotationStrategy

    n_days = 40
    series = {
        "AAAUSDT": _hourly_from_daily([100 * (1.01 ** i) for i in range(n_days)]),   # strong up
        "BBBUSDT": _hourly_from_daily([100 * (1.005 ** i) for i in range(n_days)]),  # mild up
        "CCCUSDT": _hourly_from_daily([100 * (0.995 ** i) for i in range(n_days)]),  # down
        "DDDUSDT": _hourly_from_daily([100.0] * n_days),                              # flat
    }
    store = _MultiStore(series)
    strat = WeeklyMomentumRotationStrategy(rebal_cycles=1)
    monkeypatch.setattr(type(strat), "symbols", property(lambda self: list(series)))

    sigs = strat.generate_signals(store, "bull", {})
    buys = {s.symbol for s in sigs if s.side == "BUY"}
    assert buys == {"AAAUSDT", "BBBUSDT"}   # top-2 with positive momentum

    # Next rebalance with the same data: no churn (already held).
    assert strat.generate_signals(store, "bull", {}) == []
