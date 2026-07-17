"""Regression tests for the 2026-07 system overhaul (alpha fixes)."""

import json
import numpy as np
import pandas as pd
import pytest

from strategies.base import Signal


# --------------------------------------------------------------------------- #
# Phase A — strategy-layer fixes
# --------------------------------------------------------------------------- #

def _ohlcv(n=120, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 1.5, n)
    low = close - rng.uniform(0.1, 1.5, n)
    open_ = close + rng.normal(0, 0.5, n)
    vol = rng.uniform(1e5, 5e5, n)
    return np.column_stack([open_, high, low, close, vol])


def test_knn_features_no_longer_crash():
    """The old TR computation raised ValueError on every call (dead strategy)."""
    from strategies.knn_predictor import _features

    feats = _features(_ohlcv())
    assert feats is not None
    assert len(feats) == 9
    assert np.all(np.isfinite(feats))


def test_registry_venue_gating(monkeypatch):
    from strategies import registry
    from bot.config import cfg

    monkeypatch.setattr(type(cfg), "is_alpaca", property(lambda self: True))
    alpaca_names = {s.name for s in registry.build_registry()}
    assert registry.ALPACA_INCOMPATIBLE.isdisjoint(alpaca_names)

    monkeypatch.setattr(type(cfg), "is_alpaca", property(lambda self: False))
    binance_names = {s.name for s in registry.build_registry()}
    # On Binance the gated strategies come back (those with enabled flags)
    assert "gamma_scalping" in binance_names or "market_making" in binance_names

    # Canonical names ignore venue gating entirely and are order-stable.
    canonical = registry.canonical_strategy_names()
    assert canonical == registry.canonical_strategy_names()
    assert set(binance_names) <= set(canonical)


def test_momentum_regime_killswitch_triggers_at_floor():
    from strategies.momentum import MomentumStrategy

    strat = MomentumStrategy()

    class EmptyStore:
        candles = {}

        def get_history_df(self, *a):
            raise AssertionError("should suppress before touching data")

    # ranging maps to exactly 0.3 — with `<=` this now suppresses entirely.
    assert strat.generate_signals(EmptyStore(), "ranging", {}) == []


def test_pair_legs_are_tagged():
    import inspect
    from strategies import statistical_arb, pairs_arb, rma_strategy

    for mod in (statistical_arb, pairs_arb, rma_strategy):
        assert "pair_id" in inspect.getsource(mod)


# --------------------------------------------------------------------------- #
# Phase B — orchestrator signal path
# --------------------------------------------------------------------------- #

def _sig(symbol, side, conf, strategy="momentum", meta=None):
    return Signal(symbol, "spot", side, 0.0, 100.0, conf, strategy, meta=meta)


class _OrchestratorHarness:
    """Bare object exposing just the methods under test."""

    def __new__(cls, long_only_spot=True, positions=None):
        from bot.orchestrator import Orchestrator

        inst = object.__new__(Orchestrator)
        inst.long_only_spot = long_only_spot
        inst._held = positions or {}
        inst._position_for_symbol = lambda sym: inst._held.get(sym)
        return inst


def test_netting_drops_conflicting_signals():
    orch = _OrchestratorHarness()
    signals = [
        _sig("BTCUSDT", "BUY", 0.5, "momentum"),
        _sig("BTCUSDT", "SELL", 0.45, "mean_reversion"),
        _sig("ETHUSDT", "BUY", 0.7, "breakout"),
    ]
    netted = orch._net_conflicting_signals(signals)
    symbols = [s.symbol for s in netted]
    # BTC nets to |0.5-0.45|=0.05 < 0.2 -> dropped entirely; ETH untouched.
    assert symbols == ["ETHUSDT"]


def test_netting_keeps_dominant_side():
    orch = _OrchestratorHarness()
    signals = [
        _sig("BTCUSDT", "BUY", 0.9, "momentum"),
        _sig("BTCUSDT", "BUY", 0.4, "tsmom"),
        _sig("BTCUSDT", "SELL", 0.3, "mean_reversion"),
    ]
    netted = orch._net_conflicting_signals(signals)
    assert len(netted) == 1
    assert netted[0].side == "BUY"
    assert netted[0].confidence <= 0.9


def test_orphan_pair_leg_filter(monkeypatch):
    import bot.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "record_trade_decision", lambda **k: None)
    monkeypatch.setattr(orch_mod.cfg, "is_crypto_symbol", lambda s: True, raising=False)

    orch = _OrchestratorHarness(long_only_spot=True, positions={})  # nothing held
    pair = {"pair_id": "statistical_arb:BTCUSDT/ETHUSDT"}
    signals = [
        _sig("BTCUSDT", "SELL", 0.6, "statistical_arb", meta=pair),
        _sig("ETHUSDT", "BUY", 0.6, "statistical_arb", meta=pair),
        _sig("SOLUSDT", "BUY", 0.5, "momentum"),
    ]
    kept = orch._filter_orphan_pair_legs(signals)
    assert [s.symbol for s in kept] == ["SOLUSDT"]


def test_rl_allocator_rejects_model_without_metadata(tmp_path, monkeypatch):
    from ml import rl_allocator as rl

    class FakePPO:
        pass

    alloc = rl.RLAllocator(5)
    alloc._model = FakePPO()
    alloc.metadata = None
    assert not alloc.is_valid_for(["a", "b", "c", "d", "e"])

    alloc.metadata = {"trained_on": "backtest_returns", "strategy_names": ["a", "b"]}
    assert not alloc.is_valid_for(["a", "b", "c"])  # name mismatch

    names = ["a", "b", "c"]
    alloc.metadata = {"trained_on": "backtest_returns", "strategy_names": names}
    assert alloc.is_valid_for(names)


def test_rl_metadata_sidecar_roundtrip(tmp_path):
    import json as _json
    from ml.rl_allocator import RLAllocator

    path = tmp_path / "rl_allocator"
    meta = {"trained_on": "backtest_returns", "strategy_names": ["x", "y"]}

    alloc = RLAllocator(2)
    alloc._model = type("M", (), {"save": lambda self, p: open(p + ".zip", "w").write("")})()
    alloc.save(path, metadata=meta)
    assert _json.loads(path.with_suffix(".meta.json").read_text()) == meta


# --------------------------------------------------------------------------- #
# Phase C — backtester correctness
# --------------------------------------------------------------------------- #

def _bt_frame(n=300, seed=3, trend=0.0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(trend, 0.01, n)))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open_time": idx, "open": open_, "high": high, "low": low,
        "close": close, "volume": rng.uniform(1e5, 1e6, n),
    }).set_index("open_time")


class AlwaysBuyStrategy:
    name = "always_buy"
    market = "spot"

    def generate_signals(self, store, regime, predictions):
        closes = store.get_closes("TESTUSDT", "1h", 1)
        price = closes[-1] if closes else 100.0
        return [Signal("TESTUSDT", "spot", "BUY", 0.0, price, 0.8, self.name)]


def test_backtester_exits_produce_no_phantom_positions():
    from backtester.engine import BacktestEngine

    data = {"TESTUSDT": _bt_frame()}
    engine = BacktestEngine(
        data=data, interval="1h", initial_equity=10_000,
        strategies=[AlwaysBuyStrategy()], lookback=60,
        max_open_positions=1, stop_loss_pct=0.01, take_profit_pct=0.01,
    )
    result = engine.run()
    trades = result.trades
    if len(trades):
        # Every closed trade must be LONG — the old exit path re-filled the
        # queued exit order as a phantom SHORT.
        assert set(trades["side"].unique()) == {"LONG"}
        # And stops/TPs now report an exit reason.
        assert "exit_reason" in trades.columns


def test_backtester_equity_accounting_consistent():
    """Flat-price series + no trading costs => equity stays ~initial."""
    from backtester.engine import BacktestEngine

    n = 200
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    flat = pd.DataFrame({
        "open_time": idx,
        "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
        "volume": 1e6,
    }).set_index("open_time")

    engine = BacktestEngine(
        data={"TESTUSDT": flat}, interval="1h", initial_equity=10_000,
        strategies=[AlwaysBuyStrategy()], lookback=60, max_open_positions=1,
    )
    result = engine.run()
    # With flat prices, equity should never fake-drop by position notional
    # (the old +pnl-only marking dropped the curve by full notional on entry).
    assert result.equity_curve.min() > 9_800


def test_simulated_store_history_df_advances():
    """
    Regression: SimulatedMarketDataStore.get_history_df returned a FROZEN
    frame (stale _df_cache) for the entire backtest, so strategies computing
    indicators from it saw static data and emitted zero signals historically.
    """
    from backtester.engine import SimulatedMarketDataStore

    df = _bt_frame(n=300)
    store = SimulatedMarketDataStore()
    store.load_symbol("TESTUSDT", "1h", df)

    store.set_time_index(100)
    close_at_100 = float(store.get_history_df("TESTUSDT", "1h")["close"].iloc[-1])
    store.set_time_index(200)
    close_at_200 = float(store.get_history_df("TESTUSDT", "1h")["close"].iloc[-1])

    raw = df["close"].astype(float).values
    assert close_at_100 == pytest.approx(raw[99])
    assert close_at_200 == pytest.approx(raw[199])
    assert close_at_100 != close_at_200


def test_fill_model_venue_fees():
    from backtester.fill_model import venue_fees

    assert venue_fees("alpaca") == (0.0015, 0.0025)
    assert venue_fees("binance", "spot") == (0.0010, 0.0010)
    assert venue_fees("binance", "futures") == (0.0002, 0.0004)


def test_limit_fill_requires_penetration_or_luck():
    from backtester.fill_model import FillModel, Order, OrderType

    fm = FillModel(seed=1)
    bar_touch = {"open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1e6, "atr": 1.0}
    bar_through = {"open": 101.0, "high": 102.0, "low": 99.0, "close": 101.5, "volume": 1e6, "atr": 1.0}

    def _mk():
        return Order(symbol="X", side="BUY", order_type=OrderType.LIMIT, quantity=1.0,
                     limit_price=100.0, strategy="t", bar_index=0)

    # Strict penetration always fills.
    assert fm._fill_limit(_mk(), bar_through, 1.0, 1e6, 1) is not None
    # Touch-only fills ~85% of the time; over 200 seeded tries, both outcomes occur.
    outcomes = {fm._fill_limit(_mk(), bar_touch, 1.0, 1e6, 1) is not None for _ in range(200)}
    assert outcomes == {True, False}


def test_lstm_scaler_fit_on_train_only():
    import inspect
    from ml import price_lstm

    src = inspect.getsource(price_lstm.PricePredictorTrainer.fit)
    assert "fit_transform(X)" not in src
    assert "self.scaler.fit(X[:raw_split])" in src


def test_xgb_quantiles_from_train_slice_only():
    import inspect
    from ml import xgb_alpha

    src = inspect.getsource(xgb_alpha)
    assert "y_train_ret.quantile" in src


def test_hmm_decode_causal_is_causal():
    """Appending future bars must not change already-decoded labels."""
    from ml.regime_hmm import RegimeHMM

    df = _bt_frame(n=400, seed=11)
    hmm = RegimeHMM(n_states=4)
    hmm.fit(df)

    labels_full = hmm.decode_causal(df, stride=24)
    labels_short = hmm.decode_causal(df.iloc[:300], stride=24)
    # The first 300 labels must be identical whether or not bars 300-400 exist.
    assert labels_full[:300] == labels_short


def test_optimizer_holdout_split():
    from backtester.optimizer import BacktestOptimizer

    data = {"TESTUSDT": _bt_frame(n=1000)}
    opt_data, holdout = BacktestOptimizer._split_holdout(data, 0.2, warmup_bars=100)
    assert len(opt_data["TESTUSDT"]) == 800
    assert len(holdout["TESTUSDT"]) == 300  # 200 holdout + 100 warmup
    # No optimization row overlaps the pure holdout region.
    assert opt_data["TESTUSDT"].index.max() <= holdout["TESTUSDT"].index[100]


# --------------------------------------------------------------------------- #
# Phase E — dashboard endpoints
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def dashboard_client():
    from fastapi.testclient import TestClient
    from dashboard.server import app

    return TestClient(app)


@pytest.mark.parametrize("path,required_keys", [
    ("/api/sleeves/deep_value", {"scan_available", "candidates_passing", "positions", "caps"}),
    ("/api/sleeves/cornwall", {"tickets", "count", "total_cost"}),
    ("/api/allocation/live", {"available"}),
    ("/api/crypto/book", {"positions", "strategy_activity", "meta"}),
])
def test_new_dashboard_endpoints(dashboard_client, path, required_keys):
    response = dashboard_client.get(path)
    assert response.status_code == 200
    body = response.json()
    missing = required_keys - set(body.keys())
    assert not missing, f"{path} missing keys: {missing}"
