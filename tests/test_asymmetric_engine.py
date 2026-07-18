"""Tests for the rebuilt Asymmetric Bets Engine + vol-managed momentum."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.execution import asymmetric_engine as ae
from core.execution.asymmetric_engine import (
    AsymmetricSetup,
    breakout_r_setups,
    capitulation_setups,
    manage_asym_positions,
    run_asymmetric_engine,
)


def _daily_frame(closes, highs=None, lows=None):
    n = len(closes)
    c = np.asarray(closes, dtype=float)
    idx = pd.date_range("2026-01-01", periods=n, freq="1D")
    return pd.DataFrame({
        "open": c, "high": highs if highs is not None else c * 1.01,
        "low": lows if lows is not None else c * 0.99,
        "close": c, "volume": np.full(n, 1e6),
    }, index=idx)


class FakeClient:
    def __init__(self):
        self.buys, self.sells = [], []

    def market_buy(self, symbol, qty, order_label=""):
        self.buys.append((symbol, qty))

    def market_sell(self, symbol, qty, order_label=""):
        self.sells.append((symbol, qty))


@pytest.fixture
def engine_env(monkeypatch):
    registry = {}
    monkeypatch.setattr(ae, "get_equity_overlay_metadata", lambda: dict(registry))
    monkeypatch.setattr(ae, "register_equity_overlay", lambda s, m: registry.__setitem__(s, m))
    monkeypatch.setattr(ae, "remove_equity_overlay_metadata", lambda s: registry.pop(s, None))
    monkeypatch.setattr(ae, "release_cash_from_sweep", lambda *a, **k: None)
    monkeypatch.setattr(ae, "send_alert", lambda *a, **k: None)
    monkeypatch.setattr(ae, "record_trade_decision", lambda **k: None)
    monkeypatch.setattr(ae, "load_scan_snapshot", lambda *a, **k: [])
    monkeypatch.setattr(ae, "_write_snapshot", lambda *a, **k: None)
    return registry


# --------------------------------------------------------------------------- #
# The engine's core contract
# --------------------------------------------------------------------------- #

def test_asymmetry_ratio_math():
    s = AsymmetricSetup("capitulation", "X", "equity", 100.0, 5.0, 25.0, 0.6)
    assert s.asymmetry_ratio == 5.0
    zero = AsymmetricSetup("capitulation", "X", "equity", 100.0, 0.0, 25.0, 0.6)
    assert zero.asymmetry_ratio == 0.0


def test_engine_rejects_ratio_below_gate(engine_env, monkeypatch):
    setup = AsymmetricSetup("capitulation", "LOWR", "equity", 100.0, 10.0, 20.0, 0.8)  # 2x < 4x gate
    monkeypatch.setattr(ae, "capitulation_setups", lambda frames: [setup])
    monkeypatch.setattr(ae, "breakout_r_setups", lambda frames: [])
    client = FakeClient()
    run_asymmetric_engine(client, 100_000.0, [], daily_frames={})
    assert client.buys == []


def test_engine_takes_gated_setup_with_portfolio_sizing(engine_env, monkeypatch):
    # 5x ratio setup: entry 100, defined loss 4/unit, target gain 20/unit.
    setup = AsymmetricSetup("capitulation", "GOOD", "equity", 100.0, 4.0, 20.0, 0.8,
                            stop_price=96.0, target_price=120.0)
    monkeypatch.setattr(ae, "capitulation_setups", lambda frames: [setup])
    monkeypatch.setattr(ae, "breakout_r_setups", lambda frames: [])
    client = FakeClient()
    equity = 100_000.0
    run_asymmetric_engine(client, equity, [], daily_frames={})

    assert len(client.buys) == 1
    symbol, qty = client.buys[0]
    assert symbol == "GOOD"
    # Sizing contract: worst case ≈ ASYM_PER_BET_RISK_PCT of equity.
    from config.params import ASYM_PER_BET_RISK_PCT
    worst_case = qty * setup.max_loss_per_unit
    assert worst_case <= equity * ASYM_PER_BET_RISK_PCT + setup.max_loss_per_unit
    assert worst_case >= equity * ASYM_PER_BET_RISK_PCT * 0.8   # sizable, not token
    # Registered with its stop/target for management.
    assert engine_env["GOOD"]["mode"] == "asym_r"
    assert engine_env["GOOD"]["stop_price"] == 96.0


def test_engine_respects_total_risk_budget(engine_env, monkeypatch):
    setups = [
        AsymmetricSetup("capitulation", f"S{i}", "equity", 100.0, 4.0, 20.0, 0.8,
                        stop_price=96.0, target_price=120.0)
        for i in range(10)
    ]
    monkeypatch.setattr(ae, "capitulation_setups", lambda frames: setups)
    monkeypatch.setattr(ae, "breakout_r_setups", lambda frames: [])
    client = FakeClient()
    equity = 100_000.0
    run_asymmetric_engine(client, equity, [], daily_frames={})

    from config.params import ASYM_PER_BET_RISK_PCT, ASYM_TOTAL_RISK_PCT
    total_risk = sum(q * 4.0 for _, q in client.buys)
    assert total_risk <= equity * ASYM_TOTAL_RISK_PCT + 4.0
    # ~4 bets of 1.5% risk each fit in the 6% budget.
    assert 3 <= len(client.buys) <= int(ASYM_TOTAL_RISK_PCT / ASYM_PER_BET_RISK_PCT) + 1


def test_manage_exits_on_stop_target_and_time(engine_env):
    now = datetime.now(timezone.utc)
    engine_env.update({
        "STP": {"mode": "asym_r", "stop_price": 96.0, "target_price": 120.0,
                "entered_at_utc": now.isoformat(), "source": "capitulation"},
        "TGT": {"mode": "asym_r", "stop_price": 50.0, "target_price": 110.0,
                "entered_at_utc": now.isoformat(), "source": "breakout_r"},
        "OLD": {"mode": "asym_r", "stop_price": 50.0, "target_price": 500.0,
                "entered_at_utc": (now - timedelta(days=60)).isoformat(), "source": "deep_value"},
        "OK":  {"mode": "asym_r", "stop_price": 50.0, "target_price": 500.0,
                "entered_at_utc": now.isoformat(), "source": "capitulation"},
    })
    positions = [
        SimpleNamespace(symbol="STP", qty=10, current_price=95.0),   # below stop
        SimpleNamespace(symbol="TGT", qty=10, current_price=115.0),  # above target
        SimpleNamespace(symbol="OLD", qty=10, current_price=100.0),  # time stop (45d)
        SimpleNamespace(symbol="OK", qty=10, current_price=100.0),   # healthy
    ]
    client = FakeClient()
    manage_asym_positions(client, positions)
    assert {s for s, _ in client.sells} == {"STP", "TGT", "OLD"}
    assert "OK" in engine_env


# --------------------------------------------------------------------------- #
# Setup sources on synthetic data
# --------------------------------------------------------------------------- #

def test_capitulation_source_finds_crash_in_uptrend():
    daily = [100 + 0.8 * i for i in range(120)]
    peak = daily[-1]
    daily += [peak * 0.96, peak * 0.92, peak * 0.88]
    frames = {"CRASH": _daily_frame(daily)}
    setups = capitulation_setups(frames)
    assert len(setups) == 1
    s = setups[0]
    assert s.stop_price < s.entry_price < s.target_price
    assert s.asymmetry_ratio > 1.0


def test_breakout_source_measured_move():
    # 80 days ranging 95-105, then a pop to 110.
    rng = np.random.default_rng(3)
    daily = list(100 + rng.uniform(-5, 5, 80))
    daily += [108.0, 110.0]
    frames = {"BRK": _daily_frame(daily)}
    setups = breakout_r_setups(frames)
    assert len(setups) == 1
    s = setups[0]
    assert s.target_price > s.entry_price > s.stop_price


def test_sources_silent_on_flat_data():
    frames = {"FLAT": _daily_frame([100.0] * 120)}
    assert capitulation_setups(frames) == []
    assert breakout_r_setups(frames) == []


# --------------------------------------------------------------------------- #
# Vol-managed momentum
# --------------------------------------------------------------------------- #

class _HourStore:
    def __init__(self, closes, symbol="BTCUSDT"):
        self._closes = list(map(float, closes))
        self._symbol = symbol
        self.candles = {(symbol, "1h"): object()}

    def get_closes(self, symbol, interval, n):
        return self._closes[-n:] if symbol == self._symbol else []


def test_vmm_buys_calm_uptrend(monkeypatch):
    from strategies.vol_managed_momentum import VolManagedMomentumStrategy

    rng = np.random.default_rng(5)
    hours = 35 * 24
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.004, hours)))  # calm uptrend
    strat = VolManagedMomentumStrategy(eval_every=1)
    monkeypatch.setattr(type(strat), "symbols", property(lambda self: ["BTCUSDT"]))
    sigs = strat.generate_signals(_HourStore(closes), "bull", {})
    assert len(sigs) == 1 and sigs[0].side == "BUY"
    assert 0.45 <= sigs[0].confidence <= 0.88


def test_vmm_exits_on_trend_flip(monkeypatch):
    from strategies.vol_managed_momentum import VolManagedMomentumStrategy

    rng = np.random.default_rng(6)
    up = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.004, 35 * 24)))
    strat = VolManagedMomentumStrategy(eval_every=1)
    monkeypatch.setattr(type(strat), "symbols", property(lambda self: ["BTCUSDT"]))
    assert strat.generate_signals(_HourStore(up), "bull", {})[0].side == "BUY"

    # Trend collapses: 12 days of steady decline.
    down = np.concatenate([up, up[-1] * np.exp(np.cumsum(np.full(12 * 24, -0.001)))])
    sigs = strat.generate_signals(_HourStore(down), "bear", {})
    assert len(sigs) == 1 and sigs[0].side == "SELL"
    assert sigs[0].meta["mode"] == "vmm_exit"


def test_vmm_silent_on_downtrend(monkeypatch):
    from strategies.vol_managed_momentum import VolManagedMomentumStrategy

    rng = np.random.default_rng(7)
    closes = 100 * np.exp(np.cumsum(rng.normal(-0.0004, 0.004, 35 * 24)))
    strat = VolManagedMomentumStrategy(eval_every=1)
    monkeypatch.setattr(type(strat), "symbols", property(lambda self: ["BTCUSDT"]))
    assert strat.generate_signals(_HourStore(closes), "bear", {}) == []
