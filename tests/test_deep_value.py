import json
import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from alpaca.trading.enums import AssetClass

from config import params
from core.ml import deep_value
from core.ml.deep_value import (
    DeepValueCandidate,
    ValueMetrics,
    build_candidate,
    compute_value_metrics,
    heuristic_score,
    load_scan_snapshot,
    passes_quality_filters,
    save_scan_snapshot,
)
from core.execution import deep_value_sleeve
from core.execution.deep_value_sleeve import deploy_deep_value_bets


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _net_net_fundamentals(**overrides):
    """A textbook net-net: NCAV/share = (200M - 80M) / 10M = $12, price $6."""
    base = {
        "symbol": "NETN",
        "price": 6.0,
        "shares_outstanding": 10e6,
        "current_assets": 200e6,
        "total_liabilities": 80e6,
        "cash": 120e6,
        "receivables": 40e6,
        "inventory": 20e6,
        "free_cash_flow": 5e6,
        "market_cap": 60e6,
        "avg_dollar_volume": 5e6,
        "sector": "Technology",
        "as_of": "2026-06-30",
    }
    base.update(overrides)
    return base


def _passing_candidate(symbol="NETN", score=0.8, price=6.0, ncav=12.0):
    return {
        "symbol": symbol,
        "price": price,
        "ncav_per_share": ncav,
        "liquidation_per_share": ncav * 0.7,
        "net_cash_per_share": 4.0,
        "burn_runway_quarters": -1.0,
        "price_to_ncav": price / ncav,
        "market_cap": 60e6,
        "avg_dollar_volume": 5e6,
        "sector": "Technology",
        "score": score,
        "model_probability": None,
        "failed_gates": [],
        "as_of": "2026-06-30",
    }


class FakeClient:
    def __init__(self):
        self.buys = []
        self.sells = []

    def market_buy(self, symbol, qty, order_label=""):
        self.buys.append((symbol, qty))

    def market_sell(self, symbol, qty, order_label=""):
        self.sells.append((symbol, qty))


def _fake_position(symbol, qty=100, current_price=6.0, entry_price=6.0, plpc=0.0):
    return SimpleNamespace(
        symbol=symbol,
        qty=qty,
        asset_class=AssetClass.US_EQUITY,
        current_price=current_price,
        avg_entry_price=entry_price,
        unrealized_plpc=plpc,
        market_value=qty * current_price,
    )


@pytest.fixture
def sleeve_env(monkeypatch, tmp_path):
    """Isolate the sleeve from disk registry, alerts, and the cash sweep."""
    registry = {}

    monkeypatch.setattr(deep_value_sleeve, "get_equity_overlay_metadata", lambda: dict(registry))
    monkeypatch.setattr(
        deep_value_sleeve, "register_equity_overlay", lambda symbol, meta: registry.__setitem__(symbol, meta)
    )
    monkeypatch.setattr(
        deep_value_sleeve, "remove_equity_overlay_metadata", lambda symbol: registry.pop(symbol, None)
    )
    monkeypatch.setattr(deep_value_sleeve, "release_cash_from_sweep", lambda *a, **k: None)
    monkeypatch.setattr(deep_value_sleeve, "send_alert", lambda *a, **k: None)
    return registry


# --------------------------------------------------------------------------- #
# Valuation math
# --------------------------------------------------------------------------- #

def test_ncav_math():
    metrics = compute_value_metrics(_net_net_fundamentals())
    assert metrics.ncav_per_share == pytest.approx(12.0)
    # cash 120 + recv 40*0.75 + inv 20*0.5 + other 20*0.25 - liab 80 = 85M / 10M
    assert metrics.liquidation_per_share == pytest.approx(8.5)
    assert metrics.net_cash_per_share == pytest.approx(4.0)
    assert math.isinf(metrics.burn_runway_quarters)  # FCF positive
    assert metrics.price_to_ncav == pytest.approx(0.5)


def test_burn_runway():
    metrics = compute_value_metrics(_net_net_fundamentals(free_cash_flow=-40e6))
    # cash 120M / (40M/4 per quarter) = 12 quarters
    assert metrics.burn_runway_quarters == pytest.approx(12.0)


def test_missing_required_fields_returns_none():
    assert compute_value_metrics(_net_net_fundamentals(current_assets=None)) is None
    assert compute_value_metrics(_net_net_fundamentals(shares_outstanding=0)) is None
    assert compute_value_metrics(_net_net_fundamentals(price=None)) is None


# --------------------------------------------------------------------------- #
# Quality filters
# --------------------------------------------------------------------------- #

def test_textbook_net_net_passes_all_gates():
    fundamentals = _net_net_fundamentals()
    metrics = compute_value_metrics(fundamentals)
    passed, failed = passes_quality_filters(metrics, fundamentals)
    assert passed and failed == []


@pytest.mark.parametrize(
    "overrides,expected_gate",
    [
        ({"price": 11.0}, "insufficient_margin_of_safety"),  # 11 >= 0.66*12
        ({"price": 0.80, "shares_outstanding": 75e6}, "penny_stock"),
        ({"market_cap": 10e6}, "market_cap_floor"),
        ({"avg_dollar_volume": 0.5e6}, "illiquid"),
        ({"sector": "Financial Services"}, "excluded_sector"),
        ({"free_cash_flow": -200e6}, "cash_burn_runway"),  # runway 2.4 quarters < 4
        ({"total_liabilities": 300e6}, "negative_ncav"),
        # NCAV/share $12,000 vs price $6: >95% discount = bad data or fraud
        ({"current_assets": 200e9, "total_liabilities": 80e9, "cash": 120e9, "receivables": 40e9, "inventory": 20e9, "market_cap": 60e6}, "implausible_discount"),
        # balance-sheet share count implies $600M cap vs $60M quoted: unit mismatch
        ({"shares_outstanding": 100e6, "current_assets": 2000e6, "total_liabilities": 800e6, "cash": 1200e6, "receivables": 400e6, "inventory": 200e6}, "share_count_mismatch"),
    ],
)
def test_each_gate_trips(overrides, expected_gate):
    fundamentals = _net_net_fundamentals(**overrides)
    metrics = compute_value_metrics(fundamentals)
    passed, failed = passes_quality_filters(metrics, fundamentals)
    assert not passed
    assert expected_gate in failed


def test_score_increases_with_discount():
    scores = []
    for price in (10.0, 7.0, 4.0):
        fundamentals = _net_net_fundamentals(price=price)
        scores.append(heuristic_score(compute_value_metrics(fundamentals), fundamentals))
    assert scores[0] < scores[1] < scores[2]


def test_build_candidate_zeroes_score_when_gated():
    candidate = build_candidate("TRAP", _net_net_fundamentals(sector="Financial Services"))
    assert candidate.failed_gates == ["excluded_sector"]
    assert candidate.score == 0.0


# --------------------------------------------------------------------------- #
# Snapshot persistence
# --------------------------------------------------------------------------- #

def test_snapshot_roundtrip(tmp_path):
    path = tmp_path / "scan.json"
    candidate = build_candidate("NETN", _net_net_fundamentals())
    save_scan_snapshot([candidate], path=path)
    rows = load_scan_snapshot(max_age_hours=1, path=path)
    assert rows and rows[0]["symbol"] == "NETN" and rows[0]["failed_gates"] == []


def test_snapshot_stale_returns_none(tmp_path):
    path = tmp_path / "scan.json"
    stale = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    path.write_text(json.dumps({"generated_at_utc": stale, "candidates": [{"symbol": "OLD"}]}))
    assert load_scan_snapshot(max_age_hours=20, path=path) is None


def test_snapshot_missing_or_corrupt_returns_none(tmp_path):
    assert load_scan_snapshot(max_age_hours=20, path=tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_scan_snapshot(max_age_hours=20, path=bad) is None


# --------------------------------------------------------------------------- #
# Execution sleeve
# --------------------------------------------------------------------------- #

def test_sleeve_respects_caps(monkeypatch, sleeve_env):
    candidates = [_passing_candidate(f"SYM{i}", score=0.9 - i * 0.01) for i in range(10)]
    monkeypatch.setattr(deep_value_sleeve, "load_scan_snapshot", lambda *a, **k: candidates)

    client = FakeClient()
    total_equity = 100_000.0
    deploy_deep_value_bets(client, total_equity, positions=[])

    assert len(client.buys) == params.DEEP_VALUE_MAX_POSITIONS
    per_name_cap = total_equity * params.DEEP_VALUE_MAX_SYMBOL_WEIGHT
    for symbol, qty in client.buys:
        assert qty * 6.0 <= per_name_cap + 6.0
    total_spend = sum(qty * 6.0 for _, qty in client.buys)
    assert total_spend <= total_equity * params.DEEP_VALUE_MAX_ALLOCATION + 1e-6
    assert set(sleeve_env) == {symbol for symbol, _ in client.buys}


def test_sleeve_skips_held_and_gated_and_low_score(monkeypatch, sleeve_env):
    sleeve_env["HELD1"] = {"mode": "deep_value", "ncav_per_share": 12.0, "entered_at_utc": datetime.now(timezone.utc).isoformat()}
    candidates = [
        _passing_candidate("HELD1"),
        {**_passing_candidate("GATED"), "failed_gates": ["illiquid"]},
        {**_passing_candidate("WEAK"), "score": 0.10},
        _passing_candidate("FRESH"),
    ]
    monkeypatch.setattr(deep_value_sleeve, "load_scan_snapshot", lambda *a, **k: candidates)

    client = FakeClient()
    positions = [_fake_position("HELD1")]
    deploy_deep_value_bets(client, 100_000.0, positions)

    bought = {symbol for symbol, _ in client.buys}
    assert bought == {"FRESH"}


def test_sleeve_exit_on_target_stop_and_timeout(monkeypatch, sleeve_env):
    now = datetime.now(timezone.utc)
    sleeve_env.update(
        {
            # price 11 >= 0.9 * NCAV 12 -> fair value target
            "TGT": {"mode": "deep_value", "ncav_per_share": 12.0, "entered_at_utc": now.isoformat()},
            # -35% unrealized -> hard stop
            "STP": {"mode": "deep_value", "ncav_per_share": 50.0, "entered_at_utc": now.isoformat()},
            # held 200 days -> time stop
            "OLD": {"mode": "deep_value", "ncav_per_share": 50.0, "entered_at_utc": (now - timedelta(days=200)).isoformat()},
            # healthy -> hold
            "OK": {"mode": "deep_value", "ncav_per_share": 50.0, "entered_at_utc": now.isoformat()},
        }
    )
    monkeypatch.setattr(deep_value_sleeve, "load_scan_snapshot", lambda *a, **k: [])

    client = FakeClient()
    positions = [
        _fake_position("TGT", current_price=11.0),
        _fake_position("STP", current_price=6.0, plpc=-0.35),
        _fake_position("OLD", current_price=6.0),
        _fake_position("OK", current_price=6.0),
    ]
    deploy_deep_value_bets(client, 100_000.0, positions)

    sold = {symbol for symbol, _ in client.sells}
    assert sold == {"TGT", "STP", "OLD"}
    assert "OK" in sleeve_env and all(symbol not in sleeve_env for symbol in sold)


def test_sleeve_thesis_break_exit(monkeypatch, sleeve_env):
    now = datetime.now(timezone.utc)
    sleeve_env["BRK"] = {"mode": "deep_value", "ncav_per_share": 12.0, "entered_at_utc": now.isoformat()}
    # Fresh scan re-priced BRK: NCAV/share now below current price -> floor gone
    fresh = [{**_passing_candidate("BRK"), "ncav_per_share": 4.0}]
    monkeypatch.setattr(deep_value_sleeve, "load_scan_snapshot", lambda *a, **k: fresh)

    client = FakeClient()
    deploy_deep_value_bets(client, 100_000.0, [_fake_position("BRK", current_price=6.0)])
    assert client.sells and client.sells[0][0] == "BRK"


def test_sleeve_no_entries_when_scan_stale(monkeypatch, sleeve_env):
    monkeypatch.setattr(deep_value_sleeve, "load_scan_snapshot", lambda *a, **k: None)
    client = FakeClient()
    deploy_deep_value_bets(client, 100_000.0, positions=[])
    assert client.buys == []


def test_sleeve_auto_execute_off_blocks_entries(monkeypatch, sleeve_env):
    monkeypatch.setattr(deep_value_sleeve, "load_scan_snapshot", lambda *a, **k: [_passing_candidate("NETN")])
    monkeypatch.setattr(deep_value_sleeve, "DEEP_VALUE_AUTO_EXECUTE", False)
    client = FakeClient()
    deploy_deep_value_bets(client, 100_000.0, positions=[])
    assert client.buys == []


# --------------------------------------------------------------------------- #
# Equity overlay must leave deep-value positions alone
# --------------------------------------------------------------------------- #

def test_overlay_exit_sweep_skips_deep_value(monkeypatch):
    from core.execution import equity_overlay

    sells = []
    client = SimpleNamespace(market_sell=lambda symbol, qty, order_label="": sells.append(symbol))

    monkeypatch.setattr(
        equity_overlay,
        "get_equity_overlay_metadata",
        lambda: {"DEEP": {"mode": "deep_value", "ncav_per_share": 12.0}},
    )
    monkeypatch.setattr(equity_overlay, "remove_equity_overlay_metadata", lambda symbol: None)
    monkeypatch.setattr(equity_overlay, "_market_session_open_now", lambda: True)
    monkeypatch.setattr(equity_overlay, "_is_defensive_mode", lambda *a, **k: True)  # would force-exit directional
    monkeypatch.setattr(equity_overlay, "_build_directional_targets", lambda **k: {})
    monkeypatch.setattr(equity_overlay, "_build_delta_hedge_targets", lambda **k: {})

    positions = [_fake_position("DEEP")]
    equity_overlay.rebalance_equity_overlay(
        client=client,
        positions=positions,
        movement_signals=[],
        flow_map=None,
        total_equity=100_000.0,
        buying_power=10_000.0,
        deployment_scale=1.0,
        current_vix=20.0,
        movement_bias="neutral",
        runtime_calibration=None,
    )
    assert sells == []
