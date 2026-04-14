import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from alpaca.trading.enums import AssetClass

from core.broker_client import BrokerClient
from core.manager import manage_open_spreads
from core.order_monitor import ExecutionPricingSnapshot, MonitoredOrderLeg, monitor_multileg_order
from core.runtime_env import apply_accelerator_policy, host_has_nvidia_device
from core.portfolio_risk import PortfolioRiskBlockedError, PortfolioRiskEngine, PortfolioRiskSnapshot, PortfolioTradeLeg
from core.quant_models import (
    OptionLegModel,
    analyze_long_option_tail,
    binomial_option_price,
    monte_carlo_multileg_risk,
)
from core.torch_device import resolve_torch_runtime


@dataclass
class _FakeOrder:
    id: str
    status: str
    limit_price: float
    filled_qty: float = 0.0
    qty: float = 1.0


class _SequencedTradeClient:
    def __init__(self, orders):
        self._orders = list(orders)
        self.replacements = []
        self.cancellations = []

    def get_order_by_id(self, _order_id):
        if len(self._orders) > 1:
            return self._orders.pop(0)
        return self._orders[0]

    def replace_order_by_id(self, order_id, request):
        self.replacements.append((order_id, float(request.limit_price)))
        return _FakeOrder(id=order_id, status="accepted", limit_price=float(request.limit_price))

    def cancel_order_by_id(self, order_id):
        self.cancellations.append(order_id)


class _MonitorBrokerClient:
    def __init__(self, orders):
        self.trade_client = _SequencedTradeClient(orders)


class _ManagerTradeClient:
    def get_orders(self, filter=None):
        return []

    def get_all_positions(self):
        return [
            SimpleNamespace(
                symbol="SGOV",
                asset_class=AssetClass.US_EQUITY,
                qty="12",
                unrealized_plpc="0.004",
                unrealized_pl="1.25",
            )
        ]

    def get_account(self):
        return SimpleNamespace(
            equity="10050",
            last_equity="10000",
            buying_power="8000",
            cash="1200",
            portfolio_value="10050",
        )


class _ManagerBrokerClient:
    def __init__(self):
        self.trade_client = _ManagerTradeClient()


def _build_close_series(start_price, pattern, length=120):
    values = [float(start_price)]
    for idx in range(length):
        move = float(pattern[idx % len(pattern)])
        values.append(values[-1] * (1.0 + move))
    return values


class _RiskTradeClient:
    def __init__(self, positions, account):
        self._positions = list(positions)
        self._account = account
        self.submitted_orders = []

    def get_all_positions(self):
        return list(self._positions)

    def get_account(self):
        return self._account

    def submit_order(self, request):
        self.submitted_orders.append(request)
        return request


class _RiskBrokerHarness:
    def __init__(self, positions, account, prices):
        self.trade_client = _RiskTradeClient(positions, account)
        self.prices = dict(prices)

    def get_stock_latest_trade(self, symbol):
        return {symbol: SimpleNamespace(price=float(self.prices[symbol]))}


class RuntimeHardeningTests(unittest.TestCase):
    def test_cpu_runtime_message_is_stable(self):
        with patch("core.torch_device.torch.cuda.is_available", return_value=False):
            runtime = resolve_torch_runtime()

        self.assertEqual(runtime.device.type, "cpu")
        self.assertEqual(runtime.message, "CPU detected and using.")

    def test_accelerator_policy_forces_cpu_without_gpu(self):
        with patch("core.runtime_env.host_has_nvidia_device", return_value=False):
            env, message = apply_accelerator_policy({})

        self.assertEqual(env["OPTIONS_STACK_FORCE_CPU"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(message, "CPU detected and using.")

    def test_host_gpu_detection_ignores_driver_nodes_without_real_gpu(self):
        with patch("core.runtime_env._nvidia_proc_gpu_present", return_value=False), patch(
            "core.runtime_env._nvidia_pci_gpu_present",
            return_value=False,
        ):
            self.assertFalse(host_has_nvidia_device())

    def test_quant_risk_models_return_var_metrics(self):
        risk = monte_carlo_multileg_risk(
            spot=100.0,
            legs=[
                OptionLegModel(option_type="put", side=-1, strike=100.0, years_to_expiry=30 / 365.0, volatility=0.22),
                OptionLegModel(option_type="put", side=1, strike=95.0, years_to_expiry=30 / 365.0, volatility=0.24),
            ],
            risk_free_rate=0.04,
            n_simulations=400,
            seed=7,
        )

        self.assertGreater(abs(risk.fair_value), 0.0)
        self.assertGreaterEqual(risk.var_95, 0.0)
        self.assertGreaterEqual(risk.cvar_95, risk.var_95)

    def test_long_option_tail_analytics_capture_convexity(self):
        binomial_price = binomial_option_price(
            flag="c",
            spot=100.0,
            strike=120.0,
            years_to_expiry=120 / 365.0,
            risk_free_rate=0.04,
            volatility=0.35,
            steps=80,
        )
        analytics = analyze_long_option_tail(
            flag="c",
            spot=100.0,
            strike=120.0,
            years_to_expiry=120 / 365.0,
            risk_free_rate=0.04,
            volatility=0.35,
            premium=1.25,
            n_simulations=600,
            seed=11,
            fat_tail_multiple=4.0,
        )

        self.assertGreater(binomial_price, 0.0)
        self.assertGreater(analytics.p99_payoff, analytics.p95_payoff)
        self.assertGreaterEqual(analytics.tail_payoff_multiple, 0.0)
        self.assertGreaterEqual(analytics.fat_tail_probability, 0.0)

    def test_manage_open_spreads_ignores_cash_sweep_equity(self):
        holding_status = manage_open_spreads(_ManagerBrokerClient())

        self.assertEqual(len(holding_status), 1)
        self.assertIn("SGOV [SHARES]", holding_status[0])

    @patch("core.order_monitor.send_alert")
    def test_order_monitor_reprices_then_marks_fill(self, _mock_alert):
        broker = _MonitorBrokerClient(
            [
                _FakeOrder(id="ord-1", status="new", limit_price=-1.10, filled_qty=0.0, qty=1.0),
                _FakeOrder(id="ord-1", status="new", limit_price=-1.10, filled_qty=0.0, qty=1.0),
                _FakeOrder(id="ord-1", status="filled", limit_price=-1.08, filled_qty=1.0, qty=1.0),
            ]
        )

        snapshot = ExecutionPricingSnapshot(
            natural_price=-1.00,
            fair_price=-1.12,
            pricing_confidence=0.84,
            staleness_pct=0.01,
            underlying_price=505.0,
            mc_expected_price=-1.09,
            mc_var_95=0.06,
            mc_cvar_95=0.08,
        )

        result = monitor_multileg_order(
            client=broker,
            order=_FakeOrder(id="ord-1", status="new", limit_price=-1.10, filled_qty=0.0, qty=1.0),
            order_label="Credit spread SPY put wing",
            legs=[
                MonitoredOrderLeg(symbol="SPY260619P00500000", side="sell"),
                MonitoredOrderLeg(symbol="SPY260619P00495000", side="buy"),
            ],
            is_credit=True,
            poll_seconds=0.0,
            timeout_seconds=0.25,
            max_reprices=1,
            snapshot_builder=lambda **_: snapshot,
            limit_reprice_func=lambda *_args: -1.08,
            sleep_fn=lambda *_args: None,
        )

        self.assertTrue(result.filled)
        self.assertEqual(result.reprices, 1)
        self.assertEqual(broker.trade_client.replacements, [("ord-1", -1.08)])

    @patch("core.portfolio_risk.PORTFOLIO_RISK_MC_PATHS", 300)
    def test_portfolio_risk_snapshot_detects_concentration(self):
        positions = [
            SimpleNamespace(symbol="TSLA", asset_class=AssetClass.US_EQUITY, qty="150"),
        ]
        account = SimpleNamespace(equity="10000", last_equity="10000", portfolio_value="10000")
        history_map = {
            "TSLA": _build_close_series(100.0, [0.035, -0.030, 0.024, -0.026, 0.018, -0.020]),
        }
        engine = PortfolioRiskEngine(
            _RiskBrokerHarness(positions, account, {"TSLA": history_map["TSLA"][-1]}),
            price_history_provider=lambda symbol: history_map[symbol],
        )

        snapshot = engine.build_snapshot(positions=positions, account=account)

        self.assertIn("single_underlying_concentration", snapshot.breaches)
        self.assertGreater(snapshot.var_95, 0.0)
        self.assertGreater(snapshot.correlation_concentration, 0.0)

    @patch("core.portfolio_risk.PORTFOLIO_RISK_MC_PATHS", 300)
    def test_portfolio_risk_engine_blocks_non_reducing_trade_when_breached(self):
        positions = [
            SimpleNamespace(symbol="TSLA", asset_class=AssetClass.US_EQUITY, qty="150"),
        ]
        account = SimpleNamespace(equity="10000", last_equity="10000", portfolio_value="10000")
        history_map = {
            "TSLA": _build_close_series(100.0, [0.030, -0.028, 0.026, -0.027, 0.019, -0.021]),
        }
        engine = PortfolioRiskEngine(
            _RiskBrokerHarness(positions, account, {"TSLA": history_map["TSLA"][-1]}),
            price_history_provider=lambda symbol: history_map[symbol],
        )

        decision = engine.assess_trade(
            "Add TSLA shares",
            [PortfolioTradeLeg(symbol="TSLA", side="buy", quantity=25)],
        )

        self.assertFalse(decision.allowed)
        self.assertFalse(decision.reduce_only)
        self.assertTrue(
            "risk" in decision.reason.lower() or "kill switch" in decision.reason.lower(),
        )

    @patch("core.portfolio_risk.PORTFOLIO_RISK_MC_PATHS", 300)
    def test_portfolio_risk_engine_allows_protective_put_when_it_reduces_risk(self):
        positions = [
            SimpleNamespace(symbol="SPY", asset_class=AssetClass.US_EQUITY, qty="100"),
        ]
        account = SimpleNamespace(equity="60000", last_equity="60000", portfolio_value="60000")
        history_map = {
            "SPY": _build_close_series(500.0, [0.022, -0.026, 0.018, -0.020, 0.015, -0.016]),
        }
        engine = PortfolioRiskEngine(
            _RiskBrokerHarness(positions, account, {"SPY": history_map["SPY"][-1]}),
            price_history_provider=lambda symbol: history_map[symbol],
        )

        decision = engine.assess_trade(
            "Buy SPY protective put",
            [PortfolioTradeLeg(symbol="SPY260619P00500000", side="buy", quantity=1)],
        )

        self.assertTrue(decision.allowed)
        self.assertTrue(decision.risk_reducing)
        self.assertLess(decision.projected.cvar_95, decision.current.cvar_95)

    def test_broker_market_buy_raises_when_portfolio_gate_blocks_trade(self):
        blocked_snapshot = PortfolioRiskSnapshot(
            portfolio_equity=10000.0,
            modeled_portfolio_value=9500.0,
            expected_value=9400.0,
            var_95=900.0,
            cvar_95=1200.0,
            value_volatility=400.0,
            worst_case=8000.0,
            best_case=10400.0,
            stress_losses={"gap_crash_15": 1400.0},
            worst_stress_loss=1400.0,
            max_underlying_weight=0.75,
            correlation_concentration=0.80,
            gross_exposure=18000.0,
            gross_exposure_pct_equity=1.80,
            net_delta_exposure=15000.0,
            top_underlyings=[{"symbol": "TSLA", "weight": 0.75, "delta_notional": 15000.0}],
        )
        blocked_snapshot.breaches = ["portfolio_cvar"]

        broker = BrokerClient.__new__(BrokerClient)
        broker.trade_client = _RiskTradeClient([], SimpleNamespace(equity="10000", last_equity="10000", portfolio_value="10000"))
        broker.portfolio_risk_engine = SimpleNamespace(
            assess_trade=lambda *_args, **_kwargs: SimpleNamespace(
                allowed=False,
                reason="Projected breach: portfolio_cvar.",
                order_label="Buy TSLA",
                reduce_only=False,
                risk_reducing=False,
                current=blocked_snapshot,
                projected=blocked_snapshot,
            ),
            record_post_trade_snapshot=lambda **_kwargs: None,
        )

        with self.assertRaises(PortfolioRiskBlockedError):
            broker.market_buy("TSLA", qty=1)


if __name__ == "__main__":
    unittest.main()
