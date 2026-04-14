import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from config.params import OPTION_DELAY_MIN_PRICING_CONFIDENCE
from core.delay_aware_options import UnderlyingQuoteContext, reprice_contract
from core.strategy import filter_options, score_options
from models.contract import Contract


@dataclass
class _Quote:
    bid_price: float
    ask_price: float
    timestamp: datetime


@dataclass
class _Trade:
    price: float


@dataclass
class _Greeks:
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None


@dataclass
class _Snapshot:
    latest_quote: _Quote
    latest_trade: _Trade
    greeks: _Greeks
    implied_volatility: float | None = None


class DelayAwareOptionsTests(unittest.TestCase):
    def test_reprice_contract_marks_put_richer_after_underlying_drop(self):
        contract = Contract(
            underlying="SPY",
            symbol="SPY260619P00500000",
            contract_type="put",
            dte=66,
            strike=500.0,
            bid_price=3.20,
            ask_price=3.40,
            last_price=3.30,
            oi=500,
        )
        snapshot = _Snapshot(
            latest_quote=_Quote(3.20, 3.40, datetime.now(timezone.utc) - timedelta(minutes=16)),
            latest_trade=_Trade(3.30),
            greeks=_Greeks(delta=-0.23),
            implied_volatility=0.24,
        )
        stock_context = UnderlyingQuoteContext(
            symbol="SPY",
            current_price=490.0,
            delayed_price=500.0,
            current_timestamp=datetime.now(timezone.utc),
            delayed_timestamp=datetime.now(timezone.utc) - timedelta(minutes=15),
        )

        repriced = reprice_contract(contract, snapshot=snapshot, stock_context=stock_context)

        self.assertGreater(repriced.fair_bid_price, contract.bid_price)
        self.assertGreater(repriced.fair_ask_price, contract.ask_price)
        self.assertGreater(repriced.pricing_confidence, 0.0)
        self.assertGreater(repriced.staleness_pct, 0.0)
        self.assertAlmostEqual(repriced.implied_volatility, 0.24, places=6)

    @patch("core.strategy.get_dynamic_yield", return_value=0.01)
    def test_filter_options_rejects_contracts_with_low_delay_pricing_confidence(self, _mock_yield):
        contract = Contract(
            underlying="QQQ",
            symbol="QQQ260619P00420000",
            contract_type="put",
            dte=45,
            strike=420.0,
            delta=-0.21,
            bid_price=2.0,
            ask_price=2.2,
            fair_bid_price=1.9,
            fair_ask_price=2.1,
            oi=400,
            pricing_confidence=OPTION_DELAY_MIN_PRICING_CONFIDENCE - 0.05,
            staleness_pct=0.01,
        )

        filtered = filter_options([contract])
        self.assertEqual(filtered, [])

    @patch("core.strategy.get_dynamic_yield", return_value=0.01)
    def test_score_options_penalizes_stale_low_confidence_contracts(self, _mock_yield):
        high_quality = Contract(
            underlying="IWM",
            symbol="IWM260619P00210000",
            contract_type="put",
            dte=40,
            strike=210.0,
            delta=-0.18,
            bid_price=1.4,
            ask_price=1.5,
            fair_bid_price=1.38,
            fair_ask_price=1.48,
            oi=500,
            pricing_confidence=0.92,
            staleness_pct=0.004,
        )
        low_quality = Contract(
            underlying="IWM",
            symbol="IWM260619P00205000",
            contract_type="put",
            dte=40,
            strike=205.0,
            delta=-0.18,
            bid_price=1.4,
            ask_price=1.5,
            fair_bid_price=1.38,
            fair_ask_price=1.48,
            oi=500,
            pricing_confidence=0.35,
            staleness_pct=0.03,
        )

        scores = score_options([high_quality, low_quality])
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
