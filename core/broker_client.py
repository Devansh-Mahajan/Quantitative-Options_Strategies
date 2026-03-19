from config.params import EXPIRATION_MIN, EXPIRATION_MAX
from .user_agent_mixin import UserAgentMixin 
from alpaca.trading.client import TradingClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient, StockLatestTradeRequest
from alpaca.data.requests import OptionSnapshotRequest
# ADDED: LimitOrderRequest to securely price the spreads
from alpaca.trading.requests import GetOptionContractsRequest, MarketOrderRequest, LimitOrderRequest, OptionLegRequest
from alpaca.trading.enums import ContractType, AssetStatus, AssetClass, OrderClass, OrderSide, TimeInForce
from datetime import timedelta
from zoneinfo import ZoneInfo
import datetime

class TradingClientSigned(UserAgentMixin, TradingClient):
    pass

class StockHistoricalDataClientSigned(UserAgentMixin, StockHistoricalDataClient):
    pass

class OptionHistoricalDataClientSigned(UserAgentMixin, OptionHistoricalDataClient):
    pass

class BrokerClient:
    def __init__(self, api_key, secret_key, paper=True):
        self.trade_client = TradingClientSigned(api_key=api_key, secret_key=secret_key, paper=paper)
        self.stock_client = StockHistoricalDataClientSigned(api_key=api_key, secret_key=secret_key)
        self.option_client = OptionHistoricalDataClientSigned(api_key=api_key, secret_key=secret_key)

    def get_positions(self):
        return self.trade_client.get_all_positions()

    def market_sell(self, symbol, qty=1):
        req = MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY
        )
        self.trade_client.submit_order(req)

    def market_buy(self, symbol, qty=1):
        req = MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY
        )
        self.trade_client.submit_order(req)

    def execute_credit_spread(self, short_symbol, long_symbol, limit_price, qty=1):
        """
        Executes a multi-leg options spread using a strict Limit Order.
        Per Alpaca documentation, limit_price MUST be negative to indicate a net credit.
        """
        req = LimitOrderRequest(
            qty=qty, 
            limit_price=limit_price, 
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            legs=[
                OptionLegRequest(
                    symbol=short_symbol,
                    ratio_qty=1,
                    side=OrderSide.SELL
                ),
                OptionLegRequest(
                    symbol=long_symbol,
                    ratio_qty=1,
                    side=OrderSide.BUY
                )
            ]
        )
        self.trade_client.submit_order(req)

    def get_option_snapshot(self, symbol):
        if isinstance(symbol, str):
            req = OptionSnapshotRequest(symbol_or_symbols=symbol)
            return self.option_client.get_option_snapshot(req)

        elif isinstance(symbol, list):
            all_results = {}
            for i in range(0, len(symbol), 100):
                batch = symbol[i:i+100]
                req = OptionSnapshotRequest(symbol_or_symbols=batch)
                result = self.option_client.get_option_snapshot(req)
                all_results.update(result)
            
            return all_results
        else:
            raise ValueError("Input must be a string or list of strings representing symbols.")

    def get_stock_latest_trade(self, symbol):
        req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        return self.stock_client.get_stock_latest_trade(req)

    def get_options_contracts(self, symbols, contract_type, min_days=1, max_days=60, exact_date=None):
        """Fetches options contracts filtered exactly by our target dates to save API limits."""
        from alpaca.trading.requests import GetOptionContractsRequest
        from datetime import datetime, timezone, timedelta
        import logging 
        
        logger = logging.getLogger(f"broker.{__name__}")
        all_contracts = []
        
        # --- THE FIX: Smart API Date Filtering ---
        if exact_date:
            # IV Crush & Vega: We know the exact date we want. Target it natively.
            target_min = datetime.strptime(exact_date[:10], "%Y-%m-%d").date()
            target_max = target_min
        else:
            # Theta & Cornwall: Calculate the exact DTE window.
            target_min = datetime.now(timezone.utc).date() + timedelta(days=min_days)
            target_max = datetime.now(timezone.utc).date() + timedelta(days=max_days)
        
        for sym in symbols:
            try:
                req = GetOptionContractsRequest(
                    underlying_symbols=[sym], 
                    status="active",
                    expiration_date_gte=target_min,
                    expiration_date_lte=target_max,
                    type=contract_type,
                    limit=100 # Safe at 100 because the window is surgically precise!
                )
                response = self.trade_client.get_option_contracts(req)
                if response and response.option_contracts:
                    all_contracts.extend(response.option_contracts)
            except Exception as e:
                logger.debug(f"⚠️ Alpaca rejected symbol {sym}")
                continue
                
        return all_contracts
    
    def liquidate_all_positions(self):
        positions = self.get_positions()
        to_liquidate = []
        for p in positions:
            if p.asset_class == AssetClass.US_OPTION:
                self.trade_client.close_position(p.symbol)
            else:
                to_liquidate.append(p)
        for p in to_liquidate:
            self.trade_client.close_position(p.symbol)
    def execute_debit_spread(self, call_symbol, put_symbol, limit_price, qty=1):
        """
        Executes a multi-leg debit spread or straddle.
        Unlike credit spreads, both legs are BUY orders, and the limit_price MUST be positive.
        """
        req = LimitOrderRequest(
            qty=qty, 
            limit_price=limit_price, 
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            legs=[
                OptionLegRequest(
                    symbol=call_symbol,
                    ratio_qty=1,
                    side=OrderSide.BUY  # We are BUYING the Call
                ),
                OptionLegRequest(
                    symbol=put_symbol,
                    ratio_qty=1,
                    side=OrderSide.BUY  # We are ALSO BUYING the Put
                )
            ]
        )
        self.trade_client.submit_order(req)