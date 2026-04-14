import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from alpaca.trading.enums import AssetClass, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from core.delay_aware_options import (
    build_delay_adjusted_contracts,
    effective_ask_price,
    effective_bid_price,
)
from models.contract import Contract
from .utils import parse_option_symbol
from config.params import PROFIT_TARGET, STOP_LOSS
from core.state_manager import get_equity_overlay_metadata, get_straddle_metadata, remove_straddle_metadata 
from config.params import PROFIT_TARGET, STOP_LOSS, SWEEP_TICKER, TARGET_CASH_BUFFER, MAX_RISK_BASE
from core.notifications import send_alert
logger = logging.getLogger(f"strategy.{__name__}")


@dataclass(frozen=True)
class CreditSpreadExitPlan:
    take_profit: float
    stop_loss: float
    time_stop_dte: int
    emergency_exit_risk: float


@dataclass(frozen=True)
class LongOptionExitPlan:
    take_profit: float
    stop_loss: float
    time_stop_dte: int


def build_credit_spread_exit_plan(days_to_expiry: int, pop: float, implied_risk: float) -> CreditSpreadExitPlan:
    if days_to_expiry > 35:
        take_profit = 0.28
    elif days_to_expiry > 21:
        take_profit = 0.38
    elif days_to_expiry > 10:
        take_profit = 0.50
    else:
        take_profit = 0.62

    if pop >= 75.0:
        take_profit = max(0.18, take_profit - 0.08)
        stop_loss = 1.85
        time_stop_dte = 8
    elif pop >= 60.0:
        stop_loss = 1.55
        time_stop_dte = 7
    else:
        take_profit = min(0.68, take_profit + 0.06)
        stop_loss = 1.20
        time_stop_dte = 10

    if implied_risk >= 0.75:
        stop_loss = min(stop_loss, 0.95)

    return CreditSpreadExitPlan(
        take_profit=round(take_profit, 4),
        stop_loss=round(stop_loss, 4),
        time_stop_dte=int(time_stop_dte),
        emergency_exit_risk=0.78,
    )


def build_long_option_exit_plan(days_to_expiry: int, is_cornwall: bool) -> LongOptionExitPlan:
    if is_cornwall:
        return LongOptionExitPlan(take_profit=5.0, stop_loss=0.55, time_stop_dte=3)
    if days_to_expiry > 21:
        return LongOptionExitPlan(take_profit=0.90, stop_loss=0.38, time_stop_dte=5)
    return LongOptionExitPlan(take_profit=0.75, stop_loss=0.32, time_stop_dte=3)

def get_portfolio_greeks(client, positions):
    """Calculates total Portfolio Delta, Theta, and Vega."""
    from alpaca.trading.enums import AssetClass
    
    option_positions = [p for p in positions if p.asset_class == AssetClass.US_OPTION]
    if not option_positions:
        return 0.0, 0.0, 0.0
        
    total_delta = 0.0
    total_theta = 0.0
    total_vega = 0.0  # <-- NEW
    
    try:
        priced_map = _build_position_contract_map(client, option_positions)
        
        for p in option_positions:
            qty = float(p.qty)
            contract = priced_map.get(p.symbol)
            if contract:
                delta = contract.delta or 0.0
                theta = contract.theta or 0.0
                vega = contract.vega or 0.0

                total_delta += (delta * 100 * qty)
                total_theta += (theta * 100 * qty)
                total_vega += (vega * 100 * qty)
                    
    except Exception as e:
        logger.debug(f"Could not calculate Greeks: {e}")
        
    return total_delta, total_theta, total_vega
    
def cleanup_stale_orders(client):
    """Cancels all open limit orders to free up buying power."""
    try:
        request_params = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        open_orders = client.trade_client.get_orders(filter=request_params)
        
        if not open_orders:
            logger.info("No stale orders to clean up.")
            return

        logger.info(f"=== CLEANING STALE ORDERS ({len(open_orders)}) ===")
        
        for order in open_orders:
            client.trade_client.cancel_order_by_id(order.id)
            logger.info(f"Cancelled: {order.symbol} {order.side} | Qty: {order.qty}")

        account = client.trade_client.get_account()
        logger.info(f"Cleanup Complete. Current Buying Power: ${float(account.buying_power):.2f}")
        
    except Exception as e:
        logger.error(f"Error during order cleanup: {e}")

def get_days_to_expiry(symbol):
    try:
        date_match = re.search(r'\d{6}', symbol)
        if date_match:
            date_str = date_match.group()
            expiry_date = datetime.strptime(date_str, "%y%m%d").replace(tzinfo=timezone.utc)
            return (expiry_date - datetime.now(timezone.utc)).days
    except Exception:
        pass
    return 45 # Fallback

def safe_close_position(client, symbol, reason=""):
    """Safely attempts to close a position and catches market-closed errors."""
    try:
        client.trade_client.close_position(symbol)
        return True
    except Exception as e:
        if "market hours" in str(e).lower():
            logger.warning(f"⏳ Market closed. Cannot exit {symbol} yet ({reason}).")
        else:
            logger.error(f"Failed to close {symbol}: {e}")
        return False

def calculate_dynamic_risk(current_vix):
    """Cuts position sizing automatically when market panic sets in."""
    if current_vix < 18.0:
        return MAX_RISK_BASE * 1.0   # Normal market: 100% size ($1,000)
    elif current_vix < 25.0:
        return MAX_RISK_BASE * 0.75  # Choppy market: 75% size ($750)
    elif current_vix < 35.0:
        return MAX_RISK_BASE * 0.50  # Fear market: 50% size ($500)
    else:
        return MAX_RISK_BASE * 0.25  # Panic market: 25% size ($250)

def sweep_idle_cash(client, current_equity):
    """Parks idle cash in Treasury Bills (SGOV) to earn risk-free yield."""
    try:
        account = client.trade_client.get_account()
        raw_cash = float(account.cash)
        
        # We only sweep if we have more cash than our buffer + $500 to make it worth the API call
        if raw_cash > (TARGET_CASH_BUFFER + 500):
            sweep_amount = raw_cash - TARGET_CASH_BUFFER
            
            # Get latest SGOV price to calculate shares
            latest_trade = client.get_stock_latest_trade(SWEEP_TICKER)
            sgov_price = float(latest_trade[SWEEP_TICKER].price)
            shares_to_buy = int(sweep_amount // sgov_price)
            
            if shares_to_buy > 0:
                client.market_buy(SWEEP_TICKER, qty=shares_to_buy)
                logger.info(f"🧹 CASH SWEEP: Parked ${shares_to_buy * sgov_price:.2f} of idle cash into {SWEEP_TICKER} (Yielding ~5.3%).")
                send_alert(f"🧹 **CASH SWEEP**: Parked ${shares_to_buy * sgov_price:.2f} of idle cash into {SWEEP_TICKER}.", "INFO")
    except Exception as e:
        logger.error(f"Cash sweep failed: {e}")

def manage_open_spreads(client, profit_target=PROFIT_TARGET, stop_loss=STOP_LOSS):
    cleanup_stale_orders(client)
    overlay_meta = get_equity_overlay_metadata()

    try:
        positions = client.trade_client.get_all_positions()
    except Exception as e:
        logger.error(f"Could not fetch positions: {e}")
        return []

    try:
        account = client.trade_client.get_account()
        equity, last_equity = float(account.equity), float(account.last_equity)
        daily_pnl_dollars = equity - last_equity
        daily_pnl_pct = (daily_pnl_dollars / last_equity) * 100 if last_equity > 0 else 0
        
        logger.info(f"=== ACCOUNT STATUS ===")
        logger.info(f"Today's P/L: ${daily_pnl_dollars:.2f} ({daily_pnl_pct:.2f}%) | Total Equity: ${equity:.2f}")
    except Exception as e:
        logger.warning(f"Could not fetch account daily P/L: {e}")

    if not positions:
        logger.info("No active positions to manage.")
        return []

    logger.info(f"=== PORTFOLIO MANAGER ===")

    closed_profits, closed_losses, closed_time, holding_status = [], [], [], []
    processed_straddles = set()
    priced_position_contracts = _build_position_contract_map(client, positions)

    spreads = {}
    for pos in positions:
        # --- NEW: Catch SGOV and normal stock/shares ---
        if pos.asset_class == AssetClass.US_EQUITY:
            profit_pct = float(pos.unrealized_plpc) * 100
            profit_dollars = float(pos.unrealized_pl)
            overlay_role = (overlay_meta.get(pos.symbol, {}) or {}).get("mode")
            role_suffix = f" | Role: {overlay_role}" if overlay_role else ""
            holding_status.append(
                f"{pos.symbol} [SHARES]: {profit_pct:+.2f}% (${profit_dollars:+.2f}) | Qty: {pos.qty}{role_suffix}"
            )
            continue

        if pos.asset_class != AssetClass.US_OPTION: continue
            
        underlying, opt_type, strike = parse_option_symbol(pos.symbol)
        key = f"{underlying}_{opt_type}" 
        
        if key not in spreads:
            spreads[key] = {'short': None, 'long': None, 'short_strike': 0.0, 'long_strike': 0.0}
            
        if int(pos.qty) < 0:
            spreads[key]['short'] = pos
            spreads[key]['short_strike'] = float(strike)
        elif int(pos.qty) > 0:
            spreads[key]['long'] = pos
            spreads[key]['long_strike'] = float(strike)

    for key, legs in spreads.items():
        underlying = key.split('_')[0]
        
        short_pos = legs['short']
        long_pos = legs['long']
        
        # =========================================================
        # 1. VEGA STRADDLE LOGIC (Long Call & Long Put)
        # =========================================================
        straddle_meta = get_straddle_metadata(underlying)
        if straddle_meta:
            if underlying in processed_straddles: continue
            processed_straddles.add(underlying)
            
            earnings_date = datetime.strptime(straddle_meta['earnings_date'], '%Y-%m-%d').date()
            days_to_earnings = (earnings_date - datetime.now(timezone.utc).date()).days
            
            c_pos = next((p for p in positions if p.symbol == straddle_meta['call_symbol']), None)
            p_pos = next((p for p in positions if p.symbol == straddle_meta['put_symbol']), None)
            
            if c_pos and p_pos:
                total_cost = abs(float(c_pos.cost_basis)) + abs(float(p_pos.cost_basis))
                total_val = _estimate_long_option_exit_value(c_pos, priced_position_contracts) + _estimate_long_option_exit_value(p_pos, priced_position_contracts)
                profit_pct = (total_val - total_cost) / total_cost if total_cost > 0 else 0
                
                if days_to_earnings <= 1 or profit_pct >= 0.50 or (days_to_earnings > 3 and profit_pct <= -0.35):
                    reason = (
                        "Pre-Earnings"
                        if days_to_earnings <= 1
                        else "Target Hit"
                        if profit_pct >= 0.50
                        else "Capital Protection"
                    )
                    logger.warning(f"🚨 [VEGA EXIT] {underlying} ({reason}). Attempting to exit Straddle.")
                    try:
                        client.market_sell(straddle_meta['call_symbol'])
                        client.market_sell(straddle_meta['put_symbol'])
                        closed_profits.append(f"{underlying}: Straddle Exited ({profit_pct*100:+.1f}%).")
                        remove_straddle_metadata(underlying)
                    except Exception as e:
                        if "market hours" in str(e).lower():
                            logger.warning(f"⏳ Market closed. Will exit {underlying} straddle at open.")
                        else:
                            logger.error(f"Failed to exit straddle for {underlying}: {e}")
                    continue 

                holding_status.append(f"{underlying} [STRADDLE]: {profit_pct*100:+.1f}% | Earnings in {days_to_earnings}d")
            continue
            
        # =========================================================
        # 2. THETA ENGINE LOGIC (Credit Spreads & Condors)
        # =========================================================
        if short_pos and long_pos:
            net_credit_collected = (float(short_pos.cost_basis) + float(long_pos.cost_basis)) * -1
            short_close_cost = _estimate_short_option_close_cost(short_pos, priced_position_contracts)
            long_exit_credit = _estimate_long_option_exit_value(long_pos, priced_position_contracts)
            current_cost_to_close = max(0.0, short_close_cost - long_exit_credit)
            
            if net_credit_collected <= 0: continue 
                
            profit_dollars = net_credit_collected - current_cost_to_close
            profit_pct = profit_dollars / net_credit_collected
            
            spread_width = abs(legs['short_strike'] - legs['long_strike'])
            qty = abs(int(short_pos.qty))
            current_price_per_share = current_cost_to_close / (100 * qty)
            
            implied_risk = current_price_per_share / spread_width if spread_width > 0 else 1
            pop = max(0, min(100, (1 - implied_risk) * 100)) 
            
            safety = "🟢 HIGH" if pop >= 75 else "🟡 MOD" if pop >= 50 else "🔴 RISK"
            days_to_expiry = get_days_to_expiry(short_pos.symbol)
            exit_plan = build_credit_spread_exit_plan(days_to_expiry, pop, implied_risk)
            dynamic_tp = min(float(profit_target), exit_plan.take_profit)
            dynamic_sl = min(float(stop_loss), exit_plan.stop_loss)
            status_str = (
                f"{underlying} [SPREAD]: {profit_pct*100:+.1f}% (${profit_dollars:+.2f}) | "
                f"TP: {dynamic_tp*100:.0f}% | SL: {dynamic_sl*100:.0f}% | DTE: {days_to_expiry} | "
                f"POP: {pop:.1f}% | Safety: {safety}"
            )
            
            if implied_risk >= exit_plan.emergency_exit_risk:
                if safe_close_position(client, short_pos.symbol, "Emergency Risk Exit") and safe_close_position(client, long_pos.symbol, "Emergency Risk Exit"):
                    closed_losses.append(f"{status_str} | Emergency risk exit")
            elif profit_pct >= dynamic_tp:
                if safe_close_position(client, short_pos.symbol, "Take Profit") and safe_close_position(client, long_pos.symbol, "Take Profit"):
                    closed_profits.append(status_str)
            elif profit_pct <= -dynamic_sl:
                if safe_close_position(client, short_pos.symbol, "Stop Loss") and safe_close_position(client, long_pos.symbol, "Stop Loss"):
                    closed_losses.append(status_str)
            elif days_to_expiry <= exit_plan.time_stop_dte:
                if safe_close_position(client, short_pos.symbol, "Time Stop") and safe_close_position(client, long_pos.symbol, "Time Stop"):
                    closed_time.append(f"{underlying} [SPREAD]: {days_to_expiry} DTE | POP was {pop:.1f}%")
            else:
                holding_status.append(status_str)

        # =========================================================
        # 3. ASYMMETRIC / DOOMSDAY LOGIC (Standalone Longs)
        # =========================================================
        elif long_pos and not short_pos:
            cost = abs(float(long_pos.cost_basis))
            val = _estimate_long_option_exit_value(long_pos, priced_position_contracts)
            profit_pct = (val - cost) / cost if cost > 0 else 0
            profit_dollars = val - cost
            days_to_expiry = get_days_to_expiry(long_pos.symbol)

            is_cornwall = cost < 150.0  
            exit_plan = build_long_option_exit_plan(days_to_expiry, is_cornwall)
            target_pct = exit_plan.take_profit
            strat_name = "CORNWALL" if is_cornwall else "TAIL HEDGE"

            status_str = (
                f"{underlying} [{strat_name}]: {profit_pct*100:+.1f}% (${profit_dollars:+.2f}) | "
                f"TP: {target_pct*100:.0f}% | SL: {exit_plan.stop_loss*100:.0f}% | DTE: {days_to_expiry}"
            )

            if profit_pct >= target_pct:
                if safe_close_position(client, long_pos.symbol, "Target Hit"):
                    closed_profits.append(status_str)
            elif profit_pct <= -exit_plan.stop_loss:
                if safe_close_position(client, long_pos.symbol, "Stop Loss"):
                    closed_losses.append(status_str)
            elif days_to_expiry <= exit_plan.time_stop_dte:
                if safe_close_position(client, long_pos.symbol, "Expiring"):
                    closed_time.append(f"{underlying} [{strat_name}]: Expiring in {days_to_expiry}d. Exited.")
            else:
                holding_status.append(status_str)

    # --- 4. PRINT FINAL SUMMARY ---
    if holding_status:
        logger.info("--- CURRENT OPEN POSITIONS (HOLDING) ---")
        for status in holding_status:
            logger.info(f"[HOLDING] {status}")
            
    if closed_profits or closed_losses or closed_time:
        logger.info("--- POSITIONS CLOSED THIS RUN ---")
        if closed_profits: 
            msg = " | ".join(closed_profits)
            logger.info(f"[PROFIT HIT]  : {msg}")
            send_alert(f"💰 **PROFIT TAKEN:**\n{msg}", "SUCCESS")
        if closed_losses: 
            msg = " | ".join(closed_losses)
            logger.info(f"[STOPLOSS HIT]: {msg}")
            send_alert(f"🛑 **STOP LOSS HIT:**\n{msg}", "WARNING")
        if closed_time: 
            msg = " | ".join(closed_time)
            logger.info(f"[TIME STOP]   : {msg}")
            send_alert(f"⏰ **TIME STOP (Expired):**\n{msg}", "INFO")
    else:
        logger.info("--- NO POSITIONS CLOSED THIS RUN ---")
    return holding_status


def _build_position_contract_map(client, option_positions):
    if not option_positions:
        return {}

    raw_contracts = []
    for pos in option_positions:
        underlying, opt_type, strike = parse_option_symbol(pos.symbol)
        raw_contracts.append(
            Contract(
                underlying=underlying,
                symbol=pos.symbol,
                contract_type="call" if opt_type == "C" else "put",
                dte=get_days_to_expiry(pos.symbol),
                strike=float(strike),
            )
        )

    try:
        snapshots = client.get_option_snapshot([contract.symbol for contract in raw_contracts])
        priced_contracts = build_delay_adjusted_contracts(client, raw_contracts, snapshots=snapshots)
        return {contract.symbol: contract for contract in priced_contracts}
    except Exception as exc:
        logger.debug("Could not build delay-aware position contract map: %s", exc)
        return {}


def _estimate_long_option_exit_value(position, priced_map):
    contract = priced_map.get(position.symbol)
    bid_price = effective_bid_price(contract)
    qty = abs(float(position.qty))
    if bid_price > 0:
        return bid_price * 100.0 * qty
    return max(0.0, float(position.market_value))


def _estimate_short_option_close_cost(position, priced_map):
    contract = priced_map.get(position.symbol)
    ask_price = effective_ask_price(contract)
    qty = abs(float(position.qty))
    if ask_price > 0:
        return ask_price * 100.0 * qty
    return max(0.0, -float(position.market_value))
