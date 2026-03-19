import re
from pathlib import Path
from core.broker_client import BrokerClient
from core.execution import sell_puts, sell_calls, buy_straddles, sell_iron_condors, buy_tail_hedge, deploy_asymmetric_bets
from core.state_manager import update_state, calculate_risk
from config.credentials import ALPACA_API_KEY, ALPACA_SECRET_KEY, IS_PAPER
from logging.strategy_logger import StrategyLogger
from logging.logger_setup import setup_logger
from core.cli_args import parse_args
import core.state_manager as state_manager_module

from config.params import RISK_ALLOCATION, MAX_SPREADS_PER_SYMBOL, AVOID_EARNINGS, EXPIRATION_MAX, PROFIT_TARGET, STOP_LOSS, MAX_RISK_PER_SPREAD
# --- NEW: Imported our Dashboard and Alert Tools ---
from core.manager import manage_open_spreads, get_portfolio_greeks, sweep_idle_cash, calculate_dynamic_risk
from core.notifications import send_alert
from core.sentiment import get_market_sentiment, get_vix_level
from core.earnings import analyze_market_regime

def main():
    args = parse_args()
    
    strat_logger = StrategyLogger(enabled=args.strat_log)
    logger = setup_logger(level=args.log_level, to_file=args.log_to_file)
    strat_logger.set_fresh_start(args.fresh_start)

    SYMBOLS_FILE = Path(__file__).parent.parent / "config" / "symbol_list.txt"
    with open(SYMBOLS_FILE, 'r') as file:
        SYMBOLS = [line.strip() for line in file.readlines()]

    client = BrokerClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, paper=IS_PAPER)

    # --- FIX: Initialize holding_status BEFORE the if/else block ---
    holding_status = []

    if args.fresh_start:
        logger.info("Fresh Start: Canceling pending orders and liquidating positions.")
        client.trade_client.cancel_orders() 
        client.liquidate_all_positions()
    else:
        logger.info("Step 1: Managing active positions for TP/SL/Expiry...")
        holding_status = manage_open_spreads(client, profit_target=PROFIT_TARGET, stop_loss=STOP_LOSS)
        if holding_status is None: 
            holding_status = [] # Safety catch

    positions = client.get_positions()
    strat_logger.add_current_positions(positions)

    current_risk = calculate_risk(positions)
    states = update_state(positions)
    strat_logger.add_state_dict(states)

    # Handle Assigned Shares (Wheel Fallback)
    for symbol, state in states.items():
        if state["type"] == "long_shares":
            sell_calls(client, symbol, purchase_price=state["price"], stock_qty=state["qty"], max_risk_limit=MAX_RISK_PER_SPREAD, strat_logger=strat_logger)

    # --- CONCENTRATION LIMIT (Max Spreads Per Symbol) ---
    active_counts = {}
    for pos in positions:
        match = re.match(r'^([A-Z]{1,6})\d{6}[CP]\d{8}$', pos.symbol)
        if match:
            root = match.group(1)
            active_counts[root] = active_counts.get(root, 0) + 1
            
    allowed_symbols = []
    for sym in SYMBOLS:
        if active_counts.get(sym, 0) < (MAX_SPREADS_PER_SYMBOL * 2):
            if sym not in states:
                allowed_symbols.append(sym)
        else:
            logger.info(f"Skipping {sym}: Maximum allocation ({MAX_SPREADS_PER_SYMBOL} spread) reached.")

    account = client.trade_client.get_account()
    total_equity = float(account.portfolio_value)
    
    # ==========================================================
    # --- QUANT CIRCUIT BREAKER & DRAWDOWN GUARD (DAILY P/L) ---
    # ==========================================================
    last_equity = float(account.last_equity) if account.last_equity else total_equity
    daily_pnl_dollars = total_equity - last_equity
    daily_pnl_pct = (daily_pnl_dollars / last_equity) * 100 if last_equity > 0 else 0

    # Grab VIX early so we can scale risk and print it on the dashboard
    current_vix = get_vix_level()

    if daily_pnl_pct <= -10.0:
        logger.critical(f"🔴 KILL SWITCH TRIGGERED 🔴 Daily P/L is {daily_pnl_pct:.2f}% (${daily_pnl_dollars:.2f}). Trading halted.")
        send_alert(f"🚨 **KILL SWITCH TRIGGERED** 🚨\nDaily P/L is {daily_pnl_pct:.2f}%. Trading suspended.", "ERROR")
        buying_power = 0  
        dynamic_max_risk = 0
    else:
        max_risk_allowed = total_equity * RISK_ALLOCATION
        buying_power = max_risk_allowed - current_risk

        # --- VIX-SCALED DYNAMIC RISK ---
        vix_scaled_risk = calculate_dynamic_risk(current_vix)

        if daily_pnl_pct <= -5.0:
            dynamic_max_risk = vix_scaled_risk / 2
            logger.warning(f"🟡 DRAWDOWN GUARD: Daily P/L is {daily_pnl_pct:.2f}%. Halving max risk per trade to ${dynamic_max_risk:.2f}.")
        else:
            dynamic_max_risk = vix_scaled_risk

        # --- BULLETPROOF GREEKS CALCULATION ---
        greeks = get_portfolio_greeks(client, positions)
        port_delta = greeks[0] if len(greeks) > 0 else 0.0
        port_theta = greeks[1] if len(greeks) > 1 else 0.0
        port_vega  = greeks[2] if len(greeks) > 2 else 0.0 
        
        if port_delta > 10: bias = "Bullish"
        elif port_delta < -10: bias = "Bearish"
        else: bias = "Neutral"

        # --- TERMINAL DASHBOARD ---
        logger.info(f"🟢 PORTFOLIO HEALTHY: Daily P/L is {daily_pnl_pct:+.2f}%. VIX: {current_vix:.2f} -> Risk/Trade: ${dynamic_max_risk:.2f}.")
        logger.info(f"📊 PORTFOLIO DELTA: {port_delta:+.2f} ({bias})")
        logger.info(f"⏳ PORTFOLIO THETA: ${port_theta:+.2f} / day")
        logger.info(f"🌊 PORTFOLIO VEGA: ${port_vega:+.2f} per 1% IV change")
        
        # --- DISCORD DASHBOARD ---
        holdings_str = "\n".join([f"• {h}" for h in holding_status]) if holding_status else "No open positions."
        
        dashboard_msg = (
            f"**Equity:** ${total_equity:.2f} (P/L: {daily_pnl_pct:+.2f}%)\n"
            f"**Buying Power:** ${buying_power:.2f}\n"
            f"**VIX Level:** {current_vix:.2f} (Risk Cap: ${dynamic_max_risk:.0f})\n"
            f"**Delta:** {port_delta:+.2f} ({bias})\n"
            f"**Theta (Daily Rent):** ${port_theta:+.2f}\n"
            f"**Vega (Vol Risk):** ${port_vega:+.2f}\n\n"
            f"**📦 CURRENT HOLDINGS:**\n{holdings_str}"
        )
        send_alert(dashboard_msg, "INFO")
        
        # --- IDLE CASH SWEEP ---
        sweep_idle_cash(client, total_equity)
    # ==========================================================

    strat_logger.set_buying_power(buying_power)
    strat_logger.set_allowed_symbols(allowed_symbols)

    # ==========================================================
    # --- MANAGE-ONLY GUARD & HUNTING ROUTINE ---
    # ==========================================================
    if not args.manage_only:
        if buying_power >= 50:
            logger.info(f"Step 2: Hunting for new setups with ${buying_power:.2f} BP (Total Equity: ${total_equity:.2f})...")
            
            # --- 0A. THE DOOMSDAY PROTOCOL ---
            # We pass port_delta to the hedge so it can Delta-Neutralize us
            try:
                buy_tail_hedge(client, total_equity, positions, dynamic_max_risk, port_delta) 
            except TypeError:
                logger.error("⚠️ buy_tail_hedge in execution.py needs to be updated to accept port_delta!")

            # --- 1. INTELLIGENCE GATHERING ---
            theta_candidates, vega_candidates, crush_candidates = analyze_market_regime(allowed_symbols, max_dte=EXPIRATION_MAX)
            
            logger.info(f"📊 SCAN RESULTS: Theta(Safe): {len(theta_candidates)} | Vega(Straddles): {len(vega_candidates)} | IV Crush(Condors): {len(crush_candidates)}")
            
            if not AVOID_EARNINGS:
                vega_syms = [v['symbol'] if isinstance(v, dict) else v for v in vega_candidates]
                crush_syms = [c['symbol'] if isinstance(c, dict) else c for c in crush_candidates]
                theta_candidates = [sym for sym in allowed_symbols if sym not in vega_syms and sym not in crush_syms]

            # --- 0B. ASYMMETRIC / CONVEXITY BETS (The Lottery Tickets) ---
            deploy_asymmetric_bets(client, theta_candidates, total_equity, positions)

            # --- 2A. DEPLOY VEGA ENGINE (Long Straddles) ---
            if vega_candidates:
                logger.info(f"Step 2A: Launching Vega Sniper on {len(vega_candidates)} earnings setups.")
                buying_power = buy_straddles(
                    client=client, 
                    symbols_list=vega_candidates, 
                    buying_power=buying_power, 
                    max_risk_limit=dynamic_max_risk, 
                    state_manager=state_manager_module, 
                    strat_logger=strat_logger
                )
            else:
                logger.info("Step 2A: No Vega (Straddle) opportunities found today.")

            # --- 2B. DEPLOY IV CRUSH ENGINE (Iron Condors) ---
            if crush_candidates and buying_power >= 100:
                logger.info(f"Step 2B: Launching IV Crush Condors on {len(crush_candidates)} tickers.")
                buying_power = sell_iron_condors(
                    client=client, 
                    symbols_list=crush_candidates, 
                    buying_power=buying_power, 
                    max_risk_limit=dynamic_max_risk, 
                    strat_logger=strat_logger
                )
            else:
                logger.info("Step 2B: No IV Crush (Condor) opportunities found today.")

            # --- 2C. DEPLOY THETA ENGINE (Credit Spreads) ---
            if buying_power >= 50 and theta_candidates:
                if current_vix >= 15.0:
                    logger.info(f"Step 2C: VIX is {current_vix}. Deploying Theta Engine on {len(theta_candidates)} safe symbols.")
                    sentiment = get_market_sentiment(client)
                    
                    if sentiment == "bullish":
                        logger.info("Trend is BULLISH: Deploying capital to Put Credit Spreads.")
                        sell_puts(client, theta_candidates, buying_power, dynamic_max_risk, strat_logger)
                    elif sentiment == "bearish":
                        logger.info("Trend is BEARISH: Deploying capital to Call Credit Spreads.")
                        sell_calls(client, theta_candidates, purchase_price=None, stock_qty=0, buying_power=buying_power, max_risk_limit=dynamic_max_risk, strat_logger=strat_logger)
                    else:
                        logger.info("Trend is NEUTRAL: Splitting BP evenly between Puts and Calls.")
                        half_bp = buying_power / 2
                        if half_bp >= 50:
                            sell_puts(client, theta_candidates, half_bp, dynamic_max_risk, strat_logger)
                            sell_calls(client, theta_candidates, purchase_price=None, stock_qty=0, buying_power=half_bp, max_risk_limit=dynamic_max_risk, strat_logger=strat_logger)
                        else:
                            sell_puts(client, theta_candidates, buying_power, dynamic_max_risk, strat_logger)
                else:
                    logger.info(f"Step 2C: VIX is too low ({current_vix}) for Credit Spreads. Remaining BP (${buying_power:.2f}) held in cash.")
            elif buying_power < 50:
                logger.info(f"Remaining BP (${buying_power:.2f}) exhausted by Vega/Crush trades. Skipping Theta Engine.")
                
        else:
            if buying_power == 0 and dynamic_max_risk == 0:
                pass 
            else:
                logger.info(f"Remaining BP (${buying_power:.2f}) is too low for new ${dynamic_max_risk} spreads.")
    else:
        logger.info("Manage-Only mode activated. Skipping Step 2 (Hunting for new Spreads).")
    # ==========================================================

    strat_logger.save()    

if __name__ == "__main__":
    main()