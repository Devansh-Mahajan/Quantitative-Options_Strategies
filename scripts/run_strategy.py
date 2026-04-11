import re
from pathlib import Path
import os
from core.broker_client import BrokerClient
from core.execution import sell_puts, sell_calls, buy_straddles, sell_iron_condors, buy_tail_hedge, deploy_asymmetric_bets
from core.state_manager import update_state, calculate_risk
from config.credentials import ALPACA_API_KEY, ALPACA_SECRET_KEY, IS_PAPER
from logging.strategy_logger import StrategyLogger
from logging.logger_setup import setup_logger
from core.cli_args import parse_args
import core.state_manager as state_manager_module

from config.params import (
    RISK_ALLOCATION,
    MAX_SPREADS_PER_SYMBOL,
    AVOID_EARNINGS,
    EXPIRATION_MAX,
    PROFIT_TARGET,
    STOP_LOSS,
    MAX_RISK_PER_SPREAD,
    MIN_SIGNAL_CONFIDENCE,
    LOW_CONFIDENCE_RISK_MULTIPLIER,
    MAX_NEW_TRADES_PER_CYCLE,
)
# --- Dashboard and Alert Tools ---
from core.manager import manage_open_spreads, get_portfolio_greeks, sweep_idle_cash, calculate_dynamic_risk
from core.notifications import send_alert
from core.sentiment import get_market_sentiment, get_vix_level
from core.movement_predictor import aggregate_movement_signals
from core.greeks_targeting import derive_portfolio_greek_targets

# 🧠 THE AI INFERENCE ENGINES
from core.regime_detection import get_brain_prediction # Macro market mood
from scripts.mega_screener import get_mega_brain_targets
from core.market_intelligence import estimate_institutional_flow
def main():
    args = parse_args()
    
    strat_logger = StrategyLogger(enabled=args.strat_log)
    logger = setup_logger(level=args.log_level, to_file=args.log_to_file)
    strat_logger.set_fresh_start(args.fresh_start)

    symbols_file = Path(__file__).parent.parent / "config" / "symbol_list.txt"
    weekend_symbols_file = Path(__file__).parent.parent / "config" / "volatile_symbols.txt"
    chosen_file = weekend_symbols_file if weekend_symbols_file.exists() else symbols_file
    with open(chosen_file, 'r') as file:
        SYMBOLS = [line.strip() for line in file.readlines() if line.strip()]

    client = BrokerClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, paper=IS_PAPER)

    # --- Initialize holding_status BEFORE the if/else block ---
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

    # Hedge-fund flow proxy: prioritize candidates with persistent momentum + volume surprise
    flow_map = estimate_institutional_flow(allowed_symbols) if allowed_symbols else {}
    allowed_symbols = sorted(allowed_symbols, key=lambda s: flow_map.get(s, 0.0), reverse=True)

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
    predictor_universe = allowed_symbols[:20] if allowed_symbols else SYMBOLS[:20]
    movement_signals = aggregate_movement_signals(predictor_universe, lookback="5y")

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

        greek_targets = derive_portfolio_greek_targets(
            movement_signals=movement_signals,
            equity=total_equity,
            vix_level=current_vix,
        )
        
        if port_delta > 10: bias = "Bullish"
        elif port_delta < -10: bias = "Bearish"
        else: bias = "Neutral"

        # --- 🧠 ASK THE MACRO BRAIN (Regime Gatekeeper) ---
        logger.info("🧠 Consulting the Macro Regime Gatekeeper...")
        brain_strategy, brain_confidence, brain_probs = get_brain_prediction()

        # --- TERMINAL DASHBOARD ---
        logger.info(f"🟢 PORTFOLIO HEALTHY: Daily P/L is {daily_pnl_pct:+.2f}%. VIX: {current_vix:.2f} -> Risk/Trade: ${dynamic_max_risk:.2f}.")
        logger.info(f"📊 PORTFOLIO DELTA: {port_delta:+.2f} ({bias})")
        logger.info(
            "🎯 TARGET GREEKS -> Delta: %+.2f Theta: %+.2f Vega: %+.2f Gamma: %+.2f | Bias: %s (Conf: %.1f%%)",
            greek_targets.target_delta,
            greek_targets.target_theta,
            greek_targets.target_vega,
            greek_targets.target_gamma,
            greek_targets.movement_bias,
            greek_targets.target_confidence * 100.0,
        )
        logger.info(f"⏳ PORTFOLIO THETA: ${port_theta:+.2f} / day")
        logger.info(f"🌊 PORTFOLIO VEGA: ${port_vega:+.2f} per 1% IV change")
        logger.info(f"🔮 MACRO REGIME VERDICT: {brain_strategy} ({brain_confidence:.1f}% Confidence)")

        signal_confidence = greek_targets.target_confidence
        macro_confidence = max(0.0, min(1.0, brain_confidence / 100.0))
        deployment_scale = signal_confidence * macro_confidence
        if current_vix >= 28:
            deployment_scale *= 0.75
        
        # --- DISCORD DASHBOARD ---
        holdings_str = "\n".join([f"• {h}" for h in holding_status]) if holding_status else "No open positions."
        
        # Pick an emoji based on the AI's mood
        brain_emoji = "🟢" if brain_strategy == "THETA_ENGINE" else "🟡" if brain_strategy == "VEGA_SNIPER" else "🔴"
        
        dashboard_msg = (
            f"**Equity:** ${total_equity:.2f} (P/L: {daily_pnl_pct:+.2f}%)\n"
            f"**Buying Power:** ${buying_power:.2f}\n"
            f"**VIX Level:** {current_vix:.2f} (Risk Cap: ${dynamic_max_risk:.0f})\n"
            f"**Delta:** {port_delta:+.2f} ({bias})\n"
            f"**Target Delta Bias:** {greek_targets.movement_bias} ({greek_targets.target_delta:+.2f})\n"
            f"**Theta (Daily Rent):** ${port_theta:+.2f}\n"
            f"**Vega (Vol Risk):** ${port_vega:+.2f}\n"
            f"**🧠 Macro Regime:** {brain_emoji} **{brain_strategy}** ({brain_confidence:.1f}%)\n\n"
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
            if signal_confidence < MIN_SIGNAL_CONFIDENCE:
                dynamic_max_risk *= LOW_CONFIDENCE_RISK_MULTIPLIER
                buying_power *= LOW_CONFIDENCE_RISK_MULTIPLIER
                logger.warning(
                    "🧩 LOW-CONFIDENCE REGIME: signal confidence %.2f below %.2f. Cutting risk and fresh deployment by %.0f%%.",
                    signal_confidence,
                    MIN_SIGNAL_CONFIDENCE,
                    (1.0 - LOW_CONFIDENCE_RISK_MULTIPLIER) * 100.0,
                )

            logger.info(f"Step 2: Hunting for new setups with ${buying_power:.2f} BP (Total Equity: ${total_equity:.2f})...")
            
            # --- 0A. THE DOOMSDAY PROTOCOL ---
            try:
                buy_tail_hedge(client, total_equity, positions, dynamic_max_risk, port_delta) 
            except TypeError:
                logger.error("⚠️ buy_tail_hedge in execution.py needs to be updated to accept port_delta!")

            # =======================================================
            # --- 1. RTX 5090 MEGA BRAIN INTELLIGENCE GATHERING ---
            # =======================================================
            logger.info("🧠 Awakening the RTX 5090: Scanning 150 tickers for neural setups...")
            
            ai_targets = get_mega_brain_targets(confidence_threshold=75.0)
            
            # Apply Concentration Risk Gates (Only allow symbols we aren't maxed out on)
            theta_candidates = [sym for sym in ai_targets['THETA'] if sym in allowed_symbols]
            vega_candidates  = [sym for sym in ai_targets['VEGA'] if sym in allowed_symbols]
            bull_candidates  = [sym for sym in ai_targets['BULL'] if sym in allowed_symbols]
            bear_candidates  = [sym for sym in ai_targets['BEAR'] if sym in allowed_symbols]

            throttle_n = max(
                1,
                min(
                    MAX_NEW_TRADES_PER_CYCLE,
                    int(round(MAX_NEW_TRADES_PER_CYCLE * max(0.25, deployment_scale)))
                ),
            )
            theta_candidates = theta_candidates[:throttle_n]
            vega_candidates = vega_candidates[:throttle_n]
            bull_candidates = bull_candidates[:throttle_n]
            bear_candidates = bear_candidates[:throttle_n]
            
            logger.info(f"📊 NEURAL TARGETS ALIGNED WITH RISK: Theta(Condors): {len(theta_candidates)} | Vega(Straddles): {len(vega_candidates)} | Bull(Puts): {len(bull_candidates)} | Bear(Calls): {len(bear_candidates)}")

            # --- 0B. ASYMMETRIC / CONVEXITY BETS (The Lottery Tickets) ---
            # Feed the most stable/range-bound (THETA) candidates to asymmetric generator
            deploy_asymmetric_bets(client, theta_candidates, total_equity, positions)

            # --- 2A. DEPLOY VEGA ENGINE (Long Straddles) ---
            if greek_targets.target_vega > 0:
                vega_candidates = ai_targets['VEGA'][:max(3, len(ai_targets['VEGA']) // 2)]

            if vega_candidates:
                logger.info(f"Step 2A: Launching Vega Sniper on {len(vega_candidates)} AI targets.")
                buying_power = buy_straddles(
                    client=client, 
                    symbols_list=vega_candidates, 
                    buying_power=buying_power, 
                    max_risk_limit=dynamic_max_risk, 
                    state_manager=state_manager_module, 
                    strat_logger=strat_logger
                )
            else:
                logger.info("Step 2A: No Vega (Straddle) opportunities met the AI confidence threshold.")

            # --- 2B. DEPLOY THETA ENGINE (Iron Condors) ---
            if theta_candidates and buying_power >= 100:
                # 🛑 MACRO GATEKEEPER CHECK FOR CREDIT STRATEGIES 🛑
                if brain_strategy == "TAIL_HEDGE":
                    logger.warning("🚨 MACRO AI SAYS TAIL_HEDGE: Market crash probability high. Iron Condors DEACTIVATED.")
                elif brain_strategy == "VEGA_SNIPER":
                    logger.warning("🟡 MACRO AI SAYS VEGA_SNIPER: Volatility expanding. Iron Condors DEACTIVATED to prevent blowouts.")
                else:
                    logger.info(f"Step 2B: Launching AI Theta Engine (Iron Condors) on {len(theta_candidates)} targets.")
                    buying_power = sell_iron_condors(
                        client=client, 
                        symbols_list=theta_candidates, 
                        buying_power=buying_power, 
                        max_risk_limit=dynamic_max_risk, 
                        strat_logger=strat_logger
                    )
            else:
                logger.info("Step 2B: Insufficient BP or no Theta (Condor) opportunities found.")

            # --- 2C. DEPLOY DIRECTIONAL EDGE (Bull / Bear Credit Spreads) ---
            if buying_power >= 50:
                if greek_targets.movement_bias == "bullish":
                    bear_candidates = bear_candidates[: max(1, len(bear_candidates) // 4)]
                elif greek_targets.movement_bias == "bearish":
                    bull_candidates = bull_candidates[: max(1, len(bull_candidates) // 4)]

                if brain_strategy == "TAIL_HEDGE" or brain_strategy == "VEGA_SNIPER":
                    logger.warning("🟡 MACRO AI GATE: Market is too volatile for directional short premium. Skipping Put/Call Spreads.")
                elif current_vix >= 15.0:
                    half_bp = buying_power / 2.0
                    
                    if bull_candidates and half_bp >= 50:
                        logger.info(f"🧠 Step 2C: Deploying AI Bullish Edge (Put Credit Spreads) on {len(bull_candidates)} tickers.")
                        sell_puts(client, bull_candidates, half_bp, dynamic_max_risk, strat_logger)
                        
                    if bear_candidates and half_bp >= 50:
                        logger.info(f"🧠 Step 2D: Deploying AI Bearish Edge (Call Credit Spreads) on {len(bear_candidates)} tickers.")
                        sell_calls(client, bear_candidates, purchase_price=None, stock_qty=0, buying_power=half_bp, max_risk_limit=dynamic_max_risk, strat_logger=strat_logger)
                else:
                    logger.info(f"Step 2C: VIX is too low ({current_vix}) for directional Credit Spreads. Remaining BP (${buying_power:.2f}) held in cash.")
            elif buying_power < 50:
                logger.info(f"Remaining BP (${buying_power:.2f}) exhausted. Skipping Directional Bets.")
                
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
