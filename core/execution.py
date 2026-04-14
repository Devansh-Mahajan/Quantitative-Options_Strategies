import logging
import re
from datetime import datetime, timezone, timedelta
from config.params import (
    EXPIRATION_MAX,
    EXPIRATION_MIN,
    MAX_BID_ASK_SPREAD,
    MAX_RELATIVE_SPREAD,
    OPTION_DELAY_MIN_PRICING_CONFIDENCE,
    SLIPPAGE_ALLOWANCE,
)
from core.delay_aware_options import (
    build_delay_adjusted_contracts,
    effective_ask_price,
    effective_bid_price,
    effective_mid_price,
)
from .strategy import filter_underlying, filter_options, score_options, select_options
from models.contract import Contract
import numpy as np

from core.notifications import send_alert  # <-- THE NEW DISCORD WEBHOOK

logger = logging.getLogger(f"strategy.{__name__}")

def parse_occ_symbol(symbol):
    match = re.match(r'^([A-Z]{1,6})(\d{6})([CP])(\d{8})$', symbol)
    if match:
        root, exp, type_, strike = match.groups()
        return root, exp, type_, float(strike) / 1000.0
    return None, None, None, None


def sell_puts(client, allowed_symbols, buying_power, max_risk_limit, strat_logger=None, is_condor=False, override_expiry=None):
    if not allowed_symbols or buying_power <= 0: return

    logger.info("Searching for put options...")
    filtered_symbols = allowed_symbols if is_condor else filter_underlying(client, allowed_symbols, buying_power)
    
    if strat_logger: strat_logger.set_filtered_symbols(filtered_symbols)
    if len(filtered_symbols) == 0: return
        
    option_contracts = client.get_options_contracts(
        filtered_symbols, 'put', 
        exact_date=override_expiry, 
        min_days=EXPIRATION_MIN, 
        max_days=EXPIRATION_MAX
    )
    
    snapshots = client.get_option_snapshot([c.symbol for c in option_contracts])
    priced_contracts = build_delay_adjusted_contracts(client, option_contracts, snapshots=snapshots)
    priced_contract_map = {contract.symbol: contract for contract in priced_contracts}
    
    put_options = filter_options(
        list(priced_contract_map.values()),
        target_expiry=override_expiry
    )
    
    if put_options:
        scores = score_options(put_options)
        put_options = select_options(put_options, scores)
        
        for p in put_options:
            root, exp, type_, short_strike = parse_occ_symbol(p.symbol)
            possible_longs = []
            for raw_contract in option_contracts:
                c_root, c_exp, c_type, c_strike = parse_occ_symbol(raw_contract.symbol)
                if c_root == root and c_exp == exp and c_strike < short_strike:
                    possible_longs.append((c_strike, raw_contract.symbol))
            
            if not possible_longs: continue
            possible_longs.sort(key=lambda x: x[0], reverse=True)
            long_strike, long_symbol = possible_longs[0]
            
            max_risk = round(abs(short_strike - long_strike) * 100, 2)
            if not is_condor and (max_risk > max_risk_limit or buying_power < max_risk): 
                continue 

            short_contract = priced_contract_map.get(p.symbol)
            long_contract = priced_contract_map.get(long_symbol)
            if not short_contract or not long_contract:
                continue
            if min(
                float(short_contract.pricing_confidence or 0.0),
                float(long_contract.pricing_confidence or 0.0),
            ) < OPTION_DELAY_MIN_PRICING_CONFIDENCE:
                continue

            sb, sa = effective_bid_price(short_contract), effective_ask_price(short_contract)
            lb, la = effective_bid_price(long_contract), effective_ask_price(long_contract)
            if min(sb, sa, lb, la) <= 0:
                continue
                
            short_mid, long_mid = effective_mid_price(short_contract), effective_mid_price(long_contract)
            short_rel_spread = (sa - sb) / short_mid if short_mid > 0 else 1.0
            long_rel_spread = (la - lb) / long_mid if long_mid > 0 else 1.0

            active_max_rel_spread = MAX_RELATIVE_SPREAD * 3.0 if is_condor else MAX_RELATIVE_SPREAD

            if (sa - sb) > MAX_BID_ASK_SPREAD or short_rel_spread > active_max_rel_spread: continue
            if (la - lb) > MAX_BID_ASK_SPREAD or long_rel_spread > active_max_rel_spread: continue

            mid_credit, natural_credit = short_mid - long_mid, sb - la  
            if natural_credit <= 0.0: continue
                
            limit_price = -abs(round(natural_credit + ((mid_credit - natural_credit) * 0.50 * SLIPPAGE_ALLOWANCE), 2))
            
            if abs(limit_price) < 0.06: continue
                
            logger.info(f"Executing Put Wing: SELL {p.symbol} & BUY {long_symbol} | Credit: ${abs(limit_price)}")
            try:
                client.execute_credit_spread(p.symbol, long_symbol, limit_price)
                buying_power -= max_risk 
                
                # --- DISCORD ALERT ---
                strat_name = "Condor Put Wing" if is_condor else "Put Credit Spread"
                send_alert(f"🦅 **NEW TRADE: {strat_name}**\n**Sold:** {p.symbol}\n**Bought:** {long_symbol}\n**Credit:** ${abs(limit_price):.2f}\n**Risk:** ${max_risk:.2f}", "INFO")
                
            except Exception as e:
                logger.error(f"Failed Put Wing: {e}")


def sell_calls(client, symbols, purchase_price=None, stock_qty=0, buying_power=0, max_risk_limit=300.0, strat_logger=None, is_condor=False, override_expiry=None):
    target_symbols = [symbols] if isinstance(symbols, str) else symbols
    if purchase_price is None and buying_power <= 0: return

    logger.info(f"Searching for call options...")
    filtered_symbols = target_symbols if (is_condor or purchase_price) else filter_underlying(client, target_symbols, buying_power)
    if not filtered_symbols: return

    raw_call_contracts = client.get_options_contracts(
        filtered_symbols, 'call', 
        exact_date=override_expiry, 
        min_days=EXPIRATION_MIN, 
        max_days=EXPIRATION_MAX
    )
    
    snapshots = client.get_option_snapshot([c.symbol for c in raw_call_contracts])
    priced_contracts = build_delay_adjusted_contracts(client, raw_call_contracts, snapshots=snapshots)
    priced_contract_map = {contract.symbol: contract for contract in priced_contracts}
    
    call_options = filter_options(
        list(priced_contract_map.values()), 
        min_strike=purchase_price or 0,
        target_expiry=override_expiry
    )
    
    if call_options:
        scores = score_options(call_options)
        call_options = select_options(call_options, scores)
        
        for p in call_options:
            root, exp, type_, short_strike = parse_occ_symbol(p.symbol)
            possible_longs = []
            for raw_c in raw_call_contracts:
                c_root, c_exp, c_type, c_strike = parse_occ_symbol(raw_c.symbol)
                if c_root == root and c_exp == exp and c_strike > short_strike:
                    possible_longs.append((c_strike, raw_c.symbol))
                    
            if possible_longs and purchase_price is None:
                possible_longs.sort(key=lambda x: x[0])
                long_strike, long_symbol = possible_longs[0]
                max_risk = round(abs(long_strike - short_strike) * 100, 2)
                
                if not is_condor and (max_risk > max_risk_limit or buying_power < max_risk): continue

                short_contract = priced_contract_map.get(p.symbol)
                long_contract = priced_contract_map.get(long_symbol)
                if not short_contract or not long_contract:
                    continue
                if min(
                    float(short_contract.pricing_confidence or 0.0),
                    float(long_contract.pricing_confidence or 0.0),
                ) < OPTION_DELAY_MIN_PRICING_CONFIDENCE:
                    continue

                sb, sa = effective_bid_price(short_contract), effective_ask_price(short_contract)
                lb, la = effective_bid_price(long_contract), effective_ask_price(long_contract)
                if min(sb, sa, lb, la) <= 0:
                    continue
                    
                short_mid, long_mid = effective_mid_price(short_contract), effective_mid_price(long_contract)
                short_rel_spread = (sa - sb) / short_mid if short_mid > 0 else 1.0
                long_rel_spread = (la - lb) / long_mid if long_mid > 0 else 1.0

                active_max_rel_spread = MAX_RELATIVE_SPREAD * 3.0 if is_condor else MAX_RELATIVE_SPREAD

                if (sa - sb) > MAX_BID_ASK_SPREAD or short_rel_spread > active_max_rel_spread: continue
                if (la - lb) > MAX_BID_ASK_SPREAD or long_rel_spread > active_max_rel_spread: continue

                mid_credit, natural_credit = short_mid - long_mid, sb - la 
                if natural_credit <= 0.0: continue
                    
                limit_price = -abs(round(natural_credit + ((mid_credit - natural_credit) * 0.50 * SLIPPAGE_ALLOWANCE), 2))
                
                if abs(limit_price) < 0.06: continue

                logger.info(f"Executing Call Wing: SELL {p.symbol} & BUY {long_symbol} | Credit: ${abs(limit_price)}")
                try:
                    client.execute_credit_spread(p.symbol, long_symbol, limit_price)
                    buying_power -= max_risk
                    
                    # --- DISCORD ALERT ---
                    strat_name = "Condor Call Wing" if is_condor else "Call Credit Spread"
                    send_alert(f"🦅 **NEW TRADE: {strat_name}**\n**Sold:** {p.symbol}\n**Bought:** {long_symbol}\n**Credit:** ${abs(limit_price):.2f}\n**Risk:** ${max_risk:.2f}", "INFO")

                except Exception as e:
                    logger.error(f"Failed Call Wing: {e}")
            elif purchase_price is not None:
                client.market_sell(p.symbol)

                
def buy_straddles(client, symbols_list, buying_power, max_risk_limit, state_manager=None, strat_logger=None):
    if not symbols_list or buying_power <= 0: return buying_power
    import yfinance as yf 

    for sym in symbols_list:
        if buying_power < 50: break
        symbol = sym if isinstance(sym, str) else sym.get('symbol')
        try:
            calls = client.get_options_contracts([symbol], 'call', min_days=0, max_days=45)
            puts = client.get_options_contracts([symbol], 'put', min_days=0, max_days=45)
            if not calls or not puts: continue
                
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if hist.empty: continue
            current_price = float(hist['Close'].iloc[-1])
            
            best_call, best_put = None, None
            min_diff = float('inf')
            
            for call in calls:
                _, exp, _, strike = parse_occ_symbol(call.symbol)
                diff = abs(strike - current_price)
                if diff < min_diff:
                    min_diff = diff
                    best_call = call.symbol
                    
            for put in puts:
                root, exp, _, strike = parse_occ_symbol(put.symbol)
                if best_call:
                    bc_root, bc_exp, _, bc_strike = parse_occ_symbol(best_call)
                    if root == bc_root and exp == bc_exp and strike == bc_strike:
                        best_put = put.symbol
                        break

            if not best_call or not best_put: continue

            snapshots = client.get_option_snapshot([best_call, best_put])
            raw_map = {contract.symbol: contract for contract in calls + puts}
            priced = build_delay_adjusted_contracts(client, [raw_map[best_call], raw_map[best_put]], snapshots=snapshots)
            priced_map = {contract.symbol: contract for contract in priced}
            call_contract = priced_map.get(best_call)
            put_contract = priced_map.get(best_put)
            if not call_contract or not put_contract:
                continue
            if min(
                float(call_contract.pricing_confidence or 0.0),
                float(put_contract.pricing_confidence or 0.0),
            ) < OPTION_DELAY_MIN_PRICING_CONFIDENCE:
                continue

            cb, ca = effective_bid_price(call_contract), effective_ask_price(call_contract)
            pb, pa = effective_bid_price(put_contract), effective_ask_price(put_contract)
            if min(cb, ca, pb, pa) <= 0:
                continue
                
            natural_debit = ca + pa
            mid_debit = effective_mid_price(call_contract) + effective_mid_price(put_contract)
            max_risk = round(natural_debit * 100, 2)
            if max_risk > max_risk_limit or buying_power < max_risk: continue
                
            dynamic_fill_factor = max(0.0, min(1.0, 0.50 * SLIPPAGE_ALLOWANCE))
            limit_price = round(natural_debit - ((natural_debit - mid_debit) * dynamic_fill_factor), 2)
            
            logger.info(f"🎯 Executing Long Straddle on {symbol}: BUY {best_call} & BUY {best_put} | Target Debit: ${limit_price}")
            client.execute_debit_spread(best_call, best_put, limit_price)
            buying_power -= max_risk
            
            # --- DISCORD ALERT ---
            send_alert(f"🎯 **NEW TRADE: Vega Straddle**\n**Call:** {best_call}\n**Put:** {best_put}\n**Target Debit:** ${limit_price:.2f}", "INFO")
            
            if state_manager:
                state_manager.register_straddle(symbol, best_call, best_put, "TBD")
        except Exception as e:
            logger.error(f"Failed to execute straddle for {symbol}: {e}")

    return buying_power


def sell_iron_condors(client, symbols_list, buying_power, max_risk_limit, strat_logger=None):
    if not symbols_list or buying_power <= 0: return buying_power
    for item in symbols_list:
        sym_str = item['symbol'] if isinstance(item, dict) else item
        target_expiry = item.get('expiry') if isinstance(item, dict) else None
        logger.info(f"--- 🌋 Building Condor for {sym_str} (Target Expiry: {target_expiry}) ---")
        sell_calls(client, [sym_str], buying_power=buying_power, max_risk_limit=max_risk_limit, is_condor=True, override_expiry=target_expiry)
        sell_puts(client, [sym_str], buying_power=buying_power, max_risk_limit=max_risk_limit, is_condor=True, override_expiry=target_expiry)

    try: return float(client.trade_client.get_account().buying_power)
    except: return buying_power


def buy_tail_hedge(client, total_equity, positions, max_risk_limit, port_delta):
    import math
    from datetime import datetime, timezone, timedelta
    
    # TIGHTENED LEASH: Hedge if Delta drifts past 10
    if abs(port_delta) < 10.0:
        return
        
    # If we are overly Bullish (+ Delta), we buy Puts. If Bearish (- Delta), we buy Calls.
    hedge_type = 'put' if port_delta > 0 else 'call'
    logger.info(f"🚨 PORTFOLIO DELTA IMBALANCE: {port_delta:+.2f}. Executing Delta-Neutral {hedge_type.capitalize()} Hedge...")

    # Check if we already have an active SPY hedge
    spy_hedges = sum(int(p.qty) for p in positions if p.symbol.startswith("SPY") and hedge_type.upper()[0] in p.symbol and int(p.qty) > 0)
    if spy_hedges > 0:
        logger.info(f"🛡️ Active SPY {hedge_type.capitalize()} Hedge already detected. Letting existing hedge work.")
        return
        
    try:
        # Get 60-120 day SPY options
        options_chain = client.get_options_contracts(['SPY'], hedge_type, min_days=60, max_days=120)
        target_date = datetime.now(timezone.utc).date() + timedelta(days=60)
        valid = [o for o in options_chain if datetime.strptime(parse_occ_symbol(o.symbol)[1], "%y%m%d").date() > target_date]
        
        # We want At-The-Money (ATM) options for the highest Gamma/Delta efficiency
        import yfinance as yf
        spy_price = yf.Ticker("SPY").history(period="1d")['Close'].iloc[-1]
        
        # Sort by closest to current price
        valid.sort(key=lambda x: abs(parse_occ_symbol(x.symbol)[3] - spy_price))
            
        snapshots = client.get_option_snapshot([opt.symbol for opt in valid[:10]]) 
        priced_candidates = build_delay_adjusted_contracts(client, valid[:10], snapshots=snapshots)
        priced_map = {contract.symbol: contract for contract in priced_candidates}
        
        for opt in valid[:10]:
            priced_contract = priced_map.get(opt.symbol)
            if priced_contract and effective_ask_price(priced_contract) > 0:
                # Calculate how many contracts we need to flatten the portfolio delta
                opt_delta = abs(priced_contract.delta or 0.50) * 100 # usually ~50 for ATM
                contracts_needed = math.ceil(abs(port_delta) / opt_delta)
                
                # Cap the hedge spend so we don't blow the account (2% limit)
                ask_price = effective_ask_price(priced_contract)
                max_contracts = math.floor((total_equity * 0.02) / (ask_price * 100))
                qty_to_buy = min(contracts_needed, max(1, max_contracts))
                
                if qty_to_buy > 0:
                    try:
                        client.market_buy(opt.symbol, qty=qty_to_buy)
                        logger.info(f"🛡️ DELTA NEUTRAL HEDGE DEPLOYED: Bought {qty_to_buy}x {opt.symbol} for ~${ask_price*100:.2f} each.")
                        
                        from core.notifications import send_alert
                        send_alert(f"🚨 **DELTA NEUTRAL HEDGE DEPLOYED**\nPortfolio Delta was {port_delta:+.2f}.\nBought {qty_to_buy}x {opt.symbol} to flatten exposure.", "ALERT")
                    except Exception as e:
                        if "market hours" in str(e).lower():
                            logger.warning(f"⏳ Market closed. Cannot deploy Delta-Neutral hedge {opt.symbol} yet.")
                        else:
                            logger.error(f"Failed to deploy Delta-Neutral hedge: {e}")
                    return
    except Exception as e:
        logger.error(f"Failed to search Delta-Neutral hedge: {e}")


def deploy_asymmetric_bets(client, symbols_list, total_equity, positions):
    import yfinance as yf
    import math
    from datetime import datetime, timezone, timedelta
    
    max_allocation = total_equity * 0.03
    max_price_per_leg = 0.30 
    
    active_bets_cost = sum(float(p.market_value) for p in positions if float(p.market_value) < 100 and float(p.qty) > 0)
    if active_bets_cost >= max_allocation: return
        
    budget_remaining = max_allocation - active_bets_cost
    if budget_remaining < 60.0: return
        
    lottery_targets = [s for s in ['TSLA', 'NVDA', 'SMH', 'AMD', 'COIN', 'MSTR'] if s in symbols_list]
    if not lottery_targets: return
        
    for sym in lottery_targets:
        if budget_remaining < 60.0: break
        try:
            ticker = yf.Ticker(sym)
            hist_1y = ticker.history(period="1y")
            if hist_1y.empty or len(hist_1y) < 60: continue
            
            rolling_vol = hist_1y['Close'].pct_change().rolling(window=20).std() * math.sqrt(252)
            vol_min, vol_max, current_vol = rolling_vol.min(), rolling_vol.max(), rolling_vol.iloc[-1]
            
            if vol_max > vol_min: vol_rank = ((current_vol - vol_min) / (vol_max - vol_min)) * 100
            else: continue
                
            if vol_rank > 15.0: continue

            calls = client.get_options_contracts([sym], 'call', min_days=90, max_days=180)
            puts = client.get_options_contracts([sym], 'put', min_days=90, max_days=180)
            if not calls or not puts: continue
            
            target_min = datetime.now(timezone.utc).date() + timedelta(days=90)
            target_max = datetime.now(timezone.utc).date() + timedelta(days=180)
            
            valid_calls = [o for o in calls if target_min < datetime.strptime(parse_occ_symbol(o.symbol)[1], "%y%m%d").date() < target_max]
            valid_puts = [o for o in puts if target_min < datetime.strptime(parse_occ_symbol(o.symbol)[1], "%y%m%d").date() < target_max]
            
            valid_calls.sort(key=lambda x: parse_occ_symbol(x.symbol)[3], reverse=True) 
            valid_puts.sort(key=lambda x: parse_occ_symbol(x.symbol)[3]) 
            
            call_snaps = client.get_option_snapshot([o.symbol for o in valid_calls[:30]])
            put_snaps = client.get_option_snapshot([o.symbol for o in valid_puts[:30]])
            priced_calls = build_delay_adjusted_contracts(client, valid_calls[:30], snapshots=call_snaps)
            priced_puts = build_delay_adjusted_contracts(client, valid_puts[:30], snapshots=put_snaps)
            priced_call_map = {contract.symbol: contract for contract in priced_calls}
            priced_put_map = {contract.symbol: contract for contract in priced_puts}
            
            best_call, best_put = None, None
            
            for opt in valid_calls[:30]:
                priced_contract = priced_call_map.get(opt.symbol)
                if priced_contract:
                    sb, sa = effective_bid_price(priced_contract), effective_ask_price(priced_contract)
                    if sa > 0 and sb > 0 and (sa - sb) <= 0.15: 
                        if 0.05 <= sa <= max_price_per_leg:
                            best_call = (opt.symbol, sa)
                            break
                            
            for opt in valid_puts[:30]:
                priced_contract = priced_put_map.get(opt.symbol)
                if priced_contract:
                    sb, sa = effective_bid_price(priced_contract), effective_ask_price(priced_contract)
                    if sa > 0 and sb > 0 and (sa - sb) <= 0.15: 
                        if 0.05 <= sa <= max_price_per_leg:
                            best_put = (opt.symbol, sa)
                            break
            
            if best_call or best_put:
                if best_call:
                    client.market_buy(best_call[0], qty=1)
                    budget_remaining -= (best_call[1] * 100)
                    # --- DISCORD ALERT ---
                    send_alert(f"🎲 **NEW TRADE: Cornwall Call**\n**Bought:** {best_call[0]}\n**Cost:** ~${best_call[1]*100:.2f}", "INFO")
                    
                if best_put:
                    client.market_buy(best_put[0], qty=1)
                    budget_remaining -= (best_put[1] * 100)
                    # --- DISCORD ALERT ---
                    send_alert(f"🎲 **NEW TRADE: Cornwall Put**\n**Bought:** {best_put[0]}\n**Cost:** ~${best_put[1]*100:.2f}", "INFO")
                
        except Exception as e:
            continue
