import yfinance as yf
import math
import logging
from datetime import datetime, timezone, timedelta

# --- THE MUZZLE --- 
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

logger = logging.getLogger(f"strategy.{__name__}")

# ==========================================
# --- BLACK-SCHOLES PRICING ENGINE ---
# ==========================================
def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def calculate_bs_straddle(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0.0
        
    d1 = (math.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    call_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    put_price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    
    return call_price + put_price

def get_risk_free_rate():
    try:
        irx = yf.Ticker("^IRX")
        rate = irx.history(period="5d")['Close'].iloc[-1] / 100.0
        return rate if rate > 0 else 0.045 
    except:
        return 0.045 

def analyze_vega_potential(ticker_obj, sym, target_date, risk_free_rate):
    try:
        hist_1y = ticker_obj.history(period="1y")
        if hist_1y.empty or len(hist_1y) < 60: 
            return False
            
        current_price = float(hist_1y['Close'].iloc[-1])
        realized_vol_60d = hist_1y['Close'].iloc[-60:].pct_change().std() * math.sqrt(252)
        
        rolling_vol = hist_1y['Close'].pct_change().rolling(window=20).std() * math.sqrt(252)
        vol_min = rolling_vol.min()
        vol_max = rolling_vol.max()
        
        options = ticker_obj.options
        if not options: return False
            
        valid_exps = [exp for exp in options if datetime.strptime(exp, "%Y-%m-%d").date() > target_date]
        if not valid_exps: return False
            
        target_exp = valid_exps[0]
        chain = ticker_obj.option_chain(target_exp)
        
        calls = chain.calls
        puts = chain.puts
        if calls.empty or puts.empty: return False
            
        calls['strike_diff'] = (calls['strike'] - current_price).abs()
        atm_call = calls.sort_values('strike_diff').iloc[0]
        strike = atm_call['strike']
        
        atm_put_df = puts[puts['strike'] == strike]
        if atm_put_df.empty: return False
        atm_put = atm_put_df.iloc[0]
        
        current_iv = (atm_call['impliedVolatility'] + atm_put['impliedVolatility']) / 2
        
        if vol_max > vol_min:
            vol_rank = ((current_iv - vol_min) / (vol_max - vol_min)) * 100
        else:
            vol_rank = 100.0 
            
        c_price = (atm_call['bid'] + atm_call['ask']) / 2 if (atm_call['bid'] > 0) else atm_call['lastPrice']
        p_price = (atm_put['bid'] + atm_put['ask']) / 2 if (atm_put['bid'] > 0) else atm_put['lastPrice']
        market_straddle_cost = c_price + p_price

        if market_straddle_cost <= 0: return False

        days_to_exp = (datetime.strptime(target_exp, "%Y-%m-%d").date() - datetime.now(timezone.utc).date()).days
        T = max(days_to_exp, 1) / 365.0 
        
        bs_fair_value = calculate_bs_straddle(current_price, strike, T, risk_free_rate, realized_vol_60d)
        
        if bs_fair_value > 0:
            discount_ratio = market_straddle_cost / bs_fair_value
            logger.info(f"🔍 VEGA SCAN [{sym}]: BS Ratio={discount_ratio:.2f} (Needs <0.85) | VolRank={vol_rank:.1f}% (Needs <35%)")
            
            if discount_ratio < 0.85 and vol_rank < 35.0:
                logger.info(f"💎 VEGA ALERT: {sym} passes Two-Key Auth! Historically asleep AND Mathematically cheap. Sniping!")
                return True
                
        return False
    except Exception as e:
        logger.debug(f"⚠️ Vega calculation error for {sym}: {e}")
        return False

# ==========================================
# --- THE INTELLIGENCE ROUTER ---
# ==========================================
def analyze_market_regime(symbols, max_dte=45):
    theta_list = []
    vega_list = []
    crush_list = [] 
    
    now = datetime.now(timezone.utc).date()
    
    crush_window_end = now + timedelta(days=3)
    vega_buy_window_start = now + timedelta(days=30)
    vega_buy_window_end = now + timedelta(days=45)
    theta_cutoff = now + timedelta(days=max_dte)

    logger.info("--- STARTING MARKET INTELLIGENCE SCAN ---")
    rf_rate = get_risk_free_rate()

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            cal = ticker.calendar
            found_date = None
            
            if cal is not None:
                if isinstance(cal, dict) and 'Earnings Date' in cal and cal['Earnings Date']:
                    found_date = cal['Earnings Date'][0]
                elif hasattr(cal, 'empty') and not cal.empty and 'Earnings Date' in cal:
                    found_date = cal['Earnings Date'][0]

            if found_date:
                if hasattr(found_date, 'date'): found_date = found_date.date()
                elif isinstance(found_date, str): found_date = datetime.strptime(found_date[:10], "%Y-%m-%d").date()

                # --- 1. IV CRUSH CHECK (WITH DICTIONARY FIX) ---
                if now <= found_date <= crush_window_end:
                    hist_1y = ticker.history(period="1y")
                    if not hist_1y.empty and len(hist_1y) >= 60:
                        rolling_vol = hist_1y['Close'].pct_change().rolling(window=20).std() * math.sqrt(252)
                        vol_min, vol_max = rolling_vol.min(), rolling_vol.max()
                        
                        options = ticker.options
                        if options:
                            target_exp = [e for e in options if datetime.strptime(e, "%Y-%m-%d").date() > found_date][0]
                            chain = ticker.option_chain(target_exp)
                            current_iv = (chain.calls['impliedVolatility'].mean() + chain.puts['impliedVolatility'].mean()) / 2
                            vol_rank = ((current_iv - vol_min) / (vol_max - vol_min)) * 100 if vol_max > vol_min else 100
                            
                            if vol_rank > 85.0:
                                logger.info(f"🌋 IV CRUSH ALERT: {sym} VolRank is {vol_rank:.1f}%. Routing to Iron Condors.")
                                # THE FIX: Append a dictionary containing the exact expiration string
                                crush_list.append({'symbol': sym, 'expiry': target_exp, 'earnings_date': str(found_date)})
                                continue

                # --- 2. VEGA CHECK (WITH DICTIONARY FIX) ---
                if vega_buy_window_start < found_date < vega_buy_window_end:
                    if analyze_vega_potential(ticker, sym, found_date, rf_rate):
                        # Find the target exp to pass to the engine
                        options = ticker.options
                        target_exp = [e for e in options if datetime.strptime(e, "%Y-%m-%d").date() > found_date][0] if options else str(found_date)
                        # THE FIX: Append a dictionary containing the exact expiration string
                        vega_list.append({'symbol': sym, 'expiry': target_exp, 'earnings_date': str(found_date)})
                        continue 
                
                # --- 3. THETA BLOCK ---
                if now < found_date < theta_cutoff:
                    logger.info(f"🚫 BLOCKING {sym}: Earnings too close.")
                    continue
            
            logger.info(f"✅ Safe: {sym}") 
            theta_list.append(sym)

        except Exception as e:
            logger.info(f"✅ Safe (Scan Fallback): {sym}")
            theta_list.append(sym)
            
    return theta_list, vega_list, crush_list