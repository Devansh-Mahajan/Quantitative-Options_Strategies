import logging
from config.params import (
    DELTA_MIN,
    DELTA_MAX,
    YIELD_MAX,
    OPEN_INTEREST_MIN,
    SCORE_MIN,
    EXPIRATION_MIN,
    EXPIRATION_MAX,
    MAX_RELATIVE_SPREAD,
)
from core.sentiment import get_dynamic_yield
from datetime import datetime

logger = logging.getLogger(f"strategy.{__name__}")

def filter_underlying(client, symbols, buying_power_limit):
    resp = client.get_stock_latest_trade(symbols)
    minimum_spread_risk = 50.0
    affordable = []
    for symbol, trade in resp.items():
        last_price = float(getattr(trade, "price", 0.0) or 0.0)
        minimum_required = (last_price * 100.0) + minimum_spread_risk
        if buying_power_limit >= minimum_required:
            affordable.append(symbol)
    return affordable

def filter_options(options, min_strike=0, target_expiry=None):
    dynamic_yield = get_dynamic_yield() 
    filtered_contracts = []
    
    # NEW: Added 'missing_data' and 'strike' to the tracker
    reasons = {'missing_data': 0, 'dte': 0, 'delta': 0, 'oi': 0, 'strike': 0, 'yield': 0, 'spread': 0}
    
    target_occ_date = None
    if target_expiry:
        try:
            target_occ_date = datetime.strptime(target_expiry[:10], "%Y-%m-%d").strftime("%y%m%d")
        except Exception:
            pass

    for contract in options:
        # 1. THE SILENT KILLER: Track missing Alpaca data
        if contract.delta is None or contract.oi is None:
            reasons['missing_data'] += 1
            continue
            
        # 2. EXPIRATION CHECK
        if target_occ_date:
            if target_occ_date not in contract.symbol:
                reasons['dte'] += 1; continue
        else:
            if not (EXPIRATION_MIN <= contract.dte <= EXPIRATION_MAX):
                reasons['dte'] += 1; continue

        # 3. GREEKS & LIMITS
        if not (DELTA_MIN < abs(contract.delta) < DELTA_MAX):
            reasons['delta'] += 1; continue
            
        if contract.oi < OPEN_INTEREST_MIN:
            reasons['oi'] += 1; continue
            
        if contract.strike < min_strike:
            reasons['strike'] += 1; continue
            
        if not contract.ask_price or contract.ask_price <= 0 or contract.bid_price is None:
            reasons['spread'] += 1; continue
        rel_spread = (contract.ask_price - contract.bid_price) / contract.ask_price
        if rel_spread > (MAX_RELATIVE_SPREAD * 1.2):
            reasons['spread'] += 1; continue

        # 4. YIELD CHECK 
        current_yield = (contract.bid_price / contract.strike) * (365 / (contract.dte + 1))
        if not (dynamic_yield < current_yield < YIELD_MAX):
            reasons['yield'] += 1; continue
            
        filtered_contracts.append(contract)
        
    if options:
        # THE BATCH LABEL FIX: Tell the user it's a batch scan!
        logger.debug(f"🔍 [THETA X-RAY - BATCH SCAN]: Scanned {len(options)}. Rejected -> Missing Data: {reasons['missing_data']} | DTE: {reasons['dte']} | Delta: {reasons['delta']} | OI: {reasons['oi']} | Strike: {reasons['strike']} | Spread: {reasons['spread']} | Yield: {reasons['yield']}. Passed: {len(filtered_contracts)}")
        
    return filtered_contracts

def score_options(options):
    scores = []
    for p in options:
        if not p.bid_price or not p.ask_price or not p.strike or not p.dte:
            scores.append(0.0)
            continue

        annualized_yield = (p.bid_price / p.strike) * (365.0 / (p.dte + 1.0))
        spread_quality = max(0.0, 1.0 - ((p.ask_price - p.bid_price) / max(p.ask_price, 0.01)))
        liquidity_bonus = min(1.0, float(p.oi or 0) / (OPEN_INTEREST_MIN * 4.0))
        delta_safety = max(0.0, 1.0 - (abs(p.delta) / max(DELTA_MAX, 0.01)))
        duration_efficiency = 45.0 / (p.dte + 10.0)

        score = annualized_yield * (0.50 + 0.50 * spread_quality) * (0.70 + 0.30 * liquidity_bonus) * delta_safety * duration_efficiency
        scores.append(score)
    return scores

def select_options(options, scores, n=None):
    filtered = [(option, score) for option, score in zip(options, scores) if score > SCORE_MIN]
    best_per_underlying = {}
    for option, score in filtered:
        underlying = option.underlying
        if (underlying not in best_per_underlying) or (score > best_per_underlying[underlying][1]):
            best_per_underlying[underlying] = (option, score)
    sorted_best = sorted(best_per_underlying.values(), key=lambda x: x[1], reverse=True)
    return [option for option, _ in sorted_best[:n]] if n else [option for option, _ in sorted_best]
