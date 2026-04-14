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
    OPTION_DELAY_MAX_UNDERLYING_MOVE_PCT,
    OPTION_DELAY_MIN_PRICING_CONFIDENCE,
    TCA_SPREAD_WEIGHT,
    TCA_SLIPPAGE_WEIGHT,
    TCA_LIQUIDITY_WEIGHT,
)
from core.delay_aware_options import effective_ask_price, effective_bid_price
from core.execution_quality import estimate_option_transaction_cost, execution_quality_multiplier
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
    reasons = {'missing_data': 0, 'dte': 0, 'delta': 0, 'oi': 0, 'strike': 0, 'yield': 0, 'spread': 0, 'pricing': 0}
    
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

        if (
            contract.pricing_confidence is not None
            and contract.pricing_confidence < OPTION_DELAY_MIN_PRICING_CONFIDENCE
        ):
            reasons['pricing'] += 1; continue

        if (
            contract.staleness_pct is not None
            and contract.staleness_pct > OPTION_DELAY_MAX_UNDERLYING_MOVE_PCT
        ):
            reasons['pricing'] += 1; continue

        bid_price = effective_bid_price(contract)
        ask_price = effective_ask_price(contract)
        if not ask_price or ask_price <= 0 or bid_price < 0:
            reasons['spread'] += 1; continue
        rel_spread = (ask_price - bid_price) / ask_price
        if rel_spread > (MAX_RELATIVE_SPREAD * 1.2):
            reasons['spread'] += 1; continue

        # 4. YIELD CHECK 
        current_yield = (bid_price / contract.strike) * (365 / (contract.dte + 1))
        if not (dynamic_yield < current_yield < YIELD_MAX):
            reasons['yield'] += 1; continue
            
        filtered_contracts.append(contract)
        
    if options:
        # THE BATCH LABEL FIX: Tell the user it's a batch scan!
        logger.debug(f"🔍 [THETA X-RAY - BATCH SCAN]: Scanned {len(options)}. Rejected -> Missing Data: {reasons['missing_data']} | DTE: {reasons['dte']} | Delta: {reasons['delta']} | OI: {reasons['oi']} | Strike: {reasons['strike']} | Spread: {reasons['spread']} | Yield: {reasons['yield']} | Delay Pricing: {reasons['pricing']}. Passed: {len(filtered_contracts)}")
        
    return filtered_contracts

def score_options(options):
    scores = []
    for p in options:
        bid_price = effective_bid_price(p)
        ask_price = effective_ask_price(p)
        if not bid_price or not ask_price or not p.strike or not p.dte:
            scores.append(0.0)
            continue

        annualized_yield = (bid_price / p.strike) * (365.0 / (p.dte + 1.0))
        spread_quality = max(0.0, 1.0 - ((ask_price - bid_price) / max(ask_price, 0.01)))
        liquidity_bonus = min(1.0, float(p.oi or 0) / (OPEN_INTEREST_MIN * 4.0))
        delta_safety = max(0.0, 1.0 - (abs(p.delta) / max(DELTA_MAX, 0.01)))
        duration_efficiency = 45.0 / (p.dte + 10.0)
        expected_cost_ratio = estimate_option_transaction_cost(
            bid_price=bid_price,
            ask_price=ask_price,
            open_interest=p.oi,
            notional=float(p.strike) * 100.0,
            spread_weight=TCA_SPREAD_WEIGHT,
            slippage_weight=TCA_SLIPPAGE_WEIGHT,
            liquidity_weight=TCA_LIQUIDITY_WEIGHT,
        )
        exec_quality = execution_quality_multiplier(expected_cost_ratio)
        pricing_confidence = float(p.pricing_confidence if p.pricing_confidence is not None else 1.0)
        staleness_pct = float(p.staleness_pct or 0.0)
        staleness_quality = max(0.20, 1.0 - (staleness_pct / max(OPTION_DELAY_MAX_UNDERLYING_MOVE_PCT, 0.01)) * 0.75)

        score = annualized_yield * (0.50 + 0.50 * spread_quality) * (0.70 + 0.30 * liquidity_bonus) * delta_safety * duration_efficiency * exec_quality * (0.50 + 0.50 * pricing_confidence) * staleness_quality
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
