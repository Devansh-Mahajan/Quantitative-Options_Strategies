import json
import os
import logging
from .utils import parse_option_symbol
from alpaca.trading.enums import AssetClass
from config.params import SWEEP_TICKER  # <-- NEW: Import the sweep ticker

logger = logging.getLogger(f"strategy.{__name__}")

# --- METADATA TRACKER FOR EARNINGS DATES ---
STRADDLE_META_FILE = "straddles_meta.json"

MODEL_STATE_FILE = "config/model_state.json"


def _safe_read_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def register_model_snapshot(model_name, metadata):
    """Persist model metadata for audit + weekend recalibration state."""
    data = _safe_read_json(MODEL_STATE_FILE)
    data[model_name] = metadata
    os.makedirs(os.path.dirname(MODEL_STATE_FILE), exist_ok=True)
    with open(MODEL_STATE_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def get_model_snapshot(model_name):
    return _safe_read_json(MODEL_STATE_FILE).get(model_name)


def get_all_model_snapshots():
    return _safe_read_json(MODEL_STATE_FILE)


def register_straddle(symbol, call_sym, put_sym, earnings_date):
    """Saves the straddle's earnings date so manager.py knows when to exit."""
    data = {}
    if os.path.exists(STRADDLE_META_FILE):
        with open(STRADDLE_META_FILE, 'r') as f:
            data = json.load(f)
            
    data[symbol] = {
        "call_symbol": call_sym,
        "put_symbol": put_sym,
        "earnings_date": str(earnings_date)
    }
    
    with open(STRADDLE_META_FILE, 'w') as f:
        json.dump(data, f)

def get_straddle_metadata(symbol):
    """Fetches the saved earnings date for a given symbol."""
    if os.path.exists(STRADDLE_META_FILE):
        with open(STRADDLE_META_FILE, 'r') as f:
            data = json.load(f)
            return data.get(symbol)
    return None

def remove_straddle_metadata(symbol):
    """Cleans up the file once the straddle is sold."""
    if os.path.exists(STRADDLE_META_FILE):
        with open(STRADDLE_META_FILE, 'r') as f:
            data = json.load(f)
        if symbol in data:
            del data[symbol]
            with open(STRADDLE_META_FILE, 'w') as f:
                json.dump(data, f)

# --- UPDATED RISK CALCULATOR ---
def calculate_risk(positions):
    risk = 0
    options_by_underlying = {}
    
    for p in positions:
        if p.asset_class == AssetClass.US_EQUITY:
            # --- THE FIX: Ignore our cash-sweep ETF ---
            if p.symbol == SWEEP_TICKER:
                continue 
                
            risk += float(p.avg_entry_price) * abs(int(p.qty))
            
        elif p.asset_class == AssetClass.US_OPTION:
            underlying, option_type, strike = parse_option_symbol(p.symbol)
            qty = int(p.qty)
            
            if underlying not in options_by_underlying:
                options_by_underlying[underlying] = {'shorts': [], 'longs': []}
                
            if qty < 0:
                options_by_underlying[underlying]['shorts'].append({'strike': strike, 'qty': abs(qty), 'type': option_type})
            elif qty > 0:
                options_by_underlying[underlying]['longs'].append({
                    'strike': strike, 'qty': qty, 'type': option_type, 'cost': float(p.avg_entry_price)
                })

    for underlying, legs in options_by_underlying.items():
        shorts = sorted(legs['shorts'], key=lambda x: x['strike'])
        longs = sorted(legs['longs'], key=lambda x: x['strike'])
        
        # 1. Calculate Risk for Short Spreads
        for short_leg in shorts:
            if short_leg['type'] == 'P':
                matching_long = next((l for l in longs if l['type'] == 'P' and l['strike'] < short_leg['strike']), None)
                if matching_long:
                    spread_width = round(short_leg['strike'] - matching_long['strike'], 2)
                    risk += (spread_width * 100) * short_leg['qty']
                    longs.remove(matching_long)
                else:
                    risk += (short_leg['strike'] * 100) * short_leg['qty']
                    
            elif short_leg['type'] == 'C':
                matching_long = next((l for l in longs if l['type'] == 'C' and l['strike'] > short_leg['strike']), None)
                if matching_long:
                    spread_width = round(matching_long['strike'] - short_leg['strike'], 2)
                    risk += (spread_width * 100) * short_leg['qty']
                    longs.remove(matching_long)
                else:
                    pass 
        
        # 2. Calculate Risk for leftover pure Longs (Straddles/Hedges)
        for long_leg in longs:
            risk += (long_leg['cost'] * 100) * long_leg['qty']
                    
    return risk

# --- UPDATED STATE MACHINE ---
def update_state(all_positions):    
    state = {}

    for p in all_positions:
        if p.asset_class == AssetClass.US_EQUITY:
            # --- THE FIX: Ignore our cash-sweep ETF ---
            if p.symbol == SWEEP_TICKER:
                continue 
                
            if int(p.qty) <= 0:
                logger.debug(f"Skipping short stock: {p.symbol}")
                continue

            underlying = p.symbol
            if underlying in state:
                state[underlying]["type"] = "complex_spread"
            else:
                state[underlying] = {"type": "long_shares", "price": float(p.avg_entry_price), "qty": int(p.qty)}

        elif p.asset_class == AssetClass.US_OPTION:
            underlying, option_type, _ = parse_option_symbol(p.symbol)

            # --- LONG OPTION LOGIC ---
            if int(p.qty) > 0:
                if underlying in state:
                    if state[underlying]["type"] == f"long_{'P' if option_type == 'C' else 'C'}":
                        state[underlying]["type"] = "long_straddle"
                    else:
                        state[underlying]["type"] = "complex_spread"
                else:
                    state[underlying] = {"type": f"long_{option_type}", "price": float(p.avg_entry_price)}
                continue 

            # --- SHORT OPTION LOGIC ---
            if underlying in state:
                state[underlying]["type"] = "complex_spread"
            else:
                if option_type == "C":
                    state[underlying] = {"type": "short_call_awaiting_stock", "price": None}
                elif option_type == "P":
                    state[underlying] = {"type": "short_put", "price": None}

    return state