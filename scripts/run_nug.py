import logging
import time
from scripts.mega_screener2 import get_micro_regime # Our Micro-HMM Engine
from core.regime_detection import get_brain_prediction # Our Macro-HMM Engine
from core.broker_client import BrokerClient
from core.execution import sell_iron_condors, buy_straddles, buy_tail_hedge
from core.manager import calculate_dynamic_risk
from core.sentiment import get_vix_level

logger = logging.getLogger("HMM_HolyGrail")

# --- YOUR INSTITUTIONAL UNIVERSE ---
TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "HYG", "LQD", "EEM", "FXI", "UNG", "USO", 
    "XLF", "XLK", "XLV", "XLE", "XLI", "XLP", "XLU", "XLY", "XBI", "KRE", "SMH", "GDX", "AAPL", 
    "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "QCOM", "TXN", "INTC", "MU", 
    "AMAT", "LRCX", "KLAC", "ASML", "TSM", "CRM", "ADBE", "ORCL", "IBM", "CSCO", "NOW", "PLTR", 
    "SNOW", "CRWD", "PANW", "FTNT", "TEAM", "DDOG", "MDB", "NET", "ZS", "OKTA", "SHOP", "UBER", 
    "ABNB", "DASH", "JPM", "BAC", "C", "WFC", "GS", "MS", "V", "MA", "AXP", "PYPL", "SQ", "SOFI", 
    "HOOD", "COIN", "BLK", "SCHW", "CME", "ICE", "NDAQ", "SPGI", "DFS", "SYF", "COF", "USB", 
    "PNC", "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE", "AMGN", "GILD", "VRTX", "REGN", "ISRG", 
    "MDT", "SYK", "BSX", "EW", "ZTS", "CVS", "CI", "ELV", "HUM", "BMY", "DXCM", "HD", "LOW", 
    "MCD", "SBUX", "NKE", "LULU", "CMG", "DPZ", "WMT", "TGT", "COST", "TJX", "ROST", "AZO", 
    "ORLY", "TSCO", "BBY", "BKNG", "EXPE", "MAR", "HLT", "DIS", "NFLX", "SPOT", "ROKU", "PG", 
    "KO", "PEP", "PM", "MO", "CL", "KMB", "STZ", "MDLZ", "HSY", "GIS", "K", "SYY", "BA", "CAT", 
    "DE", "GE", "HON", "MMM", "UPS", "FDX", "LMT", "NOC", "GD", "RTX", "WM", "RSG", "UNP", 
    "CSX", "NSC", "ETN", "PH", "ITW", "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "MPC", "PSX", 
    "VLO", "OXY", "DVN", "FCX", "NEM", "NUE", "DOW", "APD", "LIN", "SHW", "CTVA", "VZ", "T", 
    "TMUS", "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PEG", "AWK", "AMT", "PLD", "CCI", 
    "EQIX", "PSA", "O", "SPG", "WY", "BABA", "PDD", "JD", "BIDU", "MELI", "SE", "U", "RBLX"
]

def main():
    # 1. INIT BROKER
    client = BrokerClient(api_key="YOUR_KEY", secret_key="YOUR_SECRET", paper=True)
    
    # 2. OCEAN CHECK: Macro HMM Regime
    macro_regime, confidence, _ = get_brain_prediction()
    logger.info(f"🌊 MACRO OCEAN: {macro_regime} ({confidence:.2f}% Confidence)")
    
    # 3. THE INTERSECTION HUNT
    theta_targets = []
    vega_targets = []
    
    logger.info(f"🔎 Scanning Institutional Universe ({len(TICKERS)} tickers)...")
    
    for ticker in TICKERS:
        try:
            # We use the Micro-HMM to find locally "EXPLOSIVE" states
            micro_state = get_micro_regime(ticker)
            
            # THE DUAL-KEY LOGIC
            # Intersection A: Safe Macro + Wild Local = Selling Iron Condors
            if macro_regime == "THETA_ENGINE" and micro_state == "EXPLOSIVE":
                logger.info(f"🎯 [SIGNAL] {ticker}: Local Freakout in a Calm Ocean.")
                theta_targets.append(ticker)
            
            # Intersection B: Panic Macro + Wild Local = Buying Straddles
            elif macro_regime == "VEGA_SNIPER" and micro_state == "EXPLOSIVE":
                logger.info(f"⚡ [SIGNAL] {ticker}: Volatility Cluster Syncing.")
                vega_targets.append(ticker)
                
        except Exception as e:
            continue # Skip noisy data tickers

    # 4. EXECUTION
    buying_power = float(client.trade_client.get_account().buying_power)
    vix = get_vix_level()
    dynamic_risk = calculate_dynamic_risk(vix)

    if macro_regime == "THETA_ENGINE" and theta_targets:
        # Sort or pick top 5 targets by liquidity/preference
        logger.info(f"🚀 Deploying HMM-Aligned Theta Harvest on: {theta_targets[:5]}")
        sell_iron_condors(client, theta_targets[:5], buying_power, dynamic_risk)
        
    elif macro_regime == "VEGA_SNIPER" and vega_targets:
        logger.info(f"🎯 Deploying HMM-Aligned Vega Snipes on: {vega_targets[:3]}")
        buy_straddles(client, vega_targets[:3], buying_power, dynamic_risk)
        
    elif macro_regime == "TAIL_HEDGE":
        logger.warning("🚨 HMM MACRO ALERT: Liquidity Crunch Detected. Halting new trades and hedging.")
        # Trigger your existing tail hedge logic here

if __name__ == "__main__":
    main()