import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# --- THE HOLY GRAIL MICRO ENGINE ---
def get_micro_regime(ticker_symbol):
    # 1. Get 2 years of data for a deeper statistical baseline
    data = yf.download(ticker_symbol, period="2y", progress=False)['Close']
    if len(data) < 252: return "UNKNOWN"
    
    # 2. FEATURE ENGINEERING: Focus on Volatility, not Price
    returns = np.log(data / data.shift(1)).dropna()
    # 5-day rolling vol (Standard Deviation) is the 'sticky' signal HMMs love
    vol_signal = returns.rolling(window=5).std() * np.sqrt(252)
    X = vol_signal.dropna().values.reshape(-1, 1)
    
    # 3. ROBUST HMM TRAINING
    # We use 2 states (QUIET vs. EXPLOSIVE) for maximum convergence stability
    model = GaussianHMM(
        n_components=2, 
        covariance_type="full", 
        n_iter=1000,       # Increased for convergence
        tol=0.01,          # Slightly looser tolerance to stop 'static'
        random_state=42
    )
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled)
    
    # 4. IDENTIFY THE 'EXPLOSIVE' STATE
    # The state with the higher mean volatility is the one we want to sell against
    explosive_state = np.argmax(model.means_.flatten())
    current_state = model.predict(X_scaled)[-1]
    
    return "EXPLOSIVE" if current_state == explosive_state else "QUIET"

# --- THE EXECUTION LOOP ---
def run_dual_key_scan(tickers):
    # Check the Macro Ocean first
    # (Assuming your Macro HMM already returned 'GOLDILOCKS')
    macro_state = "GOLDILOCKS" 
    
    print(f"🌊 MACRO OCEAN: {macro_state}")
    print("--------------------------------------------------")
    
    for ticker in tickers:
        micro = get_micro_regime(ticker)
        
        if macro_state == "GOLDILOCKS" and micro == "EXPLOSIVE":
            # 🎯 THE HOLY GRAIL: Calm Macro + Local Freakout = Maximum Theta Profit
            print(f"🎯 [SIGNAL] {ticker:<5} | State: {micro} | Action: SELL IRON CONDOR")
        elif micro == "EXPLOSIVE":
            print(f"⚠️ [WATCH]  {ticker:<5} | State: {micro} | Action: WAIT (Macro not aligned)")

if __name__ == "__main__":
    # Test on a few high-vol and low-vol names
    test_universe = ["NVDA", "TSLA", "AAPL", "TLT", "KO", "JPM"]
    run_dual_key_scan(test_universe)