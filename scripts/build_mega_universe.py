import os
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("mega_matrix")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
TENSOR_PATH = os.path.join(DATA_DIR, 'mega_universe_dataset.pt')

# --- THE UNIVERSE ---
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

MACRO_TICKERS = {
    'CL=F': 'Crude_Oil',
    'ZS=F': 'Soybeans',
    'GC=F': 'Gold',
    '^TNX': 'Treasury_10Y',
    'DX-Y.NYB': 'DXY_Dollar',
    '^VIX': 'VIX'
}

SEQ_LENGTH = 60
FORWARD_LOOK = 10  # Predict 10 days out (ideal for short options)

def get_data():
    logger.info("🌍 Downloading Global Macro Backdrop...")
    macro_data = yf.download(list(MACRO_TICKERS.keys()), period="10y", progress=False)['Close']
    macro_data.rename(columns=MACRO_TICKERS, inplace=True)
    
    # Calculate Macro Returns
    macro_returns = np.log(macro_data / macro_data.shift(1)).fillna(0)
    
    logger.info(f"📈 Downloading 10 years of data for {len(TICKERS)} Universe Stocks...")
    # Group download is much faster than individual loops
    stock_data = yf.download(TICKERS, period="10y", progress=True)['Close']
    
    return stock_data, macro_returns

def build():
    stock_data, macro_returns = get_data()
    
    all_sequences = []
    all_labels = []
    
    logger.info("🧠 Processing Tickers and generating Multi-Modal Tensors...")
    
    # Process each stock individually against the macro backdrop
    for ticker in TICKERS:
        if ticker not in stock_data.columns: continue
            
        df = pd.DataFrame({'Close': stock_data[ticker]}).dropna()
        if len(df) < SEQ_LENGTH + FORWARD_LOOK + 10:
            continue # Skip stocks that are too new (need enough data for sequences)
            
        # 1. Feature Engineering (Stock Specific)
        df['Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_20'] = df['Ret'].rolling(20).std() * np.sqrt(252)
        df['SMA_50_Dist'] = (df['Close'] / df['Close'].rolling(50).mean()) - 1
        
        # 2. Merge with Global Macro
        df = df.join(macro_returns, how='left').dropna()
        
        # 3. Label Generation (Forward Looking 10 Days)
        future_return = np.log(df['Close'].shift(-FORWARD_LOOK) / df['Close'])
        
        labels = []
        for val in future_return:
            if pd.isna(val):
                labels.append(np.nan)
            elif val > 0.04:
                labels.append(2) # Class 2: Delta Bull (Call Spreads / Puts Sold)
            elif val < -0.04:
                labels.append(3) # Class 3: Delta Bear (Put Spreads / Calls Sold)
            elif abs(val) > 0.07:
                labels.append(1) # Class 1: Vega Explode (Long Straddles) - Overrides direction
            else:
                labels.append(0) # Class 0: Theta Range (Iron Condors)
                
        df['Label'] = labels
        df = df.dropna()
        
        # 4. Extract sequences
        feature_cols = ['Ret', 'Vol_20', 'SMA_50_Dist'] + list(MACRO_TICKERS.values())
        X_raw = df[feature_cols].values
        y_raw = df['Label'].values
        
        for i in range(len(X_raw) - SEQ_LENGTH):
            all_sequences.append(X_raw[i : i + SEQ_LENGTH])
            all_labels.append(y_raw[i + SEQ_LENGTH - 1])

    logger.info("⚖️ Normalizing massive dataset...")
    
    # We have to flatten, scale, and reshape because there are millions of data points
    all_seq_arr = np.array(all_sequences)
    num_samples, seq_len, num_features = all_seq_arr.shape
    
    scaler = StandardScaler()
    X_flat = all_seq_arr.reshape(-1, num_features)
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(num_samples, seq_len, num_features)
    
    logger.info("💾 Packing Tensors for the RTX 5090...")
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(np.array(all_labels), dtype=torch.long)
    
    logger.info(f"✅ FINAL SHAPE: X={X_tensor.shape}, y={y_tensor.shape}")
    
    torch.save({
        'X': X_tensor,
        'y': y_tensor,
        'scaler': scaler,
        'features_list': feature_cols
    }, TENSOR_PATH)

if __name__ == "__main__":
    build()