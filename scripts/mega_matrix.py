import os
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import logging
import joblib
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("mega_matrix_v2")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'config')
os.makedirs(DATA_DIR, exist_ok=True)
TENSOR_PATH = os.path.join(DATA_DIR, 'mega_universe_dataset.pt')
HMM_MODEL_PATH = os.path.join(CONFIG_DIR, 'hmm_macro_model.pkl')

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

MACRO_TICKERS = {'CL=F': 'Crude_Oil', 'ZS=F': 'Soybeans', 'GC=F': 'Gold', '^TNX': 'Treasury_10Y', 'DX-Y.NYB': 'DXY_Dollar', '^VIX': 'VIX'}

SEQ_LENGTH = 60
FORWARD_LOOK = 10 

def get_hmm_probabilities():
    """Loads the pre-trained HMM and generates historical daily probabilities for the 4 regimes."""
    logger.info("🧠 Loading Hidden Markov Model & recreating historical regimes...")
    
    if not os.path.exists(HMM_MODEL_PATH):
        raise FileNotFoundError(f"Missing {HMM_MODEL_PATH}. Run train_hmm.py first!")
        
    hmm_data = joblib.load(HMM_MODEL_PATH)
    hmm_model = hmm_data['model']
    scaler = hmm_data['scaler']
    features_list = hmm_data['features_list']
    
    # 1. Fetch exactly what the HMM was trained on
    tickers = {'SPY': 'SPY', '^VIX': 'VIX', '^TNX': 'TNX', 'DX-Y.NYB': 'DXY', 'HYG': 'HYG', 'LQD': 'LQD'}
    raw = yf.download(list(tickers.keys()), period="10y", progress=False)['Close']
    raw.rename(columns=tickers, inplace=True)
    raw = raw.dropna()

    # 2. Replicate the HMM features identically
    features = pd.DataFrame(index=raw.index)
    features['SPY_ret'] = np.log(raw['SPY'] / raw['SPY'].shift(1))
    features['TNX_ret'] = np.log(raw['TNX'] / raw['TNX'].shift(1))
    features['DXY_ret'] = np.log(raw['DXY'] / raw['DXY'].shift(1))
    features['VIX_lvl'] = raw['VIX']
    features['Credit_Risk_Ratio'] = raw['HYG'] / raw['LQD']
    features['Credit_Risk_Momentum'] = np.log(features['Credit_Risk_Ratio'] / features['Credit_Risk_Ratio'].shift(5))
    features = features.dropna()
    features = features[features_list] # Ensure exact column order

    # 3. Generate the Probabilities
    X_scaled = scaler.transform(features.values)
    probs = hmm_model.predict_proba(X_scaled) # Shape: [n_days, 4]
    
    # Return as a DataFrame indexed by Date
    prob_df = pd.DataFrame(probs, index=features.index, columns=[f'HMM_State_{i}' for i in range(4)])
    return prob_df

def get_data():
    logger.info("🌍 Downloading Raw Commodities & Global Macro Backdrop...")
    macro_data = yf.download(list(MACRO_TICKERS.keys()), period="10y", progress=False)['Close']
    macro_data.rename(columns=MACRO_TICKERS, inplace=True)
    macro_returns = np.log(macro_data / macro_data.shift(1)).fillna(0)
    
    logger.info(f"📈 Downloading 10 years of data for {len(TICKERS)} Universe Stocks...")
    stock_data = yf.download(TICKERS, period="10y", progress=True)['Close']
    
    return stock_data, macro_returns

def build():
    # 1. Get the Hidden State Probabilities (The Edge)
    hmm_probs = get_hmm_probabilities()
    
    # 2. Get standard market data
    stock_data, macro_returns = get_data()
    
    all_sequences = []
    all_labels = []
    all_future_returns = []
    
    logger.info("🧬 Fusing HMM Probabilities with Individual Tickers...")
    
    for ticker in TICKERS:
        if ticker not in stock_data.columns: continue
            
        df = pd.DataFrame({'Close': stock_data[ticker]}).dropna()
        if len(df) < SEQ_LENGTH + FORWARD_LOOK + 10:
            continue 
            
        # Feature Engineering (Stock Specific)
        df['Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_20'] = df['Ret'].rolling(20).std() * np.sqrt(252)
        df['SMA_50_Dist'] = (df['Close'] / df['Close'].rolling(50).mean()) - 1

        # Merge Commodities/Rates
        df = df.join(macro_returns, how='left')
        # MERGE HMM PROBABILITIES (This is the magic)
        df = df.join(hmm_probs, how='left').dropna()

        # Hidden-correlation features between stock and macro flows
        correlation_windows = [10, 20]
        for macro_col in MACRO_TICKERS.values():
            for window in correlation_windows:
                feature_name = f"Corr_{macro_col}_{window}"
                df[feature_name] = df['Ret'].rolling(window).corr(df[macro_col]).clip(-1, 1)

        # Label Generation (Forward Looking 10 Days)
        future_return = np.log(df['Close'].shift(-FORWARD_LOOK) / df['Close'])

        labels = []
        for val in future_return:
            if pd.isna(val):
                labels.append(np.nan)
            elif abs(val) > 0.07:
                labels.append(1)  # Vega: large directional move either way
            elif val > 0.03:
                labels.append(2)  # Bull
            elif val < -0.03:
                labels.append(3)  # Bear
            else:
                labels.append(0)  # Theta

        df['Label'] = labels
        df['Future_Return_10d'] = future_return
        df = df.dropna()

        # Define the dynamic feature columns so PyTorch knows how wide the tensor is
        corr_cols = [
            f"Corr_{macro_col}_{window}"
            for macro_col in MACRO_TICKERS.values()
            for window in correlation_windows
        ]
        feature_cols = ['Ret', 'Vol_20', 'SMA_50_Dist'] + list(MACRO_TICKERS.values()) + [f'HMM_State_{i}' for i in range(4)] + corr_cols
        
        X_raw = df[feature_cols].values
        y_raw = df['Label'].values
        
        for i in range(len(X_raw) - SEQ_LENGTH):
            all_sequences.append(X_raw[i : i + SEQ_LENGTH])
            label_idx = i + SEQ_LENGTH - 1
            all_labels.append(y_raw[label_idx])
            all_future_returns.append(df['Future_Return_10d'].iloc[label_idx])

    logger.info("⚖️ Normalizing massive dataset...")
    
    all_seq_arr = np.array(all_sequences)
    num_samples, seq_len, num_features = all_seq_arr.shape
    
    scaler = StandardScaler()
    X_flat = all_seq_arr.reshape(-1, num_features)
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(num_samples, seq_len, num_features)
    
    logger.info("💾 Packing HMM-Infused Tensors for the RTX 5090...")
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(np.array(all_labels), dtype=torch.long)
    r_tensor = torch.tensor(np.array(all_future_returns), dtype=torch.float32)
    
    logger.info(f"✅ FINAL SHAPE: X={X_tensor.shape}, y={y_tensor.shape}")
    
    torch.save({
        'X': X_tensor,
        'y': y_tensor,
        'scaler': scaler,
        'features_list': feature_cols,
        'future_returns': r_tensor,
        'target_annual_return': 0.05,
    }, TENSOR_PATH)

if __name__ == "__main__":
    build()