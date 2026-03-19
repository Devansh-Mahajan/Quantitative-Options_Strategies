import os
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("data_pipeline")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
TENSOR_PATH = os.path.join(DATA_DIR, 'brain_dataset.pt')

# Hyperparameters for the LSTM sequence
SEQ_LENGTH = 60  # The Brain looks at the last 60 days of data to make 1 decision
FORWARD_LOOK = 5 # We want to predict what happens over the NEXT 5 days

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def build_dataset():
    logger.info("🌍 Downloading 15 Years of Institutional Market Data...")
    
    # 1. Fetch the raw data
    tickers = {
        'SPY': 'SPY',      # Broad Market
        'QQQ': 'QQQ',      # Tech/Momentum
        'IWM': 'IWM',      # Small Caps (Canary in the coal mine)
        'VIX': '^VIX',     # Volatility
        'TNX': '^TNX',     # 10-Year Treasury Yield (Interest Rates)
        'DXY': 'DX-Y.NYB'  # US Dollar Index
    }
    
    raw_data = {}
    for name, ticker in tickers.items():
        logger.info(f"   -> Fetching {name}...")
        df = yf.download(ticker, period="15y", progress=False)
        raw_data[name] = df['Close'].squeeze()
        
    # Align all data to the same dates and drop missing rows
    df = pd.DataFrame(raw_data).dropna()
    
    logger.info("⚙️ Engineering Deep Macro Features...")
    
    # 2. Feature Engineering (What the neural net actually learns from)
    features = pd.DataFrame(index=df.index)
    
    # Price Action (Log Returns)
    features['SPY_ret'] = np.log(df['SPY'] / df['SPY'].shift(1))
    features['QQQ_ret'] = np.log(df['QQQ'] / df['QQQ'].shift(1))
    features['IWM_ret'] = np.log(df['IWM'] / df['IWM'].shift(1))
    
    # Volatility & Macro
    features['VIX_lvl'] = df['VIX']
    features['VIX_ret'] = np.log(df['VIX'] / df['VIX'].shift(1))
    features['TNX_lvl'] = df['TNX']
    features['DXY_ret'] = np.log(df['DXY'] / df['DXY'].shift(1))
    
    # Technical Indicators
    features['SPY_RSI'] = calculate_rsi(df['SPY'])
    features['VIX_RSI'] = calculate_rsi(df['VIX'])
    
    # SPY Moving Average Distances
    features['SPY_dist_20SMA'] = df['SPY'] / df['SPY'].rolling(20).mean() - 1
    features['SPY_dist_50SMA'] = df['SPY'] / df['SPY'].rolling(50).mean() - 1

    # Drop the NaN rows created by rolling windows (like the 50 SMA)
    features = features.dropna()
    
    logger.info("🎯 Generating Forward-Looking Strategy Labels...")
    
    # 3. Label Generation (Teaching The Brain what strategy wins)
    # We look FORWARD by 5 days. 
    # Class 0: Theta (Market is flat or up, Volatility drops/stays flat)
    # Class 1: Vega (Market chops, Volatility explodes > 15%)
    # Class 2: Tail Hedge / Cash (Market drops > 3%)
    
    future_spy_ret = np.log(df['SPY'].shift(-FORWARD_LOOK) / df['SPY'])
    future_vix_ret = np.log(df['VIX'].shift(-FORWARD_LOOK) / df['VIX'])
    
    labels = []
    for i in range(len(features)):
        date = features.index[i]
        if date not in future_spy_ret.index or pd.isna(future_spy_ret[date]):
            labels.append(np.nan)
            continue
            
        f_spy = future_spy_ret[date]
        f_vix = future_vix_ret[date]
        
        if f_spy < -0.03: 
            labels.append(2)  # Market dump -> Tail Hedge
        elif f_vix > 0.15:
            labels.append(1)  # Volatility spike -> Vega Long Straddles
        else:
            labels.append(0)  # Standard environment -> Theta Engine
            
    features['Label'] = labels
    features = features.dropna() # Drop the last 5 days since we don't know the future yet
    
    logger.info("🔪 Slicing data into 60-day sequence tensors for the LSTM...")
    
    # 4. Convert to PyTorch Sequence Tensors [Batch, Sequence, Features]
    X_raw = features.drop(columns=['Label']).values
    y_raw = features['Label'].values
    
    # Normalize the data (Neural networks hate large numbers)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - SEQ_LENGTH):
        X_seq.append(X_scaled[i : i + SEQ_LENGTH])
        y_seq.append(y_raw[i + SEQ_LENGTH - 1]) # The label at the end of the 60 day window
        
    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_seq), dtype=torch.long)
    
    logger.info(f"✅ Final Tensor Shape: X={X_tensor.shape}, y={y_tensor.shape}")
    
    # 5. Save it to disk
    torch.save({
        'X': X_tensor,
        'y': y_tensor,
        'scaler': scaler,
        'features_list': features.drop(columns=['Label']).columns.tolist()
    }, TENSOR_PATH)
    
    logger.info(f"💾 PyTorch Dataset saved to {TENSOR_PATH}")

if __name__ == "__main__":
    build_dataset()