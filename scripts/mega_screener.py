import os
import sys
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import logging
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.mega_neural_brain import MegaStrategyNet

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("mega_screener")

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'mega_universe_dataset.pt')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'mega_brain_weights.pth')

TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "HYG", "LQD", "EEM", "FXI", "UNG", "USO", 
    "XLF", "XLK", "XLV", "XLE", "XLI", "XLP", "XLU", "XLY", "XBI", "KRE", "SMH", "GDX", "AAPL", 
    "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "QCOM", "TXN", "INTC", "MU", 
    "AMAT", "LRCX", "KLAC", "ASML", "TSM", "CRM", "ADBE", "ORCL", "IBM", "CSCO", "NOW", "PLTR", 
    "SNOW", "CRWD", "PANW", "FTNT", "TEAM", "DDOG", "MDB", "NET", "ZS", "OKTA", "SHOP", "UBER", 
    "ABNB", "DASH", "JPM", "BAC", "C", "WFC", "GS", "MS", "V", "MA", "AXP", "PYPL", "SOFI", 
    "HOOD", "COIN", "BLK", "SCHW", "CME", "ICE", "NDAQ", "SPGI", "SYF", "COF", "USB", 
    "PNC", "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE", "AMGN", "GILD", "VRTX", "REGN", "ISRG", 
    "MDT", "SYK", "BSX", "EW", "ZTS", "CVS", "CI", "ELV", "HUM", "BMY", "DXCM", "HD", "LOW", 
    "MCD", "SBUX", "NKE", "LULU", "CMG", "DPZ", "WMT", "TGT", "COST", "TJX", "ROST", "AZO", 
    "ORLY", "TSCO", "BBY", "BKNG", "EXPE", "MAR", "HLT", "DIS", "NFLX", "SPOT", "ROKU", "PG", 
    "KO", "PEP", "PM", "MO", "CL", "KMB", "STZ", "MDLZ", "HSY", "GIS", "SYY", "BA", "CAT", 
    "DE", "GE", "HON", "MMM", "UPS", "FDX", "LMT", "NOC", "GD", "RTX", "WM", "RSG", "UNP", 
    "CSX", "NSC", "ETN", "PH", "ITW", "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "MPC", "PSX", 
    "VLO", "OXY", "DVN", "FCX", "NEM", "NUE", "DOW", "APD", "LIN", "SHW", "CTVA", "VZ", "T", 
    "TMUS", "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PEG", "AWK", "AMT", "PLD", "CCI", 
    "EQIX", "PSA", "O", "SPG", "WY", "BABA", "PDD", "JD", "BIDU", "MELI", "SE", "U", "RBLX"
]

MACRO_TICKERS = {'CL=F': 'Crude_Oil', 'ZS=F': 'Soybeans', 'GC=F': 'Gold', '^TNX': 'Treasury_10Y', 'DX-Y.NYB': 'DXY_Dollar', '^VIX': 'VIX'}

def scan_market():
    logger.info("🔌 Booting RTX 5090 Mega Inference Engine...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load(DATA_PATH, weights_only=False)
    scaler = dataset['scaler']
    feature_cols = dataset['features_list']
    
    model = MegaStrategyNet(input_size=len(feature_cols), hidden_size=256, num_layers=3, num_classes=4).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    logger.info("📥 Downloading live data for 150 tickers + Global Macro...")
    macro_data = yf.download(list(MACRO_TICKERS.keys()), period="6mo", progress=False)['Close']
    macro_data.rename(columns=MACRO_TICKERS, inplace=True)
    macro_returns = np.log(macro_data / macro_data.shift(1)).fillna(0)
    
    stock_data = yf.download(TICKERS, period="6mo", progress=False)['Close']
    
    valid_sequences = []
    valid_tickers = []
    
    for ticker in TICKERS:
        if ticker not in stock_data.columns: continue
        df = pd.DataFrame({'Close': stock_data[ticker]}).dropna()
        if len(df) < 60: continue
            
        df['Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_20'] = df['Ret'].rolling(20).std() * np.sqrt(252)
        df['SMA_50_Dist'] = (df['Close'] / df['Close'].rolling(50).mean()) - 1
        df = df.join(macro_returns, how='left').dropna()
        
        last_60 = df[feature_cols].iloc[-60:]
        if len(last_60) == 60:
            valid_sequences.append(last_60.values)
            valid_tickers.append(ticker)
            
    if not valid_sequences:
        logger.error("Failed to build sequences. Check data feeds.")
        return
        
    logger.info("🧠 Feeding live market data through the Mega Brain...")
    
    # Scale and Tensorize
    all_seq_arr = np.array(valid_sequences)
    num_samples, seq_len, num_features = all_seq_arr.shape
    X_flat = all_seq_arr.reshape(-1, num_features)
    X_scaled_flat = scaler.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(num_samples, seq_len, num_features)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    # Inference
    with torch.no_grad():
        raw_logits = model(X_tensor)
        probabilities = torch.softmax(raw_logits, dim=1).cpu().numpy()
        predictions = np.argmax(probabilities, axis=1)

    # Compile Results
    results = {'THETA (Iron Condors)': [], 'VEGA (Straddles)': [], 'BULL (Put Spreads)': [], 'BEAR (Call Spreads)': []}
    class_map = {0: 'THETA (Iron Condors)', 1: 'VEGA (Straddles)', 2: 'BULL (Put Spreads)', 3: 'BEAR (Call Spreads)'}
    
    for i in range(len(valid_tickers)):
        strat = class_map[predictions[i]]
        conf = probabilities[i][predictions[i]] * 100
        results[strat].append((valid_tickers[i], conf))
        
    print("\n" + "="*60)
    print(" 🎯 MEGA BRAIN DAILY OPTIONS SCREENER 🎯")
    print("="*60)
    
    for strat, plays in results.items():
        # Sort by highest confidence first
        plays.sort(key=lambda x: x[1], reverse=True)
        
        # Display the top 5 highest confidence plays for each strategy
        print(f"\n🔮 {strat}")
        print("-" * 30)
        if not plays:
            print("   No high-probability setups found today.")
        for ticker, conf in plays[:5]:
            print(f"   ► {ticker:<6} | Confidence: {conf:.2f}%")
            
    print("\n" + "="*60)
def get_mega_brain_targets(confidence_threshold=75.0):
    """API for the trading bot to get the AI's highest conviction setups."""
    logger.info("🧠 Booting RTX 5090 Mega Brain for Automated Execution...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load(DATA_PATH, weights_only=False)
    scaler = dataset['scaler']
    feature_cols = dataset['features_list']
    
    model = MegaStrategyNet(input_size=len(feature_cols), hidden_size=256, num_layers=3, num_classes=4).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Download Data
    macro_data = yf.download(list(MACRO_TICKERS.keys()), period="6mo", progress=False)['Close']
    macro_data.rename(columns=MACRO_TICKERS, inplace=True)
    macro_returns = np.log(macro_data / macro_data.shift(1)).fillna(0)
    
    stock_data = yf.download(TICKERS, period="6mo", progress=False)['Close']
    
    valid_sequences = []
    valid_tickers = []
    
    for ticker in TICKERS:
        if ticker not in stock_data.columns: continue
        df = pd.DataFrame({'Close': stock_data[ticker]}).dropna()
        if len(df) < 60: continue
            
        df['Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_20'] = df['Ret'].rolling(20).std() * np.sqrt(252)
        df['SMA_50_Dist'] = (df['Close'] / df['Close'].rolling(50).mean()) - 1
        df = df.join(macro_returns, how='left').dropna()
        
        last_60 = df[feature_cols].iloc[-60:]
        if len(last_60) == 60:
            valid_sequences.append(last_60.values)
            valid_tickers.append(ticker)
            
    if not valid_sequences:
        logger.error("Failed to build sequences. Check data feeds.")
        return {'THETA': [], 'VEGA': [], 'BULL': [], 'BEAR': []}
        
    # Scale and Tensorize
    all_seq_arr = np.array(valid_sequences)
    num_samples, seq_len, num_features = all_seq_arr.shape
    X_flat = all_seq_arr.reshape(-1, num_features)
    X_scaled_flat = scaler.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(num_samples, seq_len, num_features)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    # Inference
    with torch.no_grad():
        raw_logits = model(X_tensor)
        probabilities = torch.softmax(raw_logits, dim=1).cpu().numpy()
        predictions = np.argmax(probabilities, axis=1)

    # Compile Results for the Bot
    results = {'THETA': [], 'VEGA': [], 'BULL': [], 'BEAR': []}
    class_map = {0: 'THETA', 1: 'VEGA', 2: 'BULL', 3: 'BEAR'}
    
    for i in range(len(valid_tickers)):
        strat = class_map[predictions[i]]
        conf = probabilities[i][predictions[i]] * 100
        
        if conf >= confidence_threshold:
            results[strat].append((valid_tickers[i], conf))
            
    # Sort by highest confidence and strip the percentages out (bot only needs tickers)
    for strat in results:
        results[strat].sort(key=lambda x: x[1], reverse=True)
        results[strat] = [x[0] for x in results[strat]]
        
    return results

if __name__ == "__main__":
    scan_market()