import os
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import logging
import warnings

# Import your custom network architecture
from core.neural_brain import StrategySelectorNet
from core.torch_device import resolve_torch_runtime

warnings.filterwarnings("ignore")
logger = logging.getLogger(f"strategy.{__name__}")

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'brain_dataset.pt')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'brain_weights.pth')

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_brain_prediction():
    """
    Downloads the last 60 days of data, passes it to the PyTorch MoE Brain, 
    and returns the highest probability strategy.
    """
    try:
        runtime = resolve_torch_runtime()
        device = runtime.device

        # 1. Load the Scaler from our training dataset
        dataset = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
        scaler = dataset['scaler']
        features_list = dataset['features_list']
        
        # 2. Download the last 6 months (We need extra days to calculate the 50 SMA)
        tickers = {'SPY': 'SPY', 'QQQ': 'QQQ', 'IWM': 'IWM', 'VIX': '^VIX', 'TNX': '^TNX', 'DXY': 'DX-Y.NYB'}
        raw_data = {}
        for name, ticker in tickers.items():
            df = yf.download(ticker, period="6mo", progress=False)
            raw_data[name] = df['Close'].squeeze()
            
        df = pd.DataFrame(raw_data).dropna()
        
        # 3. Engineer the exact same features we trained on
        features = pd.DataFrame(index=df.index)
        features['SPY_ret'] = np.log(df['SPY'] / df['SPY'].shift(1))
        features['QQQ_ret'] = np.log(df['QQQ'] / df['QQQ'].shift(1))
        features['IWM_ret'] = np.log(df['IWM'] / df['IWM'].shift(1))
        features['VIX_lvl'] = df['VIX']
        features['VIX_ret'] = np.log(df['VIX'] / df['VIX'].shift(1))
        features['TNX_lvl'] = df['TNX']
        features['DXY_ret'] = np.log(df['DXY'] / df['DXY'].shift(1))
        features['SPY_RSI'] = calculate_rsi(df['SPY'])
        features['VIX_RSI'] = calculate_rsi(df['VIX'])
        features['SPY_dist_20SMA'] = df['SPY'] / df['SPY'].rolling(20).mean() - 1
        features['SPY_dist_50SMA'] = df['SPY'] / df['SPY'].rolling(50).mean() - 1
        features = features.dropna()
        
        # 4. Extract the final 60-day rolling sequence
        last_60_days = features.iloc[-60:]
        if len(last_60_days) < 60:
            logger.error("Not enough data to form a 60-day sequence. Fallback to THETA.")
            return "THETA_ENGINE", 0.0, [1.0, 0.0, 0.0]
            
        # 5. Scale and format into a PyTorch Tensor
        X_raw = last_60_days[features_list].values
        X_scaled = scaler.transform(X_raw)
        X_tensor = torch.tensor(np.array([X_scaled]), dtype=torch.float32) # Add Batch dimension
        
        # 6. Load the Neural Network
        X_tensor = X_tensor.to(device)
        
        input_size = len(features_list)
        model = StrategySelectorNet(input_size=input_size, hidden_size=128, num_layers=2, num_classes=3).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval() # Set to evaluation mode (turns off dropout)
        
        # 7. Ask The Brain
        with torch.no_grad():
            raw_logits = model(X_tensor)
            # SQUASH the raw math scores into perfect 0-100% probabilities
            probabilities = torch.softmax(raw_logits, dim=1)[0].cpu().numpy()
            predicted_class = np.argmax(probabilities)
            
        # 8. Decode the prediction
        class_map = {
            0: "THETA_ENGINE", # Sell Premium
            1: "VEGA_SNIPER",  # Buy Straddles
            2: "TAIL_HEDGE"    # Market Crash Imminent
        }
        
        best_strategy = class_map[predicted_class]
        confidence = probabilities[predicted_class] * 100
        
        return best_strategy, confidence, probabilities
        
    except Exception as e:
        logger.error(f"🧠 Brain Inference Failed: {e}")
        return "THETA_ENGINE", 0.0, [1.0, 0.0, 0.0]
