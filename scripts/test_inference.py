if __name__ != "__main__":
    import pytest

    pytest.importorskip("torch")

import joblib
import numpy as np
import os
import pandas as pd
import torch
import yfinance as yf

from core.mega_neural_brain import MegaStrategyNet
from core.torch_device import resolve_torch_runtime

# Path Setup
MODEL_PATH = "config/trading_model.pth"
HMM_PATH = "config/hmm_macro_model.pkl"


def run_live_test(ticker="NVDA"):
    runtime = resolve_torch_runtime()
    device = runtime.device

    # 1. Load the Brain + Scaler + HMM
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    hmm_data = joblib.load(HMM_PATH)

    scaler = checkpoint['scaler']
    feature_cols = checkpoint['features_list']

    model = MegaStrategyNet(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        num_classes=4,
        dropout=checkpoint.get('dropout', 0.35),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Download Live Macro + Stock Data (Last 90 days for technicals)
    logger_msg = f"--- 🧠 TESTING {ticker} ON LIVE DATA ---"
    print(f"\n{logger_msg}")
    print(runtime.message)

    # Macro tickers needed for the HMM
    macro_tickers = {'SPY': 'SPY', '^VIX': 'VIX', '^TNX': 'TNX', 'DX-Y.NYB': 'DXY', 'HYG': 'HYG', 'LQD': 'LQD'}
    raw_macro = yf.download(list(macro_tickers.keys()), period="90d", progress=False)['Close']
    raw_macro.rename(columns=macro_tickers, inplace=True)

    # 3. Generate HMM Probabilities for TODAY
    hmm_features = pd.DataFrame(index=raw_macro.index)
    hmm_features['SPY_ret'] = np.log(raw_macro['SPY'] / raw_macro['SPY'].shift(1))
    hmm_features['TNX_ret'] = np.log(raw_macro['TNX'] / raw_macro['TNX'].shift(1))
    hmm_features['DXY_ret'] = np.log(raw_macro['DXY'] / raw_macro['DXY'].shift(1))
    hmm_features['VIX_lvl'] = raw_macro['VIX']
    hmm_features['Credit_Risk_Ratio'] = raw_macro['HYG'] / raw_macro['LQD']
    hmm_features['Credit_Risk_Momentum'] = np.log(
        hmm_features['Credit_Risk_Ratio'] / hmm_features['Credit_Risk_Ratio'].shift(5)
    )
    hmm_features = hmm_features.dropna()[hmm_data['features_list']]

    hmm_scaled = hmm_data['scaler'].transform(hmm_features.values)
    current_regime_probs = hmm_data['model'].predict_proba(hmm_scaled)
    prob_df = pd.DataFrame(
        current_regime_probs,
        index=hmm_features.index,
        columns=[f'HMM_State_{i}' for i in range(current_regime_probs.shape[1])],
    )

    # 4. Final Feature Fusion
    stock = yf.download(ticker, period="90d", progress=False)['Close']
    df = pd.DataFrame({'Close': stock})
    df['Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Ret'].rolling(20).std() * np.sqrt(252)
    df['SMA_50_Dist'] = (df['Close'] / df['Close'].rolling(50).mean()) - 1

    # Additional macro indicators from your mega_matrix
    other_macro = yf.download(['CL=F', 'ZS=F', 'GC=F'], period="90d", progress=False)['Close']
    other_macro.rename(columns={'CL=F': 'Crude_Oil', 'ZS=F': 'Soybeans', 'GC=F': 'Gold'}, inplace=True)
    other_returns = np.log(other_macro / other_macro.shift(1))

    final_df = df.join(other_returns).join(prob_df).dropna()

    # 5. Inference
    X_raw = final_df[feature_cols].tail(60).values  # Take the most recent 60-day window
    X_scaled = scaler.transform(X_raw)
    X_tensor = torch.tensor(np.array([X_scaled]), dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    classes = {0: "THETA (Iron Condor)", 1: "VEGA (Straddle)", 2: "BULL (Put Spread)", 3: "BEAR (Call Spread)"}
    prediction = np.argmax(probs)

    print(f"Prediction: {classes[prediction]}")
    print(f"Confidence: {probs[prediction]*100:.2f}%")
    print("-" * 35)
    for i, p in enumerate(probs):
        print(f"  {classes[i]:<20}: {p*100:.2f}%")


if __name__ == "__main__":
    run_live_test("NVDA")
    run_live_test("SPY")
