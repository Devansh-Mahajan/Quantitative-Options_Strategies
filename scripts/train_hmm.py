import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import joblib
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("train_hmm")

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'hmm_model.pkl')

def train_and_save_hmm():
    logger.info("🧠 Downloading 10 Years of Market Data (SPY & VIX)...")
    
    # 1. Fetch 10 Years of Data
    spy = yf.download("SPY", period="10y", progress=False)['Close']
    vix = yf.download("^VIX", period="10y", progress=False)['Close']
    
    if isinstance(spy, pd.DataFrame): spy = spy.squeeze()
    if isinstance(vix, pd.DataFrame): vix = vix.squeeze()
    
    # 2. Feature Engineering
    logger.info("⚙️ Engineering features (Log Returns & Volatility)...")
    log_returns = np.log(spy / spy.shift(1))
    rolling_vol = log_returns.rolling(window=5).std() * np.sqrt(252)
    
    data = pd.DataFrame({
        'returns': log_returns,
        'vol': rolling_vol,
        'vix': vix
    }).dropna()
    
    X = data.values
    
    # 3. Train the HMM
    logger.info(f"🏋️ Training Hidden Markov Model on {len(X)} trading days...")
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, random_state=42)
    model.fit(X)
    
    # 4. Map the Hidden States to Human Words
    hidden_states = model.predict(X)
    state_vix_means = {}
    
    for i in range(3):
        if len(data['vix'][hidden_states == i]) > 0:
            state_vix_means[i] = data['vix'][hidden_states == i].mean()
        else:
            state_vix_means[i] = 0
            
    sorted_states = sorted(state_vix_means, key=state_vix_means.get)
    state_map = {
        sorted_states[0]: "CALM",
        sorted_states[1]: "CHOPPY",
        sorted_states[2]: "PANIC"
    }
    
    logger.info("📊 Regime Mapping Created:")
    for state_id, word in state_map.items():
        logger.info(f"   -> State {state_id} = {word} (Avg VIX: {state_vix_means[state_id]:.2f})")
    
    # 5. Save the Brain
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({'model': model, 'state_map': state_map}, MODEL_PATH)
    logger.info(f"✅ Model successfully saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_hmm()