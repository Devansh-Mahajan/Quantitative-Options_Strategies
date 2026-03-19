import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("train_hmm_macro")

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'hmm_macro_model.pkl')

def train_and_save_macro_hmm():
    logger.info("🌍 Downloading 15 Years of Global Macro Data...")
    
    # 1. The Institutional Macro Universe
    tickers = {
        'SPY': 'SPY',          # Equities (Direction)
        'VIX': '^VIX',         # Fear
        'TNX': '^TNX',         # 10Y Treasury (Cost of Capital)
        'DXY': 'DX-Y.NYB',     # US Dollar (Global Liquidity)
        'HYG': 'HYG',          # High Yield Corporate Debt
        'LQD': 'LQD'           # Investment Grade Corporate Debt
    }
    
    raw_data = yf.download(list(tickers.values()), period="15y", progress=False)['Close']
    
    # Rename columns to our readable names
    rename_map = {v: k for k, v in tickers.items()}
    raw_data.rename(columns=rename_map, inplace=True)
    raw_data = raw_data.dropna()

    logger.info("⚙️ Engineering Latent Macro Features (The Plumbing)...")
    
    # 2. Feature Engineering
    features = pd.DataFrame(index=raw_data.index)
    
    # Equity & Rates Momentum
    features['SPY_ret'] = np.log(raw_data['SPY'] / raw_data['SPY'].shift(1))
    features['TNX_ret'] = np.log(raw_data['TNX'] / raw_data['TNX'].shift(1))
    features['DXY_ret'] = np.log(raw_data['DXY'] / raw_data['DXY'].shift(1))
    
    # Absolute Fear Level
    features['VIX_lvl'] = raw_data['VIX']
    
    # THE SECRET WEAPON: Credit Spread Ratio (Risk Appetite)
    # If HYG/LQD is dropping, smart money is panicking under the surface.
    features['Credit_Risk_Ratio'] = raw_data['HYG'] / raw_data['LQD']
    features['Credit_Risk_Momentum'] = np.log(features['Credit_Risk_Ratio'] / features['Credit_Risk_Ratio'].shift(5)) # 1-week momentum

    # Drop the NA rows from the shift() calculations
    features = features.dropna()
    
    # 3. Scaling
    # HMMs will mathematically collapse if features are on wildly different scales (like VIX at 20 vs SPY_ret at 0.01)
    logger.info("⚖️ Standardizing feature scales to prevent covariance collapse...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)
    
    # 4. Train the HMM
    # We use 4 states (Classic Economic Cycle) and "full" covariance to find interlinked correlations
    n_states = 4
    logger.info(f"🏋️ Training {n_states}-State Hidden Markov Model with Full Covariance Matrix...")
    
    model = GaussianHMM(
        n_components=n_states, 
        covariance_type="full", # "full" allows it to see that DXY spikes exactly when SPY drops
        n_iter=2000, 
        random_state=42,
        tol=1e-4
    )
    model.fit(X_scaled)
    
    # 5. Map the Hidden States to Human Regimes
    hidden_states = model.predict(X_scaled)
    features['Hidden_State'] = hidden_states
    
    state_profiles = {}
    logger.info("📊 Diagnosing Discovered Regimes:")
    
    for i in range(n_states):
        state_data = features[features['Hidden_State'] == i]
        avg_vix = state_data['VIX_lvl'].mean()
        avg_spy_ret = state_data['SPY_ret'].mean() * 252 # Annualized
        avg_credit_risk = state_data['Credit_Risk_Ratio'].mean()
        
        state_profiles[i] = {
            'avg_vix': avg_vix,
            'avg_spy_ret': avg_spy_ret,
            'avg_credit_risk': avg_credit_risk
        }
        
        logger.info(f"   ► State {i} | Days: {len(state_data)} | Avg VIX: {avg_vix:.1f} | SPY Ann. Ret: {avg_spy_ret*100:+.1f}% | Credit Ratio: {avg_credit_risk:.3f}")

    # Auto-tag states based on VIX severity
    sorted_by_vix = sorted(state_profiles.items(), key=lambda x: x[1]['avg_vix'])
    
    state_map = {
        sorted_by_vix[0][0]: "GOLDILOCKS (Low Vol, Bullish)",
        sorted_by_vix[1][0]: "TRANSITION (Normal Chop)",
        sorted_by_vix[2][0]: "RISK-OFF (High Vol, Bearish)",
        sorted_by_vix[3][0]: "LIQUIDITY CRUNCH (Panic, Credit Spreads Blowing Up)"
    }
    
    logger.info("🏷️ Final Regime Mapping Assigned.")
    for state_id, word in state_map.items():
        logger.info(f"   -> State {state_id} = {word}")
    
    # 6. Save the Brain and the Scaler
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        'model': model, 
        'scaler': scaler, 
        'state_map': state_map,
        'features_list': features.drop(columns=['Hidden_State']).columns.tolist()
    }, MODEL_PATH)
    
    logger.info(f"✅ Macro HMM Engine successfully saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_macro_hmm()