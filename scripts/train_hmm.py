import os
import logging
import warnings

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("train_hmm_macro")

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'hmm_macro_model.pkl')
TARGET_ANNUAL_RETURN = 0.05


def build_macro_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=raw_data.index)

    features['SPY_ret'] = np.log(raw_data['SPY'] / raw_data['SPY'].shift(1))
    features['TNX_ret'] = np.log(raw_data['TNX'] / raw_data['TNX'].shift(1))
    features['DXY_ret'] = np.log(raw_data['DXY'] / raw_data['DXY'].shift(1))
    features['VIX_lvl'] = raw_data['VIX']
    features['Credit_Risk_Ratio'] = raw_data['HYG'] / raw_data['LQD']
    features['Credit_Risk_Momentum'] = np.log(features['Credit_Risk_Ratio'] / features['Credit_Risk_Ratio'].shift(5))

    # Hidden cross-market stress/correlation channels
    features['Corr_SPY_VIX_20'] = features['SPY_ret'].rolling(20).corr(np.log(raw_data['VIX'] / raw_data['VIX'].shift(1))).clip(-1, 1)
    features['Corr_SPY_DXY_20'] = features['SPY_ret'].rolling(20).corr(features['DXY_ret']).clip(-1, 1)
    features['Corr_SPY_TNX_20'] = features['SPY_ret'].rolling(20).corr(features['TNX_ret']).clip(-1, 1)

    return features.dropna()


def train_and_save_macro_hmm():
    logger.info("🌍 Downloading 15 Years of Global Macro Data...")

    tickers = {
        'SPY': 'SPY',
        'VIX': '^VIX',
        'TNX': '^TNX',
        'DXY': 'DX-Y.NYB',
        'HYG': 'HYG',
        'LQD': 'LQD',
    }

    raw_data = yf.download(list(tickers.values()), period="15y", progress=False)['Close']
    rename_map = {v: k for k, v in tickers.items()}
    raw_data.rename(columns=rename_map, inplace=True)
    raw_data = raw_data.dropna()

    logger.info("⚙️ Engineering Latent Macro Features (with hidden correlations)...")
    features = build_macro_features(raw_data)

    logger.info("⚖️ Standardizing feature scales to prevent covariance collapse...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)

    n_states = 4
    logger.info(f"🏋️ Training {n_states}-State Hidden Markov Model with Full Covariance Matrix...")
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=3000,
        random_state=42,
        tol=1e-4,
        min_covar=1e-5,
    )
    model.fit(X_scaled)

    hidden_states = model.predict(X_scaled)
    state_probs = model.predict_proba(X_scaled)
    features['Hidden_State'] = hidden_states

    state_profiles = {}
    logger.info("📊 Diagnosing Discovered Regimes:")

    for i in range(n_states):
        state_data = features[features['Hidden_State'] == i]
        avg_vix = state_data['VIX_lvl'].mean()
        avg_spy_ret = state_data['SPY_ret'].mean() * 252
        avg_credit_risk = state_data['Credit_Risk_Ratio'].mean()

        state_profiles[i] = {
            'avg_vix': avg_vix,
            'avg_spy_ret': avg_spy_ret,
            'avg_credit_risk': avg_credit_risk,
        }

        logger.info(
            "   ► State %d | Days: %d | Avg VIX: %.1f | SPY Ann. Ret: %+.2f%% | Credit Ratio: %.3f",
            i,
            len(state_data),
            avg_vix,
            avg_spy_ret * 100,
            avg_credit_risk,
        )

    sorted_by_vix = sorted(state_profiles.items(), key=lambda x: x[1]['avg_vix'])
    state_map = {
        sorted_by_vix[0][0]: "GOLDILOCKS",
        sorted_by_vix[1][0]: "TRANSITION",
        sorted_by_vix[2][0]: "RISK_OFF",
        sorted_by_vix[3][0]: "LIQUIDITY_CRUNCH",
    }

    state_confidence = np.mean(np.max(state_probs, axis=1))
    transition_stability = float(np.mean(np.max(model.transmat_, axis=1)))

    logger.info("🏷️ Final Regime Mapping Assigned.")
    for state_id, tag in state_map.items():
        logger.info(f"   -> State {state_id} = {tag}")

    logger.info("🔒 Avg state confidence: %.2f%% | Transition stability: %.2f%%", state_confidence * 100, transition_stability * 100)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {
            'model': model,
            'scaler': scaler,
            'state_map': state_map,
            'features_list': features.drop(columns=['Hidden_State']).columns.tolist(),
            'target_annual_return': TARGET_ANNUAL_RETURN,
            'state_confidence': state_confidence,
            'transition_stability': transition_stability,
        },
        MODEL_PATH,
    )

    logger.info(f"✅ Macro HMM Engine successfully saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_macro_hmm()
