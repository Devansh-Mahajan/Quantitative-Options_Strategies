import os
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from core.universe_maintenance import download_close_matrix, load_symbol_file

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("train_hmm_macro")

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'hmm_macro_model.pkl')
TARGET_ANNUAL_RETURN = 0.05
RANDOM_SEED = 42
OVERFIT_GAP_THRESHOLD = 0.12
SYMBOL_LIST_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'symbol_list.txt')


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


def load_universe_symbols(max_symbols: int = 100) -> list[str]:
    return load_symbol_file(Path(SYMBOL_LIST_PATH))[:max_symbols]


def _collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        flattened = []
        for col in df.columns:
            if isinstance(col, tuple):
                flattened.append(next((part for part in reversed(col) if part not in ("", None)), col[0]))
            else:
                flattened.append(col)
        df = df.copy()
        df.columns = flattened
    if df.columns.has_duplicates:
        return df.T.groupby(level=0).mean().T
    return df


def _safe_symbol_series(df: pd.DataFrame, symbol: str) -> pd.Series:
    selected = df.loc[:, df.columns == symbol]
    if selected.empty:
        raise KeyError(f"Missing symbol column: {symbol}")
    series = selected.iloc[:, 0]
    series.name = symbol
    return series


def discover_ticker_patterns() -> dict:
    symbols = load_universe_symbols(max_symbols=90)
    if len(symbols) < 12:
        logger.warning("Skipping ticker pattern discovery due to insufficient symbol universe.")
        return {}

    logger.info("🕸️ Discovering cross-ticker correlation and pairs patterns...")
    ticker_close = download_close_matrix(symbols, period="10y", auto_adjust=False, progress=False)
    if ticker_close is None or ticker_close.empty:
        logger.warning("Ticker download returned no data; skipping pairs discovery.")
        return {}

    returns = np.log(ticker_close / ticker_close.shift(1)).replace([np.inf, -np.inf], np.nan)
    # yfinance can occasionally emit duplicate or multi-indexed symbol columns;
    # normalize to a single 1-D series per symbol for downstream pair operations.
    returns = _collapse_duplicate_columns(returns)

    valid_cols = returns.columns[returns.notna().mean() > 0.80]
    returns = returns[valid_cols].dropna(how='all').ffill().dropna(axis=1, how='any')
    if returns.shape[1] < 10 or len(returns) < 200:
        logger.warning("Skipping ticker pattern discovery due to sparse returns matrix.")
        return {}

    corr = returns.corr().clip(-1, 1)
    corr_values = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().sort_values()
    top_negative = corr_values.head(8)
    top_positive = corr_values.tail(8).sort_values(ascending=False)

    top_positive_pairs = [
        {'pair': f"{a}/{b}", 'correlation': float(val)}
        for (a, b), val in top_positive.items()
    ]
    top_negative_pairs = [
        {'pair': f"{a}/{b}", 'correlation': float(val)}
        for (a, b), val in top_negative.items()
    ]

    standardized = (returns - returns.mean()) / returns.std().replace(0, 1)
    pca = PCA(n_components=min(3, standardized.shape[1]))
    pcs = pca.fit_transform(standardized.values)

    pattern_features = pd.DataFrame(index=returns.index)
    pattern_features['Universe_Mean_Return'] = returns.mean(axis=1)
    pattern_features['Universe_Dispersion'] = returns.std(axis=1)
    pattern_features['Breadth_Positive'] = (returns > 0).mean(axis=1)
    pattern_features['PC1'] = pcs[:, 0]
    pattern_features['PC2'] = pcs[:, 1] if pcs.shape[1] > 1 else 0.0

    feature_scaler = StandardScaler()
    pattern_scaled = feature_scaler.fit_transform(pattern_features.values)

    best_model = None
    best_meta = None
    for seed in [7, 13, 21, 42]:
        candidate = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=1500,
            random_state=seed,
            min_covar=1e-4,
            tol=1e-4,
        )
        candidate.fit(pattern_scaled)
        score = candidate.score(pattern_scaled) / max(len(pattern_scaled), 1)
        if best_meta is None or score > best_meta['score']:
            best_model = candidate
            best_meta = {'seed': seed, 'score': float(score)}

    states = best_model.predict(pattern_scaled)
    probs = best_model.predict_proba(pattern_scaled)
    pattern_features['Ticker_HMM_State'] = states
    pattern_prob_df = pd.DataFrame(
        probs,
        index=pattern_features.index,
        columns=[f'Ticker_HMM_State_{idx}' for idx in range(probs.shape[1])],
    )

    logger.info(
        "🔗 Ticker-pattern HMM trained | symbols=%d best_seed=%d score=%.6f",
        returns.shape[1],
        best_meta['seed'],
        best_meta['score'],
    )
    logger.info("🔝 Top positive pairs: %s", ", ".join(p['pair'] for p in top_positive_pairs[:5]))
    logger.info("🔻 Top negative pairs: %s", ", ".join(p['pair'] for p in top_negative_pairs[:5]))

    lead_lag_pairs = []
    for pair in top_positive_pairs[:20]:
        a, b = pair['pair'].split('/')
        try:
            series_a = _safe_symbol_series(returns, a)
            series_b = _safe_symbol_series(returns, b)
        except KeyError:
            continue

        pair_df = pd.concat([series_a, series_b], axis=1).dropna()
        if len(pair_df) < 120:
            continue
        series_a = pair_df.iloc[:, 0]
        series_b = pair_df.iloc[:, 1]

        corr_a_leads_b = series_a.shift(1).corr(series_b)
        corr_b_leads_a = series_b.shift(1).corr(series_a)
        if pd.isna(corr_a_leads_b) or pd.isna(corr_b_leads_a):
            continue
        if abs(corr_a_leads_b) >= abs(corr_b_leads_a):
            lead_lag_pairs.append(
                {
                    'pair': pair['pair'],
                    'leader': a,
                    'follower': b,
                    'lag1_correlation': float(corr_a_leads_b),
                }
            )
        else:
            lead_lag_pairs.append(
                {
                    'pair': pair['pair'],
                    'leader': b,
                    'follower': a,
                    'lag1_correlation': float(corr_b_leads_a),
                }
            )

    lead_lag_pairs = sorted(lead_lag_pairs, key=lambda x: abs(x['lag1_correlation']), reverse=True)[:12]
    if lead_lag_pairs:
        logger.info(
            "⛓️ Lead/Lag patterns: %s",
            ", ".join(f"{p['leader']}→{p['follower']}({p['lag1_correlation']:+.2f})" for p in lead_lag_pairs[:5]),
        )

    return {
        'symbols_used': list(returns.columns),
        'top_positive_pairs': top_positive_pairs,
        'top_negative_pairs': top_negative_pairs,
        'lead_lag_pairs': lead_lag_pairs,
        'pattern_features': pattern_features.drop(columns=['Ticker_HMM_State']),
        'pattern_probabilities': pattern_prob_df,
        'hmm_state_series': pattern_features['Ticker_HMM_State'],
        'hmm_seed': int(best_meta['seed']),
        'hmm_score': best_meta['score'],
        'explained_variance': pca.explained_variance_ratio_.tolist(),
    }


def train_and_save_macro_hmm():
    np.random.seed(RANDOM_SEED)
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

    split_idx = int(len(X_scaled) * 0.8)
    X_train = X_scaled[:split_idx]
    X_valid = X_scaled[split_idx:]

    candidate_states = [3, 4, 5]
    candidate_seeds = [11, 21, 42, 84, 168, 336]
    candidate_covars = [1e-5, 5e-5, 1e-4]
    best_model = None
    best_score = -np.inf
    best_n_states = None
    best_seed = None
    best_meta = None

    for n_states in candidate_states:
        for seed in candidate_seeds:
            for min_covar in candidate_covars:
                try:
                    candidate = GaussianHMM(
                        n_components=n_states,
                        covariance_type="full",
                        n_iter=3000,
                        random_state=seed,
                        tol=1e-4,
                        min_covar=min_covar,
                    )
                    candidate.fit(X_train)
                    train_score = candidate.score(X_train) / max(len(X_train), 1)
                    valid_score = candidate.score(X_valid) / max(len(X_valid), 1)
                    overfit_gap = train_score - valid_score
                    penalty = 0.03 * (n_states - 3)
                    objective = valid_score - penalty
                    if overfit_gap > OVERFIT_GAP_THRESHOLD:
                        objective -= overfit_gap * 0.75
                    logger.info(
                        "🧪 Candidate HMM | states=%d seed=%d min_covar=%g train=%.6f valid=%.6f gap=%.6f objective=%.6f",
                        n_states,
                        seed,
                        min_covar,
                        train_score,
                        valid_score,
                        overfit_gap,
                        objective,
                    )
                    if objective > best_score:
                        best_score = objective
                        best_model = candidate
                        best_n_states = n_states
                        best_seed = seed
                        best_meta = {
                            'min_covar': float(min_covar),
                            'train_loglike_per_step': float(train_score),
                            'valid_loglike_per_step': float(valid_score),
                            'overfit_gap': float(overfit_gap),
                            'selection_objective': float(objective),
                        }
                except Exception as exc:
                    logger.warning(
                        "Candidate training failed (states=%d seed=%d min_covar=%g): %s",
                        n_states,
                        seed,
                        min_covar,
                        exc,
                    )

    if best_model is None:
        raise RuntimeError("No viable HMM candidate could be trained.")

    model = best_model
    n_states = best_n_states
    logger.info(
        "🏆 Selected HMM: states=%d seed=%d min_covar=%g valid_loglike_per_step=%.6f train_valid_gap=%.6f",
        n_states,
        best_seed,
        best_meta['min_covar'],
        best_meta['valid_loglike_per_step'],
        best_meta['overfit_gap'],
    )

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
    base_labels = ["GOLDILOCKS", "TRANSITION", "RISK_OFF", "LIQUIDITY_CRUNCH", "PANIC"]
    if n_states > len(base_labels):
        base_labels.extend([f"REGIME_{idx}" for idx in range(len(base_labels), n_states)])
    state_map = {state_id: base_labels[idx] for idx, (state_id, _) in enumerate(sorted_by_vix)}

    state_confidence = np.mean(np.max(state_probs, axis=1))
    transition_stability = float(np.mean(np.max(model.transmat_, axis=1)))

    logger.info("🏷️ Final Regime Mapping Assigned.")
    for state_id, tag in state_map.items():
        logger.info(f"   -> State {state_id} = {tag}")

    logger.info("🔒 Avg state confidence: %.2f%% | Transition stability: %.2f%%", state_confidence * 100, transition_stability * 100)
    ticker_patterns = discover_ticker_patterns()

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
            'validation_loglike_per_step': best_meta['valid_loglike_per_step'],
            'train_loglike_per_step': best_meta['train_loglike_per_step'],
            'train_valid_loglike_gap': best_meta['overfit_gap'],
            'selection_objective': best_meta['selection_objective'],
            'selected_n_states': int(n_states),
            'selected_seed': int(best_seed),
            'selected_min_covar': best_meta['min_covar'],
            'ticker_patterns': ticker_patterns,
        },
        MODEL_PATH,
    )

    logger.info(f"✅ Macro HMM Engine successfully saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_macro_hmm()
