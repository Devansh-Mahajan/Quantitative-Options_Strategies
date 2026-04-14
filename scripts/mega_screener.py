import os
import re
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import torch
import yfinance as yf
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.mega_neural_brain import MegaStrategyNet
from core.torch_device import resolve_torch_runtime
from scripts.mega_matrix import MACRO_TICKERS, TICKERS, get_hmm_probabilities

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("mega_screener")

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'mega_universe_dataset.pt')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'trading_model.pth')
HMM_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'hmm_macro_model.pkl')
EMPTY_RESULTS = {'THETA': [], 'VEGA': [], 'BULL': [], 'BEAR': []}


def _build_feature_frame(
    df: pd.DataFrame,
    macro_returns: pd.DataFrame,
    hmm_probs: pd.DataFrame,
    feature_cols,
    ticker_prob: pd.DataFrame | None = None,
    ticker_feat: pd.DataFrame | None = None,
):
    df['Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Ret'].rolling(20).std() * np.sqrt(252)
    df['SMA_50_Dist'] = (df['Close'] / df['Close'].rolling(50).mean()) - 1
    df = df.join(macro_returns, how='left')
    df = df.join(hmm_probs, how='left')
    if isinstance(ticker_prob, pd.DataFrame):
        df = df.join(ticker_prob, how='left')
    if isinstance(ticker_feat, pd.DataFrame):
        df = df.join(ticker_feat, how='left')

    corr_pattern = re.compile(r"^Corr_(.+)_(\d+)$")
    for col in feature_cols:
        m = corr_pattern.match(col)
        if m:
            macro_col = m.group(1)
            window = int(m.group(2))
            if macro_col in df.columns:
                df[col] = df['Ret'].rolling(window).corr(df[macro_col]).clip(-1, 1)
            else:
                df[col] = 0.0

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df.dropna()


def _infer(confidence_threshold=75.0):
    runtime = resolve_torch_runtime()
    device = runtime.device
    dataset = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    scaler = dataset['scaler']
    feature_cols = dataset['features_list']

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        input_size = checkpoint.get("input_size", len(feature_cols))
        hidden_size = checkpoint.get("hidden_size", 256)
        num_layers = checkpoint.get("num_layers", 3)
        dropout = checkpoint.get("dropout", 0.35)
    else:
        model_state = checkpoint
        input_size = len(feature_cols)
        hidden_size = 256
        num_layers = 3
        dropout = 0.35

    model = MegaStrategyNet(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=4,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()

    logger.info("📥 Downloading live data + HMM probabilities...")
    macro_data = yf.download(
        list(MACRO_TICKERS.keys()),
        period="6mo",
        progress=False,
        auto_adjust=False,
        threads=False,
    )['Close']
    macro_data.rename(columns=MACRO_TICKERS, inplace=True)
    macro_returns = np.log(macro_data / macro_data.shift(1)).fillna(0)
    hmm_probs = get_hmm_probabilities().tail(180)
    hmm_data = joblib.load(HMM_PATH) if os.path.exists(HMM_PATH) else {}
    ticker_patterns = hmm_data.get('ticker_patterns', {})
    ticker_prob = ticker_patterns.get('pattern_probabilities')
    ticker_feat = ticker_patterns.get('pattern_features')
    if isinstance(ticker_prob, pd.DataFrame):
        ticker_prob = ticker_prob.tail(180)
    if isinstance(ticker_feat, pd.DataFrame):
        ticker_feat = ticker_feat.tail(180)

    stock_data = yf.download(
        TICKERS,
        period="6mo",
        progress=False,
        auto_adjust=False,
        threads=False,
    )['Close']

    valid_sequences, valid_tickers = [], []
    for ticker in TICKERS:
        if ticker not in stock_data.columns:
            continue
        df = pd.DataFrame({'Close': stock_data[ticker]}).dropna()
        if len(df) < 60:
            continue

        feat_df = _build_feature_frame(
            df,
            macro_returns,
            hmm_probs,
            feature_cols,
            ticker_prob=ticker_prob,
            ticker_feat=ticker_feat,
        )
        last_60 = feat_df[feature_cols].iloc[-60:]
        if len(last_60) == 60:
            valid_sequences.append(last_60.values)
            valid_tickers.append(ticker)

    if not valid_sequences:
        logger.error("Failed to build sequences. Check data feeds.")
        return {'THETA': [], 'VEGA': [], 'BULL': [], 'BEAR': []}

    all_seq_arr = np.array(valid_sequences)
    n, seq, f = all_seq_arr.shape
    X_flat = all_seq_arr.reshape(-1, f)
    X_scaled = scaler.transform(X_flat).reshape(n, seq, f)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    class_map = {0: 'THETA', 1: 'VEGA', 2: 'BULL', 3: 'BEAR'}
    results = {'THETA': [], 'VEGA': [], 'BULL': [], 'BEAR': []}
    for i, t in enumerate(valid_tickers):
        conf = probs[i][preds[i]] * 100
        if conf >= confidence_threshold:
            results[class_map[preds[i]]].append((t, conf))

    for key in results:
        results[key].sort(key=lambda x: x[1], reverse=True)

    return results


def scan_market():
    logger.info("🔌 Booting Mega Inference Engine...")
    logger.info(resolve_torch_runtime().message)
    results = _infer(confidence_threshold=0.0)

    pretty = {
        'THETA': 'THETA (Iron Condors)',
        'VEGA': 'VEGA (Straddles)',
        'BULL': 'BULL (Put Spreads)',
        'BEAR': 'BEAR (Call Spreads)',
    }

    print("\n" + "=" * 60)
    print(" 🎯 MEGA BRAIN DAILY OPTIONS SCREENER 🎯")
    print("=" * 60)
    for key in ['THETA', 'VEGA', 'BULL', 'BEAR']:
        print(f"\n🔮 {pretty[key]}")
        print("-" * 30)
        plays = results[key]
        if not plays:
            print("   No high-probability setups found today.")
        for ticker, conf in plays[:5]:
            print(f"   ► {ticker:<6} | Confidence: {conf:.2f}%")
    print("\n" + "=" * 60)


def get_mega_brain_targets(confidence_threshold=75.0):
    missing = [path for path in (DATA_PATH, MODEL_PATH) if not os.path.exists(path)]
    if missing:
        logger.warning("Mega Brain artifacts missing: %s", ", ".join(missing))
        return EMPTY_RESULTS.copy()

    try:
        results = _infer(confidence_threshold=confidence_threshold)
    except Exception as exc:
        logger.exception("Mega Brain inference failed: %s", exc)
        return EMPTY_RESULTS.copy()
    return {k: [x[0] for x in v] for k, v in results.items()}


if __name__ == "__main__":
    scan_market()
