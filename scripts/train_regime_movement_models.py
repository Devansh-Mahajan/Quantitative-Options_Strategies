import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from core.universe_maintenance import download_close_matrix

ROOT = Path(__file__).resolve().parents[1]
HMM_PATH = ROOT / "config" / "hmm_macro_model.pkl"
OUT_PATH = ROOT / "config" / "regime_movement_models.pkl"


def _build_macro_features(raw: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=raw.index)
    feat["SPY_ret"] = np.log(raw["SPY"] / raw["SPY"].shift(1))
    feat["TNX_ret"] = np.log(raw["TNX"] / raw["TNX"].shift(1))
    feat["DXY_ret"] = np.log(raw["DXY"] / raw["DXY"].shift(1))
    feat["VIX_lvl"] = raw["VIX"]
    feat["Credit_Risk_Ratio"] = raw["HYG"] / raw["LQD"]
    feat["Credit_Risk_Momentum"] = np.log(feat["Credit_Risk_Ratio"] / feat["Credit_Risk_Ratio"].shift(5))

    # Keep this aligned with scripts/train_hmm.py::build_macro_features.
    vix_ret = np.log(raw["VIX"] / raw["VIX"].shift(1))
    feat["Corr_SPY_VIX_20"] = feat["SPY_ret"].rolling(20).corr(vix_ret).clip(-1, 1)
    feat["Corr_SPY_DXY_20"] = feat["SPY_ret"].rolling(20).corr(feat["DXY_ret"]).clip(-1, 1)
    feat["Corr_SPY_TNX_20"] = feat["SPY_ret"].rolling(20).corr(feat["TNX_ret"]).clip(-1, 1)
    return feat


def load_hmm_probabilities():
    hmm_data = joblib.load(HMM_PATH)
    features_list = hmm_data["features_list"]
    model = hmm_data["model"]
    scaler = hmm_data["scaler"]

    raw = download_close_matrix(["SPY", "^VIX", "^TNX", "DX-Y.NYB", "HYG", "LQD"], period="10y", auto_adjust=False, progress=False).dropna()
    raw = raw.rename(columns={"^VIX": "VIX", "^TNX": "TNX", "DX-Y.NYB": "DXY"})

    feat = _build_macro_features(raw)
    missing = [col for col in features_list if col not in feat.columns]
    if missing:
        raise ValueError(
            f"HMM model expects unsupported features that are not produced by the current "
            f"pipeline: {missing}"
        )
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    feat = feat[features_list]

    probs = model.predict_proba(scaler.transform(feat.values))
    state = probs.argmax(axis=1)
    out = pd.DataFrame(probs, index=feat.index, columns=[f"hmm_state_{i}" for i in range(probs.shape[1])])
    out["state"] = state
    return out


def train(args):
    reg = load_hmm_probabilities()
    prices = download_close_matrix([args.symbol], period="10y", auto_adjust=True, progress=False)
    if prices.empty or args.symbol not in prices.columns:
        raise SystemExit(f"Failed to download prices for {args.symbol}.")
    prices = prices[args.symbol].dropna()
    df = pd.DataFrame({"close": prices}).join(reg, how="inner").dropna()
    df["ret_1d"] = np.log(df["close"] / df["close"].shift(1))
    df["ret_5d"] = np.log(df["close"] / df["close"].shift(5))
    df["mom_20"] = df["close"] / df["close"].rolling(20).mean() - 1
    df["vol_20"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna()

    feat_cols = ["ret_1d", "ret_5d", "mom_20", "vol_20"] + [c for c in df.columns if c.startswith("hmm_state_")]

    models = {}
    metrics = {}
    for state in sorted(df["state"].unique()):
        sub = df[df["state"] == state]
        if len(sub) < 120:
            continue
        split = int(len(sub) * 0.7)
        train_df, test_df = sub.iloc[:split], sub.iloc[split:]

        scaler = StandardScaler().fit(train_df[feat_cols].values)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(scaler.transform(train_df[feat_cols].values), train_df["target"].values)
        pred = clf.predict(scaler.transform(test_df[feat_cols].values))
        acc = float(accuracy_score(test_df["target"].values, pred))

        models[int(state)] = {"scaler": scaler, "model": clf, "feature_cols": feat_cols}
        metrics[int(state)] = {"accuracy": acc, "n_samples": int(len(sub)), "meets_target": acc >= args.target_accuracy}

    joblib.dump({"symbol": args.symbol, "models": models, "metrics": metrics, "target_accuracy": args.target_accuracy}, OUT_PATH)
    print(f"Saved regime models to {OUT_PATH}")
    for k, v in metrics.items():
        print(f"state={k} accuracy={v['accuracy']:.3f} samples={v['n_samples']} meets_target={v['meets_target']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train regime-specific movement models by HMM state.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--target-accuracy", type=float, default=0.56)
    train(parser.parse_args())
