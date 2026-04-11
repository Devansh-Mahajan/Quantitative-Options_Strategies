import argparse
import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS_PATH = ROOT / "config" / "symbol_list.txt"
OUT_PATH = ROOT / "config" / "correlation_alpha_model.pkl"


def load_symbols(limit: int = 180) -> list[str]:
    syms = [s.strip().upper() for s in SYMBOLS_PATH.read_text().splitlines() if s.strip()]
    return list(dict.fromkeys(syms))[:limit]


def zscore(series: pd.Series, lookback: int = 120) -> pd.Series:
    mu = series.rolling(lookback).mean()
    sd = series.rolling(lookback).std(ddof=0).replace(0, np.nan)
    return (series - mu) / sd


def evaluate_pair(log_a: pd.Series, log_b: pd.Series, entry: float = 1.5, max_hold: int = 10) -> dict | None:
    spread = (log_a - log_b).dropna()
    if len(spread) < 260:
        return None

    z = zscore(spread, lookback=120).dropna()
    if len(z) < 150:
        return None

    pnl = []
    win = 0
    n = 0
    idx = z.index
    for i in range(1, len(idx) - max_hold - 1):
        t = idx[i]
        z0 = float(z.loc[t])
        if abs(z0) < entry:
            continue
        side = -1.0 if z0 > 0 else 1.0  # short spread when rich, long when cheap
        s0 = float(spread.loc[t])

        closed = False
        for h in range(1, max_hold + 1):
            t2 = idx[i + h]
            z1 = float(z.loc[t2])
            s1 = float(spread.loc[t2])
            if z0 > 0 and z1 <= 0:
                r = side * (s1 - s0)
                pnl.append(r)
                win += 1 if r > 0 else 0
                n += 1
                closed = True
                break
            if z0 < 0 and z1 >= 0:
                r = side * (s1 - s0)
                pnl.append(r)
                win += 1 if r > 0 else 0
                n += 1
                closed = True
                break
        if not closed:
            t2 = idx[i + max_hold]
            s1 = float(spread.loc[t2])
            r = side * (s1 - s0)
            pnl.append(r)
            win += 1 if r > 0 else 0
            n += 1

    if n < 15:
        return None

    arr = np.array(pnl, dtype=float)
    return {
        "samples": int(n),
        "win_rate": float(win / n),
        "avg_spread_return": float(arr.mean()),
        "sharpe_like": float(arr.mean() / (arr.std(ddof=0) + 1e-8)),
    }


def train(symbol_limit: int = 180, min_corr: float = 0.70):
    symbols = load_symbols(limit=symbol_limit)
    prices = yf.download(symbols, period="5y", progress=False, auto_adjust=True)["Close"]
    if prices is None or prices.empty:
        raise RuntimeError("No prices returned for correlation alpha training.")

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.80)).ffill().dropna()
    logp = np.log(prices.clip(lower=1e-6))
    returns = logp.diff().dropna()
    corr = returns.corr()

    pairs = []
    for a, b in itertools.combinations(corr.columns.tolist(), 2):
        c = float(corr.loc[a, b])
        if c < min_corr:
            continue
        stats = evaluate_pair(logp[a], logp[b])
        if not stats:
            continue
        pair = f"{a}/{b}"
        pairs.append({"pair": pair, "correlation": c, **stats})

    pairs = sorted(
        pairs,
        key=lambda x: (x["win_rate"] * 0.45) + (x["sharpe_like"] * 0.35) + (x["correlation"] * 0.20),
        reverse=True,
    )

    payload = {
        "pairs": pairs[:300],
        "train_symbols": symbols,
        "num_pairs": len(pairs),
        "min_corr": min_corr,
        "entry_z": 1.5,
        "max_hold_days": 10,
    }
    joblib.dump(payload, OUT_PATH)
    print(f"Saved correlation alpha model: {OUT_PATH} | pairs={len(payload['pairs'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train correlation alpha priors for pair mean-reversion.")
    parser.add_argument("--symbol-limit", type=int, default=180)
    parser.add_argument("--min-corr", type=float, default=0.70)
    args = parser.parse_args()
    train(symbol_limit=args.symbol_limit, min_corr=args.min_corr)
