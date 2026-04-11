import logging
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(f"strategy.{__name__}")


LOOKBACK_MAP = {
    "10y": 3650,
    "5y": 1825,
    "1y": 365,
    "6mo": 182,
    "3mo": 91,
}


@dataclass
class MovementSignal:
    symbol: str
    probability_up: float
    expected_daily_move: float
    expected_direction: str


def _download_prices(symbol: str, period: str) -> pd.Series:
    data = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    close = data.get("Close")
    if close is None or close.empty:
        return pd.Series(dtype=float)
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    return close.dropna()


def _feature_frame(close: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=close.index)
    ret = np.log(close / close.shift(1))
    df["ret_1d"] = ret
    df["ret_5d"] = np.log(close / close.shift(5))
    df["ret_10d"] = np.log(close / close.shift(10))
    df["vol_20"] = ret.rolling(20).std() * np.sqrt(252)
    df["vol_60"] = ret.rolling(60).std() * np.sqrt(252)
    df["momentum_20"] = close / close.rolling(20).mean() - 1.0
    df["momentum_50"] = close / close.rolling(50).mean() - 1.0
    df["target"] = (close.shift(-1) > close).astype(int)
    return df.dropna()


def fit_symbol_movement_model(symbol: str, lookback: str = "5y") -> tuple[Pipeline | None, pd.DataFrame]:
    period = lookback if lookback in LOOKBACK_MAP else "5y"
    close = _download_prices(symbol, period=period)
    features = _feature_frame(close)

    if len(features) < 150:
        logger.warning("Insufficient data for movement model on %s.", symbol)
        return None, pd.DataFrame()

    X = features.drop(columns=["target"])
    y = features["target"]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X, y)
    return model, features


def predict_symbol_movement(symbol: str, lookback: str = "5y") -> MovementSignal | None:
    model, features = fit_symbol_movement_model(symbol, lookback=lookback)
    if model is None or features.empty:
        return None

    X_latest = features.drop(columns=["target"]).tail(1)
    prob_up = float(model.predict_proba(X_latest)[0][1])
    expected_move = float(features["ret_1d"].tail(60).mean())
    direction = "up" if prob_up >= 0.55 else "down" if prob_up <= 0.45 else "flat"
    return MovementSignal(
        symbol=symbol,
        probability_up=prob_up,
        expected_daily_move=expected_move,
        expected_direction=direction,
    )


def aggregate_movement_signals(symbols: Iterable[str], lookback: str = "5y") -> List[MovementSignal]:
    signals: List[MovementSignal] = []
    for sym in symbols:
        signal = predict_symbol_movement(sym, lookback=lookback)
        if signal:
            signals.append(signal)
    return signals


def backtest_symbol_movement(symbol: str, lookback: str = "5y") -> dict:
    """Walk-forward next-day direction backtest."""
    period = lookback if lookback in LOOKBACK_MAP else "5y"
    close = _download_prices(symbol, period=period)
    features = _feature_frame(close)
    if len(features) < 250:
        return {"symbol": symbol, "lookback": lookback, "error": "insufficient_data"}

    split_idx = int(len(features) * 0.7)
    train, test = features.iloc[:split_idx], features.iloc[split_idx:]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(train.drop(columns=["target"]), train["target"])
    probs = model.predict_proba(test.drop(columns=["target"]))[:, 1]
    preds = (probs >= 0.5).astype(int)
    y = test["target"].values

    accuracy = float((preds == y).mean())

    strategy_ret = np.where(preds == 1, test["ret_1d"].values, -test["ret_1d"].values)
    buy_hold_ret = test["ret_1d"].values

    cumulative_strategy = float(np.exp(np.nansum(strategy_ret)) - 1.0)
    cumulative_buy_hold = float(np.exp(np.nansum(buy_hold_ret)) - 1.0)
    hit_ratio = float((np.sign(strategy_ret) == np.sign(test["ret_1d"].values)).mean())

    return {
        "symbol": symbol,
        "lookback": lookback,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "accuracy": accuracy,
        "hit_ratio": hit_ratio,
        "strategy_return": cumulative_strategy,
        "buy_hold_return": cumulative_buy_hold,
    }
