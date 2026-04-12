import logging
from datetime import date, timedelta
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.universe_maintenance import download_close_matrix

logger = logging.getLogger(f"strategy.{__name__}")


LOOKBACK_MAP = {
    "10y": 3650,
    "5y": 1825,
    "3y": 1095,
    "1y": 365,
    "6mo": 182,
    "3mo": 91,
    "ytd": None,
}
LOOKBACK_PERIOD_FALLBACKS = ["6mo", "1y", "2y", "5y", "10y", "max"]
MIN_ROWS_BY_LOOKBACK = {
    "3mo": 45,
    "6mo": 70,
    "ytd": 50,
    "1y": 110,
    "3y": 140,
    "5y": 150,
    "10y": 150,
}


@dataclass
class MovementSignal:
    symbol: str
    probability_up: float
    expected_daily_move: float
    expected_direction: str


def _download_prices(symbol: str, period: str) -> pd.Series:
    close = download_close_matrix([symbol], period=period, auto_adjust=True, progress=False)
    if close.empty or symbol not in close.columns:
        return pd.Series(dtype=float)
    return close[symbol].dropna()


def lookback_days(lookback: str) -> int:
    if lookback == "ytd":
        today = date.today()
        return max(1, (today - date(today.year, 1, 1)).days + 1)
    return LOOKBACK_MAP.get(lookback, LOOKBACK_MAP["5y"])


def _lookback_cutoff(lookback: str) -> pd.Timestamp:
    if lookback == "ytd":
        return pd.Timestamp(date.today().year, 1, 1)
    return pd.Timestamp(date.today() - timedelta(days=lookback_days(lookback)))


def _download_prices_with_warmup(symbol: str, lookback: str, warmup_days: int = 252) -> pd.Series:
    """
    Download enough history to compute rolling features while still evaluating the requested lookback.
    Falls back to progressively larger periods when Yahoo returns sparse data.
    """
    required_days = lookback_days(lookback)

    close = pd.Series(dtype=float)
    for period in LOOKBACK_PERIOD_FALLBACKS:
        candidate = _download_prices(symbol, period=period)
        if not candidate.empty:
            close = candidate
            if len(candidate) >= required_days + warmup_days:
                break

    if close.empty:
        return close

    return close


def _slice_features_to_lookback(features: pd.DataFrame, lookback: str) -> pd.DataFrame:
    if features.empty:
        return features
    cutoff = _lookback_cutoff(lookback)
    sliced = features.loc[features.index >= cutoff]
    return sliced.dropna().copy()


def _min_rows_required(lookback: str) -> int:
    return MIN_ROWS_BY_LOOKBACK.get(lookback, 150)


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
    close = _download_prices_with_warmup(symbol, lookback=lookback)
    features = _slice_features_to_lookback(_feature_frame(close), lookback)

    if len(features) < _min_rows_required(lookback):
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
    close = _download_prices_with_warmup(symbol, lookback=lookback)
    features = _slice_features_to_lookback(_feature_frame(close), lookback)
    min_rows = _min_rows_required(lookback)
    if len(features) < min_rows:
        return {
            "symbol": symbol,
            "lookback": lookback,
            "error": "insufficient_data",
            "available_rows": int(len(features)),
            "required_rows": int(min_rows),
        }

    split_idx = int(len(features) * 0.7)
    if split_idx < 30 or (len(features) - split_idx) < 15:
        return {
            "symbol": symbol,
            "lookback": lookback,
            "error": "insufficient_split",
            "available_rows": int(len(features)),
        }
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
