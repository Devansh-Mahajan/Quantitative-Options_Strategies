from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from config.params import (
    ML_ALPHA_BOTTOM_PERCENTILE,
    ML_ALPHA_CACHE_MAX_AGE_MINUTES,
    ML_ALPHA_ENABLED,
    ML_ALPHA_MAX_SYMBOLS,
    ML_ALPHA_MIN_TRAIN_MONTHS,
    ML_ALPHA_TOP_PERCENTILE,
)
from core.universe_maintenance import dedupe_symbols, download_close_matrix

logger = logging.getLogger(f"strategy.{__name__}")

DEFAULT_ALPHA_CACHE_PATH = Path(".runtime/ml_alpha_snapshot.json")


@dataclass(frozen=True)
class AlphaSignal:
    symbol: str
    predicted_return: float
    percentile: float
    alpha_score: float
    direction: str
    model_dispersion: float
    source: str = "ml_alpha_ensemble"


def generate_live_alpha_signals(
    symbols: Iterable[str],
    cache_path: str | Path = DEFAULT_ALPHA_CACHE_PATH,
    max_age_minutes: int = ML_ALPHA_CACHE_MAX_AGE_MINUTES,
    max_symbols: int = ML_ALPHA_MAX_SYMBOLS,
    min_train_months: int = ML_ALPHA_MIN_TRAIN_MONTHS,
) -> list[AlphaSignal]:
    if not ML_ALPHA_ENABLED:
        return []

    requested = dedupe_symbols(symbols)[: max(1, int(max_symbols))]
    if not requested:
        return []

    cache_path = Path(cache_path)
    cached = _load_cached_signals(cache_path, requested, max_age_minutes=max_age_minutes)
    if cached:
        return cached

    try:
        daily_close = download_close_matrix(requested, period="10y", auto_adjust=True, progress=False).ffill()
        feature_frame = build_feature_frame(daily_close)
        clean = clean_feature_frame(feature_frame)
        signals = _fit_live_ensemble(clean, min_train_months=min_train_months)
    except Exception as exc:
        logger.error("ML alpha signal generation failed: %s", exc)
        return []

    if signals:
        _save_cached_signals(cache_path, requested, signals)
    return signals


def live_alpha_signal_map(
    symbols: Iterable[str],
    cache_path: str | Path = DEFAULT_ALPHA_CACHE_PATH,
    max_age_minutes: int = ML_ALPHA_CACHE_MAX_AGE_MINUTES,
    max_symbols: int = ML_ALPHA_MAX_SYMBOLS,
    min_train_months: int = ML_ALPHA_MIN_TRAIN_MONTHS,
) -> dict[str, AlphaSignal]:
    return {
        signal.symbol: signal
        for signal in generate_live_alpha_signals(
            symbols,
            cache_path=cache_path,
            max_age_minutes=max_age_minutes,
            max_symbols=max_symbols,
            min_train_months=min_train_months,
        )
    }


def backtest_alpha_strategy(
    symbols: Iterable[str],
    min_train_months: int = ML_ALPHA_MIN_TRAIN_MONTHS,
    top_percentile: float = ML_ALPHA_TOP_PERCENTILE,
    bottom_percentile: float = ML_ALPHA_BOTTOM_PERCENTILE,
) -> dict:
    requested = dedupe_symbols(symbols)[: max(4, int(ML_ALPHA_MAX_SYMBOLS))]
    if not requested:
        return {"error": "no_symbols"}

    daily_close = download_close_matrix(requested, period="10y", auto_adjust=True, progress=False).ffill()
    feature_frame = clean_feature_frame(build_feature_frame(daily_close))
    labeled = feature_frame.dropna(subset=["target"]).copy()
    if labeled.empty:
        return {"error": "no_labeled_samples"}

    dates = sorted(labeled.index.get_level_values("date").unique())
    if len(dates) < max(24, int(min_train_months) + 6):
        return {"error": "insufficient_months", "available_months": len(dates)}

    monthly_long_short = []
    monthly_long_only = []
    monthly_ic = []
    rows = []

    for i, test_date in enumerate(dates):
        if i < int(min_train_months):
            continue
        train_dates = dates[:i]
        train = labeled.loc[labeled.index.get_level_values("date").isin(train_dates)]
        test = labeled.loc[labeled.index.get_level_values("date") == test_date]
        if len(train) < 250 or len(test) < 6:
            continue

        prediction_frame = _predict_frame(train, test.drop(columns=["target"]))
        if prediction_frame.empty:
            continue

        merged = prediction_frame.join(test[["target"]], how="inner")
        if merged.empty:
            continue

        rank_pct = merged["ensemble"].rank(method="average", pct=True)
        long_bucket = merged.loc[rank_pct >= float(top_percentile)]
        short_bucket = merged.loc[rank_pct <= float(bottom_percentile)]
        if long_bucket.empty:
            continue

        long_ret = float(long_bucket["target"].mean())
        short_ret = float(-short_bucket["target"].mean()) if not short_bucket.empty else 0.0
        long_short_ret = (0.5 * long_ret) + (0.5 * short_ret if not short_bucket.empty else 0.0)
        ic = merged["ensemble"].rank().corr(merged["target"].rank(), method="spearman")

        monthly_long_only.append(long_ret)
        monthly_long_short.append(long_short_ret)
        if not pd.isna(ic):
            monthly_ic.append(float(ic))
        rows.append(
            {
                "date": str(test_date.date() if hasattr(test_date, "date") else test_date),
                "n_test": int(len(merged)),
                "long_bucket": int(len(long_bucket)),
                "short_bucket": int(len(short_bucket)),
                "long_return": long_ret,
                "long_short_return": long_short_ret,
                "information_coefficient": float(ic) if not pd.isna(ic) else 0.0,
            }
        )

    if not monthly_long_only:
        return {"error": "no_backtest_windows"}

    long_only_series = pd.Series(monthly_long_only)
    long_short_series = pd.Series(monthly_long_short)
    summary = {
        "months_tested": int(len(rows)),
        "avg_information_coefficient": _safe_mean(monthly_ic),
        "long_only": _monthly_return_summary(long_only_series),
        "long_short": _monthly_return_summary(long_short_series),
    }
    return {"summary": summary, "rows": rows[-36:]}


def build_feature_frame(daily_close: pd.DataFrame) -> pd.DataFrame:
    if daily_close.empty:
        return pd.DataFrame()

    close = daily_close.copy()
    if isinstance(close.index, pd.DatetimeIndex) and close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close = close.sort_index().dropna(how="all")
    monthly_close = close.resample("ME").last().dropna(how="all")
    daily_ret = close.pct_change()

    rows: list[dict[str, object]] = []
    for idx in range(24, len(monthly_close.index)):
        asof_date = monthly_close.index[idx]
        next_date = monthly_close.index[idx + 1] if idx + 1 < len(monthly_close.index) else None
        for symbol in monthly_close.columns:
            history = monthly_close[symbol].iloc[: idx + 1].dropna()
            if len(history) < 25:
                continue

            current_price = float(history.iloc[-1])
            if current_price <= 0:
                continue

            row = {
                "date": asof_date,
                "ticker": symbol,
                "ret_1m": (history.iloc[-1] / history.iloc[-2]) - 1.0,
                "ret_12_1": (history.iloc[-2] / history.iloc[-13]) - 1.0,
                "ret_6_1": (history.iloc[-2] / history.iloc[-7]) - 1.0 if len(history) >= 7 else np.nan,
                "ret_3_1": (history.iloc[-2] / history.iloc[-4]) - 1.0 if len(history) >= 4 else np.nan,
                "ret_24m": (history.iloc[-1] / history.iloc[-25]) - 1.0 if len(history) >= 25 else np.nan,
                "price_to_52w_high": current_price / float(history.iloc[-12:].max()) if len(history) >= 12 else np.nan,
                "dist_from_sma12": ((current_price / float(history.iloc[-12:].mean())) - 1.0) if len(history) >= 12 else np.nan,
                "dist_from_sma6": ((current_price / float(history.iloc[-6:].mean())) - 1.0) if len(history) >= 6 else np.nan,
            }

            daily_series = daily_ret[symbol].loc[:asof_date].dropna()
            if len(daily_series) >= 21:
                vol_1m = float(daily_series.iloc[-21:].std() * np.sqrt(252.0))
                row["vol_1m"] = vol_1m
            else:
                vol_1m = np.nan
                row["vol_1m"] = np.nan
            if len(daily_series) >= 63:
                vol_3m = float(daily_series.iloc[-63:].std() * np.sqrt(252.0))
                row["vol_3m"] = vol_3m
                row["vol_ratio"] = (vol_1m / vol_3m) if pd.notna(vol_1m) and vol_3m > 1e-9 else np.nan
            else:
                row["vol_3m"] = np.nan
                row["vol_ratio"] = np.nan

            if next_date is not None:
                next_price = monthly_close[symbol].loc[next_date]
                row["target"] = (float(next_price) / current_price) - 1.0 if pd.notna(next_price) else np.nan
            else:
                row["target"] = np.nan
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    return frame


def clean_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    df = frame.copy()
    feature_cols = [col for col in df.columns if col != "target"]
    row_threshold = max(6, int(len(feature_cols) * 0.70))
    df = df.dropna(thresh=row_threshold)
    df[feature_cols] = df.groupby(level="date")[feature_cols].transform(lambda series: series.fillna(series.median()))

    for col in feature_cols:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lo, upper=hi)

    df = df.dropna(subset=feature_cols, how="any")
    return df.sort_index()


def _fit_live_ensemble(frame: pd.DataFrame, min_train_months: int) -> list[AlphaSignal]:
    if frame.empty:
        return []

    latest_date = frame.index.get_level_values("date").max()
    live = frame.loc[frame.index.get_level_values("date") == latest_date].drop(columns=["target"], errors="ignore")
    labeled = frame.dropna(subset=["target"]).copy()
    if live.empty or labeled.empty:
        return []

    n_train_months = len(sorted(labeled.index.get_level_values("date").unique()))
    if n_train_months < int(min_train_months):
        logger.warning("ML alpha skipped: only %d train months available.", n_train_months)
        return []

    predictions = _predict_frame(labeled, live)
    if predictions.empty:
        return []

    ranked = predictions.sort_values("ensemble", ascending=False).copy()
    ranked["percentile"] = ranked["ensemble"].rank(method="average", pct=True)

    signals: list[AlphaSignal] = []
    for symbol, row in ranked.iterrows():
        percentile = float(row["percentile"])
        alpha_score = abs(percentile - 0.5) * 2.0
        direction = "up" if percentile >= 0.60 else "down" if percentile <= 0.40 else "flat"
        signals.append(
            AlphaSignal(
                symbol=str(symbol).upper(),
                predicted_return=float(row["ensemble"]),
                percentile=percentile,
                alpha_score=alpha_score,
                direction=direction,
                model_dispersion=float(row["dispersion"]),
            )
        )
    return signals


def _predict_frame(train: pd.DataFrame, live_features: pd.DataFrame) -> pd.DataFrame:
    if train.empty or live_features.empty:
        return pd.DataFrame()

    feature_cols = [col for col in train.columns if col != "target"]
    X_train = train[feature_cols]
    y_train = train["target"]
    X_live = live_features[feature_cols]

    ridge = Pipeline(
        [
            ("scaler", RobustScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    forest = RandomForestRegressor(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=1,
    )
    booster = GradientBoostingRegressor(
        random_state=42,
        n_estimators=160,
        learning_rate=0.05,
        max_depth=2,
    )

    model_predictions: dict[str, np.ndarray] = {}
    for name, model in (("ridge", ridge), ("forest", forest), ("booster", booster)):
        model.fit(X_train, y_train)
        model_predictions[name] = np.asarray(model.predict(X_live), dtype=float)

    pred_frame = pd.DataFrame(index=X_live.index.get_level_values("ticker").astype(str))
    for name, values in model_predictions.items():
        pred_frame[name] = values

    pred_frame["ensemble"] = (
        (0.20 * pred_frame["ridge"])
        + (0.45 * pred_frame["forest"])
        + (0.35 * pred_frame["booster"])
    )
    pred_frame["dispersion"] = pred_frame[["ridge", "forest", "booster"]].std(axis=1).fillna(0.0)
    pred_frame.index.name = "ticker"
    return pred_frame


def _save_cached_signals(cache_path: Path, requested_symbols: list[str], signals: list[AlphaSignal]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_symbols": requested_symbols,
        "signals": [asdict(signal) for signal in signals],
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_cached_signals(cache_path: Path, requested_symbols: list[str], max_age_minutes: int) -> list[AlphaSignal]:
    if not cache_path.exists():
        return []
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        generated = datetime.fromisoformat(str(payload.get("generated_at_utc")))
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
        age_minutes = (datetime.now(timezone.utc) - generated.astimezone(timezone.utc)).total_seconds() / 60.0
        if age_minutes > max(1, int(max_age_minutes)):
            return []

        cached_signals = [AlphaSignal(**row) for row in payload.get("signals", [])]
        cached_map = {signal.symbol: signal for signal in cached_signals}
        subset = [cached_map[symbol] for symbol in requested_symbols if symbol in cached_map]
        return subset if subset else []
    except Exception:
        return []


def _monthly_return_summary(returns: pd.Series) -> dict[str, float]:
    clean = returns.dropna()
    if clean.empty:
        return {
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }

    equity = (1.0 + clean).cumprod()
    years = max(len(clean) / 12.0, 1e-9)
    annualized_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    annualized_volatility = float(clean.std(ddof=0) * np.sqrt(12.0))
    sharpe = float((clean.mean() * 12.0) / annualized_volatility) if annualized_volatility > 1e-9 else 0.0
    running_max = equity.cummax()
    max_drawdown = float(((equity / running_max) - 1.0).min())
    win_rate = float((clean > 0).mean())
    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


def _safe_mean(values: Iterable[float]) -> float:
    numeric = [float(value) for value in values]
    return float(sum(numeric) / len(numeric)) if numeric else 0.0
