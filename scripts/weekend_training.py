"""
Weekend self-learning pipeline (GPU-accelerated, RTX 5090).
Downloads fresh historical data, retrains all ML models, validates, then deploys.

Schedule via cron:
    0 1 * * 6  cd /path/to/bot && /path/to/venv/bin/python -m scripts.weekend_training

Usage:
    python -m scripts.weekend_training
    # or:
    weekend-train
"""

from __future__ import annotations
import asyncio
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bot.config import cfg
from bot.logger import setup_logging
from bot import state, notifications

setup_logging(cfg.log_dir, "INFO")
log = logging.getLogger("scripts.weekend_training")


# --------------------------------------------------------------------------- #
# Data download
# --------------------------------------------------------------------------- #

async def download_data(client, symbols: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for all symbols from Binance."""
    data: dict[str, pd.DataFrame] = {}
    limit = lookback_days * 24  # hourly candles
    for symbol in symbols:
        try:
            klines = await client.get_klines(symbol, interval="1h", limit=min(limit, 1500), futures=True)
            df = pd.DataFrame(klines)
            df = df[["open_time", "open", "high", "low", "close", "volume", "taker_buy_base"]]
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("open_time", inplace=True)
            data[symbol] = df
            log.info("Downloaded %d bars for %s", len(df), symbol)
        except Exception as exc:
            log.warning("Failed to download %s: %s", symbol, exc)
    return data


# --------------------------------------------------------------------------- #
# Feature + target construction
# --------------------------------------------------------------------------- #

def build_training_data(df: pd.DataFrame, funding: float = 0.0, oi: float = 0.0):
    from ml.features import compute_feature_matrix

    X = compute_feature_matrix(df, funding_rate=funding, open_interest=oi)
    closes = df["close"].astype(float).values

    # Forward returns at 1h, 4h, 24h (clipped to prevent outlier labels)
    n = len(X)
    max_idx = len(closes) - 25  # need 24 bars ahead
    X = X[:max_idx]

    y = np.zeros((max_idx, 3), dtype=np.float32)
    for i in range(max_idx):
        raw_idx = len(df) - n + i
        if raw_idx + 24 < len(closes):
            y[i, 0] = np.clip((closes[raw_idx + 1] / closes[raw_idx]) - 1, -0.5, 0.5)   # 1h
            y[i, 1] = np.clip((closes[raw_idx + 4] / closes[raw_idx]) - 1, -0.5, 0.5)   # 4h
            y[i, 2] = np.clip((closes[raw_idx + 24] / closes[raw_idx]) - 1, -0.5, 0.5)  # 24h

    return X, y


# --------------------------------------------------------------------------- #
# Model validation
# --------------------------------------------------------------------------- #

def validate_lstm(predictor, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    import torch
    from ml.features import build_sequences

    Xs, ys = build_sequences(X_val, y_val, lookback=cfg.lstm_lookback)
    if len(Xs) == 0:
        return {"mae": 999.0}

    tensor = torch.tensor(Xs[:200], dtype=torch.float32, device=predictor.device)
    predictor.model.eval()
    with torch.no_grad():
        preds = predictor.model(tensor).cpu().numpy()
    mae = float(np.abs(preds - ys[:200]).mean())
    dir_acc = float((np.sign(preds[:, 0]) == np.sign(ys[:200, 0])).mean())
    return {"mae": mae, "dir_acc_1h": dir_acc}


# --------------------------------------------------------------------------- #
# Main training pipeline
# --------------------------------------------------------------------------- #

async def run_training() -> None:
    from exchange.client import BinanceClient
    from ml.regime_hmm import RegimeHMM
    from ml.price_lstm import PricePredictorTrainer
    from ml.rl_allocator import RLAllocator, StrategyAllocEnv
    from ml.vol_model import GARCHVolModel
    from ml.features import FEATURE_NAMES

    log.info("=" * 60)
    log.info("WEEKEND TRAINING PIPELINE — GPU=%s", cfg.device)
    log.info("=" * 60)

    await notifications.alert("Weekend training started", title="Training", level="INFO")

    client = BinanceClient()
    await client.start()

    symbols = cfg.futures_symbols
    lookback = cfg.retrain_lookback_days

    # 1. Download data
    log.info("[1/6] Downloading %d days of 1h data for %d symbols...", lookback, len(symbols))
    data = await download_data(client, symbols, lookback_days=lookback)

    if not data:
        log.error("No data downloaded — aborting training")
        return

    training_dir = cfg.training_dir
    training_dir.mkdir(parents=True, exist_ok=True)

    # 2. Retrain HMM for each symbol
    log.info("[2/6] Training HMM regime detectors...")
    for symbol, df in data.items():
        try:
            hmm = RegimeHMM(n_states=cfg.hmm_states)
            hmm.fit(df)
            hmm.save(training_dir / f"hmm_{symbol}.pkl")
            log.info("  HMM saved for %s", symbol)
        except Exception as exc:
            log.warning("  HMM training failed for %s: %s", symbol, exc)

    # 3. Retrain LSTM price predictors
    log.info("[3/6] Training LSTM price predictors (GPU)...")
    n_features = len(FEATURE_NAMES)
    all_X, all_y = [], []

    for symbol, df in list(data.items())[:4]:  # top-4 symbols
        try:
            X, y = build_training_data(df)
            if len(X) < 200:
                continue
            split = int(len(X) * 0.8)
            X_tr, y_tr = X[:split], y[:split]
            X_val, y_val = X[split:], y[split:]

            predictor = PricePredictorTrainer(n_features)
            history = predictor.fit(X_tr, y_tr, epochs=80, batch_size=512)

            metrics = validate_lstm(predictor, X_val, y_val)
            log.info("  %s LSTM — val MAE=%.5f dir_acc=%.1f%%",
                     symbol, metrics["mae"], metrics.get("dir_acc_1h", 0) * 100)

            # Save even if not perfect; production bot will use it
            predictor.save(training_dir / f"lstm_{symbol}.pt")

            # Collect returns for RL training
            all_X.append(X_tr)
            all_y.append(y_tr)

        except Exception as exc:
            log.warning("  LSTM training failed for %s: %s", symbol, exc)

    # 4. Retrain GARCH volatility models
    log.info("[4/6] Fitting GARCH volatility models...")
    for symbol, df in list(data.items())[:4]:
        try:
            rets = df["close"].astype(float).pct_change().dropna()
            garch = GARCHVolModel()
            garch.fit(rets)
            garch.save(training_dir / f"garch_{symbol}.pkl")
            log.info("  %s GARCH fitted — current vol=%.1f%%", symbol, garch.current_vol * 100)
        except Exception as exc:
            log.warning("  GARCH failed for %s: %s", symbol, exc)

    # 5. Retrain RL strategy allocator
    log.info("[5/6] Training RL strategy allocator (PPO)...")
    try:
        if all_X:
            X_combined = np.vstack(all_X)
            # Mock per-strategy returns (in prod, use actual strategy PnL history)
            from strategies.registry import build_registry
            strategies = build_registry()
            n_strats = len(strategies)
            # Proxy: each strategy return = some function of features
            T = len(X_combined)
            mock_rets = np.random.randn(T, n_strats) * 0.001  # small random for now

            # Regime index (0=bull,1=bear,2=ranging,3=volatile) — use HMM output
            mock_regimes = np.zeros(T, dtype=np.int32)

            allocator = RLAllocator(n_strats)
            allocator.train(mock_rets, X_combined, mock_regimes, timesteps=cfg.rl_timesteps)
            allocator.save(training_dir / "rl_allocator")
            log.info("  RL allocator saved")
    except Exception as exc:
        log.warning("  RL training failed: %s", exc)

    # 6. Deploy: copy training_dir models -> model_dir (live)
    log.info("[6/6] Deploying new models to live directory...")
    live_dir = cfg.model_dir
    live_dir.mkdir(parents=True, exist_ok=True)

    deployed = 0
    for model_file in training_dir.glob("*"):
        try:
            dest = live_dir / model_file.name
            shutil.copy2(model_file, dest)
            deployed += 1
        except Exception as exc:
            log.warning("  Failed to copy %s: %s", model_file.name, exc)

    log.info("Deployed %d model files to %s", deployed, live_dir)

    state.set_model_meta("last_training_run", pd.Timestamp.now().isoformat())
    state.set_model_meta("training_symbols", symbols)
    state.set_model_meta("training_lookback_days", lookback)

    await notifications.alert(
        f"Weekend training complete — deployed {deployed} model files",
        title="Training Complete",
        level="INFO",
    )

    await client.close()
    log.info("Training pipeline complete")


def main() -> None:
    try:
        asyncio.run(run_training())
    except KeyboardInterrupt:
        log.info("Training interrupted")
    except Exception as exc:
        log.critical("Training pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
