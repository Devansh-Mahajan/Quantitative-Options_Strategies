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
from core.telemetry.torch_device import resolve_torch_runtime

setup_logging(cfg.log_dir, "INFO")
log = logging.getLogger("scripts.weekend_training")


# --------------------------------------------------------------------------- #
# Data download
# --------------------------------------------------------------------------- #

async def download_data(symbols: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for a mixed crypto + stock universe."""
    from backtester.data_loader import load_multi

    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=lookback_days)
    raw = await asyncio.to_thread(
        load_multi,
        symbols,
        "1h",
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )

    data: dict[str, pd.DataFrame] = {}
    for symbol, df in raw.items():
        if df is None or df.empty:
            log.warning("Failed to download %s: empty dataset", symbol)
            continue
        keep_cols = [c for c in ("open", "high", "low", "close", "volume", "taker_buy_base") if c in df.columns]
        data[symbol] = df.loc[:, keep_cols].copy()
        log.info("Downloaded %d bars for %s", len(df), symbol)
    return data


# --------------------------------------------------------------------------- #
# Feature + target construction
# --------------------------------------------------------------------------- #

def build_training_data(df: pd.DataFrame, funding: float = 0.0, oi: float = 0.0):
    from ml.features import FEATURE_NAMES, compute_features

    feat_df = compute_features(df, funding_rate=funding, open_interest=oi)
    close = df["close"].astype(float)
    targets = pd.DataFrame(
        {
            "target_1h": ((close.shift(-1) / close) - 1).clip(-0.5, 0.5),
            "target_4h": ((close.shift(-4) / close) - 1).clip(-0.5, 0.5),
            "target_24h": ((close.shift(-24) / close) - 1).clip(-0.5, 0.5),
        },
        index=df.index,
    )
    aligned = feat_df.join(targets, how="inner").dropna()
    X = aligned.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = aligned.loc[:, ["target_1h", "target_4h", "target_24h"]].to_numpy(dtype=np.float32)
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

def build_rl_training_set(
    data: dict[str, pd.DataFrame],
    rl_symbols: list[str],
    rl_months: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
    """
    Per-strategy return series from single-strategy backtests over real data.

    Runs the BacktestEngine once per canonical strategy (each run whitelisted
    to that one strategy), aligns the resulting equity curves, and returns
    (returns T×n, features T×k, regime_idx T, strategy_names). Regime labels
    come from RegimeHMM.decode_causal (no lookahead). Strategies that never
    trade contribute flat (zero) return columns — the policy learns to ignore
    them rather than being fed fabricated noise.
    """
    from backtester.engine import BacktestEngine
    from ml.regime_hmm import RegimeHMM
    from strategies.registry import _build_all_enabled, canonical_strategy_names

    strategy_names = canonical_strategy_names()
    rl_data = {}
    max_bars = max(1, int(rl_months * 30 * 24))
    for sym in rl_symbols:
        df = data.get(sym)
        if df is not None and len(df) >= 500:
            rl_data[sym] = df.tail(max_bars)
    if not rl_data:
        return None, None, None, strategy_names

    equity_curves: dict[str, pd.Series] = {}
    for name in strategy_names:
        try:
            instances = [s for s in _build_all_enabled() if s.name == name]
            if not instances:
                continue
            engine = BacktestEngine(data=rl_data, interval="1h", strategies=instances)
            result = engine.run()
            equity_curves[name] = result.equity_curve
        except Exception as exc:
            log.warning("  RL returns backtest failed for strategy %s: %s", name, exc)

    curves = {n: c for n, c in equity_curves.items() if c is not None and len(c) > 100}
    if len(curves) < max(3, len(strategy_names) // 4):
        return None, None, None, strategy_names

    frame = pd.DataFrame(curves).ffill().dropna()
    rets = frame.pct_change().fillna(0.0)
    # Canonical column order; missing strategies (failed backtest) = flat zero.
    returns_matrix = np.column_stack([
        rets[name].to_numpy(dtype=np.float32) if name in rets.columns else np.zeros(len(rets), dtype=np.float32)
        for name in strategy_names
    ])

    # Market features from the anchor symbol, aligned to the returns index.
    anchor_sym = next(iter(rl_data))
    anchor = rl_data[anchor_sym]
    closes = anchor["close"].astype(float)
    feat = pd.DataFrame(index=anchor.index)
    feat["ret_1"] = closes.pct_change()
    feat["ret_24"] = closes.pct_change(24)
    feat["vol_24"] = closes.pct_change().rolling(24).std()
    feat["mom_72"] = closes.pct_change(72)
    feat["rng"] = ((anchor["high"] - anchor["low"]) / closes).rolling(24).mean()
    feat = feat.fillna(0.0).clip(-10, 10)
    feature_matrix = (
        feat.reindex(rets.index).ffill().fillna(0.0).to_numpy(dtype=np.float32)
        if not feat.empty
        else np.zeros((len(rets), 5), dtype=np.float32)
    )

    # Causal regime labels for the same window.
    regime_map = {"bull": 0, "bear": 1, "ranging": 2, "volatile": 3}
    try:
        hmm = RegimeHMM(n_states=cfg.hmm_states)
        hmm.fit(anchor)
        labels = hmm.decode_causal(anchor)
        label_series = pd.Series(labels, index=anchor.index).reindex(rets.index).ffill()
        regime_idx = np.array([regime_map.get(r, 2) for r in label_series], dtype=np.int32)
    except Exception as exc:
        log.warning("  RL regime labeling failed (%s) — defaulting to 'ranging'", exc)
        regime_idx = np.full(len(rets), 2, dtype=np.int32)

    return returns_matrix, feature_matrix, regime_idx, strategy_names


async def run_training(rl_months: int = 12, rl_symbols_cap: int = 4) -> None:
    from ml.regime_hmm import RegimeHMM
    from ml.price_lstm import PricePredictorTrainer
    from ml.rl_allocator import RLAllocator, StrategyAllocEnv
    from ml.vol_model import GARCHVolModel
    from ml.features import FEATURE_NAMES

    runtime = resolve_torch_runtime()
    log.info("=" * 60)
    log.info(
        "WEEKEND TRAINING PIPELINE — target=%s resolved=%s (%s)",
        cfg.device,
        runtime.device.type,
        runtime.accelerator_name,
    )
    log.info("=" * 60)

    await notifications.alert("Weekend training started", title="Training", level="INFO")

    symbols = cfg.model_symbols
    lookback = cfg.retrain_lookback_days
    rl_symbols = [s for s in symbols][: max(1, rl_symbols_cap)]

    # 1. Download data
    log.info("[1/8] Downloading %d days of 1h data for %d symbols...", lookback, len(symbols))
    data = await download_data(symbols, lookback_days=lookback)

    if not data:
        log.error("No data downloaded — aborting training")
        return

    training_dir = cfg.training_dir
    training_dir.mkdir(parents=True, exist_ok=True)

    # 2. Retrain HMM for each symbol
    log.info("[2/8] Training HMM regime detectors...")
    for symbol, df in data.items():
        try:
            hmm = RegimeHMM(n_states=cfg.hmm_states)
            hmm.fit(df)
            hmm.save(training_dir / f"hmm_{symbol}.pkl")
            log.info("  HMM saved for %s", symbol)
        except Exception as exc:
            log.warning("  HMM training failed for %s: %s", symbol, exc)

    # 3. Retrain LSTM price predictors
    log.info("[3/8] Training LSTM price predictors (GPU)...")
    n_features = len(FEATURE_NAMES)
    all_X, all_y = [], []

    for symbol in cfg.model_symbols:
        try:
            df = data.get(symbol)
            if df is None:
                continue
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
    log.info("[4/8] Fitting GARCH volatility models...")
    for symbol in cfg.model_symbols:
        try:
            df = data.get(symbol)
            if df is None:
                continue
            rets = df["close"].astype(float).pct_change().dropna()
            garch = GARCHVolModel()
            garch.fit(rets)
            garch.save(training_dir / f"garch_{symbol}.pkl")
            log.info("  %s GARCH fitted — current vol=%.1f%%", symbol, garch.current_vol * 100)
        except Exception as exc:
            log.warning("  GARCH failed for %s: %s", symbol, exc)

    # 5. Retrain RL strategy allocator on REAL per-strategy backtest returns.
    # (The previous implementation trained PPO on np.random.randn mock returns
    # and a constant regime — a policy learned from noise. Models produced that
    # way carried no metadata and are now rejected at load time.)
    log.info("[5/8] Training RL strategy allocator (PPO) on real strategy returns...")
    try:
        returns_matrix, feature_matrix, regime_idx, strategy_names = build_rl_training_set(
            data,
            rl_symbols=rl_symbols,
            rl_months=rl_months,
        )
        if returns_matrix is None:
            log.warning("  RL training skipped — could not build a usable returns matrix")
        else:
            allocator = RLAllocator(len(strategy_names))
            allocator.train(returns_matrix, feature_matrix, regime_idx, timesteps=cfg.rl_timesteps)
            allocator.save(
                training_dir / "rl_allocator",
                metadata={
                    "strategy_names": strategy_names,
                    "trained_on": "backtest_returns",
                    "trained_at_utc": pd.Timestamp.utcnow().isoformat(),
                    "n_strategies": len(strategy_names),
                    "bars": int(len(returns_matrix)),
                    "symbols": rl_symbols,
                },
            )
            log.info("  RL allocator saved (%d strategies × %d bars of real returns)",
                     len(strategy_names), len(returns_matrix))
    except Exception as exc:
        log.warning("  RL training failed: %s", exc)

    # 6. Train MegaStrategyNet (NeuralSelector backbone)
    log.info("[6/8] Training MegaStrategyNet macro classifier (GPU)...")
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from core.ml.mega_neural_brain import MegaStrategyNet
        from ml.regime_hmm import RegimeHMM

        mega_X_list, mega_y_list = [], []
        for symbol in cfg.model_symbols:
            df = data.get(symbol)
            if df is None:
                continue
            X, y = build_training_data(df)
            if len(X) < 100:
                continue
            # Derive per-bar macro label from HMM regime + return direction
            # Label: 0=THETA (range), 1=VEGA (explosive), 2=BULL, 3=BEAR
            hmm = RegimeHMM(n_states=cfg.hmm_states)
            hmm.fit(df)
            # decode_causal: labels at bar t use only data <= t. The old
            # predict_sequence used global Viterbi, leaking future bars into
            # training labels.
            regime_seq = hmm.decode_causal(df)  # list of str per bar
            regime_map = {"ranging": 0, "volatile": 1, "bull": 2, "bear": 3}
            if regime_seq is None or len(regime_seq) < len(X):
                labels = np.zeros(len(X), dtype=np.int64)
            else:
                labels = np.array(
                    [regime_map.get(r, 0) for r in regime_seq[-len(X):]],
                    dtype=np.int64,
                )
            mega_X_list.append(X)
            mega_y_list.append(labels)

        if mega_X_list:
            X_mega = np.vstack(mega_X_list)
            y_mega = np.concatenate(mega_y_list)
            n_features_mega = X_mega.shape[1]
            seq_len = min(64, len(X_mega) // 4)

            # Build sequences (T, seq_len, n_features)
            Xs_mega, ys_mega = [], []
            for i in range(seq_len, len(X_mega)):
                Xs_mega.append(X_mega[i - seq_len: i])
                ys_mega.append(y_mega[i])
            Xs_mega = np.array(Xs_mega, dtype=np.float32)
            ys_mega = np.array(ys_mega, dtype=np.int64)

            split = int(len(Xs_mega) * 0.85)
            t_X = torch.tensor(Xs_mega[:split])
            t_y = torch.tensor(ys_mega[:split])
            dataset = TensorDataset(t_X, t_y)
            loader = DataLoader(dataset, batch_size=256, shuffle=True)

            device = cfg.device if runtime.device.type != "cpu" else "cpu"
            mega_net = MegaStrategyNet(input_size=n_features_mega).to(device)
            optimizer = optim.AdamW(mega_net.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            mega_net.train()
            for epoch in range(30):
                epoch_loss = 0.0
                for batch_X, batch_y in loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()
                    out = mega_net(batch_X)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(mega_net.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                if (epoch + 1) % 10 == 0:
                    log.info("  MegaNet epoch %d/%d loss=%.4f", epoch + 1, 30, epoch_loss / max(len(loader), 1))

            save_path = training_dir / "mega_net.pt"
            torch.save({
                "state_dict": mega_net.state_dict(),
                "input_size": n_features_mega,
            }, save_path)
            log.info("  MegaStrategyNet saved to %s", save_path)
        else:
            log.warning("  MegaStrategyNet: no training data collected")
    except Exception as exc:
        log.warning("  MegaStrategyNet training failed: %s", exc)

    # 7. Generate alpha cache (ml_alpha ensemble snapshot)
    log.info("[7/8] Generating ML alpha cache...")
    try:
        from core.ml.ml_alpha import generate_live_alpha_signals
        from ml.alpha_cache import DEFAULT_CACHE_PATH

        alpha_signals = generate_live_alpha_signals(
            symbols=cfg.model_symbols,
            cache_path=DEFAULT_CACHE_PATH,
            max_age_minutes=0,   # force regenerate
        )
        log.info("  Alpha cache: %d signals written to %s", len(alpha_signals), DEFAULT_CACHE_PATH)
    except Exception as exc:
        log.warning("  Alpha cache generation failed: %s", exc)

    # 8. Deploy: copy training_dir models -> model_dir (live)
    log.info("[8/8] Deploying new models to live directory...")
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

    log.info("Training pipeline complete")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Weekend GPU training pipeline.")
    parser.add_argument("--rl-months", type=int, default=12, help="months of data for RL returns backtests")
    parser.add_argument("--rl-symbols", type=int, default=4, help="max symbols for RL returns backtests")
    args = parser.parse_args()

    try:
        asyncio.run(run_training(rl_months=args.rl_months, rl_symbols_cap=args.rl_symbols))
    except KeyboardInterrupt:
        log.info("Training interrupted")
    except Exception as exc:
        log.critical("Training pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
