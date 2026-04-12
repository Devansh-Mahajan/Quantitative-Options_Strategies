import os
import random
import sys
import logging
import argparse
from typing import Optional
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.mega_neural_brain import MegaStrategyNet

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("Mega_Trainer")

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'mega_universe_dataset.pt')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'trading_model.pth')


def resolve_training_device() -> tuple[torch.device, bool]:
    try:
        gpu_available = torch.cuda.is_available()
    except Exception as exc:
        logger.error("❌ GPU detection failed (%s). Falling back to CPU.", exc)
        return torch.device("cpu"), False

    if gpu_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA device"
        logger.info("✅ GPU detected: %s. Training will run on CUDA.", gpu_name)
        return torch.device("cuda"), True

    logger.error("❌ No NVIDIA GPU detected. Falling back to CPU training.")
    return torch.device("cpu"), False


DEVICE, CUDA_AVAILABLE = resolve_training_device()

BATCH_SIZE = 512
EPOCHS = 120
LEARNING_RATE = 7e-4
WEIGHT_DECAY = 1e-3
TARGET_ANNUAL_RETURN = 0.05
TARGET_ACCURACY = 0.55
SEED = 42
MAX_RESTARTS = 3
OVERFIT_GAP_THRESHOLD = 0.08
MIN_GENERALIZATION_RATIO = 0.65
MAX_DYNAMIC_ADJUSTMENTS = 8
MAX_LABEL_SMOOTHING = 0.14
MIN_LEARNING_RATE = 1e-5
MAX_WEIGHT_DECAY = 1e-2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed_all(seed)


def annualized_from_horizon_return(horizon_return: torch.Tensor, horizon_days: int = 10) -> float:
    if horizon_return.numel() == 0:
        return 0.0
    avg_10d = torch.mean(horizon_return).item()
    return (1.0 + avg_10d) ** (252 / horizon_days) - 1.0


def evaluate(model, loader, criterion, future_returns):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    selected_returns = []

    with torch.no_grad():
        for b_x, b_y, b_idx in loader:
            b_x, b_y = b_x.to(DEVICE), b_y.to(DEVICE)
            logits = model(b_x)
            loss = criterion(logits, b_y)
            total_loss += loss.item()

            pred = logits.argmax(1)
            correct += (pred == b_y).sum().item()
            total += b_y.size(0)

            fr = future_returns[b_idx]
            directional_mask = (pred.cpu() == 2) | (pred.cpu() == 3)
            if directional_mask.any():
                signed = fr.clone()
                signed[pred.cpu() == 3] *= -1.0
                selected_returns.append(signed[directional_mask])

    avg_loss = total_loss / max(1, len(loader))
    acc = correct / max(1, total)
    if selected_returns:
        selected_returns = torch.cat(selected_returns)
        ann = annualized_from_horizon_return(selected_returns)
    else:
        ann = 0.0

    return avg_loss, acc, ann


def build_training_attempts() -> list[dict]:
    return [
        {
            "hidden_size": 256,
            "num_layers": 3,
            "dropout": 0.25,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "label_smoothing": 0.02,
            "patience": 10,
        },
        {
            "hidden_size": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": LEARNING_RATE * 0.6,
            "weight_decay": WEIGHT_DECAY * 1.8,
            "batch_size": 384,
            "label_smoothing": 0.04,
            "patience": 8,
        },
        {
            "hidden_size": 160,
            "num_layers": 2,
            "dropout": 0.45,
            "lr": LEARNING_RATE * 0.45,
            "weight_decay": WEIGHT_DECAY * 2.4,
            "batch_size": 320,
            "label_smoothing": 0.06,
            "patience": 7,
        },
    ]


def build_class_weights(y_train: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(y_train, minlength=4).float()
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    return weights / weights.mean()


def build_weighted_loss_for_attempt(
    y_train: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    if class_weights is None:
        if y_train is None:
            raise ValueError("Either y_train or class_weights must be provided.")
        class_weights = build_class_weights(y_train)
    return nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=label_smoothing)


def is_overfitting(train_loss: float, val_loss: float, train_acc: float, val_acc: float) -> bool:
    loss_gap = val_loss - train_loss
    if loss_gap >= OVERFIT_GAP_THRESHOLD:
        return True
    if train_acc < 1e-8:
        return False
    generalization_ratio = val_acc / train_acc
    return generalization_ratio < MIN_GENERALIZATION_RATIO and (train_acc - val_acc) > 0.06


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def build_adaptive_attempt(
    current_attempt: dict,
    train_loss: float,
    val_loss: float,
    train_acc: float,
    val_acc: float,
) -> dict:
    """
    Generate a stronger-regularization configuration when persistent overfitting is detected.
    """
    loss_gap = max(0.0, val_loss - train_loss)
    acc_gap = max(0.0, train_acc - val_acc)
    severity = clamp(max(loss_gap / OVERFIT_GAP_THRESHOLD, acc_gap / 0.06), 1.0, 2.5)
    scale = severity - 1.0

    tuned = deepcopy(current_attempt)
    tuned['hidden_size'] = max(128, int(round(current_attempt['hidden_size'] * (0.90 - 0.10 * scale))))
    tuned['num_layers'] = max(2, current_attempt['num_layers'] - (1 if severity >= 1.8 else 0))
    tuned['dropout'] = clamp(current_attempt['dropout'] + 0.06 + 0.05 * scale, 0.20, 0.60)
    tuned['lr'] = clamp(current_attempt['lr'] * (0.78 - 0.10 * scale), 1e-5, LEARNING_RATE)
    tuned['weight_decay'] = clamp(current_attempt['weight_decay'] * (1.4 + 0.5 * scale), 5e-4, 1e-2)
    tuned['batch_size'] = max(192, int(round(current_attempt['batch_size'] * (0.88 - 0.08 * scale))))
    tuned['label_smoothing'] = clamp(current_attempt['label_smoothing'] + 0.015 + 0.02 * scale, 0.0, 0.12)
    tuned['patience'] = max(4, current_attempt['patience'] - 1)
    tuned['adaptive_from_overfit'] = True
    tuned['overfit_severity'] = severity
    return tuned


def compute_overfit_pressure(
    train_loss: float,
    val_loss: float,
    train_acc: float,
    val_acc: float,
    overfit_streak: int,
    stale: int,
) -> float:
    loss_gap = max(0.0, val_loss - train_loss)
    acc_gap = max(0.0, train_acc - val_acc)
    loss_pressure = loss_gap / max(OVERFIT_GAP_THRESHOLD, 1e-6)
    acc_pressure = acc_gap / 0.06
    streak_pressure = min(1.5, overfit_streak * 0.35)
    stale_pressure = min(1.0, stale * 0.10)
    return clamp(0.45 * loss_pressure + 0.35 * acc_pressure + 0.15 * streak_pressure + 0.05 * stale_pressure, 0.1, 2.0)


def apply_dynamic_regularization(
    optimizer: optim.Optimizer,
    attempt: dict,
    class_weights: torch.Tensor,
    pressure: float,
) -> nn.Module:
    lr_mult = clamp(1.0 - 0.12 * pressure, 0.65, 0.98)
    wd_mult = clamp(1.0 + 0.22 * pressure, 1.03, 1.65)
    smooth_add = 0.006 + 0.010 * pressure

    attempt['lr'] = clamp(attempt['lr'] * lr_mult, MIN_LEARNING_RATE, LEARNING_RATE)
    attempt['weight_decay'] = clamp(attempt['weight_decay'] * wd_mult, WEIGHT_DECAY, MAX_WEIGHT_DECAY)
    attempt['label_smoothing'] = clamp(attempt['label_smoothing'] + smooth_add, 0.0, MAX_LABEL_SMOOTHING)
    attempt['patience'] = max(4, int(round(attempt['patience'] - pressure * 0.35)))

    for group in optimizer.param_groups:
        group['lr'] = attempt['lr']
        group['weight_decay'] = attempt['weight_decay']

    return build_weighted_loss_for_attempt(
        label_smoothing=attempt['label_smoothing'],
        class_weights=class_weights,
    )


def train(target_annual_return: Optional[float] = None, target_accuracy: float = TARGET_ACCURACY):
    set_seed(SEED)
    logger.info(f"🚀 Training on {DEVICE} | seed={SEED}")

    checkpoint = torch.load(DATA_PATH, weights_only=False)
    X, y = checkpoint['X'], checkpoint['y']
    logger.info("📦 Dataset loaded: samples=%d, sequence_len=%d, features=%d", X.shape[0], X.shape[1], X.shape[2])
    future_returns = checkpoint.get('future_returns', torch.zeros(len(y), dtype=torch.float32))
    target_annual = float(target_annual_return if target_annual_return is not None else checkpoint.get('target_annual_return', TARGET_ANNUAL_RETURN))

    split = int(len(X) * 0.8)
    train_idx = torch.arange(0, split)
    val_idx = torch.arange(split, len(X))

    train_ds = TensorDataset(X[:split], y[:split], train_idx)
    val_ds = TensorDataset(X[split:], y[split:], val_idx)
    logger.info("🧪 Train/validation split: %d / %d", len(train_ds), len(val_ds))

    input_dim = X.shape[2]
    attempts = build_training_attempts()[:MAX_RESTARTS]
    global_best_score = -float('inf')
    overfit_restart_requested = False
    last_metrics = None

    for attempt_idx, attempt in enumerate(attempts, start=1):
        if overfit_restart_requested and last_metrics is not None and attempt_idx > 1:
            adaptive_attempt = build_adaptive_attempt(attempt, *last_metrics)
            logger.info(
                "🧪 Adaptive tuning from overfit signal: severity=%.2f dropout=%.2f lr=%.6f wd=%.6f batch=%d",
                adaptive_attempt['overfit_severity'],
                adaptive_attempt['dropout'],
                adaptive_attempt['lr'],
                adaptive_attempt['weight_decay'],
                adaptive_attempt['batch_size'],
            )
            attempt = adaptive_attempt
            overfit_restart_requested = False

        set_seed(SEED + attempt_idx)
        pin_memory = DEVICE.type == 'cuda'
        train_loader = DataLoader(
            train_ds,
            batch_size=attempt['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=attempt['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=pin_memory,
        )
        logger.info(
            "⚙️ Attempt %d/%d | hidden=%d layers=%d dropout=%.2f lr=%.6f wd=%.6f batch=%d",
            attempt_idx,
            len(attempts),
            attempt['hidden_size'],
            attempt['num_layers'],
            attempt['dropout'],
            attempt['lr'],
            attempt['weight_decay'],
            attempt['batch_size'],
        )
        logger.info("⚙️ Dataloaders ready: train_batches=%d, val_batches=%d", len(train_loader), len(val_loader))

        model = MegaStrategyNet(
            input_size=input_dim,
            hidden_size=attempt['hidden_size'],
            num_layers=attempt['num_layers'],
            num_classes=4,
            dropout=attempt['dropout'],
        ).to(DEVICE)
        class_weights = build_class_weights(y[:split])
        criterion = build_weighted_loss_for_attempt(class_weights=class_weights, label_smoothing=attempt['label_smoothing'])
        optimizer = optim.AdamW(model.parameters(), lr=attempt['lr'], weight_decay=attempt['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=12, T_mult=2)
        scaler = torch.amp.GradScaler(device="cuda", enabled=(DEVICE.type == 'cuda'))
        logger.info("🧠 Model initialized. Starting optimization...")

        stale = 0
        best_score_this_attempt = -float('inf')
        overfit_streak = 0
        dynamic_adjustments = 0

        for epoch in range(EPOCHS):
            model.train()
            running_loss, running_correct, seen = 0.0, 0, 0
            total_batches = max(1, len(train_loader))
            progress_interval = max(1, total_batches // 10)

            for batch_idx, (b_x, b_y, _) in enumerate(train_loader, start=1):
                b_x, b_y = b_x.to(DEVICE), b_y.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == 'cuda')):
                    logits = model(b_x)
                    loss = criterion(logits, b_y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                preds = logits.argmax(1)
                running_correct += (preds == b_y).sum().item()
                seen += b_y.size(0)

                if batch_idx % progress_interval == 0 or batch_idx == total_batches:
                    pct = int((batch_idx / total_batches) * 100)
                    logger.info("⏱️ Epoch %03d progress: %d%% (%d/%d batches)", epoch + 1, pct, batch_idx, total_batches)

            scheduler.step(epoch + 1)
            train_loss = running_loss / max(1, len(train_loader))
            train_acc = running_correct / max(1, seen)
            val_loss, val_acc, val_ann = evaluate(model, val_loader, criterion, future_returns)
            score = (1.0 - val_loss) + min(val_acc, target_accuracy) * 2.0 + min(val_ann, target_annual) * 2.0

            logger.info(
                "Attempt %d Epoch %03d | train_loss=%.4f val_loss=%.4f train_acc=%.2f%% val_acc=%.2f%% implied_ann=%.2f%%",
                attempt_idx,
                epoch + 1,
                train_loss,
                val_loss,
                train_acc * 100,
                val_acc * 100,
                val_ann * 100,
            )

            if score > best_score_this_attempt:
                best_score_this_attempt = score
                stale = 0
            else:
                stale += 1

            if score > global_best_score:
                global_best_score = score
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'input_size': input_dim,
                        'hidden_size': attempt['hidden_size'],
                        'num_layers': attempt['num_layers'],
                        'dropout': attempt['dropout'],
                        'scaler': checkpoint['scaler'],
                        'features_list': checkpoint['features_list'],
                        'target_annual_return': target_annual,
                        'target_accuracy': target_accuracy,
                        'best_val_loss': val_loss,
                        'best_val_acc': val_acc,
                        'best_implied_annual_return': val_ann,
                        'selected_attempt': attempt_idx,
                    },
                    MODEL_SAVE_PATH,
                )

            if is_overfitting(train_loss, val_loss, train_acc, val_acc):
                overfit_streak += 1
                logger.warning(
                    "⚠️ Overfitting signal detected (attempt=%d epoch=%d streak=%d).",
                    attempt_idx,
                    epoch + 1,
                    overfit_streak,
                )
                if dynamic_adjustments < MAX_DYNAMIC_ADJUSTMENTS:
                    pressure = compute_overfit_pressure(
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_acc=train_acc,
                        val_acc=val_acc,
                        overfit_streak=overfit_streak,
                        stale=stale,
                    )
                    criterion = apply_dynamic_regularization(
                        optimizer=optimizer,
                        attempt=attempt,
                        class_weights=class_weights,
                        pressure=pressure,
                    )
                    dynamic_adjustments += 1
                    logger.warning(
                        "🧭 Dynamic regulation applied (attempt=%d epoch=%d adjust=%d/%d pressure=%.2f lr=%.6f wd=%.6f smooth=%.3f patience=%d).",
                        attempt_idx,
                        epoch + 1,
                        dynamic_adjustments,
                        MAX_DYNAMIC_ADJUSTMENTS,
                        pressure,
                        attempt['lr'],
                        attempt['weight_decay'],
                        attempt['label_smoothing'],
                        attempt['patience'],
                    )
                if overfit_streak >= 2:
                    logger.warning("🔁 Restarting with adjusted hyperparameters.")
                    overfit_restart_requested = True
                    last_metrics = (train_loss, val_loss, train_acc, val_acc)
                    break
            else:
                overfit_streak = 0

            if stale >= attempt['patience']:
                logger.warning("🛑 Early stopping triggered for attempt %d (no validation improvement).", attempt_idx)
                break

        if global_best_score > -float('inf') and best_score_this_attempt > -float('inf') and overfit_streak == 0:
            logger.info("✅ Attempt %d completed without persistent overfitting.", attempt_idx)
            break

    if global_best_score == -float('inf'):
        logger.error("Training failed before producing a checkpoint.")
        return

    logger.info("✅ Training complete. Checkpoint saved to %s", MODEL_SAVE_PATH)
    if checkpoint.get('future_returns') is not None and target_annual > 0:
        logger.info("🎯 Strategy target annual return floor used in model selection: %.2f%%", target_annual * 100)
        logger.info("🎯 Strategy target validation accuracy floor: %.2f%%", target_accuracy * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mega strategy network.")
    parser.add_argument("--target-annual-return", type=float, default=TARGET_ANNUAL_RETURN)
    parser.add_argument("--target-accuracy", type=float, default=TARGET_ACCURACY)
    args = parser.parse_args()
    train(target_annual_return=args.target_annual_return, target_accuracy=args.target_accuracy)
