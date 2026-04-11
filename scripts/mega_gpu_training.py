import os
import random
import sys
import logging
import argparse
from typing import Optional

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512
EPOCHS = 120
LEARNING_RATE = 7e-4
WEIGHT_DECAY = 1e-3
TARGET_ANNUAL_RETURN = 0.05
TARGET_ACCURACY = 0.55
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def annualized_from_horizon_return(horizon_return: torch.Tensor, horizon_days: int = 10) -> float:
    if horizon_return.numel() == 0:
        return 0.0
    avg_10d = torch.mean(horizon_return).item()
    return (1.0 + avg_10d) ** (252 / horizon_days) - 1.0


def build_weighted_loss(y_train: torch.Tensor) -> nn.Module:
    counts = torch.bincount(y_train, minlength=4).float()
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()
    return nn.CrossEntropyLoss(weight=weights.to(DEVICE))


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


def train(target_annual_return: Optional[float] = None, target_accuracy: float = TARGET_ACCURACY):
    set_seed(SEED)
    logger.info(f"🚀 Training on {DEVICE} | seed={SEED}")

    checkpoint = torch.load(DATA_PATH, weights_only=False)
    X, y = checkpoint['X'], checkpoint['y']
    future_returns = checkpoint.get('future_returns', torch.zeros(len(y), dtype=torch.float32))
    target_annual = float(target_annual_return if target_annual_return is not None else checkpoint.get('target_annual_return', TARGET_ANNUAL_RETURN))

    split = int(len(X) * 0.8)
    train_idx = torch.arange(0, split)
    val_idx = torch.arange(split, len(X))

    train_ds = TensorDataset(X[:split], y[:split], train_idx)
    val_ds = TensorDataset(X[split:], y[split:], val_idx)

    pin_memory = DEVICE.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin_memory)

    input_dim = X.shape[2]
    model = MegaStrategyNet(input_size=input_dim, hidden_size=256, num_layers=3, num_classes=4).to(DEVICE)

    criterion = build_weighted_loss(y[:split])
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(DEVICE.type == 'cuda'))

    best_score = -float('inf')
    patience, stale = 10, 0

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0

        for b_x, b_y, _ in train_loader:
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
            running += loss.item()

        scheduler.step(epoch + 1)

        val_loss, val_acc, val_ann = evaluate(model, val_loader, criterion, future_returns)
        train_loss = running / max(1, len(train_loader))

        score = (1.0 - val_loss) + min(val_acc, target_accuracy) * 2.0 + min(val_ann, target_annual) * 2.0
        logger.info(
            "Epoch %03d | train_loss=%.4f val_loss=%.4f val_acc=%.2f%% implied_ann=%.2f%%",
            epoch + 1,
            train_loss,
            val_loss,
            val_acc * 100,
            val_ann * 100,
        )

        if score > best_score:
            best_score = score
            stale = 0
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'input_size': input_dim,
                    'hidden_size': 256,
                    'num_layers': 3,
                    'scaler': checkpoint['scaler'],
                    'features_list': checkpoint['features_list'],
                    'target_annual_return': target_annual,
                    'target_accuracy': target_accuracy,
                    'best_val_loss': val_loss,
                    'best_val_acc': val_acc,
                    'best_implied_annual_return': val_ann,
                },
                MODEL_SAVE_PATH,
            )
        else:
            stale += 1
            if stale >= patience:
                logger.warning("🛑 Early stopping triggered (no validation improvement).")
                break

    if best_score == -float('inf'):
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
