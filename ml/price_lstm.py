"""
Bidirectional LSTM + multi-head attention price predictor.
Predicts next-bar return direction + magnitude for 1h, 4h, 24h horizons.
GPU-accelerated (RTX 5090 / CUDA).
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler

from bot.config import cfg
from core.telemetry.torch_device import resolve_torch_runtime

log = logging.getLogger("ml.price_lstm")


# --------------------------------------------------------------------------- #
# Model architecture
# --------------------------------------------------------------------------- #

class AttentionLayer(nn.Module):
    def __init__(self, hidden: int, heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


class CryptoPricePredictor(nn.Module):
    """
    Input:  (batch, seq_len, n_features)
    Output: (batch, 3)  — predicted returns at 1h, 4h, 24h horizons
    """

    def __init__(self, n_features: int, seq_len: int = 50, hidden: int = 256, layers: int = 3) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features

        self.input_proj = nn.Linear(n_features, hidden)
        self.lstm = nn.LSTM(
            hidden, hidden, layers,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        # bidirectional => hidden * 2
        self.attn = AttentionLayer(hidden * 2, heads=8)
        self.dropout = nn.Dropout(0.3)

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),   # 1h, 4h, 24h
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.input_proj(x))
        x, _ = self.lstm(x)
        x = self.attn(x)
        x = x[:, -1, :]        # last timestep
        x = self.dropout(x)
        return self.head(x)

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 30
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MC-Dropout uncertainty estimate. Returns (mean, std) both shape (batch, 3)."""
        self.train()           # enable dropout
        with torch.no_grad():
            preds = torch.stack([self(x) for _ in range(n_samples)], dim=0)
        self.eval()
        return preds.mean(0), preds.std(0)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

class PricePredictorTrainer:
    def __init__(self, n_features: int) -> None:
        self.runtime = resolve_torch_runtime()
        self.device = self.runtime.device
        self.scaler = RobustScaler()
        self.model = CryptoPricePredictor(n_features).to(self.device)
        log.info("PricePredictor on device: %s", self.device)

    def fit(
        self,
        X: np.ndarray,  # (T, F)
        y: np.ndarray,  # (T, 3)  — forward returns at 1h/4h/24h
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 3e-4,
    ) -> dict[str, list[float]]:
        from ml.features import build_sequences

        X_scaled = self.scaler.fit_transform(X)
        Xs, ys = build_sequences(X_scaled, y, lookback=cfg.lstm_lookback)

        split = int(len(Xs) * 0.9)
        Xtr, ytr = Xs[:split], ys[:split]
        Xval, yval = Xs[split:], ys[split:]

        tr_ds = torch.utils.data.TensorDataset(
            torch.tensor(Xtr, device=self.device),
            torch.tensor(ytr, device=self.device),
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.tensor(Xval, device=self.device),
            torch.tensor(yval, device=self.device),
        )
        tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.HuberLoss()

        history: dict[str, list[float]] = {"train": [], "val": []}
        best_val = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            tr_loss = 0.0
            for xb, yb in tr_loader:
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tr_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += loss_fn(self.model(xb), yb).item()

            sched.step()
            tr_loss /= len(tr_loader)
            val_loss /= max(len(val_loader), 1)
            history["train"].append(tr_loss)
            history["val"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if epoch % 20 == 0:
                log.info("Epoch %3d/%d — train=%.6f val=%.6f", epoch, epochs, tr_loss, val_loss)

        if best_state:
            self.model.load_state_dict(best_state)
        return history

    def predict(self, X_recent: np.ndarray) -> dict[str, float]:
        """
        X_recent: last cfg.lstm_lookback rows of features (unscaled).
        Returns {h1_ret, h4_ret, h24_ret, h1_std, h4_std, h24_std}.
        """
        if len(X_recent) < cfg.lstm_lookback:
            return {"h1_ret": 0.0, "h4_ret": 0.0, "h24_ret": 0.0}

        X_scaled = self.scaler.transform(X_recent[-cfg.lstm_lookback:])
        tensor = torch.tensor(X_scaled[np.newaxis], dtype=torch.float32, device=self.device)

        self.model.eval()
        mean, std = self.model.predict_with_uncertainty(tensor, n_samples=20)
        mean_np = mean.cpu().numpy()[0]
        std_np = std.cpu().numpy()[0]
        return {
            "h1_ret": float(mean_np[0]),  "h1_std": float(std_np[0]),
            "h4_ret": float(mean_np[1]),  "h4_std": float(std_np[1]),
            "h24_ret": float(mean_np[2]), "h24_std": float(std_np[2]),
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler": self.scaler,
            "n_features": self.model.n_features,
        }, path)
        log.info("PricePredictor saved to %s", path)

    def load(self, path: Path | str) -> "PricePredictorTrainer":
        data = torch.load(path, map_location=self.device, weights_only=False)
        loaded_n_features = int(data["n_features"])
        if loaded_n_features != self.model.n_features:
            raise ValueError(
                f"Feature-count mismatch for {path}: checkpoint expects {loaded_n_features}, "
                f"runtime expects {self.model.n_features}"
            )
        self.model = CryptoPricePredictor(loaded_n_features).to(self.device)
        self.model.load_state_dict(data["model_state"])
        self.scaler = data["scaler"]
        self.model.eval()
        log.info("PricePredictor loaded from %s", path)
        return self


def load_or_init_predictor(symbol: str, n_features: int) -> PricePredictorTrainer:
    path = cfg.model_dir / f"lstm_{symbol}.pt"
    trainer = PricePredictorTrainer(n_features)
    if path.exists():
        try:
            trainer.load(path)
        except Exception as exc:
            log.warning("Could not load LSTM from %s: %s", path, exc)
    return trainer
