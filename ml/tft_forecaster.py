"""
Temporal Fusion Transformer (TFT) — Multi-Horizon Price Forecaster.

Architecture follows Lim et al. (2021) "Temporal Fusion Transformers for
Interpretable Multi-Horizon Time Series Forecasting" (IJF).

Key design decisions:
  - Variable Selection Networks (VSN) weight each input feature per time-step
  - Gated Residual Networks (GRN) enable dynamic non-linear transformations
  - Multi-Head Attention over encoder context (interpretable temporal weights)
  - Quantile outputs: P10 / P50 / P90 → uncertainty band around point forecast
  - Static covariates (symbol embedding) modulate temporal processing
  - No look-ahead: all features built from t-L to t-1, forecast t to t+H

Horizons: 1h, 4h, 24h (bars, not wall-clock — adapts to any frequency)
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger("ml.tft_forecaster")

_ROOT    = Path(__file__).resolve().parents[1]
CKPT_DIR = _ROOT / "config" / "tft"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
LOOKBACK   = 60        # encoder sequence length (bars)
HORIZONS   = [1, 4, 24]
D_MODEL    = 64        # hidden size throughout
N_HEADS    = 4         # attention heads
N_LAYERS   = 2         # encoder layers
DROPOUT    = 0.1
QUANTILES  = [0.10, 0.50, 0.90]
BATCH      = 128
EPOCHS     = 20
LR         = 3e-4
MIN_ROWS   = LOOKBACK + max(HORIZONS) + 100


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class GRN(nn.Module):
    """Gated Residual Network — core TFT building block."""
    def __init__(self, d: int, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.fc1  = nn.Linear(d, d)
        self.fc2  = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.ln   = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        # c = static context that modulates the GRN
        h = F.elu(self.fc1(x if c is None else x + c))
        h = self.drop(self.fc2(h))
        g = torch.sigmoid(self.gate(x if c is None else x + c))
        return self.ln(g * h + (1 - g) * x)


class VSN(nn.Module):
    """Variable Selection Network — soft-gates input features per time-step."""
    def __init__(self, n_features: int, d: int, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(1, d) for _ in range(n_features)])
        self.grns        = nn.ModuleList([GRN(d, dropout) for _ in range(n_features)])
        self.selector    = nn.Sequential(nn.Linear(n_features * d, n_features), nn.Softmax(dim=-1))

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, F)
        B, T, F = x.shape
        embeds = []
        for i, (proj, grn) in enumerate(zip(self.projections, self.grns)):
            xi = x[..., i:i+1]                     # (B, T, 1)
            e  = grn(proj(xi), context)             # (B, T, D)
            embeds.append(e)
        stacked = torch.stack(embeds, dim=-2)       # (B, T, F, D)
        flat    = stacked.view(B, T, -1)            # (B, T, F*D)
        weights = self.selector(flat).unsqueeze(-1) # (B, T, F, 1)
        return (stacked * weights).sum(dim=-2)      # (B, T, D)


class StaticCovariateEncoder(nn.Module):
    """Maps static symbol embedding → 4 context vectors used by TFT layers."""
    def __init__(self, n_symbols: int, d: int) -> None:
        super().__init__()
        self.embed    = nn.Embedding(n_symbols, d)
        self.grn_enc  = GRN(d)    # context for encoder VSN
        self.grn_h    = GRN(d)    # LSTM init hidden
        self.grn_c    = GRN(d)    # LSTM init cell
        self.grn_enr  = GRN(d)    # enrichment context

    def forward(self, sym_idx: torch.Tensor):
        e    = self.embed(sym_idx)   # (B, D)
        return self.grn_enc(e), self.grn_h(e), self.grn_c(e), self.grn_enr(e)


# ─────────────────────────────────────────────────────────────────────────────
# Full TFT Model
# ─────────────────────────────────────────────────────────────────────────────

class TFT(nn.Module):
    def __init__(self, n_features: int, n_symbols: int, n_horizons: int,
                 d: int = D_MODEL, n_heads: int = N_HEADS,
                 n_layers: int = N_LAYERS, dropout: float = DROPOUT,
                 quantiles: list[float] = QUANTILES) -> None:
        super().__init__()
        self.n_horizons = n_horizons
        self.quantiles  = quantiles
        Q = len(quantiles)

        self.static_enc = StaticCovariateEncoder(n_symbols, d)
        self.vsn_enc    = VSN(n_features, d, dropout)
        self.lstm       = nn.LSTM(d, d, batch_first=True, num_layers=n_layers,
                                  dropout=dropout if n_layers > 1 else 0.0)
        self.post_lstm  = GRN(d, dropout)
        attn_layer      = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=d*4,
                                                     dropout=dropout, batch_first=True)
        self.attn       = nn.TransformerEncoder(attn_layer, n_layers)
        self.enrichment = GRN(d, dropout)
        self.out        = nn.Linear(d, n_horizons * Q)

    def forward(self, x: torch.Tensor, sym_idx: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)   sym_idx: (B,)
        B, T, _ = x.shape
        ctx_enc, h0, c0, ctx_enr = self.static_enc(sym_idx)
        ctx_enc_t = ctx_enc.unsqueeze(1).expand(B, T, -1)  # broadcast over time

        selected = self.vsn_enc(x, ctx_enc_t)              # (B, T, D)
        h0 = h0.unsqueeze(0).expand(self.lstm.num_layers, B, -1).contiguous()
        c0 = c0.unsqueeze(0).expand(self.lstm.num_layers, B, -1).contiguous()
        enc, _  = self.lstm(selected, (h0, c0))            # (B, T, D)
        enc     = self.post_lstm(enc)
        enc     = self.attn(enc)
        ctx_enr_t = ctx_enr.unsqueeze(1).expand(B, T, -1)
        out     = self.enrichment(enc, ctx_enr_t)          # (B, T, D)
        last    = out[:, -1, :]                            # (B, D) — final bar
        preds   = self.out(last)                           # (B, H*Q)
        return preds.view(B, self.n_horizons, len(self.quantiles))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TFTDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, sym_ids: np.ndarray) -> None:
        self.X   = torch.tensor(X,       dtype=torch.float32)
        self.y   = torch.tensor(y,       dtype=torch.float32)
        self.sym = torch.tensor(sym_ids, dtype=torch.long)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.sym[i]


# ─────────────────────────────────────────────────────────────────────────────
# Quantile loss
# ─────────────────────────────────────────────────────────────────────────────

def quantile_loss(pred: torch.Tensor, target: torch.Tensor,
                  quantiles: list[float]) -> torch.Tensor:
    """
    pred:   (B, H, Q)
    target: (B, H)
    """
    target = target.unsqueeze(-1).expand_as(pred)
    q      = torch.tensor(quantiles, device=pred.device).view(1, 1, -1)
    err    = target - pred
    return torch.max((q - 1) * err, q * err).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder (shared with XGBAlphaEngine)
# ─────────────────────────────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> np.ndarray:
    """Returns (T, F) float32 with no infs/nans."""
    c  = df["close"].astype(float)
    lr = np.log(c / c.shift(1))
    f  = pd.DataFrame(index=df.index)
    for w in [1, 3, 5, 10, 20, 60]:
        f[f"ret_{w}"] = c.pct_change(w)
    for w in [5, 10, 20, 60]:
        f[f"rv_{w}"] = lr.rolling(w).std() * np.sqrt(252 * 24)
    for span in [9, 21, 55]:
        ema = c.ewm(span=span, adjust=False).mean()
        f[f"ema_dev_{span}"] = (c - ema) / (ema + 1e-10)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-10))
    macd_line   = c.ewm(12, adjust=False).mean() - c.ewm(26, adjust=False).mean()
    f["macd"]   = macd_line / (c + 1e-10)
    bb_m = c.rolling(20).mean()
    bb_s = c.rolling(20).std()
    f["bb_pct"] = (c - bb_m) / (2 * bb_s + 1e-10)
    f["vol_ratio"] = df["volume"].astype(float) / (df["volume"].rolling(20).mean() + 1e-10)
    f.replace([np.inf, -np.inf], np.nan, inplace=True)
    f.fillna(0.0, inplace=True)
    return f.values.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# High-level engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TFTResult:
    symbol:   str
    horizon:  int
    p10:      float       # 10th percentile forecast
    p50:      float       # median forecast
    p90:      float       # 90th percentile forecast
    direction: str        # "LONG" | "SHORT" | "NEUTRAL"


@dataclass
class TFTEngine:
    """
    Multi-symbol TFT engine. Trains a single shared model using symbol embeddings.
    Saves / loads PyTorch checkpoint.
    """
    model:        Optional[TFT] = field(default=None, repr=False)
    sym_map:      dict = field(default_factory=dict)   # symbol → int id
    feature_mean: Optional[np.ndarray] = field(default=None, repr=False)
    feature_std:  Optional[np.ndarray] = field(default=None, repr=False)
    n_features:   int = 0
    device:       str = "cpu"

    def __post_init__(self):
        try:
            from core.torch_device import resolve_torch_runtime
            self.device = str(resolve_torch_runtime().device)
        except Exception:
            self.device = "cpu"

    # ── Normalise ─────────────────────────────────────────────────────────

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        if self.feature_mean is None:
            self.feature_mean = X.reshape(-1, X.shape[-1]).mean(0)
            self.feature_std  = X.reshape(-1, X.shape[-1]).std(0) + 1e-8
        return (X - self.feature_mean) / self.feature_std

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, df_dict: dict[str, pd.DataFrame],
              epochs: int = EPOCHS, verbose: bool = True) -> dict:
        if not df_dict:
            return {}

        # Assign symbol ids
        self.sym_map = {s: i for i, s in enumerate(sorted(df_dict.keys()))}
        n_syms = len(self.sym_map)

        # Build sequences
        all_X, all_y, all_sym = [], [], []
        for sym, df in df_dict.items():
            if len(df) < MIN_ROWS:
                continue
            feats = _build_features(df)
            c     = df["close"].astype(float).values
            self.n_features = feats.shape[1]
            fwd_rets = np.zeros((len(feats), len(HORIZONS)))
            for hi, h in enumerate(HORIZONS):
                fr = np.zeros(len(feats))
                for t in range(len(feats) - h):
                    fr[t] = c[t + h] / c[t] - 1.0 if c[t] > 0 else 0.0
                fwd_rets[:, hi] = fr

            for t in range(LOOKBACK, len(feats) - max(HORIZONS)):
                all_X.append(feats[t - LOOKBACK: t])
                all_y.append(fwd_rets[t])
                all_sym.append(self.sym_map[sym])

        if not all_X:
            log.warning("TFT train: no sequences built")
            return {}

        X  = np.array(all_X, dtype=np.float32)
        y  = np.array(all_y, dtype=np.float32)
        ids= np.array(all_sym, dtype=np.int64)

        X  = self._normalise(X)

        split = int(len(X) * 0.85)
        ds_tr = TFTDataset(X[:split], y[:split], ids[:split])
        ds_va = TFTDataset(X[split:], y[split:], ids[split:])
        dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True,  num_workers=0)
        dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, num_workers=0)

        self.model = TFT(
            n_features=self.n_features,
            n_symbols=n_syms,
            n_horizons=len(HORIZONS),
        ).to(self.device)

        opt = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_val = float("inf")
        metrics: dict = {}

        for ep in range(1, epochs + 1):
            self.model.train()
            tr_loss = 0.0
            for Xb, yb, sb in dl_tr:
                Xb, yb, sb = Xb.to(self.device), yb.to(self.device), sb.to(self.device)
                pred = self.model(Xb, sb)
                loss = quantile_loss(pred, yb, QUANTILES)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tr_loss += loss.item()
            sched.step()

            self.model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for Xb, yb, sb in dl_va:
                    Xb, yb, sb = Xb.to(self.device), yb.to(self.device), sb.to(self.device)
                    pred = self.model(Xb, sb)
                    va_loss += quantile_loss(pred, yb, QUANTILES).item()

            va_loss /= max(len(dl_va), 1)
            tr_loss /= max(len(dl_tr), 1)
            if verbose:
                log.info("TFT epoch %2d/%d  train=%.6f  val=%.6f", ep, epochs, tr_loss, va_loss)
            if va_loss < best_val:
                best_val = va_loss
                metrics = {"best_val_loss": best_val, "epoch": ep, "n_symbols": n_syms,
                           "n_sequences": len(X)}

        return metrics

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, df_dict: dict[str, pd.DataFrame]) -> list[TFTResult]:
        if self.model is None or not self.sym_map:
            return []
        self.model.eval()
        results = []
        for sym, df in df_dict.items():
            if sym not in self.sym_map or len(df) < LOOKBACK:
                continue
            feats = _build_features(df)[-LOOKBACK:]       # (L, F)
            X     = self._normalise(feats[np.newaxis])    # (1, L, F)
            Xt    = torch.tensor(X, dtype=torch.float32).to(self.device)
            sid   = torch.tensor([self.sym_map[sym]], dtype=torch.long).to(self.device)
            with torch.no_grad():
                pred = self.model(Xt, sid)[0]             # (H, Q)
            for hi, h in enumerate(HORIZONS):
                p10, p50, p90 = [float(pred[hi, qi]) for qi in range(len(QUANTILES))]
                direction = ("LONG" if p50 > 0.005 else
                             "SHORT" if p50 < -0.005 else "NEUTRAL")
                results.append(TFTResult(sym, h, p10, p50, p90, direction))
        return results

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path = CKPT_DIR / "tft.pt") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":   self.model.state_dict() if self.model else None,
            "sym_map":      self.sym_map,
            "n_features":   self.n_features,
            "feature_mean": self.feature_mean,
            "feature_std":  self.feature_std,
        }, path)
        log.info("TFT saved → %s", path)

    def load(self, path: Path = CKPT_DIR / "tft.pt") -> "TFTEngine":
        if not Path(path).exists():
            return self
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.sym_map      = ckpt["sym_map"]
        self.n_features   = ckpt["n_features"]
        self.feature_mean = ckpt["feature_mean"]
        self.feature_std  = ckpt["feature_std"]
        if ckpt["state_dict"] and self.n_features:
            self.model = TFT(
                n_features=self.n_features,
                n_symbols=len(self.sym_map),
                n_horizons=len(HORIZONS),
            ).to(self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
        log.info("TFT loaded from %s (%d symbols)", path, len(self.sym_map))
        return self


# ── Module-level singleton ────────────────────────────────────────────────────
_engine: Optional[TFTEngine] = None


def get_tft_engine() -> TFTEngine:
    global _engine
    if _engine is None:
        _engine = TFTEngine()
        _engine.load()
    return _engine
