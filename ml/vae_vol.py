"""
Variational Autoencoder for Vol Surface Anomaly Detection.

Research basis:
  - Kingma & Welling (2013) "Auto-Encoding Variational Bayes"
  - Cont & da Fonseca (2002) "Dynamics of implied volatility surfaces" — vol surface
    lives on a low-dimensional manifold; VAE learns this manifold.
  - Application: reconstruction error flags when observed vol surface deviates
    significantly from the learned manifold → actionable anomaly signal.

Architecture:
  - Input: flattened vol surface snapshot (N_STRIKES × N_EXPIRIES grid)
  - Encoder: 3-layer MLP → (mu, log_var) in latent space Z_DIM
  - Decoder: 3-layer MLP → reconstructed surface
  - Training objective: ELBO = -Reconstruction + beta * KL(q||p)
    (beta-VAE with beta=4 for better disentanglement)
  - Anomaly score: MSE(observed, reconstructed) + KL term
    Calibrate threshold at 95th percentile of training distribution

Usage:
    engine = VAEVolEngine()
    engine.train(surface_snapshots)   # list of np.ndarray grids
    score, is_anomaly = engine.score(surface_snapshot)
    engine.save() / engine.load()
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger("ml.vae_vol")

_ROOT    = Path(__file__).resolve().parents[1]
CKPT_DIR = _ROOT / "config" / "vae_vol"

# ── Grid dimensions — must match build_demo_surface output ───────────────────
N_STRIKES  = 20
N_EXPIRIES = 10
INPUT_DIM  = N_STRIKES * N_EXPIRIES

# ── Architecture ─────────────────────────────────────────────────────────────
Z_DIM      = 16        # latent space dimension
H_DIMS     = [256, 128, 64]   # encoder hidden layer sizes
BETA       = 4.0       # KL weight (beta-VAE — stronger disentanglement)
LR         = 1e-3
BATCH      = 64
EPOCHS     = 100
ANOMALY_PERCENTILE = 95.0     # threshold calibration percentile


# ─────────────────────────────────────────────────────────────────────────────
# Encoder / Decoder
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, input_dim: int, h_dims: list[int], z_dim: int) -> None:
        super().__init__()
        layers = []
        in_d = input_dim
        for h in h_dims:
            layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2)]
            in_d = h
        self.net    = nn.Sequential(*layers)
        self.mu     = nn.Linear(in_d, z_dim)
        self.logvar = nn.Linear(in_d, z_dim)

    def forward(self, x: torch.Tensor):
        h      = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, z_dim: int, h_dims: list[int], output_dim: int) -> None:
        super().__init__()
        layers = []
        in_d = z_dim
        for h in reversed(h_dims):
            layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2)]
            in_d = h
        layers += [nn.Linear(in_d, output_dim), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BetaVAE(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM,
                 h_dims: list[int] = H_DIMS,
                 z_dim: int = Z_DIM,
                 beta: float = BETA) -> None:
        super().__init__()
        self.beta    = beta
        self.encoder = Encoder(input_dim, h_dims, z_dim)
        self.decoder = Decoder(z_dim, h_dims, input_dim)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparametrization trick: z = mu + eps * std."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z          = self.reparametrize(mu, logvar)
        x_hat      = self.decoder(z)
        return x_hat, mu, logvar

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
        kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon + self.beta * kl, float(recon.item()), float(kl.item())

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction error per sample (lower = normal)."""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            z          = mu   # use mean at inference (no noise)
            x_hat      = self.decoder(z)
            per_sample = F.mse_loss(x_hat, x, reduction="none").mean(dim=1)
            kl_per     = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
            return per_sample + self.beta * kl_per / x.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# Surface preparation helpers
# ─────────────────────────────────────────────────────────────────────────────

def surface_to_array(surface_grid: dict) -> Optional[np.ndarray]:
    """
    Convert surface_grid dict (from VolSurface.surface_grid()) to flat array.
    Returns (INPUT_DIM,) float32 or None if shape mismatch.
    """
    try:
        vols = np.array(surface_grid["vols"], dtype=np.float32)     # (n_exp, n_k)
        if vols.size == 0:
            return None
        # Resize to fixed grid via bilinear interp
        import cv2
        grid = cv2.resize(vols, (N_STRIKES, N_EXPIRIES), interpolation=cv2.INTER_LINEAR)
        return grid.flatten().astype(np.float32)
    except ImportError:
        # Fallback without cv2: nearest-neighbour re-sampling
        vols = np.array(surface_grid["vols"], dtype=np.float32)
        if vols.size == 0:
            return None
        from scipy.ndimage import zoom
        zy = N_EXPIRIES / vols.shape[0]
        zx = N_STRIKES  / vols.shape[1]
        grid = zoom(vols, (zy, zx), order=1)
        return grid.flatten().astype(np.float32)
    except Exception as e:
        log.debug("surface_to_array failed: %s", e)
        return None


def normalize_surface(surfaces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale surfaces to [0,1] range using training min/max per feature."""
    lo  = surfaces.min(axis=0)
    hi  = surfaces.max(axis=0)
    rng = np.maximum(hi - lo, 1e-8)
    return (surfaces - lo) / rng, lo, rng


# ─────────────────────────────────────────────────────────────────────────────
# High-level engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VAEVolEngine:
    model:             Optional[BetaVAE] = field(default=None, repr=False)
    _threshold:        float = 0.0
    _norm_lo:          Optional[np.ndarray] = field(default=None, repr=False)
    _norm_rng:         Optional[np.ndarray] = field(default=None, repr=False)
    _train_scores:     Optional[np.ndarray] = field(default=None, repr=False)
    device:            str = "cpu"

    def __post_init__(self):
        try:
            from core.telemetry.torch_device import resolve_torch_runtime
            self.device = str(resolve_torch_runtime().device)
        except Exception:
            self.device = "cpu"

    # ── Pre-process ───────────────────────────────────────────────────────

    def _prep(self, surfaces: np.ndarray) -> torch.Tensor:
        if self._norm_lo is None:
            surfaces, self._norm_lo, self._norm_rng = normalize_surface(surfaces)
        else:
            surfaces = (surfaces - self._norm_lo) / self._norm_rng
        return torch.tensor(np.clip(surfaces, 0.0, 1.0), dtype=torch.float32)

    def _prep_single(self, arr: np.ndarray) -> torch.Tensor:
        if self._norm_lo is None:
            return torch.tensor(arr[np.newaxis], dtype=torch.float32).to(self.device)
        normed = np.clip((arr - self._norm_lo) / self._norm_rng, 0.0, 1.0)
        return torch.tensor(normed[np.newaxis], dtype=torch.float32).to(self.device)

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, surfaces: list[np.ndarray],
              epochs: int = EPOCHS, verbose: bool = True) -> dict:
        """
        surfaces: list of flat (INPUT_DIM,) arrays — one per historical snapshot.
        """
        if len(surfaces) < 50:
            log.warning("VAE vol: need >= 50 surface snapshots, got %d", len(surfaces))
            return {}

        X = np.vstack(surfaces).astype(np.float32)
        Xt = self._prep(X).to(self.device)

        split = int(len(Xt) * 0.85)
        ds_tr = TensorDataset(Xt[:split])
        ds_va = TensorDataset(Xt[split:])
        dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True,  num_workers=0)
        dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, num_workers=0)

        self.model = BetaVAE().to(self.device)
        opt   = torch.optim.Adam(self.model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10)

        best_val = float("inf")
        for ep in range(1, epochs + 1):
            self.model.train()
            tr_loss = 0.0
            for (xb,) in dl_tr:
                xhat, mu, lv = self.model(xb)
                loss, _, _   = self.model.loss(xb, xhat, mu, lv)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tr_loss += loss.item()

            self.model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for (xb,) in dl_va:
                    xhat, mu, lv = self.model(xb)
                    va_loss += self.model.loss(xb, xhat, mu, lv)[0].item()

            va_loss /= max(len(dl_va), 1)
            sched.step(va_loss)
            if verbose and ep % 10 == 0:
                log.info("VAE epoch %3d/%d  val_loss=%.5f", ep, epochs, va_loss)
            if va_loss < best_val:
                best_val = va_loss

        # Calibrate anomaly threshold from training distribution
        self.model.eval()
        with torch.no_grad():
            scores = self.model.anomaly_score(Xt).cpu().numpy()
        self._train_scores = scores
        self._threshold    = float(np.percentile(scores, ANOMALY_PERCENTILE))
        log.info("VAE calibrated  threshold=%.6f  (p95 of %d train samples)",
                 self._threshold, len(scores))
        return {"best_val_loss": best_val, "threshold": self._threshold,
                "n_train": len(Xt)}

    # ── Scoring ───────────────────────────────────────────────────────────

    def score(self, surface_arr: np.ndarray) -> tuple[float, bool, str]:
        """
        Returns (anomaly_score, is_anomaly, label).
        surface_arr: flat (INPUT_DIM,) array.
        """
        if self.model is None:
            return 0.0, False, "not_trained"
        Xt = self._prep_single(surface_arr)
        sc = float(self.model.anomaly_score(Xt).item())
        is_anom = sc > self._threshold
        label = "ANOMALY" if is_anom else "NORMAL"
        return sc, is_anom, label

    def latent(self, surface_arr: np.ndarray) -> np.ndarray:
        """Return latent representation (mu) for dimensionality reduction / visualization."""
        if self.model is None:
            return np.zeros(Z_DIM)
        self.model.eval()
        Xt = self._prep_single(surface_arr)
        with torch.no_grad():
            mu, _ = self.model.encoder(Xt)
        return mu.squeeze(0).cpu().numpy()

    def reconstruct(self, surface_arr: np.ndarray) -> np.ndarray:
        """Reconstruct surface from latent code (for comparison / visualization)."""
        if self.model is None:
            return surface_arr
        self.model.eval()
        Xt = self._prep_single(surface_arr)
        with torch.no_grad():
            mu, _ = self.model.encoder(Xt)
            x_hat = self.model.decoder(mu).squeeze(0).cpu().numpy()
        # Denormalise
        if self._norm_lo is not None:
            x_hat = x_hat * self._norm_rng + self._norm_lo
        return x_hat.reshape(N_EXPIRIES, N_STRIKES)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path = CKPT_DIR / "vae_vol.pt") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":    self.model.state_dict() if self.model else None,
            "threshold":     self._threshold,
            "norm_lo":       self._norm_lo,
            "norm_rng":      self._norm_rng,
            "train_scores":  self._train_scores,
        }, path)
        log.info("VAE vol saved → %s", path)

    def load(self, path: Path = CKPT_DIR / "vae_vol.pt") -> "VAEVolEngine":
        if not Path(path).exists():
            return self
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._threshold   = ckpt.get("threshold", 0.0)
        self._norm_lo     = ckpt.get("norm_lo")
        self._norm_rng    = ckpt.get("norm_rng")
        self._train_scores= ckpt.get("train_scores")
        if ckpt.get("state_dict"):
            self.model = BetaVAE().to(self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
        log.info("VAE vol loaded from %s  threshold=%.6f", path, self._threshold)
        return self


# ── Module-level singleton ────────────────────────────────────────────────────
_engine: Optional[VAEVolEngine] = None


def get_vae_engine() -> VAEVolEngine:
    global _engine
    if _engine is None:
        _engine = VAEVolEngine()
        _engine.load()
    return _engine
