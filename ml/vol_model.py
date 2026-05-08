"""
Volatility models:
- GARCH(1,1) per symbol for conditional vol estimation
- Implied vol surface interpolation for options strategies
- Vol regime classifier (low/mid/high)
"""

from __future__ import annotations
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("ml.vol_model")

try:
    from arch import arch_model
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False
    log.warning("arch library not installed — GARCH disabled, using rolling std")


# --------------------------------------------------------------------------- #
# GARCH volatility estimator
# --------------------------------------------------------------------------- #

class GARCHVolModel:
    """
    Fits GARCH(1,1) on log returns. Forecasts 1-step-ahead conditional volatility.
    Volatility is annualised (crypto: 365 * 24 for hourly data).
    """

    def __init__(self, freq_multiplier: float = 365 * 24) -> None:
        self.freq_mult = freq_multiplier  # annualisation factor
        self._result = None
        self._last_vol: float = 0.0

    def fit(self, returns: pd.Series | np.ndarray) -> None:
        if not _ARCH_AVAILABLE:
            arr = np.array(returns)
            self._last_vol = float(arr[-20:].std() * np.sqrt(self.freq_mult))
            return
        ret_pct = np.array(returns) * 100  # ARCH expects percentage returns
        model = arch_model(ret_pct, vol="Garch", p=1, q=1, dist="t")
        self._result = model.fit(disp="off", show_warning=False)
        h = self._result.conditional_volatility[-1] / 100  # back to decimal
        self._last_vol = float(h * np.sqrt(self.freq_mult))

    def forecast_vol(self, horizon: int = 1) -> float:
        """Return annualised conditional volatility forecast."""
        if self._result is None:
            return self._last_vol
        try:
            fc = self._result.forecast(horizon=horizon, reindex=False)
            h = float(fc.variance.values[-1, -1] ** 0.5) / 100
            return h * np.sqrt(self.freq_mult)
        except Exception:
            return self._last_vol

    @property
    def current_vol(self) -> float:
        return self._last_vol

    def vol_regime(self) -> str:
        """Classify current vol: low (<40% ann), mid (<80%), high (>80%)."""
        v = self._last_vol
        if v < 0.40:
            return "low"
        if v < 0.80:
            return "mid"
        return "high"

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"result": self._result, "last_vol": self._last_vol, "freq_mult": self.freq_mult}, f)

    def load(self, path: Path | str) -> "GARCHVolModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._result = data.get("result")
        self._last_vol = data.get("last_vol", 0.0)
        self.freq_mult = data.get("freq_mult", self.freq_mult)
        return self


# --------------------------------------------------------------------------- #
# Implied volatility surface (for options)
# --------------------------------------------------------------------------- #

class IVSurface:
    """
    Simple implied vol surface built from discrete option quotes.
    Uses 2D linear interpolation over (strike_ratio, days_to_expiry).
    """

    def __init__(self) -> None:
        self._points: list[tuple[float, float, float]] = []  # (k_ratio, dte, iv)

    def update(self, options: list[dict]) -> None:
        """
        options: list of dicts with keys: strike, underlying_price, dte, implied_volatility
        """
        self._points = []
        for opt in options:
            try:
                k_ratio = float(opt["strike"]) / float(opt["underlying_price"])
                dte = float(opt["dte"])
                iv = float(opt["implied_volatility"])
                if iv > 0 and dte > 0:
                    self._points.append((k_ratio, dte, iv))
            except (KeyError, ZeroDivisionError, ValueError):
                continue

    def get_iv(self, strike_ratio: float, dte: float) -> float | None:
        """Interpolate IV for a given (moneyness, dte). Returns None if no data."""
        if len(self._points) < 3:
            return None
        try:
            from scipy.interpolate import LinearNDInterpolator
            pts = np.array([(k, d) for k, d, _ in self._points])
            ivs = np.array([iv for _, _, iv in self._points])
            interp = LinearNDInterpolator(pts, ivs)
            iv = float(interp([[strike_ratio, dte]])[0])
            return iv if not np.isnan(iv) else None
        except Exception:
            return None

    def term_structure(self, strike_ratio: float = 1.0) -> list[tuple[float, float]]:
        """Return [(dte, iv)] sorted by dte for ATM strike_ratio ≈ 1.0."""
        pts = [(d, iv) for k, d, iv in self._points if abs(k - strike_ratio) < 0.05]
        return sorted(pts, key=lambda x: x[0])

    def atm_vol(self, dte: float = 7) -> float | None:
        return self.get_iv(1.0, dte)

    def skew(self, dte: float = 7) -> float | None:
        """25-delta risk reversal proxy: IV(0.9) - IV(1.1)."""
        iv_otm_put = self.get_iv(0.90, dte)
        iv_otm_call = self.get_iv(1.10, dte)
        if iv_otm_put is None or iv_otm_call is None:
            return None
        return iv_otm_put - iv_otm_call


# --------------------------------------------------------------------------- #
# Black-Scholes helpers (for options pricing)
# --------------------------------------------------------------------------- #

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European call price (Black-Scholes)."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European put price (Black-Scholes)."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return 1.0 if (option_type == "call" and S > K) else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T) + 1e-10))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    from scipy.stats import norm
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(S * norm.pdf(d1) * np.sqrt(T) / 100)  # per 1% IV move
