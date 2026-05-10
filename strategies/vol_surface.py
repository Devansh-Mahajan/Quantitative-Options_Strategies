"""
Volatility Surface Construction & Analytics.

Implements:
  - SABR model (Hagan 2002) — industry standard for equity/rates vol surfaces
  - SVI parametrisation (Gatheral 2004) — raw SVI for single expiry slices
  - Term structure construction — forward vol, variance term structure
  - Vol surface interpolation — bilinear + cubic spline
  - Skew analytics — risk-reversal, butterfly, skew slope
  - Vol regime signals — IV/RV ratio, vol-of-vol, term structure slope

Used by the GS Quant dashboard tab and strategies/options_vol.py.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.interpolate import RectBivariateSpline, CubicSpline

log = logging.getLogger("vol_surface")


# ─────────────────────────────────────────────────────────────────────────────
# SABR Model (Hagan, Kumar, Lesniewski, Woodward 2002)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SABRParams:
    alpha: float     # initial vol (> 0)
    beta:  float     # CEV exponent ∈ [0, 1]
    rho:   float     # correlation ∈ (-1, 1)
    nu:    float     # vol-of-vol (> 0)


def sabr_vol(F: float, K: float, T: float, params: SABRParams) -> float:
    """
    Hagan SABR implied vol approximation.
    F: forward price, K: strike, T: time to expiry, params: SABR params.
    """
    alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu

    if T <= 0 or F <= 0 or K <= 0:
        return alpha

    if abs(F - K) < 1e-9:
        # ATM approximation
        Fm = F ** (1 - beta)
        term1 = 1 + ((1 - beta)**2 / 24 * alpha**2 / Fm**2
                     + rho * beta * nu * alpha / (4 * Fm)
                     + (2 - 3 * rho**2) / 24 * nu**2) * T
        return alpha / Fm * term1

    log_FK = math.log(F / K)
    FK_mid = math.sqrt(F * K)
    FK_b   = FK_mid ** (1 - beta)
    z      = nu / alpha * FK_b * log_FK
    x_z    = math.log((math.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
    zx     = z / x_z if abs(x_z) > 1e-12 else 1.0

    num_factor = alpha / (FK_b * (1 + (1-beta)**2/24 * log_FK**2
                                    + (1-beta)**4/1920 * log_FK**4))
    correction = 1 + ((1-beta)**2/24 * alpha**2 / FK_b**2
                      + rho*beta*nu*alpha/(4*FK_b)
                      + (2-3*rho**2)/24*nu**2) * T

    return num_factor * zx * correction


def fit_sabr(F: float, T: float, strikes: np.ndarray,
             market_vols: np.ndarray, beta: float = 0.5) -> SABRParams:
    """Fit SABR to a slice of market implied vols."""

    def objective(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or not (-1 < rho < 1):
            return np.ones(len(strikes)) * 1e6
        p = SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
        model_vols = np.array([sabr_vol(F, K, T, p) for K in strikes])
        return model_vols - market_vols

    x0     = [market_vols.mean(), -0.3, 0.4]
    bounds = ([1e-6, -0.999, 1e-6], [5.0, 0.999, 5.0])
    try:
        res = least_squares(objective, x0, bounds=bounds, method="trf", max_nfev=2000)
        a, r, n = res.x
        return SABRParams(alpha=a, beta=beta, rho=r, nu=n)
    except Exception:
        return SABRParams(alpha=market_vols.mean(), beta=beta, rho=-0.3, nu=0.4)


# ─────────────────────────────────────────────────────────────────────────────
# SVI Parametrisation (Gatheral 2004)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SVIParams:
    a: float   # overall variance level
    b: float   # ATM variance slope
    rho: float # rotation
    m: float   # translation (moneyness shift)
    sigma: float # ATM curvature

    def total_variance(self, k: float) -> float:
        """Raw SVI total variance as a function of log-moneyness k = log(K/F)."""
        w = self.a + self.b * (
            self.rho * (k - self.m)
            + math.sqrt((k - self.m)**2 + self.sigma**2)
        )
        return max(w, 0.0)

    def implied_vol(self, k: float, T: float) -> float:
        w = self.total_variance(k)
        return math.sqrt(w / max(T, 1e-9))


def fit_svi(F: float, T: float, strikes: np.ndarray,
            market_vols: np.ndarray) -> SVIParams:
    """Fit SVI to a single expiry slice."""
    k = np.log(strikes / F)
    market_vars = (market_vols ** 2) * T

    def objective(params):
        a, b, rho, m, sigma = params
        if b < 0 or sigma <= 0 or not (-1 < rho < 1):
            return 1e9
        p = SVIParams(a, b, rho, m, sigma)
        model_vars = np.array([p.total_variance(ki) for ki in k])
        return float(np.sum((model_vars - market_vars)**2))

    x0 = [market_vars.mean(), 0.1, -0.3, 0.0, 0.1]
    bounds = [
        (-1.0, 1.0), (1e-6, 5.0), (-0.999, 0.999), (-2.0, 2.0), (1e-6, 5.0)
    ]
    try:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "ftol": 1e-14})
        a, b, rho, m, sigma = res.x
        return SVIParams(a, b, rho, m, sigma)
    except Exception:
        return SVIParams(market_vars.mean(), 0.1, -0.3, 0.0, 0.1)


# ─────────────────────────────────────────────────────────────────────────────
# internal aliases so method parameter names don't shadow module functions
_fit_sabr = fit_sabr
_fit_svi  = fit_svi

# Vol Surface
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VolSlice:
    """A single expiry slice of the vol surface."""
    expiry:  float          # years to expiry
    forward: float          # forward price
    strikes: np.ndarray
    vols:    np.ndarray     # market implied vols
    sabr:    Optional[SABRParams] = None
    svi:     Optional[SVIParams]  = None

    def atm_vol(self) -> float:
        """ATM vol via interpolation at F."""
        if self.sabr:
            return sabr_vol(self.forward, self.forward, self.expiry, self.sabr)
        idx = np.argmin(np.abs(self.strikes - self.forward))
        return float(self.vols[idx])

    def skew(self, delta: float = 0.25) -> float:
        """25Δ risk reversal proxy: vol(0.25Δ call strike) - vol(0.25Δ put strike)."""
        if self.sabr is None:
            return 0.0
        from strategies.options_pricer import BlackScholesEngine, OptionType
        spot   = self.forward
        T      = self.expiry
        r      = 0.05
        atm_iv = self.atm_vol()
        # approx 25Δ strikes via BS delta equation
        from scipy.optimize import brentq
        from scipy.stats import norm
        sqT = math.sqrt(T)
        def call_delta(K, iv):
            d1 = (math.log(spot / K) + (r + 0.5*iv**2)*T) / (iv*sqT)
            return norm.cdf(d1)
        try:
            K_25c = brentq(lambda K: call_delta(K, atm_iv) - delta, spot*0.5, spot*2)
            K_25p = brentq(lambda K: call_delta(K, atm_iv) - (1 - delta), spot*0.5, spot*2)
            iv_25c = sabr_vol(spot, K_25c, T, self.sabr)
            iv_25p = sabr_vol(spot, K_25p, T, self.sabr)
            return round(iv_25c - iv_25p, 6)
        except Exception:
            return 0.0

    def vol_smile(self, n: int = 50) -> list[dict]:
        """Return vol smile across moneyness grid."""
        K_grid = np.linspace(self.strikes.min(), self.strikes.max(), n)
        smile = []
        for K in K_grid:
            if self.sabr:
                iv = sabr_vol(self.forward, K, self.expiry, self.sabr)
            else:
                # linear interpolation on raw strikes
                iv = float(np.interp(K, self.strikes, self.vols))
            smile.append({
                "strike":    round(float(K), 4),
                "moneyness": round(float(K / self.forward), 6),
                "log_money": round(float(math.log(K / self.forward)), 6),
                "vol":       round(float(iv), 6),
            })
        return smile


class VolSurface:
    """
    2D volatility surface: strikes × expiries with SABR/SVI per slice.
    Supports interpolation, skew analytics, and term structure analysis.
    """

    def __init__(self, spot: float, r: float = 0.05, q: float = 0.0):
        self.spot   = spot
        self.r      = r
        self.q      = q
        self.slices: list[VolSlice] = []

    def add_slice(self, expiry: float, strikes: np.ndarray, vols: np.ndarray,
                  fit_sabr: bool = True, fit_svi: bool = True) -> VolSlice:
        """Add an expiry slice and fit SABR + SVI to it."""
        F = self.spot * math.exp((self.r - self.q) * expiry)
        sabr_p = _fit_sabr(F, expiry, strikes, vols) if fit_sabr else None
        svi_p  = _fit_svi(F, expiry, strikes, vols)  if fit_svi  else None
        sl = VolSlice(expiry=expiry, forward=F, strikes=strikes,
                      vols=vols, sabr=sabr_p, svi=svi_p)
        self.slices.append(sl)
        self.slices.sort(key=lambda s: s.expiry)
        return sl

    def atm_vol(self, expiry: float) -> float:
        """ATM vol at given expiry via linear interpolation of ATM vols."""
        if not self.slices:
            return 0.0
        expiries = np.array([s.expiry for s in self.slices])
        atm_vols = np.array([s.atm_vol() for s in self.slices])
        return float(np.interp(expiry, expiries, atm_vols))

    def interpolate(self, expiry: float, strike: float) -> float:
        """Bilinear interpolation: find vol at arbitrary (expiry, strike)."""
        if not self.slices:
            return 0.0
        # Find surrounding slices
        expiries = [s.expiry for s in self.slices]
        if expiry <= expiries[0]:
            sl = self.slices[0]
            if sl.sabr:
                return sabr_vol(sl.forward, strike, sl.expiry, sl.sabr)
            return float(np.interp(strike, sl.strikes, sl.vols))
        if expiry >= expiries[-1]:
            sl = self.slices[-1]
            if sl.sabr:
                return sabr_vol(sl.forward, strike, sl.expiry, sl.sabr)
            return float(np.interp(strike, sl.strikes, sl.vols))

        # Interpolate between two surrounding slices
        idx = np.searchsorted(expiries, expiry)
        sl_lo, sl_hi = self.slices[idx-1], self.slices[idx]
        w = (expiry - sl_lo.expiry) / max(sl_hi.expiry - sl_lo.expiry, 1e-9)
        F_lo = sl_lo.forward
        F_hi = sl_hi.forward

        if sl_lo.sabr:
            v_lo = sabr_vol(F_lo, strike, sl_lo.expiry, sl_lo.sabr)
        else:
            v_lo = float(np.interp(strike, sl_lo.strikes, sl_lo.vols))
        if sl_hi.sabr:
            v_hi = sabr_vol(F_hi, strike, sl_hi.expiry, sl_hi.sabr)
        else:
            v_hi = float(np.interp(strike, sl_hi.strikes, sl_hi.vols))

        return (1 - w) * v_lo + w * v_hi

    def term_structure(self) -> list[dict]:
        """ATM forward vol term structure."""
        out = []
        for sl in self.slices:
            out.append({
                "expiry_years": round(sl.expiry, 4),
                "expiry_days":  round(sl.expiry * 365, 1),
                "atm_vol":      round(sl.atm_vol(), 6),
                "forward":      round(sl.forward, 4),
                "sabr_alpha":   round(sl.sabr.alpha, 6) if sl.sabr else None,
                "sabr_rho":     round(sl.sabr.rho, 6)   if sl.sabr else None,
                "sabr_nu":      round(sl.sabr.nu, 6)    if sl.sabr else None,
                "skew_25d":     round(sl.skew(), 6),
            })
        return out

    def surface_grid(self, n_strikes: int = 20, n_expiries: int = 10) -> dict:
        """
        Return a 2D grid of (strike, expiry) → vol for surface visualization.
        """
        if not self.slices:
            return {"available": False}

        all_strikes = np.concatenate([s.strikes for s in self.slices])
        K_min, K_max = float(all_strikes.min()), float(all_strikes.max())
        T_min = self.slices[0].expiry
        T_max = self.slices[-1].expiry

        K_grid = np.linspace(K_min, K_max, n_strikes)
        T_grid = np.linspace(T_min, T_max, n_expiries)

        surface = []
        for T in T_grid:
            row = []
            for K in K_grid:
                v = self.interpolate(T, K)
                row.append(round(v, 6))
            surface.append(row)

        return {
            "available":   True,
            "strikes":     [round(float(k), 4) for k in K_grid],
            "expiries":    [round(float(t), 4) for t in T_grid],
            "expiry_days": [round(float(t * 365), 1) for t in T_grid],
            "vols":        surface,       # [n_expiries][n_strikes]
            "spot":        self.spot,
        }

    def skew_report(self) -> list[dict]:
        """25Δ skew, butterfly, slope for each expiry slice."""
        report = []
        for sl in self.slices:
            skew25 = sl.skew(0.25)
            atm    = sl.atm_vol()
            report.append({
                "expiry_days": round(sl.expiry * 365, 1),
                "atm_vol":     round(atm, 6),
                "skew_25d":    round(skew25, 6),
                "slope":       round(skew25 / max(atm, 1e-9), 4),
                "sabr_rho":    round(sl.sabr.rho, 4) if sl.sabr else None,
                "sabr_nu":     round(sl.sabr.nu, 4)  if sl.sabr else None,
            })
        return report


# ─────────────────────────────────────────────────────────────────────────────
# Vol Regime Analytics
# ─────────────────────────────────────────────────────────────────────────────

def vol_regime_analytics(atm_iv: float, realized_vol_21: float,
                          term_structure: list[dict]) -> dict:
    """
    Classify the vol regime from IV/RV ratio and term structure slope.
    Returns actionable signals for the vol surface arb strategy.
    """
    iv_rv_ratio = atm_iv / max(realized_vol_21, 1e-6)

    # Term structure slope: front ATM vs back ATM
    ts_slope = 0.0
    if len(term_structure) >= 2:
        front = term_structure[0]["atm_vol"]
        back  = term_structure[-1]["atm_vol"]
        days_span = term_structure[-1]["expiry_days"] - term_structure[0]["expiry_days"]
        ts_slope = (back - front) / max(days_span, 1) * 30  # per month

    # Classify
    if iv_rv_ratio > 1.3 and atm_iv > 0.25:
        regime = "RICH_IV"
        signal = "SELL_VOL"
    elif iv_rv_ratio < 0.8:
        regime = "CHEAP_IV"
        signal = "BUY_VOL"
    elif ts_slope < -0.005:
        regime = "INVERTED_TS"
        signal = "CALENDAR_SPREAD"
    elif ts_slope > 0.01:
        regime = "STEEP_TS"
        signal = "SELL_NEAR_BUY_FAR"
    else:
        regime = "NORMAL"
        signal = "NEUTRAL"

    return {
        "atm_iv":       round(atm_iv, 6),
        "realized_vol": round(realized_vol_21, 6),
        "iv_rv_ratio":  round(iv_rv_ratio, 4),
        "ts_slope":     round(ts_slope, 6),
        "regime":       regime,
        "signal":       signal,
        "premium_pct":  round((iv_rv_ratio - 1) * 100, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build a demo vol surface from ATM vol + skew assumptions
# ─────────────────────────────────────────────────────────────────────────────

def build_demo_surface(spot: float, atm_vol: float,
                        skew: float = -0.05,
                        expiries_days: list[int] | None = None) -> VolSurface:
    """
    Construct a demo vol surface using SABR with parameterised skew.
    Used when live options chain is not available.
    """
    if expiries_days is None:
        expiries_days = [7, 14, 21, 30, 45, 60, 90, 120, 180, 252]

    surface = VolSurface(spot=spot)

    for dte in expiries_days:
        T = dte / 365.0
        K_range = np.linspace(spot * 0.7, spot * 1.3, 30)

        # Synthetic vol smile: ATM + skew × log-moneyness + curvature
        smile_vols = np.array([
            max(0.01, atm_vol + skew * math.log(K / spot) + 0.5 * 0.15 * math.log(K / spot)**2)
            for K in K_range
        ])

        surface.add_slice(T, K_range, smile_vols, fit_sabr=True, fit_svi=True)

    return surface
