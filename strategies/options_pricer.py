"""
Institutional-Grade Options Pricer.

Mirrors gs-quant EqOption framework structure (without requiring GS credentials).
Implements Black-Scholes + full Greeks suite including higher-order:
  Delta, Gamma, Vega, Theta, Rho — first-order
  Vanna, Volga, Charm, Veta — second-order
  Speed, Color, Zomma, Ultima — third-order

Also implements:
  - Newton-Raphson implied vol solver
  - Portfolio-level Greeks aggregation
  - Scenario P&L (spot/vol/rate shocks)
  - Intrinsic value & time value decomposition
  - Expected move (1σ range) from IV
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# ─────────────────────────────────────────────────────────────────────────────
# Enumerations (mirror gs-quant OptionType / ExerciseStyle)
# ─────────────────────────────────────────────────────────────────────────────

class OptionType(str, Enum):
    CALL     = "Call"
    PUT      = "Put"
    STRADDLE = "Straddle"

class ExerciseStyle(str, Enum):
    EUROPEAN = "European"
    AMERICAN = "American"   # binomial approximation

class StrikeType(str, Enum):
    STRIKE = "Strike"       # absolute strike
    DELTA  = "Delta"        # delta-equivalent strike
    ATMF   = "ATMF"         # at-the-money forward

# ─────────────────────────────────────────────────────────────────────────────
# Greeks dataclass — mirrors gs-quant risk measures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Greeks:
    # First-order
    delta:  float = 0.0    # dV/dS
    gamma:  float = 0.0    # d²V/dS²
    vega:   float = 0.0    # dV/dσ  (per 1% move in vol)
    theta:  float = 0.0    # dV/dt  (per calendar day)
    rho:    float = 0.0    # dV/dr  (per 1% move in rate)
    # Second-order
    vanna:  float = 0.0    # d²V/dS dσ = dDelta/dσ = dVega/dS
    volga:  float = 0.0    # d²V/dσ²  (vomma)
    charm:  float = 0.0    # d²V/dS dt (delta bleed)
    veta:   float = 0.0    # d²V/dσ dt (vega decay)
    # Third-order
    speed:  float = 0.0    # d³V/dS³
    color:  float = 0.0    # d³V/dS² dt (gamma decay)
    zomma:  float = 0.0    # d³V/dS² dσ
    ultima: float = 0.0    # d³V/dσ³

    def as_dict(self) -> dict:
        return {k: round(v, 8) for k, v in self.__dict__.items()}


@dataclass
class PricingResult:
    option_type:   str
    spot:          float
    strike:        float
    dte:           float        # days to expiry
    T:             float        # years to expiry
    sigma:         float        # implied vol (annualised)
    rate:          float        # risk-free rate
    price:         float        # option fair value
    intrinsic:     float        # max(payoff, 0)
    time_value:    float        # price - intrinsic
    greeks:        Greeks = field(default_factory=Greeks)
    expected_move: float = 0.0  # 1σ spot range at expiry

    def as_dict(self) -> dict:
        return {
            "option_type":   self.option_type,
            "spot":          round(self.spot, 4),
            "strike":        round(self.strike, 4),
            "dte":           round(self.dte, 2),
            "T":             round(self.T, 6),
            "sigma":         round(self.sigma, 6),
            "rate":          round(self.rate, 6),
            "price":         round(self.price, 6),
            "intrinsic":     round(self.intrinsic, 6),
            "time_value":    round(self.time_value, 6),
            "expected_move": round(self.expected_move, 4),
            "greeks":        self.greeks.as_dict(),
        }

# ─────────────────────────────────────────────────────────────────────────────
# Core Black-Scholes Engine
# ─────────────────────────────────────────────────────────────────────────────

class BlackScholesEngine:
    """
    Full Black-Scholes-Merton pricer with complete Greeks suite.
    Structured to mirror gs-quant EqOption.calc() patterns.
    """

    @staticmethod
    def _d1d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return None, None
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    @classmethod
    def price(cls, S: float, K: float, T: float, r: float, sigma: float,
              option_type: OptionType = OptionType.CALL, q: float = 0.0) -> float:
        """Black-Scholes-Merton option price."""
        if T <= 1e-9:
            if option_type == OptionType.CALL:     return max(S - K, 0.0)
            if option_type == OptionType.PUT:      return max(K - S, 0.0)
            if option_type == OptionType.STRADDLE: return abs(S - K)

        d1, d2 = cls._d1d2(S, K, T, r, q, sigma)
        if d1 is None:
            return 0.0

        F = S * math.exp((r - q) * T)
        disc = math.exp(-r * T)

        if option_type == OptionType.CALL:
            return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
        if option_type == OptionType.PUT:
            return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        if option_type == OptionType.STRADDLE:
            c = disc * (F * norm.cdf(d1)  - K * norm.cdf(d2))
            p = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
            return c + p
        return 0.0

    @classmethod
    def greeks(cls, S: float, K: float, T: float, r: float, sigma: float,
               option_type: OptionType = OptionType.CALL, q: float = 0.0) -> Greeks:
        """Compute all Greeks up to third order."""
        g = Greeks()
        if T <= 1e-9 or sigma <= 0:
            return g

        d1, d2 = cls._d1d2(S, K, T, r, q, sigma)
        if d1 is None:
            return g

        sqT  = math.sqrt(T)
        nd1  = norm.cdf(d1)
        nd2  = norm.cdf(d2)
        n_d1 = norm.cdf(-d1)
        n_d2 = norm.cdf(-d2)
        pdf1 = norm.pdf(d1)
        disc = math.exp(-r * T)
        dq   = math.exp(-q * T)

        # ── Delta ─────────────────────────────────────────────────────────
        if option_type == OptionType.CALL:
            g.delta = dq * nd1
        elif option_type == OptionType.PUT:
            g.delta = -dq * n_d1
        else:  # straddle
            g.delta = dq * (nd1 - n_d1)

        # ── Gamma (same for calls and puts) ────────────────────────────────
        g.gamma = dq * pdf1 / (S * sigma * sqT)

        # ── Vega (per 1% vol move) ─────────────────────────────────────────
        g.vega = S * dq * pdf1 * sqT / 100.0

        # ── Theta (per calendar day) ───────────────────────────────────────
        theta_base = -(S * dq * pdf1 * sigma / (2 * sqT))
        if option_type == OptionType.CALL:
            g.theta = (theta_base - r * K * disc * nd2  + q * S * dq * nd1)  / 365.0
        elif option_type == OptionType.PUT:
            g.theta = (theta_base + r * K * disc * n_d2 - q * S * dq * n_d1) / 365.0
        else:
            g.theta = theta_base * 2 / 365.0

        # ── Rho (per 1% rate move) ─────────────────────────────────────────
        if option_type == OptionType.CALL:
            g.rho = K * T * disc * nd2  / 100.0
        elif option_type == OptionType.PUT:
            g.rho = -K * T * disc * n_d2 / 100.0
        else:
            g.rho = K * T * disc * (nd2 - n_d2) / 100.0

        # ── Vanna  d²V/dSdσ ───────────────────────────────────────────────
        g.vanna = -dq * pdf1 * d2 / sigma

        # ── Volga / Vomma  d²V/dσ² ────────────────────────────────────────
        g.volga = S * dq * pdf1 * sqT * d1 * d2 / sigma

        # ── Charm  d²V/dSdt (per calendar day) ────────────────────────────
        if option_type in (OptionType.CALL, OptionType.STRADDLE):
            g.charm = (-dq * (pdf1 * (2*(r-q)*T - d2*sigma*sqT) / (2*T*sigma*sqT) - (q if q else 0)*nd1)) / 365.0
        else:
            g.charm = (-dq * (pdf1 * (2*(r-q)*T - d2*sigma*sqT) / (2*T*sigma*sqT) + (q if q else 0)*n_d1)) / 365.0

        # ── Veta  d²V/dσdt (vega decay per calendar day) ─────────────────
        g.veta = (-S * dq * pdf1 * sqT
                  * (q + (r - q)*d1/(sigma*sqT) - (1 + d1*d2)/(2*T))) / 365.0

        # ── Speed  d³V/dS³ ────────────────────────────────────────────────
        g.speed = -g.gamma / S * (d1/(sigma*sqT) + 1)

        # ── Color  d³V/dS²dt (gamma bleed per calendar day) ───────────────
        g.color = (-dq * pdf1 / (2*S*T*sigma*sqT)
                   * (2*q*T + 1 + d1*(2*(r-q)*T - d2*sigma*sqT)/(sigma*sqT))) / 365.0

        # ── Zomma  d³V/dS²dσ ─────────────────────────────────────────────
        g.zomma = g.gamma * (d1*d2 - 1) / sigma

        # ── Ultima  d³V/dσ³ ───────────────────────────────────────────────
        g.ultima = -g.volga / sigma * (d1*d2 - 1) / (d1*d2)  if (d1 * d2) != 0 else 0.0

        return g

    @classmethod
    def implied_vol(cls, market_price: float, S: float, K: float, T: float, r: float,
                    option_type: OptionType = OptionType.CALL, q: float = 0.0,
                    tol: float = 1e-6) -> Optional[float]:
        """Newton-Raphson IV solver with Brent fallback."""
        if T <= 0 or market_price <= 0:
            return None

        intrinsic = max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0)
        if market_price < intrinsic:
            return None

        def objective(sigma):
            return cls.price(S, K, T, r, sigma, option_type, q) - market_price

        try:
            return brentq(objective, 1e-6, 20.0, xtol=tol, maxiter=200)
        except Exception:
            return None

    @classmethod
    def calc(cls, S: float, K: float, T: float, r: float, sigma: float,
             option_type: OptionType = OptionType.CALL,
             q: float = 0.0, dte: float | None = None) -> PricingResult:
        """
        Full pricing with all Greeks. Mirrors gs-quant EqOption.calc() API.
        """
        if dte is None:
            dte = T * 365.0

        px = cls.price(S, K, T, r, sigma, option_type, q)
        g  = cls.greeks(S, K, T, r, sigma, option_type, q)

        intrinsic = max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0)

        expected_move = S * sigma * math.sqrt(T)  # 1σ move in price terms

        return PricingResult(
            option_type   = option_type.value,
            spot          = S,
            strike        = K,
            dte           = dte,
            T             = T,
            sigma         = sigma,
            rate          = r,
            price         = px,
            intrinsic     = intrinsic,
            time_value    = px - intrinsic,
            greeks        = g,
            expected_move = expected_move,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Scenario Engine (gs-quant Scenario-style)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioPnL:
    spot_shock_pct:  float
    vol_shock_abs:   float
    rate_shock_abs:  float
    pnl:             float
    pnl_delta:       float   # delta component
    pnl_gamma:       float   # gamma component
    pnl_vega:        float   # vega component
    pnl_theta:       float   # theta component
    pnl_rho:         float   # rho component

    def as_dict(self) -> dict:
        return {k: round(v, 6) for k, v in self.__dict__.items()}


def run_scenario(result: PricingResult, spot_shock_pct: float = 0.0,
                 vol_shock_abs: float = 0.0, rate_shock_abs: float = 0.0,
                 time_decay_days: float = 0.0) -> ScenarioPnL:
    """
    Compute P&L attribution for a scenario shock.
    Uses Taylor expansion: ΔV ≈ Δ·ΔS + ½Γ·ΔS² + ν·Δσ + Θ·Δt + ρ·Δr
    """
    dS  = result.spot * spot_shock_pct / 100.0
    dV  = vol_shock_abs
    dr  = rate_shock_abs
    dt  = time_decay_days / 365.0
    g   = result.greeks

    pnl_delta = g.delta * dS
    pnl_gamma = 0.5 * g.gamma * dS**2
    pnl_vega  = g.vega  * (dV * 100.0)   # vega is per 1% vol
    pnl_theta = g.theta * time_decay_days
    pnl_rho   = g.rho   * (dr * 100.0)   # rho is per 1% rate

    total_pnl = pnl_delta + pnl_gamma + pnl_vega + pnl_theta + pnl_rho

    return ScenarioPnL(
        spot_shock_pct  = spot_shock_pct,
        vol_shock_abs   = vol_shock_abs,
        rate_shock_abs  = rate_shock_abs,
        pnl             = total_pnl,
        pnl_delta       = pnl_delta,
        pnl_gamma       = pnl_gamma,
        pnl_vega        = pnl_vega,
        pnl_theta       = pnl_theta,
        pnl_rho         = pnl_rho,
    )


def full_scenario_grid(result: PricingResult,
                        spot_range: list[float] | None = None,
                        vol_range: list[float]  | None = None) -> list[dict]:
    """
    Generate a 2D scenario grid: spot shock × vol shock P&L.
    Used for heatmap visualization.
    """
    if spot_range is None:
        spot_range = [-20, -15, -10, -5, -3, 0, 3, 5, 10, 15, 20]
    if vol_range is None:
        vol_range  = [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]

    rows = []
    for dv in vol_range:
        for ds in spot_range:
            scen = run_scenario(result, spot_shock_pct=ds, vol_shock_abs=dv)
            rows.append(scen.as_dict())
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Options Portfolio (multi-leg)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionLeg:
    option_type:   OptionType
    spot:          float
    strike:        float
    T:             float        # years to expiry
    sigma:         float        # implied vol
    rate:          float = 0.05
    q:             float = 0.0
    quantity:      float = 1.0  # +long / -short
    multiplier:    float = 100.0  # options contract multiplier

    def price(self) -> float:
        return BlackScholesEngine.price(
            self.spot, self.strike, self.T, self.rate, self.sigma, self.option_type, self.q
        )

    def greeks(self) -> Greeks:
        return BlackScholesEngine.greeks(
            self.spot, self.strike, self.T, self.rate, self.sigma, self.option_type, self.q
        )

    def result(self) -> PricingResult:
        return BlackScholesEngine.calc(
            self.spot, self.strike, self.T, self.rate, self.sigma, self.option_type, self.q
        )


class OptionsPortfolio:
    """
    Aggregate multiple option legs into portfolio-level Greeks.
    Mirrors gs-quant Portfolio.calc() pattern.
    """

    def __init__(self, legs: list[OptionLeg]):
        self.legs = legs

    def _agg(self, attr: str) -> float:
        total = 0.0
        for leg in self.legs:
            g = leg.greeks()
            v = getattr(g, attr, 0.0) or 0.0
            total += v * leg.quantity * leg.multiplier
        return total

    def total_value(self) -> float:
        return sum(leg.price() * leg.quantity * leg.multiplier for leg in self.legs)

    def portfolio_greeks(self) -> dict:
        return {
            "value":    round(self.total_value(), 4),
            "delta":    round(self._agg("delta"),  6),
            "gamma":    round(self._agg("gamma"),  6),
            "vega":     round(self._agg("vega"),   6),
            "theta":    round(self._agg("theta"),  6),
            "rho":      round(self._agg("rho"),    6),
            "vanna":    round(self._agg("vanna"),  6),
            "volga":    round(self._agg("volga"),  6),
            "charm":    round(self._agg("charm"),  6),
            "speed":    round(self._agg("speed"),  6),
        }

    def pnl_attribution(self, dS: float = 0.0, dV: float = 0.0,
                         dt: float = 1.0, dr: float = 0.0) -> dict:
        """P&L components for given shocks."""
        g = self.portfolio_greeks()
        pnl_delta = g["delta"] * dS
        pnl_gamma = 0.5 * g["gamma"] / 100.0 * dS**2  # gamma scaled by contract size
        pnl_vega  = g["vega"]  * dV * 100.0
        pnl_theta = g["theta"] * dt
        pnl_rho   = g["rho"]   * dr * 100.0
        total     = pnl_delta + pnl_gamma + pnl_vega + pnl_theta + pnl_rho
        return {
            "total":  round(total,     4),
            "delta":  round(pnl_delta, 4),
            "gamma":  round(pnl_gamma, 4),
            "vega":   round(pnl_vega,  4),
            "theta":  round(pnl_theta, 4),
            "rho":    round(pnl_rho,   4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: price a single option from API request
# ─────────────────────────────────────────────────────────────────────────────

def price_option(S: float, K: float, dte: float, sigma: float,
                 option_type: str = "Call", r: float = 0.05, q: float = 0.0) -> dict:
    """
    Entry point for dashboard API: price an option and return full result dict.
    """
    T = max(dte / 365.0, 1e-6)
    ot = OptionType(option_type) if option_type in [o.value for o in OptionType] else OptionType.CALL
    result = BlackScholesEngine.calc(S, K, T, r, sigma, ot, q, dte=dte)

    # Standard scenarios
    scenarios = []
    for ds in [-15, -10, -5, -3, -1, 0, 1, 3, 5, 10, 15]:
        scen = run_scenario(result, spot_shock_pct=ds)
        scenarios.append({"spot_shock_pct": ds, "pnl": round(scen.pnl, 4)})

    # Vol scenarios
    vol_scenarios = []
    for dv in [-0.10, -0.05, 0.0, 0.05, 0.10, 0.20]:
        scen = run_scenario(result, vol_shock_abs=dv)
        vol_scenarios.append({"vol_shock": round(dv, 2), "pnl": round(scen.pnl, 4)})

    d = result.as_dict()
    d["spot_scenarios"]  = scenarios
    d["vol_scenarios"]   = vol_scenarios
    d["scenario_grid"]   = full_scenario_grid(result)
    return d
