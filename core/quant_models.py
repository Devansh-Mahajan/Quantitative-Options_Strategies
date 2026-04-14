from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OptionLegModel:
    option_type: str
    side: int
    strike: float
    years_to_expiry: float
    volatility: float
    quantity: int = 1


@dataclass(frozen=True)
class MonteCarloRiskSnapshot:
    fair_value: float
    expected_value: float
    var_95: float
    cvar_95: float
    value_volatility: float
    worst_case: float
    best_case: float


@dataclass(frozen=True)
class LongOptionTailSnapshot:
    premium: float
    bs_fair_value: float
    binomial_fair_value: float
    mc_fair_value: float
    model_edge: float
    expected_pnl: float
    var_95: float
    cvar_95: float
    profit_probability: float
    fat_tail_probability: float
    p95_payoff: float
    p99_payoff: float
    tail_payoff_multiple: float


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def black_scholes_price(
    flag: str,
    spot: float,
    strike: float,
    years_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
) -> float:
    if spot <= 0 or strike <= 0 or years_to_expiry <= 0 or volatility <= 0:
        intrinsic = max(0.0, spot - strike) if flag == "c" else max(0.0, strike - spot)
        return intrinsic

    variance_term = volatility * math.sqrt(years_to_expiry)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility * volatility) * years_to_expiry) / variance_term
    d2 = d1 - variance_term

    if flag == "c":
        return (spot * _norm_cdf(d1)) - (strike * math.exp(-risk_free_rate * years_to_expiry) * _norm_cdf(d2))
    return (strike * math.exp(-risk_free_rate * years_to_expiry) * _norm_cdf(-d2)) - (spot * _norm_cdf(-d1))


def option_payoff(flag: str, spot: float, strike: float) -> float:
    if flag == "c":
        return max(0.0, float(spot) - float(strike))
    return max(0.0, float(strike) - float(spot))


def binomial_option_price(
    flag: str,
    spot: float,
    strike: float,
    years_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    steps: int = 100,
) -> float:
    if spot <= 0 or strike <= 0 or years_to_expiry <= 0 or volatility <= 0:
        return option_payoff(flag, spot, strike)

    n_steps = max(2, int(steps))
    dt = years_to_expiry / n_steps
    u = math.exp(volatility * math.sqrt(dt))
    d = 1.0 / u
    if abs(u - d) < 1e-9:
        return black_scholes_price(flag, spot, strike, years_to_expiry, risk_free_rate, volatility)

    discount = math.exp(-risk_free_rate * dt)
    p = (math.exp(risk_free_rate * dt) - d) / (u - d)
    p = max(0.0, min(1.0, p))

    terminal_prices = [
        spot * (u ** (n_steps - down_moves)) * (d ** down_moves)
        for down_moves in range(n_steps + 1)
    ]
    option_values = [option_payoff(flag, terminal_spot, strike) for terminal_spot in terminal_prices]

    for step in range(n_steps - 1, -1, -1):
        option_values = [
            discount * ((p * option_values[idx]) + ((1.0 - p) * option_values[idx + 1]))
            for idx in range(step + 1)
        ]
    return float(option_values[0])


def monte_carlo_option_price(
    flag: str,
    spot: float,
    strike: float,
    years_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    n_simulations: int = 5000,
    seed: int | None = 42,
) -> float:
    if spot <= 0 or strike <= 0 or years_to_expiry <= 0 or volatility <= 0:
        return black_scholes_price(flag, spot, strike, years_to_expiry, risk_free_rate, volatility)

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(max(1, int(n_simulations)))
    drift = (risk_free_rate - 0.5 * volatility * volatility) * years_to_expiry
    diffusion = volatility * math.sqrt(years_to_expiry) * shocks
    terminal_prices = spot * np.exp(drift + diffusion)

    if flag == "c":
        payoffs = np.maximum(terminal_prices - strike, 0.0)
    else:
        payoffs = np.maximum(strike - terminal_prices, 0.0)

    return float(math.exp(-risk_free_rate * years_to_expiry) * np.mean(payoffs))


def price_multileg_position(spot: float, legs: list[OptionLegModel], risk_free_rate: float) -> float:
    total = 0.0
    for leg in legs:
        flag = "c" if str(leg.option_type).lower().startswith("c") else "p"
        option_value = black_scholes_price(
            flag=flag,
            spot=spot,
            strike=float(leg.strike),
            years_to_expiry=max(1e-6, float(leg.years_to_expiry)),
            risk_free_rate=risk_free_rate,
            volatility=max(1e-4, float(leg.volatility)),
        )
        total += int(leg.side) * int(leg.quantity) * option_value
    return float(total)


def simulate_terminal_prices(
    spot: float,
    horizon_years: float,
    risk_free_rate: float,
    volatility: float,
    n_simulations: int = 2000,
    seed: int | None = 42,
    use_student_t: bool = True,
    degrees_of_freedom: int = 5,
) -> np.ndarray:
    if spot <= 0:
        return np.array([], dtype=float)
    if horizon_years <= 0 or volatility <= 0:
        return np.full(max(1, int(n_simulations)), float(spot), dtype=float)

    rng = np.random.default_rng(seed)
    if use_student_t:
        df = max(3, int(degrees_of_freedom))
        shocks = rng.standard_t(df=df, size=max(1, int(n_simulations)))
        shocks = shocks * math.sqrt((df - 2) / df)
    else:
        shocks = rng.standard_normal(max(1, int(n_simulations)))

    drift = (risk_free_rate - 0.5 * volatility * volatility) * horizon_years
    diffusion = volatility * math.sqrt(horizon_years) * shocks
    return spot * np.exp(drift + diffusion)


def monte_carlo_multileg_risk(
    spot: float,
    legs: list[OptionLegModel],
    risk_free_rate: float,
    horizon_years: float = 1.0 / 252.0,
    confidence: float = 0.95,
    n_simulations: int = 2000,
    seed: int | None = 42,
) -> MonteCarloRiskSnapshot:
    if spot <= 0 or not legs:
        return MonteCarloRiskSnapshot(
            fair_value=0.0,
            expected_value=0.0,
            var_95=0.0,
            cvar_95=0.0,
            value_volatility=0.0,
            worst_case=0.0,
            best_case=0.0,
        )

    current_fair_value = price_multileg_position(spot=spot, legs=legs, risk_free_rate=risk_free_rate)
    average_vol = float(np.mean([max(1e-4, float(leg.volatility)) for leg in legs]))
    terminal_spots = simulate_terminal_prices(
        spot=spot,
        horizon_years=horizon_years,
        risk_free_rate=risk_free_rate,
        volatility=average_vol,
        n_simulations=n_simulations,
        seed=seed,
        use_student_t=True,
    )

    simulated_values = np.zeros(len(terminal_spots), dtype=float)
    for idx, terminal_spot in enumerate(terminal_spots):
        rolled_legs = [
            OptionLegModel(
                option_type=leg.option_type,
                side=leg.side,
                strike=leg.strike,
                years_to_expiry=max(1e-6, float(leg.years_to_expiry) - horizon_years),
                volatility=leg.volatility,
                quantity=leg.quantity,
            )
            for leg in legs
        ]
        simulated_values[idx] = price_multileg_position(
            spot=float(terminal_spot),
            legs=rolled_legs,
            risk_free_rate=risk_free_rate,
        )

    pnl = simulated_values - current_fair_value
    confidence_floor = max(0.50, min(0.99, float(confidence)))
    downside_cut = float(np.quantile(pnl, 1.0 - confidence_floor))
    downside_tail = pnl[pnl <= downside_cut]

    var_95 = max(0.0, -downside_cut)
    cvar_95 = max(0.0, -float(np.mean(downside_tail))) if len(downside_tail) else var_95

    return MonteCarloRiskSnapshot(
        fair_value=float(current_fair_value),
        expected_value=float(np.mean(simulated_values)),
        var_95=float(var_95),
        cvar_95=float(cvar_95),
        value_volatility=float(np.std(simulated_values)),
        worst_case=float(np.min(simulated_values)),
        best_case=float(np.max(simulated_values)),
    )


def analyze_long_option_tail(
    flag: str,
    spot: float,
    strike: float,
    years_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    premium: float,
    n_simulations: int = 4000,
    seed: int | None = 42,
    fat_tail_multiple: float = 5.0,
) -> LongOptionTailSnapshot:
    safe_premium = max(0.01, float(premium))
    bs_fair_value = black_scholes_price(flag, spot, strike, years_to_expiry, risk_free_rate, volatility)
    binomial_fair_value = binomial_option_price(flag, spot, strike, years_to_expiry, risk_free_rate, volatility)
    mc_fair_value = monte_carlo_option_price(
        flag,
        spot,
        strike,
        years_to_expiry,
        risk_free_rate,
        volatility,
        n_simulations=n_simulations,
        seed=seed,
    )

    terminal_spots = simulate_terminal_prices(
        spot=spot,
        horizon_years=years_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        n_simulations=n_simulations,
        seed=seed,
        use_student_t=True,
    )
    payoffs = np.array([option_payoff(flag, terminal_spot, strike) for terminal_spot in terminal_spots], dtype=float)
    pnl = payoffs - safe_premium
    downside_cut = float(np.quantile(pnl, 0.05))
    downside_tail = pnl[pnl <= downside_cut]

    p95_payoff = float(np.quantile(payoffs, 0.95))
    p99_payoff = float(np.quantile(payoffs, 0.99))
    tail_payoff_multiple = p99_payoff / safe_premium if safe_premium > 0 else 0.0
    fat_tail_threshold = safe_premium * max(1.0, float(fat_tail_multiple))

    return LongOptionTailSnapshot(
        premium=safe_premium,
        bs_fair_value=float(bs_fair_value),
        binomial_fair_value=float(binomial_fair_value),
        mc_fair_value=float(mc_fair_value),
        model_edge=float(max(bs_fair_value, binomial_fair_value, mc_fair_value) - safe_premium),
        expected_pnl=float(np.mean(pnl)),
        var_95=float(max(0.0, -downside_cut)),
        cvar_95=float(max(0.0, -float(np.mean(downside_tail)))) if len(downside_tail) else float(max(0.0, -downside_cut)),
        profit_probability=float(np.mean(pnl > 0.0)),
        fat_tail_probability=float(np.mean(payoffs >= fat_tail_threshold)),
        p95_payoff=p95_payoff,
        p99_payoff=p99_payoff,
        tail_payoff_multiple=float(tail_payoff_multiple),
    )
