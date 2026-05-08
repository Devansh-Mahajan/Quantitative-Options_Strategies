"""
Options volatility strategies on Binance EAPI (BTC/ETH European options).
- Short straddle/strangle when IV > RV (sell premium in low-vol regimes)
- Long straddle before expected high-vol events
- Delta-neutral maintenance via futures hedge
"""

from __future__ import annotations
import logging
import math

import numpy as np

from bot.config import cfg
from strategies.base import BaseStrategy, Signal
from ml.vol_model import bs_delta, bs_call_price, bs_put_price

log = logging.getLogger("strategy.options_vol")

IV_RV_THRESHOLD = 1.25    # IV must be > 1.25x RV to sell premium
MIN_IV = 0.20              # ignore options with implied vol < 20%
RISK_FREE = 0.05           # 5% risk-free rate proxy


class OptionsVolatilityStrategy(BaseStrategy):
    name = "options_vol"
    market = "options"

    @property
    def symbols(self) -> list[str]:
        return cfg.options_underlying  # ["BTC", "ETH"]

    def generate_signals(self, store, regime: str, predictions: dict) -> list[Signal]:
        if not cfg.enable_options:
            return []

        signals = []

        for underlying in self.symbols:
            try:
                symbol = f"{underlying}USDT"
                closes = store.get_closes(symbol, "1h", 30)
                if len(closes) < 20:
                    continue

                rets = np.diff(np.log(closes))
                realised_vol = float(rets.std() * np.sqrt(365 * 24))
                spot_price = float(closes[-1])

                # ATM IV from vol surface (stored by orchestrator)
                iv_surface = getattr(store, "iv_surfaces", {}).get(underlying)
                if iv_surface is None:
                    continue

                atm_iv = iv_surface.atm_vol(dte=7)
                if atm_iv is None or atm_iv < MIN_IV:
                    continue

                iv_rv_ratio = atm_iv / max(realised_vol, 0.01)
                vol_regime = "low" if realised_vol < 0.4 else ("high" if realised_vol > 0.8 else "mid")

                # --- Short straddle: IV >> RV, low-vol / ranging regime ---
                if iv_rv_ratio > IV_RV_THRESHOLD and regime in ("ranging", "bull") and vol_regime != "high":
                    dte = 7
                    T = dte / 365

                    # Find near-ATM call and put symbols
                    call_sym = _find_atm_option(store, underlying, spot_price, "CALL", dte)
                    put_sym = _find_atm_option(store, underlying, spot_price, "PUT", dte)

                    if call_sym and put_sym:
                        call_price = bs_call_price(spot_price, spot_price, T, RISK_FREE, atm_iv)
                        put_price = bs_put_price(spot_price, spot_price, T, RISK_FREE, atm_iv)

                        confidence = min(0.80, 0.50 + (iv_rv_ratio - IV_RV_THRESHOLD) * 0.30)

                        # Sell call
                        signals.append(Signal(
                            symbol=call_sym, market="options", side="SELL",
                            quantity=0.01, price=call_price,
                            confidence=confidence, strategy=self.name,
                            meta={"iv": atm_iv, "rv": realised_vol, "leg": "short_call"},
                        ))
                        # Sell put
                        signals.append(Signal(
                            symbol=put_sym, market="options", side="SELL",
                            quantity=0.01, price=put_price,
                            confidence=confidence, strategy=self.name,
                            meta={"iv": atm_iv, "rv": realised_vol, "leg": "short_put"},
                        ))

                # --- Long straddle: very low IV, volatile regime likely incoming ---
                elif atm_iv < 0.35 and regime == "volatile":
                    dte = 14
                    T = dte / 365
                    call_sym = _find_atm_option(store, underlying, spot_price, "CALL", dte)
                    put_sym = _find_atm_option(store, underlying, spot_price, "PUT", dte)

                    if call_sym and put_sym:
                        call_price = bs_call_price(spot_price, spot_price, T, RISK_FREE, atm_iv)
                        put_price = bs_put_price(spot_price, spot_price, T, RISK_FREE, atm_iv)
                        confidence = 0.65

                        signals.append(Signal(
                            symbol=call_sym, market="options", side="BUY",
                            quantity=0.01, price=call_price,
                            confidence=confidence, strategy=self.name,
                            meta={"iv": atm_iv, "leg": "long_call"},
                        ))
                        signals.append(Signal(
                            symbol=put_sym, market="options", side="BUY",
                            quantity=0.01, price=put_price,
                            confidence=confidence, strategy=self.name,
                            meta={"iv": atm_iv, "leg": "long_put"},
                        ))

            except Exception as exc:
                log.debug("OptionsVol %s error: %s", underlying, exc)

        return signals


def _find_atm_option(store, underlying: str, spot: float, option_type: str, target_dte: int) -> str | None:
    """Look up nearest ATM option symbol from stored options chain."""
    chain = getattr(store, "options_chains", {}).get(underlying, [])
    best = None
    best_score = float("inf")
    for opt in chain:
        if opt.get("optionType", "").upper() != option_type:
            continue
        dte = float(opt.get("daysToExpiry", 0))
        strike = float(opt.get("strikePrice", 0))
        if dte <= 0 or strike <= 0:
            continue
        score = abs(dte - target_dte) + abs(strike / spot - 1.0) * 10
        if score < best_score:
            best_score = score
            best = opt.get("symbol")
    return best
