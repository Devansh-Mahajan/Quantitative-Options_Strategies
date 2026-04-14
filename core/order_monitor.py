from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

from alpaca.trading.requests import ReplaceOrderRequest

from config.params import (
    ORDER_MONITOR_ENABLED,
    ORDER_MONITOR_MAX_REPRICES,
    ORDER_MONITOR_MC_PATHS,
    ORDER_MONITOR_MIN_PRICE_STEP,
    ORDER_MONITOR_POLL_SECONDS,
    ORDER_MONITOR_REPRICE_FRACTION,
    ORDER_MONITOR_TIMEOUT_SECONDS,
    ORDER_MONITOR_VAR_CONFIDENCE,
    ORDER_MONITOR_VAR_HORIZON_DAYS,
    OPTION_PRICING_RISK_FREE_RATE,
)
from core.delay_aware_options import (
    build_delay_adjusted_contracts,
    effective_ask_price,
    effective_bid_price,
    effective_mid_price,
)
from core.notifications import send_alert
from core.quant_models import OptionLegModel, monte_carlo_multileg_risk
from core.utils import get_option_days_to_expiry, parse_option_symbol

logger = logging.getLogger(f"strategy.{__name__}")

OPEN_ORDER_STATUSES = {
    "new",
    "accepted",
    "accepted_for_bidding",
    "pending_new",
    "pending_replace",
    "partially_filled",
    "held",
}
FILLED_ORDER_STATUSES = {"filled"}
TERMINAL_ORDER_STATUSES = {
    "canceled",
    "done_for_day",
    "expired",
    "rejected",
    "replaced",
    "stopped",
    "suspended",
    "calculated",
}


@dataclass(frozen=True)
class MonitoredOrderLeg:
    symbol: str
    side: str


@dataclass(frozen=True)
class ExecutionPricingSnapshot:
    natural_price: float
    fair_price: float
    pricing_confidence: float
    staleness_pct: float
    underlying_price: float
    mc_expected_price: float
    mc_var_95: float
    mc_cvar_95: float


@dataclass(frozen=True)
class MonitoredOrderResult:
    order_id: str
    final_status: str
    filled_qty: float
    limit_price: float
    reprices: int = 0
    filled: bool = False
    partial_fill: bool = False
    pricing_confidence: float | None = None
    natural_price: float | None = None
    fair_price: float | None = None
    mc_expected_price: float | None = None
    mc_var_95: float | None = None
    mc_cvar_95: float | None = None


def build_execution_pricing_snapshot(
    client,
    legs: list[MonitoredOrderLeg],
    risk_free_rate: float = OPTION_PRICING_RISK_FREE_RATE,
) -> ExecutionPricingSnapshot | None:
    if not legs:
        return None

    synthetic_contracts = []
    for leg in legs:
        underlying, option_type, strike = parse_option_symbol(leg.symbol)
        synthetic_contracts.append(
            SimpleNamespace(
                symbol=leg.symbol,
                underlying_symbol=underlying,
                type="call" if option_type == "C" else "put",
                strike_price=float(strike),
                dte=float(max(-1, get_option_days_to_expiry(leg.symbol))),
            )
        )

    symbols = [contract.symbol for contract in synthetic_contracts]
    snapshots = client.get_option_snapshot(symbols)
    priced_contracts = build_delay_adjusted_contracts(client, synthetic_contracts, snapshots=snapshots)
    priced_map = {contract.symbol: contract for contract in priced_contracts}
    if len(priced_map) != len(legs):
        return None

    signed_natural = 0.0
    signed_fair = 0.0
    min_confidence = 1.0
    max_staleness = 0.0
    risk_legs: list[OptionLegModel] = []
    underlying_price = 0.0

    for leg in legs:
        contract = priced_map[leg.symbol]
        if leg.side == "buy":
            signed_natural += effective_ask_price(contract)
            signed_fair += effective_mid_price(contract)
            side = 1
        else:
            signed_natural -= effective_bid_price(contract)
            signed_fair -= effective_mid_price(contract)
            side = -1

        min_confidence = min(min_confidence, float(contract.pricing_confidence or 0.0))
        max_staleness = max(max_staleness, float(contract.staleness_pct or 0.0))
        underlying_price = max(
            underlying_price,
            float(contract.underlying_price or contract.delayed_underlying_price or 0.0),
        )
        risk_legs.append(
            OptionLegModel(
                option_type=str(contract.contract_type),
                side=side,
                strike=float(contract.strike or 0.0),
                years_to_expiry=max(1e-6, float(contract.dte or 0.0) / 365.0),
                volatility=max(1e-4, float(contract.implied_volatility or 0.25)),
            )
        )

    risk = monte_carlo_multileg_risk(
        spot=max(0.01, underlying_price),
        legs=risk_legs,
        risk_free_rate=risk_free_rate,
        horizon_years=max(1.0 / 252.0, float(ORDER_MONITOR_VAR_HORIZON_DAYS) / 252.0),
        confidence=float(ORDER_MONITOR_VAR_CONFIDENCE),
        n_simulations=int(ORDER_MONITOR_MC_PATHS),
    )

    return ExecutionPricingSnapshot(
        natural_price=round(float(signed_natural), 4),
        fair_price=round(float(signed_fair), 4),
        pricing_confidence=round(float(min_confidence), 4),
        staleness_pct=round(float(max_staleness), 4),
        underlying_price=round(float(underlying_price), 4),
        mc_expected_price=round(float(risk.expected_value), 4),
        mc_var_95=round(float(risk.var_95), 4),
        mc_cvar_95=round(float(risk.cvar_95), 4),
    )


def suggest_repriced_limit(
    snapshot: ExecutionPricingSnapshot | None,
    current_limit_price: float,
    attempt_number: int,
    is_credit: bool,
) -> float | None:
    if snapshot is None:
        return None

    natural_abs = abs(float(snapshot.natural_price))
    fair_abs = abs(float(snapshot.fair_price))
    mc_abs = abs(float(snapshot.mc_expected_price))
    current_abs = abs(float(current_limit_price or 0.0))

    if natural_abs <= 0 and fair_abs <= 0 and current_abs <= 0:
        return None

    preferred_abs = fair_abs if fair_abs > 0 else current_abs
    if mc_abs > 0:
        preferred_abs = (0.65 * preferred_abs) + (0.35 * mc_abs)
    preferred_abs = max(preferred_abs, ORDER_MONITOR_MIN_PRICE_STEP)

    gap = abs(natural_abs - preferred_abs)
    if gap <= 1e-9:
        target_abs = max(natural_abs, preferred_abs)
    else:
        progress = min(1.0, float(attempt_number) * float(ORDER_MONITOR_REPRICE_FRACTION))
        confidence = max(0.0, min(1.0, float(snapshot.pricing_confidence)))
        tail_risk_ratio = min(1.0, float(snapshot.mc_cvar_95 or 0.0) / max(preferred_abs, 0.01))

        if is_credit:
            urgency = max(0.35, min(1.0, 0.75 + (0.25 * confidence) - (0.30 * tail_risk_ratio)))
            target_abs = preferred_abs - (gap * progress * urgency)
            target_abs = max(natural_abs, target_abs)
        else:
            urgency = max(0.45, min(1.20, 0.80 + (0.20 * (1.0 - confidence)) + (0.25 * tail_risk_ratio)))
            target_abs = preferred_abs + (gap * progress * urgency)
            target_abs = min(max(natural_abs, preferred_abs), target_abs)

    target_abs = round(max(0.01, target_abs), 2)
    current_abs = round(max(0.0, current_abs), 2)
    step = round(float(ORDER_MONITOR_MIN_PRICE_STEP), 2)

    if abs(target_abs - current_abs) < step:
        if is_credit:
            target_abs = min(natural_abs or target_abs, current_abs + step)
        else:
            cap = natural_abs if natural_abs > 0 else current_abs + step
            target_abs = min(cap, current_abs + step)
        target_abs = round(max(0.01, target_abs), 2)

    if is_credit:
        return -target_abs
    return target_abs


def _status_text(value) -> str:
    if hasattr(value, "value"):
        value = value.value
    return str(value or "").strip().lower()


def monitor_multileg_order(
    client,
    order,
    order_label: str,
    legs: list[MonitoredOrderLeg],
    is_credit: bool,
    poll_seconds: float = ORDER_MONITOR_POLL_SECONDS,
    timeout_seconds: float = ORDER_MONITOR_TIMEOUT_SECONDS,
    max_reprices: int = ORDER_MONITOR_MAX_REPRICES,
    snapshot_builder: Callable[..., ExecutionPricingSnapshot | None] = build_execution_pricing_snapshot,
    limit_reprice_func: Callable[[ExecutionPricingSnapshot | None, float, int, bool], float | None] = suggest_repriced_limit,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> MonitoredOrderResult:
    active_order = order
    active_order_id = str(getattr(order, "id", ""))
    current_limit = float(getattr(order, "limit_price", 0.0) or 0.0)
    reprices = 0
    last_status = ""
    last_filled_qty = -1.0
    last_snapshot: ExecutionPricingSnapshot | None = None

    if not ORDER_MONITOR_ENABLED:
        status = _status_text(getattr(order, "status", None))
        filled_qty = float(getattr(order, "filled_qty", 0.0) or 0.0)
        return MonitoredOrderResult(
            order_id=active_order_id,
            final_status=status,
            filled_qty=filled_qty,
            limit_price=current_limit,
            reprices=0,
            filled=status in FILLED_ORDER_STATUSES,
            partial_fill=filled_qty > 0 and status not in FILLED_ORDER_STATUSES,
        )

    logger.info(
        "📡 Order monitor engaged for %s | order_id=%s | initial_limit=%+.2f",
        order_label,
        active_order_id,
        current_limit,
    )

    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while time.monotonic() <= deadline:
        active_order = client.trade_client.get_order_by_id(active_order_id)
        status = _status_text(getattr(active_order, "status", None))
        filled_qty = float(getattr(active_order, "filled_qty", 0.0) or 0.0)
        order_qty = float(getattr(active_order, "qty", 0.0) or 0.0)
        current_limit = float(getattr(active_order, "limit_price", current_limit) or current_limit)

        if status != last_status or not math.isclose(filled_qty, last_filled_qty, abs_tol=1e-9):
            logger.info(
                "📈 Order status | %s | order_id=%s | status=%s | filled=%.2f/%.2f | limit=%+.2f",
                order_label,
                active_order_id,
                status or "unknown",
                filled_qty,
                order_qty,
                current_limit,
            )
            last_status = status
            last_filled_qty = filled_qty

        if status in FILLED_ORDER_STATUSES:
            send_alert(f"✅ **ORDER FILLED**\n{order_label}\nOrder ID: `{active_order_id}`", "SUCCESS")
            return MonitoredOrderResult(
                order_id=active_order_id,
                final_status=status,
                filled_qty=filled_qty,
                limit_price=current_limit,
                reprices=reprices,
                filled=True,
                partial_fill=False,
                pricing_confidence=getattr(last_snapshot, "pricing_confidence", None),
                natural_price=getattr(last_snapshot, "natural_price", None),
                fair_price=getattr(last_snapshot, "fair_price", None),
                mc_expected_price=getattr(last_snapshot, "mc_expected_price", None),
                mc_var_95=getattr(last_snapshot, "mc_var_95", None),
                mc_cvar_95=getattr(last_snapshot, "mc_cvar_95", None),
            )

        if status in TERMINAL_ORDER_STATUSES:
            break

        if reprices < max_reprices and status in OPEN_ORDER_STATUSES:
            snapshot = snapshot_builder(client=client, legs=legs)
            proposed_limit = limit_reprice_func(snapshot, current_limit, reprices + 1, is_credit)
            if snapshot is not None:
                last_snapshot = snapshot
            if proposed_limit is not None and abs(proposed_limit - current_limit) >= float(ORDER_MONITOR_MIN_PRICE_STEP) - 1e-9:
                replacement = client.trade_client.replace_order_by_id(
                    active_order_id,
                    ReplaceOrderRequest(limit_price=round(float(proposed_limit), 2)),
                )
                active_order_id = str(getattr(replacement, "id", active_order_id))
                current_limit = round(float(proposed_limit), 2)
                reprices += 1
                logger.info(
                    "🔁 Repriced %s | attempt=%d/%d | new_limit=%+.2f | fair=%+.2f | natural=%+.2f | conf=%.2f | mc_var=%.2f",
                    order_label,
                    reprices,
                    max_reprices,
                    current_limit,
                    float(snapshot.fair_price if snapshot else 0.0),
                    float(snapshot.natural_price if snapshot else 0.0),
                    float(snapshot.pricing_confidence if snapshot else 0.0),
                    float(snapshot.mc_var_95 if snapshot else 0.0),
                )

        if poll_seconds > 0:
            sleep_fn(float(poll_seconds))

    final_status = _status_text(getattr(active_order, "status", None))
    final_filled_qty = float(getattr(active_order, "filled_qty", 0.0) or 0.0)

    if final_status in OPEN_ORDER_STATUSES:
        try:
            client.trade_client.cancel_order_by_id(active_order_id)
            final_status = "canceled"
        except Exception as exc:
            logger.warning("Could not cancel unfilled order %s: %s", active_order_id, exc)

    send_alert(
        f"⚠️ **ORDER NOT FILLED**\n{order_label}\nFinal status: `{final_status or 'unknown'}`\n"
        f"Filled qty: `{final_filled_qty}`\nReprices: `{reprices}`",
        "WARNING",
    )
    return MonitoredOrderResult(
        order_id=active_order_id,
        final_status=final_status,
        filled_qty=final_filled_qty,
        limit_price=current_limit,
        reprices=reprices,
        filled=False,
        partial_fill=final_filled_qty > 0.0,
        pricing_confidence=getattr(last_snapshot, "pricing_confidence", None),
        natural_price=getattr(last_snapshot, "natural_price", None),
        fair_price=getattr(last_snapshot, "fair_price", None),
        mc_expected_price=getattr(last_snapshot, "mc_expected_price", None),
        mc_var_95=getattr(last_snapshot, "mc_var_95", None),
        mc_cvar_95=getattr(last_snapshot, "mc_cvar_95", None),
    )
