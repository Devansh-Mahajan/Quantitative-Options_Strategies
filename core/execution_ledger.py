from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from core.execution_quality import assess_execution_quality


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTION_LEDGER_PATH = ROOT / ".runtime" / "execution_ledger.json"
DEFAULT_EXECUTION_SUMMARY_PATH = ROOT / ".runtime" / "execution_quality_snapshot.json"
DEFAULT_EXECUTION_LEDGER_MAX_RECORDS = 600


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_timestamp(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "tzinfo"):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _normalize_fill_price(fill_price: float | None, limit_price: float | None, is_credit: bool) -> float | None:
    price = _safe_float(fill_price)
    if price is None:
        return None
    if is_credit:
        return -abs(price)
    if limit_price is not None and float(limit_price) < 0:
        return -abs(price)
    return abs(price)


def _leg_payload(legs: list[object] | None = None) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for leg in list(legs or []):
        symbol = getattr(leg, "symbol", None)
        side = getattr(leg, "side", None)
        if symbol:
            payload.append(
                {
                    "symbol": str(symbol),
                    "side": str(side or "").lower(),
                }
            )
    return payload


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [dict(item) for item in payload["records"] if isinstance(item, dict)]
    return []


def _record_key(record: dict) -> str:
    order_id = str(record.get("order_id") or "").strip()
    if order_id:
        return order_id
    return json.dumps(
        [
            record.get("order_label"),
            record.get("submitted_at_utc"),
            record.get("limit_price"),
            record.get("legs"),
        ],
        sort_keys=True,
    )


def _sort_key(record: dict) -> str:
    return str(
        record.get("filled_at_utc")
        or record.get("updated_at_utc")
        or record.get("submitted_at_utc")
        or record.get("recorded_at_utc")
        or ""
    )


def _adaptive_reprice_factor(fill_rate: float | None, avg_quality: float | None) -> float:
    factor = 1.0
    if fill_rate is not None and fill_rate < 0.45:
        factor += min(0.18, (0.45 - fill_rate) * 0.45)
    if avg_quality is not None and avg_quality < 0.56:
        factor -= min(0.12, (0.56 - avg_quality) * 0.40)
    if avg_quality is not None and avg_quality > 0.72 and fill_rate is not None and fill_rate < 0.60:
        factor += min(0.05, (avg_quality - 0.72) * 0.20)
    return round(max(0.85, min(1.20, factor)), 4)


def summarize_execution_records(records: list[dict]) -> dict:
    fills = [record for record in records if float(record.get("filled_qty") or 0.0) > 0.0]
    full_fills = [record for record in fills if record.get("filled") is True]
    partial_fills = [record for record in fills if record.get("partial_fill") is True]

    quality_scores = [
        float((record.get("execution_quality") or {}).get("score"))
        for record in fills
        if (record.get("execution_quality") or {}).get("score") is not None
    ]
    limit_edges = [
        float((record.get("execution_quality") or {}).get("limit_edge_bps"))
        for record in fills
        if (record.get("execution_quality") or {}).get("limit_edge_bps") is not None
    ]
    reference_edges = [
        float((record.get("execution_quality") or {}).get("reference_edge_bps"))
        for record in fills
        if (record.get("execution_quality") or {}).get("reference_edge_bps") is not None
    ]
    pricing_confidences = [
        float((record.get("pricing_snapshot") or {}).get("pricing_confidence"))
        for record in fills
        if (record.get("pricing_snapshot") or {}).get("pricing_confidence") is not None
    ]
    staleness_values = [
        float((record.get("pricing_snapshot") or {}).get("staleness_pct"))
        for record in fills
        if (record.get("pricing_snapshot") or {}).get("staleness_pct") is not None
    ]
    tiers = [str((record.get("execution_quality") or {}).get("tier")) for record in fills if (record.get("execution_quality") or {}).get("tier")]
    tier_counts = {tier: tiers.count(tier) for tier in sorted(set(tiers))}
    degraded_count = sum(1 for tier in tiers if tier in {"degraded", "poor"})
    broker_fill_coverage = sum(
        1
        for record in fills
        if bool((record.get("execution_quality") or {}).get("broker_fill_observed"))
    )

    fill_rate = (len(fills) / len(records)) if records else None
    full_fill_rate = (len(full_fills) / len(records)) if records else None
    avg_quality = (sum(quality_scores) / len(quality_scores)) if quality_scores else None

    latest_fill_record = None
    if fills:
        latest_fill_record = max(fills, key=_sort_key)

    return {
        "generated_at_utc": _utc_now_iso(),
        "records": len(records),
        "fill_events": len(fills),
        "full_fills": len(full_fills),
        "partial_fills": len(partial_fills),
        "fill_rate": round(fill_rate, 6) if fill_rate is not None else None,
        "full_fill_rate": round(full_fill_rate, 6) if full_fill_rate is not None else None,
        "broker_fill_price_coverage": round(broker_fill_coverage / len(fills), 6) if fills else None,
        "avg_execution_quality_score": round(avg_quality, 6) if avg_quality is not None else None,
        "avg_limit_edge_bps": round(sum(limit_edges) / len(limit_edges), 4) if limit_edges else None,
        "avg_reference_edge_bps": round(sum(reference_edges) / len(reference_edges), 4) if reference_edges else None,
        "avg_pricing_confidence": round(sum(pricing_confidences) / len(pricing_confidences), 6) if pricing_confidences else None,
        "avg_staleness_pct": round(sum(staleness_values) / len(staleness_values), 6) if staleness_values else None,
        "degraded_execution_count": degraded_count,
        "tier_counts": tier_counts,
        "adaptive_reprice_factor": _adaptive_reprice_factor(fill_rate, avg_quality),
        "latest_fill_at_utc": None if latest_fill_record is None else latest_fill_record.get("filled_at_utc") or latest_fill_record.get("updated_at_utc"),
        "latest_order_id": None if latest_fill_record is None else latest_fill_record.get("order_id"),
        "note": "Execution summary blends broker-reported fills with delay-aware pricing references. Lower-confidence records are penalized conservatively.",
    }


def _persist_records(
    records: list[dict],
    *,
    ledger_path: Path,
    summary_path: Path,
    max_records: int = DEFAULT_EXECUTION_LEDGER_MAX_RECORDS,
) -> dict:
    trimmed = sorted(records, key=_sort_key)[-max(1, int(max_records)) :]
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")
    summary = summarize_execution_records(trimmed)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def upsert_execution_records(
    new_records: list[dict],
    *,
    ledger_path: Path = DEFAULT_EXECUTION_LEDGER_PATH,
    summary_path: Path = DEFAULT_EXECUTION_SUMMARY_PATH,
    max_records: int = DEFAULT_EXECUTION_LEDGER_MAX_RECORDS,
) -> dict:
    merged = {_record_key(record): dict(record) for record in _read_records(ledger_path)}
    for record in new_records:
        merged[_record_key(record)] = dict(record)
    return _persist_records(list(merged.values()), ledger_path=ledger_path, summary_path=summary_path, max_records=max_records)


def load_execution_feedback(summary_path: Path = DEFAULT_EXECUTION_SUMMARY_PATH) -> dict:
    if not summary_path.exists():
        return {"adaptive_reprice_factor": 1.0}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"adaptive_reprice_factor": 1.0}
    if not isinstance(payload, dict):
        return {"adaptive_reprice_factor": 1.0}
    payload.setdefault("adaptive_reprice_factor", 1.0)
    return payload


def record_monitored_order(
    *,
    order_label: str,
    order,
    result,
    legs: list[object],
    is_credit: bool,
    pricing_snapshot=None,
    source: str = "order_monitor",
    ledger_path: Path = DEFAULT_EXECUTION_LEDGER_PATH,
    summary_path: Path = DEFAULT_EXECUTION_SUMMARY_PATH,
) -> dict:
    fill_ratio = 1.0 if result.filled else 0.0
    order_qty = _safe_float(getattr(order, "qty", None))
    if fill_ratio <= 0.0:
        if order_qty and order_qty > 0:
            fill_ratio = max(0.0, min(1.0, float(result.filled_qty or 0.0) / order_qty))
        elif float(result.filled_qty or 0.0) > 0.0:
            fill_ratio = 1.0

    fill_price = _normalize_fill_price(
        getattr(order, "filled_avg_price", getattr(result, "filled_avg_price", None)),
        result.limit_price,
        is_credit=is_credit,
    )
    assessment = assess_execution_quality(
        fill_price=fill_price,
        limit_price=result.limit_price,
        reference_price=getattr(pricing_snapshot, "fair_price", getattr(result, "fair_price", None)),
        pricing_confidence=getattr(pricing_snapshot, "pricing_confidence", getattr(result, "pricing_confidence", None)),
        staleness_pct=getattr(pricing_snapshot, "staleness_pct", None),
        is_credit=is_credit,
        fill_ratio=fill_ratio,
        broker_fill_observed=fill_price is not None,
    )

    record = {
        "recorded_at_utc": _utc_now_iso(),
        "source": source,
        "order_id": str(result.order_id or getattr(order, "id", "")),
        "order_label": order_label,
        "status": result.final_status,
        "filled": bool(result.filled),
        "partial_fill": bool(result.partial_fill),
        "filled_qty": float(result.filled_qty or 0.0),
        "qty": _safe_float(getattr(order, "qty", None)),
        "limit_price": _safe_float(result.limit_price),
        "filled_avg_price": fill_price,
        "fill_source": assessment.fill_source,
        "is_credit": bool(is_credit),
        "reprices": int(result.reprices or 0),
        "submitted_at_utc": _normalize_timestamp(getattr(order, "submitted_at", getattr(result, "submitted_at", None))),
        "updated_at_utc": _normalize_timestamp(getattr(order, "updated_at", getattr(result, "updated_at", None))),
        "filled_at_utc": _normalize_timestamp(getattr(order, "filled_at", getattr(result, "filled_at", None))),
        "legs": _leg_payload(legs),
        "pricing_snapshot": None
        if pricing_snapshot is None
        else {
            "natural_price": _safe_float(getattr(pricing_snapshot, "natural_price", None)),
            "fair_price": _safe_float(getattr(pricing_snapshot, "fair_price", None)),
            "pricing_confidence": _safe_float(getattr(pricing_snapshot, "pricing_confidence", None)),
            "staleness_pct": _safe_float(getattr(pricing_snapshot, "staleness_pct", None)),
            "underlying_price": _safe_float(getattr(pricing_snapshot, "underlying_price", None)),
            "mc_expected_price": _safe_float(getattr(pricing_snapshot, "mc_expected_price", None)),
            "mc_var_95": _safe_float(getattr(pricing_snapshot, "mc_var_95", None)),
            "mc_cvar_95": _safe_float(getattr(pricing_snapshot, "mc_cvar_95", None)),
        },
        "execution_quality": {
            "score": assessment.score,
            "tier": assessment.tier,
            "fill_ratio": assessment.fill_ratio,
            "fill_source": assessment.fill_source,
            "broker_fill_observed": assessment.broker_fill_observed,
            "limit_edge_bps": assessment.limit_edge_bps,
            "reference_edge_bps": assessment.reference_edge_bps,
        },
    }
    summary = upsert_execution_records([record], ledger_path=ledger_path, summary_path=summary_path)
    return {"record": record, "summary": summary}


def _extract_broker_order_legs(order) -> list[dict[str, str]]:
    nested_legs = getattr(order, "legs", None)
    if nested_legs:
        return _leg_payload(nested_legs)

    symbol = getattr(order, "symbol", None)
    if symbol:
        return [{"symbol": str(symbol), "side": str(getattr(order, "side", "") or "").lower()}]
    return []


def reconcile_recent_order_fills(
    client,
    *,
    lookback_hours: float = 36.0,
    max_orders: int = 128,
    ledger_path: Path = DEFAULT_EXECUTION_LEDGER_PATH,
    summary_path: Path = DEFAULT_EXECUTION_SUMMARY_PATH,
) -> dict:
    existing_records = _read_records(ledger_path)
    existing_map = {_record_key(record): dict(record) for record in existing_records}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1.0, float(lookback_hours)))
    request = GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=cutoff, limit=max(1, int(max_orders)))
    orders = list(client.trade_client.get_orders(filter=request) or [])

    reconciled = 0
    broker_fill_updates = 0
    for order in orders:
        status = str(getattr(order, "status", "") or "").strip().lower()
        filled_qty = float(getattr(order, "filled_qty", 0.0) or 0.0)
        order_id = str(getattr(order, "id", "") or "")
        if not order_id:
            continue
        if filled_qty <= 0.0 and status not in {"filled", "partially_filled"}:
            continue

        limit_price = _safe_float(getattr(order, "limit_price", None))
        fill_price = _normalize_fill_price(
            getattr(order, "filled_avg_price", None),
            limit_price,
            is_credit=bool(limit_price is not None and limit_price < 0),
        )
        qty = _safe_float(getattr(order, "qty", None))
        fill_ratio = 1.0 if status == "filled" else 0.0
        if fill_ratio <= 0.0:
            if qty and qty > 0:
                fill_ratio = max(0.0, min(1.0, filled_qty / qty))
            elif filled_qty > 0.0:
                fill_ratio = 1.0

        assessment = assess_execution_quality(
            fill_price=fill_price,
            limit_price=limit_price,
            is_credit=bool(limit_price is not None and limit_price < 0),
            fill_ratio=fill_ratio,
            broker_fill_observed=fill_price is not None,
        )

        key = _record_key({"order_id": order_id})
        existing = existing_map.get(key) or {}
        if fill_price is not None and not bool((existing.get("execution_quality") or {}).get("broker_fill_observed")):
            broker_fill_updates += 1

        existing_map[key] = {
            "recorded_at_utc": existing.get("recorded_at_utc") or _utc_now_iso(),
            "source": "broker_reconciliation",
            "order_id": order_id,
            "order_label": str(existing.get("order_label") or getattr(order, "client_order_id", "") or getattr(order, "symbol", "") or order_id),
            "status": status,
            "filled": status == "filled",
            "partial_fill": status == "partially_filled" or (filled_qty > 0.0 and status != "filled"),
            "filled_qty": filled_qty,
            "qty": qty,
            "limit_price": limit_price,
            "filled_avg_price": fill_price,
            "fill_source": assessment.fill_source,
            "is_credit": bool(limit_price is not None and limit_price < 0),
            "reprices": int(existing.get("reprices") or 0),
            "submitted_at_utc": _normalize_timestamp(getattr(order, "submitted_at", None)),
            "updated_at_utc": _normalize_timestamp(getattr(order, "updated_at", None)),
            "filled_at_utc": _normalize_timestamp(getattr(order, "filled_at", None)),
            "legs": existing.get("legs") or _extract_broker_order_legs(order),
            "pricing_snapshot": existing.get("pricing_snapshot"),
            "execution_quality": {
                "score": assessment.score,
                "tier": assessment.tier,
                "fill_ratio": assessment.fill_ratio,
                "fill_source": assessment.fill_source,
                "broker_fill_observed": assessment.broker_fill_observed,
                "limit_edge_bps": assessment.limit_edge_bps,
                "reference_edge_bps": assessment.reference_edge_bps,
            },
        }
        reconciled += 1

    summary = _persist_records(list(existing_map.values()), ledger_path=ledger_path, summary_path=summary_path)
    return {
        "checked_orders": len(orders),
        "reconciled_records": reconciled,
        "broker_fill_updates": broker_fill_updates,
        "ledger_records": len(existing_map),
        "adaptive_reprice_factor": summary.get("adaptive_reprice_factor"),
        "fill_rate": summary.get("fill_rate"),
        "avg_execution_quality_score": summary.get("avg_execution_quality_score"),
    }
