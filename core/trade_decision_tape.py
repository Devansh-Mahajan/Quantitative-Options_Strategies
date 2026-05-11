from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECISION_TAPE_PATH = ROOT / ".runtime" / "trade_decision_tape.jsonl"


def record_trade_decision(
    *,
    status: str,
    strategy: str,
    symbol: str | None = None,
    reason: str = "",
    action: str = "",
    confidence: float | None = None,
    risk_cap: float | None = None,
    buying_power: float | None = None,
    details: dict[str, Any] | None = None,
    path: Path = DEFAULT_DECISION_TAPE_PATH,
) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "status": str(status or "INFO").upper(),
        "strategy": str(strategy or "unknown"),
        "symbol": str(symbol or ""),
        "action": str(action or ""),
        "reason": str(reason or ""),
        "confidence": confidence,
        "risk_cap": risk_cap,
        "buying_power": buying_power,
        "details": details or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")


def read_trade_decisions(
    *,
    path: Path = DEFAULT_DECISION_TAPE_PATH,
    limit: int = 500,
    status: str | None = None,
    strategy: str | None = None,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    status_filter = str(status or "").strip().upper()
    strategy_filter = str(strategy or "").strip().lower()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if status_filter and str(row.get("status", "")).upper() != status_filter:
                continue
            if strategy_filter and strategy_filter not in str(row.get("strategy", "")).lower():
                continue
            rows.append(row)
    return rows[-max(1, int(limit)) :]
