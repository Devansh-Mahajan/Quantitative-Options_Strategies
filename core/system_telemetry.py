from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from core.resource_profile import ResourceProfile, load_resource_profile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEM_SNAPSHOT_PATH = ROOT / ".runtime" / "system_resource_snapshot.json"
DEFAULT_RISK_SNAPSHOT_PATH = ROOT / ".runtime" / "risk_snapshot.json"
DEFAULT_PORTFOLIO_RISK_GUARD_SNAPSHOT_PATH = ROOT / ".runtime" / "portfolio_risk_guard.json"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _memory_snapshot() -> dict[str, float]:
    metrics = {
        "total_gb": 0.0,
        "available_gb": 0.0,
        "used_gb": 0.0,
        "usage_pct": 0.0,
    }
    try:
        meminfo = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                meminfo[key.strip()] = int(value.strip().split()[0])
        total_kb = float(meminfo.get("MemTotal", 0.0))
        available_kb = float(meminfo.get("MemAvailable", meminfo.get("MemFree", 0.0)))
        used_kb = max(0.0, total_kb - available_kb)
        metrics["total_gb"] = round(total_kb / (1024 ** 2), 2)
        metrics["available_gb"] = round(available_kb / (1024 ** 2), 2)
        metrics["used_gb"] = round(used_kb / (1024 ** 2), 2)
        metrics["usage_pct"] = round((used_kb / total_kb) * 100.0, 2) if total_kb > 0 else 0.0
    except OSError:
        pass
    return metrics


def _disk_snapshot(path: Path) -> dict[str, float]:
    try:
        usage = shutil.disk_usage(path)
        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)
        free_gb = usage.free / (1024 ** 3)
        usage_pct = (used_gb / total_gb) * 100.0 if total_gb > 0 else 0.0
        return {
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(free_gb, 2),
            "usage_pct": round(usage_pct, 2),
        }
    except OSError:
        return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0, "usage_pct": 0.0}


def build_system_resource_snapshot(
    *,
    repo_root: Path = ROOT,
    profile: ResourceProfile | None = None,
) -> dict:
    profile = profile or load_resource_profile(repo_root)
    loadavg = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
    normalized_load = round((loadavg[0] / max(1, profile.cpu_count)) * 100.0, 2)
    return {
        "generated_at_utc": _utc_now(),
        "resource_profile": profile.to_dict(),
        "host_metrics": {
            "loadavg_1m": round(loadavg[0], 2),
            "loadavg_5m": round(loadavg[1], 2),
            "loadavg_15m": round(loadavg[2], 2),
            "normalized_cpu_load_pct": normalized_load,
            "memory": _memory_snapshot(),
            "disk": _disk_snapshot(repo_root),
        },
        "status": {
            "pressure": "high" if normalized_load >= 85.0 else "elevated" if normalized_load >= 65.0 else "normal",
            "note": "Host utilization is a planning signal, not a guarantee of task health.",
        },
    }


def write_system_resource_snapshot(
    path: Path = DEFAULT_SYSTEM_SNAPSHOT_PATH,
    *,
    repo_root: Path = ROOT,
    profile: ResourceProfile | None = None,
) -> dict:
    snapshot = build_system_resource_snapshot(repo_root=repo_root, profile=profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot


def write_risk_snapshot(path: Path = DEFAULT_RISK_SNAPSHOT_PATH, payload: dict | None = None) -> dict:
    snapshot = dict(payload or {})
    snapshot.setdefault("generated_at_utc", _utc_now())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot
