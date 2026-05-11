#!/usr/bin/env python3
"""Verify the legacy dashboard keeps its required tabs and JS hooks.

This is intentionally lightweight: it catches the common failure mode where a
visual overhaul replaces the full dashboard with a condensed mock that keeps
the header/nav but drops tab panels or loader functions.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "dashboard" / "static" / "index.html"

REQUIRED_TABS = [
    "live",
    "options",
    "stocks",
    "odds",
    "backtests",
    "params",
    "risk",
    "simlab",
    "stratlab",
    "ftest",
    "training",
    "intelligence",
    "tradeanalysis",
    "cmdcenter",
    "gsquant",
    "logs",
]

OPTIONAL_BUT_EXPECTED_TABS = ["dtwin"]

REQUIRED_HOOKS = [
    "loadLive",
    "connectLiveWs",
    "loadOptions",
    "loadStocks",
    "loadOdds",
    "loadBacktests",
    "loadParams",
    "loadRisk",
    "loadSimLab",
    "loadStratLab",
    "loadTraining",
    "loadIntelligence",
    "loadTradeAnalysis",
    "loadCmdCenter",
    "loadGsQuant",
    "initLogs",
    "triggerMasterRecal",
    "api",
    "TAB_INIT",
]


def has_function_or_binding(source: str, name: str) -> bool:
    if name == "TAB_INIT":
        return "const TAB_INIT" in source or "let TAB_INIT" in source or "var TAB_INIT" in source
    patterns = [
        rf"\bfunction\s+{re.escape(name)}\s*\(",
        rf"\basync\s+function\s+{re.escape(name)}\s*\(",
        rf"\bconst\s+{re.escape(name)}\s*=",
        rf"\blet\s+{re.escape(name)}\s*=",
    ]
    return any(re.search(pattern, source) for pattern in patterns)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    source = path.read_text(encoding="utf-8")

    missing_tabs = [
        tab
        for tab in REQUIRED_TABS
        if f'data-tab="{tab}"' not in source or f'id="tab-{tab}"' not in source
    ]
    missing_hooks = [hook for hook in REQUIRED_HOOKS if not has_function_or_binding(source, hook)]
    missing_optional = [
        tab
        for tab in OPTIONAL_BUT_EXPECTED_TABS
        if f'data-tab="{tab}"' not in source or f'id="tab-{tab}"' not in source
    ]

    stats = {
        "lines": len(source.splitlines()),
        "nav_tabs": source.count('class="ntab'),
        "tab_panels": source.count('class="tab'),
        "api_call_sites": source.count("api('/api/") + source.count("api(`/api/"),
        "canvas_tags": source.count("<canvas"),
        "websocket_refs": source.count("/ws/"),
    }

    print("Dashboard contract stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if missing_tabs:
        print(f"Missing required tabs: {', '.join(missing_tabs)}", file=sys.stderr)
    if missing_hooks:
        print(f"Missing required hooks: {', '.join(missing_hooks)}", file=sys.stderr)
    if missing_optional:
        print(f"Missing expected extension tabs: {', '.join(missing_optional)}", file=sys.stderr)

    if stats["tab_panels"] < len(REQUIRED_TABS):
        print("Too few tab panels; likely truncated dashboard HTML.", file=sys.stderr)
        return 1
    if stats["api_call_sites"] < 50:
        print("Too few API call sites; backend functionality may have been replaced by stubs.", file=sys.stderr)
        return 1
    if missing_tabs or missing_hooks:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
