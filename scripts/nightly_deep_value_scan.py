"""
Nightly deep-value (Graham net-net) scan.

Pulls the day's movers from Yahoo's screener, fetches balance-sheet
fundamentals for the shortlist, and writes scored candidates to
.runtime/deep_value_scan.json for the next trading day's strategy cycle
(core/execution/deep_value_sleeve.py). Every scanned ticker is also appended
to .runtime/deep_value_history.jsonl as future ML training data.

Invoked overnight by scripts/automation_controller.py (--deep-value-scan-command);
safe to run manually: python -m scripts.nightly_deep_value_scan --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import DEEP_VALUE_ALERT_SCORE, DEEP_VALUE_MIN_MARKET_CAP, DEEP_VALUE_MIN_PRICE
from core.ml.deep_value import (
    append_training_snapshot,
    build_candidate,
    fetch_fundamentals,
    fetch_screener_candidates,
    save_scan_snapshot,
)
from core.telemetry.notifications import send_alert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nightly Graham net-net deep-value scan.")
    parser.add_argument("--limit-per-screen", type=int, default=50)
    parser.add_argument("--max-fundamentals", type=int, default=60, help="cap on per-ticker balance-sheet fetches")
    parser.add_argument("--fetch-sleep", type=float, default=0.5, help="seconds between fundamentals fetches")
    parser.add_argument("--dry-run", action="store_true", help="scan and print, but write no snapshot/history")
    return parser.parse_args()


def _prefilter(quotes: list[dict]) -> list[dict]:
    """Cheap gate on screener fields only — no extra network per ticker."""
    kept = []
    for quote in quotes:
        price = quote.get("price")
        if price is None or price < DEEP_VALUE_MIN_PRICE:
            continue
        market_cap = quote.get("market_cap") or 0.0
        if market_cap and market_cap < DEEP_VALUE_MIN_MARKET_CAP:
            continue
        kept.append(quote)
    return kept


def main() -> int:
    args = parse_args()

    try:
        quotes = fetch_screener_candidates(limit_per_screen=args.limit_per_screen)
    except Exception as exc:
        print(f"[deep-value-scan] FATAL: {exc}", file=sys.stderr)
        return 1

    shortlist = _prefilter(quotes)[: max(1, args.max_fundamentals)]
    print(f"[deep-value-scan] screener rows={len(quotes)} shortlist={len(shortlist)}")

    candidates = []
    history_rows = []
    for index, quote in enumerate(shortlist):
        symbol = quote["symbol"]
        fundamentals = fetch_fundamentals(symbol, quote)
        if index < len(shortlist) - 1 and args.fetch_sleep > 0:
            time.sleep(args.fetch_sleep)
        if fundamentals is None:
            continue
        candidate = build_candidate(symbol, fundamentals)
        if candidate is None:
            continue
        candidates.append(candidate)
        history_rows.append(asdict(candidate))

    passing = [c for c in candidates if not c.failed_gates]
    passing.sort(key=lambda c: c.score, reverse=True)
    print(f"[deep-value-scan] fundamentals ok={len(candidates)} passing all gates={len(passing)}")

    if not args.dry_run:
        save_scan_snapshot(candidates)
        append_training_snapshot(history_rows)
        for candidate in passing:
            if candidate.score >= DEEP_VALUE_ALERT_SCORE:
                send_alert(
                    f"💎 Deep value candidate {candidate.symbol}: price ${candidate.price:.2f} vs "
                    f"NCAV/share ${candidate.ncav_per_share:.2f} (score {candidate.score:.2f})",
                    level="WARNING",
                )

    summary = {
        "screener_rows": len(quotes),
        "shortlist": len(shortlist),
        "with_fundamentals": len(candidates),
        "passing": [
            {
                "symbol": c.symbol,
                "price": c.price,
                "ncav_per_share": c.ncav_per_share,
                "liquidation_per_share": c.liquidation_per_share,
                "score": c.score,
                "model_probability": c.model_probability,
            }
            for c in passing[:10]
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
