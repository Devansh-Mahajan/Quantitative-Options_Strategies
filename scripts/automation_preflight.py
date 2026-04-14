from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.system_preflight import DEFAULT_STATE_PATH, REPO_ROOT, run_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the automation stack before live actions run.")
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--max-age-seconds", type=int, default=300)
    parser.add_argument(
        "--deep-model-checks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load critical model artifacts during validation instead of only checking that they exist.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Suppress progress lines and only print the final JSON payload.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    def _callback(percent: int, message: str, detail: str | None = None) -> None:
        if args.json_only:
            return
        suffix = f" | {detail}" if detail else ""
        print(f"PRECHECK {percent:3d}% | {message}{suffix}")

    result = run_preflight(
        root=REPO_ROOT,
        state_path=Path(args.state_path),
        max_age_seconds=args.max_age_seconds,
        progress_callback=_callback,
        deep_model_checks=args.deep_model_checks,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
