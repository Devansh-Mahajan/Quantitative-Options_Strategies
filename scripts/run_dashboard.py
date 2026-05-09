#!/usr/bin/env python3
"""
Elite Quant Trading Dashboard — FastAPI server on port 8080.

Usage:
    python -m scripts.run_dashboard              # Start dashboard server
    python -m scripts.run_dashboard --port 9000  # Custom port
"""

import argparse
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant Trading Dashboard")
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run dashboard on (default: 8080)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0 — network accessible)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on file changes (development only)"
    )
    args = parser.parse_args()

    uvicorn.run(
        "dashboard.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
