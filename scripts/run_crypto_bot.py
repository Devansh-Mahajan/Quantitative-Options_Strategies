"""
Supervised Alpaca crypto daemon.

This keeps the async strategy stack in a crypto-only, spot-safe mode so it can
share one Alpaca account with the options/equity runner without pretending the
broker supports perpetual shorts.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from bot.config import cfg
from bot.logger import setup_logging
from bot.orchestrator import ALPACA_CRYPTO_SPOT_STRATEGIES, Orchestrator

log = logging.getLogger("scripts.run_crypto_bot")


def _parse_symbols(raw: str, fallback: list[str]) -> list[str]:
    symbols = [str(item or "").strip().upper() for item in str(raw or "").split(",")]
    cleaned = [symbol for symbol in symbols if symbol]
    return cleaned or list(fallback)


def parse_args() -> argparse.Namespace:
    default_runtime = list(cfg.crypto_symbols)
    default_models = default_runtime[: max(1, min(4, len(default_runtime)))]
    parser = argparse.ArgumentParser(description="Run the Alpaca crypto daemon in spot-safe mode.")
    parser.add_argument("--symbols", default=",".join(default_runtime))
    parser.add_argument("--model-symbols", default=",".join(default_models))
    parser.add_argument(
        "--primary-regime-symbol",
        default=(default_models[0] if default_models else "BTCUSDT"),
    )
    parser.add_argument("--log-level", default=cfg.log_level)
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    setup_logging(cfg.log_dir, args.log_level)
    runtime_symbols = _parse_symbols(args.symbols, cfg.crypto_symbols)
    if not runtime_symbols:
        raise RuntimeError("No crypto symbols configured for the Alpaca crypto daemon.")
    model_symbols = _parse_symbols(args.model_symbols, runtime_symbols[: max(1, min(4, len(runtime_symbols)))])
    orchestrator = Orchestrator(
        runtime_symbols=runtime_symbols,
        model_symbols=model_symbols,
        primary_regime_symbol=str(args.primary_regime_symbol or runtime_symbols[0]).upper(),
        book_name="crypto",
        long_only_spot=True,
        strategy_whitelist=ALPACA_CRYPTO_SPOT_STRATEGIES,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(_shutdown(orchestrator)))

    log.info(
        "Launching Alpaca crypto daemon | runtime_symbols=%s | model_symbols=%s | strategies=%s",
        runtime_symbols,
        model_symbols,
        sorted(ALPACA_CRYPTO_SPOT_STRATEGIES),
    )
    await orchestrator.run()


async def _shutdown(orchestrator: Orchestrator) -> None:
    log.info("Crypto daemon shutdown signal received")
    loop = asyncio.get_event_loop()
    try:
        if orchestrator.stream_manager:
            await orchestrator.stream_manager.stop()
        await orchestrator.client.close()
    except Exception as exc:
        log.warning("Cleanup error during crypto daemon shutdown: %s", exc)
    finally:
        loop.stop()


def main() -> None:
    if not cfg.is_alpaca:
        log.critical("scripts.run_crypto_bot requires BROKER=alpaca.")
        raise SystemExit(2)

    args = parse_args()
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        log.info("Crypto daemon stopped by keyboard interrupt")
    except Exception as exc:
        log.critical("Crypto daemon fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
