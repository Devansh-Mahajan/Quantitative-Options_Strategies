"""
Main bot entry point.
Usage:
    python -m scripts.run_bot
    # or after pip install -e .:
    run-bot
"""

from __future__ import annotations
import asyncio
import logging
import signal
import sys

from bot.config import cfg
from bot.logger import setup_logging

setup_logging(cfg.log_dir, cfg.log_level)
log = logging.getLogger("scripts.run_bot")


async def _main() -> None:
    from bot.orchestrator import Orchestrator

    orch = Orchestrator()

    # Graceful shutdown on SIGTERM / SIGINT
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(_shutdown(orch)))

    await orch.run()


async def _shutdown(orch) -> None:
    log.info("Shutdown signal received — closing positions and stopping...")
    try:
        if orch.stream_manager:
            await orch.stream_manager.stop()
        await orch.client.close()
    finally:
        sys.exit(0)


def main() -> None:
    log.info("Binance Quant Bot v2.0 — starting")
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        log.info("Bot stopped by keyboard interrupt")
    except SystemExit:
        pass
    except Exception as exc:
        log.critical("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
