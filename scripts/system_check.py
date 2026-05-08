"""
Pre-flight system check — run before starting the bot.
Validates: credentials, GPU, Python packages, disk space, Binance connectivity.

Usage:
    python -m scripts.system_check
    # or:
    system-check
"""

from __future__ import annotations
import asyncio
import logging
import shutil
import sys

from bot.config import cfg
from bot.logger import setup_logging

setup_logging(cfg.log_dir, "INFO")
log = logging.getLogger("scripts.system_check")


REQUIRED_PACKAGES = [
    "torch", "numpy", "pandas", "hmmlearn",
    "sklearn", "stable_baselines3", "arch", "httpx",
]

# Broker-specific packages checked separately
ALPACA_PACKAGES = ["alpaca"]
BINANCE_PACKAGES = ["binance"]

MIN_DISK_GB = 5.0


def check_packages() -> bool:
    ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
            log.info("  [OK] %s", pkg)
        except ImportError:
            log.error("  [MISSING] %s — run: pip install -e .", pkg)
            ok = False

    broker_pkgs = ALPACA_PACKAGES if cfg.is_alpaca else BINANCE_PACKAGES
    for pkg in broker_pkgs:
        try:
            __import__(pkg)
            log.info("  [OK] %s (broker=%s)", pkg, cfg.broker)
        except ImportError:
            install_hint = 'pip install -e ".[alpaca]"' if cfg.is_alpaca else "pip install python-binance"
            log.error("  [MISSING] %s — run: %s", pkg, install_hint)
            ok = False
    return ok


def check_gpu() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info("  [OK] GPU: %s (%.1f GB VRAM)", name, vram)
            return True
        else:
            log.warning("  [WARN] CUDA not available — will use CPU (slower training)")
            return True  # not fatal
    except ImportError:
        log.warning("  [WARN] PyTorch not installed")
        return False


def check_disk() -> bool:
    usage = shutil.disk_usage(".")
    free_gb = usage.free / 1e9
    if free_gb < MIN_DISK_GB:
        log.error("  [FAIL] Only %.1f GB free — need at least %.1f GB", free_gb, MIN_DISK_GB)
        return False
    log.info("  [OK] Disk: %.1f GB free", free_gb)
    return True


def check_env() -> bool:
    try:
        cfg.validate()
        log.info("  [OK] .env loaded — API key: %s...%s", cfg.api_key[:4], cfg.api_key[-4:])
        return True
    except ValueError as exc:
        log.error("  [FAIL] %s", exc)
        return False


async def check_broker_connectivity() -> bool:
    from exchange import get_client
    client = get_client()
    try:
        await client.start()
        ts = await client.get_server_time()
        price = await client.get_price("BTCUSDT")
        await client.close()
        log.info("  [OK] %s API connected — BTC price=%.2f USD(T), server_time=%d",
                 cfg.broker.upper(), price, ts)
        return True
    except Exception as exc:
        log.error("  [FAIL] %s API error: %s", cfg.broker.upper(), exc)
        return False


async def _run_checks() -> int:
    log.info("=" * 55)
    log.info("SYSTEM PRE-FLIGHT CHECK")
    log.info("=" * 55)

    failures = 0

    log.info("\n[1] Python packages")
    if not check_packages():
        failures += 1

    log.info("\n[2] GPU / CUDA")
    if not check_gpu():
        failures += 1

    log.info("\n[3] Disk space")
    if not check_disk():
        failures += 1

    log.info("\n[4] Environment (.env)")
    if not check_env():
        failures += 1

    log.info("\n[5] %s API connectivity", cfg.broker.upper())
    if not await check_broker_connectivity():
        failures += 1

    log.info("\n" + "=" * 55)
    if failures == 0:
        log.info("ALL CHECKS PASSED — bot is ready to run")
    else:
        log.error("%d CHECK(S) FAILED — fix the above issues before running the bot", failures)
    log.info("=" * 55)

    return failures


def main() -> None:
    failures = asyncio.run(_run_checks())
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
