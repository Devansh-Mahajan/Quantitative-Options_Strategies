"""Structured, colour-coded logging with daily rotation."""

from __future__ import annotations
import logging
import logging.handlers
import sys
from pathlib import Path

try:
    import colorlog
    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False


_INITIALIZED = False


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    log_dir.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # Console handler
    if _HAS_COLORLOG:
        color_fmt = (
            "%(log_color)s%(asctime)s | %(levelname)-8s%(reset)s | "
            "%(cyan)s%(name)s%(reset)s | %(message)s"
        )
        ch = colorlog.StreamHandler(sys.stdout)
        ch.setFormatter(
            colorlog.ColoredFormatter(
                color_fmt,
                datefmt=date_fmt,
                log_colors={
                    "DEBUG": "white",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
    else:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))

    ch.setLevel(numeric_level)
    root.addHandler(ch)

    # Rotating file handler — new file each day, keep 30 days
    fh = logging.handlers.TimedRotatingFileHandler(
        log_dir / "bot.log", when="midnight", backupCount=30, utc=True
    )
    fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    fh.setLevel(numeric_level)
    root.addHandler(fh)

    # Separate error log
    eh = logging.handlers.TimedRotatingFileHandler(
        log_dir / "errors.log", when="midnight", backupCount=30, utc=True
    )
    eh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    eh.setLevel(logging.ERROR)
    root.addHandler(eh)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "asyncio", "binance", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
