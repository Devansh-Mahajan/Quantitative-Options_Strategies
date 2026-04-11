"""Compatibility wrapper around the stdlib ``logging`` package.

This repository historically created a top-level ``logging`` package for strategy
helpers, which shadows Python's standard-library ``logging`` module on import
resolution. Many third-party dependencies (e.g. pandas) import ``logging`` and
expect the stdlib API to be available.

To keep local imports like ``logging.logger_setup`` working while preserving
stdlib behavior, we load stdlib ``logging`` directly from the Python stdlib
path and re-export its public attributes from this package namespace.
"""

from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path

_STDLIB_LOGGING_PATH = Path(sysconfig.get_path("stdlib")) / "logging" / "__init__.py"
_spec = importlib.util.spec_from_file_location("_stdlib_logging", _STDLIB_LOGGING_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load stdlib logging module from {_STDLIB_LOGGING_PATH}")

_stdlib_logging = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stdlib_logging)

# Re-export stdlib logging symbols without clobbering package metadata that makes
# local submodules (e.g. logging.logger_setup) importable.
for _name, _value in _stdlib_logging.__dict__.items():
    if _name in {"__name__", "__package__", "__path__", "__file__", "__spec__"}:
        continue
    globals()[_name] = _value

# Mirror stdlib surface as closely as possible for dir(logging) and wildcard imports.
__all__ = getattr(_stdlib_logging, "__all__", [])
